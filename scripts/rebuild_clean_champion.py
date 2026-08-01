"""严格 purge 下重建 M1 Clean Champion，并量化 E21R 路由协调。

该脚本只生成独立研究运行目录，不触碰 ``results/best``。正式晋级必须在
后续 Production Gate 通过后由机械 promotion 入口执行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import legacy_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.experiments import (
    build_fingerprints,
    finalize_run,
    new_run_dir,
    write_json,
)
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.oof import build_legacy_oof
from gas_forecast.routing import (
    leave_one_fold_out_route,
    reconcile_post_route,
)
from gas_forecast.scoring import (
    absolute_percentage_error,
    block_bootstrap_improvement_probability,
    score_oof_long,
)
from gas_forecast.selection_competition import choose_competition_candidate
from gas_forecast.orchestration import audit_future_perturbation


def _fixed_target_route(rows: pd.DataFrame, *, reconcile: bool) -> pd.DataFrame:
    output = rows.copy()
    output["v2_v3_target_pred"] = output["v3_pred"]
    output.loc[output["target"].eq("generator_1"), "v2_v3_target_pred"] = output.loc[
        output["target"].eq("generator_1"), "v2_pred"
    ]
    if reconcile:
        output = reconcile_post_route(output, prediction_column="v2_v3_target_pred")
    return output


def _fixed_target_route_spec() -> dict[str, object]:
    """返回部署时可复现的 generator_1→V2、generator_all→V3 路由。"""

    cells = {
        f"{target}|{horizon}": {
            "selected": "v2_pred" if target == "generator_1" else "v3_pred"
        }
        for target in ("generator_1", "generator_all")
        for horizon in range(15, 121, 15)
    }
    return {
        "policy": "fixed_v2_v3_target_route",
        "global": {"selected": "v3_pred"},
        "targets": {
            "generator_1": {"selected": "v2_pred"},
            "generator_all": {"selected": "v3_pred"},
        },
        "cells": cells,
        "post_route_reconciliation": {"enabled": True, "max_generator_rest": 240.0},
    }


def _attach_column(target: pd.DataFrame, source: pd.DataFrame, column: str) -> None:
    keys = ["fold", "origin_time", "target", "horizon"]
    target_index = pd.MultiIndex.from_frame(target[keys])
    source_index = pd.MultiIndex.from_frame(source[keys])
    aligned = source.set_index(source_index)[column].reindex(target_index)
    if aligned.isna().any():
        raise RuntimeError(f"路由候选未覆盖全部 OOF 行: {column}")
    target[column] = aligned.to_numpy(dtype=float)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重建严格 purge 的 Clean Champion")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def _resolve_data_dir(path: Path) -> Path:
    """兼容 Windows 执行器的 ASCII 父目录传参。"""

    if (path / "Pre_gas.csv").exists():
        return path
    matches = sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and (child / "Pre_gas.csv").exists()
    )
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"无法从数据目录解析官方表格: {path}")


def main() -> None:
    args = _parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs 必须大于等于 1")
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "clean_champion_oof")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = legacy_forecast_config()
    data_dir = _resolve_data_dir(args.data_dir)
    dataset = align_tables(data_dir, config.feature.frequency)
    prices = sorted(data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    features = build_causal_features(dataset.frame, config.feature, price)
    feature_audit = audit_future_perturbation(
        dataset.frame,
        config,
        price=price,
        baseline_features=features,
    )
    if not feature_audit["passed"]:
        raise RuntimeError(f"特征未来扰动审计失败: {feature_audit['changed_columns']}")

    result = build_legacy_oof(
        dataset.frame,
        features,
        versions=("v1", "v2", "v25", "v3"),
        config=config,
        n_jobs=args.jobs,
        checkpoint_dir=run_dir / "checkpoints",
    )
    rows = result.rows.copy()
    fixed_raw = _fixed_target_route(rows, reconcile=False)
    fixed_reconciled = _fixed_target_route(rows, reconcile=True)
    lofo_raw, lofo_report_raw = leave_one_fold_out_route(
        rows,
        ("persistence_pred", "v1_pred", "v2_pred", "v25_pred", "v3_pred"),
        post_route_reconciliation=False,
    )
    lofo_reconciled, lofo_report_reconciled = leave_one_fold_out_route(
        rows,
        ("persistence_pred", "v1_pred", "v2_pred", "v25_pred", "v3_pred"),
        post_route_reconciliation=True,
    )
    _attach_column(rows, fixed_raw, "v2_v3_target_pred")
    _attach_column(
        rows,
        fixed_reconciled.rename(columns={"v2_v3_target_pred": "v2_v3_target_reconciled_pred"}),
        "v2_v3_target_reconciled_pred",
    )
    _attach_column(rows, lofo_raw.rename(columns={"routed_pred": "lofo_raw_pred"}), "lofo_raw_pred")
    _attach_column(
        rows,
        lofo_reconciled.rename(columns={"routed_pred": "lofo_reconciled_pred"}),
        "lofo_reconciled_pred",
    )
    rows["routed_pred"] = rows["lofo_reconciled_pred"]
    candidates = {
        "persistence": "persistence_pred",
        "v1": "v1_pred",
        "v2": "v2_pred",
        "v25": "v25_pred",
        "v3": "v3_pred",
        "v2_v3_target_raw": "v2_v3_target_pred",
        "v2_v3_target_reconciled": "v2_v3_target_reconciled_pred",
        "lofo_raw": "lofo_raw_pred",
        "lofo_reconciled": "lofo_reconciled_pred",
    }
    reports = {
        name: score_oof_long(rows, column) for name, column in candidates.items()
    }
    selection = choose_competition_candidate(rows, candidates)
    selected = str(selection["selected_candidate"])
    runner_up = str(selection.get("runner_up", selected))
    stability = {
        "selected": selected,
        "runner_up": runner_up,
    }
    if runner_up != selected:
        selected_column = candidates[selected]
        runner_column = candidates[runner_up]
        stability.update(
            {
                "day_block_bootstrap": block_bootstrap_improvement_probability(
                    rows, selected_column, runner_column, block="day"
                ),
                "fold_block_bootstrap": block_bootstrap_improvement_probability(
                    rows, selected_column, runner_column, block="fold"
                ),
            }
        )
        development = rows.loc[rows["fold"].ne("blind")]
        differences = (
            score_oof_long(development, selected_column)["pooled_mape"]
            - score_oof_long(development, runner_column)["pooled_mape"]
        )
        selected_ape = absolute_percentage_error(
            development["actual"], development[selected_column]
        )
        runner_ape = absolute_percentage_error(
            development["actual"], development[runner_column]
        )
        fold_scores = (
            pd.DataFrame(
                {"fold": development["fold"], "difference": selected_ape - runner_ape}
            )
            .groupby("fold", sort=True)["difference"]
            .mean()
        )
        stability.update(
            {
                "development_difference": float(differences),
                "development_fold_win_rate": float((fold_scores < 0.0).mean()),
                "development_worst_fold_regression": float(fold_scores.max()),
                "development_recent_folds": {
                    str(key): float(value) for key, value in fold_scores.tail(5).items()
                },
            }
        )
    route_report = {
        "fixed_target_route": _fixed_target_route_spec(),
        "fixed_target_raw": {
            "pooled_mape": reports["v2_v3_target_raw"]["pooled_mape"],
            "by_target": reports["v2_v3_target_raw"]["by_target"],
        },
        "fixed_target_reconciled": {
            "pooled_mape": reports["v2_v3_target_reconciled"]["pooled_mape"],
            "by_target": reports["v2_v3_target_reconciled"]["by_target"],
        },
        "lofo_raw": lofo_report_raw,
        "lofo_reconciled": lofo_report_reconciled,
        "reconciliation_delta": {
            "modified_cells": int(
                (
                    ~np.isclose(
                    rows["lofo_raw_pred"],
                    rows["lofo_reconciled_pred"],
                    equal_nan=True,
                    )
                ).sum()
            ),
            "raw_pooled_mape": reports["lofo_raw"]["pooled_mape"],
            "reconciled_pooled_mape": reports["lofo_reconciled"]["pooled_mape"],
        },
    }
    rows.to_csv(run_dir / "oof.csv", index=False, encoding="utf-8")
    write_json(run_dir / "oof_report.json", result.report)
    rows.to_csv(run_dir / "oof_with_routes.csv", index=False, encoding="utf-8")
    report = {
        "stage": "Phase_0_clean_champion",
        "candidate_reports": reports,
        "selection": selection,
        "selected_candidate": selected,
        "route_report": route_report,
        "stability": stability,
        "feature_future_perturbation": feature_audit,
        "strict_label_purge": True,
        "purge_minutes": 15 * (max(config.feature.horizons) + 1),
        "folds": result.report["folds"],
        "dataset_audit": dataset.audit.to_dict(),
        "fingerprints": build_fingerprints(
            config=config,
            dataset=dataset.frame,
            features=features,
            model_params={"versions": ["v1", "v2", "v25", "v3"]},
        ),
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "selection.json", selection)
    finalize_run(
        run_dir,
        {
            "run_type": "oof",
            "stage": "Phase_0_clean_champion",
            "is_smoke": False,
            "outer_folds": len(result.report["folds"]),
            "pooled_mape": float(reports[selected]["pooled_mape"]),
            "candidate": selected,
            "config": {
                "targets": list(config.targets),
                "feature": config.feature.__dict__,
                "model": config.model.__dict__,
                "validation": config.validation.__dict__,
            },
            **report["fingerprints"],
            "report": "report.json",
            "oof": "oof_with_routes.csv",
            "strict_label_purge": True,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
