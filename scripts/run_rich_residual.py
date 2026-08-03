"""运行 generator_1 RichResidual 的严格 OOF 分组筛选或完整开发验证。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.features import load_price_schedule
from gas_forecast.rich_residual import (
    RICH_FEATURE_GROUPS,
    RichResidualSpec,
    build_rich_residual_oof,
)


OOF_KEYS = ["fold", "origin_time", "target", "horizon"]


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time"])


def _load_config(path: Path | None) -> ForecastConfig:
    if path is None:
        return ForecastConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config 必须是 ForecastConfig JSON 对象")
    return forecast_config_from_dict(payload)


def _parse_group_set(value: str) -> frozenset[str]:
    normalized = value.strip().lower()
    if normalized in {"", "base", "none"}:
        return frozenset()
    groups = frozenset(item.strip() for item in normalized.split(",") if item.strip())
    invalid = sorted(groups.difference(RICH_FEATURE_GROUPS))
    if invalid:
        raise ValueError(f"未知 Rich 特征组: {invalid}")
    return groups


def _name_for_groups(groups: frozenset[str]) -> str:
    return "rich_base" if not groups else "rich_" + "_".join(sorted(groups))


def _selection_payload(reports: dict[str, dict[str, object]]) -> dict[str, object]:
    """仅从 screening 结果选择可进入组合验证的至多两个组。"""

    best_by_group: dict[str, dict[str, object]] = {}
    for group_name, report in reports.items():
        models = report.get("models", {})
        if not isinstance(models, dict):
            continue
        ranked = sorted(
            (
                {
                    "candidate": candidate,
                    "pooled_difference": float(metrics["pooled_difference"]),
                    "generator_1_difference": float(metrics["generator_1_difference"]),
                }
                for candidate, metrics in models.items()
                if isinstance(metrics, dict)
            ),
            key=lambda item: (
                float(item["generator_1_difference"]),
                float(item["pooled_difference"]),
                str(item["candidate"]),
            ),
        )
        if ranked:
            best_by_group[group_name] = ranked[0]
    eligible = [
        {"group": group, **result}
        for group, result in best_by_group.items()
        if group != "base"
        and float(result["pooled_difference"]) < 0.0
        and float(result["generator_1_difference"]) <= 0.0
    ]
    eligible.sort(
        key=lambda item: (
            float(item["generator_1_difference"]),
            float(item["pooled_difference"]),
            str(item["group"]),
        )
    )
    return {
        "selection_scope": "screening",
        "best_by_group": best_by_group,
        "selected_groups": [item["group"] for item in eligible[:2]],
        "selection_rule": (
            "只保留 pooled 与 generator_1 均不退化的单组，按 generator_1 差值、"
            "pooled 差值和组名排序，最多选择两个进入 development 组合。"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument(
        "--baseline-column",
        default="aggressive_r75_lgb20_pred",
        help="严格 Champion OOF 中的同折基线预测列",
    )
    parser.add_argument("--config", type=Path, help="冻结 Champion 的 config.json")
    parser.add_argument(
        "--scope",
        choices=("screening", "development", "final"),
        default="screening",
    )
    parser.add_argument(
        "--group-set",
        action="append",
        default=[],
        help="一个 Rich 组组合，例如 base、quantile、ramp、gas、quantile,ramp；可重复提供",
    )
    parser.add_argument(
        "--screen-all",
        action="store_true",
        help="快捷运行 base、quantile、ramp、gas 四个独立筛选组",
    )
    parser.add_argument(
        "--frozen-final",
        action="store_true",
        help="确认本次只对单一已冻结组执行一次 final/blind 报告，不用于选参",
    )
    parser.add_argument("--min-train-rows", type=int, default=256)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_train_rows < 16:
        raise ValueError("--min-train-rows 必须至少为 16")
    groups = [_parse_group_set(value) for value in args.group_set]
    if args.screen_all:
        groups.extend([frozenset(), frozenset({"quantile"}), frozenset({"ramp"}), frozenset({"gas"})])
    if not groups:
        groups = [frozenset()]
    group_sets = list(dict.fromkeys(groups))
    if args.scope == "final" and (not args.frozen_final or len(group_sets) != 1):
        raise ValueError("final/blind 只能显式确认且只运行一个已冻结的 Rich 特征组")
    config = _load_config(args.config)
    dataset = align_tables(args.data_dir, config.feature.frequency)
    price_paths = sorted(args.data_dir.glob("*price*.xlsx"))
    price_schedule = load_price_schedule(price_paths[0]) if price_paths else None
    champion = _read_frame(args.baseline_oof)
    run_dir = args.run_dir or new_run_dir("results", "experiment_rich_residual")
    run_dir.mkdir(parents=True, exist_ok=True)

    merged: pd.DataFrame | None = None
    reports: dict[str, dict[str, object]] = {}
    effective_configs: dict[str, dict[str, object]] = {}
    group_dir = run_dir / "groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    for group_set in group_sets:
        name = _name_for_groups(group_set)
        spec = RichResidualSpec(
            name=name,
            feature_groups=group_set,
            min_train_rows=args.min_train_rows,
            n_estimators=args.n_estimators,
        )
        result = build_rich_residual_oof(
            dataset.frame,
            champion,
            config=config,
            spec=spec,
            baseline_column=args.baseline_column,
            scope=args.scope,
            price_schedule=price_schedule,
        )
        candidate_columns = [column for column in result.rows if column.startswith(name)]
        if merged is None:
            base_columns = [
                column
                for column in (
                    *OOF_KEYS,
                    "train_end",
                    "actual",
                    "current_value",
                    "persistence_pred",
                    args.baseline_column,
                )
                if column in result.rows
            ]
            merged = result.rows.loc[:, base_columns + candidate_columns].copy()
        else:
            addition = result.rows.loc[:, OOF_KEYS + candidate_columns]
            merged = merged.merge(addition, on=OOF_KEYS, how="inner", validate="one_to_one")
            if len(merged) != len(result.rows):
                raise ValueError("RichResidual 候选 OOF 键不完整，拒绝合并")
        reports["base" if not group_set else "+".join(sorted(group_set))] = result.report
        effective_configs[name] = asdict(result.feature_config)
        # 长开发验证按组落盘，进程意外结束时仍保留已完成的严格 OOF 收据。
        result.rows.to_csv(group_dir / f"{name}_oof.csv", index=False, encoding="utf-8")
        write_json(
            group_dir / f"{name}_report.json",
            {
                "name": name,
                "feature_groups": sorted(group_set),
                "report": result.report,
                "effective_config": asdict(result.feature_config),
            },
        )

    assert merged is not None
    output_path = run_dir / "oof.csv"
    report_path = run_dir / "report.json"
    config_path = run_dir / "config.json"
    merged.to_csv(output_path, index=False, encoding="utf-8")
    payload: dict[str, object] = {
        "scope": args.scope,
        "baseline_oof": str(args.baseline_oof.resolve()),
        "baseline_column": args.baseline_column,
        "candidates": reports,
        "effective_configs": effective_configs,
        "strict_oof": True,
        "blind_used_for_selection": False,
    }
    if args.scope == "screening" and args.screen_all:
        selection = _selection_payload(reports)
        payload["screening_selection"] = selection
        write_json(run_dir / "screening_selection.json", selection)
    write_json(report_path, payload)
    write_json(config_path, asdict(config))
    best_metrics = [
        float(metrics["pooled_mape"])
        for report in reports.values()
        for metrics in report.get("models", {}).values()
        if isinstance(metrics, dict) and isinstance(metrics.get("pooled_mape"), (int, float))
    ]
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": "rich_residual",
            "scope": args.scope,
            "is_smoke": args.scope == "screening",
            "blind_included": args.scope == "final",
            "formal_candidate": False,
            "pooled_mape": min(best_metrics) if best_metrics else None,
            "baseline": args.baseline_column,
            "config": asdict(config),
            "report": "report.json",
            "oof": "oof.csv",
            "config_file": "config.json",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "scope": args.scope,
                "groups": ["base" if not item else "+".join(sorted(item)) for item in group_sets],
                "rows": len(merged),
                "selection": payload.get("screening_selection"),
            },
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
    )


if __name__ == "__main__":
    main()
