"""在严格 development OOF 上运行 A56 A51 长步长特征组消融。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import (
    config_fingerprint,
    dataframe_fingerprint,
    finalize_run,
    write_json,
)
from gas_forecast.features import load_price_schedule
from gas_forecast.research import compare_research_candidate
from gas_forecast.rich_residual import (
    LONG_HORIZON_ABLATION_GROUP_ORDER,
    RichResidualSpec,
    build_rich_residual_oof,
)


A56_ACTIVE_HORIZONS = (75, 90, 105, 120)
A56_BLEND_WEIGHT = 0.30
A56_MIN_TRAIN_ROWS = 256
A56_MIN_RICH_GAS_IMPROVEMENT_PP = 0.005
A56_MAX_A51_POOLED_REGRESSION_PP = 0.001
A56_MIN_RECENT5_WINS = 3
OOF_KEYS = ["fold", "origin_time", "target", "horizon"]


def _read_rows(path: Path) -> pd.DataFrame:
    """读取 A51 splice OOF，并保留严格训练边界的时间类型。"""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def _load_config(path: Path) -> ForecastConfig:
    """恢复 A51 的冻结配置，拒绝由 A56 隐式改写模型容量。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config 必须是 ForecastConfig JSON 对象")
    return forecast_config_from_dict(payload)


def _price_schedule(data_dir: Path):
    """读取唯一已知未来价格表；没有价格表时明确关闭该输入。"""

    paths = sorted(data_dir.glob("*price*.xlsx"))
    if len(paths) > 1:
        raise ValueError(f"A56 发现多个 price 文件: {paths}")
    return load_price_schedule(paths[0]) if paths else None


def _validate_development_rows(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    a51_column: str,
) -> pd.DataFrame:
    """验证输入只含 development OOF 且具备 A51 对照列。"""

    required = {
        *OOF_KEYS,
        "train_end",
        "actual",
        "current_value",
        "persistence_pred",
        baseline_column,
        a51_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"A56 输入 OOF 缺少字段: {missing}")
    work = rows.copy()
    work["fold"] = work["fold"].astype(str)
    if work["fold"].eq("blind").any():
        raise ValueError("A56 只接受 development OOF，输入不得含 blind 行")
    for column in ("origin_time", "train_end"):
        work[column] = pd.to_datetime(work[column], errors="coerce")
        if work[column].isna().any():
            raise ValueError(f"A56 输入含非法 {column}")
    if work.duplicated(OOF_KEYS).any():
        raise ValueError("A56 输入存在重复 fold×origin×target×horizon")
    numeric_columns = [
        "actual",
        "current_value",
        "persistence_pred",
        baseline_column,
        a51_column,
    ]
    numeric = work.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("A56 输入的真实值或预测含缺失/非有限数")
    work.loc[:, numeric.columns] = numeric
    work["horizon"] = pd.to_numeric(work["horizon"], errors="raise").astype(int)
    return work.sort_values(["origin_time", "target", "horizon", "fold"]).reset_index(drop=True)


def _eligible_long_g1(rows: pd.DataFrame) -> pd.Series:
    """返回 A56 唯一允许改写的 g1 长步长单元。"""

    return rows["target"].eq("generator_1") & rows["horizon"].isin(A56_ACTIVE_HORIZONS)


def _route_audit(
    rows: pd.DataFrame,
    *,
    raw_column: str,
    baseline_column: str,
) -> dict[str, object]:
    """独立审计每个消融候选不会改写 12 个非目标预测单元。"""

    changed = ~np.isclose(
        rows[raw_column].to_numpy(dtype=float),
        rows[baseline_column].to_numpy(dtype=float),
    )
    noneligible = int((changed & ~_eligible_long_g1(rows).to_numpy(dtype=bool)).sum())
    if noneligible:
        raise RuntimeError("A56 原始候选修改了 generator_1 长步长以外的单元")
    return {
        "raw_changed_cells": int(changed.sum()),
        "noneligible_raw_changed_cells": noneligible,
        "selector_only_changes_g1_long": bool(noneligible == 0),
    }


def _recent5_wins(comparison: dict[str, object]) -> int:
    """按现有实验口径统计候选相对父模型最近五折胜数。"""

    recent = comparison["recent_5_folds_difference"]
    if not isinstance(recent, dict):
        raise TypeError("A56 比较报告缺少最近五折差值")
    return int(sum(float(value) < 0.0 for value in recent.values()))


def _status(
    rich_gas_comparison: dict[str, object],
    a51_comparison: dict[str, object],
) -> dict[str, object]:
    """执行 A56 的固定稳定性门槛，不对消融结果进行排序选优。"""

    rich_gas_improvement_pp = -float(rich_gas_comparison["pooled_difference"]) * 100.0
    a51_regression_pp = float(a51_comparison["pooled_difference"]) * 100.0
    recent5_wins = _recent5_wins(rich_gas_comparison)
    retained = bool(
        rich_gas_improvement_pp >= A56_MIN_RICH_GAS_IMPROVEMENT_PP
        and a51_regression_pp <= A56_MAX_A51_POOLED_REGRESSION_PP
        and recent5_wins >= A56_MIN_RECENT5_WINS
    )
    return {
        "rich_gas_pooled_improvement_pp": rich_gas_improvement_pp,
        "a51_pooled_regression_pp": a51_regression_pp,
        "recent5_wins_vs_rich_gas": recent5_wins,
        "status": "RETAIN_STABILITY" if retained else "DO_NOT_RETAIN",
        "acceptance": {
            "rich_gas_pooled_improvement_pp_at_least": A56_MIN_RICH_GAS_IMPROVEMENT_PP,
            "a51_pooled_regression_pp_at_most": A56_MAX_A51_POOLED_REGRESSION_PP,
            "recent5_wins_vs_rich_gas_at_least": A56_MIN_RECENT5_WINS,
        },
    }


def _spec(group: str) -> RichResidualSpec:
    """构造唯一允许的 A51 删除一整组字段的固定配置。"""

    return RichResidualSpec(
        name=f"a56_without_{group}",
        feature_groups=frozenset({"quantile", "ramp", "gas"}),
        feature_profile="long_horizon",
        active_horizons=A56_ACTIVE_HORIZONS,
        include_champion_prediction=True,
        exclude_long_feature_groups=frozenset({group}),
        min_train_rows=A56_MIN_TRAIN_ROWS,
        blend_weights=(A56_BLEND_WEIGHT,),
    )


def _spec_payload(spec: RichResidualSpec) -> dict[str, object]:
    """将不可变 spec 转为 JSON 原生结构，保留消融组收据。"""

    return {
        "name": spec.name,
        "target": spec.target,
        "feature_groups": sorted(spec.feature_groups),
        "feature_profile": spec.feature_profile,
        "active_horizons": list(spec.active_horizons or ()),
        "include_champion_prediction": spec.include_champion_prediction,
        "exclude_long_feature_groups": sorted(spec.exclude_long_feature_groups),
        "min_train_rows": spec.min_train_rows,
        "n_estimators": spec.n_estimators,
        "blend_weights": list(spec.blend_weights),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-column", default="rich_gas_blend_30_pred")
    parser.add_argument("--a51-column", default="rich_short00_long100_pred")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    dataset = align_tables(args.data_dir, config.feature.frequency)
    input_rows = _validate_development_rows(
        _read_rows(args.input),
        baseline_column=args.baseline_column,
        a51_column=args.a51_column,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = args.run_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    base_columns = [
        *OOF_KEYS,
        "train_end",
        "actual",
        "current_value",
        "persistence_pred",
        args.baseline_column,
        args.a51_column,
    ]
    merged = input_rows.loc[:, base_columns].copy()
    group_reports: dict[str, dict[str, object]] = {}
    retained: list[dict[str, object]] = []
    price_schedule = _price_schedule(args.data_dir)
    for group in LONG_HORIZON_ABLATION_GROUP_ORDER:
        spec = _spec(group)
        result = build_rich_residual_oof(
            dataset.frame,
            input_rows,
            config=config,
            spec=spec,
            baseline_column=args.baseline_column,
            scope="development",
            price_schedule=price_schedule,
        )
        raw_column = f"{spec.name}_residual_raw_pred"
        candidate_column = f"{spec.name}_blend_30_pred"
        candidate_columns = [column for column in result.rows if column.startswith(spec.name)]
        comparison_to_rich_gas = result.report["models"][candidate_column]
        comparison_to_a51 = compare_research_candidate(
            result.rows,
            candidate_column,
            args.a51_column,
            scope="development",
        )
        route_audit = _route_audit(
            result.rows,
            raw_column=raw_column,
            baseline_column=args.baseline_column,
        )
        status = _status(comparison_to_rich_gas, comparison_to_a51)
        record = {
            "group": group,
            "spec": _spec_payload(spec),
            "candidate_column": candidate_column,
            "feature_columns": result.report["feature_columns"],
            "removed_feature_columns": 249 - int(result.report["feature_columns"]),
            "feature_group_counts": result.report["long_horizon_feature_group_counts"],
            "route_audit": route_audit,
            "comparison_to_rich_gas": comparison_to_rich_gas,
            "comparison_to_a51": comparison_to_a51,
            "status": status,
            "strict_oof_contract": result.report["strict_oof_contract"],
        }
        group_reports[group] = record
        if status["status"] == "RETAIN_STABILITY":
            retained.append(
                {
                    "group": group,
                    "candidate": candidate_column,
                    **status,
                }
            )
        group_output_columns = list(dict.fromkeys(base_columns + candidate_columns))
        result.rows.loc[:, group_output_columns].to_csv(
            groups_dir / f"{spec.name}_oof.csv",
            index=False,
            encoding="utf-8",
        )
        write_json(groups_dir / f"{spec.name}_report.json", record)
        addition = result.rows.loc[:, OOF_KEYS + candidate_columns]
        merged = merged.merge(addition, on=OOF_KEYS, how="inner", validate="one_to_one")
        if len(merged) != len(input_rows):
            raise RuntimeError("A56 候选 OOF 键不完整，拒绝合并")

    report = {
        "stage": "A56_a51_group_ablation",
        "scope": "development",
        "input": str(args.input.resolve()),
        "baseline_column": args.baseline_column,
        "a51_column": args.a51_column,
        "target_scope": "generator_1",
        "eligible_horizons": list(A56_ACTIVE_HORIZONS),
        "ablation_groups": list(LONG_HORIZON_ABLATION_GROUP_ORDER),
        "fixed_spec": {
            "feature_groups": ["gas", "quantile", "ramp"],
            "feature_profile": "long_horizon",
            "include_champion_prediction": True,
            "blend_weight": A56_BLEND_WEIGHT,
            "min_train_rows": A56_MIN_TRAIN_ROWS,
        },
        "pre_registered_acceptance": {
            "goal": "寻找 recent5 稳定性改善而不实质损失 A51 总收益的整组删除",
            "rich_gas_pooled_improvement_pp_at_least": A56_MIN_RICH_GAS_IMPROVEMENT_PP,
            "a51_pooled_regression_pp_at_most": A56_MAX_A51_POOLED_REGRESSION_PP,
            "recent5_wins_vs_rich_gas_at_least": A56_MIN_RECENT5_WINS,
            "no_feature_subset_search": True,
            "no_weight_search": True,
        },
        "rows": int(len(merged)),
        "eligible_rows": int(_eligible_long_g1(merged).sum()),
        "groups": group_reports,
        "retained_stability_candidates": retained,
        "formal_candidate": False,
        "blind_used": False,
        "strict_oof_contract": {
            "development_only": True,
            "blind_rows_accepted": False,
            "same_a51_long_horizon_spec": True,
            "only_g1_long_raw_cells_changed": True,
            "capacity_projection": "每个候选与固定 30% 融合均使用生产一致的容量投影",
        },
    }
    merged.to_csv(args.run_dir / "oof.csv", index=False, encoding="utf-8")
    write_json(args.run_dir / "report.json", report)
    write_json(args.run_dir / "config.json", asdict(config))
    pooled_scores = [
        float(record["comparison_to_rich_gas"]["pooled_mape"])
        for record in group_reports.values()
    ]
    finalize_run(
        args.run_dir,
        {
            "run_type": "experiment",
            "stage": "A56_a51_group_ablation",
            "scope": "development",
            "is_smoke": False,
            "formal_candidate": False,
            "blind_used": False,
            "pooled_mape": min(pooled_scores),
            "input": str(args.input.resolve()),
            "baseline": args.baseline_column,
            "a51_parent": args.a51_column,
            "data_dir": str(args.data_dir.resolve()),
            "data_hash": dataframe_fingerprint(dataset.frame),
            "config": asdict(config),
            "config_hash": config_fingerprint(config),
            "report": "report.json",
            "oof": "oof.csv",
            "config_file": "config.json",
            "groups_dir": "groups",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "retained_stability_candidates": retained,
                "blind_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
