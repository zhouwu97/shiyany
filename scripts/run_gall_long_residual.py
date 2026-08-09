"""在严格 development OOF 上运行 A60 generator_all 长步长残差专模。"""

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
from gas_forecast.rich_residual import RichResidualSpec, build_rich_residual_oof


A60_TARGET = "generator_all"
A60_ACTIVE_HORIZONS = (75, 90, 105, 120)
A60_BLEND_WEIGHT = 0.30
A60_MIN_TRAIN_ROWS = 256
A60_RETAIN_POOLED_IMPROVEMENT_PP = 0.005
A60_MIN_RECENT5_WINS = 3
OOF_KEYS = ["fold", "origin_time", "target", "horizon"]


def _read_rows(path: Path) -> pd.DataFrame:
    """读取包含当前 g1/gall 强基线的 development OOF。"""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def _load_config(path: Path) -> ForecastConfig:
    """恢复 A51 同源冻结配置，禁止 A60 在运行时改模型容量。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config 必须是 ForecastConfig JSON 对象")
    return forecast_config_from_dict(payload)


def _price_schedule(data_dir: Path):
    """读取唯一已知未来价格表；缺失时显式不使用价格。"""

    paths = sorted(data_dir.glob("*price*.xlsx"))
    if len(paths) > 1:
        raise ValueError(f"A60 发现多个 price 文件: {paths}")
    return load_price_schedule(paths[0]) if paths else None


def _validate_development_rows(
    rows: pd.DataFrame,
    *,
    parent_column: str,
    rich_gas_column: str,
) -> pd.DataFrame:
    """验证 A60 只消费无 blind 的完整 development 基线 OOF。"""

    required = {
        *OOF_KEYS,
        "train_end",
        "actual",
        "current_value",
        "persistence_pred",
        parent_column,
        rich_gas_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"A60 输入 OOF 缺少字段: {missing}")
    work = rows.copy()
    work["fold"] = work["fold"].astype(str)
    if work["fold"].eq("blind").any():
        raise ValueError("A60 只接受 development OOF，输入不得含 blind 行")
    for column in ("origin_time", "train_end"):
        work[column] = pd.to_datetime(work[column], errors="coerce")
        if work[column].isna().any():
            raise ValueError(f"A60 输入含非法 {column}")
    if work.duplicated(OOF_KEYS).any():
        raise ValueError("A60 输入存在重复 fold×origin×target×horizon")
    numeric_columns = [
        "actual",
        "current_value",
        "persistence_pred",
        parent_column,
        rich_gas_column,
    ]
    numeric = work.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("A60 输入的真实值或预测含缺失/非有限数")
    work.loc[:, numeric.columns] = numeric
    work["horizon"] = pd.to_numeric(work["horizon"], errors="raise").astype(int)
    return work.sort_values(["origin_time", "target", "horizon", "fold"]).reset_index(drop=True)


def _eligible_long_gall(rows: pd.DataFrame) -> pd.Series:
    """返回 A60 唯一允许改写的 generator_all 长步长单元。"""

    return rows["target"].eq(A60_TARGET) & rows["horizon"].isin(A60_ACTIVE_HORIZONS)


def _route_audit(
    rows: pd.DataFrame,
    *,
    raw_column: str,
    parent_column: str,
) -> dict[str, object]:
    """确认原始 A60 只修改 gall 长步长，容量投影另行透明记录。"""

    changed = ~np.isclose(
        rows[raw_column].to_numpy(dtype=float),
        rows[parent_column].to_numpy(dtype=float),
    )
    noneligible = int((changed & ~_eligible_long_gall(rows).to_numpy(dtype=bool)).sum())
    if noneligible:
        raise RuntimeError("A60 原始候选修改了 generator_all 长步长以外的单元")
    return {
        "raw_changed_cells": int(changed.sum()),
        "noneligible_raw_changed_cells": noneligible,
        "selector_only_changes_gall_long": bool(noneligible == 0),
    }


def _recent5_wins(comparison: dict[str, object]) -> int:
    """统计候选相对冻结父模型的最近五折胜数。"""

    recent = comparison["recent_5_folds_difference"]
    if not isinstance(recent, dict):
        raise TypeError("A60 比较报告缺少最近五折差值")
    return int(sum(float(value) < 0.0 for value in recent.values()))


def _status(comparison: dict[str, object]) -> dict[str, object]:
    """执行 A60 的固定保留门槛，不授予 blind 或生产权限。"""

    pooled_improvement_pp = -float(comparison["pooled_difference"]) * 100.0
    pairwise = comparison["pairwise"]
    if not isinstance(pairwise, dict):
        raise TypeError("A60 比较报告缺少 pairwise 指标")
    by_target = pairwise["by_target"]
    if not isinstance(by_target, dict) or A60_TARGET not in by_target:
        raise TypeError("A60 比较报告缺少 generator_all 指标")
    target_metrics = by_target[A60_TARGET]
    if not isinstance(target_metrics, dict):
        raise TypeError("A60 generator_all 指标格式错误")
    target_improvement_pp = -float(target_metrics["difference"]) * 100.0
    recent5_wins = _recent5_wins(comparison)
    retained = bool(
        pooled_improvement_pp >= A60_RETAIN_POOLED_IMPROVEMENT_PP
        and target_improvement_pp > 0.0
        and recent5_wins >= A60_MIN_RECENT5_WINS
    )
    return {
        "pooled_improvement_pp": pooled_improvement_pp,
        "generator_all_improvement_pp": target_improvement_pp,
        "recent5_wins": recent5_wins,
        "status": "RETAIN_GALL_DIVERSITY" if retained else "DO_NOT_RETAIN",
        "acceptance": {
            "pooled_improvement_pp_at_least": A60_RETAIN_POOLED_IMPROVEMENT_PP,
            "generator_all_improvement_pp_greater_than": 0.0,
            "recent5_wins_at_least": A60_MIN_RECENT5_WINS,
        },
    }


def _spec_payload(spec: RichResidualSpec) -> dict[str, object]:
    """写入 A60 唯一冻结模型规格，避免把 defaults 当作可调参数。"""

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
    parser.add_argument("--parent-column", default="rich_short00_long100_pred")
    parser.add_argument("--rich-gas-column", default="rich_gas_blend_30_pred")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    dataset = align_tables(args.data_dir, config.feature.frequency)
    rows = _validate_development_rows(
        _read_rows(args.input),
        parent_column=args.parent_column,
        rich_gas_column=args.rich_gas_column,
    )
    spec = RichResidualSpec(
        name="a60_gall_long",
        target=A60_TARGET,
        feature_groups=frozenset({"quantile", "ramp", "gas"}),
        feature_profile="long_horizon",
        active_horizons=A60_ACTIVE_HORIZONS,
        include_champion_prediction=True,
        min_train_rows=A60_MIN_TRAIN_ROWS,
        blend_weights=(A60_BLEND_WEIGHT,),
    )
    result = build_rich_residual_oof(
        dataset.frame,
        rows,
        config=config,
        spec=spec,
        baseline_column=args.parent_column,
        scope="development",
        price_schedule=_price_schedule(args.data_dir),
    )
    raw_column = f"{spec.name}_residual_raw_pred"
    candidate_column = f"{spec.name}_blend_30_pred"
    comparison_to_parent = result.report["models"][candidate_column]
    comparison_to_rich_gas = compare_research_candidate(
        result.rows,
        candidate_column,
        args.rich_gas_column,
        scope="development",
    )
    parent_vs_rich_gas = compare_research_candidate(
        rows,
        args.parent_column,
        args.rich_gas_column,
        scope="development",
    )
    route_audit = _route_audit(
        result.rows,
        raw_column=raw_column,
        parent_column=args.parent_column,
    )
    status = _status(comparison_to_parent)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "A60_generator_all_long_residual",
        "scope": "development",
        "input": str(args.input.resolve()),
        "parent_column": args.parent_column,
        "rich_gas_column": args.rich_gas_column,
        "target_scope": A60_TARGET,
        "eligible_horizons": list(A60_ACTIVE_HORIZONS),
        "spec": _spec_payload(spec),
        "rows": int(len(result.rows)),
        "eligible_rows": int(_eligible_long_gall(result.rows).sum()),
        "feature_columns": result.report["feature_columns"],
        "selected_feature_columns": result.report["selected_feature_columns"],
        "fold_training_rows": result.report["fold_training_rows"],
        "trained_horizons": result.report["trained_horizons"],
        "champion_prediction_feature": result.report["champion_prediction_feature"],
        "route_audit": route_audit,
        "parent_vs_rich_gas": parent_vs_rich_gas,
        "comparison_to_parent": comparison_to_parent,
        "comparison_to_rich_gas": comparison_to_rich_gas,
        "status": status,
        "formal_candidate": False,
        "blind_used": False,
        "strict_oof_contract": result.report["strict_oof_contract"],
    }
    result.rows.to_csv(args.run_dir / "oof.csv", index=False, encoding="utf-8")
    write_json(args.run_dir / "report.json", report)
    write_json(args.run_dir / "config.json", asdict(config))
    finalize_run(
        args.run_dir,
        {
            "run_type": "experiment",
            "stage": "A60_generator_all_long_residual",
            "scope": "development",
            "is_smoke": False,
            "formal_candidate": False,
            "blind_used": False,
            "pooled_mape": float(comparison_to_parent["pooled_mape"]),
            "input": str(args.input.resolve()),
            "parent": args.parent_column,
            "rich_gas": args.rich_gas_column,
            "data_dir": str(args.data_dir.resolve()),
            "data_hash": dataframe_fingerprint(dataset.frame),
            "config": asdict(config),
            "config_hash": config_fingerprint(config),
            "report": "report.json",
            "oof": "oof.csv",
            "config_file": "config.json",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "status": status,
                "blind_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
