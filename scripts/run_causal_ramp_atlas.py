"""在严格 development OOF 上运行 A54 因果 disagreement/ramp 图谱。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import (
    config_fingerprint,
    dataframe_fingerprint,
    feature_schema_fingerprint,
    finalize_run,
    new_run_dir,
    write_json,
)
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.ramp_router import build_a54_causal_signal_atlas
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config


def _read_rows(path: Path) -> pd.DataFrame:
    """读取 CSV 或 Parquet OOF 长表并解析预测起点和训练边界。"""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def _load_config(path: Path | None) -> ForecastConfig:
    """恢复冻结 Champion 配置；未提供时使用项目默认配置。"""

    if path is None:
        return ForecastConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config 必须是 ForecastConfig JSON 对象")
    return forecast_config_from_dict(payload)


def _price_schedule(data_dir: Path):
    """读取训练目录中唯一的已知未来价格表；缺失时显式不使用价格。"""

    paths = sorted(data_dir.glob("*price*.xlsx"))
    if len(paths) > 1:
        raise ValueError(f"A54 发现多个 price 文件: {paths}")
    return load_price_schedule(paths[0]) if paths else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--champion-column", default="aggressive_r75_lgb20_pred")
    parser.add_argument("--rich-gas-column", default="rich_gas_blend_30_pred")
    parser.add_argument("--specialist-column", default="rich_g1_long_blend_30_pred")
    parser.add_argument("--min-history-rows", type=int, default=128)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    dataset = align_tables(args.data_dir, config.feature.frequency)
    causal_config = rich_feature_config(config, RICH_FEATURE_GROUPS, feature_profile="long_horizon")
    features = build_causal_features(
        dataset.frame,
        causal_config.feature,
        _price_schedule(args.data_dir),
    )
    result = build_a54_causal_signal_atlas(
        _read_rows(args.input),
        features,
        baseline_column=args.champion_column,
        rich_gas_column=args.rich_gas_column,
        specialist_column=args.specialist_column,
        min_history_rows=args.min_history_rows,
    )
    run_dir = args.run_dir or new_run_dir("results", "experiment_a54_causal_ramp_atlas")
    run_dir.mkdir(parents=True, exist_ok=True)
    result.cells.to_csv(run_dir / "cells.csv", index=False, encoding="utf-8")
    result.table.to_csv(run_dir / "signal_quintiles.csv", index=False, encoding="utf-8")
    result.ramp_table.to_csv(run_dir / "signal_quintile_ramps.csv", index=False, encoding="utf-8")
    result.cutoffs.to_csv(run_dir / "forward_quantile_cutoffs.csv", index=False, encoding="utf-8")
    write_json(run_dir / "report.json", result.report)
    write_json(run_dir / "config.json", asdict(causal_config))
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": "A54_causal_disagreement_ramp_atlas",
            "scope": "development",
            "is_smoke": False,
            "formal_candidate": False,
            "diagnostic_only": True,
            "blind_used": False,
            "pooled_mape": float(result.report["long_horizon_pairwise"]["specialist_mape"]),
            "baseline": args.rich_gas_column,
            "specialist": args.specialist_column,
            "input": str(args.input.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "config": asdict(causal_config),
            "config_file": "config.json",
            "data_hash": dataframe_fingerprint(dataset.frame),
            "feature_schema_hash": feature_schema_fingerprint(features),
            "config_hash": config_fingerprint(causal_config),
            "dataset_hash": dataframe_fingerprint(dataset.frame),
            "feature_hash": feature_schema_fingerprint(features),
            "report": "report.json",
            "cells": "cells.csv",
            "signal_quintiles": "signal_quintiles.csv",
            "signal_quintile_ramps": "signal_quintile_ramps.csv",
            "forward_quantile_cutoffs": "forward_quantile_cutoffs.csv",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "rows": result.report["rows"],
                "signals": result.report["signals"],
                "formal_candidate": result.report["formal_candidate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
