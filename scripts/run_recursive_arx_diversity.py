"""在严格 development OOF 上运行 A61 Recursive ARX diversity。"""

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
    write_json,
)
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.recursive_arx import build_recursive_arx_diversity
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config


def _read_rows(path: Path):
    """读取 A60 development OOF，并解析时间字段。"""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def _load_config(path: Path) -> ForecastConfig:
    """恢复父模型冻结配置，禁止 A61 改写特征开关。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config 必须是 ForecastConfig JSON 对象")
    return forecast_config_from_dict(payload)


def _price_schedule(data_dir: Path):
    """读取唯一的官方已知未来价格表；多个文件时直接失败。"""

    paths = sorted(data_dir.glob("*price*.xlsx"))
    if len(paths) > 1:
        raise ValueError(f"A61 发现多个 price 文件: {paths}")
    if not paths:
        raise ValueError("A61 Recursive ARX 必须有唯一官方 price.xlsx")
    return load_price_schedule(paths[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-column", default="a60_gall_long_blend_30_pred")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    dataset = align_tables(args.data_dir, config.feature.frequency)
    effective_config = rich_feature_config(
        config,
        RICH_FEATURE_GROUPS,
        feature_profile="long_horizon",
    )
    price_schedule = _price_schedule(args.data_dir)
    features = build_causal_features(
        dataset.frame,
        effective_config.feature,
        price_schedule,
    )
    result = build_recursive_arx_diversity(
        dataset.frame,
        features,
        _read_rows(args.input),
        baseline_column=args.baseline_column,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    report = dict(result.report)
    report["input"] = str(args.input.resolve())
    report["effective_feature_config"] = asdict(effective_config)
    result.rows.to_csv(args.run_dir / "oof.csv", index=False, encoding="utf-8")
    result.training_trace.to_csv(args.run_dir / "training_trace.csv", index=False, encoding="utf-8")
    write_json(args.run_dir / "report.json", report)
    write_json(args.run_dir / "config.json", asdict(effective_config))
    finalize_run(
        args.run_dir,
        {
            "run_type": "experiment",
            "stage": "A61_recursive_arx_diversity",
            "scope": "development",
            "is_smoke": False,
            "formal_candidate": False,
            "blind_used": False,
            "status": report["status"],
            "baseline": args.baseline_column,
            "retained_fixed_blends": report["retained_fixed_blends"],
            "input": str(args.input.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "data_hash": dataframe_fingerprint(dataset.frame),
            "feature_schema_hash": feature_schema_fingerprint(features),
            "config": asdict(effective_config),
            "config_hash": config_fingerprint(effective_config),
            "report": "report.json",
            "oof": "oof.csv",
            "training_trace": "training_trace.csv",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "status": report["status"],
                "retained_fixed_blends": report["retained_fixed_blends"],
                "blind_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
