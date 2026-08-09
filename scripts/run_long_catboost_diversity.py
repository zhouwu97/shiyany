"""在严格 development OOF 上运行 A57a/b 长步长 CatBoost 多样性实验。"""

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
from gas_forecast.long_catboost import build_a57_long_catboost_diversity
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config


def _read_rows(path: Path) -> pd.DataFrame:
    """读取 CSV 或 Parquet OOF 长表并解析预测起点和训练边界。"""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def _load_config(path: Path) -> ForecastConfig:
    """恢复冻结 Champion 配置，拒绝隐式改写训练边界。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config 必须是 ForecastConfig JSON 对象")
    return forecast_config_from_dict(payload)


def _price_schedule(data_dir: Path):
    """读取训练目录唯一的已知未来价格表；缺失时显式不使用价格。"""

    paths = sorted(data_dir.glob("*price*.xlsx"))
    if len(paths) > 1:
        raise ValueError(f"A57 发现多个 price 文件: {paths}")
    return load_price_schedule(paths[0]) if paths else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-column", default="rich_gas_blend_30_pred")
    parser.add_argument("--a51-column", default="rich_short00_long100_pred")
    parser.add_argument("--run-dir", type=Path)
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
    features = build_causal_features(
        dataset.frame,
        effective_config.feature,
        _price_schedule(args.data_dir),
    )
    result = build_a57_long_catboost_diversity(
        dataset.frame,
        features,
        _read_rows(args.input),
        baseline_column=args.baseline_column,
        a51_column=args.a51_column,
    )
    run_dir = args.run_dir or new_run_dir("results", "experiment_a57_long_catboost_diversity")
    run_dir.mkdir(parents=True, exist_ok=True)
    report = dict(result.report)
    report["effective_feature_config"] = asdict(effective_config)
    report["input"] = str(args.input.resolve())
    result.rows.to_csv(run_dir / "oof.csv", index=False, encoding="utf-8")
    result.training_trace.to_csv(run_dir / "training_trace.csv", index=False, encoding="utf-8")
    result.residual_correlation.to_csv(
        run_dir / "residual_correlation.csv", index=False, encoding="utf-8"
    )
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", asdict(effective_config))
    candidate_metrics = report["models"]
    best_reported_pooled_mape = min(
        float(metrics["pooled_mape"])
        for metrics in candidate_metrics.values()
    )
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": "A57_long_horizon_catboost_diversity",
            "scope": "development",
            "is_smoke": False,
            "formal_candidate": False,
            "blind_used": False,
            "pooled_mape": best_reported_pooled_mape,
            "best_reported_pooled_mape": best_reported_pooled_mape,
            "baseline": args.baseline_column,
            "a51_parent": args.a51_column,
            "input": str(args.input.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "config": asdict(effective_config),
            "config_file": "config.json",
            "data_hash": dataframe_fingerprint(dataset.frame),
            "feature_schema_hash": feature_schema_fingerprint(features),
            "config_hash": config_fingerprint(effective_config),
            "dataset_hash": dataframe_fingerprint(dataset.frame),
            "feature_hash": feature_schema_fingerprint(features),
            "report": "report.json",
            "oof": "oof.csv",
            "training_trace": "training_trace.csv",
            "residual_correlation": "residual_correlation.csv",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "retained_fixed_blends": report["retained_fixed_blends"],
                "feature_column_count": report["feature_column_count"],
                "blind_used": report["blind_used"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
