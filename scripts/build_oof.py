"""为旧 V1/V2/V2.5/V3 生成共享外层折逐行 OOF。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.oof import SUPPORTED_LEGACY_MODELS, build_legacy_oof, write_oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成旧版本逐行 OOF 长表")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--versions", nargs="+", choices=SUPPORTED_LEGACY_MODELS, default=SUPPORTED_LEGACY_MODELS
    )
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "legacy_oof")
    output = args.output or run_dir / "oof.csv"
    report = args.report or run_dir / "report.json"
    checkpoint_dir = run_dir / "checkpoints"
    config = ForecastConfig()
    dataset = align_tables(args.data_dir, config.feature.frequency)
    prices = sorted(args.data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    # M1 只复现旧 V1/V2/V2.5/V3 报告；增强特征由 run_experiment 单独登记。
    legacy_feature = replace(
        config.feature,
        lags=(1, 2, 4, 8, 16, 32, 96),
        enable_anomaly_features=False,
        enable_physical_features=False,
        enable_long_cycle_features=False,
    )
    features = build_causal_features(dataset.frame, legacy_feature, price)
    result = build_legacy_oof(
        dataset.frame,
        features,
        versions=args.versions,
        config=config,
        max_folds=args.max_folds,
        n_jobs=args.jobs,
        checkpoint_dir=checkpoint_dir,
    )
    write_oof(result, output, report)
    write_json(report, result.report)
    finalize_run(
        run_dir,
        {
            "run_type": "oof",
            "stage": "M1",
            "is_smoke": args.max_folds is not None and args.max_folds < 20,
            "outer_folds": len(result.report["folds"]),
            "pooled_mape": float(result.report.get("pooled_mape", 0.0)),
            "report": str(report.relative_to(run_dir)),
            "oof": str(output.relative_to(run_dir)),
        },
    )
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
