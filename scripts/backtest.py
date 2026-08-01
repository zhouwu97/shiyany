"""运行前向滚动验证。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.validation import backtest_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行短周期预测滚动验证")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--version", choices=["v1", "v2", "v25", "v3"], default="v1")
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", f"backtest_{args.version}")
    output = args.output or run_dir / "report.json"
    config = ForecastConfig()
    dataset = align_tables(args.data_dir, config.feature.frequency)
    price_paths = sorted(args.data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(price_paths[0]) if price_paths else None
    features = build_causal_features(dataset.frame, config.feature, price)
    result = backtest_model(
        dataset.frame,
        features,
        args.version,
        config,
        max_folds=args.max_folds,
        n_jobs=args.jobs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finalize_run(
        run_dir,
        {
            "run_type": "oof",
            "stage": args.version,
            "is_smoke": args.max_folds is not None and args.max_folds < 20,
            "outer_folds": len(result.get("folds", [])),
            "report": str(output.relative_to(run_dir)),
        },
    )
    print(json.dumps({key: value for key, value in result.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
