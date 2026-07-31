"""运行前向滚动验证。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.validation import backtest_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行短周期预测滚动验证")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--version", choices=["v1", "v2", "v25", "v3"], default="v1")
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/raw/backtest_v1.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ForecastConfig()
    dataset = align_tables(args.data_dir, config.feature.frequency)
    price_paths = sorted(args.data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(price_paths[0]) if price_paths else None
    features = build_causal_features(dataset.frame, config.feature, price)
    result = backtest_model(
        dataset.frame, features, args.version, config, max_folds=args.max_folds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
