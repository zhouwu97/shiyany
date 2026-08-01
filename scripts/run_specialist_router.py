"""E111：在严格时间顺序门控下生成 generator_1 Ramp specialist OOF。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.config import legacy_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, write_json
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.specialist_router import build_ramp_specialist_oof


def _resolve_data_dir(path: Path) -> Path:
    if (path / "Pre_gas.csv").exists():
        return path
    matches = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "Pre_gas.csv").exists()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"无法解析官方数据目录: {path}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 E111 Ramp specialist gate")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--champion-column", required=True)
    parser.add_argument("--specialist-column", required=True)
    parser.add_argument("--coverage", type=float, default=0.20)
    parser.add_argument("--blend-weight", type=float, default=0.50)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = legacy_forecast_config()
    data_dir = _resolve_data_dir(args.data_dir)
    dataset = align_tables(data_dir, config.feature.frequency)
    prices = sorted(data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    features = build_causal_features(dataset.frame, config.feature, price)
    rows = pd.read_csv(args.oof, parse_dates=["origin_time"])
    routed, report = build_ramp_specialist_oof(
        rows,
        features,
        champion_column=args.champion_column,
        specialist_column=args.specialist_column,
        coverage=args.coverage,
        blend_weight=args.blend_weight,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    routed.to_csv(args.run_dir / "oof.csv", index=False, encoding="utf-8")
    write_json(args.run_dir / "report.json", {"experiment_id": "E111", **report})
    finalize_run(
        args.run_dir,
        {
            "run_type": "experiment",
            "stage": "E111",
            "scope": "development",
            "is_smoke": False,
            "candidate": "ramp_specialist",
            "pooled_mape": report["overall"]["pooled_mape"],
            "report": "report.json",
            "oof": "oof.csv",
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
