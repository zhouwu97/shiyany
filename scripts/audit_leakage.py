"""执行至少 50 个起点、5 种未来扰动的因果审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.leakage import audit_future_perturbations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行多起点未来扰动审计")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--origins", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "leakage_audit")
    output = args.output or run_dir / "report.json"
    config = ForecastConfig()
    dataset = align_tables(args.data_dir, config.feature.frequency)
    prices = sorted(args.data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None

    def builder(frame):
        return build_causal_features(frame, config.feature, price)

    predictor = None
    if args.model:
        model = joblib.load(args.model)

        def predictor(features, current):
            return model.predict(features, current.loc[:, list(model.config.targets)])
    result = audit_future_perturbations(
        dataset.frame,
        builder,
        predictor=predictor,
        origins=args.origins,
        n_jobs=args.jobs,
    )
    write_json(output, result)
    finalize_run(
        run_dir,
        {
            "run_type": "audit",
            "stage": "leakage",
            "passed": bool(result["passed"]),
            "report": str(output.relative_to(run_dir)),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
