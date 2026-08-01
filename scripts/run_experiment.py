"""运行 cross-fitting、OOF 残差、融合、协调及可选增强实验。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.experiments import (
    build_experimental_oof,
    finalize_run,
    new_run_dir,
    register_experiment,
)
from gas_forecast.features import build_causal_features, load_price_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行新训练体系外层 OOF 实验")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        help="每个外层折内树模型线程数；默认按逻辑核心数和实际并行折数自动分配",
    )
    parser.add_argument("--include-catboost", action="store_true")
    parser.add_argument("--include-gas-trajectory", action="store_true")
    parser.add_argument(
        "--lgb-objective", choices=["regression_l1", "huber", "fair"], default="regression_l1"
    )
    parser.add_argument("--no-mape-weights", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", f"experiment_{args.experiment_id}")
    output = args.output or run_dir / "oof.csv"
    report = args.report or run_dir / "report.json"
    if args.jobs < 1:
        raise ValueError("--jobs 必须大于等于 1")
    active_workers = min(args.jobs, args.max_folds or args.jobs)
    threads_per_worker = args.threads_per_worker or max(
        1, (os.cpu_count() or 1) // active_workers
    )
    if threads_per_worker < 1:
        raise ValueError("--threads-per-worker 必须大于等于 1")
    base_config = ForecastConfig()
    config = replace(
        base_config,
        model=replace(
            base_config.model,
            lgb_objective=args.lgb_objective,
            lgb_use_mape_weights=not args.no_mape_weights,
            lgb_use_early_stopping=not args.no_early_stopping,
            tree_threads_per_worker=threads_per_worker,
            inner_folds=args.inner_folds,
        ),
    )
    dataset = align_tables(args.data_dir, config.feature.frequency)
    prices = sorted(args.data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    features = build_causal_features(dataset.frame, config.feature, price)
    rows, duration = build_experimental_oof(
        dataset.frame,
        features,
        config=config,
        max_folds=args.max_folds,
        n_jobs=args.jobs,
        include_catboost=args.include_catboost,
        include_gas_trajectory=args.include_gas_trajectory,
        checkpoint_dir=run_dir / "checkpoints",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False, encoding="utf-8")
    payload = register_experiment(
        output,
        report,
        experiment_id=args.experiment_id,
        config=config,
        training_command=" ".join(sys.argv),
        training_time=duration,
    )
    record = payload["record"]
    pooled = record.get("pooled_mape", {})
    best_mape = min(pooled.values()) if pooled else None
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": args.experiment_id,
            "is_smoke": args.max_folds is not None and args.max_folds < 20,
            "outer_folds": record.get("outer_folds"),
            "pooled_mape": best_mape,
            "best_candidate": payload.get("best_candidate"),
            "report": str(report.relative_to(run_dir)),
            "oof": str(output.relative_to(run_dir)),
        },
    )
    print(json.dumps(payload["record"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
