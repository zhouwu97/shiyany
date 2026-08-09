"""在严格 OOF 上生成 generator_1 的 Ramp Error Atlas。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.ramp_atlas import build_ramp_error_atlas


def _read_rows(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline-column", required=True)
    parser.add_argument("--candidate-column", required=True)
    parser.add_argument("--target", default="generator_1")
    parser.add_argument("--scope", choices=("development", "final"), default="development")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input)
    result = build_ramp_error_atlas(
        rows,
        baseline_column=args.baseline_column,
        candidate_column=args.candidate_column,
        target=args.target,
        scope=args.scope,
    )
    run_dir = args.run_dir or new_run_dir("results", "experiment_ramp_error_atlas")
    run_dir.mkdir(parents=True, exist_ok=True)
    result.cells.to_csv(run_dir / "cells.csv", index=False, encoding="utf-8")
    result.table.to_csv(run_dir / "ramp_atlas.csv", index=False, encoding="utf-8")
    write_json(run_dir / "report.json", result.report)
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": "A50_ramp_error_atlas",
            "scope": args.scope,
            "is_smoke": False,
            "candidate": args.candidate_column.removesuffix("_pred"),
            "pooled_mape": float(result.report["overall"]["candidate_mape"]),
            "baseline": args.baseline_column,
            "input": str(args.input.resolve()),
            "report": "report.json",
            "cells": "cells.csv",
            "ramp_atlas": "ramp_atlas.csv",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "scope": args.scope,
                "overall": result.report["overall"],
                "rows_by_ramp_band": result.report["rows_by_ramp_band"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
