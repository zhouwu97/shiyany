"""在严格 development OOF 上运行 A53 Perfect Ramp Router oracle 诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.ramp_router import build_a53_oracle_ramp_router


def _read_rows(path: Path) -> pd.DataFrame:
    """读取 CSV 或 Parquet OOF 长表并解析预测起点。"""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline-column", default="rich_gas_blend_30_pred")
    parser.add_argument("--specialist-column", default="rich_g1_long_blend_30_pred")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_a53_oracle_ramp_router(
        _read_rows(args.input),
        baseline_column=args.baseline_column,
        specialist_column=args.specialist_column,
    )
    run_dir = args.run_dir or new_run_dir("results", "experiment_a53_oracle_ramp_router")
    run_dir.mkdir(parents=True, exist_ok=True)
    result.rows.to_csv(run_dir / "oof.csv", index=False, encoding="utf-8")
    result.bucket_table.to_csv(run_dir / "ramp_buckets.csv", index=False, encoding="utf-8")
    write_json(run_dir / "report.json", result.report)
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": "A53_perfect_true_ramp_router",
            "scope": "development",
            "is_smoke": False,
            "formal_candidate": False,
            "oracle_only": True,
            "actual_ramp_used": True,
            "deployable": False,
            "pooled_mape": float(result.report["comparison"]["pooled_mape"]),
            "baseline": args.baseline_column,
            "specialist": args.specialist_column,
            "input": str(args.input.resolve()),
            "report": "report.json",
            "oof": "oof.csv",
            "ramp_buckets": "ramp_buckets.csv",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "oracle_headroom": result.report["oracle_headroom"],
                "formal_candidate": result.report["formal_candidate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
