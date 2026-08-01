"""E112/E140：对已冻结 OOF 候选执行低相关融合和时间顺序 stack。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.blending import residual_correlation, time_ordered_stack_oof, weighted_blend
from gas_forecast.experiments import finalize_run, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 OOF 融合实验")
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float)
    parser.add_argument("--active-column")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = pd.read_csv(args.oof, parse_dates=["origin_time"])
    columns = tuple(args.columns)
    weights = args.weights or [1.0] * len(columns)
    if len(weights) != len(columns):
        raise ValueError("--weights 数量必须与 --columns 一致")
    blended, blend_report = weighted_blend(
        rows,
        columns,
        weights,
        output_column="blend_pred",
        active_column=args.active_column,
    )
    stacked, stack_report = time_ordered_stack_oof(blended, columns, output_column="stack_pred")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stacked.to_csv(args.run_dir / "oof.csv", index=False, encoding="utf-8")
    report = {
        "experiment_id": "E112_E140",
        "columns": list(columns),
        "residual_correlation": residual_correlation(rows, columns).to_dict(),
        "blend": blend_report,
        "time_ordered_stack": stack_report,
        "blind_used_for_selection": False,
    }
    write_json(args.run_dir / "report.json", report)
    finalize_run(
        args.run_dir,
        {
            "run_type": "experiment",
            "stage": "E112_E140",
            "scope": "development",
            "is_smoke": False,
            "candidate": "time_ordered_stack",
            "pooled_mape": stack_report["score"]["pooled_mape"],
            "report": "report.json",
            "oof": "oof.csv",
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
