"""为已完成的 C0 OOF 报告补齐 runner-up 稳定性诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.experiments import finalize_run, write_json
from gas_forecast.scoring import absolute_percentage_error, block_bootstrap_improvement_probability


def main() -> None:
    parser = argparse.ArgumentParser(description="补齐 C0 稳定性评分卡")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.run_dir / "report.json"
    rows_path = args.run_dir / "oof_with_routes.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = pd.read_csv(rows_path, parse_dates=["origin_time"])
    selection = report["selection"]
    selected = str(selection["selected_candidate"])
    runner = str(selection.get("runner_up", selected))
    candidate_columns = {
        str(name): f"{name}_pred"
        for name in report["candidate_reports"]
        if f"{name}_pred" in rows
    }
    candidate_columns.update(
        {
            "lofo_raw": "lofo_raw_pred",
            "lofo_reconciled": "lofo_reconciled_pred",
        }
    )
    stability: dict[str, object] = {"selected": selected, "runner_up": runner}
    if runner != selected:
        selected_column = candidate_columns[selected]
        runner_column = candidate_columns[runner]
        stability["day_block_bootstrap"] = block_bootstrap_improvement_probability(
            rows, selected_column, runner_column, block="day"
        )
        stability["fold_block_bootstrap"] = block_bootstrap_improvement_probability(
            rows, selected_column, runner_column, block="fold"
        )
        development = rows.loc[rows["fold"].ne("blind")].copy()
        selected_ape = absolute_percentage_error(
            development["actual"], development[selected_column]
        )
        runner_ape = absolute_percentage_error(
            development["actual"], development[runner_column]
        )
        fold_scores = (
            pd.DataFrame(
                {"fold": development["fold"], "difference": selected_ape - runner_ape}
            )
            .groupby("fold", sort=True)["difference"]
            .mean()
        )
        stability.update(
            {
                "development_fold_win_rate": float((fold_scores < 0.0).mean()),
                "development_worst_fold_regression": float(fold_scores.max()),
                "development_recent_folds": {
                    str(key): float(value) for key, value in fold_scores.tail(5).items()
                },
            }
        )
    report["stability"] = stability
    write_json(report_path, report)
    write_json(args.run_dir / "selection.json", report["selection"])
    finalize_run(
        args.run_dir,
        {
            "stability": stability,
            "report": "report.json",
            "oof": "oof_with_routes.csv",
        },
    )
    print(json.dumps(stability, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
