"""从任意逐行 OOF 表直接选择 pooled MAPE 最低候选。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.selection_competition import choose_competition_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 pooled OOF 候选选择")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--columns", nargs="+")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "select_candidate")
    output = args.output or run_dir / "selection.json"
    rows = pd.read_csv(args.input, parse_dates=["origin_time"])
    columns = args.columns or [column for column in rows if column.endswith("_pred")]
    candidates = {column.removesuffix("_pred"): column for column in columns}
    result = choose_competition_candidate(rows, candidates)
    write_json(output, result)
    selected = result["selected_candidate"]
    finalize_run(
        run_dir,
        {
            "run_type": "comparison",
            "stage": "candidate_selection",
            "is_smoke": False,
            "pooled_mape": float(result["reports"][selected]["pooled_mape"]),
            "candidate": selected,
            "report": str(output.relative_to(run_dir)),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
