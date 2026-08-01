"""按共享 OOF 键合并多个独立实验并直接比较候选。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.experiments import new_run_dir, write_json
from gas_forecast.selection_competition import choose_competition_candidate


KEYS = ("fold", "origin_time", "target", "horizon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并并比较多个共享外层折 OOF 实验")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--prefixes", nargs="+")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefixes = args.prefixes or [f"experiment_{index + 1}" for index in range(len(args.inputs))]
    if len(prefixes) != len(args.inputs):
        raise ValueError("--prefixes 数量必须与 --inputs 相同")
    merged: pd.DataFrame | None = None
    candidates: dict[str, str] = {}
    for path, prefix in zip(args.inputs, prefixes):
        rows = pd.read_csv(path, parse_dates=["origin_time"])
        required = set(KEYS).union({"actual"})
        missing = sorted(required.difference(rows.columns))
        if missing:
            raise ValueError(f"{path} 缺少 OOF 键: {missing}")
        prediction_columns = [column for column in rows if column.endswith("_pred")]
        renamed = {
            column: f"{prefix}_{column}" for column in prediction_columns
        }
        part = rows.loc[:, [*KEYS, "actual", *prediction_columns]].rename(columns=renamed)
        for original, renamed_column in renamed.items():
            candidates[f"{prefix}_{original.removesuffix('_pred')}"] = renamed_column
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=list(KEYS), how="inner", suffixes=("", "_right"))
            if not np.allclose(
                merged["actual"], merged["actual_right"], equal_nan=True
            ):
                raise ValueError(f"{path} 与前序实验真实值不一致")
            merged = merged.drop(columns="actual_right")
    if merged is None or merged.empty:
        raise ValueError("实验没有共同 OOF 行")
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "compare_experiments")
    merged.to_csv(run_dir / "merged_oof.csv", index=False, encoding="utf-8")
    result = choose_competition_candidate(merged, candidates)
    payload = {
        "inputs": [str(path) for path in args.inputs],
        "shared_rows": int(len(merged)),
        "selection": result,
    }
    write_json(run_dir / "report.json", payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
