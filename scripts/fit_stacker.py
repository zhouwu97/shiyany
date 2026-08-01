"""从基础分支 OOF 表冻结逐目标逐步长 simplex 权重。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.experiments import new_run_dir, write_json
from gas_forecast.stacking import fit_simplex_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拟合受约束 OOF simplex")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--regularization", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "fit_stacker")
    output = args.output or run_dir / "weights.json"
    rows = pd.read_csv(args.input)
    weights: dict[str, object] = {}
    for target, target_rows in rows.groupby("target", sort=True):
        horizons = sorted(target_rows["horizon"].unique())
        grouped = [target_rows.loc[target_rows["horizon"].eq(horizon)] for horizon in horizons]
        length = min(len(part) for part in grouped)
        predictions = np.stack(
            [part.iloc[:length][args.columns].to_numpy() for part in grouped], axis=2
        )
        actual = np.column_stack([part.iloc[:length]["actual"].to_numpy() for part in grouped])
        state = fit_simplex_state(
            predictions,
            actual,
            tuple(args.columns),
            regularization=args.regularization,
        )
        weights[str(target)] = {
            "horizons": [int(value) for value in horizons],
            "branches": list(state.branch_names),
            "target_weights": state.target_weights.tolist(),
            "horizon_weights": state.horizon_weights.tolist(),
            "regularized_weights": state.regularized_weights.tolist(),
        }
    payload = {"weights": weights, "reporting_policy": "weights_only_no_in_sample_score"}
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
