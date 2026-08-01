"""比较旧模型、V2/V3 目标路由与稳定 LOFO 目标×步长路由。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.routing import leave_one_fold_out_route
from gas_forecast.selection_competition import choose_competition_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较共享 OOF 表中的竞赛候选")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "compare_candidates")
    output = args.output or run_dir / "routed.csv"
    report = args.report or run_dir / "report.json"
    rows = pd.read_csv(args.input, parse_dates=["origin_time", "train_end"])
    required = ("persistence_pred", "v1_pred", "v2_pred", "v25_pred", "v3_pred")
    missing = [column for column in required if column not in rows]
    if missing:
        raise ValueError(f"OOF 表缺少候选列: {missing}")
    rows["v2_v3_target_pred"] = rows["v3_pred"]
    rows.loc[rows["target"].eq("generator_1"), "v2_v3_target_pred"] = rows.loc[
        rows["target"].eq("generator_1"), "v2_pred"
    ]
    routed, route_report = leave_one_fold_out_route(rows, required)
    candidates = {
        "persistence": "persistence_pred",
        "v1": "v1_pred",
        "v2": "v2_pred",
        "v25": "v25_pred",
        "v3": "v3_pred",
        "v2_v3_target": "v2_v3_target_pred",
        "stable_target_horizon_lofo": "routed_pred",
    }
    selection = choose_competition_candidate(routed, candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    routed.to_csv(output, index=False, encoding="utf-8")
    payload = {"selection": selection, "routing": route_report, "rows": int(len(routed))}
    write_json(report, payload)
    selected = selection["selected_candidate"]
    selected_mape = selection["reports"][selected]["pooled_mape"]
    finalize_run(
        run_dir,
        {
            "run_type": "comparison",
            "stage": "M1",
            "is_smoke": False,
            "pooled_mape": float(selected_mape),
            "candidate": selected,
            "report": str(report.relative_to(run_dir)),
            "prediction": str(output.relative_to(run_dir)),
        },
    )
    print(json.dumps(payload["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
