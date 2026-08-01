"""训练并保存模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.experiments import new_run_dir, write_json
from gas_forecast.workflow import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="训练煤气发电预测模型")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--version", choices=["v1", "v2", "v25", "v3", "auto", "routed"], default="v1"
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "train_model")
    output = args.output or run_dir / "model.joblib"
    version = args.version
    if version == "auto":
        if args.selection is None:
            parser.error("--version auto 必须同时提供 --selection")
        decision = json.loads(args.selection.read_text(encoding="utf-8"))
        version = decision.get("selected_version", decision.get("selected_candidate"))
    route = None
    if version == "routed":
        if args.selection is None:
            parser.error("--version routed 必须同时提供 --selection")
        decision = json.loads(args.selection.read_text(encoding="utf-8"))
        route = decision.get("final_route") or decision.get("routing", {}).get("final_route")
        if route is None:
            parser.error("选择文件不包含 final_route")
    train_model(args.data_dir, output, version, route=route, n_jobs=args.jobs)
    summary = {
        "version": version,
        "model": str(output),
        "selection": str(args.selection) if args.selection else None,
        "jobs": args.jobs,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"{version} 模型已保存: {output}")


if __name__ == "__main__":
    main()
