"""训练并保存模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.workflow import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="训练煤气发电预测模型")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--version",
        choices=[
            "v1",
            "v2",
            "v25",
            "v3",
            "horizon_ridge",
            "generator1_horizon",
            "generator1_catboost",
            "generator1_lgb",
            "generator1_state_expert",
            "generator1_incremental",
            "generator1_direct_incremental",
            "generator1_path",
            "auto",
            "routed",
        ],
        default="v1",
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="冻结配置 JSON，或 run_research_experiment 生成的 report.json",
    )
    parser.add_argument(
        "--candidate-name",
        help="当 --config 指向研究 report.json 时，指定冻结的候选名称",
    )
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
    training_config: ForecastConfig | None = None
    if args.config is not None:
        config_payload = json.loads(args.config.read_text(encoding="utf-8"))
        if "feature" in config_payload:
            raw_config = config_payload
        else:
            candidates = config_payload.get("candidates", {})
            if args.candidate_name is None:
                parser.error("研究 report.json 必须同时提供 --candidate-name")
            candidate = candidates.get(args.candidate_name)
            if not isinstance(candidate, dict) or "config" not in candidate:
                parser.error("研究报告中没有指定的 candidate config")
            raw_config = candidate["config"]
        training_config = forecast_config_from_dict(raw_config)
    train_model(
        args.data_dir,
        output,
        version,
        route=route,
        n_jobs=args.jobs,
        config=training_config,
    )
    summary = {
        "version": version,
        "model": str(output),
        "selection": str(args.selection) if args.selection else None,
        "jobs": args.jobs,
        "config": str(args.config) if args.config else None,
        "candidate_name": args.candidate_name,
    }
    write_json(run_dir / "summary.json", summary)
    finalize_run(
        run_dir,
        {
            "run_type": "training",
            "stage": version,
            "is_smoke": False,
            "model": str(output.relative_to(run_dir)),
            "summary": "summary.json",
        },
    )
    print(f"{version} 模型已保存: {output}")


if __name__ == "__main__":
    main()
