"""将严格 OOF 的 C0 选型重训为可审计的正式候选运行。"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from gas_forecast.config import legacy_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.experiments import build_fingerprints, finalize_run, write_json
from gas_forecast.submission import package_submission, validate_submission_frame
from gas_forecast.workflow import predict_rolling, train_model


def _resolve_data_dir(path: Path, marker: str = "Pre_gas.csv") -> Path:
    if (path / marker).exists():
        return path
    matches = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / marker).exists()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"无法解析数据目录: {path}")
    return matches[0]


def _fixed_target_route() -> dict[str, object]:
    cells = {
        f"{target}|{horizon}": {
            "selected": "v2_pred" if target == "generator_1" else "v3_pred"
        }
        for target in ("generator_1", "generator_all")
        for horizon in range(15, 121, 15)
    }
    return {
        "policy": "fixed_v2_v3_target_route",
        "global": {"selected": "v3_pred"},
        "targets": {
            "generator_1": {"selected": "v2_pred"},
            "generator_all": {"selected": "v3_pred"},
        },
        "cells": cells,
        "post_route_reconciliation": {"enabled": True, "max_generator_rest": 240.0},
    }


def _single_model_route(candidate: str) -> dict[str, object]:
    column = f"{candidate}_pred"
    return {
        "policy": "single_model_route",
        "global": {"selected": column},
        "targets": {},
        "cells": {},
        "post_route_reconciliation": {"enabled": True, "max_generator_rest": 240.0},
    }


def _deployment_route(c0_report: dict[str, object], candidate: str) -> tuple[dict[str, object], str]:
    route_report = c0_report.get("route_report", {})
    if not isinstance(route_report, dict):
        route_report = {}
    if candidate in {"v2_v3_target_raw", "v2_v3_target_reconciled"}:
        return _fixed_target_route(), "fixed_v2_v3_target_reconciled"
    if candidate in {"lofo_raw", "lofo_reconciled"}:
        source = route_report.get("lofo_reconciled")
        if isinstance(source, dict) and isinstance(source.get("final_route"), dict):
            return dict(source["final_route"]), "lofo_reconciled_final_route"
    if candidate in {"persistence", "v1", "v2", "v25", "v3"}:
        return _single_model_route(candidate), f"single_{candidate}"
    raise ValueError(f"C0 候选没有可部署路由: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser(description="重训并打包严格 OOF 的 Clean Champion")
    parser.add_argument("--c0-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs 必须大于等于 1")

    c0_report = json.loads((args.c0_run / "report.json").read_text(encoding="utf-8"))
    candidate = str(c0_report["selected_candidate"])
    route, route_name = _deployment_route(c0_report, candidate)
    train_dir = _resolve_data_dir(args.data_dir)
    test_dir = _resolve_data_dir(args.test_dir, "Pre_test_gas.csv")
    config = legacy_forecast_config()
    dataset = align_tables(train_dir, config.feature.frequency)
    model_path = args.run_dir / "model.joblib"
    result_path = args.run_dir / "submission" / "result.csv"
    archive_path = args.run_dir / "submission.zip"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    train_model(
        train_dir,
        model_path,
        "routed",
        route=route,
        n_jobs=args.jobs,
        config=config,
    )
    _, predictions = predict_rolling(train_dir, test_dir, model_path)
    result_frame = predictions.reset_index()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(result_path, index=False, encoding="utf-8")
    validation = validate_submission_frame(result_frame, config)
    package_submission(result_path, archive_path)
    shutil.copy2(args.c0_run / "oof_with_routes.csv", args.run_dir / "oof.csv")
    shutil.copy2(args.c0_run / "report.json", args.run_dir / "report.json")
    shutil.copy2(args.c0_run / "selection.json", args.run_dir / "selection.json")
    write_json(args.run_dir / "config.json", asdict(config))
    fingerprints = build_fingerprints(
        config=config,
        dataset=dataset.frame,
        model_params={"candidate": candidate, "route_name": route_name, "route": route},
    )
    write_json(
        args.run_dir / "summary.json",
        {
            "candidate": candidate,
            "deployment_route": route_name,
            "model": "model.joblib",
            "oof": "oof.csv",
            "result": "submission/result.csv",
            "submission": "submission.zip",
            "validation": validation,
            "c0_run": str(args.c0_run.resolve()),
            **fingerprints,
        },
    )
    finalize_run(
        args.run_dir,
        {
            "run_type": "training",
            "stage": "C0_formal_candidate",
            "is_smoke": False,
            "candidate": candidate,
            "deployment_route": route_name,
            "pooled_mape": float(c0_report["candidate_reports"][candidate]["pooled_mape"]),
            "config": asdict(config),
            "best_files": {
                "model": "model.joblib",
                "result": "submission/result.csv",
                "submission": "submission.zip",
                "report": "report.json",
                "selection": "selection.json",
            },
            "leakage_passed": False,
            "tests_passed": False,
            "submission_valid": True,
            **fingerprints,
            "summary": "summary.json",
            "oof": "oof.csv",
        },
    )
    print(json.dumps({"candidate": candidate, "route": route_name, "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
