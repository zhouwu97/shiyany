"""把冻结的 R75 + 20% LGB 候选组装为可审计的初赛生产运行。"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gas_forecast.aggressive import evaluate_candidate, project_long_candidate
from gas_forecast.aggressive_model import AggressiveR75LGBForecaster
from gas_forecast.data import align_tables
from gas_forecast.experiments import build_fingerprints, finalize_run, write_json
from gas_forecast.submission import package_submission, validate_submission_frame
from gas_forecast.submission_quality import (
    COMPETITION_QUALITY_POLICY,
    prepare_submission_input,
)
from gas_forecast.workflow import predict_rolling


def _resolve_data_dir(path: Path, marker: str) -> Path:
    if (path / marker).exists():
        return path
    matches = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / marker).exists()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"无法解析数据目录: {path}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-model", type=Path, required=True)
    parser.add_argument("--e21-model", type=Path, required=True)
    parser.add_argument("--oof-confirmation", type=Path, required=True)
    parser.add_argument("--blind-report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate = "aggressive_r75_lgb20"
    prediction_column = f"{candidate}_pred"
    train_dir = _resolve_data_dir(args.data_dir, "Pre_gas.csv")
    test_dir = _resolve_data_dir(args.test_dir, "Pre_test_gas.csv")
    c0_model = joblib.load(args.c0_model)
    e21_model = joblib.load(args.e21_model)
    model = AggressiveR75LGBForecaster(c0_model, e21_model)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.run_dir / "model.joblib"
    input_path = args.run_dir / "submission" / "input.csv"
    result_path = args.run_dir / "submission" / "s_result.csv"
    archive_path = args.run_dir / "submission.zip"
    joblib.dump(model, model_path)

    input_features, predictions = predict_rolling(train_dir, test_dir, model_path)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_missing_cells = int(input_features.isna().sum().sum())
    input_export = input_features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    quality_input, quality_report = prepare_submission_input(
        input_export.reset_index(),
        COMPETITION_QUALITY_POLICY,
    )
    quality_input.to_csv(input_path, index=False, encoding="utf-8")
    result_frame = predictions.reset_index()
    result_frame.to_csv(result_path, index=False, encoding="utf-8")
    validation = validate_submission_frame(result_frame, model.config)
    package_submission(
        input_path,
        result_path,
        archive_path,
        quality_policy=COMPETITION_QUALITY_POLICY,
    )

    confirmed = pd.read_parquet(args.oof_confirmation)
    raw_column = "confirmed_blend_lgb_residual_pred_20"
    projected = project_long_candidate(
        confirmed,
        raw_column,
        output_column=prediction_column,
    )
    oof_path = args.run_dir / "oof.csv"
    projected.to_csv(oof_path, index=False, encoding="utf-8")
    evaluation = evaluate_candidate(projected, prediction_column, baseline_column="c0_pred")
    report_path = args.run_dir / "report.json"
    write_json(
        report_path,
        {
            "candidate": candidate,
            "strict_label_purge": True,
            "route": "R75",
            "lgb_weight": 0.20,
            "capacity_projection": True,
            "evaluation": evaluation,
            "blind_confirmation": "blind_confirmation.json",
        },
    )
    shutil.copy2(args.blind_report, args.run_dir / "blind_confirmation.json")
    write_json(args.run_dir / "config.json", asdict(model.config))

    dataset = align_tables(train_dir, model.config.feature.frequency)
    fingerprints = build_fingerprints(
        config=model.config,
        dataset=dataset.frame,
        model_params={
            "candidate": candidate,
            "route": "R75",
            "crossing_minutes": 75,
            "lgb_weight": 0.20,
            "c0_model": str(args.c0_model.resolve()),
            "e21_model": str(args.e21_model.resolve()),
        },
    )
    manifest = finalize_run(
        args.run_dir,
        {
            "run_type": "training",
            "stage": "aggressive_prelim_candidate",
            "is_smoke": False,
            "candidate": candidate,
            "pooled_mape": float(evaluation["candidate"]["pooled_mape"]),
            "config": asdict(model.config),
            "best_files": {
                "model": "model.joblib",
                "input": "submission/input.csv",
                "result": "submission/s_result.csv",
                "submission": "submission.zip",
                "report": "report.json",
            },
            "leakage_passed": False,
            "tests_passed": False,
            "submission_valid": True,
            "oof": "oof.csv",
            "report": "report.json",
            "blind_confirmation": "blind_confirmation.json",
            "input_export_forward_filled_cells": input_missing_cells,
            "submission_quality": quality_report,
            "validation": validation,
            **fingerprints,
        },
    )
    print(
        json.dumps(
            {
                "candidate": candidate,
                "pooled_mape": manifest["pooled_mape"],
                "validation": validation,
                "run_dir": str(args.run_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
