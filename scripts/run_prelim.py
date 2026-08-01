"""按训练期选择结果执行初赛训练、预测、校验与打包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.experiments import finalize_run, new_run_dir
from gas_forecast.submission import (
    export_legacy_json,
    package_submission,
    validate_submission_frame,
)
from gas_forecast.workflow import predict_rolling, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="执行初赛端到端离线流水线")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "prelim_pipeline")
    model_path = args.model or run_dir / "model.joblib"
    output_dir = args.output_dir or run_dir / "submission"
    archive_path = args.archive or run_dir / "submission.zip"
    metrics_path = args.metrics or run_dir / "summary.json"

    decision = json.loads(args.selection.read_text(encoding="utf-8"))
    version = str(decision["selected_version"])
    train_model(args.train_dir, model_path, version)
    features, predictions = predict_rolling(args.train_dir, args.test_dir, model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.csv"
    result_path = output_dir / "s_result.csv"
    features.reset_index().to_csv(input_path, index=False, encoding="utf-8")
    result_frame = predictions.reset_index()
    result_frame.to_csv(result_path, index=False, encoding="utf-8")
    validation = validate_submission_frame(result_frame)
    archive = package_submission(result_path, archive_path)
    legacy_json = export_legacy_json(result_path, output_dir / "result_legacy.json")

    payload = {
        "selected_version": version,
        "selection_file": str(args.selection),
        "run_dir": str(run_dir),
        "model": str(model_path),
        "input_csv": str(input_path),
        "result_csv": str(result_path),
        "validation": validation,
        "archive": archive,
        "legacy_json": legacy_json,
        "test_labels_used": False,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finalize_run(
        run_dir,
        {
            "run_type": "training",
            "stage": "prelim",
            "is_smoke": False,
            "submission_valid": bool(validation.get("valid", True)),
            "model": str(model_path.relative_to(run_dir)),
            "result": str(result_path.relative_to(run_dir)),
            "submission": str(archive_path.relative_to(run_dir)),
            "summary": str(metrics_path.relative_to(run_dir)),
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
