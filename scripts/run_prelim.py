"""按训练期选择结果执行初赛训练、预测、校验与打包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("submissions/final"))
    parser.add_argument(
        "--archive", type=Path, default=Path("submissions/teamname_gas_predict_prelim.zip")
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("results/raw/final_pipeline.json")
    )
    args = parser.parse_args()

    decision = json.loads(args.selection.read_text(encoding="utf-8"))
    version = str(decision["selected_version"])
    train_model(args.train_dir, args.model, version)
    features, predictions = predict_rolling(args.train_dir, args.test_dir, args.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "input.csv"
    result_path = args.output_dir / "s_result.csv"
    features.reset_index().to_csv(input_path, index=False, encoding="utf-8")
    result_frame = predictions.reset_index()
    result_frame.to_csv(result_path, index=False, encoding="utf-8")
    validation = validate_submission_frame(result_frame)
    archive = package_submission(result_path, args.archive)
    legacy_json = export_legacy_json(result_path, args.output_dir / "result_legacy.json")

    payload = {
        "selected_version": version,
        "selection_file": str(args.selection),
        "model": str(args.model),
        "input_csv": str(input_path),
        "result_csv": str(result_path),
        "validation": validation,
        "archive": archive,
        "legacy_json": legacy_json,
        "test_labels_used": False,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
