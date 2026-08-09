"""训练未来行重建模型，生成独立候选与平台提交包。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gas_forecast.data import align_tables
from gas_forecast.future_reconstruction import FutureRowReconstructionForecaster
from gas_forecast.submission import package_submission, validate_submission_frame
from gas_forecast.submission_quality import COMPETITION_QUALITY_POLICY, prepare_submission_input
from gas_forecast.workflow import predict_rolling


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_score(path: Path, prediction: pd.DataFrame) -> dict[str, object]:
    """在预测和模型写盘后读取外部参考，仅用于最终保留评估。"""

    with zipfile.ZipFile(path) as archive:
        reference = pd.read_csv(io.BytesIO(archive.read("s_result.csv")))
    reference["datetime"] = pd.to_datetime(reference["datetime"])
    reference = reference.set_index("datetime")
    if not reference.index.equals(prediction.index):
        raise ValueError("参考答案与候选预测时间轴不一致")
    if set(reference.columns) != set(prediction.columns):
        raise ValueError("参考答案与候选预测字段不一致")
    reference = reference.reindex(columns=prediction.columns)
    actual = reference.to_numpy(dtype=float)
    predicted = prediction.to_numpy(dtype=float)
    errors = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-6)
    by_target = {}
    for target in ("generator_1", "generator_all"):
        positions = [
            position
            for position, column in enumerate(prediction.columns)
            if column.startswith(f"{target}_")
        ]
        by_target[target] = float(errors[:, positions].mean())
    by_horizon = {}
    for minutes in range(15, 121, 15):
        positions = [
            position
            for position, column in enumerate(prediction.columns)
            if f"_t+{minutes}_" in column
        ]
        by_horizon[f"t+{minutes}"] = float(errors[:, positions].mean())
    return {
        "pooled_mape": float(errors.mean()),
        "score_100_one_minus_mape": float(100.0 * (1.0 - errors.mean())),
        "by_target": by_target,
        "by_horizon": by_horizon,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--delivery-dir", type=Path, default=Path("提交这个_训练优化"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--team-name", default="咕咕嘎嘎")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model = joblib.load(args.base_model)
    config = base_model.config
    train_frame = align_tables(args.train_dir, config.feature.frequency).frame
    scoring_frame = align_tables(args.test_dir, config.feature.frequency).frame

    # 外部参考在模型训练与候选预测完成前不读取。
    model = FutureRowReconstructionForecaster(config).fit(train_frame)
    input_features, base_predictions = predict_rolling(
        args.train_dir,
        args.test_dir,
        args.base_model,
    )
    predictions, reconstruction = model.predict(scoring_frame, base_predictions)
    validation = validate_submission_frame(predictions.reset_index(), config)

    submission_dir = args.run_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.run_dir / "model.joblib"
    base_copy = args.run_dir / "base_model.joblib"
    input_path = submission_dir / "input.csv"
    result_path = submission_dir / "s_result.csv"
    archive_path = args.run_dir / "submission.zip"
    report_path = args.run_dir / "report.json"

    joblib.dump(model, model_path)
    shutil.copy2(args.base_model, base_copy)
    export_features = input_features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    quality_input, quality_report = prepare_submission_input(
        export_features.reset_index(),
        COMPETITION_QUALITY_POLICY,
    )
    quality_input.to_csv(input_path, index=False, encoding="utf-8")
    predictions.reset_index().to_csv(result_path, index=False, encoding="utf-8")
    package_submission(
        input_path,
        result_path,
        archive_path,
        quality_policy=COMPETITION_QUALITY_POLICY,
    )

    report: dict[str, object] = {
        "candidate": model.version,
        "training": model.training_report(),
        "reconstruction": reconstruction,
        "validation": validation,
        "quality": quality_report,
        "reference_used_for_training": False,
        "base_model": str(args.base_model.resolve()),
    }
    if args.reference is not None:
        report["reference_evaluation"] = _reference_score(args.reference, predictions)
        report["reference"] = str(args.reference.resolve())
    report["sha256"] = {
        "model": _sha256(model_path),
        "base_model": _sha256(base_copy),
        "result": _sha256(result_path),
        "submission": _sha256(archive_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.team_name.strip() or any(
        character in args.team_name for character in '<>:"/\\|?*'
    ):
        raise ValueError("队伍名称为空或含 Windows 文件名非法字符")
    args.delivery_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_path, args.delivery_dir / f"{args.team_name}_gas_predict_prelim.zip")
    shutil.copy2(input_path, args.delivery_dir / "input.csv")
    shutil.copy2(result_path, args.delivery_dir / "s_result.csv")
    shutil.copy2(report_path, args.delivery_dir / "report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
