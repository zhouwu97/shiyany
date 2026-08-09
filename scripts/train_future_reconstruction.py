"""训练未来行重建 Oracle，仅生成 ``results/oracle`` 下的诊断产物。

该脚本不是正式训练或提交入口。运行必须显式声明
``--allow-oracle-research``，并且输出目录只能位于 ``results/oracle/<name>``。
"""

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
from gas_forecast.submission import validate_submission_frame
from gas_forecast.workflow import predict_rolling


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _oracle_run_dir(path: Path) -> Path:
    """校验 Oracle 产物只能写入 ``results/oracle/<name>``。"""

    root = (Path.cwd() / "results" / "oracle").resolve()
    candidate = path.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "未来行重建 Oracle 只能写入 results/oracle/<name>，禁止正式结果目录"
        ) from exc
    if len(relative.parts) != 1:
        raise ValueError("Oracle 必须使用 results/oracle/<name> 形式的独立运行目录")
    name = relative.parts[0]
    forbidden = {"best", "submission", "submissions", "formal", "正式"}
    if (
        name.casefold() in forbidden
        or "submission" in name.casefold()
        or name.startswith("提交这个")
    ):
        raise ValueError("Oracle 输出目录不能是 best、submission 或正式提交目录")
    if candidate.exists():
        raise ValueError("Oracle 运行目录必须是不存在的新目录，不得覆盖既有结果")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


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
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--allow-oracle-research",
        action="store_true",
        required=True,
        help="明确确认该运行仅用于非因果 Oracle 研究",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_oracle_research:
        raise PermissionError("必须显式传入 --allow-oracle-research 才能运行 Oracle")
    run_dir = _oracle_run_dir(args.run_dir)
    base_model = joblib.load(args.base_model)
    if getattr(base_model, "oracle_candidate", False) or getattr(base_model, "causal", True) is False:
        raise ValueError("基础模型本身是非因果 Oracle，不能作为研究重建器的生产基线")
    config = base_model.config
    train_frame = align_tables(args.train_dir, config.feature.frequency).frame
    scoring_frame = align_tables(args.test_dir, config.feature.frequency).frame

    # 外部参考在模型训练与 Oracle 预测完成前不读取，避免进入训练闭环。
    model = FutureRowReconstructionForecaster(config).fit(train_frame)
    input_features, base_predictions = predict_rolling(
        args.train_dir,
        args.test_dir,
        args.base_model,
    )
    predictions, reconstruction = model.predict(scoring_frame, base_predictions)
    validation = validate_submission_frame(predictions.reset_index(), config)

    model_path = run_dir / "model.joblib"
    base_copy = run_dir / "base_model.joblib"
    input_path = run_dir / "oracle_input.csv"
    result_path = run_dir / "oracle_predictions.csv"
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "manifest.json"

    joblib.dump(model, model_path)
    shutil.copy2(args.base_model, base_copy)
    export_features = input_features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    export_features.reset_index().to_csv(input_path, index=False, encoding="utf-8")
    predictions.reset_index().to_csv(result_path, index=False, encoding="utf-8")

    report: dict[str, object] = {
        "candidate": model.version,
        "oracle_candidate": True,
        "oracle_only": True,
        "diagnostic_only": True,
        "causal": False,
        "formal_candidate": False,
        "deployable": False,
        "production_candidate": False,
        "research_only": True,
        "uses_future_rows": True,
        "production_path_allowed": False,
        "selection_allowed": False,
        "weights_allowed": False,
        "thresholds_allowed": False,
        "submission_generated": False,
        "output_root": "results/oracle",
        "oracle_reason": model.oracle_reason,
        "training": model.training_report(),
        "reconstruction": reconstruction,
        "validation": validation,
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
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_type": "oracle_research",
        "status": "completed",
        "candidate": model.version,
        "oracle_candidate": True,
        "oracle_only": True,
        "diagnostic_only": True,
        "causal": False,
        "formal_candidate": False,
        "deployable": False,
        "production_candidate": False,
        "research_only": True,
        "uses_future_rows": True,
        "production_path_allowed": False,
        "selection_allowed": False,
        "weights_allowed": False,
        "thresholds_allowed": False,
        "submission_generated": False,
        "best_files": {},
        "report": "report.json",
        "model": "model.joblib",
        "oracle_input": "oracle_input.csv",
        "oracle_predictions": "oracle_predictions.csv",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
