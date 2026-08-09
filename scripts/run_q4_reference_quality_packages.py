"""生成并审计共享冻结预测的 Q_CAUSAL/Q_REFERENCE A/B 提交包。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from zipfile import ZipFile

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from gas_forecast.data import align_tables  # noqa: E402
from gas_forecast.features import build_causal_features, load_price_schedule  # noqa: E402
from gas_forecast.submission import (  # noqa: E402
    SUBMISSION_MEMBERS,
    package_submission,
    prepare_submission_chain,
    validate_submission_archive,
    validate_submission_input,
)
from gas_forecast.submission_quality import (  # noqa: E402
    COMPETITION_QUALITY_POLICY,
    audit_submission_quality,
    inspect_submission_input_quality,
)
from gas_forecast.workflow import resolve_prediction_feature_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scoring-input",
        type=Path,
        required=True,
        help="正式评分 input.csv，或根目录含 input.csv 的正式 ZIP",
    )
    parser.add_argument(
        "--source-zip",
        type=Path,
        required=True,
        help="已合法提交的冻结预测 ZIP；只读取其中 s_result.csv",
    )
    parser.add_argument("--training-input", type=Path, help="独立训练期 input.csv")
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        help="未提供 --training-input 时，用项目正式特征 API 重建训练 input",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="与 --training-data-dir 配套的冻结模型，用于取得正式特征配置",
    )
    parser.add_argument("--train-end", help="Q_CAUSAL 训练统计截止时间")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    using_csv = args.training_input is not None
    using_rebuild = args.training_data_dir is not None or args.model is not None
    if using_csv == using_rebuild:
        parser.error("必须二选一：--training-input，或 --training-data-dir 与 --model")
    if using_rebuild and (args.training_data_dir is None or args.model is None):
        parser.error("--training-data-dir 与 --model 必须同时提供")
    return args


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_submission_member(archive_path: Path, member: str) -> bytes:
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if names != list(SUBMISSION_MEMBERS):
            raise ValueError(f"正式 ZIP 成员不符合双 CSV 契约: {names}")
        return archive.read(member)


def _materialize_scoring_input(source: Path, destination: Path) -> dict[str, object]:
    if source.suffix.casefold() == ".zip":
        payload = _read_submission_member(source, "input.csv")
        source_kind = "zip_member"
    else:
        payload = source.read_bytes()
        source_kind = "csv"
    destination.write_bytes(payload)
    frame = pd.read_csv(io.BytesIO(payload))
    return {
        "source": str(source),
        "source_kind": source_kind,
        "sha256": _sha256_bytes(payload),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
    }


def _build_training_input(data_dir: Path, model_path: Path, destination: Path) -> dict[str, object]:
    """用冻结模型配置和项目正式特征 API 重建训练期输入。"""

    model = joblib.load(model_path)
    feature_config = resolve_prediction_feature_config(model)
    dataset = align_tables(data_dir, feature_config.frequency)
    price_paths = sorted(data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(price_paths[0]) if price_paths else None
    features = build_causal_features(dataset.frame, feature_config, price)
    training = features.reset_index()
    if training.columns[0] != "datetime":
        raise ValueError("正式特征 API 未生成首列 datetime")
    training.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")
    return {
        "mode": "rebuilt_with_formal_feature_api",
        "data_dir": str(data_dir),
        "model": str(model_path),
        "model_sha256": _sha256_file(model_path),
        "rows": int(len(training)),
        "columns": int(len(training.columns)),
        "start": str(training["datetime"].iloc[0]),
        "end": str(training["datetime"].iloc[-1]),
        "sha256": _sha256_file(destination),
    }


def _materialize_training_input(
    destination: Path,
    *,
    training_input: Path | None,
    training_data_dir: Path | None,
    model_path: Path | None,
) -> dict[str, object]:
    if training_input is not None:
        shutil.copyfile(training_input, destination)
        frame = pd.read_csv(destination)
        return {
            "mode": "provided_training_input",
            "source": str(training_input),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "start": str(frame["datetime"].iloc[0]),
            "end": str(frame["datetime"].iloc[-1]),
            "sha256": _sha256_file(destination),
        }
    if training_data_dir is None or model_path is None:
        raise ValueError("缺少训练 input，且未提供训练数据目录与冻结模型")
    return _build_training_input(training_data_dir, model_path, destination)


def _mechanical_metrics(frame: pd.DataFrame, result: pd.DataFrame) -> dict[str, object]:
    """报告本地可推导量；字段名对应平台细项，但不推断平台计分公式。"""

    quality = inspect_submission_input_quality(frame)
    schema = audit_submission_quality(frame, COMPETITION_QUALITY_POLICY)
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    valid_timestamps = timestamps.dropna()
    intervals = valid_timestamps.diff().dropna()
    interval_violations = int(intervals.ne(pd.Timedelta(minutes=15)).sum())
    duplicate_timestamps = int(timestamps.duplicated().sum())
    invalid_columns = sorted(quality["nonfinite_by_column"])
    result_times = pd.to_datetime(result["datetime"], errors="coerce")
    aligned = bool(
        len(frame) == len(result)
        and timestamps.reset_index(drop=True).equals(result_times.reset_index(drop=True))
    )
    return {
        "score_inference": None,
        "score_inference_reason": "平台细项公式未知，仅报告可机械复核的本地计数与布尔量",
        "miss": {
            "nonfinite_cells": int(quality["nonfinite_cells"]),
            "affected_columns": invalid_columns,
        },
        "dup": {
            "duplicate_timestamps": duplicate_timestamps,
            "duplicate_columns": int(len(quality["duplicate_columns"])),
            "duplicate_column_pairs": quality["duplicate_column_pairs"],
        },
        "out": {
            "iqr_outlier_cells_linear": int(quality["iqr_outlier_cells"]),
            "iqr_outlier_cells_all_methods": int(quality["iqr_outlier_cells_all_methods"]),
            "abs_z_gt_3_cells": int(quality["zscore_outlier_cells"]),
        },
        "intv": {
            "invalid_timestamps": int(timestamps.isna().sum()),
            "non_15_minute_intervals": interval_violations,
            "strictly_increasing": bool(
                not timestamps.isna().any()
                and not timestamps.duplicated().any()
                and timestamps.is_monotonic_increasing
            ),
        },
        "invalid_col": {
            "nonfinite_or_nonnumeric_columns": invalid_columns,
            "unexpected_raw_columns": schema["unexpected_raw_columns"],
            "count": int(len(set(invalid_columns).union(schema["unexpected_raw_columns"]))),
        },
        "feat": {
            "feature_columns": int(schema["feature_column_count"]),
            "raw_columns": int(schema["raw_column_count"]),
            "input_columns_excluding_datetime": int(len(frame.columns) - 1),
        },
        "comp": {
            "rows": int(len(frame)),
            "required_rows": 192,
            "row_count_complete": len(frame) == 192,
            "timestamps_align_with_s_result": aligned,
            "missing_required_raw_columns": schema["missing_required_raw_columns"],
        },
    }


def _assert_package_directory(path: Path) -> None:
    members = sorted(item.name for item in path.iterdir())
    if members != sorted(SUBMISSION_MEMBERS):
        raise ValueError(f"提交目录只能包含 input.csv 与 s_result.csv: {path} -> {members}")


def _archive_member_hashes(path: Path) -> dict[str, str]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if names != list(SUBMISSION_MEMBERS):
            raise ValueError(f"ZIP 根目录成员不符合契约: {path} -> {names}")
        return {name: _sha256_bytes(archive.read(name)) for name in names}


def run_q4(
    scoring_input: Path,
    source_zip: Path,
    run_dir: Path,
    *,
    training_input: Path | None = None,
    training_data_dir: Path | None = None,
    model_path: Path | None = None,
    train_end: str | None = None,
) -> dict[str, object]:
    """走正式 submission 链生成 A/B 包，并冻结全部比较证据。"""

    output = run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt_root = output / "receipts"
    chain_dir = receipt_root / "formal_chain"
    receipt_root.mkdir()
    chain_dir.mkdir()

    scoring_source = scoring_input.resolve()
    source_archive = source_zip.resolve()
    for source in (scoring_source, source_archive):
        if not source.is_file():
            raise FileNotFoundError(f"Q4 来源文件不存在: {source}")
    source_archive_before = _sha256_file(source_archive)
    scoring_file = receipt_root / "source_scoring_input.csv"
    scoring_record = _materialize_scoring_input(scoring_source, scoring_file)
    scoring_hash_before = _sha256_file(scoring_file)

    training_file = receipt_root / "training_input.csv"
    training_record = _materialize_training_input(
        training_file,
        training_input=None if training_input is None else training_input.resolve(),
        training_data_dir=(None if training_data_dir is None else training_data_dir.resolve()),
        model_path=None if model_path is None else model_path.resolve(),
    )
    training_hash_before = _sha256_file(training_file)

    frozen_result_bytes = _read_submission_member(source_archive, "s_result.csv")
    frozen_result = receipt_root / "source_s_result.csv"
    frozen_result.write_bytes(frozen_result_bytes)
    source_result_hash = _sha256_bytes(frozen_result_bytes)

    chain = prepare_submission_chain(
        training_file,
        scoring_file,
        frozen_result,
        chain_dir,
        train_end=train_end,
        policy=COMPETITION_QUALITY_POLICY,
    )
    causal_input_path = Path(chain["causal_input_path"])
    reference_input_path = Path(chain["input_path"])
    causal_hash_before_packaging = _sha256_file(causal_input_path)

    sub_a = output / "SUB_A_Q_CAUSAL"
    sub_b = output / "SUB_B_Q_REFERENCE"
    sub_a.mkdir()
    sub_b.mkdir()
    shutil.copyfile(causal_input_path, sub_a / "input.csv")
    (sub_a / "s_result.csv").write_bytes(frozen_result_bytes)
    shutil.copyfile(reference_input_path, sub_b / "input.csv")
    (sub_b / "s_result.csv").write_bytes(frozen_result_bytes)
    _assert_package_directory(sub_a)
    _assert_package_directory(sub_b)

    result_frame = pd.read_csv(frozen_result)
    input_a_frame = pd.read_csv(sub_a / "input.csv")
    input_b_frame = pd.read_csv(sub_b / "input.csv")
    validate_submission_input(input_a_frame, result_frame)
    validate_submission_input(input_b_frame, result_frame)

    zip_a = output / "SUB_A_Q_CAUSAL.zip"
    zip_b = output / "SUB_B_Q_REFERENCE.zip"
    package_a = package_submission(
        sub_a / "input.csv",
        sub_a / "s_result.csv",
        zip_a,
        result_freeze=chain["result_freeze"],
    )
    package_b = package_submission(
        sub_b / "input.csv",
        sub_b / "s_result.csv",
        zip_b,
        quality_receipt_path=chain["quality_receipt_path"],
        result_freeze=chain["result_freeze"],
    )
    archive_a = validate_submission_archive(
        zip_a,
        expected_input_path=sub_a / "input.csv",
        expected_result_path=sub_a / "s_result.csv",
        result_freeze=chain["result_freeze"],
    )
    archive_b = validate_submission_archive(
        zip_b,
        expected_input_path=sub_b / "input.csv",
        expected_result_path=sub_b / "s_result.csv",
        quality_receipt_path=chain["quality_receipt_path"],
        result_freeze=chain["result_freeze"],
    )
    archived_hashes_a = _archive_member_hashes(zip_a)
    archived_hashes_b = _archive_member_hashes(zip_b)

    result_hashes = {
        "source_zip_member": source_result_hash,
        "SUB_A_file": _sha256_file(sub_a / "s_result.csv"),
        "SUB_B_file": _sha256_file(sub_b / "s_result.csv"),
        "SUB_A_zip_member": archived_hashes_a["s_result.csv"],
        "SUB_B_zip_member": archived_hashes_b["s_result.csv"],
    }
    if len(set(result_hashes.values())) != 1:
        raise ValueError(f"A/B 的 s_result 未保持逐字节一致: {result_hashes}")
    if _sha256_file(causal_input_path) != causal_hash_before_packaging:
        raise ValueError("打包阶段改写了 Q_CAUSAL 输入")
    if _sha256_file(scoring_file) != scoring_hash_before:
        raise ValueError("正式链改写了评分输入副本")
    if _sha256_file(training_file) != training_hash_before:
        raise ValueError("正式链改写了训练输入副本")
    if _sha256_file(source_archive) != source_archive_before:
        raise ValueError("正式链改写了源提交 ZIP")

    reference_report = chain["quality_receipt"]["reference_normalization"]
    final_quality = reference_report["final_quality"]
    expected_zeros = {
        "nonfinite_cells": 0,
        "constant_columns": [],
        "duplicate_columns": [],
        "iqr_outlier_cells_all_methods": 0,
        "zscore_outlier_cells": 0,
    }
    if any(final_quality.get(name) != expected for name, expected in expected_zeros.items()):
        raise ValueError(f"Q_REFERENCE 终态机械门禁失败: {final_quality}")

    report: dict[str, Any] = {
        "experiment": "Q4_reference_quality_ab",
        "status": "LOCAL_AB_READY_PLATFORM_NOT_SUBMITTED",
        "prediction_provenance": {
            "source_zip": str(source_archive),
            "source_zip_sha256": source_archive_before,
            "s_result_sha256": source_result_hash,
            "declared_model": "current_legally_submitted_frozen_result_not_a61",
            "a61_production_s_result_available": False,
            "reason": (
                "仓库 A61 只有 development/verification OOF 且 formal_candidate=false；"
                "复用已有合法提交且仓库记录平台准确率 49.9/50 的冻结预测"
            ),
        },
        "inputs": {
            "scoring": scoring_record,
            "training": training_record,
            "train_end": train_end,
            "scoring_copy_immutable": _sha256_file(scoring_file) == scoring_hash_before,
            "training_copy_immutable": _sha256_file(training_file) == training_hash_before,
        },
        "formal_chain": {
            "api": "gas_forecast.submission.prepare_submission_chain",
            "causal_receipt": str(chain["causal_receipt_path"]),
            "reference_receipt": str(chain["quality_receipt_path"]),
            "future_perturbation_passed": bool(
                chain["causal_receipt"]["future_perturbation"]["passed"]
            ),
            "q_reference_feeds_model": bool(chain["quality_receipt"]["feeds_model"]),
            "causal_input_immutable_after_reference": bool(
                chain["quality_receipt"]["causal_input_immutable_after_prediction"]["passed"]
            ),
            "prediction_mode": chain["quality_receipt"]["prediction_input"],
        },
        "s_result_freeze": {
            "hashes": result_hashes,
            "all_byte_identical": True,
        },
        "SUB_A": {
            "mode": "Q_CAUSAL",
            "directory": str(sub_a),
            "directory_members": sorted(item.name for item in sub_a.iterdir()),
            "zip": str(zip_a),
            "zip_sha256": _sha256_file(zip_a),
            "input_sha256": _sha256_file(sub_a / "input.csv"),
            "mechanical_metrics": _mechanical_metrics(input_a_frame, result_frame),
            "package": package_a,
            "archive_validation": archive_a,
            "archive_member_hashes": archived_hashes_a,
        },
        "SUB_B": {
            "mode": "Q_REFERENCE",
            "directory": str(sub_b),
            "directory_members": sorted(item.name for item in sub_b.iterdir()),
            "zip": str(zip_b),
            "zip_sha256": _sha256_file(zip_b),
            "input_sha256": _sha256_file(sub_b / "input.csv"),
            "mechanical_metrics": _mechanical_metrics(input_b_frame, result_frame),
            "terminal_quality": final_quality,
            "write_read_back": chain["quality_receipt"]["write_read_back"],
            "package": package_b,
            "archive_validation": archive_b,
            "archive_member_hashes": archived_hashes_b,
        },
        "platform": {
            "submitted": False,
            "quality_score": None,
            "accuracy_score": None,
            "reason": "未获外部提交授权；本地机械指标不等同于平台计分",
        },
    }
    _write_json(output / "q4_ab_report.json", report)
    return report


def main() -> int:
    args = _parse_args()
    report = run_q4(
        args.scoring_input,
        args.source_zip,
        args.run_dir,
        training_input=args.training_input,
        training_data_dir=args.training_data_dir,
        model_path=args.model,
        train_end=args.train_end,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
