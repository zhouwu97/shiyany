"""生成 R0（当前 Q5 等价）与 R1（Exact Reference Input Clone）A/B 提交包。

R1 目标：把 ``diaofenyuan/aic-gangtie`` 从官方评分原表到最终 ``input.csv``
的完整构造语义一层不漏地复刻（除项目字段映射外）：

    raw 加载（官方原表全列，无 allowlist）
    -> prepare_submission_sources（Hampel 672/96/6 + median + 无限制 ffill）
    -> 特征 sanitize（训练期 fit all-nonfinite/constant/duplicate/median，
       评分期只套 schema）
    -> concat(raw sources, sanitized features)
    -> 全矩阵 Q_REFERENCE 归一化（IQR 1.5 / clip 1.0 / 五插值 / Z>3 / 10 轮 /
       残余异常列直接删）

R0 与 R1 共享同一份冻结 ``s_result.csv`` 字节；只有 ``input.csv`` 不同。
全部收据与终态门禁通过后才标记 ``LEGAL_R1_READY_FOR_PLATFORM``。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# 复用 Q4/Q5 已验证的预测源校验、哈希与打包工具，避免重复实现。
from run_q4_reference_quality_packages import (  # noqa: E402
    _archive_member_hashes,
    _assert_package_directory,
    _build_training_input,
    _materialize_scoring_input,
    _mechanical_metrics,
    _read_submission_member,
    _result_bytes_frame,
    _sha256_bytes,
    _sha256_file,
    _validate_prediction_source,
    _write_json,
)

from gas_forecast.data import (  # noqa: E402
    align_tables,
    combine_context,
    load_original_input_frame,
)
from gas_forecast.features import build_causal_features, load_price_schedule  # noqa: E402
from gas_forecast.submission import (  # noqa: E402
    package_submission,
    prepare_submission_chain,
    validate_submission_archive,
    validate_submission_input,
)
from gas_forecast.submission_quality import (  # noqa: E402
    COMPETITION_QUALITY_POLICY,
    audit_submission_quality,
    inspect_submission_input_quality,
    prepare_exact_reference_input,
)
from gas_forecast.workflow import resolve_prediction_feature_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-zip",
        type=Path,
        required=True,
        help="通过 Production Gate 的冻结预测 ZIP；只读取其中 s_result.csv",
    )
    parser.add_argument(
        "--prediction-manifest",
        type=Path,
        required=True,
        help="与冻结预测 ZIP 配套、含文件哈希和 Production Gate 结论的 manifest.json",
    )
    parser.add_argument(
        "--declared-result-file",
        type=Path,
        default=None,
        help="manifest hashes.result 指向的更高精度本地 s_result 副本",
    )
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        required=True,
        help="官方训练数据目录（含 Pre_gas.csv 等四表与 price.xlsx）",
    )
    parser.add_argument(
        "--scoring-data-dir",
        type=Path,
        required=True,
        help="官方评分数据目录（含 Pre_test_*.csv 四表）",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="冻结模型，用于取得正式特征配置",
    )
    parser.add_argument("--train-end", required=True, help="Q_CAUSAL 训练统计截止时间")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印阶段进度；默认逐阶段输出时间戳与耗时",
    )
    return parser.parse_args()


def _feature_only(frame: pd.DataFrame) -> pd.DataFrame:
    """只保留 ``feat_`` 前缀的工程特征列（与参考仓库纯特征语义一致）。"""

    return frame.loc[:, [str(column) for column in frame.columns if str(column).startswith("feat_")]]


def _origins_from_result(frozen_result_bytes: bytes) -> pd.DatetimeIndex:
    frame = _result_bytes_frame(frozen_result_bytes)
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], errors="raise"))
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("冻结 s_result 的 datetime 必须唯一且严格递增")
    timestamps.name = "datetime"
    return timestamps


def _audit_rows(label: str, frame: pd.DataFrame, result: pd.DataFrame) -> dict[str, object]:
    """生成平台可机械复核的本地审计行（不推断平台计分公式）。"""

    quality = inspect_submission_input_quality(frame)
    schema = audit_submission_quality(frame, COMPETITION_QUALITY_POLICY)
    return {
        "label": label,
        "raw_columns": int(schema["raw_column_count"]),
        "feature_columns": int(schema["feature_column_count"]),
        "input_columns_excluding_datetime": int(len(frame.columns) - 1),
        "all_nonfinite_cells": int(quality["nonfinite_cells"]),
        "constant_columns": list(quality["constant_columns"]),
        "duplicate_columns": list(quality["duplicate_columns"]),
        "iqr_outlier_cells_linear": int(quality["iqr_outlier_cells"]),
        "iqr_outlier_cells_all_methods": int(quality["iqr_outlier_cells_all_methods"]),
        "abs_z_gt_3_cells": int(quality["zscore_outlier_cells"]),
        "residual_dropped_columns": [],
        "repaired_cells": 0,
        "mechanical": _mechanical_metrics(frame, result),
    }


def run_r1(
    source_zip: Path,
    run_dir: Path,
    *,
    prediction_manifest: Path,
    training_data_dir: Path,
    scoring_data_dir: Path,
    model_path: Path,
    train_end: str,
    declared_result_file: Path | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """生成 R0（当前 Q5 等价）与 R1（Exact Reference Input Clone）包。

    ``verbose`` 为真时逐阶段打印进度与耗时，便于长实验监控。
    """

    def progress(message: str, started: float | None = None) -> float:
        """打印阶段进度；传入上一阶段时间戳时附带耗时。"""

        elapsed = f"  [{time.monotonic() - started:.1f}s]" if started is not None else ""
        if verbose:
            print(f"[R1 {datetime.now().strftime('%H:%M:%S')}] {message}{elapsed}", flush=True)
        return time.monotonic()

    timer = progress("校验预测源与 manifest")
    source_archive = source_zip.resolve()
    source_manifest = prediction_manifest.resolve()
    for source in (source_archive, source_manifest, model_path, training_data_dir, scoring_data_dir):
        if not source.is_file() and not source.is_dir():
            raise FileNotFoundError(f"R1 来源文件不存在: {source}")
    source_archive_before = _sha256_file(source_archive)
    frozen_result_bytes = _read_submission_member(source_archive, "s_result.csv")
    source_result_hash = _sha256_bytes(frozen_result_bytes)
    provenance = _validate_prediction_source(
        source_manifest,
        archive_sha256=source_archive_before,
        result_sha256=source_result_hash,
        frozen_result_bytes=frozen_result_bytes,
        declared_result_file=None
        if declared_result_file is None
        else declared_result_file.resolve(),
    )
    timer = progress(f"预测源校验通过（s_result SHA256={source_result_hash[:12]}…）", timer)

    output = run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt_root = output / "receipts"
    r0_chain_dir = receipt_root / "r0_formal_chain"
    receipt_root.mkdir()
    r0_chain_dir.mkdir()

    # ---- 共享冻结结果 ----
    frozen_result = receipt_root / "source_s_result.csv"
    frozen_result.write_bytes(frozen_result_bytes)
    result_frame = _result_bytes_frame(frozen_result_bytes)
    origins = _origins_from_result(frozen_result_bytes)
    if len(origins) != 192:
        raise ValueError(f"冻结 s_result 的 origin 数必须为 192，实际 {len(origins)}")
    timer = progress(f"共享冻结结果就绪（{len(origins)} 个 origin）", timer)

    # ---- R0：当前 Q5 等价（Q_CAUSAL 21 列 allowlist -> Q_REFERENCE 副本） ----
    timer = progress("R0：重建训练 input（align_tables + build_causal_features）")
    scoring_file = receipt_root / "source_scoring_input.csv"
    scoring_record = _materialize_scoring_input(source_archive, scoring_file)
    timer = progress("R0：重建训练 input（align_tables + build_causal_features）", timer)
    training_file = receipt_root / "r0_training_input.csv"
    training_record = _build_training_input(training_data_dir, model_path, training_file)
    timer = progress("R0：正式链 Q_CAUSAL → 冻结结果 → Q_REFERENCE 副本", timer)
    r0_chain = prepare_submission_chain(
        training_file,
        scoring_file,
        frozen_result,
        r0_chain_dir,
        train_end=train_end,
        policy=COMPETITION_QUALITY_POLICY,
    )
    r0_input_path = Path(r0_chain["input_path"])
    r0_input_hash = _sha256_file(r0_input_path)
    timer = progress("R0 完成（21 列 allowlist + Q_REFERENCE）", timer)

    # ---- R1：Exact Reference Input Clone ----
    timer = progress("R1：加载官方训练/评分原表（load_original_input_frame）")
    model = joblib.load(model_path)
    feature_config = resolve_prediction_feature_config(model)
    training_raw = load_original_input_frame(training_data_dir)
    scoring_raw = load_original_input_frame(scoring_data_dir)
    timer = progress(f"R1：原表加载完成（训练 {training_raw.shape[1]} 列 / 评分 {scoring_raw.shape[1]} 列）", timer)

    timer = progress("R1：对齐四表并构建上下文")
    train_aligned = align_tables(training_data_dir, feature_config.frequency).frame
    test_aligned = align_tables(scoring_data_dir, feature_config.frequency).frame
    price_paths = sorted(training_data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(price_paths[0]) if price_paths else None
    context = combine_context(train_aligned, test_aligned)
    timer = progress(f"R1：上下文就绪（{len(context)} 行）", timer)

    # 参考语义：特征在完整上下文（训练历史 + 评分期）上构建，训练特征用于
    # 拟合 schema，评分特征只取 origins 行。评分期特征允许存在 NaN，由
    # sanitize_submission_features 用训练中位数填充。
    timer = progress("R1：构建训练期特征（build_causal_features）")
    train_features = _feature_only(build_causal_features(train_aligned, feature_config, price))
    timer = progress(f"R1：训练特征完成（{train_features.shape[1]} 列）", timer)
    timer = progress("R1：构建完整上下文特征（评分期特征取 origins）")
    scoring_features = _feature_only(
        build_causal_features(context, feature_config, price)
    ).reindex(origins)
    timer = progress(f"R1：评分特征完成（{scoring_features.shape[1]} 列）", timer)
    missing_origins = scoring_features.index.difference(origins)
    if len(missing_origins):
        raise ValueError(f"R1 评分特征未覆盖全部预测起点: {missing_origins[:3].tolist()}")

    timer = progress("R1：prepare_submission_sources（Hampel 672/96/6）+ sanitize + concat + 全矩阵归一化")
    r1_input, r1_report = prepare_exact_reference_input(
        training_raw,
        scoring_raw,
        origins,
        train_features,
        scoring_features,
    )
    timer = progress(
        f"R1 链完成：concat {r1_report['concat']['input_columns_excluding_datetime']} 列 → 终态 {r1_report['final_quality']['columns']} 列",
        timer,
    )

    r1_dir = output / "R1_EXACT_REFERENCE_CLONE"
    r1_dir.mkdir()
    r1_input_path = r1_dir / "input.csv"
    (r1_dir / "s_result.csv").write_bytes(frozen_result_bytes)
    r1_input.to_csv(
        r1_input_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    _assert_package_directory(r1_dir)
    validate_submission_input(pd.read_csv(r1_input_path), result_frame)
    timer = progress("R1 input.csv 写盘并 read-back 复核完成", timer)

    # ---- R0/R1 共享结果字节验证 ----
    r1_result_hash = _sha256_file(r1_dir / "s_result.csv")
    if r1_result_hash != source_result_hash:
        raise ValueError("R1 包改写了冻结 s_result 字节")
    r0_result_hash = _sha256_file(r0_chain["result_path"])
    if r0_result_hash != source_result_hash:
        raise ValueError("R0 包改写了冻结 s_result 字节")
    if _sha256_file(scoring_file) != scoring_record["sha256"]:
        raise ValueError("正式链改写了评分输入副本")
    if _sha256_file(training_file) != training_record["sha256"]:
        raise ValueError("正式链改写了训练输入副本")
    timer = progress("R0/R1 共享 s_result 字节一致性验证通过", timer)

    # ---- ZIP 封装与复核 ----
    timer = progress("ZIP 封装与 archive 复核（R0 / R1）")
    zip_r0 = output / "R0_Q5_EQUIVALENT.zip"
    zip_r1 = output / "R1_EXACT_REFERENCE.zip"
    package_r0 = package_submission(
        r0_input_path,
        r0_chain["result_path"],
        zip_r0,
        quality_receipt_path=r0_chain["quality_receipt_path"],
        result_freeze=r0_chain["result_freeze"],
    )
    package_r1 = package_submission(
        r1_input_path,
        r1_dir / "s_result.csv",
        zip_r1,
        result_freeze=r0_chain["result_freeze"],
    )
    archive_r0 = validate_submission_archive(
        zip_r0,
        expected_input_path=r0_input_path,
        expected_result_path=r0_chain["result_path"],
        quality_receipt_path=r0_chain["quality_receipt_path"],
        result_freeze=r0_chain["result_freeze"],
    )
    archive_r1 = validate_submission_archive(
        zip_r1,
        expected_input_path=r1_input_path,
        expected_result_path=r1_dir / "s_result.csv",
        result_freeze=r0_chain["result_freeze"],
    )
    hashes_r0 = _archive_member_hashes(zip_r0)
    hashes_r1 = _archive_member_hashes(zip_r1)
    if hashes_r0["s_result.csv"] != hashes_r1["s_result.csv"]:
        raise ValueError("R0/R1 的 s_result ZIP 成员字节不一致")
    if hashes_r0["s_result.csv"] != source_result_hash:
        raise ValueError("ZIP 内 s_result 与冻结字节不一致")
    if hashes_r0["input.csv"] == hashes_r1["input.csv"]:
        raise ValueError("R0/R1 的 input.csv 不应相同")
    timer = progress("ZIP 封装完成（s_result 成员字节一致、input.csv 不同）", timer)

    # ---- 审计表 ----
    timer = progress("生成审计表与终态门禁检查")
    r0_input_frame = pd.read_csv(r0_input_path)
    r1_input_frame = pd.read_csv(r1_input_path)
    r0_row = _audit_rows("R0_Q5_EQUIVALENT", r0_input_frame, result_frame)
    r1_row = _audit_rows("R1_EXACT_REFERENCE_CLONE", r1_input_frame, result_frame)
    r0_reference = r0_chain["quality_receipt"]["reference_normalization"]
    r1_matrix = r1_report["matrix_normalization"]
    r0_row["residual_dropped_columns"] = list(r0_reference["dropped_residual_bad_columns"])
    r1_row["residual_dropped_columns"] = list(r1_matrix["dropped_residual_bad_columns"])
    r0_row["repaired_cells"] = int(
        sum(r0_reference["winsorized_by_column"].values())
    )
    r1_row["repaired_cells"] = int(
        sum(r1_report["raw_sources"]["missing_repairs"].values())
        + sum(r1_report["raw_sources"]["outlier_repairs"].values())
        + sum(r1_report["raw_sources"]["median_fallbacks"].values())
        + sum(r1_matrix["winsorized_by_column"].values())
    )

    # ---- R1 终态机械门禁 ----
    expected_zeros = {
        "nonfinite_cells": 0,
        "constant_columns": [],
        "duplicate_columns": [],
        "iqr_outlier_cells_all_methods": 0,
        "zscore_outlier_cells": 0,
    }
    final_quality = r1_matrix["final_quality"]
    if any(final_quality.get(name) != expected for name, expected in expected_zeros.items()):
        raise ValueError(f"R1 终态机械门禁失败: {final_quality}")
    if r1_report["raw_sources"]["nonfinite_after"] != 0:
        raise ValueError("R1 raw sources 修复后仍含非有限值")

    ready_status = "LEGAL_R1_READY_FOR_PLATFORM"
    timer = progress(f"终态门禁通过，状态 {ready_status}（R1 input SHA256={_sha256_file(r1_input_path)[:12]}…）", timer)

    report: dict[str, object] = {
        "experiment": "R1_EXACT_REFERENCE_INPUT_CLONE",
        "status": ready_status,
        "prediction_provenance": {
            **provenance,
            "source_zip": str(source_archive),
            "reason": "预测源与通过 Production Gate 的 manifest 哈希一致，且不含 Oracle 标记",
        },
        "inputs": {
            "scoring": scoring_record,
            "r0_training": training_record,
            "train_end": train_end,
            "origins": len(origins),
            "training_raw_columns": int(training_raw.shape[1]),
            "scoring_raw_columns": int(scoring_raw.shape[1]),
            "scoring_copy_immutable": _sha256_file(scoring_file) == scoring_record["sha256"],
            "training_copy_immutable": _sha256_file(training_file) == training_record["sha256"],
        },
        "s_result_freeze": {
            "source_sha256": source_result_hash,
            "R0_sha256": r0_result_hash,
            "R1_sha256": r1_result_hash,
            "R0_zip_member": hashes_r0["s_result.csv"],
            "R1_zip_member": hashes_r1["s_result.csv"],
            "all_byte_identical": True,
        },
        "R0_Q5_EQUIVALENT": {
            "mode": "Q_CAUSAL_21col_allowlist_then_Q_REFERENCE",
            "directory": str(r0_chain["result_path"].parent),
            "input_sha256": r0_input_hash,
            "audit": r0_row,
            "package": package_r0,
            "archive_validation": archive_r0,
            "archive_member_hashes": hashes_r0,
        },
        "R1_EXACT_REFERENCE_CLONE": {
            "mode": "exact_reference_clone_v1",
            "directory": str(r1_dir),
            "input_sha256": _sha256_file(r1_input_path),
            "audit": r1_row,
            "r1_report": r1_report,
            "package": package_r1,
            "archive_validation": archive_r1,
            "archive_member_hashes": hashes_r1,
        },
        "comparison_table": {
            "fields": [
                "raw_columns",
                "feature_columns",
                "all_nonfinite_cells",
                "constant_columns",
                "duplicate_columns",
                "iqr_outlier_cells_linear",
                "iqr_outlier_cells_all_methods",
                "abs_z_gt_3_cells",
                "residual_dropped_columns",
                "repaired_cells",
            ],
            "R0": {key: r0_row[key] for key in (
                "raw_columns",
                "feature_columns",
                "all_nonfinite_cells",
                "constant_columns",
                "duplicate_columns",
                "iqr_outlier_cells_linear",
                "iqr_outlier_cells_all_methods",
                "abs_z_gt_3_cells",
                "residual_dropped_columns",
                "repaired_cells",
            )},
            "R1": {key: r1_row[key] for key in (
                "raw_columns",
                "feature_columns",
                "all_nonfinite_cells",
                "constant_columns",
                "duplicate_columns",
                "iqr_outlier_cells_linear",
                "iqr_outlier_cells_all_methods",
                "abs_z_gt_3_cells",
                "residual_dropped_columns",
                "repaired_cells",
            )},
        },
        "platform": {
            "submitted": False,
            "quality_score": None,
            "accuracy_score": None,
            "reason": "未获外部提交授权；本地机械指标不等同于平台计分",
        },
    }
    _write_json(output / "r1_report.json", report)
    return report


def main() -> int:
    args = _parse_args()
    report = run_r1(
        args.source_zip,
        args.run_dir,
        prediction_manifest=args.prediction_manifest,
        training_data_dir=args.training_data_dir,
        scoring_data_dir=args.scoring_data_dir,
        model_path=args.model,
        train_end=args.train_end,
        declared_result_file=args.declared_result_file,
        verbose=not args.quiet,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
