"""初赛结果宽表校验和提交包生成。"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig

SUBMISSION_MEMBERS = ["input.csv", "s_result.csv"]


def expected_prediction_columns(config: ForecastConfig | None = None) -> list[str]:
    config = config or ForecastConfig()
    return [
        f"{target}_t+{15 * horizon}_pred"
        for target in config.targets
        for horizon in config.feature.horizons
    ]


def validate_submission_frame(
    frame: pd.DataFrame,
    config: ForecastConfig | None = None,
) -> dict[str, object]:
    config = config or ForecastConfig()
    expected = ["datetime", *expected_prediction_columns(config)]
    if list(frame.columns) != expected:
        missing = [column for column in expected if column not in frame.columns]
        extra = [column for column in frame.columns if column not in expected]
        raise ValueError(f"结果字段或顺序不合法，缺少={missing}，多余={extra}")
    if frame.empty:
        raise ValueError("结果文件为空")

    timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("datetime 含非法时间戳")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("datetime 必须唯一且严格递增")
    spacing = timestamps.diff().dropna()
    if not spacing.empty and not spacing.eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("滚动起点必须保持 15 分钟连续")

    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("预测字段含缺失值、非数值或非有限值")

    for horizon in config.feature.horizons:
        minutes = 15 * horizon
        generator_1 = values[f"generator_1_t+{minutes}_pred"]
        generator_all = values[f"generator_all_t+{minutes}_pred"]
        if not generator_1.between(0, 200).all():
            raise ValueError(f"generator_1 t+{minutes} 超出 [0, 200]")
        if not generator_all.between(0, 440).all():
            raise ValueError(f"generator_all t+{minutes} 超出 [0, 440]")
        if not (generator_all >= generator_1).all():
            raise ValueError(f"t+{minutes} 存在 generator_all < generator_1")

    return {
        "rows": len(frame),
        "prediction_columns": len(values.columns),
        "start": str(timestamps.iloc[0]),
        "end": str(timestamps.iloc[-1]),
    }


def validate_submission_input(
    input_frame: pd.DataFrame,
    result_frame: pd.DataFrame,
) -> dict[str, object]:
    """校验提交输入特征，并确认它与预测结果逐行对应。"""

    if input_frame.empty:
        raise ValueError("input.csv 为空")
    if input_frame.columns[0] != "datetime":
        raise ValueError("input.csv 第一列必须为 datetime")
    if input_frame.columns.duplicated().any():
        raise ValueError("input.csv 含重复字段")

    timestamps = pd.to_datetime(input_frame["datetime"], errors="coerce")
    result_timestamps = pd.to_datetime(result_frame["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("input.csv 的 datetime 含非法时间戳")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("input.csv 的 datetime 必须唯一且严格递增")
    if len(input_frame) != len(result_frame) or not timestamps.reset_index(drop=True).equals(
        result_timestamps.reset_index(drop=True)
    ):
        raise ValueError("input.csv 与 s_result.csv 的行数或时间戳不一致")

    features = input_frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if features.empty:
        raise ValueError("input.csv 不含模型输入字段")
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("input.csv 的模型输入字段含缺失值、非数值或非有限值")

    return {
        "rows": len(input_frame),
        "input_columns": len(features.columns),
        "start": str(timestamps.iloc[0]),
        "end": str(timestamps.iloc[-1]),
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def validate_submission_archive(
    archive_path: str | Path,
    *,
    expected_input_path: str | Path | None = None,
    expected_result_path: str | Path | None = None,
) -> dict[str, object]:
    """校验 ZIP 成员、内部 CSV，以及可选的磁盘源文件一致性。"""

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if names != SUBMISSION_MEMBERS:
            raise ValueError(f"提交 ZIP 必须依次包含 {SUBMISSION_MEMBERS}，实际为: {names}")
        archived_input = pd.read_csv(io.BytesIO(archive.read("input.csv")))
        archived_result = pd.read_csv(io.BytesIO(archive.read("s_result.csv")))

    result_summary = validate_submission_frame(archived_result)
    input_summary = validate_submission_input(archived_input, archived_result)
    if expected_input_path is not None:
        expected_input = pd.read_csv(expected_input_path)
        validate_submission_input(expected_input, archived_result)
        if list(expected_input.columns) != list(archived_input.columns):
            raise ValueError("磁盘 input.csv 与 ZIP 内 input.csv 字段不一致")
        np.testing.assert_allclose(
            expected_input.iloc[:, 1:].to_numpy(float),
            archived_input.iloc[:, 1:].to_numpy(float),
            rtol=0.0,
            atol=5e-10,
        )
    if expected_result_path is not None:
        expected_result = pd.read_csv(expected_result_path)
        validate_submission_frame(expected_result)
        if list(expected_result.columns) != list(archived_result.columns):
            raise ValueError("磁盘 s_result.csv 与 ZIP 内 s_result.csv 字段不一致")
        if not pd.to_datetime(expected_result["datetime"]).equals(
            pd.to_datetime(archived_result["datetime"])
        ):
            raise ValueError("磁盘 s_result.csv 与 ZIP 内 s_result.csv 时间戳不一致")
        np.testing.assert_allclose(
            expected_result.iloc[:, 1:].to_numpy(float),
            archived_result.iloc[:, 1:].to_numpy(float),
            rtol=0.0,
            atol=5e-7,
        )
    return {
        "valid": True,
        "members": list(SUBMISSION_MEMBERS),
        "validation": result_summary,
        "input": input_summary,
    }


def package_submission(
    input_path: str | Path,
    result_path: str | Path,
    output_zip: str | Path,
) -> dict[str, object]:
    input_frame = pd.read_csv(input_path)
    result_frame = pd.read_csv(result_path)
    result_summary = validate_submission_frame(result_frame)
    input_summary = validate_submission_input(input_frame, result_frame)
    input_buffer = io.StringIO()
    result_buffer = io.StringIO()
    input_frame.to_csv(input_buffer, index=False, lineterminator="\n")
    result_frame.to_csv(
        result_buffer,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
    )
    destination = Path(output_zip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            _zip_info("input.csv"),
            input_buffer.getvalue().encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
        archive.writestr(
            _zip_info("s_result.csv"),
            result_buffer.getvalue().encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    archive_summary = validate_submission_archive(
        destination,
        expected_input_path=input_path,
        expected_result_path=result_path,
    )
    return {
        **result_summary,
        **input_summary,
        "prediction_columns": result_summary["prediction_columns"],
        "archive_members": archive_summary["members"],
        "archive": str(destination),
    }


def export_legacy_json(result_path: str | Path, output_path: str | Path) -> dict[str, object]:
    """导出数据字典旧版 columns/data JSON；不得与正式 CSV 同包提交。"""

    frame = pd.read_csv(result_path)
    summary = validate_submission_frame(frame)
    payload = {
        "columns": list(frame.columns),
        "data": frame.astype(object).where(pd.notna(frame), None).values.tolist(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    summary["json"] = str(destination)
    return summary
