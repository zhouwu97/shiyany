"""初赛结果宽表校验和提交包生成。"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig


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


def package_submission(
    result_path: str | Path,
    output_zip: str | Path,
) -> dict[str, object]:
    frame = pd.read_csv(result_path)
    summary = validate_submission_frame(frame)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, float_format="%.6f", lineterminator="\n")
    destination = Path(output_zip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("result.csv", buffer.getvalue().encode("utf-8"))
    summary["archive"] = str(destination)
    return summary


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
