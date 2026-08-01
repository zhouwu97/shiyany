"""可复现合规预处理与因果异常双通道。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessingAudit:
    rows: int
    columns: int
    duplicate_timestamps: int
    missing_cells: int
    missing_rows: int
    discontinuities: int
    outlier_cells: int
    frozen_cells: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def causal_hampel(
    series: pd.Series,
    *,
    window: int = 16,
    threshold: float = 4.5,
) -> pd.DataFrame:
    """仅用 t 之前窗口生成 Hampel 清洗值、标记和异常分数。"""

    numeric = pd.to_numeric(series, errors="coerce")
    history = numeric.shift(1)
    rolling = history.rolling(window, min_periods=max(4, window // 2))
    median = rolling.median()
    mad = (history - median).abs().rolling(window, min_periods=max(4, window // 2)).median()
    scale = (1.4826 * mad).replace(0.0, np.nan)
    score = (numeric - median).abs() / scale
    outlier = score.gt(threshold) & median.notna()
    clean = numeric.where(~outlier, median)
    positions = pd.Series(np.arange(len(numeric), dtype=float), index=numeric.index)
    last_outlier = positions.where(outlier).ffill()
    steps_since_outlier = (positions - last_outlier).where(last_outlier.notna(), -1)
    return pd.DataFrame(
        {
            "clean_value": clean,
            "is_outlier": outlier.astype("int8"),
            "outlier_score": score,
            "steps_since_outlier": steps_since_outlier.astype("int32"),
            "local_median_deviation": numeric - median,
        },
        index=series.index,
    )


def causal_freeze_features(series: pd.Series, *, tolerance: float = 1e-9) -> pd.DataFrame:
    """检测当前已观察到的精确或近似冻结长度。"""

    numeric = pd.to_numeric(series, errors="coerce")
    exact_same = numeric.eq(numeric.shift(1)) & numeric.notna()
    near_same = numeric.sub(numeric.shift(1)).abs().le(tolerance) & numeric.notna()

    def run_length(mask: pd.Series) -> pd.Series:
        groups = (~mask).cumsum()
        return mask.astype("int32").groupby(groups).cumsum().astype("int32")

    exact_length = run_length(exact_same)
    near_length = run_length(near_same)
    return pd.DataFrame(
        {
            "freeze_length": exact_length,
            "near_freeze_length": near_length,
            "freeze_started": (exact_length.eq(1)).astype("int8"),
            "freeze_ended": ((exact_length.eq(0)) & exact_length.shift(1).gt(0)).astype("int8"),
            "post_freeze_jump": (
                exact_length.eq(0) & exact_length.shift(1).gt(0) & numeric.sub(numeric.shift(1)).abs().gt(tolerance)
            ).astype("int8"),
        },
        index=series.index,
    )


def build_preprocessing_audit(
    frame: pd.DataFrame,
    *,
    frequency: str = "15min",
    hampel_window: int = 16,
    hampel_threshold: float = 4.5,
) -> PreprocessingAudit:
    """汇总去重、连续性、缺失、异常与冻结诊断，不覆盖原始值。"""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("预处理审计需要 DatetimeIndex")
    duplicate_timestamps = int(frame.index.duplicated(keep=False).sum())
    unique = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    expected = pd.date_range(unique.index.min(), unique.index.max(), freq=frequency)
    discontinuities = int(len(expected.difference(unique.index)))
    numeric = unique.select_dtypes(include=[np.number])
    outliers = 0
    frozen = 0
    for column in numeric.columns:
        outliers += int(causal_hampel(numeric[column], window=hampel_window, threshold=hampel_threshold)["is_outlier"].sum())
        frozen += int(causal_freeze_features(numeric[column])["freeze_length"].gt(0).sum())
    return PreprocessingAudit(
        rows=int(len(unique)),
        columns=int(unique.shape[1]),
        duplicate_timestamps=duplicate_timestamps,
        missing_cells=int(unique.isna().sum().sum()),
        missing_rows=int(unique.isna().any(axis=1).sum()),
        discontinuities=discontinuities,
        outlier_cells=outliers,
        frozen_cells=frozen,
    )


def build_anomaly_channels(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    window: int = 16,
    threshold: float = 4.5,
) -> dict[str, pd.Series]:
    """为登记字段生成 clean、异常与冻结通道，同时保留原始字段。"""

    output: dict[str, pd.Series] = {}
    for column in sorted(set(columns).intersection(frame.columns)):
        hampel = causal_hampel(frame[column], window=window, threshold=threshold)
        freeze = causal_freeze_features(frame[column])
        output[f"feat_{column}_clean"] = hampel["clean_value"]
        output[f"feat_{column}_is_outlier"] = hampel["is_outlier"]
        output[f"feat_{column}_outlier_score"] = hampel["outlier_score"]
        output[f"feat_{column}_steps_since_outlier"] = hampel["steps_since_outlier"]
        output[f"feat_{column}_local_median_deviation"] = hampel["local_median_deviation"]
        for feature_name in freeze.columns:
            output[f"feat_{column}_{feature_name}"] = freeze[feature_name]
    return output
