"""仅使用预测起点可见特征定义诊断工况。"""

from __future__ import annotations

import pandas as pd


def classify_origin_regimes(features: pd.DataFrame) -> pd.Series:
    """按稳定优先级标记缺失、异常、冻结、煤气切换、电价切换或常规工况。"""

    regime = pd.Series("normal", index=features.index, dtype="object")
    missing_columns = [column for column in features if column.startswith("feat_missing_")]
    outlier_columns = [column for column in features if column.endswith("_is_outlier")]
    freeze_columns = [column for column in features if column.endswith("_freeze_length")]
    if missing_columns:
        regime.loc[features[missing_columns].fillna(0).gt(0).any(axis=1)] = "missing_input"
    if outlier_columns:
        regime.loc[features[outlier_columns].fillna(0).gt(0).any(axis=1)] = "causal_outlier"
    if freeze_columns:
        regime.loc[features[freeze_columns].fillna(0).ge(4).any(axis=1)] = "frozen_signal"
    if "feat_dominant_gas_changed" in features:
        regime.loc[features["feat_dominant_gas_changed"].fillna(0).gt(0)] = "gas_switch"
    if "feat_price_switch_within_120" in features:
        regime.loc[features["feat_price_switch_within_120"].fillna(0).gt(0)] = "price_switch"
    return regime


def attach_regimes(rows: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """按 origin_time 将因果工况标签附到逐行 OOF。"""

    output = rows.copy()
    regimes = classify_origin_regimes(features)
    mapped = pd.to_datetime(output["origin_time"]).map(regimes)
    output["regime"] = mapped.fillna("unknown").astype(str)
    return output
