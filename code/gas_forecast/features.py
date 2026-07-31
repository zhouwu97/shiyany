"""只使用当前及历史信息的特征工程。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig


@dataclass(frozen=True)
class PriceSchedule:
    """按月份和半小时槽位索引的已知电价计划。"""

    values: np.ndarray

    def lookup(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        months = timestamps.month.to_numpy() - 1
        slots = timestamps.hour.to_numpy() * 2 + timestamps.minute.to_numpy() // 30
        return self.values[slots, months]


@dataclass(frozen=True)
class FeatureAvailability:
    """特征相对预测起点的最大信息时间。"""

    max_offset_minutes: int
    known_in_advance: bool = False


def load_price_schedule(path: str | Path) -> PriceSchedule:
    """读取官方 48 行、12 个月电价表。"""

    frame = pd.read_excel(path)
    if frame.shape[0] != 48 or frame.shape[1] < 13:
        raise ValueError(f"电价表应包含 48 个半小时槽位和 12 个月: {path}")
    values = frame.iloc[:, 1:13].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    return PriceSchedule(values=values)


def _sum_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[existing].sum(axis=1, min_count=1)


def _add_history_features(
    output: dict[str, pd.Series | np.ndarray],
    series: pd.Series,
    prefix: str,
    config: FeatureConfig,
    *,
    include_range: bool,
) -> None:
    for lag in config.lags:
        output[f"feat_{prefix}_lag_{lag}"] = series.shift(lag)
    for lag in config.diff_lags:
        output[f"feat_{prefix}_diff_{lag}"] = series - series.shift(lag)

    # 历史窗口统一先移一格，避免窗口统计意外读取预测起点之后的数据。
    history = series.shift(1)
    for window in config.rolling_windows:
        rolling = history.rolling(window, min_periods=max(2, window // 2))
        output[f"feat_{prefix}_mean_{window}"] = rolling.mean()
        output[f"feat_{prefix}_std_{window}"] = rolling.std()
        if include_range:
            output[f"feat_{prefix}_min_{window}"] = rolling.min()
            output[f"feat_{prefix}_max_{window}"] = rolling.max()
        output[f"feat_{prefix}_vs_mean_{window}"] = series - rolling.mean()

    for window in (4, 8):
        output[f"feat_{prefix}_slope_{window}"] = (
            series - series.shift(window - 1)
        ) / (window - 1)


def _add_zero_features(
    output: dict[str, pd.Series | np.ndarray], series: pd.Series, prefix: str
) -> None:
    is_zero = series.eq(0)
    groups = (~is_zero).cumsum()
    age = is_zero.groupby(groups).cumcount().add(1).where(is_zero, 0)
    output[f"feat_{prefix}_is_zero"] = is_zero.astype("int8")
    output[f"feat_{prefix}_zero_age"] = age.astype("int32")
    output[f"feat_{prefix}_zero_started"] = (is_zero & ~is_zero.shift(1, fill_value=False)).astype(
        "int8"
    )
    output[f"feat_{prefix}_zero_ended"] = (~is_zero & is_zero.shift(1, fill_value=False)).astype(
        "int8"
    )


def build_causal_features(
    frame: pd.DataFrame,
    config: FeatureConfig | None = None,
    price_schedule: PriceSchedule | None = None,
) -> pd.DataFrame:
    """生成在每个时间点均可在线计算的特征。"""

    config = config or FeatureConfig()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("特征输入必须使用 DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("特征输入时间轴必须严格递增且唯一")

    numeric = frame.select_dtypes(include=[np.number]).copy()
    missing = numeric.isna()
    filled = numeric.ffill(limit=config.max_forward_fill_steps)
    feature_values: dict[str, pd.Series | np.ndarray] = {
        column: filled[column] for column in filled.columns
    }

    # 缺失标记的字段集合只由输入结构决定，不能因未来区间是否出现缺失而变化。
    for column in numeric.columns:
        feature_values[f"feat_missing_{column}"] = missing[column].astype("int8")

    if {"generator_1", "generator_all"}.issubset(filled.columns):
        feature_values["feat_generator_rest"] = filled["generator_all"] - filled["generator_1"]

    bf_production = _sum_existing(
        filled, ["blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5"]
    )
    air_heater_use = _sum_existing(
        filled, ["air_heater_1", "air_heater_2", "air_heater_4", "air_heater_5"]
    )
    bf_user_use = _sum_existing(
        filled,
        [
            "blast_furnace_user1",
            "blast_furnace_user2",
            "blast_furnace_user3",
            "blast_furnace_user4",
        ],
    )
    mixed_bf = filled.get(
        "into_gas_mixed_blast_furnace", pd.Series(np.nan, index=filled.index)
    )
    feature_values["feat_bf_production"] = bf_production
    feature_values["feat_air_heater_use"] = air_heater_use
    feature_values["feat_bf_user_use"] = bf_user_use
    feature_values["feat_bf_surplus_proxy"] = (
        bf_production - air_heater_use - bf_user_use - mixed_bf
    )

    generator_gas_columns = [
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
    ]
    generator_gas_total = _sum_existing(filled, generator_gas_columns)
    feature_values["feat_generator_gas_total"] = generator_gas_total
    for column in generator_gas_columns:
        if column in filled:
            denominator = generator_gas_total.replace(0, np.nan)
            feature_values[f"feat_{column}_share"] = filled[column] / denominator

    if all(column in filled for column in generator_gas_columns):
        gas_matrix = filled[generator_gas_columns].fillna(0.0).clip(lower=0.0)
        gas_share_matrix = gas_matrix.div(gas_matrix.sum(axis=1).replace(0, np.nan), axis=0)
        entropy = -(gas_share_matrix * np.log(gas_share_matrix.clip(lower=1e-12))).sum(axis=1)
        dominant = gas_matrix.to_numpy().argmax(axis=1)
        dominant_series = pd.Series(dominant, index=filled.index, dtype="int8")
        dominant_changed = dominant_series.ne(dominant_series.shift(1))
        dominant_changed.iloc[0] = False
        switch_groups = dominant_changed.cumsum()
        feature_values["feat_gas_mix_entropy"] = entropy
        feature_values["feat_dominant_gas_type"] = dominant_series
        feature_values["feat_dominant_gas_changed"] = dominant_changed.astype("int8")
        feature_values["feat_steps_since_gas_switch"] = dominant_changed.groupby(
            switch_groups
        ).cumcount().astype("int32")

        blast = filled["generator_use_blast_furnace_gas"]
        coke = filled["generator_use_coke_gas"]
        converter = filled["generator_use_converter_gas"]
        feature_values["feat_coke_down_blast_up"] = (
            (coke < coke.shift(1)) & (blast > blast.shift(1))
        ).astype("int8")
        feature_values["feat_coke_up_blast_down"] = (
            (coke > coke.shift(1)) & (blast < blast.shift(1))
        ).astype("int8")
        feature_values["feat_converter_down_blast_up"] = (
            (converter < converter.shift(1)) & (blast > blast.shift(1))
        ).astype("int8")

    history_series: dict[str, tuple[pd.Series, bool]] = {}
    for target in ("generator_1", "generator_all"):
        if target in filled:
            history_series[target] = (filled[target], True)
    if "feat_generator_rest" in feature_values:
        history_series["generator_rest"] = (feature_values["feat_generator_rest"], True)
    for column in generator_gas_columns:
        if column in filled:
            history_series[column] = (filled[column], False)
    history_series["bf_surplus_proxy"] = (feature_values["feat_bf_surplus_proxy"], False)
    if "blast_furnace_gas_holder_2" in filled:
        history_series["blast_furnace_gas_holder_2"] = (
            filled["blast_furnace_gas_holder_2"],
            True,
        )

    for name, (series, include_range) in history_series.items():
        _add_history_features(feature_values, series, name, config, include_range=include_range)

    zero_candidates = generator_gas_columns + [
        "air_heater_5",
        "into_gas_mixed_blast_furnace",
        "converter_user1",
        "converter_user2",
    ]
    for column in zero_candidates:
        if column in filled:
            _add_zero_features(feature_values, filled[column], column)

    index = frame.index
    minute_of_day = index.hour * 60 + index.minute
    feature_values["feat_hour"] = index.hour.astype("int8")
    feature_values["feat_minute_slot"] = (index.minute // 15).astype("int8")
    feature_values["feat_day_of_week"] = index.dayofweek.astype("int8")
    feature_values["feat_month"] = index.month.astype("int8")
    feature_values["feat_time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    feature_values["feat_time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    if price_schedule is not None:
        prices: list[np.ndarray] = []
        for horizon in config.horizons:
            target_index = index + pd.Timedelta(minutes=15 * horizon)
            price = price_schedule.lookup(target_index)
            prices.append(price)
            feature_values[f"feat_target_price_tplus_{15 * horizon}"] = price
        price_matrix = np.column_stack(prices)
        changed = price_matrix != price_matrix[:, [0]]
        first_change = np.where(changed.any(axis=1), changed.argmax(axis=1) + 1, 0)
        feature_values["feat_price_switch_within_120"] = changed.any(axis=1).astype("int8")
        feature_values["feat_steps_to_price_switch"] = first_change.astype("int8")

    output = pd.DataFrame(feature_values, index=frame.index)
    output = output.replace([np.inf, -np.inf], np.nan)
    audit_feature_availability(list(output.columns))
    return output


def audit_feature_availability(columns: list[str]) -> dict[str, FeatureAvailability]:
    """登记特征最大信息时间，并拒绝未知生产变量的未来偏移。"""

    metadata: dict[str, FeatureAvailability] = {}
    for column in columns:
        if column.startswith("feat_target_price_tplus_"):
            minutes = int(column.rsplit("_", 1)[-1])
            metadata[column] = FeatureAvailability(minutes, known_in_advance=True)
        elif "_lag_" in column:
            steps = int(column.rsplit("_", 1)[-1])
            metadata[column] = FeatureAvailability(-15 * steps)
        else:
            metadata[column] = FeatureAvailability(0)

    violations = [
        column
        for column, item in metadata.items()
        if item.max_offset_minutes > 0 and not item.known_in_advance
    ]
    if violations:
        raise ValueError(f"发现未知未来生产特征: {violations}")
    return metadata
