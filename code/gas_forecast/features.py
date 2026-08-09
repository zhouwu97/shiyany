"""只使用当前及历史信息的特征工程。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig
from gas_forecast.preprocessing import build_anomaly_channels


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


def _add_target_aligned_features(
    output: dict[str, pd.Series | np.ndarray],
    series: pd.Series,
    target: str,
    config: FeatureConfig,
) -> None:
    """为每个预测步长登记历史上对应目标时刻的周期锚点。"""

    for horizon in config.horizons:
        aligned: list[pd.Series] = []
        for days in config.target_aligned_cycle_days:
            lag = 96 * days - horizon
            if lag <= 0:
                continue
            value = series.shift(lag)
            aligned.append(value)
            output[f"feat_{target}_aligned_h{horizon}_lag_{lag}"] = value
        if not aligned:
            continue
        matrix = pd.concat(aligned, axis=1)
        output[f"feat_{target}_aligned_h{horizon}_mean"] = matrix.mean(axis=1)
        output[f"feat_{target}_aligned_h{horizon}_median"] = matrix.median(axis=1)
        output[f"feat_{target}_aligned_h{horizon}_vs_current"] = series - matrix.iloc[:, 0]


def _is_core_dynamic_column(column: str) -> bool:
    """识别生产侧的核心动态字段，不把所有原始字段默认扩展一遍。"""

    return (
        column in {"generator_1", "generator_all"}
        or column.startswith("generator_use_")
        or "gas_holder" in column
        or "blast_furnace" in column
        or "coke" in column
        or "converter" in column
    )


def _add_dynamic_features(
    output: dict[str, pd.Series | np.ndarray],
    filled: pd.DataFrame,
    config: FeatureConfig,
) -> None:
    """为核心或全部生产字段加入轻量、严格因果的动态统计。"""

    scope = config.dynamic_feature_scope
    if scope not in {"none", "core", "all"}:
        raise ValueError(f"不支持的动态特征范围: {scope}")
    if scope == "none":
        return
    columns = list(filled.columns) if scope == "all" else [
        column for column in filled.columns if _is_core_dynamic_column(column)
    ]
    for column in columns:
        series = filled[column]
        prefix = f"feat_dynamic_{column}"
        for lag in config.dynamic_lags:
            output[f"{prefix}_lag_{lag}"] = series.shift(lag)
            output[f"{prefix}_diff_{lag}"] = series - series.shift(lag)
        history = series.shift(1)
        for window in config.dynamic_rolling_windows:
            rolling = history.rolling(window, min_periods=max(2, window // 2))
            output[f"{prefix}_mean_{window}"] = rolling.mean()
            output[f"{prefix}_std_{window}"] = rolling.std()
            output[f"{prefix}_slope_{window}"] = (
                series - series.shift(window - 1)
            ) / max(window - 1, 1)


def _run_length(mask: pd.Series) -> pd.Series:
    """返回截至当前时刻连续 True 的长度，缺失和 False 均重置。"""

    mask = mask.fillna(False).astype(bool)
    groups = (~mask).cumsum()
    return mask.groupby(groups, sort=False).cumcount().add(1).where(mask, 0).astype("int16")


def _add_ramp_features(
    output: dict[str, pd.Series | np.ndarray],
    series: pd.Series,
) -> None:
    """增加 generator_1 的方向、加速度、运行长度和波动率特征。"""

    for lag in (1, 2, 4):
        output[f"feat_generator_1_ramp_diff_{lag}"] = series - series.shift(lag)
    diff = series.diff()
    output["feat_generator_1_acceleration"] = diff - diff.shift(1)
    for window in (4, 8, 16):
        history = series.shift(1)
        rolling = history.rolling(window, min_periods=max(2, window // 2))
        output[f"feat_generator_1_ramp_volatility_{window}"] = rolling.std()
        output[f"feat_generator_1_ramp_range_{window}"] = rolling.max() - rolling.min()
        output[f"feat_generator_1_ramp_q10_{window}"] = rolling.quantile(0.10)
        output[f"feat_generator_1_ramp_q90_{window}"] = rolling.quantile(0.90)
        output[f"feat_generator_1_ewma_{window}"] = series.ewm(
            span=window, adjust=False, min_periods=max(2, window // 2)
        ).mean()
    output["feat_generator_1_ramp_up_run_length"] = _run_length(diff.gt(0))
    output["feat_generator_1_ramp_down_run_length"] = _run_length(diff.lt(0))


def _steps_since_event(event: pd.Series) -> pd.Series:
    """返回距最近一次事件的步数；首次事件前保持缺失而不伪造历史。"""

    event = event.fillna(False).astype(bool)
    groups = event.cumsum()
    age = event.groupby(groups, sort=False).cumcount()
    return age.where(groups.gt(0)).astype(float)


def _add_rich_quantile_features(
    output: dict[str, pd.Series | np.ndarray],
    series: pd.Series,
    prefix: str,
    config: FeatureConfig,
) -> None:
    """增加稳健历史分位数，窗口末端始终停在预测起点之前。"""

    history = series.shift(1)
    for window in config.rich_quantile_windows:
        if window < 2:
            raise ValueError("rich_quantile_windows 中的窗口必须至少为 2")
        rolling = history.rolling(window, min_periods=max(2, window // 2))
        output[f"feat_rich_quantile_{prefix}_q10_{window}"] = rolling.quantile(0.10)
        output[f"feat_rich_quantile_{prefix}_q90_{window}"] = rolling.quantile(0.90)


def _add_rich_ramp_state_features(
    output: dict[str, pd.Series | np.ndarray],
    series: pd.Series,
    prefix: str,
) -> None:
    """将连续变化率转换为严格因果的上升、稳定和下降运行状态。"""

    rate = series.diff()
    # 阈值由过去两小时的变化率分位数决定，当前变化率只用于判定当前状态。
    threshold = rate.abs().shift(1).rolling(8, min_periods=4).quantile(0.75)
    ramp_up = rate.gt(threshold)
    ramp_down = rate.lt(-threshold)
    stable = rate.abs().le(threshold)
    ramp_event = ramp_up | ramp_down
    output[f"feat_rich_ramp_{prefix}_rate"] = rate
    output[f"feat_rich_ramp_{prefix}_threshold"] = threshold
    output[f"feat_rich_ramp_{prefix}_up"] = ramp_up.astype("int8")
    output[f"feat_rich_ramp_{prefix}_down"] = ramp_down.astype("int8")
    output[f"feat_rich_ramp_{prefix}_stable"] = stable.astype("int8")
    output[f"feat_rich_ramp_{prefix}_time_since_event"] = _steps_since_event(ramp_event)
    output[f"feat_rich_ramp_{prefix}_up_run_length"] = _run_length(ramp_up)
    output[f"feat_rich_ramp_{prefix}_down_run_length"] = _run_length(ramp_down)
    output[f"feat_rich_ramp_{prefix}_stable_run_length"] = _run_length(stable)


def _add_rich_gas_resource_features(
    output: dict[str, pd.Series | np.ndarray],
    filled: pd.DataFrame,
    *,
    bf_production: pd.Series,
    air_heater_use: pd.Series,
    bf_user_use: pd.Series,
    mixed_bf: pd.Series,
    converter_users: pd.Series,
) -> None:
    """构造煤气产量、优先需求和可供发电余量等资源状态。"""

    unavailable = pd.Series(np.nan, index=filled.index, dtype=float)
    coke_production = filled.get("coke_oven_1", unavailable)
    converter_production = filled.get("converter_1", unavailable)
    mixed_coke = filled.get("into_gas_mixed_coke", unavailable)
    mixed_converter = filled.get("into_gas_mixed_converter", unavailable)
    total_production = pd.concat(
        [bf_production, coke_production, converter_production], axis=1
    ).sum(axis=1, min_count=1)
    priority_demand = pd.concat(
        [
            air_heater_use,
            bf_user_use,
            mixed_bf,
            mixed_coke,
            converter_users,
            mixed_converter,
        ],
        axis=1,
    ).sum(axis=1, min_count=1)
    available_for_generation = total_production - priority_demand
    demand_denominator = priority_demand.replace(0.0, np.nan)
    production_denominator = total_production.replace(0.0, np.nan)
    output["feat_rich_gas_total_production"] = total_production
    output["feat_rich_gas_priority_demand"] = priority_demand
    output["feat_rich_gas_available_for_generation"] = available_for_generation
    output["feat_rich_gas_production_demand_ratio"] = total_production / demand_denominator
    output["feat_rich_gas_available_production_ratio"] = (
        available_for_generation / production_denominator
    )

    production_parts = {
        "blast": bf_production,
        "coke": coke_production,
        "converter": converter_production,
    }
    for name, value in production_parts.items():
        output[f"feat_rich_gas_{name}_production_share"] = value / production_denominator

    holder = filled.get("blast_furnace_gas_holder_2")
    if holder is not None:
        output["feat_rich_gas_holder_buffer"] = holder
        output["feat_rich_gas_holder_to_production_ratio"] = holder / production_denominator
        output["feat_rich_gas_holder_to_available_ratio"] = holder / available_for_generation.replace(
            0.0, np.nan
        )


def _add_relation_features(
    output: dict[str, pd.Series | np.ndarray],
    filled: pd.DataFrame,
    relation_specs: tuple[str, ...],
) -> None:
    """生成已冻结的工业时延关系特征。

    每个 spec 形如 ``source|lag|horizon``，只引用 ``source[t-lag]``；horizon
    仅用于让逐步长模型拥有独立关系字段，不会读取任何未来生产值。
    """

    for spec in relation_specs:
        parts = str(spec).split("|")
        if len(parts) != 3:
            raise ValueError(f"relation feature 规格应为 source|lag|horizon: {spec}")
        source, lag_text, horizon_text = parts
        if source in filled:
            source_series = filled[source]
        elif source == "generator_rest" and {"generator_all", "generator_1"}.issubset(
            filled.columns
        ):
            source_series = filled["generator_all"] - filled["generator_1"]
        elif source in output:
            source_series = pd.Series(output[source], index=filled.index)
        else:
            raise ValueError(f"relation feature 来源字段不存在: {source}")
        lag = int(lag_text)
        horizon = int(horizon_text)
        if lag < 0 or horizon < 1:
            raise ValueError(f"relation feature 的 lag/horizon 无效: {spec}")
        safe_source = re.sub(r"[^0-9A-Za-z_]+", "_", source)
        output[f"feat_relation_{safe_source}_lag_{lag}_h{horizon}"] = source_series.shift(lag)


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

    anomaly_columns = [
        "generator_1",
        "generator_all",
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
        "blast_furnace_gas_holder_2",
    ]
    if config.enable_anomaly_features:
        feature_values.update(
            build_anomaly_channels(
                numeric,
                anomaly_columns,
                window=config.anomaly_window,
                threshold=config.anomaly_threshold,
            )
        )

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

    # 三类煤气分别计算当前可见的供需守恒残差，不对缺少的用户字段作零值臆测。
    blast_balance = (
        bf_production
        - air_heater_use
        - bf_user_use
        - mixed_bf
        - filled.get("generator_use_blast_furnace_gas", np.nan)
    )
    coke_balance = (
        filled.get("coke_oven_1", pd.Series(np.nan, index=filled.index))
        - filled.get("into_gas_mixed_coke", 0.0)
        - filled.get("generator_use_coke_gas", 0.0)
    )
    converter_users = _sum_existing(filled, ["converter_user1", "converter_user2"])
    converter_balance = (
        filled.get("converter_1", pd.Series(np.nan, index=filled.index))
        - converter_users
        - filled.get("into_gas_mixed_converter", 0.0)
        - filled.get("generator_use_converter_gas", 0.0)
    )
    balances = {
        "blast_balance": blast_balance,
        "coke_balance": coke_balance,
        "converter_balance": converter_balance,
    }
    if config.enable_rich_gas_resource_features:
        _add_rich_gas_resource_features(
            feature_values,
            filled,
            bf_production=bf_production,
            air_heater_use=air_heater_use,
            bf_user_use=bf_user_use,
            mixed_bf=mixed_bf,
            converter_users=converter_users,
        )
    if config.enable_physical_features:
        for name, balance in balances.items():
            feature_values[f"feat_{name}"] = balance
            history = balance.shift(1)
            for window in (4, 8, 16):
                feature_values[f"feat_{name}_mean_{window}"] = history.rolling(
                    window, min_periods=max(2, window // 2)
                ).mean()
            feature_values[f"feat_{name}_diff_1"] = balance.diff()
            feature_values[f"feat_{name}_positive"] = balance.gt(0).astype("int8")
            state_change = balance.gt(0).ne(balance.gt(0).shift(1))
            feature_values[f"feat_{name}_state_changed"] = state_change.astype("int8")
            feature_values[f"feat_{name}_state_age"] = state_change.groupby(
                state_change.cumsum()
            ).cumcount().astype("int32")
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
    if config.enable_physical_features:
        for name, balance in balances.items():
            history_series[name] = (balance, False)
    if "blast_furnace_gas_holder_2" in filled:
        history_series["blast_furnace_gas_holder_2"] = (
            filled["blast_furnace_gas_holder_2"],
            True,
        )

    if config.enable_physical_features and {"generator_1", "generator_all"}.issubset(filled.columns):
        efficiency_series = {
            "generator_1_efficiency": filled["generator_1"] / generator_gas_total.replace(0, np.nan),
            "generator_all_efficiency": filled["generator_all"] / generator_gas_total.replace(0, np.nan),
            "generator_rest_efficiency": (
                filled["generator_all"] - filled["generator_1"]
            ) / generator_gas_total.replace(0, np.nan),
        }
        for name, efficiency in efficiency_series.items():
            feature_values[f"feat_{name}"] = efficiency
            history_series[name] = (efficiency, False)

    for name, (series, include_range) in history_series.items():
        _add_history_features(feature_values, series, name, config, include_range=include_range)
        if config.enable_long_cycle_features and name in {
            "generator_1",
            "generator_all",
            "generator_rest",
        }:
            same_slots = pd.concat(
                [series.shift(96 * day) for day in range(1, 15)], axis=1
            )
            feature_values[f"feat_{name}_same_slot_median_14d"] = same_slots.median(axis=1)
            for days in (3, 7, 14):
                baseline = same_slots.iloc[:, :days].mean(axis=1)
                feature_values[f"feat_{name}_same_slot_mean_{days}d"] = baseline
                feature_values[f"feat_{name}_vs_same_slot_mean_{days}d"] = series - baseline

    if config.enable_target_aligned_features:
        for target in ("generator_1", "generator_all"):
            if target in filled:
                _add_target_aligned_features(feature_values, filled[target], target, config)

    if config.enable_rich_quantile_features:
        rich_quantile_series: dict[str, pd.Series] = {
            "generator_gas_total": generator_gas_total,
            "bf_surplus_proxy": feature_values["feat_bf_surplus_proxy"],
            **balances,
        }
        if "generator_1" in filled:
            rich_quantile_series["generator_1"] = filled["generator_1"]
        if "generator_all" in filled:
            rich_quantile_series["generator_all"] = filled["generator_all"]
        if "feat_generator_rest" in feature_values:
            rich_quantile_series["generator_rest"] = feature_values["feat_generator_rest"]
        if "blast_furnace_gas_holder_2" in filled:
            rich_quantile_series["gas_holder"] = filled["blast_furnace_gas_holder_2"]
        for name, series in rich_quantile_series.items():
            _add_rich_quantile_features(feature_values, series, name, config)

    if config.enable_rich_ramp_state_features:
        rich_ramp_series: dict[str, pd.Series] = {}
        if "generator_1" in filled:
            rich_ramp_series["generator_1"] = filled["generator_1"]
        if "generator_all" in filled:
            rich_ramp_series["generator_all"] = filled["generator_all"]
        if "feat_generator_rest" in feature_values:
            rich_ramp_series["generator_rest"] = feature_values["feat_generator_rest"]
        for name, series in rich_ramp_series.items():
            _add_rich_ramp_state_features(feature_values, series, name)

    if config.enable_ramp_features and "generator_1" in filled:
        _add_ramp_features(feature_values, filled["generator_1"])

    if config.relation_features:
        _add_relation_features(feature_values, filled, config.relation_features)

    _add_dynamic_features(feature_values, filled, config)

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
    if config.enable_slot_one_hot:
        slot = index.hour * 4 + index.minute // 15
        for value in range(96):
            feature_values[f"feat_slot_{value}"] = (slot == value).astype("int8")
        weekday = index.dayofweek
        for value in range(7):
            feature_values[f"feat_weekday_{value}"] = (weekday == value).astype("int8")
        feature_values["feat_is_weekend"] = (weekday >= 5).astype("int8")
    if config.enable_time_fourier:
        week_minutes = index.dayofweek * 1440 + minute_of_day
        for harmonic in range(1, 5):
            phase = 2 * np.pi * harmonic * minute_of_day / 1440
            feature_values[f"feat_day_fourier_sin_{harmonic}"] = np.sin(phase)
            feature_values[f"feat_day_fourier_cos_{harmonic}"] = np.cos(phase)
        for harmonic in range(1, 3):
            phase = 2 * np.pi * harmonic * week_minutes / (7 * 1440)
            feature_values[f"feat_week_fourier_sin_{harmonic}"] = np.sin(phase)
            feature_values[f"feat_week_fourier_cos_{harmonic}"] = np.cos(phase)

    if price_schedule is not None:
        current_price = price_schedule.lookup(index)
        if config.enable_price_delta_features or config.enable_price_interactions:
            feature_values["feat_current_price"] = current_price
        prices: list[np.ndarray] = []
        for horizon in config.horizons:
            target_index = index + pd.Timedelta(minutes=15 * horizon)
            price = price_schedule.lookup(target_index)
            prices.append(price)
            feature_values[f"feat_target_price_tplus_{15 * horizon}"] = price
        price_matrix = np.column_stack(prices)
        changed = price_matrix != current_price[:, None]
        first_change = np.where(changed.any(axis=1), changed.argmax(axis=1) + 1, 0)
        feature_values["feat_price_switch_within_120"] = changed.any(axis=1).astype("int8")
        feature_values["feat_steps_to_price_switch"] = first_change.astype("int8")
        if config.enable_price_delta_features:
            denominator = np.where(np.abs(current_price) > 1e-6, current_price, np.nan)
            for step, horizon in enumerate(config.horizons):
                minutes = 15 * horizon
                delta = price_matrix[:, step] - current_price
                feature_values[f"feat_price_delta_tplus_{minutes}"] = delta
                feature_values[f"feat_price_ratio_tplus_{minutes}"] = (
                    price_matrix[:, step] / denominator
                )
                feature_values[f"feat_price_direction_tplus_{minutes}"] = np.sign(delta).astype(
                    "int8"
                )
            feature_values["feat_next_2h_price_max_delta"] = (
                price_matrix.max(axis=1) - current_price
            )
            feature_values["feat_next_2h_price_min_delta"] = (
                price_matrix.min(axis=1) - current_price
            )
        if config.enable_price_interactions:
            for step, horizon in enumerate(config.horizons):
                minutes = 15 * horizon
                delta = price_matrix[:, step] - current_price
                if "generator_1" in filled:
                    feature_values[f"feat_generator1_price_delta_tplus_{minutes}"] = (
                        filled["generator_1"] * delta
                    )
                feature_values[f"feat_generator_gas_total_price_delta_tplus_{minutes}"] = (
                    generator_gas_total * delta
                )
                holder = filled.get("blast_furnace_gas_holder_2")
                if holder is not None:
                    feature_values[f"feat_gas_holder_price_delta_tplus_{minutes}"] = holder * delta
                slope = feature_values.get("feat_generator_1_slope_4")
                if slope is not None:
                    feature_values[f"feat_generator1_slope_price_delta_tplus_{minutes}"] = (
                        slope * delta
                    )

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
        elif "_price_delta_tplus_" in column or "_price_ratio_tplus_" in column or (
            "_price_direction_tplus_" in column
        ):
            minutes = int(column.rsplit("_", 1)[-1])
            metadata[column] = FeatureAvailability(minutes, known_in_advance=True)
        elif column.startswith("feat_next_2h_price_"):
            metadata[column] = FeatureAvailability(120, known_in_advance=True)
        elif "_lag_" in column:
            tail = column.rsplit("_lag_", 1)[-1]
            steps = int(tail.split("_", 1)[0])
            metadata[column] = FeatureAvailability(-15 * steps)
        elif column.startswith("feat_relation_"):
            match = re.search(r"_lag_(\d+)_h\d+$", column)
            steps = int(match.group(1)) if match else 0
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
