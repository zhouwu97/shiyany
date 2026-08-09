"""A64 直接增量模型。

本模块是一个独立的研究入口，不改写正式模型。每个 ``target x horizon``
拥有自己的模型，标签恒为 ``y[t+h] - y[t]``。所有特征在原点只使用
``timestamp <= origin`` 的生产观测；训练期的缺失填补、标准化和增量裁剪
统计均在当前折训练集内拟合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gas_forecast.config import ForecastConfig
from gas_forecast.scoring import competition_mape
from gas_forecast.splits import TimeFold, assert_label_safe_fold, make_outer_folds


TARGETS: tuple[str, str] = ("generator_1", "generator_all")
HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
MODEL_NAMES: tuple[str, str] = ("ridge", "lightgbm")


@dataclass(frozen=True)
class DirectDeltaConfig:
    """A64 固定配置；不提供连续调参接口，避免筛选过拟合。"""

    targets: tuple[str, ...] = TARGETS
    horizons: tuple[int, ...] = HORIZONS
    ridge_alpha: float = 20.0
    lgb_n_estimators: int = 120
    lgb_learning_rate: float = 0.03
    lgb_num_leaves: int = 15
    lgb_max_depth: int = 5
    lgb_min_child_samples: int = 40
    random_state: int = 20250731
    inner_folds: int = 3
    min_train_rows: int = 96
    enable_nonlinear_state: bool = True
    allow_future_price: bool = False

    def __post_init__(self) -> None:
        if tuple(self.targets) != TARGETS:
            raise ValueError("A64 固定使用 generator_1 和 generator_all 两个目标")
        if tuple(self.horizons) != HORIZONS:
            raise ValueError("A64 固定使用 1..8 共八个步长")
        if self.ridge_alpha <= 0 or self.lgb_n_estimators < 1:
            raise ValueError("模型参数必须为正数")
        if self.inner_folds < 2 or self.min_train_rows < 16:
            raise ValueError("nested cross-fitting 参数过小")


@dataclass(frozen=True)
class FeatureBuildResult:
    """特征及其可用性收据。"""

    frame: pd.DataFrame
    columns: tuple[str, ...]
    price_enabled: bool
    price_reason: str
    source_columns: tuple[str, ...]


@dataclass
class _ModelState:
    model: object
    imputer: SimpleImputer
    lower: float
    upper: float
    feature_columns: tuple[str, ...]


def _as_numeric_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """返回指定字段的数值序列；缺失字段保持全 NaN 且不改变 schema。"""

    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=frame.index, dtype=float, name=name)


def _sum_matching(frame: pd.DataFrame, needles: Sequence[str]) -> pd.Series:
    columns = [
        column
        for column in frame.columns
        if any(token in str(column).lower() for token in needles)
    ]
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def _history_features(output: dict[str, pd.Series], series: pd.Series, prefix: str) -> None:
    """为一个状态量添加固定、严格向后的动态统计。"""

    current = series
    for lag in (1, 2, 3, 4, 8, 16):
        output[f"{prefix}_lag{lag}"] = current.shift(lag)
    for lag in (1, 2, 4):
        output[f"{prefix}_diff{lag}"] = current - current.shift(lag)
    diff1 = current - current.shift(1)
    output[f"{prefix}_second_diff"] = diff1 - diff1.shift(1)
    for window in (4, 8, 16):
        output[f"{prefix}_slope{window}"] = (
            current - current.shift(window - 1)
        ) / float(window - 1)
        output[f"{prefix}_std{window}"] = current.shift(1).rolling(
            window, min_periods=max(2, window // 2)
        ).std()
    output[f"{prefix}_ewma_level"] = current.ewm(
        span=8, adjust=False, min_periods=2
    ).mean()
    trend = current.ewm(span=8, adjust=False, min_periods=3).mean()
    output[f"{prefix}_ewma_trend"] = trend - trend.shift(1)


def _price_columns(
    index: pd.DatetimeIndex,
    price_schedule: object | None,
    *,
    allow_future_price: bool,
    price_known_in_advance: bool,
) -> tuple[dict[str, np.ndarray], bool, str]:
    """只在调用方同时给出显式证明时启用未来电价。"""

    if price_schedule is None:
        return {}, False, "未提供价格计划"
    if not allow_future_price:
        return {}, False, "配置明确禁用未来电价"
    if not price_known_in_advance:
        return {}, False, "缺少 origin 已知的严格证明，未来电价禁用"
    lookup = getattr(price_schedule, "lookup", None)
    if not callable(lookup):
        raise TypeError("price_schedule 必须提供 lookup(DatetimeIndex) 方法")
    values: dict[str, np.ndarray] = {}
    for horizon in HORIZONS:
        target_index = index + pd.to_timedelta(15 * horizon, unit="min")
        values[f"known_future_price_{15 * horizon}"] = np.asarray(
            lookup(target_index), dtype=float
        )
    return values, True, "调用方提供了 origin 已知的价格证明"


def build_direct_delta_features(
    frame: pd.DataFrame,
    *,
    price_schedule: object | None = None,
    allow_future_price: bool = False,
    price_known_in_advance: bool = False,
    include_nonlinear_state: bool = True,
    return_metadata: bool = False,
) -> pd.DataFrame | FeatureBuildResult:
    """构造 A64 特征。

    ``frame`` 可以包含训练和生产输入的联合时间轴。函数只用当前行及其
    左侧历史行；未来行被删除、打乱、置空或改值不会影响任何已存在原点。
    """

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("输入必须使用 DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("时间轴必须严格递增且唯一")
    numeric = frame.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce")
    if numeric.empty:
        raise ValueError("输入至少需要一个数值生产字段")
    filled = numeric.ffill()
    values: dict[str, pd.Series | np.ndarray] = {}

    # 当前目标和当前 generator gas usage 是唯一直接读取的生产原点值。
    source_names = ["generator_1", "generator_all"]
    source_names.extend(
        sorted(
            column
            for column in numeric.columns
            if str(column).lower().startswith("generator_use_")
        )
    )
    holder_name = next(
        (column for column in numeric.columns if "gas_holder" in str(column).lower()),
        None,
    )
    if holder_name is not None:
        source_names.append(holder_name)
    source_names = list(dict.fromkeys(source_names))
    for name in source_names:
        series = _as_numeric_series(filled, name)
        values[f"{name}_current"] = series
        _history_features(values, series, name)

    # 供需错配只由当前已观测的生产、优先用户和发电用气构造。
    supply = _sum_matching(
        filled,
        ("blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5", "coke_oven", "converter_1"),
    )
    demand = _sum_matching(
        filled,
        ("air_heater", "blast_furnace_user", "into_gas_mixed", "converter_user", "generator_use_"),
    )
    mismatch = supply - demand
    values["gas_supply_demand_mismatch"] = mismatch
    _history_features(values, mismatch, "gas_supply_demand_mismatch")
    if holder_name is not None:
        holder = _as_numeric_series(filled, holder_name)
        values["holder_momentum"] = holder - holder.shift(1)
        values["holder_momentum_accel"] = values["holder_momentum"] - values["holder_momentum"].shift(1)
    else:
        values["holder_momentum"] = pd.Series(np.nan, index=numeric.index)
        values["holder_momentum_accel"] = pd.Series(np.nan, index=numeric.index)

    # 状态/非线性项仅是原点状态的确定变换，禁止用未来真值做门控。
    if include_nonlinear_state:
        values["generator_1_abs_diff1"] = values["generator_1_diff1"].abs()
        values["generator_all_abs_diff1"] = values["generator_all_diff1"].abs()
        values["generator_1_ramp_up"] = (values["generator_1_diff1"] > 0).astype("int8")
        values["generator_1_ramp_down"] = (values["generator_1_diff1"] < 0).astype("int8")
        values["gas_mismatch_positive"] = (mismatch > 0).astype("int8")
        values["gas_mismatch_abs"] = mismatch.abs()
        values["generator_1_x_gas_mismatch"] = (
            values["generator_1_current"] * mismatch
        )

    price_values, price_enabled, price_reason = _price_columns(
        numeric.index,
        price_schedule,
        allow_future_price=allow_future_price,
        price_known_in_advance=price_known_in_advance,
    )
    if price_enabled:
        values.update(price_values)
    result = pd.DataFrame(values, index=numeric.index)
    # 保证列顺序和类型稳定；无穷值在训练时由 imputer 统一处理。
    result = result.replace([np.inf, -np.inf], np.nan).astype(float)
    if return_metadata:
        return FeatureBuildResult(
            frame=result,
            columns=tuple(result.columns),
            price_enabled=price_enabled,
            price_reason=price_reason,
            source_columns=tuple(source_names),
        )
    return result


def build_direct_delta_targets(
    frame: pd.DataFrame,
    targets: Sequence[str] = TARGETS,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """构造严格的绝对增量标签 ``y[t+h] - y[t]``。"""

    labels: dict[str, pd.Series] = {}
    for target in targets:
        if target not in frame:
            raise ValueError(f"缺少目标字段: {target}")
        current = pd.to_numeric(frame[target], errors="coerce")
        for horizon in horizons:
            if int(horizon) <= 0:
                raise ValueError("horizon 必须为正整数")
            labels[f"{target}_delta_{int(horizon)}"] = current.shift(-int(horizon)) - current
    return pd.DataFrame(labels, index=frame.index)


def _target_column(target: str, horizon: int) -> str:
    return f"{target}_delta_{int(horizon)}"


def _make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _make_lgb(config: DirectDeltaConfig, seed_offset: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=config.lgb_n_estimators,
        learning_rate=config.lgb_learning_rate,
        num_leaves=config.lgb_num_leaves,
        max_depth=config.lgb_max_depth,
        min_child_samples=config.lgb_min_child_samples,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=config.random_state + seed_offset,
        n_jobs=1,
        verbosity=-1,
    )


def _fit_state(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    config: DirectDeltaConfig,
    seed_offset: int,
) -> _ModelState:
    valid = y.notna()
    if int(valid.sum()) < config.min_train_rows:
        raise ValueError(f"有效增量训练样本不足: {int(valid.sum())}")
    matrix = x.loc[valid]
    target = y.loc[valid].to_numpy(dtype=float)
    lower = float(np.nanquantile(target, 0.001))
    upper = float(np.nanquantile(target, 0.999))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("训练期增量裁剪统计无效")
    if model_name == "ridge":
        model: object = _make_ridge(config.ridge_alpha)
        model.fit(matrix, target)
        imputer = model.named_steps["imputer"]
    elif model_name == "lightgbm":
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        imputed = imputer.fit_transform(matrix)
        model = _make_lgb(config, seed_offset)
        model.fit(imputed, target)
    else:
        raise ValueError(f"A64 仅支持 Ridge/LightGBM，收到 {model_name}")
    return _ModelState(
        model=model,
        imputer=imputer,
        lower=lower,
        upper=upper,
        feature_columns=tuple(x.columns),
    )


def _predict_state(state: _ModelState, x: pd.DataFrame) -> np.ndarray:
    matrix = x.reindex(columns=state.feature_columns)
    if isinstance(state.model, Pipeline):
        prediction = state.model.predict(matrix)
    else:
        prediction = state.model.predict(state.imputer.transform(matrix))
    return np.clip(np.asarray(prediction, dtype=float), state.lower, state.upper)


class DirectDeltaForecaster:
    """16 个独立的目标×步长直接增量模型。"""

    version = "a64_direct_delta"

    def __init__(self, config: DirectDeltaConfig | None = None) -> None:
        self.config = config or DirectDeltaConfig()
        self.feature_columns_: tuple[str, ...] = ()
        self.states_: dict[tuple[str, int, str], _ModelState] = {}
        self.trace_: list[dict[str, object]] = []
        self.last_prediction_metadata_: dict[str, object] = {}

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
        *,
        train_end: pd.Timestamp | None = None,
        model_names: Iterable[str] = MODEL_NAMES,
    ) -> "DirectDeltaForecaster":
        """在给定训练边界内拟合；不读取边界之后的任何标签或统计量。"""

        names = tuple(dict.fromkeys(model_names))
        if not names or any(name not in MODEL_NAMES for name in names):
            raise ValueError("model_names 只能包含 ridge 和 lightgbm")
        index = features.index
        if train_end is not None:
            mask = index <= pd.Timestamp(train_end)
            features = features.loc[mask]
            deltas = deltas.loc[mask]
            current = current.loc[mask]
        self.feature_columns_ = tuple(features.columns)
        self.states_.clear()
        for target_i, target in enumerate(self.config.targets):
            if target not in current:
                raise ValueError(f"current 缺少目标字段: {target}")
            for horizon_i, horizon in enumerate(self.config.horizons):
                column = _target_column(target, horizon)
                if column not in deltas:
                    raise ValueError(f"deltas 缺少列: {column}")
                for model_i, model_name in enumerate(names):
                    state = _fit_state(
                        features,
                        deltas[column],
                        model_name,
                        self.config,
                        target_i * 100 + horizon_i * 10 + model_i,
                    )
                    self.states_[(target, horizon, model_name)] = state
                    self.trace_.append(
                        {
                            "event": "fit",
                            "target": target,
                            "horizon": 15 * horizon,
                            "model": model_name,
                            "train_end": str(train_end) if train_end is not None else None,
                            "train_rows": int(len(features)),
                            "feature_count": int(len(self.feature_columns_)),
                        }
                    )
        if len(self.states_) != len(self.config.targets) * len(self.config.horizons) * len(names):
            raise RuntimeError("A64 模型数量不符合登记范围")
        return self

    def predict_deltas(
        self,
        features: pd.DataFrame,
        *,
        model_name: str = "ridge",
    ) -> pd.DataFrame:
        if not self.states_:
            raise RuntimeError("A64 尚未训练")
        if model_name not in MODEL_NAMES:
            raise ValueError("model_name 只能是 ridge 或 lightgbm")
        output: dict[str, np.ndarray] = {}
        for target in self.config.targets:
            for horizon in self.config.horizons:
                key = (target, horizon, model_name)
                if key not in self.states_:
                    raise RuntimeError(f"模型未训练: {key}")
                output[f"{target}_delta_{horizon}_{model_name}"] = _predict_state(
                    self.states_[key], features
                )
        return pd.DataFrame(output, index=features.index)

    def predict(
        self,
        features: pd.DataFrame,
        current: pd.DataFrame,
        *,
        model_name: str = "ridge",
    ) -> pd.DataFrame:
        """将预测增量加回当前值，返回统一的绝对预测列。"""

        deltas = self.predict_deltas(features, model_name=model_name)
        output: dict[str, np.ndarray] = {}
        for target in self.config.targets:
            anchor = pd.to_numeric(current[target], errors="coerce").ffill().to_numpy(dtype=float)
            for horizon in self.config.horizons:
                delta = deltas[f"{target}_delta_{horizon}_{model_name}"].to_numpy()
                output[f"{target}_t+{15 * horizon}_pred"] = anchor + delta
        result = pd.DataFrame(output, index=features.index)
        _apply_capacity_constraints(result)
        return result

    def predict_at_origin(
        self,
        history_until_origin: pd.DataFrame,
        *,
        model_name: str = "ridge",
        price_schedule: object | None = None,
        price_known_in_advance: bool = False,
    ) -> pd.DataFrame:
        """只以当前 origin 及其历史构造一行 A64 预测。

        这是 P3 正式集成使用的入口。它不接受整段评分期特征矩阵，因而未来
        generator、gas、holder 或 users 行无法进入特征构造或 ``predict``。
        """

        if not isinstance(history_until_origin.index, pd.DatetimeIndex):
            raise TypeError("history_until_origin 必须使用 DatetimeIndex")
        if (
            history_until_origin.empty
            or not history_until_origin.index.is_monotonic_increasing
            or not history_until_origin.index.is_unique
        ):
            raise ValueError("history_until_origin 时间轴必须非空、唯一且递增")
        if self.config.allow_future_price and price_schedule is not None and not price_known_in_advance:
            raise ValueError("A64 未提供 origin 已知证明时禁止未来电价")
        origin = pd.Timestamp(history_until_origin.index[-1])
        features = build_direct_delta_features(
            history_until_origin,
            price_schedule=price_schedule,
            allow_future_price=self.config.allow_future_price,
            price_known_in_advance=price_known_in_advance,
            include_nonlinear_state=self.config.enable_nonlinear_state,
        )
        if not isinstance(features, pd.DataFrame):
            raise RuntimeError("A64 单 origin 特征构造没有返回 DataFrame")
        prediction = self.predict(
            features.loc[[origin]],
            history_until_origin.loc[[origin], list(self.config.targets)],
            model_name=model_name,
        )
        self.last_prediction_metadata_ = {
            "origin": origin,
            "history_rows": int(len(history_until_origin)),
            "used_future_observations": False,
            "model": model_name,
        }
        return prediction


def _apply_capacity_constraints(output: pd.DataFrame) -> None:
    """沿用物理容量硬约束，仅影响最终绝对预测，不改变增量训练目标。"""

    for horizon in HORIZONS:
        g1 = f"generator_1_t+{15 * horizon}_pred"
        gall = f"generator_all_t+{15 * horizon}_pred"
        if g1 in output:
            output[g1] = output[g1].clip(0.0, 200.0)
        if gall in output:
            output[gall] = output[gall].clip(0.0, 440.0)
        if g1 in output and gall in output:
            output[gall] = np.maximum(output[gall], output[g1])
            output[gall] = np.minimum(output[gall], output[g1] + 240.0)


def _fold_rows(
    frame: pd.DataFrame,
    fold: TimeFold,
    predictions: Mapping[str, pd.DataFrame],
    config: DirectDeltaConfig,
) -> pd.DataFrame:
    _, validation_mask = fold.masks(frame.index)
    origins = frame.index[validation_mask]
    records: list[pd.DataFrame] = []
    for target in config.targets:
        current = pd.to_numeric(frame.loc[origins, target], errors="coerce")
        for horizon in config.horizons:
            actual = pd.to_numeric(frame[target].shift(-horizon).loc[origins], errors="coerce")
            valid = current.notna() & actual.notna()
            if not valid.any():
                continue
            part = pd.DataFrame(
                {
                    "fold": fold.name,
                    "origin_time": origins[valid],
                    "train_end": fold.train_end,
                    "target": target,
                    "horizon": 15 * horizon,
                    "actual": actual[valid].to_numpy(dtype=float),
                    "current_value": current[valid].to_numpy(dtype=float),
                    "actual_delta": (actual[valid] - current[valid]).to_numpy(dtype=float),
                }
            )
            for name, prediction in predictions.items():
                column = f"{target}_t+{15 * horizon}_pred"
                part[f"{name}_prediction"] = prediction.loc[origins[valid], column].to_numpy(dtype=float)
            records.append(part)
    if not records:
        raise ValueError(f"折 {fold.name} 没有可评分标签")
    return pd.concat(records, ignore_index=True)


def _inner_fold_trace(
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    config: DirectDeltaConfig,
) -> list[dict[str, object]]:
    """固定模型的 nested cross-fitting 收据；选择只读取训练期 OOF。"""

    if len(train_index) < config.min_train_rows + config.inner_folds * 8:
        return [{"event": "inner_skip", "reason": "训练样本不足", "rows": int(len(train_index))}]
    boundaries = np.linspace(config.min_train_rows, len(train_index), config.inner_folds + 1, dtype=int)
    trace: list[dict[str, object]] = []
    for position in range(config.inner_folds):
        valid_start = int(boundaries[position])
        valid_end = int(boundaries[position + 1])
        if valid_end <= valid_start:
            continue
        # 内折训练集在验证块前保留最大步长 purge。
        fit_end = valid_start - max(config.horizons) - 1
        if fit_end < config.min_train_rows:
            trace.append({"event": "inner_skip", "inner_fold": position + 1, "reason": "purge 后训练不足"})
            continue
        fit_idx = train_index[:fit_end]
        valid_idx = train_index[valid_start:valid_end]
        # inner OOF 的标签也必须在外层训练历史内结束，不能借用外层
        # train_end 之后、哪怕尚未进入 held fold 的生产真值。
        last_safe_origin = train_index[-1] - pd.Timedelta(minutes=15 * max(config.horizons))
        valid_idx = valid_idx[valid_idx <= last_safe_origin]
        if len(valid_idx) == 0:
            trace.append(
                {
                    "event": "inner_skip",
                    "inner_fold": position + 1,
                    "reason": "没有标签完全位于外层训练历史内的验证原点",
                }
            )
            continue
        for target in config.targets:
            for horizon in config.horizons:
                column = _target_column(target, horizon)
                actual_delta = deltas.loc[valid_idx, column]
                valid = actual_delta.notna()
                if not valid.any():
                    continue
                for model_name in MODEL_NAMES:
                    state = _fit_state(
                        features.loc[fit_idx],
                        deltas.loc[fit_idx, column],
                        model_name,
                        config,
                        position * 100 + horizon,
                    )
                    pred = _predict_state(state, features.loc[valid_idx[valid]])
                    trace.append(
                        {
                            "event": "inner_score",
                            "inner_fold": position + 1,
                            "target": target,
                            "horizon": 15 * horizon,
                            "model": model_name,
                            "train_end": str(fit_idx[-1]),
                            "origin_min": str(valid_idx[0]),
                            "rows": int(valid.sum()),
                            "delta_mape": competition_mape(
                                actual_delta[valid].to_numpy(dtype=float), pred
                            ),
                        }
                    )
    return trace


def build_direct_delta_oof(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    config: DirectDeltaConfig | None = None,
    folds: Sequence[TimeFold] | None = None,
    include_blind: bool = False,
    nested: bool = True,
    origin_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """严格滚动 OOF；默认只使用 development 折，永不自动读取 blind。"""

    config = config or DirectDeltaConfig()
    if not frame.index.equals(features.index):
        raise ValueError("frame 与 features 时间轴不一致")
    deltas = build_direct_delta_targets(frame, config.targets, config.horizons)
    if include_blind:
        raise ValueError("A64 路线不读取 blind 标签；只允许 development OOF")
    if origin_only and config.allow_future_price:
        raise ValueError("A64 正式逐 origin OOF 不允许未来电价")
    all_folds = (
        list(folds)
        if folds is not None
        else make_outer_folds(frame.index, ForecastConfig())
    )
    selected_folds = [fold for fold in all_folds if not fold.blind]
    if not selected_folds:
        raise ValueError("没有可用的 development 折")
    rows_parts: list[pd.DataFrame] = []
    trace: list[dict[str, object]] = []
    for fold in selected_folds:
        assert_label_safe_fold(fold, max(config.horizons))
        train_mask, validation_mask = fold.masks(frame.index)
        train_index = frame.index[train_mask]
        if len(train_index) < config.min_train_rows:
            trace.append({"event": "fold_skip", "fold": fold.name, "reason": "训练样本不足"})
            continue
        train_features = (
            build_direct_delta_features(
                frame.loc[train_mask],
                include_nonlinear_state=config.enable_nonlinear_state,
            )
            if origin_only
            else features.loc[train_mask]
        )
        if not isinstance(train_features, pd.DataFrame):
            raise RuntimeError("A64 训练特征构造没有返回 DataFrame")
        model = DirectDeltaForecaster(config).fit(
            train_features,
            deltas.loc[train_mask],
            frame.loc[train_mask, list(config.targets)],
            train_end=fold.train_end,
        )
        validation_origins = frame.index[validation_mask]
        if origin_only:
            predictions = {
                name: pd.concat(
                    [
                        model.predict_at_origin(frame.loc[:origin], model_name=name)
                        for origin in validation_origins
                    ],
                    axis=0,
                )
                for name in MODEL_NAMES
            }
        else:
            predictions = {
                name: model.predict(
                    features.loc[validation_mask],
                    frame.loc[validation_mask, list(config.targets)],
                    model_name=name,
                )
                for name in MODEL_NAMES
            }
        rows_parts.append(_fold_rows(frame, fold, predictions, config))
        trace.extend({"fold": fold.name, **item} for item in model.trace_)
        if nested:
            trace.extend({"fold": fold.name, **item} for item in _inner_fold_trace(
                features, deltas, train_index, config
            ))
    if not rows_parts:
        raise ValueError("没有生成 OOF 行")
    rows = pd.concat(rows_parts, ignore_index=True).sort_values(
        ["origin_time", "target", "horizon", "fold"], kind="stable"
    ).reset_index(drop=True)
    reports: dict[str, object] = {}
    for model_name in MODEL_NAMES:
        reports[model_name] = _score_candidate(rows, f"{model_name}_prediction")
    report: dict[str, object] = {
        "version": DirectDeltaForecaster.version,
        "targets": list(config.targets),
        "horizons": [15 * h for h in config.horizons],
        "models": reports,
        "folds": [fold.name for fold in selected_folds],
        "blind_included": False,
        "blind_labels_used": False,
        "nested_cross_fitting": bool(nested),
        "origin_only_prediction": bool(origin_only),
        "trace_rows": int(len(trace)),
        "trace": trace,
    }
    return rows, report


def _score_candidate(rows: pd.DataFrame, prediction_column: str) -> dict[str, object]:
    scored = rows.copy()
    scored["ape"] = np.abs(scored["actual"] - scored[prediction_column]) / np.maximum(
        np.abs(scored["actual"]), 1e-6
    )
    by_fold = scored.groupby("fold", sort=True)["ape"].mean()
    by_target = scored.groupby("target", sort=True)["ape"].mean()
    by_horizon = scored.groupby("horizon", sort=True)["ape"].mean()
    return {
        "rows": int(len(scored)),
        "pooled_mape": float(scored["ape"].mean()),
        "by_fold": {str(k): float(v) for k, v in by_fold.items()},
        "by_target": {str(k): float(v) for k, v in by_target.items()},
        "by_horizon": {f"t+{int(k)}": float(v) for k, v in by_horizon.items()},
    }


def screen_direct_delta(
    rows: pd.DataFrame,
    *,
    candidate: str = "lightgbm",
    parent: str = "ridge",
    min_improvement_pp: float = 0.02,
    min_wins: int = 3,
) -> dict[str, object]:
    """按预注册前五折门槛决定 PASS/STOP，不读取 blind 或平台成绩。"""

    candidate_column = f"{candidate}_prediction"
    parent_column = f"{parent}_prediction"
    required = {"fold", candidate_column, parent_column, "actual"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"screening 输入缺少列: {missing}")
    dev = rows.loc[rows["fold"].astype(str).str.lower().ne("blind")].copy()
    folds = sorted(dev["fold"].astype(str).unique())[:5]
    subset = dev.loc[dev["fold"].astype(str).isin(folds)]
    if len(folds) < 5:
        return {
            "status": "STOP",
            "reason": "development 折少于五折",
            "folds": folds,
            "pooled_improvement_pp": float("nan"),
            "fold_wins": 0,
        }
    # screening 的预注册门槛只依赖 pooled 与逐折胜负，不能因完整
    # OOF 诊断字段（target/horizon）缺失而改变判定。
    candidate_ape = np.abs(subset["actual"] - subset[candidate_column]) / np.maximum(
        np.abs(subset["actual"]), 1e-6
    )
    parent_ape = np.abs(subset["actual"] - subset[parent_column]) / np.maximum(
        np.abs(subset["actual"]), 1e-6
    )
    candidate_score = float(candidate_ape.mean())
    parent_score = float(parent_ape.mean())
    improvement_pp = (float(parent_score) - float(candidate_score)) * 100.0
    candidate_by_fold = subset.assign(
        _candidate=np.abs(subset["actual"] - subset[candidate_column])
        / np.maximum(np.abs(subset["actual"]), 1e-6),
        _parent=np.abs(subset["actual"] - subset[parent_column])
        / np.maximum(np.abs(subset["actual"]), 1e-6),
    ).groupby("fold", sort=True)[["_candidate", "_parent"]].mean()
    wins = int((candidate_by_fold["_candidate"] < candidate_by_fold["_parent"]).sum())
    passed = improvement_pp >= min_improvement_pp and wins >= min_wins
    return {
        "status": "PASS" if passed else "STOP",
        "reason": "达到 pooled 改善和胜折门槛" if passed else "未达到 pooled 改善 0.02pp 且至少三胜",
        "folds": folds,
        "candidate": candidate,
        "parent": parent,
        "pooled_improvement_pp": float(improvement_pp),
        "fold_wins": wins,
        "min_improvement_pp": float(min_improvement_pp),
        "min_wins": int(min_wins),
        "fold_improvement_pp": {
            str(k): float((row["_parent"] - row["_candidate"]) * 100.0)
            for k, row in candidate_by_fold.iterrows()
        },
    }


def audit_direct_delta_future_perturbations(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    model: DirectDeltaForecaster,
    *,
    origins: Sequence[pd.Timestamp] | None = None,
    random_state: int = 20250731,
    model_name: str = "ridge",
) -> dict[str, object]:
    """验证 extreme/shuffle/null/delete 后 16 个预测逐元素不变。"""

    selected = list(origins) if origins is not None else list(frame.index[:: max(1, len(frame) // 16)])
    selected = [pd.Timestamp(value) for value in selected if value in frame.index]
    methods = ("extreme", "shuffle", "null", "delete")
    failures: list[dict[str, object]] = []
    baseline = model.predict(
        features.loc[selected],
        frame.loc[selected, list(model.config.targets)],
        model_name=model_name,
    )
    rng = np.random.default_rng(random_state)
    for origin in selected:
        changed = frame.copy()
        future = changed.index > origin
        numeric_columns = list(changed.select_dtypes(include=[np.number]).columns)
        for method in methods:
            perturbed = changed.copy()
            if method == "extreme":
                perturbed.loc[future, numeric_columns] = -999999.0
            elif method == "shuffle":
                block = perturbed.loc[future, numeric_columns].to_numpy(copy=True)
                if len(block):
                    perturbed.loc[future, numeric_columns] = block[rng.permutation(len(block))]
            elif method == "null":
                perturbed.loc[future, numeric_columns] = np.nan
            else:
                perturbed = perturbed.loc[perturbed.index <= origin].copy()
            changed_features = build_direct_delta_features(
                perturbed,
                include_nonlinear_state=model.config.enable_nonlinear_state,
            )
            changed_rows = model.predict(
                changed_features.loc[[origin]],
                perturbed.loc[[origin], list(model.config.targets)],
                model_name=model_name,
            )
            base_row = baseline.loc[[origin]]
            equal = np.isclose(base_row.to_numpy(), changed_rows.to_numpy(), equal_nan=True).all()
            if not equal:
                failures.append({"origin": str(origin), "method": method})
    return {
        "passed": not failures,
        "origins": int(len(selected)),
        "methods": list(methods),
        "cases_checked": int(len(selected) * len(methods)),
        "failures": failures,
    }


# 便于实验脚本和外部审计使用的稳定别名。
make_direct_delta_features = build_direct_delta_features
make_direct_delta_targets = build_direct_delta_targets
DirectDeltaModel = DirectDeltaForecaster


__all__ = [
    "DirectDeltaConfig",
    "FeatureBuildResult",
    "DirectDeltaForecaster",
    "DirectDeltaModel",
    "build_direct_delta_features",
    "make_direct_delta_features",
    "build_direct_delta_targets",
    "make_direct_delta_targets",
    "build_direct_delta_oof",
    "screen_direct_delta",
    "audit_direct_delta_future_perturbations",
]
