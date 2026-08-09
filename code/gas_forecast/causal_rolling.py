"""严格因果的 15 分钟滚动重建预测器。

本模块故意不接入公共注册表或提交链路。它的唯一正式推理入口是
``predict_at_origin(history_until_origin)``：调用方只能提供截至当前 origin 的
生产历史。模型在每个 15 分钟步长后仅使用自己生成的机组状态和最后可见的
资源状态推进，绝不读取未来的 generator、煤气、holder 或用户真值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gas_forecast.config import ForecastConfig
from gas_forecast.splits import TimeFold, assert_label_safe_fold, make_outer_folds


TARGETS: tuple[str, str] = ("generator_1", "generator_all")
HORIZONS: tuple[int, ...] = tuple(range(1, 9))
FEATURE_VERSION = "causal_current_state_v1"


@dataclass(frozen=True)
class CausalRollingConfig:
    """滚动重建的固定、可审计配置。"""

    targets: tuple[str, ...] = TARGETS
    horizons: tuple[int, ...] = HORIZONS
    step_minutes: int = 15
    ridge_alpha: float = 20.0
    min_train_rows: int = 64
    min_history_rows: int = 17
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999
    train_window_rows: int | None = None
    generator_1_capacity: float = 200.0
    generator_all_capacity: float = 440.0
    generator_extra_capacity: float = 240.0

    def __post_init__(self) -> None:
        if tuple(self.targets) != TARGETS:
            raise ValueError("滚动重建固定使用 generator_1 和 generator_all")
        if tuple(self.horizons) != HORIZONS:
            raise ValueError("滚动重建固定输出 1..8 共八个 15 分钟步长")
        if self.step_minutes != 15:
            raise ValueError("滚动重建固定使用 15 分钟步长")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha 不能为负数")
        if self.min_train_rows < 8:
            raise ValueError("min_train_rows 至少为 8")
        if self.min_history_rows < 1:
            raise ValueError("min_history_rows 必须为正数")
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("增量裁剪分位数无效")
        if self.train_window_rows is not None and self.train_window_rows < self.min_train_rows:
            raise ValueError("train_window_rows 不能小于 min_train_rows")
        if min(
            self.generator_1_capacity,
            self.generator_all_capacity,
            self.generator_extra_capacity,
        ) <= 0.0:
            raise ValueError("机组容量必须为正数")

    @property
    def max_horizon(self) -> int:
        """返回最长预测步数。"""

        return max(self.horizons)


@dataclass(frozen=True)
class CurrentStateSourceMapping:
    """从训练期 schema 固化的资源字段映射。

    推理时复用这一映射；即使线上少列，也只把相应特征视作缺失，而不会临时
    改选其他字段，保证 feature manifest 可复现。
    """

    holder_columns: tuple[str, ...]
    gas_supply_columns: tuple[str, ...]
    gas_demand_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "holder_columns": list(self.holder_columns),
            "gas_supply_columns": list(self.gas_supply_columns),
            "gas_demand_columns": list(self.gas_demand_columns),
        }


@dataclass(frozen=True)
class FeatureManifest:
    """稳定记录特征列、来源映射和时间语义。"""

    version: str
    feature_columns: tuple[str, ...]
    targets: tuple[str, ...]
    step_minutes: int
    source_mapping: CurrentStateSourceMapping

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "feature_columns": list(self.feature_columns),
            "targets": list(self.targets),
            "step_minutes": self.step_minutes,
            "source_mapping": self.source_mapping.to_dict(),
        }


@dataclass(frozen=True)
class CausalFeatureBuildResult:
    """特征矩阵和固定来源清单。"""

    frame: pd.DataFrame
    manifest: FeatureManifest


@dataclass(frozen=True)
class HorizonMetadata:
    """单个预测步长的增量与绝对输出列契约。"""

    step: int
    minutes: int
    delta_columns: tuple[str, str]
    prediction_columns: tuple[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "minutes": self.minutes,
            "delta_columns": list(self.delta_columns),
            "prediction_columns": list(self.prediction_columns),
        }


@dataclass
class _DeltaState:
    """一个目标的一步增量模型及其仅训练期统计量。"""

    model: Pipeline
    lower: float
    upper: float
    train_rows: int
    train_origin_start: pd.Timestamp
    train_origin_end: pd.Timestamp
    label_maturity_end: pd.Timestamp


def _validate_frame(frame: pd.DataFrame, *, context: str) -> None:
    """统一检查时间轴，避免按行 shift 时隐式改变时间语义。"""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context} 必须是 DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{context} 必须使用 DatetimeIndex")
    if frame.empty:
        raise ValueError(f"{context} 不能为空")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{context} 时间轴必须严格递增且唯一")
    if frame.columns.duplicated().any():
        raise ValueError(f"{context} 包含重复字段名")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """读取数值列；缺列和非有限值统一显式为缺失。"""

    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float, name=column)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def _aggregate_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """因果地聚合当前行资源字段；没有来源时返回全缺失。"""

    present = [column for column in columns if column in frame]
    if not present:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.concat([_numeric_series(frame, column) for column in present], axis=1)
    return values.sum(axis=1, min_count=1).astype(float)


def _stable_columns(columns: Sequence[object]) -> tuple[str, ...]:
    """按字段名稳定排序，避免输入列顺序影响 manifest。"""

    names = {str(column) for column in columns}
    return tuple(sorted(names, key=lambda value: (value.casefold(), value)))


def resolve_current_state_mapping(columns: Sequence[object]) -> CurrentStateSourceMapping:
    """仅按字段名解析 holder、煤气供给和需求来源。

    该规则不读取任意数值、时间范围或未来标签，因此可以安全地在训练边界处
    固化。缺少某类字段不是隐式回填，而是在特征中保留 NaN，交由训练期 imputer
    处理。
    """

    names = _stable_columns(columns)
    holders: list[str] = []
    supply: list[str] = []
    demand: list[str] = []
    for name in names:
        lowered = name.casefold()
        if name in TARGETS:
            continue
        is_holder = "holder" in lowered
        is_demand = any(
            token in lowered
            for token in (
                "user",
                "demand",
                "air_heater",
                "into_gas",
                "mixed",
                "generator_use",
                "generator_gas",
                "gas_consumption",
            )
        )
        is_supply = any(
            token in lowered
            for token in (
                "production",
                "supply",
                "source",
                "blast_furnace_",
                "coke_oven",
                "bf_gas",
                "coke_gas",
                "converter_",
            )
        ) or lowered in {"gas", "gas_supply"}
        if is_holder:
            holders.append(name)
        elif is_demand:
            demand.append(name)
        elif is_supply:
            supply.append(name)
    return CurrentStateSourceMapping(
        holder_columns=tuple(holders),
        gas_supply_columns=tuple(supply),
        gas_demand_columns=tuple(demand),
    )


def _add_generator_current_state(
    values: dict[str, pd.Series],
    series: pd.Series,
    target: str,
) -> None:
    """加入当前机组状态、向后差分和显式滞后统计。"""

    values[f"{target}_current"] = series
    for steps, minutes in ((1, 15), (2, 30), (3, 45), (4, 60)):
        values[f"{target}_delta_{minutes}"] = series - series.shift(steps)
    for steps in (4, 8, 16):
        values[f"{target}_slope_{steps}"] = (series - series.shift(steps)) / float(steps)
    delta_15 = series - series.shift(1)
    values[f"{target}_acceleration"] = delta_15 - delta_15.shift(1)

    # 先把当前点移出 EWMA/rolling 窗口，再统计 t-1 及更早历史。
    strictly_past = series.shift(1)
    ewma_level = strictly_past.ewm(span=8, adjust=False, min_periods=2).mean()
    values[f"{target}_ewma_trend"] = ewma_level - ewma_level.shift(1)
    values[f"{target}_volatility"] = strictly_past.rolling(8, min_periods=2).std(ddof=0)


def _feature_columns() -> tuple[str, ...]:
    """返回固定的 Current-State 特征列顺序。"""

    columns: list[str] = []
    for target in TARGETS:
        columns.append(f"{target}_current")
        columns.extend(f"{target}_delta_{minutes}" for minutes in (15, 30, 45, 60))
        columns.extend(f"{target}_slope_{steps}" for steps in (4, 8, 16))
        columns.extend(
            (
                f"{target}_acceleration",
                f"{target}_ewma_trend",
                f"{target}_volatility",
            )
        )
    columns.extend(
        (
            "holder_current",
            "holder_momentum",
            "holder_volatility",
            "gas_supply_current",
            "gas_demand_current",
            "gas_mismatch",
            "gas_mismatch_delta_15",
            "gas_mismatch_volatility",
        )
    )
    return tuple(columns)


CURRENT_STATE_FEATURE_COLUMNS = _feature_columns()


def _build_features(
    frame: pd.DataFrame,
    source_mapping: CurrentStateSourceMapping,
) -> pd.DataFrame:
    """基于已固定的来源映射构建严格因果特征。"""

    values: dict[str, pd.Series] = {}
    for target in TARGETS:
        _add_generator_current_state(values, _numeric_series(frame, target), target)

    holder = _aggregate_columns(frame, source_mapping.holder_columns)
    holder_past = holder.shift(1)
    values["holder_current"] = holder
    values["holder_momentum"] = holder - holder.shift(1)
    values["holder_volatility"] = holder_past.rolling(8, min_periods=2).std(ddof=0)

    supply = _aggregate_columns(frame, source_mapping.gas_supply_columns)
    demand = _aggregate_columns(frame, source_mapping.gas_demand_columns)
    mismatch = supply - demand
    values["gas_supply_current"] = supply
    values["gas_demand_current"] = demand
    values["gas_mismatch"] = mismatch
    values["gas_mismatch_delta_15"] = mismatch - mismatch.shift(1)
    values["gas_mismatch_volatility"] = mismatch.shift(1).rolling(8, min_periods=2).std(
        ddof=0
    )

    result = pd.DataFrame(values, index=frame.index)
    result = result.reindex(columns=CURRENT_STATE_FEATURE_COLUMNS)
    return result.replace([np.inf, -np.inf], np.nan).astype(float)


def build_causal_rolling_features(
    frame: pd.DataFrame,
    *,
    source_mapping: CurrentStateSourceMapping | None = None,
    return_manifest: bool = False,
) -> pd.DataFrame | CausalFeatureBuildResult:
    """构造稳定的 Current-State 特征。

    函数只使用当前行及其左侧数据。对任意 origin，修改、打乱、置空或删除
    origin 之后的行，都不会改变该 origin 的特征。
    """

    _validate_frame(frame, context="特征输入")
    mapping = source_mapping or resolve_current_state_mapping(frame.columns)
    features = _build_features(frame, mapping)
    manifest = FeatureManifest(
        version=FEATURE_VERSION,
        feature_columns=tuple(features.columns),
        targets=TARGETS,
        step_minutes=15,
        source_mapping=mapping,
    )
    if return_manifest:
        return CausalFeatureBuildResult(frame=features, manifest=manifest)
    return features


def _make_ridge_pipeline(alpha: float) -> Pipeline:
    """训练期缺失填补、缩放和 Ridge 必须在同一因果训练切片内拟合。"""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _horizon_metadata(config: CausalRollingConfig) -> tuple[HorizonMetadata, ...]:
    """生成稳定的八步输出元数据。"""

    result: list[HorizonMetadata] = []
    for step in config.horizons:
        minutes = step * config.step_minutes
        result.append(
            HorizonMetadata(
                step=step,
                minutes=minutes,
                delta_columns=tuple(
                    f"{target}_delta_t+{minutes}" for target in config.targets
                ),
                prediction_columns=tuple(
                    f"{target}_t+{minutes}_pred" for target in config.targets
                ),
            )
        )
    return tuple(result)


def _successor_index(index: pd.DatetimeIndex) -> pd.Series:
    """把后一行时间对齐到当前 origin，用于排除有时间缺口的伪标签。"""

    return pd.Series(index, index=index).shift(-1)


def _finite_number(value: object) -> float | None:
    """把单个原点状态转换为有限浮点数。"""

    try:
        number = float(pd.to_numeric(value, errors="coerce"))
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


class CausalRollingReconstructionForecaster:
    """只以 origin 历史为输入的递归增量重建模型。"""

    version = "p1_causal_rolling_reconstruction_v1"

    def __init__(self, config: CausalRollingConfig | None = None) -> None:
        self.config = config or CausalRollingConfig()
        self.states_: dict[str, _DeltaState] = {}
        self.feature_manifest_: FeatureManifest | None = None
        self.horizon_metadata_: tuple[HorizonMetadata, ...] = _horizon_metadata(self.config)
        self.fit_metadata_: dict[str, object] = {}
        self.last_prediction_metadata_: dict[str, object] = {}
        self.last_delta_trajectory_: pd.DataFrame | None = None

    def prediction_columns(self) -> tuple[str, ...]:
        """返回 16 个绝对预测列的稳定顺序。"""

        return tuple(
            f"{target}_t+{metadata.minutes}_pred"
            for target in self.config.targets
            for metadata in self.horizon_metadata_
        )

    def delta_columns(self) -> tuple[str, ...]:
        """返回最近一次滚动预测对应的 16 个累计增量列名。"""

        return tuple(
            f"{target}_delta_t+{metadata.minutes}"
            for target in self.config.targets
            for metadata in self.horizon_metadata_
        )

    def horizon_metadata(self) -> tuple[HorizonMetadata, ...]:
        """返回不可变的步长契约，供 OOF 和下游审计使用。"""

        return self.horizon_metadata_

    def feature_manifest(self) -> FeatureManifest:
        """返回训练完成后固化的 feature manifest。"""

        if self.feature_manifest_ is None:
            raise RuntimeError("模型尚未训练，没有 feature manifest")
        return self.feature_manifest_

    def fit(
        self,
        training_frame: pd.DataFrame,
        *,
        train_end: pd.Timestamp | None = None,
        train_start: pd.Timestamp | None = None,
    ) -> "CausalRollingReconstructionForecaster":
        """在指定训练 origin 边界前拟合一步增量模型。

        对任一训练 candidate ``t``，只接受 ``t + 15min <= train_end`` 的成熟
        标签。即使调用方传入了更长的 frame，边界之后的数值、标签和统计量均
        不会进入拟合过程。
        """

        _validate_frame(training_frame, context="训练输入")
        index = training_frame.index
        effective_end = pd.Timestamp(train_end) if train_end is not None else index[-1]
        if effective_end not in index:
            raise ValueError("train_end 必须位于训练时间轴")
        if effective_end > index[-1]:
            raise ValueError("train_end 不能晚于训练数据末尾")
        effective_start = pd.Timestamp(train_start) if train_start is not None else index[0]
        if effective_start > effective_end:
            raise ValueError("train_start 不能晚于 train_end")

        # 先截断，再解析来源和构造特征，防止训练期统计量触及边界之后数据。
        train_frame = training_frame.loc[training_frame.index <= effective_end]
        built = build_causal_rolling_features(train_frame, return_manifest=True)
        if not isinstance(built, CausalFeatureBuildResult):
            raise RuntimeError("内部特征构造未返回 manifest")
        features = built.frame
        successor = _successor_index(train_frame.index)
        step_offset = pd.Timedelta(minutes=self.config.step_minutes)
        expected_successor = train_frame.index + step_offset
        label_mature = successor.eq(expected_successor) & successor.le(effective_end)
        origin_allowed = pd.Series(train_frame.index >= effective_start, index=train_frame.index)
        candidate_mask = label_mature & origin_allowed

        if self.config.train_window_rows is not None:
            candidate_positions = np.flatnonzero(candidate_mask.to_numpy())
            if len(candidate_positions) > self.config.train_window_rows:
                keep = candidate_positions[-self.config.train_window_rows :]
                candidate_mask = pd.Series(False, index=train_frame.index)
                candidate_mask.iloc[keep] = True

        fitted_states: dict[str, _DeltaState] = {}
        target_rows: dict[str, int] = {}
        for target in self.config.targets:
            if target not in train_frame:
                raise ValueError(f"训练输入缺少目标字段: {target}")
            current = _numeric_series(train_frame, target)
            delta = current.shift(-1) - current
            valid = candidate_mask & current.notna() & delta.notna()
            valid &= np.isfinite(delta.to_numpy(dtype=float))
            valid_index = train_frame.index[valid.to_numpy()]
            if len(valid_index) < self.config.min_train_rows:
                raise ValueError(
                    f"{target} 的成熟一步增量训练样本不足: {len(valid_index)} "
                    f"< {self.config.min_train_rows}"
                )
            x_train = features.loc[valid_index, list(built.manifest.feature_columns)]
            y_train = delta.loc[valid_index].to_numpy(dtype=float)
            model = _make_ridge_pipeline(self.config.ridge_alpha)
            model.fit(x_train, y_train)
            lower = float(np.nanquantile(y_train, self.config.lower_quantile))
            upper = float(np.nanquantile(y_train, self.config.upper_quantile))
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError(f"{target} 的训练期增量裁剪统计无效")
            fitted_states[target] = _DeltaState(
                model=model,
                lower=lower,
                upper=upper,
                train_rows=int(len(valid_index)),
                train_origin_start=pd.Timestamp(valid_index[0]),
                train_origin_end=pd.Timestamp(valid_index[-1]),
                label_maturity_end=effective_end,
            )
            target_rows[target] = int(len(valid_index))

        self.states_ = fitted_states
        self.feature_manifest_ = built.manifest
        self.fit_metadata_ = {
            "version": self.version,
            "train_start": effective_start,
            "train_end": effective_end,
            "step_minutes": self.config.step_minutes,
            "mature_label_rule": "origin + 15min <= train_end 且存在连续后继时间点",
            "target_train_rows": target_rows,
            "feature_manifest": built.manifest.to_dict(),
            "horizon_metadata": [item.to_dict() for item in self.horizon_metadata_],
        }
        return self

    @staticmethod
    def _declared_origin(
        history_until_origin: pd.DataFrame,
        origin: pd.Timestamp | None,
    ) -> pd.Timestamp:
        """解析可选边界声明；有未来行时必须在进入特征构造前失败。"""

        attribute_values = [
            history_until_origin.attrs[key]
            for key in ("origin_time", "forecast_origin", "causal_origin")
            if key in history_until_origin.attrs
        ]
        declared = pd.Timestamp(origin) if origin is not None else None
        for value in attribute_values:
            candidate = pd.Timestamp(value)
            if declared is not None and candidate != declared:
                raise ValueError("origin 参数与 history_until_origin.attrs 的边界声明不一致")
            declared = candidate
        return declared if declared is not None else pd.Timestamp(history_until_origin.index[-1])

    def _validate_history_boundary(
        self,
        history_until_origin: pd.DataFrame,
        origin: pd.Timestamp | None,
    ) -> tuple[pd.DataFrame, pd.Timestamp]:
        """验证唯一正式推理入口的 history-only 契约。"""

        _validate_frame(history_until_origin, context="history_until_origin")
        resolved_origin = self._declared_origin(history_until_origin, origin)
        if resolved_origin not in history_until_origin.index:
            raise ValueError("origin 必须位于 history_until_origin 时间轴")
        if (history_until_origin.index > resolved_origin).any():
            raise ValueError("history_until_origin 包含 origin 之后的行，已拒绝未来输入")
        return history_until_origin.loc[:resolved_origin].copy(deep=True), resolved_origin

    def _origin_targets(self, history: pd.DataFrame) -> dict[str, float]:
        """取得真实可用的当前机组状态；目标缺失不能静默填补。"""

        result: dict[str, float] = {}
        for target in self.config.targets:
            if target not in history:
                raise ValueError(f"history_until_origin 缺少当前目标字段: {target}")
            value = _finite_number(history[target].iloc[-1])
            if value is None:
                raise ValueError(f"origin 当前 {target} 为缺失或非有限值")
            result[target] = value
        return result

    def _predict_one_step(self, target: str, feature_row: pd.DataFrame) -> float:
        """使用训练期模型和裁剪统计量预测一个 15 分钟增量。"""

        state = self.states_.get(target)
        if state is None:
            raise RuntimeError(f"模型未训练目标: {target}")
        prediction = np.asarray(
            state.model.predict(feature_row.reindex(columns=self.feature_manifest().feature_columns)),
            dtype=float,
        )
        if prediction.shape != (1,) or not np.isfinite(prediction[0]):
            raise RuntimeError(f"{target} 的一步增量预测无效")
        return float(np.clip(prediction[0], state.lower, state.upper))

    def _constrain_pair(self, values: dict[str, float]) -> dict[str, float]:
        """施加与现有 A61/DirectDelta 相同口径的弱物理容量约束。"""

        generator_1 = float(np.clip(values["generator_1"], 0.0, self.config.generator_1_capacity))
        generator_all = float(
            np.clip(values["generator_all"], 0.0, self.config.generator_all_capacity)
        )
        generator_all = max(generator_all, generator_1)
        generator_all = min(generator_all, generator_1 + self.config.generator_extra_capacity)
        return {"generator_1": generator_1, "generator_all": generator_all}

    @staticmethod
    def _append_causal_state(
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
        generated_targets: dict[str, float],
    ) -> pd.DataFrame:
        """把预测状态追加到历史末尾，资源字段仅持有最后可见值。

        这里不能借用原始 frame 中 ``timestamp`` 的任何行。对 gas、holder、
        users 等外生状态的持有，是一个显式且可审计的递推假设。
        """

        carried = history.ffill().iloc[-1].copy()
        for target, value in generated_targets.items():
            carried[target] = value
        next_row = pd.DataFrame([carried], index=pd.DatetimeIndex([timestamp]))
        return pd.concat([history, next_row], axis=0)

    def _persistent_prediction(
        self,
        origin: pd.Timestamp,
        current: dict[str, float],
        *,
        reason: str,
    ) -> pd.DataFrame:
        """短历史的明确降级行为：返回当前状态的持续性轨迹。"""

        constrained = self._constrain_pair(current)
        output = {
            f"{target}_t+{metadata.minutes}_pred": constrained[target]
            for target in self.config.targets
            for metadata in self.horizon_metadata_
        }
        delta_output = {
            f"{target}_delta_t+{metadata.minutes}": float(constrained[target] - current[target])
            for target in self.config.targets
            for metadata in self.horizon_metadata_
        }
        self.last_delta_trajectory_ = pd.DataFrame(
            [delta_output],
            index=pd.DatetimeIndex([origin]),
        ).reindex(columns=self.delta_columns())
        self.last_prediction_metadata_ = {
            "origin": origin,
            "mode": "persistence",
            "reason": reason,
            "used_future_observations": False,
            "delta_trajectory": {
                target: [float(constrained[target] - current[target])] * len(self.horizon_metadata_)
                for target in self.config.targets
            },
        }
        return pd.DataFrame([output], index=pd.DatetimeIndex([origin]))

    def predict_at_origin(
        self,
        history_until_origin: pd.DataFrame,
        *,
        origin: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """预测当前 origin 后 15..120 分钟的两目标绝对轨迹。

        正常调用只传 ``history_until_origin``，其最后一行即为 origin。可选
        ``origin`` 仅用于把调用方已知的边界再次断言；若输入含有更晚行会立即
        抛错。模块不提供 ``predict(scoring_frame)`` 这类整段未来推理接口。
        """

        if not self.states_ or self.feature_manifest_ is None:
            raise RuntimeError("滚动重建模型尚未训练")
        history, resolved_origin = self._validate_history_boundary(history_until_origin, origin)
        current = self._origin_targets(history)
        if len(history) < self.config.min_history_rows:
            return self._persistent_prediction(
                resolved_origin,
                current,
                reason=f"历史行数 {len(history)} < {self.config.min_history_rows}",
            )

        synthetic = history
        anchors = current.copy()
        absolute_paths: dict[str, list[float]] = {target: [] for target in self.config.targets}
        delta_paths: dict[str, list[float]] = {target: [] for target in self.config.targets}
        for metadata in self.horizon_metadata_:
            features = build_causal_rolling_features(
                synthetic,
                source_mapping=self.feature_manifest_.source_mapping,
            )
            one_step_delta = {
                target: self._predict_one_step(target, features.iloc[[-1]])
                for target in self.config.targets
            }
            state_now = self._origin_targets(synthetic)
            generated = self._constrain_pair(
                {target: state_now[target] + one_step_delta[target] for target in self.config.targets}
            )
            for target in self.config.targets:
                absolute_paths[target].append(generated[target])
                delta_paths[target].append(float(generated[target] - anchors[target]))
            synthetic = self._append_causal_state(
                synthetic,
                resolved_origin + pd.Timedelta(minutes=metadata.minutes),
                generated,
            )

        output: dict[str, float] = {}
        for target in self.config.targets:
            for metadata, value in zip(self.horizon_metadata_, absolute_paths[target], strict=True):
                output[f"{target}_t+{metadata.minutes}_pred"] = value
        result = pd.DataFrame([output], index=pd.DatetimeIndex([resolved_origin]))
        if tuple(result.columns) != self.prediction_columns():
            result = result.reindex(columns=self.prediction_columns())
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise RuntimeError("滚动重建输出包含非有限值")
        delta_output = {
            f"{target}_delta_t+{metadata.minutes}": value
            for target in self.config.targets
            for metadata, value in zip(
                self.horizon_metadata_,
                delta_paths[target],
                strict=True,
            )
        }
        self.last_delta_trajectory_ = pd.DataFrame(
            [delta_output],
            index=pd.DatetimeIndex([resolved_origin]),
        ).reindex(columns=self.delta_columns())
        self.last_prediction_metadata_ = {
            "origin": resolved_origin,
            "mode": "rolling_reconstruction",
            "history_rows": int(len(history)),
            "used_future_observations": False,
            "resource_transition": "hold_last_observed_state",
            "delta_trajectory": delta_paths,
        }
        return result

    @classmethod
    def build_oof(
        cls,
        frame: pd.DataFrame,
        *,
        config: CausalRollingConfig | None = None,
        folds: Sequence[TimeFold] | None = None,
        include_blind: bool = False,
        forward_refit: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """提供类级 OOF 入口，语义与模块函数完全一致。"""

        return build_causal_rolling_oof(
            frame,
            config=config,
            folds=folds,
            include_blind=include_blind,
            forward_refit=forward_refit,
        )


def _oof_records_for_origin(
    frame: pd.DataFrame,
    model: CausalRollingReconstructionForecaster,
    *,
    fold: TimeFold,
    origin: pd.Timestamp,
) -> list[dict[str, object]]:
    """把一个 origin 的 16 个绝对预测转成可评分长表。"""

    prediction = model.predict_at_origin(frame.loc[:origin]).iloc[0]
    records: list[dict[str, object]] = []
    for metadata in model.horizon_metadata_:
        target_time = origin + pd.Timedelta(minutes=metadata.minutes)
        if target_time not in frame.index:
            continue
        for target in model.config.targets:
            current = _finite_number(frame.at[origin, target])
            actual = _finite_number(frame.at[target_time, target])
            if current is None or actual is None:
                continue
            value = float(prediction[f"{target}_t+{metadata.minutes}_pred"])
            records.append(
                {
                    "fold": fold.name,
                    "origin_time": origin,
                    "train_end": model.fit_metadata_["train_end"],
                    "label_maturity_end": model.fit_metadata_["train_end"],
                    "target": target,
                    "horizon": metadata.minutes,
                    "actual": actual,
                    "current_value": current,
                    "actual_delta": actual - current,
                    "prediction": value,
                    "causal_rolling_prediction": value,
                }
            )
    return records


def build_causal_rolling_oof(
    frame: pd.DataFrame,
    *,
    config: CausalRollingConfig | None = None,
    folds: Sequence[TimeFold] | None = None,
    include_blind: bool = False,
    forward_refit: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """构造按 origin 前向训练的严格 OOF。

    默认 ``forward_refit=True``：每个验证 origin 都仅以当时已成熟的历史标签
    重训，模拟线上持续学习。设为 False 时则按外层折固定 ``train_end`` 拟合，
    同样保留成熟标签约束。
    """

    _validate_frame(frame, context="OOF 输入")
    selected_config = config or CausalRollingConfig()
    if include_blind:
        raise ValueError("CausalRolling OOF 不读取 blind 标签")
    all_folds = list(folds) if folds is not None else make_outer_folds(frame.index, ForecastConfig())
    selected_folds = [fold for fold in all_folds if not fold.blind]
    if not selected_folds:
        raise ValueError("没有可用的 development 折")

    records: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    last_model: CausalRollingReconstructionForecaster | None = None
    fixed_models: dict[str, CausalRollingReconstructionForecaster] = {}
    for fold in selected_folds:
        assert_label_safe_fold(fold, selected_config.max_horizon)
        _, validation_mask = fold.masks(frame.index)
        origins = frame.index[validation_mask]
        if len(origins) == 0:
            trace.append({"fold": fold.name, "event": "skip", "reason": "验证 origin 为空"})
            continue
        if not forward_refit:
            try:
                fixed_models[fold.name] = CausalRollingReconstructionForecaster(selected_config).fit(
                    frame,
                    train_start=fold.train_start,
                    train_end=fold.train_end,
                )
            except ValueError as exc:
                trace.append({"fold": fold.name, "event": "skip", "reason": str(exc)})
                continue
        for origin in origins:
            if forward_refit:
                try:
                    model = CausalRollingReconstructionForecaster(selected_config).fit(
                        frame,
                        train_start=fold.train_start,
                        train_end=origin,
                    )
                except ValueError as exc:
                    trace.append(
                        {
                            "fold": fold.name,
                            "origin_time": origin,
                            "event": "origin_skip",
                            "reason": str(exc),
                        }
                    )
                    continue
            else:
                model = fixed_models[fold.name]
            origin_records = _oof_records_for_origin(frame, model, fold=fold, origin=origin)
            records.extend(origin_records)
            trace.append(
                {
                    "fold": fold.name,
                    "origin_time": origin,
                    "event": "origin_fit_predict",
                    "train_end": model.fit_metadata_["train_end"],
                    "rows": len(origin_records),
                }
            )
            last_model = model

    if not records or last_model is None:
        raise ValueError("没有生成可评分的 CausalRolling OOF 行")
    rows = pd.DataFrame.from_records(records).sort_values(
        ["origin_time", "target", "horizon", "fold"],
        kind="stable",
    ).reset_index(drop=True)
    report: dict[str, object] = {
        "version": CausalRollingReconstructionForecaster.version,
        "targets": list(selected_config.targets),
        "horizons": [item.minutes for item in last_model.horizon_metadata_],
        "feature_manifest": last_model.feature_manifest().to_dict(),
        "horizon_metadata": [item.to_dict() for item in last_model.horizon_metadata_],
        "forward_refit": bool(forward_refit),
        "mature_label_rule": "每个训练 origin 的一步标签结束时刻 <= 当时 train_end",
        "folds": [fold.name for fold in selected_folds],
        "blind_included": False,
        "blind_labels_used": False,
        "trace_rows": len(trace),
        "trace": trace,
    }
    return rows, report


def _future_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    """按 hard-audit 要求划分未来字段组。"""

    groups: dict[str, list[str]] = {
        "generator": [],
        "gas": [],
        "holder": [],
        "users": [],
        "all_features": list(frame.select_dtypes(include=[np.number]).columns),
    }
    for column in frame.columns:
        name = str(column).casefold()
        if name.startswith("generator"):
            groups["generator"].append(column)
        if any(token in name for token in ("gas", "blast_furnace", "coke", "converter")):
            groups["gas"].append(column)
        if "holder" in name:
            groups["holder"].append(column)
        if any(token in name for token in ("user", "air_heater", "demand", "mixed")):
            groups["users"].append(column)
    return groups


def audit_causal_rolling_future_perturbations(
    model: CausalRollingReconstructionForecaster,
    frame: pd.DataFrame,
    origins: Sequence[pd.Timestamp] | None = None,
) -> dict[str, object]:
    """验证未来 generator/gas/holder/users/全部特征均不能影响 origin 预测。"""

    _validate_frame(frame, context="未来扰动审计输入")
    selected = list(origins) if origins is not None else list(frame.index[:: max(1, len(frame) // 8)])
    selected = [pd.Timestamp(origin) for origin in selected if pd.Timestamp(origin) in frame.index]
    if not selected:
        raise ValueError("未来扰动审计至少需要一个有效 origin")
    groups = _future_groups(frame)
    failures: list[dict[str, object]] = []
    maximum = 0.0
    cases = 0
    for origin in selected:
        baseline = model.predict_at_origin(frame.loc[:origin])
        for group_name, columns in groups.items():
            if not columns:
                continue
            for operation in ("perturb", "delete"):
                changed = frame.copy(deep=True)
                future = changed.index > origin
                if operation == "perturb":
                    changed.loc[future, columns] = -9_999_991.0
                else:
                    changed.loc[future, columns] = np.nan
                candidate = model.predict_at_origin(changed.loc[:origin])
                difference = np.abs(
                    baseline.to_numpy(dtype=float) - candidate.to_numpy(dtype=float)
                )
                maximum = max(maximum, float(np.max(difference)))
                cases += 1
                if not np.array_equal(baseline.to_numpy(), candidate.to_numpy()):
                    failures.append(
                        {
                            "origin": str(origin),
                            "group": group_name,
                            "operation": operation,
                            "max_abs_difference": float(np.max(difference)),
                        }
                    )
        deleted_future = frame.loc[:origin].copy(deep=True)
        candidate = model.predict_at_origin(deleted_future)
        difference = np.abs(baseline.to_numpy(dtype=float) - candidate.to_numpy(dtype=float))
        maximum = max(maximum, float(np.max(difference)))
        cases += 1
        if not np.array_equal(baseline.to_numpy(), candidate.to_numpy()):
            failures.append(
                {
                    "origin": str(origin),
                    "group": "all_features",
                    "operation": "delete_future_rows",
                    "max_abs_difference": float(np.max(difference)),
                }
            )
    return {
        "passed": not failures,
        "origins": len(selected),
        "cases_checked": cases,
        "max_abs_difference": maximum,
        "prediction_columns": list(model.prediction_columns()),
        "failures": failures,
    }


# 独立研究模块的稳定别名；不接入公共模型注册表。
CausalRollingModel = CausalRollingReconstructionForecaster
make_causal_rolling_features = build_causal_rolling_features


__all__ = [
    "CausalFeatureBuildResult",
    "CausalRollingConfig",
    "CausalRollingModel",
    "CausalRollingReconstructionForecaster",
    "CurrentStateSourceMapping",
    "FEATURE_VERSION",
    "FeatureManifest",
    "HORIZONS",
    "HorizonMetadata",
    "TARGETS",
    "audit_causal_rolling_future_perturbations",
    "build_causal_rolling_features",
    "build_causal_rolling_oof",
    "make_causal_rolling_features",
    "resolve_current_state_mapping",
]
