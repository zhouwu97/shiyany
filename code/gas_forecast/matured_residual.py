"""严格因果的成熟残差状态与 OOF 修正接口。

一个预测只能在它的 ``target_datetime`` 到达时结算。因而在时间 ``t``
生成状态时，本模块只读取更早 origin 发出的、且
``target_datetime == t`` 的 OOF 预测误差。它不会把同一 origin 的未来
标签、晚于 ``t`` 的标签，或模型训练集内拟合得到的残差混入状态。

本模块仅提供状态、特征和逐行修正；不在这里选择 P3 的最终融合权重。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


STEP_MINUTES = 15
DEFAULT_HORIZONS_MINUTES: tuple[int, ...] = tuple(range(15, 121, STEP_MINUTES))


@dataclass(frozen=True)
class MaturedResidualConfig:
    """成熟残差状态的固定参数。

    ``horizons`` 使用分钟而非步数，与项目统一 OOF 长表的 ``horizon``
    字段一致。误差定义为 ``actual - prediction``：正值表示基础预测低估，
    负值表示基础预测高估。
    """

    horizons: tuple[int, ...] = DEFAULT_HORIZONS_MINUTES
    ewma_alpha: float = 0.25
    slope_window: int = 4
    correction_slope_weight: float = 0.0
    correction_clip: float | None = None

    def __post_init__(self) -> None:
        horizons = tuple(int(value) for value in self.horizons)
        if not horizons:
            raise ValueError("成熟残差至少需要一个预测步长")
        if len(set(horizons)) != len(horizons):
            raise ValueError("成熟残差步长不能重复")
        if any(value <= 0 or value % STEP_MINUTES for value in horizons):
            raise ValueError("成熟残差步长必须是正的 15 分钟整数倍")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha 必须位于 (0, 1]")
        if self.slope_window < 2:
            raise ValueError("slope_window 至少为 2")
        if self.correction_clip is not None and self.correction_clip <= 0.0:
            raise ValueError("correction_clip 必须为正数或 None")
        object.__setattr__(self, "horizons", horizons)


@dataclass(frozen=True)
class MaturedResidualOOFResult:
    """带成熟残差特征和修正预测的 OOF 结果。"""

    rows: pd.DataFrame
    state_trace: pd.DataFrame
    report: dict[str, object]


@dataclass
class _ErrorState:
    """一个 target×horizon 的可变在线统计量。"""

    matured_count: int = 0
    latest_error: float = 0.0
    error_ewma: float = 0.0
    consecutive_overestimate_count: int = 0
    consecutive_underestimate_count: int = 0
    last_matured_origin: pd.Timestamp | None = None
    last_matured_target_datetime: pd.Timestamp | None = None
    history: deque[tuple[pd.Timestamp, float]] = field(default_factory=deque)


def _as_timestamp_series(values: pd.Series, name: str) -> pd.Series:
    """解析时间列，并拒绝不能用于严格时间比较的值。"""

    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{name} 包含无法解析的时间")
    return parsed


def _as_horizon_series(values: pd.Series, allowed: tuple[int, ...]) -> pd.Series:
    """校验分钟粒度的 horizon，避免把步数误当成分钟。"""

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.equal(numeric % 1.0, 0.0).all():
        raise ValueError("horizon 必须是整数分钟")
    horizon = numeric.astype(int)
    unsupported = sorted(set(horizon).difference(allowed))
    if unsupported:
        raise ValueError(f"成熟残差不支持 horizon: {unsupported}")
    return horizon


def _required_columns(
    rows: pd.DataFrame,
    *,
    prediction_column: str,
    actual_column: str,
    require_cross_fit: bool,
) -> None:
    """统一检查 OOF 台账的最小契约。"""

    if prediction_column == actual_column:
        raise ValueError("基础预测列和实际值列必须不同")
    required = {"origin_time", "target", "horizon", prediction_column, actual_column}
    if require_cross_fit:
        required.add("train_end")
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"成熟残差 OOF 台账缺少字段: {missing}")


def _prepare_ledger(
    rows: pd.DataFrame,
    *,
    config: MaturedResidualConfig,
    prediction_column: str,
    actual_column: str,
    require_cross_fit: bool,
    cutoff: pd.Timestamp | None,
) -> pd.DataFrame:
    """规范化一个可见时间窗内的 OOF 预测台账。

    ``cutoff`` 用于单 origin 查询。先按 forecast origin 截断，再校验数值，
    所以 cutoff 之后的标签即使被篡改、删除或置空也不会参与当前输出。
    """

    _required_columns(
        rows,
        prediction_column=prediction_column,
        actual_column=actual_column,
        require_cross_fit=require_cross_fit,
    )
    work = rows.copy()
    work["origin_time"] = _as_timestamp_series(work["origin_time"], "origin_time")
    if cutoff is not None:
        work = work.loc[work["origin_time"].le(pd.Timestamp(cutoff))].copy()
    if work.empty:
        raise ValueError("可见 OOF 台账为空")
    if work["target"].isna().any():
        raise ValueError("target 不能为空")
    work["target"] = work["target"].astype(str)
    if work["target"].eq("").any():
        raise ValueError("target 不能为空")
    work["horizon"] = _as_horizon_series(work["horizon"], config.horizons)
    if "target_datetime" in work.columns:
        work["target_datetime"] = _as_timestamp_series(
            work["target_datetime"], "target_datetime"
        )
    else:
        work["target_datetime"] = work["origin_time"] + pd.to_timedelta(
            work["horizon"], unit="min"
        )
    expected_target_datetime = work["origin_time"] + pd.to_timedelta(
        work["horizon"], unit="min"
    )
    if not work["target_datetime"].eq(expected_target_datetime).all():
        raise ValueError("target_datetime 必须等于 origin_time + horizon")
    if not work["origin_time"].lt(work["target_datetime"]).all():
        raise ValueError("预测 target_datetime 必须严格晚于 origin_time")
    if require_cross_fit:
        work["train_end"] = _as_timestamp_series(work["train_end"], "train_end")
        if not work["train_end"].lt(work["origin_time"]).all():
            raise ValueError("成熟残差拒绝 in-sample 残差：train_end 必须早于 origin_time")
        if "fold" in work.columns and work["fold"].astype(str).str.lower().str.contains("blind").any():
            raise ValueError("成熟残差 OOF 不允许包含 blind 折")
    work[prediction_column] = pd.to_numeric(work[prediction_column], errors="coerce")
    if not np.isfinite(work[prediction_column].to_numpy(dtype=float)).all():
        raise ValueError("成熟残差 OOF 包含非有限基础预测")
    if work.duplicated(["origin_time", "target", "horizon"]).any():
        raise ValueError("成熟残差 OOF 存在重复的 origin_time/target/horizon 行")

    # 只要求截至可见截止点已经成熟的标签有效；当前 origin 的未来标签
    # 是训练/评测台账中的占位信息，不能影响当前状态或触发校验失败。
    visible_end = pd.Timestamp(cutoff) if cutoff is not None else work["origin_time"].max()
    matured = work["target_datetime"].le(visible_end)
    if matured.any():
        numeric_actual = pd.to_numeric(work.loc[matured, actual_column], errors="coerce")
        if not np.isfinite(numeric_actual.to_numpy(dtype=float)).all():
            raise ValueError("已经成熟的 OOF 标签包含 NaN/Inf")
        work.loc[matured, actual_column] = numeric_actual.to_numpy(dtype=float)
    return work.sort_values(["origin_time", "target", "horizon"], kind="stable").reset_index(
        drop=True
    )


class MaturedResidualState:
    """只由已经成熟 OOF 误差推进的逐 origin 状态机。

    :meth:`advance` 的唯一更新源是更早 origin 且
    ``target_datetime == origin`` 的行。调用者可在没有当前预测的虚拟时刻
    调用它，以穿越 OOF 块之间的时间缺口；这不会引入任何未来标签。
    """

    def __init__(self, config: MaturedResidualConfig | None = None) -> None:
        self.config = config or MaturedResidualConfig()
        self._states: dict[tuple[str, int], _ErrorState] = {}
        self._last_origin: pd.Timestamp | None = None

    @property
    def last_origin(self) -> pd.Timestamp | None:
        """返回最近一次已推进的 origin。"""

        return self._last_origin

    def _state_for(self, target: str, horizon: int) -> _ErrorState:
        key = (str(target), int(horizon))
        if key not in self._states:
            self._states[key] = _ErrorState()
        return self._states[key]

    def _slope(self, state: _ErrorState) -> float:
        """以 15 分钟步数为横轴计算最近成熟误差的最小二乘斜率。"""

        if len(state.history) < 2:
            return 0.0
        times = np.asarray([item[0].value for item in state.history], dtype=np.float64)
        values = np.asarray([item[1] for item in state.history], dtype=float)
        steps = (times - times[0]) / (STEP_MINUTES * 60.0 * 1_000_000_000.0)
        centered_steps = steps - steps.mean()
        denominator = float(np.dot(centered_steps, centered_steps))
        if denominator <= 1e-12:
            return 0.0
        return float(np.dot(centered_steps, values - values.mean()) / denominator)

    def _update(self, row: pd.Series, *, prediction_column: str, actual_column: str) -> None:
        """结算一条已经到期的 OOF 预测，并更新对应 horizon 的状态。"""

        target = str(row["target"])
        horizon = int(row["horizon"])
        actual = float(row[actual_column])
        prediction = float(row[prediction_column])
        if not np.isfinite(actual) or not np.isfinite(prediction):
            raise ValueError("成熟残差不能用 NaN/Inf 结算已成熟预测")
        error = actual - prediction
        state = self._state_for(target, horizon)
        if state.matured_count == 0:
            state.error_ewma = error
        else:
            alpha = self.config.ewma_alpha
            state.error_ewma = alpha * error + (1.0 - alpha) * state.error_ewma
        state.matured_count += 1
        state.latest_error = error
        state.last_matured_origin = pd.Timestamp(row["origin_time"])
        state.last_matured_target_datetime = pd.Timestamp(row["target_datetime"])
        state.history.append((state.last_matured_target_datetime, error))
        while len(state.history) > self.config.slope_window:
            state.history.popleft()
        if error < 0.0:
            state.consecutive_overestimate_count += 1
            state.consecutive_underestimate_count = 0
        elif error > 0.0:
            state.consecutive_underestimate_count += 1
            state.consecutive_overestimate_count = 0
        else:
            state.consecutive_overestimate_count = 0
            state.consecutive_underestimate_count = 0

    def advance(
        self,
        origin: pd.Timestamp | str,
        ledger: pd.DataFrame,
        *,
        prediction_column: str = "prediction",
        actual_column: str = "actual",
    ) -> int:
        """推进到一个 origin，并只结算恰在该时刻成熟的更早预测。

        返回本次真正结算的 OOF 行数。时间必须严格递增，避免同一成熟
        残差被重复写入 EWMA 和连续高估/低估计数。
        """

        timestamp = pd.Timestamp(origin)
        if self._last_origin is not None and timestamp <= self._last_origin:
            raise ValueError("成熟残差状态 advance 时间必须严格递增")
        required = {
            "origin_time",
            "target_datetime",
            "target",
            "horizon",
            prediction_column,
            actual_column,
        }
        missing = sorted(required.difference(ledger.columns))
        if missing:
            raise ValueError(f"成熟残差台账缺少字段: {missing}")

        # 这里必须是严格相等而非 <=。较早到期的行应该在它们自己的
        # origin 时刻结算，不能因 OOF 块缺口而被延迟、成批灌入当前状态。
        target_times = _as_timestamp_series(ledger["target_datetime"], "target_datetime")
        origin_times = _as_timestamp_series(ledger["origin_time"], "origin_time")
        due = ledger.loc[target_times.eq(timestamp) & origin_times.lt(timestamp)].copy()
        if not due.empty:
            if due.duplicated(["origin_time", "target", "horizon"]).any():
                raise ValueError("同一成熟 OOF 预测重复出现在台账中")
            if due.duplicated(["target", "horizon"]).any():
                raise ValueError("同一 target/horizon 在一个成熟时刻有多条 OOF 预测")
            due["horizon"] = _as_horizon_series(due["horizon"], self.config.horizons)
            due[actual_column] = pd.to_numeric(due[actual_column], errors="coerce")
            due[prediction_column] = pd.to_numeric(due[prediction_column], errors="coerce")
            if not np.isfinite(due[[actual_column, prediction_column]].to_numpy(dtype=float)).all():
                raise ValueError("已成熟 OOF 行包含 NaN/Inf，拒绝用不完整残差更新状态")
            actual_counts = due.groupby("target", sort=False)[actual_column].nunique(dropna=False)
            if actual_counts.gt(1).any():
                raise ValueError("同一 target_datetime/target 的成熟实际值不一致")
            for _, row in due.sort_values(["target", "horizon", "origin_time"], kind="stable").iterrows():
                self._update(row, prediction_column=prediction_column, actual_column=actual_column)
        self._last_origin = timestamp
        return int(len(due))

    def correction(self, target: str, horizon: int) -> float:
        """返回用于该 target×horizon 基础预测的因果误差修正量。"""

        horizon = int(horizon)
        if horizon not in self.config.horizons:
            raise ValueError(f"成熟残差不支持 horizon: {horizon}")
        state = self._state_for(str(target), horizon)
        if state.matured_count == 0:
            return 0.0
        correction = state.error_ewma + self.config.correction_slope_weight * self._slope(state)
        if self.config.correction_clip is not None:
            correction = float(
                np.clip(correction, -self.config.correction_clip, self.config.correction_clip)
            )
        return float(correction)

    def features_for(self, origin: pd.Timestamp | str, pairs: pd.DataFrame) -> pd.DataFrame:
        """返回指定 target×horizon 对在当前 origin 的因果状态特征。"""

        required = {"target", "horizon"}
        missing = sorted(required.difference(pairs.columns))
        if missing:
            raise ValueError(f"状态特征请求缺少字段: {missing}")
        timestamp = pd.Timestamp(origin)
        records: list[dict[str, object]] = []
        for row in pairs.loc[:, ["target", "horizon"]].itertuples(index=False):
            target = str(row.target)
            horizon = int(row.horizon)
            if horizon not in self.config.horizons:
                raise ValueError(f"成熟残差不支持 horizon: {horizon}")
            state = self._state_for(target, horizon)
            records.append(
                {
                    "origin_time": timestamp,
                    "target": target,
                    "horizon": horizon,
                    "matured_error_count": state.matured_count,
                    "latest_matured_error": state.latest_error,
                    "error_ewma": state.error_ewma,
                    "error_slope": self._slope(state),
                    "consecutive_overestimate_count": state.consecutive_overestimate_count,
                    "consecutive_underestimate_count": state.consecutive_underestimate_count,
                    "matured_residual_correction": self.correction(target, horizon),
                    "last_matured_origin": state.last_matured_origin,
                    "last_matured_target_datetime": state.last_matured_target_datetime,
                }
            )
        result = pd.DataFrame.from_records(records)
        for column in ("last_matured_origin", "last_matured_target_datetime"):
            result[column] = pd.to_datetime(result[column])
        return result


class MaturedResidualPredictor:
    """把 :class:`MaturedResidualState` 应用于当前 origin 基础预测的预测器。"""

    def __init__(self, config: MaturedResidualConfig | None = None) -> None:
        self.state = MaturedResidualState(config)

    def step(
        self,
        origin: pd.Timestamp | str,
        ledger: pd.DataFrame,
        current_predictions: pd.DataFrame,
        *,
        prediction_column: str = "prediction",
        actual_column: str = "actual",
        output_column: str = "matured_residual_pred",
    ) -> tuple[pd.DataFrame, int]:
        """先结算当前成熟误差，再修正当前 origin 的基础预测。

        ``current_predictions`` 可以包含尚未成熟的 ``actual`` 列，但该列
        在本方法中不会被读取；状态只由 ``ledger`` 中恰好到期的更早 OOF
        行推进。
        """

        timestamp = pd.Timestamp(origin)
        required = {"origin_time", "target", "horizon", prediction_column}
        missing = sorted(required.difference(current_predictions.columns))
        if missing:
            raise ValueError(f"当前预测缺少字段: {missing}")
        current = current_predictions.copy()
        current["origin_time"] = _as_timestamp_series(current["origin_time"], "origin_time")
        if not current["origin_time"].eq(timestamp).all():
            raise ValueError("当前预测的 origin_time 必须全部等于 step origin")
        current["horizon"] = _as_horizon_series(current["horizon"], self.state.config.horizons)
        if current.duplicated(["target", "horizon"]).any():
            raise ValueError("当前 origin 存在重复的 target/horizon 基础预测")
        current[prediction_column] = pd.to_numeric(current[prediction_column], errors="coerce")
        if not np.isfinite(current[prediction_column].to_numpy(dtype=float)).all():
            raise ValueError("当前基础预测包含 NaN/Inf")
        reserved_columns = {
            "matured_residual_base_prediction",
            "matured_residual_correction",
            "matured_error_count",
            "latest_matured_error",
            "error_ewma",
            "error_slope",
            "consecutive_overestimate_count",
            "consecutive_underestimate_count",
            "last_matured_origin",
            "last_matured_target_datetime",
        }
        if not isinstance(output_column, str) or not output_column:
            raise ValueError("成熟残差输出列名不能为空")
        if output_column in reserved_columns:
            raise ValueError(f"成熟残差输出列名为保留字段: {output_column}")
        generated_columns = set(reserved_columns)
        if output_column != prediction_column:
            generated_columns.add(output_column)
        collisions = sorted(generated_columns.intersection(current.columns))
        if collisions:
            raise ValueError(f"当前预测已包含成熟残差输出字段: {collisions}")

        matured_rows = self.state.advance(
            timestamp,
            ledger,
            prediction_column=prediction_column,
            actual_column=actual_column,
        )
        features = self.state.features_for(timestamp, current)
        result = current.merge(
            features,
            on=["origin_time", "target", "horizon"],
            how="left",
            validate="one_to_one",
        )
        result["matured_residual_base_prediction"] = result[prediction_column].to_numpy(dtype=float)
        result[output_column] = (
            result["matured_residual_base_prediction"]
            + result["matured_residual_correction"]
        )
        if not np.isfinite(result[output_column].to_numpy(dtype=float)).all():
            raise RuntimeError("成熟残差预测器没有生成有限修正预测")
        return result, matured_rows


def _event_times(ledger: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DatetimeIndex:
    """生成到 cutoff 为止的真实 origin 与仅用于结算的成熟时刻。"""

    origins = ledger.loc[ledger["origin_time"].le(cutoff), "origin_time"]
    maturities = ledger.loc[ledger["target_datetime"].le(cutoff), "target_datetime"]
    return pd.DatetimeIndex(sorted(set(origins).union(set(maturities))))


def _walk_visible_ledger(
    ledger: pd.DataFrame,
    *,
    config: MaturedResidualConfig,
    prediction_column: str,
    actual_column: str,
    output_column: str,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间事件逐个回放，必要时穿越没有当前 OOF 行的成熟时刻。"""

    predictor = MaturedResidualPredictor(config)
    output_parts: list[pd.DataFrame] = []
    trace: list[dict[str, object]] = []
    for timestamp in _event_times(ledger, cutoff):
        current = ledger.loc[ledger["origin_time"].eq(timestamp)].copy()
        if current.empty:
            matured_rows = predictor.state.advance(
                timestamp,
                ledger,
                prediction_column=prediction_column,
                actual_column=actual_column,
            )
            trace.append(
                {
                    "origin_time": timestamp,
                    "matured_rows": matured_rows,
                    "has_current_oof_prediction": False,
                }
            )
            continue
        corrected, matured_rows = predictor.step(
            timestamp,
            ledger,
            current,
            prediction_column=prediction_column,
            actual_column=actual_column,
            output_column=output_column,
        )
        output_parts.append(corrected)
        trace.append(
            {
                "origin_time": timestamp,
                "matured_rows": matured_rows,
                "has_current_oof_prediction": True,
            }
        )
    if not output_parts:
        raise ValueError("没有可输出的当前 OOF 预测")
    rows = pd.concat(output_parts, ignore_index=True).sort_values(
        ["origin_time", "target", "horizon"], kind="stable"
    )
    return rows.reset_index(drop=True), pd.DataFrame(trace)


def build_matured_residual_oof(
    rows: pd.DataFrame,
    *,
    config: MaturedResidualConfig | None = None,
    prediction_column: str = "prediction",
    actual_column: str = "actual",
    output_column: str = "matured_residual_pred",
) -> MaturedResidualOOFResult:
    """从按 origin 生成的 OOF 台账构造严格成熟的残差特征和修正预测。

    输入必须提供 ``train_end < origin_time``，因此任何用于状态的误差都
    来自交叉拟合/OOF 预测而非 in-sample fitted residual。返回行保留原始
    OOF 键、标签和基础预测，P3 可把 ``output_column`` 作为候选路线预测列
    传给统一融合契约。
    """

    effective = config or MaturedResidualConfig()
    ledger = _prepare_ledger(
        rows,
        config=effective,
        prediction_column=prediction_column,
        actual_column=actual_column,
        require_cross_fit=True,
        cutoff=None,
    )
    cutoff = pd.Timestamp(ledger["origin_time"].max())
    corrected, trace = _walk_visible_ledger(
        ledger,
        config=effective,
        prediction_column=prediction_column,
        actual_column=actual_column,
        output_column=output_column,
        cutoff=cutoff,
    )
    report: dict[str, object] = {
        "experiment": "matured_residual_state",
        "rows": int(len(corrected)),
        "prediction_column": output_column,
        "base_prediction_column": prediction_column,
        "horizons_minutes": list(effective.horizons),
        "cross_fitted_oof_required": True,
        "in_sample_residual_used": False,
        "maturity_rule": "earlier_origin and target_datetime == current_origin",
        "future_labels_used_for_current_origin": False,
        "state_trace_rows": int(len(trace)),
    }
    return MaturedResidualOOFResult(rows=corrected, state_trace=trace, report=report)


def predict_matured_residual_at_origin(
    rows: pd.DataFrame,
    origin: pd.Timestamp | str,
    *,
    config: MaturedResidualConfig | None = None,
    prediction_column: str = "prediction",
    actual_column: str = "actual",
    output_column: str = "matured_residual_pred",
    require_cross_fit: bool = True,
) -> pd.DataFrame:
    """在单个 origin 计算严格历史成熟残差状态和修正预测。

    本入口先删除 forecast origin 晚于请求时刻的行，再进行任何数值校验或
    状态回放。故请求时刻之后的实际值、特征列或整行被改变/删除时，当前
    输出保持逐元素不变。
    """

    timestamp = pd.Timestamp(origin)
    effective = config or MaturedResidualConfig()
    ledger = _prepare_ledger(
        rows,
        config=effective,
        prediction_column=prediction_column,
        actual_column=actual_column,
        require_cross_fit=require_cross_fit,
        cutoff=timestamp,
    )
    if not ledger["origin_time"].eq(timestamp).any():
        raise ValueError("请求的 origin 没有当前 OOF 预测")
    corrected, _ = _walk_visible_ledger(
        ledger,
        config=effective,
        prediction_column=prediction_column,
        actual_column=actual_column,
        output_column=output_column,
        cutoff=timestamp,
    )
    current = corrected.loc[corrected["origin_time"].eq(timestamp)].copy()
    # 单 origin 预测是在线接口，当前行的 actual 属于未来标签；不把它回传
    # 给调用者，才能保证未来标签被扰动时整个输出对象仍保持不变。
    return current.drop(columns=[actual_column], errors="ignore").reset_index(drop=True)


def audit_matured_residual_future_perturbation(
    rows: pd.DataFrame,
    *,
    origin: pd.Timestamp | str,
    config: MaturedResidualConfig | None = None,
    prediction_column: str = "prediction",
    actual_column: str = "actual",
) -> dict[str, object]:
    """审计未来标签、未来特征和未来 OOF 行不会改变一个 origin 的输出。"""

    timestamp = pd.Timestamp(origin)
    effective = config or MaturedResidualConfig()
    baseline = predict_matured_residual_at_origin(
        rows,
        timestamp,
        config=effective,
        prediction_column=prediction_column,
        actual_column=actual_column,
    )
    origin_times = pd.to_datetime(rows["origin_time"], errors="coerce")
    if "target_datetime" in rows.columns:
        target_times = pd.to_datetime(rows["target_datetime"], errors="coerce")
    else:
        horizons = pd.to_numeric(rows["horizon"], errors="coerce")
        target_times = origin_times + pd.to_timedelta(horizons, unit="min")
    future_label_mask = target_times.gt(timestamp)
    future_origin_mask = origin_times.gt(timestamp)
    required = {
        "origin_time",
        "target",
        "horizon",
        "train_end",
        prediction_column,
        actual_column,
    }
    feature_columns = [
        column
        for column in rows.columns
        if column not in required and pd.api.types.is_numeric_dtype(rows[column])
    ]
    variants: dict[str, pd.DataFrame] = {}
    changed = rows.copy()
    changed.loc[future_label_mask, actual_column] = -999_999.0
    if feature_columns:
        changed.loc[future_origin_mask, feature_columns] = -999_999.0
    variants["modified"] = changed

    nulled = rows.copy()
    nulled.loc[future_label_mask, actual_column] = np.nan
    if feature_columns:
        nulled.loc[future_origin_mask, feature_columns] = np.nan
    variants["nulled"] = nulled
    variants["deleted"] = rows.loc[~future_origin_mask].copy()

    cases: dict[str, dict[str, object]] = {}
    for name, variant in variants.items():
        predicted = predict_matured_residual_at_origin(
            variant,
            timestamp,
            config=effective,
            prediction_column=prediction_column,
            actual_column=actual_column,
        )
        equal = baseline.equals(predicted)
        cases[name] = {
            "passed": bool(equal),
            "rows": int(len(predicted)),
        }
    return {
        "origin": str(timestamp),
        "prediction_count": int(len(baseline)),
        "passed": all(item["passed"] for item in cases.values()),
        "cases": cases,
    }


# 为交互式探索保留简短别名；OOF 构造仍使用完整函数名以强调其来源约束。
build_matured_residual_features = build_matured_residual_oof
predict_at_origin = predict_matured_residual_at_origin
audit_future_perturbation = audit_matured_residual_future_perturbation
