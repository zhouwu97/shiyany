"""严格逐时刻模拟的在线预测校准。"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _prediction_column(target: str, horizon: int) -> str:
    return f"{target}_t+{15 * horizon}_pred"


@dataclass(frozen=True)
class MaturedForecast:
    target: str
    horizon: int
    anchor: float
    base_prediction: float
    published_prediction: float


class OnlineForecastCalibrator:
    """一次只推进一个时间点的因果校准状态机。

    ``observe`` 先结算已经到达目标时刻的历史预测，``transform`` 再修正
    当前预测，``register`` 最后登记本次发布结果。调用顺序由
    :meth:`walk_forward` 固定，避免整块预先计算误差。
    """

    def __init__(
        self,
        horizons: tuple[int, ...],
        *,
        mode: str = "bias",
        half_life: float = 16.0,
        bias_clip: float = 12.0,
        gain_clip: tuple[float, float] = (0.0, 1.3),
        vintage_weight: float = 0.25,
        history_size: int = 64,
    ) -> None:
        if mode not in {"bias", "gain", "vintage"}:
            raise ValueError("在线校准模式必须是 bias、gain 或 vintage")
        if half_life <= 0 or history_size < 1:
            raise ValueError("half_life 和 history_size 必须为正数")
        if not 0.0 <= vintage_weight <= 1.0:
            raise ValueError("vintage_weight 必须位于 [0, 1]")
        self.horizons = tuple(horizons)
        self.mode = mode
        self.decay = float(np.exp(np.log(0.5) / half_life))
        self.bias_clip = float(bias_clip)
        self.gain_clip = tuple(float(value) for value in gain_clip)
        self.vintage_weight = float(vintage_weight)
        self.pending: dict[pd.Timestamp, list[MaturedForecast]] = defaultdict(list)
        self.vintages: dict[pd.Timestamp, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=history_size))
        )
        self.bias: dict[tuple[str, int], float] = defaultdict(float)
        self.gain_numerator: dict[tuple[str, str], float] = defaultdict(float)
        self.gain_denominator: dict[tuple[str, str], float] = defaultdict(float)
        self.last_observed_timestamp: pd.Timestamp | None = None

    def _gain_group(self, horizon: int) -> str:
        return "short" if self.horizons.index(horizon) < len(self.horizons) // 2 else "long"

    def observe(self, timestamp: pd.Timestamp, actual: pd.Series) -> None:
        """结算截至当前时刻已经揭晓且真实值可用的历史预测。

        EMA 的时间单位是观测时间戳，而不是“成熟了一条 forecast”。同一
        15 分钟内成熟的多个 horizon 先按状态组聚合，再只推进一次衰减。
        """

        timestamp = pd.Timestamp(timestamp)
        if self.last_observed_timestamp is not None and timestamp < self.last_observed_timestamp:
            raise ValueError("在线校准 observe 时间必须递增")
        if self.last_observed_timestamp is None:
            elapsed_steps = 0
        else:
            elapsed_minutes = (
                timestamp - self.last_observed_timestamp
            ).total_seconds() / 60.0
            elapsed_steps = max(0, int(round(elapsed_minutes / 15.0)))
        if elapsed_steps:
            time_decay = self.decay**elapsed_steps
            for key in list(self.bias):
                self.bias[key] *= time_decay
            for key in list(self.gain_numerator):
                self.gain_numerator[key] *= time_decay
            for key in list(self.gain_denominator):
                self.gain_denominator[key] *= time_decay
        self.last_observed_timestamp = timestamp

        bias_errors: dict[tuple[str, int], list[float]] = defaultdict(list)
        gain_stats: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        due_times = sorted(due for due in self.pending if due <= timestamp)
        for due_time in due_times:
            matured = self.pending.pop(due_time, [])
            still_pending: list[MaturedForecast] = []
            for item in matured:
                try:
                    value = float(actual.get(item.target, np.nan))
                except (TypeError, ValueError):
                    value = float("nan")
                if not np.isfinite(value):
                    still_pending.append(item)
                    continue
                key = (item.target, item.horizon)
                if self.mode == "bias":
                    bias_errors[key].append(value - item.published_prediction)
                elif self.mode == "gain":
                    delta = item.base_prediction - item.anchor
                    group = (item.target, self._gain_group(item.horizon))
                    weight = 1.0 / max(abs(value), 1.0)
                    gain_stats[group].append(
                        (weight * delta * (value - item.anchor), weight * delta * delta)
                    )
            if still_pending:
                self.pending[due_time].extend(still_pending)
        update_weight = 1.0 - self.decay
        for key, errors in bias_errors.items():
            self.bias[key] = float(
                np.clip(
                    self.bias[key] + update_weight * float(np.mean(errors)),
                    -self.bias_clip,
                    self.bias_clip,
                )
            )
        for group, stats in gain_stats.items():
            numerator = float(np.mean([item[0] for item in stats]))
            denominator = float(np.mean([item[1] for item in stats]))
            self.gain_numerator[group] += update_weight * numerator
            self.gain_denominator[group] += update_weight * denominator

    def transform(
        self,
        timestamp: pd.Timestamp,
        target: str,
        horizon: int,
        anchor: float,
        base_prediction: float,
    ) -> float:
        """使用截至当前时刻的状态修正一个基础预测。"""

        timestamp = pd.Timestamp(timestamp)
        prediction = float(base_prediction)
        if self.mode == "bias":
            prediction += self.bias[(target, horizon)]
        elif self.mode == "gain":
            group = (target, self._gain_group(horizon))
            denominator = self.gain_denominator[group]
            gain = 1.0 if denominator <= 1e-12 else self.gain_numerator[group] / denominator
            gain = float(np.clip(gain, *self.gain_clip))
            prediction = float(anchor + gain * (base_prediction - anchor))
        elif self.mode == "vintage":
            target_time = timestamp + pd.Timedelta(minutes=15 * horizon)
            previous = self.vintages[target_time][target]
            if previous:
                prediction = (1.0 - self.vintage_weight) * prediction + self.vintage_weight * float(
                    np.mean(previous)
                )
        return prediction

    def register(
        self,
        timestamp: pd.Timestamp,
        target: str,
        horizon: int,
        anchor: float,
        base_prediction: float,
        published_prediction: float,
    ) -> None:
        """登记本次发布的预测，等待目标时刻揭晓后再使用。"""

        timestamp = pd.Timestamp(timestamp)
        target_time = timestamp + pd.Timedelta(minutes=15 * horizon)
        self.pending[target_time].append(
            MaturedForecast(
                target=target,
                horizon=horizon,
                anchor=float(anchor),
                base_prediction=float(base_prediction),
                published_prediction=float(published_prediction),
            )
        )
        self.vintages[target_time][target].append(float(published_prediction))

    def walk_forward(
        self,
        base_predictions: pd.DataFrame,
        current: pd.DataFrame,
        targets: tuple[str, ...],
        *,
        allow_missing: bool = False,
    ) -> pd.DataFrame:
        """逐行生成校准结果，当前行真实值只在该行开始时可用。"""

        if not base_predictions.index.equals(current.index):
            raise ValueError("基础预测和当前观测必须使用相同时间索引")
        output = base_predictions.copy()
        for timestamp in base_predictions.index:
            self.observe(timestamp, current.loc[timestamp])
            for target in targets:
                if target not in current.columns:
                    continue
                anchor = float(current.loc[timestamp, target])
                if not np.isfinite(anchor):
                    continue
                for horizon in self.horizons:
                    column = _prediction_column(target, horizon)
                    if column not in base_predictions.columns:
                        continue
                    base = float(base_predictions.loc[timestamp, column])
                    if not np.isfinite(base):
                        continue
                    published = self.transform(timestamp, target, horizon, anchor, base)
                    output.loc[timestamp, column] = published
                    self.register(timestamp, target, horizon, anchor, base, published)
        if not allow_missing and not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("在线校准结果包含非有限值")
        return output


def apply_online_calibration(
    base_predictions: pd.DataFrame,
    current: pd.DataFrame,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
    *,
    mode: str = "bias",
    allow_missing: bool = False,
    **kwargs: object,
) -> pd.DataFrame:
    """便捷的冷启动 walk-forward 入口。"""

    calibrator = OnlineForecastCalibrator(horizons, mode=mode, **kwargs)
    return calibrator.walk_forward(
        base_predictions, current, targets, allow_missing=allow_missing
    )


def apply_online_calibration_hot_start(
    base_predictions: pd.DataFrame,
    current: pd.DataFrame,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
    *,
    calibration_predictions: pd.DataFrame,
    calibration_current: pd.DataFrame,
    mode: str = "bias",
    allow_missing: bool = False,
    **kwargs: object,
) -> pd.DataFrame:
    """以验证块之前真正 OOF 的预测历史初始化在线状态后再预测验证块。

    ``calibration_predictions`` 必须由每个起点之前训练的基础模型产生；本函数
    只负责保证时间边界与状态推进，不能接受训练集 fitted residual 的替代品。
    """

    if calibration_predictions.empty:
        raise ValueError("真正 hot start 需要非空的 OOF calibration history")
    if not calibration_predictions.index.equals(calibration_current.index):
        raise ValueError("calibration 预测和当前观测必须使用相同时间索引")
    if not base_predictions.index.equals(current.index):
        raise ValueError("验证预测和当前观测必须使用相同时间索引")
    if not calibration_predictions.index.is_monotonic_increasing:
        raise ValueError("calibration history 时间必须递增")
    if not base_predictions.index.is_monotonic_increasing:
        raise ValueError("验证块时间必须递增")
    if calibration_predictions.index.max() >= base_predictions.index.min():
        raise ValueError("calibration history 必须严格早于验证块")
    calibrator = OnlineForecastCalibrator(horizons, mode=mode, **kwargs)
    calibrator.walk_forward(
        calibration_predictions,
        calibration_current,
        targets,
        allow_missing=allow_missing,
    )
    return calibrator.walk_forward(
        base_predictions,
        current,
        targets,
        allow_missing=allow_missing,
    )


def apply_online_calibration_hot_start_pipeline(
    base_predictions: pd.DataFrame,
    current: pd.DataFrame,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
    *,
    calibration_predictions: pd.DataFrame,
    calibration_current: pd.DataFrame,
    modes: tuple[str, ...],
    allow_missing: bool = False,
    **kwargs: object,
) -> pd.DataFrame:
    """按给定顺序组合不超过两个真正 hot-start 在线模块。"""

    if not modes or len(modes) > 2:
        raise ValueError("在线组合必须包含一到两个模块")
    if len(set(modes)) != len(modes):
        raise ValueError("在线组合不能重复同一模块")
    if calibration_predictions.empty:
        raise ValueError("真正 hot start 需要非空的 OOF calibration history")
    if not calibration_predictions.index.equals(calibration_current.index):
        raise ValueError("calibration 预测和当前观测必须使用相同时间索引")
    if not base_predictions.index.equals(current.index):
        raise ValueError("验证预测和当前观测必须使用相同时间索引")
    if calibration_predictions.index.max() >= base_predictions.index.min():
        raise ValueError("calibration history 必须严格早于验证块")
    history = calibration_predictions.copy()
    output = base_predictions.copy()
    for mode in modes:
        calibrator = OnlineForecastCalibrator(horizons, mode=mode, **kwargs)
        history = calibrator.walk_forward(
            history,
            calibration_current,
            targets,
            allow_missing=allow_missing,
        )
        output = calibrator.walk_forward(
            output,
            current,
            targets,
            allow_missing=allow_missing,
        )
    return output


def apply_online_calibration_to_oof(
    rows: pd.DataFrame,
    base_column: str,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
    *,
    mode: str = "bias",
    output_column: str | None = None,
    warmup_rows: int = 0,
    **kwargs: object,
) -> pd.DataFrame:
    """将在线校准严格应用到 OOF 长表。

    每个外层折都从空状态开始，按 ``origin_time`` 逐行推进。状态只读取
    ``current_value``，而不是 OOF 行中的 ``actual``，因此未来真实值不会
    参与当前时刻的校准。``warmup_rows`` 用于折内 warm-up 评估：每折前若干个
    origin 仅用于填充状态，并通过标记列排除出正式评分；它不是跨折外部热启动。
    """

    required = {
        "fold",
        "origin_time",
        "target",
        "horizon",
        "actual",
        "current_value",
        base_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"OOF 长表缺少在线校准字段: {missing}")
    if not targets or not horizons:
        raise ValueError("在线校准至少需要一个目标和一个预测步长")
    if not isinstance(warmup_rows, int) or warmup_rows < 0:
        raise ValueError("warmup_rows 必须是非负整数")

    output_column = output_column or (
        f"{base_column.removesuffix('_pred')}_online_{mode}_pred"
    )
    warmup_column = f"{output_column}_is_warmup"
    fallback_column = f"{output_column}_is_fallback"
    if (
        output_column in rows.columns
        or warmup_column in rows.columns
        or fallback_column in rows.columns
    ):
        raise ValueError(f"OOF 长表已存在在线校准输出列: {output_column}")

    work = rows.reset_index(drop=True).copy()
    work["origin_time"] = pd.to_datetime(work["origin_time"])
    work[output_column] = pd.to_numeric(work[base_column], errors="raise")
    work[warmup_column] = False
    work[fallback_column] = False
    expected_pairs = {(target, 15 * horizon) for target in targets for horizon in horizons}

    for fold_name, fold_rows in work.groupby("fold", sort=False):
        positions = fold_rows.index
        part = fold_rows.sort_values(
            ["origin_time", "target", "horizon"], kind="stable"
        )
        observed_pairs = set(zip(part["target"], part["horizon"].astype(int)))
        if observed_pairs != expected_pairs:
            missing_pairs = sorted(expected_pairs.difference(observed_pairs))
            extra_pairs = sorted(observed_pairs.difference(expected_pairs))
            raise ValueError(
                f"折 {fold_name} 的目标×步长不完整，缺失={missing_pairs}，多余={extra_pairs}"
            )
        if part.duplicated(["origin_time", "target", "horizon"]).any():
            raise ValueError(f"折 {fold_name} 存在重复的 origin/target/horizon OOF 行")
        if part["horizon"].astype(int).mod(15).ne(0).any():
            raise ValueError("OOF horizon 必须使用 15 分钟为单位")

        current_rows = part[["origin_time", "target", "current_value"]].copy()
        current_variants = current_rows.groupby(
            ["origin_time", "target"], sort=False, dropna=False
        )["current_value"].nunique(dropna=False)
        if current_variants.gt(1).any():
            raise ValueError(f"折 {fold_name} 同一 origin/target 的 current_value 不一致")
        current_wide = (
            current_rows.drop_duplicates(["origin_time", "target"])
            .pivot(index="origin_time", columns="target", values="current_value")
            .reindex(columns=list(targets))
            .sort_index()
        )
        prediction_wide = part.pivot(
            index="origin_time", columns=["target", "horizon"], values=base_column
        )
        prediction_columns = [
            _prediction_column(target, horizon) for target in targets for horizon in horizons
        ]
        prediction_wide.columns = [
            _prediction_column(str(target), int(horizon) // 15)
            for target, horizon in prediction_wide.columns
        ]
        prediction_wide = prediction_wide.reindex(columns=prediction_columns).sort_index()
        prediction_wide = prediction_wide.apply(pd.to_numeric, errors="coerce")
        current_wide = current_wide.apply(pd.to_numeric, errors="coerce")
        complete_mask = prediction_wide.notna().all(axis=1) & current_wide.notna().all(axis=1)
        complete_mask &= np.isfinite(prediction_wide).all(axis=1)
        complete_mask &= np.isfinite(current_wide).all(axis=1)
        calibrated = apply_online_calibration(
            prediction_wide,
            current_wide,
            targets,
            horizons,
            mode=mode,
            allow_missing=True,
            **kwargs,
        )
        calibrated_records: list[pd.DataFrame] = []
        for target in targets:
            for horizon in horizons:
                column = _prediction_column(target, horizon)
                record = calibrated[[column]].rename(columns={column: "value"})
                record["origin_time"] = record.index
                record["target"] = target
                record["horizon"] = 15 * horizon
                calibrated_records.append(record.reset_index(drop=True))
        calibrated_long = pd.concat(calibrated_records, ignore_index=True).set_index(
            ["origin_time", "target", "horizon"]
        )["value"]
        part_keys = pd.MultiIndex.from_frame(
            part[["origin_time", "target", "horizon"]].assign(
                horizon=part["horizon"].astype(int)
            )
        )
        values = calibrated_long.reindex(part_keys).to_numpy(dtype=float)
        finite_values = np.isfinite(values)
        if finite_values.any():
            work.loc[part.index[finite_values], output_column] = values[finite_values]

        warmup_origins = set(prediction_wide.index[:warmup_rows])
        work.loc[positions, warmup_column] = work.loc[positions, "origin_time"].isin(
            warmup_origins
        )
        fallback_origins = set(prediction_wide.index[~complete_mask])
        work.loc[positions, fallback_column] = work.loc[positions, "origin_time"].isin(
            fallback_origins
        )

    if work[output_column].isna().any() or not np.isfinite(work[output_column]).all():
        raise ValueError("在线校准没有为所有 OOF 行生成有限预测")
    work.index = rows.index
    return work
