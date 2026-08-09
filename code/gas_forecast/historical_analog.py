"""严格因果的历史轨迹相似样本实验。

这个模块故意不接入正式训练或提交入口。它只登记六个预注册的
``context x neighbors`` 组合，并在每个外层折中用训练期内的嵌套滚动
验证冻结当折配置。相似样本的完整 120 分钟轨迹必须在 held fold 开始
前结束，因此预测时不会读取 held fold 或 origin 之后的生产观测。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from gas_forecast.config import ForecastConfig
from gas_forecast.scoring import absolute_percentage_error, competition_mape, score_oof_long
from gas_forecast.splits import TimeFold, make_inner_folds, make_outer_folds


TARGETS: tuple[str, ...] = ("generator_1", "generator_all")
HORIZONS: tuple[int, ...] = tuple(range(1, 9))
STEP_MINUTES = 15
MAX_TRAJECTORY_MINUTES = 120
SCREENING_FOLDS = 5
SCREENING_MIN_IMPROVEMENT_PP = 0.02
SCREENING_MIN_WINS = 3


@dataclass(frozen=True, order=True)
class HistoricalAnalogSpec:
    """一个不可变的、预注册的历史相似样本配置。"""

    context: int
    neighbors: int
    metric: str = "euclidean"

    @property
    def name(self) -> str:
        """返回适合 OOF 列名和 trace 的稳定名称。"""

        return f"analog_c{self.context}_k{self.neighbors}"

    def prediction_column(self) -> str:
        """返回该固定配置在统一 OOF 长表中的预测列名。"""

        return f"{self.name}_pred"


PRE_REGISTERED_SPECS: tuple[HistoricalAnalogSpec, ...] = tuple(
    HistoricalAnalogSpec(context=context, neighbors=neighbors)
    for context in (16, 32)
    for neighbors in (8, 16, 32)
)


@dataclass(frozen=True)
class HistoricalAnalogResult:
    """历史相似样本 OOF、嵌套选择 trace 和统一报告。"""

    rows: pd.DataFrame
    trace: pd.DataFrame
    report: dict[str, object]


@dataclass(frozen=True)
class _PreparedTarget:
    """一个目标的全量因果特征和历史增量轨迹缓存。"""

    index: pd.DatetimeIndex
    values: np.ndarray
    features_by_context: Mapping[int, np.ndarray]
    trajectories: np.ndarray


def validate_pre_registered_specs(
    specs: Iterable[HistoricalAnalogSpec] = PRE_REGISTERED_SPECS,
) -> tuple[HistoricalAnalogSpec, ...]:
    """拒绝未预注册配置，防止运行后扩展上下文或近邻数。"""

    values = tuple(specs)
    if not values:
        raise ValueError("历史相似样本至少需要一个配置")
    expected = set(PRE_REGISTERED_SPECS)
    invalid = [spec for spec in values if spec not in expected or spec.metric != "euclidean"]
    if invalid:
        raise ValueError(f"发现未预注册的历史相似样本配置: {invalid}")
    if len(values) != len(set(values)):
        raise ValueError("历史相似样本配置不能重复")
    return values


def _validate_frame(frame: pd.DataFrame, targets: tuple[str, ...]) -> pd.DataFrame:
    """校验 15 分钟生产序列与本实验固定的两个目标。"""

    missing = sorted(set(targets).difference(frame.columns))
    if missing:
        raise ValueError(f"历史相似样本缺少目标列: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("历史相似样本输入必须使用 DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("历史相似样本时间索引必须严格递增且不重复")
    if len(frame.index) < max(HORIZONS) + max(spec.context for spec in PRE_REGISTERED_SPECS) + 1:
        raise ValueError("历史相似样本数据不足以构造上下文和 120 分钟轨迹")
    gaps = frame.index.to_series().diff().dropna()
    if not gaps.eq(pd.Timedelta(minutes=STEP_MINUTES)).all():
        raise ValueError("历史相似样本仅支持连续的 15 分钟生产网格")
    return frame


def _causal_feature_matrix(values: np.ndarray, context: int) -> np.ndarray:
    """构造只在行末使用当前及历史观测的相似度特征。

    特征依次是 level、一阶差分、最近四点斜率、context 内滚动标准差和
    相对当前值、按滚动标准差归一化的形状向量。这里不调用居中窗口，
    因而修改该时刻后的生产值不会改变这一行。
    """

    if context not in (16, 32):
        raise ValueError("context 只能是预注册的 16 或 32")
    result = np.full((len(values), 4 + context), np.nan, dtype=float)
    if len(values) < context:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, context)
    valid = np.isfinite(windows).all(axis=1)
    if not valid.any():
        return result
    rows = np.arange(context - 1, len(values), dtype=int)[valid]
    selected = windows[valid]
    level = selected[:, -1]
    diff_1 = selected[:, -1] - selected[:, -2]
    slope_4 = (selected[:, -1] - selected[:, -4]) / 3.0
    rolling_std = selected.std(axis=1, ddof=0)
    scale = np.maximum(rolling_std, 1e-6)
    normalized_shape = (selected - level[:, None]) / scale[:, None]
    result[rows, :4] = np.column_stack((level, diff_1, slope_4, rolling_std))
    result[rows, 4:] = normalized_shape
    return result


def _trajectory_deltas(values: np.ndarray) -> np.ndarray:
    """生成每个历史 origin 的八步绝对增量，末尾不可用位置显式为 NaN。"""

    result = np.full((len(values), len(HORIZONS)), np.nan, dtype=float)
    for offset, horizon in enumerate(HORIZONS):
        result[:-horizon, offset] = values[horizon:] - values[:-horizon]
    return result


def _prepare_targets(frame: pd.DataFrame, targets: tuple[str, ...]) -> dict[str, _PreparedTarget]:
    """一次性缓存各目标的因果向量，避免折间重算且不共享未来统计量。"""

    prepared: dict[str, _PreparedTarget] = {}
    for target in targets:
        values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        prepared[target] = _PreparedTarget(
            index=frame.index,
            values=values,
            features_by_context={
                context: _causal_feature_matrix(values, context) for context in (16, 32)
            },
            trajectories=_trajectory_deltas(values),
        )
    return prepared


def _strict_candidate_positions(
    prepared: _PreparedTarget,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    context: int,
) -> np.ndarray:
    """返回候选窗口和完整未来轨迹都位于训练截止点前的 origins。

    ``train_end`` 是本次类比搜索可见历史的严格截止点，而不是“最后一个
    可以拿来当候选 origin 的时刻”。候选 ``j`` 的 context 窗口必须从
    ``train_start`` 之后开始，且第八步真值 ``j + 120min`` 必须严格早于
    ``train_end``。这会拒绝虽然早于 held fold、但借用了训练截止点之后
    标签的伪历史候选。
    """

    trajectory_end = prepared.index + pd.Timedelta(minutes=MAX_TRAJECTORY_MINUTES)
    context_start = prepared.index - pd.Timedelta(minutes=STEP_MINUTES * (context - 1))
    before_training_cutoff = trajectory_end < pd.Timestamp(train_end)
    before_held_fold = trajectory_end < pd.Timestamp(validation_start)
    window_in_training_history = context_start >= pd.Timestamp(train_start)
    features = prepared.features_by_context[context]
    valid = (
        before_training_cutoff
        & before_held_fold
        & window_in_training_history
        & np.isfinite(features).all(axis=1)
        & np.isfinite(prepared.trajectories).all(axis=1)
    )
    return np.flatnonzero(valid)


def _standardize_from_training(
    train: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """只用候选轨迹拟合特征统计量，再变换候选和查询向量。"""

    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (train - mean) / scale, (query - mean) / scale


def _predict_target_for_origins(
    prepared: _PreparedTarget,
    origins: pd.DatetimeIndex,
    spec: HistoricalAnalogSpec,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    """以一个固定配置对若干 held origins 预测八步增量轨迹。

    重要边界：候选 ``j`` 的 context 窗口与最后标签 ``j + 120min`` 都
    必须位于 ``train_end`` 严格之前；随后仍额外验证它早于 held fold。
    特征标准化也只在这些完整历史候选轨迹上拟合。
    """

    positions = prepared.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("预测 origin 不在生产时间网格中")
    features = prepared.features_by_context[spec.context]
    candidate_positions = _strict_candidate_positions(
        prepared,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        context=spec.context,
    )
    query = features[positions]
    count = len(origins)
    if len(candidate_positions) < spec.neighbors:
        raise ValueError(
            "严格历史候选不足："
            f"配置 {spec.name} 需要 {spec.neighbors} 条，实际只有 {len(candidate_positions)} 条"
        )
    prediction_delta = np.zeros((count, len(HORIZONS)), dtype=float)
    nearest_distance = np.full(count, np.nan, dtype=float)
    median_neighbor_distance = np.full(count, np.nan, dtype=float)
    effective_neighbor_count = np.zeros(count, dtype=float)
    uncertainty = np.full((count, len(HORIZONS)), np.nan, dtype=float)
    fallback = np.zeros(count, dtype=bool)

    valid_query = np.isfinite(query).all(axis=1) & np.isfinite(prepared.values[positions])
    if not valid_query.all():
        invalid_count = int((~valid_query).sum())
        raise ValueError(f"严格历史类比查询缺少完整因果窗口或当前值: {invalid_count} 条")
    train = features[candidate_positions]
    scaled_train, scaled_query = _standardize_from_training(train, query)
    # metric 固定为 Euclidean；NearestNeighbors 仅消费训练期冻结后的向量。
    finder = NearestNeighbors(n_neighbors=spec.neighbors, metric="euclidean")
    finder.fit(scaled_train)
    distances, local_indices = finder.kneighbors(scaled_query, return_distance=True)
    selected_deltas = prepared.trajectories[candidate_positions[local_indices]]
    prediction_delta[:] = np.median(selected_deltas, axis=1)
    nearest_distance[:] = distances[:, 0]
    median_neighbor_distance[:] = np.median(distances, axis=1)
    effective_neighbor_count[:] = float(spec.neighbors)
    uncertainty[:] = selected_deltas.std(axis=1, ddof=0)

    anchor = prepared.values[positions]
    predictions = anchor[:, None] + prediction_delta
    data: dict[str, object] = {
        "origin_time": origins,
        "nearest_distance": nearest_distance,
        "median_neighbor_distance": median_neighbor_distance,
        "effective_neighbor_count": effective_neighbor_count,
        "fallback_to_persistence": fallback,
        "candidate_pool_count": np.full(count, len(candidate_positions), dtype=int),
        "candidate_window_and_trajectory_before_train_cutoff": np.full(
            count, True, dtype=bool
        ),
        "candidate_trajectory_end_before_train_cutoff": np.full(count, True, dtype=bool),
        "candidate_trajectory_end_before_validation": np.full(count, True, dtype=bool),
    }
    for offset, horizon in enumerate(HORIZONS):
        minutes = STEP_MINUTES * horizon
        data[f"prediction_t+{minutes}"] = predictions[:, offset]
        data[f"analog_uncertainty_t+{minutes}"] = uncertainty[:, offset]
    return pd.DataFrame(data)


def _prediction_long(
    prepared: Mapping[str, _PreparedTarget],
    origins: pd.DatetimeIndex,
    spec: HistoricalAnalogSpec,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """将两目标八步预测统一成长表，并保留每行的距离诊断。"""

    parts: list[pd.DataFrame] = []
    for target in targets:
        wide = _predict_target_for_origins(
            prepared[target],
            origins,
            spec,
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
        )
        for horizon in horizons:
            minutes = STEP_MINUTES * horizon
            part = wide.loc[
                :,
                [
                    "origin_time",
                    "nearest_distance",
                    "median_neighbor_distance",
                    "effective_neighbor_count",
                    "fallback_to_persistence",
                    "candidate_pool_count",
                    "candidate_window_and_trajectory_before_train_cutoff",
                    "candidate_trajectory_end_before_train_cutoff",
                    "candidate_trajectory_end_before_validation",
                    f"analog_uncertainty_t+{minutes}",
                    f"prediction_t+{minutes}",
                ],
            ].copy()
            part["target"] = target
            part["horizon"] = minutes
            part = part.rename(
                columns={
                    f"analog_uncertainty_t+{minutes}": "analog_uncertainty",
                    f"prediction_t+{minutes}": "prediction",
                }
            )
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _inner_selection(
    prepared: Mapping[str, _PreparedTarget],
    frame: pd.DataFrame,
    fold: TimeFold,
    specs: tuple[HistoricalAnalogSpec, ...],
    *,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
    inner_folds: int,
) -> tuple[HistoricalAnalogSpec, dict[str, object]]:
    """仅用 outer-train 内的嵌套 expanding folds 冻结 held fold 配置。"""

    train_index = frame.index[(frame.index >= fold.train_start) & (frame.index <= fold.train_end)]
    try:
        nested_folds = make_inner_folds(
            train_index,
            folds=inner_folds,
            purge_steps=max(HORIZONS),
        )
    except ValueError as error:
        # 数据不足时不允许偷看 held fold；使用字典序第一个预注册配置并登记原因。
        selected = specs[0]
        return selected, {
            "nested_cross_fitting": False,
            "selection_reason": "insufficient_inner_history",
            "selection_error": str(error),
            "selected_spec": selected.name,
            "inner_folds": [],
            "scores": {},
        }

    scores: dict[str, list[float]] = {spec.name: [] for spec in specs}
    for nested in nested_folds:
        origins = train_index[(train_index >= nested.validation_start) & (train_index < nested.validation_end)]
        for spec in specs:
            prediction = _prediction_long(
                prepared,
                origins,
                spec,
                train_start=nested.train_start,
                train_end=nested.train_end,
                validation_start=nested.validation_start,
                targets=targets,
                horizons=horizons,
            )
            actual_parts: list[pd.DataFrame] = []
            for target in targets:
                values = prepared[target].values
                positions = prepared[target].index.get_indexer(origins)
                for horizon in horizons:
                    actual = np.full(len(origins), np.nan, dtype=float)
                    valid = positions + horizon < len(values)
                    actual[valid] = values[positions[valid] + horizon]
                    actual_parts.append(
                        pd.DataFrame(
                            {
                                "origin_time": origins,
                                "target": target,
                                "horizon": STEP_MINUTES * horizon,
                                "actual": actual,
                            }
                        )
                    )
            actuals = pd.concat(actual_parts, ignore_index=True)
            joined = prediction.merge(
                actuals,
                on=["origin_time", "target", "horizon"],
                how="left",
                validate="one_to_one",
            )
            score = competition_mape(joined["actual"], joined["prediction"])
            if np.isfinite(score):
                scores[spec.name].append(float(score))

    mean_scores = {
        name: float(np.mean(values)) if values else float("inf") for name, values in scores.items()
    }
    selected = min(specs, key=lambda spec: (mean_scores[spec.name], spec.context, spec.neighbors))
    return selected, {
        "nested_cross_fitting": True,
        "selection_reason": "lowest_inner_pooled_mape",
        "selected_spec": selected.name,
        "inner_folds": [item.name for item in nested_folds],
        "scores": mean_scores,
    }


def select_historical_analog_folds(
    index: pd.DatetimeIndex,
    config: ForecastConfig,
    *,
    scope: str,
) -> list[TimeFold]:
    """选择非 blind 的开发折；screening 固定为最早五个开发折。"""

    if scope not in {"screening", "development"}:
        raise ValueError("历史相似样本只支持 screening 或 development，禁止读取 blind")
    development = [fold for fold in make_outer_folds(index, config) if not fold.blind]
    if scope == "screening":
        return development[:SCREENING_FOLDS]
    return development


def _base_rows(
    prepared: Mapping[str, _PreparedTarget],
    fold: TimeFold,
    *,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """构造与项目统一 OOF 契约一致的真实值与 persistence 行。"""

    origins = prepared[targets[0]].index[
        (prepared[targets[0]].index >= fold.validation_start)
        & (prepared[targets[0]].index < fold.validation_end)
    ]
    parts: list[pd.DataFrame] = []
    for target in targets:
        target_prepared = prepared[target]
        positions = target_prepared.index.get_indexer(origins)
        current = target_prepared.values[positions]
        for horizon in horizons:
            actual = np.full(len(origins), np.nan, dtype=float)
            valid = positions + horizon < len(target_prepared.values)
            actual[valid] = target_prepared.values[positions[valid] + horizon]
            keep = np.isfinite(current) & np.isfinite(actual)
            parts.append(
                pd.DataFrame(
                    {
                        "fold": fold.name,
                        "origin_time": origins[keep],
                        "train_end": fold.train_end,
                        "target": target,
                        "horizon": STEP_MINUTES * horizon,
                        "actual": actual[keep],
                        "current_value": current[keep],
                        "persistence_pred": current[keep],
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def _screening_summary(rows: pd.DataFrame, prediction_column: str) -> dict[str, object]:
    """按预注册的前五折 pooled 改善和折胜数给出机械 STOP/PASS。"""

    candidate = score_oof_long(rows, prediction_column)["pooled_mape"]
    baseline = score_oof_long(rows, "persistence_pred")["pooled_mape"]
    by_fold: list[dict[str, object]] = []
    wins = 0
    for fold_name, part in rows.groupby("fold", sort=True):
        candidate_score = competition_mape(part["actual"], part[prediction_column])
        baseline_score = competition_mape(part["actual"], part["persistence_pred"])
        win = bool(candidate_score < baseline_score)
        wins += int(win)
        by_fold.append(
            {
                "fold": str(fold_name),
                "candidate_mape": float(candidate_score),
                "persistence_mape": float(baseline_score),
                "improvement_pp": float((baseline_score - candidate_score) * 100.0),
                "win": win,
            }
        )
    improvement_pp = float((baseline - candidate) * 100.0)
    available_folds = len(by_fold)
    passed = (
        available_folds >= SCREENING_FOLDS
        and improvement_pp >= SCREENING_MIN_IMPROVEMENT_PP
        and wins >= SCREENING_MIN_WINS
    )
    return {
        "pooled_candidate_mape": float(candidate),
        "pooled_persistence_mape": float(baseline),
        "pooled_improvement_pp": improvement_pp,
        "fold_wins": int(wins),
        "folds_evaluated": available_folds,
        "required_folds": SCREENING_FOLDS,
        "required_improvement_pp": SCREENING_MIN_IMPROVEMENT_PP,
        "required_wins": SCREENING_MIN_WINS,
        "status": "PASS" if passed else "STOP",
        "passed": passed,
        "by_fold": by_fold,
    }


def build_historical_analog_oof(
    frame: pd.DataFrame,
    *,
    config: ForecastConfig | None = None,
    scope: str = "screening",
    specs: Iterable[HistoricalAnalogSpec] = PRE_REGISTERED_SPECS,
) -> HistoricalAnalogResult:
    """运行六个固定配置的严格 OOF，并返回每折嵌套选择后的统一预测。

    ``scope`` 永远不包含 blind。每个 outer fold 的配置由 outer-train 内的
    expanding inner folds 决定；所有距离标准化统计量都在对应候选轨迹池上
    拟合，候选轨迹的最后一个真值严格早于 held fold。
    """

    config = config or ForecastConfig()
    targets = tuple(config.targets)
    if targets != TARGETS:
        raise ValueError("历史相似样本第一版只允许 generator_1 和 generator_all")
    horizons = tuple(config.feature.horizons)
    if not horizons or any(horizon not in HORIZONS for horizon in horizons):
        raise ValueError("历史相似样本只支持 15 到 120 分钟的八个预注册步长")
    values = _validate_frame(frame, targets)
    registered = validate_pre_registered_specs(specs)
    folds = select_historical_analog_folds(values.index, config, scope=scope)
    if not folds:
        raise ValueError("没有可用的非 blind 开发折")
    prepared = _prepare_targets(values, targets)
    parts: list[pd.DataFrame] = []
    trace_rows: list[dict[str, object]] = []

    for fold in folds:
        selected, selection_trace = _inner_selection(
            prepared,
            values,
            fold,
            registered,
            targets=targets,
            horizons=horizons,
            inner_folds=config.model.inner_folds,
        )
        base = _base_rows(prepared, fold, targets=targets, horizons=horizons)
        validation_origins = pd.DatetimeIndex(sorted(base["origin_time"].unique()))
        keys = ["origin_time", "target", "horizon"]
        selected_diagnostics: pd.DataFrame | None = None
        for spec in registered:
            prediction = _prediction_long(
                prepared,
                validation_origins,
                spec,
                train_start=fold.train_start,
                train_end=fold.train_end,
                validation_start=fold.validation_start,
                targets=targets,
                horizons=horizons,
            )
            prediction_column = spec.prediction_column()
            renamed = prediction.rename(
                columns={
                    "prediction": prediction_column,
                    "nearest_distance": f"{spec.name}_nearest_distance",
                    "median_neighbor_distance": f"{spec.name}_median_neighbor_distance",
                    "effective_neighbor_count": f"{spec.name}_effective_neighbor_count",
                    "analog_uncertainty": f"{spec.name}_analog_uncertainty",
                    "fallback_to_persistence": f"{spec.name}_fallback_to_persistence",
                    "candidate_pool_count": f"{spec.name}_candidate_pool_count",
                    "candidate_window_and_trajectory_before_train_cutoff": (
                        f"{spec.name}_window_and_trajectory_before_train_cutoff"
                    ),
                    "candidate_trajectory_end_before_train_cutoff": (
                        f"{spec.name}_trajectory_before_train_cutoff"
                    ),
                    "candidate_trajectory_end_before_validation": (
                        f"{spec.name}_trajectory_safe"
                    ),
                }
            )
            base = base.merge(renamed, on=keys, how="left", validate="one_to_one")
            if spec == selected:
                selected_diagnostics = prediction
        if selected_diagnostics is None:
            raise RuntimeError("嵌套选择没有找到可用的预注册配置")
        selected_prediction = selected.prediction_column()
        base["historical_analog_pred"] = base[selected_prediction]
        # 统一 OOF 契约要求 generic prediction 字段保存当折冻结配置的输出。
        base["prediction"] = base["historical_analog_pred"]
        generic_diagnostics = selected_diagnostics.loc[
            :,
            keys
            + [
                "nearest_distance",
                "median_neighbor_distance",
                "effective_neighbor_count",
                "analog_uncertainty",
                "fallback_to_persistence",
                "candidate_pool_count",
                "candidate_window_and_trajectory_before_train_cutoff",
                "candidate_trajectory_end_before_train_cutoff",
                "candidate_trajectory_end_before_validation",
            ],
        ]
        base = base.merge(generic_diagnostics, on=keys, how="left", validate="one_to_one")
        base["selected_context"] = selected.context
        base["selected_neighbors"] = selected.neighbors
        base["selected_metric"] = selected.metric
        base["selection_used_held_fold_labels"] = False
        if not base["candidate_window_and_trajectory_before_train_cutoff"].all():
            raise RuntimeError(f"折 {fold.name} 存在越过训练截止点的 analog 候选")
        if not base["candidate_trajectory_end_before_train_cutoff"].all():
            raise RuntimeError(f"折 {fold.name} 存在越过训练截止点的 analog 未来标签")
        if not base["candidate_trajectory_end_before_validation"].all():
            raise RuntimeError(f"折 {fold.name} 存在跨 held fold 的 analog trajectory")
        parts.append(base)
        trace_rows.append(
            {
                "fold": fold.name,
                "scope": scope,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "candidate_trajectory_latest_end": fold.train_end
                - pd.Timedelta(minutes=STEP_MINUTES),
                "candidate_training_cutoff": fold.train_end,
                "candidate_window_and_trajectory_before_train_cutoff": True,
                "selected_context": selected.context,
                "selected_neighbors": selected.neighbors,
                "selected_metric": selected.metric,
                "selection_used_held_fold_labels": False,
                "selection_data_end": fold.train_end,
                "candidate_pool_max_time": fold.train_end
                - pd.Timedelta(minutes=MAX_TRAJECTORY_MINUTES + STEP_MINUTES),
                "nested_cross_fitting": bool(selection_trace["nested_cross_fitting"]),
                "selection_reason": str(selection_trace["selection_reason"]),
                "inner_folds": json.dumps(selection_trace["inner_folds"]),
                "inner_scores": json.dumps(selection_trace["scores"], sort_keys=True),
                "selection_error": selection_trace.get("selection_error"),
            }
        )

    rows = pd.concat(parts, ignore_index=True).sort_values(
        ["fold", "origin_time", "target", "horizon"], kind="stable"
    )
    rows["persistence_ape"] = absolute_percentage_error(
        rows["actual"], rows["persistence_pred"]
    )
    rows["historical_analog_ape"] = absolute_percentage_error(
        rows["actual"], rows["historical_analog_pred"]
    )
    metrics = {
        "persistence": score_oof_long(rows, "persistence_pred"),
        "historical_analog": score_oof_long(rows, "historical_analog_pred"),
        **{
            spec.name: score_oof_long(rows, spec.prediction_column()) for spec in registered
        },
    }
    screening_rows = rows.loc[rows["fold"].isin([fold.name for fold in folds[:SCREENING_FOLDS]])]
    screening = _screening_summary(screening_rows, "historical_analog_pred")
    report: dict[str, object] = {
        "experiment": "historical_analog",
        "scope": scope,
        "blind_included": False,
        "blind_labels_used": False,
        "future_generator_truth_used_for_prediction": False,
        "platform_reference_used": False,
        "targets": list(targets),
        "horizons_minutes": [STEP_MINUTES * horizon for horizon in horizons],
        "pre_registered_search": [asdict(spec) for spec in registered],
        "distance_metric": "euclidean",
        "trajectory_rule": "candidate_window_start >= train_start and candidate_j + 120min < train_end",
        "nested_cross_fitting": True,
        "folds": [fold.name for fold in folds],
        "metrics": metrics,
        "screening": screening,
        "full_development_gate": screening if scope == "screening" else _screening_summary(
            rows, "historical_analog_pred"
        ),
        "oof_rows": int(len(rows)),
        "trace_rows": int(len(trace_rows)),
    }
    return HistoricalAnalogResult(
        rows=rows.reset_index(drop=True),
        trace=pd.DataFrame(trace_rows),
        report=report,
    )


def predict_historical_analog_at_origin(
    frame: pd.DataFrame,
    origin: pd.Timestamp | str,
    *,
    spec: HistoricalAnalogSpec = HistoricalAnalogSpec(16, 8),
    targets: tuple[str, ...] = TARGETS,
) -> pd.DataFrame:
    """在单个 origin 产出 16 个严格因果预测和四项距离诊断。

    该公共入口用于未来扰动审计。它把下一个 15 分钟刻度作为严格训练
    截止点，因此候选完整轨迹至多结束在 origin 本身，绝不读取 origin
    之后的生产值。
    """

    validate_pre_registered_specs((spec,))
    targets = tuple(targets)
    if targets != TARGETS:
        raise ValueError("历史相似样本第一版只允许两个 generator 目标")
    values = _validate_frame(frame, targets)
    timestamp = pd.Timestamp(origin)
    if timestamp not in values.index:
        raise ValueError("origin 不在生产时间网格中")
    prepared = _prepare_targets(values, targets)
    prediction = _prediction_long(
        prepared,
        pd.DatetimeIndex([timestamp]),
        spec,
        train_start=values.index.min(),
        train_end=timestamp + pd.Timedelta(minutes=STEP_MINUTES),
        validation_start=timestamp + pd.Timedelta(minutes=STEP_MINUTES),
        targets=targets,
        horizons=HORIZONS,
    )
    result = prediction.rename(columns={"prediction": "prediction"}).copy()
    result["context"] = spec.context
    result["neighbors"] = spec.neighbors
    return result.sort_values(["target", "horizon"], kind="stable").reset_index(drop=True)


def audit_historical_analog_future_perturbation(
    frame: pd.DataFrame,
    *,
    origin: pd.Timestamp | str | None = None,
    spec: HistoricalAnalogSpec = HistoricalAnalogSpec(16, 8),
) -> dict[str, object]:
    """对修改、打乱、置空和删除 origin 后生产数据做 16 元素不变性审计。"""

    values = _validate_frame(frame, TARGETS)
    if origin is None:
        # 保留足够的后续行供四种扰动实际发生，同时让候选池有较长历史。
        position = max(32 + max(HORIZONS), int(len(values) * 0.75))
        position = min(position, len(values) - 2)
        timestamp = values.index[position]
    else:
        timestamp = pd.Timestamp(origin)
    baseline = predict_historical_analog_at_origin(values, timestamp, spec=spec)
    expected = baseline["prediction"].to_numpy(dtype=float)
    if len(expected) != len(TARGETS) * len(HORIZONS):
        raise RuntimeError("未来扰动审计必须覆盖两个目标的 16 个预测")
    future = values.index > timestamp
    numeric = list(values.select_dtypes(include=[np.number]).columns)
    variants: dict[str, pd.DataFrame] = {}

    modified = values.copy()
    modified.loc[future, numeric] = -999_999.0
    variants["modified"] = modified

    shuffled = values.copy()
    future_rows = shuffled.loc[future, numeric]
    shuffled.loc[future, numeric] = future_rows.iloc[::-1].to_numpy()
    variants["shuffled"] = shuffled

    nulled = values.copy()
    nulled.loc[future, numeric] = np.nan
    variants["nulled"] = nulled

    variants["deleted"] = values.loc[~future].copy()

    cases: dict[str, dict[str, object]] = {}
    for name, candidate in variants.items():
        predicted = predict_historical_analog_at_origin(candidate, timestamp, spec=spec)
        observed = predicted["prediction"].to_numpy(dtype=float)
        equal = np.isclose(expected, observed, rtol=0.0, atol=0.0, equal_nan=True)
        cases[name] = {
            "passed": bool(equal.all()),
            "changed_prediction_positions": np.flatnonzero(~equal).astype(int).tolist(),
            "prediction_count": int(len(observed)),
        }
    return {
        "passed": all(case["passed"] for case in cases.values()),
        "origin": str(timestamp),
        "context": spec.context,
        "neighbors": spec.neighbors,
        "prediction_count": int(len(expected)),
        "cases": cases,
    }


# 简短别名使外部审计和交互式使用不必依赖内部函数命名。
predict_at_origin = predict_historical_analog_at_origin
audit_future_perturbation = audit_historical_analog_future_perturbation
