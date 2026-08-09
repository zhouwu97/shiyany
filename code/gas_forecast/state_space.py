"""严格因果的 Local Linear Trend / Kalman 多样性实验。

本模块只从 ``frame.loc[:origin]`` 读取生产观测。所有阻尼参数和融合比例
均为预登记有限集合；选择只在 development 折的历史结果上前向进行，拒绝
读取 blind 标签或未来生产量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from gas_forecast.scoring import competition_mape


STATE_SPACE_HORIZONS: Final[tuple[int, ...]] = (15, 30, 45, 60, 75, 90, 105, 120)
STATE_SPACE_TARGETS: Final[tuple[str, ...]] = ("generator_1", "generator_all")
STATE_SPACE_DAMPING: Final[tuple[float, ...]] = (0.70, 0.85, 0.95)
STATE_SPACE_BLEND_WEIGHTS: Final[tuple[float, ...]] = (0.05, 0.10, 0.20)
SCREENING_MIN_IMPROVEMENT_PP: Final[float] = 0.02
SCREENING_MIN_WINS: Final[int] = 3
SCREENING_MAX_TARGET_REGRESSION_PP: Final[float] = 0.10
FULL_MIN_IMPROVEMENT_PP: Final[float] = 0.0
FULL_MIN_RECENT5_WINS: Final[int] = 3
FULL_MAX_WORST_REGRESSION_PP: Final[float] = 0.10
FULL_MAX_TARGET_REGRESSION_PP: Final[float] = 0.10


def _steps(horizons: Sequence[int]) -> tuple[int, ...]:
    """将步长或分钟形式统一为 1--8 个 15 分钟步。"""

    result: list[int] = []
    for value in horizons:
        number = int(value)
        if number > 8:
            if number % 15:
                raise ValueError(f"非法 horizon: {value}")
            number //= 15
        if number < 1 or number > 8:
            raise ValueError(f"非法 horizon: {value}")
        result.append(number)
    if len(set(result)) != len(result):
        raise ValueError("horizon 不得重复")
    return tuple(result)


def _finite_history(values: Iterable[float], *, window: int | None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if window is not None and window > 0:
        array = array[-window:]
    return array


class LocalLinearTrend:
    """独立的一维局部线性趋势模型，状态为 level/trend。"""

    def __init__(self, damping: float = 0.85, *, window: int = 96, min_history: int = 3) -> None:
        if not 0.0 < float(damping) <= 1.0:
            raise ValueError("damping 必须位于 (0, 1]")
        if window < 1 or min_history < 1:
            raise ValueError("window 和 min_history 必须为正数")
        self.damping = float(damping)
        self.window = int(window)
        self.min_history = int(min_history)
        self.level_: float | None = None
        self.trend_: float | None = None
        self.history_rows_: int = 0
        self.fallback_: bool = False

    def fit(self, values: Iterable[float]) -> "LocalLinearTrend":
        history = _finite_history(values, window=self.window)
        self.history_rows_ = int(len(history))
        if len(history) == 0:
            raise ValueError("LocalLinearTrend 没有有限观测")
        self.level_ = float(history[-1])
        self.fallback_ = len(history) < self.min_history
        if len(history) < 2:
            self.trend_ = 0.0
            return self
        x = np.arange(len(history), dtype=float)
        centered = x - x.mean()
        denominator = float(np.dot(centered, centered))
        slope = float(np.dot(centered, history - history.mean()) / denominator) if denominator else 0.0
        # 极端坏点不应让一个短窗口产生无界外推。
        self.trend_ = float(np.clip(slope, -1_000.0, 1_000.0))
        return self

    def forecast(self, horizons: Sequence[int] = STATE_SPACE_HORIZONS) -> np.ndarray:
        if self.level_ is None or self.trend_ is None:
            raise RuntimeError("LocalLinearTrend 尚未拟合")
        steps = _steps(horizons)
        output = []
        for step in steps:
            multiplier = sum(self.damping**index for index in range(step))
            output.append(self.level_ + self.trend_ * multiplier)
        return np.asarray(output, dtype=float)


class KalmanLocalLinearTrend:
    """两状态 Kalman 局部线性趋势模型（level + trend）。"""

    def __init__(
        self,
        damping: float = 0.85,
        *,
        window: int = 512,
        min_history: int = 3,
        process_floor: float = 1e-5,
        observation_floor: float = 1e-5,
    ) -> None:
        if not 0.0 < float(damping) <= 1.0:
            raise ValueError("damping 必须位于 (0, 1]")
        if window < 1 or min_history < 1:
            raise ValueError("window 和 min_history 必须为正数")
        self.damping = float(damping)
        self.window = int(window)
        self.min_history = int(min_history)
        self.process_floor = float(process_floor)
        self.observation_floor = float(observation_floor)
        self.state_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.history_rows_: int = 0
        self.fallback_: bool = False

    def fit(self, values: Iterable[float]) -> "KalmanLocalLinearTrend":
        history = _finite_history(values, window=self.window)
        self.history_rows_ = int(len(history))
        if len(history) == 0:
            raise ValueError("KalmanLocalLinearTrend 没有有限观测")
        self.fallback_ = len(history) < self.min_history
        if len(history) == 1:
            initial_trend = 0.0
        else:
            differences = np.diff(history)
            initial_trend = float(np.median(differences[-min(16, len(differences)) :]))
        differences = np.diff(history) if len(history) > 1 else np.array([0.0])
        diff_scale = float(np.nanmedian(np.abs(differences - np.nanmedian(differences))))
        if not np.isfinite(diff_scale):
            diff_scale = 0.0
        observation = max(self.observation_floor, diff_scale**2 + self.process_floor)
        level_noise = max(self.process_floor, diff_scale**2 * 0.25 + self.process_floor)
        trend_noise = max(self.process_floor, diff_scale**2 * 0.05 + self.process_floor)
        transition = np.array([[1.0, self.damping], [0.0, self.damping]], dtype=float)
        process = np.diag([level_noise, trend_noise])
        state = np.array([history[0], initial_trend], dtype=float)
        covariance = np.diag([observation, max(observation, trend_noise * 10.0)])
        for observation_value in history:
            predicted_state = transition @ state
            predicted_covariance = transition @ covariance @ transition.T + process
            innovation = float(observation_value - predicted_state[0])
            innovation_variance = float(predicted_covariance[0, 0] + observation)
            gain = predicted_covariance[:, 0] / max(innovation_variance, self.observation_floor)
            state = predicted_state + gain * innovation
            covariance = predicted_covariance - np.outer(gain, predicted_covariance[0, :])
            covariance = (covariance + covariance.T) * 0.5
            covariance[np.diag_indices(2)] = np.maximum(np.diag(covariance), self.process_floor)
        self.state_ = state
        self.covariance_ = covariance
        return self

    def forecast(self, horizons: Sequence[int] = STATE_SPACE_HORIZONS) -> np.ndarray:
        if self.state_ is None:
            raise RuntimeError("KalmanLocalLinearTrend 尚未拟合")
        steps = _steps(horizons)
        transition = np.array([[1.0, self.damping], [0.0, self.damping]], dtype=float)
        state = self.state_.copy()
        output: list[float] = []
        for step in range(1, max(steps) + 1):
            state = transition @ state
            if step in steps:
                output.append(float(state[0]))
        return np.asarray(output, dtype=float)


KalmanLocalTrend = KalmanLocalLinearTrend
LocalTrend = LocalLinearTrend


def forecast_state_space(
    history: Iterable[float],
    *,
    model: str,
    damping: float,
    horizons: Sequence[int] = STATE_SPACE_HORIZONS,
    local_window: int = 96,
    kalman_window: int = 512,
) -> np.ndarray:
    """拟合一个独立状态模型并返回固定未来步长预测。"""

    if model == "local_trend":
        estimator: LocalLinearTrend | KalmanLocalLinearTrend = LocalLinearTrend(
            damping, window=local_window
        )
    elif model == "kalman":
        estimator = KalmanLocalLinearTrend(damping, window=kalman_window)
    else:
        raise ValueError("model 只支持 local_trend 或 kalman")
    horizon_values = tuple(horizons)
    try:
        return estimator.fit(history).forecast(horizon_values)
    except ValueError:
        return np.full(len(horizon_values), np.nan, dtype=float)


def forecast_at_origin(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    target: str,
    *,
    model: str,
    damping: float,
    horizons: Sequence[int] = STATE_SPACE_HORIZONS,
    local_window: int = 96,
    kalman_window: int = 512,
) -> np.ndarray:
    """严格按 ``frame.loc[:origin]`` 拟合，禁止读取 origin 之后的生产观测。"""

    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError("state-space frame 必须使用唯一、递增 DatetimeIndex")
    if target not in frame.columns:
        raise ValueError(f"frame 缺少目标列: {target}")
    timestamp = pd.Timestamp(origin)
    history = frame.loc[:timestamp, target]
    if history.empty:
        raise ValueError(f"origin 之前没有 {target} 历史观测: {timestamp}")
    return forecast_state_space(
        history.to_numpy(dtype=float),
        model=model,
        damping=damping,
        horizons=horizons,
        local_window=local_window,
        kalman_window=kalman_window,
    )


def _normalise_parent_rows(rows: pd.DataFrame, parent_column: str) -> pd.DataFrame:
    if parent_column not in rows.columns and parent_column != "parent_pred" and "parent_pred" in rows.columns:
        # 测试/临时 OOF 常用通用别名；不改变正式 A61 列名记录。
        parent_column = "parent_pred"
    required = {
        "fold",
        "origin_time",
        "train_end",
        "target",
        "horizon",
        "actual",
        "current_value",
        "persistence_pred",
        parent_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"state-space OOF 缺少字段: {missing}")
    work = rows.copy()
    work["fold"] = work["fold"].astype(str)
    if work["fold"].str.lower().eq("blind").any():
        raise ValueError("state-space 只接受 development OOF，不得读取 blind")
    for column in ("origin_time", "train_end"):
        work[column] = pd.to_datetime(work[column], errors="coerce")
        if work[column].isna().any():
            raise ValueError(f"state-space OOF 含非法 {column}")
    work["horizon"] = pd.to_numeric(work["horizon"], errors="raise").astype(int)
    if work["horizon"].max() <= 8:
        work["horizon"] = work["horizon"] * 15
    if not work["horizon"].isin(STATE_SPACE_HORIZONS).all():
        raise ValueError("state-space 只接受 15--120 分钟八个步长")
    if not work["target"].isin(STATE_SPACE_TARGETS).all():
        raise ValueError("state-space 只接受两个生产目标")
    if work.duplicated(["fold", "origin_time", "target", "horizon"]).any():
        raise ValueError("state-space OOF 存在重复键")
    numeric_columns = ["actual", "current_value", "persistence_pred", parent_column]
    numeric = work.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("state-space OOF 的实际值或父预测含缺失/非有限数")
    work.loc[:, numeric_columns] = numeric
    counts = work.groupby(["fold", "origin_time"], observed=True).size()
    if not counts.eq(16).all():
        raise ValueError("每个 fold×origin 必须完整包含 2 个目标×8 个步长")
    for fold, group in work.groupby("fold", sort=False, observed=True):
        if group["train_end"].nunique() != 1:
            raise ValueError(f"fold {fold} 含多个 train_end")
    work = work.rename(columns={parent_column: "parent_pred"})
    return work.sort_values(["origin_time", "target", "horizon", "fold"], kind="stable").reset_index(drop=True)


def _fold_order(rows: pd.DataFrame) -> list[str]:
    order = rows.groupby("fold", sort=False, observed=True)["origin_time"].min().sort_values()
    return order.index.astype(str).tolist()


def _fold_metrics(rows: pd.DataFrame, prediction_columns: Sequence[str], parent_column: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for candidate in prediction_columns:
        for fold, part in rows.groupby("fold", sort=False, observed=True):
            score = competition_mape(part["actual"], part[candidate])
            parent = competition_mape(part["actual"], part[parent_column])
            records.append(
                {
                    "candidate": candidate.removesuffix("_pred"),
                    "fold": str(fold),
                    "mape": float(score),
                    "parent_mape": float(parent),
                    "improvement_pp": float((parent - score) * 100.0),
                    "win": bool(score < parent),
                }
            )
    return pd.DataFrame(records)


def _group_metrics(rows: pd.DataFrame, prediction_columns: Sequence[str], parent_column: str, group: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for candidate in prediction_columns:
        for value, part in rows.groupby(group, sort=True, observed=True):
            score = competition_mape(part["actual"], part[candidate])
            parent = competition_mape(part["actual"], part[parent_column])
            records.append(
                {
                    "candidate": candidate.removesuffix("_pred"),
                    group: value,
                    "mape": float(score),
                    "parent_mape": float(parent),
                    "improvement_pp": float((parent - score) * 100.0),
                }
            )
    return pd.DataFrame(records)


def candidate_report(rows: pd.DataFrame, candidate: str, parent_column: str = "parent_pred") -> dict[str, object]:
    """返回一个候选相对 A61-5% 父模型的固定诊断。"""

    if candidate not in rows or parent_column not in rows:
        raise ValueError("候选或父模型列不存在")
    candidate_mape = competition_mape(rows["actual"], rows[candidate])
    parent_mape = competition_mape(rows["actual"], rows[parent_column])
    fold_order = (
        rows.groupby("fold", observed=True)["origin_time"]
        .min()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    folds = []
    for fold in fold_order:
        part = rows.loc[rows["fold"].astype(str).eq(fold)]
        c = competition_mape(part["actual"], part[candidate])
        p = competition_mape(part["actual"], part[parent_column])
        folds.append({"fold": str(fold), "difference_pp": float((c - p) * 100.0), "win": bool(c < p)})
    targets = {}
    for target, part in rows.groupby("target", sort=True, observed=True):
        c = competition_mape(part["actual"], part[candidate])
        p = competition_mape(part["actual"], part[parent_column])
        targets[str(target)] = {
            "candidate_mape": float(c),
            "parent_mape": float(p),
            "improvement_pp": float((p - c) * 100.0),
            "regression_pp": float((c - p) * 100.0),
        }
    recent = folds[-5:]
    differences = [float(item["difference_pp"]) for item in folds]
    return {
        "candidate": candidate.removesuffix("_pred"),
        "pooled_mape": float(candidate_mape),
        "parent_pooled_mape": float(parent_mape),
        "improvement_pp": float((parent_mape - candidate_mape) * 100.0),
        "fold_wins": int(sum(bool(item["win"]) for item in folds)),
        "fold_count": int(len(folds)),
        "recent5_wins": int(sum(bool(item["win"]) for item in recent)),
        "worst_fold_regression_pp": float(max(differences) if differences else float("nan")),
        "target_metrics": targets,
        "folds": folds,
    }


def screening_gate(report: Mapping[str, object]) -> dict[str, object]:
    """执行前五 development 折的硬停止门槛。"""

    targets = report.get("target_metrics", {})
    target_regressions = [
        float(value.get("regression_pp", 0.0))
        for value in targets.values()
        if isinstance(value, Mapping)
    ]
    checks = {
        "pooled_improvement": float(report["improvement_pp"]) >= SCREENING_MIN_IMPROVEMENT_PP,
        "minimum_wins": int(report["fold_wins"]) >= SCREENING_MIN_WINS,
        "target_stability": max(target_regressions, default=0.0) <= SCREENING_MAX_TARGET_REGRESSION_PP,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "pooled_improvement_pp": SCREENING_MIN_IMPROVEMENT_PP,
            "wins": SCREENING_MIN_WINS,
            "max_target_regression_pp": SCREENING_MAX_TARGET_REGRESSION_PP,
        },
    }


def full_development_gate(report: Mapping[str, object], *, perturbation_passed: bool) -> dict[str, object]:
    """执行完整 development 的固定晋级门槛。"""

    targets = report.get("target_metrics", {})
    target_regressions = [
        float(value.get("regression_pp", 0.0))
        for value in targets.values()
        if isinstance(value, Mapping)
    ]
    checks = {
        "pooled_improvement": float(report["improvement_pp"]) > FULL_MIN_IMPROVEMENT_PP,
        "recent5_wins": int(report["recent5_wins"]) >= FULL_MIN_RECENT5_WINS,
        "worst_fold_regression": float(report["worst_fold_regression_pp"]) <= FULL_MAX_WORST_REGRESSION_PP,
        "target_stability": max(target_regressions, default=0.0) <= FULL_MAX_TARGET_REGRESSION_PP,
        "future_perturbation": bool(perturbation_passed),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "pooled_improvement_pp_gt": FULL_MIN_IMPROVEMENT_PP,
            "recent5_wins": FULL_MIN_RECENT5_WINS,
            "worst_fold_regression_pp_at_most": FULL_MAX_WORST_REGRESSION_PP,
            "max_target_regression_pp": FULL_MAX_TARGET_REGRESSION_PP,
        },
    }


def _predict_bundle(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    phi_by_target: Mapping[tuple[str, str], float],
    *,
    horizons: Sequence[int] = STATE_SPACE_HORIZONS,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for model in ("local_trend", "kalman"):
        values = []
        for target in STATE_SPACE_TARGETS:
            values.append(
                forecast_at_origin(
                    frame,
                    origin,
                    target,
                    model=model,
                    damping=float(phi_by_target[(model, target)]),
                    horizons=horizons,
                )
            )
        output[model] = np.concatenate(values)
    return output


def future_perturbation_audit(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    phi_by_fold: Mapping[str, Mapping[tuple[str, str], float]],
    max_origins: int | None = 50,
) -> dict[str, object]:
    """验证 extreme/shuffle/null/delete 后每个 origin 的 16 个预测不变。"""

    origins = pd.DatetimeIndex(sorted(pd.to_datetime(rows["origin_time"]).unique()))
    if max_origins is not None and len(origins) > max_origins:
        positions = np.linspace(0, len(origins) - 1, max_origins, dtype=int)
        origins = origins[np.unique(positions)]
    methods = ("extreme", "shuffle", "null", "delete_future")
    failures: list[dict[str, object]] = []
    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    for position, origin in enumerate(origins):
        matching = rows.loc[rows["origin_time"].eq(origin)]
        if matching.empty:
            continue
        fold = str(matching["fold"].iloc[0])
        baseline = _predict_bundle(frame, origin, phi_by_fold[fold])
        for method in methods:
            changed = frame.copy()
            future = changed.index > origin
            if method == "extreme":
                changed.loc[future, numeric_columns] = -999_999.0
            elif method == "shuffle":
                values = changed.loc[future, numeric_columns].to_numpy(copy=True)
                if len(values):
                    rng = np.random.default_rng(20250731 + position)
                    changed.loc[future, numeric_columns] = values[rng.permutation(len(values))]
            elif method == "null":
                changed.loc[future, numeric_columns] = np.nan
            else:
                changed = changed.loc[:origin].copy()
            observed = _predict_bundle(changed, origin, phi_by_fold[fold])
            for model in ("local_trend", "kalman"):
                if not np.allclose(baseline[model], observed[model], rtol=0.0, atol=1e-12, equal_nan=True):
                    failures.append({"origin": str(origin), "method": method, "model": model})
    return {
        "passed": not failures,
        "methods": list(methods),
        "origins_checked": int(len(origins)),
        "cases_checked": int(len(origins) * len(methods)),
        "prediction_cells_checked": int(len(origins) * len(methods) * 16 * 2),
        "failures": failures,
    }


@dataclass(frozen=True)
class StateSpaceResult:
    """state-space 运行器的可追溯产物。"""

    rows: pd.DataFrame
    fold_metrics: pd.DataFrame
    target_metrics: pd.DataFrame
    horizon_metrics: pd.DataFrame
    training_trace: pd.DataFrame
    report: dict[str, object]


def _run_folds(
    frame: pd.DataFrame,
    parent: pd.DataFrame,
    folds: Sequence[str],
    *,
    horizons: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[tuple[str, str], float]]]:
    """按时间顺序生成 OOF；阻尼选择只消费更早折的误差。"""

    active: dict[tuple[str, str], float] = {
        (model, target): 0.85
        for model in ("local_trend", "kalman")
        for target in STATE_SPACE_TARGETS
    }
    history_scores: dict[tuple[str, str, float], list[float]] = {
        (model, target, phi): []
        for model in ("local_trend", "kalman")
        for target in STATE_SPACE_TARGETS
        for phi in STATE_SPACE_DAMPING
    }
    output_parts: list[pd.DataFrame] = []
    traces: list[dict[str, object]] = []
    phi_by_fold: dict[str, dict[tuple[str, str], float]] = {}
    for fold in folds:
        held = parent.loc[parent["fold"].eq(fold)].copy()
        fold_phi = dict(active)
        phi_by_fold[fold] = fold_phi
        variant_tables: dict[tuple[str, str, float], pd.DataFrame] = {}
        for model in ("local_trend", "kalman"):
            for target in STATE_SPACE_TARGETS:
                target_origins = pd.DatetimeIndex(sorted(held.loc[held["target"].eq(target), "origin_time"].unique()))
                for phi in STATE_SPACE_DAMPING:
                    records: list[dict[str, object]] = []
                    fallback_count = 0
                    min_history = None
                    max_history = 0
                    for origin in target_origins:
                        history = frame.loc[:origin, target]
                        finite_count = int(np.isfinite(history.to_numpy(dtype=float)).sum())
                        min_history = finite_count if min_history is None else min(min_history, finite_count)
                        max_history = max(max_history, finite_count)
                        if finite_count < 3:
                            fallback_count += 1
                        prediction = forecast_at_origin(
                            frame,
                            origin,
                            target,
                            model=model,
                            damping=phi,
                            horizons=horizons,
                        )
                        for horizon, value in zip(horizons, prediction, strict=True):
                            records.append({"origin_time": origin, "target": target, "horizon": 15 * int(horizon), "value": float(value)})
                    table = pd.DataFrame.from_records(records)
                    variant_tables[(model, target, phi)] = table
                    if phi == fold_phi[(model, target)]:
                        # 训练回执逐 origin×horizon 保存，便于逐元素复核实际值与预测值。
                        actual_lookup = held.loc[held["target"].eq(target)].set_index(
                            ["origin_time", "horizon"]
                        )["actual"]
                        for item in records:
                            key = (item["origin_time"], item["horizon"])
                            traces.append(
                                {
                                    "fold": fold,
                                    "origin_time": item["origin_time"],
                                    "train_end": held["train_end"].iloc[0],
                                    "target": target,
                                    "horizon": int(item["horizon"]),
                                    "actual": float(actual_lookup.loc[key]),
                                    "prediction": float(item["value"]),
                                    "model": model,
                                    "selected_phi": float(phi),
                                    "selection_source": "prior_development_folds" if history_scores[(model, target, phi)] else "registered_default_0.85",
                                    "history_rows_min": int(min_history or 0),
                                    "history_rows_max": int(max_history),
                                    "history_after_train_end": 0,
                                    "labels_from_held_fold": False,
                                    "fallback_origins": int(fallback_count),
                                }
                            )
        fold_out = held.copy()
        for model in ("local_trend", "kalman"):
            for target in STATE_SPACE_TARGETS:
                table = variant_tables[(model, target, fold_phi[(model, target)])]
                key = pd.MultiIndex.from_frame(fold_out.loc[fold_out["target"].eq(target), ["origin_time", "target", "horizon"]])
                values = table.set_index(["origin_time", "target", "horizon"])["value"].reindex(key).to_numpy(float)
                mask = fold_out["target"].eq(target)
                fold_out.loc[mask, f"{model}_pred"] = values
                for phi in STATE_SPACE_DAMPING:
                    diagnostic = variant_tables[(model, target, phi)].set_index(["origin_time", "target", "horizon"])["value"].reindex(key).to_numpy(float)
                    fold_out.loc[mask, f"{model}_phi_{int(phi * 100):02d}_pred"] = diagnostic
        fold_out["parent_pred"] = fold_out["parent_pred"].astype(float)
        for model in ("local_trend", "kalman"):
            for weight in STATE_SPACE_BLEND_WEIGHTS:
                column = f"parent_{model}_blend_{int(weight * 100):02d}_pred"
                fold_out[column] = (1.0 - weight) * fold_out["parent_pred"] + weight * fold_out[f"{model}_pred"]
        output_parts.append(fold_out)
        for model in ("local_trend", "kalman"):
            for target in STATE_SPACE_TARGETS:
                target_part = fold_out.loc[fold_out["target"].eq(target)]
                for phi in STATE_SPACE_DAMPING:
                    column = f"{model}_phi_{int(phi * 100):02d}_pred"
                    history_scores[(model, target, phi)].append(
                        competition_mape(target_part["actual"], target_part[column])
                    )
        # 当前折标签只在此处用于更新下一折的参数选择。
        for key in active:
            model, target = key
            scores = {
                phi: float(np.mean(history_scores[(model, target, phi)]))
                for phi in STATE_SPACE_DAMPING
                if history_scores[(model, target, phi)]
            }
            active[key] = min(scores, key=lambda phi: (scores[phi], phi))
    result = pd.concat(output_parts, ignore_index=True)
    return result, pd.DataFrame(traces), phi_by_fold


def build_state_space_diversity(
    frame: pd.DataFrame,
    parent_rows: pd.DataFrame,
    *,
    parent_column: str = "a61_recursive_blend_05_pred",
    scope: str = "screening",
    horizons: Sequence[int] = STATE_SPACE_HORIZONS,
) -> StateSpaceResult:
    """构造 screening 或完整 development 的 state-space OOF。"""

    if scope not in {"screening", "development"}:
        raise ValueError("scope 只支持 screening 或 development")
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique:
        raise ValueError("state-space frame 必须使用唯一 DatetimeIndex")
    steps = _steps(horizons)
    parent = _normalise_parent_rows(parent_rows, parent_column)
    missing_targets = sorted(set(STATE_SPACE_TARGETS).difference(frame.columns))
    if missing_targets:
        raise ValueError(f"frame 缺少目标列: {missing_targets}")
    all_folds = _fold_order(parent)
    screening_folds = all_folds[:5]
    if len(screening_folds) < min(5, len(all_folds)):
        raise ValueError("development folds 为空")
    screen_rows, screen_trace, screen_phi = _run_folds(frame, parent, screening_folds, horizons=steps)
    candidate_columns = [
        "persistence_pred",
        "local_trend_pred",
        "kalman_pred",
        "parent_pred",
        *[f"parent_{model}_blend_{int(weight * 100):02d}_pred" for model in ("local_trend", "kalman") for weight in STATE_SPACE_BLEND_WEIGHTS],
    ]
    blend_columns = [
        column
        for column in candidate_columns
        if column.startswith("parent_") and column != "parent_pred"
    ]
    all_comparison_columns = [column for column in candidate_columns if column in screen_rows.columns]
    screen_reports = {
        column: candidate_report(screen_rows, column)
        for column in all_comparison_columns
        if column != "parent_pred"
    }
    screen_gates = {column: screening_gate(report) for column, report in screen_reports.items()}
    passing = [column for column in blend_columns if screen_gates[column]["passed"]]
    selected = min(passing, key=lambda column: (-screen_reports[column]["improvement_pp"], column)) if passing else None
    if scope == "screening" or selected is None:
        rows = screen_rows
        trace = screen_trace
        phi_by_fold = screen_phi
        full_reports: dict[str, object] = {}
        perturbation = {"passed": False, "skipped": True, "reason": "screening 未通过或尚未进入 full development"}
        status = "SCREENING_PASS" if selected else "STOP_SCREENING"
        full_gate = {"passed": False, "skipped": True}
    else:
        full_folds = all_folds
        rows, trace, phi_by_fold = _run_folds(frame, parent, full_folds, horizons=steps)
        full_reports = {
            column: candidate_report(rows, column)
            for column in all_comparison_columns
            if column != "parent_pred"
        }
        selected_full = selected
        perturbation = future_perturbation_audit(frame, rows, phi_by_fold=phi_by_fold)
        full_gate = full_development_gate(full_reports[selected_full], perturbation_passed=bool(perturbation["passed"]))
        status = "PROMOTE_CANDIDATE" if full_gate["passed"] else "STOP_DEVELOPMENT"
    prediction_columns = [column for column in candidate_columns if column in rows.columns]
    fold_metrics = _fold_metrics(rows, prediction_columns, "parent_pred")
    target_metrics = _group_metrics(rows, prediction_columns, "parent_pred", "target")
    horizon_metrics = _group_metrics(rows, prediction_columns, "parent_pred", "horizon")
    selected_model = None
    if selected:
        selected_model = selected.split("parent_", 1)[1].split("_blend", 1)[0]
        rows["state_space_pred"] = rows[f"{selected_model}_pred"]
        for weight in STATE_SPACE_BLEND_WEIGHTS:
            rows[f"parent_state_blend_{int(weight * 100):02d}_pred"] = rows[f"parent_{selected_model}_blend_{int(weight * 100):02d}_pred"]
    prediction_column = selected if selected is not None else "parent_pred"
    rows["prediction"] = rows[prediction_column].to_numpy(dtype=float)
    report = {
        "stage": "A62_state_space_diversity",
        "scope": scope,
        "status": status,
        "formal_candidate": bool(full_gate.get("passed", False)),
        "blind_used": False,
        "parent_column": parent_column,
        "selected_candidate": selected,
        "prediction_column": prediction_column,
        "comparison_candidates": [column.removesuffix("_pred") for column in all_comparison_columns],
        "selected_state_model": selected_model,
        "damping_values": list(STATE_SPACE_DAMPING),
        "blend_weights": list(STATE_SPACE_BLEND_WEIGHTS),
        "screening_folds": screening_folds,
        "screening_reports": screen_reports,
        "screening_gates": screen_gates,
        "full_reports": full_reports,
        "full_gate": full_gate,
        "future_perturbation": perturbation,
        "rows": int(len(rows)),
        "strict_oof_contract": {
            "development_only": True,
            "blind_rows_accepted": False,
            "history_rule": "frame.loc[:origin] for every target/model/origin",
            "damping_selection_rule": "only earlier development folds; first fold uses 0.85",
            "future_production_labels_used": False,
            "future_generator_truth_used": False,
            "capacity_projection": "not applied; raw state-space and fixed blends are reported",
        },
    }
    return StateSpaceResult(
        rows=rows.reset_index(drop=True),
        fold_metrics=fold_metrics,
        target_metrics=target_metrics,
        horizon_metrics=horizon_metrics,
        training_trace=trace.reset_index(drop=True),
        report=report,
    )
