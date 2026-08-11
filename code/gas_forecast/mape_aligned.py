"""X3 MAPE-aligned baseline models.

围绕竞赛 MAPE 实现的固定参数新基模型。每个 ``target x horizon`` 独立建模，
学习绝对增量 ``y[t+h] - y[t]``（absolute delta 目标不变），以
``1 / max(abs(y_future), epsilon)`` 作为固定样本权重把训练损失对齐到竞赛
MAPE。epsilon 只能由训练侧固定规则确定并写入收据，不能在训练后按 held、
blind 或未来评分行选择。

本模块只实现三条固定分支：

- ``lgb_l1``：LightGBM ``regression_l1``；
- ``lgb_huber``：LightGBM ``huber``（alpha 固定为 1.0）；
- ``cat_mae``：CatBoost ``MAE``。

每条分支一套固定参数，不做 Optuna / 网格搜索，不使用 early stopping，
不使用 held / blind / 未来评分行的标签、特征、权重或阈值。

严格复用冻结 A61 verification 的 19 个 development 折：训练特征
``origin <= train_end`` 且标签结束 ``origin + horizon <= train_end``，
held 折、blind 与未来评分行不得进入训练集。复用 causal features 白名单
（``long_horizon`` profile）与生产一致的容量投影。产物写入独立
experiments run 目录，记录训练行数、权重分布、耗时、覆盖与 OOF hash。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Final

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer

from gas_forecast.aggressive import (
    project_long_candidate,
    project_production_predictions,
)
from gas_forecast.research import compare_research_candidate
from gas_forecast.rich_residual import select_rich_feature_columns
from joblib import Parallel, delayed


X3_TARGETS: Final[tuple[str, ...]] = ("generator_1", "generator_all")
X3_HORIZONS: Final[tuple[int, ...]] = (15, 30, 45, 60, 75, 90, 105, 120)
X3_BRANCHES: Final[tuple[str, ...]] = ("lgb_l1", "lgb_huber", "cat_mae")
X3_STEP_MINUTES: Final[int] = 15
X3_MIN_TRAIN_ROWS: Final[int] = 128
X3_RANDOM_SEED: Final[int] = 20250731

# epsilon 固定训练侧规则：epsilon = max(floor, fraction * quantile(|y_future|))，
# 全部统计只取该 cell 的训练行（origin<=train_end 且标签结束<=train_end）。
X3_EPSILON_FLOOR: Final[float] = 1e-6
X3_EPSILON_QUANTILE: Final[float] = 0.05
X3_EPSILON_FRACTION: Final[float] = 0.05

# 预注册固定融合：A61 parent 80% + branch 20%。
X3_A61_BLEND_WEIGHT: Final[float] = 0.20
X3_A61_PARENT_WEIGHT: Final[float] = 1.0 - X3_A61_BLEND_WEIGHT

# 预注册保留门槛：pooled 改善 >= 0.05pp 才标记 RETAIN_MAPE_ALIGNED。
X3_RETAIN_IMPROVEMENT_PP: Final[float] = 0.05

# 三条分支的固定参数字典（仅 objective / loss 不同，全参数冻结）。
# 参数在真实运行前一次性冻结：LGB 120 棵 / CatBoost 100 轮，全程不使用
# early stopping，避免 held / blind / 未来评分行进入早停选择。
X3_LGB_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 120,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "n_jobs": 1,
    "verbosity": -1,
}
X3_CAT_PARAMS: Final[dict[str, object]] = {
    "loss_function": "MAE",
    "iterations": 100,
    "depth": 6,
    "learning_rate": 0.05,
    "thread_count": 2,
    "has_time": True,
    "verbose": False,
    "allow_writing_files": False,
}


@dataclass(frozen=True)
class MapeAlignedResult:
    """X3 完整 development OOF、训练轨迹、审计与报告。"""

    rows: pd.DataFrame
    training_trace: pd.DataFrame
    weight_trace: pd.DataFrame
    report: dict[str, object]


def _validate_parent_rows(rows: pd.DataFrame, *, parent_column: str) -> pd.DataFrame:
    """校验 A61 parent development OOF 契约，拒绝任何 blind 行。"""

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
        raise ValueError(f"X3 parent OOF 缺少字段: {missing}")
    work = rows.copy()
    work["fold"] = work["fold"].astype(str)
    if work["fold"].eq("blind").any():
        raise ValueError("X3 只接受 development OOF，输入不得含 blind 行")
    for column in ("origin_time", "train_end"):
        work[column] = pd.to_datetime(work[column], errors="coerce")
        if work[column].isna().any():
            raise ValueError(f"X3 输入含非法 {column}")
    keys = ["fold", "origin_time", "target", "horizon"]
    if work.duplicated(keys).any():
        raise ValueError("X3 输入存在重复 fold×origin×target×horizon")
    work["horizon"] = pd.to_numeric(work["horizon"], errors="raise").astype(int)
    if not work["horizon"].isin(X3_HORIZONS).all():
        raise ValueError("X3 输入只能包含 15--120 分钟的八个登记步长")
    if not work["target"].isin(X3_TARGETS).all():
        raise ValueError("X3 输入只能包含 generator_1 和 generator_all")
    numeric_columns = [
        "actual",
        "current_value",
        "persistence_pred",
        parent_column,
    ]
    numeric = work.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("X3 输入的真实值或 parent 预测含缺失/非有限数")
    work.loc[:, numeric.columns] = numeric
    counts = work.groupby(["fold", "origin_time"], observed=True).size()
    if not counts.eq(len(X3_TARGETS) * len(X3_HORIZONS)).all():
        raise ValueError("X3 每个 fold×origin 必须包含两个目标和八个步长")
    for fold, group in work.groupby("fold", sort=False, observed=True):
        if group["train_end"].nunique() != 1:
            raise ValueError(f"X3 fold {fold} 含多个 train_end")
    return work.sort_values(["origin_time", "target", "horizon", "fold"]).reset_index(
        drop=True
    )


def _fold_order(rows: pd.DataFrame) -> list[str]:
    """按 held 起点时间恢复 development 折顺序。"""

    order = (
        rows.groupby("fold", sort=False, observed=True)["origin_time"]
        .min()
        .sort_values()
    )
    return order.index.astype(str).tolist()


def _fold_train_end(rows: pd.DataFrame, fold: str) -> pd.Timestamp:
    """读取一个折唯一的严格训练边界。"""

    values = pd.DatetimeIndex(rows.loc[rows["fold"].eq(fold), "train_end"].unique())
    if len(values) != 1:
        raise ValueError(f"X3 fold {fold} 含多个 train_end")
    return pd.Timestamp(values[0])


def _feature_matrix(
    features: pd.DataFrame,
    origins: pd.DatetimeIndex,
    feature_columns: list[str],
) -> pd.DataFrame:
    """按预测起点取固定静态白名单，保留缺失给训练期插补器处理。"""

    matrix = features.reindex(origins).reindex(columns=feature_columns).copy()
    return matrix.replace([np.inf, -np.inf], np.nan)


def _cell_training_data(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    target: str,
    horizon: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, dict[str, object]]:
    """构造一个 cell 的严格训练区绝对增量标签。

    只保留 ``origin <= train_end`` 且标签结束 ``origin + horizon <= train_end``
    的 origin；``y_future = y[t+h]`` 同时用于增量标签和 MAPE 样本权重。任何
    越界或 held 折混入都会立即失败。
    """

    step = horizon // X3_STEP_MINUTES
    series = pd.to_numeric(frame[target], errors="coerce")
    y_future = series.shift(-step)
    last_origin = pd.Timestamp(train_end) - pd.Timedelta(minutes=horizon)
    origins = pd.DatetimeIndex(
        frame.index[(frame.index <= last_origin) & y_future.notna().to_numpy()]
    )
    origin_series = series.reindex(origins)
    future = y_future.reindex(origins)
    if (origins > last_origin).any():
        raise RuntimeError(f"X3 {target} t+{horizon} 训练 origin 越过标签成熟边界")
    if origins.max() + pd.Timedelta(minutes=horizon) > pd.Timestamp(train_end):
        raise RuntimeError(f"X3 {target} t+{horizon} 训练标签结束越过 train_end")
    valid = np.isfinite(origin_series.to_numpy(dtype=float)) & np.isfinite(
        future.to_numpy(dtype=float)
    )
    origins = pd.DatetimeIndex(origins.to_numpy()[valid])
    origin_values = origin_series.reindex(origins).to_numpy(dtype=float)
    future_values = future.reindex(origins).to_numpy(dtype=float)
    delta = future_values - origin_values
    trace = {
        "history_rows": int(len(origins)),
        "training_rows": int(len(origins)),
        "history_max_time": (
            pd.NaT if not len(origins) else pd.Timestamp(origins.max())
        ),
        "label_max_time": (
            pd.NaT
            if not len(origins)
            else pd.Timestamp(origins.max() + pd.Timedelta(minutes=horizon))
        ),
        "history_after_train_end": int((origins > pd.Timestamp(train_end)).sum()),
        "labels_from_held_fold": False,
    }
    return origins, delta, future_values, trace


def _epsilon_rule(future_values: np.ndarray) -> float:
    """固定训练侧 epsilon 规则，只消费当前 cell 的训练行。

    规则：``epsilon = max(X3_EPSILON_FLOOR, X3_EPSILON_FRACTION *
    quantile(X3_EPSILON_QUANTILE, |y_future|))``。该公式完全由模块常量确定，
    不读取 held / blind / 未来评分行。
    """

    magnitude = np.abs(np.asarray(future_values, dtype=float))
    scale = float(np.nanquantile(magnitude, X3_EPSILON_QUANTILE)) * X3_EPSILON_FRACTION
    if not np.isfinite(scale):
        scale = 0.0
    return max(X3_EPSILON_FLOOR, scale)


def _sample_weights(future_values: np.ndarray, epsilon: float) -> np.ndarray:
    """固定 MAPE 对齐权重 ``1 / max(|y_future|, epsilon)``。"""

    weights = 1.0 / np.maximum(np.abs(np.asarray(future_values, dtype=float)), epsilon)
    if not np.isfinite(weights).all():
        raise RuntimeError("X3 样本权重含非有限值")
    return weights


def _fit_branch(
    branch: str,
    training_features: pd.DataFrame,
    training_delta: np.ndarray,
    sample_weight: np.ndarray,
    held_features: pd.DataFrame,
    *,
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """以固定参数训练一个分支并预测 held grid，不使用 early stopping。"""

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_matrix = imputer.fit_transform(training_features)
    held_matrix = imputer.transform(held_features)
    if branch == "lgb_l1":
        params = dict(X3_LGB_PARAMS)
        params["objective"] = "regression_l1"
        params["random_state"] = X3_RANDOM_SEED + seed_offset
        model = LGBMRegressor(**params)
        model.fit(train_matrix, training_delta, sample_weight=sample_weight)
    elif branch == "lgb_huber":
        params = dict(X3_LGB_PARAMS)
        params["objective"] = "huber"
        params["alpha"] = 1.0
        params["random_state"] = X3_RANDOM_SEED + seed_offset
        model = LGBMRegressor(**params)
        model.fit(train_matrix, training_delta, sample_weight=sample_weight)
    elif branch == "cat_mae":
        params = dict(X3_CAT_PARAMS)
        params["random_seed"] = X3_RANDOM_SEED + seed_offset
        model = CatBoostRegressor(**params)
        model.fit(
            train_matrix,
            training_delta,
            sample_weight=sample_weight,
            verbose=False,
        )
    else:
        raise ValueError(f"X3 未知分支: {branch}")
    prediction = np.asarray(model.predict(held_matrix), dtype=float)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"X3 {branch} 产生非有限预测")
    return prediction, {"branch": branch, "imputed_inf_to_nan": True}


def _process_fold(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    work: pd.DataFrame,
    *,
    fold: str,
    feature_columns: list[str],
    parent_column: str,
    fold_position: int,
) -> tuple[
    str,
    list[tuple[int, str, float]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """在单个 development 折上训练 16 cell x 3 分支并返回 held 预测更新。

    joblib worker：每个折独立处理，返回按 ``work`` 行索引的 raw 预测更新、
    训练轨迹与权重分布记录。不读取本身 ``work``，只写入调用方合并。
    """

    train_end = _fold_train_end(work, fold)
    fold_started = time.perf_counter()
    updates: list[tuple[int, str, float]] = []
    training_records: list[dict[str, object]] = []
    weight_records: list[dict[str, object]] = []
    for target in X3_TARGETS:
        for horizon in X3_HORIZONS:
            held_mask = (
                work["fold"].eq(fold)
                & work["target"].eq(target)
                & work["horizon"].eq(horizon)
            )
            positions = work.index[held_mask].to_numpy()
            held = work.loc[held_mask]
            if held.empty:
                raise RuntimeError(f"X3 {fold} {target} t+{horizon} 缺少 held 行")
            held_origins = pd.DatetimeIndex(held["origin_time"])
            train_origins, delta, future_values, trace = _cell_training_data(
                frame,
                train_end=train_end,
                target=target,
                horizon=horizon,
            )
            if len(train_origins) < X3_MIN_TRAIN_ROWS:
                trace.update(
                    {
                        "fold": fold,
                        "target": target,
                        "horizon_minutes": horizon,
                        "status": "parent_fallback",
                        "held_rows": int(len(held)),
                        "elapsed_seconds": time.perf_counter() - fold_started,
                    }
                )
                training_records.append(trace)
                continue
            epsilon = _epsilon_rule(future_values)
            weights = _sample_weights(future_values, epsilon)
            training_features = _feature_matrix(features, train_origins, feature_columns)
            held_features = _feature_matrix(features, held_origins, feature_columns)
            usable = np.asarray(
                np.isfinite(delta)
                & training_features.notna().any(axis=1).to_numpy(dtype=bool),
                dtype=bool,
            )
            trained_rows = int(usable.sum())
            if trained_rows < X3_MIN_TRAIN_ROWS:
                trace.update(
                    {
                        "fold": fold,
                        "target": target,
                        "horizon_minutes": horizon,
                        "status": "parent_fallback_insufficient_feature_rows",
                        "held_rows": int(len(held)),
                        "elapsed_seconds": time.perf_counter() - fold_started,
                    }
                )
                training_records.append(trace)
                continue
            held_current = held["current_value"].to_numpy(dtype=float)
            for branch in X3_BRANCHES:
                raw_column = f"x3_{branch}_raw_pred"
                seed_offset = (
                    fold_position * 1000
                    + X3_TARGETS.index(target) * 100
                    + X3_HORIZONS.index(horizon)
                )
                branch_started = time.perf_counter()
                prediction, _ = _fit_branch(
                    branch,
                    training_features.loc[usable],
                    delta[usable],
                    weights[usable],
                    held_features,
                    seed_offset=seed_offset,
                )
                # 绝对增量预测重建绝对目标。
                absolute = held_current + prediction
                for position, value in zip(positions, absolute, strict=True):
                    updates.append((int(position), raw_column, float(value)))
                training_records.append(
                    {
                        "fold": fold,
                        "target": target,
                        "horizon_minutes": horizon,
                        "branch": branch,
                        "train_end": train_end,
                        "epsilon": epsilon,
                        **trace,
                        "trained_rows": trained_rows,
                        "weight_min": float(np.min(weights[usable])),
                        "weight_max": float(np.max(weights[usable])),
                        "weight_mean": float(np.mean(weights[usable])),
                        "weight_std": float(np.std(weights[usable])),
                        "held_rows": int(len(held)),
                        "status": "trained",
                        "elapsed_seconds": time.perf_counter() - branch_started,
                        "held_fold_labels_used": False,
                    }
                )
                weight_records.append(
                    {
                        "fold": fold,
                        "target": target,
                        "horizon_minutes": horizon,
                        "branch": branch,
                        "epsilon": epsilon,
                        "training_rows": trained_rows,
                        "weight_q05": float(np.quantile(weights[usable], 0.05)),
                        "weight_q50": float(np.quantile(weights[usable], 0.50)),
                        "weight_q95": float(np.quantile(weights[usable], 0.95)),
                        "weight_sum": float(np.sum(weights[usable])),
                    }
                )
    return fold, updates, training_records, weight_records


def _fold_held_mask(rows: pd.DataFrame, fold: str) -> pd.Series:
    return rows["fold"].eq(fold)


def _route_audit(
    rows: pd.DataFrame,
    *,
    parent_column: str,
    raw_column: str,
) -> dict[str, object]:
    """确认每个分支原始预测覆盖全部登记 cell（2 target x 8 horizon）。"""

    changed = ~np.isclose(
        rows[raw_column].to_numpy(dtype=float),
        rows[parent_column].to_numpy(dtype=float),
    )
    return {
        "parent_column": parent_column,
        "raw_column": raw_column,
        "changed_cells": int(changed.sum()),
        "total_cells": int(len(rows)),
        "coverage": float(changed.mean()),
    }


def _sha256_frame(rows: pd.DataFrame) -> str:
    """对 OOF 长表生成稳定的内容 hash。"""

    normalized = rows.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("datetime64[ns]").astype("int64")
    row_hash = pd.util.hash_pandas_object(normalized, index=True).to_numpy(np.uint64)
    return hashlib.sha256(row_hash.tobytes()).hexdigest()


def _add_blend_column(
    rows: pd.DataFrame,
    *,
    parent_column: str,
    branch_raw_column: str,
    output_column: str,
) -> pd.DataFrame:
    """添加预注册 A61 parent 80% + branch 20% 的固定融合列。"""

    blended = (
        X3_A61_PARENT_WEIGHT * rows[parent_column].to_numpy(dtype=float)
        + X3_A61_BLEND_WEIGHT * rows[branch_raw_column].to_numpy(dtype=float)
    )
    output = rows.copy()
    output[output_column + "_raw"] = blended
    output = project_long_candidate(
        output,
        output_column + "_raw",
        output_column=output_column,
    )
    return output


def _candidate_gate(comparison: dict[str, object]) -> dict[str, object]:
    """执行预注册 0.05pp 保留门槛（只针对与 A61 parent 的比价）。"""

    improvement_pp = -float(comparison["pooled_difference"]) * 100.0
    recent5_wins = int(
        sum(float(value) < 0.0 for value in comparison["recent_5_folds_difference"].values())
    )
    worst_fold_regression_pp = float(comparison["worst_fold_regression"]) * 100.0
    generator_1_regression_pp = float(comparison["generator_1_difference"]) * 100.0
    target_regressions = {
        str(target): float(ape["difference"]) * 100.0
        for target, ape in comparison["pairwise"]["by_target"].items()
    }
    retained = improvement_pp >= X3_RETAIN_IMPROVEMENT_PP
    return {
        "pooled_improvement_pp": improvement_pp,
        "recent5_wins": recent5_wins,
        "worst_fold_regression_pp": worst_fold_regression_pp,
        "generator_1_regression_pp": generator_1_regression_pp,
        "target_regression_pp": target_regressions,
        "retain_threshold_pp": X3_RETAIN_IMPROVEMENT_PP,
        "status": "RETAIN_MAPE_ALIGNED" if retained else "DO_NOT_RETAIN",
    }


def build_mape_aligned_oof(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    parent_rows: pd.DataFrame,
    *,
    parent_column: str,
    n_jobs: int = 1,
) -> MapeAlignedResult:
    """在 A61 固定 19 个 development 折上构造 X3 三条分支的 OOF。

    ``frame`` 是官方训练生产数据（DatetimeIndex），``features`` 是复用构建
    的 causal features 矩阵，``parent_rows`` 是冻结 A61 verification
    development OOF。三条分支对每个 target×horizon 独立训练，然后与 A61
    做固定 80%/20% 融合，全部通过生产一致的容量投影。
    """

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("X3 原始 frame 必须使用 DatetimeIndex")
    if features.index.equals(frame.index) is False:
        raise ValueError("X3 frame 与 features 的索引必须一致")
    work = _validate_parent_rows(parent_rows, parent_column=parent_column)
    if not pd.DatetimeIndex(work["origin_time"]).isin(frame.index).all():
        raise ValueError("X3 parent OOF origin 不在原始时间轴中")
    feature_columns = select_rich_feature_columns(features, "long_horizon")
    if not feature_columns:
        raise ValueError("X3 没有可用 long_horizon 因果特征")

    fold_order = _fold_order(work)
    for branch in X3_BRANCHES:
        raw_column = f"x3_{branch}_raw_pred"
        work[raw_column] = work[parent_column].to_numpy(dtype=float)
    training_records: list[dict[str, object]] = []
    weight_records: list[dict[str, object]] = []
    worker_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_fold)(
            frame,
            features,
            work,
            fold=fold,
            feature_columns=feature_columns,
            parent_column=parent_column,
            fold_position=position,
        )
        for position, fold in enumerate(fold_order)
    )
    for fold_name, updates, trace, weights in worker_results:
        for row_index, raw_column, value in updates:
            work.loc[row_index, raw_column] = value
        training_records.extend(trace)
        weight_records.extend(weights)

    candidate_rows = work.copy()
    reports: dict[str, dict[str, object]] = {}
    audits: dict[str, dict[str, object]] = {}
    for branch in X3_BRANCHES:
        raw_column = f"x3_{branch}_raw_pred"
        standalone_column = f"x3_{branch}_pred"
        audits[standalone_column] = _route_audit(
            candidate_rows,
            parent_column=parent_column,
            raw_column=raw_column,
        )
        candidate_rows = project_long_candidate(
            candidate_rows,
            raw_column,
            output_column=standalone_column,
        )
        reports[standalone_column] = compare_research_candidate(
            candidate_rows,
            standalone_column,
            parent_column,
            scope="development",
        )
        blend_column = f"x3_{branch}_blend_a61_pred"
        candidate_rows = _add_blend_column(
            candidate_rows,
            parent_column=parent_column,
            branch_raw_column=raw_column,
            output_column=blend_column,
        )
        reports[blend_column] = compare_research_candidate(
            candidate_rows,
            blend_column,
            parent_column,
            scope="development",
        )

    trace = pd.DataFrame(training_records).sort_values(
        ["fold", "target", "horizon_minutes", "branch"], kind="stable"
    )
    weight_trace = pd.DataFrame(weight_records).sort_values(
        ["fold", "target", "horizon_minutes", "branch"], kind="stable"
    )
    gates = {column: _candidate_gate(report) for column, report in reports.items()}
    # 预注册保留判定对全部已报告候选（三条 standalone 与三条 A61 80% + 20% 融合）
    # 统一执行：只要 pooled 改善 >= 0.05pp 即标记 RETAIN_MAPE_ALIGNED。
    retained = [
        column
        for column in reports
        if gates[column]["status"] == "RETAIN_MAPE_ALIGNED"
    ]
    oof_hash = _sha256_frame(candidate_rows)
    report: dict[str, object] = {
        "stage": "X3_mape_aligned_baselines",
        "scope": "development",
        "parent_column": parent_column,
        "targets": list(X3_TARGETS),
        "horizons": [int(h) for h in X3_HORIZONS],
        "branches": {
            "lgb_l1": {
                "objective": "regression_l1",
                "params": dict(X3_LGB_PARAMS, objective="regression_l1"),
                "early_stopping": False,
                "optuna": False,
            },
            "lgb_huber": {
                "objective": "huber",
                "alpha": 1.0,
                "params": dict(X3_LGB_PARAMS, objective="huber", alpha=1.0),
                "early_stopping": False,
                "optuna": False,
            },
            "cat_mae": {
                "loss_function": "MAE",
                "params": dict(X3_CAT_PARAMS),
                "early_stopping": False,
                "optuna": False,
            },
        },
        "fixed_blend": {
            "parent": parent_column,
            "parent_weight": X3_A61_PARENT_WEIGHT,
            "branch_weight": X3_A61_BLEND_WEIGHT,
        },
        "sample_weight_formula": "1 / max(abs(y_future), epsilon)",
        "epsilon_rule": {
            "formula": (
                "max(X3_EPSILON_FLOOR, X3_EPSILON_FRACTION * "
                "quantile(X3_EPSILON_QUANTILE, abs(y_future)))"
            ),
            "floor": X3_EPSILON_FLOOR,
            "quantile": X3_EPSILON_QUANTILE,
            "fraction": X3_EPSILON_FRACTION,
            "statistics_source": "cell training rows only (origin<=train_end, label end<=train_end)",
        },
        "rows": int(len(candidate_rows)),
        "folds": fold_order,
        "feature_profile": "long_horizon",
        "feature_columns": feature_columns,
        "feature_column_count": int(len(feature_columns)),
        "blind_labels_used": False,
        "held_validation_labels_used": False,
        "future_scoring_rows_used": False,
        "training_trace_summary": {
            "records": int(len(trace)),
            "trained_records": int(trace["status"].eq("trained").sum()),
            "fallback_records": int(~trace["status"].eq("trained").sum()),
            "history_after_train_end": int(
                trace["history_after_train_end"].astype(float).sum()
            ),
            "labels_from_held_fold": int(trace["labels_from_held_fold"].astype(int).sum()),
        },
        "models": reports,
        "candidate_gates": gates,
        "raw_route_audits": audits,
        "retained_candidates": retained,
        "status": (
            "RETAIN_MAPE_ALIGNED" if retained else "STOP_DO_NOT_RETAIN"
        ),
        "formal_candidate": False,
        "oof_hash_sha256": oof_hash,
        "strict_oof_contract": {
            "development_only": True,
            "blind_rows_accepted": False,
            "training_rule": "origin <= train_end AND origin + horizon <= train_end",
            "held_fold_labels_used": False,
            "early_stopping_used": False,
            "weight_threshold_selection_from_held": False,
            "capacity_projection": "production-identical projection (g1 [0,200], gall [0,440], gall in [g1, g1+240])",
        },
    }
    return MapeAlignedResult(
        rows=candidate_rows.reset_index(drop=True),
        training_trace=trace.reset_index(drop=True),
        weight_trace=weight_trace.reset_index(drop=True),
        report=report,
    )


def fit_x3_cell_production(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    target: str,
    horizon: int,
    seed_position: int,
    feature_columns: list[str] | None = None,
) -> tuple[CatBoostRegressor, dict[str, object]]:
    """在单个冻结 cutoff 上拟合一个 cat_mae cell 模型（fit-once / replay 共用）。

    ``seed_position`` 来自 seed_contract：replay = frozen fold position，
    production = PRODUCTION_SEED_SLOT。标签只取 ``origin <= cutoff`` 且
    ``origin + horizon <= cutoff``；不读取 held / blind / 未来评分行。
    """

    from gas_forecast.seed_contract import seed_offset

    if feature_columns is None:
        feature_columns = select_rich_feature_columns(features, "long_horizon")
    train_origins, delta, future_values, trace = _cell_training_data(
        frame, train_end=cutoff, target=target, horizon=horizon
    )
    if len(train_origins) < X3_MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"X3 production {target} t+{horizon} 训练行不足: {len(train_origins)}"
        )
    epsilon = _epsilon_rule(future_values)
    weights = _sample_weights(future_values, epsilon)
    training_features = _feature_matrix(features, train_origins, feature_columns)
    usable = np.asarray(
        np.isfinite(delta)
        & training_features.notna().any(axis=1).to_numpy(dtype=bool),
        dtype=bool,
    )
    trained_rows = int(usable.sum())
    if trained_rows < X3_MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"X3 production {target} t+{horizon} 可用特征行不足: {trained_rows}"
        )
    target_idx = X3_TARGETS.index(target)
    horizon_idx = X3_HORIZONS.index(horizon)
    offset = seed_offset(seed_position, target_idx, horizon_idx)
    params = dict(X3_CAT_PARAMS)
    params["random_seed"] = X3_RANDOM_SEED + offset
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_matrix = imputer.fit_transform(training_features.loc[usable])
    model = CatBoostRegressor(**params)
    model.fit(
        train_matrix,
        delta[usable],
        sample_weight=weights[usable],
        verbose=False,
    )
    receipt = {
        "seed_position": int(seed_position),
        "target": target,
        "horizon_minutes": int(horizon),
        "seed_offset": int(offset),
        "effective_seed": int(X3_RANDOM_SEED + offset),
        "trained_rows": int(trained_rows),
        "history_rows": int(len(train_origins)),
        "epsilon": float(epsilon),
    }
    return model, imputer, receipt


def predict_x3_cell_production(
    model: CatBoostRegressor,
    imputer: SimpleImputer,
    features: pd.DataFrame,
    *,
    origins: pd.DatetimeIndex,
    current_values: np.ndarray,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    """用冻结 X3 cell 模型预测绝对目标 = current + delta_pred。

    与 OOF ``_fit_branch`` 一致：held 特征先经同一训练侧 median imputer。
    """

    if feature_columns is None:
        feature_columns = select_rich_feature_columns(features, "long_horizon")
    held_features = _feature_matrix(features, origins, feature_columns)
    held_matrix = imputer.transform(held_features)
    delta_pred = np.asarray(model.predict(held_matrix), dtype=float)
    if not np.isfinite(delta_pred).all():
        raise RuntimeError("X3 production 产生非有限 delta 预测")
    return current_values + delta_pred


def build_x3_production_predictions(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    origins: pd.DatetimeIndex,
    seed_position: int,
    fold_label: str = "production",
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """fit-once 16 cell cat_mae 并对给定 origin 输出 x3_cat_mae_pred（含容量投影）。

    长表列：fold / origin_time / target / horizon / current_value /
    x3_cat_mae_raw_pred / x3_cat_mae_pred。**不含 actual**（评分期 actual 是
    未来真值；replay 由调用方从冻结 OOF 合并，production 由平台评分）。
    """

    if feature_columns is None:
        feature_columns = select_rich_feature_columns(features, "long_horizon")
    parts: list[pd.DataFrame] = []
    receipts: dict[str, object] = {}
    for target in X3_TARGETS:
        for horizon in X3_HORIZONS:
            model, imputer, receipt = fit_x3_cell_production(
                frame,
                features,
                cutoff=cutoff,
                target=target,
                horizon=horizon,
                seed_position=seed_position,
                feature_columns=feature_columns,
            )
            receipts[f"{target}_t+{horizon}"] = receipt
            current = (
                pd.to_numeric(frame[target], errors="coerce")
                .reindex(origins)
                .to_numpy(dtype=float)
            )
            pred = predict_x3_cell_production(
                model,
                imputer,
                features,
                origins=origins,
                current_values=current,
                feature_columns=feature_columns,
            )
            parts.append(
                pd.DataFrame(
                    {
                        "fold": fold_label,
                        "origin_time": origins,
                        "target": target,
                        "horizon": horizon,
                        "train_end": cutoff,
                        "current_value": current,
                        "x3_cat_mae_raw_pred": pred,
                    }
                )
            )
    long = pd.concat(parts, ignore_index=True)
    long = project_production_predictions(long, "x3_cat_mae_raw_pred", output_column="x3_cat_mae_pred")
    return long, receipts


__all__ = [
    "MapeAlignedResult",
    "X3_A61_BLEND_WEIGHT",
    "X3_A61_PARENT_WEIGHT",
    "X3_BRANCHES",
    "X3_CAT_PARAMS",
    "X3_EPSILON_FLOOR",
    "X3_EPSILON_FRACTION",
    "X3_EPSILON_QUANTILE",
    "X3_HORIZONS",
    "X3_LGB_PARAMS",
    "X3_MIN_TRAIN_ROWS",
    "X3_RANDOM_SEED",
    "X3_RETAIN_IMPROVEMENT_PP",
    "X3_TARGETS",
    "build_mape_aligned_oof",
    "build_x3_production_predictions",
    "fit_x3_cell_production",
    "predict_x3_cell_production",
]
