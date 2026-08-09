"""A57 长步长 CatBoost 多样性实验。

本模块只实现已预注册的四个 ``generator_1`` 长步长模型：absolute target
和相对 RichGas 的严格历史 OOF residual。它不提供 Optuna、阈值或连续融合
搜索入口，所有融合权重和训练参数均为模块常量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer

from gas_forecast.aggressive import project_long_candidate
from gas_forecast.research import compare_research_candidate
from gas_forecast.rich_residual import select_rich_feature_columns


A57_LONG_HORIZONS: Final[tuple[int, ...]] = (75, 90, 105, 120)
A57_MIN_TRAIN_ROWS: Final[int] = 128
A57_ITERATIONS: Final[int] = 600
A57_DEPTH: Final[int] = 6
A57_LEARNING_RATE: Final[float] = 0.03
A57_RANDOM_SEED: Final[int] = 20250731
A57_THREAD_COUNT: Final[int] = 1
A57_RICH_GAS_WEIGHTS: Final[tuple[float, ...]] = (0.05, 0.10, 0.15, 0.20)
A57_A51_WEIGHTS: Final[tuple[float, ...]] = (0.05, 0.10, 0.15)
A57_RETAIN_IMPROVEMENT_PP: Final[float] = 0.005


@dataclass(frozen=True)
class LongCatBoostDiversityResult:
    """A57 的完整 development OOF、训练轨迹、相关性审计与报告。"""

    rows: pd.DataFrame
    training_trace: pd.DataFrame
    residual_correlation: pd.DataFrame
    report: dict[str, object]


def _validate_development_rows(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    a51_column: str,
) -> pd.DataFrame:
    """校验 A57 的 development OOF 契约，并拒绝任何 blind 行。"""

    required = {
        "fold",
        "origin_time",
        "train_end",
        "target",
        "horizon",
        "actual",
        "current_value",
        "persistence_pred",
        baseline_column,
        a51_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"A57 输入 OOF 缺少字段: {missing}")
    work = rows.copy()
    work["fold"] = work["fold"].astype(str)
    if work["fold"].eq("blind").any():
        raise ValueError("A57 只接受 development OOF，输入不得含 blind 行")
    for column in ("origin_time", "train_end"):
        work[column] = pd.to_datetime(work[column], errors="coerce")
        if work[column].isna().any():
            raise ValueError(f"A57 输入含非法 {column}")
    keys = ["fold", "origin_time", "target", "horizon"]
    if work.duplicated(keys).any():
        raise ValueError("A57 输入存在重复 fold×origin×target×horizon")
    numeric_columns = [
        "actual",
        "current_value",
        "persistence_pred",
        baseline_column,
        a51_column,
    ]
    numeric = work.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("A57 输入的真实值或预测含缺失/非有限数")
    work.loc[:, numeric.columns] = numeric
    work["horizon"] = pd.to_numeric(work["horizon"], errors="raise").astype(int)
    return work.sort_values(["origin_time", "target", "horizon", "fold"]).reset_index(
        drop=True
    )


def _eligible_long_g1(rows: pd.DataFrame) -> pd.Series:
    """返回 A57 唯一允许替换的四个 generator_1 长步长单元。"""

    return rows["target"].eq("generator_1") & rows["horizon"].isin(A57_LONG_HORIZONS)


def _chronological_folds(rows: pd.DataFrame) -> list[str]:
    """以 held origin 的时间顺序恢复 development 外层折次序。"""

    order = (
        rows.groupby("fold", sort=False)
        .agg(first_origin=("origin_time", "min"), train_end=("train_end", "first"))
        .reset_index()
        .sort_values(["first_origin", "train_end", "fold"], kind="stable")
    )
    return order["fold"].astype(str).tolist()


def _fold_train_end(rows: pd.DataFrame, fold: str) -> pd.Timestamp:
    """读取一个 held fold 的唯一严格训练边界。"""

    values = pd.DatetimeIndex(rows.loc[rows["fold"].eq(fold), "train_end"].unique())
    if len(values) != 1:
        raise ValueError(f"A57 fold {fold} 含多个 train_end")
    return pd.Timestamp(values[0])


def _feature_matrix(
    features: pd.DataFrame,
    origins: pd.DatetimeIndex,
    feature_columns: list[str],
) -> pd.DataFrame:
    """按预测起点取固定静态白名单，保留缺失给训练期插补器处理。"""

    matrix = features.reindex(origins).reindex(columns=feature_columns).copy()
    return matrix.replace([np.inf, -np.inf], np.nan)


def _fit_fixed_catboost(
    training_features: pd.DataFrame,
    training_target: np.ndarray,
    held_features: pd.DataFrame,
    *,
    horizon: int,
) -> np.ndarray:
    """以 A57 固定参数训练一个 CatBoost，不做早停或参数选择。"""

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_matrix = imputer.fit_transform(training_features)
    held_matrix = imputer.transform(held_features)
    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=A57_ITERATIONS,
        depth=A57_DEPTH,
        learning_rate=A57_LEARNING_RATE,
        random_seed=A57_RANDOM_SEED + horizon,
        has_time=True,
        thread_count=A57_THREAD_COUNT,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_matrix, training_target, verbose=False)
    prediction = np.asarray(model.predict(held_matrix), dtype=float)
    if not np.isfinite(prediction).all():
        raise RuntimeError("A57 CatBoost 产生非有限预测")
    return prediction


def _usable_training_rows(
    features: pd.DataFrame,
    target: np.ndarray,
) -> np.ndarray:
    """仅排除无标签或整行特征完全不可用的训练样本。"""

    return np.isfinite(target) & features.notna().any(axis=1).to_numpy(dtype=bool)


def _absolute_training_data(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    horizon: int,
    first_held_origin: pd.Timestamp,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """构造绝对目标版本的纯训练区标签，并验证最长标签隔离。"""

    step = horizon // 15
    target = pd.to_numeric(frame["generator_1"].shift(-step), errors="coerce")
    origins = frame.index[(frame.index <= train_end) & target.notna()]
    if len(origins):
        label_end = origins.max() + pd.Timedelta(minutes=horizon)
        if label_end >= first_held_origin:
            raise ValueError(
                "A57 absolute CatBoost 训练标签越过 held fold；train_end 未满足 purge 契约"
            )
    return pd.DatetimeIndex(origins), target.reindex(origins).to_numpy(dtype=float)


def _residual_training_data(
    rows: pd.DataFrame,
    *,
    fold: str,
    train_end: pd.Timestamp,
    horizon: int,
    baseline_column: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, pd.DataFrame]:
    """只用此前 RichGas OOF 构造 A57b 的残差标签。"""

    history = rows.loc[
        rows["target"].eq("generator_1")
        & rows["horizon"].eq(horizon)
        & rows["fold"].ne(fold)
        & rows["origin_time"].le(train_end)
    ].sort_values("origin_time")
    if history["fold"].eq(fold).any():
        raise RuntimeError("A57b residual 历史混入 held fold")
    if (history["origin_time"] > train_end).any():
        raise RuntimeError("A57b residual 历史越过 outer-fold train_end")
    residual = (
        history["actual"].to_numpy(dtype=float) - history[baseline_column].to_numpy(dtype=float)
    )
    return pd.DatetimeIndex(history["origin_time"]), residual, history


def _route_audit(
    rows: pd.DataFrame,
    *,
    raw_column: str,
    parent_column: str,
) -> dict[str, object]:
    """确认候选的原始路由只改变四个目标单元。"""

    changed = ~np.isclose(
        rows[raw_column].to_numpy(dtype=float), rows[parent_column].to_numpy(dtype=float)
    )
    noneligible = int((changed & ~rows["a57_route_eligible"].to_numpy(dtype=bool)).sum())
    if noneligible:
        raise RuntimeError("A57 原始候选修改了 generator_1 长步长以外的单元")
    return {
        "parent_column": parent_column,
        "raw_changed_cells": int(changed.sum()),
        "noneligible_raw_changed_cells": noneligible,
        "selector_only_changes_g1_long": bool(noneligible == 0),
    }


def _add_fixed_blends(
    rows: pd.DataFrame,
    *,
    variant: str,
    cat_raw_column: str,
    parent_column: str,
    parent_label: str,
    weights: tuple[float, ...],
    reports: dict[str, dict[str, object]],
    audits: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """为一个固定父模型写入预注册小权重融合并统一投影。"""

    output = rows
    eligible = output["a57_route_eligible"]
    for weight in weights:
        label = f"{int(round(weight * 100)):02d}"
        raw_column = f"{variant}_{parent_label}_cat{label}_raw_pred"
        prediction_column = f"{variant}_{parent_label}_cat{label}_pred"
        output[raw_column] = output[parent_column].to_numpy(dtype=float)
        output.loc[eligible, raw_column] = (
            (1.0 - weight) * output.loc[eligible, parent_column].to_numpy(dtype=float)
            + weight * output.loc[eligible, cat_raw_column].to_numpy(dtype=float)
        )
        audits[prediction_column] = _route_audit(
            output,
            raw_column=raw_column,
            parent_column=parent_column,
        )
        output = project_long_candidate(output, raw_column, output_column=prediction_column)
        reports[prediction_column] = compare_research_candidate(
            output,
            prediction_column,
            parent_column,
            scope="development",
        )
    return output


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    """计算误差 Pearson 相关；常量向量没有可定义相关性。"""

    if len(left) < 2 or np.isclose(np.std(left), 0.0) or np.isclose(np.std(right), 0.0):
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _residual_correlation_table(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    a51_column: str,
    cat_columns: dict[str, str],
) -> pd.DataFrame:
    """按 pooled g1-long 与单步长审计三条模型误差的相关性。"""

    eligible = rows.loc[rows["a57_route_eligible"]].copy()
    records: list[dict[str, object]] = []
    for variant, cat_column in cat_columns.items():
        for horizon in (None, *A57_LONG_HORIZONS):
            part = eligible if horizon is None else eligible.loc[eligible["horizon"].eq(horizon)]
            actual = part["actual"].to_numpy(dtype=float)
            rich_error = actual - part[baseline_column].to_numpy(dtype=float)
            a51_error = actual - part[a51_column].to_numpy(dtype=float)
            cat_error = actual - part[cat_column].to_numpy(dtype=float)
            records.append(
                {
                    "variant": variant,
                    "scope": "pooled_g1_long" if horizon is None else f"t+{horizon}",
                    "horizon": horizon,
                    "rows": int(len(part)),
                    "corr_richgas_a51": _pearson(rich_error, a51_error),
                    "corr_richgas_cat": _pearson(rich_error, cat_error),
                    "corr_a51_cat": _pearson(a51_error, cat_error),
                }
            )
    return pd.DataFrame(records)


def _retained_blends(
    reports: dict[str, dict[str, object]],
    *,
    parent_label: str,
) -> list[dict[str, object]]:
    """仅按预注册 0.005pp 门槛标记可保留的固定融合，不选择权重。"""

    records: list[dict[str, object]] = []
    for candidate, comparison in reports.items():
        improvement_pp = -float(comparison["pooled_difference"]) * 100.0
        if improvement_pp >= A57_RETAIN_IMPROVEMENT_PP:
            records.append(
                {
                    "candidate": candidate,
                    "parent": parent_label,
                    "pooled_improvement_pp": improvement_pp,
                    "status": "RETAIN_DIVERSITY",
                }
            )
    return records


def build_a57_long_catboost_diversity(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    a51_column: str,
) -> LongCatBoostDiversityResult:
    """运行 A57a/b 的四步长 CatBoost 及固定多样性融合。

    ``features`` 必须是官方输入构造出的因果特征矩阵；函数内部固定复用 A51
    的静态 ``long_horizon`` 白名单。A57a 从各外层折的原始训练区读取绝对
    标签；A57b 只从已完成 RichGas OOF 读取 residual 标签。
    """

    if "generator_1" not in frame:
        raise ValueError("A57 原始训练数据缺少 generator_1")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("A57 原始训练数据必须使用 DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("A57 原始训练时间轴必须严格递增且唯一")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("A57 因果特征必须使用 DatetimeIndex")
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("A57 因果特征时间轴必须严格递增且唯一")

    work = _validate_development_rows(
        rows,
        baseline_column=baseline_column,
        a51_column=a51_column,
    )
    if not pd.DatetimeIndex(work["origin_time"]).isin(features.index).all():
        raise ValueError("A57 OOF origin 不完全包含在因果特征时间轴中")
    feature_columns = select_rich_feature_columns(features, "long_horizon")
    if not feature_columns:
        raise ValueError("A57 没有可用数值因果特征")
    work["a57_route_eligible"] = _eligible_long_g1(work)
    if not work["a57_route_eligible"].any():
        raise ValueError("A57 没有 generator_1 的 75/90/105/120 分钟单元")

    fold_order = _chronological_folds(work)
    training_records: list[dict[str, object]] = []
    variants = {
        "a57a_absolute": "absolute",
        "a57b_residual": "residual",
    }
    for variant, target_mode in variants.items():
        raw_column = f"{variant}_cat_raw_pred"
        work[raw_column] = work[baseline_column].to_numpy(dtype=float)
        for fold in fold_order:
            train_end = _fold_train_end(work, fold)
            for horizon in A57_LONG_HORIZONS:
                held_mask = (
                    work["fold"].eq(fold)
                    & work["a57_route_eligible"]
                    & work["horizon"].eq(horizon)
                )
                held = work.loc[held_mask]
                if held.empty:
                    raise RuntimeError(f"A57 {fold} t+{horizon} 缺少 held g1-long 行")
                held_origins = pd.DatetimeIndex(held["origin_time"])
                if target_mode == "absolute":
                    train_origins, training_target = _absolute_training_data(
                        frame,
                        train_end=train_end,
                        horizon=horizon,
                        first_held_origin=pd.Timestamp(held_origins.min()),
                    )
                    history = pd.DataFrame(
                        {
                            "origin_time": train_origins,
                            "fold": "raw_training",
                        }
                    )
                    label_source = "raw_absolute_target"
                else:
                    train_origins, training_target, history = _residual_training_data(
                        work,
                        fold=fold,
                        train_end=train_end,
                        horizon=horizon,
                        baseline_column=baseline_column,
                    )
                    label_source = "historical_richgas_oof_residual"
                training_features = _feature_matrix(features, train_origins, feature_columns)
                held_features = _feature_matrix(features, held_origins, feature_columns)
                usable = _usable_training_rows(training_features, training_target)
                trained_rows = int(usable.sum())
                status = "trained" if trained_rows >= A57_MIN_TRAIN_ROWS else "rich_gas_fallback"
                if status == "trained":
                    prediction = _fit_fixed_catboost(
                        training_features.loc[usable],
                        training_target[usable],
                        held_features,
                        horizon=horizon,
                    )
                    if len(prediction) != len(held):
                        raise RuntimeError("A57 CatBoost 预测长度与 held 行数不一致")
                    if target_mode == "residual":
                        prediction = prediction + held[baseline_column].to_numpy(dtype=float)
                    work.loc[held_mask, raw_column] = prediction
                training_records.append(
                    {
                        "variant": variant,
                        "target_mode": target_mode,
                        "fold": fold,
                        "horizon": horizon,
                        "train_end": train_end,
                        "label_source": label_source,
                        "history_rows": int(len(history)),
                        "training_rows": trained_rows,
                        "history_max_time": (
                            pd.NaT
                            if history.empty
                            else pd.Timestamp(pd.to_datetime(history["origin_time"]).max())
                        ),
                        "history_folds": ",".join(
                            history["fold"].drop_duplicates().astype(str).tolist()
                        ),
                        "held_rows": int(len(held)),
                        "status": status,
                        "held_fold_used_for_residual_history": False,
                        "labels_from_held_fold": False,
                    }
                )

    candidate_rows = work.copy()
    reports: dict[str, dict[str, object]] = {}
    raw_audits: dict[str, dict[str, object]] = {}
    standalone_columns: dict[str, str] = {}
    for variant in variants:
        raw_column = f"{variant}_cat_raw_pred"
        prediction_column = f"{variant}_cat_pred"
        raw_audits[prediction_column] = _route_audit(
            candidate_rows,
            raw_column=raw_column,
            parent_column=baseline_column,
        )
        candidate_rows = project_long_candidate(
            candidate_rows,
            raw_column,
            output_column=prediction_column,
        )
        reports[prediction_column] = compare_research_candidate(
            candidate_rows,
            prediction_column,
            baseline_column,
            scope="development",
        )
        standalone_columns[variant] = prediction_column
        candidate_rows = _add_fixed_blends(
            candidate_rows,
            variant=variant,
            cat_raw_column=raw_column,
            parent_column=baseline_column,
            parent_label="richgas",
            weights=A57_RICH_GAS_WEIGHTS,
            reports=reports,
            audits=raw_audits,
        )
        candidate_rows = _add_fixed_blends(
            candidate_rows,
            variant=variant,
            cat_raw_column=raw_column,
            parent_column=a51_column,
            parent_label="a51",
            weights=A57_A51_WEIGHTS,
            reports=reports,
            audits=raw_audits,
        )

    correlation = _residual_correlation_table(
        candidate_rows,
        baseline_column=baseline_column,
        a51_column=a51_column,
        cat_columns=standalone_columns,
    )
    trace = pd.DataFrame(training_records).sort_values(
        ["variant", "fold", "horizon"], kind="stable"
    )
    richgas_reports = {
        candidate: report
        for candidate, report in reports.items()
        if "_richgas_" in candidate
    }
    a51_reports = {
        candidate: report for candidate, report in reports.items() if "_a51_" in candidate
    }
    retained = [
        *_retained_blends(richgas_reports, parent_label="rich_gas"),
        *_retained_blends(a51_reports, parent_label="a51_splice"),
    ]
    report = {
        "stage": "A57_long_horizon_catboost_diversity",
        "scope": "development",
        "baseline_column": baseline_column,
        "a51_column": a51_column,
        "target_scope": "generator_1",
        "eligible_horizons": list(A57_LONG_HORIZONS),
        "rows": int(len(candidate_rows)),
        "eligible_rows": int(candidate_rows["a57_route_eligible"].sum()),
        "folds": fold_order,
        "feature_columns": feature_columns,
        "feature_column_count": int(len(feature_columns)),
        "feature_profile": (
            "A51 long_horizon 的静态因果白名单；不含 A51 专用的 Champion 预测合成特征"
        ),
        "fixed_catboost": {
            "loss_function": "MAE",
            "iterations": A57_ITERATIONS,
            "depth": A57_DEPTH,
            "learning_rate": A57_LEARNING_RATE,
            "random_seed": "20250731 + horizon_minutes",
            "thread_count": A57_THREAD_COUNT,
            "allow_writing_files": False,
            "early_stopping": False,
            "optuna": False,
            "minimum_training_rows": A57_MIN_TRAIN_ROWS,
        },
        "pre_registered_evaluation": {
            "a57a_target": "generator_1[t+h]",
            "a57b_target": "actual - same_horizon_RichGas_OOF_prediction",
            "rich_gas_cat_weights": list(A57_RICH_GAS_WEIGHTS),
            "a51_cat_weights": list(A57_A51_WEIGHTS),
            "retain_fixed_blend_if_pooled_improvement_pp_at_least": A57_RETAIN_IMPROVEMENT_PP,
            "standalone_is_not_a_stop_rule": True,
        },
        "training_trace_summary": {
            "records": int(len(trace)),
            "trained_records": int(trace["status"].eq("trained").sum()),
            "fallback_records": int(trace["status"].eq("rich_gas_fallback").sum()),
            "residual_history_after_train_end": int(
                (
                    (trace["target_mode"].eq("residual"))
                    & (
                        pd.to_datetime(trace["history_max_time"], errors="coerce")
                        > pd.to_datetime(trace["train_end"])
                    )
                ).sum()
            ),
        },
        "models": reports,
        "raw_route_audits": raw_audits,
        "residual_correlation_summary": correlation.to_dict(orient="records"),
        "retained_fixed_blends": retained,
        "formal_candidate": False,
        "blind_used": False,
        "strict_oof_contract": {
            "development_only": True,
            "blind_rows_accepted": False,
            "absolute_training": (
                "每个 held fold 只使用 origin_time <= train_end 的原始训练区绝对标签，"
                "并验证标签结束时间严格早于 held 起点"
            ),
            "residual_training": (
                "每个 held fold 只使用此前 origin_time <= train_end 的同一 horizon "
                "RichGas OOF residual；不使用 raw future target 或 held fold residual"
            ),
            "only_g1_long_raw_cells_changed": True,
            "capacity_projection": "每个 standalone 和固定融合都使用生产一致的容量投影",
        },
    }
    return LongCatBoostDiversityResult(
        rows=candidate_rows.reset_index(drop=True),
        training_trace=trace.reset_index(drop=True),
        residual_correlation=correlation.reset_index(drop=True),
        report=report,
    )
