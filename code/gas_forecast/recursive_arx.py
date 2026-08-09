"""A61 Recursive ARX 异构分支。

本模块只实现一套预注册的低自由度递归模型：两个目标各训练一个一步
Ridge ARX，预测时把自己的输出递归回填到下一步。模型不读取 OOF 标签，
外生量只使用训练时已登记的历史状态和官方已知未来价格。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from gas_forecast.aggressive import project_long_candidate
from gas_forecast.research import compare_research_candidate
from gas_forecast.second_tier import RecursiveARX, RecursiveARXSpec, fixed_recursive_blends


A61_HORIZONS: Final[tuple[int, ...]] = (15, 30, 45, 60, 75, 90, 105, 120)
A61_TARGETS: Final[tuple[str, ...]] = ("generator_1", "generator_all")
A61_BLEND_WEIGHTS: Final[tuple[float, ...]] = (0.05, 0.10, 0.20)
A61_ALPHA: Final[float] = 20.0
A61_MIN_TRAIN_ROWS: Final[int] = 200
A61_STEP_MINUTES: Final[int] = 15
A61_RETAIN_IMPROVEMENT_PP: Final[float] = 0.005
A61_MIN_RECENT5_WINS: Final[int] = 3
A61_MAX_WORST_REGRESSION_PP: Final[float] = 0.100
A61_COMMON_STATIC_COLUMNS: Final[tuple[str, ...]] = (
    "feat_generator_rest",
    "feat_generator_gas_total",
    "feat_rich_gas_available_for_generation",
    "feat_rich_gas_holder_buffer",
    "feat_rich_ramp_generator_1_rate",
    "feat_rich_ramp_generator_all_rate",
)


@dataclass(frozen=True)
class RecursiveARXResult:
    """A61 的逐行 OOF、训练轨迹和固定候选报告。"""

    rows: pd.DataFrame
    training_trace: pd.DataFrame
    report: dict[str, object]


def _validate_rows(rows: pd.DataFrame, *, baseline_column: str) -> pd.DataFrame:
    """验证 A61 只接收完整 development OOF，并保留严格训练边界。"""

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
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"A61 输入 OOF 缺少字段: {missing}")
    work = rows.copy()
    work["fold"] = work["fold"].astype(str)
    if work["fold"].eq("blind").any():
        raise ValueError("A61 只接受 development OOF，输入不得含 blind 行")
    for column in ("origin_time", "train_end"):
        work[column] = pd.to_datetime(work[column], errors="coerce")
        if work[column].isna().any():
            raise ValueError(f"A61 输入含非法 {column}")
    keys = ["fold", "origin_time", "target", "horizon"]
    if work.duplicated(keys).any():
        raise ValueError("A61 输入存在重复 fold×origin×target×horizon")
    work["horizon"] = pd.to_numeric(work["horizon"], errors="raise").astype(int)
    if not work["horizon"].isin(A61_HORIZONS).all():
        raise ValueError("A61 输入只能包含 15--120 分钟的八个登记步长")
    if not work["target"].isin(A61_TARGETS).all():
        raise ValueError("A61 输入只能包含 generator_1 和 generator_all")
    numeric_columns = [
        "actual",
        "current_value",
        "persistence_pred",
        baseline_column,
    ]
    numeric = work.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("A61 输入的真实值或父模型预测含缺失/非有限数")
    work.loc[:, numeric.columns] = numeric
    counts = work.groupby(["fold", "origin_time"], observed=True).size()
    if not counts.eq(len(A61_TARGETS) * len(A61_HORIZONS)).all():
        raise ValueError("A61 每个 fold×origin 必须包含两个目标和八个步长")
    for fold, group in work.groupby("fold", sort=False, observed=True):
        if group["train_end"].nunique() != 1:
            raise ValueError(f"A61 fold {fold} 含多个 train_end")
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
        raise ValueError(f"A61 fold {fold} 含多个 train_end")
    return pd.Timestamp(values[0])


def _static_columns(target: str) -> tuple[str, ...]:
    """返回冻结的低维系统状态列，避免按结果添加字段。"""

    if target not in A61_TARGETS:
        raise ValueError(f"A61 未登记目标: {target}")
    other_target = "generator_all" if target == "generator_1" else "generator_1"
    return (other_target, *A61_COMMON_STATIC_COLUMNS)


def _spec(target: str) -> RecursiveARXSpec:
    """构造一个目标对应的固定一步 ARX 规格。"""

    return RecursiveARXSpec(
        current_column=target,
        lag_columns=(f"feat_{target}_lag_1", f"feat_{target}_lag_2"),
        future_price_columns=tuple(
            f"feat_target_price_tplus_{horizon}" for horizon in A61_HORIZONS
        ),
        static_columns=_static_columns(target),
    )


def _feature_columns(spec: RecursiveARXSpec) -> list[str]:
    """按 ARX 规格展开需要从完整因果特征中读取的列。"""

    return [
        spec.current_column,
        *spec.lag_columns,
        *spec.future_price_columns,
        *spec.static_columns,
    ]


def _training_data(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    target: str,
    train_end: pd.Timestamp,
    first_held_origin: pd.Timestamp,
    spec: RecursiveARXSpec,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """构造一步训练集，并验证标签结束时间没有越过 held 起点。"""

    next_actual = pd.to_numeric(frame[target], errors="coerce").shift(-1)
    origins = frame.index[frame.index <= train_end]
    labels = next_actual.reindex(origins)
    valid = labels.notna() & np.isfinite(labels.to_numpy(dtype=float))
    training_origins = pd.DatetimeIndex(origins[valid.to_numpy()])
    training_features = features.reindex(training_origins).reindex(
        columns=_feature_columns(spec)
    )
    training_labels = labels.reindex(training_origins).astype(float)
    if len(training_origins):
        label_end = training_origins.max() + pd.Timedelta(minutes=A61_STEP_MINUTES)
        if label_end >= first_held_origin:
            raise ValueError(
                f"A61 {target} 训练标签越过 held fold: {label_end} >= {first_held_origin}"
            )
    trace = {
        "history_rows": int(len(training_origins)),
        "training_rows": int(len(training_origins)),
        "history_max_time": (
            pd.NaT if not len(training_origins) else pd.Timestamp(training_origins.max())
        ),
        "label_max_time": (
            pd.NaT
            if not len(training_origins)
            else pd.Timestamp(training_origins.max() + pd.Timedelta(minutes=A61_STEP_MINUTES))
        ),
        "label_end_max_time": (
            pd.NaT
            if not len(training_origins)
            else pd.Timestamp(training_origins.max() + pd.Timedelta(minutes=A61_STEP_MINUTES))
        ),
        "history_after_train_end": int((training_origins > train_end).sum()),
        "labels_from_held_fold": False,
    }
    return training_features, training_labels, trace


def _route_audit(
    rows: pd.DataFrame,
    *,
    raw_column: str,
    parent_column: str,
    eligible_column: str,
) -> dict[str, object]:
    """审计 ARX 原始预测是否只改写预注册 cell。"""

    changed = ~np.isclose(
        rows[raw_column].to_numpy(dtype=float), rows[parent_column].to_numpy(dtype=float)
    )
    eligible = rows[eligible_column].to_numpy(dtype=bool)
    noneligible = int((changed & ~eligible).sum())
    if noneligible:
        raise RuntimeError("A61 原始候选修改了登记目标和步长以外的 cell")
    return {
        "raw_changed_cells": int(changed.sum()),
        "noneligible_raw_changed_cells": noneligible,
        "only_registered_cells_changed": bool(noneligible == 0),
    }


def _correlation_rows(
    rows: pd.DataFrame,
    *,
    parent_column: str,
    recursive_column: str,
) -> list[dict[str, object]]:
    """记录父模型与递归分支的误差相关性，作为 diversity 审计而非调参依据。"""

    output: list[dict[str, object]] = []
    for target in (*A61_TARGETS, "pooled"):
        subset = rows if target == "pooled" else rows.loc[rows["target"].eq(target)]
        parent_error = subset["actual"].to_numpy(dtype=float) - subset[parent_column].to_numpy(
            dtype=float
        )
        recursive_error = subset["actual"].to_numpy(dtype=float) - subset[recursive_column].to_numpy(
            dtype=float
        )
        valid = np.isfinite(parent_error) & np.isfinite(recursive_error)
        parent_valid = parent_error[valid]
        recursive_valid = recursive_error[valid]
        if (
            len(parent_valid) < 2
            or np.isclose(np.std(parent_valid), 0.0)
            or np.isclose(np.std(recursive_valid), 0.0)
        ):
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(parent_valid, recursive_valid)[0, 1])
        output.append(
            {
                "target": target,
                "rows": int(valid.sum()),
                "parent_recursive_error_correlation": correlation,
            }
        )
    return output


def _candidate_status(comparison: dict[str, object]) -> dict[str, object]:
    """执行预注册的固定融合保留门槛。"""

    pooled_improvement_pp = -float(comparison["pooled_difference"]) * 100.0
    recent = comparison.get("recent_5_folds_difference")
    if not isinstance(recent, dict):
        raise TypeError("A61 比较报告缺少最近五折差值")
    recent5_wins = int(sum(float(value) < 0.0 for value in recent.values()))
    worst_fold_regression_pp = float(comparison["worst_fold_regression"]) * 100.0
    retained = bool(
        pooled_improvement_pp >= A61_RETAIN_IMPROVEMENT_PP
        and recent5_wins >= A61_MIN_RECENT5_WINS
        and worst_fold_regression_pp <= A61_MAX_WORST_REGRESSION_PP
    )
    return {
        "pooled_improvement_pp": pooled_improvement_pp,
        "recent5_wins": recent5_wins,
        "worst_fold_regression_pp": worst_fold_regression_pp,
        "status": "RETAIN_RECURSIVE_DIVERSITY" if retained else "DO_NOT_RETAIN",
        "acceptance": {
            "pooled_improvement_pp_at_least": A61_RETAIN_IMPROVEMENT_PP,
            "recent5_wins_at_least": A61_MIN_RECENT5_WINS,
            "worst_fold_regression_pp_at_most": A61_MAX_WORST_REGRESSION_PP,
        },
    }


def _trace_payload(trace: pd.DataFrame) -> list[dict[str, object]]:
    """把训练回执中的 pandas 时间转换为稳定的 JSON 文本。"""

    output = trace.copy()
    for column in ("train_end", "first_held_origin", "history_max_time", "label_max_time", "label_end_max_time"):
        output[column] = output[column].map(
            lambda value: None if pd.isna(value) else pd.Timestamp(value).isoformat()
        )
    return output.to_dict(orient="records")


def build_recursive_arx_diversity(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    parent_rows: pd.DataFrame,
    *,
    baseline_column: str,
) -> RecursiveARXResult:
    """在 development 外层折上构造 A61 的递归 ARX 与固定融合 OOF。"""

    work = _validate_rows(parent_rows, baseline_column=baseline_column)
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique:
        raise ValueError("A61 原始 frame 必须使用唯一 DatetimeIndex")
    if not features.index.equals(frame.index):
        raise ValueError("A61 原始 frame 与因果特征的时间索引必须完全一致")
    fold_order = _fold_order(work)
    work["a61_route_eligible"] = True
    recursive_raw_column = "a61_recursive_raw_pred"
    work[recursive_raw_column] = work[baseline_column].to_numpy(dtype=float)
    traces: list[dict[str, object]] = []

    for fold in fold_order:
        held = work.loc[work["fold"].eq(fold)]
        train_end = _fold_train_end(work, fold)
        first_held_origin = pd.Timestamp(held["origin_time"].min())
        for target in A61_TARGETS:
            spec = _spec(target)
            training_features, training_labels, trace = _training_data(
                frame,
                features,
                target=target,
                train_end=train_end,
                first_held_origin=first_held_origin,
                spec=spec,
            )
            model = RecursiveARX(spec, alpha=A61_ALPHA)
            status = "trained"
            if len(training_labels) < A61_MIN_TRAIN_ROWS:
                status = "parent_fallback"
            else:
                model.fit(training_features, training_labels)
                target_origins = pd.DatetimeIndex(
                    held.loc[held["target"].eq(target), "origin_time"].unique()
                ).sort_values()
                held_features = features.reindex(target_origins).reindex(
                    columns=_feature_columns(spec)
                )
                prediction = model.predict(held_features)
                if prediction.shape != (len(target_origins), len(A61_HORIZONS)):
                    raise RuntimeError("A61 Recursive ARX 输出形状不符合八步规格")
                for step, horizon in enumerate(A61_HORIZONS):
                    target_mask = (
                        work["fold"].eq(fold)
                        & work["target"].eq(target)
                        & work["horizon"].eq(horizon)
                    )
                    values = pd.Series(prediction[:, step], index=target_origins)
                    work.loc[target_mask, recursive_raw_column] = (
                        work.loc[target_mask, "origin_time"].map(values).to_numpy(dtype=float)
                    )
            traces.append(
                {
                    "fold": fold,
                    "target": target,
                    "train_end": train_end,
                    "first_held_origin": first_held_origin,
                    **trace,
                    "held_rows": int(held["target"].eq(target).sum()),
                    "status": status,
                    "alpha": A61_ALPHA,
                    "min_train_rows": A61_MIN_TRAIN_ROWS,
                }
            )

    recursive_column = "a61_recursive_pred"
    audits: dict[str, dict[str, object]] = {}
    reports: dict[str, dict[str, object]] = {}
    audits[recursive_column] = _route_audit(
        work,
        raw_column=recursive_raw_column,
        parent_column=baseline_column,
        eligible_column="a61_route_eligible",
    )
    work = project_long_candidate(work, recursive_raw_column, output_column=recursive_column)
    reports[recursive_column] = compare_research_candidate(
        work, recursive_column, baseline_column, scope="development"
    )
    for blend_name, blend_values in fixed_recursive_blends(
        work[baseline_column].to_numpy(dtype=float),
        work[recursive_raw_column].to_numpy(dtype=float),
        A61_BLEND_WEIGHTS,
    ).items():
        raw_column = f"a61_{blend_name}_raw_pred"
        prediction_column = f"a61_{blend_name}_pred"
        work[raw_column] = blend_values
        audits[prediction_column] = _route_audit(
            work,
            raw_column=raw_column,
            parent_column=baseline_column,
            eligible_column="a61_route_eligible",
        )
        work = project_long_candidate(work, raw_column, output_column=prediction_column)
        reports[prediction_column] = compare_research_candidate(
            work, prediction_column, baseline_column, scope="development"
        )

    status_by_candidate = {
        candidate: _candidate_status(comparison)
        for candidate, comparison in reports.items()
        if candidate != recursive_column
    }
    retained = [
        candidate
        for candidate, status in status_by_candidate.items()
        if status["status"] == "RETAIN_RECURSIVE_DIVERSITY"
    ]
    trace = pd.DataFrame(traces).sort_values(["fold", "target"], kind="stable")
    report = {
        "stage": "A61_recursive_arx_diversity",
        "scope": "development",
        "baseline_column": baseline_column,
        "target_scope": list(A61_TARGETS),
        "eligible_horizons": list(A61_HORIZONS),
        "rows": int(len(work)),
        "eligible_rows": int(work["a61_route_eligible"].sum()),
        "spec": {
            "model": "RecursiveARX",
            "alpha": A61_ALPHA,
            "minimum_training_rows": A61_MIN_TRAIN_ROWS,
            "current_and_lags": {
                target: [
                    _spec(target).current_column,
                    *_spec(target).lag_columns,
                ]
                for target in A61_TARGETS
            },
            "static_columns": {
                target: list(_static_columns(target)) for target in A61_TARGETS
            },
            "future_price_columns": [
                f"feat_target_price_tplus_{horizon}" for horizon in A61_HORIZONS
            ],
            "blend_weights": list(A61_BLEND_WEIGHTS),
            "price_route_search": False,
        },
        "training_trace_summary": {
            "records": int(len(trace)),
            "trained_records": int(trace["status"].eq("trained").sum()),
            "fallback_records": int(trace["status"].eq("parent_fallback").sum()),
            "history_after_train_end": int(trace["history_after_train_end"].sum()),
            "labels_from_held_fold": int(trace["labels_from_held_fold"].sum()),
        },
        "training_trace": _trace_payload(trace),
        "models": reports,
        "candidate_status": status_by_candidate,
        "raw_route_audits": audits,
        "error_correlation": _correlation_rows(
            work,
            parent_column=baseline_column,
            recursive_column=recursive_column,
        ),
        "retained_fixed_blends": retained,
        "status": "RETAIN_RECURSIVE_DIVERSITY" if retained else "STOP_RECURSIVE_DIVERSITY",
        "formal_candidate": False,
        "blind_used": False,
        "strict_oof_contract": {
            "development_only": True,
            "blind_rows_accepted": False,
            "training_history_rule": "raw origin_time <= outer_fold.train_end",
            "one_step_label_rule": "actual[t+15min]，其 label_end 必须早于 held 起点",
            "held_fold_labels_used": False,
            "future_exogenous_inputs": "registered official known-future price only",
            "recursive_feedback": "only own previous predictions; no held actual feedback",
            "capacity_projection": "all standalone and fixed blends use production projection",
        },
    }
    return RecursiveARXResult(
        rows=work.reset_index(drop=True),
        training_trace=trace.reset_index(drop=True),
        report=report,
    )
