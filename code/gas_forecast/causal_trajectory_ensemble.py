"""A69 严格因果轨迹集成。

本模块只消费已经完成的 development OOF。它不训练基础预测器，也不读取
评分集、blind 标签或任何平台参考值。所有集成权重来自其他 development
fold 的 ``actual/prediction``，并且仅在预注册的小型非负单纯形中选择。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from gas_forecast.scoring import competition_mape


KEY_COLUMNS: tuple[str, ...] = (
    "fold",
    "origin_time",
    "train_end",
    "target",
    "horizon",
    "actual",
)
IDENTITY_COLUMNS: tuple[str, ...] = KEY_COLUMNS[:-1]
TARGETS: tuple[str, ...] = ("generator_1", "generator_all")
HORIZONS: tuple[int, ...] = (15, 30, 45, 60, 75, 90, 105, 120)
PARENT_ROUTE = "a61_parent"
PRE_REGISTERED_AUXILIARY_WEIGHTS: tuple[float, ...] = (0.10, 0.20)
STATIC_MIN_IMPROVEMENT_PP = 0.02
STATIC_MIN_RECENT5_WINS = 3
STATIC_MAX_WORST_FOLD_REGRESSION_PP = 0.10
STATIC_MAX_TARGET_REGRESSION_PP = 0.10


@dataclass(frozen=True)
class RouteReceipt:
    """一条候选路线的准入收据。"""

    name: str
    source: str
    status: str
    accepted: bool
    reason: str
    rows: int
    blind_labels_used: bool
    future_perturbation_passed: bool | None


@dataclass(frozen=True)
class EnsembleResult:
    """A69 统一 OOF、指标表、训练收据和报告。"""

    rows: pd.DataFrame
    fold_metrics: pd.DataFrame
    target_metrics: pd.DataFrame
    horizon_metrics: pd.DataFrame
    residual_correlation: pd.DataFrame
    training_trace: pd.DataFrame
    route_receipts: pd.DataFrame
    report: dict[str, object]


def read_oof(path: str | Path) -> pd.DataFrame:
    """读取 CSV 或 Parquet OOF，不对原始文件做任何写入。"""

    value = Path(path)
    if value.suffix.lower() == ".parquet":
        return pd.read_parquet(value)
    return pd.read_csv(value)


def _as_timestamp(column: pd.Series, name: str) -> pd.Series:
    """解析审计时间列，并拒绝无法解析的键。"""

    parsed = pd.to_datetime(column, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{name} 包含无法解析的时间")
    return parsed


def _require_columns(rows: pd.DataFrame, required: Iterable[str], source: str) -> None:
    """统一报告 OOF 契约缺失字段。"""

    missing = sorted(set(required).difference(rows.columns))
    if missing:
        raise ValueError(f"{source} OOF 缺少字段: {missing}")


def canonicalize_oof(
    rows: pd.DataFrame,
    *,
    source: str,
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """将一条路线压缩为统一主键和稳定预测列。

    ``actual`` 是显式主键的一部分，故各路线在合并时必须同时拥有相同标签。
    这会拒绝日期边界、目标或 horizon 不一致的候选，避免隐式 inner join。
    """

    _require_columns(rows, (*KEY_COLUMNS, prediction_column), source)
    result = rows.loc[:, [*KEY_COLUMNS, prediction_column]].copy()
    result["fold"] = result["fold"].astype(str)
    result["origin_time"] = _as_timestamp(result["origin_time"], "origin_time")
    result["train_end"] = _as_timestamp(result["train_end"], "train_end")
    result["target"] = result["target"].astype(str)
    result["horizon"] = pd.to_numeric(result["horizon"], errors="coerce")
    if result["horizon"].isna().any() or not np.equal(result["horizon"] % 1, 0).all():
        raise ValueError(f"{source} OOF 的 horizon 必须是整数")
    result["horizon"] = result["horizon"].astype(int)
    result["actual"] = pd.to_numeric(result["actual"], errors="coerce")
    result[prediction_column] = pd.to_numeric(result[prediction_column], errors="coerce")
    if result.loc[:, ["actual", prediction_column]].isna().any().any():
        raise ValueError(f"{source} OOF 含空标签或预测")
    if not np.isfinite(result.loc[:, ["actual", prediction_column]].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} OOF 含 NaN/Inf")
    if result.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{source} OOF 主键含空值")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{source} OOF 主键不唯一")
    if result["fold"].str.lower().str.contains("blind").any():
        raise ValueError(f"{source} OOF 包含 blind 折")
    if not result["train_end"].lt(result["origin_time"]).all():
        raise ValueError(f"{source} OOF 存在 train_end 不早于 origin_time 的行")
    coverage = result.groupby("target", sort=True)["horizon"].agg(lambda values: tuple(sorted(values.unique())))
    if set(coverage.index) != set(TARGETS) or any(values != HORIZONS for values in coverage):
        raise ValueError(f"{source} OOF 未覆盖两个目标和八个 horizon")
    result = result.rename(columns={prediction_column: f"{source}__prediction"})
    return result.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)


def validate_oof_contract(
    rows: pd.DataFrame,
    *,
    source: str,
    prediction_column: str = "prediction",
) -> dict[str, object]:
    """验证并摘要一条 OOF 的严格主键、折边界和覆盖范围。"""

    canonical = canonicalize_oof(rows, source=source, prediction_column=prediction_column)
    fold_bounds = (
        canonical.groupby("fold", sort=True)
        .agg(
            rows=("fold", "size"),
            origin_start=("origin_time", "min"),
            origin_end=("origin_time", "max"),
            train_end=("train_end", "first"),
        )
        .reset_index()
    )
    return {
        "source": source,
        "rows": int(len(canonical)),
        "folds": fold_bounds.to_dict(orient="records"),
        "targets": sorted(canonical["target"].unique().tolist()),
        "horizons": sorted(canonical["horizon"].unique().tolist()),
        "unique_key": True,
        "blind_labels_used": False,
    }


def _same_keys(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, dict[str, int]]:
    """比较两条 OOF 的完整主键集合，并给出可写入报告的计数。"""

    left_keys = left.loc[:, list(KEY_COLUMNS)]
    right_keys = right.loc[:, list(KEY_COLUMNS)]
    compared = left_keys.merge(right_keys, on=list(KEY_COLUMNS), how="outer", indicator=True)
    counts = compared["_merge"].value_counts().to_dict()
    return (
        bool((compared["_merge"] == "both").all()),
        {
            "parent_only": int(counts.get("left_only", 0)),
            "route_only": int(counts.get("right_only", 0)),
            "shared": int(counts.get("both", 0)),
        },
    )


def collect_matching_oofs(
    parent_rows: pd.DataFrame,
    candidate_rows: Mapping[str, pd.DataFrame],
    *,
    parent_prediction_column: str = "prediction",
    candidate_prediction_columns: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """收集与父模型主键完全一致的已准入 development OOF。

    错配路线不会静默丢行；调用者可根据 ``route_checks`` 记录拒绝理由，
    然后只把通过检查的路线再次传入本函数。
    """

    candidate_prediction_columns = candidate_prediction_columns or {}
    parent = canonicalize_oof(
        parent_rows,
        source=PARENT_ROUTE,
        prediction_column=parent_prediction_column,
    )
    merged = parent.copy()
    checks: dict[str, dict[str, object]] = {}
    for name in sorted(candidate_rows):
        if name == PARENT_ROUTE:
            raise ValueError("候选路线不能覆盖 A61 父模型名称")
        candidate = canonicalize_oof(
            candidate_rows[name],
            source=name,
            prediction_column=candidate_prediction_columns.get(name, "prediction"),
        )
        matching, counts = _same_keys(parent, candidate)
        checks[name] = {"matching_keys": matching, **counts}
        if not matching:
            raise ValueError(
                f"{name} 与 A61 的 fold/origin/train_end/target/horizon/actual 不完全一致: {counts}"
            )
        merged = merged.merge(candidate, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    return merged.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True), checks


def pre_registered_weight_candidates(route_names: Sequence[str]) -> list[tuple[str, dict[str, float]]]:
    """返回小型、固定、非负且权重和为一的静态候选集合。

    候选只包括父模型、父模型加一条路线的 10%/20%，以及父模型加两条路线
    各 10%。这避免在 development 结束后扩张组合空间。
    """

    routes = tuple(sorted(set(route_names)))
    if PARENT_ROUTE in routes:
        raise ValueError("route_names 不应包含父模型")
    candidates: list[tuple[str, dict[str, float]]] = [("parent_only", {PARENT_ROUTE: 1.0})]
    for route in routes:
        for weight in PRE_REGISTERED_AUXILIARY_WEIGHTS:
            weights = {PARENT_ROUTE: 1.0 - weight, route: weight}
            candidates.append((f"parent_{int((1.0 - weight) * 100):02d}_{route}_{int(weight * 100):02d}", weights))
    for first, second in combinations(routes, 2):
        candidates.append(
            (
                f"parent_80_{first}_10_{second}_10",
                {PARENT_ROUTE: 0.80, first: 0.10, second: 0.10},
            )
        )
    for _, weights in candidates:
        values = np.asarray(list(weights.values()), dtype=float)
        if (values < 0.0).any() or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise AssertionError("预注册权重不是非负单纯形")
    return candidates


def _capacity_project(rows: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    """在同一 origin/horizon 内施加 ``generator_all >= generator_1`` 约束。"""

    result = rows.copy()
    pairing_keys = ["fold", "origin_time", "train_end", "horizon"]
    g1 = result.loc[result["target"].eq("generator_1"), pairing_keys + [prediction_column]].rename(
        columns={prediction_column: "_generator_1_prediction"}
    )
    all_rows = result.loc[
        result["target"].eq("generator_all"), pairing_keys + [prediction_column]
    ].copy()
    paired = all_rows.merge(g1, on=pairing_keys, how="left", validate="one_to_one")
    if paired["_generator_1_prediction"].isna().any():
        raise ValueError("generator_all 缺少同 origin/horizon 的 generator_1 预测")
    all_indices = result.index[result["target"].eq("generator_all")]
    result.loc[all_indices, prediction_column] = np.maximum(
        paired[prediction_column].to_numpy(dtype=float),
        paired["_generator_1_prediction"].to_numpy(dtype=float),
    )
    return result


def _prediction_for_weights(rows: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    """按权重生成预测，并立即应用确定性的容量投影。"""

    result = rows.loc[:, list(KEY_COLUMNS)].copy()
    prediction = np.zeros(len(rows), dtype=float)
    for route, weight in weights.items():
        column = f"{route}__prediction"
        if column not in rows:
            raise ValueError(f"权重引用未收集路线: {route}")
        prediction += float(weight) * rows[column].to_numpy(dtype=float)
    result["prediction"] = prediction
    return _capacity_project(result, "prediction")


def cross_fitted_static_blend(
    rows: pd.DataFrame,
    *,
    route_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对每个 held fold 仅由其他 fold 的标签选择一组固定权重。"""

    candidates = pre_registered_weight_candidates(route_names)
    folds = sorted(rows["fold"].astype(str).unique().tolist())
    outputs: list[pd.DataFrame] = []
    trace_rows: list[dict[str, object]] = []
    for held_fold in folds:
        train_rows = rows.loc[rows["fold"].astype(str).ne(held_fold)]
        held_rows = rows.loc[rows["fold"].astype(str).eq(held_fold)]
        if train_rows.empty or held_rows.empty:
            raise ValueError(f"fold {held_fold} 无法做 cross-fitted 融合")
        scores: list[tuple[float, str, dict[str, float]]] = []
        for candidate_name, weights in candidates:
            predicted_train = _prediction_for_weights(train_rows, weights)
            scores.append(
                (
                    competition_mape(predicted_train["actual"], predicted_train["prediction"]),
                    candidate_name,
                    weights,
                )
            )
        selected_score, selected_name, selected_weights = min(scores, key=lambda item: (item[0], item[1]))
        held_prediction = _prediction_for_weights(held_rows, selected_weights)
        held_prediction["selected_candidate"] = selected_name
        held_prediction["selection_rows"] = int(len(train_rows))
        for route in (PARENT_ROUTE, *sorted(route_names)):
            held_prediction[f"weight_{route}"] = float(selected_weights.get(route, 0.0))
        outputs.append(held_prediction)
        trace_rows.append(
            {
                "held_fold": held_fold,
                "selection_rows": int(len(train_rows)),
                "selection_fold_count": int(train_rows["fold"].nunique()),
                "selected_candidate": selected_name,
                "selected_training_mape": float(selected_score),
                "candidate_scores": json.dumps(
                    {name: float(score) for score, name, _ in scores},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "weights": json.dumps(selected_weights, ensure_ascii=False, sort_keys=True),
                "held_fold_labels_used": False,
            }
        )
    result = pd.concat(outputs, ignore_index=True).sort_values(list(IDENTITY_COLUMNS), kind="stable")
    return result.reset_index(drop=True), pd.DataFrame(trace_rows)


def _add_ape(rows: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    """复制并加上统一 epsilon 口径的绝对百分比误差。"""

    result = rows.copy()
    result["ape"] = np.abs(result["actual"] - result[prediction_column]) / np.maximum(
        np.abs(result["actual"]), 1e-6
    )
    return result


def static_fusion_gate(
    rows: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    parent_prediction_column: str = f"{PARENT_ROUTE}__prediction",
) -> dict[str, object]:
    """按预注册 pooled、recent5、最差折和目标回归门槛评估静态融合。"""

    _require_columns(rows, (prediction_column, parent_prediction_column), "A69")
    candidate = _add_ape(rows, prediction_column)
    parent = _add_ape(rows, parent_prediction_column)
    by_fold_candidate = candidate.groupby("fold", sort=True)["ape"].mean()
    by_fold_parent = parent.groupby("fold", sort=True)["ape"].mean()
    fold_delta_pp = (by_fold_parent - by_fold_candidate) * 100.0
    by_target_candidate = candidate.groupby("target", sort=True)["ape"].mean()
    by_target_parent = parent.groupby("target", sort=True)["ape"].mean()
    target_delta_pp = (by_target_parent - by_target_candidate) * 100.0
    recent = fold_delta_pp.sort_index().tail(5)
    pooled_candidate = float(candidate["ape"].mean())
    pooled_parent = float(parent["ape"].mean())
    improvement_pp = float((pooled_parent - pooled_candidate) * 100.0)
    worst_regression_pp = float((-fold_delta_pp).max())
    max_target_regression_pp = float((-target_delta_pp).max())
    checks = {
        "pooled_improvement": improvement_pp >= STATIC_MIN_IMPROVEMENT_PP,
        "recent5_wins": int((recent > 0.0).sum()) >= STATIC_MIN_RECENT5_WINS,
        "worst_fold_regression": worst_regression_pp <= STATIC_MAX_WORST_FOLD_REGRESSION_PP,
        "target_regression": max_target_regression_pp <= STATIC_MAX_TARGET_REGRESSION_PP,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "pooled_candidate_mape": pooled_candidate,
        "pooled_parent_mape": pooled_parent,
        "improvement_pp": improvement_pp,
        "fold_wins": int((fold_delta_pp > 0.0).sum()),
        "recent5_wins": int((recent > 0.0).sum()),
        "worst_fold_regression_pp": worst_regression_pp,
        "max_target_regression_pp": max_target_regression_pp,
        "by_fold_improvement_pp": {str(key): float(value) for key, value in fold_delta_pp.items()},
        "by_target_improvement_pp": {str(key): float(value) for key, value in target_delta_pp.items()},
    }


def _metric_tables(rows: pd.DataFrame, prediction_columns: Mapping[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成 pooled 折、目标及 target×horizon 指标表。"""

    fold_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    horizon_parts: list[pd.DataFrame] = []
    for model, column in prediction_columns.items():
        scored = _add_ape(rows, column)
        fold = scored.groupby("fold", sort=True)["ape"].agg([("mape", "mean"), ("rows", "size")]).reset_index()
        fold.insert(0, "model", model)
        target = scored.groupby("target", sort=True)["ape"].agg([("mape", "mean"), ("rows", "size")]).reset_index()
        target.insert(0, "model", model)
        horizon = (
            scored.groupby(["target", "horizon"], sort=True)["ape"]
            .agg([("mape", "mean"), ("rows", "size")])
            .reset_index()
        )
        horizon.insert(0, "model", model)
        fold_parts.append(fold)
        target_parts.append(target)
        horizon_parts.append(horizon)
    return (
        pd.concat(fold_parts, ignore_index=True),
        pd.concat(target_parts, ignore_index=True),
        pd.concat(horizon_parts, ignore_index=True),
    )


def _first_round_summary(
    rows: pd.DataFrame,
    prediction_columns: Mapping[str, str],
) -> dict[str, object]:
    """将第一轮单模型 pooled、target×horizon、折和稳健性摘要写入报告。"""

    parent_column = prediction_columns[PARENT_ROUTE]
    parent_scores = _add_ape(rows, parent_column).groupby("fold", sort=True)["ape"].mean()
    models: dict[str, dict[str, object]] = {}
    for model, column in prediction_columns.items():
        scored = _add_ape(rows, column)
        by_fold = scored.groupby("fold", sort=True)["ape"].mean()
        by_target_horizon = scored.groupby(["target", "horizon"], sort=True)["ape"].mean()
        recent = by_fold.sort_index().tail(5)
        models[model] = {
            "pooled_mape": float(scored["ape"].mean()),
            "by_fold_mape": {str(key): float(value) for key, value in by_fold.items()},
            "by_target_horizon_mape": {
                target: {
                    str(horizon): float(by_target_horizon.loc[(target, horizon)])
                    for horizon in HORIZONS
                }
                for target in TARGETS
            },
            "worst_fold": {
                "fold": str(by_fold.idxmax()),
                "mape": float(by_fold.max()),
            },
            "recent5_wins_vs_parent": (
                None
                if model == PARENT_ROUTE
                else int((recent < parent_scores.reindex(recent.index)).sum())
            ),
        }
    return {"models": models, "score_definition": "pooled cell MAPE; epsilon=1e-6"}


def residual_correlation(rows: pd.DataFrame, prediction_columns: Mapping[str, str]) -> pd.DataFrame:
    """以 ``actual - prediction`` 计算各模型 OOF 残差相关性矩阵。"""

    residuals = pd.DataFrame(
        {
            name: rows["actual"].to_numpy(dtype=float) - rows[column].to_numpy(dtype=float)
            for name, column in prediction_columns.items()
        }
    )
    matrix = residuals.corr(method="pearson")
    matrix.index.name = "model"
    return matrix.reset_index()


def validate_prediction_contract(rows: pd.DataFrame, prediction_columns: Iterable[str]) -> None:
    """验证最终预测无 NaN/Inf、完整覆盖并满足两目标的物理次序。"""

    _require_columns(rows, KEY_COLUMNS, "A69")
    if rows.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("A69 OOF 主键不唯一")
    coverage = rows.groupby("target", sort=True)["horizon"].agg(lambda values: tuple(sorted(values.unique())))
    if set(coverage.index) != set(TARGETS) or any(values != HORIZONS for values in coverage):
        raise ValueError("A69 OOF 没有完整覆盖两目标×八 horizon")
    for column in prediction_columns:
        _require_columns(rows, (column,), "A69")
        values = rows[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"A69 {column} 含 NaN/Inf")
        projected = _capacity_project(rows.loc[:, [*KEY_COLUMNS, column]], column)
        if not np.array_equal(projected[column].to_numpy(dtype=float), values):
            raise ValueError(f"A69 {column} 不满足 generator_all >= generator_1")


def _route_receipt_frame(receipts: Sequence[RouteReceipt]) -> pd.DataFrame:
    """将不可变收据转为易审计的 CSV 表。"""

    return pd.DataFrame(
        [
            {
                "route": item.name,
                "source": item.source,
                "status": item.status,
                "accepted": item.accepted,
                "reason": item.reason,
                "rows": item.rows,
                "blind_labels_used": item.blind_labels_used,
                "future_perturbation_passed": item.future_perturbation_passed,
            }
            for item in receipts
        ]
    )


def build_causal_trajectory_ensemble(
    parent_rows: pd.DataFrame,
    accepted_routes: Mapping[str, pd.DataFrame],
    *,
    parent_prediction_column: str = "prediction",
    route_prediction_columns: Mapping[str, str] | None = None,
    route_receipts: Sequence[RouteReceipt] = (),
) -> EnsembleResult:
    """构造 A69 静态 cross-fitted 融合，并在门槛失败时明确 STOP。

    ``accepted_routes`` 只能包含已经通过各自 screening、development 和未来扰动
    收据的路线。没有候选路线时仍输出完整父模型审计产物，便于复现 STOP。
    """

    route_prediction_columns = route_prediction_columns or {}
    merged, route_checks = collect_matching_oofs(
        parent_rows,
        accepted_routes,
        parent_prediction_column=parent_prediction_column,
        candidate_prediction_columns=route_prediction_columns,
    )
    route_names = tuple(sorted(accepted_routes))
    raw_columns = {
        PARENT_ROUTE: f"{PARENT_ROUTE}__prediction",
        **{route: f"{route}__prediction" for route in route_names},
    }
    # 每条基础路线先投影，保证所有报告的比较对象都满足物理不等式。
    for column in raw_columns.values():
        projected = _capacity_project(merged.loc[:, [*KEY_COLUMNS, column]], column)
        merged[column] = projected[column].to_numpy(dtype=float)
    blended, trace = cross_fitted_static_blend(merged, route_names=route_names)
    output = merged.merge(
        blended.drop(columns=list(KEY_COLUMNS)),
        left_index=True,
        right_index=True,
        how="inner",
        validate="one_to_one",
    )
    output = output.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    prediction_columns = {**raw_columns, "a69_cross_fitted_static": "prediction"}
    validate_prediction_contract(output, prediction_columns.values())
    gate = static_fusion_gate(output)
    selected_per_fold = trace.loc[:, ["held_fold", "selected_candidate", "weights"]].to_dict(orient="records")
    unique_candidates = sorted(output["selected_candidate"].astype(str).unique().tolist())
    static_selected = "a69_cross_fitted_static" if gate["passed"] else PARENT_ROUTE
    status = "FROZEN_STATIC_ENSEMBLE" if gate["passed"] else "STOP_STATIC_FUSION"
    fold_metrics, target_metrics, horizon_metrics = _metric_tables(output, prediction_columns)
    correlation = residual_correlation(output, prediction_columns)
    route_receipt_table = _route_receipt_frame(route_receipts)
    first_round = _first_round_summary(output, prediction_columns)
    rejection_reasons = {
        item.name: item.reason for item in route_receipts if not item.accepted
    }
    report: dict[str, object] = {
        "experiment": "A69_causal_trajectory_ensemble",
        "status": status,
        "blind_labels_used": False,
        "platform_reference_used": False,
        "future_generator_truth_used": False,
        "static_only": True,
        "regime_gate": {
            "status": "NOT_EVALUATED" if not gate["passed"] else "NOT_IMPLEMENTED_BY_FROZEN_PLAN",
            "reason": (
                "静态融合没有可靠 development 改善，不允许追加动态 gate 搜索"
                if not gate["passed"]
                else "冻结方案只保留静态融合，未追加 regime gate 搜索"
            ),
            "allowed_origin_features": [
                "slope",
                "volatility",
                "recent_change",
                "holder_mismatch",
                "gas_mismatch",
                "analog_distance",
                "model_disagreement",
            ],
        },
        "routes": {
            "accepted": list(route_names),
            "rejected": [item.name for item in route_receipts if not item.accepted],
            "key_checks": route_checks,
            "rejection_reasons": rejection_reasons,
        },
        "first_round": first_round,
        "weight_search": {
            "type": "leave_one_fold_out_cross_fitted_static",
            "candidate_weights": [
                {"name": name, "weights": weights}
                for name, weights in pre_registered_weight_candidates(route_names)
            ],
            "held_fold_labels_used_for_weights": False,
            "selected_per_fold": selected_per_fold,
            "unique_selected_candidates": unique_candidates,
        },
        "static_gate": gate,
        "frozen_solution": {
            "name": static_selected,
            "model_structure": "nonnegative static convex blend with capacity projection",
            "capacity_projection": "generator_all = max(generator_all, generator_1)",
            "weights": ({PARENT_ROUTE: 1.0} if not gate["passed"] else "per-fold cross-fitted values in training_trace.csv"),
            "code": [
                "code/gas_forecast/causal_trajectory_ensemble.py",
                "scripts/run_causal_trajectory_plan.py",
            ],
            "origin_features_used_for_static_blend": [],
            "thresholds": {
                "min_improvement_pp": STATIC_MIN_IMPROVEMENT_PP,
                "min_recent5_wins": STATIC_MIN_RECENT5_WINS,
                "max_worst_fold_regression_pp": STATIC_MAX_WORST_FOLD_REGRESSION_PP,
                "max_target_regression_pp": STATIC_MAX_TARGET_REGRESSION_PP,
            },
        },
        "future_perturbation": {
            "status": "INPUT_ROUTE_RECEIPTS_ONLY",
            "passed": bool(
                any(item.accepted for item in route_receipts)
                and all(
                    item.future_perturbation_passed is True
                    for item in route_receipts
                    if item.accepted
                )
            ),
            "methods": ["extreme", "shuffle", "null", "delete_future"],
            "note": "A69 只混合已审计 OOF；基础路线审计收据见 route_receipts.csv",
        },
        "prediction_contract": {
            "targets": list(TARGETS),
            "horizons_minutes": list(HORIZONS),
            "no_nan_inf": True,
            "generator_all_gte_generator_1": True,
            "deterministic": True,
        },
        "rows": int(len(output)),
        "route_receipts": route_receipt_table.to_dict(orient="records"),
    }
    return EnsembleResult(
        rows=output,
        fold_metrics=fold_metrics,
        target_metrics=target_metrics,
        horizon_metrics=horizon_metrics,
        residual_correlation=correlation,
        training_trace=trace,
        route_receipts=route_receipt_table,
        report=report,
    )


def write_ensemble_artifacts(result: EnsembleResult, run_dir: str | Path) -> Path:
    """写入 A69 规定的独立实验产物，不更新 results/best 或提交目录。"""

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    result.rows.to_parquet(output / "oof.parquet", index=False)
    result.rows.to_csv(output / "oof.csv", index=False, encoding="utf-8")
    result.fold_metrics.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8")
    result.target_metrics.to_csv(output / "target_metrics.csv", index=False, encoding="utf-8")
    result.horizon_metrics.to_csv(output / "horizon_metrics.csv", index=False, encoding="utf-8")
    result.residual_correlation.to_csv(output / "residual_correlation.csv", index=False, encoding="utf-8")
    result.training_trace.to_csv(output / "training_trace.csv", index=False, encoding="utf-8")
    result.route_receipts.to_csv(output / "route_receipts.csv", index=False, encoding="utf-8")
    (output / "report.json").write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "EnsembleResult",
    "HORIZONS",
    "KEY_COLUMNS",
    "PARENT_ROUTE",
    "RouteReceipt",
    "TARGETS",
    "build_causal_trajectory_ensemble",
    "canonicalize_oof",
    "collect_matching_oofs",
    "cross_fitted_static_blend",
    "pre_registered_weight_candidates",
    "read_oof",
    "residual_correlation",
    "static_fusion_gate",
    "validate_oof_contract",
    "validate_prediction_contract",
    "write_ensemble_artifacts",
]
