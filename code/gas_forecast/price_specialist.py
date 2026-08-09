"""初赛 Price-Switch Error Atlas 与严格前向残差专家。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from gas_forecast.aggressive import (
    DEFAULT_BRANCH_COLUMNS,
    _fold_order,
    evaluate_candidate,
    normalize_branch_frame,
    validate_oof_contract,
)
from gas_forecast.scoring import absolute_percentage_error


PRICE_HORIZONS = (15, 30, 45, 60, 75, 90, 105, 120)


@dataclass(frozen=True)
class PriceSpecialistConfig:
    model: str
    alpha: float
    clip_mw: float

    def __post_init__(self) -> None:
        if self.model not in {"ridge", "huber", "gam"}:
            raise ValueError(f"不支持的 price specialist: {self.model}")
        if self.alpha not in {0.25, 0.50, 0.75, 1.00}:
            raise ValueError("alpha 必须来自冻结集合 0.25/0.50/0.75/1.00")
        if self.clip_mw not in {5.0, 10.0, 15.0}:
            raise ValueError("clip 必须来自冻结集合 5/10/15 MW")


def _price_column(rows: pd.DataFrame, horizon: int) -> str | None:
    candidates = (
        f"future_price_{horizon}",
        f"feat_target_price_tplus_{horizon}",
        f"price_tplus_{horizon}",
    )
    return next((column for column in candidates if column in rows), None)


def build_price_switch_features(rows: pd.DataFrame) -> pd.DataFrame:
    """从官方 known-future price 构造固定、可解释的 switch 特征。"""

    output = normalize_branch_frame(rows)
    current_column = next(
        (column for column in ("current_price", "feat_current_price") if column in output),
        None,
    )
    price_columns = [_price_column(output, horizon) for horizon in PRICE_HORIZONS]
    if current_column is None or any(column is None for column in price_columns):
        raise ValueError("Price Specialist 需要 current_price 和未来 15–120 分钟价格")
    current = output[current_column].to_numpy(float)
    matrix = output.loc[:, price_columns].to_numpy(float)
    changed = ~np.isclose(matrix, current[:, None], rtol=0.0, atol=1e-12)
    switch = changed.any(axis=1)
    first_step = np.where(switch, changed.argmax(axis=1) + 1, 0)
    first_index = np.maximum(first_step - 1, 0)
    first_price = matrix[np.arange(len(output)), first_index]
    first_delta = np.where(switch, first_price - current, 0.0)
    adjacent = np.column_stack(
        [~np.isclose(matrix[:, 0], current, atol=1e-12), np.diff(matrix, axis=1) != 0]
    )

    output["current_price"] = current
    for position, horizon in enumerate(PRICE_HORIZONS):
        output[f"future_price_{horizon}"] = matrix[:, position]
        output[f"price_delta_{horizon}"] = matrix[:, position] - current
    output["max_price_next_120"] = matrix.max(axis=1)
    output["min_price_next_120"] = matrix.min(axis=1)
    output["price_range_next_120"] = matrix.max(axis=1) - matrix.min(axis=1)
    output["switch_within_120"] = switch.astype("int8")
    output["steps_to_first_switch"] = first_step.astype("int8")
    output["switch_direction"] = np.sign(first_delta).astype("int8")
    output["first_switch_delta"] = first_delta
    output["n_switches_120"] = adjacent.sum(axis=1).astype("int8")
    output["switch_category"] = pd.cut(
        first_step,
        bins=[-1, 0, 1, 2, 3, 5, 8],
        labels=[
            "no_switch",
            "immediate_switch",
            "switch_in_1_step",
            "switch_in_2_steps",
            "switch_in_3_4_steps",
            "switch_in_5_8_steps",
        ],
    ).astype(str)
    horizon_step = np.ceil(output["horizon"].to_numpy(float) / 15.0).astype(int)
    relation = np.full(len(output), "no_switch", dtype=object)
    relation[switch & (horizon_step < first_step)] = "before_switch"
    relation[switch & (horizon_step == first_step)] = "at_switch"
    relation[switch & (horizon_step > first_step)] = "after_switch"
    output["horizon_switch_relation"] = relation

    output["c0_delta_from_persistence"] = output["c0_pred"] - output["persistence_pred"]
    if {"v2_pred", "v3_pred"}.issubset(output.columns):
        output["v2_v3_disagreement"] = output["v2_pred"] - output["v3_pred"]
    else:
        output["v2_v3_disagreement"] = 0.0
    available_branches = [column for column in DEFAULT_BRANCH_COLUMNS if column in output]
    if available_branches:
        output["branch_prediction_std"] = output[available_branches].std(axis=1, ddof=0)
    else:
        output["branch_prediction_std"] = 0.0
    return output


def price_error_atlas(
    rows: pd.DataFrame,
    *,
    baseline_column: str = "c0_pred",
    include_blind: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """按 switch 类别、方向、幅度和相对 horizon 汇总 Strict C0 误差。"""

    work = build_price_switch_features(rows)
    validate_oof_contract(work, (baseline_column,))
    work["ape"] = absolute_percentage_error(work["actual"], work[baseline_column])
    denominator = work["actual"].abs().clip(lower=1e-6)
    work["signed_percentage_bias"] = (work[baseline_column] - work["actual"]) / denominator
    analysis = work if include_blind else work.loc[~work["fold"].astype(str).eq("blind")]
    switched = analysis["switch_within_120"].eq(1)
    magnitude = analysis.loc[switched, "first_switch_delta"].abs()
    work["price_magnitude_quantile"] = "no_switch"
    if magnitude.nunique() >= 4:
        work.loc[magnitude.index, "price_magnitude_quantile"] = pd.qcut(
            magnitude.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]
        ).astype(str)
    else:
        work.loc[magnitude.index, "price_magnitude_quantile"] = "Q1"
    analysis = work.loc[analysis.index]

    def grouped(columns: Sequence[str]) -> list[dict[str, object]]:
        result = (
            analysis.groupby(list(columns), dropna=False, sort=True)
            .agg(rows=("ape", "size"), mape=("ape", "mean"), bias=("signed_percentage_bias", "mean"))
            .reset_index()
        )
        return result.to_dict(orient="records")

    switch_mask = analysis["switch_within_120"].eq(1)
    switch_part = analysis.loc[switch_mask]
    non_switch_part = analysis.loc[~switch_mask]
    report = {
        "scope": "all" if include_blind else "development_only",
        "rows": int(len(analysis)),
        "switch_coverage": float(switch_mask.mean()),
        "mape_switch": float(switch_part["ape"].mean()) if len(switch_part) else None,
        "mape_non_switch": float(non_switch_part["ape"].mean()) if len(non_switch_part) else None,
        "bias_switch": float(switch_part["signed_percentage_bias"].mean())
        if len(switch_part)
        else None,
        "bias_before_switch": float(
            analysis.loc[analysis["horizon_switch_relation"].eq("before_switch"), "signed_percentage_bias"].mean()
        ),
        "bias_after_switch": float(
            analysis.loc[analysis["horizon_switch_relation"].eq("after_switch"), "signed_percentage_bias"].mean()
        ),
        "by_switch_category": grouped(("switch_category",)),
        "by_direction": grouped(("switch_direction",)),
        "by_magnitude": grouped(("price_magnitude_quantile",)),
        "by_horizon_relation": grouped(("horizon_switch_relation",)),
    }
    return work, report


def select_price_specialist_features(rows: pd.DataFrame) -> pd.DataFrame:
    """提取计划冻结的低自由度 price residual 特征并固定 one-hot 结构。"""

    direct = {
        "current_price",
        "max_price_next_120",
        "min_price_next_120",
        "price_range_next_120",
        "switch_within_120",
        "steps_to_first_switch",
        "switch_direction",
        "first_switch_delta",
        "n_switches_120",
        "current_value",
        "c0_pred",
        "c0_delta_from_persistence",
        "v2_v3_disagreement",
        "branch_prediction_std",
    }
    prefixes = (
        "future_price_",
        "price_delta_",
        "generator_1_",
        "generator_all_",
        "generator_rest_",
        "feat_generator_1_",
        "feat_generator_rest_",
        "feat_gas_holder",
        "feat_generator_gas_total",
        "feat_gas_balance",
    )
    columns = [
        column
        for column in rows.columns
        if column in direct or any(column.startswith(prefix) for prefix in prefixes)
    ]
    numeric = rows.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    categorical = pd.get_dummies(
        rows[["target", "horizon"]].astype(str),
        prefix=["target", "horizon"],
        dtype=float,
    )
    return pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)


def _make_model(name: str) -> Pipeline:
    if name == "ridge":
        estimator = Ridge(alpha=20.0)
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    elif name == "huber":
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000)),
        ]
    elif name == "gam":
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
            ("model", Ridge(alpha=20.0)),
        ]
    else:
        raise ValueError(f"不支持的模型: {name}")
    return Pipeline(steps)


def time_ordered_price_corrections(
    rows: pd.DataFrame,
    *,
    model: str,
    baseline_column: str = "c0_pred",
    use_blind_for_reporting: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """仅用历史折的 switch OOF residual 拟合当前折 correction。"""

    work = build_price_switch_features(rows).reset_index(drop=True)
    validate_oof_contract(work, (baseline_column,))
    features = select_price_specialist_features(work)
    correction = np.zeros(len(work), dtype=float)
    folds = _fold_order(work)
    trajectory: list[dict[str, object]] = []
    for position, fold in enumerate(folds):
        held = work["fold"].astype(str).eq(fold).to_numpy()
        active = held & work["switch_within_120"].eq(1).to_numpy()
        if position == 0 or (fold == "blind" and not use_blind_for_reporting):
            trajectory.append({"fold": fold, "fallback": "zero_correction"})
            continue
        history_folds = [value for value in folds[:position] if value != "blind"]
        train = work["fold"].astype(str).isin(history_folds).to_numpy()
        train &= work["switch_within_120"].eq(1).to_numpy()
        if int(train.sum()) < 64 or not active.any():
            trajectory.append({"fold": fold, "fallback": "zero_correction"})
            continue
        estimator = _make_model(model)
        residual = work["actual"].to_numpy(float) - work[baseline_column].to_numpy(float)
        estimator.fit(features.loc[train], residual[train])
        correction[active] = estimator.predict(features.loc[active])
        fitted_model = estimator.named_steps["model"]
        iteration_count = getattr(fitted_model, "n_iter_", None)
        iteration_limit = getattr(fitted_model, "max_iter", None)
        converged = (
            iteration_count is None
            or iteration_limit is None
            or int(np.max(np.atleast_1d(iteration_count))) < int(iteration_limit)
        )
        trajectory.append(
            {
                "fold": fold,
                "train_switch_rows": int(train.sum()),
                "active_rows": int(active.sum()),
                "converged": converged,
            }
        )
    column = f"price_{model}_raw_correction"
    work[column] = correction
    fitted = [item for item in trajectory if "converged" in item]
    return work, {
        "model": model,
        "feature_columns": list(features.columns),
        "fold_trajectory": trajectory,
        "active_rows": int(work["switch_within_120"].sum()),
        "converged": bool(fitted) and all(bool(item["converged"]) for item in fitted),
    }


def run_price_specialist_grid(
    rows: pd.DataFrame,
    *,
    baseline_column: str = "c0_pred",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """运行 Ridge/Huber/GAM 与冻结的 4×3 alpha/clip 小网格。"""

    atlas_rows, atlas = price_error_atlas(rows, baseline_column=baseline_column)
    output = atlas_rows.reset_index(drop=True)
    ranking: list[dict[str, object]] = []
    model_reports: dict[str, object] = {}
    for model in ("ridge", "huber", "gam"):
        corrected, model_report = time_ordered_price_corrections(
            output, model=model, baseline_column=baseline_column
        )
        raw_column = f"price_{model}_raw_correction"
        output[raw_column] = corrected[raw_column]
        model_reports[model] = model_report
        for alpha in (0.25, 0.50, 0.75, 1.00):
            for clip_mw in (5.0, 10.0, 15.0):
                config = PriceSpecialistConfig(model, alpha, clip_mw)
                column = f"price_{model}_a{int(alpha * 100):03d}_c{int(clip_mw):02d}_pred"
                active = output["switch_within_120"].eq(1)
                clipped = output[raw_column].clip(-clip_mw, clip_mw)
                output[column] = output[baseline_column]
                output.loc[active, column] = (
                    output.loc[active, baseline_column] + alpha * clipped.loc[active]
                )
                evaluation = evaluate_candidate(
                    output, column, baseline_column=baseline_column
                )
                ranking.append(
                    {
                        "candidate": column,
                        "config": config.__dict__,
                        "pooled_mape": evaluation["candidate"]["pooled_mape"],
                        "delta_pp": evaluation["delta_pp"],
                        "fold_wins": evaluation["fold_wins"],
                        "recent5_wins": evaluation["recent5_wins"],
                        "max_fold_regression_pp": evaluation["max_fold_regression_pp"],
                        "valid": bool(model_report["converged"]),
                    }
                )
    ranking.sort(key=lambda value: (not value["valid"], value["pooled_mape"]))
    best = ranking[0]["candidate"]
    active = output["switch_within_120"].eq(1)
    correction = output[best] - output[baseline_column]
    absolute = correction.loc[active].abs()
    distribution = {
        "p50": float(absolute.quantile(0.50)),
        "p90": float(absolute.quantile(0.90)),
        "p95": float(absolute.quantile(0.95)),
        "p99": float(absolute.quantile(0.99)),
        "max": float(absolute.max()),
    }
    if not np.allclose(
        output.loc[~active, best].to_numpy(float),
        output.loc[~active, baseline_column].to_numpy(float),
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("Price Specialist 非 switch 区域被意外修改")
    return output, {
        "atlas": atlas,
        "ranking": ranking,
        "best_candidate": best,
        "best_evaluation": evaluate_candidate(output, best, baseline_column=baseline_column),
        "correction_distribution_mw": distribution,
        "models": model_reports,
    }
