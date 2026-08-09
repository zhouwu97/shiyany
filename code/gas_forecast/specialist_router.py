"""E111/E112：面向 generator_1 长步长的低覆盖率专家门控。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from gas_forecast.scoring import score_oof_long


def _gate_columns(features: pd.DataFrame) -> list[str]:
    preferred = (
        "feat_generator_1_slope_4",
        "feat_generator_1_slope_8",
        "feat_generator_1_slope_16",
        "feat_generator_1_diff_1",
        "feat_generator_1_diff_2",
        "feat_generator_1_diff_4",
        "feat_generator_1_acceleration",
        "feat_generator_1_ramp_up_run_length",
        "feat_generator_1_ramp_down_run_length",
        "feat_generator_1_ramp_volatility_8",
        "feat_generator_1_ramp_range_8",
        "feat_generator_gas_total",
        "feat_bf_surplus_proxy",
        "feat_blast_balance",
        "feat_coke_balance",
        "feat_converter_balance",
        "blast_furnace_gas_holder_2",
    )
    columns = [column for column in preferred if column in features.columns]
    if not columns:
        columns = [column for column in features.columns if "generator_1" in column][:16]
    if not columns:
        raise ValueError("Ramp gate 没有可用因果特征")
    return columns


def _ramp_labels(rows: pd.DataFrame, *, threshold: float = 3.0) -> pd.Series:
    target = rows.loc[rows["target"].eq("generator_1")]
    if target.empty:
        raise ValueError("Ramp gate 缺少 generator_1 标签行")
    future = target.loc[target["horizon"].astype(int).ge(75)].copy()
    grouped = future.assign(
        move=(future["actual"] - future["current_value"]).abs()
    ).groupby("origin_time", sort=True)["move"].max()
    return grouped.gt(float(threshold)).astype(int)


def _top_q(values: pd.Series, q: float) -> pd.Series:
    if not 0.0 < q <= 1.0:
        raise ValueError("specialist coverage q 必须位于 (0, 1]")
    count = max(1, int(np.ceil(len(values) * q)))
    order = values.sort_values(ascending=False, kind="stable").index[:count]
    selected = pd.Series(False, index=values.index)
    selected.loc[order] = True
    return selected


def build_ramp_specialist_oof(
    rows: pd.DataFrame,
    features: pd.DataFrame,
    *,
    champion_column: str,
    specialist_column: str,
    coverage: float,
    blend_weight: float,
    long_horizons: tuple[int, ...] = (5, 6, 7, 8),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """用严格时间顺序 inner-style gate 生成 specialist OOF。"""

    required = {
        "fold",
        "origin_time",
        "target",
        "horizon",
        "actual",
        "current_value",
        champion_column,
        specialist_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Ramp specialist 输入缺少字段: {missing}")
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("specialist blend_weight 必须位于 [0, 1]")
    work = rows.copy().reset_index(drop=True)
    work["origin_time"] = pd.to_datetime(work["origin_time"])
    gate_columns = _gate_columns(features)
    origin_index = pd.DatetimeIndex(sorted(work["origin_time"].unique()))
    gate_frame = features.reindex(origin_index).loc[:, gate_columns]
    labels = _ramp_labels(work)
    fold_order = (
        work.loc[:, ["fold", "origin_time"]]
        .drop_duplicates()
        .sort_values("origin_time")
        .groupby("fold", sort=False)["origin_time"]
        .min()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    probabilities = pd.Series(0.0, index=origin_index)
    active = pd.Series(False, index=origin_index)
    for position, held_out in enumerate(fold_order):
        held_times = pd.DatetimeIndex(
            work.loc[work["fold"].astype(str).eq(held_out), "origin_time"].unique()
        )
        if position == 0:
            continue
        previous_folds = set(fold_order[:position])
        train_times = pd.DatetimeIndex(
            work.loc[work["fold"].astype(str).isin(previous_folds), "origin_time"].unique()
        )
        train_times = train_times.intersection(labels.index).intersection(gate_frame.index)
        if len(train_times) < 32 or labels.loc[train_times].nunique() < 2:
            continue
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(C=0.2, max_iter=500, random_state=20250731)),
            ]
        )
        estimator.fit(gate_frame.loc[train_times], labels.loc[train_times])
        probabilities.loc[held_times] = estimator.predict_proba(
            gate_frame.reindex(held_times)
        )[:, list(estimator.named_steps["logistic"].classes_).index(1)]
        active.loc[held_times] = _top_q(probabilities.loc[held_times], coverage)

    work["ramp_probability"] = work["origin_time"].map(probabilities).fillna(0.0)
    work["specialist_active"] = work["origin_time"].map(active).fillna(False).astype(bool)
    work["ramp_specialist_pred"] = work[champion_column].astype(float)
    target_mask = work["target"].eq("generator_1") & work["horizon"].isin(
        [15 * horizon for horizon in long_horizons]
    ) & work["specialist_active"]
    work.loc[target_mask, "ramp_specialist_pred"] = (
        (1.0 - blend_weight) * work.loc[target_mask, champion_column].to_numpy(dtype=float)
        + blend_weight * work.loc[target_mask, specialist_column].to_numpy(dtype=float)
    )
    scored_active = work.loc[work["specialist_active"]]
    report = {
        "coverage_target": float(coverage),
        "blend_weight": float(blend_weight),
        "active_origin_coverage": float(active.mean()),
        "active_origins": int(active.sum()),
        "total_origins": int(len(active)),
        "gate_features": gate_columns,
        "specialist_column": specialist_column,
        "champion_column": champion_column,
        "overall": score_oof_long(work, "ramp_specialist_pred"),
        "active_subset": (
            score_oof_long(scored_active, "ramp_specialist_pred")
            if not scored_active.empty
            else None
        ),
    }
    return work, report
