"""A53 Oracle Ramp Router 与 A54 因果信号图谱。

两项工具均只服务于 development OOF 的研究诊断。A53 有意使用真实未来
ramp，因此永远不可部署；A54 的分位数阈值则严格限制在各外层折的训练期。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from gas_forecast.aggressive import project_long_candidate
from gas_forecast.ramp_atlas import RAMP_BANDS, assign_ramp_band
from gas_forecast.research import compare_research_candidate
from gas_forecast.scoring import absolute_percentage_error, competition_mape


G1_LONG_HORIZONS: Final[frozenset[int]] = frozenset({75, 90, 105, 120})
A55_MIN_ORACLE_HEADROOM_PP: Final[float] = 0.005
A58_DISAGREEMENT_QUANTILE: Final[float] = 0.80
A58_MIN_HISTORY_ROWS: Final[int] = 128
A58_MIN_POOLED_IMPROVEMENT_PP: Final[float] = 0.005
A58_MIN_RECENT5_WINS: Final[int] = 3
A58_MAX_FOLD_REGRESSION_PP: Final[float] = 0.100
A58_RAW_COLUMN: Final[str] = "a58_forward_q5_disagreement_raw_pred"
A58_PREDICTION_COLUMN: Final[str] = "a58_forward_q5_disagreement_pred"
QUINTILE_PROBABILITIES: Final[tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
QUINTILE_LABELS: Final[tuple[str, ...]] = ("Q1", "Q2", "Q3", "Q4", "Q5")
UNASSIGNED_QUINTILE: Final[str] = "unassigned"


@dataclass(frozen=True)
class OracleRampRouterResult:
    """A53 的不可部署 oracle 预测、分档统计和审计报告。"""

    rows: pd.DataFrame
    bucket_table: pd.DataFrame
    report: dict[str, object]


@dataclass(frozen=True)
class CausalSignalAtlasResult:
    """A54 的逐信号分位数单元、阈值收据和汇总表。"""

    cells: pd.DataFrame
    table: pd.DataFrame
    ramp_table: pd.DataFrame
    cutoffs: pd.DataFrame
    report: dict[str, object]


@dataclass(frozen=True)
class ForwardDisagreementSpecialistResult:
    """A58 的严格前向分歧路由预测、阈值轨迹和验收报告。"""

    rows: pd.DataFrame
    threshold_trace: pd.DataFrame
    report: dict[str, object]


def _validate_oof_rows(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    specialist_column: str,
    require_train_end: bool,
) -> pd.DataFrame:
    """校验 router 输入的长表键、预测和可选外层训练边界。"""

    required = {
        "fold",
        "origin_time",
        "target",
        "horizon",
        "actual",
        "current_value",
        "persistence_pred",
        baseline_column,
        specialist_column,
    }
    if require_train_end:
        required.add("train_end")
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Ramp Router 输入缺少字段: {missing}")
    work = rows.copy()
    for column in ("origin_time", "train_end") if require_train_end else ("origin_time",):
        work[column] = pd.to_datetime(work[column], errors="coerce")
        if work[column].isna().any():
            raise ValueError(f"Ramp Router 输入含非法 {column}")
    keys = ["fold", "origin_time", "target", "horizon"]
    if work.duplicated(keys).any():
        raise ValueError("Ramp Router 输入存在重复 fold×origin×target×horizon")
    numeric_columns = [
        "actual",
        "current_value",
        "persistence_pred",
        baseline_column,
        specialist_column,
    ]
    numeric = work.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Ramp Router 输入含缺失或非有限真实值/预测")
    work.loc[:, numeric.columns] = numeric
    work["horizon"] = pd.to_numeric(work["horizon"], errors="raise").astype(int)
    return work


def _eligible_long_g1(rows: pd.DataFrame) -> pd.Series:
    """返回 A51 唯一允许修改的 generator_1 长步长单元。"""

    return rows["target"].eq("generator_1") & rows["horizon"].isin(G1_LONG_HORIZONS)


def _bucket_metrics(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    specialist_column: str,
    oracle_column: str,
    epsilon: float,
) -> pd.DataFrame:
    """汇总 A51、Oracle 相对 RichGas 在真实 ramp 档的条件表现。"""

    eligible = rows.loc[rows["oracle_route_eligible"]].copy()
    if eligible.empty:
        raise ValueError("A53 没有 generator_1 长步长可路由单元")
    records: list[dict[str, object]] = []
    denominator = np.maximum(eligible["actual"].abs().to_numpy(dtype=float), epsilon)
    eligible["baseline_ape"] = np.abs(
        eligible[baseline_column].to_numpy(dtype=float) - eligible["actual"].to_numpy(dtype=float)
    ) / denominator
    eligible["specialist_ape"] = np.abs(
        eligible[specialist_column].to_numpy(dtype=float) - eligible["actual"].to_numpy(dtype=float)
    ) / denominator
    eligible["oracle_ape"] = np.abs(
        eligible[oracle_column].to_numpy(dtype=float) - eligible["actual"].to_numpy(dtype=float)
    ) / denominator
    for band, group in eligible.groupby("true_ramp_band", observed=False, sort=False):
        if group.empty:
            continue
        baseline_mape = float(group["baseline_ape"].mean())
        specialist_mape = float(group["specialist_ape"].mean())
        oracle_mape = float(group["oracle_ape"].mean())
        records.append(
            {
                "ramp_band": str(band),
                "rows": int(len(group)),
                "coverage": float(len(group) / len(eligible)),
                "oracle_switch_coverage": float(group["oracle_switched"].mean()),
                "baseline_mape": baseline_mape,
                "specialist_mape": specialist_mape,
                "oracle_mape": oracle_mape,
                "specialist_minus_baseline": specialist_mape - baseline_mape,
                "oracle_minus_baseline": oracle_mape - baseline_mape,
                "specialist_better_fraction": float(
                    (group["specialist_ape"] < group["baseline_ape"]).mean()
                ),
            }
        )
    output = pd.DataFrame(records)
    output["ramp_band"] = pd.Categorical(
        output["ramp_band"],
        categories=[name for name, _, _ in RAMP_BANDS],
        ordered=True,
    )
    return output.sort_values("ramp_band").reset_index(drop=True)


def build_a53_oracle_ramp_router(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    specialist_column: str,
    epsilon: float = 1e-6,
) -> OracleRampRouterResult:
    """构造只供理论上限诊断的真实 ramp Oracle Router。

    真实 ``actual - current_value`` 决定是否切换 A51，故该函数只接受
    development 语义：blind 行无条件剔除，结果也永远不会被标为正式候选。
    """

    if epsilon <= 0.0:
        raise ValueError("A53 epsilon 必须为正数")
    work = _validate_oof_rows(
        rows,
        baseline_column=baseline_column,
        specialist_column=specialist_column,
        require_train_end=False,
    )
    work = work.loc[work["fold"].ne("blind")].copy()
    if work.empty:
        raise ValueError("A53 development OOF 没有可评分行")
    work["actual_delta"] = work["actual"] - work["current_value"]
    work["true_ramp_band"] = assign_ramp_band(work["actual_delta"])
    work["oracle_route_eligible"] = _eligible_long_g1(work)
    if not work["oracle_route_eligible"].any():
        raise ValueError("A53 没有 generator_1 的 75/90/105/120 分钟单元")
    work["oracle_switched"] = work["oracle_route_eligible"] & work["true_ramp_band"].ne(
        "stable"
    )
    raw_column = "a53_oracle_ramp_raw_pred"
    output_column = "a53_oracle_ramp_pred"
    work[raw_column] = work[baseline_column].to_numpy(dtype=float)
    switched = work["oracle_switched"]
    work.loc[switched, raw_column] = work.loc[switched, specialist_column].to_numpy(
        dtype=float
    )
    raw_changed = ~np.isclose(
        work[raw_column].to_numpy(dtype=float),
        work[baseline_column].to_numpy(dtype=float),
    )
    noneligible_raw_changes = int((raw_changed & ~work["oracle_route_eligible"].to_numpy()).sum())
    projected = project_long_candidate(work, raw_column, output_column=output_column)
    comparison = compare_research_candidate(
        projected,
        output_column,
        baseline_column,
        scope="development",
    )
    bucket_table = _bucket_metrics(
        projected,
        baseline_column=baseline_column,
        specialist_column=specialist_column,
        oracle_column=output_column,
        epsilon=epsilon,
    )
    difference = float(comparison["pooled_difference"])
    improvement_pp = -difference * 100.0
    report = {
        "stage": "A53_perfect_true_ramp_router",
        "scope": "development",
        "baseline_column": baseline_column,
        "specialist_column": specialist_column,
        "oracle_column": output_column,
        "target_scope": "generator_1",
        "eligible_horizons": sorted(G1_LONG_HORIZONS),
        "rows": int(len(projected)),
        "eligible_rows": int(projected["oracle_route_eligible"].sum()),
        "oracle_switch_rows": int(projected["oracle_switched"].sum()),
        "oracle_switch_coverage": float(projected["oracle_switched"].mean()),
        "eligible_switch_coverage": float(
            projected["oracle_switched"].sum() / projected["oracle_route_eligible"].sum()
        ),
        "raw_route_audit": {
            "raw_changed_cells": int(raw_changed.sum()),
            "noneligible_raw_changed_cells": noneligible_raw_changes,
            "selector_only_changes_g1_long": bool(noneligible_raw_changes == 0),
            "capacity_projection_modified_cells": int(
                (~np.isclose(
                    projected[raw_column].to_numpy(dtype=float),
                    projected[output_column].to_numpy(dtype=float),
                )).sum()
            ),
        },
        "comparison": comparison,
        "oracle_headroom": {
            "pooled_difference": difference,
            "pooled_improvement_pp": improvement_pp,
            "a55_minimum_headroom_pp": A55_MIN_ORACLE_HEADROOM_PP,
            "meets_a55_headroom": bool(improvement_pp >= A55_MIN_ORACLE_HEADROOM_PP),
        },
        "bucket_summary": bucket_table.to_dict(orient="records"),
        "oracle_only": True,
        "actual_ramp_used": True,
        "deployable": False,
        "formal_candidate": False,
        "strict_oof_contract": (
            "仅使用 development OOF；真实未来 ramp 只用于理论上限，不能进入训练、"
            "阈值选择或生产推理。容量投影与生产一致。"
        ),
    }
    return OracleRampRouterResult(
        rows=projected.reset_index(drop=True),
        bucket_table=bucket_table,
        report=report,
    )


def _require_feature_series(features: pd.DataFrame, column: str) -> pd.Series:
    """读取一个固定因果特征列，缺列即中止而非静默降级。"""

    if column not in features:
        raise ValueError(f"A54 因果特征矩阵缺少字段: {column}")
    values = pd.to_numeric(features[column], errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def _causal_feature_signals(features: pd.DataFrame) -> dict[str, pd.Series]:
    """构造 A54 预注册的低维可观测 ramp-risk 信号。"""

    g1_diff_1 = _require_feature_series(features, "feat_generator_1_diff_1")
    g1_diff_2 = _require_feature_series(features, "feat_generator_1_diff_2")
    g1_diff_4 = _require_feature_series(features, "feat_generator_1_diff_4")
    g1_max_4 = _require_feature_series(features, "feat_generator_1_max_4")
    g1_min_4 = _require_feature_series(features, "feat_generator_1_min_4")
    g1_max_8 = _require_feature_series(features, "feat_generator_1_max_8")
    g1_min_8 = _require_feature_series(features, "feat_generator_1_min_8")
    available = _require_feature_series(features, "feat_rich_gas_available_for_generation")
    return {
        "g1_abs_diff_1": g1_diff_1.abs(),
        "g1_abs_diff_2": g1_diff_2.abs(),
        "g1_abs_diff_4": g1_diff_4.abs(),
        "g1_range_4": (g1_max_4 - g1_min_4).abs(),
        "g1_range_8": (g1_max_8 - g1_min_8).abs(),
        "g1_std_4": _require_feature_series(features, "feat_generator_1_std_4").abs(),
        "g1_std_8": _require_feature_series(features, "feat_generator_1_std_8").abs(),
        "holder_abs_slope_4": _require_feature_series(
            features, "feat_blast_furnace_gas_holder_2_slope_4"
        ).abs(),
        "available_gas_abs_change": available.diff().abs(),
        "rest_abs_slope_4": _require_feature_series(
            features, "feat_generator_rest_slope_4"
        ).abs(),
    }


def _quantile_edges(values: pd.Series, *, minimum_rows: int) -> np.ndarray | None:
    """从历史有限值计算固定 Q1--Q5 边界，历史不足时返回空。"""

    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if len(finite) < minimum_rows:
        return None
    return np.quantile(finite.to_numpy(dtype=float), QUINTILE_PROBABILITIES)


def _assign_quintiles(values: pd.Series, edges: np.ndarray | None) -> pd.Series:
    """按过去期边界给当前单元编号；不在当前折内重新排序。"""

    output = pd.Series(UNASSIGNED_QUINTILE, index=values.index, dtype="object")
    if edges is None:
        return output
    numeric = values.to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    if finite.any():
        bins = np.searchsorted(edges[1:-1], numeric[finite], side="right")
        output.loc[output.index[finite]] = [QUINTILE_LABELS[int(value)] for value in bins]
    return output


def _fold_train_ends(rows: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """验证每个 OOF fold 对应唯一的严格训练结束时刻。"""

    output: dict[str, pd.Timestamp] = {}
    for fold, part in rows.groupby("fold", sort=True):
        ends = pd.DatetimeIndex(part["train_end"].unique())
        if len(ends) != 1:
            raise ValueError(f"A54 fold {fold} 含多个 train_end，拒绝推断历史边界")
        output[str(fold)] = pd.Timestamp(ends[0])
    return output


def _cutoff_record(
    *,
    fold: str,
    horizon: int | None,
    signal_name: str,
    signal_source: str,
    train_end: pd.Timestamp,
    history: pd.Series,
    edges: np.ndarray | None,
) -> dict[str, object]:
    """把每折分位数历史范围写为可审计的扁平记录。"""

    finite = history[np.isfinite(history.to_numpy(dtype=float))]
    record: dict[str, object] = {
        "fold": fold,
        "horizon": horizon,
        "signal_name": signal_name,
        "signal_source": signal_source,
        "train_end": train_end,
        "history_rows": int(len(finite)),
        "history_max_time": (
            pd.NaT if finite.empty else pd.Timestamp(finite.index.max())
        ),
        "status": "ready" if edges is not None else "insufficient_history",
    }
    edge_values = () if edges is None else edges
    for label, value in zip(("min", "q20", "q40", "q60", "q80", "max"), edge_values):
        record[f"cutoff_{label}"] = float(value)
    return record


def _summarize_atlas_cells(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """按信号和历史分位数汇总 A51 相对 RichGas 的条件收益。"""

    assigned = cells.loc[cells["quintile"].ne(UNASSIGNED_QUINTILE)].copy()
    totals = cells.groupby("signal_name", sort=True).size().to_dict()
    records: list[dict[str, object]] = []
    for keys, group in assigned.groupby(
        ["signal_name", "signal_source", "quintile"], observed=False, sort=True
    ):
        signal_name, signal_source, quintile = keys
        baseline_mape = float(group["baseline_ape"].mean())
        specialist_mape = float(group["specialist_ape"].mean())
        records.append(
            {
                "signal_name": str(signal_name),
                "signal_source": str(signal_source),
                "quintile": str(quintile),
                "rows": int(len(group)),
                "coverage": float(len(group) / int(totals[str(signal_name)])),
                "mean_signal_value": float(group["signal_value"].mean()),
                "baseline_mape": baseline_mape,
                "specialist_mape": specialist_mape,
                "mape_difference": specialist_mape - baseline_mape,
                "specialist_better_fraction": float(
                    (group["specialist_ape"] < group["baseline_ape"]).mean()
                ),
                "true_ramp_fraction": float(group["true_ramp_band"].ne("stable").mean()),
                "folds": int(group["fold"].nunique()),
            }
        )
    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("A54 没有具备足够历史的因果信号分位数")
    table["quintile"] = pd.Categorical(
        table["quintile"], categories=QUINTILE_LABELS, ordered=True
    )
    table = table.sort_values(["signal_name", "quintile"]).reset_index(drop=True)

    ramp_records: list[dict[str, object]] = []
    for keys, group in assigned.groupby(
        ["signal_name", "quintile", "true_ramp_band"], observed=False, sort=True
    ):
        signal_name, quintile, band = keys
        if group.empty:
            continue
        baseline_mape = float(group["baseline_ape"].mean())
        specialist_mape = float(group["specialist_ape"].mean())
        ramp_records.append(
            {
                "signal_name": str(signal_name),
                "quintile": str(quintile),
                "true_ramp_band": str(band),
                "rows": int(len(group)),
                "baseline_mape": baseline_mape,
                "specialist_mape": specialist_mape,
                "mape_difference": specialist_mape - baseline_mape,
            }
        )
    ramp_table = pd.DataFrame(ramp_records)
    if not ramp_table.empty:
        ramp_table["quintile"] = pd.Categorical(
            ramp_table["quintile"], categories=QUINTILE_LABELS, ordered=True
        )
        ramp_table["true_ramp_band"] = pd.Categorical(
            ramp_table["true_ramp_band"],
            categories=[name for name, _, _ in RAMP_BANDS],
            ordered=True,
        )
        ramp_table = ramp_table.sort_values(
            ["signal_name", "quintile", "true_ramp_band"]
        ).reset_index(drop=True)

    signal_summary: list[dict[str, object]] = []
    for signal_name, group in table.groupby("signal_name", observed=True, sort=True):
        ordered = group.sort_values("quintile")
        differences = ordered["mape_difference"].to_numpy(dtype=float)
        q1 = ordered.loc[ordered["quintile"].eq("Q1"), "mape_difference"]
        q5 = ordered.loc[ordered["quintile"].eq("Q5"), "mape_difference"]
        signal_summary.append(
            {
                "signal_name": str(signal_name),
                "assigned_coverage": float(int(ordered["rows"].sum()) / int(totals[str(signal_name)])),
                "quintiles_observed": [str(value) for value in ordered["quintile"]],
                "q5_minus_q1_difference_pp": (
                    None
                    if q1.empty or q5.empty
                    else float((q5.iloc[0] - q1.iloc[0]) * 100.0)
                ),
                "nonincreasing_specialist_difference": bool(
                    np.all(np.diff(differences) <= 1e-12)
                ),
            }
        )
    return table, ramp_table, signal_summary


def build_a54_causal_signal_atlas(
    rows: pd.DataFrame,
    features: pd.DataFrame,
    *,
    baseline_column: str,
    rich_gas_column: str,
    specialist_column: str,
    min_history_rows: int = 128,
    epsilon: float = 1e-6,
) -> CausalSignalAtlasResult:
    """构造每折训练期分位数的 A54 disagreement/ramp 诊断图谱。

    因果传感器信号从 ``features.index <= train_end`` 取阈值；两种模型分歧
    则只从同一目标/步长的既完成 OOF 预测取阈值。真实未来只在最终误差和
    ramp 分层展示中出现，绝不参与阈值计算。
    """

    if min_history_rows < 1:
        raise ValueError("A54 min_history_rows 必须为正数")
    if epsilon <= 0.0:
        raise ValueError("A54 epsilon 必须为正数")
    work = _validate_oof_rows(
        rows,
        baseline_column=baseline_column,
        specialist_column=specialist_column,
        require_train_end=True,
    )
    if rich_gas_column not in work:
        raise ValueError(f"A54 输入缺少 RichGas 预测列: {rich_gas_column}")
    rich_gas = pd.to_numeric(work[rich_gas_column], errors="coerce")
    if not np.isfinite(rich_gas.to_numpy(dtype=float)).all():
        raise ValueError("A54 RichGas 预测含缺失或非有限数")
    work[rich_gas_column] = rich_gas.astype(float)
    work = work.loc[work["fold"].ne("blind") & _eligible_long_g1(work)].copy()
    if work.empty:
        raise ValueError("A54 development OOF 没有 generator_1 长步长行")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("A54 因果特征必须使用 DatetimeIndex")
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("A54 因果特征时间轴必须严格递增且唯一")
    work = work.sort_values(["fold", "origin_time", "horizon"]).reset_index(drop=True)
    train_ends = _fold_train_ends(work)
    feature_signals = _causal_feature_signals(features)
    model_signals = {
        "richgas_champion_abs_disagreement": (
            work[rich_gas_column] - work[baseline_column]
        ).abs(),
        "a51_richgas_abs_disagreement": (
            work[specialist_column] - work[rich_gas_column]
        ).abs(),
    }
    work["actual_delta"] = work["actual"] - work["current_value"]
    work["true_ramp_band"] = assign_ramp_band(work["actual_delta"])
    work["baseline_ape"] = absolute_percentage_error(
        work["actual"], work[rich_gas_column], epsilon=epsilon
    )
    work["specialist_ape"] = absolute_percentage_error(
        work["actual"], work[specialist_column], epsilon=epsilon
    )

    cell_parts: list[pd.DataFrame] = []
    cutoff_records: list[dict[str, object]] = []
    for signal_name, complete_signal in feature_signals.items():
        for fold, held in work.groupby("fold", sort=True):
            train_end = train_ends[str(fold)]
            history = complete_signal.loc[complete_signal.index <= train_end]
            edges = _quantile_edges(history, minimum_rows=min_history_rows)
            cutoff_records.append(
                _cutoff_record(
                    fold=str(fold),
                    horizon=None,
                    signal_name=signal_name,
                    signal_source="causal_feature",
                    train_end=train_end,
                    history=history,
                    edges=edges,
                )
            )
            values = complete_signal.reindex(pd.DatetimeIndex(held["origin_time"]))
            part = held.copy()
            part["signal_name"] = signal_name
            part["signal_source"] = "causal_feature"
            part["signal_value"] = values.to_numpy(dtype=float)
            part["quintile"] = _assign_quintiles(part["signal_value"], edges).to_numpy()
            cell_parts.append(part)

    for signal_name, values in model_signals.items():
        work[signal_name] = values.to_numpy(dtype=float)
        for fold, held in work.groupby("fold", sort=True):
            train_end = train_ends[str(fold)]
            for horizon, held_horizon in held.groupby("horizon", sort=True):
                history = work.loc[
                    work["origin_time"].le(train_end) & work["horizon"].eq(horizon), signal_name
                ]
                history.index = pd.DatetimeIndex(
                    work.loc[
                        work["origin_time"].le(train_end) & work["horizon"].eq(horizon),
                        "origin_time",
                    ]
                )
                edges = _quantile_edges(history, minimum_rows=min_history_rows)
                cutoff_records.append(
                    _cutoff_record(
                        fold=str(fold),
                        horizon=int(horizon),
                        signal_name=signal_name,
                        signal_source="historical_oof_prediction",
                        train_end=train_end,
                        history=history,
                        edges=edges,
                    )
                )
                part = held_horizon.copy()
                part["signal_name"] = signal_name
                part["signal_source"] = "historical_oof_prediction"
                part["signal_value"] = part[signal_name].to_numpy(dtype=float)
                part["quintile"] = _assign_quintiles(part["signal_value"], edges).to_numpy()
                cell_parts.append(part)

    cells = pd.concat(cell_parts, ignore_index=True)
    cells["quintile"] = pd.Categorical(
        cells["quintile"], categories=[*QUINTILE_LABELS, UNASSIGNED_QUINTILE], ordered=True
    )
    cutoffs = pd.DataFrame(cutoff_records).sort_values(
        ["signal_name", "fold", "horizon"], na_position="first"
    ).reset_index(drop=True)
    table, ramp_table, signal_summary = _summarize_atlas_cells(cells)
    baseline_mape = competition_mape(work["actual"], work[rich_gas_column], epsilon=epsilon)
    specialist_mape = competition_mape(work["actual"], work[specialist_column], epsilon=epsilon)
    report = {
        "stage": "A54_causal_disagreement_ramp_atlas",
        "scope": "development",
        "target_scope": "generator_1",
        "eligible_horizons": sorted(G1_LONG_HORIZONS),
        "baseline_column": rich_gas_column,
        "champion_column": baseline_column,
        "specialist_column": specialist_column,
        "min_history_rows": int(min_history_rows),
        "quantile_probabilities": list(QUINTILE_PROBABILITIES),
        "rows": int(len(work)),
        "signals": signal_summary,
        "long_horizon_pairwise": {
            "baseline_mape": baseline_mape,
            "specialist_mape": specialist_mape,
            "mape_difference": specialist_mape - baseline_mape,
        },
        "diagnostic_only": True,
        "formal_candidate": False,
        "blind_used": False,
        "strict_oof_contract": {
            "feature_quantile_history": "features.index <= outer_fold.train_end",
            "prediction_quantile_history": (
                "同一 generator_1 长步长、origin_time <= outer_fold.train_end 的历史 OOF 预测"
            ),
            "labels_used_for_quantiles": False,
            "held_fold_used_for_cutoffs": False,
            "blind_used": False,
            "true_ramp_usage": "仅用于诊断表的分层，不参与分位数或门控选择",
        },
    }
    return CausalSignalAtlasResult(
        cells=cells.reset_index(drop=True),
        table=table,
        ramp_table=ramp_table,
        cutoffs=cutoffs,
        report=report,
    )


def _chronological_development_folds(rows: pd.DataFrame) -> list[str]:
    """按验证起点排序 development 折，供严格前向选择使用。"""

    order = (
        rows.groupby("fold", sort=False)
        .agg(first_origin=("origin_time", "min"), train_end=("train_end", "first"))
        .reset_index()
        .sort_values(["first_origin", "train_end", "fold"], kind="stable")
    )
    return order["fold"].astype(str).tolist()


def _forward_fold_differences(
    comparison: dict[str, object], fold_order: list[str]
) -> dict[str, float]:
    """按真实时间顺序提取完整候选相对基线的逐折差值。"""

    by_fold = comparison["pairwise"]["by_fold"]
    return {
        fold: float(by_fold[fold]["difference"])
        for fold in fold_order
        if fold in by_fold
    }


def build_a58_forward_disagreement_specialist(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    specialist_column: str,
) -> ForwardDisagreementSpecialistResult:
    """构造 A58 固定 q80 的严格前向 disagreement specialist。

    每个 held fold 的同一 horizon 阈值只使用此前 development folds 且
    ``origin_time <= train_end`` 的 ``|A51-RichGas|``。q80、最小历史长度、
    候选范围和验收门槛均在模块常量中冻结；此函数不提供阈值或权重搜索入口。
    """

    work = _validate_oof_rows(
        rows,
        baseline_column=baseline_column,
        specialist_column=specialist_column,
        require_train_end=True,
    )
    work["fold"] = work["fold"].astype(str)
    work = work.loc[work["fold"].ne("blind")].copy()
    if work.empty:
        raise ValueError("A58 development OOF 没有可评分行")
    work = work.sort_values(["origin_time", "target", "horizon", "fold"]).reset_index(
        drop=True
    )
    work["a58_route_eligible"] = _eligible_long_g1(work)
    if not work["a58_route_eligible"].any():
        raise ValueError("A58 没有 generator_1 的 75/90/105/120 分钟单元")
    work["a58_abs_disagreement"] = (
        work[specialist_column] - work[baseline_column]
    ).abs()
    work[A58_RAW_COLUMN] = work[baseline_column].to_numpy(dtype=float)
    work["a58_switched"] = False

    fold_order = _chronological_development_folds(work)
    trace_records: list[dict[str, object]] = []
    for position, fold in enumerate(fold_order):
        held_mask = work["fold"].eq(fold)
        held = work.loc[held_mask]
        train_end = _fold_train_ends(held)[fold]
        prior_folds = fold_order[:position]
        for horizon in sorted(G1_LONG_HORIZONS):
            held_horizon_mask = held_mask & work["a58_route_eligible"] & work["horizon"].eq(
                horizon
            )
            history_mask = (
                work["fold"].isin(prior_folds)
                & work["a58_route_eligible"]
                & work["horizon"].eq(horizon)
                & work["origin_time"].le(train_end)
            )
            history_rows = work.loc[history_mask, ["fold", "origin_time", "a58_abs_disagreement"]]
            if history_rows["fold"].eq(fold).any():
                raise RuntimeError("A58 阈值历史混入当前 held fold")
            if (history_rows["origin_time"] > train_end).any():
                raise RuntimeError("A58 阈值历史越过 outer-fold train_end")
            finite_history = history_rows.loc[
                np.isfinite(history_rows["a58_abs_disagreement"].to_numpy(dtype=float))
            ]
            ready = len(finite_history) >= A58_MIN_HISTORY_ROWS
            q80 = (
                float(
                    np.quantile(
                        finite_history["a58_abs_disagreement"].to_numpy(dtype=float),
                        A58_DISAGREEMENT_QUANTILE,
                    )
                )
                if ready
                else None
            )
            switch_mask = held_horizon_mask & False
            if q80 is not None:
                switch_mask = held_horizon_mask & work["a58_abs_disagreement"].ge(q80)
                work.loc[switch_mask, A58_RAW_COLUMN] = work.loc[
                    switch_mask, specialist_column
                ].to_numpy(dtype=float)
                work.loc[switch_mask, "a58_switched"] = True
            held_rows = int(held_horizon_mask.sum())
            used_history_folds = finite_history["fold"].drop_duplicates().astype(str).tolist()
            trace_records.append(
                {
                    "fold": fold,
                    "horizon": int(horizon),
                    "train_end": train_end,
                    "history_rows": int(len(finite_history)),
                    "history_max_time": (
                        pd.NaT
                        if finite_history.empty
                        else pd.Timestamp(finite_history["origin_time"].max())
                    ),
                    "history_folds": ",".join(used_history_folds),
                    "history_fold_count": int(len(used_history_folds)),
                    "q80": q80,
                    "status": "ready" if ready else "insufficient_history",
                    "fallback": "none" if ready else "rich_gas",
                    "held_rows": held_rows,
                    "switch_rows": int(switch_mask.sum()),
                    "coverage": (
                        float(switch_mask.sum() / held_rows) if held_rows else 0.0
                    ),
                    "held_fold_used_for_threshold": False,
                    "future_rows_used_for_threshold": False,
                    "threshold_used_labels": False,
                }
            )

    raw_changed = ~np.isclose(
        work[A58_RAW_COLUMN].to_numpy(dtype=float),
        work[baseline_column].to_numpy(dtype=float),
    )
    noneligible_raw_changes = int(
        (raw_changed & ~work["a58_route_eligible"].to_numpy(dtype=bool)).sum()
    )
    if noneligible_raw_changes:
        raise RuntimeError("A58 原始路由修改了 generator_1 长步长以外的单元")
    projected = project_long_candidate(work, A58_RAW_COLUMN, output_column=A58_PREDICTION_COLUMN)
    comparison = compare_research_candidate(
        projected,
        A58_PREDICTION_COLUMN,
        baseline_column,
        scope="development",
    )
    eligible = projected.loc[projected["a58_route_eligible"]].copy()
    g1_long_comparison = compare_research_candidate(
        eligible,
        A58_PREDICTION_COLUMN,
        baseline_column,
        scope="development",
    )
    fold_differences = _forward_fold_differences(comparison, fold_order)
    g1_long_fold_differences = _forward_fold_differences(g1_long_comparison, fold_order)
    recent_folds = list(fold_differences)[-5:]
    recent5_wins = int(sum(fold_differences[fold] < 0.0 for fold in recent_folds))
    pooled_improvement_pp = -float(comparison["pooled_difference"]) * 100.0
    g1_long_improvement_pp = -float(g1_long_comparison["pooled_difference"]) * 100.0
    worst_fold_regression_pp = max(fold_differences.values(), default=0.0) * 100.0
    g1_long_worst_fold_regression_pp = (
        max(g1_long_fold_differences.values(), default=0.0) * 100.0
    )
    acceptance = {
        "pooled_improvement_at_least_0_005pp": bool(
            pooled_improvement_pp >= A58_MIN_POOLED_IMPROVEMENT_PP
        ),
        "recent5_at_least_3_of_5": bool(recent5_wins >= A58_MIN_RECENT5_WINS),
        "g1_long_improves": bool(g1_long_improvement_pp > 0.0),
        "no_extreme_fold_regression": bool(
            worst_fold_regression_pp <= A58_MAX_FOLD_REGRESSION_PP
            and g1_long_worst_fold_regression_pp <= A58_MAX_FOLD_REGRESSION_PP
        ),
    }
    blind_eligible = bool(all(acceptance.values()))
    trace = pd.DataFrame(trace_records).sort_values(["fold", "horizon"], kind="stable")
    report = {
        "stage": "A58_strict_forward_disagreement_specialist",
        "scope": "development",
        "baseline_column": baseline_column,
        "specialist_column": specialist_column,
        "raw_column": A58_RAW_COLUMN,
        "prediction_column": A58_PREDICTION_COLUMN,
        "target_scope": "generator_1",
        "eligible_horizons": sorted(G1_LONG_HORIZONS),
        "rows": int(len(projected)),
        "eligible_rows": int(projected["a58_route_eligible"].sum()),
        "switched_rows": int(projected["a58_switched"].sum()),
        "eligible_switch_coverage": float(
            projected["a58_switched"].sum() / projected["a58_route_eligible"].sum()
        ),
        "frozen_rule": {
            "disagreement": f"abs({specialist_column} - {baseline_column})",
            "quantile": "q80",
            "quantile_probability": A58_DISAGREEMENT_QUANTILE,
            "minimum_history_rows": A58_MIN_HISTORY_ROWS,
            "switch_rule": "同一 g1-long horizon 中 D >= 前向 q80 时使用 A51，否则 RichGas",
            "blend_weight_search": False,
            "threshold_grid_search": False,
            "classifier_or_soft_gate": False,
        },
        "threshold_trace_summary": {
            "records": int(len(trace)),
            "ready_records": int(trace["status"].eq("ready").sum()),
            "fallback_records": int(trace["status"].eq("insufficient_history").sum()),
            "history_max_time_after_train_end": int(
                (
                    pd.to_datetime(trace["history_max_time"], errors="coerce")
                    > pd.to_datetime(trace["train_end"])
                ).sum()
            ),
        },
        "raw_route_audit": {
            "raw_changed_cells": int(raw_changed.sum()),
            "noneligible_raw_changed_cells": noneligible_raw_changes,
            "selector_only_changes_g1_long": bool(noneligible_raw_changes == 0),
            "capacity_projection_modified_cells": int(
                (~np.isclose(
                    projected[A58_RAW_COLUMN].to_numpy(dtype=float),
                    projected[A58_PREDICTION_COLUMN].to_numpy(dtype=float),
                )).sum()
            ),
        },
        "comparison": comparison,
        "g1_long_comparison": g1_long_comparison,
        "chronological_fold_differences": fold_differences,
        "chronological_g1_long_fold_differences": g1_long_fold_differences,
        "recent5_folds": recent_folds,
        "recent5_wins": recent5_wins,
        "acceptance": {
            **acceptance,
            "pooled_improvement_pp": pooled_improvement_pp,
            "g1_long_improvement_pp": g1_long_improvement_pp,
            "worst_fold_regression_pp": worst_fold_regression_pp,
            "g1_long_worst_fold_regression_pp": g1_long_worst_fold_regression_pp,
            "max_allowed_fold_regression_pp": A58_MAX_FOLD_REGRESSION_PP,
        },
        "router_series_status": (
            "A58_BLIND_ELIGIBLE" if blind_eligible else "STOP_ROUTER_SERIES"
        ),
        "blind_eligible": blind_eligible,
        "formal_candidate": False,
        "blind_used": False,
        "threshold_used_labels": False,
        "held_fold_used_for_threshold": False,
        "strict_oof_contract": {
            "threshold_history": (
                "仅此前 development folds、同一 generator_1 长步长且 "
                "origin_time <= 当前 outer_fold.train_end 的历史 OOF 预测"
            ),
            "threshold_used_labels": False,
            "held_fold_used_for_threshold": False,
            "future_rows_used_for_threshold": False,
            "blind_used": False,
            "all_non_g1_long_raw_cells_keep_rich_gas": True,
            "capacity_projection": "原始路由后统一使用生产一致的容量投影",
        },
    }
    return ForwardDisagreementSpecialistResult(
        rows=projected.reset_index(drop=True),
        threshold_trace=trace.reset_index(drop=True),
        report=report,
    )
