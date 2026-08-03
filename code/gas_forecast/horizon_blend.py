"""严格 OOF 的短长权重和四组 horizon 路由。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gas_forecast.aggressive import project_long_candidate
from gas_forecast.research import compare_research_candidate


SHORT_HORIZONS = frozenset({15, 30, 45, 60})
FOUR_HORIZON_BANDS: dict[str, frozenset[int]] = {
    "g1_15_30": frozenset({15, 30}),
    "g2_45_60": frozenset({45, 60}),
    "g3_75_90": frozenset({75, 90}),
    "g4_105_120": frozenset({105, 120}),
}


@dataclass(frozen=True)
class HorizonBlendResult:
    """短长权重或前向路由产生的 OOF 与统一比较报告。"""

    rows: pd.DataFrame
    report: dict[str, object]
    route_trace: list[dict[str, object]] = field(default_factory=list)


def _validate_weights(weights: tuple[float, ...]) -> None:
    if not weights:
        raise ValueError("horizon blend 权重不能为空")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("horizon blend 权重必须位于 [0, 1]")


def _validate_rows(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    branch_column: str,
    comparison_column: str,
) -> pd.DataFrame:
    required = {
        "fold",
        "origin_time",
        "target",
        "horizon",
        "actual",
        "persistence_pred",
        baseline_column,
        branch_column,
        comparison_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"horizon blend 输入缺少字段: {missing}")
    output = rows.copy()
    output["origin_time"] = pd.to_datetime(output["origin_time"], errors="coerce")
    if output["origin_time"].isna().any():
        raise ValueError("horizon blend 输入含非法 origin_time")
    keys = ["fold", "origin_time", "target", "horizon"]
    if output.duplicated(keys).any():
        raise ValueError("horizon blend 输入存在重复 fold×origin×target×horizon")
    numeric_columns = list(
        dict.fromkeys(
            ["actual", "persistence_pred", baseline_column, branch_column, comparison_column]
        )
    )
    numeric = output.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("horizon blend 输入含缺失/非有限真实值或预测")
    output.loc[:, numeric.columns] = numeric
    output["horizon"] = pd.to_numeric(output["horizon"], errors="raise").astype(int)
    return output


def _candidate_values(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    branch_column: str,
    short_weight: float,
    long_weight: float,
) -> np.ndarray:
    baseline = rows[baseline_column].to_numpy(dtype=float)
    branch = rows[branch_column].to_numpy(dtype=float)
    generator_1 = rows["target"].eq("generator_1").to_numpy()
    short = rows["horizon"].isin(SHORT_HORIZONS).to_numpy()
    weights = np.where(short, short_weight, long_weight)
    weights = np.where(generator_1, weights, 0.0)
    return baseline + weights * (branch - baseline)


def _weight_label(weight: float) -> str:
    return f"{int(round(weight * 100)):02d}"


def build_two_band_blend_grid(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    branch_column: str,
    comparison_column: str,
    short_weights: tuple[float, ...] = (0.10, 0.15, 0.20),
    long_weights: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35),
    scope: str,
) -> HorizonBlendResult:
    """比较低自由度的短长两段 blend；generator_all 保持现有基线。"""

    if scope not in {"screening", "development", "final"}:
        raise ValueError("horizon blend scope 无效")
    _validate_weights(short_weights)
    _validate_weights(long_weights)
    work = _validate_rows(
        rows,
        baseline_column=baseline_column,
        branch_column=branch_column,
        comparison_column=comparison_column,
    )
    candidates: list[str] = []
    for short_weight in short_weights:
        for long_weight in long_weights:
            raw_column = (
                f"rich_short{_weight_label(short_weight)}_long{_weight_label(long_weight)}_raw_pred"
            )
            output_column = (
                f"rich_short{_weight_label(short_weight)}_long{_weight_label(long_weight)}_pred"
            )
            work[raw_column] = _candidate_values(
                work,
                baseline_column=baseline_column,
                branch_column=branch_column,
                short_weight=short_weight,
                long_weight=long_weight,
            )
            work = project_long_candidate(work, raw_column, output_column=output_column)
            candidates.append(output_column)
    reports = {
        column: compare_research_candidate(work, column, comparison_column, scope=scope)
        for column in candidates
    }
    return HorizonBlendResult(
        rows=work,
        report={
            "scope": scope,
            "baseline_column": baseline_column,
            "branch_column": branch_column,
            "comparison_column": comparison_column,
            "short_weights": list(short_weights),
            "long_weights": list(long_weights),
            "models": reports,
            "strict_oof_contract": "权重网格不读取 blind，也不按单步长自由搜索",
        },
    )


def _fold_order(rows: pd.DataFrame) -> list[str]:
    return (
        rows.groupby("fold", sort=False)["origin_time"]
        .min()
        .sort_values()
        .index.astype(str)
        .tolist()
    )


def _choose_weight(
    history: pd.DataFrame,
    weights: tuple[float, ...],
    *,
    baseline_column: str,
    branch_column: str,
) -> tuple[float, dict[str, float]]:
    """在历史 OOF 上选择固定候选中的最低 MAPE 权重。"""

    actual = history["actual"].to_numpy(dtype=float)
    baseline = history[baseline_column].to_numpy(dtype=float)
    branch = history[branch_column].to_numpy(dtype=float)
    denominator = np.maximum(np.abs(actual), 1e-6)
    scores = {
        _weight_label(weight): float(
            np.mean(np.abs(actual - (baseline + weight * (branch - baseline))) / denominator)
        )
        for weight in weights
    }
    selected = min(weights, key=lambda weight: (scores[_weight_label(weight)], weight))
    return selected, scores


def time_ordered_four_band_router(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    branch_column: str,
    comparison_column: str,
    short_weight: float,
    long_weight: float,
    scope: str,
    min_history_rows: int = 128,
) -> HorizonBlendResult:
    """按外层折时间前向选择四个 horizon 组的 Champion/branch/blend。"""

    if scope not in {"screening", "development", "final"}:
        raise ValueError("horizon router scope 无效")
    _validate_weights((short_weight, long_weight))
    if min_history_rows < 1:
        raise ValueError("min_history_rows 必须为正数")
    work = _validate_rows(
        rows,
        baseline_column=baseline_column,
        branch_column=branch_column,
        comparison_column=comparison_column,
    )
    if "train_end" not in work:
        raise ValueError("四组路由需要每折的严格 train_end 作为历史标签边界")
    work["train_end"] = pd.to_datetime(work["train_end"], errors="coerce")
    if work["train_end"].isna().any():
        raise ValueError("四组路由输入含非法 train_end")
    if scope != "final":
        work = work.loc[work["fold"].ne("blind")].copy()
    folds = _fold_order(work)
    raw_column = "rich_four_band_route_raw_pred"
    output_column = "rich_four_band_route_pred"
    work[raw_column] = work[baseline_column].to_numpy(dtype=float)
    route_trace: list[dict[str, object]] = []
    for position, fold in enumerate(folds):
        held = work["fold"].eq(fold)
        held_rows = work.loc[held]
        train_end = held_rows["train_end"].min()
        history = work.loc[
            work["target"].eq("generator_1") & work["origin_time"].le(train_end)
        ]
        for band, horizons in FOUR_HORIZON_BANDS.items():
            current_weight = short_weight if max(horizons) <= 60 else long_weight
            choices = (0.0, current_weight, 1.0)
            band_history = history.loc[history["horizon"].isin(horizons)]
            fallback = len(band_history) < min_history_rows
            if fallback:
                selected, scores = 0.0, {}
            else:
                selected, scores = _choose_weight(
                    band_history,
                    choices,
                    baseline_column=baseline_column,
                    branch_column=branch_column,
                )
            target_mask = held & work["target"].eq("generator_1") & work["horizon"].isin(horizons)
            baseline = work.loc[target_mask, baseline_column].to_numpy(dtype=float)
            branch = work.loc[target_mask, branch_column].to_numpy(dtype=float)
            work.loc[target_mask, raw_column] = baseline + selected * (branch - baseline)
            route_trace.append(
                {
                    "fold": fold,
                    "position": position,
                    "band": band,
                    "horizons": sorted(horizons),
                    "history_rows": int(len(band_history)),
                    "selected_weight": float(selected),
                    "scores": scores,
                    "fallback": fallback,
                }
            )
    projected = project_long_candidate(work, raw_column, output_column=output_column)
    report = compare_research_candidate(projected, output_column, comparison_column, scope=scope)
    return HorizonBlendResult(
        rows=projected,
        report={
            "scope": scope,
            "baseline_column": baseline_column,
            "branch_column": branch_column,
            "comparison_column": comparison_column,
            "short_weight": short_weight,
            "long_weight": long_weight,
            "models": {output_column: report},
            "strict_oof_contract": (
                "每个 fold/band 只使用 origin_time <= train_end 的历史 OOF 标签选择候选权重"
            ),
        },
        route_trace=route_trace,
    )
