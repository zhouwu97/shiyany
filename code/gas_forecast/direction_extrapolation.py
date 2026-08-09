"""按目标和预测步长对两个冻结提交做方向外推。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from gas_forecast.scoring import competition_mape
from gas_forecast.submission import validate_submission_frame


SHORT_HORIZONS = frozenset({15, 30, 45, 60})
LONG_HORIZONS = frozenset({75, 90, 105, 120})
ALL_HORIZONS = SHORT_HORIZONS | LONG_HORIZONS


def _validate_multipliers(multipliers: Mapping[int, float]) -> dict[int, float]:
    normalized = {int(horizon): float(value) for horizon, value in multipliers.items()}
    if set(normalized) != ALL_HORIZONS:
        raise ValueError(f"方向外推倍率必须完整覆盖 {sorted(ALL_HORIZONS)}")
    if any(not np.isfinite(value) or value < 0.0 or value > 2.0 for value in normalized.values()):
        raise ValueError("方向外推倍率必须是 [0, 2] 内的有限数")
    return normalized


def extrapolate_submission_result(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    target: str,
    multipliers: Mapping[int, float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """沿 baseline 到 candidate 的方向按步长外推，其他目标保持 baseline。"""

    validate_submission_frame(baseline)
    validate_submission_frame(candidate)
    if list(baseline.columns) != list(candidate.columns):
        raise ValueError("方向外推的两个结果文件字段不一致")
    baseline_time = pd.to_datetime(baseline["datetime"])
    candidate_time = pd.to_datetime(candidate["datetime"])
    if not baseline_time.equals(candidate_time):
        raise ValueError("方向外推的两个结果文件时间戳不一致")

    normalized = _validate_multipliers(multipliers)
    output = baseline.copy()
    changed_by_horizon: dict[str, int] = {}
    for horizon in sorted(ALL_HORIZONS):
        column = f"{target}_t+{horizon}_pred"
        if column not in output:
            raise ValueError(f"方向外推目标字段不存在: {column}")
        baseline_values = baseline[column].to_numpy(dtype=float)
        candidate_values = candidate[column].to_numpy(dtype=float)
        output[column] = baseline_values + normalized[horizon] * (
            candidate_values - baseline_values
        )
        changed_by_horizon[f"t+{horizon}"] = int(
            np.count_nonzero(~np.isclose(output[column], baseline_values, rtol=0.0, atol=1e-12))
        )

    validation = validate_submission_frame(output)
    prediction_columns = [column for column in output.columns if column != "datetime"]
    unchanged_columns = [
        column for column in prediction_columns if not column.startswith(f"{target}_t+")
    ]
    unchanged_cells = int(
        np.count_nonzero(
            ~np.isclose(
                output[unchanged_columns].to_numpy(dtype=float),
                baseline[unchanged_columns].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        )
    )
    if unchanged_cells:
        raise RuntimeError("方向外推意外修改了非目标预测")
    return output, {
        "target": target,
        "multipliers": {f"t+{key}": value for key, value in sorted(normalized.items())},
        "changed_cells_vs_baseline": int(sum(changed_by_horizon.values())),
        "changed_cells_by_horizon": changed_by_horizon,
        "non_target_changed_cells": unchanged_cells,
        "validation": validation,
    }


def evaluate_direction_policy(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    candidate_column: str,
    target: str,
    multipliers: Mapping[int, float],
) -> dict[str, object]:
    """在现有严格 OOF 上评估冻结策略，不用 blind 选择倍率。"""

    required = {
        "fold",
        "target",
        "horizon",
        "actual",
        baseline_column,
        candidate_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"方向外推 OOF 缺少字段: {missing}")
    normalized = _validate_multipliers(multipliers)
    work = rows.copy()
    horizon = pd.to_numeric(work["horizon"], errors="raise").astype(int)
    unknown = sorted(set(horizon).difference(ALL_HORIZONS))
    if unknown:
        raise ValueError(f"方向外推 OOF 含未知步长: {unknown}")
    base = pd.to_numeric(work[baseline_column], errors="raise").to_numpy(dtype=float)
    branch = pd.to_numeric(work[candidate_column], errors="raise").to_numpy(dtype=float)
    weights = np.zeros(len(work), dtype=float)
    target_mask = work["target"].eq(target).to_numpy()
    weights[target_mask] = horizon[target_mask].map(normalized).to_numpy(dtype=float)
    policy_column = "direction_policy_pred"
    work[policy_column] = base + weights * (branch - base)

    def score(part: pd.DataFrame, column: str) -> float:
        return competition_mape(part["actual"], part[column])

    scopes: dict[str, object] = {}
    scope_rows = {
        "development": work.loc[work["fold"].ne("blind")],
        "blind": work.loc[work["fold"].eq("blind")],
        "full": work,
    }
    for name, part in scope_rows.items():
        if part.empty:
            continue
        baseline_mape = score(part, baseline_column)
        candidate_mape = score(part, candidate_column)
        policy_mape = score(part, policy_column)
        scopes[name] = {
            "rows": int(len(part)),
            "baseline_mape": baseline_mape,
            "candidate_mape": candidate_mape,
            "policy_mape": policy_mape,
            "policy_local_score": 100.0 * (1.0 - policy_mape),
            "difference_vs_baseline_pp": 100.0 * (policy_mape - baseline_mape),
            "difference_vs_candidate_pp": 100.0 * (policy_mape - candidate_mape),
        }

    development = scope_rows["development"]
    fold_differences = {
        str(fold): 100.0 * (score(part, policy_column) - score(part, baseline_column))
        for fold, part in development.groupby("fold", sort=True)
    }
    ordered_folds = sorted(fold_differences)
    recent_folds = ordered_folds[-5:]
    return {
        "policy_column": policy_column,
        "target": target,
        "multipliers": {f"t+{key}": value for key, value in sorted(normalized.items())},
        "scopes": scopes,
        "development_fold_differences_pp": fold_differences,
        "development_fold_wins": int(sum(value < 0.0 for value in fold_differences.values())),
        "development_fold_count": int(len(fold_differences)),
        "recent_5_folds": recent_folds,
        "recent_5_wins": int(sum(fold_differences[fold] < 0.0 for fold in recent_folds)),
        "worst_development_fold_regression_pp": float(max(fold_differences.values())),
        "selection_used_blind": False,
    }
