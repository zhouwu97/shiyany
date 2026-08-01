"""多起点、多扰动方式的因果特征与预测审计。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]
Predictor = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def select_audit_origins(frame: pd.DataFrame, count: int = 50) -> pd.DatetimeIndex:
    """覆盖均匀时间点，并补入缺失、零值和大跳变附近起点。"""

    if len(frame) < 3:
        raise ValueError("泄漏审计至少需要3行")
    numeric = frame.select_dtypes(include=[np.number])
    positions = set(np.linspace(1, len(frame) - 2, min(count, len(frame) - 2), dtype=int))
    if not numeric.empty:
        missing_positions = np.flatnonzero(numeric.isna().any(axis=1).to_numpy())
        zero_positions = np.flatnonzero(numeric.eq(0).any(axis=1).to_numpy())
        jump = numeric.diff().abs().median(axis=1)
        jump_positions = np.flatnonzero(jump.ge(jump.quantile(0.99)).to_numpy())
        for values in (missing_positions, zero_positions, jump_positions):
            positions.update(int(value) for value in values[: max(1, count // 10)])
    valid = sorted(position for position in positions if 0 < position < len(frame) - 1)
    if len(valid) > count:
        selected = np.linspace(0, len(valid) - 1, count, dtype=int)
        valid = [valid[position] for position in selected]
    return frame.index[valid]


def _perturb(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    method: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if method == "delete_future":
        return frame.loc[:origin].copy()
    output = frame.copy()
    future = output.index > origin
    numeric_columns = list(output.select_dtypes(include=[np.number]).columns)
    output[numeric_columns] = output[numeric_columns].astype(float)
    if method == "extreme":
        output.loc[future, numeric_columns] = -999_999.0
    elif method == "shuffle":
        values = output.loc[future, numeric_columns].to_numpy(copy=True)
        output.loc[future, numeric_columns] = values[rng.permutation(len(values))]
    elif method == "null":
        output.loc[future, numeric_columns] = np.nan
    elif method == "single_field":
        if numeric_columns:
            output.loc[future, numeric_columns[0]] = 999_999.0
    else:
        raise ValueError(f"未知扰动方式: {method}")
    return output


def audit_future_perturbations(
    frame: pd.DataFrame,
    feature_builder: FeatureBuilder,
    *,
    predictor: Predictor | None = None,
    origins: int = 50,
    random_state: int = 20250731,
    n_jobs: int = 1,
) -> dict[str, object]:
    """验证所有起点的原点特征和可选最终预测不受未来生产值影响。"""

    baseline_features = feature_builder(frame)
    selected = select_audit_origins(frame, origins)
    methods = ("extreme", "shuffle", "null", "single_field", "delete_future")
    def audit_origin(position: int, origin: pd.Timestamp) -> tuple[int, list[dict[str, object]]]:
        rng = np.random.default_rng(random_state + position)
        origin_failures: list[dict[str, object]] = []
        baseline_row = baseline_features.loc[[origin]]
        baseline_prediction = (
            predictor(baseline_row, frame.loc[[origin]]) if predictor is not None else None
        )
        for method in methods:
            changed_frame = _perturb(frame, origin, method, rng)
            changed_features = feature_builder(changed_frame).loc[[origin]]
            left, right = baseline_row.align(changed_features, axis=1)
            feature_equal = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
            ).all()
            prediction_equal = True
            if predictor is not None:
                changed_prediction = predictor(changed_features, changed_frame.loc[[origin]])
                prediction_equal = np.isclose(
                    baseline_prediction.to_numpy(dtype=float),
                    changed_prediction.to_numpy(dtype=float),
                    equal_nan=True,
                ).all()
            if not feature_equal or not prediction_equal:
                origin_failures.append(
                    {
                        "origin": str(origin),
                        "method": method,
                        "features_unchanged": bool(feature_equal),
                        "predictions_unchanged": bool(prediction_equal),
                    }
                )
        return len(methods), origin_failures

    if n_jobs == 1:
        results = [audit_origin(position, origin) for position, origin in enumerate(selected)]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(audit_origin)(position, origin)
            for position, origin in enumerate(selected)
        )
    checked = sum(item[0] for item in results)
    failures = [failure for _, item in results for failure in item]
    return {
        "passed": not failures,
        "origins": int(len(selected)),
        "methods": list(methods),
        "cases_checked": checked,
        "jobs": n_jobs,
        "failures": failures,
    }
