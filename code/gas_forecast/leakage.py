"""多起点、多扰动方式的因果特征与预测审计。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]
Predictor = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
OriginPredictor = Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame | np.ndarray]


_PREDICTION_MUTATIONS = ("extreme", "shuffle", "null", "single_field", "delete_future")
_ALL_NUMERIC_COLUMNS = "__all_numeric_columns__"


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


def _prediction_for_origin(
    predictor: OriginPredictor,
    frame: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """规范化模型级审计预测，拒绝非 1x16 的不完整回执。"""

    raw = predictor(frame.copy(deep=True), origin)
    if isinstance(raw, pd.DataFrame):
        prediction = raw.copy()
    else:
        values = np.asarray(raw)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        prediction = pd.DataFrame(values)
    if prediction.shape != (1, 16):
        raise ValueError(
            "模型级泄漏审计要求 predictor(frame, origin) 返回 1x16 预测，"
            f"实际为 {prediction.shape}"
        )
    return prediction


def _prediction_mutations(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    rng: np.random.Generator,
) -> list[tuple[str, str, pd.DataFrame]]:
    """生成未来生产观测扰动；单字段路径覆盖每个数值生产列。"""

    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    if not numeric_columns:
        raise ValueError("模型级泄漏审计需要至少一个数值生产列")
    future = frame.index > origin
    if not future.any():
        raise ValueError(f"审计起点 {origin} 后没有可扰动的生产观测")

    def numeric_copy() -> pd.DataFrame:
        output = frame.copy(deep=True)
        output[numeric_columns] = output[numeric_columns].astype(float)
        return output

    extreme = numeric_copy()
    extreme.loc[future, numeric_columns] = -999_999.0

    shuffled = numeric_copy()
    future_positions = np.flatnonzero(future)
    if len(future_positions) > 1:
        for column in numeric_columns:
            permutation = rng.permutation(len(future_positions))
            if np.array_equal(permutation, np.arange(len(future_positions))):
                permutation = np.roll(permutation, 1)
            values = shuffled.loc[future, column].to_numpy(copy=True)
            shuffled.loc[future, column] = values[permutation]

    null = numeric_copy()
    null.loc[future, numeric_columns] = np.nan

    mutations: list[tuple[str, str, pd.DataFrame]] = [
        ("extreme", _ALL_NUMERIC_COLUMNS, extreme),
        ("shuffle", _ALL_NUMERIC_COLUMNS, shuffled),
        ("null", _ALL_NUMERIC_COLUMNS, null),
    ]
    for column in numeric_columns:
        single_field = numeric_copy()
        single_field.loc[future, column] = 999_999.0
        mutations.append(("single_field", str(column), single_field))
    mutations.append(("delete_future", _ALL_NUMERIC_COLUMNS, frame.loc[:origin].copy()))
    return mutations


def _prediction_difference(
    baseline: pd.DataFrame,
    changed: pd.DataFrame,
) -> tuple[bool, list[str], float]:
    """严格逐元素比较两个预测，并返回发生变化的预测字段和最大差值。"""

    if not baseline.columns.equals(changed.columns):
        return False, ["__prediction_schema__"], float("inf")
    before = baseline.to_numpy(dtype=float)
    after = changed.to_numpy(dtype=float)
    equal = (before == after) | (np.isnan(before) & np.isnan(after))
    if equal.all():
        return True, [], 0.0

    changed_columns = [
        str(column) for column, is_equal in zip(baseline.columns, equal[0], strict=True) if not is_equal
    ]
    difference = np.abs(before - after)
    nan_mismatch = np.isnan(before) ^ np.isnan(after)
    if np.isinf(difference).any() or nan_mismatch.any():
        max_abs_diff = float("inf")
    else:
        finite_difference = difference[np.isfinite(difference)]
        max_abs_diff = float(finite_difference.max()) if finite_difference.size else float("inf")
    return False, changed_columns, max_abs_diff


def audit_origin_predictor(
    frame: pd.DataFrame,
    predictor: OriginPredictor,
    origins: int = 50,
    *,
    random_state: int = 20250731,
) -> dict[str, object]:
    """审计模型在每个起点是否读取未来生产观测。

    ``predictor`` 接收完整输入和预测起点。审计不会预先截断基线输入，因此任何
    读取 ``origin`` 之后发电量、煤气或其他数值生产列的实现都会被扰动捕获。
    """

    if origins < 1:
        raise ValueError("模型级泄漏审计至少需要一个起点")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("模型级泄漏审计输入必须使用 DatetimeIndex")
    if frame.empty or frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError("模型级泄漏审计时间轴必须非空、唯一且递增")
    if len(frame) < 3:
        raise ValueError("模型级泄漏审计至少需要3行")
    selected = select_audit_origins(frame, origins)
    failures: list[dict[str, object]] = []
    cases_checked = 0
    numeric_columns = [str(column) for column in frame.select_dtypes(include=[np.number]).columns]

    for position, origin in enumerate(selected):
        try:
            baseline = _prediction_for_origin(predictor, frame, origin)
        except Exception as exc:  # noqa: BLE001 - 审计需把模型契约错误显式暴露给调用方。
            raise RuntimeError(f"模型级泄漏审计无法生成基线预测，origin={origin}") from exc
        rng = np.random.default_rng(random_state + position)
        for mutation, column, changed_frame in _prediction_mutations(frame, origin, rng):
            cases_checked += 1
            try:
                changed = _prediction_for_origin(predictor, changed_frame, origin)
                unchanged, changed_columns, max_abs_diff = _prediction_difference(baseline, changed)
            except Exception as exc:  # noqa: BLE001 - 未来数据删改后不能预测即不满足生产契约。
                failures.append(
                    {
                        "origin": str(origin),
                        "mutation": mutation,
                        "method": mutation,
                        "column": column,
                        "changed_prediction_columns": ["__prediction_error__"],
                        "max_abs_diff": float("inf"),
                        "max_diff": float("inf"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if not unchanged:
                failures.append(
                    {
                        "origin": str(origin),
                        "mutation": mutation,
                        "method": mutation,
                        "column": column,
                        "changed_prediction_columns": changed_columns,
                        "max_abs_diff": max_abs_diff,
                        "max_diff": max_abs_diff,
                    }
                )

    return {
        "passed": not failures,
        "origins": int(len(selected)),
        "methods": list(_PREDICTION_MUTATIONS),
        "numeric_production_columns": numeric_columns,
        "cases_checked": int(cases_checked),
        "failures": failures,
    }


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
