"""低相关 OOF 融合与严格时间顺序 stacking。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.scoring import score_oof_long


def _validate_prediction_columns(rows: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(rows.columns))
    if missing:
        raise ValueError(f"融合输入缺少预测列: {missing}")
    if not columns:
        raise ValueError("融合至少需要一个预测列")


def residual_correlation(
    rows: pd.DataFrame,
    prediction_columns: tuple[str, ...],
) -> pd.DataFrame:
    """计算真实 OOF raw residual 的成对相关矩阵。"""

    _validate_prediction_columns(rows, prediction_columns)
    residuals = {}
    actual = pd.to_numeric(rows["actual"], errors="coerce")
    for column in prediction_columns:
        residuals[column] = actual - pd.to_numeric(rows[column], errors="coerce")
    matrix = pd.DataFrame(residuals).corr(min_periods=32)
    return matrix


def weighted_blend(
    rows: pd.DataFrame,
    prediction_columns: tuple[str, ...],
    weights: tuple[float, ...] | list[float],
    *,
    output_column: str = "blend_pred",
    active_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """生成固定权重融合；可限制在 specialist active 子空间。"""

    _validate_prediction_columns(rows, prediction_columns)
    values = np.asarray(weights, dtype=float)
    if values.shape != (len(prediction_columns),) or not np.isfinite(values).all():
        raise ValueError("融合权重长度或数值不合法")
    if np.any(values < 0) or values.sum() <= 0:
        raise ValueError("融合权重必须非负且总和为正")
    values = values / values.sum()
    output = rows.copy()
    candidate = np.column_stack([output[column].to_numpy(dtype=float) for column in prediction_columns])
    blended = candidate @ values
    if active_column is not None:
        if active_column not in output:
            raise ValueError(f"融合 active 字段不存在: {active_column}")
        active = output[active_column].astype(bool).to_numpy()
        blended = np.where(active, blended, candidate[:, 0])
    output[output_column] = blended
    report = {
        "prediction_columns": list(prediction_columns),
        "weights": values.tolist(),
        "active_column": active_column,
        "score": score_oof_long(output, output_column),
    }
    return output, report


def _fit_simplex_weights(x: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    """用加权最小二乘的非负投影拟合低容量融合权重。"""

    scale = 1.0 / np.maximum(np.abs(y), epsilon)
    weighted_x = x * scale[:, None]
    weighted_y = y * scale
    try:
        weights = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
    except np.linalg.LinAlgError:
        weights = np.ones(x.shape[1], dtype=float)
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
        weights = np.ones(x.shape[1], dtype=float)
    return weights / weights.sum()


def time_ordered_stack_oof(
    rows: pd.DataFrame,
    prediction_columns: tuple[str, ...],
    *,
    output_column: str = "stack_pred",
    epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """按外层折先后拟合 stacking 权重，禁止用 held-out 折学习自身权重。"""

    _validate_prediction_columns(rows, prediction_columns)
    if "fold" not in rows or "origin_time" not in rows:
        raise ValueError("time-ordered stack 需要 fold 和 origin_time")
    output = rows.copy()
    output["origin_time"] = pd.to_datetime(output["origin_time"])
    fold_order = (
        output.groupby("fold", sort=False)["origin_time"]
        .min()
        .sort_values()
        .index.tolist()
    )
    weights_by_fold: dict[str, list[float]] = {}
    predictions = np.full(len(output), np.nan, dtype=float)
    for position, fold in enumerate(fold_order):
        held_mask = output["fold"].eq(fold).to_numpy()
        if position == 0:
            weights = np.zeros(len(prediction_columns), dtype=float)
            weights[0] = 1.0
        else:
            train_mask = output["fold"].isin(fold_order[:position]).to_numpy()
            train = output.loc[train_mask]
            valid = np.isfinite(train["actual"].to_numpy(dtype=float))
            matrix = train.loc[:, list(prediction_columns)].to_numpy(dtype=float)
            valid &= np.isfinite(matrix).all(axis=1)
            if int(valid.sum()) < max(32, len(prediction_columns) * 4):
                weights = np.zeros(len(prediction_columns), dtype=float)
                weights[0] = 1.0
            else:
                weights = _fit_simplex_weights(
                    matrix[valid], train["actual"].to_numpy(dtype=float)[valid], epsilon
                )
        held_matrix = output.loc[held_mask, list(prediction_columns)].to_numpy(dtype=float)
        if not np.isfinite(held_matrix).all():
            raise ValueError(f"stack fold {fold} 的候选预测不完整")
        predictions[held_mask] = held_matrix @ weights
        weights_by_fold[str(fold)] = weights.tolist()
    output[output_column] = predictions
    return output, {
        "prediction_columns": list(prediction_columns),
        "weights_by_fold": weights_by_fold,
        "score": score_oof_long(output, output_column),
    }
