"""generator_1、generator_rest 与 generator_all 的结构协调。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from gas_forecast.scoring import competition_mape


SUMMING_MATRIX = np.asarray(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)


@dataclass(frozen=True)
class ReconciliationState:
    direct_bottom_weights: np.ndarray
    diagonal_variances: np.ndarray
    covariance: np.ndarray | None = None


def _reconciliation_projection(error_covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(error_covariance, dtype=float)
    inverse = np.linalg.pinv(covariance)
    middle = np.linalg.pinv(SUMMING_MATRIX.T @ inverse @ SUMMING_MATRIX)
    return SUMMING_MATRIX @ middle @ SUMMING_MATRIX.T @ inverse


def fit_reconciliation(
    actual: np.ndarray,
    base_predictions: np.ndarray,
    *,
    estimate_full_covariance: bool = False,
) -> ReconciliationState:
    """输入形状均为 样本×3目标×步长，目标顺序为 1/rest/all。"""

    if actual.shape != base_predictions.shape or actual.ndim != 3 or actual.shape[1] != 3:
        raise ValueError("协调输入必须同为 样本×3目标×步长")
    horizons = actual.shape[2]
    blend_weights = np.zeros(horizons)
    diagonal_variances = np.zeros((horizons, 3))
    covariance = np.zeros((horizons, 3, 3)) if estimate_full_covariance else None
    for step in range(horizons):
        direct = base_predictions[:, 2, step]
        bottom_up = base_predictions[:, 0, step] + base_predictions[:, 1, step]
        truth = actual[:, 2, step]

        def objective(weight: float) -> float:
            return competition_mape(truth, weight * direct + (1.0 - weight) * bottom_up)

        result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
        blend_weights[step] = float(result.x if result.success else 0.5)
        errors = actual[:, :, step] - base_predictions[:, :, step]
        diagonal_variances[step] = np.maximum(np.nanvar(errors, axis=0), 1e-6)
        if covariance is not None:
            raw = np.cov(errors, rowvar=False)
            shrinkage = 0.1 * np.trace(raw) / 3.0
            covariance[step] = raw + np.eye(3) * max(shrinkage, 1e-6)
    return ReconciliationState(blend_weights, diagonal_variances, covariance)


def reconcile_predictions(
    base_predictions: np.ndarray,
    state: ReconciliationState,
) -> dict[str, np.ndarray]:
    """输出 direct、bottom-up、固定融合、对角协调和可选 full-cov 协调。"""

    if base_predictions.ndim != 3 or base_predictions.shape[1] != 3:
        raise ValueError("基础预测必须为 样本×3目标×步长")
    direct = base_predictions[:, 2, :]
    bottom_up = base_predictions[:, 0, :] + base_predictions[:, 1, :]
    blended = state.direct_bottom_weights * direct + (
        1.0 - state.direct_bottom_weights
    ) * bottom_up
    diagonal = np.empty_like(base_predictions)
    full = np.empty_like(base_predictions) if state.covariance is not None else None
    for step in range(base_predictions.shape[2]):
        diagonal_projection = _reconciliation_projection(
            np.diag(state.diagonal_variances[step])
        )
        diagonal[:, :, step] = base_predictions[:, :, step] @ diagonal_projection.T
        if full is not None:
            projection = _reconciliation_projection(state.covariance[step])
            full[:, :, step] = base_predictions[:, :, step] @ projection.T
    output = {
        "direct_all": direct,
        "bottom_up_all": bottom_up,
        "blended_all": blended,
        "diagonal_reconciled": diagonal,
    }
    if full is not None:
        output["full_covariance_reconciled"] = full
    return output
