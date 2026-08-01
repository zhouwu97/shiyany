"""仅依赖时间 OOF 预测的低自由度融合器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SimplexState:
    branch_names: tuple[str, ...]
    target_weights: np.ndarray
    horizon_weights: np.ndarray
    regularized_weights: np.ndarray
    regularization: float


@dataclass
class DynamicGateState:
    feature_columns: tuple[str, ...]
    models: list[Pipeline | None]
    constants: np.ndarray


def select_gate_feature_columns(columns: list[str], target: str) -> tuple[str, ...]:
    """显式登记且稳定排序门控字段，不再按构造顺序截取前 N 列。"""

    keywords = (
        target,
        "generator_rest",
        "generator_gas_total",
        "gas_switch",
        "dominant_gas",
        "gas_mix_entropy",
        "balance",
        "surplus",
        "holder",
        "outlier",
        "freeze",
        "time_",
        "hour",
        "minute",
        "price",
    )
    return tuple(sorted(column for column in columns if any(key in column for key in keywords)))


def _gate_matrix(
    features: pd.DataFrame,
    anchor: np.ndarray,
    branches: np.ndarray,
    columns: tuple[str, ...],
    step: int,
) -> pd.DataFrame:
    matrix = features.reindex(columns=columns).reset_index(drop=True).copy()
    corrections = branches[:, :, step] - anchor[:, None]
    for branch in range(branches.shape[1]):
        matrix[f"branch_{branch}_correction"] = corrections[:, branch]
    median = np.median(branches[:, :, step], axis=1)
    matrix["branch_disagreement_mad"] = np.median(
        np.abs(branches[:, :, step] - median[:, None]), axis=1
    )
    return matrix


def fit_dynamic_gate(
    features: pd.DataFrame,
    anchor: np.ndarray,
    branches: np.ndarray,
    actual: np.ndarray,
    base_prediction: np.ndarray,
    *,
    target: str,
    alpha: float = 40.0,
) -> DynamicGateState:
    """用 OOF 分支拟合连续 oracle coefficient 的低容量 Ridge 门控。"""

    columns = select_gate_feature_columns(list(features.columns), target)
    models: list[Pipeline | None] = []
    constants = np.zeros(actual.shape[1])
    for step in range(actual.shape[1]):
        correction = base_prediction[:, step] - anchor
        valid = np.abs(correction) > 1e-6
        oracle = np.zeros(len(anchor))
        oracle[valid] = (actual[valid, step] - anchor[valid]) / correction[valid]
        oracle = np.clip(oracle, 0.0, 1.0)
        constants[step] = float(np.mean(oracle))
        if float(np.std(oracle)) < 1e-6:
            models.append(None)
            continue
        matrix = _gate_matrix(features, anchor, branches, columns, step)
        sample_weight = np.clip(
            np.abs(correction) / np.maximum(np.abs(actual[:, step]), 1.0), 0.05, 1.0
        )
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(matrix, oracle, ridge__sample_weight=sample_weight)
        models.append(model)
    return DynamicGateState(columns, models, constants)


def apply_dynamic_gate(
    state: DynamicGateState,
    features: pd.DataFrame,
    anchor: np.ndarray,
    branches: np.ndarray,
    base_prediction: np.ndarray,
    *,
    gate_min: float = 0.0,
    gate_max: float = 1.0,
) -> np.ndarray:
    output = np.empty_like(base_prediction)
    for step, model in enumerate(state.models):
        if model is None:
            gate = np.full(len(anchor), state.constants[step])
        else:
            matrix = _gate_matrix(
                features, anchor, branches, state.feature_columns, step
            )
            gate = model.predict(matrix)
        gate = np.clip(gate, gate_min, gate_max)
        output[:, step] = anchor + gate * (base_prediction[:, step] - anchor)
    return output


def _solve(
    predictions: np.ndarray,
    actual: np.ndarray,
    *,
    reference: np.ndarray,
    regularization: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    branches = predictions.shape[1]
    denominator = np.maximum(np.abs(actual), epsilon)

    def objective(weights: np.ndarray) -> float:
        blended = predictions @ weights
        mape = np.mean(np.abs(actual - blended) / denominator)
        penalty = regularization * np.sum((weights - reference) ** 2)
        return float(mape + penalty)

    result = minimize(
        objective,
        reference,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * branches,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if result.success:
        weights = np.clip(result.x, 0.0, 1.0)
        return weights / weights.sum()
    scores = [objective(np.eye(branches)[index]) for index in range(branches)]
    return np.eye(branches)[int(np.argmin(scores))]


def fit_simplex_state(
    branch_predictions: np.ndarray,
    actual: np.ndarray,
    branch_names: tuple[str, ...],
    *,
    regularization: float = 0.002,
) -> SimplexState:
    """拟合目标级、步长级和向目标权重收缩的步长级 simplex。"""

    if branch_predictions.ndim != 3:
        raise ValueError("分支预测应为 样本×分支×步长")
    if actual.shape != (branch_predictions.shape[0], branch_predictions.shape[2]):
        raise ValueError("真实值形状与分支预测不一致")
    valid = np.isfinite(actual) & np.isfinite(branch_predictions).all(axis=1)
    if not valid.any(axis=0).all():
        raise ValueError("至少一个步长没有完整 OOF 预测")

    flat_prediction = np.concatenate(
        [branch_predictions[valid[:, step], :, step] for step in range(actual.shape[1])]
    )
    flat_actual = np.concatenate(
        [actual[valid[:, step], step] for step in range(actual.shape[1])]
    )
    uniform = np.full(branch_predictions.shape[1], 1.0 / branch_predictions.shape[1])
    target_weights = _solve(
        flat_prediction,
        flat_actual,
        reference=uniform,
        regularization=0.0,
    )
    horizon_weights = np.zeros((actual.shape[1], branch_predictions.shape[1]))
    regularized_weights = np.zeros_like(horizon_weights)
    for step in range(actual.shape[1]):
        prediction = branch_predictions[valid[:, step], :, step]
        truth = actual[valid[:, step], step]
        horizon_weights[step] = _solve(
            prediction, truth, reference=uniform, regularization=0.0
        )
        regularized_weights[step] = _solve(
            prediction,
            truth,
            reference=target_weights,
            regularization=regularization,
        )
    return SimplexState(
        branch_names=branch_names,
        target_weights=target_weights,
        horizon_weights=horizon_weights,
        regularized_weights=regularized_weights,
        regularization=regularization,
    )


def apply_simplex(branch_predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """应用目标固定权重或逐步长权重。"""

    if weights.ndim == 1:
        return np.einsum("nbh,b->nh", branch_predictions, weights)
    return np.einsum("nbh,hb->nh", branch_predictions, weights)
