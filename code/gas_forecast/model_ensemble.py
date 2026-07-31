"""煤气增强残差集成、V2.5 连续门控与 V3 状态事件模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gas_forecast.config import ForecastConfig
from gas_forecast.model_v1 import RidgeDeltaForecaster, make_ridge_pipeline
from gas_forecast.targets import target_columns


BRANCH_NAMES = ("persistence", "ridge", "recent", "gas", "lgb_residual")


@dataclass
class BranchModels:
    full_ridge: Pipeline
    recent_ridge: Pipeline
    gas_ridge: Pipeline
    residual_models: list[LGBMRegressor]
    delta_lower: np.ndarray
    delta_upper: np.ndarray


@dataclass
class EventClassifier:
    model: Pipeline | None
    constant_class: int | None = None


@dataclass
class EventState:
    gmm: GaussianMixture
    feature_columns: list[str]
    threshold: float
    classifiers: list[EventClassifier]


@dataclass
class GateState:
    model: Pipeline | None
    constant_probability: float | None = None


@dataclass
class TargetEnsembleState:
    branches: BranchModels
    blend_weights: np.ndarray
    correction_weights: np.ndarray
    rest_gmm: GaussianMixture | None
    event: EventState | None
    gates: list[GateState]
    gate_feature_columns: list[str]
    uncertainty_low: np.ndarray
    uncertainty_high: np.ndarray


def _select_gas_features(columns: list[str]) -> list[str]:
    keywords = (
        "gas",
        "furnace",
        "heater",
        "user",
        "holder",
        "surplus",
        "price",
        "time_",
        "hour",
        "month",
        "minute",
    )
    selected = [column for column in columns if any(key in column for key in keywords)]
    if not selected:
        raise ValueError("未找到煤气分支可用特征")
    return selected


def _select_gate_features(columns: list[str], target: str) -> list[str]:
    keywords = (
        target,
        "generator_rest",
        "generator_gas_total",
        "gas_switch",
        "dominant_gas",
        "gas_mix_entropy",
        "coke_down",
        "surplus",
        "holder_2",
        "time_",
        "hour",
        "minute",
        "price",
    )
    return [column for column in columns if any(key in column for key in keywords)][:72]


def _fit_lgb_residuals(
    features: pd.DataFrame,
    residuals: np.ndarray,
    config: ForecastConfig,
) -> list[LGBMRegressor]:
    models: list[LGBMRegressor] = []
    for step in range(residuals.shape[1]):
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=config.model.lgb_n_estimators,
            learning_rate=0.03,
            num_leaves=config.model.lgb_num_leaves,
            max_depth=config.model.lgb_max_depth,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=config.model.random_state + step,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(features, residuals[:, step])
        models.append(model)
    return models


def _fit_simplex_weights(branches: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """逐步长拟合非负且和为 1 的权重，再沿步长平滑。"""

    branch_count = branches.shape[1]
    horizon_count = branches.shape[2]
    weights = np.zeros((branch_count, horizon_count), dtype=float)
    denominator = np.maximum(np.abs(actual), 1e-6)
    bounds = [(0.0, 1.0)] * branch_count
    constraint = {"type": "eq", "fun": lambda value: value.sum() - 1.0}

    for step in range(horizon_count):
        prediction_matrix = branches[:, :, step]

        def objective(value: np.ndarray) -> float:
            predicted = prediction_matrix @ value
            return float(np.mean(np.abs(actual[:, step] - predicted) / denominator[:, step]))

        result = minimize(
            objective,
            np.full(branch_count, 1.0 / branch_count),
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if result.success:
            weights[:, step] = result.x
        else:
            single_scores = [objective(np.eye(branch_count)[branch]) for branch in range(branch_count)]
            weights[int(np.argmin(single_scores)), step] = 1.0

    smoothed = pd.DataFrame(weights.T).rolling(3, center=True, min_periods=1).mean().to_numpy().T
    return smoothed / smoothed.sum(axis=0, keepdims=True)


def _robust_disagreement(branches: np.ndarray) -> np.ndarray:
    """返回样本×步长的分支预测中位绝对偏差。"""

    median = np.median(branches, axis=1, keepdims=True)
    return np.median(np.abs(branches - median), axis=1)


def _fit_best_gmm(
    values: np.ndarray,
    max_components: int,
    random_state: int,
    components: int | None = None,
) -> GaussianMixture:
    finite = values[np.isfinite(values)].reshape(-1, 1)
    if len(finite) < 100:
        raise ValueError("状态模型有效样本不足 100 行")
    candidates = [components] if components is not None else list(range(2, max_components + 1))
    fitted = []
    for count in candidates:
        model = GaussianMixture(
            n_components=count,
            covariance_type="diag",
            reg_covar=1e-3,
            random_state=random_state,
        ).fit(finite)
        fitted.append((model.bic(finite), model))
    return min(fitted, key=lambda item: item[0])[1]


def _state_matrix(gmm: GaussianMixture, values: np.ndarray, prefix: str) -> pd.DataFrame:
    probabilities = gmm.predict_proba(values[:, None])
    distances = np.abs(values[:, None] - gmm.means_.reshape(1, -1))
    output: dict[str, np.ndarray] = {}
    for state in range(probabilities.shape[1]):
        output[f"{prefix}_probability_{state}"] = probabilities[:, state]
        output[f"{prefix}_distance_{state}"] = distances[:, state]
    return pd.DataFrame(output)


def _event_candidates(target: str) -> np.ndarray:
    return np.asarray([5, 8, 10, 12, 15] if target == "generator_1" else [8, 12, 15, 20, 25])


class GasAwareEnsembleForecaster(RidgeDeltaForecaster):
    """V2 固定集成、V2.5 低参数门控和 V3 事件增强门控。"""

    def __init__(self, version: str = "v2", config: ForecastConfig | None = None) -> None:
        if version not in {"v2", "v25", "v3"}:
            raise ValueError("增强模型版本只能是 v2、v25 或 v3")
        super().__init__(config)
        self.version = version
        self.ensemble_states_: dict[str, TargetEnsembleState] = {}
        self.gas_feature_columns_: list[str] = []

    def _fit_branches(self, x: pd.DataFrame, y: pd.DataFrame) -> BranchModels:
        full_ridge = make_ridge_pipeline(self.config.model.ridge_alpha)
        full_ridge.fit(x, y)
        full_delta = full_ridge.predict(x)

        recent_start = x.index.max() - pd.Timedelta(days=self.config.model.recent_days)
        recent_mask = x.index >= recent_start
        if int(recent_mask.sum()) < 200:
            recent_mask = np.ones(len(x), dtype=bool)
        recent_ridge = make_ridge_pipeline(self.config.model.ridge_alpha)
        recent_ridge.fit(x.loc[recent_mask], y.loc[recent_mask])

        gas_ridge = make_ridge_pipeline(self.config.model.ridge_alpha)
        gas_ridge.fit(x[self.gas_feature_columns_], y)
        residual_models = _fit_lgb_residuals(x, y.to_numpy() - full_delta, self.config)
        return BranchModels(
            full_ridge=full_ridge,
            recent_ridge=recent_ridge,
            gas_ridge=gas_ridge,
            residual_models=residual_models,
            delta_lower=y.quantile(self.config.model.lower_quantile).to_numpy(),
            delta_upper=y.quantile(self.config.model.upper_quantile).to_numpy(),
        )

    def _predict_branches(
        self,
        models: BranchModels,
        x: pd.DataFrame,
        anchor: np.ndarray,
    ) -> np.ndarray:
        ridge = models.full_ridge.predict(x)
        recent = models.recent_ridge.predict(x)
        gas = models.gas_ridge.predict(x[self.gas_feature_columns_])
        residual = np.column_stack([model.predict(x) for model in models.residual_models])
        deltas = [np.zeros_like(ridge), ridge, recent, gas, ridge + residual]
        clipped = [np.clip(delta, models.delta_lower, models.delta_upper) for delta in deltas]
        return np.stack([anchor[:, None] + delta for delta in clipped], axis=1)

    def _event_matrix(
        self,
        x: pd.DataFrame,
        anchor: np.ndarray,
        gmm: GaussianMixture,
        columns: list[str],
    ) -> pd.DataFrame:
        matrix = x.reindex(columns=columns).reset_index(drop=True).copy()
        return pd.concat([matrix, _state_matrix(gmm, anchor, "target_state")], axis=1)

    def _fit_event_state(
        self,
        x: pd.DataFrame,
        anchor: np.ndarray,
        y: np.ndarray,
        target: str,
        columns: list[str],
        *,
        components: int | None = None,
        threshold: float | None = None,
    ) -> EventState:
        gmm = _fit_best_gmm(
            anchor,
            self.config.model.state_components,
            self.config.model.random_state,
            components,
        )
        if threshold is None:
            reference = float(np.nanquantile(np.abs(y), 0.65))
            candidates = _event_candidates(target)
            threshold = float(candidates[np.argmin(np.abs(candidates - reference))])
        event_x = self._event_matrix(x, anchor, gmm, columns)
        classifiers: list[EventClassifier] = []

        for step in range(y.shape[1]):
            labels = np.where(y[:, step] < -threshold, -1, np.where(y[:, step] > threshold, 1, 0))
            classes = np.unique(labels)
            if len(classes) < 2:
                classifiers.append(EventClassifier(model=None, constant_class=int(classes[0])))
                continue
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=0.2,
                            class_weight="balanced",
                            max_iter=400,
                            random_state=self.config.model.random_state + step,
                        ),
                    ),
                ]
            )
            model.fit(event_x, labels)
            classifiers.append(EventClassifier(model=model))
        return EventState(gmm, columns, threshold, classifiers)

    def _predict_events(
        self,
        event: EventState,
        x: pd.DataFrame,
        anchor: np.ndarray,
    ) -> np.ndarray:
        event_x = self._event_matrix(x, anchor, event.gmm, event.feature_columns)
        result = np.zeros((len(x), len(event.classifiers), 3), dtype=float)
        class_to_column = {-1: 0, 0: 1, 1: 2}
        for step, classifier in enumerate(event.classifiers):
            if classifier.model is None:
                result[:, step, class_to_column[classifier.constant_class]] = 1.0
                continue
            probabilities = classifier.model.predict_proba(event_x)
            classes = classifier.model.named_steps["logistic"].classes_
            for class_index, label in enumerate(classes):
                result[:, step, class_to_column[int(label)]] = probabilities[:, class_index]
        return result

    def _gate_matrix(
        self,
        x: pd.DataFrame,
        anchor: np.ndarray,
        branches: np.ndarray,
        rest_values: np.ndarray,
        rest_gmm: GaussianMixture,
        event_probabilities: np.ndarray | None,
        correction_weights: np.ndarray,
        columns: list[str],
        step: int,
    ) -> pd.DataFrame:
        matrix = x.reindex(columns=columns).reset_index(drop=True).copy()
        corrected = branches[:, 1:, step] @ correction_weights[:, step]
        matrix["model_correction"] = corrected - anchor
        matrix["model_disagreement_mad"] = _robust_disagreement(branches[:, 1:, :])[:, step]
        matrix["ridge_correction"] = branches[:, 1, step] - anchor
        matrix["gas_correction"] = branches[:, 3, step] - anchor
        matrix["lgb_correction"] = branches[:, 4, step] - anchor
        matrix = pd.concat([matrix, _state_matrix(rest_gmm, rest_values, "rest_state")], axis=1)
        if event_probabilities is not None:
            matrix["event_drop_probability"] = event_probabilities[:, step, 0]
            matrix["event_stable_probability"] = event_probabilities[:, step, 1]
            matrix["event_rise_probability"] = event_probabilities[:, step, 2]
        return matrix

    def _fit_gates(
        self,
        x: pd.DataFrame,
        anchor: np.ndarray,
        actual: np.ndarray,
        branches: np.ndarray,
        rest_values: np.ndarray,
        rest_gmm: GaussianMixture,
        event_probabilities: np.ndarray | None,
        correction_weights: np.ndarray,
        columns: list[str],
    ) -> list[GateState]:
        gates: list[GateState] = []
        for step in range(actual.shape[1]):
            corrected = branches[:, 1:, step] @ correction_weights[:, step]
            correction = corrected - anchor
            valid = np.abs(correction) > 1e-3
            oracle = np.zeros(len(x), dtype=float)
            oracle[valid] = (actual[valid, step] - anchor[valid]) / correction[valid]
            oracle = np.clip(oracle, 0.0, 1.0)
            sample_weight = np.clip(
                np.abs(correction) / np.maximum(np.abs(actual[:, step]), 1.0), 0.05, 1.0
            )
            if float(np.std(oracle)) < 1e-6:
                gates.append(GateState(model=None, constant_probability=float(oracle[0])))
                continue
            gate_x = self._gate_matrix(
                x,
                anchor,
                branches,
                rest_values,
                rest_gmm,
                event_probabilities,
                correction_weights,
                columns,
                step,
            )
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=40.0)),
                ]
            )
            model.fit(gate_x, oracle, ridge__sample_weight=sample_weight)
            gates.append(GateState(model=model))
        return gates

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "GasAwareEnsembleForecaster":
        self.feature_columns_ = list(features.columns)
        self.gas_feature_columns_ = _select_gas_features(self.feature_columns_)
        max_horizon = max(self.config.feature.horizons)
        full_rest = current["generator_all"] - current["generator_1"]

        for target in self.config.targets:
            columns = target_columns(target, self.config.feature.horizons)
            valid = (
                current[target].notna()
                & full_rest.notna()
                & deltas[columns].notna().all(axis=1)
            )
            x = features.loc[valid]
            y = deltas.loc[valid, columns]
            anchor = current.loc[valid, target]
            rest = full_rest.loc[valid]
            if len(x) < 400:
                raise ValueError(f"{target} 的增强模型有效训练样本不足 400 行")

            calibration_rows = max(192, int(len(x) * self.config.model.calibration_fraction))
            calibration_start = len(x) - calibration_rows
            development_end = max(200, calibration_start - max_horizon)
            development_x = x.iloc[:development_end]
            development_y = y.iloc[:development_end]
            development_anchor = anchor.iloc[:development_end].to_numpy(dtype=float)
            development_rest = rest.iloc[:development_end].to_numpy(dtype=float)
            calibration_x = x.iloc[calibration_start:]
            calibration_y = y.iloc[calibration_start:].to_numpy(dtype=float)
            calibration_anchor = anchor.iloc[calibration_start:].to_numpy(dtype=float)
            calibration_rest = rest.iloc[calibration_start:].to_numpy(dtype=float)
            actual = calibration_anchor[:, None] + calibration_y

            development_branches = self._fit_branches(development_x, development_y)
            calibration_branches = self._predict_branches(
                development_branches, calibration_x, calibration_anchor
            )
            blend_weights = _fit_simplex_weights(calibration_branches, actual)
            correction_weights = _fit_simplex_weights(calibration_branches[:, 1:, :], actual)
            disagreement = _robust_disagreement(calibration_branches[:, 1:, :])
            uncertainty_low = np.quantile(disagreement, 0.70, axis=0)
            uncertainty_high = np.quantile(disagreement, 0.95, axis=0)

            gate_columns = _select_gate_features(self.feature_columns_, target)
            rest_gmm: GaussianMixture | None = None
            event_state: EventState | None = None
            gates: list[GateState] = []
            if self.version in {"v25", "v3"}:
                development_rest_gmm = _fit_best_gmm(
                    development_rest,
                    self.config.model.state_components,
                    self.config.model.random_state,
                )
                calibration_events = None
                development_event = None
                if self.version == "v3":
                    development_event = self._fit_event_state(
                        development_x,
                        development_anchor,
                        development_y.to_numpy(),
                        target,
                        gate_columns,
                    )
                    calibration_events = self._predict_events(
                        development_event, calibration_x, calibration_anchor
                    )
                gates = self._fit_gates(
                    calibration_x,
                    calibration_anchor,
                    actual,
                    calibration_branches,
                    calibration_rest,
                    development_rest_gmm,
                    calibration_events,
                    correction_weights,
                    gate_columns,
                )
                rest_gmm = _fit_best_gmm(
                    rest.to_numpy(dtype=float),
                    self.config.model.state_components,
                    self.config.model.random_state,
                    development_rest_gmm.n_components,
                )
                if development_event is not None:
                    event_state = self._fit_event_state(
                        x,
                        anchor.to_numpy(dtype=float),
                        y.to_numpy(),
                        target,
                        gate_columns,
                        components=development_event.gmm.n_components,
                        threshold=development_event.threshold,
                    )

            final_branches = self._fit_branches(x, y)
            self.ensemble_states_[target] = TargetEnsembleState(
                branches=final_branches,
                blend_weights=blend_weights,
                correction_weights=correction_weights,
                rest_gmm=rest_gmm,
                event=event_state,
                gates=gates,
                gate_feature_columns=gate_columns,
                uncertainty_low=uncertainty_low,
                uncertainty_high=uncertainty_high,
            )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if not self.ensemble_states_:
            raise RuntimeError("增强模型尚未训练")
        x = features.reindex(columns=self.feature_columns_)
        rest_values = (current["generator_all"] - current["generator_1"]).ffill().to_numpy(float)
        predictions: dict[str, np.ndarray] = {}

        for target in self.config.targets:
            state = self.ensemble_states_[target]
            anchor = current[target].ffill().to_numpy(dtype=float)
            branches = self._predict_branches(state.branches, x, anchor)
            if self.version == "v2":
                absolute = np.einsum("nbh,bh->nh", branches, state.blend_weights)
            else:
                if state.rest_gmm is None:
                    raise RuntimeError("动态门控缺少 generator_rest 状态模型")
                events = (
                    self._predict_events(state.event, x, anchor) if state.event is not None else None
                )
                absolute = np.empty((len(x), len(self.config.feature.horizons)))
                disagreement = _robust_disagreement(branches[:, 1:, :])
                max_shrink = (
                    self.config.model.generator_1_max_shrink
                    if target == "generator_1"
                    else self.config.model.generator_all_max_shrink
                )
                for step, gate in enumerate(state.gates):
                    corrected = branches[:, 1:, step] @ state.correction_weights[:, step]
                    if gate.model is None:
                        probability = np.full(len(x), gate.constant_probability)
                    else:
                        gate_x = self._gate_matrix(
                            x,
                            anchor,
                            branches,
                            rest_values,
                            state.rest_gmm,
                            events,
                            state.correction_weights,
                            state.gate_feature_columns,
                            step,
                        )
                        probability = gate.model.predict(gate_x)
                    probability = np.clip(
                        probability, self.config.model.gate_min, self.config.model.gate_max
                    )
                    shrink = (
                        (disagreement[:, step] - state.uncertainty_low[step])
                        / max(
                            state.uncertainty_high[step] - state.uncertainty_low[step], 1e-6
                        )
                    )
                    shrink = np.clip(shrink, 0.0, 1.0) * max_shrink
                    effective_gate = probability * (1.0 - shrink)
                    absolute[:, step] = anchor + effective_gate * (corrected - anchor)

            for step, horizon in enumerate(self.config.feature.horizons):
                predictions[f"{target}_t+{15 * horizon}_pred"] = absolute[:, step]

        output = pd.DataFrame(predictions, index=features.index)
        self._apply_weak_constraints(output)
        return output
