"""内层时间 cross-fitting 基础分支与 OOF 残差 LightGBM。"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from gas_forecast.config import ForecastConfig
from gas_forecast.model_v1 import make_ridge_pipeline
from gas_forecast.reconciliation import (
    ReconciliationState,
    fit_reconciliation,
    reconcile_predictions,
)
from gas_forecast.splits import TimeFold, make_inner_folds
from gas_forecast.stacking import (
    DynamicGateState,
    SimplexState,
    apply_dynamic_gate,
    apply_simplex,
    fit_dynamic_gate,
    fit_simplex_state,
)
from gas_forecast.targets import target_columns


BRANCH_NAMES = ("persistence", "ridge", "recent_ridge", "gas_ridge", "oof_residual_lgb")


@dataclass
class FittedBaseModels:
    ridge: Pipeline
    recent_ridge: Pipeline
    gas_ridge: Pipeline
    residual_models: list[LGBMRegressor]
    delta_lower: np.ndarray
    delta_upper: np.ndarray


@dataclass
class CrossFittedTargetState:
    models: FittedBaseModels
    stacker: SimplexState
    inner_folds: tuple[TimeFold, ...]
    branch_oof_rows: int
    residual_oof_rows: int
    distribution: dict[str, object]
    oof_index: pd.DatetimeIndex
    branch_oof: np.ndarray
    actual_oof: np.ndarray
    dynamic_gate: DynamicGateState


def select_gas_feature_columns(columns: list[str]) -> list[str]:
    """显式、稳定地登记煤气分支字段，避免按构造顺序截断。"""

    keywords = (
        "gas",
        "furnace",
        "heater",
        "user",
        "holder",
        "surplus",
        "balance",
        "price",
        "time_",
        "hour",
        "month",
        "minute",
    )
    selected = sorted(column for column in columns if any(key in column for key in keywords))
    if not selected:
        raise ValueError("未找到煤气分支可用特征")
    return selected


def _recent_mask(index: pd.DatetimeIndex, days: int) -> np.ndarray:
    mask = np.asarray(index >= index.max() - pd.Timedelta(days=days))
    return mask if int(mask.sum()) >= 200 else np.ones(len(index), dtype=bool)


def _fit_ridges(
    x: pd.DataFrame,
    y: pd.DataFrame,
    gas_columns: list[str],
    config: ForecastConfig,
) -> tuple[Pipeline, Pipeline, Pipeline]:
    ridge = make_ridge_pipeline(config.model.ridge_alpha).fit(x, y)
    recent = make_ridge_pipeline(config.model.ridge_alpha).fit(
        x.loc[_recent_mask(x.index, config.model.recent_days)],
        y.loc[_recent_mask(x.index, config.model.recent_days)],
    )
    gas = make_ridge_pipeline(config.model.ridge_alpha).fit(x[gas_columns], y)
    return ridge, recent, gas


def _make_lgb(config: ForecastConfig, step: int, estimators: int | None = None) -> LGBMRegressor:
    return LGBMRegressor(
        objective=config.model.lgb_objective,
        n_estimators=estimators or config.model.lgb_max_estimators,
        learning_rate=config.model.lgb_learning_rate,
        num_leaves=config.model.lgb_num_leaves,
        max_depth=config.model.lgb_max_depth,
        min_child_samples=config.model.lgb_min_child_samples,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=config.model.random_state + step,
        n_jobs=config.model.tree_threads_per_worker,
        verbosity=-1,
    )


def _fit_residual_model(
    x: pd.DataFrame,
    residual: np.ndarray,
    future_absolute: np.ndarray,
    config: ForecastConfig,
    step: int,
) -> LGBMRegressor:
    """用最终绝对目标近似 MAPE 权重，并通过时间尾块 early stopping。"""

    split = max(200, int(len(x) * 0.8))
    split = min(split, len(x) - 96)
    lower = max(float(np.nanquantile(np.abs(future_absolute), 0.05)), 1e-6)
    upper = max(float(np.nanquantile(np.abs(future_absolute), 0.95)), lower)
    weights = (
        1.0 / np.clip(np.abs(future_absolute), lower, upper)
        if config.model.lgb_use_mape_weights
        else np.ones(len(future_absolute))
    )
    if (
        config.model.lgb_use_early_stopping
        and split >= 200
        and len(x) - split >= 96
    ):
        probe = _make_lgb(config, step)
        probe.fit(
            x.iloc[:split],
            residual[:split],
            sample_weight=weights[:split],
            eval_X=x.iloc[split:],
            eval_y=residual[split:],
            eval_sample_weight=[weights[split:]],
            eval_metric="l1",
            callbacks=[
                lgb.early_stopping(
                    config.model.lgb_early_stopping_rounds,
                    verbose=False,
                )
            ],
        )
        estimators = max(20, int(probe.best_iteration_ or config.model.lgb_n_estimators))
    else:
        estimators = config.model.lgb_n_estimators
    model = _make_lgb(config, step, estimators)
    model.fit(x, residual, sample_weight=weights)
    return model


def _predict_ridge_branches(
    models: tuple[Pipeline, Pipeline, Pipeline],
    x: pd.DataFrame,
    anchor: np.ndarray,
    gas_columns: list[str],
) -> np.ndarray:
    ridge, recent, gas = models
    ridge_delta = ridge.predict(x)
    deltas = [
        np.zeros_like(ridge_delta),
        ridge_delta,
        recent.predict(x),
        gas.predict(x[gas_columns]),
    ]
    return np.stack([anchor[:, None] + delta for delta in deltas], axis=1)


class CrossFittedBranchForecaster:
    """用内层时间 OOF 训练残差模型和低自由度融合器。"""

    version = "crossfit"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.feature_columns_: list[str] = []
        self.gas_feature_columns_: list[str] = []
        self.states_: dict[str, CrossFittedTargetState] = {}
        self.reconciliation_state_: ReconciliationState | None = None

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "CrossFittedBranchForecaster":
        self.feature_columns_ = list(features.columns)
        self.gas_feature_columns_ = select_gas_feature_columns(self.feature_columns_)
        for target in self.config.targets:
            columns = target_columns(target, self.config.feature.horizons)
            valid = current[target].notna() & deltas[columns].notna().all(axis=1)
            x = features.loc[valid]
            y = deltas.loc[valid, columns]
            anchor = current.loc[valid, target].to_numpy(dtype=float)
            if len(x) < 800:
                raise ValueError(f"{target} 的 cross-fitting 有效样本不足 800 行")
            inner_folds = make_inner_folds(
                x.index,
                folds=self.config.model.inner_folds,
                purge_steps=max(self.config.feature.horizons),
            )
            ridge_oof = np.full((len(x), 4, len(columns)), np.nan)
            residual_targets = np.full((len(x), len(columns)), np.nan)
            fold_positions: list[np.ndarray] = []
            for fold in inner_folds:
                train_mask, validation_mask = fold.masks(x.index)
                models = _fit_ridges(
                    x.loc[train_mask], y.loc[train_mask], self.gas_feature_columns_, self.config
                )
                ridge_oof[validation_mask] = _predict_ridge_branches(
                    models,
                    x.loc[validation_mask],
                    anchor[validation_mask],
                    self.gas_feature_columns_,
                )
                residual_targets[validation_mask] = (
                    y.loc[validation_mask].to_numpy() - models[0].predict(x.loc[validation_mask])
                )
                fold_positions.append(np.flatnonzero(validation_mask))

            residual_oof_prediction = np.zeros((len(x), len(columns)), dtype=float)
            seen_positions: list[int] = []
            for positions in fold_positions:
                if len(seen_positions) >= 200:
                    train_positions = np.asarray(seen_positions)
                    for step in range(len(columns)):
                        model = _fit_residual_model(
                            x.iloc[train_positions],
                            residual_targets[train_positions, step],
                            anchor[train_positions] + y.iloc[train_positions, step].to_numpy(),
                            self.config,
                            step,
                        )
                        residual_oof_prediction[positions, step] = model.predict(x.iloc[positions])
                seen_positions.extend(positions.tolist())

            oof_mask = np.isfinite(ridge_oof).all(axis=(1, 2))
            branch_oof = np.concatenate(
                [
                    ridge_oof[oof_mask],
                    (
                        ridge_oof[oof_mask, 1, :]
                        + residual_oof_prediction[oof_mask]
                    )[:, None, :],
                ],
                axis=1,
            )
            actual_oof = anchor[oof_mask, None] + y.iloc[oof_mask].to_numpy()
            stacker = fit_simplex_state(
                branch_oof,
                actual_oof,
                BRANCH_NAMES,
                regularization=self.config.model.simplex_regularization,
            )
            regularized_oof = apply_simplex(
                branch_oof, stacker.regularized_weights
            )
            dynamic_gate = fit_dynamic_gate(
                x.loc[oof_mask],
                anchor[oof_mask],
                branch_oof,
                actual_oof,
                regularized_oof,
                target=target,
            )

            final_ridges = _fit_ridges(x, y, self.gas_feature_columns_, self.config)
            residual_train = np.flatnonzero(np.isfinite(residual_targets).all(axis=1))
            residual_models = [
                _fit_residual_model(
                    x.iloc[residual_train],
                    residual_targets[residual_train, step],
                    anchor[residual_train] + y.iloc[residual_train, step].to_numpy(),
                    self.config,
                    step,
                )
                for step in range(len(columns))
            ]
            distribution = {
                "branch_oof_mean": np.nanmean(branch_oof, axis=0).tolist(),
                "branch_oof_std": np.nanstd(branch_oof, axis=0).tolist(),
                "gate_feature_columns": [],
                "gas_feature_columns": self.gas_feature_columns_,
            }
            self.states_[target] = CrossFittedTargetState(
                models=FittedBaseModels(
                    ridge=final_ridges[0],
                    recent_ridge=final_ridges[1],
                    gas_ridge=final_ridges[2],
                    residual_models=residual_models,
                    delta_lower=y.quantile(self.config.model.lower_quantile).to_numpy(),
                    delta_upper=y.quantile(self.config.model.upper_quantile).to_numpy(),
                ),
                stacker=stacker,
                inner_folds=tuple(inner_folds),
                branch_oof_rows=int(oof_mask.sum()),
                residual_oof_rows=int(len(residual_train)),
                distribution=distribution,
                oof_index=x.index[oof_mask],
                branch_oof=branch_oof,
                actual_oof=actual_oof,
                dynamic_gate=dynamic_gate,
            )
        structural_targets = ("generator_1", "generator_rest", "generator_all")
        if all(target in self.states_ for target in structural_targets):
            common_index = self.states_[structural_targets[0]].oof_index
            for target in structural_targets[1:]:
                common_index = common_index.intersection(self.states_[target].oof_index)
            base_predictions = []
            actual_values = []
            for target in structural_targets:
                state = self.states_[target]
                positions = state.oof_index.get_indexer(common_index)
                base_predictions.append(
                    apply_simplex(
                        state.branch_oof[positions], state.stacker.regularized_weights
                    )
                )
                actual_values.append(state.actual_oof[positions])
            self.reconciliation_state_ = fit_reconciliation(
                np.stack(actual_values, axis=1),
                np.stack(base_predictions, axis=1),
                estimate_full_covariance=False,
            )
        return self

    def predict_candidates(
        self,
        features: pd.DataFrame,
        current: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """返回每个基础分支及三种 simplex 候选的绝对量预测。"""

        if not self.states_:
            raise RuntimeError("cross-fitting 模型尚未训练")
        x = features.reindex(columns=self.feature_columns_)
        outputs: dict[str, dict[str, np.ndarray]] = {
            name: {}
            for name in (
                *BRANCH_NAMES,
                "simplex_target",
                "simplex_horizon",
                "simplex_regularized",
                "dynamic_gate",
            )
        }
        for target in self.config.targets:
            state = self.states_[target]
            anchor = current[target].ffill().to_numpy(dtype=float)
            ridge_models = (
                state.models.ridge,
                state.models.recent_ridge,
                state.models.gas_ridge,
            )
            ridge_branches = _predict_ridge_branches(
                ridge_models, x, anchor, self.gas_feature_columns_
            )
            residual = np.column_stack([model.predict(x) for model in state.models.residual_models])
            lgb_absolute = anchor[:, None] + np.clip(
                state.models.ridge.predict(x) + residual,
                state.models.delta_lower,
                state.models.delta_upper,
            )
            branches = np.concatenate([ridge_branches, lgb_absolute[:, None, :]], axis=1)
            blends = {
                "simplex_target": apply_simplex(branches, state.stacker.target_weights),
                "simplex_horizon": apply_simplex(branches, state.stacker.horizon_weights),
                "simplex_regularized": apply_simplex(branches, state.stacker.regularized_weights),
            }
            blends["dynamic_gate"] = apply_dynamic_gate(
                state.dynamic_gate,
                x,
                anchor,
                branches,
                blends["simplex_regularized"],
                gate_min=self.config.model.gate_min,
                gate_max=self.config.model.gate_max,
            )
            for step, horizon in enumerate(self.config.feature.horizons):
                key = f"{target}_t+{15 * horizon}_pred"
                for branch_index, name in enumerate(BRANCH_NAMES):
                    outputs[name][key] = branches[:, branch_index, step]
                for name, values in blends.items():
                    outputs[name][key] = values[:, step]
        if self.reconciliation_state_ is not None:
            structural_targets = ("generator_1", "generator_rest", "generator_all")
            base = np.stack(
                [
                    np.column_stack(
                        [
                            outputs["simplex_regularized"][f"{target}_t+{15 * horizon}_pred"]
                            for horizon in self.config.feature.horizons
                        ]
                    )
                    for target in structural_targets
                ],
                axis=1,
            )
            reconciled = reconcile_predictions(base, self.reconciliation_state_)
            for candidate in ("struct_bottom_up", "struct_blended", "struct_diagonal"):
                outputs[candidate] = dict(outputs["simplex_regularized"])
            for step, horizon in enumerate(self.config.feature.horizons):
                all_key = f"generator_all_t+{15 * horizon}_pred"
                outputs["struct_bottom_up"][all_key] = reconciled["bottom_up_all"][:, step]
                outputs["struct_blended"][all_key] = reconciled["blended_all"][:, step]
                for target_index, target in enumerate(structural_targets):
                    key = f"{target}_t+{15 * horizon}_pred"
                    outputs["struct_diagonal"][key] = reconciled[
                        "diagonal_reconciled"
                    ][:, target_index, step]
        return {
            name: pd.DataFrame(values, index=features.index)
            for name, values in outputs.items()
        }

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        return self.predict_candidates(features, current)["simplex_regularized"]
