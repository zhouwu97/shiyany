"""未来煤气资源轨迹的严格 OOF 两阶段候选。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from gas_forecast.config import ForecastConfig
from gas_forecast.model_v1 import make_ridge_pipeline
from gas_forecast.splits import TimeFold, make_inner_folds
from gas_forecast.targets import target_columns


DEFAULT_RESOURCE_COLUMNS = (
    "generator_use_blast_furnace_gas",
    "generator_use_coke_gas",
    "generator_use_converter_gas",
    "blast_furnace_1",
    "blast_furnace_2",
    "blast_furnace_4",
    "blast_furnace_5",
    "coke_oven_1",
    "converter_1",
    "blast_furnace_gas_holder_2",
)


@dataclass
class GasStageTargetState:
    model: Pipeline
    delta_lower: np.ndarray
    delta_upper: np.ndarray


def _resource_targets(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for column in columns:
        for horizon in horizons:
            values[f"resource_{column}_tplus_{15 * horizon}"] = frame[column].shift(-horizon)
    return pd.DataFrame(values, index=frame.index)


class GasTrajectoryForecaster:
    """第一阶段 Ridge 预测资源轨迹，第二阶段仅用第一阶段时间 OOF 特征。"""

    version = "gas_trajectory"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.feature_columns_: list[str] = []
        self.resource_columns_: tuple[str, ...] = ()
        self.resource_target_columns_: list[str] = []
        self.resource_model_: Pipeline | None = None
        self.stage2_states_: dict[str, GasStageTargetState] = {}
        self.inner_folds_: tuple[TimeFold, ...] = ()
        self.stage2_rows_: int = 0

    def fit(
        self,
        frame: pd.DataFrame,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "GasTrajectoryForecaster":
        self.feature_columns_ = list(features.columns)
        self.resource_columns_ = tuple(
            column for column in DEFAULT_RESOURCE_COLUMNS if column in frame.columns
        )
        if len(self.resource_columns_) < 4:
            raise ValueError("未来煤气轨迹可用资源字段不足 4 个")
        resource_y = _resource_targets(
            frame, self.resource_columns_, self.config.feature.horizons
        )
        self.resource_target_columns_ = list(resource_y.columns)
        valid_resource = resource_y.notna().all(axis=1)
        valid_index = features.index[valid_resource]
        inner_folds = make_inner_folds(
            valid_index,
            folds=self.config.model.inner_folds,
            purge_steps=max(self.config.feature.horizons),
        )
        resource_oof = pd.DataFrame(np.nan, index=features.index, columns=resource_y.columns)
        for fold in inner_folds:
            train = (
                valid_resource
                & (features.index >= fold.train_start)
                & (features.index <= fold.train_end)
            )
            validation = (
                valid_resource
                & (features.index >= fold.validation_start)
                & (features.index < fold.validation_end)
            )
            model = make_ridge_pipeline(self.config.model.ridge_alpha).fit(
                features.loc[train], resource_y.loc[train]
            )
            resource_oof.loc[validation] = model.predict(features.loc[validation])

        stage2_valid = resource_oof.notna().all(axis=1)
        stage2_x = pd.concat(
            [
                features.loc[stage2_valid].add_prefix("origin_"),
                resource_oof.loc[stage2_valid],
            ],
            axis=1,
        )
        self.stage2_rows_ = int(stage2_valid.sum())
        if self.stage2_rows_ < 300:
            raise ValueError("两阶段模型的第一阶段 OOF 行数不足 300")
        for target in self.config.targets:
            columns = target_columns(target, self.config.feature.horizons)
            valid = current.loc[stage2_valid, target].notna() & deltas.loc[
                stage2_valid, columns
            ].notna().all(axis=1)
            x = stage2_x.loc[valid]
            y = deltas.loc[x.index, columns]
            model = make_ridge_pipeline(self.config.model.ridge_alpha).fit(x, y)
            self.stage2_states_[target] = GasStageTargetState(
                model=model,
                delta_lower=y.quantile(self.config.model.lower_quantile).to_numpy(),
                delta_upper=y.quantile(self.config.model.upper_quantile).to_numpy(),
            )
        self.resource_model_ = make_ridge_pipeline(self.config.model.ridge_alpha).fit(
            features.loc[valid_resource], resource_y.loc[valid_resource]
        )
        self.inner_folds_ = tuple(inner_folds)
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if self.resource_model_ is None or not self.stage2_states_:
            raise RuntimeError("两阶段煤气轨迹模型尚未训练")
        aligned = features.reindex(columns=self.feature_columns_)
        resource = pd.DataFrame(
            self.resource_model_.predict(aligned),
            index=features.index,
            columns=self.resource_target_columns_,
        )
        stage2_x = pd.concat([aligned.add_prefix("origin_"), resource], axis=1)
        predictions: dict[str, np.ndarray] = {}
        for target in self.config.targets:
            state = self.stage2_states_[target]
            delta = np.clip(
                state.model.predict(stage2_x), state.delta_lower, state.delta_upper
            )
            absolute = current[target].ffill().to_numpy()[:, None] + delta
            for step, horizon in enumerate(self.config.feature.horizons):
                predictions[f"{target}_t+{15 * horizon}_pred"] = absolute[:, step]
        return pd.DataFrame(predictions, index=features.index)
