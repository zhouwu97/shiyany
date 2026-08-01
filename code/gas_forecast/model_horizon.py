"""目标时刻对齐的逐目标逐步长 Ridge。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig
from gas_forecast.model_v1 import _mape, make_ridge_pipeline
from gas_forecast.targets import target_columns


@dataclass
class HorizonTargetState:
    """单个目标的逐步长模型状态。"""

    models: list
    weights: np.ndarray
    delta_lower: np.ndarray
    delta_upper: np.ndarray


class HorizonSpecificRidgeForecaster:
    """每个目标和预测步长独立建模，避免共享不匹配的周期特征。"""

    version = "horizon_ridge"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.feature_columns_: list[str] = []
        self.states_: dict[str, HorizonTargetState] = {}

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "HorizonSpecificRidgeForecaster":
        self.feature_columns_ = list(features.columns)
        horizons = self.config.feature.horizons
        max_horizon = max(horizons)

        for target in self.config.targets:
            columns = target_columns(target, horizons)
            valid = current[target].notna() & deltas[columns].notna().all(axis=1)
            x = features.loc[valid]
            y = deltas.loc[valid, columns]
            anchor = current.loc[valid, target]
            if len(x) < 200:
                raise ValueError(f"{target} 的逐步长 Ridge 有效训练样本不足 200 行")

            calibration_rows = max(96, int(len(x) * self.config.model.calibration_fraction))
            calibration_start = len(x) - calibration_rows
            development_end = max(100, calibration_start - max_horizon)
            development_x = x.iloc[:development_end]
            calibration_x = x.iloc[calibration_start:]
            calibration_anchor = anchor.iloc[calibration_start:].to_numpy(dtype=float)
            calibration_actual = calibration_anchor[:, None] + y.iloc[calibration_start:].to_numpy()
            grid = np.linspace(0.0, 1.0, 51)
            models = []
            weights = []
            lowers = []
            uppers = []
            for step, column in enumerate(columns):
                probe = make_ridge_pipeline(self.config.model.ridge_alpha)
                probe.fit(development_x, y.iloc[:development_end][column])
                probe_prediction = calibration_anchor + probe.predict(calibration_x)
                scores = [
                    _mape(
                        calibration_actual[:, step],
                        calibration_anchor + weight * (probe_prediction - calibration_anchor),
                    )
                    for weight in grid
                ]
                weights.append(float(grid[int(np.argmin(scores))]))
                model = make_ridge_pipeline(self.config.model.ridge_alpha)
                model.fit(x, y[column])
                models.append(model)
                lowers.append(float(y[column].quantile(self.config.model.lower_quantile)))
                uppers.append(float(y[column].quantile(self.config.model.upper_quantile)))
            self.states_[target] = HorizonTargetState(
                models=models,
                weights=np.asarray(weights, dtype=float),
                delta_lower=np.asarray(lowers, dtype=float),
                delta_upper=np.asarray(uppers, dtype=float),
            )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if not self.states_:
            raise RuntimeError("逐步长 Ridge 尚未训练")
        x = features.reindex(columns=self.feature_columns_)
        predictions: dict[str, np.ndarray] = {}
        horizons = self.config.feature.horizons
        for target in self.config.targets:
            state = self.states_[target]
            anchor = current[target].ffill().to_numpy(dtype=float)
            for step, horizon in enumerate(horizons):
                delta = state.models[step].predict(x)
                delta = np.clip(delta, state.delta_lower[step], state.delta_upper[step])
                absolute = anchor + state.weights[step] * delta
                predictions[f"{target}_t+{15 * horizon}_pred"] = absolute
        output = pd.DataFrame(predictions, index=features.index)
        self._apply_constraints(output)
        return output

    @staticmethod
    def _apply_constraints(output: pd.DataFrame) -> None:
        for horizon in sorted(
            int(column.rsplit("_t+", 1)[1].removesuffix("_pred"))
            for column in output.columns
            if column.startswith("generator_1_t+")
        ):
            gen1 = f"generator_1_t+{horizon}_pred"
            total = f"generator_all_t+{horizon}_pred"
            output[gen1] = output[gen1].clip(0.0, 200.0)
            output[total] = output[total].clip(0.0, 440.0)
            output[total] = np.maximum(output[total], output[gen1])
            output[total] = np.minimum(output[total], output[gen1] + 240.0)
        if not np.isfinite(output.to_numpy()).all():
            raise ValueError("逐步长 Ridge 预测包含非有限值")
