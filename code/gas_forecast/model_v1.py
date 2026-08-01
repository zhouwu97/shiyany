"""持续性锚点下的直接绝对增量 Ridge。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gas_forecast.config import ForecastConfig
from gas_forecast.targets import target_columns


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.maximum(np.abs(actual), 1e-6)
    return float(np.mean(np.abs(actual - predicted) / denominator))


def make_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def make_weighted_lad_pipeline(alpha: float) -> Pipeline:
    """返回中位数 LAD 管线；样本权重由调用方按未来绝对量提供。"""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("lad", QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")),
        ]
    )


@dataclass
class TargetV1State:
    model: Pipeline
    weights: np.ndarray
    delta_lower: np.ndarray
    delta_upper: np.ndarray


class RidgeDeltaForecaster:
    """为两个目标分别训练 8 输出增量模型。"""

    version = "v1"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.feature_columns_: list[str] = []
        self.states_: dict[str, TargetV1State] = {}

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        return features.reindex(columns=self.feature_columns_)

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "RidgeDeltaForecaster":
        self.feature_columns_ = list(features.columns)
        max_horizon = max(self.config.feature.horizons)

        for target in self.config.targets:
            columns = target_columns(target, self.config.feature.horizons)
            valid = current[target].notna() & deltas[columns].notna().all(axis=1)
            x = features.loc[valid]
            y = deltas.loc[valid, columns]
            anchor = current.loc[valid, target]
            if len(x) < 200:
                raise ValueError(f"{target} 的有效训练样本不足 200 行")

            calibration_rows = max(96, int(len(x) * self.config.model.calibration_fraction))
            calibration_start = len(x) - calibration_rows
            development_end = max(100, calibration_start - max_horizon)
            initial = make_ridge_pipeline(self.config.model.ridge_alpha)
            initial.fit(x.iloc[:development_end], y.iloc[:development_end])

            calibration_x = x.iloc[calibration_start:]
            calibration_y = y.iloc[calibration_start:].to_numpy()
            calibration_anchor = anchor.iloc[calibration_start:].to_numpy()[:, None]
            ridge_absolute = calibration_anchor + initial.predict(calibration_x)
            actual = calibration_anchor + calibration_y

            raw_weights = []
            grid = np.linspace(0.0, 1.0, 51)
            for step in range(len(columns)):
                scores = [
                    _mape(
                        actual[:, step],
                        calibration_anchor[:, 0]
                        + weight * (ridge_absolute[:, step] - calibration_anchor[:, 0]),
                    )
                    for weight in grid
                ]
                raw_weights.append(grid[int(np.argmin(scores))])
            weights = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit_transform(
                np.arange(len(columns)), raw_weights
            )

            final_model = make_ridge_pipeline(self.config.model.ridge_alpha)
            final_model.fit(x, y)
            lower = y.quantile(self.config.model.lower_quantile).to_numpy()
            upper = y.quantile(self.config.model.upper_quantile).to_numpy()
            self.states_[target] = TargetV1State(
                model=final_model,
                weights=np.asarray(weights),
                delta_lower=lower,
                delta_upper=upper,
            )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if not self.states_:
            raise RuntimeError("模型尚未训练")
        x = self._align_features(features)
        predictions: dict[str, np.ndarray] = {}

        for target in self.config.targets:
            state = self.states_[target]
            anchor = current[target].ffill().to_numpy(dtype=float)[:, None]
            delta = state.model.predict(x)
            delta = np.clip(delta, state.delta_lower, state.delta_upper)
            absolute = anchor + state.weights * delta
            for step, horizon in enumerate(self.config.feature.horizons):
                predictions[f"{target}_t+{15 * horizon}_pred"] = absolute[:, step]

        output = pd.DataFrame(predictions, index=features.index)
        self._apply_weak_constraints(output)
        return output

    def _apply_weak_constraints(self, output: pd.DataFrame) -> None:
        for horizon in self.config.feature.horizons:
            minutes = 15 * horizon
            generator_1 = f"generator_1_t+{minutes}_pred"
            generator_all = f"generator_all_t+{minutes}_pred"
            if generator_1 in output:
                output[generator_1] = output[generator_1].clip(lower=0.0, upper=200.0)
            if generator_all in output:
                output[generator_all] = output[generator_all].clip(lower=0.0, upper=440.0)
            if generator_1 in output and generator_all in output:
                output[generator_all] = np.maximum(output[generator_all], output[generator_1])
                # 其余两套机组总容量为 240MW，修正 generator_rest 的物理上界。
                output[generator_all] = np.minimum(
                    output[generator_all], output[generator_1] + 240.0
                )
        if not np.isfinite(output.to_numpy()).all():
            raise ValueError("预测结果包含非有限值")
