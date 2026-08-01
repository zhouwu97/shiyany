"""Analog 与 Damped Trend generator_1 专家候选。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from gas_forecast.config import ForecastConfig
from gas_forecast.research_models import (
    _fit_generator_all_baseline,
    _predict_generator_all_baseline,
    apply_capacity_projection,
    merge_target_route_predictions,
)
from gas_forecast.targets import target_columns


def select_analog_features(features: pd.DataFrame) -> pd.DataFrame:
    """选择形状、煤气、气柜和守恒趋势组成的低维相似工况向量。"""

    prefixes = (
        "feat_generator_1_lag_",
        "feat_generator_1_diff_",
        "feat_generator_1_slope_",
        "feat_generator_1_ramp_",
        "feat_generator_1_ewma_",
        "feat_generator_rest_lag_",
        "feat_generator_rest_diff_",
        "feat_generator_rest_slope_",
        "feat_generator_use_",
        "feat_blast_balance",
        "feat_coke_balance",
        "feat_converter_balance",
        "feat_bf_surplus_proxy",
    )
    direct = {"generator_1", "feat_generator_rest", "feat_generator_gas_total"}
    columns = [
        column
        for column in features.columns
        if column in direct
        or column.startswith(prefixes)
        or "gas_holder" in column
    ]
    if not columns:
        raise ValueError("Analog 没有匹配到相似工况特征")
    # 保持固定列顺序并去掉完全重复字段，避免距离被重复列支配。
    return features.loc[:, list(dict.fromkeys(columns))]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = 0.5 * float(sorted_weights.sum())
    position = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


class AnalogGenerator1Forecaster:
    """只对 generator_1 使用 outer-train 历史相似工况的加权中位数。"""

    version = "generator1_analog"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.feature_columns_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.neighbors_: NearestNeighbors | None = None
        self.train_scaled_: np.ndarray | None = None
        self.future_deltas_: np.ndarray | None = None
        self.generator_all_model_: object | None = None

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "AnalogGenerator1Forecaster":
        analog = select_analog_features(features)
        columns = target_columns("generator_1", self.config.feature.horizons)
        valid = current["generator_1"].notna() & deltas[columns].notna().all(axis=1)
        if int(valid.sum()) < max(200, self.config.model.analog_k):
            raise ValueError("Analog outer-train 有效样本不足")
        train_x = analog.loc[valid]
        self.feature_columns_ = list(train_x.columns)
        self.imputer_ = SimpleImputer(strategy="median").fit(train_x)
        imputed = self.imputer_.transform(train_x)
        self.scaler_ = StandardScaler().fit(imputed)
        scaled = self.scaler_.transform(imputed)
        self.train_scaled_ = scaled
        k = min(int(self.config.model.analog_k), len(train_x))
        if k < 1:
            raise ValueError("Analog k 必须为正数")
        self.neighbors_ = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(scaled)
        self.future_deltas_ = deltas.loc[valid, columns].to_numpy(dtype=float)
        self.generator_all_model_ = _fit_generator_all_baseline(
            self.config,
            features,
            deltas,
            current,
        )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if (
            self.imputer_ is None
            or self.scaler_ is None
            or self.neighbors_ is None
            or self.train_scaled_ is None
            or self.future_deltas_ is None
            or self.generator_all_model_ is None
        ):
            raise RuntimeError("Analog 尚未训练")
        analog = select_analog_features(features).reindex(columns=self.feature_columns_)
        query = self.scaler_.transform(self.imputer_.transform(analog))
        distances, indices = self.neighbors_.kneighbors(query)
        output: dict[str, np.ndarray] = {}
        anchor = current["generator_1"].ffill().to_numpy(dtype=float)
        deltas = np.zeros((len(query), len(self.config.feature.horizons)), dtype=float)
        for row in range(len(query)):
            weights = 1.0 / np.maximum(distances[row], 1e-6)
            for step in range(deltas.shape[1]):
                neighbor_indices = indices[row]
                if self.config.model.analog_mode == "weighted_median":
                    deltas[row, step] = _weighted_median(
                        self.future_deltas_[neighbor_indices, step], weights
                    )
                elif self.config.model.analog_mode == "local_ridge":
                    local = Ridge(
                        alpha=float(self.config.model.analog_local_ridge_alpha),
                    )
                    local.fit(
                        self.train_scaled_[neighbor_indices],
                        self.future_deltas_[neighbor_indices, step],
                        sample_weight=weights,
                    )
                    deltas[row, step] = float(local.predict(query[row : row + 1])[0])
                else:
                    raise ValueError(
                        f"不支持的 Analog 模式: {self.config.model.analog_mode}"
                    )
        for step, horizon in enumerate(self.config.feature.horizons):
            output[f"generator_1_t+{15 * horizon}_pred"] = anchor + deltas[:, step]
        generator_1 = apply_capacity_projection(pd.DataFrame(output, index=features.index)).loc[
            :, [column for column in output]
        ]
        generator_all = _predict_generator_all_baseline(
            self.generator_all_model_, features, current
        )
        return merge_target_route_predictions(self.config, generator_1, generator_all)


class DampedTrendGenerator1Forecaster:
    """用近期斜率外推 generator_1 的低自由度 specialist。"""

    version = "generator1_damped_trend"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.generator_all_model_: object | None = None

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "DampedTrendGenerator1Forecaster":
        self.generator_all_model_ = _fit_generator_all_baseline(
            self.config,
            features,
            deltas,
            current,
        )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if self.generator_all_model_ is None:
            raise RuntimeError("Damped Trend 尚未训练")
        window = int(self.config.model.damped_trend_window)
        damping = float(self.config.model.damped_trend_damping)
        lag_column = f"feat_generator_1_lag_{max(window - 1, 1)}"
        if lag_column in features:
            slope = (
                features["generator_1"].to_numpy(dtype=float)
                - features[lag_column].to_numpy(dtype=float)
            ) / max(window - 1, 1)
        else:
            slope = np.zeros(len(features), dtype=float)
        anchor = current["generator_1"].ffill().to_numpy(dtype=float)
        output: dict[str, np.ndarray] = {}
        for horizon in self.config.feature.horizons:
            steps = np.arange(1, horizon + 1, dtype=float)
            multiplier = float(np.sum(damping ** (steps - 1)))
            output[f"generator_1_t+{15 * horizon}_pred"] = anchor + slope * multiplier
        generator_1 = apply_capacity_projection(pd.DataFrame(output, index=features.index)).loc[
            :, list(output)
        ]
        generator_all = _predict_generator_all_baseline(
            self.generator_all_model_, features, current
        )
        return merge_target_route_predictions(self.config, generator_1, generator_all)
