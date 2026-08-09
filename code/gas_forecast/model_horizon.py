"""目标时刻对齐的逐目标逐步长 Ridge。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig
from gas_forecast.model_v1 import _mape, make_ridge_pipeline, make_weighted_lad_pipeline
from gas_forecast.targets import target_columns


@dataclass
class HorizonTargetState:
    """单个目标的逐步长模型状态。"""

    models: list
    weights: np.ndarray
    delta_lower: np.ndarray
    delta_upper: np.ndarray
    alphas: np.ndarray


class HorizonSpecificRidgeForecaster:
    """每个目标和预测步长独立建模，避免共享不匹配的周期特征。"""

    version = "horizon_ridge"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.feature_columns_: list[str] = []
        self.states_: dict[str, HorizonTargetState] = {}

    def _alpha_for(self, target: str, horizon: int) -> float:
        """解析 generator_1 的短/长步长正则组，其余目标保持基础正则。"""

        alpha = self.config.model.ridge_alpha
        if target != "generator_1":
            return alpha
        split = len(self.config.feature.horizons) // 2
        position = self.config.feature.horizons.index(horizon)
        configured = (
            self.config.model.generator1_short_alpha
            if position < split
            else self.config.model.generator1_long_alpha
        )
        if configured is None:
            return alpha
        if configured <= 0:
            raise ValueError("Ridge alpha 必须为正数")
        return float(configured)

    def _apply_hard_recency(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        anchor: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """仅在明确要求时截取固定近期窗口。"""

        mode = self.config.model.ridge_recency_mode
        if mode not in {"all", "hard", "exp", "grouped_exp"}:
            raise ValueError(f"不支持的 Ridge 时间漂移模式: {mode}")
        if mode != "hard":
            return x, y, anchor
        days = self.config.model.ridge_hard_window_days
        if days is None or days <= 0:
            raise ValueError("hard recency 需要正数 ridge_hard_window_days")
        recent = x.index >= x.index.max() - pd.Timedelta(days=days)
        if int(recent.sum()) < 200:
            raise ValueError(f"{days} 天硬窗口的逐步长 Ridge 有效样本不足 200 行")
        return x.loc[recent], y.loc[recent], anchor.loc[recent]

    def _sample_weight(
        self,
        index: pd.DatetimeIndex,
        future_absolute: np.ndarray,
        *,
        target: str | None = None,
        horizon: int | None = None,
    ) -> np.ndarray:
        """组合指数时间衰减和未来绝对量权重，并归一化为均值一。"""

        weights = np.ones(len(index), dtype=float)
        if self.config.model.ridge_recency_mode in {"exp", "grouped_exp"}:
            if self.config.model.ridge_recency_mode == "grouped_exp":
                if target != "generator_1" or horizon is None:
                    half_life = self.config.model.ridge_half_life_days
                else:
                    split = len(self.config.feature.horizons) // 2
                    position = self.config.feature.horizons.index(horizon)
                    half_life = (
                        self.config.model.ridge_short_half_life_days
                        if position < split
                        else self.config.model.ridge_long_half_life_days
                    )
            else:
                half_life = self.config.model.ridge_half_life_days
            if half_life is None or half_life <= 0:
                raise ValueError("指数 recency 需要正数半衰期配置")
            age = (index.max() - index).total_seconds() / 86_400.0
            weights *= np.exp(-np.log(2.0) * np.asarray(age, dtype=float) / half_life)

        magnitude = np.abs(np.asarray(future_absolute, dtype=float))
        lower = max(float(np.nanquantile(magnitude, 0.05)), 1e-6)
        upper = max(float(np.nanquantile(magnitude, 0.95)), lower)
        clipped = np.clip(magnitude, lower, upper)
        mode = self.config.model.ridge_magnitude_weighting
        if mode == "uniform":
            pass
        elif mode in {"inverse_absolute", "inverse_abs"}:
            weights /= clipped
        elif mode in {"inverse_squared", "inverse_abs_squared"}:
            weights /= clipped**2
        else:
            raise ValueError(f"不支持的 Ridge 绝对量权重: {mode}")
        if not np.isfinite(weights).all() or float(weights.mean()) <= 0:
            raise ValueError("Ridge 样本权重无效")
        return weights / weights.mean()

    def _make_pipeline(self, alpha: float):
        if self.config.model.ridge_loss == "ridge":
            return make_ridge_pipeline(alpha), "ridge"
        if self.config.model.ridge_loss == "weighted_lad":
            return make_weighted_lad_pipeline(self.config.model.weighted_lad_alpha), "lad"
        raise ValueError(f"不支持的线性损失: {self.config.model.ridge_loss}")

    @staticmethod
    def _fit_pipeline(
        pipeline: object,
        step_name: str,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray,
    ) -> None:
        pipeline.fit(x, y, **{f"{step_name}__sample_weight": sample_weight})

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
            x, y, anchor = self._apply_hard_recency(x, y, anchor)
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
            alphas = []
            for step, column in enumerate(columns):
                horizon = horizons[step]
                alpha = self._alpha_for(target, horizon)
                future_absolute = anchor.to_numpy(dtype=float) + y[column].to_numpy(dtype=float)
                development_weight = self._sample_weight(
                    development_x.index,
                    future_absolute[:development_end],
                    target=target,
                    horizon=horizon,
                )
                probe, probe_step = self._make_pipeline(alpha)
                self._fit_pipeline(
                    probe,
                    probe_step,
                    development_x,
                    y.iloc[:development_end][column],
                    development_weight,
                )
                probe_prediction = calibration_anchor + probe.predict(calibration_x)
                scores = [
                    _mape(
                        calibration_actual[:, step],
                        calibration_anchor + weight * (probe_prediction - calibration_anchor),
                    )
                    for weight in grid
                ]
                weights.append(float(grid[int(np.argmin(scores))]))
                model, model_step = self._make_pipeline(alpha)
                self._fit_pipeline(
                    model,
                    model_step,
                    x,
                    y[column],
                    self._sample_weight(
                        x.index,
                        future_absolute,
                        target=target,
                        horizon=horizon,
                    ),
                )
                models.append(model)
                alphas.append(alpha)
                lowers.append(float(y[column].quantile(self.config.model.lower_quantile)))
                uppers.append(float(y[column].quantile(self.config.model.upper_quantile)))
            self.states_[target] = HorizonTargetState(
                models=models,
                weights=np.asarray(weights, dtype=float),
                delta_lower=np.asarray(lowers, dtype=float),
                delta_upper=np.asarray(uppers, dtype=float),
                alphas=np.asarray(alphas, dtype=float),
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
        horizons = sorted(
            int(column.rsplit("_t+", 1)[1].removesuffix("_pred"))
            for column in output.columns
            if column.startswith("generator_1_t+") or column.startswith("generator_all_t+")
        )
        for horizon in horizons:
            gen1 = f"generator_1_t+{horizon}_pred"
            total = f"generator_all_t+{horizon}_pred"
            if gen1 in output:
                output[gen1] = output[gen1].clip(0.0, 200.0)
            if total in output:
                output[total] = output[total].clip(0.0, 440.0)
            if gen1 in output and total in output:
                output[total] = np.maximum(output[total], output[gen1])
                output[total] = np.minimum(output[total], output[gen1] + 240.0)
        if not np.isfinite(output.to_numpy()).all():
            raise ValueError("逐步长 Ridge 预测包含非有限值")
