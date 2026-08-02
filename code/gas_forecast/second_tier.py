"""前三条主线完成后才允许启用的初赛第二梯队异构模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CATBOOST_PRESETS: Mapping[str, dict[str, float | int]] = {
    "small": {"depth": 4, "learning_rate": 0.03, "iterations": 500},
    "medium": {"depth": 6, "learning_rate": 0.03, "iterations": 700},
    "more_nonlinear": {"depth": 7, "learning_rate": 0.02, "iterations": 800},
}


class AbsoluteCatBoostMAPE:
    """只预测 generator_1 未来绝对量的三套冻结 CatBoost 候选。"""

    def __init__(self, preset: str, *, random_state: int = 20250731, threads: int = 1) -> None:
        if preset not in CATBOOST_PRESETS:
            raise ValueError(f"CatBoost preset 只允许: {sorted(CATBOOST_PRESETS)}")
        self.preset = preset
        self.random_state = random_state
        self.threads = threads
        self.feature_columns_: list[str] = []
        self.models_: list[object] = []

    def fit(self, features: pd.DataFrame, absolute_targets: pd.DataFrame) -> "AbsoluteCatBoostMAPE":
        from catboost import CatBoostRegressor

        if len(absolute_targets.columns) != 8:
            raise ValueError("Absolute CatBoost-MAPE 必须一次固定训练 8 个 horizon")
        valid = absolute_targets.notna().all(axis=1) & features.notna().any(axis=1)
        if int(valid.sum()) < 400:
            raise ValueError("Absolute CatBoost-MAPE 有效训练样本不足 400 行")
        self.feature_columns_ = list(features.columns)
        x = features.loc[valid].replace([np.inf, -np.inf], np.nan)
        parameters = CATBOOST_PRESETS[self.preset]
        self.models_ = []
        for step, column in enumerate(absolute_targets.columns):
            model = CatBoostRegressor(
                loss_function="MAPE",
                eval_metric="MAPE",
                depth=int(parameters["depth"]),
                learning_rate=float(parameters["learning_rate"]),
                iterations=int(parameters["iterations"]),
                random_seed=self.random_state + step,
                thread_count=self.threads,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(x, absolute_targets.loc[valid, column])
            self.models_.append(model)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if len(self.models_) != 8:
            raise RuntimeError("Absolute CatBoost-MAPE 尚未训练")
        x = features.reindex(columns=self.feature_columns_).replace([np.inf, -np.inf], np.nan)
        return np.column_stack([model.predict(x) for model in self.models_])


@dataclass(frozen=True)
class RecursiveARXSpec:
    current_column: str
    lag_columns: tuple[str, ...]
    future_price_columns: tuple[str, ...]
    static_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.future_price_columns) != 8:
            raise ValueError("Recursive ARX 必须登记未来 8 步官方价格")
        if not self.lag_columns:
            raise ValueError("Recursive ARX 至少需要一个历史 lag")


class RecursiveARX:
    """一步 ARX 递归八次；未知未来生产量只使用自身预测回填。"""

    def __init__(self, spec: RecursiveARXSpec, *, alpha: float = 20.0) -> None:
        self.spec = spec
        self.alpha = alpha
        self.model_: Pipeline | None = None
        self.training_columns_: list[str] = []

    def _one_step_frame(
        self,
        frame: pd.DataFrame,
        *,
        current: np.ndarray | None = None,
        lags: np.ndarray | None = None,
        price: np.ndarray | None = None,
    ) -> pd.DataFrame:
        current_values = (
            frame[self.spec.current_column].to_numpy(float) if current is None else current
        )
        lag_values = (
            frame.loc[:, list(self.spec.lag_columns)].to_numpy(float) if lags is None else lags
        )
        price_values = (
            frame[self.spec.future_price_columns[0]].to_numpy(float) if price is None else price
        )
        output = pd.DataFrame(
            np.column_stack([current_values, lag_values, price_values]),
            index=frame.index,
            columns=["arx_current", *[f"arx_lag_{i + 1}" for i in range(lag_values.shape[1])], "arx_price"],
        )
        for column in self.spec.static_columns:
            output[column] = pd.to_numeric(frame[column], errors="coerce")
        return output

    def fit(self, features: pd.DataFrame, next_actual: pd.Series) -> "RecursiveARX":
        required = {
            self.spec.current_column,
            *self.spec.lag_columns,
            *self.spec.future_price_columns,
            *self.spec.static_columns,
        }
        missing = sorted(required.difference(features.columns))
        if missing:
            raise ValueError(f"Recursive ARX 输入缺少字段: {missing}")
        x = self._one_step_frame(features)
        valid = next_actual.notna() & np.isfinite(next_actual.to_numpy(float))
        if int(valid.sum()) < 200:
            raise ValueError("Recursive ARX 有效一步训练样本不足 200 行")
        self.training_columns_ = list(x.columns)
        self.model_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )
        self.model_.fit(x.loc[valid], next_actual.loc[valid])
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Recursive ARX 尚未训练")
        current = features[self.spec.current_column].to_numpy(float).copy()
        lags = features.loc[:, list(self.spec.lag_columns)].to_numpy(float).copy()
        predictions = np.empty((len(features), 8), dtype=float)
        for step, price_column in enumerate(self.spec.future_price_columns):
            price = features[price_column].to_numpy(float)
            one_step = self._one_step_frame(features, current=current, lags=lags, price=price)
            one_step = one_step.reindex(columns=self.training_columns_)
            next_prediction = self.model_.predict(one_step)
            predictions[:, step] = next_prediction
            if lags.shape[1] > 1:
                lags[:, 1:] = lags[:, :-1]
            lags[:, 0] = current
            current = next_prediction
        return predictions


def fixed_recursive_blends(
    direct: np.ndarray,
    recursive: np.ndarray,
    weights: Sequence[float] = (0.05, 0.10, 0.20),
) -> dict[str, np.ndarray]:
    """只生成计划允许的 95/5、90/10、80/20 三条递归融合路径。"""

    direct_values = np.asarray(direct, dtype=float)
    recursive_values = np.asarray(recursive, dtype=float)
    if direct_values.shape != recursive_values.shape:
        raise ValueError("direct 与 recursive 预测形状不一致")
    return {
        f"recursive_blend_{int(weight * 100):02d}": (
            (1.0 - weight) * direct_values + weight * recursive_values
        )
        for weight in weights
    }
