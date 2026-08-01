"""后期候选模型：CatBoost 与因果变化点近期窗口。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import ruptures as rpt
from catboost import CatBoostRegressor, Pool
from sklearn.impute import SimpleImputer

from gas_forecast.config import ForecastConfig
from gas_forecast.targets import target_columns


@dataclass
class CatBoostTargetState:
    imputer: SimpleImputer
    models: list[CatBoostRegressor]
    delta_lower: np.ndarray
    delta_upper: np.ndarray


def select_recent_training_start(
    series: pd.Series,
    *,
    candidate_days: tuple[int, ...] = (15, 30, 45, 60),
    penalty: float = 8.0,
) -> tuple[pd.Timestamp, dict[str, object]]:
    """只在当前训练历史中检测最后变化点，并回缩到预设窗口集合。"""

    clean = pd.to_numeric(series, errors="coerce").ffill().dropna()
    if len(clean) < 384:
        return clean.index.min(), {"method": "all_history", "reason": "有效样本不足"}
    standardized = ((clean - clean.median()) / max(float(clean.std()), 1e-6)).to_numpy()[:, None]
    changes = rpt.Pelt(model="rbf", min_size=96, jump=16).fit(standardized).predict(pen=penalty)
    last_position = changes[-2] if len(changes) >= 2 else 0
    change_start = clean.index[last_position]
    candidates = {
        f"recent_{days}d": clean.index.max() - pd.Timedelta(days=days)
        for days in candidate_days
    }
    candidates["last_change"] = change_start
    valid = {name: max(start, clean.index.min()) for name, start in candidates.items()}
    # 变化点过近时回退到15天，避免近期模型样本量失控。
    selected_name = "last_change"
    selected = valid[selected_name]
    if int((clean.index >= selected).sum()) < 384:
        selected_name = "recent_15d"
        selected = valid[selected_name]
    return selected, {
        "method": selected_name,
        "selected_start": str(selected),
        "last_change": str(change_start),
        "candidate_starts": {name: str(value) for name, value in valid.items()},
    }


class CatBoostDeltaForecaster:
    """浅层逐目标逐步长 CatBoost，仅作为独立 OOF 候选。"""

    version = "catboost"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.feature_columns_: list[str] = []
        self.states_: dict[str, CatBoostTargetState] = {}

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "CatBoostDeltaForecaster":
        self.feature_columns_ = sorted(features.columns)
        for target in self.config.targets:
            columns = target_columns(target, self.config.feature.horizons)
            valid = current[target].notna() & deltas[columns].notna().all(axis=1)
            x = features.loc[valid, self.feature_columns_]
            y = deltas.loc[valid, columns]
            if len(x) < 400:
                raise ValueError(f"{target} 的 CatBoost 有效样本不足 400 行")
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            matrix = imputer.fit_transform(x)
            split = min(max(300, int(len(x) * 0.85)), len(x) - 96)
            models: list[CatBoostRegressor] = []
            for step, column in enumerate(columns):
                future_absolute = current.loc[valid, target].to_numpy() + y[column].to_numpy()
                lower = max(float(np.nanquantile(np.abs(future_absolute), 0.05)), 1e-6)
                upper = max(float(np.nanquantile(np.abs(future_absolute), 0.95)), lower)
                weights = 1.0 / np.clip(np.abs(future_absolute), lower, upper)
                probe = CatBoostRegressor(
                    loss_function="MAE",
                    # 增量标签上的 MAPE 与比赛绝对负荷 MAPE 不同；
                    # 带绝对负荷权重的 MAE 才是可解释的早停近似。
                    eval_metric="MAE",
                    iterations=self.config.model.catboost_iterations,
                    depth=self.config.model.catboost_depth,
                    learning_rate=self.config.model.catboost_learning_rate,
                    random_seed=self.config.model.random_state + step,
                    has_time=True,
                    thread_count=self.config.model.tree_threads_per_worker,
                    verbose=False,
                    allow_writing_files=False,
                )
                probe.fit(
                    matrix[:split],
                    y[column].to_numpy()[:split],
                    sample_weight=weights[:split],
                    eval_set=Pool(
                        matrix[split:],
                        y[column].to_numpy()[split:],
                        weight=weights[split:],
                    ),
                    early_stopping_rounds=self.config.model.catboost_early_stopping_rounds,
                    verbose=False,
                )
                iterations = max(20, probe.get_best_iteration() + 1)
                model = CatBoostRegressor(
                    loss_function="MAE",
                    iterations=iterations,
                    depth=self.config.model.catboost_depth,
                    learning_rate=self.config.model.catboost_learning_rate,
                    random_seed=self.config.model.random_state + step,
                    has_time=True,
                    thread_count=self.config.model.tree_threads_per_worker,
                    verbose=False,
                    allow_writing_files=False,
                )
                model.fit(matrix, y[column].to_numpy(), sample_weight=weights, verbose=False)
                models.append(model)
            self.states_[target] = CatBoostTargetState(
                imputer=imputer,
                models=models,
                delta_lower=y.quantile(self.config.model.lower_quantile).to_numpy(),
                delta_upper=y.quantile(self.config.model.upper_quantile).to_numpy(),
            )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if not self.states_:
            raise RuntimeError("CatBoost 模型尚未训练")
        predictions: dict[str, np.ndarray] = {}
        for target in self.config.targets:
            state = self.states_[target]
            matrix = state.imputer.transform(features.reindex(columns=self.feature_columns_))
            delta = np.column_stack([model.predict(matrix) for model in state.models])
            delta = np.clip(delta, state.delta_lower, state.delta_upper)
            absolute = current[target].ffill().to_numpy()[:, None] + delta
            for step, horizon in enumerate(self.config.feature.horizons):
                predictions[f"{target}_t+{15 * horizon}_pred"] = absolute[:, step]
        return pd.DataFrame(predictions, index=features.index)
