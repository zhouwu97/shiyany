"""Strict C0 后冻结的 R75 与 LGB 小权重生产组合。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gas_forecast.model_ensemble import BRANCH_NAMES, GasAwareEnsembleForecaster
from gas_forecast.model_routed import RoutedLegacyForecaster
from gas_forecast.research_models import apply_capacity_projection


class AggressiveR75LGBForecaster:
    """复用冻结 C0/E21，仅替换 generator_1 并保持生产容量约束。"""

    version = "generator1_aggressive_r75_lgb"

    def __init__(
        self,
        c0_model: RoutedLegacyForecaster,
        e21_model: Any,
        *,
        crossing_minutes: int = 75,
        lgb_weight: float = 0.20,
    ) -> None:
        if crossing_minutes not in {75, 90, 105}:
            raise ValueError("crossing_minutes 只允许冻结的 75/90/105")
        if lgb_weight not in {0.05, 0.10, 0.15, 0.20, 0.30}:
            raise ValueError("lgb_weight 必须来自冻结 Diversity 网格")
        if "v2" not in c0_model.models_:
            raise ValueError("Strict C0 模型缺少可复用的 V2 分支")
        if not hasattr(e21_model, "predict_generator1_only"):
            raise TypeError("E21 模型缺少 generator_1 单目标预测接口")
        if tuple(c0_model.config.feature.horizons) != tuple(
            e21_model.config.feature.horizons
        ):
            raise ValueError("C0 与 E21 的 horizon 配置不一致")
        self.c0_model = c0_model
        self.e21_model = e21_model
        self.crossing_minutes = crossing_minutes
        self.lgb_weight = lgb_weight
        self.config = e21_model.config

    def _lgb_generator1(self, features: pd.DataFrame, current: pd.DataFrame) -> np.ndarray:
        v2_model = self.c0_model.models_["v2"]
        if not isinstance(v2_model, GasAwareEnsembleForecaster):
            raise TypeError("Strict C0 的 V2 子模型类型不支持分支提取")
        state = v2_model.ensemble_states_.get("generator_1")
        if state is None:
            raise RuntimeError("V2 子模型缺少 generator_1 ensemble state")
        x = features.reindex(columns=v2_model.feature_columns_)
        anchor = current["generator_1"].ffill().to_numpy(dtype=float)
        branches = v2_model._predict_branches(state.branches, x, anchor)
        return branches[:, BRANCH_NAMES.index("lgb_residual"), :]

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        """执行 C0、R75、LGB 20% 融合，并按正式推理语义完成容量投影。"""

        c0 = self.c0_model.predict(features, current)
        e21 = self.e21_model.predict_generator1_only(features, current)
        lgb = self._lgb_generator1(features, current)
        horizons = tuple(self.config.feature.horizons)
        generator1_columns = [
            f"generator_1_t+{15 * horizon}_pred" for horizon in horizons
        ]
        baseline = c0.loc[:, generator1_columns].to_numpy(dtype=float)
        e21_values = e21.loc[:, generator1_columns].to_numpy(dtype=float)
        for position, horizon in enumerate(horizons):
            if 15 * horizon >= self.crossing_minutes:
                baseline[:, position] = e21_values[:, position]
        blended = (1.0 - self.lgb_weight) * baseline + self.lgb_weight * lgb
        output = c0.copy()
        output.loc[:, generator1_columns] = blended
        return apply_capacity_projection(output)
