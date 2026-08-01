"""冻结目标/步长路由的多模型推理包装器。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gas_forecast.config import ForecastConfig, legacy_forecast_config
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster
from gas_forecast.model_v1 import RidgeDeltaForecaster


def _fit_legacy_model(
    version: str,
    config: ForecastConfig,
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    current: pd.DataFrame,
):
    model = (
        RidgeDeltaForecaster(config)
        if version == "v1"
        else GasAwareEnsembleForecaster(version, config)
    )
    return version, model.fit(features, deltas, current)


class RoutedLegacyForecaster:
    """按 OOF 冻结路由加载必要旧模型；Persistence 无需训练。"""

    version = "routed_legacy"

    def __init__(
        self,
        route: Mapping[str, object],
        config: ForecastConfig | None = None,
        *,
        n_jobs: int = 4,
    ) -> None:
        self.config = config or legacy_forecast_config()
        self.route = dict(route)
        self.n_jobs = n_jobs
        self.models_: dict[str, object] = {}

    def _required_versions(self) -> tuple[str, ...]:
        selected = {str(self.route["global"]["selected"])}
        selected.update(
            str(item["selected"]) for item in self.route.get("targets", {}).values()
        )
        selected.update(
            str(item["selected"]) for item in self.route.get("cells", {}).values()
        )
        versions = sorted(value.removesuffix("_pred") for value in selected)
        invalid = sorted(set(versions).difference({"persistence", "v1", "v2", "v25", "v3"}))
        if invalid:
            raise ValueError(f"路由含不支持的旧模型: {invalid}")
        return tuple(version for version in versions if version != "persistence")

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "RoutedLegacyForecaster":
        versions = self._required_versions()
        fitted = Parallel(n_jobs=min(self.n_jobs, max(1, len(versions))))(
            delayed(_fit_legacy_model)(version, self.config, features, deltas, current)
            for version in versions
        )
        self.models_ = dict(fitted)
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        model_predictions = {
            f"{version}_pred": model.predict(features, current)
            for version, model in self.models_.items()
        }
        output: dict[str, np.ndarray] = {}
        cells = self.route.get("cells", {})
        global_model = str(self.route["global"]["selected"])
        for target in self.config.targets:
            for horizon in self.config.feature.horizons:
                minutes = 15 * horizon
                key = f"{target}|{minutes}"
                selected = str(cells.get(key, {}).get("selected", global_model))
                output_column = f"{target}_t+{minutes}_pred"
                if selected == "persistence_pred":
                    output[output_column] = current[target].ffill().to_numpy(dtype=float)
                else:
                    output[output_column] = model_predictions[selected][output_column].to_numpy()
        result = pd.DataFrame(output, index=features.index)
        for target, upper in (("generator_1", 200.0), ("generator_all", 440.0)):
            columns = [column for column in result if column.startswith(f"{target}_")]
            result[columns] = result[columns].clip(lower=0.0, upper=upper)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("路由预测包含非有限值")
        return result
