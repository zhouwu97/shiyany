"""训练、序列化与滚动推理公共流程。"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from gas_forecast.config import ForecastConfig, horizon_ridge_forecast_config, legacy_forecast_config
from gas_forecast.data import align_tables, combine_context
from gas_forecast.features import (
    build_causal_features,
    load_price_schedule,
)
from gas_forecast.model_v1 import RidgeDeltaForecaster
from gas_forecast.model_horizon import HorizonSpecificRidgeForecaster
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster
from gas_forecast.model_routed import RoutedLegacyForecaster
from gas_forecast.targets import build_delta_targets


def _find_price(data_dir: str | Path) -> Path | None:
    matches = sorted(Path(data_dir).glob("*price*.xlsx"))
    return matches[0] if matches else None


def resolve_training_config(
    version: str, config: ForecastConfig | None = None
) -> ForecastConfig:
    """统一训练入口配置，确保选型、重训和推理使用同一对象。"""

    if config is not None:
        return config
    return horizon_ridge_forecast_config() if version == "horizon_ridge" else legacy_forecast_config()


def train_model(
    data_dir: str | Path,
    output: str | Path,
    version: str = "v1",
    *,
    route: dict[str, object] | None = None,
    n_jobs: int = 4,
    config: ForecastConfig | None = None,
) -> RidgeDeltaForecaster | HorizonSpecificRidgeForecaster | GasAwareEnsembleForecaster | RoutedLegacyForecaster:
    config = resolve_training_config(version, config)
    dataset = align_tables(data_dir, config.feature.frequency)
    price_path = _find_price(data_dir)
    price = load_price_schedule(price_path) if price_path else None
    features = build_causal_features(dataset.frame, config.feature, price)
    deltas = build_delta_targets(dataset.frame, config.targets, config.feature.horizons)
    if version == "routed":
        if route is None:
            raise ValueError("训练 routed 模型必须提供冻结路由")
        model = RoutedLegacyForecaster(route, config, n_jobs=n_jobs)
    else:
        model = (
            RidgeDeltaForecaster(config)
            if version == "v1"
            else HorizonSpecificRidgeForecaster(config)
            if version == "horizon_ridge"
            else GasAwareEnsembleForecaster(version, config)
        )
    model.fit(
        features,
        deltas,
        dataset.frame.loc[:, list(config.targets)],
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


def train_v1(data_dir: str | Path, output: str | Path) -> RidgeDeltaForecaster:
    return train_model(data_dir, output, "v1")


def predict_rolling(
    train_dir: str | Path,
    test_dir: str | Path,
    model_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model: RidgeDeltaForecaster | HorizonSpecificRidgeForecaster | GasAwareEnsembleForecaster | RoutedLegacyForecaster = joblib.load(
        model_path
    )
    train = align_tables(train_dir, model.config.feature.frequency).frame
    test = align_tables(test_dir, model.config.feature.frequency).frame
    context = combine_context(train, test)
    price_path = _find_price(train_dir)
    price = load_price_schedule(price_path) if price_path else None
    features = build_causal_features(context, model.config.feature, price)
    test_features = features.reindex(test.index)
    predictions = model.predict(
        test_features,
        test_features.loc[:, list(model.config.targets)],
    )
    return test_features, predictions
