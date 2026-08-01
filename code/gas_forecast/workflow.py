"""训练、序列化与滚动推理公共流程。"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from gas_forecast.config import (
    ForecastConfig,
    horizon_ridge_forecast_config,
    legacy_forecast_config,
    research_feature_superset,
    research_forecast_config,
)
from gas_forecast.data import align_tables, combine_context
from gas_forecast.features import (
    build_causal_features,
    load_price_schedule,
)
from gas_forecast.model_v1 import RidgeDeltaForecaster
from gas_forecast.model_horizon import HorizonSpecificRidgeForecaster
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster
from gas_forecast.model_routed import RoutedLegacyForecaster
from gas_forecast.research_models import (
    DirectIncrementalBlendForecaster,
    Generator1CatBoostForecaster,
    Generator1HorizonRidgeForecaster,
    Generator1IncrementalPathForecaster,
    Generator1LightGBMForecaster,
    Generator1StateExpertForecaster,
    PathSmoothedGenerator1HorizonRidgeForecaster,
)
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
    if version == "horizon_ridge":
        return horizon_ridge_forecast_config()
    if version in {
        "generator1_horizon",
        "generator1_catboost",
        "generator1_lgb",
        "generator1_state_expert",
        "generator1_incremental",
        "generator1_direct_incremental",
        "generator1_path",
    }:
        return research_forecast_config()
    return legacy_forecast_config()


def train_model(
    data_dir: str | Path,
    output: str | Path,
    version: str = "v1",
    *,
    route: dict[str, object] | None = None,
    n_jobs: int = 4,
    config: ForecastConfig | None = None,
) -> object:
    config = resolve_training_config(version, config)
    dataset = align_tables(data_dir, config.feature.frequency)
    price_path = _find_price(data_dir)
    price = load_price_schedule(price_path) if price_path else None
    feature_config = (
        research_feature_superset(config.feature)
        if version.startswith("generator1_")
        else config.feature
    )
    features = build_causal_features(dataset.frame, feature_config, price)
    deltas = build_delta_targets(dataset.frame, config.targets, config.feature.horizons)
    if version == "routed":
        if route is None:
            raise ValueError("训练 routed 模型必须提供冻结路由")
        model = RoutedLegacyForecaster(route, config, n_jobs=n_jobs)
    else:
        research_models = {
            "generator1_horizon": Generator1HorizonRidgeForecaster,
            "generator1_catboost": Generator1CatBoostForecaster,
            "generator1_lgb": Generator1LightGBMForecaster,
            "generator1_state_expert": Generator1StateExpertForecaster,
            "generator1_incremental": Generator1IncrementalPathForecaster,
            "generator1_direct_incremental": DirectIncrementalBlendForecaster,
            "generator1_path": PathSmoothedGenerator1HorizonRidgeForecaster,
        }
        if version == "v1":
            model = RidgeDeltaForecaster(config)
        elif version == "horizon_ridge":
            model = HorizonSpecificRidgeForecaster(config)
        elif version in research_models:
            model = research_models[version](config)
        else:
            model = GasAwareEnsembleForecaster(version, config)
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
    model = joblib.load(model_path)
    train = align_tables(train_dir, model.config.feature.frequency).frame
    test = align_tables(test_dir, model.config.feature.frequency).frame
    context = combine_context(train, test)
    price_path = _find_price(train_dir)
    price = load_price_schedule(price_path) if price_path else None
    feature_config = (
        research_feature_superset(model.config.feature)
        if str(getattr(model, "version", "")).startswith("generator1_")
        else model.config.feature
    )
    features = build_causal_features(context, feature_config, price)
    test_features = features.reindex(test.index)
    predictions = model.predict(
        test_features,
        test_features.loc[:, list(model.config.targets)],
    )
    return test_features, predictions
