from __future__ import annotations

from gas_forecast.config import ForecastConfig, horizon_ridge_forecast_config, legacy_forecast_config
from gas_forecast.workflow import resolve_training_config


def test_training_config_is_explicit_and_stable_across_retrain() -> None:
    custom = ForecastConfig()
    assert resolve_training_config("v1", custom) is custom


def test_horizon_ridge_default_config_enables_target_alignment() -> None:
    assert resolve_training_config("v1").feature == legacy_forecast_config().feature
    assert resolve_training_config("horizon_ridge").feature == horizon_ridge_forecast_config().feature
    assert resolve_training_config("horizon_ridge").feature.enable_target_aligned_features is True
