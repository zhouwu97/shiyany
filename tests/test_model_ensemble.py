from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.config import FeatureConfig, ForecastConfig, ModelConfig
from gas_forecast.features import build_causal_features, build_delta_targets
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_enhanced_models_produce_finite_predictions(version: str) -> None:
    rows = 520
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows)
    state = np.where((phase // 60) % 2 == 0, 0.0, 12.0)
    frame = pd.DataFrame(
        {
            "generator_1": 100 + state + 4 * np.sin(phase / 12),
            "generator_all": 230 + 1.5 * state + 7 * np.sin(phase / 15),
            "generator_use_blast_furnace_gas": 600_000 + state * 1_000,
            "generator_use_coke_gas": 10_000 + 100 * np.sin(phase / 7),
            "generator_use_converter_gas": 20_000 + 200 * np.cos(phase / 9),
            "blast_furnace_gas_holder_2": 100_000 + 500 * np.sin(phase / 20),
        },
        index=index,
    )
    config = ForecastConfig(
        feature=FeatureConfig(horizons=(1, 2), lags=(1, 2, 4), rolling_windows=(4, 8)),
        model=ModelConfig(
            recent_days=3,
            calibration_fraction=0.2,
            lgb_n_estimators=12,
            state_components=3,
        ),
    )
    features = build_causal_features(frame, config.feature)
    deltas = build_delta_targets(frame, config.targets, config.feature.horizons)
    model = GasAwareEnsembleForecaster(version, config).fit(
        features, deltas, frame[list(config.targets)]
    )

    predicted = model.predict(features.iloc[-12:], frame[list(config.targets)].iloc[-12:])

    assert predicted.shape == (12, 4)
    assert np.isfinite(predicted.to_numpy()).all()

