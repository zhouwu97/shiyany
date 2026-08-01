from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig, ForecastConfig
from gas_forecast.features import build_causal_features
from gas_forecast.model_horizon import HorizonSpecificRidgeForecaster
from gas_forecast.targets import build_delta_targets


def test_horizon_specific_ridge_fits_and_enforces_capacity_constraints() -> None:
    rows = 420
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    signal = np.sin(np.arange(rows) / 20)
    frame = pd.DataFrame(
        {
            "generator_1": 100 + 10 * signal,
            "generator_all": 220 + 15 * signal,
            "generator_use_blast_furnace_gas": 600_000 + 1000 * signal,
        },
        index=index,
    )
    feature_config = FeatureConfig(
        horizons=(1, 2),
        rolling_windows=(4, 8),
        enable_target_aligned_features=True,
    )
    config = ForecastConfig(feature=feature_config)
    features = build_causal_features(frame, feature_config)
    deltas = build_delta_targets(frame, config.targets, feature_config.horizons)
    model = HorizonSpecificRidgeForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    )

    predicted = model.predict(features.iloc[-10:], frame[list(config.targets)].iloc[-10:])

    assert predicted.shape == (10, 4)
    assert np.isfinite(predicted.to_numpy()).all()
    for horizon in feature_config.horizons:
        gen1 = predicted[f"generator_1_t+{15 * horizon}_pred"]
        total = predicted[f"generator_all_t+{15 * horizon}_pred"]
        assert (total >= gen1).all()
        assert (total - gen1 <= 240.0 + 1e-9).all()
