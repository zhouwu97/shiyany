from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig, ForecastConfig
from gas_forecast.features import build_causal_features
from gas_forecast.model_v1 import RidgeDeltaForecaster
from gas_forecast.targets import build_delta_targets


def test_v1_predicts_complete_constrained_wide_table() -> None:
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
    config = ForecastConfig(feature=FeatureConfig(rolling_windows=(4, 8, 16)))
    features = build_causal_features(frame, config.feature)
    deltas = build_delta_targets(frame, config.targets, config.feature.horizons)
    model = RidgeDeltaForecaster(config).fit(features, deltas, frame[list(config.targets)])

    predicted = model.predict(features.iloc[-10:], frame[list(config.targets)].iloc[-10:])

    assert predicted.shape == (10, 16)
    assert np.isfinite(predicted.to_numpy()).all()
    for horizon in config.feature.horizons:
        assert (
            predicted[f"generator_all_t+{15 * horizon}_pred"]
            >= predicted[f"generator_1_t+{15 * horizon}_pred"]
        ).all()
