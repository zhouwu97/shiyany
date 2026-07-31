import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig, ForecastConfig
from gas_forecast.features import build_causal_features
from gas_forecast.model_v1 import RidgeDeltaForecaster
from gas_forecast.targets import build_delta_targets


def test_v1_repeated_training_is_reproducible() -> None:
    rows = 360
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows)
    frame = pd.DataFrame(
        {
            "generator_1": 100 + 5 * np.sin(phase / 11),
            "generator_all": 230 + 8 * np.sin(phase / 13),
            "generator_use_blast_furnace_gas": 600_000 + 500 * np.cos(phase / 9),
        },
        index=index,
    )
    config = ForecastConfig(
        feature=FeatureConfig(horizons=(1, 2), lags=(1, 2, 4), rolling_windows=(4, 8))
    )
    features = build_causal_features(frame, config.feature)
    deltas = build_delta_targets(frame, config.targets, config.feature.horizons)
    current = frame[list(config.targets)]

    first = RidgeDeltaForecaster(config).fit(features, deltas, current)
    second = RidgeDeltaForecaster(config).fit(features, deltas, current)
    first_prediction = first.predict(features.iloc[-16:], current.iloc[-16:])
    second_prediction = second.predict(features.iloc[-16:], current.iloc[-16:])

    np.testing.assert_allclose(first_prediction, second_prediction, rtol=0.0, atol=1e-12)
