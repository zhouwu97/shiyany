from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig, ForecastConfig, ModelConfig
from gas_forecast.features import build_causal_features
from gas_forecast.gas_stage import GasTrajectoryForecaster
from gas_forecast.targets import build_delta_targets


def test_gas_trajectory_stage_two_is_trained_from_oof_resources() -> None:
    rows = 1050
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100 + np.sin(phase / 10),
            "generator_all": 220 + 2 * np.sin(phase / 12),
            "generator_use_blast_furnace_gas": 500_000 + phase,
            "generator_use_coke_gas": 10_000 + phase,
            "generator_use_converter_gas": 20_000 + phase,
            "blast_furnace_1": 600_000 + phase,
            "blast_furnace_2": 500_000 + phase,
            "blast_furnace_4": 400_000 + phase,
            "blast_furnace_5": 300_000 + phase,
            "coke_oven_1": 50_000 + phase,
            "converter_1": 30_000 + phase,
            "blast_furnace_gas_holder_2": 100_000 + phase,
        },
        index=index,
    )
    config = ForecastConfig(
        feature=FeatureConfig(horizons=(1, 2), lags=(1, 2), rolling_windows=(4,)),
        model=ModelConfig(inner_folds=2),
    )
    features = build_causal_features(frame, config.feature)
    deltas = build_delta_targets(frame, config.targets, config.feature.horizons)

    model = GasTrajectoryForecaster(config).fit(
        frame, features, deltas, frame[list(config.targets)]
    )
    predicted = model.predict(features.iloc[-8:], frame[list(config.targets)].iloc[-8:])

    assert model.stage2_rows_ >= 300
    assert len(model.inner_folds_) == 2
    assert predicted.shape == (8, 4)
    assert np.isfinite(predicted.to_numpy()).all()
