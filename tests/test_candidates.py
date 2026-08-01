from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.candidates import CatBoostDeltaForecaster, select_recent_training_start
from gas_forecast.config import FeatureConfig, ForecastConfig, ModelConfig
from gas_forecast.features import build_causal_features
from gas_forecast.targets import build_delta_targets


def _frame(rows: int) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows)
    return pd.DataFrame(
        {
            "generator_1": 100 + np.sin(phase / 10),
            "generator_all": 220 + 2 * np.sin(phase / 12),
            "generator_use_blast_furnace_gas": 500_000 + phase,
            "generator_use_coke_gas": 10_000 + phase,
            "generator_use_converter_gas": 20_000 + phase,
            "blast_furnace_gas_holder_2": 100_000 + phase,
        },
        index=index,
    )


def test_change_point_recent_window_stays_inside_training_history() -> None:
    series = _frame(1000)["generator_1"].copy()
    series.iloc[700:] += 20

    start, report = select_recent_training_start(series)

    assert series.index.min() <= start <= series.index.max()
    assert "last_change" in report


def test_catboost_candidate_produces_finite_predictions() -> None:
    frame = _frame(520)
    config = ForecastConfig(
        feature=FeatureConfig(horizons=(1, 2), lags=(1, 2), rolling_windows=(4,)),
        model=ModelConfig(
            catboost_iterations=12,
            catboost_early_stopping_rounds=3,
            catboost_depth=3,
            tree_threads_per_worker=1,
        ),
    )
    features = build_causal_features(frame, config.feature)
    deltas = build_delta_targets(frame, config.targets, config.feature.horizons)

    model = CatBoostDeltaForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    )
    predicted = model.predict(features.iloc[-8:], frame[list(config.targets)].iloc[-8:])

    assert predicted.shape == (8, 4)
    assert np.isfinite(predicted.to_numpy()).all()
