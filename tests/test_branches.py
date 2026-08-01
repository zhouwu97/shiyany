from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.branches import BRANCH_NAMES, CrossFittedBranchForecaster
from gas_forecast.config import FeatureConfig, ForecastConfig, ModelConfig
from gas_forecast.features import build_causal_features
from gas_forecast.targets import build_delta_targets


def test_crossfitted_branches_produce_all_candidates() -> None:
    rows = 1050
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows)
    frame = pd.DataFrame(
        {
            "generator_1": 100 + 5 * np.sin(phase / 20),
            "generator_all": 230 + 8 * np.sin(phase / 25),
            "generator_use_blast_furnace_gas": 600_000 + 500 * np.sin(phase / 11),
            "generator_use_coke_gas": 10_000 + 100 * np.cos(phase / 9),
            "generator_use_converter_gas": 20_000 + 200 * np.sin(phase / 7),
            "blast_furnace_gas_holder_2": 100_000 + 300 * np.sin(phase / 17),
        },
        index=index,
    )
    config = ForecastConfig(
        feature=FeatureConfig(horizons=(1, 2), lags=(1, 2), rolling_windows=(4, 8)),
        model=ModelConfig(
            recent_days=3,
            inner_folds=2,
            lgb_n_estimators=8,
            lgb_max_estimators=12,
            lgb_early_stopping_rounds=3,
            lgb_min_child_samples=20,
        ),
    )
    features = build_causal_features(frame, config.feature)
    deltas = build_delta_targets(frame, config.targets, config.feature.horizons)

    model = CrossFittedBranchForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    )
    candidates = model.predict_candidates(
        features.iloc[-10:], frame[list(config.targets)].iloc[-10:]
    )

    assert set(BRANCH_NAMES).issubset(candidates)
    assert {"simplex_target", "simplex_horizon", "simplex_regularized"}.issubset(candidates)
    assert all(value.shape == (10, 4) for value in candidates.values())
    assert all(np.isfinite(value.to_numpy()).all() for value in candidates.values())
