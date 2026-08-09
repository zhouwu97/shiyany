from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig
from gas_forecast.features import PriceSchedule, build_causal_features


def test_research_feature_flags_add_causal_time_price_and_core_dynamic_fields() -> None:
    index = pd.date_range("2025-01-01", periods=120, freq="15min")
    values = np.tile(np.arange(48, dtype=float)[:, None], (1, 12))
    frame = pd.DataFrame(
        {
            "generator_1": np.linspace(100.0, 130.0, len(index)),
            "generator_all": np.linspace(220.0, 250.0, len(index)),
            "generator_use_blast_furnace_gas": np.linspace(500_000.0, 501_000.0, len(index)),
            "blast_furnace_gas_holder_2": np.linspace(30_000.0, 31_000.0, len(index)),
            "unrelated_measurement": np.arange(len(index), dtype=float),
        },
        index=index,
    )
    config = FeatureConfig(
        horizons=(1, 2),
        rolling_windows=(4, 8),
        enable_slot_one_hot=True,
        enable_time_fourier=True,
        enable_price_delta_features=True,
        enable_price_interactions=True,
        dynamic_feature_scope="core",
    )

    features = build_causal_features(frame, config, PriceSchedule(values))

    assert "feat_slot_0" in features
    assert "feat_weekday_2" in features
    assert "feat_is_weekend" in features
    assert "feat_day_fourier_sin_4" in features
    assert "feat_week_fourier_cos_2" in features
    assert "feat_price_delta_tplus_30" in features
    assert features.iloc[0]["feat_price_delta_tplus_30"] == 1.0
    assert "feat_generator1_price_delta_tplus_15" in features
    assert "feat_dynamic_generator_1_lag_1" in features
    assert "feat_dynamic_generator_use_blast_furnace_gas_mean_4" in features
    assert "feat_dynamic_unrelated_measurement_lag_1" not in features


def test_ramp_and_relation_features_are_causal_and_frozen_by_config() -> None:
    index = pd.date_range("2025-01-01", periods=120, freq="15min")
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + np.sin(np.arange(len(index)) / 4.0),
            "generator_all": 220.0 + np.sin(np.arange(len(index)) / 5.0),
            "generator_use_blast_furnace_gas": 500_000.0 + np.arange(len(index)),
            "blast_furnace_gas_holder_2": 30_000.0 + np.arange(len(index)),
        },
        index=index,
    )
    config = FeatureConfig(
        horizons=(1, 2),
        rolling_windows=(4, 8),
        enable_ramp_features=True,
        relation_features=("generator_1|1|1",),
    )

    baseline = build_causal_features(frame, config)
    changed = frame.copy()
    changed.iloc[-1, :] = -999_999.0
    perturbed = build_causal_features(changed, config)

    assert "feat_generator_1_ramp_up_run_length" in baseline
    assert "feat_generator_1_acceleration" in baseline
    assert "feat_relation_generator_1_lag_1_h1" in baseline
    pd.testing.assert_series_equal(
        baseline.loc[index[-2]], perturbed.loc[index[-2]], check_names=False
    )
