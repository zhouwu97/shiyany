from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.config import FeatureConfig, ForecastConfig
from gas_forecast.features import build_causal_features
from gas_forecast.research_models import select_generator1_features


def _rich_frame(rows: int = 128) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    values = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "generator_1": 100.0 + values,
            "generator_all": 230.0 + 1.5 * values,
            "generator_use_blast_furnace_gas": 500_000.0 + 50.0 * values,
            "generator_use_coke_gas": 20_000.0 + 5.0 * values,
            "generator_use_converter_gas": 30_000.0 + 3.0 * values,
            "blast_furnace_1": 250_000.0 + 10.0 * values,
            "blast_furnace_2": 200_000.0 + 8.0 * values,
            "blast_furnace_4": 180_000.0 + 6.0 * values,
            "blast_furnace_5": 160_000.0 + 4.0 * values,
            "coke_oven_1": 70_000.0 + 2.0 * values,
            "converter_1": 40_000.0 + values,
            "air_heater_1": 80_000.0 + values,
            "air_heater_2": 60_000.0 + values,
            "air_heater_4": 40_000.0 + values,
            "air_heater_5": 20_000.0 + values,
            "blast_furnace_user1": 30_000.0 + values,
            "blast_furnace_user2": 20_000.0 + values,
            "blast_furnace_user3": 10_000.0 + values,
            "blast_furnace_user4": 5_000.0 + values,
            "converter_user1": 2_000.0 + values,
            "converter_user2": 3_000.0 + values,
            "into_gas_mixed_blast_furnace": 15_000.0 + values,
            "into_gas_mixed_coke": 10_000.0 + values,
            "into_gas_mixed_converter": 5_000.0 + values,
            "blast_furnace_gas_holder_2": 90_000.0 + 20.0 * values,
        },
        index=index,
    )


def _rich_config() -> FeatureConfig:
    return FeatureConfig(
        lags=(1, 2, 4),
        diff_lags=(1, 2),
        rolling_windows=(4, 8),
        enable_rich_quantile_features=True,
        enable_rich_ramp_state_features=True,
        enable_rich_gas_resource_features=True,
        rich_quantile_windows=(8,),
    )


def test_rich_quantile_features_use_only_history() -> None:
    frame = _rich_frame()
    features = build_causal_features(frame, _rich_config())

    expected = frame["generator_1"].iloc[:8].quantile(0.10)
    assert features["feat_rich_quantile_generator_1_q10_8"].iloc[8] == pytest.approx(
        expected
    )
    assert "feat_rich_ramp_generator_1_up" in features
    assert "feat_rich_gas_available_for_generation" in features


def test_rich_feature_groups_remain_causal_under_future_perturbation() -> None:
    frame = _rich_frame()
    config = _rich_config()
    origin = frame.index[80]
    baseline = build_causal_features(frame, config)
    changed = frame.copy()
    changed.loc[changed.index > origin] = -999_999.0
    perturbed = build_causal_features(changed, config)

    pd.testing.assert_series_equal(baseline.loc[origin], perturbed.loc[origin])


def test_generator1_selector_only_includes_enabled_rich_groups() -> None:
    frame = _rich_frame()
    disabled = FeatureConfig(
        lags=(1,), diff_lags=(1,), rolling_windows=(4,), rich_quantile_windows=(8,)
    )
    enabled = _rich_config()

    disabled_features = build_causal_features(frame, disabled)
    enabled_features = build_causal_features(frame, enabled)
    disabled_selected = select_generator1_features(
        disabled_features, "core", ForecastConfig(feature=disabled)
    )
    enabled_selected = select_generator1_features(
        enabled_features, "core", ForecastConfig(feature=enabled)
    )

    assert not any(column.startswith("feat_rich_") for column in disabled_selected)
    assert "feat_rich_quantile_generator_1_q10_8" in enabled_selected
    assert "feat_rich_ramp_generator_1_rate" in enabled_selected
    assert "feat_rich_gas_production_demand_ratio" in enabled_selected
