from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.config import FeatureConfig
from gas_forecast.features import audit_feature_availability, build_causal_features
from gas_forecast.targets import build_delta_targets


def _frame(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    values = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "generator_1": 100 + values * 0.1,
            "generator_all": 220 + values * 0.2,
            "generator_use_blast_furnace_gas": 500_000 + values,
            "generator_use_coke_gas": np.where(values % 5 == 0, 0, 10_000 + values),
            "generator_use_converter_gas": 20_000 + values,
            "blast_furnace_gas_holder_2": 100_000 + values,
        },
        index=index,
    )


def test_future_perturbation_does_not_change_origin_features() -> None:
    frame = _frame()
    origin = frame.index[80]
    baseline = build_causal_features(frame)
    perturbed = frame.copy()
    perturbed.loc[perturbed.index > origin] = -999_999
    changed = build_causal_features(perturbed)

    pd.testing.assert_series_equal(baseline.loc[origin], changed.loc[origin])


def test_missing_indicator_schema_does_not_depend_on_future_missing_values() -> None:
    frame = _frame()
    origin = frame.index[80]
    baseline = build_causal_features(frame)
    changed = frame.copy()
    changed.loc[changed.index > origin, "generator_1"] = np.nan
    perturbed = build_causal_features(changed)

    assert "feat_missing_generator_1" in baseline.columns
    assert list(baseline.columns) == list(perturbed.columns)
    pd.testing.assert_series_equal(baseline.loc[origin], perturbed.loc[origin])


def test_rolling_mean_excludes_current_value() -> None:
    frame = _frame(20)
    features = build_causal_features(
        frame,
        FeatureConfig(lags=(1,), diff_lags=(1,), rolling_windows=(4,)),
    )
    expected = frame["generator_1"].iloc[4:8].mean()
    assert features["feat_generator_1_mean_4"].iloc[8] == pytest.approx(expected)


def test_delta_targets_are_direct_horizon_differences() -> None:
    frame = _frame(20)
    labels = build_delta_targets(frame, ("generator_1",), (1, 8))
    assert labels.iloc[0]["generator_1_tplus_15"] == pytest.approx(0.1)
    assert labels.iloc[0]["generator_1_tplus_120"] == pytest.approx(0.8)


def test_only_known_price_features_may_have_positive_information_offset() -> None:
    columns = ["generator_1", "feat_generator_1_lag_4", "feat_target_price_tplus_120"]
    metadata = audit_feature_availability(columns)
    assert metadata["feat_generator_1_lag_4"].max_offset_minutes == -60
    assert metadata["feat_target_price_tplus_120"].known_in_advance is True
