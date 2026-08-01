from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig
from gas_forecast.features import build_causal_features
from gas_forecast.leakage import audit_future_perturbations


def test_multi_origin_future_perturbations_leave_features_unchanged() -> None:
    index = pd.date_range("2025-01-01", periods=100, freq="15min")
    phase = np.arange(100, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100 + phase,
            "generator_all": 220 + phase,
            "generator_use_blast_furnace_gas": 500_000 + phase,
            "generator_use_coke_gas": 10_000 + phase,
            "generator_use_converter_gas": 20_000 + phase,
            "blast_furnace_gas_holder_2": 100_000 + phase,
        },
        index=index,
    )
    config = FeatureConfig(horizons=(1, 2), lags=(1, 2), rolling_windows=(4,))

    report = audit_future_perturbations(
        frame,
        lambda value: build_causal_features(value, config),
        origins=5,
    )

    assert report["passed"] is True
    assert report["origins"] == 5
    assert report["cases_checked"] == 25
