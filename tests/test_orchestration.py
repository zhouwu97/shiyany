from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig
from gas_forecast.orchestration import audit_future_perturbation


def test_future_perturbation_guard_checks_all_origin_features() -> None:
    rows = 160
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    values = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100 + values * 0.1,
            "generator_all": 220 + values * 0.2,
            "generator_use_blast_furnace_gas": 500_000 + values,
            "generator_use_coke_gas": 10_000 + values,
            "generator_use_converter_gas": 20_000 + values,
            "blast_furnace_gas_holder_2": 100_000 + values,
        },
        index=index,
    )

    result = audit_future_perturbation(frame, ForecastConfig())

    assert result["passed"] is True
    assert result["future_rows_perturbed"] > 0
    assert result["checked_feature_columns"] > len(frame.columns)
    assert result["changed_columns"] == []
