from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.relations import (
    add_stability_diagnostics,
    build_residual_relation_scan,
    freeze_relation_features,
)


def test_relation_scan_uses_oof_residual_and_freezes_stable_specs() -> None:
    index = pd.date_range("2025-01-01", periods=96, freq="15min")
    source = np.sin(np.arange(len(index)) / 5.0)
    frame = pd.DataFrame({"generator_1": source}, index=index)
    rows = pd.DataFrame(
        {
            "fold": ["a"] * len(index),
            "origin_time": index,
            "target": ["generator_1"] * len(index),
            "horizon": [15] * len(index),
            "actual": source,
            "current_value": np.zeros(len(index)),
            "champion_pred": np.zeros(len(index)),
        }
    )
    scan = build_residual_relation_scan(
        frame,
        rows,
        prediction_column="champion_pred",
        sources=("generator_1",),
        max_lag=2,
        max_horizon=1,
    )
    stable = add_stability_diagnostics(scan, rows, frame, prediction_column="champion_pred")
    frozen = freeze_relation_features(stable, max_features=2, min_month_count=1)

    assert not stable.empty
    assert stable["corr_residual"].notna().any()
    assert len(frozen) <= 2
