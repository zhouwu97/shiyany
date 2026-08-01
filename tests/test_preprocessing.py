from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.preprocessing import causal_freeze_features, causal_hampel


def test_causal_hampel_does_not_change_past_when_future_changes() -> None:
    index = pd.date_range("2025-01-01", periods=50, freq="15min")
    baseline = pd.Series(np.sin(np.arange(50) / 5), index=index)
    changed = baseline.copy()
    changed.iloc[31:] = 9999.0

    before = causal_hampel(baseline, window=8)
    after = causal_hampel(changed, window=8)

    pd.testing.assert_frame_equal(before.iloc[:31], after.iloc[:31])


def test_freeze_features_detect_completed_run() -> None:
    series = pd.Series([1.0, 1.0, 1.0, 2.0])

    features = causal_freeze_features(series)

    assert features["freeze_length"].tolist() == [0, 1, 2, 0]
    assert features["freeze_ended"].tolist() == [0, 0, 0, 1]
