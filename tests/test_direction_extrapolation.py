from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.direction_extrapolation import (
    LONG_HORIZONS,
    SHORT_HORIZONS,
    extrapolate_submission_result,
)
from gas_forecast.submission import expected_prediction_columns


def _submission(generator_1: float, generator_all: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-05-01", periods=2, freq="15min"),
            **{
                column: generator_1 if column.startswith("generator_1") else generator_all
                for column in expected_prediction_columns()
            },
        }
    )
    return frame


def test_direction_extrapolation_only_changes_selected_target() -> None:
    baseline = _submission(100.0, 220.0)
    candidate = _submission(110.0, 230.0)
    multipliers = {
        **{horizon: 1.6 for horizon in SHORT_HORIZONS},
        **{horizon: 1.0 for horizon in LONG_HORIZONS},
    }

    result, report = extrapolate_submission_result(
        baseline,
        candidate,
        target="generator_1",
        multipliers=multipliers,
    )

    np.testing.assert_allclose(result["generator_1_t+15_pred"], 116.0)
    np.testing.assert_allclose(result["generator_1_t+120_pred"], 110.0)
    np.testing.assert_allclose(
        result.filter(like="generator_all").to_numpy(dtype=float),
        baseline.filter(like="generator_all").to_numpy(dtype=float),
    )
    assert report["non_target_changed_cells"] == 0


def test_direction_extrapolation_rejects_timestamp_mismatch() -> None:
    baseline = _submission(100.0, 220.0)
    candidate = _submission(110.0, 230.0)
    candidate.loc[1, "datetime"] = pd.Timestamp("2025-05-01 00:45:00")
    multipliers = {horizon: 1.0 for horizon in SHORT_HORIZONS | LONG_HORIZONS}

    with pytest.raises(ValueError, match="时间戳不一致|15 分钟连续"):
        extrapolate_submission_result(
            baseline,
            candidate,
            target="generator_1",
            multipliers=multipliers,
        )
