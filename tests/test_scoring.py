from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.scoring import ScoreSpec, competition_mape, score_oof_long


def test_competition_mape_uses_explicit_epsilon_and_drops_invalid_pairs() -> None:
    actual = np.array([100.0, 0.0, np.nan])
    predicted = np.array([90.0, 1.0, 5.0])

    score = competition_mape(actual, predicted, epsilon=1.0)

    assert score == 0.55


def test_score_oof_long_reports_pooled_and_legacy_fold_mean() -> None:
    rows = pd.DataFrame(
        {
            "fold": ["a", "b", "b", "b"],
            "origin_time": pd.date_range("2025-01-01", periods=4, freq="15min"),
            "target": ["x", "x", "x", "x"],
            "horizon": [15, 15, 15, 15],
            "actual": [100.0] * 4,
            "pred": [90.0, 80.0, 80.0, 80.0],
        }
    )

    report = score_oof_long(rows, "pred", spec=ScoreSpec())

    assert np.isclose(report["pooled_mape"], 0.175)
    assert np.isclose(report["legacy_fold_mean_mape"], 0.15)
    assert report["score_spec"]["missing_policy"] == "drop_pairwise"
