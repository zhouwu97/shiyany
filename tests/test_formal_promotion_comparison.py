from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import legacy_forecast_config
from gas_forecast.research import _candidate_comparison, make_formal_routed_candidate


def test_formal_routed_candidate_preserves_the_legacy_champion_configuration() -> None:
    route = {"global": {"selected": "v3_pred"}, "targets": {}, "cells": {}}

    candidate = make_formal_routed_candidate(route)

    assert candidate.kind == "formal_routed"
    assert candidate.config == legacy_forecast_config()
    assert candidate.route == route


def test_final_pairwise_comparison_requires_blind_and_majority_fold_support() -> None:
    rows = pd.DataFrame(
        {
            "fold": ["dev_01", "dev_01", "blind", "blind"],
            "origin_time": pd.to_datetime(
                ["2025-03-20", "2025-03-20", "2025-03-22", "2025-03-22"]
            ),
            "target": ["generator_1", "generator_all", "generator_1", "generator_all"],
            "horizon": [15, 15, 15, 15],
            "actual": [100.0, 200.0, 100.0, 200.0],
            "challenger_pred": [99.0, 198.0, 99.0, 198.0],
            "champion_pred": [102.0, 204.0, 102.0, 204.0],
        }
    )

    report = _candidate_comparison(
        rows,
        "challenger_pred",
        "champion_pred",
        scope="final",
    )

    pairwise = report["pairwise"]
    assert report["formal_candidate"] is True
    assert report["fold_wins"] == 2
    assert report["blind_difference"] < 0.0
    assert pairwise["by_target"]["generator_1"]["difference"] < 0.0
    assert pairwise["by_horizon"]["t+15"]["difference"] < 0.0
    assert np.isclose(report["day_block_bootstrap"]["probability_candidate_better"], 1.0)
