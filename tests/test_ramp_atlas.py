from __future__ import annotations

import pandas as pd

from gas_forecast.ramp_atlas import build_ramp_error_atlas


def _rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    origin = pd.Timestamp("2025-03-20")
    for fold, current, actual, baseline, candidate in (
        ("dev_01", 100.0, 101.0, 100.0, 101.0),
        ("dev_01", 100.0, 105.0, 100.0, 104.0),
        ("dev_02", 100.0, 110.0, 100.0, 109.0),
        ("dev_02", 100.0, 120.0, 100.0, 119.0),
        ("blind", 100.0, 130.0, 100.0, 100.0),
    ):
        records.append(
            {
                "fold": fold,
                "origin_time": origin,
                "target": "generator_1",
                "horizon": 75,
                "actual": actual,
                "current_value": current,
                "champion_pred": baseline,
                "candidate_pred": candidate,
            }
        )
        origin += pd.Timedelta(minutes=15)
    return pd.DataFrame(records)


def test_ramp_atlas_uses_registered_bands_and_excludes_blind_in_development() -> None:
    result = build_ramp_error_atlas(
        _rows(),
        baseline_column="champion_pred",
        candidate_column="candidate_pred",
        scope="development",
    )

    assert len(result.cells) == 4
    assert result.report["rows_by_ramp_band"] == {
        "stable": 1,
        "mild": 1,
        "medium": 1,
        "large": 1,
    }
    large = result.table.loc[result.table["ramp_band"].eq("large")].iloc[0]
    assert large["candidate_mape"] < large["baseline_mape"]
    assert large["candidate_direction_accuracy"] == 1.0


def test_ramp_atlas_final_scope_retains_blind_rows() -> None:
    result = build_ramp_error_atlas(
        _rows(),
        baseline_column="champion_pred",
        candidate_column="candidate_pred",
        scope="final",
    )

    assert len(result.cells) == 5
    assert result.report["rows_by_ramp_band"]["large"] == 2
