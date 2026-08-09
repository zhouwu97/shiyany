from __future__ import annotations

import pandas as pd

from gas_forecast.blending import residual_correlation, time_ordered_stack_oof, weighted_blend


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": ["a", "a", "b", "b", "c", "c"],
            "origin_time": pd.date_range("2025-01-01", periods=6, freq="15min"),
            "target": ["generator_1"] * 6,
            "horizon": [15] * 6,
            "actual": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "champion_pred": [100.0, 109.0, 119.0, 129.0, 139.0, 149.0],
            "specialist_pred": [99.0, 111.0, 121.0, 131.0, 141.0, 151.0],
        }
    )


def test_weighted_blend_and_residual_correlation_are_oof_only() -> None:
    rows = _rows()
    blended, report = weighted_blend(
        rows,
        ("champion_pred", "specialist_pred"),
        (0.75, 0.25),
    )

    assert blended["blend_pred"].notna().all()
    assert report["score"]["pooled_mape"] >= 0
    assert set(residual_correlation(rows, ("champion_pred", "specialist_pred")).columns) == {
        "champion_pred",
        "specialist_pred",
    }


def test_time_ordered_stack_never_uses_current_fold_for_weights() -> None:
    rows, report = time_ordered_stack_oof(
        _rows(),
        ("champion_pred", "specialist_pred"),
    )

    assert rows["stack_pred"].notna().all()
    assert list(report["weights_by_fold"]) == ["a", "b", "c"]
    assert report["weights_by_fold"]["a"] == [1.0, 0.0]
