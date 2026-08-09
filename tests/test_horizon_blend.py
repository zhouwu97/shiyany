from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.horizon_blend import (
    build_two_band_blend_grid,
    build_two_band_blend_pairs,
    time_ordered_four_band_router,
)


def _rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    start = pd.Timestamp("2025-03-20")
    for fold_position, fold in enumerate(("dev_01", "dev_02", "dev_03", "dev_04")):
        for origin_position in range(12):
            origin = start + pd.Timedelta(days=fold_position * 2, minutes=15 * origin_position)
            for horizon in (15, 30, 45, 60, 75, 90, 105, 120):
                for target, baseline in (("generator_1", 100.0), ("generator_all", 220.0)):
                    residual = (horizon / 120.0) * 2.0 if target == "generator_1" else 0.0
                    records.append(
                        {
                            "fold": fold,
                            "origin_time": origin,
                            "train_end": origin - pd.Timedelta(minutes=135),
                            "target": target,
                            "horizon": horizon,
                            "actual": baseline + residual,
                            "current_value": baseline,
                            "persistence_pred": baseline,
                            "champion_pred": baseline,
                            "rich_branch_pred": baseline + residual,
                        }
                    )
    return pd.DataFrame(records)


def test_two_band_grid_keeps_generator_all_at_champion_before_projection() -> None:
    rows = _rows()

    result = build_two_band_blend_grid(
        rows,
        baseline_column="champion_pred",
        branch_column="rich_branch_pred",
        comparison_column="champion_pred",
        short_weights=(0.10,),
        long_weights=(0.30,),
        scope="development",
    )

    candidate = "rich_short10_long30_pred"
    generator_all = result.rows["target"].eq("generator_all")
    np.testing.assert_allclose(
        result.rows.loc[generator_all, candidate],
        result.rows.loc[generator_all, "champion_pred"],
    )
    assert result.report["models"][candidate]["generator_1_difference"] < 0.0


def test_four_band_router_does_not_use_current_fold_labels_for_route_choice() -> None:
    rows = _rows()
    baseline = time_ordered_four_band_router(
        rows,
        baseline_column="champion_pred",
        branch_column="rich_branch_pred",
        comparison_column="champion_pred",
        short_weight=0.10,
        long_weight=0.30,
        scope="development",
        min_history_rows=8,
    )
    changed = rows.copy()
    changed.loc[changed["fold"].eq("dev_03"), "actual"] += 1_000.0
    perturbed = time_ordered_four_band_router(
        changed,
        baseline_column="champion_pred",
        branch_column="rich_branch_pred",
        comparison_column="champion_pred",
        short_weight=0.10,
        long_weight=0.30,
        scope="development",
        min_history_rows=8,
    )
    candidate = "rich_four_band_route_pred"
    original = baseline.rows.loc[baseline.rows["fold"].eq("dev_03"), candidate]
    altered = perturbed.rows.loc[perturbed.rows["fold"].eq("dev_03"), candidate]
    np.testing.assert_allclose(original, altered)
    assert all(item["fold"] != "dev_01" or item["fallback"] for item in baseline.route_trace)


def test_two_band_pairs_only_emit_pre_registered_combinations() -> None:
    rows = _rows()

    result = build_two_band_blend_pairs(
        rows,
        baseline_column="champion_pred",
        branch_column="rich_branch_pred",
        comparison_column="champion_pred",
        weight_pairs=((0.20, 0.30), (0.30, 0.50)),
        scope="development",
    )

    assert set(result.report["models"]) == {
        "rich_short20_long30_pred",
        "rich_short30_long50_pred",
    }
    assert result.report["weight_pairs"] == [
        {"short_weight": 0.20, "long_weight": 0.30},
        {"short_weight": 0.30, "long_weight": 0.50},
    ]
