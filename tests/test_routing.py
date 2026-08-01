from __future__ import annotations

import pandas as pd

from gas_forecast.routing import (
    RoutingConfig,
    leave_one_fold_out_route,
    reconcile_post_route,
)


def test_lofo_route_uses_other_folds_and_falls_back_when_unstable() -> None:
    rows = []
    for fold, best in [("a", "v1_pred"), ("b", "v2_pred"), ("c", "v2_pred")]:
        for origin in pd.date_range("2025-01-01", periods=4, freq="15min"):
            rows.append(
                {
                    "fold": fold,
                    "origin_time": origin,
                    "target": "generator_1",
                    "horizon": 15,
                    "actual": 100.0,
                    "v1_pred": 100.0 if best == "v1_pred" else 90.0,
                    "v2_pred": 100.0 if best == "v2_pred" else 90.0,
                }
            )
    frame = pd.DataFrame(rows)

    routed, report = leave_one_fold_out_route(
        frame,
        ("v1_pred", "v2_pred"),
        config=RoutingConfig(min_relative_improvement=0.0, min_fold_win_rate=0.5),
    )

    assert len(routed) == len(frame)
    assert routed["routed_pred"].notna().all()
    assert set(report["fold_routes"]) == {"a", "b", "c"}
    assert "pooled_mape" in report["unbiased_oof"]


def test_post_route_reconciliation_preserves_generator_order_and_cap() -> None:
    frame = pd.DataFrame(
        {
            "fold": ["a", "a"],
            "origin_time": [pd.Timestamp("2025-01-01")] * 2,
            "target": ["generator_1", "generator_all"],
            "horizon": [15, 15],
            "actual": [100.0, 150.0],
            "v1_pred": [100.0, 100.0],
            "v2_pred": [100.0, 500.0],
        }
    )
    routed = frame.assign(raw_routed_pred=[100.0, 500.0])
    reconciled = reconcile_post_route(routed, prediction_column="raw_routed_pred")
    gen1 = reconciled.loc[reconciled["target"].eq("generator_1"), "raw_routed_pred"].iloc[0]
    genall = reconciled.loc[reconciled["target"].eq("generator_all"), "raw_routed_pred"].iloc[0]
    assert genall == gen1 + 240.0
