from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.causal_trajectory_ensemble import PARENT_ROUTE
from gas_forecast.p4_robust_fusion import (
    ROUTE_NAMES,
    evaluate_training_gate,
    robust_cross_fitted_fusion,
    validate_matching_keys,
)


def _rows(
    *,
    folds: int = 7,
    route_prediction: float = 96.0,
    route_name: str = "route",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fold_index in range(1, folds + 1):
        origin = pd.Timestamp("2025-01-01") + pd.Timedelta(days=fold_index)
        for target in ("generator_1", "generator_all"):
            records.append(
                {
                    "fold": f"fold_{fold_index:02d}",
                    "origin_time": origin,
                    "train_end": origin - pd.Timedelta(minutes=15),
                    "target": target,
                    "horizon": 15,
                    "actual": 100.0,
                    f"{PARENT_ROUTE}__prediction": 101.0,
                    f"{route_name}__prediction": route_prediction,
                }
            )
    return pd.DataFrame(records)


def test_changing_held_actual_does_not_change_that_fold_selection() -> None:
    rows = _rows()
    original = robust_cross_fitted_fusion(rows, route_names=("route",))
    changed = rows.copy()
    changed.loc[changed["fold"].eq("fold_04"), "actual"] = 250.0

    rerun = robust_cross_fitted_fusion(changed, route_names=("route",))

    original_choice = original.selections.set_index("held_fold").loc["fold_04"]
    changed_choice = rerun.selections.set_index("held_fold").loc["fold_04"]
    assert original_choice["selected_candidate"] == changed_choice["selected_candidate"]
    assert original_choice["weights"] == changed_choice["weights"]
    held_trace = rerun.trace.loc[rerun.trace["held_fold"].eq("fold_04")]
    assert not held_trace["held_fold_labels_used"].any()
    assert all("fold_04" not in value for value in held_trace["training_folds"])


def test_training_side_dangerous_candidate_is_filtered_before_ranking() -> None:
    rows = _rows(folds=7)
    # 20% 在多数折更优，但 fold_01 回退超过 0.100pp；10% 仍在边界内。
    rows.loc[rows["fold"].eq("fold_01"), "route__prediction"] = 101.75

    result = robust_cross_fitted_fusion(rows, route_names=("route",))
    held = result.trace.loc[result.trace["held_fold"].eq("fold_02")].set_index("candidate")

    assert (
        held.loc["parent_80_route_20", "pooled_candidate_mape"]
        < held.loc["parent_90_route_10", "pooled_candidate_mape"]
    )
    assert not bool(held.loc["parent_80_route_20", "check_worst_fold_regression"])
    assert not bool(held.loc["parent_80_route_20", "eligible"])
    assert bool(held.loc["parent_90_route_10", "eligible"])
    assert bool(held.loc["parent_90_route_10", "selected"])


def test_no_eligible_candidate_falls_back_to_parent() -> None:
    rows = _rows(route_prediction=120.0)

    result = robust_cross_fitted_fusion(rows, route_names=("route",))

    assert set(result.selections["selected_candidate"]) == {"parent_only"}
    assert result.selections["selected_reason"].str.contains("回退 A61 parent-only").all()
    assert np.allclose(result.rows["prediction"], result.rows[f"{PARENT_ROUTE}__prediction"])


def test_fixed_point_one_pp_worst_fold_boundary_is_inclusive() -> None:
    rows = _rows(folds=6)
    rows["prediction"] = 100.0
    rows.loc[rows["fold"].eq("fold_06"), "prediction"] = 101.1

    on_boundary = evaluate_training_gate(rows)
    rows.loc[rows["fold"].eq("fold_06"), "prediction"] = 101.100001
    over_boundary = evaluate_training_gate(rows)

    assert on_boundary["worst_fold_regression_pp"] == pytest.approx(0.100, abs=1e-12)
    assert bool(on_boundary["checks"]["worst_fold_regression"])
    assert over_boundary["worst_fold_regression_pp"] > 0.100
    assert not bool(over_boundary["checks"]["worst_fold_regression"])


def test_fixed_point_one_pp_target_boundary_is_inclusive() -> None:
    rows = _rows(folds=6)
    rows["prediction"] = 100.0
    rows.loc[rows["target"].eq("generator_1"), "prediction"] = 101.1

    on_boundary = evaluate_training_gate(rows)
    rows.loc[rows["target"].eq("generator_1"), "prediction"] = 101.100001
    over_boundary = evaluate_training_gate(rows)

    assert on_boundary["max_target_regression_pp"] == pytest.approx(0.100, abs=1e-12)
    assert bool(on_boundary["checks"]["target_regression"])
    assert over_boundary["max_target_regression_pp"] > 0.100
    assert not bool(over_boundary["checks"]["target_regression"])


def test_blind_and_complete_key_mismatch_fail_closed() -> None:
    blind = _rows()
    blind.loc[0, "fold"] = "blind_01"
    with pytest.raises(ValueError, match="blind"):
        robust_cross_fitted_fusion(blind, route_names=("route",))

    integration = _rows(folds=1).rename(columns={"route__prediction": "unused"})
    raw = integration.loc[:, ["fold", "origin_time", "train_end", "target", "horizon", "actual"]]
    route_rows = {route: raw.copy() for route in ROUTE_NAMES}
    route_rows[ROUTE_NAMES[0]].loc[0, "actual"] += 1.0
    with pytest.raises(ValueError, match="完整键不一致"):
        validate_matching_keys(integration, route_rows)
