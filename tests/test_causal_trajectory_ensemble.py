from __future__ import annotations

import json

import pandas as pd
import pytest

from gas_forecast.causal_trajectory_ensemble import (
    HORIZONS,
    PARENT_ROUTE,
    RouteReceipt,
    build_causal_trajectory_ensemble,
    collect_matching_oofs,
    cross_fitted_static_blend,
    pre_registered_weight_candidates,
    validate_oof_contract,
    write_ensemble_artifacts,
)


def _rows(folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造完整两目标×八 horizon 的小型 development OOF。"""

    parent_records: list[dict[str, object]] = []
    route_records: list[dict[str, object]] = []
    for fold_number in range(1, folds + 1):
        fold = f"dev_{fold_number:02d}"
        train_end = pd.Timestamp("2025-01-01") + pd.Timedelta(days=fold_number)
        for origin_offset in range(2):
            origin = train_end + pd.Timedelta(minutes=135 + origin_offset * 15)
            for target, actual in (("generator_1", 100.0), ("generator_all", 220.0)):
                for horizon in HORIZONS:
                    parent_error = 4.0 if fold_number % 2 else -4.0
                    parent_records.append(
                        {
                            "fold": fold,
                            "origin_time": origin,
                            "train_end": train_end,
                            "target": target,
                            "horizon": horizon,
                            "actual": actual,
                            "prediction": actual + parent_error,
                        }
                    )
                    route_records.append(
                        {
                            "fold": fold,
                            "origin_time": origin,
                            "train_end": train_end,
                            "target": target,
                            "horizon": horizon,
                            "actual": actual,
                            "prediction": actual + 1.0,
                        }
                    )
    return pd.DataFrame(parent_records), pd.DataFrame(route_records)


def test_validate_contract_requires_complete_development_coverage() -> None:
    parent, _ = _rows()
    summary = validate_oof_contract(parent, source=PARENT_ROUTE)
    assert summary["rows"] == len(parent)
    assert summary["blind_labels_used"] is False

    incomplete = parent.loc[~parent["horizon"].eq(120)].copy()
    with pytest.raises(ValueError, match="未覆盖"):
        validate_oof_contract(incomplete, source="a64_direct_delta")


def test_collect_matching_oofs_rejects_fold_boundary_mismatch() -> None:
    parent, route = _rows()
    route.loc[route["fold"].eq("dev_03"), "train_end"] += pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="不完全一致"):
        collect_matching_oofs(parent, {"a64_direct_delta": route})


def test_cross_fitted_weights_do_not_consume_their_held_fold_labels() -> None:
    parent, route = _rows()
    merged, _ = collect_matching_oofs(parent, {"a64_direct_delta": route})
    _, trace = cross_fitted_static_blend(merged, route_names=("a64_direct_delta",))

    altered = merged.copy()
    altered.loc[altered["fold"].eq("dev_03"), "actual"] *= 10.0
    _, altered_trace = cross_fitted_static_blend(altered, route_names=("a64_direct_delta",))

    original = trace.set_index("held_fold").loc["dev_03", ["selected_candidate", "weights"]]
    changed = altered_trace.set_index("held_fold").loc["dev_03", ["selected_candidate", "weights"]]
    assert original.to_dict() == changed.to_dict()
    assert trace["held_fold_labels_used"].eq(False).all()


def test_weight_registry_is_small_nonnegative_simplex() -> None:
    candidates = pre_registered_weight_candidates(("a62_state_space", "a64_direct_delta"))
    assert len(candidates) == 6
    for _, weights in candidates:
        assert all(value >= 0.0 for value in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)


def test_stop_result_writes_required_audit_artifacts_and_projects_capacity(tmp_path) -> None:
    parent, _ = _rows()
    # 刻意破坏总量约束，验证 A69 的确定性容量投影。
    parent.loc[parent["target"].eq("generator_all"), "prediction"] = 80.0
    result = build_causal_trajectory_ensemble(
        parent,
        {},
        route_receipts=(
            RouteReceipt(
                name=PARENT_ROUTE,
                source="synthetic",
                status="FROZEN_DEVELOPMENT_PARENT",
                accepted=True,
                reason="test",
                rows=len(parent),
                blind_labels_used=False,
                future_perturbation_passed=True,
            ),
        ),
    )
    assert result.report["status"] == "STOP_STATIC_FUSION"
    all_rows = result.rows.loc[result.rows["target"].eq("generator_all"), "prediction"]
    g1_rows = result.rows.loc[result.rows["target"].eq("generator_1"), "prediction"]
    assert (all_rows.to_numpy() >= g1_rows.to_numpy()).all()

    output = write_ensemble_artifacts(result, tmp_path / "a69")
    for name in (
        "oof.parquet",
        "oof.csv",
        "fold_metrics.csv",
        "target_metrics.csv",
        "horizon_metrics.csv",
        "residual_correlation.csv",
        "training_trace.csv",
        "report.json",
    ):
        assert (output / name).is_file()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["blind_labels_used"] is False
