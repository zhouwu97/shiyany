from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.recursive_arx import (
    A61_HORIZONS,
    A61_TARGETS,
    _candidate_status,
    _spec,
    _training_data,
    build_recursive_arx_diversity,
)


def _frame(rows: int = 1_000) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    generator_1 = 100.0 + 0.01 * phase + 2.0 * np.sin(phase / 12.0)
    generator_all = 220.0 + 0.02 * phase + 3.0 * np.sin(phase / 15.0)
    return pd.DataFrame(
        {
            "generator_1": generator_1,
            "generator_all": generator_all,
            "feat_generator_1_lag_1": pd.Series(generator_1, index=index).shift(1),
            "feat_generator_1_lag_2": pd.Series(generator_1, index=index).shift(2),
            "feat_generator_all_lag_1": pd.Series(generator_all, index=index).shift(1),
            "feat_generator_all_lag_2": pd.Series(generator_all, index=index).shift(2),
            "feat_generator_rest": generator_all - generator_1,
            "feat_generator_gas_total": 1_000.0 + phase,
            "feat_rich_gas_available_for_generation": 2_000.0 + phase,
            "feat_rich_gas_holder_buffer": 100.0 + np.sin(phase / 20.0),
            "feat_rich_ramp_generator_1_rate": np.gradient(generator_1),
            "feat_rich_ramp_generator_all_rate": np.gradient(generator_all),
            **{
                f"feat_target_price_tplus_{horizon}": np.full(rows, 1.0 + horizon / 1_000.0)
                for horizon in A61_HORIZONS
            },
        },
        index=index,
    )


def _rows(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    folds = (
        ("dev_01", frame.index[400], frame.index[500:520]),
        ("dev_02", frame.index[500], frame.index[600:620]),
    )
    for fold, train_end, origins in folds:
        for origin in origins:
            for target in A61_TARGETS:
                current = float(frame.loc[origin, target])
                for horizon in A61_HORIZONS:
                    records.append(
                        {
                            "fold": fold,
                            "origin_time": origin,
                            "train_end": train_end,
                            "target": target,
                            "horizon": horizon,
                            "actual": current,
                            "current_value": current,
                            "persistence_pred": current,
                            "parent_pred": current,
                        }
                    )
    return pd.DataFrame(records)


def test_recursive_arx_spec_has_fixed_eight_step_contract() -> None:
    spec = _spec("generator_1")
    assert spec.current_column == "generator_1"
    assert spec.lag_columns == ("feat_generator_1_lag_1", "feat_generator_1_lag_2")
    assert len(spec.future_price_columns) == 8


def test_training_data_excludes_held_labels_and_checks_label_end() -> None:
    frame = _frame()
    features = frame.copy()
    train_end = frame.index[400]
    first_held = frame.index[500]
    training_features, labels, trace = _training_data(
        frame,
        features,
        target="generator_1",
        train_end=train_end,
        first_held_origin=first_held,
        spec=_spec("generator_1"),
    )
    assert len(labels) == 401
    assert training_features.index.max() == train_end
    assert trace["history_after_train_end"] == 0
    perturbed = frame.copy()
    perturbed.loc[frame.index[500:], "generator_1"] += 10_000.0
    _, perturbed_labels, _ = _training_data(
        perturbed,
        perturbed.copy(),
        target="generator_1",
        train_end=train_end,
        first_held_origin=first_held,
        spec=_spec("generator_1"),
    )
    np.testing.assert_allclose(labels.to_numpy(), perturbed_labels.to_numpy())


def test_recursive_arx_builder_keeps_held_fold_changes_out_of_training() -> None:
    frame = _frame()
    rows = _rows(frame)
    baseline = build_recursive_arx_diversity(
        frame,
        frame,
        rows,
        baseline_column="parent_pred",
    )
    changed = frame.copy()
    changed.loc[frame.index[521:], "generator_1"] += 10_000.0
    changed_result = build_recursive_arx_diversity(
        changed,
        changed,
        rows,
        baseline_column="parent_pred",
    )
    first_fold = baseline.rows["fold"].eq("dev_01") & baseline.rows["target"].eq(
        "generator_1"
    )
    np.testing.assert_allclose(
        baseline.rows.loc[first_fold, "a61_recursive_raw_pred"].to_numpy(),
        changed_result.rows.loc[first_fold, "a61_recursive_raw_pred"].to_numpy(),
    )
    assert baseline.report["training_trace_summary"]["labels_from_held_fold"] == 0
    assert baseline.report["raw_route_audits"]["a61_recursive_pred"][
        "noneligible_raw_changed_cells"
    ] == 0


def test_recursive_arx_builder_rejects_incomplete_origin_cells() -> None:
    frame = _frame()
    rows = _rows(frame).iloc[:-1]
    with pytest.raises(ValueError, match="两个目标和八个步长"):
        build_recursive_arx_diversity(frame, frame, rows, baseline_column="parent_pred")


def test_a61_candidate_status_requires_all_fixed_guards() -> None:
    comparison = {
        "pooled_difference": -0.00006,
        "recent_5_folds_difference": {
            "dev_1": -0.1,
            "dev_2": 0.1,
            "dev_3": -0.1,
            "dev_4": -0.1,
            "dev_5": 0.1,
        },
        "worst_fold_regression": 0.0009,
    }
    result = _candidate_status(comparison)
    assert result["status"] == "RETAIN_RECURSIVE_DIVERSITY"
    comparison["worst_fold_regression"] = 0.0011
    assert _candidate_status(comparison)["status"] == "DO_NOT_RETAIN"
