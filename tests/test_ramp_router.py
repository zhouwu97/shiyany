from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from gas_forecast.ramp_router import (
    A58_RAW_COLUMN,
    UNASSIGNED_QUINTILE,
    build_a53_oracle_ramp_router,
    build_a54_causal_signal_atlas,
    build_a58_forward_disagreement_specialist,
)


def _router_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    start = pd.Timestamp("2025-03-20 00:00:00")
    for fold_index, fold in enumerate(("dev_01", "dev_02", "dev_03")):
        train_end = start + pd.Timedelta(days=fold_index * 2 - 1, minutes=45)
        for offset in range(2):
            origin = start + pd.Timedelta(days=fold_index * 2, minutes=15 * offset)
            for target, current in (("generator_1", 100.0), ("generator_all", 220.0)):
                for horizon in (60, 75, 90):
                    ramp = 5.0 if horizon == 75 else 1.0
                    actual = current + ramp
                    baseline = current
                    specialist = current + (ramp if target == "generator_1" and horizon == 75 else -8.0)
                    records.append(
                        {
                            "fold": fold,
                            "origin_time": origin,
                            "train_end": train_end,
                            "target": target,
                            "horizon": horizon,
                            "actual": actual,
                            "current_value": current,
                            "persistence_pred": current,
                            "champion_pred": baseline,
                            "rich_gas_pred": baseline,
                            "specialist_pred": specialist,
                        }
                    )
    return pd.DataFrame(records)


def _causal_features() -> pd.DataFrame:
    index = pd.date_range("2025-03-01", "2025-03-26", freq="15min")
    value = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "feat_generator_1_diff_1": value,
            "feat_generator_1_diff_2": value + 1.0,
            "feat_generator_1_diff_4": value + 2.0,
            "feat_generator_1_max_4": value + 10.0,
            "feat_generator_1_min_4": value,
            "feat_generator_1_max_8": value + 20.0,
            "feat_generator_1_min_8": value,
            "feat_generator_1_std_4": value / 10.0,
            "feat_generator_1_std_8": value / 8.0,
            "feat_blast_furnace_gas_holder_2_slope_4": value / 4.0,
            "feat_rich_gas_available_for_generation": value * 2.0,
            "feat_generator_rest_slope_4": value / 3.0,
        },
        index=index,
    )


def _a58_rows() -> pd.DataFrame:
    """构造足够 q80 历史的两折 OOF，覆盖非目标单元和首折回退。"""

    records: list[dict[str, object]] = []
    start = pd.Timestamp("2025-03-20 00:00:00")
    for fold_index, fold in enumerate(("dev_01", "dev_02")):
        fold_start = start + pd.Timedelta(days=fold_index * 3)
        train_end = fold_start - pd.Timedelta(minutes=15)
        for offset in range(128):
            origin = fold_start + pd.Timedelta(minutes=15 * offset)
            for target, current in (("generator_1", 100.0), ("generator_all", 220.0)):
                for horizon in (60, 75, 90, 105, 120):
                    disagreement = (offset + 1) / 20.0
                    specialist = current
                    if target == "generator_1" and horizon in (75, 90, 105, 120):
                        specialist = current + disagreement
                    records.append(
                        {
                            "fold": fold,
                            "origin_time": origin,
                            "train_end": train_end,
                            "target": target,
                            "horizon": horizon,
                            "actual": current + 4.0,
                            "current_value": current,
                            "persistence_pred": current,
                            "rich_gas_pred": current,
                            "specialist_pred": specialist,
                        }
                    )
    return pd.DataFrame(records)


def test_a53_oracle_only_switches_g1_long_true_ramps() -> None:
    rows = _router_rows()
    blind = rows.loc[rows["fold"].eq("dev_03")].copy()
    blind["fold"] = "blind"
    rows = pd.concat([rows, blind], ignore_index=True)
    result = build_a53_oracle_ramp_router(
        rows,
        baseline_column="rich_gas_pred",
        specialist_column="specialist_pred",
    )

    raw_column = "a53_oracle_ramp_raw_pred"
    eligible = result.rows["target"].eq("generator_1") & result.rows["horizon"].eq(75)
    np.testing.assert_allclose(
        result.rows.loc[eligible, raw_column], result.rows.loc[eligible, "specialist_pred"]
    )
    np.testing.assert_allclose(
        result.rows.loc[~eligible, raw_column], result.rows.loc[~eligible, "rich_gas_pred"]
    )
    assert result.rows.loc[eligible, "oracle_switched"].all()
    assert result.rows["fold"].ne("blind").all()
    assert result.report["oracle_only"] is True
    assert result.report["actual_ramp_used"] is True
    assert result.report["deployable"] is False
    assert result.report["formal_candidate"] is False
    assert result.report["raw_route_audit"]["selector_only_changes_g1_long"] is True


def test_a54_quantiles_only_use_each_fold_prior_history() -> None:
    rows = _router_rows()
    blind = rows.loc[rows["fold"].eq("dev_03")].copy()
    blind["fold"] = "blind"
    rows = pd.concat([rows, blind], ignore_index=True)
    features = _causal_features()
    result = build_a54_causal_signal_atlas(
        rows,
        features,
        baseline_column="champion_pred",
        rich_gas_column="rich_gas_pred",
        specialist_column="specialist_pred",
        min_history_rows=1,
    )

    cutoff = result.cutoffs.loc[
        result.cutoffs["fold"].eq("dev_02")
        & result.cutoffs["signal_name"].eq("richgas_champion_abs_disagreement")
        & result.cutoffs["horizon"].eq(75)
    ].iloc[0]
    assert pd.Timestamp(cutoff["history_max_time"]) <= pd.Timestamp(cutoff["train_end"])
    assert int(cutoff["history_rows"]) == 2
    first_model_signal = result.cells.loc[
        result.cells["fold"].eq("dev_01")
        & result.cells["signal_name"].eq("richgas_champion_abs_disagreement")
    ]
    assert first_model_signal["quintile"].eq(UNASSIGNED_QUINTILE).all()
    ready = result.cutoffs.dropna(subset=["history_max_time"])
    assert (
        pd.to_datetime(ready["history_max_time"]) <= pd.to_datetime(ready["train_end"])
    ).all()
    assert result.cells["fold"].ne("blind").all()
    assert result.report["strict_oof_contract"]["labels_used_for_quantiles"] is False
    assert result.report["formal_candidate"] is False


def test_a58_forward_q80_uses_only_prior_folds_and_only_routes_g1_long() -> None:
    rows = _a58_rows()
    blind = rows.loc[rows["fold"].eq("dev_02")].copy()
    blind["fold"] = "blind"
    result = build_a58_forward_disagreement_specialist(
        pd.concat([rows, blind], ignore_index=True),
        baseline_column="rich_gas_pred",
        specialist_column="specialist_pred",
    )

    trace = result.threshold_trace
    first_fold = trace.loc[trace["fold"].eq("dev_01")]
    second_fold = trace.loc[trace["fold"].eq("dev_02")]
    assert first_fold["status"].eq("insufficient_history").all()
    assert first_fold["fallback"].eq("rich_gas").all()
    assert second_fold["status"].eq("ready").all()
    assert second_fold["history_rows"].eq(128).all()
    assert (
        pd.to_datetime(second_fold["history_max_time"])
        <= pd.to_datetime(second_fold["train_end"])
    ).all()
    assert second_fold["history_folds"].eq("dev_01").all()
    expected_q80 = float(np.quantile(np.arange(1, 129, dtype=float) / 20.0, 0.80))
    np.testing.assert_allclose(second_fold["q80"], expected_q80)

    output = result.rows
    eligible = output["target"].eq("generator_1") & output["horizon"].isin(
        (75, 90, 105, 120)
    )
    np.testing.assert_allclose(
        output.loc[~eligible, A58_RAW_COLUMN], output.loc[~eligible, "rich_gas_pred"]
    )
    assert output.loc[output["fold"].eq("dev_01"), "a58_switched"].eq(False).all()
    assert output.loc[output["fold"].eq("dev_02") & eligible, "a58_switched"].any()
    assert output["fold"].ne("blind").all()
    assert result.report["threshold_used_labels"] is False
    assert result.report["held_fold_used_for_threshold"] is False
    assert result.report["blind_used"] is False
    assert result.report["raw_route_audit"]["selector_only_changes_g1_long"] is True
    assert "quantile" not in inspect.signature(
        build_a58_forward_disagreement_specialist
    ).parameters


def test_a58_held_and_future_values_do_not_change_current_fold_threshold() -> None:
    rows = _a58_rows()
    original = build_a58_forward_disagreement_specialist(
        rows,
        baseline_column="rich_gas_pred",
        specialist_column="specialist_pred",
    )
    perturbed = rows.copy()
    future_or_held = perturbed["fold"].eq("dev_02")
    perturbed.loc[future_or_held, "specialist_pred"] += 10_000.0
    repeated = build_a58_forward_disagreement_specialist(
        perturbed,
        baseline_column="rich_gas_pred",
        specialist_column="specialist_pred",
    )
    original_q80 = original.threshold_trace.loc[
        original.threshold_trace["fold"].eq("dev_02"), ["horizon", "q80"]
    ].sort_values("horizon")
    repeated_q80 = repeated.threshold_trace.loc[
        repeated.threshold_trace["fold"].eq("dev_02"), ["horizon", "q80"]
    ].sort_values("horizon")
    np.testing.assert_allclose(
        original_q80["q80"].to_numpy(dtype=float),
        repeated_q80["q80"].to_numpy(dtype=float),
    )
