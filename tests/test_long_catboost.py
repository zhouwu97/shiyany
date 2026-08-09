from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import gas_forecast.long_catboost as long_catboost


def _frame_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=200, freq="15min")
    generator_1 = 100.0 + np.linspace(0.0, 20.0, len(index))
    frame = pd.DataFrame(
        {
            "generator_1": generator_1,
            "generator_all": generator_1 + 120.0,
        },
        index=index,
    )
    features = pd.DataFrame(
        {
            "generator_1": generator_1,
            "generator_all": generator_1 + 120.0,
            "feat_generator_1_lag_1": pd.Series(generator_1, index=index).shift(1),
        },
        index=index,
    )
    return frame, features


def _development_rows(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    folds = (
        ("dev_01", frame.index[60], frame.index[80:84]),
        ("dev_02", frame.index[130], frame.index[150:154]),
    )
    for fold, train_end, origins in folds:
        for origin in origins:
            for target in ("generator_1", "generator_all"):
                current = float(frame.loc[origin, target])
                for horizon in (75, 90, 105, 120):
                    actual = float(frame.loc[origin + pd.Timedelta(minutes=horizon), target])
                    rich_gas = current
                    a51 = current + 0.5 if target == "generator_1" else rich_gas
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
                            "rich_gas_pred": rich_gas,
                            "a51_pred": a51,
                        }
                    )
    return pd.DataFrame(records)


def test_a57_residual_history_excludes_held_fold_and_future_data() -> None:
    frame, _ = _frame_and_features()
    rows = _development_rows(frame)
    train_end = pd.Timestamp(frame.index[130])
    origins, residual, history = long_catboost._residual_training_data(
        rows,
        fold="dev_02",
        train_end=train_end,
        horizon=75,
        baseline_column="rich_gas_pred",
    )
    assert history["fold"].eq("dev_01").all()
    assert (origins <= train_end).all()
    changed = rows.copy()
    changed.loc[changed["fold"].eq("dev_02"), "actual"] += 10_000.0
    repeated_origins, repeated_residual, repeated_history = long_catboost._residual_training_data(
        changed,
        fold="dev_02",
        train_end=train_end,
        horizon=75,
        baseline_column="rich_gas_pred",
    )
    assert repeated_history["fold"].eq("dev_01").all()
    np.testing.assert_array_equal(origins, repeated_origins)
    np.testing.assert_allclose(residual, repeated_residual)


def test_a57_only_routes_g1_long_and_uses_fixed_blends(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, features = _frame_and_features()
    rows = _development_rows(frame)
    calls: list[tuple[int, int]] = []

    def fake_fit(
        training_features: pd.DataFrame,
        training_target: np.ndarray,
        held_features: pd.DataFrame,
        *,
        horizon: int,
    ) -> np.ndarray:
        calls.append((horizon, len(training_target)))
        return np.full(len(held_features), float(np.mean(training_target)))

    monkeypatch.setattr(long_catboost, "A57_MIN_TRAIN_ROWS", 1)
    monkeypatch.setattr(long_catboost, "_fit_fixed_catboost", fake_fit)
    result = long_catboost.build_a57_long_catboost_diversity(
        frame,
        features,
        rows,
        baseline_column="rich_gas_pred",
        a51_column="a51_pred",
    )

    output = result.rows
    eligible = output["target"].eq("generator_1") & output["horizon"].isin(
        long_catboost.A57_LONG_HORIZONS
    )
    for raw_column in ("a57a_absolute_cat_raw_pred", "a57b_residual_cat_raw_pred"):
        np.testing.assert_allclose(
            output.loc[~eligible, raw_column], output.loc[~eligible, "rich_gas_pred"]
        )
    trace = result.training_trace
    first_residual = trace.loc[
        trace["variant"].eq("a57b_residual") & trace["fold"].eq("dev_01")
    ]
    second_residual = trace.loc[
        trace["variant"].eq("a57b_residual") & trace["fold"].eq("dev_02")
    ]
    assert first_residual["status"].eq("rich_gas_fallback").all()
    assert second_residual["status"].eq("trained").all()
    assert second_residual["history_folds"].eq("dev_01").all()
    assert len(calls) == 12
    assert len(result.residual_correlation) == 10
    assert result.report["fixed_catboost"]["iterations"] == 600
    assert result.report["pre_registered_evaluation"]["rich_gas_cat_weights"] == [
        0.05,
        0.1,
        0.15,
        0.2,
    ]
    assert result.report["pre_registered_evaluation"]["a51_cat_weights"] == [0.05, 0.1, 0.15]
    assert all(
        audit["selector_only_changes_g1_long"]
        for audit in result.report["raw_route_audits"].values()
    )
    assert "weights" not in inspect.signature(
        long_catboost.build_a57_long_catboost_diversity
    ).parameters


def test_a57_rejects_blind_rows_before_training() -> None:
    frame, features = _frame_and_features()
    rows = _development_rows(frame)
    blind = rows.iloc[:8].copy()
    blind["fold"] = "blind"
    with pytest.raises(ValueError, match="不得含 blind 行"):
        long_catboost.build_a57_long_catboost_diversity(
            frame,
            features,
            pd.concat([rows, blind], ignore_index=True),
            baseline_column="rich_gas_pred",
            a51_column="a51_pred",
        )
