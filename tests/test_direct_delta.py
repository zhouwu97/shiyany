from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.direct_delta import (
    DirectDeltaConfig,
    DirectDeltaForecaster,
    audit_direct_delta_future_perturbations,
    build_direct_delta_features,
    build_direct_delta_oof,
    build_direct_delta_targets,
    screen_direct_delta,
)
from gas_forecast.splits import TimeFold


def _frame(rows: int = 280) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    point = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "generator_1": 95.0 + 8.0 * np.sin(point / 9.0) + point * 0.02,
            "generator_all": 215.0 + 10.0 * np.sin(point / 11.0) + point * 0.03,
            "generator_use_blast_furnace_gas": 500_000.0 + point * 20.0,
            "generator_use_coke_gas": 12_000.0 + 100.0 * np.cos(point / 8.0),
            "generator_use_converter_gas": 18_000.0 + 60.0 * np.sin(point / 7.0),
            "blast_furnace_1": 800_000.0 + point * 10.0,
            "blast_furnace_user1": 100_000.0 + point * 4.0,
            "blast_furnace_gas_holder_2": 25_000.0 + 50.0 * np.sin(point / 6.0),
        },
        index=index,
    )


def _config() -> DirectDeltaConfig:
    return DirectDeltaConfig(
        min_train_rows=48,
        lgb_n_estimators=8,
        lgb_min_child_samples=12,
        inner_folds=2,
    )


def test_direct_delta_targets_are_absolute_differences_and_feature_schema_is_causal() -> None:
    frame = _frame()
    labels = build_direct_delta_targets(frame)
    origin = frame.index[100]
    assert labels.loc[origin, "generator_1_delta_3"] == (
        frame.loc[frame.index[103], "generator_1"] - frame.loc[origin, "generator_1"]
    )

    baseline = build_direct_delta_features(frame)
    altered = frame.copy()
    altered.loc[altered.index > origin] = -999_999.0
    changed = build_direct_delta_features(altered)
    assert list(baseline.columns) == list(changed.columns)
    pd.testing.assert_series_equal(baseline.loc[origin], changed.loc[origin])
    assert "known_future_price_15" not in baseline.columns


def test_future_price_requires_explicit_origin_known_proof() -> None:
    class Schedule:
        def lookup(self, index: pd.DatetimeIndex) -> np.ndarray:
            return np.arange(len(index), dtype=float)

    frame = _frame()
    blocked = build_direct_delta_features(
        frame,
        price_schedule=Schedule(),
        allow_future_price=True,
        price_known_in_advance=False,
        return_metadata=True,
    )
    enabled = build_direct_delta_features(
        frame,
        price_schedule=Schedule(),
        allow_future_price=True,
        price_known_in_advance=True,
        return_metadata=True,
    )
    assert blocked.price_enabled is False
    assert "known_future_price_15" not in blocked.frame
    assert enabled.price_enabled is True
    assert "known_future_price_15" in enabled.frame


def test_direct_delta_forecaster_has_16_models_per_model_family_and_feasible_predictions() -> None:
    frame = _frame()
    config = _config()
    features = build_direct_delta_features(frame, include_nonlinear_state=True)
    labels = build_direct_delta_targets(frame)
    model = DirectDeltaForecaster(config).fit(
        features.iloc[:220],
        labels.iloc[:220],
        frame.loc[frame.index[:220], list(config.targets)],
        train_end=frame.index[219],
    )

    assert len(model.states_) == 32
    predicted = model.predict(
        features.iloc[220:228],
        frame.loc[frame.index[220:228], list(config.targets)],
        model_name="lightgbm",
    )
    assert predicted.shape == (8, 16)
    assert np.isfinite(predicted.to_numpy()).all()
    for step in config.horizons:
        g1 = predicted[f"generator_1_t+{15 * step}_pred"]
        total = predicted[f"generator_all_t+{15 * step}_pred"]
        assert (total >= g1).all()
        assert (total - g1 <= 240.0 + 1e-9).all()


def test_future_mutations_leave_each_origin_prediction_unchanged() -> None:
    frame = _frame()
    config = _config()
    features = build_direct_delta_features(frame)
    labels = build_direct_delta_targets(frame)
    model = DirectDeltaForecaster(config).fit(
        features.iloc[:200],
        labels.iloc[:200],
        frame.loc[frame.index[:200], list(config.targets)],
    )
    audit = audit_direct_delta_future_perturbations(
        frame.iloc[:220],
        features.iloc[:220],
        model,
        origins=[frame.index[120], frame.index[160]],
        model_name="lightgbm",
    )
    assert audit["passed"] is True
    assert audit["cases_checked"] == 8


def test_predict_at_origin_rebuilds_only_history_features() -> None:
    frame = _frame()
    config = _config()
    features = build_direct_delta_features(frame)
    labels = build_direct_delta_targets(frame)
    model = DirectDeltaForecaster(config).fit(
        features.iloc[:200],
        labels.iloc[:200],
        frame.loc[frame.index[:200], list(config.targets)],
        train_end=frame.index[199],
    )
    origin = frame.index[220]
    baseline = model.predict_at_origin(frame.loc[:origin], model_name="ridge")
    changed = frame.copy()
    changed.loc[changed.index > origin] = -999_999.0
    observed = model.predict_at_origin(changed.loc[:origin], model_name="ridge")

    np.testing.assert_array_equal(baseline.to_numpy(dtype=float), observed.to_numpy(dtype=float))
    assert model.last_prediction_metadata_["origin"] == origin
    assert model.last_prediction_metadata_["used_future_observations"] is False


def test_oof_records_fold_origin_train_end_target_horizon_actual_and_predictions() -> None:
    frame = _frame()
    config = _config()
    features = build_direct_delta_features(frame)
    start = frame.index[149]
    fold = TimeFold(
        name="dev_01",
        train_start=frame.index[0],
        train_end=start - pd.Timedelta(minutes=135),
        validation_start=start,
        validation_end=frame.index[170],
    )
    rows, report = build_direct_delta_oof(
        frame,
        features,
        config=config,
        folds=[fold],
        nested=True,
        origin_only=True,
    )

    required = {
        "fold",
        "origin_time",
        "train_end",
        "target",
        "horizon",
        "actual",
        "actual_delta",
        "ridge_prediction",
        "lightgbm_prediction",
    }
    assert required.issubset(rows.columns)
    assert set(rows["target"]) == {"generator_1", "generator_all"}
    assert set(rows["horizon"]) == {15, 30, 45, 60, 75, 90, 105, 120}
    assert (rows["train_end"] + pd.Timedelta(minutes=120) < rows["origin_time"]).all()
    assert report["blind_labels_used"] is False
    assert report["nested_cross_fitting"] is True
    assert report["origin_only_prediction"] is True
    blind = TimeFold(
        name="blind",
        train_start=fold.train_start,
        train_end=fold.train_end,
        validation_start=fold.validation_start,
        validation_end=fold.validation_end,
        blind=True,
    )
    with pytest.raises(ValueError, match="blind"):
        build_direct_delta_oof(
            frame,
            features,
            config=config,
            folds=[blind],
            include_blind=True,
        )


def test_screening_requires_both_pooled_improvement_and_three_fold_wins() -> None:
    records: list[dict[str, object]] = []
    for fold_index in range(5):
        for position in range(8):
            actual = 100.0
            records.append(
                {
                    "fold": f"dev_{fold_index + 1:02d}",
                    "actual": actual,
                    "parent_ridge_prediction": 101.0,
                    "lightgbm_prediction": 100.0 if fold_index < 3 else 102.0,
                }
            )
    report = screen_direct_delta(
        pd.DataFrame(records), candidate="lightgbm", parent="parent_ridge"
    )
    assert report["status"] == "PASS"
    assert report["fold_wins"] == 3
    assert report["pooled_improvement_pp"] >= 0.02

    stopped = screen_direct_delta(
        pd.DataFrame(records).assign(lightgbm_prediction=100.99),
        candidate="lightgbm",
        parent="parent_ridge",
    )
    assert stopped["status"] == "STOP"
