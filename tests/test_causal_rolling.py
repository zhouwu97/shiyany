from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.causal_rolling import (
    CausalFeatureBuildResult,
    CausalRollingConfig,
    CausalRollingReconstructionForecaster,
    audit_causal_rolling_future_perturbations,
    build_causal_rolling_features,
    build_causal_rolling_oof,
)
from gas_forecast.splits import TimeFold


def _frame(rows: int = 280) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    point = np.arange(rows, dtype=float)
    generator_1 = 95.0 + 6.0 * np.sin(point / 9.0) + point * 0.03
    return pd.DataFrame(
        {
            "generator_1": generator_1,
            "generator_all": generator_1 + 112.0 + 4.0 * np.cos(point / 13.0),
            "blast_furnace_1": 500_000.0 + 20.0 * point,
            "coke_oven_1": 70_000.0 + 11.0 * point,
            "converter_1": 35_000.0 + 6.0 * point,
            "blast_furnace_gas_holder_2": 120_000.0 + 80.0 * np.sin(point / 7.0),
            "blast_furnace_user1": 95_000.0 + 8.0 * point,
            "air_heater_1": 18_000.0 + 4.0 * point,
            "into_gas_mixed_blast_furnace": 2_000.0 + point,
            "generator_use_blast_furnace_gas": 125_000.0 + 7.0 * point,
        },
        index=index,
    )


def _config() -> CausalRollingConfig:
    return CausalRollingConfig(min_train_rows=48, min_history_rows=17, ridge_alpha=5.0)


def _assert_exact(left: pd.DataFrame, right: pd.DataFrame) -> None:
    np.testing.assert_array_equal(left.to_numpy(dtype=float), right.to_numpy(dtype=float))


def test_current_state_manifest_and_recursive_absolute_trajectory_are_stable() -> None:
    frame = _frame()
    built = build_causal_rolling_features(frame, return_manifest=True)
    assert isinstance(built, CausalFeatureBuildResult)
    required = {
        "generator_1_current",
        "generator_1_delta_15",
        "generator_1_delta_30",
        "generator_1_delta_45",
        "generator_1_delta_60",
        "generator_1_slope_4",
        "generator_1_slope_8",
        "generator_1_slope_16",
        "generator_1_acceleration",
        "generator_1_ewma_trend",
        "generator_1_volatility",
        "holder_momentum",
        "gas_mismatch",
    }
    assert required.issubset(built.frame.columns)
    assert tuple(built.frame.columns) == built.manifest.feature_columns
    assert built.manifest.source_mapping.holder_columns == ("blast_furnace_gas_holder_2",)

    origin = frame.index[200]
    changed = frame.copy()
    changed.loc[changed.index > origin] = -999_999.0
    changed_features = build_causal_rolling_features(changed)
    pd.testing.assert_series_equal(built.frame.loc[origin], changed_features.loc[origin])

    model = CausalRollingReconstructionForecaster(_config()).fit(frame.iloc[:180])
    prediction = model.predict_at_origin(frame.loc[:origin])
    assert prediction.shape == (1, 16)
    assert tuple(prediction.columns) == model.prediction_columns()
    assert np.isfinite(prediction.to_numpy(dtype=float)).all()
    assert [item.minutes for item in model.horizon_metadata()] == list(range(15, 121, 15))
    assert model.last_prediction_metadata_["mode"] == "rolling_reconstruction"
    assert len(model.last_prediction_metadata_["delta_trajectory"]["generator_1"]) == 8
    assert model.last_delta_trajectory_ is not None
    assert tuple(model.last_delta_trajectory_.columns) == model.delta_columns()
    for minutes in range(15, 121, 15):
        generator_1 = prediction.at[origin, f"generator_1_t+{minutes}_pred"]
        generator_all = prediction.at[origin, f"generator_all_t+{minutes}_pred"]
        assert 0.0 <= generator_1 <= 200.0
        assert generator_1 <= generator_all <= generator_1 + 240.0


def test_origin_history_contract_rejects_future_and_all_future_groups_are_bitwise_inert() -> None:
    frame = _frame()
    model = CausalRollingReconstructionForecaster(_config()).fit(frame.iloc[:180])
    origin = frame.index[210]
    baseline = model.predict_at_origin(frame.loc[:origin])
    future = frame.index > origin
    groups = {
        "generator": ["generator_1", "generator_all"],
        "gas": ["blast_furnace_1", "coke_oven_1", "converter_1"],
        "holder": ["blast_furnace_gas_holder_2"],
        "users": ["blast_furnace_user1", "air_heater_1", "into_gas_mixed_blast_furnace"],
        "all_features": list(frame.columns),
    }
    for columns in groups.values():
        for value in (-999_999.0, np.nan):
            changed = frame.copy()
            changed.loc[future, columns] = value
            _assert_exact(baseline, model.predict_at_origin(changed.loc[:origin]))
    _assert_exact(baseline, model.predict_at_origin(frame.loc[:origin].copy()))

    with pytest.raises(ValueError, match="origin 之后"):
        model.predict_at_origin(frame, origin=origin)
    declared = frame.copy()
    declared.attrs["origin_time"] = origin
    with pytest.raises(ValueError, match="origin 之后"):
        model.predict_at_origin(declared)
    assert not hasattr(model, "predict")

    audit = audit_causal_rolling_future_perturbations(model, frame, origins=[origin])
    assert audit["passed"] is True
    assert audit["max_abs_difference"] == 0.0


def test_oof_refits_forward_with_only_matured_labels_and_keeps_audit_metadata() -> None:
    frame = _frame(240)
    validation_start = frame.index[145]
    fold = TimeFold(
        name="dev_01",
        train_start=frame.index[0],
        train_end=validation_start - pd.Timedelta(minutes=135),
        validation_start=validation_start,
        validation_end=frame.index[153],
    )
    rows, report = build_causal_rolling_oof(
        frame,
        config=_config(),
        folds=[fold],
        forward_refit=True,
    )
    required = {
        "fold",
        "origin_time",
        "train_end",
        "label_maturity_end",
        "target",
        "horizon",
        "actual",
        "actual_delta",
        "prediction",
        "causal_rolling_prediction",
    }
    assert required.issubset(rows.columns)
    assert len(rows) == 8 * 2 * 8
    assert set(rows["horizon"]) == set(range(15, 121, 15))
    assert (pd.to_datetime(rows["label_maturity_end"]) <= rows["origin_time"]).all()
    assert report["forward_refit"] is True
    assert report["blind_labels_used"] is False
    assert report["feature_manifest"]["feature_columns"]
    assert [item["minutes"] for item in report["horizon_metadata"]] == list(range(15, 121, 15))

    blind = TimeFold(
        name="blind",
        train_start=fold.train_start,
        train_end=fold.train_end,
        validation_start=fold.validation_start,
        validation_end=fold.validation_end,
        blind=True,
    )
    with pytest.raises(ValueError, match="blind"):
        build_causal_rolling_oof(frame, config=_config(), folds=[blind], include_blind=True)


def test_short_history_persists_and_invalid_current_target_is_rejected() -> None:
    frame = _frame()
    model = CausalRollingReconstructionForecaster(_config()).fit(frame.iloc[:180])
    origin = frame.index[205]
    short_history = frame.loc[[origin]]
    prediction = model.predict_at_origin(short_history)
    assert model.last_prediction_metadata_["mode"] == "persistence"
    for minutes in range(15, 121, 15):
        assert prediction.at[origin, f"generator_1_t+{minutes}_pred"] == frame.at[origin, "generator_1"]

    invalid = frame.loc[:origin].copy()
    invalid.at[origin, "generator_1"] = np.inf
    with pytest.raises(ValueError, match="非有限"):
        model.predict_at_origin(invalid)


def test_missing_optional_resources_and_nonfinite_optional_values_are_imputed() -> None:
    frame = _frame()
    reduced = frame.loc[:, ["generator_1", "generator_all"]].copy()
    reduced["unused_sensor"] = np.inf
    model = CausalRollingReconstructionForecaster(_config()).fit(reduced.iloc[:180])
    assert model.feature_manifest().source_mapping.holder_columns == ()
    assert model.feature_manifest().source_mapping.gas_supply_columns == ()
    prediction = model.predict_at_origin(reduced.loc[: reduced.index[205]])
    assert np.isfinite(prediction.to_numpy(dtype=float)).all()
