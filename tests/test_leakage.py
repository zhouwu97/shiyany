from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig, ForecastConfig
from gas_forecast.features import build_causal_features
from gas_forecast.future_reconstruction import FutureRowReconstructionForecaster
from gas_forecast.leakage import audit_future_perturbations, audit_origin_predictor
from gas_forecast.model_v1 import RidgeDeltaForecaster
from gas_forecast.submission import expected_prediction_columns
from gas_forecast.targets import build_delta_targets
from scripts.production_gate import _expand_policy_sources, _required_false


def test_multi_origin_future_perturbations_leave_features_unchanged() -> None:
    index = pd.date_range("2025-01-01", periods=100, freq="15min")
    phase = np.arange(100, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100 + phase,
            "generator_all": 220 + phase,
            "generator_use_blast_furnace_gas": 500_000 + phase,
            "generator_use_coke_gas": 10_000 + phase,
            "generator_use_converter_gas": 20_000 + phase,
            "blast_furnace_gas_holder_2": 100_000 + phase,
        },
        index=index,
    )
    config = FeatureConfig(horizons=(1, 2), lags=(1, 2), rolling_windows=(4,))

    report = audit_future_perturbations(
        frame,
        lambda value: build_causal_features(value, config),
        origins=5,
    )

    assert report["passed"] is True
    assert report["origins"] == 5
    assert report["cases_checked"] == 25


def _prediction_columns() -> list[str]:
    return expected_prediction_columns()


def _production_frame(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "generator_1": 80.0 + phase * 0.03,
            "generator_all": 220.0 + phase * 0.05,
            "gas": 500_000.0 + phase * 7.0,
        },
        index=index,
    )


def test_model_level_audit_passes_causal_predictor_and_covers_all_single_fields() -> None:
    frame = _production_frame()
    config = ForecastConfig(
        feature=FeatureConfig(lags=(1, 2), diff_lags=(1,), rolling_windows=(4,))
    )
    features = build_causal_features(frame, config.feature)
    deltas = build_delta_targets(frame, config.targets, config.feature.horizons)
    model = RidgeDeltaForecaster(config).fit(features, deltas, frame.loc[:, list(config.targets)])

    def predictor(value: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        origin_features = build_causal_features(value, config.feature).loc[[origin]]
        current = value.loc[[origin], list(config.targets)]
        return model.predict(origin_features, current)

    report = audit_origin_predictor(frame, predictor, origins=5)

    assert report["passed"] is True
    assert report["cases_checked"] == 5 * (4 + 3)
    assert report["methods"] == ["extreme", "shuffle", "null", "single_field", "delete_future"]


def test_model_level_audit_rejects_future_row_reconstruction_with_precise_failure() -> None:
    training = _production_frame(rows=900)
    scoring = _production_frame(rows=320).set_axis(
        pd.date_range("2025-05-01", periods=320, freq="15min")
    )
    model = FutureRowReconstructionForecaster(validation_rows=96).fit(training)

    def predictor(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        base = pd.DataFrame(
            {
                column: 100.0 if column.startswith("generator_1") else 300.0
                for column in _prediction_columns()
            },
            index=frame.index,
        )
        prediction, _ = model.predict(frame, base)
        return prediction.loc[[origin]]

    report = audit_origin_predictor(scoring, predictor, origins=5)

    assert report["passed"] is False
    assert report["cases_checked"] == 5 * (4 + 3)
    failure = next(
        item
        for item in report["failures"]
        if item["mutation"] == "single_field" and item["column"] == "generator_1"
    )
    assert failure["max_abs_diff"] > 0.0
    assert failure["max_diff"] == failure["max_abs_diff"]
    assert failure["changed_prediction_columns"]
    assert any(item["mutation"] == "delete_future" for item in report["failures"])


def test_model_level_audit_prioritizes_future_gas_and_delete_future() -> None:
    frame = _production_frame()

    def predictor(value: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        future_gas = float(value.loc[value.index > origin, "gas"].iloc[0])
        return pd.DataFrame([[future_gas] * 16], columns=_prediction_columns(), index=[origin])

    report = audit_origin_predictor(frame, predictor, origins=1)

    assert report["passed"] is False
    assert any(
        item["mutation"] == "single_field" and item["column"] == "gas"
        for item in report["failures"]
    )
    deletion = next(item for item in report["failures"] if item["mutation"] == "delete_future")
    assert deletion["column"] == "__all_numeric_columns__"
    assert deletion["max_diff"] == float("inf")


def test_production_declarations_require_explicit_non_conflicting_false() -> None:
    assert _required_false("oracle_candidate", {"oracle_candidate": False}, {}) is True
    assert _required_false("oracle_candidate", {"oracle_candidate": False}, {"oracle_candidate": True}) is False
    assert _required_false("blind_labels_used", {}, {}) is False
    sources = _expand_policy_sources(
        {"causal_prediction_audit": {"oracle_candidate": False}},
        {},
    )
    assert _required_false("oracle_candidate", *sources) is True
