from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gas_forecast.aggressive import (
    ExperimentRegistry,
    StackingConfig,
    confirm_frozen_blend_on_blind,
    decide_experiment_status,
    diversity_sweep,
    e21_crossing_routes,
    freeze_research_base,
    oracle_gap_diagnostics,
    project_long_candidate,
    read_research_base,
    time_ordered_persistence_stack,
)
from gas_forecast.aggressive_model import AggressiveR75LGBForecaster
from gas_forecast.config import ForecastConfig
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster
from gas_forecast.model_routed import RoutedLegacyForecaster
from gas_forecast.physical_rest import run_x1_blend_grid, time_ordered_physical_rest_oof
from gas_forecast.price_specialist import (
    build_price_switch_features,
    time_ordered_price_corrections,
)
from gas_forecast.second_tier import RecursiveARX, RecursiveARXSpec, fixed_recursive_blends
from scripts.run_aggressive_plan import _merge_candidate_file


def _research_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    start = pd.Timestamp("2025-01-01")
    folds = ("dev_01", "dev_02", "dev_03", "blind")
    horizons = (15, 30, 45, 60)
    for fold_position, fold in enumerate(folds):
        for origin_position in range(60):
            origin = start + pd.Timedelta(days=fold_position * 3, minutes=15 * origin_position)
            current_price = 1.0
            prices = np.ones(8)
            if origin_position % 3 == 0:
                prices[2:] = 1.2
            g1_current = 100.0 + 0.05 * origin_position
            rest_current = 85.0 + 10.0 * np.sin(origin_position / 8)
            for horizon in horizons:
                step = horizon // 15
                g1_actual = g1_current + 0.6 * step + rng.normal(0, 0.2)
                rest_actual = rest_current + 0.4 * step + rng.normal(0, 0.3)
                for target, current, actual in (
                    ("generator_1", g1_current, g1_actual),
                    ("generator_all", g1_current + rest_current, g1_actual + rest_actual),
                ):
                    persistence = current
                    correction = actual - persistence
                    row: dict[str, object] = {
                        "fold": fold,
                        "origin_time": origin,
                        "target": target,
                        "horizon": horizon,
                        "actual": actual,
                        "current_value": current,
                        "persistence_pred": persistence,
                        "ridge_pred": persistence + 0.75 * correction,
                        "recent_ridge_pred": persistence + 0.90 * correction,
                        "gas_ridge_pred": persistence + 0.60 * correction,
                        "lgb_residual_pred": persistence + 1.05 * correction,
                        "c0_pred": persistence + 0.70 * correction,
                        "v2_pred": persistence + 0.65 * correction,
                        "v3_pred": persistence + 0.75 * correction,
                        "feat_current_price": current_price,
                        "feat_generator_gas_total": 20.0 + origin_position / 10,
                        "feat_gas_holder_slope": np.sin(origin_position / 5),
                    }
                    for price_position, price_horizon in enumerate(
                        (15, 30, 45, 60, 75, 90, 105, 120)
                    ):
                        row[f"feat_target_price_tplus_{price_horizon}"] = prices[price_position]
                    records.append(row)
    return pd.DataFrame(records)


def test_phase0_freeze_roundtrip_and_fingerprints(tmp_path) -> None:
    rows = _research_rows()
    metrics = freeze_research_base(
        rows,
        rows,
        tmp_path,
        expected_pooled_mape=None,
        split_payload={"purge_minutes": 135},
    )
    c0, branches = read_research_base(tmp_path)

    assert metrics["rows"] == len(rows)
    assert len(c0) == len(rows)
    assert len(branches) == len(rows)
    fingerprint = json.loads((tmp_path / "split_fingerprint.json").read_text("utf-8"))
    assert fingerprint["payload"]["purge_minutes"] == 135


def test_oracle_and_stacking_are_strictly_time_ordered() -> None:
    rows = _research_rows()
    oracle_rows, oracle = oracle_gap_diagnostics(rows)
    assert oracle["full_oracle_mape"] <= oracle["current_c0_mape"]
    assert np.isfinite(oracle["split_half_oracle_mape"])
    blind = oracle_rows["fold"].eq("blind")
    np.testing.assert_allclose(
        oracle_rows.loc[blind, "full_oracle_pred"], oracle_rows.loc[blind, "c0_pred"]
    )

    config = StackingConfig(
        "target_horizon", lambda_global=0.5, lambda_target=0.25, smooth_horizon=True
    )
    original, report = time_ordered_persistence_stack(rows, config=config)
    changed = rows.copy()
    changed.loc[changed["fold"].eq("dev_03"), "actual"] += 1000.0
    perturbed, _ = time_ordered_persistence_stack(changed, config=config)

    earlier = original["fold"].isin(["dev_01", "dev_02"])
    np.testing.assert_allclose(original.loc[earlier, "stack_pred"], perturbed.loc[earlier, "stack_pred"])
    assert report["weight_trajectory"][0]["fallback"] == "strict_c0"


def test_price_correction_changes_only_switch_rows_and_is_forward_only() -> None:
    rows = _research_rows()
    featured = build_price_switch_features(rows)
    corrected, _ = time_ordered_price_corrections(featured, model="ridge")
    non_switch = corrected["switch_within_120"].eq(0)
    assert (corrected.loc[non_switch, "price_ridge_raw_correction"] == 0).all()

    changed = featured.copy()
    changed.loc[changed["fold"].eq("dev_03"), "actual"] += 500.0
    perturbed, _ = time_ordered_price_corrections(changed, model="ridge")
    earlier = corrected["fold"].isin(["dev_01", "dev_02"])
    np.testing.assert_allclose(
        corrected.loc[earlier, "price_ridge_raw_correction"],
        perturbed.loc[earlier, "price_ridge_raw_correction"],
    )


def test_physical_rest_probabilities_and_x1_contract() -> None:
    rows = _research_rows()
    rest, _ = time_ordered_physical_rest_oof(rows)
    probability_columns = [
        "prob_state_0",
        "prob_state_1",
        "prob_state_2",
        "prob_transition",
    ]
    np.testing.assert_allclose(rest[probability_columns].sum(axis=1), 1.0, atol=1e-12)
    assert rest["physical_rest_pred"].between(0.0, 240.0).all()

    blended, report = run_x1_blend_grid(rows)
    generator_all = blended["target"].eq("generator_all")
    for candidate in [item["candidate"] for item in report["ranking"]]:
        np.testing.assert_allclose(
            blended.loc[generator_all, candidate], blended.loc[generator_all, "c0_pred"]
        )


def test_long_candidate_projection_matches_production_capacity_contract() -> None:
    rows = _research_rows()
    rows["candidate_pred"] = rows["c0_pred"]
    generator_1 = rows["target"].eq("generator_1")
    generator_all = rows["target"].eq("generator_all")
    rows.loc[generator_1, "candidate_pred"] = -10.0
    rows.loc[generator_all, "candidate_pred"] = 500.0

    projected = project_long_candidate(
        rows,
        "candidate_pred",
        output_column="projected_pred",
    )
    assert projected.loc[generator_1, "projected_pred"].eq(0.0).all()
    assert projected.loc[generator_all, "projected_pred"].eq(240.0).all()


def test_aggressive_production_model_applies_r75_lgb20_and_projection() -> None:
    config = ForecastConfig()
    route = {"global": {"selected": "v2_pred"}, "cells": {}}
    c0 = RoutedLegacyForecaster(route, config)
    v2 = GasAwareEnsembleForecaster("v2", config)
    v2.feature_columns_ = []
    v2.ensemble_states_["generator_1"] = SimpleNamespace(branches=object())

    def branch_predictions(models, features, anchor):
        del models, features
        values = np.zeros((len(anchor), 5, 8), dtype=float)
        values[:, 4, :] = 90.0
        return values

    v2._predict_branches = branch_predictions
    c0.models_["v2"] = v2

    columns = [
        f"{target}_t+{15 * horizon}_pred"
        for target in config.targets
        for horizon in config.feature.horizons
    ]

    def c0_prediction(features, current):
        del current
        values = {
            column: np.full(len(features), 100.0 if column.startswith("generator_1") else 350.0)
            for column in columns
        }
        return pd.DataFrame(values, index=features.index)

    c0.predict = c0_prediction

    def e21_prediction(features, current):
        del current
        return pd.DataFrame(
            {
                f"generator_1_t+{15 * horizon}_pred": np.full(len(features), 110.0)
                for horizon in config.feature.horizons
            },
            index=features.index,
        )

    e21 = SimpleNamespace(config=config, predict_generator1_only=e21_prediction)
    model = AggressiveR75LGBForecaster(c0, e21)
    current = pd.DataFrame({"generator_1": [100.0], "generator_all": [350.0]})
    prediction = model.predict(pd.DataFrame(index=[0]), current)

    np.testing.assert_allclose(
        prediction.loc[0, [f"generator_1_t+{15 * h}_pred" for h in range(1, 5)]],
        98.0,
    )
    np.testing.assert_allclose(
        prediction.loc[0, [f"generator_1_t+{15 * h}_pred" for h in range(5, 9)]],
        106.0,
    )
    assert prediction.loc[0, "generator_all_t+15_pred"] == 338.0
    assert prediction.loc[0, "generator_all_t+120_pred"] == 346.0


def test_diversity_registry_and_mechanical_status(tmp_path) -> None:
    rows = _research_rows()
    blended, ranking = diversity_sweep(rows, ("recent_ridge_pred",))
    assert ranking.iloc[0]["best_weight"] in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}
    blind = blended["fold"].eq("blind")
    blend_columns = [column for column in blended if column.startswith("blend_recent")]
    for column in blend_columns:
        np.testing.assert_allclose(blended.loc[blind, column], blended.loc[blind, "c0_pred"])

    confirmed, blind_report = confirm_frozen_blend_on_blind(
        blended,
        challenger_column="recent_ridge_pred",
        baseline_column="c0_pred",
        weight=float(ranking.iloc[0]["best_weight"] or 0.05),
    )
    assert blind_report["selection_used_blind"] is False
    assert blind_report["blind_rows"] == int(blind.sum())
    assert confirmed[blind_report["candidate"]].notna().all()

    rows["e21_pred"] = rows["recent_ridge_pred"]
    e21, _ = e21_crossing_routes(rows)
    for column in [
        value for value in e21 if value.startswith("e21_") and value != "e21_pred"
    ]:
        np.testing.assert_allclose(e21.loc[blind, column], e21.loc[blind, "c0_pred"])

    status, _ = decide_experiment_status(
        delta_pp=-0.006, fold_wins=12, total_folds=20, recent5_wins=3
    )
    assert status == "PROMOTE"
    registry = ExperimentRegistry(tmp_path / "aggressive_registry.csv")
    output = registry.append(
        {
            "experiment_id": "S3-test",
            "model": "stacking",
            "status": status,
            "blind_used": False,
            "leakage_passed": True,
        }
    )
    assert output.iloc[0]["status"] == "PROMOTE"


def test_diversity_candidate_merge_requires_complete_oof_keys(tmp_path) -> None:
    rows = _research_rows()
    candidate = rows.loc[:, ["fold", "origin_time", "target", "horizon"]].copy()
    candidate["external_pred"] = rows["recent_ridge_pred"]
    path = tmp_path / "candidate.parquet"
    candidate.to_parquet(path, index=False)

    merged = _merge_candidate_file(rows, "external_pred", path)
    np.testing.assert_allclose(merged["external_pred"], rows["recent_ridge_pred"])

    candidate.loc[candidate["target"].eq("generator_all"), "external_pred"] = np.nan
    candidate.to_parquet(path, index=False)
    merged = _merge_candidate_file(
        rows,
        "external_pred",
        path,
        fallback_column="c0_pred",
    )
    missing_branch = merged["target"].eq("generator_all")
    np.testing.assert_allclose(
        merged.loc[missing_branch, "external_pred"],
        merged.loc[missing_branch, "c0_pred"],
    )

    candidate.iloc[:-1].to_parquet(path, index=False)
    with np.testing.assert_raises_regex(ValueError, "OOF 键不完整"):
        _merge_candidate_file(rows, "external_pred", path)


def test_recursive_arx_only_uses_registered_price_and_own_predictions() -> None:
    rows = 260
    current = np.linspace(90.0, 110.0, rows)
    features = pd.DataFrame(
        {
            "current": current,
            "lag_1": current - 0.2,
            "lag_2": current - 0.4,
            **{f"price_{step}": np.full(rows, 1.0 + step / 100) for step in range(1, 9)},
        }
    )
    target = pd.Series(current + 0.3)
    spec = RecursiveARXSpec(
        current_column="current",
        lag_columns=("lag_1", "lag_2"),
        future_price_columns=tuple(f"price_{step}" for step in range(1, 9)),
    )
    model = RecursiveARX(spec).fit(features, target)
    prediction = model.predict(features.iloc[-3:])

    assert prediction.shape == (3, 8)
    blends = fixed_recursive_blends(prediction, prediction + 1.0)
    assert set(blends) == {"recursive_blend_05", "recursive_blend_10", "recursive_blend_20"}
