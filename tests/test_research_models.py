from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig, ForecastConfig, ModelConfig
from gas_forecast.features import build_causal_features
from gas_forecast.research_models import (
    Generator1IncrementalPathForecaster,
    Generator1CatBoostForecaster,
    Generator1HorizonRidgeForecaster,
    Generator1LightGBMForecaster,
    Generator1StateExpertForecaster,
    select_generator_all_features,
    select_generator1_features,
    smooth_prediction_paths,
)
from gas_forecast.targets import build_delta_targets


def test_generator1_horizon_ridge_routes_generator_all_and_respects_capacity() -> None:
    rows = 480
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + 10.0 * np.sin(phase / 18.0),
            "generator_all": 230.0 + 14.0 * np.sin(phase / 21.0),
            "generator_use_blast_furnace_gas": 600_000.0 + 500.0 * phase,
            "blast_furnace_gas_holder_2": 30_000.0 + 100.0 * np.cos(phase / 12.0),
        },
        index=index,
    )
    feature_config = FeatureConfig(
        horizons=(1, 2),
        rolling_windows=(4, 8),
        enable_target_aligned_features=True,
    )
    config = ForecastConfig(
        feature=feature_config,
        model=ModelConfig(
            generator1_feature_profile="core",
            generator_all_route_model="horizon_ridge",
        ),
    )
    features = build_causal_features(frame, feature_config)
    deltas = build_delta_targets(frame, config.targets, feature_config.horizons)

    model = Generator1HorizonRidgeForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    )
    predicted = model.predict(features.iloc[-12:], frame[list(config.targets)].iloc[-12:])

    assert set(predicted.columns) == {
        "generator_1_t+15_pred",
        "generator_1_t+30_pred",
        "generator_all_t+15_pred",
        "generator_all_t+30_pred",
    }
    assert np.isfinite(predicted.to_numpy()).all()
    for horizon in feature_config.horizons:
        first = predicted[f"generator_1_t+{15 * horizon}_pred"]
        total = predicted[f"generator_all_t+{15 * horizon}_pred"]
        assert (total >= first).all()
        assert (total - first <= 240.0 + 1e-9).all()


def test_generator1_route_keeps_v3_generator_all_baseline_by_default() -> None:
    rows = 440
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + 4.0 * np.sin(phase / 11.0),
            "generator_all": 220.0 + 7.0 * np.sin(phase / 13.0),
            "generator_use_blast_furnace_gas": 500_000.0 + 10.0 * phase,
        },
        index=index,
    )
    feature_config = FeatureConfig(horizons=(1, 2), rolling_windows=(4, 8))
    config = ForecastConfig(
        feature=feature_config,
        model=ModelConfig(lgb_n_estimators=8, tree_threads_per_worker=1),
    )
    features = build_causal_features(frame, feature_config)
    deltas = build_delta_targets(frame, config.targets, feature_config.horizons)

    model = Generator1HorizonRidgeForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    )
    predicted = model.predict(features.iloc[-4:], frame[list(config.targets)].iloc[-4:])

    assert np.isfinite(predicted.to_numpy()).all()


def test_generator1_core_profile_ignores_unrelated_measurements() -> None:
    rows = 420
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + 7.0 * np.sin(phase / 15.0),
            "generator_all": 220.0 + 10.0 * np.sin(phase / 17.0),
            "generator_use_blast_furnace_gas": 500_000.0 + phase,
            "unrelated_measurement": phase,
        },
        index=index,
    )
    feature_config = FeatureConfig(horizons=(1, 2), rolling_windows=(4, 8))
    config = ForecastConfig(
        feature=feature_config,
        model=ModelConfig(
            generator1_feature_profile="core",
            generator_all_route_model="horizon_ridge",
        ),
    )
    features = build_causal_features(frame, feature_config)
    deltas = build_delta_targets(frame, config.targets, feature_config.horizons)
    model = Generator1HorizonRidgeForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    )
    altered = features.iloc[-8:].copy()
    altered["unrelated_measurement"] = 1_000_000_000.0

    baseline = model.predict(features.iloc[-8:], frame[list(config.targets)].iloc[-8:])
    changed = model.predict(altered, frame[list(config.targets)].iloc[-8:])

    pd.testing.assert_frame_equal(
        baseline.filter(like="generator_1_"), changed.filter(like="generator_1_")
    )


def test_generator_all_baseline_excludes_generator1_research_only_features() -> None:
    features = pd.DataFrame(
        {
            "generator_all": [220.0],
            "feat_generator_1_aligned_h1_mean": [100.0],
            "feat_slot_0": [1.0],
            "feat_price_delta_tplus_15": [2.0],
            "feat_dynamic_unrelated_measurement_lag_1": [3.0],
        }
    )

    selected = select_generator_all_features(features)

    assert list(selected.columns) == ["generator_all"]


def test_generator1_selector_hides_superset_features_until_their_phase_is_enabled() -> None:
    features = pd.DataFrame(
        {
            "generator_1": [100.0],
            "feat_generator_1_same_slot_mean_3d": [99.0],
            "feat_generator_1_aligned_h1_mean": [101.0],
            "feat_slot_0": [1.0],
            "feat_dynamic_unrelated_measurement_lag_1": [2.0],
        }
    )
    config = ForecastConfig(
        feature=FeatureConfig(
            enable_target_aligned_features=False,
            enable_long_cycle_features=False,
            enable_slot_one_hot=False,
            dynamic_feature_scope="none",
        )
    )

    selected = select_generator1_features(features, "core", config)

    assert list(selected.columns) == ["generator_1"]


def test_generator1_weighted_lad_supports_magnitude_and_recency_weights() -> None:
    rows = 420
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + 4.0 * np.sin(phase / 11.0),
            "generator_all": 220.0 + 7.0 * np.sin(phase / 13.0),
            "generator_use_blast_furnace_gas": 500_000.0 + phase,
        },
        index=index,
    )
    feature_config = FeatureConfig(horizons=(1, 2), rolling_windows=(4, 8))
    config = ForecastConfig(
        feature=feature_config,
        model=ModelConfig(
            generator1_feature_profile="core",
            generator_all_route_model="horizon_ridge",
            ridge_loss="weighted_lad",
            ridge_magnitude_weighting="inverse_absolute",
            ridge_recency_mode="exp",
            ridge_half_life_days=30.0,
        ),
    )
    features = build_causal_features(frame, feature_config)
    deltas = build_delta_targets(frame, config.targets, feature_config.horizons)

    predicted = Generator1HorizonRidgeForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    ).predict(features.iloc[-6:], frame[list(config.targets)].iloc[-6:])

    assert np.isfinite(predicted.to_numpy()).all()


def test_path_smoothing_preserves_zero_penalty_and_reduces_path_curvature() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="15min")
    prediction = pd.DataFrame(
        {
            "generator_1_t+15_pred": [100.0, 101.0],
            "generator_1_t+30_pred": [150.0, 151.0],
            "generator_1_t+45_pred": [100.0, 101.0],
            "generator_1_t+60_pred": [150.0, 151.0],
            "generator_all_t+15_pred": [220.0, 221.0],
            "generator_all_t+30_pred": [260.0, 261.0],
            "generator_all_t+45_pred": [220.0, 221.0],
            "generator_all_t+60_pred": [260.0, 261.0],
        },
        index=index,
    )

    unchanged = smooth_prediction_paths(prediction, (1, 2, 3, 4), penalty=0.0)
    smoothed = smooth_prediction_paths(prediction, (1, 2, 3, 4), penalty=1.0)

    pd.testing.assert_frame_equal(unchanged, prediction)
    raw_path = prediction.filter(like="generator_1_").to_numpy()
    smooth_path = smoothed.filter(like="generator_1_").to_numpy()
    raw_curvature = np.abs(np.diff(raw_path, n=2, axis=1)).sum()
    smooth_curvature = np.abs(np.diff(smooth_path, n=2, axis=1)).sum()
    assert smooth_curvature < raw_curvature


def test_incremental_path_forecaster_returns_finite_feasible_paths() -> None:
    rows = 440
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 90.0 + 8.0 * np.sin(phase / 10.0),
            "generator_all": 210.0 + 12.0 * np.sin(phase / 12.0),
            "generator_use_blast_furnace_gas": 550_000.0 + 20.0 * phase,
        },
        index=index,
    )
    feature_config = FeatureConfig(horizons=(1, 2, 3), rolling_windows=(4, 8))
    config = ForecastConfig(
        feature=feature_config,
        model=ModelConfig(
            generator1_feature_profile="core",
            generator_all_route_model="horizon_ridge",
        ),
    )
    features = build_causal_features(frame, feature_config)
    deltas = build_delta_targets(frame, config.targets, feature_config.horizons)

    model = Generator1IncrementalPathForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    )
    predicted = model.predict(features.iloc[-10:], frame[list(config.targets)].iloc[-10:])

    assert predicted.shape == (10, 6)
    assert np.isfinite(predicted.to_numpy()).all()
    assert (predicted["generator_all_t+15_pred"] >= predicted["generator_1_t+15_pred"]).all()


def test_generator1_direct_lightgbm_uses_target_route() -> None:
    rows = 440
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 95.0 + 6.0 * np.sin(phase / 12.0),
            "generator_all": 215.0 + 8.0 * np.sin(phase / 14.0),
            "generator_use_blast_furnace_gas": 510_000.0 + 50.0 * phase,
        },
        index=index,
    )
    feature_config = FeatureConfig(horizons=(1, 2), rolling_windows=(4, 8))
    config = ForecastConfig(
        feature=feature_config,
        model=ModelConfig(
            generator1_feature_profile="core",
            generator_all_route_model="horizon_ridge",
            lgb_n_estimators=12,
            lgb_max_estimators=24,
            lgb_use_early_stopping=False,
        ),
    )
    features = build_causal_features(frame, feature_config)
    deltas = build_delta_targets(frame, config.targets, feature_config.horizons)

    predicted = Generator1LightGBMForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    ).predict(features.iloc[-6:], frame[list(config.targets)].iloc[-6:])

    assert predicted.shape == (6, 4)
    assert np.isfinite(predicted.to_numpy()).all()


def test_generator1_catboost_uses_target_route() -> None:
    rows = 440
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 96.0 + 5.0 * np.sin(phase / 12.0),
            "generator_all": 216.0 + 9.0 * np.sin(phase / 14.0),
            "generator_use_blast_furnace_gas": 515_000.0 + 30.0 * phase,
        },
        index=index,
    )
    feature_config = FeatureConfig(horizons=(1, 2), rolling_windows=(4, 8))
    config = ForecastConfig(
        feature=feature_config,
        model=ModelConfig(
            generator1_feature_profile="core",
            generator_all_route_model="horizon_ridge",
            catboost_iterations=20,
            catboost_early_stopping_rounds=5,
            tree_threads_per_worker=1,
        ),
    )
    features = build_causal_features(frame, feature_config)
    deltas = build_delta_targets(frame, config.targets, feature_config.horizons)

    predicted = Generator1CatBoostForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    ).predict(features.iloc[-6:], frame[list(config.targets)].iloc[-6:])

    assert predicted.shape == (6, 4)
    assert np.isfinite(predicted.to_numpy()).all()


def test_generator1_state_expert_uses_cross_fitted_state_probabilities() -> None:
    rows = 900
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + 8.0 * np.sin(phase / 9.0),
            "generator_all": 225.0 + 12.0 * np.sin(phase / 11.0),
            "generator_use_blast_furnace_gas": 520_000.0 + 100.0 * np.cos(phase / 13.0),
            "blast_furnace_gas_holder_2": 32_000.0 + 200.0 * np.sin(phase / 15.0),
        },
        index=index,
    )
    feature_config = FeatureConfig(horizons=(1, 2, 3, 4), rolling_windows=(4, 8))
    config = ForecastConfig(
        feature=feature_config,
        model=ModelConfig(
            generator1_feature_profile="core",
            generator_all_route_model="horizon_ridge",
            state_expert_inner_folds=2,
        ),
    )
    features = build_causal_features(frame, feature_config)
    deltas = build_delta_targets(frame, config.targets, feature_config.horizons)
    model = Generator1StateExpertForecaster(config).fit(
        features, deltas, frame[list(config.targets)]
    )
    predicted = model.predict(features.iloc[-8:], frame[list(config.targets)].iloc[-8:])

    assert model.generator1_model_.oof_probability_rows_ > 0
    assert predicted.shape == (8, 8)
    assert np.isfinite(predicted.to_numpy()).all()
