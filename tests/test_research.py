from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd

import numpy as np

import gas_forecast.research as research_module
import gas_forecast.research_models as research_models_module
from gas_forecast.config import FeatureConfig, ForecastConfig, ModelConfig, ValidationConfig
from gas_forecast.research import (
    build_research_oof,
    _generator_all_predictions_by_fold,
    make_online_combination_candidate,
    make_research_candidates,
    select_research_folds,
)


def test_research_registry_covers_planned_experiments_and_hides_blind_until_final() -> None:
    config = ForecastConfig(
        validation=ValidationConfig(
            first_validation_date="2025-03-20",
            fold_spacing_days=2,
            validation_days=2,
            blind_days=4,
            min_train_days=45,
        )
    )
    index = pd.date_range("2025-01-01", "2025-05-31 23:45:00", freq="15min")

    screening = select_research_folds(index, config, scope="screening")
    development = select_research_folds(index, config, scope="development")
    final = select_research_folds(index, config, scope="final")
    catboost = make_research_candidates("E70_catboost_gen1_fixed_metric", config)
    all_ids = {
        candidate.experiment_id
        for experiment_id in (
            "E10_gen1_hridge_base",
            "E11_gen1_hridge_aligned",
            "E12_gen1_hridge_aligned_longcycle",
            "E13_gen1_alpha_group",
            "E20_gen1_recency_hard",
            "E21_gen1_recency_exp",
            "E30_gen1_time_slot",
            "E31_gen1_fourier",
            "E32_gen1_slot_fourier",
            "E40_price_delta",
            "E41_price_interactions",
            "E50_gen1_weighted_ridge",
            "E51_gen1_weighted_lad",
            "E60_aligned_recency",
            "E61_aligned_recency_time",
            "E62_aligned_recency_time_price",
            "E63_best_linear",
            "E70_catboost_gen1_fixed_metric",
            "E80_lgb_direct_gen1",
            "E90_online_bias_true_hot",
            "E91_online_gain_true_hot",
            "E92_online_vintage_true_hot",
            "E100_dynamic_core",
            "E101_dynamic_all",
            "E110_gen1_moe",
            "E120_capacity_projection",
            "E121_path_smoothing",
            "E130_incremental_path",
            "E131_direct_incremental_blend",
        )
        for candidate in make_research_candidates(experiment_id, config)
    }

    assert 1 <= len(screening) <= 5
    assert len(development) >= len(screening)
    assert all(not fold.blind for fold in screening)
    assert all(not fold.blind for fold in development)
    assert final[-1].blind is True
    assert len(catboost) == 4
    assert "E131_direct_incremental_blend" in all_ids


def test_research_oof_scores_a_registered_candidate_on_shared_screening_folds() -> None:
    index = pd.date_range("2025-01-01", periods=12 * 96, freq="15min")
    phase = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + 8.0 * np.sin(phase / 12.0),
            "generator_all": 225.0 + 12.0 * np.sin(phase / 14.0),
            "generator_use_blast_furnace_gas": 500_000.0 + 20.0 * phase,
        },
        index=index,
    )
    config = ForecastConfig(
        feature=FeatureConfig(horizons=(1, 2), lags=(1, 2, 4, 8), rolling_windows=(4, 8)),
        model=ModelConfig(generator_all_route_model="horizon_ridge"),
        validation=ValidationConfig(
            first_validation_date="2025-01-04",
            fold_spacing_days=1,
            validation_days=1,
            blind_days=2,
            min_train_days=2,
        ),
    )
    candidates = make_research_candidates("E10_gen1_hridge_base", config)

    result = build_research_oof(frame, None, candidates, scope="screening")

    assert "e10_base_pred" in result.rows
    assert result.report["scope"] == "screening"
    assert result.report["blind_included"] is False
    assert result.report["models"]["e10_base"]["rows"] == len(result.rows)


def test_research_oof_reuses_feature_matrix_for_candidates_with_same_feature_config(
    monkeypatch,
) -> None:
    """同一特征配置的 alpha 消融不应重复生成因果特征。"""

    index = pd.date_range("2025-01-01", periods=12 * 96, freq="15min")
    phase = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + 8.0 * np.sin(phase / 12.0),
            "generator_all": 225.0 + 12.0 * np.sin(phase / 14.0),
            "generator_use_blast_furnace_gas": 500_000.0 + 20.0 * phase,
        },
        index=index,
    )
    config = ForecastConfig(
        feature=FeatureConfig(horizons=(1, 2), lags=(1, 2, 4, 8), rolling_windows=(4, 8)),
        model=ModelConfig(generator_all_route_model="horizon_ridge"),
        validation=ValidationConfig(
            first_validation_date="2025-01-04",
            fold_spacing_days=1,
            validation_days=1,
            blind_days=2,
            min_train_days=2,
        ),
    )
    first = make_research_candidates("E10_gen1_hridge_base", config)[0]
    second = replace(
        first,
        name="e10_alpha_variant",
        config=replace(first.config, model=replace(first.config.model, ridge_alpha=10.0)),
    )
    original_builder = research_module.build_causal_features
    builds: list[FeatureConfig] = []

    def count_builds(*args, **kwargs):
        builds.append(args[1])
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(research_module, "build_causal_features", count_builds)

    result = build_research_oof(frame, None, [first, second], scope="screening")

    assert result.rows[["e10_base_pred", "e10_alpha_variant_pred"]].notna().all().all()
    assert len(builds) == 1


def test_research_oof_builds_true_hot_start_history_before_validation() -> None:
    index = pd.date_range("2025-01-01", periods=12 * 96, freq="15min")
    phase = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + 7.0 * np.sin(phase / 10.0),
            "generator_all": 224.0 + 10.0 * np.sin(phase / 12.0),
            "generator_use_blast_furnace_gas": 500_000.0 + 25.0 * phase,
        },
        index=index,
    )
    config = ForecastConfig(
        feature=FeatureConfig(horizons=(1, 2), lags=(1, 2, 4, 8), rolling_windows=(4, 8)),
        model=ModelConfig(
            generator_all_route_model="horizon_ridge",
            online_calibration_rows=32,
            online_refit_stride=16,
        ),
        validation=ValidationConfig(
            first_validation_date="2025-01-04",
            fold_spacing_days=1,
            validation_days=1,
            blind_days=2,
            min_train_days=2,
        ),
    )
    candidate = make_research_candidates("E90_online_bias_true_hot", config)[0]

    result = build_research_oof(frame, None, [candidate], scope="screening")

    column = f"{candidate.name}_pred"
    assert result.rows[column].notna().all()
    assert result.report["models"][candidate.name]["pooled_mape"] >= 0.0


def test_online_combination_candidate_is_limited_to_two_frozen_modules() -> None:
    candidate = make_online_combination_candidate(ForecastConfig(), ("bias", "vintage"))

    assert candidate.kind == "online_hot_start"
    assert candidate.online_modes == ("bias", "vintage")


def test_alpha_group_inherits_the_frozen_predecessor_feature_choice() -> None:
    """已被拒绝的对齐特征不能在 alpha 消融中被重新打开。"""

    config = ForecastConfig(
        feature=FeatureConfig(
            enable_target_aligned_features=False,
            enable_long_cycle_features=False,
        )
    )

    candidates = make_research_candidates("E13_gen1_alpha_group", config)

    assert all(
        not candidate.config.feature.enable_target_aligned_features for candidate in candidates
    )
    assert all(not candidate.config.feature.enable_long_cycle_features for candidate in candidates)


def test_alpha_variants_reuse_frozen_generator_all_oof_predictions(monkeypatch) -> None:
    """仅改 generator_1 alpha 时，不能反复拟合冻结的 generator_all 路由。"""

    index = pd.date_range("2025-01-01", periods=12 * 96, freq="15min")
    phase = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + 8.0 * np.sin(phase / 12.0),
            "generator_all": 225.0 + 12.0 * np.sin(phase / 14.0),
            "generator_use_blast_furnace_gas": 500_000.0 + 20.0 * phase,
        },
        index=index,
    )
    config = ForecastConfig(
        feature=FeatureConfig(horizons=(1, 2), lags=(1, 2, 4, 8), rolling_windows=(4, 8)),
        model=ModelConfig(generator_all_route_model="v3"),
        validation=ValidationConfig(
            first_validation_date="2025-01-04",
            fold_spacing_days=1,
            validation_days=1,
            blind_days=2,
            min_train_days=2,
        ),
    )
    baseline = make_research_candidates("E10_gen1_hridge_base", config)[0]
    alpha_variant = replace(
        baseline,
        name="e13_short_alpha_10",
        config=replace(
            baseline.config,
            model=replace(baseline.config.model, generator1_short_alpha=10.0),
        ),
    )
    calls: list[int] = []

    class FrozenGeneratorAll:
        """用于验证路由复用次数的轻量 generator_all 预测器。"""

        def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    f"generator_all_t+{15 * horizon}_pred": current["generator_all"].to_numpy()
                    for horizon in (1, 2)
                },
                index=features.index,
            )

    def fake_fit_generator_all(*args, **kwargs):
        calls.append(1)
        return FrozenGeneratorAll()

    monkeypatch.setattr(
        research_models_module,
        "_fit_generator_all_baseline",
        fake_fit_generator_all,
    )

    result = build_research_oof(
        frame,
        None,
        [baseline, alpha_variant],
        scope="screening",
        baseline_name=baseline.name,
    )

    assert len(calls) == len(result.report["folds"])
    assert result.rows["e13_short_alpha_10_pred"].notna().all()


def test_frozen_generator_all_reuse_allows_generator1_only_recency_changes() -> None:
    """V3 不使用专项 Ridge 的 recency 参数，因而可复用其固定路由。"""

    baseline = make_research_candidates("E10_gen1_hridge_base", ForecastConfig())[0]
    recency_variant = replace(
        baseline,
        name="e21_exp_half_life_30d",
        config=replace(
            baseline.config,
            model=replace(
                baseline.config.model,
                ridge_recency_mode="exp",
                ridge_half_life_days=30.0,
            ),
        ),
    )
    changed_v3_variant = replace(
        baseline,
        name="changed_v3_alpha",
        config=replace(
            baseline.config,
            model=replace(baseline.config.model, ridge_alpha=10.0),
        ),
    )

    assert research_module._can_reuse_frozen_generator_all(baseline, recency_variant)
    assert not research_module._can_reuse_frozen_generator_all(
        baseline, changed_v3_variant
    )


def test_missing_blind_generator_all_rows_fall_back_from_route_reuse() -> None:
    """blind 缺少可评分真值时，不复用基线路由而应交给模型独立预测。"""

    rows = pd.DataFrame(
        {
            "fold": ["dev_01"],
            "origin_time": [pd.Timestamp("2025-03-20")],
            "target": ["generator_all"],
            "horizon": [15],
            "e10_base_pred": [220.0],
        }
    )

    cached = _generator_all_predictions_by_fold(
        rows,
        "e10_base",
        [SimpleNamespace(name="blind")],
        (1,),
    )

    assert cached == {}
