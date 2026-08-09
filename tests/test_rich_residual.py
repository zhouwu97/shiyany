from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gas_forecast.config import FeatureConfig, ForecastConfig, ValidationConfig
from gas_forecast.research import select_research_folds
from gas_forecast.rich_residual import (
    LONG_HORIZON_ABLATION_GROUP_ORDER,
    RichResidualSpec,
    build_rich_residual_oof,
    fit_full_rich_residual_corrector,
    long_horizon_feature_group,
)


def _frame(rows: int = 13 * 96) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "generator_1": 100.0 + 7.0 * np.sin(phase / 8.0),
            "generator_all": 220.0 + 9.0 * np.sin(phase / 10.0),
            "generator_use_blast_furnace_gas": 500_000.0 + 100.0 * phase,
            "generator_use_coke_gas": 20_000.0 + 20.0 * phase,
            "generator_use_converter_gas": 30_000.0 + 5.0 * phase,
        },
        index=index,
    )


def _config() -> ForecastConfig:
    return ForecastConfig(
        feature=FeatureConfig(
            horizons=(1,),
            lags=(1, 2, 4),
            diff_lags=(1,),
            rolling_windows=(4, 8),
            rich_quantile_windows=(8,),
        ),
        validation=ValidationConfig(
            first_validation_date="2025-01-04",
            fold_spacing_days=1,
            validation_days=1,
            blind_days=2,
            min_train_days=2,
        ),
    )


def _champion_oof(
    frame: pd.DataFrame,
    config: ForecastConfig,
    *,
    scope: str = "development",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fold in select_research_folds(frame.index, config, scope=scope):
        _, validation_mask = fold.masks(frame.index)
        for origin in frame.index[validation_mask]:
            signal = float(frame.loc[origin, "generator_1"] - 100.0)
            for target, base in (("generator_1", 100.0), ("generator_all", 220.0)):
                residual = 0.5 * signal if target == "generator_1" else 0.0
                records.append(
                    {
                        "fold": fold.name,
                        "origin_time": origin,
                        "target": target,
                        "horizon": 15,
                        "actual": base + residual,
                        "current_value": base,
                        "persistence_pred": base,
                        "aggressive_r75_lgb20_pred": base,
                    }
                )
    return pd.DataFrame(records)


def test_rich_residual_uses_only_prior_oof_folds_for_each_prediction() -> None:
    frame = _frame()
    config = _config()
    champion = _champion_oof(frame, config)
    spec = RichResidualSpec(
        name="test_rich",
        feature_groups=frozenset({"quantile"}),
        min_train_rows=16,
        n_estimators=12,
        blend_weights=(0.10,),
    )

    baseline = build_rich_residual_oof(frame, champion, config=config, spec=spec)
    candidate_column = "test_rich_residual_pred"
    assert candidate_column in baseline.rows
    assert (baseline.rows["target"].eq("generator_all")).any()
    all_rows = baseline.rows["target"].eq("generator_all")
    np.testing.assert_allclose(
        baseline.rows.loc[all_rows, candidate_column],
        baseline.rows.loc[all_rows, "aggressive_r75_lgb20_pred"],
    )

    checked_fold = baseline.report["folds"][2]
    changed = champion.copy()
    changed.loc[changed["fold"].eq(checked_fold), "actual"] += 1_000.0
    perturbed = build_rich_residual_oof(frame, changed, config=config, spec=spec)
    original_values = baseline.rows.loc[
        baseline.rows["fold"].eq(checked_fold), candidate_column
    ]
    changed_values = perturbed.rows.loc[
        perturbed.rows["fold"].eq(checked_fold), candidate_column
    ]
    np.testing.assert_allclose(original_values, changed_values)
    assert baseline.report["fold_training_rows"][checked_fold] > 0


def test_rich_residual_rejects_duplicate_oof_keys() -> None:
    frame = _frame()
    config = _config()
    champion = _champion_oof(frame, config)
    duplicated = pd.concat([champion, champion.iloc[[0]]], ignore_index=True)
    spec = RichResidualSpec(name="test_rich", min_train_rows=16, n_estimators=12)

    with pytest.raises(ValueError, match="重复"):
        build_rich_residual_oof(frame, duplicated, config=config, spec=spec)


def test_full_rich_fit_requires_explicit_blind_oof_authorization() -> None:
    frame = _frame()
    config = _config()
    champion = _champion_oof(frame, config, scope="final")
    spec = RichResidualSpec(name="test_rich", min_train_rows=16, n_estimators=12)

    default_corrector = fit_full_rich_residual_corrector(
        frame,
        champion,
        config=config,
        spec=spec,
    )
    confirmed_corrector = fit_full_rich_residual_corrector(
        frame,
        champion,
        config=config,
        spec=spec,
        allow_confirmed_blind_oof=True,
    )

    default_rows = int(
        champion.loc[
            champion["target"].eq("generator_1") & champion["fold"].ne("blind")
        ].shape[0]
    )
    confirmed_rows = int(champion.loc[champion["target"].eq("generator_1")].shape[0])
    assert default_corrector.states_[15].training_rows == default_rows
    assert confirmed_corrector.states_[15].training_rows == confirmed_rows
    assert confirmed_rows > default_rows


def test_long_horizon_profile_only_modifies_registered_generator1_horizons() -> None:
    frame = _frame()
    config = replace(_config(), feature=replace(_config().feature, horizons=(1, 5, 6, 7, 8)))
    champion = _champion_oof(frame, config)
    spec = RichResidualSpec(
        name="test_long",
        feature_groups=frozenset({"quantile", "ramp", "gas"}),
        feature_profile="long_horizon",
        active_horizons=(75, 90, 105, 120),
        include_champion_prediction=True,
        min_train_rows=16,
        n_estimators=12,
        blend_weights=(0.30,),
    )

    result = build_rich_residual_oof(frame, champion, config=config, spec=spec)
    raw_column = "test_long_residual_raw_pred"
    inactive_generator1 = result.rows["target"].eq("generator_1") & ~result.rows[
        "horizon"
    ].isin(spec.active_horizons)
    generator_all = result.rows["target"].eq("generator_all")
    np.testing.assert_allclose(
        result.rows.loc[inactive_generator1, raw_column],
        result.rows.loc[inactive_generator1, "aggressive_r75_lgb20_pred"],
    )
    np.testing.assert_allclose(
        result.rows.loc[generator_all, raw_column],
        result.rows.loc[generator_all, "aggressive_r75_lgb20_pred"],
    )
    assert result.report["feature_profile"] == "long_horizon"
    assert 1 <= int(result.report["feature_columns"]) <= 250
    assert "feat_champion_prediction" in result.report["selected_feature_columns"]
    assert result.report["active_horizons"] == [75, 90, 105, 120]
    assert result.report["strict_oof_contract"]["blind_labels_used"] is False
    assert result.report["strict_oof_contract"]["champion_prediction_is_production_available"]
    for horizons in result.report["trained_horizons"].values():
        assert set(horizons).issubset(set(spec.active_horizons))


def test_long_horizon_profile_can_target_generator_all() -> None:
    frame = _frame()
    config = replace(_config(), feature=replace(_config().feature, horizons=(1, 5, 6, 7, 8)))
    champion = _champion_oof(frame, config)
    spec = RichResidualSpec(
        name="test_gall_long",
        target="generator_all",
        feature_groups=frozenset({"quantile", "ramp", "gas"}),
        feature_profile="long_horizon",
        active_horizons=(75, 90, 105, 120),
        include_champion_prediction=True,
        min_train_rows=16,
        n_estimators=12,
        blend_weights=(0.30,),
    )

    result = build_rich_residual_oof(frame, champion, config=config, spec=spec)

    raw_column = "test_gall_long_residual_raw_pred"
    generator_1 = result.rows["target"].eq("generator_1")
    inactive_generator_all = result.rows["target"].eq("generator_all") & ~result.rows[
        "horizon"
    ].isin(spec.active_horizons)
    np.testing.assert_allclose(
        result.rows.loc[generator_1, raw_column],
        result.rows.loc[generator_1, "aggressive_r75_lgb20_pred"],
    )
    np.testing.assert_allclose(
        result.rows.loc[inactive_generator_all, raw_column],
        result.rows.loc[inactive_generator_all, "aggressive_r75_lgb20_pred"],
    )
    assert result.report["target_scope"] == "generator_all"
    assert result.report["strict_oof_contract"]["target_scope"] == "generator_all"


def test_generator_all_residual_history_excludes_held_fold_labels() -> None:
    frame = _frame()
    config = replace(_config(), feature=replace(_config().feature, horizons=(1, 5, 6, 7, 8)))
    champion = _champion_oof(frame, config)
    spec = RichResidualSpec(
        name="test_gall_history",
        target="generator_all",
        feature_groups=frozenset({"quantile", "ramp", "gas"}),
        feature_profile="long_horizon",
        active_horizons=(75, 90, 105, 120),
        include_champion_prediction=True,
        min_train_rows=16,
        n_estimators=12,
        blend_weights=(0.30,),
    )

    baseline = build_rich_residual_oof(frame, champion, config=config, spec=spec)
    checked_fold = baseline.report["folds"][2]
    changed = champion.copy()
    changed.loc[
        changed["fold"].eq(checked_fold) & changed["target"].eq("generator_all"), "actual"
    ] += 10_000.0
    perturbed = build_rich_residual_oof(frame, changed, config=config, spec=spec)
    column = "test_gall_history_residual_pred"
    original_values = baseline.rows.loc[
        baseline.rows["fold"].eq(checked_fold) & baseline.rows["target"].eq("generator_all"),
        column,
    ]
    changed_values = perturbed.rows.loc[
        perturbed.rows["fold"].eq(checked_fold) & perturbed.rows["target"].eq("generator_all"),
        column,
    ]
    np.testing.assert_allclose(original_values, changed_values)


def test_long_horizon_spec_rejects_invalid_minute_step() -> None:
    with pytest.raises(ValueError, match="15 分钟"):
        RichResidualSpec(name="test_long", active_horizons=(80,))


def test_long_horizon_ablation_excludes_only_registered_group() -> None:
    frame = _frame()
    config = replace(_config(), feature=replace(_config().feature, horizons=(1, 5, 6, 7, 8)))
    champion = _champion_oof(frame, config)
    spec = RichResidualSpec(
        name="test_ablation",
        feature_groups=frozenset({"quantile", "ramp", "gas"}),
        feature_profile="long_horizon",
        active_horizons=(75, 90, 105, 120),
        include_champion_prediction=True,
        exclude_long_feature_groups=frozenset({"branch_prediction_disagreement"}),
        min_train_rows=16,
        n_estimators=12,
        blend_weights=(0.30,),
    )

    result = build_rich_residual_oof(frame, champion, config=config, spec=spec)

    assert result.report["excluded_long_feature_groups"] == [
        "branch_prediction_disagreement"
    ]
    assert result.report["champion_prediction_feature"] is False
    assert "feat_champion_prediction" not in result.report["selected_feature_columns"]
    counts = result.report["long_horizon_feature_group_counts"]
    assert set(counts) == set(LONG_HORIZON_ABLATION_GROUP_ORDER)
    assert counts["branch_prediction_disagreement"] == 0
    assert long_horizon_feature_group("feat_target_price_tplus_75") == "time_price"
    assert long_horizon_feature_group("feat_rich_ramp_generator_1_rate") == "quantile_ramp_state"


def test_long_horizon_ablation_rejects_unknown_or_wrong_profile_group() -> None:
    with pytest.raises(ValueError, match="未知 long_horizon"):
        RichResidualSpec(
            name="test_unknown_group",
            exclude_long_feature_groups=frozenset({"unknown"}),
        )
    with pytest.raises(ValueError, match="只能用于 long_horizon"):
        RichResidualSpec(
            name="test_wrong_profile",
            exclude_long_feature_groups=frozenset({"generation_dynamics"}),
        )
