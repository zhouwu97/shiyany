from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.config import FeatureConfig, ForecastConfig, ValidationConfig
from gas_forecast.research import select_research_folds
from gas_forecast.rich_residual import (
    RichResidualSpec,
    build_rich_residual_oof,
    fit_full_rich_residual_corrector,
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
