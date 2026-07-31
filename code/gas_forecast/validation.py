"""前向滚动切分与误差汇总。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig
from gas_forecast.features import build_delta_targets
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster
from gas_forecast.model_v1 import RidgeDeltaForecaster


@dataclass(frozen=True)
class RollingFold:
    name: str
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_end: pd.Timestamp
    blind: bool = False


def make_rolling_folds(index: pd.DatetimeIndex, config: ForecastConfig) -> list[RollingFold]:
    rule = config.validation
    first = max(pd.Timestamp(rule.first_validation_date), index.min() + pd.Timedelta(days=rule.min_train_days))
    blind_start = index.max().normalize() - pd.Timedelta(days=rule.blind_days - 1)
    folds: list[RollingFold] = []
    start = first
    number = 1
    while start + pd.Timedelta(days=rule.validation_days) <= blind_start:
        folds.append(
            RollingFold(
                name=f"dev_{number:02d}",
                validation_start=start,
                validation_end=start + pd.Timedelta(days=rule.validation_days),
                train_end=start - pd.Timedelta(minutes=15 * max(config.feature.horizons)),
            )
        )
        number += 1
        start += pd.Timedelta(days=rule.fold_spacing_days)
    folds.append(
        RollingFold(
            name="blind",
            validation_start=blind_start,
            validation_end=index.max() + pd.Timedelta(minutes=15),
            train_end=blind_start - pd.Timedelta(minutes=15 * max(config.feature.horizons)),
            blind=True,
        )
    )
    return folds


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.maximum(np.abs(actual), 1e-6)
    return float(np.mean(np.abs(actual - predicted) / denominator))


def backtest_model(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    version: str,
    config: ForecastConfig | None = None,
    *,
    max_folds: int | None = None,
) -> dict[str, object]:
    config = config or ForecastConfig()
    deltas = build_delta_targets(frame, config.targets, config.feature.horizons)
    folds = make_rolling_folds(frame.index, config)
    if max_folds is not None:
        folds = folds[-max_folds:]
    fold_results: list[dict[str, object]] = []

    for fold in folds:
        train_mask = features.index < fold.train_end
        validation_mask = (features.index >= fold.validation_start) & (
            features.index < fold.validation_end
        )
        model = (
            RidgeDeltaForecaster(config)
            if version == "v1"
            else GasAwareEnsembleForecaster(version, config)
        ).fit(
            features.loc[train_mask],
            deltas.loc[train_mask],
            frame.loc[train_mask, list(config.targets)],
        )
        predicted = model.predict(
            features.loc[validation_mask],
            features.loc[validation_mask, list(config.targets)],
        )

        scores: dict[str, float] = {}
        persistence_scores: dict[str, float] = {}
        for target in config.targets:
            for horizon in config.feature.horizons:
                minutes = 15 * horizon
                actual = frame[target].shift(-horizon).loc[validation_mask]
                valid = actual.notna() & frame.loc[validation_mask, target].notna()
                key = f"{target}_t+{minutes}"
                scores[key] = mape(
                    actual.loc[valid].to_numpy(),
                    predicted.loc[valid, f"{target}_t+{minutes}_pred"].to_numpy(),
                )
                persistence_scores[key] = mape(
                    actual.loc[valid].to_numpy(),
                    frame.loc[validation_mask, target].loc[valid].to_numpy(),
                )
        fold_results.append(
            {
                **asdict(fold),
                "validation_start": str(fold.validation_start),
                "validation_end": str(fold.validation_end),
                "train_end": str(fold.train_end),
                "mape": float(np.mean(list(scores.values()))),
                "persistence_mape": float(np.mean(list(persistence_scores.values()))),
                "by_target_horizon": scores,
            }
        )

    return {
        "version": version,
        "folds": fold_results,
        "mean_mape": float(np.mean([item["mape"] for item in fold_results])),
        "mean_persistence_mape": float(
            np.mean([item["persistence_mape"] for item in fold_results])
        ),
        "wins": int(sum(item["mape"] < item["persistence_mape"] for item in fold_results)),
    }


def backtest_v1(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    config: ForecastConfig | None = None,
    *,
    max_folds: int | None = None,
) -> dict[str, object]:
    return backtest_model(frame, features, "v1", config, max_folds=max_folds)
