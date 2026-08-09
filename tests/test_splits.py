from __future__ import annotations

import pandas as pd

from gas_forecast.config import ForecastConfig
from gas_forecast.splits import label_end, make_inner_folds, make_outer_folds


def test_inner_folds_are_expanding_and_purged() -> None:
    index = pd.date_range("2025-01-01", periods=1400, freq="15min")

    folds = make_inner_folds(index, folds=4, purge_steps=8)

    assert len(folds) == 4
    assert all(
        fold.train_end <= fold.validation_start - pd.Timedelta(minutes=120)
        for fold in folds
    )
    assert [fold.train_end for fold in folds] == sorted(fold.train_end for fold in folds)


def test_outer_folds_keep_the_longest_training_label_before_validation() -> None:
    index = pd.date_range("2025-01-01", periods=1800, freq="15min")

    folds = make_outer_folds(index, ForecastConfig())

    assert folds
    assert all(
        label_end(fold.train_end, 8) < fold.validation_start
        for fold in folds
    )
    assert all(
        fold.train_end + pd.Timedelta(minutes=120) < fold.validation_start
        for fold in folds
    )
