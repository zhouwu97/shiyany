from __future__ import annotations

import pandas as pd

from gas_forecast.splits import make_inner_folds


def test_inner_folds_are_expanding_and_purged() -> None:
    index = pd.date_range("2025-01-01", periods=1400, freq="15min")

    folds = make_inner_folds(index, folds=4, purge_steps=8)

    assert len(folds) == 4
    assert all(
        fold.train_end <= fold.validation_start - pd.Timedelta(minutes=120)
        for fold in folds
    )
    assert [fold.train_end for fold in folds] == sorted(fold.train_end for fold in folds)
