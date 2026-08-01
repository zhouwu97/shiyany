"""带 purge 的外层滚动折与内层 expanding cross-fitting。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig


@dataclass(frozen=True)
class TimeFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    blind: bool = False

    def masks(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        train = (index >= self.train_start) & (index <= self.train_end)
        validation = (index >= self.validation_start) & (index < self.validation_end)
        return np.asarray(train), np.asarray(validation)


def make_outer_folds(index: pd.DatetimeIndex, config: ForecastConfig) -> list[TimeFold]:
    """生成与旧验证边界兼容、但显式登记训练起止时间的外层折。"""

    rule = config.validation
    purge = pd.Timedelta(minutes=15 * max(config.feature.horizons))
    first = max(pd.Timestamp(rule.first_validation_date), index.min() + pd.Timedelta(days=rule.min_train_days))
    blind_start = index.max().normalize() - pd.Timedelta(days=rule.blind_days - 1)
    folds: list[TimeFold] = []
    start = first
    number = 1
    while start + pd.Timedelta(days=rule.validation_days) <= blind_start:
        folds.append(
            TimeFold(
                name=f"dev_{number:02d}",
                train_start=index.min(),
                train_end=start - purge,
                validation_start=start,
                validation_end=start + pd.Timedelta(days=rule.validation_days),
            )
        )
        number += 1
        start += pd.Timedelta(days=rule.fold_spacing_days)
    folds.append(
        TimeFold(
            name="blind",
            train_start=index.min(),
            train_end=blind_start - purge,
            validation_start=blind_start,
            validation_end=index.max() + pd.Timedelta(minutes=15),
            blind=True,
        )
    )
    return folds


def make_inner_folds(
    index: pd.DatetimeIndex,
    *,
    folds: int = 5,
    purge_steps: int = 8,
    min_train_rows: int = 384,
    min_validation_rows: int = 96,
) -> list[TimeFold]:
    """在单个外层训练集内部生成 expanding 子折。"""

    if folds < 2:
        raise ValueError("内层 cross-fitting 至少需要2折")
    if len(index) < min_train_rows + folds * min_validation_rows + purge_steps:
        available = len(index) - min_train_rows - purge_steps
        folds = min(folds, available // min_validation_rows)
    if folds < 2:
        raise ValueError("训练数据不足以生成至少2个内层时间折")

    validation_rows = max(min_validation_rows, (len(index) - min_train_rows - purge_steps) // folds)
    first_validation = len(index) - folds * validation_rows
    result: list[TimeFold] = []
    for position in range(folds):
        validation_start_position = first_validation + position * validation_rows
        validation_end_position = min(len(index), validation_start_position + validation_rows)
        train_end_position = validation_start_position - purge_steps
        if train_end_position < min_train_rows:
            continue
        result.append(
            TimeFold(
                name=f"inner_{position + 1:02d}",
                train_start=index[0],
                train_end=index[train_end_position - 1],
                validation_start=index[validation_start_position],
                validation_end=(
                    index[validation_end_position]
                    if validation_end_position < len(index)
                    else index[-1] + (index[-1] - index[-2])
                ),
            )
        )
    if len(result) < 2:
        raise ValueError("有效内层时间折少于2个")
    return result
