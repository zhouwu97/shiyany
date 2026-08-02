"""Sanity check: 打印 generator_1 每个预测步长的绝对发电量分布。

检查对象是 actual = anchor + delta（未来绝对发电量），不是 residual/delta。
用于评估 A1 competition sample_weight 的极端权重风险。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.splits import make_outer_folds
from gas_forecast.targets import build_delta_targets, target_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 generator_1 绝对发电量分布")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--fold-index",
        type=int,
        default=-1,
        help="折索引；-1 = blind fold，0..N-1 = dev folds；默认 -1",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="聚合所有 development fold 的 validation 样本",
    )
    return parser.parse_args()


def _describe_actual(actual: np.ndarray, horizons: list[int]) -> None:
    """打印每个 horizon 的分布统计。"""
    header = f"{'horizon':>8}  {'min':>8}  {'p0.1':>8}  {'p1':>8}  {'p5':>8}  {'median':>8}  {'max':>8}  {'<1':>5}  {'<5':>5}  {'<10':>5}"
    print(header)
    print("-" * len(header))
    for i, h in enumerate(horizons):
        col = actual[:, i]
        abs_col = np.abs(col)
        print(
            f"  t+{h:>3d}  "
            f"{col.min():>8.2f}  "
            f"{np.percentile(col, 0.1):>8.2f}  "
            f"{np.percentile(col, 1):>8.2f}  "
            f"{np.percentile(col, 5):>8.2f}  "
            f"{np.median(col):>8.2f}  "
            f"{col.max():>8.2f}  "
            f"{int((abs_col < 1).sum()):>5d}  "
            f"{int((abs_col < 5).sum()):>5d}  "
            f"{int((abs_col < 10).sum()):>5d}"
        )


def main() -> None:
    args = parse_args()
    config = ForecastConfig()
    dataset = align_tables(args.data_dir, config.feature.frequency)
    prices = sorted(args.data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    features = build_causal_features(dataset.frame, config.feature, price)

    target = "generator_1"
    horizons = config.feature.horizons
    columns = target_columns(target, horizons)

    # build_delta_targets constructs shift-based delta labels (same as training)
    all_deltas = build_delta_targets(dataset.frame, config.targets, horizons)

    folds = make_outer_folds(dataset.frame.index, config)

    if args.all_folds:
        all_actual: list[np.ndarray] = []
        dev_folds = folds[:-1]  # last fold is blind; preceding ones are dev
        print(f"\n=== absolute generator_1 target — pooled {len(dev_folds)} dev folds ===\n")
        for fold in dev_folds:
            _, val_mask = fold.masks(dataset.frame.index)
            current = dataset.frame.loc[val_mask, target]
            deltas = all_deltas.loc[val_mask, columns]
            valid = current.notna() & deltas.notna().all(axis=1)
            anchor = current.loc[valid].to_numpy(dtype=float)
            dy = deltas.loc[valid].to_numpy(dtype=float)
            all_actual.append(anchor[:, None] + dy)
        combined = np.concatenate(all_actual, axis=0)
        _describe_actual(combined, horizons)
    else:
        target_fold = folds[args.fold_index]
        _, val_mask = target_fold.masks(dataset.frame.index)
        label = "blind" if args.fold_index == -1 else f"dev fold {args.fold_index}"
        print(f"\n=== absolute generator_1 target — {label} ===\n")
        current = dataset.frame.loc[val_mask, target]
        deltas = all_deltas.loc[val_mask, columns]
        valid = current.notna() & deltas.notna().all(axis=1)
        anchor = current.loc[valid].to_numpy(dtype=float)
        dy = deltas.loc[valid].to_numpy(dtype=float)
        actual = anchor[:, None] + dy
        _describe_actual(actual, horizons)

    print()


if __name__ == "__main__":
    main()
