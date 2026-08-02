"""P2-0 诊断: V2 generator_1 branch 权重与 MAPE。

用法:
    python scripts/diagnose_v2_branches.py \\
        --data-dir "data/raw/official/初赛-参赛者使用" \\
        --fold-index -1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import legacy_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.model_ensemble import BRANCH_NAMES, GasAwareEnsembleForecaster
from gas_forecast.splits import make_outer_folds
from gas_forecast.targets import build_delta_targets


_EPS = 1e-6


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(pred)
    return float(
        np.mean(np.abs(actual[valid] - pred[valid]) / np.maximum(np.abs(actual[valid]), _EPS))
    ) * 100.0


def _resolve_data_dir(path: Path) -> Path:
    if (path / "Pre_gas.csv").exists():
        return path
    matches = sorted(c for c in path.iterdir() if c.is_dir() and (c / "Pre_gas.csv").exists())
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"无法解析官方数据目录: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 branch 权重与 MAPE 诊断 (P2-0)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--fold-index", type=int, default=-1,
        help="外层折索引 (Python 列表索引), 默认 -1 = 最后一折 (blind)",
    )
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    config = legacy_forecast_config()
    data_dir = _resolve_data_dir(args.data_dir)
    dataset = align_tables(data_dir, config.feature.frequency)
    prices = sorted(data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    features = build_causal_features(dataset.frame, config.feature, price)
    deltas = build_delta_targets(dataset.frame, config.targets, config.feature.horizons)

    folds = make_outer_folds(dataset.frame.index, config)
    fold = folds[args.fold_index]
    train_mask, validation_mask = fold.masks(dataset.frame.index)
    print(f"fold={fold.name} train_end={fold.train_end} "
          f"validation=[{fold.validation_start}, {fold.validation_end}) "
          f"n_train={int(train_mask.sum())} n_val={int(validation_mask.sum())}")

    model = GasAwareEnsembleForecaster("v2", config).fit(
        features.loc[train_mask],
        deltas.loc[train_mask],
        dataset.frame.loc[train_mask, list(config.targets)],
    )

    target = "generator_1"
    state = model.ensemble_states_[target]
    horizons = list(config.feature.horizons)

    # === 表 1: blend_weights (5 branches x 8 horizons) ===
    weights = pd.DataFrame(
        state.blend_weights,
        index=list(BRANCH_NAMES),
        columns=[f"t+{15 * h}" for h in horizons],
    ).T
    print("\n=== blend_weights (generator_1) ===")
    print(weights.round(4).to_string())
    print("\nmean weight per branch:")
    print(weights.mean(axis=0).round(4).to_string())

    # === 验证集上逐 branch MAPE ===
    validation_features = features.loc[validation_mask]
    x = validation_features.reindex(columns=model.feature_columns_)
    current = validation_features[list(config.targets)]
    anchor = current[target].ffill().to_numpy(dtype=float)
    branches = model._predict_branches(state.branches, x, anchor)  # (n, 5, 8)
    blended = np.einsum("nbh,bh->nh", branches, state.blend_weights)

    validation_index = dataset.frame.index[validation_mask]
    actual = np.column_stack([
        dataset.frame[target].shift(-h).loc[validation_index].to_numpy(dtype=float)
        for h in horizons
    ])

    mape_rows = []
    for step, horizon in enumerate(horizons):
        row = {"horizon": f"t+{15 * horizon}"}
        for b, name in enumerate(BRANCH_NAMES):
            row[name] = _mape(actual[:, step], branches[:, b, step])
        row["BLEND"] = _mape(actual[:, step], blended[:, step])
        mape_rows.append(row)
    mape_table = pd.DataFrame(mape_rows).set_index("horizon")
    print("\n=== per-branch MAPE %% (generator_1, fold=%s) ===" % fold.name)
    print(mape_table.round(4).to_string())
    pooled = {name: _mape(actual.ravel(), branches[:, b, :].ravel())
              for b, name in enumerate(BRANCH_NAMES)}
    pooled["BLEND"] = _mape(actual.ravel(), blended.ravel())
    print("\npooled MAPE per branch:")
    print(pd.Series(pooled).round(4).to_string())

    # === branch 残差相关性 (pooled over horizons) ===
    residuals = np.stack(
        [branches[:, b, :] - actual for b in range(len(BRANCH_NAMES))], axis=0
    ).reshape(len(BRANCH_NAMES), -1)
    finite = np.isfinite(residuals).all(axis=0)
    corr = pd.DataFrame(
        np.corrcoef(residuals[:, finite]),
        index=list(BRANCH_NAMES),
        columns=list(BRANCH_NAMES),
    )
    print("\n=== branch residual correlation (generator_1) ===")
    print(corr.round(3).to_string())

    # === LGB residual 重要性 ===
    print("\n=== lgb_residual feature importance top-20 (mean over 8 horizons) ===")
    importance = np.zeros(len(model.feature_columns_), dtype=float)
    for lgb in state.branches.residual_models:
        importance += np.asarray(lgb.feature_importances_, dtype=float)
    importance /= max(len(state.branches.residual_models), 1)
    ranked = pd.Series(importance, index=model.feature_columns_).sort_values(ascending=False)
    print(ranked.head(20).round(2).to_string())
    print(f"\nzero-importance features: {int((ranked == 0).sum())}/{len(ranked)}")


if __name__ == "__main__":
    main()
