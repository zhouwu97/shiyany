"""Gate E1 layer replay：X3 production runner 对全部 19 个历史 cutoff 复现冻结 OOF。

对每个 development fold k：
  build_x3_production_predictions(cutoff=train_end_k, seed_position=k)
  → 预测该折 held origins
  → 与冻结 X3 replay OOF 的 x3_cat_mae_pred 逐 cell 比较

接受标准：max|pred_production - pred_oof| <= 1e-6（浮点机器精度），
pooled MAPE diff == 0（实际上应逐位一致，因 B1 已证明 OOF builder 逐字节复现）。

用法：
  python scripts/pred1_e1_x3_layer_replay.py --output <report.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.mape_aligned import build_x3_production_predictions
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config
from gas_forecast.seed_contract import resolve_seed_position

CONFIG_PATH = Path("results/raw/runs/audits/pred1_asset_audit_20260810/x3_config.json")
OOF_PATH = Path("results/raw/runs/experiments/pred1_x3_replay_20260810/oof.csv")
DATA_DIR = Path("data/raw/official/初赛-参赛者使用")


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred) / np.maximum(np.abs(actual), 1e-6)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="", help="逗号分隔折子集（默认全部 19）")
    args = parser.parse_args()

    config = forecast_config_from_dict(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    dataset = align_tables(DATA_DIR, config.feature.frequency)
    effective = rich_feature_config(config, RICH_FEATURE_GROUPS, feature_profile="long_horizon")
    price_paths = sorted(DATA_DIR.glob("*price*.xlsx"))
    if len(price_paths) != 1:
        raise ValueError(f"X3 E1 需要唯一 price.xlsx: {price_paths}")
    features = build_causal_features(dataset.frame, effective.feature, load_price_schedule(price_paths[0]))

    oof = pd.read_csv(OOF_PATH, parse_dates=["origin_time", "train_end"])
    folds = args.folds.split(",") if args.folds else [f"dev_{i:02d}" for i in range(1, 20)]

    rows = []
    all_max = 0.0
    for fold in folds:
        part = oof[oof["fold"] == fold]
        cutoff = part["train_end"].unique()[0]
        origins = pd.DatetimeIndex(sorted(part["origin_time"].unique()))
        position = resolve_seed_position("replay", cutoff=cutoff)
        long, receipts = build_x3_production_predictions(
            dataset.frame,
            features,
            cutoff=cutoff,
            origins=origins,
            seed_position=position,
            fold_label=fold,
        )
        merged = long.merge(
            part[["origin_time", "target", "horizon", "actual", "x3_cat_mae_pred"]].rename(
                columns={"x3_cat_mae_pred": "oof_pred"}
            ),
            on=["origin_time", "target", "horizon"],
            how="left",
        )
        diff = (merged["x3_cat_mae_pred"] - merged["oof_pred"]).abs()
        pooled = _mape(merged["actual"].to_numpy(dtype=float), merged["x3_cat_mae_pred"].to_numpy(dtype=float))
        oof_pooled = _mape(merged["actual"].to_numpy(dtype=float), merged["oof_pred"].to_numpy(dtype=float))
        maxd = float(diff.max())
        all_max = max(all_max, maxd)
        rows.append(
            {
                "fold": fold,
                "cutoff": str(cutoff),
                "seed_position": int(position),
                "origins": int(merged["origin_time"].nunique()),
                "cells": int(len(merged)),
                "max_abs_pred_diff": maxd,
                "mean_abs_pred_diff": float(diff.mean()),
                "pooled_mape_production": pooled,
                "pooled_mape_oof": oof_pooled,
                "pass": maxd <= 1e-6,
            }
        )
        print(f"{fold}: max|diff|={maxd:.3e} pooled={pooled:.6f} vs oof={oof_pooled:.6f} {'PASS' if maxd<=1e-6 else 'FAIL'}")

    report = {
        "layer": "X3",
        "layer_index": 5,
        "seed_contract": "replay: cutoff->frozen fold position",
        "acceptance": "max_abs_pred_diff <= 1e-6",
        "all_folds_max_abs_pred_diff": all_max,
        "pass": all_max <= 1e-6,
        "folds": rows,
        "n_folds": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_max": all_max, "pass": report["pass"]}))


if __name__ == "__main__":
    main()
