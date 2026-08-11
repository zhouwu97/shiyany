"""Gate E2：SAFE60 全链端到端 production replay。

在历史 cutoff（development fold）上按依赖序跑整条生产链：
  aggressive → RichGas → A51 → splice → A60 → A61 → X3 → SAFE60
并与冻结 OOF 的 SAFE60（0.6*X3 + 0.4*A61）逐 cell 比较。

接受标准：max_abs_pred_diff <= 1e-6（各层已逐位复现，端到端应精确）。
另输出 pooled MAPE diff / correlation / median/p99/max 预测差（E2 契约）。

用法：
  python scripts/pred1_e2_full_chain_replay.py --output <report.json> [--folds dev_19]
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
from gas_forecast.production_runner import (
    A51_OUTPUT_COLUMN,
    RICH_GAS_OUTPUT_COLUMN,
    apply_short_long_splice,
    build_a51_production_predictions,
    build_a60_production_predictions,
    build_rich_gas_production_predictions,
)
from gas_forecast.recursive_arx import build_a61_production_predictions
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config
from gas_forecast.seed_contract import resolve_seed_position

CONFIG_PATH = Path("results/raw/runs/experiments/a60_generator_all_long_residual_verification_20260804/config.json")
X3_CONFIG_PATH = Path("results/raw/runs/audits/pred1_asset_audit_20260810/x3_config.json")
DATA_DIR = Path("data/raw/official/初赛-参赛者使用")
RICH_GAS_OOF = Path("results/raw/runs/experiments/a51_g1_long_rich_residual_development_20260803/oof.csv")
A51_OOF = Path("results/raw/runs/experiments/a51_g1_long_rich_residual_development_20260803/oof.csv")
A60_OOF = Path("results/raw/runs/experiments/a60_generator_all_long_residual_verification_20260804/oof.csv")
A61_OOF = Path("results/raw/runs/experiments/pred1_a61_replay_20260810/oof.csv")
X3_OOF = Path("results/raw/runs/experiments/pred1_x3_replay_20260810/oof.csv")


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p) / np.maximum(np.abs(a), 1e-6)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="dev_19")
    args = parser.parse_args()

    config = forecast_config_from_dict(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    x3_config = forecast_config_from_dict(json.loads(X3_CONFIG_PATH.read_text(encoding="utf-8")))
    dataset = align_tables(DATA_DIR, config.feature.frequency)
    price_paths = sorted(DATA_DIR.glob("*price*.xlsx"))
    if len(price_paths) != 1:
        raise ValueError(f"E2 需要唯一 price.xlsx: {price_paths}")
    price = load_price_schedule(price_paths[0])

    # X3 用全组 long_horizon 特征（与 mape_aligned 一致）
    x3_eff = rich_feature_config(x3_config, RICH_FEATURE_GROUPS, feature_profile="long_horizon")
    x3_features = build_causal_features(dataset.frame, x3_eff.feature, price)

    # 冻结 OOF 参考（重放链产物）
    a61_oof = pd.read_csv(A61_OOF, parse_dates=["origin_time"])
    x3_oof = pd.read_csv(X3_OOF, parse_dates=["origin_time"])

    rows_out = []
    for fold in [f.strip() for f in args.folds.split(",")]:
        cutoff_rows = pd.read_csv(A60_OOF, parse_dates=["origin_time"])
        part = cutoff_rows[cutoff_rows["fold"] == fold].copy()
        cutoff = part["train_end"].unique()[0]
        parent = part[["fold", "origin_time", "target", "horizon", "aggressive_r75_lgb20_pred"]].copy()

        # 1. RichGas
        rg = build_rich_gas_production_predictions(
            dataset.frame, pd.read_csv(RICH_GAS_OOF, parse_dates=["origin_time"]),
            config=config, cutoff=cutoff, scoring_rows=parent, price_schedule=price, fold_label=fold)
        # 2. A51（baseline=aggressive）
        a51 = build_a51_production_predictions(
            dataset.frame, pd.read_csv(A51_OOF, parse_dates=["origin_time"]),
            config=config, cutoff=cutoff, scoring_rows=parent, price_schedule=price, fold_label=fold)
        # 3. splice（baseline=rich_gas_blend_30, branch=rich_g1_long_blend_30）
        a51.rows[RICH_GAS_OUTPUT_COLUMN] = rg.rows[RICH_GAS_OUTPUT_COLUMN].to_numpy(dtype=float)
        spliced = apply_short_long_splice(
            a51.rows, baseline_column=RICH_GAS_OUTPUT_COLUMN, branch_column=A51_OUTPUT_COLUMN)
        spliced = spliced.merge(
            part[["origin_time", "target", "horizon", "actual"]], on=["origin_time", "target", "horizon"], how="left")
        # 4. A60（parent=rich_short00_long100）
        a60 = build_a60_production_predictions(
            dataset.frame, pd.read_csv(A60_OOF, parse_dates=["origin_time"]),
            config=config, cutoff=cutoff, scoring_rows=spliced, price_schedule=price, fold_label=fold)
        # 5. A61（parent=a60_gall_long_blend_30）返回 (rows, receipt)
        a61_rows, _ = build_a61_production_predictions(
            dataset.frame, x3_features, cutoff=cutoff, parent=a60.rows, fold_label=fold)
        # 6. X3
        origins = pd.DatetimeIndex(sorted(part["origin_time"].unique()))
        pos = resolve_seed_position("replay", cutoff=cutoff)
        x3_long, _ = build_x3_production_predictions(
            dataset.frame, x3_features, cutoff=cutoff, origins=origins, seed_position=pos, fold_label=fold)
        # 7. SAFE60
        merged = a61_rows.merge(
            x3_long[["origin_time", "target", "horizon", "x3_cat_mae_pred"]],
            on=["origin_time", "target", "horizon"], how="left")
        merged["safe60_pred"] = 0.6 * merged["x3_cat_mae_pred"] + 0.4 * merged["a61_recursive_blend_05_pred"]
        # 参考 SAFE60：冻结 X3/A61 OOF
        ref = a61_oof[a61_oof["fold"] == fold][["origin_time", "target", "horizon", "actual", "a61_recursive_blend_05_pred"]]
        ref = ref.merge(
            x3_oof[x3_oof["fold"] == fold][["origin_time", "target", "horizon", "x3_cat_mae_pred"]],
            on=["origin_time", "target", "horizon"], how="left")
        ref["safe60_ref"] = 0.6 * ref["x3_cat_mae_pred"] + 0.4 * ref["a61_recursive_blend_05_pred"]
        cmp = merged.merge(ref, on=["origin_time", "target", "horizon"], suffixes=("", "_ref"))
        diff = (cmp["safe60_pred"] - cmp["safe60_ref"]).abs()
        corr = np.corrcoef(cmp["safe60_pred"], cmp["safe60_ref"])[0, 1]
        row = {
            "fold": fold, "cutoff": str(cutoff), "cells": int(len(cmp)),
            "max_abs_pred_diff": float(diff.max()), "median_abs_pred_diff": float(diff.median()),
            "p99_abs_pred_diff": float(diff.quantile(0.99)), "correlation": float(corr),
            "pooled_mape_production": _mape(cmp["actual"].to_numpy(float), cmp["safe60_pred"].to_numpy(float)),
            "pooled_mape_ref": _mape(cmp["actual"].to_numpy(float), cmp["safe60_ref"].to_numpy(float)),
            "pass": bool(diff.max() <= 1e-6),
        }
        rows_out.append(row)
        print(f"{fold}: max|diff|={diff.max():.3e} corr={corr:.10f} pooled={row['pooled_mape_production']:.6f} vs ref={row['pooled_mape_ref']:.6f} {'PASS' if row['pass'] else 'FAIL'}")

    report = {"stage": "E2_full_chain", "seed_contract": "replay", "acceptance": "max_abs_pred_diff<=1e-6",
              "all_pass": bool(all(r["pass"] for r in rows_out)), "folds": rows_out}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"], "n_folds": len(rows_out)}))


if __name__ == "__main__":
    main()
