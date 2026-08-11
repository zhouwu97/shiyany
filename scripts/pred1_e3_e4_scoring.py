"""Gate E3/E4：SAFE60 最终 fit once + 评分期 192 origin 推理。

依赖序：aggressive → RichGas → A51 → splice → A60 → A61 → X3 → SAFE60。
- final cutoff = 2025-05-01 00:00（评分起点），seed = PRODUCTION_SEED_SLOT(100)。
- missing_current_policy = causal_forward_fill（3 个隐藏 origin，禁止 R1 重建）。
- blind 政策：RichGas 用 final OOF（rich_residual_final_gas，含 confirmed blind）；
  A51/A60 corrector 用 dev OOF（无 final OOF，注明）；X3/A61 fit 原始帧 ≤ cutoff
  （天然含盲区数据）。
- 输出评分期 SAFE60 长表（fold=scoring）+ 每层 receipt + SAFE60 = 0.6*X3+0.4*A61。

用法：
  python scripts/pred1_e3_e4_scoring.py --output <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import forecast_config_from_dict
from gas_forecast.data import align_tables, combine_context
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
from gas_forecast.workflow import predict_rolling

TRAIN_DIR = Path("data/raw/official/初赛-参赛者使用")
TEST_DIR = Path("煤气发电预测优化-初赛训练和测试集/初赛-评分所用测试集")
AGGRESSIVE_MODEL = Path("results/raw/runs/training/aggressive_r75_lgb20_20260802/model.joblib")
A60_CONFIG = Path("results/raw/runs/experiments/a60_generator_all_long_residual_verification_20260804/config.json")
X3_CONFIG = Path("results/raw/runs/audits/pred1_asset_audit_20260810/x3_config.json")
RICH_GAS_FINAL_OOF = Path("results/raw/runs/experiments/rich_residual_final_gas_20260803/oof.csv")
A51_DEV_OOF = Path("results/raw/runs/experiments/a51_g1_long_rich_residual_development_20260803/oof.csv")
A60_DEV_OOF = Path("results/raw/runs/experiments/a60_generator_all_long_residual_verification_20260804/oof.csv")

FINAL_CUTOFF = pd.Timestamp("2025-05-01 00:00:00")
SAFE60_X3_WEIGHT = 0.60
SAFE60_A61_WEIGHT = 0.40


def _wide_to_long(pred: pd.DataFrame, *, fold: str) -> pd.DataFrame:
    """把 16 列宽表转成长表（origin/target/horizon）。"""
    rows = []
    for col in pred.columns:
        if not col.endswith("_pred"):
            continue
        target = col.split("_t+")[0]
        horizon = int(col.split("_t+")[1].split("_")[0])
        s = pred[col].to_frame().reset_index()
        s.columns = ["origin_time", "value"]
        s["target"] = target
        s["horizon"] = horizon
        s["fold"] = fold
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = forecast_config_from_dict(json.loads(A60_CONFIG.read_text(encoding="utf-8")))
    x3_config = forecast_config_from_dict(json.loads(X3_CONFIG.read_text(encoding="utf-8")))
    train = align_tables(TRAIN_DIR, config.feature.frequency).frame
    test = align_tables(TEST_DIR, config.feature.frequency).frame
    ctx = combine_context(train, test)
    ctx[["generator_1", "generator_all"]] = ctx[["generator_1", "generator_all"]].ffill()
    price_paths = sorted(TRAIN_DIR.glob("*price*.xlsx"))
    if len(price_paths) != 1:
        raise ValueError(f"E3/E4 需要唯一 price.xlsx: {price_paths}")
    price = load_price_schedule(price_paths[0])

    x3_eff = rich_feature_config(x3_config, RICH_FEATURE_GROUPS, feature_profile="long_horizon")
    x3_features = build_causal_features(ctx, x3_eff.feature, price)

    # 1. aggressive at scoring origins（部署模型）
    _, agg_pred = predict_rolling(TRAIN_DIR, TEST_DIR, AGGRESSIVE_MODEL)
    agg_long = _wide_to_long(agg_pred, fold="scoring")
    agg_long = agg_long.rename(columns={"value": "aggressive_r75_lgb20_pred"})
    scoring_origins = pd.DatetimeIndex(sorted(agg_long["origin_time"].unique()))
    if len(scoring_origins) != 192:
        raise ValueError(f"评分 origin 数量异常: {len(scoring_origins)}")

    # 2. RichGas（final OOF 含 confirmed blind）
    rg = build_rich_gas_production_predictions(
        ctx, pd.read_csv(RICH_GAS_FINAL_OOF, parse_dates=["origin_time"]),
        config=config, cutoff=FINAL_CUTOFF, scoring_rows=agg_long, price_schedule=price, fold_label="scoring")
    # 3. A51（dev OOF，baseline=aggressive）
    a51 = build_a51_production_predictions(
        ctx, pd.read_csv(A51_DEV_OOF, parse_dates=["origin_time"]),
        config=config, cutoff=FINAL_CUTOFF, scoring_rows=agg_long, price_schedule=price, fold_label="scoring")
    # 4. splice
    a51.rows[RICH_GAS_OUTPUT_COLUMN] = rg.rows[RICH_GAS_OUTPUT_COLUMN].to_numpy(dtype=float)
    spliced = apply_short_long_splice(
        a51.rows, baseline_column=RICH_GAS_OUTPUT_COLUMN, branch_column=A51_OUTPUT_COLUMN)
    # 5. A60
    a60 = build_a60_production_predictions(
        ctx, pd.read_csv(A60_DEV_OOF, parse_dates=["origin_time"]),
        config=config, cutoff=FINAL_CUTOFF, scoring_rows=spliced, price_schedule=price, fold_label="scoring")
    # 6. A61（one-step 标签 origin+15min 须 < first_held；train_end = first_held - 30min）
    a61_cutoff = FINAL_CUTOFF - pd.Timedelta(minutes=30)
    a61_rows, _ = build_a61_production_predictions(
        ctx, x3_features, cutoff=a61_cutoff, parent=a60.rows, fold_label="scoring")
    # 7. X3（production seed slot 100）
    pos = resolve_seed_position("production")
    x3_long, _ = build_x3_production_predictions(
        ctx, x3_features, cutoff=FINAL_CUTOFF, origins=scoring_origins, seed_position=pos, fold_label="scoring")
    # 8. SAFE60
    merged = a61_rows.merge(
        x3_long[["origin_time", "target", "horizon", "x3_cat_mae_pred"]],
        on=["origin_time", "target", "horizon"], how="left")
    merged["safe60_pred"] = SAFE60_X3_WEIGHT * merged["x3_cat_mae_pred"] + SAFE60_A61_WEIGHT * merged["a61_recursive_blend_05_pred"]
    merged["fold"] = "scoring"

    if not np.isfinite(merged["safe60_pred"]).all():
        raise ValueError("SAFE60 评分预测含非有限值")
    if len(merged) != 3072 or merged["origin_time"].nunique() != 192:
        raise ValueError(f"SAFE60 评分长表结构异常: {len(merged)} rows")

    receipt = {
        "final_cutoff": str(FINAL_CUTOFF),
        "a61_final_cutoff": str(a61_cutoff),  # one-step label 成熟边界
        "seed_position": int(pos),
        "seed_contract": "production",
        "missing_current_policy": {
            "policy": "causal_forward_fill",
            "affected_origins": 3,
            "uses_future_data": False,
            "uses_reference_reconstruction": False,
        },
        "blind_policy": {
            "confirmed_blind_labels_used_for_selection": False,
            "confirmed_blind_oof_used_for_refit": True,
            "post_blind_tuning_allowed": False,
            "rich_gas_used_final_oof": True,
            "a51_a60_used_dev_oof_no_final_exists": True,
        },
        "scoring_origins": int(len(scoring_origins)),
        "cells": int(len(merged)),
        "layers": {
            "rich_gas": rg.receipts["fit"],
            "a51": a51.receipts["fit"],
            "a60": a60.receipts["fit"],
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "safe60_scoring_oof.csv", index=False, encoding="utf-8")
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    x3_long.to_csv(args.output / "x3_scoring.csv", index=False, encoding="utf-8")
    print(json.dumps({"cells": len(merged), "origins": merged["origin_time"].nunique(),
                      "finite": bool(np.isfinite(merged["safe60_pred"]).all()),
                      "g1_mean": float(merged[merged.target=="generator_1"]["safe60_pred"].mean()),
                      "gall_mean": float(merged[merged.target=="generator_all"]["safe60_pred"].mean())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
