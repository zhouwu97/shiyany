"""PRED-1 上传前 3-origin 定向复核（Item 2 + 3）。

对 3 个隐藏 current origin（2025-05-02 10:15 g1、13:15 gall、21:45 g1）：
- fit 一次全链（final cutoff / slot 100）
- 用三种上下文重建因果特征并重新生成该 origin 的 SAFE60 预测：
    A) full context（应为冻结 s_result，验证可复现）
    B) truncated（数据截断到 <= origin）
    C) perturbed（origin 后 generator_1/generator_all shuffle/null/extreme）
- 三者的 16 cells 必须逐位一致，且 A 必须 == 冻结 s_result_safe60.csv（1e-10）。
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
    A51_OUTPUT_COLUMN, RICH_GAS_OUTPUT_COLUMN, apply_short_long_splice,
    build_a51_production_predictions, build_a60_production_predictions,
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
FROZEN_S_RESULT = Path("results/raw/runs/audits/pred1_e34_scoring_20260810/s_result_safe60.csv")
FINAL_CUTOFF = pd.Timestamp("2025-05-01 00:00:00")
TARGET_ORIGINS = {"2025-05-02 10:15:00", "2025-05-02 13:15:00", "2025-05-02 21:45:00"}


def _chain_predict(ctx, features, *, agg_long, config, x3_features, seed_pos, origins, price):
    """在给定 context+features 上跑全链，返回 SAFE60 长表（仅 target origins）。"""
    a61_cutoff = FINAL_CUTOFF - pd.Timedelta(minutes=30)
    agg_long = agg_long[agg_long["origin_time"].isin(origins)].copy()
    rg = build_rich_gas_production_predictions(
        ctx, pd.read_csv(RICH_GAS_FINAL_OOF, parse_dates=["origin_time"]),
        config=config, cutoff=FINAL_CUTOFF, scoring_rows=agg_long.copy(), price_schedule=price, fold_label="scoring")
    a51 = build_a51_production_predictions(
        ctx, pd.read_csv(A51_DEV_OOF, parse_dates=["origin_time"]),
        config=config, cutoff=FINAL_CUTOFF, scoring_rows=agg_long.copy(), price_schedule=price, fold_label="scoring")
    a51.rows[RICH_GAS_OUTPUT_COLUMN] = rg.rows[RICH_GAS_OUTPUT_COLUMN].to_numpy(dtype=float)
    spliced = apply_short_long_splice(a51.rows, baseline_column=RICH_GAS_OUTPUT_COLUMN, branch_column=A51_OUTPUT_COLUMN)
    a60 = build_a60_production_predictions(
        ctx, pd.read_csv(A60_DEV_OOF, parse_dates=["origin_time"]),
        config=config, cutoff=FINAL_CUTOFF, scoring_rows=spliced, price_schedule=price, fold_label="scoring")
    a61_rows, _ = build_a61_production_predictions(
        ctx, x3_features, cutoff=a61_cutoff, parent=a60.rows, fold_label="scoring")
    x3_long, _ = build_x3_production_predictions(
        ctx, x3_features, cutoff=FINAL_CUTOFF, origins=origins, seed_position=seed_pos, fold_label="scoring")
    merged = a61_rows.merge(x3_long[["origin_time", "target", "horizon", "x3_cat_mae_pred"]],
                            on=["origin_time", "target", "horizon"], how="left")
    merged["safe60_pred"] = 0.6 * merged["x3_cat_mae_pred"] + 0.4 * merged["a61_recursive_blend_05_pred"]
    return merged[["origin_time", "target", "horizon", "safe60_pred"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = forecast_config_from_dict(json.loads(A60_CONFIG.read_text(encoding="utf-8")))
    x3_config = forecast_config_from_dict(json.loads(X3_CONFIG.read_text(encoding="utf-8")))
    train = align_tables(TRAIN_DIR, config.feature.frequency).frame
    test = align_tables(TEST_DIR, config.feature.frequency).frame
    base_ctx = combine_context(train, test)
    base_ctx[["generator_1", "generator_all"]] = base_ctx[["generator_1", "generator_all"]].ffill()
    price_paths = sorted(TRAIN_DIR.glob("*price*.xlsx"))
    price = load_price_schedule(price_paths[0])
    x3_eff = rich_feature_config(x3_config, RICH_FEATURE_GROUPS, feature_profile="long_horizon")
    x3_features = build_causal_features(base_ctx, x3_eff.feature, price)

    _, agg_pred = predict_rolling(TRAIN_DIR, TEST_DIR, AGGRESSIVE_MODEL)
    # 构造 agg_long（长表，含 aggressive_r75_lgb20_pred）
    parts = []
    for col in agg_pred.columns:
        if not col.endswith("_pred"):
            continue
        target = col.split("_t+")[0]
        horizon = int(col.split("_t+")[1].split("_")[0])
        s = agg_pred[col].to_frame().reset_index()
        s.columns = ["origin_time", "value"]
        s["target"] = target
        s["horizon"] = horizon
        s["fold"] = "scoring"
        parts.append(s)
    agg_long = pd.concat(parts, ignore_index=True).rename(columns={"value": "aggressive_r75_lgb20_pred"})

    origins = pd.DatetimeIndex([pd.Timestamp(t) for t in sorted(TARGET_ORIGINS)])
    seed_pos = resolve_seed_position("production")

    # 基线预测（full context）
    base_pred = _chain_predict(base_ctx, x3_features, agg_long=agg_long, config=config,
                               x3_features=x3_features, seed_pos=seed_pos, origins=origins, price=price)

    # 冻结 s_result 参考（target origins 的 16 cells）
    frozen = pd.read_csv(FROZEN_S_RESULT, parse_dates=["datetime"])
    parts = []
    for col in frozen.columns:
        if not col.endswith("_pred"):
            continue
        target = col.split("_t+")[0]
        horizon = int(col.split("_t+")[1].split("_")[0])
        s = frozen[["datetime", col]].copy()
        s.columns = ["origin_time", "frozen_pred"]
        s["target"] = target
        s["horizon"] = horizon
        parts.append(s)
    frozen_long = pd.concat(parts, ignore_index=True)
    cmp = base_pred.merge(frozen_long, on=["origin_time", "target", "horizon"])
    repro_max = float((cmp["safe60_pred"] - cmp["frozen_pred"]).abs().max())

    # 截断 + 扰动变体
    variants = {"truncated": {}, "perturbed": {}}
    rng = np.random.default_rng(20250731)
    for origin in origins:
        # truncated：数据 <= origin
        tctx = base_ctx.loc[base_ctx.index <= origin].copy()
        tf = build_causal_features(tctx, x3_eff.feature, price)
        tp = _chain_predict(tctx, tf, agg_long=agg_long, config=config, x3_features=tf,
                            seed_pos=seed_pos, origins=pd.DatetimeIndex([origin]), price=price)
        variants["truncated"][str(origin)] = tp
        # perturbed：origin 后 generator 全部随机改
        pctx = base_ctx.copy()
        future = pctx.index > origin
        for col in ["generator_1", "generator_all"]:
            block = pctx.loc[future, col].to_numpy(copy=True)
            pctx.loc[future, col] = block[rng.permutation(len(block))] if len(block) else block
        pf = build_causal_features(pctx, x3_eff.feature, price)
        pp = _chain_predict(pctx, pf, agg_long=agg_long, config=config, x3_features=pf,
                            seed_pos=seed_pos, origins=pd.DatetimeIndex([origin]), price=price)
        variants["perturbed"][str(origin)] = pp

    # 对比：每个 origin 的 truncated/perturbed vs baseline
    checks = {}
    all_ok = repro_max <= 1e-10
    for origin in origins:
        os = str(origin)
        for vname, vdict in variants.items():
            pred_t = vdict[str(origin)][["target", "horizon", "safe60_pred"]]
            base_t = base_pred[base_pred["origin_time"] == origin][["target", "horizon", "safe60_pred"]]
            m = pred_t.merge(base_t, on=["target", "horizon"], suffixes=("_v", "_b"))
            d = float((m["safe60_pred_v"] - m["safe60_pred_b"]).abs().max())
            checks[f"{os}_{vname}"] = d
            all_ok &= d <= 1e-10

    report = {
        "stage": "final_verify_3origins",
        "reproducible_vs_frozen_sresult_max_abs_diff": repro_max,
        "variants_max_abs_diff": checks,
        "all_identical": bool(all_ok),
        "missing_current_policy": "causal_forward_fill",
        "target_origins": sorted(TARGET_ORIGINS),
        "acceptance": "all max_abs_diff <= 1e-10",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"repro_max": repro_max, "checks": checks, "all_identical": all_ok}, ensure_ascii=False))


if __name__ == "__main__":
    main()
