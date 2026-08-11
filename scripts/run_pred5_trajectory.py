"""PRED-5：PCA / MultiOutput Trajectory（第二增益源）。

假设：未来 8 步绝对 delta 轨迹共享低维结构，联合预测 delta_15..delta_120
比独立逐步预测更稳，且产生与 SAFE60 低相关的残差结构。

严格 forward：held fold 只用 origin <= train_end - 120（8 标签全成熟）的
训练 origin 拟合，预测 held origins。

模型（≤6 主实验，禁止扩大网格）：
  5A: MultiOutput Ridge(alpha=20) / MultiOutput ExtraTrees（固定配置）
  5B: PCA(components 2/3/4) + Ridge / CatBoost MAE 预测系数

评估：standalone MAPE vs SAFE60 anchor、残差相关、5%/10%/15% blend。

Retain rule：standalone <= anchor + 0.15pp 且残差相关显著更低，或 5-15% blend
稳定改善 → 保留为 specialist。不要求单体冠军。

用法：
  python scripts/run_pred5_trajectory.py --output <report.json> [--folds dev_01,dev_19]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from gas_forecast.config import forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config

TRAIN_DIR = Path("data/raw/official/初赛-参赛者使用")
X3_CONFIG = Path("results/raw/runs/audits/pred1_asset_audit_20260810/x3_config.json")
SAFE60_OOF = Path("results/raw/runs/audits/pred1_gate_c_20260810/merged_safe60_eval.csv")
TARGETS = ("generator_1", "generator_all")
STEP = 15
N_STEPS = 8
HORIZONS = tuple(STEP * k for k in range(1, N_STEPS + 1))


def _trajectory_deltas(frame: pd.DataFrame, target: str, origins: pd.DatetimeIndex) -> np.ndarray:
    """返回每个 origin 的 8 步绝对 delta 矩阵 (n, 8)。delta_k = y[t+15k] - y[t]。"""
    series = pd.to_numeric(frame[target], errors="coerce")
    cols = []
    for k in range(1, N_STEPS + 1):
        future = series.shift(-k)
        cols.append((future - series).reindex(origins).to_numpy(dtype=float))
    return np.column_stack(cols)


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p) / np.maximum(np.abs(a), 1e-6)))


def _fit_predict(Xtr, Ytr, Xte, kind: str, seed: int = 20250731):
    if kind == "ridge":
        m = MultiOutputRegressor(Ridge(alpha=20.0), n_jobs=2)
        m.fit(Xtr, Ytr)
        return m.predict(Xte)
    if kind == "et":
        m = MultiOutputRegressor(
            ExtraTreesRegressor(n_estimators=150, max_depth=8, min_samples_leaf=20,
                                random_state=seed, n_jobs=2),
            n_jobs=2)
        m.fit(Xtr, Ytr)
        return m.predict(Xte)
    if kind.startswith("pca"):
        comps = int(kind.split("_")[1])
        pca = PCA(n_components=comps, random_state=seed).fit(Ytr)
        Coef = pca.transform(Ytr)
        if "cat" in kind:
            models = [CatBoostRegressor(iterations=100, depth=6, learning_rate=0.05,
                                        random_seed=seed + i, verbose=False)
                      for i in range(comps)]
            coef_te = np.column_stack([m.fit(Xtr, Coef[:, i]).predict(Xte) for i, m in enumerate(models)])
        else:
            models = [Ridge(alpha=20.0).fit(Xtr, Coef[:, i]) for i in range(comps)]
            coef_te = np.column_stack([m.predict(Xte) for m in models])
        return pca.inverse_transform(coef_te)
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="")
    args = parser.parse_args()

    cfg = forecast_config_from_dict(json.loads(X3_CONFIG.read_text(encoding="utf-8")))
    dataset = align_tables(TRAIN_DIR, cfg.feature.frequency)
    price = load_price_schedule(sorted(TRAIN_DIR.glob("*price*.xlsx"))[0])
    eff = rich_feature_config(cfg, RICH_FEATURE_GROUPS, feature_profile="long_horizon")
    features = build_causal_features(dataset.frame, eff.feature, price)
    feature_cols = [c for c in features.columns if not c.startswith("feat_target_price")]
    X = features[feature_cols].to_numpy(dtype=float)

    safe60 = pd.read_csv(SAFE60_OOF, parse_dates=["origin_time", "train_end"])
    fold_order = [f"dev_{i:02d}" for i in range(1, 20)]
    folds = args.folds.split(",") if args.folds else fold_order

    kinds = ["ridge", "et", "pca_2_ridge", "pca_3_ridge", "pca_4_ridge", "pca_3_cat"]
    results = {k: [] for k in kinds}

    for fold in folds:
        part = safe60[safe60["fold"] == fold]
        train_end = part["train_end"].unique()[0]
        held_origins = pd.DatetimeIndex(sorted(part["origin_time"].unique()))
        # 训练 origin：8 标签全成熟；NaN 特征由 median imputer 处理
        train_origins = dataset.frame.index[
            (dataset.frame.index <= train_end - pd.Timedelta(minutes=120))
        ]
        train_origins = pd.DatetimeIndex(train_origins)

        for target in TARGETS:
            Ytr = _trajectory_deltas(dataset.frame, target, train_origins)
            valid = np.isfinite(Ytr).all(axis=1)
            if valid.sum() < 200:
                continue
            Xtr = X[dataset.frame.index.get_indexer(train_origins[valid])]
            Ytr_v = Ytr[valid]
            imputer = SimpleImputer(strategy="median").fit(Xtr)
            Xtr = imputer.transform(Xtr)
            Xte = imputer.transform(X[dataset.frame.index.get_indexer(held_origins)])
            cur = pd.to_numeric(dataset.frame[target], errors="coerce").reindex(held_origins).to_numpy(float)
            for kind in kinds:
                pred_delta = _fit_predict(Xtr, Ytr_v, Xte, kind)
                pred = cur[:, None] + pred_delta  # (n_origins, 8)
                for k, h in enumerate(HORIZONS):
                    mask = part["target"].eq(target) & part["horizon"].eq(h)
                    idx = held_origins.get_indexer(part.loc[mask, "origin_time"])
                    results[kind].append(
                        pd.DataFrame({
                            "fold": fold, "target": target, "horizon": h,
                            "actual": part.loc[mask, "actual"].to_numpy(float),
                            "anchor": part.loc[mask, "safe60_pred"].to_numpy(float),
                            "pred": pred[idx, k],
                        })
                    )

    # 汇总
    summary = {}
    for kind in kinds:
        r = pd.concat(results[kind], ignore_index=True)
        r["resid_anchor"] = r["actual"] - r["anchor"]
        r["resid_pred"] = r["actual"] - r["pred"]
        corr = float(np.corrcoef(r["resid_anchor"], r["resid_pred"])[0, 1]) if len(r) else None
        summary[kind] = {
            "standalone_mape": _mape(r["actual"].to_numpy(float), r["pred"].to_numpy(float)),
            "anchor_mape": _mape(r["actual"].to_numpy(float), r["anchor"].to_numpy(float)),
            "residual_corr_with_anchor": corr,
            "cells": len(r),
        }
        for w in (0.05, 0.10, 0.15):
            blend = (1 - w) * r["anchor"] + w * r["pred"]
            summary[kind][f"blend{w:.2f}_mape"] = _mape(r["actual"].to_numpy(float), blend.to_numpy(float))
        print(f"{kind}: standalone={summary[kind]['standalone_mape']:.4f} anchor={summary[kind]['anchor_mape']:.4f} "
              f"corr={corr:.4f} blend05={summary[kind]['blend0.05_mape']:.4f} blend10={summary[kind]['blend0.10_mape']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"anchor_mape": summary["ridge"]["anchor_mape"],
                                       "kinds": summary, "folds": folds}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "folds": len(folds)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
