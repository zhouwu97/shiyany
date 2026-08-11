"""PRED-6A：Target Joint Structure —— rest = gall - g1 独立 delta 模型。

用 SAFE60 冻结 g1 预测 + 独立 rest delta 模型 → 重构 gall = g1_pred + rest_pred。
对比 SAFE60_gall。关键比较基线 = 隐式 rest（SAFE60_gall - SAFE60_g1）。

严格 forward：held fold 只用 origin<=train_end-120（8 标签成熟）的训练 origin。
rest delta 模型：预测 Δrest[t+15k] = rest[t+15k] - rest[t]（8 步），重建
rest_pred = rest[t] + Δpred，gall_pred = g1_safe60 + rest_pred。

用法：
  python scripts/run_pred6a_rest_joint.py --output <report.json> [--folds ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
STEP = 15
N_STEPS = 8
HORIZONS = tuple(STEP * k for k in range(1, N_STEPS + 1))


def _trajectory_deltas(series: pd.Series, origins: pd.DatetimeIndex) -> np.ndarray:
    """8 步绝对 delta：delta_k = s[t+15k] - s[t]。"""
    cols = []
    for k in range(1, N_STEPS + 1):
        future = series.shift(-k)
        cols.append((future - series).reindex(origins).to_numpy(dtype=float))
    return np.column_stack(cols)


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p) / np.maximum(np.abs(a), 1e-6)))


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

    frame = dataset.frame
    rest_series = pd.to_numeric(frame["generator_all"], errors="coerce") - pd.to_numeric(frame["generator_1"], errors="coerce")
    g1_series = pd.to_numeric(frame["generator_1"], errors="coerce")

    safe60 = pd.read_csv(SAFE60_OOF, parse_dates=["origin_time", "train_end"])
    fold_order = [f"dev_{i:02d}" for i in range(1, 20)]
    folds = args.folds.split(",") if args.folds else fold_order

    results = {"ridge": [], "et": []}
    for fold in folds:
        part = safe60[safe60["fold"] == fold]
        train_end = part["train_end"].unique()[0]
        held_origins = pd.DatetimeIndex(sorted(part["origin_time"].unique()))
        train_origins = pd.DatetimeIndex(
            frame.index[frame.index <= train_end - pd.Timedelta(minutes=120)])
        Xtr = X[frame.index.get_indexer(train_origins)]
        Ytr = _trajectory_deltas(rest_series, train_origins)
        valid = np.isfinite(Ytr).all(axis=1)
        Xtr, Ytr = Xtr[valid], Ytr[valid]
        imp = SimpleImputer(strategy="median").fit(Xtr)
        Xtr = imp.transform(Xtr)
        Xte = imp.transform(X[frame.index.get_indexer(held_origins)])
        rest_t = rest_series.reindex(held_origins).to_numpy(float)

        for kind in ("ridge", "et"):
            if kind == "ridge":
                m = MultiOutputRegressor(Ridge(alpha=20.0), n_jobs=2).fit(Xtr, Ytr)
            else:
                m = MultiOutputRegressor(
                    ExtraTreesRegressor(n_estimators=150, max_depth=8, min_samples_leaf=20,
                                        random_state=20250731, n_jobs=2), n_jobs=2).fit(Xtr, Ytr)
            rest_pred = rest_t[:, None] + m.predict(Xte)  # (n, 8)
            # SAFE60 的 g1 预测（按 origin×horizon 映射）
            g1_safe60 = part.loc[part["target"].eq("generator_1"), ["origin_time", "horizon", "safe60_pred"]]
            g1_map = dict(zip(zip(g1_safe60["origin_time"], g1_safe60["horizon"]), g1_safe60["safe60_pred"]))
            for k, h in enumerate(HORIZONS):
                gall_mask = part["target"].eq("generator_all") & part["horizon"].eq(h)
                g1_mask = part["target"].eq("generator_1") & part["horizon"].eq(h)
                gall_idx = held_origins.get_indexer(part.loc[gall_mask, "origin_time"])
                g1_vals = np.array([g1_map.get((o, h), np.nan) for o in part.loc[gall_mask, "origin_time"]])
                gall_pred = g1_vals + rest_pred[gall_idx, k]
                results[kind].append(pd.DataFrame({
                    "fold": fold, "target": "generator_all", "horizon": h,
                    "actual": part.loc[gall_mask, "actual"].to_numpy(float),
                    "anchor": part.loc[gall_mask, "safe60_pred"].to_numpy(float),
                    "pred": gall_pred,
                }))
                results[kind].append(pd.DataFrame({
                    "fold": fold, "target": "generator_1", "horizon": h,
                    "actual": part.loc[g1_mask, "actual"].to_numpy(float),
                    "anchor": part.loc[g1_mask, "safe60_pred"].to_numpy(float),
                    "pred": part.loc[g1_mask, "safe60_pred"].to_numpy(float),  # g1 不变
                }))

    summary = {}
    for kind in results:
        r = pd.concat(results[kind], ignore_index=True)
        gall = r[r.target == "generator_all"]
        g1 = r[r.target == "generator_1"]
        summary[kind] = {
            "pooled_mape": _mape(r.actual.to_numpy(float), r.pred.to_numpy(float)),
            "anchor_mape": _mape(r.actual.to_numpy(float), r.anchor.to_numpy(float)),
            "gall_mape": _mape(gall.actual.to_numpy(float), gall.pred.to_numpy(float)),
            "gall_anchor": _mape(gall.actual.to_numpy(float), gall.anchor.to_numpy(float)),
            "g1_mape": _mape(g1.actual.to_numpy(float), g1.pred.to_numpy(float)),
            "pooled_delta_pp": _mape(r.actual.to_numpy(float), r.anchor.to_numpy(float)) - _mape(r.actual.to_numpy(float), r.pred.to_numpy(float)),
        }
        print(f"{kind}: pooled={summary[kind]['pooled_mape']:.4f} (anchor={summary[kind]['anchor_mape']:.4f} delta={summary[kind]['pooled_delta_pp']:+.4f}pp) "
              f"gall={summary[kind]['gall_mape']:.4f} (anchor={summary[kind]['gall_anchor']:.4f}) g1={summary[kind]['g1_mape']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "folds": folds}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "folds": len(folds)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
