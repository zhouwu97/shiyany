"""X0 — Archived OOF Diversity Audit.

零训练成本。把所有已存档的 strict OOF 预测列拉到同一批 (fold, origin, horizon) 行上，
算:
  - standalone MAPE (dev folds)
  - 残差与 champion 的相关性
  - in-sample 最优凸 blend (上界诊断)
  - LOO-fold 诚实 blend (权重在其余折拟合、应用到当前折; 真实因果估计)

防泄漏:
  - champion 列 = v2_v3_target_reconciled_pred (已验证与 research 的 c0_champion_pred bit-identical)
  - blend weight 永不在与评估相同的行上拟合 (LOO 按 fold)
  - 只用 dev folds 做结论; blind 仅观察
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent

# (描述, oof 路径, [预测列])  — 只登记确实存在的列
SOURCES: list[dict] = [
    {"name": "champion", "path": "results/raw/runs/oof/clean_c0_strict_20260801_v2/oof.csv", "cols": ["v2_v3_target_reconciled_pred"]},
    {"name": "v1", "path": "results/raw/runs/oof/clean_c0_strict_20260801_v2/oof.csv", "cols": ["v1_pred"]},
    {"name": "v2", "path": "results/raw/runs/oof/clean_c0_strict_20260801_v2/oof.csv", "cols": ["v2_pred"]},
    {"name": "v25", "path": "results/raw/runs/oof/clean_c0_strict_20260801_v2/oof.csv", "cols": ["v25_pred"]},
    {"name": "v3", "path": "results/raw/runs/oof/clean_c0_strict_20260801_v2/oof.csv", "cols": ["v3_pred"]},
    {"name": "e13_alpha", "path": "results/raw/runs/experiments/e13_alpha_screening_20260801_1609/oof.csv", "cols": ["e13_short_alpha_5_pred", "e13_short_alpha_10_pred", "e13_short_alpha_20_pred", "e13_short_alpha_40_pred", "e13_short_alpha_80_pred"]},
    {"name": "e20_e21_recency", "path": "results/raw/runs/experiments/e20_e21_recency_screening_20260801_1631/oof.csv", "cols": ["e20_hard_30d_pred", "e20_hard_60d_pred", "e20_hard_90d_pred", "e21_exp_half_life_30d_pred", "e21_exp_half_life_60d_pred", "e21_exp_half_life_90d_pred"]},
    {"name": "e22_e25", "path": "results/raw/runs/experiments/e22_e25_screening_20260801/oof.csv", "cols": ["e22_window_4_damping_0.7_pred", "e22_window_4_damping_0.85_pred", "e22_window_8_damping_0.7_pred", "e22_window_8_damping_0.85_pred", "e25_analog_k10_pred", "e25_analog_k20_pred", "e25_analog_k40_pred", "e25_analog_k80_pred"]},
    {"name": "e25_dev", "path": "results/raw/runs/experiments/e25_development_20260801/oof.csv", "cols": ["e25_analog_k40_pred", "e25_analog_k80_pred"]},
    {"name": "e50_e51", "path": "results/raw/runs/experiments/e50_e51_screening_20260801/oof.csv", "cols": ["e50_inverse_absolute_pred", "e50_inverse_squared_pred", "e51_weighted_lad_pred"]},
    {"name": "e91_e92", "path": "results/raw/runs/experiments/e91_e92_screening_20260801/oof.csv", "cols": ["e91_online_gain_true_hot_hl4_vw0.25_pred", "e91_online_gain_true_hot_hl16_vw0.25_pred", "e92_online_vintage_true_hot_hl4_vw0.25_pred", "e92_online_vintage_true_hot_hl8_vw0.25_pred", "e92_online_vintage_true_hot_hl16_vw0.25_pred"]},
]


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual), 1e-6)
    return float(np.mean(np.abs(actual - pred) / denom))


def convex_blend_mape(
    pred_matrix: np.ndarray, actual: np.ndarray, w0: np.ndarray
) -> float:
    def objective(w: np.ndarray) -> float:
        return mape(actual, pred_matrix @ w)

    result = minimize(
        objective, w0, method="SLSQP",
        bounds=[(0.0, 1.0)] * pred_matrix.shape[1],
        constraints={"type": "eq", "fun": lambda v: v.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return float(objective(result.x)) if result.success else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cols", type=int, default=8, help="单 run 最多取的预测列数（控制规模）")
    args = parser.parse_args()

    # ---- 加载并聚合所有预测列到 dev 长表 ----
    dev_parts: list[pd.DataFrame] = []
    blind_parts: list[pd.DataFrame] = []
    source_cols: dict[str, str] = {}  # {source: column}
    for src in SOURCES:
        path = ROOT / src["path"]
        if not path.exists():
            print(f"SKIP missing: {src['name']}")
            continue
        df = pd.read_csv(path, parse_dates=["origin_time"])
        keys = ["fold", "origin_time", "target", "horizon"]
        available = [c for c in src["cols"] if c in df.columns][: args.max_cols]
        if not available:
            print(f"SKIP no cols: {src['name']}")
            continue
        for col in available:
            colname = f"{src['name']}::{col}"
            cols = keys + [col]
            if src["name"] == "champion":
                cols = keys + ["actual", col]  # champion 带 actual 作为锚
            part = df[cols].rename(columns={col: colname})
            dev = part.loc[part["fold"].ne("blind")]
            blind = part.loc[part["fold"].eq("blind")]
            if len(dev) < 1000:
                print(f"SKIP too small: {src['name']}::{col} rows={len(dev)}")
                continue
            dev_parts.append(dev)
            blind_parts.append(blind)
            source_cols[colname] = col
            print(f"  load {src['name']}::{col} dev_rows={len(dev)}")
    print(f"total candidate columns: {len(source_cols)}")

    # champion column is the anchor
    champ_col = "champion::v2_v3_target_reconciled_pred"
    if champ_col not in source_cols:
        raise ValueError("champion column missing")

    dev = dev_parts[0]
    for part in dev_parts[1:]:
        dev = dev.merge(part, on=["fold", "origin_time", "target", "horizon"], how="outer")
    dev = dev.sort_values(["fold", "origin_time", "target", "horizon"]).reset_index(drop=True)

    # drop rows missing champion OR any candidate (research OOFs are 5-fold screening;
    # restrict to the fold intersection so every blend is apples-to-apples)
    pred_cols = [c for c in source_cols if c != champ_col]
    dev = dev.loc[dev[champ_col].notna()]
    for col in pred_cols:
        dev = dev.loc[dev[col].notna()]
    dev = dev.reset_index(drop=True)
    dev_folds = sorted(dev["fold"].unique())
    # actual comes from the champion part (first loaded)
    actual = dev["actual"].to_numpy(dtype=float)
    if not np.isfinite(actual).all():
        raise ValueError("actual 存在 NaN")
    print(f"merged dev rows: {len(dev)}, folds: {dev_folds}, candidates: {len(pred_cols)}")

    # ---- per-candidate diagnostics ----
    champ_mape = mape(actual, dev[champ_col].to_numpy(dtype=float))
    results: dict[str, object] = {
        "champion_col": champ_col,
        "champion_dev_mape": champ_mape,
        "n_rows": int(len(dev)),
        "folds": dev_folds,
        "candidates": {},
    }

    # residual correlation vs champion
    champ_resid = actual - dev[champ_col].to_numpy(dtype=float)

    for col in pred_cols:
        pred = dev[col].to_numpy(dtype=float)
        standalone = mape(actual, pred)
        resid = actual - pred
        corr = float(np.corrcoef(champ_resid, resid)[0, 1])
        # in-sample convex blend (champion + this candidate)
        pm = np.column_stack([dev[champ_col].to_numpy(dtype=float), pred])
        insample = convex_blend_mape(pm, actual, np.array([0.9, 0.1]))
        insample_gain = (champ_mape - insample) * 100
        results["candidates"][col] = {
            "standalone_mape": standalone,
            "resid_corr_vs_champion": round(corr, 4),
            "insample_2way_blend_mape": insample,
            "insample_blend_gain_pp": round(insample_gain, 4),
            "better_than_champion": standalone < champ_mape,
        }
        print(f"{col:46s} mape={standalone:.5f} corr={corr:+.3f} insample_blend={insample:.5f} gain={insample_gain:+.4f}pp")

    # ---- LOO-fold honest blend: champion + best single candidate ----
    # for each dev fold, fit weight on all OTHER folds, apply to this fold
    best_single = max(
        results["candidates"],
        key=lambda c: results["candidates"][c]["insample_blend_gain_pp"],
    )
    honest = {}
    for holdout in dev_folds:
        train = dev.loc[dev["fold"].ne(holdout)]
        test = dev.loc[dev["fold"].eq(holdout)]
        pm_train = np.column_stack([train[champ_col].to_numpy(dtype=float), train[best_single].to_numpy(dtype=float)])
        pm_test = np.column_stack([test[champ_col].to_numpy(dtype=float), test[best_single].to_numpy(dtype=float)])

        def obj(w: np.ndarray) -> float:
            return mape(train["actual"].to_numpy(dtype=float), pm_train @ w)

        r = minimize(
            obj, np.array([0.9, 0.1]), method="SLSQP",
            bounds=[(0.0, 1.0)] * 2,
            constraints={"type": "eq", "fun": lambda v: v.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        w = r.x if r.success else np.array([1.0, 0.0])
        blended = pm_test @ w
        test_mape = mape(test["actual"].to_numpy(dtype=float), blended)
        test_base = mape(test["actual"].to_numpy(dtype=float), test[champ_col].to_numpy(dtype=float))
        honest[holdout] = {"w": [round(float(w[0]), 4), round(float(w[1]), 4)], "blend_mape": test_mape, "champion_mape": test_base}
    honest_gain = sum((honest[f]["champion_mape"] - honest[f]["blend_mape"]) for f in honest) * 100 / len(honest)
    results["loo_honest"] = {
        "best_single": best_single,
        "per_fold": honest,
        "mean_blend_gain_pp": round(honest_gain, 4),
        "note": "LOO weight (fit on other folds) applied to holdout; real causal estimate",
    }
    print(f"\nLOO honest blend ({best_single}): mean gain {honest_gain:+.4f}pp over champion")

    # ---- blind observation (reference only) ----
    blind = blind_parts[0]
    for part in blind_parts[1:]:
        blind = blind.merge(part, on=["fold", "origin_time", "target", "horizon"], how="outer")
    if not blind.empty:
        blind = blind.loc[blind[champ_col].notna()].reset_index(drop=True)
        results["blind"] = {
            "rows": int(len(blind)),
            "champion_mape": mape(blind["actual"].to_numpy(dtype=float), blind[champ_col].to_numpy(dtype=float)),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
