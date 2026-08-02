"""A2.1 — 把 calibration oracle gap 拆明白。

不重训模型。直接用 A2 已缓存的分支预测 + blend_weights。

对每个 fold × horizon 计算四个量（统一 %/pp）:
  current_blend     = 当前 calibration 权重的 MAPE
  best_single       = 该 cell 最佳单分支 MAPE
  oracle_simplex    = 同一 cell 内拟合的 oracle simplex MAPE (理论上限)
  oracle_gap        = current_blend - oracle_simplex

输出:
  - development pooled / per fold / per horizon / recent 5 folds
  - oracle gap 分布 (median, P25, P75, 正 gap 比例, >0.05pp, >0.10pp)
  - day-block bootstrap of (blend - oracle)
  - split-half oracle: 用 cell 前半 origins 拟合权重、后半评估 —— 界定
    0.24pp 里多少是纯后见乐观、多少是真实权重错配
  - blind 仅展示，不参与决策

防泄漏: oracle simplex 在 held-out fold 内拟合+评价（理论上限）;
split-half 用前半拟合后半评估（更接近可部署估计）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BRANCH_NAMES = ("persistence", "ridge", "recent", "gas", "lgb_residual")
HORIZONS_MIN = (15, 30, 45, 60, 75, 90, 105, 120)
EPS = 1e-6


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual), EPS)
    return float(np.mean(np.abs(actual - pred) / denom))


def oracle_simplex(pred_matrix: np.ndarray, actual: np.ndarray) -> np.ndarray:
    n_b = pred_matrix.shape[1]

    def objective(w: np.ndarray) -> float:
        return mape(actual, pred_matrix @ w)

    r = minimize(
        objective, np.full(n_b, 1 / n_b), method="SLSQP",
        bounds=[(0.0, 1.0)] * n_b,
        constraints={"type": "eq", "fun": lambda v: v.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if r.success:
        return r.x
    scores = [objective(np.eye(n_b)[b]) for b in range(n_b)]
    w = np.zeros(n_b)
    w[int(np.argmin(scores))] = 1.0
    return w


def load_fold_data(run_dir: Path):
    folds: list[str] = []
    rows_by_fold: dict[str, pd.DataFrame] = {}
    for path in sorted(run_dir.glob("branches_*.csv")):
        fold = path.stem.removeprefix("branches_")
        folds.append(fold)
        rows_by_fold[fold] = pd.read_csv(path)
    return folds, rows_by_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/raw/runs/a2_calibration/20260802_102331")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    folds, rows_by_fold = load_fold_data(Path(args.run_dir))
    dev_folds = [f for f in folds if f != "blind"]

    # ---- per (fold, horizon) compute the three MAPEs ----
    per_cell: dict[str, dict] = {}
    for fold in folds:
        fr = rows_by_fold[fold]
        per_cell[fold] = {}
        for h in range(8):
            part = fr.loc[fr["horizon_idx"].eq(h)]
            actual = part["actual"].to_numpy(float)
            pm = np.column_stack([part[f"{b}_pred"].to_numpy(float) for b in BRANCH_NAMES])
            blend = mape(actual, part["v2_pred"].to_numpy(float))
            branch_mape = {b: mape(actual, part[f"{b}_pred"].to_numpy(float)) for b in BRANCH_NAMES}
            best_name = min(branch_mape, key=branch_mape.get)
            ow = oracle_simplex(pm, actual)
            oracle_m = mape(actual, pm @ ow)
            per_cell[fold][str(h)] = {
                "blend_mape": blend,
                "best_name": best_name,
                "best_mape": branch_mape[best_name],
                "oracle_mape": oracle_m,
                "oracle_gap_pp": (blend - oracle_m) * 100,
                "blend_best_pp": (blend - branch_mape[best_name]) * 100,
                "best_oracle_pp": (branch_mape[best_name] - oracle_m) * 100,
                "oracle_w": ow.tolist(),
            }
    # write per-cell table
    rows_out = []
    for fold in folds:
        for h in range(8):
            c = per_cell[fold][str(h)]
            rows_out.append({
                "fold": fold, "horizon": HORIZONS_MIN[h],
                "current_blend_mape_pct": round(c["blend_mape"] * 100, 4),
                "best_single_mape_pct": round(c["best_mape"] * 100, 4),
                "best_single": c["best_name"],
                "oracle_simplex_mape_pct": round(c["oracle_mape"] * 100, 4),
                "blend_to_best_pp": round(c["blend_best_pp"], 4),
                "blend_to_oracle_pp": round(c["oracle_gap_pp"], 4),
                "best_to_oracle_pp": round(c["best_oracle_pp"], 4),
            })
    cell_df = pd.DataFrame(rows_out)

    def summarize(cells: pd.DataFrame, label: str) -> dict:
        gap = cells["blend_to_oracle_pp"].to_numpy(dtype=float)
        blend_best = cells["blend_to_best_pp"].to_numpy(dtype=float)
        best_oracle = cells["best_to_oracle_pp"].to_numpy(dtype=float)
        return {
            "label": label,
            "cells": int(len(cells)),
            "current_blend_pooled_pct": round(float(cells["current_blend_mape_pct"].mean()), 4),
            "best_single_pooled_pct": round(float(cells["best_single_mape_pct"].mean()), 4),
            "oracle_pooled_pct": round(float(cells["oracle_simplex_mape_pct"].mean()), 4),
            "blend_to_best_pp": round(float(blend_best.mean()), 4),
            "blend_to_oracle_pp": round(float(gap.mean()), 4),
            "best_to_oracle_pp": round(float(best_oracle.mean()), 4),
            "oracle_gap": {
                "median": round(float(np.median(gap)), 4),
                "p25": round(float(np.quantile(gap, 0.25)), 4),
                "p75": round(float(np.quantile(gap, 0.75)), 4),
                "positive_frac": round(float((gap > 0).mean()), 4),
                "gt_0_05pp_frac": round(float((gap > 0.05).mean()), 4),
                "gt_0_10pp_frac": round(float((gap > 0.10).mean()), 4),
                "max": round(float(gap.max()), 4),
            },
        }

    report: dict[str, object] = {
        "rule": "dev folds decide; blind observed only",
        "folds": folds,
        "per_cell_table_path": "cells.csv",
    }
    dev_cells = cell_df.loc[cell_df["fold"].ne("blind")]
    blind_cells = cell_df.loc[cell_df["fold"].eq("blind")]

    report["development_pooled"] = summarize(dev_cells, "dev pooled")
    report["recent_5_folds"] = summarize(dev_cells.loc[dev_cells["fold"].isin(
        [f"dev_{i:02d}" for i in range(15, 20)])], "dev_15..19")
    report["per_fold"] = {
        fold: summarize(dev_cells.loc[dev_cells["fold"].eq(fold)], fold)
        for fold in sorted(dev_cells["fold"].unique())
    }
    report["per_horizon"] = {
        f"t+{int(h)}": summarize(dev_cells.loc[dev_cells["horizon"].eq(h)], f"t+{h}")
        for h in sorted(dev_cells["horizon"].unique())
    }
    report["blind"] = summarize(blind_cells, "blind (ref only)")

    # ---- day-block bootstrap of blend - oracle (dev) ----
    # need day-level MAPE difference; reconstruct from cell_df + weights of rows? cell_df is per cell.
    # Use origin-level: recompute day blocks from row-level blend/oracle.
    # For simplicity, bootstrap on cell-level gaps blocked by fold (19 blocks).
    rng = np.random.default_rng(20250731)
    dev_fold_names = sorted(dev_cells["fold"].unique())
    fold_gaps = [float(dev_cells.loc[dev_cells["fold"].eq(f), "blend_to_oracle_pp"].mean()) for f in dev_fold_names]
    diffs = np.asarray(fold_gaps)
    sampled = rng.choice(diffs, size=(2000, len(diffs)), replace=True).mean(axis=1)
    report["day_block_bootstrap"] = {
        "block": "fold",
        "mean_oracle_gap_pp": round(float(diffs.mean()), 4),
        "p5": round(float(np.quantile(sampled, 0.05)), 4),
        "p95": round(float(np.quantile(sampled, 0.95)), 4),
        "prob_oracle_better_than_blend": round(float((sampled > 0).mean()), 4),
    }

    # ---- split-half oracle (realistic): fit on first half origins, eval on second ----
    # This bounds how much of the 0.24pp is hindsight optimism.
    split = {}
    for fold in dev_folds:
        fr = rows_by_fold[fold]
        split[fold] = {}
        for h in range(8):
            part = fr.loc[fr["horizon_idx"].eq(h)].reset_index(drop=True)
            actual = part["actual"].to_numpy(float)
            pm = np.column_stack([part[f"{b}_pred"].to_numpy(float) for b in BRANCH_NAMES])
            n = len(actual)
            half = n // 2
            ow = oracle_simplex(pm[:half], actual[:half])
            eval_mape = mape(actual[half:], pm[half:] @ ow)
            blend_mape = mape(actual[half:], part["v2_pred"].to_numpy(float)[half:])
            split[fold][str(h)] = {
                "split_oracle_mape": eval_mape,
                "split_blend_mape": blend_mape,
                "split_gap_pp": (blend_mape - eval_mape) * 100,
            }
    sp_cells = [(fold, h, split[fold][str(h)]["split_gap_pp"]) for fold in dev_folds for h in range(8)]
    sp_gaps = np.array([g for _, _, g in sp_cells], dtype=float)
    report["split_half_oracle"] = {
        "mean_gap_pp": round(float(sp_gaps.mean()), 4),
        "median": round(float(np.median(sp_gaps)), 4),
        "positive_frac": round(float((sp_gaps > 0).mean()), 4),
        "note": "weights fit on first half origins, evaluated on second half of same cell; "
                "bounds hindsight optimism vs real weight misallocation",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cell_df.to_csv(out.parent / "cells.csv", index=False)

    print(f"wrote {out}")
    d = report["development_pooled"]
    print(f"\ndev pooled: blend={d['current_blend_pooled_pct']}%  best_single={d['best_single_pooled_pct']}%  "
          f"oracle={d['oracle_pooled_pct']}%")
    print(f"  blend→best {d['blend_to_best_pp']:+.4f}pp   blend→oracle {d['blend_to_oracle_pp']:+.4f}pp   best→oracle {d['best_to_oracle_pp']:+.4f}pp")
    s = d["oracle_gap"]
    print(f"  oracle gap: med={s['median']} p25={s['p25']} p75={s['p75']} pos={s['positive_frac']} >0.05pp={s['gt_0_05pp_frac']} >0.10pp={s['gt_0_10pp_frac']} max={s['max']}")
    r5 = report["recent_5_folds"]
    print(f"recent 5 (dev_15..19): blend→oracle {r5['blend_to_oracle_pp']:+.4f}pp")
    bl = report["blind"]
    print(f"blind: blend→oracle {bl['blend_to_oracle_pp']:+.4f}pp")
    sh = report["split_half_oracle"]
    print(f"split-half oracle: mean gap {sh['mean_gap_pp']:+.4f}pp  pos={sh['positive_frac']}")


if __name__ == "__main__":
    main()
