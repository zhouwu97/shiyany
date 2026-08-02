"""A2 分析 — 从仪器化 fold 分支数据计算 8 项 calibration 稳定性指标。

输入: results/raw/runs/a2_calibration/<stamp>/branches_<fold>.csv + weights_<fold>.json
输出: results/runs/<stamp>/a2_metrics.json

指标:
  1. per-fold, per-branch held-out MAPE (branch 质量随 fold 的方差)
  2. per-fold simplex weights            (权重随 fold 漂移)
  3. persistence weight vs next-fold persistence error (绝对)
  4. lgb weight vs next-fold lgb error                   (绝对)
  5. blend regret: MAPE(blend) - MAPE(best_branch) per fold x horizon
  6. weight turnover: 相邻 fold 的 L1 sum|w_k - w_{k-1}|
  7. weight -> next-fold advantage (相对, 去整体难度混淆)
  8. oracle gap: 同一 fold 上 (a)历史calib权重 (b)单最佳branch (c)该fold oracle simplex

规则: 结论只基于 development folds; blind 仅观察。

输出三汇总:
  S1: blend 胜过最佳单分支的 fold x horizon 单元格占比
  S2: persistence 高权重 -> 下一 fold persistence 相对 advantage 的方向
  S3: lgb 高权重     -> 下一 fold lgb 相对 advantage 的方向
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

BRANCH_NAMES = ("persistence", "ridge", "recent", "gas", "lgb_residual")
HORIZON_IDX = list(range(8))
BRANCH_COLS = [f"{name}_pred" for name in BRANCH_NAMES]
EPS = 1e-6


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual), EPS)
    return float(np.mean(np.abs(actual - pred) / denom))


def oracle_simplex(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    n_b = pred.shape[1]
    def objective(value: np.ndarray) -> float:
        return _mape(actual, pred @ value)
    x0 = np.full(n_b, 1.0 / n_b)
    result = minimize(
        objective, x0, method="SLSQP",
        bounds=[(0.0, 1.0)] * n_b,
        constraints={"type": "eq", "fun": lambda v: v.sum() - 1.0},
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if result.success:
        return result.x
    scores = [objective(np.eye(n_b)[b]) for b in range(n_b)]
    winner = int(np.argmin(scores))
    weights = np.zeros(n_b)
    weights[winner] = 1.0
    return weights


def load_fold_data(run_dir: Path) -> tuple[list[str], dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """返回 dev 有序列表 + {fold: rows} + {fold: blend_weights (5,8)}。"""
    fold_names: list[str] = []
    rows_by_fold: dict[str, pd.DataFrame] = {}
    weights_by_fold: dict[str, np.ndarray] = {}
    for path in sorted(run_dir.glob("branches_*.csv")):
        fold = path.stem.removeprefix("branches_")
        fold_names.append(fold)
        rows_by_fold[fold] = pd.read_csv(path)
        weight_path = run_dir / f"weights_{fold}.json"
        if not weight_path.exists():
            raise FileNotFoundError(f"missing {weight_path}")
        weights_by_fold[fold] = np.asarray(
            json.loads(weight_path.read_text(encoding="utf-8")), dtype=float
        )
    return fold_names, rows_by_fold, weights_by_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fold_names, rows_by_fold, weights_by_fold = load_fold_data(run_dir)
    dev_folds = [f for f in fold_names if f != "blind"]

    # 预计算每个 fold x horizon 的逐分支 MAPE + blend + oracle
    per_cell: dict[str, dict[str, dict]] = {}
    for fold in fold_names:
        fr = rows_by_fold[fold]
        per_cell[fold] = {}
        for h in HORIZON_IDX:
            part = fr.loc[fr["horizon_idx"].eq(h)]
            actual = part["actual"].to_numpy(float)
            pred_matrix = part[BRANCH_COLS].to_numpy(float)
            branch_mape = {name: _mape(actual, pred_matrix[:, i]) for i, name in enumerate(BRANCH_NAMES)}
            best_name = min(branch_mape, key=branch_mape.get)
            blend = _mape(actual, part["v2_pred"].to_numpy(float))
            oracle_w = oracle_simplex(pred_matrix, actual)
            oracle_m = _mape(actual, pred_matrix @ oracle_w)
            per_cell[fold][str(h)] = {
                "branch_mape": {k: round(100.0 * v, 4) for k, v in branch_mape.items()},
                "best_branch": best_name,
                "best_branch_mape": round(100.0 * branch_mape[best_name], 4),
                "blend_mape": round(100.0 * blend, 4),
                "regret_pp": round(100.0 * (blend - branch_mape[best_name]), 4),
                "oracle_mape": round(100.0 * oracle_m, 4),
                "oracle_weights": [round(float(w), 4) for w in oracle_w],
                "calib_weights": [round(float(w), 4) for w in weights_by_fold[fold][:, h]],
            }

    # ---- S1: blend beats best single branch fraction (dev cells) ----
    dev_cells = [per_cell[f][str(h)] for f in dev_folds for h in HORIZON_IDX]
    s1_beat = sum(1 for c in dev_cells if c["regret_pp"] < 0.0)
    s1 = {
        "blend_beats_best_branch_cells": s1_beat,
        "total_dev_cells": len(dev_cells),
        "fraction": round(s1_beat / len(dev_cells), 4) if dev_cells else None,
    }

    # ---- metric 6: weight turnover over adjacent dev folds ----
    turnover = {}
    for h in HORIZON_IDX:
        series = [weights_by_fold[f][:, h] for f in dev_folds]
        steps = [
            float(np.abs(series[i + 1] - series[i]).sum())
            for i in range(len(series) - 1)
        ]
        turnover[str(h)] = {
            "per_adjacent_fold": [round(v, 4) for v in steps],
            "mean": round(float(np.mean(steps)), 4) if steps else None,
            "max": round(float(np.max(steps)), 4) if steps else None,
        }
    turnover_mean_all_h = np.mean(
        [turnover[str(h)]["mean"] for h in HORIZON_IDX if turnover[str(h)]["mean"] is not None]
    ) if any(turnover[str(h)]["mean"] is not None for h in HORIZON_IDX) else None

    # ---- metric 7 + S2/S3: weight -> next-fold relative advantage ----
    # advantage_k+1(b, h) = (median 分支 MAPE - 该分支 MAPE) 在下一 fold, 去整体难度
    # correlation across dev folds between weight_k(b,h) and advantage_k+1(b,h)
    def next_fold_advantage(fold: str, h: int, rows_by_fold, dev_folds) -> dict[str, float] | None:
        i = dev_folds.index(fold)
        if i + 1 >= len(dev_folds):
            return None
        nxt = dev_folds[i + 1]
        part = rows_by_fold[nxt]
        part = part.loc[part["horizon_idx"].eq(h)]
        actual = part["actual"].to_numpy(float)
        pm = part[BRANCH_COLS].to_numpy(float)
        mape = {name: _mape(actual, pm[:, i]) for i, name in enumerate(BRANCH_NAMES)}
        med = float(np.median(list(mape.values())))
        return {name: med - mape[name] for name in BRANCH_NAMES}

    weight_next: dict[str, list[tuple[float, float]]] = {name: [] for name in BRANCH_NAMES}
    for fold in dev_folds[:-1]:
        for h in HORIZON_IDX:
            adv = next_fold_advantage(fold, h, rows_by_fold, dev_folds)
            if adv is None:
                continue
            w = weights_by_fold[fold][:, h]
            for i, name in enumerate(BRANCH_NAMES):
                weight_next[name].append((float(w[i]), adv[name]))

    direction: dict[str, float] = {}
    sample_n: dict[str, int] = {}
    for name in BRANCH_NAMES:
        pairs = weight_next[name]
        sample_n[name] = len(pairs)
        if len(pairs) >= 5 and np.std([p[0] for p in pairs]) > 0 and np.std([p[1] for p in pairs]) > 0:
            rho, _ = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
            direction[name] = round(float(rho), 4)
        else:
            direction[name] = None

    s2 = {
        "persistence": {
            "spearman_weight_vs_next_advantage": direction["persistence"],
            "n_pairs": sample_n["persistence"],
            "high_weight_implies_advantage": direction["persistence"] is not None and direction["persistence"] > 0,
        }
    }
    s3 = {
        "lgb_residual": {
            "spearman_weight_vs_next_advantage": direction["lgb_residual"],
            "n_pairs": sample_n["lgb_residual"],
            "high_weight_implies_advantage": direction["lgb_residual"] is not None and direction["lgb_residual"] > 0,
        }
    }

    # ---- metric 8: oracle gap (dev folds) ----
    oracle_gap = {}
    for fold in dev_folds:
        oracle_gap[fold] = {}
        for h in HORIZON_IDX:
            c = per_cell[fold][str(h)]
            oracle_gap[fold][str(h)] = {
                "calib_blend_pp": c["blend_mape"],
                "best_branch_pp": c["best_branch_mape"],
                "oracle_pp": c["oracle_mape"],
                "gap_calib_vs_oracle_pp": round(c["blend_mape"] - c["oracle_mape"], 4),
                "gap_best_vs_oracle_pp": round(c["best_branch_mape"] - c["oracle_mape"], 4),
            }

    # ---- metric 2: weight drift summary ----
    weight_drift = {}
    for h in HORIZON_IDX:
        series = np.stack([weights_by_fold[f][:, h] for f in dev_folds], axis=1)  # (5, n_dev)
        weight_drift[str(h)] = {
            "first": [round(float(v), 4) for v in series[:, 0]],
            "last": [round(float(v), 4) for v in series[:, -1]],
            "mean": [round(float(v), 4) for v in series.mean(axis=1)],
            "std_across_folds": [round(float(v), 4) for v in series.std(axis=1)],
        }

    result = {
        "rule": "conclusions from development folds only; blind reported for observation",
        "folds": fold_names,
        "s1_blend_vs_best": s1,
        "s2_persistence_weight_advantage": s2,
        "s3_lgb_weight_advantage": s3,
        "per_cell": per_cell,
        "oracle_gap": oracle_gap,
        "weight_drift": weight_drift,
        "weight_turnover": turnover,
        "weight_turnover_mean_across_horizons": (
            round(float(turnover_mean_all_h), 4) if turnover_mean_all_h is not None else None
        ),
        "next_fold_advantage_spearman_by_branch": direction,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"S1: blend beats best single branch in {s1_beat}/{len(dev_cells)} dev cells "
          f"({s1['fraction']})")
    print(f"S2 persistence: rho={direction['persistence']}, n={sample_n['persistence']}")
    print(f"S3 lgb:         rho={direction['lgb_residual']}, n={sample_n['lgb_residual']}")


if __name__ == "__main__":
    main()
