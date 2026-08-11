"""PRED-8A：Dynamic Opportunity Audit（纯诊断，零 ML）。

在 SAFE60 时代重新测量动态选择空间。专家：SAFE60/X3/A61/A64。

输出：
1) SAFE60-inclusive origin oracle（raw + 30/60/120min min-dwell 约束）
2) winner/regret 持续性（repeat prob、dwell、转移矩阵、regret 自相关 k=1/2/4/8）
3) 严格 chronological matured-loss 延迟追踪（trailing 30/60/120/240min + EMA）
三数字 gate：constrained oracle / regret 自相关 / delayed tracker MAPE。

严格因果：origin t 只用 target_time<=t 的 matured loss；不读取未来 actual。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SAFE60_OOF = Path("results/raw/runs/audits/pred1_gate_c_20260810/merged_safe60_eval.csv")
A64_OOF = Path("results/raw/runs/experiments/p3_rolling_training_20260809_190558/a64_direct_delta_oof.csv")
EXPERTS = ("SAFE60", "X3", "A61", "A64")
TRACKER_WINDOWS = (30, 60, 120, 240)  # minutes
TRACKER_EMA = 0.25


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p) / np.maximum(np.abs(a), 1e-6)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    m = pd.read_csv(SAFE60_OOF, parse_dates=["origin_time", "train_end"])
    a64_path = Path("results/raw/runs/experiments/p3_rolling_training_20260809_190558/a64_direct_delta_oof.csv")
    a64 = pd.read_csv(a64_path, parse_dates=["origin_time"])
    key = ["fold", "origin_time", "target", "horizon"]
    work = m[key + ["actual", "x3_cat_mae_pred", "a61_recursive_blend_05_pred", "safe60_pred"]].merge(
        a64[key + ["ridge_prediction"]], on=key, how="inner")
    work["A64"] = work["ridge_prediction"]
    work["X3"] = work["x3_cat_mae_pred"]
    work["A61"] = work["a61_recursive_blend_05_pred"]
    work["SAFE60"] = work["safe60_pred"]
    work = work.sort_values(["origin_time", "target", "horizon"]).reset_index(drop=True)

    # 每 origin 聚合 APE 矩阵
    ape = pd.DataFrame({"origin_time": work["origin_time"]})
    for e in EXPERTS:
        ape[e] = (work["actual"] - work[e]).abs().div(work["actual"].abs().clip(lower=1e-6))
    per_origin = ape.groupby("origin_time")[[*EXPERTS]].mean()  # L_j(t) = mean APE over 16 cells
    per_origin = per_origin.sort_index()
    origins = list(per_origin.index)
    n_origins = len(origins)

    # --- 1) Oracle ---
    oracle_raw = per_origin.min(axis=1).mean() * 100
    safe60_all = per_origin["SAFE60"].mean() * 100

    def dwell_oracle(min_dwell: int) -> float:
        """只能每 min_dwell 个 origin 切换一次 winner（从上一段起点延续）。"""
        loss_sum = 0.0
        count = 0
        idx = 0
        while idx < n_origins:
            block = per_origin.iloc[idx:min(idx + min_dwell, n_origins)]
            winner = block.mean(axis=0).idxmin()
            loss_sum += block[winner].sum()
            count += block.shape[0]
            idx += min_dwell
        return loss_sum / count * 100

    oracles = {
        "safe60_all": round(safe60_all, 6),
        "origin_oracle_raw": round(oracle_raw, 6),
        "oracle_dwell2_30min": round(dwell_oracle(2), 6),
        "oracle_dwell4_60min": round(dwell_oracle(4), 6),
        "oracle_dwell8_120min": round(dwell_oracle(8), 6),
    }

    # --- 2) winner persistence ---
    winner = per_origin.idxmin(axis=1)
    repeat1 = float((winner.shift(1) == winner).dropna().mean())
    repeat2 = float((winner.shift(2) == winner).dropna().mean())
    # dwell
    run_len = []
    cur = 1
    for i in range(1, n_origins):
        if winner.iloc[i] == winner.iloc[i - 1]:
            cur += 1
        else:
            run_len.append(cur)
            cur = 1
    run_len.append(cur)
    dwell = pd.Series(run_len)
    # transition matrix
    from collections import Counter
    trans = Counter(zip(winner.values[:-1], winner.values[1:]))
    tmat = {}
    for e in EXPERTS:
        total = sum(v for (i, j), v in trans.items() if i == e)
        tmat[e] = {j: round(trans.get((e, j), 0) / total, 4) if total else None for j in EXPERTS}
    # regret 自相关（对非 NaN 序列手动计算）
    regret = per_origin.sub(per_origin["SAFE60"], axis=0)  # R_j(t) = L_SAFE60 - L_j
    regret_ac = {}
    for k in (1, 2, 4, 8):
        ac = {}
        for e in EXPERTS:
            if e == "SAFE60":
                continue
            rv = regret[e].to_numpy(dtype=float)
            rv = rv[~np.isnan(rv)]
            if len(rv) > k + 5:
                ac[e] = round(float(np.corrcoef(rv[:-k], rv[k:])[0, 1]), 4)
            else:
                ac[e] = None
        regret_ac[f"lag{k}"] = ac

    # --- 3) chronological matured-loss delayed tracker ---
    # 账本：每条 (expert, origin, target, horizon, pred, actual, matured_time)
    work["matured_time"] = work["origin_time"] + pd.to_timedelta(work["horizon"], unit="m")
    work = work.sort_values("origin_time")
    # 每个 origin 的 16 个 cell 的 expert 预测（用于最终选择后评估）
    # 逐 origin 前向：在 t，可见 matured_time<=t 的行
    tracker_results = {}
    for win in TRACKER_WINDOWS:
        sel_preds = []
        for i, t in enumerate(origins):
            visible = work[work["matured_time"] <= t]
            recent = visible[visible["matured_time"] > t - pd.Timedelta(minutes=win)]
            if recent.empty:
                sel = "SAFE60"
            else:
                loss = {}
                for e in EXPERTS:
                    v = recent
                    loss[e] = float((v["actual"] - v[e]).abs().div(v["actual"].abs().clip(lower=1e-6)).mean())
                sel = min(loss, key=loss.get)
            # 当前 origin 的该专家预测
            cur = work[(work["origin_time"] == t)]
            sel_preds.append(cur[["target", "horizon", "actual", sel]].rename(columns={sel: "pred"}))
        r = pd.concat(sel_preds)
        tracker_results[f"trail{win}min"] = _mape(r["actual"].to_numpy(float), r["pred"].to_numpy(float))
    # EMA tracker
    ema_state = {e: None for e in EXPERTS}
    sel_preds = []
    for i, t in enumerate(origins):
        visible = work[work["matured_time"] <= t]
        if visible.empty:
            sel = "SAFE60"
        else:
            for e in EXPERTS:
                v = visible
                val = float((v["actual"] - v[e]).abs().div(v["actual"].abs().clip(lower=1e-6)).mean())
                ema_state[e] = val if ema_state[e] is None else TRACKER_EMA * val + (1 - TRACKER_EMA) * ema_state[e]
            sel = min(ema_state, key=ema_state.get)
        cur = work[(work["origin_time"] == t)]
        sel_preds.append(cur[["target", "horizon", "actual", sel]].rename(columns={sel: "pred"}))
    r = pd.concat(sel_preds)
    tracker_results["ema025"] = _mape(r["actual"].to_numpy(float), r["pred"].to_numpy(float))

    report = {
        "experiment": "PRED-8A_dynamic_opportunity_audit",
        "baseline": "SAFE60",
        "formal_candidate": False,
        "causal": True,
        "experts": list(EXPERTS),
        "oracles_pp": oracles,
        "winner_persistence": {"repeat_lag1": repeat1, "repeat_lag2": repeat2,
                               "dwell_mean": round(float(dwell.mean()), 2), "dwell_median": round(float(dwell.median()), 1),
                               "dwell_p75": round(float(dwell.quantile(0.75)), 1), "dwell_p90": round(float(dwell.quantile(0.90)), 1),
                               "dwell_max": int(dwell.max())},
        "transition_matrix": tmat,
        "regret_autocorr": regret_ac,
        "delayed_tracker_mape": {k: round(v, 6) for k, v in tracker_results.items()},
        "safe60_mape_pp": round(safe60_all, 6),
        "three_numbers": {
            "constrained_oracle_pp": oracles["oracle_dwell4_60min"],
            "regret_ac_lag1_mean": round(float(np.mean([v for v in regret_ac["lag1"].values() if v is not None])), 4) if any(v is not None for v in regret_ac["lag1"].values()) else None,
            "delayed_tracker_best": min(tracker_results.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
