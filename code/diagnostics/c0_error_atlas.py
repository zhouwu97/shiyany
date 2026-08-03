"""
c0_error_atlas.py — 无需训练的 D1 + D3 诊断
输出到 results/diagnostics/c0_error_atlas/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "results/raw/runs/oof/clean_c0_strict_20260801_v2/oof_with_routes.csv"
PRICE_PATH = ROOT / "data/raw/official/初赛-参赛者使用/price.xlsx"
OUT_DIR = ROOT / "results/diagnostics/c0_error_atlas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [15, 30, 45, 60, 75, 90, 105, 120]
EPS = 1e-6

# ──────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────

def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.where(np.abs(actual) < EPS, EPS, np.abs(actual))
    return float(np.mean(np.abs(actual - pred) / denom) * 100)


def fold_is_blind(fold: str) -> bool:
    return fold == "fold_blind" or fold.startswith("blind")


# ──────────────────────────────────────────────────
# D1 — V2 / V3 / Persistence × generator_1 × horizon × fold
# ──────────────────────────────────────────────────

def run_d1(oof: pd.DataFrame) -> None:
    print("[D1] 开始计算 generator_1 对照矩阵 …")

    g1 = oof[oof["target"] == "generator_1"].copy()
    print(f"  generator_1 行数: {len(g1)}")

    # --- 每格 (horizon, fold) 的 MAPE ---
    records = []
    folds_ordered = sorted(g1["fold"].unique(), key=lambda f: (fold_is_blind(f), f))

    for h in HORIZONS:
        sub_h = g1[g1["horizon"] == h]
        v2_folds, v3_folds, pers_folds = [], [], []
        for fold in folds_ordered:
            sub = sub_h[sub_h["fold"] == fold]
            if sub.empty:
                continue
            act = sub["actual"].to_numpy()
            v2  = mape(act, sub["v2_pred"].to_numpy())
            v3  = mape(act, sub["v3_pred"].to_numpy())
            ps  = mape(act, sub["persistence_pred"].to_numpy())
            v2_folds.append(v2)
            v3_folds.append(v3)
            pers_folds.append(ps)

        v2_arr  = np.array(v2_folds)
        v3_arr  = np.array(v3_folds)
        ps_arr  = np.array(pers_folds)
        delta   = v3_arr - v2_arr          # 正 = V3 更差

        dev_mask  = np.array([not fold_is_blind(f) for f in folds_ordered
                               if not sub_h[sub_h["fold"]==f].empty])
        blind_mask = ~dev_mask

        v3_beats_v2      = int((delta < 0).sum())          # V3 MAPE < V2 MAPE
        recent5_v3_win   = int((delta[dev_mask][-5:] < 0).sum()) if dev_mask.sum() >= 5 else None

        records.append(dict(
            horizon         = h,
            V2_MAPE_all     = round(float(v2_arr.mean()), 4),
            V3_MAPE_all     = round(float(v3_arr.mean()), 4),
            V3_minus_V2_pp  = round(float(delta.mean()), 4),
            Pers_MAPE_all   = round(float(ps_arr.mean()), 4),
            V3_beats_V2_folds        = v3_beats_v2,
            V3_beats_V2_folds_total  = len(v2_folds),
            recent5_V3_wins = recent5_v3_win,
            blind_V2_MAPE   = round(float(v2_arr[blind_mask].mean()), 4) if blind_mask.any() else None,
            blind_V3_MAPE   = round(float(v3_arr[blind_mask].mean()), 4) if blind_mask.any() else None,
            blind_delta_pp  = round(float(delta[blind_mask].mean()), 4)  if blind_mask.any() else None,
        ))

    df_target_horizon = pd.DataFrame(records)
    df_target_horizon.to_csv(OUT_DIR / "target_horizon_scores.csv", index=False)
    print(f"  → target_horizon_scores.csv ({len(df_target_horizon)} 行)")

    # --- V2 vs V3 逐格胜负明细 ---
    cell_records = []
    for fold in folds_ordered:
        for h in HORIZONS:
            sub = g1[(g1["fold"] == fold) & (g1["horizon"] == h)]
            if sub.empty:
                continue
            act = sub["actual"].to_numpy()
            v2  = mape(act, sub["v2_pred"].to_numpy())
            v3  = mape(act, sub["v3_pred"].to_numpy())
            ps  = mape(act, sub["persistence_pred"].to_numpy())
            cell_records.append(dict(
                fold=fold, horizon=h,
                V2_MAPE=round(v2, 4), V3_MAPE=round(v3, 4),
                V3_minus_V2=round(v3 - v2, 4),
                Pers_MAPE=round(ps, 4),
                winner="V3" if v3 < v2 else "V2",
                is_blind=fold_is_blind(fold),
            ))

    df_cell = pd.DataFrame(cell_records)
    df_cell.to_csv(OUT_DIR / "v2_v3_cell_comparison.csv", index=False)
    print(f"  → v2_v3_cell_comparison.csv ({len(df_cell)} 行)")

    # --- fold × horizon 胜负矩阵 ---
    pivot = df_cell.pivot(index="fold", columns="horizon", values="V3_minus_V2")
    pivot.to_csv(OUT_DIR / "fold_horizon_matrix.csv")
    print("  → fold_horizon_matrix.csv")

    # --- 打印摘要供快速判断 ---
    print("\n  ── horizon 汇总 (generator_1) ──")
    print(f"  {'h':>5}  {'V2%':>7}  {'V3%':>7}  {'Δpp':>7}  {'V3>V2':>6}  {'最近5':>5}  {'blind_V2':>9}  {'blind_V3':>9}  {'blind_Δ':>8}")
    for r in records:
        recent5 = f"{r['recent5_V3_wins']}/5" if r['recent5_V3_wins'] is not None else "  N/A"
        bv2 = f"{r['blind_V2_MAPE']:.4f}" if r['blind_V2_MAPE'] is not None else "   N/A"
        bv3 = f"{r['blind_V3_MAPE']:.4f}" if r['blind_V3_MAPE'] is not None else "   N/A"
        bd  = f"{r['blind_delta_pp']:+.4f}" if r['blind_delta_pp'] is not None else "   N/A"
        total = r['V3_beats_V2_folds_total']
        print(f"  t+{r['horizon']:>3}  {r['V2_MAPE_all']:7.4f}  {r['V3_MAPE_all']:7.4f}  "
              f"{r['V3_minus_V2_pp']:+7.4f}  {r['V3_beats_V2_folds']:>3}/{total}  {recent5}  "
              f"{bv2}  {bv3}  {bd}")

    print()
    return df_target_horizon, df_cell


# ──────────────────────────────────────────────────
# D3 — Price switch 语义审计
# ──────────────────────────────────────────────────

def run_d3() -> None:
    print("[D3] 加载电价表 …")
    frame = pd.read_excel(PRICE_PATH)
    # 48 行 × 13 列 (第1列标签，后12列=月份1..12)
    if frame.shape[0] != 48:
        print(f"  警告: 期望 48 行，实际 {frame.shape[0]} 行")

    values = frame.iloc[:, 1:13].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    # values[slot, month-1]  slot=0..47  (每半小时)

    # 用一整年的 15min 时间戳枚举 switch 事件
    # 电价以半小时为分辨率，OOF 数据从 2025-03 开始
    # 生成 2025-01-01 ~ 2025-12-31 所有 15min 点
    ts = pd.date_range("2025-01-01", "2025-12-31 23:45", freq="15min")

    def lookup(timestamps: pd.DatetimeIndex) -> np.ndarray:
        months = timestamps.month.to_numpy() - 1
        slots  = timestamps.hour.to_numpy() * 2 + timestamps.minute.to_numpy() // 30
        return values[slots, months]

    current_price = lookup(ts)

    # horizons = [15, 30, 45, 60, 75, 90, 105, 120] 分钟
    horizon_minutes = HORIZONS
    target_prices = {}
    for hm in horizon_minutes:
        tp = lookup(ts + pd.Timedelta(minutes=hm))
        target_prices[hm] = tp

    # 构建价格矩阵 N×8
    price_matrix = np.column_stack([target_prices[hm] for hm in horizon_minutes])

    # ── 当前代码行为 (BUG): baseline = price_matrix[:,0] = t+15 价格
    changed_bug  = price_matrix != price_matrix[:, [0]]
    sw_bug       = changed_bug.any(axis=1)
    first_bug    = np.where(sw_bug, changed_bug.argmax(axis=1) + 1, 0)

    # ── 修正后行为: baseline = current_price (= t 时刻价格)
    changed_fix  = price_matrix != current_price[:, None]
    sw_fix       = changed_fix.any(axis=1)
    first_fix    = np.where(sw_fix, changed_fix.argmax(axis=1) + 1, 0)

    # 找所有有 switch 的行 (按修正定义或 bug 定义)
    any_switch = sw_bug | sw_fix

    events = []
    for i in np.where(any_switch)[0]:
        t = ts[i]
        cp = float(current_price[i])
        tp_row = {f"t+{hm}": float(target_prices[hm][i]) for hm in horizon_minutes}
        events.append(dict(
            timestamp         = str(t),
            month             = int(t.month),
            half_slot         = int(t.hour * 2 + t.minute // 30),
            current_price     = cp,
            **tp_row,
            # BUG 代码输出
            bug_price_switch_within_120 = int(sw_bug[i]),
            bug_steps_to_price_switch   = int(first_bug[i]),
            # 修正后输出
            fix_price_switch_within_120 = int(sw_fix[i]),
            fix_steps_to_price_switch   = int(first_fix[i]),
            # 分歧标志
            diverges = int(sw_bug[i] != sw_fix[i] or first_bug[i] != first_fix[i]),
        ))

    df_events = pd.DataFrame(events)
    df_events.to_csv(OUT_DIR / "price_switch_events.csv", index=False)
    print(f"  → price_switch_events.csv ({len(df_events)} 行)")

    # --- 统计 ---
    n_switch_fix  = int(sw_fix.sum())
    n_switch_bug  = int(sw_bug.sum())
    n_diverge     = int(df_events["diverges"].sum()) if not df_events.empty else 0
    # 立即 switch: fix 检测到但 bug 看不见
    immediate_switch_missed = int(((sw_fix) & (~sw_bug)).sum())
    # 延迟 switch 误差 (步数不同)
    step_wrong = int(((sw_fix & sw_bug) & (first_bug != first_fix)).sum())

    print("\n  ── price switch 统计 ──")
    print(f"  修正后检测到 switch 时间步: {n_switch_fix} / {len(ts)}")
    print(f"  BUG 代码检测到 switch 时间步: {n_switch_bug} / {len(ts)}")
    print(f"  立即 switch (t→t+15) BUG 漏掉: {immediate_switch_missed}")
    print(f"  步数错误 (switch 存在但步数不同): {step_wrong}")
    print(f"  总分歧时间步: {n_diverge}")

    # 按月汇总
    if not df_events.empty:
        monthly = df_events.groupby("month").agg(
            switch_events_fix=("fix_price_switch_within_120", "sum"),
            switch_events_bug=("bug_price_switch_within_120", "sum"),
            diverge_count=("diverges", "sum"),
        ).reset_index()
        monthly.to_csv(OUT_DIR / "price_switch_monthly.csv", index=False)
        print("\n  按月分布:")
        print(monthly.to_string(index=False))
    print()

    return {
        "n_switch_fix": n_switch_fix,
        "n_switch_bug": n_switch_bug,
        "immediate_switch_missed": immediate_switch_missed,
        "step_wrong": step_wrong,
        "n_diverge": n_diverge,
    }


# ──────────────────────────────────────────────────
# 写 summary.json
# ──────────────────────────────────────────────────

def write_summary(d1_df: pd.DataFrame, d3_stats: dict) -> None:
    # D1 关键判断
    long_h = d1_df[d1_df["horizon"].isin([90, 105, 120])]
    v3_wins_long = long_h["V3_beats_V2_folds"].tolist()
    v3_total_long = long_h["V3_beats_V2_folds_total"].tolist()
    horizons_long = long_h["horizon"].tolist()

    # 用户定义的通过条件 (保守)
    # h90: ≥14/20, h105: ≥13/20, h120: ≥15/20
    thresholds = {90: 14, 105: 13, 120: 15}
    pass_d1 = all(
        wins >= thresholds.get(h, 11)
        for h, wins, total in zip(horizons_long, v3_wins_long, v3_total_long)
    )

    summary = {
        "run_oof": str(OOF_PATH),
        "d1_horizon_summary": [
            {"horizon": int(h), "V3_beats_V2": int(w), "total_folds": int(t),
             "V3_minus_V2_pp": float(d1_df[d1_df["horizon"]==h]["V3_minus_V2_pp"].iloc[0])}
            for h, w, t in zip(horizons_long, v3_wins_long, v3_total_long)
        ],
        "d1_gate_pass": pass_d1,
        "d1_verdict": "V3_long_horizon_advantage_confirmed" if pass_d1 else "V3_long_horizon_NOT_confirmed_stop_route",
        "d3_stats": d3_stats,
        "d3_verdict": "price_fix_worthwhile" if d3_stats["immediate_switch_missed"] > 0 else "no_immediate_miss",
        "next_action": "proceed_P1_price_fix_and_V3_route" if pass_d1 else "stop_V3_long_route_investigate_P1_price_only",
        "integrity": {
            "test_labels_used": False,
            "leaderboard_feedback_used": False,
            "manual_prediction_edits": False,
        }
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  → summary.json")
    print(f"\n  ══ D1 GATE: {'PASS ✓' if pass_d1 else 'FAIL ✗ — 停止 V3 长程路线'} ══")


# ──────────────────────────────────────────────────

def main() -> None:
    print(f"OOF  : {OOF_PATH}")
    print(f"PRICE: {PRICE_PATH}")
    print(f"OUT  : {OUT_DIR}\n")

    print("加载 oof_with_routes.csv …")
    oof = pd.read_csv(OOF_PATH)
    print(f"  总行数: {len(oof)}, 列: {list(oof.columns)}\n")

    d1_df, _ = run_d1(oof)
    d3_stats  = run_d3()
    write_summary(d1_df, d3_stats)

    print(f"\n完成。输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
