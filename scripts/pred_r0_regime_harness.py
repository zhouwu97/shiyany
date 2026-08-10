"""PRED-R0 Causal Regime Transfer Harness(v4.1 冻结版)。

训练期 regime 审计 + 三口径评估。**合规边界:** regime 阈值与密度模型只在
``< dev_01`` 训练数据上拟合,评估 held fold 只 transform;绝不读测试协变量
画像或平台反馈做任何选择。

阶段分离:
  Phase 1 —— 冻结 regime 定义 + 阈值 + Metric C 密度权重(不加载任何候选误差)
  Phase 2 —— 载入候选 OOF,输出三口径 + ESS + matched-days + bootstrap

三口径:
  Metric A  full dev pooled (robustness floor)
  Metric B  hard regime match (HIGH_STABLE 子集,解释用)
  Metric C  regime continuous-weighted MAPE (主指标)
            w(origin) = p_hat(HIGH_STABLE | X), 密度模型 fit 在 < dev_01

用法:
  python scripts/pred_r0_regime_harness.py \
    --data-dir "data/raw/official/初赛-参赛者使用" \
    --input <gate_merged_oof.csv> \
    --columns safe60_pred,a61_pred,aggressive_r75_lgb20_pred \
    --cutoff 2025-03-20 \
    --run-dir <run_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- 冻结参数(与 PLAN v4.1 §10 决策 #1 一致,决策后不得改动) ---
GAS_QUANTILES: tuple[float, float, float] = (0.75, 0.50, 0.25)
RAMP_QUANTILES: tuple[float, float] = (0.25, 0.75)
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260810
DENSITY_FEATURES = ["bf_total", "bf_ramp_max", "bf_slope8", "bf_missing4"]
RAMP_WINDOW = 8

# 时序信赖门禁(v4.1 §8.1,决策 #5)
MATCHED_DAYS_MIN = 8
DAY_ESS_MIN = 5.0
CELL_ESS_COVERAGE = 200.0


def _sha256(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _load_gas(data_dir: Path) -> pd.DataFrame:
    """从训练输入构建 origin 级协变量(BF 总气 level/ramp/slope/missing)。"""
    gas = pd.read_csv(data_dir / "Pre_gas.csv", parse_dates=["datetime"])
    bf_cols = [c for c in gas.columns if c.startswith("blast_furnace_")]
    b = gas[bf_cols].fillna(0).sum(axis=1)
    out = pd.DataFrame({"origin_time": gas["datetime"], "bf_total": b.values})
    out["bf_slope8"] = b.diff(RAMP_WINDOW).values / np.maximum(b.shift(RAMP_WINDOW), 1).values
    out["bf_ramp_max"] = b.diff().rolling(RAMP_WINDOW).max().values / np.maximum(b, 1).values
    out["bf_missing4"] = gas[bf_cols].isna().rolling(RAMP_WINDOW, min_periods=1).mean().mean(axis=1).values
    return out


def _freeze_regime(gas: pd.DataFrame, cutoff: str) -> dict[str, object]:
    """Phase 1: 只在 < cutoff 数据上拟合阈值;返回冻结 spec + 全期 regime 标签。"""
    pre = gas[gas["origin_time"] < pd.Timestamp(cutoff)]
    if len(pre) < 200:
        raise ValueError(f"< cutoff 训练样本不足: {len(pre)}")
    gas_hi, gas_mid, gas_lo = pre["bf_total"].quantile(list(GAS_QUANTILES)).tolist()
    ramp_lo, ramp_hi = pre["bf_ramp_max"].quantile(list(RAMP_QUANTILES)).tolist()

    def level(x):
        if x >= gas_hi:
            return "HIGH"
        if x <= gas_lo:
            return "LOW"
        return "MID"

    def ramp(x):
        if x <= ramp_lo:
            return "STABLE"
        if x >= ramp_hi:
            return "RAMP"
        return "NORMAL"

    gas["gas_level"] = gas["bf_total"].map(level)
    gas["ramp_state"] = gas["bf_ramp_max"].map(ramp)
    gas["regime"] = np.where(
        (gas["gas_level"] == "HIGH") & (gas["ramp_state"] == "STABLE"),
        "HIGH_STABLE",
        "OTHER",
    )

    # Metric C density: 在 < cutoff 上 fit P(HIGH_STABLE | X)
    pre_lab = pre["bf_total"] >= gas_hi
    pre_stab = pre["bf_ramp_max"] <= ramp_lo
    y = (pre_lab & pre_stab).astype(int)
    X = pre[DENSITY_FEATURES].fillna(0).to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    logit = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X), y)
    weights = logit.predict_proba(scaler.transform(gas[DENSITY_FEATURES].fillna(0)))[:, 1]
    gas["hs_weight"] = weights

    spec = {
        "cutoff": str(cutoff),
        "fit_rows": int(len(pre)),
        "gas_quantiles": list(GAS_QUANTILES),
        "ramp_quantiles": list(RAMP_QUANTILES),
        "thresholds": {
            "bf_total_hi": float(gas_hi),
            "bf_total_mid": float(gas_mid),
            "bf_total_lo": float(gas_lo),
            "bf_ramp_lo": float(ramp_lo),
            "bf_ramp_hi": float(ramp_hi),
        },
        "density_features": DENSITY_FEATURES,
        "regime_label": "HIGH_STABLE = HIGH gas & STABLE ramp",
        "fit_spec_sha256": None,
    }
    spec["fit_spec_sha256"] = _sha256(spec)
    return {"spec": spec, "gas": gas}


def _ess(weights: np.ndarray) -> float:
    s = weights.sum()
    return float(s * s / np.maximum((weights * weights).sum(), 1e-12))


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred) / np.maximum(np.abs(actual), 1e-6)))


def _weighted_mape(actual: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    num = float(np.sum(w * np.abs(actual - pred) / np.maximum(np.abs(actual), 1e-6)))
    den = float(np.sum(w))
    return num / den


def _evaluate(rows: pd.DataFrame, gas: pd.DataFrame, spec: dict[str, object], cols: list[str]) -> dict[str, object]:
    work = rows.copy()
    work["origin_time"] = pd.to_datetime(work["origin_time"])
    work = work.merge(gas[["origin_time", "regime", "hs_weight"]], on="origin_time", how="left")
    work["_day"] = work["origin_time"].dt.floor("D")
    if work["regime"].isna().any():
        raise ValueError("OOF 中存在无法匹配到 regime 的 origin_time(训练输入未覆盖?)")

    result: dict[str, object] = {"regime_spec": spec, "report_sha256": None}
    for col in cols:
        m = work["actual"].notna() & work[col].notna()
        a = work.loc[m, "actual"].to_numpy(dtype=float)
        p = work.loc[m, col].to_numpy(dtype=float)
        w = work.loc[m, "hs_weight"].to_numpy(dtype=float)
        hs = work.loc[m, "regime"].eq("HIGH_STABLE").to_numpy(dtype=bool)

        matched_days = int(work.loc[m & hs, "_day"].nunique())
        day_ess = _ess(
            work.loc[m, "hs_weight"].groupby(work.loc[m, "_day"]).sum().to_numpy(dtype=float)
        )
        origin_ess = _ess(work.loc[m, "hs_weight"].groupby(work.loc[m, "origin_time"]).sum().to_numpy(dtype=float))
        cell_ess = _ess(w)

        entry = {
            "full_dev_mape": round(_mape(a, p) * 100, 6),
            "regime_hard_mape": round(_mape(a[hs], p[hs]) * 100, 6),
            "regime_weighted_mape": round(_weighted_mape(a, p, w) * 100, 6),
            "matched_cells": int(hs.sum()),
            "matched_origins": int(work.loc[m & hs, "origin_time"].nunique()),
            "matched_days": matched_days,
            "cell_ess": round(cell_ess, 2),
            "origin_ess": round(origin_ess, 2),
            "day_ess": round(day_ess, 2),
            "temporal_support": "OK" if (matched_days >= MATCHED_DAYS_MIN and day_ess >= DAY_ESS_MIN) else "INSUFFICIENT_TEMPORAL_SUPPORT",
        }
        result[col] = entry
    result["report_sha256"] = _sha256({k: v for k, v in result.items() if k != "report_sha256"})
    return result


def _bootstrap_weighted(
    rows: pd.DataFrame, gas: pd.DataFrame, cand_col: str, base_col: str
) -> dict[str, object]:
    work = rows.copy()
    work["origin_time"] = pd.to_datetime(work["origin_time"])
    work = work.merge(gas[["origin_time", "hs_weight"]], on="origin_time", how="left")
    work["_day"] = work["origin_time"].dt.floor("D")
    days = pd.DatetimeIndex(sorted(work["_day"].unique()))
    day_rows = {d: g for d, g in work.groupby("_day", sort=True)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def weighted_improve(idx: np.ndarray) -> float:
        blocks = pd.concat([day_rows[d] for d in days[idx]])
        w = blocks["hs_weight"].to_numpy(dtype=float)
        a = blocks["actual"].to_numpy(dtype=float)
        base = _weighted_mape(a, blocks[base_col].to_numpy(dtype=float), w)
        cand = _weighted_mape(a, blocks[cand_col].to_numpy(dtype=float), w)
        return base - cand

    vals = np.empty(BOOTSTRAP_SAMPLES)
    for r in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, days.size, size=days.size)
        vals[r] = weighted_improve(idx)
    return {
        "block": "day",
        "blocks": int(days.size),
        "samples": int(BOOTSTRAP_SAMPLES),
        "random_seed": BOOTSTRAP_SEED,
        "mean_improvement_pp": float(vals.mean() * 100.0),
        "median_improvement_pp": float(np.median(vals) * 100.0),
        "ci95_low_pp": float(np.quantile(vals, 0.025) * 100.0),
        "ci95_high_pp": float(np.quantile(vals, 0.975) * 100.0),
        "probability_candidate_better": float(np.mean(vals > 0.0)),
        "observed_improvement_pp": None,  # filled by caller with weighted deltas
    }


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--columns", required=True, help="候选预测列,逗号分隔")
    parser.add_argument("--cutoff", default="2025-03-20", help="dev_01 起始,阈值只在其前拟合")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cols = [c.strip() for c in args.columns.split(",")]
    rows = _read_input(args.input)
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        raise ValueError(f"输入缺少候选列: {missing}")

    # Phase 1: 冻结 regime,不加载候选误差
    gas = _load_gas(args.data_dir)
    phase1 = _freeze_regime(gas, args.cutoff)
    spec = phase1["spec"]

    # Phase 2: 评估
    report = _evaluate(rows, phase1["gas"], spec, cols)

    # bootstrap: 主候选 vs 每个 baseline 的 regime-weighted 比较
    base_cols = [c for c in cols if c != cols[0]]
    report["bootstrap_weighted_vs"] = {}
    for b in base_cols:
        report["bootstrap_weighted_vs"][b] = _bootstrap_weighted(rows, phase1["gas"], cols[0], b)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "regime_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
