"""Wave 0b: P0 — Physical × SAFE60 重锚 audit（因果）。

把 physical_rest 路径重锚到 SAFE60-era OOF（X3 replay OOF 为基，safe60=0.6X3+0.4A61，
含 current_value）。旧 physical 用 `c0_pred` 锚，这次必须用 SAFE60 锚。

输出四组数字：
1) residual corr(res_safe60, res_physical): Pearson+Spearman, overall/by horizon/by fold/by recent5
2) physical standalone MAPE（g1=gall_safe60-rest_pred，gall=SAFE60）
3) SAFE60+physical 两专家 origin oracle + dwell4(60min) oracle + win coverage × margin
4) **causal-selective gate**（决定性数字）：更早折拟合的 origin 级 gate，
   只用 origin 可得特征 + 已知电价，在 held 上选择 physical 是否赢。不比 hindsight oracle。

预注册 gate（写进 run_meta，先于运行）：
- GO 候选需同时满足：
  a) pooled residual corr (Pearson) ≤ 0.70；
  b) physical g1 standalone 不差于 SAFE60 g1 超 0.5pp（防"弱预测器买来的便宜低相关"）。
- 条件价值需满足任一：
  c) SAFE60+physical dwell4(60min) origin oracle headroom ≥ 0.08pp；
  d) causal-selective gate 在 g1 上 ≥ 0.05pp 增益且 coverage ≥ 10%，recent folds 不反转。
- 若因果 gate 失败（哪怕 oracle 大）→ 诚实负结果，动态/物理线保持关闭。

产物：results/runs/<stamp>/。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from scipy.stats import spearmanr  # noqa: E402

from gas_forecast.config import FeatureConfig  # noqa: E402
from gas_forecast.data import align_tables  # noqa: E402
from gas_forecast.features import build_causal_features, load_price_schedule  # noqa: E402
from gas_forecast.physical_rest import time_ordered_physical_rest_oof  # noqa: E402
from gas_forecast.scoring import competition_mape  # noqa: E402

DATA_DIR = Path("data/raw/official/初赛-参赛者使用")
X3_REPLAY_OOF = Path("results/raw/runs/experiments/pred1_x3_replay_20260810/oof.csv")
HORIZONS = (15, 30, 45, 60, 75, 90, 105, 120)

# 预注册的 physical feature 集合（origin 级，全部历史+已知电价构造）
BASE_PREFIXES = (
    "feat_generator_rest_",
    "feat_generator_1_lag_",
    "feat_generator_1_diff_",
    "feat_generator_1_slope_",
    "feat_generator_1_std_",
    "feat_generator_1_mean_",
)
BASE_DIRECT = {
    "feat_generator_gas_total",
    "feat_bf_production",
    "feat_bf_user_use",
    "feat_air_heater_use",
    "feat_bf_surplus_proxy",
    "feat_target_price_tplus_15",
    "feat_target_price_tplus_30",
    "feat_target_price_tplus_45",
    "feat_target_price_tplus_60",
    "feat_target_price_tplus_75",
    "feat_target_price_tplus_90",
    "feat_target_price_tplus_105",
    "feat_target_price_tplus_120",
    "feat_price_switch_within_120",
    "feat_steps_to_price_switch",
}


def _build_origin_features() -> pd.DataFrame:
    """构建 origin 级 physical feature 帧（DatetimeIndex, 15min grid）。"""
    dataset = align_tables(DATA_DIR)
    frame = dataset.frame
    config = FeatureConfig(
        enable_anomaly_features=False,
        enable_physical_features=True,
        enable_long_cycle_features=True,
        enable_target_aligned_features=False,
        dynamic_feature_scope="core",
    )
    feat = build_causal_features(frame, config, price_schedule=load_price_schedule(DATA_DIR / "price.xlsx"))

    cols = [c for c in feat.columns if c in BASE_DIRECT or any(c.startswith(p) for p in BASE_PREFIXES)]
    out = feat.loc[:, cols].copy()

    # holder：从 blast_furnace_gas_holder_2 显式构造
    holder = pd.to_numeric(frame["blast_furnace_gas_holder_2"], errors="coerce").ffill(limit=8)
    out["feat_gas_holder"] = holder
    for lag in (1, 2, 4, 8, 16):
        out[f"feat_gas_holder_lag_{lag}"] = holder.shift(lag)
    for lag in (1, 2, 4):
        out[f"feat_gas_holder_diff_{lag}"] = holder - holder.shift(lag)
    for win in (4, 8, 16):
        out[f"feat_gas_holder_slope_{win}"] = (holder - holder.shift(win - 1)) / float(win - 1)

    # gas balance：可用气 = 高炉产量 - 厂内消费 - 发电消费（origin 可见近似）
    surplus = out["feat_bf_surplus_proxy"]
    gen_gas = out["feat_generator_gas_total"]
    out["feat_gas_balance"] = surplus - gen_gas

    # price：current + future + delta（已知电价，合法未来信息）
    price = load_price_schedule(DATA_DIR / "price.xlsx")
    idx = frame.index
    out["current_price"] = price.lookup(idx)
    for h in HORIZONS:
        target_idx = idx + pd.to_timedelta(h, unit="min")
        fp = np.asarray(price.lookup(target_idx), dtype=float)
        out[f"future_price_{h}"] = fp
        out[f"price_delta_{h}"] = fp - out["current_price"]
    return out


def _assemble_physical_frame() -> pd.DataFrame:
    """把 origin 级 features 合并到 SAFE60-era OOF 上，构造 physical 输入。"""
    m = pd.read_csv(X3_REPLAY_OOF, parse_dates=["origin_time", "train_end"])
    m["fold"] = m["fold"].astype(str)
    m["safe60_pred"] = 0.6 * m["x3_cat_mae_pred"] + 0.4 * m["a61_recursive_blend_05_pred"]
    origin_feat = _build_origin_features()
    missing = sorted(set(m["origin_time"]) - set(origin_feat.index))
    if missing:
        raise ValueError(f"{len(missing)} 个 OOF origin 不在 feature grid 上: {missing[:5]}")
    merged = m.merge(origin_feat, left_on="origin_time", right_index=True, how="left", validate="many_to_one")
    return merged


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p) / np.maximum(np.abs(a), 1e-6)))


def _dwell_oracle(ape_frame: pd.DataFrame, min_dwell: int) -> float:
    origins = list(ape_frame.index)
    n = len(origins)
    loss_sum = 0.0
    count = 0
    idx = 0
    while idx < n:
        block = ape_frame.iloc[idx:min(idx + min_dwell, n)]
        winner = block.mean(axis=0).idxmin()
        loss_sum += float(block[winner].sum())
        count += block.shape[0]
        idx += min_dwell
    return loss_sum / count


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{stamp}_wave0b_p0_physical")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== [1/5] assemble physical frame ===")
    rows = _assemble_physical_frame()
    print(f"rows: {len(rows)}, folds: {rows['fold'].nunique()}, origins: {rows['origin_time'].nunique()}")

    print("=== [2/5] run causal physical OOF (anchored on SAFE60) ===")
    physical, report = time_ordered_physical_rest_oof(
        rows, baseline_column="safe60_pred"
    )
    # physical 输出是 pivoted 帧（一列 per target）：每行 = (fold, origin, horizon)
    physical["physical_g1_pred"] = physical["x1_indirect_g1_pred"]
    physical.to_csv(run_dir / "physical_oof.csv", index=False, encoding="utf-8")

    print("=== [3/5] residual correlation ===")
    g1 = physical
    res_safe60 = (g1["actual_generator_1"] - g1["direct_g1_pred"]).to_numpy(float)
    res_phys = (g1["actual_generator_1"] - g1["physical_g1_pred"]).to_numpy(float)

    def corr_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        pear = float(np.corrcoef(a, b)[0, 1]) if len(a) >= 2 else float("nan")
        spear, _ = spearmanr(a, b)
        return pear, float(spear)

    corr_rows: list[dict[str, object]] = []
    p, s = corr_pair(res_safe60, res_phys)
    corr_rows.append({"scope": "g1_overall", "value": "all", "pearson": p, "spearman": s})
    for horizon, part in g1.groupby("horizon", sort=True):
        a = (part["actual_generator_1"] - part["direct_g1_pred"]).to_numpy(float)
        b = (part["actual_generator_1"] - part["physical_g1_pred"]).to_numpy(float)
        pp, ss = corr_pair(a, b)
        corr_rows.append({"scope": "g1_horizon", "value": str(horizon), "pearson": pp, "spearman": ss})
    fold_order = g1.groupby("fold", sort=False)["origin_time"].min().sort_values().index.astype(str).tolist()
    for fold in fold_order:
        part = g1.loc[g1["fold"].eq(fold)]
        a = (part["actual_generator_1"] - part["direct_g1_pred"]).to_numpy(float)
        b = (part["actual_generator_1"] - part["physical_g1_pred"]).to_numpy(float)
        pp, ss = corr_pair(a, b)
        corr_rows.append({"scope": "g1_fold", "value": fold, "pearson": pp, "spearman": ss})
    recent5 = fold_order[-5:]
    r5 = g1.loc[g1["fold"].isin(recent5)]
    a = (r5["actual_generator_1"] - r5["direct_g1_pred"]).to_numpy(float)
    b = (r5["actual_generator_1"] - r5["physical_g1_pred"]).to_numpy(float)
    pp, ss = corr_pair(a, b)
    corr_rows.append({"scope": "g1_recent5", "value": "recent5", "pearson": pp, "spearman": ss})
    corr_df = pd.DataFrame.from_records(corr_rows)
    corr_df.to_csv(run_dir / "residual_correlation.csv", index=False, encoding="utf-8")
    overall = corr_df.loc[corr_df["scope"].eq("g1_overall")].iloc[0]
    print(f"g1 overall corr: pearson={overall['pearson']:.4f} spearman={overall['spearman']:.4f}")

    print("=== [4/5] standalone + two-expert oracle ===")
    safe60_g1 = _mape(g1["actual_generator_1"].to_numpy(float), g1["direct_g1_pred"].to_numpy(float))
    phys_g1 = _mape(g1["actual_generator_1"].to_numpy(float), g1["physical_g1_pred"].to_numpy(float))
    safe60_gall = _mape(g1["actual_generator_all"].to_numpy(float), g1["gall_c0_pred"].to_numpy(float))
    # pooled = 两 target 全部 cell 的 mean APE（同 competition_mape 定义）
    safe60_pooled = (safe60_g1 + safe60_gall) / 2.0
    phys_pooled = (phys_g1 + safe60_gall) / 2.0
    print(f"g1   SAFE60={safe60_g1*100:.4f}%  physical={phys_g1*100:.4f}%  (delta {(phys_g1-safe60_g1)*100:+.4f}pp)")
    print(f"gall SAFE60={safe60_gall*100:.4f}%")
    print(f"pooled SAFE60={safe60_pooled*100:.4f}%  physical={phys_pooled*100:.4f}%")

    # 两专家 origin oracle（g1，按 origin 聚合 APE）
    ape_s = g1["actual_generator_1"].sub(g1["direct_g1_pred"]).abs().div(g1["actual_generator_1"].abs().clip(lower=1e-6))
    ape_p = g1["actual_generator_1"].sub(g1["physical_g1_pred"]).abs().div(g1["actual_generator_1"].abs().clip(lower=1e-6))
    per_origin = pd.DataFrame({"SAFE60": ape_s, "PHYS": ape_p}).groupby(g1["origin_time"]).mean().sort_index()
    oracle_raw = per_origin.min(axis=1).mean()
    oracle_dwell4 = _dwell_oracle(per_origin, 4)
    win = per_origin["PHYS"] < per_origin["SAFE60"]
    coverage = float(win.mean())
    margin = float((per_origin.loc[win, "SAFE60"] - per_origin.loc[win, "PHYS"]).mean())
    loss_where_phys_loses = float((per_origin.loc[~win, "PHYS"] - per_origin.loc[~win, "SAFE60"]).mean())
    oracle = {
        "g1_safe60_ape": safe60_g1 * 100,
        "g1_phys_ape": phys_g1 * 100,
        "g1_origin_oracle_raw": oracle_raw * 100,
        "g1_origin_oracle_dwell4_60min": oracle_dwell4 * 100,
        "g1_origin_oracle_headroom_vs_safe60_dwell4": (safe60_g1 - oracle_dwell4) * 100,
        "phys_win_coverage": coverage,
        "phys_win_mean_margin_pp": margin * 100,
        "phys_loss_mean_pp": loss_where_phys_loses * 100,
    }
    print("oracle:", {k: round(v, 4) for k, v in oracle.items()})

    print("=== [5/5] causal-selective gate (决定性) ===")
    # origin 级 gate：label=physical 赢，features=origin 可得（4 态概率 + 物理特征）
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    state_cols = ["prob_state_0", "prob_state_1", "prob_state_2", "prob_transition"]
    gate_feats = state_cols + [
        "rest_current", "feat_gas_holder", "feat_gas_balance",
        "feat_generator_gas_total", "feat_price_switch_within_120", "feat_steps_to_price_switch",
        "current_price", "feat_generator_rest_same_slot_median_7d",
    ]
    gate_feats = [c for c in gate_feats if c in g1.columns]
    g1_work = g1.copy()
    g1_work["safe60_ape"] = g1_work["actual_generator_1"].sub(g1_work["direct_g1_pred"]).abs().div(
        g1_work["actual_generator_1"].abs().clip(lower=1e-6))
    g1_work["phys_ape"] = g1_work["actual_generator_1"].sub(g1_work["physical_g1_pred"]).abs().div(
        g1_work["actual_generator_1"].abs().clip(lower=1e-6))
    agg = {c: ("mean" if c.endswith("_ape") else "first") for c in ["safe60_ape", "phys_ape"] + gate_feats}
    per = g1_work.groupby("origin_time", sort=True).agg(
        fold=("fold", "first"),
        safe60_ape=("safe60_ape", "mean"),
        phys_ape=("phys_ape", "mean"),
        **{c: (c, "first") for c in gate_feats},
    )
    per["label"] = (per["phys_ape"] < per["safe60_ape"]).astype(int)

    causal_pred = pd.Series(index=per.index, dtype=float)
    gate_trajectory = []
    for position, fold in enumerate(fold_order):
        held = per["fold"].eq(fold)
        if position == 0:
            causal_pred.loc[per.index[held]] = 0.0  # 首折回退 SAFE60
            gate_trajectory.append({"fold": fold, "fallback": True})
            continue
        history = per.loc[per["fold"].isin(fold_order[:position])]
        Xtr = history.loc[:, gate_feats].apply(pd.to_numeric, errors="coerce")
        ytr = history["label"].to_numpy(int)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.2, class_weight="balanced", max_iter=1500)),
        ])
        model.fit(Xtr, ytr)
        Xte = per.loc[held, gate_feats].apply(pd.to_numeric, errors="coerce")
        causal_pred.loc[per.index[held]] = model.predict_proba(Xte)[:, 1]
        sel = causal_pred.loc[per.index[held]] > 0.5
        gate_trajectory.append({
            "fold": fold,
            "covered": float(sel.mean()),
            "train_rows": int(len(Xtr)),
        })

    selected_phys = causal_pred > 0.5
    per["causal_pred"] = causal_pred
    per["selected_phys"] = selected_phys
    causal_ape = np.where(selected_phys, per["phys_ape"], per["safe60_ape"])
    causal_g1 = float(np.mean(causal_ape))
    causal_coverage = float(selected_phys.mean())
    causal = {
        "g1_safe60_ape": safe60_g1 * 100,
        "g1_causal_selective_ape": causal_g1 * 100,
        "causal_gain_vs_safe60_pp": (safe60_g1 - causal_g1) * 100,
        "causal_coverage": causal_coverage,
        "recent5_gain": None,
    }
    # recent5 反转检查
    r5_mask = per["fold"].isin(recent5)
    if r5_mask.any():
        r5_sel = per.loc[r5_mask, "selected_phys"]
        r5_safe = per.loc[r5_mask, "safe60_ape"].to_numpy(float)
        r5_phys = per.loc[r5_mask, "phys_ape"].to_numpy(float)
        r5_ape = np.where(r5_sel, r5_phys, r5_safe)
        causal["recent5_gain"] = (float(np.mean(r5_safe)) - float(np.mean(r5_ape))) * 100
    print("causal-selective:", {k: round(v, 4) if isinstance(v, float) else v for k, v in causal.items()})

    # 预注册判定
    corr_ok = float(overall["pearson"]) <= 0.70
    standalone_ok = (phys_g1 - safe60_g1) * 100 <= 0.5
    oracle_ok = oracle["g1_origin_oracle_headroom_vs_safe60_dwell4"] >= 0.08
    causal_ok = causal["causal_gain_vs_safe60_pp"] >= 0.05 and causal_coverage >= 0.10 and (causal["recent5_gain"] or 0) >= 0.0
    decision = {
        "go_candidate": bool(corr_ok and standalone_ok),
        "checks": {
            "corr_le_0.70": corr_ok,
            "standalone_g1_within_0.5pp": standalone_ok,
            "dwell4_oracle_headroom_ge_0.08pp": oracle_ok,
            "causal_gate_gain_ge_0.05pp_and_coverage_and_recent5_stable": causal_ok,
        },
        "verdict": "GO_IF_causal_ok" if (corr_ok and standalone_ok and causal_ok) else
                   ("LOW_CORR_BUT_NO_CAUSAL_UTILITY" if corr_ok and standalone_ok else "STOP"),
    }

    report_out = {
        "experiment": "wave0b_p0_physical_safe60_audit",
        "stamp": stamp,
        "baseline": "SAFE60",
        "correlation": corr_rows,
        "oracle": oracle,
        "causal_selective": causal,
        "physical_rest_report": {k: v for k, v in report.items() if k in ("feature_columns", "state_semantics")},
        "decision": decision,
    }
    (run_dir / "report.json").write_text(json.dumps(report_out, ensure_ascii=False, indent=2), encoding="utf-8")
    per.reset_index().to_csv(run_dir / "origin_gate_table.csv", index=False, encoding="utf-8")

    run_meta = {
        "run": "wave0b_p0_physical_safe60_audit",
        "stamp": stamp,
        "date": datetime.now().isoformat(),
        "python": sys.executable,
        "baseline": "SAFE60 = 0.6*X3 + 0.4*A61 (from X3 replay OOF)",
        "physical_anchor": "time_ordered_physical_rest_oof(baseline_column=safe60_pred), strictly earlier folds only",
        "feature_columns": sorted(gate_feats),
        "pre_registered_gates": decision["checks"],
        "caveats": [
            "state labels in _physical_labels use future rest_actual -> regime NOT origin-identifiable; causal gate is the decisive test, not oracle",
            "causal-selective gate: LogisticRegression fit on strictly earlier folds, origin-level, features=4 state probs + physical + price",
        ],
        "status": "complete",
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== DECISION: {decision['verdict']} ===")
    print("checks:", {k: v for k, v in decision["checks"].items()})
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
