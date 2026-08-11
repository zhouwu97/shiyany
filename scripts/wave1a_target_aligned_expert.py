"""Wave 1a: P1 — Target-Aligned Long-Horizon Cycle Expert（75-120，低 DF）。

纯粹基于 target-clock alignment：ŷ_{t+h} = y_t + median_{d=1,2,3,7}(y_{t+h-96d} - y_{t-96d})

四种预测器：
  anchor_median      — 4 日锚点中位数（无训练）
  anchor_wmedian     — 日均权重中位数（近者权大，无训练）
  horizon_ridge      — 每 horizon 独立 Ridge（5 特征：4 锚点 + current）
  horizon_huber      — 每 horizon 独立 Huber（同上，更抗噪声）

严格隔离：h120 模型只看到 h120 对齐特征，不跨 horizon 共享。
仅评估 g1 × {75,90,105,120}；gall 保持 SAFE60。

Gate（预注册，先于运行写入 run_meta）：
  - GO: standalone long-horizon MAPE < SAFE60 long-horizon MAPE
  - ELIF corr(res,res_SAFE60) ≤ 0.70:
    - standalone ≤ SAFE60 + 0.5pp → oracle headroom ≥ 0.10pp 为 GO
    - standalone > SAFE60 + 0.5pp → oracle headroom ≥ 0.15pp 为 GO（防便宜低相关）
  - ELSE STOP

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

from gas_forecast.data import align_tables  # noqa: E402
from gas_forecast.scoring import competition_mape  # noqa: E402

DATA_DIR = Path("data/raw/official/初赛-参赛者使用")
X3_REPLAY_OOF = Path("results/raw/runs/experiments/pred1_x3_replay_20260810/oof.csv")
LONG_HORIZONS = (75, 90, 105, 120)
CYCLE_DAYS = (1, 2, 3, 7)
W_SUM = 10.0  # 4+3+2+1


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p) / np.maximum(np.abs(a), 1e-6)))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """权重中位数：累积权重过半时取值。"""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cs = np.cumsum(w)
    idx = np.searchsorted(cs, cs[-1] / 2.0)
    return float(v[min(idx, len(v) - 1)])


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


def _build_anchors(raw_gen1: pd.Series, origins, horizons) -> pd.DataFrame:
    """对每个 origin×horizon 构造 4 日历史锚点 + delta。"""
    records = []
    h_steps = {h: h // 15 for h in horizons}
    gen1 = raw_gen1.copy()
    for origin in origins:
        if origin not in gen1.index:
            continue
        y_t = float(gen1.loc[origin])
        for h, step in h_steps.items():
            anchors = {}
            deltas = {}
            for d in CYCLE_DAYS:
                lag = 96 * d
                tgt_idx = origin + pd.to_timedelta(h, unit="min") - pd.to_timedelta(lag * 15, unit="min")
                cur_idx = origin - pd.to_timedelta(lag * 15, unit="min")
                if tgt_idx not in gen1.index or cur_idx not in gen1.index:
                    continue
                anc = float(gen1.loc[tgt_idx])
                cur = float(gen1.loc[cur_idx])
                anchors[f"anchor_{d}d"] = anc
                deltas[f"delta_{d}d"] = anc - cur
                deltas[f"pred_{d}d"] = y_t + anc - cur
            if len(anchors) < 4:
                continue
            rec = {"origin_time": origin, "horizon": h, "y_t": y_t}
            rec.update(anchors)
            rec.update(deltas)
            # median / wmedian
            preds = np.array([rec[f"pred_{d}d"] for d in CYCLE_DAYS])
            ws = np.array([4.0, 3.0, 2.0, 1.0])  # 1d 权重最大
            rec["anchor_median_pred"] = float(np.median(preds))
            rec["anchor_wmedian_pred"] = _weighted_median(preds, ws)
            records.append(rec)
    return pd.DataFrame.from_records(records).set_index(["origin_time", "horizon"])


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{stamp}_wave1a_target_aligned")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== [1/4] load data ===")
    dataset = align_tables(DATA_DIR)
    gen1 = pd.to_numeric(dataset.frame["generator_1"], errors="coerce").ffill(limit=8)

    m = pd.read_csv(X3_REPLAY_OOF, parse_dates=["origin_time", "train_end"])
    m["fold"] = m["fold"].astype(str)
    m["safe60_pred"] = 0.6 * m["x3_cat_mae_pred"] + 0.4 * m["a61_recursive_blend_05_pred"]
    fold_order = m.groupby("fold", sort=False)["origin_time"].min().sort_values().index.astype(str).tolist()
    recent5 = fold_order[-5:]

    # 只保留 g1 long horizon
    g1_long = m.loc[m["target"].eq("generator_1") & m["horizon"].isin(LONG_HORIZONS)].copy()
    print(f"g1 long-horizon rows: {len(g1_long)}, folds: {len(fold_order)}, origins: {g1_long['origin_time'].nunique()}")

    print("=== [2/4] build target-aligned anchors ===")
    origins = pd.DatetimeIndex(sorted(g1_long["origin_time"].unique()))
    anchors = _build_anchors(gen1, origins, LONG_HORIZONS)
    print(f"anchor rows: {len(anchors)}")

    joined = g1_long.set_index(["origin_time", "horizon"]).join(anchors, how="inner").reset_index()
    print(f"joined: {len(joined)}")

    # 无训练预测（anchor median/wmedian）— 纯公式
    for col in ("anchor_median_pred", "anchor_wmedian_pred"):
        joined[col] = joined[col].clip(0, 200)

    # 每 horizon 独立 Ridge / Huber（forward cross-fit）
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    ridge_feats = ["anchor_1d", "anchor_2d", "anchor_3d", "anchor_7d", "y_t"]
    for model_name, clf in (("horizon_ridge", Ridge(alpha=20.0)),
                             ("horizon_huber", HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000))):
        col = f"{model_name}_pred"
        joined[col] = float("nan")
        for h in LONG_HORIZONS:
            h_mask = joined["horizon"].eq(h)
            for position, fold in enumerate(fold_order):
                held = joined["fold"].eq(fold).to_numpy() & h_mask.to_numpy()
                if position == 0:
                    joined.loc[held, col] = joined.loc[held, "y_t"]  # 首折回退 persistence
                    continue
                train = joined["fold"].isin(fold_order[:position]).to_numpy() & h_mask.to_numpy()
                tr = joined.loc[train]
                he = joined.loc[held]
                Xtr = tr.loc[:, ridge_feats].apply(pd.to_numeric, errors="coerce")
                ytr = tr["actual"].to_numpy(float)
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                    ("model", clf),
                ])
                model.fit(Xtr, ytr)
                Xte = he.loc[:, ridge_feats].apply(pd.to_numeric, errors="coerce")
                joined.loc[held, col] = np.clip(model.predict(Xte), 0, 200)

    # 同时算 SAFE60 long-horizon MAPE
    safe60_lh = _mape(joined["actual"].to_numpy(float), joined["safe60_pred"].to_numpy(float))
    print(f"\nSAFE60 long-horizon g1 MAPE = {safe60_lh*100:.4f}%")

    print("\n=== [3/4] evaluate all experts ===")
    experts = {
        "anchor_median": "anchor_median_pred",
        "anchor_wmedian": "anchor_wmedian_pred",
        "horizon_ridge": "horizon_ridge_pred",
        "horizon_huber": "horizon_huber_pred",
    }

    for name, col in experts.items():
        mape_val = _mape(joined["actual"].to_numpy(float), joined[col].to_numpy(float))
        delta_pp = (mape_val - safe60_lh) * 100
        # residual corr
        res_c = joined["actual"] - joined[col]
        res_s = joined["actual"] - joined["safe60_pred"]
        pear = float(np.corrcoef(res_c, res_s)[0, 1])
        spear, _ = spearmanr(res_c, res_s)
        # oracle
        ape_e = joined["actual"].sub(joined[col]).abs().div(joined["actual"].abs().clip(lower=1e-6))
        ape_safe = joined["actual"].sub(joined["safe60_pred"]).abs().div(joined["actual"].abs().clip(lower=1e-6))
        per = pd.DataFrame({"EXPERT": ape_e, "SAFE60": ape_safe}).groupby(joined["origin_time"]).mean().sort_index()
        oracle_dwell4 = _dwell_oracle(per, 4)
        oracle_headroom = (safe60_lh - oracle_dwell4) * 100
        win_mask = per["EXPERT"] < per["SAFE60"]
        coverage = float(win_mask.mean())
        margin = float((per.loc[win_mask, "SAFE60"] - per.loc[win_mask, "EXPERT"]).mean()) * 100
        loss_margin = float((per.loc[~win_mask, "EXPERT"] - per.loc[~win_mask, "SAFE60"]).mean()) * 100
        print(f"\n-- {name} --")
        print(f"  standalone long-horizon g1 = {mape_val*100:.4f}% (Δ {delta_pp:+.2f}pp vs SAFE60)")
        print(f"  residual corr: pearson={pear:.4f} spearman={spear:.4f}")
        print(f"  dwell4 oracle headroom: {oracle_headroom:.4f}pp")
        print(f"  win coverage: {coverage:.3f}, win margin: {margin:.2f}pp, loss: {loss_margin:.2f}pp")

        # recent5 / per-horizon / per-fold corr
        r5 = joined.loc[joined["fold"].isin(recent5)]
        r5p, _ = spearmanr(r5["actual"] - r5[col], r5["actual"] - r5["safe60_pred"])
        print(f"  recent5 corr: pearson={float(np.corrcoef(r5['actual']-r5[col], r5['actual']-r5['safe60_pred'])[0,1]):.4f}")

    best_idx = min(experts, key=lambda k: _mape(joined["actual"].to_numpy(float), joined[experts[k]].to_numpy(float)))
    best_col = experts[best_idx]
    best_mape = _mape(joined["actual"].to_numpy(float), joined[best_col].to_numpy(float))
    best_delta = (best_mape - safe60_lh) * 100
    # gate
    best_pear = float(np.corrcoef(joined["actual"]-joined[best_col], joined["actual"]-joined["safe60_pred"])[0,1])
    best_oracle_headroom = safe60_lh - _dwell_oracle(
        pd.DataFrame({"E": joined["actual"].sub(joined[best_col]).abs().div(joined["actual"].abs().clip(lower=1e-6)),
                      "S": joined["actual"].sub(joined["safe60_pred"]).abs().div(joined["actual"].abs().clip(lower=1e-6))}
        ).groupby(joined["origin_time"]).mean().sort_index(), 4)
    best_oracle_headroom = (best_oracle_headroom) * 100

    cheap_thresh = 0.15 if best_delta > 0.5 else 0.10
    go = (best_delta < 0) or (best_pear <= 0.70 and best_oracle_headroom >= cheap_thresh)
    verdict = "GO" if go else "STOP"

    print(f"\n=== [4/4] Decision ===")
    print(f"best: {best_idx} | MAPE={best_mape*100:.4f}% (Δ {best_delta:+.2f}pp) | "
          f"corr={best_pear:.4f} | dwell4 headroom={best_oracle_headroom:.4f}pp")
    print(f"gate: delta<0={best_delta<0}, corr≤0.70={best_pear<=0.70}, "
          f"headroom≥{cheap_thresh:.2f}={best_oracle_headroom>=cheap_thresh}")
    print(f"verdict: {verdict}")

    joined.to_csv(run_dir / "p1_predictions.csv", index=False, encoding="utf-8")
    run_meta = {
        "run": "wave1a_target_aligned_long_horizon_75_120",
        "stamp": stamp,
        "date": datetime.now().isoformat(),
        "python": sys.executable,
        "baseline": "SAFE60 on g1 × {75,90,105,120}",
        "config": {"cycle_days": list(CYCLE_DAYS), "horizons": list(LONG_HORIZONS),
                   "ridge_features": ridge_feats, "weights_wmedian": [4, 3, 2, 1]},
        "best_expert": best_idx,
        "best_mape_pp": best_mape * 100,
        "best_delta_pp": best_delta,
        "best_residual_corr_pearson": best_pear,
        "best_dwell4_oracle_headroom_pp": best_oracle_headroom,
        "pre_registered_gates": {
            "go_if_standalone_beats_safe60": "delta_pp < 0",
            "go_if_corr_le_0.70_and_headroom_ge_0.10pp": "if delta_pp ≤ 0.5pp",
            "go_if_corr_le_0.70_and_headroom_ge_0.15pp": "if delta_pp > 0.5pp (防便宜低相关)",
        },
        "verdict": verdict,
        "status": "complete",
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
