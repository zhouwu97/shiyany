"""Wave 1b: 价格 hazard 消融 —— 已知未来电价能否改善物理状态 transition 预测。

三组独立物理 transition label（均从 future actual 定义，只用于监督标签，不泄漏到特征）：
  1. rest_transition: |future_rest_delta| >= 20 MW
  2. holder_slope_flip: current holder slope 方向反转
  3. avail_contraction: 可用气在未来 horizon 内显著缩减

对每组 label，两个分类器（forward cross-fit）：
  A (state only): origin 物理状态
  B (state + price): A + 已知未来电价 schedule（唯一合法未来信息通道）

比较 AUC / logloss Δ。Gate：
  - 若三组 labels 的 ΔAUC 均 < 0.02 且 Δlogloss 均 > −0.01
    → 价格对物理 transition 预测无增益 → 价格线永久关闭。
  - 否则 → 价格有独立信息价值，允许进入 MoE 作为 hazard augmentation。

这是决定性消融：因为已知电价是规则下唯一合法的未来信息通道，
如果在物理 transition 这一核心 state 识别任务上都不增加预测力，
整个价格线不应再被任何新包装重新开门。

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

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, log_loss  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from gas_forecast.data import align_tables  # noqa: E402
from gas_forecast.features import load_price_schedule  # noqa: E402

DATA_DIR = Path("data/raw/official/初赛-参赛者使用")
HORIZONS = (15, 30, 45, 60, 75, 90, 105, 120)

STATE_FEATURES = [
    "holder", "holder_slope_4", "rest_current", "avail_B", "generator_gas",
    "rest_same_slot_7d", "current_price",
]
PRICE_FEATURES = [
    "price_switch_120", "steps_to_price_switch", "n_switches_120",
    "first_switch_magnitude", "price_range_120", "price_trend_60",
]


def _physical_series(frame: pd.DataFrame):
    f = frame.ffill(limit=8)
    holder = pd.to_numeric(f["blast_furnace_gas_holder_2"], errors="coerce")
    prod = sum(pd.to_numeric(f[c], errors="coerce") for c in
               ("blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5"))
    nongen = (
        sum(pd.to_numeric(f[c], errors="coerce") for c in
            ("air_heater_1", "air_heater_2", "air_heater_4", "air_heater_5"))
        + sum(pd.to_numeric(f[c], errors="coerce") for c in
              ("blast_furnace_user1", "blast_furnace_user2", "blast_furnace_user3", "blast_furnace_user4"))
        + pd.to_numeric(f["into_gas_mixed_blast_furnace"], errors="coerce")
    )
    gen_gas = sum(pd.to_numeric(f[c], errors="coerce") for c in
                  ("generator_use_blast_furnace_gas", "generator_use_coke_gas", "generator_use_converter_gas"))
    avail = prod - nongen
    rest = pd.to_numeric(f["generator_all"], errors="coerce") - pd.to_numeric(f["generator_1"], errors="coerce")
    return pd.DataFrame({"holder": holder, "prod": prod, "avail_B": avail, "generator_gas": gen_gas, "rest": rest},
                        index=f.index)


def _build_labels_and_features(raw, price, origins) -> pd.DataFrame:
    phy = _physical_series(raw)
    state = pd.DataFrame(index=raw.index)
    state["holder"] = phy["holder"]
    state["holder_slope_4"] = phy["holder"] - phy["holder"].shift(3)
    state["rest_current"] = phy["rest"]
    state["avail_B"] = phy["avail_B"]
    state["generator_gas"] = phy["generator_gas"]
    state["rest_same_slot_7d"] = phy["rest"].shift(96 * 7)
    state["current_price"] = price.lookup(raw.index)

    # price features
    pm = np.column_stack([price.lookup(raw.index + pd.to_timedelta(h, unit="m")) for h in HORIZONS])
    cur_p = state["current_price"].to_numpy(float)
    changed = ~np.isclose(pm, cur_p[:, None], rtol=0.0, atol=1e-12)
    state["price_switch_120"] = changed.any(axis=1).astype("int8")
    first_step = np.where(changed.any(axis=1), changed.argmax(axis=1) + 1, 0)
    state["steps_to_price_switch"] = first_step.astype("int8")
    state["n_switches_120"] = changed.sum(axis=1).astype("int8")
    fp = pm[np.arange(len(pm)), np.maximum(first_step - 1, 0)]
    state["first_switch_magnitude"] = np.where(first_step > 0, fp - cur_p, 0.0)
    state["price_range_120"] = pm.max(axis=1) - pm.min(axis=1)
    pm60 = pm[:, :4]
    state["price_trend_60"] = np.where(first_step > 0, pm60.max(axis=1) - pm60.min(axis=1), 0.0)

    records = []
    h_steps = {h: h // 15 for h in HORIZONS}
    for origin in origins:
        if origin not in phy.index:
            continue
        s = state.loc[origin]
        for h, step in h_steps.items():
            tgt = origin + pd.to_timedelta(h, unit="min")
            if tgt not in phy.index:
                continue
            f_phy = phy.loc[tgt]
            f_holder_delta = float(f_phy["holder"] - s["holder"])
            f_rest_delta = float(f_phy["rest"] - s["rest_current"])
            f_avail_delta = float(f_phy["avail_B"] - s["avail_B"])

            # labels
            rest_trans = abs(f_rest_delta) >= 20.0
            holder_flip = float(s["holder_slope_4"]) * f_holder_delta < 0 and abs(f_holder_delta) > 10.0
            avail_contract = f_avail_delta < -5.0

            rec = {"origin_time": origin, "horizon": h,
                   "label_rest_trans": int(rest_trans), "label_holder_flip": int(holder_flip),
                   "label_avail_contract": int(avail_contract)}
            for c in STATE_FEATURES + PRICE_FEATURES:
                rec[c] = s[c]
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
    ])


def _forward_evaluate(rows: pd.DataFrame, features_a: list[str], features_b: list[str],
                      labels: list[str], fold_order: list[str]) -> list[dict[str, object]]:
    """对每组 label×features 做 forward cross-fit，返回 AUC/logloss。"""
    out = []
    for label in labels:
        y = rows[label].to_numpy(int)
        for feat_name, feats in ("state_only", features_a), ("state_plus_price", features_b):
            pred_p = np.zeros(len(rows), dtype=float)
            covered = np.zeros(len(rows), dtype=bool)
            for pos, fold in enumerate(fold_order):
                held = rows["fold"].eq(fold).to_numpy()
                if pos == 0:
                    continue
                train = rows["fold"].isin(fold_order[:pos]).to_numpy()
                m = _classifier()
                m.fit(rows.loc[train, feats], y[train])
                pp = m.predict_proba(rows.loc[held, feats])[:, 1]
                pred_p[held] = pp
                covered[held] = True
            y_c = y[covered]
            pp_c = pred_p[covered]
            valid = len(np.unique(y_c)) >= 2
            auc = float(roc_auc_score(y_c, pp_c)) if valid else float("nan")
            ll = float(log_loss(y_c, np.clip(pp_c, 1e-12, 1 - 1e-12))) if valid else float("nan")
            base_rate = float(y_c.mean())
            out.append({"label": label, "features": feat_name, "auc": auc, "logloss": ll,
                        "base_rate": base_rate, "rows": int(len(y_c))})
    return out


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{stamp}_wave1b_price_hazard")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== [1/3] load data + build labels ===")
    price = load_price_schedule(DATA_DIR / "price.xlsx")
    dataset = align_tables(DATA_DIR)
    raw = dataset.frame

    m = pd.read_csv(Path("results/raw/runs/experiments/pred1_x3_replay_20260810/oof.csv"),
                    parse_dates=["origin_time", "train_end"])
    m["fold"] = m["fold"].astype(str)
    fold_order = m.groupby("fold", sort=False)["origin_time"].min().sort_values().index.astype(str).tolist()

    origins = pd.DatetimeIndex(sorted(m["origin_time"].unique()))
    data = _build_labels_and_features(raw, price, origins)
    print(f"label rows: {len(data)}, origins: {data['origin_time'].nunique()}")

    # merge fold
    orig_fold = m[["origin_time", "fold"]].drop_duplicates("origin_time").set_index("origin_time")
    data["fold"] = data["origin_time"].map(orig_fold["fold"])

    print("\nbase rates:")
    for lbl in ("label_rest_trans", "label_holder_flip", "label_avail_contract"):
        print(f"  {lbl}: {data[lbl].mean():.3f}")

    print("=== [2/3] forward cross-fit ablation ===")
    labels = ["label_rest_trans", "label_holder_flip", "label_avail_contract"]
    feats_a = STATE_FEATURES
    feats_b = STATE_FEATURES + PRICE_FEATURES
    results = _forward_evaluate(data, feats_a, feats_b, labels, fold_order)

    print(f"\n{'label':<25s} {'features':<20s} {'AUC':>7s} {'logloss':>8s} {'base':>6s}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<25s} {r['features']:<20s} {r['auc']:7.4f} {r['logloss']:8.4f} {r['base_rate']:6.3f}")

    # Δ
    deltas = []
    for label in labels:
        a = next(r for r in results if r["label"] == label and r["features"] == "state_only")
        b = next(r for r in results if r["label"] == label and r["features"] == "state_plus_price")
        da = b["auc"] - a["auc"]
        dl = a["logloss"] - b["logloss"]  # positive = improvement
        deltas.append({"label": label, "delta_auc": da, "delta_logloss": dl})
        print(f"\n{label}: ΔAUC = {da:+.4f}  Δlogloss = {dl:+.4f} (positive=improvement)")

    print("\n=== [3/3] Gate ===")
    all_small = all(abs(d["delta_auc"]) < 0.02 for d in deltas) and all(d["delta_logloss"] < 0.01 for d in deltas)
    any_improvement = any(d["delta_logloss"] > 0.01 for d in deltas) or any(abs(d["delta_auc"]) >= 0.02 for d in deltas)
    max_dauc = max(abs(d["delta_auc"]) for d in deltas)
    max_dll = max(d["delta_logloss"] for d in deltas)

    if all_small:
        verdict = "STOP_PRICE_PERMANENTLY_CLOSED"
    elif any_improvement:
        verdict = "PRICE_HAS_INDEPENDENT_INFO"
    else:
        verdict = "STOP_PRICE_NEGATIVE"

    print(f"max |ΔAUC|: {max_dauc:.4f}, max Δlogloss: {max_dll:.4f}")
    print(f"verdict: {verdict}")

    data.to_csv(run_dir / "hazard_labels.csv", index=False, encoding="utf-8")
    report = {
        "experiment": "wave1b_price_hazard_ablation",
        "stamp": stamp,
        "state_features": STATE_FEATURES,
        "price_features": PRICE_FEATURES,
        "results": results,
        "deltas": deltas,
        "verdict": verdict,
        "pre_registered_gate": {
            "price_closed_if": "all |ΔAUC|<0.02 AND all Δlogloss<0.01 across all 3 labels",
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    run_meta = {
        "run": "wave1b_price_hazard_ablation",
        "stamp": stamp,
        "date": datetime.now().isoformat(),
        "python": sys.executable,
        "note": "已知未来电价是唯一合法未来信息通道；若对物理 transition 无预测增益则永久关闭",
        "status": "complete",
        "verdict": verdict,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrun_dir: {run_dir}")


if __name__ == "__main__":
    main()
