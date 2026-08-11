"""Wave 0c: stock-flow residual-R² 证伪（分层，因果 vs 事后上限）。

假设：SAFE60 的 g1/gall 误差能被物理链（未来生产/需求/气柜/可用气）解释。
测试三层：
  A) 因果：residual ~ origin 可得物理状态（holder/rest/balance/ramp/price）
  B) 因果+OOF预测未来：residual ~ origin 状态 + 更早折预测的未来物理量
  C) 事后上限(ORACLE, 明确标注)：residual ~ origin 状态 + 真实未来物理轨迹

判定（预注册，先于运行）：
  - 若 C 相对 A 的 ΔR² < 0.05（整体）且 transition regime ΔR² < 0.10
    → stock-flow 物理链对 SAFE60 误差无内容，线永久关闭。
  - 若 C 有内容(ΔR²≥0.05)但 B 相对 A ΔR² ≈ 0 → 物理链有内容但因果不可榨，线保持关闭。
  - 仅当 C 有内容 且 B 相对 A 也有 ΔR² ≥ 0.03 → 才有资格建完整 stock-flow expert。

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
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from gas_forecast.data import align_tables  # noqa: E402

DATA_DIR = Path("data/raw/official/初赛-参赛者使用")
X3_REPLAY_OOF = Path("results/raw/runs/experiments/pred1_x3_replay_20260810/oof.csv")
HORIZONS = (15, 30, 45, 60, 75, 90, 105, 120)

ORIGIN_STATE = [
    "holder", "holder_slope_4", "rest_current", "avail_B", "generator_gas",
    "rest_same_slot_7d", "price_switch_120", "steps_to_price_switch", "current_price",
]
FUTURE_DELTAS = ["f_holder_delta", "f_prod_delta", "f_avail_delta", "f_rest_delta", "f_gen_gas_delta"]


def _physical_series(frame: pd.DataFrame) -> pd.DataFrame:
    """构建 origin 级物理序列（15min grid, DatetimeIndex）。"""
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
    out = pd.DataFrame({"holder": holder, "prod": prod, "nongen": nongen,
                        "generator_gas": gen_gas, "avail_B": avail, "rest": rest}, index=frame.index)
    return out


def _build_features(raw: pd.DataFrame, price, origins, horizons) -> pd.DataFrame:
    """对每个 origin×horizon 构造 origin 状态 + 未来实际(Oracle) 特征。"""
    phy = _physical_series(raw)
    # origin 级状态
    state = pd.DataFrame(index=raw.index)
    state["holder"] = phy["holder"]
    state["holder_slope_4"] = phy["holder"] - phy["holder"].shift(3)
    state["rest_current"] = phy["rest"]
    state["avail_B"] = phy["avail_B"]
    state["generator_gas"] = phy["generator_gas"]
    state["rest_same_slot_7d"] = phy["rest"].shift(96 * 7)
    state["current_price"] = price.lookup(raw.index)
    # price switch: 未来 120min 内是否有已知电价切换（origin 可得，合法）
    price_mat = np.column_stack([price.lookup(raw.index + pd.to_timedelta(h, unit="m")) for h in HORIZONS])
    cur = state["current_price"].to_numpy(float)
    changed = ~np.isclose(price_mat, cur[:, None], rtol=0.0, atol=1e-12)
    state["price_switch_120"] = changed.any(axis=1).astype("int8")
    first_step = np.where(changed.any(axis=1), changed.argmax(axis=1) + 1, 0)
    state["steps_to_price_switch"] = first_step.astype("int8")

    records = []
    for origin in origins:
        if origin not in phy.index:
            continue
        s = state.loc[origin]
        for h in horizons:
            tgt = origin + pd.to_timedelta(h, unit="min")
            if tgt not in phy.index:
                continue
            f_phy = phy.loc[tgt]
            rec = {"origin_time": origin, "horizon": h}
            for c in ORIGIN_STATE:
                rec[c] = s[c]
            rec.update({
                "f_holder_delta": float(f_phy["holder"] - s["holder"]),
                "f_prod_delta": float(f_phy["prod"] - phy.loc[origin, "prod"]),
                "f_avail_delta": float(f_phy["avail_B"] - s["avail_B"]),
                "f_rest_delta": float(f_phy["rest"] - s["rest_current"]),
                "f_gen_gas_delta": float(f_phy["generator_gas"] - s["generator_gas"]),
            })
            records.append(rec)
    return pd.DataFrame.from_records(records).set_index(["origin_time", "horizon"])


def _ridge() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=20.0)),
    ])


def _r2(r_hat: np.ndarray, r: np.ndarray) -> float:
    ss_res = float(np.mean((r - r_hat) ** 2))
    ss_tot = float(np.mean(r ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _forward_crossfit(rows: pd.DataFrame, y: pd.Series, features: list[str],
                      fold_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full(len(rows), 0.0, dtype=float)
    covered = np.zeros(len(rows), dtype=bool)
    for position, fold in enumerate(fold_order):
        held = rows["fold"].eq(fold).to_numpy()
        if position == 0:
            continue
        train = rows["fold"].isin(fold_order[:position]).to_numpy()
        m = _ridge()
        m.fit(rows.loc[train, features], y[train])
        pred[held] = m.predict(rows.loc[held, features])
        covered[held] = True
    return pred, covered


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{stamp}_wave0c_stockflow")
    run_dir.mkdir(parents=True, exist_ok=True)

    from gas_forecast.features import load_price_schedule
    price = load_price_schedule(DATA_DIR / "price.xlsx")
    print("=== [1/4] load data ===")
    dataset = align_tables(DATA_DIR)
    raw = dataset.frame

    m = pd.read_csv(X3_REPLAY_OOF, parse_dates=["origin_time", "train_end"])
    m["fold"] = m["fold"].astype(str)
    m["safe60_pred"] = 0.6 * m["x3_cat_mae_pred"] + 0.4 * m["a61_recursive_blend_05_pred"]
    fold_order = m.groupby("fold", sort=False)["origin_time"].min().sort_values().index.astype(str).tolist()
    print(f"OOF rows: {len(m)}, folds: {len(fold_order)}")

    print("=== [2/4] build physical features ===")
    origins = pd.DatetimeIndex(sorted(m["origin_time"].unique()))
    feat = _build_features(raw, price, origins, HORIZONS)
    print(f"feature rows: {len(feat)}")

    # 合并到 OOF（只保留 g1 目标）
    g1 = m.loc[m["target"].eq("generator_1")].copy()
    key = g1.set_index(["origin_time", "horizon"])
    missing = sorted(set(g1.set_index(["origin_time", "horizon"]).index) - set(feat.index))
    if missing:
        print(f"WARNING: {len(missing)} (origin,horizon) missing features, dropping")
    joined = key.join(feat, how="inner").reset_index()
    joined["fold"] = joined["fold"].astype(str)
    print(f"joined g1 rows: {len(joined)}")

    r = (joined["actual"] - joined["safe60_pred"]).to_numpy(float)
    r_abs = np.abs(r)
    joined["residual"] = r
    joined["residual_abs"] = r_abs
    y_labels = {"signed": r, "abs": r_abs}

    print("=== [3/4] R² regressions (cross-fit causal / cross-fit ceiling / in-sample ceiling) ===")
    results: list[dict[str, object]] = []
    # 未来物理量可预测性诊断：origin 状态 → 每个 future delta 的 cross-fit R²
    delta_pred_r2 = {}
    for fd in FUTURE_DELTAS:
        pfd, _ = _forward_crossfit(joined, joined[fd].to_numpy(float), ORIGIN_STATE, fold_order)
        delta_pred_r2[fd] = _r2(pfd, joined[fd].to_numpy(float))
    print("future-delta predictability (cross-fit R² from origin state):")
    for fd, v in delta_pred_r2.items():
        print(f"  {fd}: {v:.4f}")

    for scope, y in y_labels.items():
        y = joined["residual" if scope == "signed" else "residual_abs"].to_numpy(float)
        # A) 因果：origin 状态
        pred_a, cov_a = _forward_crossfit(joined, y, ORIGIN_STATE, fold_order)
        r2_a = _r2(pred_a[cov_a], y[cov_a])
        # C_cf) 诚实 ceiling：cross-fit，给真实未来（"完美预见物理路径"上限，仍 forward）
        pred_ccf, cov_ccf = _forward_crossfit(joined, y, ORIGIN_STATE + FUTURE_DELTAS, fold_order)
        r2_ccf = _r2(pred_ccf[cov_ccf], y[cov_ccf])
        # C_insample) 事后上限（in-sample ORACLE，明确标注，只作参考）
        m_c = _ridge()
        m_c.fit(joined.loc[:, ORIGIN_STATE + FUTURE_DELTAS], y)
        pred_c = m_c.predict(joined.loc[:, ORIGIN_STATE + FUTURE_DELTAS])
        r2_c = _r2(pred_c, y)
        # B) 因果 + OOF 预测未来：先预测未来 delta，再 regress residual on predicted
        joined_pred_fut = joined.copy()
        for fd in FUTURE_DELTAS:
            pfd, _ = _forward_crossfit(joined, joined[fd].to_numpy(float), ORIGIN_STATE, fold_order)
            joined_pred_fut[fd + "_pred"] = pfd
        feats_b = ORIGIN_STATE + [fd + "_pred" for fd in FUTURE_DELTAS]
        pred_b, cov_b = _forward_crossfit(joined_pred_fut, y, feats_b, fold_order)
        r2_b = _r2(pred_b[cov_b], y[cov_b])

        results.append({
            "scope": scope,
            "r2_causal_origin_state": r2_a,
            "r2_causal_plus_pred_future": r2_b,
            "r2_crossfit_ceiling_oracle_future": r2_ccf,
            "r2_insample_ceiling_oracle_future": r2_c,
            "delta_r2_crossfit_ceiling_vs_causal": r2_ccf - r2_a,
            "delta_r2_pred_future_vs_causal": r2_b - r2_a,
            "covered_causal_rows": int(cov_a.sum()),
        })
        print(f"[{scope}] A(causal) R²={r2_a:.4f} | B(+pred_future) R²={r2_b:.4f} | "
              f"C_cf(crossfit oracle) R²={r2_ccf:.4f} | C_insample R²={r2_c:.4f} | "
              f"ΔC_cf-A={r2_ccf-r2_a:+.4f} | ΔB-A={r2_b-r2_a:+.4f}")

    # regime 分层：transition / holder 快速 / price switch（held-only, causal A vs ceiling C）
    joined["transition"] = joined["f_rest_delta"].abs() >= 20.0
    holder_q = joined["holder_slope_4"].quantile([1 / 3, 2 / 3])
    joined["holder_fast"] = joined["holder_slope_4"].abs() > joined["holder_slope_4"].abs().quantile(2 / 3)
    joined["price_switch"] = joined["price_switch_120"].eq(1)
    y_signed = joined["residual"].to_numpy(float)

    regime_results: list[dict[str, object]] = []
    regimes = {
        "all": joined["fold"].notna(),
        "transition_rest": joined["transition"],
        "holder_fast": joined["holder_fast"],
        "price_switch": joined["price_switch"],
    }
    pred_a_all, cov_a_all = _forward_crossfit(joined, y_signed, ORIGIN_STATE, fold_order)
    pred_ccf_all, cov_ccf_all = _forward_crossfit(joined, y_signed, ORIGIN_STATE + FUTURE_DELTAS, fold_order)
    for name, mask in regimes.items():
        if mask.sum() < 100:
            regime_results.append({"regime": name, "rows": int(mask.sum()), "r2_causal": None, "r2_ceiling": None, "delta": None})
            continue
        r2a = _r2(pred_a_all[mask & cov_a_all], y_signed[mask & cov_a_all]) if (mask & cov_a_all).any() else float("nan")
        r2ccf = _r2(pred_ccf_all[mask & cov_ccf_all], y_signed[mask & cov_ccf_all]) if (mask & cov_ccf_all).any() else float("nan")
        regime_results.append({
            "regime": name, "rows": int(mask.sum()),
            "r2_causal_origin": float(r2a), "r2_crossfit_ceiling_oracle": float(r2ccf),
            "delta_ceiling_vs_causal": float(r2ccf - r2a),
        })
    print("\n--- regime breakdown (signed residual) ---")
    for row in regime_results:
        print(row)

    print("=== [4/4] 判定 ===")
    signed = results[0]
    ceiling_content = signed["delta_r2_crossfit_ceiling_vs_causal"]
    causal_content = signed["delta_r2_pred_future_vs_causal"]
    regime_delta = max((row.get("delta_ceiling_vs_causal") or 0) for row in regime_results)
    decision = {
        "ceiling_has_content": bool(ceiling_content >= 0.05),
        "transition_regime_ceiling_delta": float(regime_delta),
        "causal_pred_future_content": bool(causal_content >= 0.03),
    }
    if not decision["ceiling_has_content"] and regime_delta < 0.10:
        verdict = "STOP_PHYSICAL_CHAIN_DEAD"
    elif decision["ceiling_has_content"] and not decision["causal_pred_future_content"]:
        verdict = "STOP_CONTENT_BUT_CAUSALLY_UNCAPTURABLE"
    elif decision["ceiling_has_content"] and decision["causal_pred_future_content"]:
        verdict = "GO_BUILD_STOCK_FLOW_EXPERT"
    else:
        verdict = "STOP_BORDERLINE"
    print(f"verdict: {verdict}, ceiling ΔR²={ceiling_content:.4f}, transition ΔR²={regime_delta:.4f}, "
          f"causal pred-future ΔR²={causal_content:.4f}")

    joined.reset_index().to_csv(run_dir / "residual_phys_table.csv", index=False, encoding="utf-8")
    report = {
        "experiment": "wave0c_stock_flow_residual_r2",
        "stamp": stamp,
        "baseline": "SAFE60",
        "note": "crossfit ceiling = forward cross-fit given ACTUAL future physical trajectory (perfect-foresight upper bound); causal = forward cross-fit, origin state only",
        "results": results,
        "regimes": regime_results,
        "future_delta_predictability": delta_pred_r2,
        "decision": decision,
        "verdict": verdict,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    run_meta = {
        "run": "wave0c_stock_flow_residual_r2",
        "stamp": stamp,
        "date": datetime.now().isoformat(),
        "python": sys.executable,
        "pre_registered_gates": {
            "ceiling_content_pp_r2_ge_0.05": 0.05,
            "transition_regime_ceiling_delta_ge_0.10": 0.10,
            "causal_pred_future_delta_r2_ge_0.03": 0.03,
        },
        "caveats": ["ceiling is hindsight ORACLE, in-sample; only causal cross-fit R² counts for GO"],
        "status": "complete",
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrun_dir: {run_dir}")


if __name__ == "__main__":
    main()
