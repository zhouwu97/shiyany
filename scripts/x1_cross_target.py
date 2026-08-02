"""X1 — Cross-target decomposition: generator_all → generator_1 indirect.

核心:
    rest = generator_all - generator_1
    g1_indirect(t+h) = gall_V3_OOF(t+h) - rest_pred(t+h)
    物理约束: max(0, gall-240) <= g1_indirect <= gall

两条路径独立:
  - V3 generator_all OOF 预测 (champion OOF 的 v3_pred 列, generator_all target)
  - rest predictor: X1-P persistence / X1-R rest delta ridge

报告每个候选:
  1. indirect gen1 standalone MAPE (gen1 cells)
  2. C0+indirect 与 C0+analog+indirect 的 LOO blend
     (generator_all cells 用 C0+analog, 因为 X1 只改 gen1)
  3. residual correlation C0↔X1, analog↔X1

防泄漏:
  - rest predictor 只在 fold train fit, 预测 fold val
  - V3 gall 用 champion OOF 的 OOF 预测 (非真实 future)
  - blend 权重 LOO (其他 folds 学)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gas_forecast.config import legacy_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.model_v1 import make_ridge_pipeline
from gas_forecast.splits import make_outer_folds
from gas_forecast.targets import build_delta_targets

HORIZONS_MIN = (15, 30, 45, 60, 75, 90, 105, 120)
CHAMPION_DIR = Path("results/raw/runs/oof/clean_c0_strict_20260801_v2")
E25_DEV = Path("results/raw/runs/experiments/e25_development_20260801")


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual), 1e-6)
    return float(np.mean(np.abs(actual - pred) / denom))


def _clip_indirect(gall: np.ndarray, g1_indirect: np.ndarray) -> np.ndarray:
    return np.clip(g1_indirect, np.maximum(0.0, gall - 240.0), gall)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/official/初赛-参赛者使用")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rest-model", choices=("persistence", "ridge"), default="persistence")
    parser.add_argument("--folds", default=None, help="逗号分隔 fold 名（冒烟用）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "Pre_gas.csv").exists():
        matches = sorted(c for c in data_dir.iterdir() if c.is_dir() and (c / "Pre_gas.csv").exists())
        if len(matches) == 1:
            data_dir = matches[0]
        else:
            raise FileNotFoundError(f"无法解析数据目录: {args.data_dir}")

    config = legacy_forecast_config()
    dataset = align_tables(data_dir, config.feature.frequency)
    prices = sorted(data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    features = build_causal_features(dataset.frame, config.feature, price)
    all_folds = make_outer_folds(dataset.frame.index, config)
    dev_folds = [f for f in all_folds if not f.blind]
    if args.folds is not None:
        allowed = set(args.folds.split(","))
        dev_folds = [f for f in dev_folds if f.name in allowed]

    # ---- load champion OOF (C0, V3 gall) + analog ----
    champ = pd.read_csv(CHAMPION_DIR / "oof.csv", parse_dates=["origin_time"])
    champ = champ[["fold", "origin_time", "target", "horizon", "actual",
                   "v3_pred", "v2_v3_target_reconciled_pred"]].rename(
        columns={"v2_v3_target_reconciled_pred": "c0", "v3_pred": "gall"})
    e25 = pd.read_csv(E25_DEV / "oof.csv", parse_dates=["origin_time"])
    e25 = e25[["fold", "origin_time", "target", "horizon", "e25_analog_k40_pred"]].rename(
        columns={"e25_analog_k40_pred": "analog"})
    ref = champ.merge(e25, on=["fold", "origin_time", "target", "horizon"], how="inner")
    ref = ref.loc[ref.fold.isin([f.name for f in dev_folds])].reset_index(drop=True)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- per fold: build indirect gen1 ----
    g1_indirect_by_fold: dict[str, pd.DataFrame] = {}
    for fold in dev_folds:
        train_mask, validation_mask = fold.masks(dataset.frame.index)
        x_tr = features.loc[train_mask]
        x_val = features.loc[validation_mask]
        origin_times = pd.DatetimeIndex(dataset.frame.index[validation_mask])
        rest_anchor = (dataset.frame.loc[validation_mask, "generator_all"]
                       - dataset.frame.loc[validation_mask, "generator_1"]).ffill().to_numpy(dtype=float)

        if args.rest_model == "persistence":
            rest_pred = np.repeat(rest_anchor[:, None], 8, axis=1)
        else:
            # rest delta ridge: fit on train, predict val
            rest_full = dataset.frame["generator_all"] - dataset.frame["generator_1"]
            rest_delta = build_delta_targets(
                dataset.frame.assign(rest=rest_full), ("rest",), config.feature.horizons
            )
            rest_cols = [f"rest_tplus_{m}" for m in HORIZONS_MIN]
            valid_tr = rest_delta.loc[train_mask, rest_cols].notna().all(axis=1)
            pipe = make_ridge_pipeline(config.model.ridge_alpha)
            pipe.fit(x_tr.loc[valid_tr], rest_delta.loc[train_mask].loc[valid_tr, rest_cols])
            rest_delta_pred = pipe.predict(x_val)
            rest_pred = rest_anchor[:, None] + rest_delta_pred

        # V3 gall OOF for generator_all, pivoted to (n_origins, 8)
        gall_part = ref.loc[(ref.fold == fold.name) & (ref.target == "generator_all")]
        gall_pivot = gall_part.pivot(index="origin_time", columns="horizon", values="gall")
        gall_full = gall_pivot.reindex(origin_times).to_numpy(dtype=float)

        g1_indirect = _clip_indirect(gall_full, gall_full - rest_pred)  # (n_origins, 8)
        pred_long = pd.DataFrame(
            {
                "fold": fold.name,
                "origin_time": np.repeat(origin_times.to_numpy(), len(HORIZONS_MIN)),
                "horizon": np.tile(HORIZONS_MIN, len(origin_times)),
                "x1_indirect": g1_indirect.reshape(-1),
            }
        )
        g1_indirect_by_fold[fold.name] = pred_long
        print(f"{fold.name}: indirect built", flush=True)

    x1_all = pd.concat(g1_indirect_by_fold.values(), ignore_index=True)

    # ---- attach x1 to ref rows (gen1 cells only) ----
    rows = ref.merge(x1_all, on=["fold", "origin_time", "horizon"], how="left")
    # x1 only defined for gen1
    rows.loc[rows["target"].eq("generator_all"), "x1_indirect"] = np.nan
    rows = rows.loc[rows[["c0", "analog", "actual"]].notna().all(axis=1)].reset_index(drop=True)
    rows.to_csv(run_dir / "x1_with_ref.csv", index=False)

    actual = rows["actual"].to_numpy(dtype=float)
    c0 = rows["c0"].to_numpy(dtype=float)
    analog = rows["analog"].to_numpy(dtype=float)
    g1_idx = rows["target"].eq("generator_1").to_numpy()
    x1 = rows["x1_indirect"].to_numpy(dtype=float)

    report: dict[str, object] = {
        "rest_model": args.rest_model,
        "folds": [f.name for f in dev_folds],
        "c0_pooled_mape": mape(actual, c0),
        "analog_pooled_mape": mape(actual, analog),
        "x1_indirect_gen1_mape": mape(actual[g1_idx], x1[g1_idx]),
        "c0_gen1_mape": mape(actual[g1_idx], c0[g1_idx]),
        "resid_corr": {
            "c0_x1": round(float(np.corrcoef(actual[g1_idx] - c0[g1_idx], actual[g1_idx] - x1[g1_idx])[0, 1]), 4),
            "analog_x1": round(float(np.corrcoef(actual[g1_idx] - analog[g1_idx], actual[g1_idx] - x1[g1_idx])[0, 1]), 4),
            "c0_analog": round(float(np.corrcoef(actual - c0, actual - analog)[0, 1]), 4),
        },
        "blends": {},
    }

    folds_ordered = [f.name for f in dev_folds]

    def loo_blend(cols: list[str], on_gen1: bool) -> float:
        """LOO blend; if on_gen1, weights fit only on gen1 cells and applied to gen1,
        gen_all cells get C0+analog weights (fit on gen_all)."""
        out = np.zeros(len(rows))
        for tgt in ["generator_1", "generator_all"]:
            idx = rows["target"].eq(tgt).to_numpy()
            sub = rows.loc[idx]
            use_cols = cols if (tgt == "generator_1" or not on_gen1 or "x1" not in cols) else ["c0", "analog"]
            use_cols = [c for c in use_cols if c in rows.columns and rows[c].notna().all()]
            for hold in folds_ordered:
                tr = sub.loc[sub.fold != hold]
                te = sub.loc[sub.fold == hold]
                m_tr = np.column_stack([tr[c].to_numpy(float) for c in use_cols])
                m_te = np.column_stack([te[c].to_numpy(float) for c in use_cols])
                r = minimize(lambda w: mape(tr["actual"].to_numpy(float), m_tr @ w),
                             np.full(len(use_cols), 1 / len(use_cols)),
                             method="SLSQP", bounds=[(0, 1)] * len(use_cols),
                             constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
                             options={"maxiter": 500, "ftol": 1e-10})
                w = r.x if r.success else np.eye(len(use_cols))[0]
                out[idx & (rows["fold"] == hold).to_numpy()] = m_te @ w
        return mape(actual, out)

    for combo in (["c0", "analog"], ["c0", "x1"], ["c0", "analog", "x1"]):
        report["blends"]["+".join(combo)] = loo_blend(combo, on_gen1=True)

    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
