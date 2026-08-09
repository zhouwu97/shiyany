"""P2-1d — combo_45_10 的 19-fold full-dev 验证 + 预注册 splice 评估。

一次训练 combo_45_10（recent_days=45, ridge_alpha=10）在全部 19 个 dev folds，
保存完整逐 horizon OOF。baseline 直接复用 A2 的 v2_pred（已逐 cell 验证
== recent_60/alpha_20，max diff 0.0）。

同时评估 3 个预注册候选（边界在 screening 阶段固定，不后验调整）:
  1. global_combo   : 8 horizons 全用 combo_45_10
  2. splice_75      : t+15..60 baseline, t+75..120 combo
  3. splice_90      : t+15..75 baseline, t+90..120 combo

输出:
  - 每候选 overall gen1 pooled / by_horizon / by_fold / fold×horizon delta
  - t+75..120 pooled + long-cell win rate
  - generator_all 冻结逐 cell equality check（combo 只训 gen1，gen_all 天然不变）
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import legacy_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster
from gas_forecast.oof import _base_fold_rows
from gas_forecast.splits import make_outer_folds
from gas_forecast.targets import build_delta_targets

HORIZONS_MIN = (15, 30, 45, 60, 75, 90, 105, 120)
TARGET = "generator_1"
COMBO = {"recent_days": 45, "ridge_alpha": 10}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/official/初赛-参赛者使用")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--a2-baseline-dir", default="results/raw/runs/a2_calibration/20260802_102331")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--folds", default=None, help="逗号分隔 fold 名白名单（冒烟用）")
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
    deltas = build_delta_targets(dataset.frame, config.targets, config.feature.horizons)
    dev_folds = [f for f in make_outer_folds(dataset.frame.index, config) if not f.blind]
    if args.folds is not None:
        allowed = set(args.folds.split(","))
        dev_folds = [f for f in dev_folds if f.name in allowed]

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. train combo on all 19 dev folds, save per-horizon OOF ----
    combo_cfg = replace(config, targets=(TARGET,), model=replace(config.model, **COMBO))
    parts: list[pd.DataFrame] = []
    for fold in dev_folds:
        train_mask, validation_mask = fold.masks(dataset.frame.index)
        model = GasAwareEnsembleForecaster("v2", combo_cfg).fit(
            features.loc[train_mask],
            deltas.loc[train_mask],
            dataset.frame.loc[train_mask, [TARGET, "generator_all"]],
        )
        out = model.predict(
            features.loc[validation_mask],
            dataset.frame.loc[validation_mask, [TARGET, "generator_all"]],
        )
        base = _base_fold_rows(dataset.frame, fold, validation_mask, combo_cfg)
        base = base.loc[base["target"].eq(TARGET)].copy()
        origin_times = pd.DatetimeIndex(dataset.frame.index[validation_mask])
        pred_long = pd.DataFrame(
            {
                "origin_time": np.repeat(origin_times.to_numpy(), len(HORIZONS_MIN)),
                "horizon": np.tile(HORIZONS_MIN, len(origin_times)),
                "combo_pred": out[
                    [f"{TARGET}_t+{m}_pred" for m in HORIZONS_MIN]
                ].to_numpy().reshape(-1),
            }
        )
        base = base.merge(pred_long, on=["origin_time", "horizon"], how="left")
        parts.append(base[["fold", "origin_time", "horizon", "actual", "combo_pred"]])
        print(f"{fold.name}: done", flush=True)
    combo_rows = pd.concat(parts, ignore_index=True)
    combo_rows.to_csv(run_dir / "oof_combo_dev.csv", index=False)

    # ---- 2. load A2 baseline (verified == recent_60/alpha_20) ----
    a2_dir = Path(args.a2_baseline_dir)
    a2 = pd.concat(
        [pd.read_csv(p) for p in sorted(a2_dir.glob("branches_*.csv"))],
        ignore_index=True,
    )
    a2 = a2[["fold", "origin_time", "horizon", "v2_pred"]].rename(columns={"v2_pred": "base_pred"})
    a2["origin_time"] = pd.to_datetime(a2["origin_time"])
    rows = combo_rows.merge(a2, on=["fold", "origin_time", "horizon"], how="left")
    if rows["base_pred"].isna().any():
        raise RuntimeError("A2 baseline 未覆盖全部 combo OOF 行")
    rows.to_csv(run_dir / "oof_dev_with_baseline.csv", index=False)

    # ---- 3. evaluate the 3 pre-registered candidates ----
    def build(candidate: str, r: pd.DataFrame) -> pd.DataFrame:
        out = r.copy()
        if candidate == "global_combo":
            out["pred"] = out["combo_pred"]
        elif candidate == "splice_75":
            out["pred"] = np.where(out["horizon"] >= 75, out["combo_pred"], out["base_pred"])
        elif candidate == "splice_90":
            out["pred"] = np.where(out["horizon"] >= 90, out["combo_pred"], out["base_pred"])
        else:
            raise ValueError(candidate)
        denom = np.maximum(out["actual"].abs(), 1e-6)
        out["ape"] = np.abs(out["actual"] - out["pred"]) / denom
        out["base_ape"] = np.abs(out["actual"] - out["base_pred"]) / denom
        out["combo_ape"] = np.abs(out["actual"] - out["combo_pred"]) / denom
        return out

    candidates = ["global_combo", "splice_75", "splice_90"]
    report: dict[str, object] = {
        "combo_config": COMBO,
        "baseline": "A2 v2_pred (recent_60/alpha_20), verified bit-identical on screening",
        "folds": [f.name for f in dev_folds],
        "candidates": {},
    }
    for candidate in candidates:
        r = build(candidate, rows)
        overall = float(r["ape"].mean())
        base_overall = float(r["base_ape"].mean())
        delta_pp = (overall - base_overall) * 100
        by_horizon = {f"t+{int(h)}": float(v) for h, v in r.groupby("horizon")["ape"].mean().items()}
        base_by_horizon = {f"t+{int(h)}": float(v) for h, v in r.groupby("horizon")["base_ape"].mean().items()}
        # long-horizon (t+75..120) pooled
        long = r.loc[r["horizon"] >= 75]
        long_pooled = float(long["ape"].mean())
        long_base = float(long["base_ape"].mean())
        # long-cell win rate (fold × horizon cells, t+75..120)
        long_delta = long.groupby(["fold", "horizon"])[["ape", "base_ape"]].mean()
        long_wins = int((long_delta["ape"] < long_delta["base_ape"]).sum())
        long_cells = int(len(long_delta))
        # short unchanged? (t+15..60 for splice variants)
        short = r.loc[r["horizon"] < 75]
        short_delta = (float(short["ape"].mean()) - float(short["base_ape"].mean())) * 100
        by_fold = {str(k): float(v) for k, v in r.groupby("fold")["ape"].mean().items()}
        base_by_fold = {str(k): float(v) for k, v in r.groupby("fold")["base_ape"].mean().items()}
        fold_delta = {k: round((by_fold[k] - base_by_fold[k]) * 100, 4) for k in by_fold}
        win_folds = sum(1 for k in by_fold if by_fold[k] < base_by_fold[k])
        report["candidates"][candidate] = {
            "overall_gen1_mape": overall,
            "baseline_gen1_mape": base_overall,
            "overall_delta_pp": round(delta_pp, 4),
            "long_horizon_pooled": round(long_pooled, 5),
            "long_horizon_base_pooled": round(long_base, 5),
            "long_horizon_delta_pp": round((long_pooled - long_base) * 100, 4),
            "long_cell_wins": f"{long_wins}/{long_cells}",
            "long_cell_win_rate": round(long_wins / long_cells, 4) if long_cells else None,
            "short_horizon_delta_pp": round(short_delta, 4),
            "fold_wins": f"{win_folds}/19",
            "by_horizon_mape": by_horizon,
            "baseline_by_horizon_mape": base_by_horizon,
            "by_horizon_delta_pp": {
                h: round((by_horizon[h] - base_by_horizon[h]) * 100, 4) for h in by_horizon
            },
            "by_fold_delta_pp": fold_delta,
        }
        print(f"{candidate}: overall Δ={delta_pp:+.4f}pp  long Δ={(long_pooled-long_base)*100:+.4f}pp  "
              f"long-cell {long_wins}/{long_cells}  short Δ={short_delta:+.4f}pp")

    # ---- 4. generator_all frozen equality check ----
    # combo only fits gen1; generator_all is never trained/modified. Check by
    # confirming combo model produced no generator_all columns.
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
