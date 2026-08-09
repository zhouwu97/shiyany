"""P2-1 — generator_1 V2 超参微搜索 (recent_days / ridge_alpha / 组合确认)。

只动 generator_1 V2。generator_all 冻结（不重训）——通过 config.targets=('generator_1',)
使 fit() 只为 gen1 建状态；generator_all 仅作为 current 输入供 _fit_branches 的
full_rest / rest 使用，但永远不训练、不修改。

本脚本生成一个候选在同一批外层折上的逐行 OOF，并输出：
  - 候选与 baseline 的 pooled MAPE / by_fold / by_horizon 对比
  - screening 判定（5 个固定 screening folds: dev_01,05,10,15,19）
  - generator_1 的 8 个 horizon ΔMAPE

使用:
  python scripts/p21_micro_search.py --data-dir ... --run-dir ... \
      --candidates recent_30,recent_45,recent_90 --baseline-name recent_60
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
SCREEN_FOLDS = ("dev_01", "dev_05", "dev_10", "dev_15", "dev_19")


def _recent_variants() -> dict[str, dict[str, int]]:
    return {f"recent_{d}": {"recent_days": d} for d in (30, 45, 60, 90)}


def _alpha_variants(recent_days: int) -> dict[str, dict[str, float | int]]:
    return {
        "alpha_10": {"recent_days": recent_days, "ridge_alpha": 10},
        "alpha_20": {"recent_days": recent_days, "ridge_alpha": 20},
        "alpha_40": {"recent_days": recent_days, "ridge_alpha": 40},
    }


def _combo_variants(recent_days: int) -> dict[str, dict[str, float | int]]:
    """P2-1c 局部组合确认：recent_days winner + alpha winner + baseline。"""
    return {
        "combo_45_10": {"recent_days": recent_days, "ridge_alpha": 10},
        "recent_45": {"recent_days": 45, "ridge_alpha": 20},
        "alpha_10": {"recent_days": 60, "ridge_alpha": 10},
        "baseline_60_20": {"recent_days": 60, "ridge_alpha": 20},
    }


def evaluate_variant(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    config,
    overrides: dict,
    variant: str,
    folds: list,
) -> pd.DataFrame:
    """在给定折集合上运行一个变体的 gen1 OOF，返回长表 (fold, origin, horizon, actual, pred)。"""
    cfg = replace(config, targets=(TARGET,), model=replace(config.model, **overrides))
    parts: list[pd.DataFrame] = []
    for fold in folds:
        train_mask, validation_mask = fold.masks(frame.index)
        model = GasAwareEnsembleForecaster("v2", cfg).fit(
            features.loc[train_mask],
            deltas.loc[train_mask],
            frame.loc[train_mask, [TARGET, "generator_all"]],
        )
        out = model.predict(
            features.loc[validation_mask],
            frame.loc[validation_mask, [TARGET, "generator_all"]],
        )
        base = _base_fold_rows(frame, fold, validation_mask, cfg)
        base = base.loc[base["target"].eq(TARGET)].copy()
        origin_times = pd.DatetimeIndex(frame.index[validation_mask])
        pred_long = pd.DataFrame(
            {
                "origin_time": np.repeat(origin_times.to_numpy(), len(HORIZONS_MIN)),
                "horizon": np.tile(HORIZONS_MIN, len(origin_times)),
                "prediction": out[
                    [f"{TARGET}_t+{minutes}_pred" for minutes in HORIZONS_MIN]
                ].to_numpy().reshape(-1),
            }
        )
        base = base.merge(pred_long, on=["origin_time", "horizon"], how="left")
        parts.append(base[["fold", "origin_time", "horizon", "actual", "prediction"]])
    return pd.concat(parts, ignore_index=True)


def score(rows: pd.DataFrame) -> dict[str, object]:
    rows = rows.copy()
    denom = np.maximum(rows["actual"].abs(), 1e-6)
    rows["ape"] = np.abs(rows["actual"] - rows["prediction"]) / denom
    dev = rows.loc[rows["fold"].ne("blind")]
    pooled = float(dev["ape"].mean())
    by_fold = {str(k): float(v) for k, v in dev.groupby("fold")["ape"].mean().items()}
    by_horizon = {
        f"t+{int(h)}": float(v) for h, v in dev.groupby("horizon")["ape"].mean().items()
    }
    return {
        "pooled_mape": pooled,
        "by_fold": by_fold,
        "by_horizon": by_horizon,
        "n_rows": int(len(dev)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/official/初赛-参赛者使用")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--candidates",
        required=True,
        help="逗号分隔变体名，来自 _variants()",
    )
    parser.add_argument("--baseline", required=True, help="基线变体名")
    parser.add_argument(
        "--mode",
        choices=("recent", "alpha", "combo"),
        default="recent",
        help="recent: 扫 recent_days; alpha: 在固定 recent_days 上扫 ridge_alpha; combo: 组合确认",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=60,
        help="alpha 模式下固定的 recent_days（沿用 P2-1a winner）",
    )
    parser.add_argument(
        "--scope",
        choices=("screening", "development"),
        default="screening",
        help="screening 只跑 5 折；development 跑全部 19 折",
    )
    parser.add_argument("--jobs", type=int, default=1)
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
    all_folds = make_outer_folds(dataset.frame.index, config)
    dev_folds = [f for f in all_folds if not f.blind]
    if args.scope == "screening":
        folds = [f for f in dev_folds if f.name in SCREEN_FOLDS]
    else:
        folds = dev_folds

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates = args.candidates.split(",")
    if args.mode == "recent":
        variant_map = _recent_variants()
    elif args.mode == "alpha":
        variant_map = _alpha_variants(args.recent_days)
    else:
        variant_map = _combo_variants(args.recent_days)
    def overrides_for(name: str):
        return variant_map[name]

    assert args.baseline in variant_map
    assert all(c in variant_map for c in candidates)

    scores: dict[str, dict] = {}
    rows: dict[str, pd.DataFrame] = {}
    for variant in [args.baseline] + [c for c in candidates if c != args.baseline]:
        df = evaluate_variant(
            dataset.frame, features, deltas, config, overrides_for(variant), variant, folds
        )
        df.to_csv(run_dir / f"oof_{variant}.csv", index=False)
        rows[variant] = df
        scores[variant] = score(df)
        print(f"{variant}: pooled={scores[variant]['pooled_mape']:.6f}")

    base = scores[args.baseline]
    comparison = {
        "baseline": args.baseline,
        "scope": args.scope,
        "folds": [f.name for f in folds],
        "baseline_pooled_mape": base["pooled_mape"],
        "candidates": {},
    }
    for variant in candidates:
        if variant == args.baseline:
            continue
        s = scores[variant]
        delta_pp = (s["pooled_mape"] - base["pooled_mape"]) * 100
        # screening: wins on the 5 screening folds
        wins = sum(1 for f in SCREEN_FOLDS if f in s["by_fold"] and f in base["by_fold"]
                   and s["by_fold"][f] < base["by_fold"][f])
        by_horizon_delta = {
            h: round((s["by_horizon"][h] - base["by_horizon"][h]) * 100, 4)
            for h in s["by_horizon"]
        }
        comparison["candidates"][variant] = {
            "pooled_mape": s["pooled_mape"],
            "delta_pp": round(delta_pp, 4),
            "screening_wins": wins,
            "by_horizon_delta_pp": by_horizon_delta,
            "by_fold_delta_pp": {
                str(f): round((s["by_fold"][f] - base["by_fold"][f]) * 100, 4)
                for f in s["by_fold"] if f in base["by_fold"]
            },
        }
        print(f"  vs baseline: Δ={delta_pp:+.4f}pp  screen_wins={wins}/5")

    (run_dir / "report.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
