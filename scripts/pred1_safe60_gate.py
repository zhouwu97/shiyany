"""PRED-1 Gate C: SAFE60 重放评估 + 双 baseline + day-block bootstrap。

输入：重放 X3 OOF 与 A61 OOF（或冻结 A61 verification OOF），在本脚本内实时
构造 SAFE60 = 0.60*replay_X3 + 0.40*replay_A61，执行线性恒等校验、pooled /
fold / recent5 / target / horizon 分解，以及 day-block bootstrap
（5000 reps, seed 20260810）双 baseline（研究基线 A61、promotion 基线
aggressive）概率比较。不读取外部 blend 列。

用法：
  python scripts/pred1_safe60_gate.py \
    --x3-oof <replay x3 oof.csv> \
    --a61-oof <a61 oof.csv> \
    --output <report.json> \
    [--aggressive-oof <optional; default = use aggressive column in x3 oof>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.scoring import (
    ScoreSpec,
    competition_mape,
    score_oof_long,
)

SAFE60_X3_WEIGHT = 0.60
SAFE60_A61_WEIGHT = 0.40
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260810


def _read_oof(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def _fold_order(rows: pd.DataFrame) -> list[str]:
    order = (
        rows.groupby("fold", sort=False, observed=True)["origin_time"]
        .min()
        .sort_values()
    )
    return order.index.astype(str).tolist()


def _merge_oofs(
    x3: pd.DataFrame, a61: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = ["fold", "origin_time", "target", "horizon"]
    merged = x3.merge(a61, on=keys, suffixes=("_x3", "_a61"))
    audit = {
        "x3_rows": int(len(x3)),
        "a61_rows": int(len(a61)),
        "merged_rows": int(len(merged)),
        "actual_identical": bool((merged["actual_x3"] == merged["actual_a61"]).all()),
    }
    work = pd.DataFrame(
        {
            "fold": merged["fold"],
            "origin_time": pd.to_datetime(merged["origin_time"]),
            "target": merged["target"],
            "horizon": merged["horizon"],
            "actual": merged["actual_x3"],
            "x3_cat_mae_pred": merged["x3_cat_mae_pred"].astype(float),
            "a61_pred": merged["a61_recursive_blend_05_pred_a61"].astype(float),
        }
    )
    # aggressive is present in BOTH oofs -> suffixed; prefer the X3-side parent.
    for cand in ("aggressive_r75_lgb20_pred_x3", "aggressive_r75_lgb20_pred_a61"):
        if cand in merged.columns:
            work["aggressive_r75_lgb20_pred"] = merged[cand].astype(float)
            break
    return work, audit


def _safe60(work: pd.DataFrame) -> pd.DataFrame:
    work = work.copy()
    work["safe60_pred"] = (
        SAFE60_X3_WEIGHT * work["x3_cat_mae_pred"]
        + SAFE60_A61_WEIGHT * work["a61_pred"]
    )
    linear_check = (
        np.abs(
            work["safe60_pred"]
            - (
                SAFE60_X3_WEIGHT * work["x3_cat_mae_pred"]
                + SAFE60_A61_WEIGHT * work["a61_pred"]
            )
        ).max()
    )
    work.attrs["linear_identity_max_abs_diff"] = float(linear_check)
    return work


def _recent5(work: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    return work[work["fold"].isin(order[-5:])]


def _day_block_bootstrap(
    rows: pd.DataFrame,
    candidate_col: str,
    baseline_col: str,
) -> dict[str, object]:
    """PRED-0 规范 day-block bootstrap。

    重采样单位 = origin_time.date；同一天所有 cell（2 target x 8 horizon x 各
    origin）整体保留。每次重采样：按天有放回抽样（天数相同），拼接被抽中的全部
    cell，计算 pooled `improvement = MAPE_baseline - MAPE_candidate`。
    输出 mean / median / 95% CI / P(improvement > 0)。
    """

    spec = ScoreSpec()
    work = rows.copy()
    work["origin_time"] = pd.to_datetime(work["origin_time"])
    work["_day"] = work["origin_time"].dt.floor("D")
    days = pd.DatetimeIndex(sorted(work["_day"].unique()))
    day_rows = {day: part for day, part in work.groupby("_day", sort=True)}
    if not days.size:
        raise ValueError("bootstrap 没有可用的天块")

    def pooled_improvement(day_index: np.ndarray) -> float:
        blocks = pd.concat([day_rows[d] for d in days[day_index]])
        base = competition_mape(blocks["actual"], blocks[baseline_col], epsilon=spec.epsilon)
        cand = competition_mape(blocks["actual"], blocks[candidate_col], epsilon=spec.epsilon)
        return float(base - cand)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    improvement = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for r in range(BOOTSTRAP_SAMPLES):
        index = rng.integers(0, days.size, size=days.size)
        improvement[r] = pooled_improvement(index)

    return {
        "block": "day",
        "blocks": int(days.size),
        "samples": int(BOOTSTRAP_SAMPLES),
        "random_seed": BOOTSTRAP_SEED,
        "mean_improvement_pp": float(improvement.mean() * 100.0),
        "median_improvement_pp": float(np.median(improvement) * 100.0),
        "ci95_low_pp": float(np.quantile(improvement, 0.025) * 100.0),
        "ci95_high_pp": float(np.quantile(improvement, 0.975) * 100.0),
        "probability_candidate_better": float(np.mean(improvement > 0.0)),
        "observed_improvement_pp": float(
            (
                competition_mape(work["actual"], work[baseline_col], epsilon=spec.epsilon)
                - competition_mape(work["actual"], work[candidate_col], epsilon=spec.epsilon)
            )
            * 100.0
        ),
    }


def _gate_report(
    work: pd.DataFrame,
    *,
    a61_oof: pd.DataFrame,
    aggressive_col: str,
) -> dict[str, object]:
    order = _fold_order(work)
    recent5 = _recent5(work, order)

    safe60 = _safe60(work)
    safe60_recent5 = _safe60(recent5)

    x3_score = score_oof_long(work, "x3_cat_mae_pred")["pooled_mape"]
    a61_score = score_oof_long(work, "a61_pred")["pooled_mape"]
    safe60_score = score_oof_long(safe60, "safe60_pred")["pooled_mape"]
    aggressive_score = score_oof_long(work, aggressive_col)["pooled_mape"]

    by_fold_safe60 = score_oof_long(safe60, "safe60_pred")["by_fold"]
    by_fold_a61 = score_oof_long(work, "a61_pred")["by_fold"]
    by_fold_aggressive = score_oof_long(work, aggressive_col)["by_fold"]

    def wins(a: dict[str, float], b: dict[str, float]) -> int:
        return int(sum(b[f] < a[f] for f in order))

    def worst_regression(a: dict[str, float], b: dict[str, float]) -> float:
        return max((b[f] - a[f]) for f in order)  # positive = baseline better than candidate

    def target_wins(work_df, cand_col, base_col) -> dict[str, int]:
        scored = score_oof_long(work_df, cand_col)["by_target"]
        base = score_oof_long(work_df, base_col)["by_target"]
        return {str(t): int(base[t] < scored[t]) for t in sorted(scored)}

    bootstrap_vs_a61 = _day_block_bootstrap(
        safe60, "safe60_pred", "a61_pred"
    )
    bootstrap_vs_aggressive = _day_block_bootstrap(
        safe60, "safe60_pred", aggressive_col
    )

    return {
        "linear_identity_max_abs_diff": safe60.attrs["linear_identity_max_abs_diff"],
        "pooled_mape": {
            "x3": round(x3_score * 100, 6),
            "a61": round(a61_score * 100, 6),
            "safe60": round(safe60_score * 100, 6),
            "aggressive": round(aggressive_score * 100, 6),
        },
        "improvement_vs_a61_pp": round((a61_score - safe60_score) * 100, 6),
        "improvement_vs_aggressive_pp": round((aggressive_score - safe60_score) * 100, 6),
        "fold_wins_safe60_vs_a61": wins(by_fold_a61, by_fold_safe60),
        "fold_wins_safe60_vs_aggressive": wins(by_fold_aggressive, by_fold_safe60),
        "recent5_wins_safe60_vs_a61": int(
            sum(
                by_fold_a61[f] > by_fold_safe60[f]
                for f in order[-5:]
            )
        ),
        "worst_fold_regression_vs_a61_pp": round(worst_regression(by_fold_a61, by_fold_safe60) * 100, 6),
        "worst_fold_regression_vs_aggressive_pp": round(
            worst_regression(by_fold_aggressive, by_fold_safe60) * 100, 6
        ),
        "target_mape": {
            "a61": {str(t): round(v * 100, 6) for t, v in score_oof_long(work, "a61_pred")["by_target"].items()},
            "safe60": {str(t): round(v * 100, 6) for t, v in score_oof_long(safe60, "safe60_pred")["by_target"].items()},
            "aggressive": {str(t): round(v * 100, 6) for t, v in score_oof_long(work, aggressive_col)["by_target"].items()},
        },
        "bootstrap_vs_a61": bootstrap_vs_a61,
        "bootstrap_vs_aggressive": bootstrap_vs_aggressive,
        "recent5": {
            "folds": order[-5:],
            "pooled_vs_a61_pp": round(
                (score_oof_long(work[work["fold"].isin(order[-5:])], "a61_pred")["pooled_mape"]
                 - score_oof_long(safe60_recent5, "safe60_pred")["pooled_mape"]) * 100, 6
            ),
        },
        "folds": order,
        "bootstrap_spec": {
            "block": "day",
            "samples": BOOTSTRAP_SAMPLES,
            "random_seed": BOOTSTRAP_SEED,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x3-oof", type=Path, required=True)
    parser.add_argument("--a61-oof", type=Path, required=True)
    parser.add_argument("--aggressive-oof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    x3 = _read_oof(args.x3_oof)
    a61 = _read_oof(args.a61_oof)
    if args.aggressive_oof:
        aggressive = _read_oof(args.aggressive_oof)
        aggressive_col = "aggressive_r75_lgb20_pred"
        # align aggressive onto x3 identities
        keys = ["fold", "origin_time", "target", "horizon"]
        m = x3.merge(aggressive[keys + [aggressive_col]].drop_duplicates(keys), on=keys, how="left")
        x3[aggressive_col] = m[aggressive_col]
    else:
        aggressive_col = "aggressive_r75_lgb20_pred"

    if aggressive_col not in x3.columns:
        raise ValueError(f"X3 OOF 缺少 aggressive 列 {aggressive_col}")

    work, merge_audit = _merge_oofs(x3, a61)
    report = _gate_report(work, a61_oof=a61, aggressive_col=aggressive_col)
    report["merge_audit"] = merge_audit
    report["gate_criteria"] = {
        "reproduce_x3_pooled": "5.119696 +/- 0.005pp",
        "reproduce_a61_pooled": "5.195745 +/- 0.005pp",
        "reproduce_safe60_pooled": "5.099520 +/- 0.005pp",
        "vs_a61_min_improvement_pp": 0.050,
        "vs_aggressive_min_improvement_pp": 0.020,
        "bootstrap_probability_better_min": 0.95,
        "worst_fold_regression_max_pp": 0.100,
        "recent5_wins_min": "3/5",
        "future_diff": "exact zero",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(
        {
            "pooled_mape": report["pooled_mape"],
            "improvement_vs_a61_pp": report["improvement_vs_a61_pp"],
            "improvement_vs_aggressive_pp": report["improvement_vs_aggressive_pp"],
            "fold_wins": report["fold_wins_safe60_vs_a61"],
            "recent5_wins_vs_a61": report["recent5_wins_safe60_vs_a61"],
            "worst_fold_regression_vs_a61_pp": report["worst_fold_regression_vs_a61_pp"],
            "bootstrap_vs_a61_p": report["bootstrap_vs_a61"]["probability_candidate_better"],
            "bootstrap_vs_aggressive_p": report["bootstrap_vs_aggressive"]["probability_candidate_better"],
            "linear_identity_max_abs_diff": report["linear_identity_max_abs_diff"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
