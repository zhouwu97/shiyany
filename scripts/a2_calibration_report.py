"""A2 汇总 — 从 a2_metrics.json 生成最终报告 + A3 方向门评估。

用法: python scripts/a2_calibration_report.py --metrics results/runs/<stamp>/a2_metrics.json --out results/runs/<stamp>/a2_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BRANCH_NAMES = ("persistence", "ridge", "recent", "gas", "lgb_residual")
HORIZONS_MIN = (15, 30, 45, 60, 75, 90, 105, 120)


def pp(value, suffix="%") -> str:
    return f"{value:.4f}{suffix}" if value is not None else "n/a"


def render_table(metrics: dict, dev_folds: list[str], out: list[str]) -> None:
    """每 fold x horizon 一行: blend | best branch | regret | calib w | next best."""
    header = (
        "| fold | horizon | blend | best branch | regret | persistence w | lgb w "
        "| next best branch |"
    )
    sep = "|------|---------|-------|-------------|--------|---------------|-------|------------------|"
    out.append(header)
    out.append(sep)
    for fold in dev_folds:
        for h in range(8):
            c = metrics["per_cell"][fold][str(h)]
            branch = c["branch_mape"]
            best = min(branch, key=branch.get)
            second = sorted(branch, key=branch.get)[1]
            w = c["calib_weights"]
            out.append(
                f"| {fold} | t+{HORIZONS_MIN[h]} | {pp(c['blend_mape'])} "
                f"| {best} {pp(c['best_branch_mape'])} | {pp(c['regret_pp'], 'pp')} "
                f"| {w[0]:.3f} | {w[4]:.3f} | {second} {pp(branch[second])} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    folds = [f for f in metrics["folds"]]
    dev_folds = [f for f in folds if f != "blind"]

    out: list[str] = []
    out.append("# A2 — simplex calibration stability audit (generator_1 / v2)\n")
    out.append(f"Rule: conclusions from development folds only; blind observed only.\n")

    out.append("## Per fold × horizon table (dev folds)\n")
    render_table(metrics, dev_folds, out)

    out.append("\n## Oracle gap (dev folds)\n")
    out.append(
        "| fold | horizon | calib blend | best branch | oracle | calib→oracle gap | best→oracle gap |"
    )
    out.append("|------|---------|-------------|-------------|--------|------------------|-----------------|")
    for fold in dev_folds:
        for h in range(8):
            og = metrics["oracle_gap"][fold][str(h)]
            out.append(
                f"| {fold} | t+{HORIZONS_MIN[h]} | {pp(og['calib_blend_pp'])} "
                f"| {pp(og['best_branch_pp'])} | {pp(og['oracle_pp'])} "
                f"| {pp(og['gap_calib_vs_oracle_pp'], 'pp')} "
                f"| {pp(og['gap_best_vs_oracle_pp'], 'pp')} |"
            )

    s1 = metrics["s1_blend_vs_best"]
    out.append("\n## Summary statistics\n")
    out.append(
        f"- **S1 (blend vs best single branch):** {s1['fraction']:.1%} of dev cells "
        f"({s1['blend_beats_best_branch_cells']}/{s1['total_dev_cells']})"
    )
    s2 = metrics["s2_persistence_weight_advantage"]["persistence"]
    s3 = metrics["s3_lgb_weight_advantage"]["lgb_residual"]
    out.append(
        f"- **S2 (persistence weight → next-fold advantage):** spearman="
        f"{s2['spearman_weight_vs_next_advantage']}, n={s2['n_pairs']}"
    )
    out.append(
        f"- **S3 (lgb weight → next-fold advantage):** spearman="
        f"{s3['spearman_weight_vs_next_advantage']}, n={s3['n_pairs']}"
    )
    turnover = metrics["weight_turnover"]
    out.append(
        f"- **Weight turnover (adjacent-fold L1):** mean across horizons "
        f"{pp(metrics['weight_turnover_mean_across_horizons'])}; "
        + "; ".join(
            f"t+{HORIZONS_MIN[h]} mean={pp(turnover[str(h)]['mean'])}"
            for h in range(8)
        )
    )

    out.append("\n## A3 direction gate\n")
    # 阈值按 A2 授权口径: 任一命中即 "rebuild calibration"
    regret_cells = [metrics["per_cell"][f][str(h)] for f in dev_folds for h in range(8)]
    regret_positive_frac = sum(1 for c in regret_cells if c["regret_pp"] > 0) / len(regret_cells)
    regret_trigger = regret_positive_frac > 0.40

    # 条件 2: persistence 平均权重高但下一 fold 优势相关 <= 0
    weight_drift = metrics["weight_drift"]
    persistence_mean_weight = float(np.mean(
        [weight_drift[str(h)]["mean"][0] for h in range(8)]
    ))
    s2_rho = s2["spearman_weight_vs_next_advantage"]
    persistence_trigger = (
        persistence_mean_weight > 0.5
        and (s2_rho is None or s2_rho <= 0)
    )

    # 条件 3: 高 weight turnover（相邻 fold L1 均值）
    turnover_mean = metrics["weight_turnover_mean_across_horizons"]
    turnover_trigger = turnover_mean is not None and turnover_mean > 0.5

    # 条件 4: oracle 明显优于当前 calibration, 且最佳单分支未被降级
    oracle_gaps = [metrics["oracle_gap"][f][str(h)] for f in dev_folds for h in range(8)]
    gap_calib_mean = float(np.mean([og["gap_calib_vs_oracle_pp"] for og in oracle_gaps]))
    gap_best_mean = float(np.mean([og["gap_best_vs_oracle_pp"] for og in oracle_gaps]))
    oracle_trigger = gap_calib_mean > 0.5 and gap_best_mean < 0.5

    triggers = {
        "regret_positive_fraction": round(regret_positive_frac, 4),
        "trigger_1_regret_gt_40pct": regret_trigger,
        "persistence_mean_weight": round(float(persistence_mean_weight), 4),
        "persistence_next_adv_rho": s2_rho,
        "trigger_2_persist_w_high_but_adv_le0": persistence_trigger,
        "turnover_mean": turnover_mean,
        "trigger_3_turnover_gt_0.5": turnover_trigger,
        "mean_gap_calib_vs_oracle_pp": round(gap_calib_mean, 4),
        "mean_gap_best_vs_oracle_pp": round(gap_best_mean, 4),
        "trigger_4_oracle_beats_calib_but_not_best": oracle_trigger,
        "gate_hits_rebuild_calibration": any(
            [regret_trigger, persistence_trigger, turnover_trigger, oracle_trigger]
        ),
    }
    for key, value in triggers.items():
        out.append(f"- **{key}:** {value}")

    out.append("\n## A3 conclusion\n")
    if triggers["gate_hits_rebuild_calibration"]:
        out.append("**Direction: rebuild calibration.** At least one gate hit. "
                   "The gap is in weight estimation, not branch strength.")
    else:
        out.append("**Direction: continue strengthening the LGB branch.** "
                   "No calibration gate hit; branches are the limiting factor.")

    out.append("\n> NOTE: numbers are generated from the instrumented A2 run. "
               "The A3 decision is made by the operator using these gates, "
               "on development folds only.")
    out.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
