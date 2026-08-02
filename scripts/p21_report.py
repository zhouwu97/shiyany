"""P2-1 汇总 — 从 p21 报告 JSON 渲染筛选判定与 horizon 结构。

用法: python scripts/p21_report.py --report <run>/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HORIZONS_MIN = (15, 30, 45, 60, 75, 90, 105, 120)
SCREEN_FOLDS = ("dev_01", "dev_05", "dev_10", "dev_15", "dev_19")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    base = report["baseline"]
    base_mape = report["baseline_pooled_mape"]
    print(f"baseline: {base}  pooled={base_mape:.6f}")
    print()
    for name, c in report["candidates"].items():
        d = c["delta_pp"]
        wins = c["screening_wins"]
        verdict = "SCREEN-PASS" if (d <= -0.015 and wins >= 3) else ("borderline" if d <= -0.01 else "no-pass")
        print(f"{name}: pooled={c['pooled_mape']:.6f}  Δ={d:+.4f}pp  wins={wins}/5  -> {verdict}")
        byh = c["by_horizon_delta_pp"]
        short = [byh[f"t+{h}"] for h in (15, 30, 45)]
        med = [byh[f"t+{h}"] for h in (60, 75, 90)]
        long = [byh[f"t+{h}"] for h in (105, 120)]
        print(f"   Δ by horizon (short t15-45 / med t60-90 / long t105-120):")
        print(f"     short={sum(short)/3:+.4f}  med={sum(med)/3:+.4f}  long={sum(long)/2:+.4f}")
        horizon_row = " ".join(f"{byh['t+' + str(h)]:+.3f}" for h in HORIZONS_MIN)
        print(f"     t+15..t+120: {horizon_row}")
        print(f"   by fold: {' '.join(f'{k}:{v:+.3f}' for k, v in sorted(c['by_fold_delta_pp'].items()))}")
        print()


if __name__ == "__main__":
    main()
