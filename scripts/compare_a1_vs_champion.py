"""
compare_a1_vs_champion.py
Compare A1 (competition-weighted L1 residual LGB) vs champion baseline.

Champion: results/raw/runs/oof/clean_c0_strict_20260801_v2/report.json
A1:       results/raw/runs/oof/20260802_072137_a1_weighted_l1/report.json

Screening criteria (dev folds only, blind fold excluded):
  - gen1 mean improvement >= 0.02pp
  - >= 3/5 dev folds win on gen1
  - no blowup (any fold MAPE > 2x champion)

Full dev criteria (after screening passes):
  - pooled improvement >= 0.015-0.02pp
  - dev win rate > 50% (ideally > 60%)
  - recent folds (dev_15..dev_19) not degraded
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

CHAMP_PATH = ROOT / "results/raw/runs/oof/clean_c0_strict_20260801_v2/report.json"
A1_PATH    = ROOT / "results/raw/runs/oof/20260802_072137_a1_weighted_l1/report.json"


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def pp(val: float) -> str:
    return f"{val*100:.4f}%"


def main() -> None:
    if not A1_PATH.exists():
        print(f"A1 report not found: {A1_PATH}")
        sys.exit(1)

    champ = load(CHAMP_PATH)
    a1    = load(A1_PATH)

    cand = "v2_v3_target_reconciled"

    champ_cr = champ["candidate_reports"][cand]
    a1_cr    = a1["candidate_reports"][cand]

    # ── Pooled MAPE ──────────────────────────────────────────────────────────
    champ_pooled = champ_cr["pooled_mape"]
    a1_pooled    = a1_cr["pooled_mape"]
    delta_pooled = (champ_pooled - a1_pooled) * 100

    print("=" * 60)
    print("POOLED MAPE")
    print(f"  champion : {pp(champ_pooled)}")
    print(f"  A1       : {pp(a1_pooled)}")
    print(f"  delta    : {delta_pooled:+.4f}pp  ({'IMPROVED' if delta_pooled > 0 else 'DEGRADED'})")

    # ── By-target ────────────────────────────────────────────────────────────
    print()
    print("BY TARGET")
    for tgt in ("generator_1", "generator_all"):
        c_t = champ_cr.get("by_target", {}).get(tgt)
        a_t = a1_cr.get("by_target", {}).get(tgt)
        if c_t is None or a_t is None:
            print(f"  {tgt}: missing in report")
            continue
        d = (c_t - a_t) * 100
        print(f"  {tgt:15s}  champ={pp(c_t)}  a1={pp(a_t)}  Δ={d:+.4f}pp")

    # ── By-fold ───────────────────────────────────────────────────────────────
    champ_folds = champ_cr.get("by_fold", {})
    a1_folds    = a1_cr.get("by_fold", {})

    fold_names = sorted(champ_folds.keys())
    print()
    print("BY FOLD")

    dev_folds   = [n for n in fold_names if not n.startswith("blind")]
    blind_folds = [n for n in fold_names if n.startswith("blind")]

    wins = 0
    losses = 0
    deltas = []
    blowup = False

    for name in dev_folds:
        c_v = champ_folds.get(name)
        a_v = a1_folds.get(name)
        if c_v is None or a_v is None:
            print(f"  {name}: missing")
            continue
        d = (c_v - a_v) * 100
        deltas.append(d)
        marker = "✓" if d > 0 else "✗"
        if d > 0:
            wins += 1
        else:
            losses += 1
        if a_v > champ_pooled * 2:
            blowup = True
            marker += " BLOWUP"
        print(f"  {name:12s}  champ={pp(c_v)}  a1={pp(a_v)}  Δ={d:+.4f}pp  {marker}")

    total_dev = wins + losses
    win_rate  = wins / total_dev if total_dev > 0 else 0
    mean_d    = sum(deltas) / len(deltas) if deltas else 0

    print()
    print("BLIND FOLDS (for reference — not used for selection)")
    for name in blind_folds:
        c_v = champ_folds.get(name)
        a_v = a1_folds.get(name)
        if c_v is None or a_v is None:
            print(f"  {name}: missing")
            continue
        d = (c_v - a_v) * 100
        marker = "✓" if d > 0 else "✗"
        print(f"  {name:12s}  champ={pp(c_v)}  a1={pp(a_v)}  Δ={d:+.4f}pp  {marker}")

    # ── Stability from report ─────────────────────────────────────────────────
    stab_champ = champ.get("stability", {})
    stab_a1    = a1.get("stability", {})

    print()
    print("STABILITY (from report.json)")
    for key in ("development_fold_win_rate",):
        c_s = stab_champ.get(key)
        a_s = stab_a1.get(key)
        print(f"  {key}: champ={c_s}  a1={a_s}")

    rec_champ = stab_champ.get("development_recent_folds", {})
    rec_a1    = stab_a1.get("development_recent_folds", {})
    if rec_champ or rec_a1:
        print("  recent folds (dev_15..dev_19):")
        all_rec = sorted(set(list(rec_champ.keys()) + list(rec_a1.keys())))
        for k in all_rec:
            c_v = rec_champ.get(k)
            a_v = rec_a1.get(k)
            d_str = ""
            if c_v is not None and a_v is not None:
                d_str = f"  Δ={((c_v - a_v)*100):+.4f}pp"
            print(f"    {k}: champ={c_v}  a1={a_v}{d_str}")

    # ── Screening verdict ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SCREENING VERDICT (dev folds)")
    print(f"  mean gen1 improvement : {mean_d:+.4f}pp  (need >= +0.02pp)")
    print(f"  dev win rate          : {wins}/{total_dev} = {win_rate:.0%}  (need >= 3/5)")
    print(f"  blowup detected       : {blowup}  (must be False)")

    pass_screen = (mean_d >= 0.02) and (wins >= 3) and (not blowup)
    print()
    if pass_screen:
        print("✅  SCREENING PASSED — proceed to full dev eval")
        if delta_pooled >= 1.5:  # 0.015pp
            print("✅  FULL DEV PASSED — A1 is a valid champion candidate")
        else:
            print("⚠️   FULL DEV: pooled improvement below 0.015pp threshold")
    else:
        reasons = []
        if mean_d < 0.02:
            reasons.append(f"mean improvement {mean_d:+.4f}pp < 0.02pp")
        if wins < 3:
            reasons.append(f"win rate {wins}/{total_dev} < 3/5")
        if blowup:
            reasons.append("blowup in at least one fold")
        print("❌  SCREENING FAILED: " + "; ".join(reasons))


if __name__ == "__main__":
    main()
