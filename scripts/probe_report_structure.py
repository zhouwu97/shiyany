"""Probe the champion report.json structure to see what a calibration audit can use.

Read-only. Prints a schema outline, not the payload.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORT = Path("results/raw/runs/oof/clean_c0_strict_20260801_v2/report.json")


def outline(obj: object, depth: int = 0, max_depth: int = 4, prefix: str = "") -> None:
    pad = "  " * depth
    if depth > max_depth:
        print(f"{pad}{prefix}...")
        return
    if isinstance(obj, dict):
        print(f"{pad}{prefix}dict[{len(obj)}] keys={list(obj)[:14]}")
        for key in list(obj)[:6]:
            outline(obj[key], depth + 1, max_depth, prefix=f"{key}: ")
    elif isinstance(obj, list):
        print(f"{pad}{prefix}list[{len(obj)}]")
        if obj:
            outline(obj[0], depth + 1, max_depth, prefix="[0]: ")
    else:
        text = repr(obj)
        if len(text) > 90:
            text = text[:90] + "..."
        print(f"{pad}{prefix}{type(obj).__name__} = {text}")


def main() -> int:
    if not REPORT.exists():
        print(f"MISSING: {REPORT}")
        return 1
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    print("=" * 70)
    print("TOP-LEVEL KEYS")
    print("=" * 70)
    for key, value in report.items():
        kind = type(value).__name__
        size = len(value) if isinstance(value, (dict, list)) else ""
        print(f"  {key:34s} {kind:6s} {size}")

    for key in ("folds", "stability", "selection", "route_report"):
        if key not in report:
            continue
        print()
        print("=" * 70)
        print(f"OUTLINE: {key}")
        print("=" * 70)
        outline(report[key], max_depth=4)

    # hunt for anything weight-ish anywhere in the tree
    print()
    print("=" * 70)
    print("PATHS CONTAINING weight / branch / calib / horizon")
    print("=" * 70)
    hits: list[str] = []

    def walk(obj: object, path: str = "") -> None:
        if len(hits) > 80:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                low = str(key).lower()
                child = f"{path}.{key}"
                if any(tok in low for tok in ("weight", "branch", "calib", "horizon")):
                    kind = type(value).__name__
                    size = len(value) if isinstance(value, (dict, list)) else value
                    hits.append(f"  {child}  ({kind}) {str(size)[:70]}")
                walk(value, child)
        elif isinstance(obj, list) and obj:
            walk(obj[0], f"{path}[0]")

    walk(report)
    for hit in hits[:80]:
        print(hit)
    if not hits:
        print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
