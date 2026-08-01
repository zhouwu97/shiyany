"""比较两次冻结清单是否逐项一致。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.freeze import compare_reproductions
from gas_forecast.experiments import new_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="比较主运行与干净环境复现产物")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "compare_reproduction")
    output = args.output or run_dir / "report.json"
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    comparison = compare_reproductions(reference, candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if not comparison["identical"]:
        raise SystemExit("复现产物未逐项一致")


if __name__ == "__main__":
    main()
