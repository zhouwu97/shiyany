"""比较两次冻结清单是否逐项一致。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.freeze import compare_reproductions


def main() -> None:
    parser = argparse.ArgumentParser(description="比较主运行与干净环境复现产物")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    comparison = compare_reproductions(reference, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if not comparison["identical"]:
        raise SystemExit("复现产物未逐项一致")


if __name__ == "__main__":
    main()
