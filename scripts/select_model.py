"""从训练期滚动报告生成可审计的版本选择文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.selection import choose_version


def main() -> None:
    parser = argparse.ArgumentParser(description="选择通过滚动与盲折门槛的最高模型版本")
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path)
    parser.add_argument("--v25", type=Path)
    parser.add_argument("--v3", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/raw/model_selection.json"))
    args = parser.parse_args()

    reports = {}
    for version in ("v1", "v2", "v25", "v3"):
        path = getattr(args, version)
        if path:
            reports[version] = json.loads(path.read_text(encoding="utf-8"))
    decision = choose_version(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
