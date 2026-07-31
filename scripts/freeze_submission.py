"""生成不包含赛事原始数据的本地冻结清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.freeze import build_freeze_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结模型与提交产物哈希")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("requirements-lock.txt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_freeze_manifest(
        args.model, args.result, args.archive, args.selection, args.lock
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
