"""输出赛事四表的数据契约审计结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.data import align_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计赛事四表的时间轴和缺失情况")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = align_tables(args.data_dir)
    payload = result.audit.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

