"""输出赛事四表的数据契约审计结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.data import align_tables
from gas_forecast.experiments import new_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计赛事四表的时间轴和缺失情况")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "data_audit")
    output = args.output or run_dir / "report.json"
    result = align_tables(args.data_dir)
    payload = result.audit.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
