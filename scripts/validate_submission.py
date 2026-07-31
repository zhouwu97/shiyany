"""独立校验预测结果格式与弱物理约束。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.submission import validate_submission_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="校验初赛短周期结果")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_submission_frame(pd.read_csv(args.input))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

