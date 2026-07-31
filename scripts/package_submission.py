"""将宽表结果打包为只含 result.csv 的初赛 ZIP。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.submission import package_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="生成初赛提交压缩包")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_submission(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

