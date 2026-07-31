"""按数据字典旧说明导出 columns/data JSON。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.submission import export_legacy_json


def main() -> None:
    parser = argparse.ArgumentParser(description="生成旧版 JSON 结果，正式提交前以平台要求为准")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_legacy_json(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
