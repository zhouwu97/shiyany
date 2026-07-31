"""训练并保存模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.workflow import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="训练煤气发电预测模型")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--version", choices=["v1", "v2", "v25", "v3", "auto"], default="v1"
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_v1.joblib"))
    args = parser.parse_args()
    version = args.version
    if version == "auto":
        if args.selection is None:
            parser.error("--version auto 必须同时提供 --selection")
        decision = json.loads(args.selection.read_text(encoding="utf-8"))
        version = decision["selected_version"]
    train_model(args.data_dir, args.output, version)
    print(f"{version} 模型已保存: {args.output}")


if __name__ == "__main__":
    main()
