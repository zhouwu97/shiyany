"""训练并保存模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

from gas_forecast.workflow import train_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="训练煤气发电预测模型")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--version", choices=["v1"], default="v1")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_v1.joblib"))
    args = parser.parse_args()
    train_v1(args.data_dir, args.output)
    print(f"模型已保存: {args.output}")


if __name__ == "__main__":
    main()

