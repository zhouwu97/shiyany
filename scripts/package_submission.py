"""将模型输入与宽表结果打包为初赛 ZIP。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.submission import package_submission
from gas_forecast.submission_quality import COMPETITION_QUALITY_POLICY


def main() -> None:
    parser = argparse.ArgumentParser(description="生成初赛提交压缩包")
    parser.add_argument("--input", type=Path, required=True, help="模型实际使用的 input.csv")
    parser.add_argument("--result", type=Path, required=True, help="预测结果 s_result.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quality-policy",
        choices=("competition", "none"),
        default="competition",
        help="competition 会执行初赛 raw schema 与异常质量门槛",
    )
    args = parser.parse_args()
    policy = COMPETITION_QUALITY_POLICY if args.quality_policy == "competition" else None
    print(
        json.dumps(
            package_submission(
                args.input,
                args.result,
                args.output,
                quality_policy=policy,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
