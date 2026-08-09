"""将已冻结、已质量处理的模型输入与结果封装为初赛 ZIP。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 直接执行脚本时优先使用当前工作树，避免导入其他 worktree 的已安装包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CODE = PROJECT_ROOT / "code"
if str(LOCAL_CODE) not in sys.path:
    sys.path.insert(0, str(LOCAL_CODE))

from gas_forecast.submission import (  # noqa: E402
    SUBMISSION_QUALITY_RECEIPT,
    package_submission,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="封装已冻结的初赛提交压缩包")
    parser.add_argument("--input", type=Path, required=True, help="已完成 Q_REFERENCE 的 input.csv")
    parser.add_argument("--result", type=Path, required=True, help="已冻结的 s_result.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quality-receipt",
        type=Path,
        help="Q_REFERENCE 阶段生成的 submission_quality_receipt.json；提供后执行哈希复核",
    )
    parser.add_argument(
        "--result-freeze",
        type=Path,
        help="可选的独立 s_result SHA256 冻结记录；正式链通常由质量收据包含此信息",
    )
    parser.add_argument(
        "--quality-policy",
        choices=("competition", "none"),
        default="none",
        help="旧版兼容参数；打包阶段不会拟合、审计或修复质量策略",
    )
    args = parser.parse_args()
    if args.quality_policy != "none":
        print(
            "提示：--quality-policy 已不在打包阶段生效；请先通过 prepare_submission 生成质量收据。",
            file=sys.stderr,
        )
    if not args.input.is_file():
        raise SystemExit(f"input.csv 不存在: {args.input}")
    if not args.result.is_file():
        raise SystemExit(f"s_result.csv 不存在: {args.result}")
    quality_receipt = args.quality_receipt
    if quality_receipt is None:
        sidecar = args.input.parent / SUBMISSION_QUALITY_RECEIPT
        if sidecar.is_file():
            quality_receipt = sidecar
        else:
            raise SystemExit(
                "缺少 Q_REFERENCE 提交质量收据；请提供 --quality-receipt，"
                "或先运行 scripts/prepare_submission.py 生成已冻结的正式副本。"
            )
    print(
        json.dumps(
            package_submission(
                args.input,
                args.result,
                args.output,
                quality_receipt_path=quality_receipt,
                result_freeze=args.result_freeze,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
