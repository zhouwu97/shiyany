"""显示当前正式最优模型和唯一提交文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="显示 results/best 当前正式提交信息")
    parser.add_argument("--best-dir", type=Path, default=Path("results/best"))
    args = parser.parse_args()
    summary_path = args.best_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"尚未建立正式 best：{summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mape = float(summary["pooled_mape"])
    print("当前正式最优模型")
    print(f"模型：{summary.get('candidate', summary.get('stage', 'unknown'))}")
    print(f"离线 pooled MAPE：{mape:.4%}")
    print(f"离线预测得分：{100 * (1 - mape):.4f}")
    print(f"来源运行：{summary.get('source_run', 'unknown')}")
    delivery_summary = Path("提交这个/summary.json")
    submission = Path("提交这个/咕咕嘎嘎_gas_predict_prelim.zip")
    if delivery_summary.exists():
        delivery = json.loads(delivery_summary.read_text(encoding="utf-8"))
        submission = Path(str(delivery.get("submission", submission)))
    if not submission.exists():
        submission = args.best_dir / "submission.zip"
    print(f"唯一提交文件：{submission.resolve()}")


if __name__ == "__main__":
    main()
