"""自动执行初赛训练期选型、最终训练、预测、校验与打包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.orchestration import (
    SUPPORTED_VERSIONS,
    run_automated_pipeline,
    run_competition_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行不使用测试未来标签的完整自动流水线")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=SUPPORTED_VERSIONS,
        default=list(SUPPORTED_VERSIONS),
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--selection-policy", choices=["pooled_oof", "legacy"], default="pooled_oof"
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument(
        "--selection",
        type=Path,
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--archive",
        type=Path,
    )
    parser.add_argument(
        "--summary",
        type=Path,
    )
    parser.add_argument("--expected-rows", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selection_policy == "pooled_oof":
        summary = run_competition_pipeline(
            args.train_dir,
            args.test_dir,
            versions=args.versions,
            run_dir=args.run_dir,
            jobs=args.jobs,
            max_folds=args.max_folds,
            expected_rows=args.expected_rows,
        )
    else:
        summary = run_automated_pipeline(
            args.train_dir,
            args.test_dir,
            versions=args.versions,
            run_dir=args.run_dir,
            reports_dir=args.reports_dir,
            selection_path=args.selection,
            model_path=args.model,
            output_dir=args.output_dir,
            archive_path=args.archive,
            summary_path=args.summary,
            jobs=args.jobs,
            max_folds=args.max_folds,
            expected_rows=args.expected_rows,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
