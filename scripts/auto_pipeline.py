"""自动执行初赛训练期选型、最终训练、预测、校验与打包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.orchestration import SUPPORTED_VERSIONS, run_automated_pipeline


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
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--reports-dir", type=Path, default=Path("results/raw/auto"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/raw/model_selection_auto.json"),
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("submissions/final"))
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("submissions/teamname_gas_predict_prelim.zip"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/raw/auto_pipeline.json"),
    )
    parser.add_argument("--expected-rows", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_automated_pipeline(
        args.train_dir,
        args.test_dir,
        versions=args.versions,
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
