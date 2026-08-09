"""以固定预测结果生成 Q0/Q1/Q2/Q3 提交质量消融包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.submission import package_submission, validate_submission_frame
from gas_forecast.submission_quality import (
    COMPETITION_QUALITY_POLICY,
    fit_quality_policy,
    policy_with_raw_columns,
    prepare_full_matrix_submission_input,
    raw_columns,
    transform_submission_input,
)


def _write_candidate(
    name: str,
    input_frame: pd.DataFrame,
    result_frame: pd.DataFrame,
    output_dir: Path,
    *,
    report: dict[str, object],
    frozen_policy: dict[str, object],
) -> dict[str, object]:
    destination = output_dir / name
    destination.mkdir(parents=True, exist_ok=True)
    input_path = destination / "input.csv"
    result_path = destination / "s_result.csv"
    archive_path = destination / "submission.zip"
    input_frame.to_csv(input_path, index=False, encoding="utf-8")
    result_frame.to_csv(result_path, index=False, encoding="utf-8")
    archive = package_submission(input_path, result_path, archive_path)
    return {
        "input": str(input_path.resolve()),
        "result": str(result_path.resolve()),
        "archive": str(archive_path.resolve()),
        "archive_summary": archive,
        "quality": report,
        "frozen_policy": frozen_policy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="当前冻结模型的 input.csv")
    parser.add_argument("--result", type=Path, required=True, help="固定 s_result.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--training-input",
        type=Path,
        help="仅含训练期生产观测的 input.csv；未提供时只允许显式演示模式",
    )
    parser.add_argument(
        "--train-end",
        help="训练期最后一个可用 origin；与 --training-input 一起冻结统计",
    )
    parser.add_argument(
        "--allow-fit-on-scoring-input",
        action="store_true",
        help="仅用于历史 Q0/Q1/Q2/Q3 复现实验，不可作为生产质量策略",
    )
    args = parser.parse_args()

    source_input = pd.read_csv(args.input)
    result = pd.read_csv(args.result)
    validate_submission_frame(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.training_input is None:
        if not args.allow_fit_on_scoring_input:
            raise SystemExit("必须提供 --training-input；历史复现请显式传 --allow-fit-on-scoring-input")
        training_input = source_input
        fit_mode = "demo_fit_on_scoring_input"
    else:
        training_input = pd.read_csv(args.training_input)
        fit_mode = "production_training_period"

    repair_only_policy = policy_with_raw_columns(
        COMPETITION_QUALITY_POLICY,
        raw_columns(training_input),
    )
    q1_policy = fit_quality_policy(training_input, repair_only_policy, train_end=args.train_end)
    q2_policy = fit_quality_policy(
        training_input,
        COMPETITION_QUALITY_POLICY,
        train_end=args.train_end,
    )
    q1_input, q1_report = transform_submission_input(source_input, q1_policy, strict=False)
    q2_input, q2_report = transform_submission_input(source_input, q2_policy)
    q3_input, q3_report = prepare_full_matrix_submission_input(source_input, COMPETITION_QUALITY_POLICY)
    q1_report["fit_mode"] = fit_mode
    q2_report["fit_mode"] = fit_mode
    q3_report["fit_mode"] = fit_mode
    q3_report["production_eligible"] = False
    payload = {
        "q0": _write_candidate(
            "Q0_current",
            source_input,
            result,
            args.output_dir,
            report={"policy": "none", "repaired_cells": 0},
            frozen_policy={"fit_mode": fit_mode, "applied": False},
        ),
        "q1": _write_candidate(
            "Q1_raw_repair",
            q1_input,
            result,
            args.output_dir,
            report=q1_report,
            frozen_policy=q1_policy.to_dict(),
        ),
        "q2": _write_candidate(
            "Q2_schema_and_repair",
            q2_input,
            result,
            args.output_dir,
            report=q2_report,
            frozen_policy=q2_policy.to_dict(),
        ),
        "q3": _write_candidate(
            "Q3_full_matrix_quality",
            q3_input,
            result,
            args.output_dir,
            report=q3_report,
            frozen_policy={"fit_mode": fit_mode, "full_matrix": "historical_ablation_only"},
        ),
    }
    report_path = args.output_dir / "quality_ablation.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
