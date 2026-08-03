"""以固定预测结果生成 Q0/Q1/Q2 提交质量消融包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.submission import package_submission, validate_submission_frame
from gas_forecast.submission_quality import (
    COMPETITION_QUALITY_POLICY,
    policy_with_raw_columns,
    prepare_submission_input,
    raw_columns,
)


def _write_candidate(
    name: str,
    input_frame: pd.DataFrame,
    result_frame: pd.DataFrame,
    output_dir: Path,
    *,
    report: dict[str, object],
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="当前冻结模型的 input.csv")
    parser.add_argument("--result", type=Path, required=True, help="固定 s_result.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_input = pd.read_csv(args.input)
    result = pd.read_csv(args.result)
    validate_submission_frame(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repair_only_policy = policy_with_raw_columns(
        COMPETITION_QUALITY_POLICY,
        raw_columns(source_input),
    )
    q1_input, q1_report = prepare_submission_input(
        source_input,
        repair_only_policy,
        strict=False,
    )
    q2_input, q2_report = prepare_submission_input(
        source_input,
        COMPETITION_QUALITY_POLICY,
    )
    payload = {
        "q0": _write_candidate(
            "Q0_current",
            source_input,
            result,
            args.output_dir,
            report={"policy": "none", "repaired_cells": 0},
        ),
        "q1": _write_candidate(
            "Q1_raw_repair",
            q1_input,
            result,
            args.output_dir,
            report=q1_report,
        ),
        "q2": _write_candidate(
            "Q2_schema_and_repair",
            q2_input,
            result,
            args.output_dir,
            report=q2_report,
        ),
    }
    report_path = args.output_dir / "quality_ablation.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
