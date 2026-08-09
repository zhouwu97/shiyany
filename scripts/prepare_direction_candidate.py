"""生成冻结短长方向外推候选，并附带 OOF 与输入质量收据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gas_forecast.direction_extrapolation import (
    LONG_HORIZONS,
    SHORT_HORIZONS,
    evaluate_direction_policy,
    extrapolate_submission_result,
)
from gas_forecast.submission import package_submission
from gas_forecast.submission_quality import (
    COMPETITION_QUALITY_POLICY,
    prepare_full_matrix_submission_input,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-column", required=True)
    parser.add_argument("--candidate-column", required=True)
    parser.add_argument("--target", default="generator_1")
    parser.add_argument("--short-multiplier", type=float, default=1.6)
    parser.add_argument("--long-multiplier", type=float, default=1.0)
    parser.add_argument("--archive-name", default="咕咕嘎嘎_gas_predict_prelim.zip")
    args = parser.parse_args()

    multipliers = {
        **{horizon: args.short_multiplier for horizon in SHORT_HORIZONS},
        **{horizon: args.long_multiplier for horizon in LONG_HORIZONS},
    }
    baseline = pd.read_csv(args.baseline_result)
    candidate = pd.read_csv(args.candidate_result)
    result, result_report = extrapolate_submission_result(
        baseline,
        candidate,
        target=args.target,
        multipliers=multipliers,
    )
    oof_report = evaluate_direction_policy(
        pd.read_csv(args.oof),
        baseline_column=args.baseline_column,
        candidate_column=args.candidate_column,
        target=args.target,
        multipliers=multipliers,
    )
    quality_input, quality_report = prepare_full_matrix_submission_input(
        pd.read_csv(args.input),
        COMPETITION_QUALITY_POLICY,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "input.csv"
    result_path = args.output_dir / "s_result.csv"
    archive_path = args.output_dir / args.archive_name
    quality_input.to_csv(input_path, index=False, encoding="utf-8")
    result.to_csv(result_path, index=False, encoding="utf-8")
    archive_report = package_submission(input_path, result_path, archive_path)
    sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    report = {
        "strategy": "frozen_two_band_direction_extrapolation",
        "result": result_report,
        "oof": oof_report,
        "input_quality": quality_report,
        "archive": archive_report,
        "archive_sha256": sha256,
        "formal_submission_modified": False,
    }
    report_path = args.output_dir / "direction_candidate_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
