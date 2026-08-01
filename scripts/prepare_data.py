"""生成基础合规预处理报告与可复现对齐产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.preprocessing import build_preprocessing_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行去重、连续性、缺失和异常基础审计")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-frame", type=Path)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "preprocess")
    output_frame = args.output_frame or run_dir / "aligned.csv"
    output_report = args.output_report or run_dir / "report.json"
    config = ForecastConfig()
    dataset = align_tables(args.data_dir, config.feature.frequency)
    audit = build_preprocessing_audit(dataset.frame, frequency=config.feature.frequency)
    output_frame.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.reset_index().to_csv(output_frame, index=False, encoding="utf-8")
    payload = {
        "source_audit": dataset.audit.to_dict(),
        "preprocessing_audit": audit.to_dict(),
        "output_frame": str(output_frame),
        "raw_values_preserved": True,
        "duplicate_policy": "keep_last_per_source_table",
        "continuity_policy": "complete_15min_grid",
        "missing_policy": "preserve_nan_and_emit_flags",
        "anomaly_policy": "causal_raw_clean_flag_dual_channel",
    }
    write_json(output_report, payload)
    finalize_run(
        run_dir,
        {
            "run_type": "audit",
            "stage": "preprocess",
            "passed": True,
            "report": str(output_report.relative_to(run_dir)),
            "aligned_frame": str(output_frame.relative_to(run_dir)),
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
