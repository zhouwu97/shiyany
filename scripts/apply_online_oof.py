"""在已生成的 OOF 长表上运行严格因果在线校准。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.online import apply_online_calibration_to_oof
from gas_forecast.scoring import score_oof_long


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对已有 OOF 长表执行冷启动/折内 warm-up 在线校准"
    )
    parser.add_argument("--input", type=Path, required=True, help="已有 OOF CSV")
    parser.add_argument("--base-column", required=True, help="基础预测列，例如 v1_pred")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("bias", "gain", "vintage"),
        default=("bias", "gain", "vintage"),
    )
    parser.add_argument("--warmup-rows", type=int, default=0)
    parser.add_argument("--half-life", type=float, default=16.0)
    parser.add_argument("--bias-clip", type=float, default=12.0)
    parser.add_argument("--vintage-weight", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--run-dir", type=Path, help="可选的实验运行目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.input, parse_dates=["origin_time", "train_end"])
    targets = tuple(sorted(rows["target"].dropna().unique().tolist()))
    horizons = tuple(sorted({int(value) // 15 for value in rows["horizon"].dropna().unique()}))
    if not targets or not horizons:
        raise ValueError("OOF 长表没有可用的 target/horizon")
    if f"{args.base_column}" not in rows.columns:
        raise ValueError(f"基础预测列不存在: {args.base_column}")

    run_dir = args.run_dir
    if run_dir is None and args.output is None and args.report is None:
        run_dir = new_run_dir("results/raw/runs", "online_oof")
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (
        run_dir / "oof_online.csv"
        if run_dir is not None
        else args.input.with_name(f"{args.input.stem}_online.csv")
    )
    report_path = args.report or (
        run_dir / "report.json" if run_dir is not None else output.with_suffix(".json")
    )
    reports: dict[str, object] = {}
    for mode in args.modes:
        rows = apply_online_calibration_to_oof(
            rows,
            args.base_column,
            targets,
            horizons,
            mode=mode,
            warmup_rows=args.warmup_rows,
            half_life=args.half_life,
            bias_clip=args.bias_clip,
            vintage_weight=args.vintage_weight,
        )
        output_column = f"{args.base_column.removesuffix('_pred')}_online_{mode}_pred"
        warmup_column = f"{output_column}_is_warmup"
        fallback_column = f"{output_column}_is_fallback"
        scored = rows.loc[~rows[warmup_column]].copy()
        reports[output_column.removesuffix("_pred")] = {
            **score_oof_long(scored, output_column),
            "base_column": args.base_column,
            "mode": mode,
            "warmup_rows_per_fold": args.warmup_rows,
            "scored_rows": int(len(scored)),
            "fallback_rows": int(rows[fallback_column].sum()),
            "baseline_on_same_scored_rows": score_oof_long(scored, args.base_column),
            "evaluation_mode": (
                "cold_start" if args.warmup_rows == 0 else "within_fold_warmup"
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False, encoding="utf-8")
    write_json(
        report_path,
        {
            "input": str(args.input),
            "output": str(output),
            "base_column": args.base_column,
            "targets": list(targets),
            "horizons": list(horizons),
            "warmup_rows_per_fold": args.warmup_rows,
            "evaluation_mode": (
                "cold_start" if args.warmup_rows == 0 else "within_fold_warmup"
            ),
            "models": reports,
        },
    )
    if run_dir is not None:
        finalize_run(
            run_dir,
            {
                "run_type": "oof",
                "stage": "P2_online",
                "is_smoke": False,
                "input": str(args.input),
                "output": str(output.relative_to(run_dir) if output.is_relative_to(run_dir) else output),
                "report": str(report_path.relative_to(run_dir) if report_path.is_relative_to(run_dir) else report_path),
                "base_column": args.base_column,
                "warmup_rows_per_fold": args.warmup_rows,
                "evaluation_mode": (
                    "cold_start" if args.warmup_rows == 0 else "within_fold_warmup"
                ),
                "models": {
                    name: report["pooled_mape"] for name, report in reports.items()
                },
            },
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
