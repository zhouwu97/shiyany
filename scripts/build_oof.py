"""为旧 V1/V2/V2.5/V3 生成共享外层折逐行 OOF。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from gas_forecast.config import ForecastConfig, horizon_ridge_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.oof import (
    OOFResult,
    SUPPORTED_LEGACY_MODELS,
    SUPPORTED_OOF_MODELS,
    build_legacy_oof,
    write_oof,
)
from gas_forecast.online import apply_online_calibration_to_oof
from gas_forecast.scoring import score_oof_long


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成逐行外层 OOF 长表")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--versions", nargs="+", choices=SUPPORTED_OOF_MODELS, default=SUPPORTED_LEGACY_MODELS
    )
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--online-base",
        choices=("persistence", *SUPPORTED_OOF_MODELS),
        help="在已有 OOF 上附加在线校准的基础预测列；不指定则不运行在线校准",
    )
    parser.add_argument(
        "--online-modes",
        nargs="+",
        choices=("bias", "gain", "vintage"),
        help="在线校准模式，可同时比较多个模式",
    )
    parser.add_argument(
        "--online-warmup-rows",
        type=int,
        default=0,
        help="每个外层折用于折内 warm-up 但不计分的 origin 行数；0 表示冷启动",
    )
    parser.add_argument("--online-half-life", type=float, default=16.0)
    parser.add_argument("--online-bias-clip", type=float, default=12.0)
    parser.add_argument("--online-vintage-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "legacy_oof")
    output = args.output or run_dir / "oof.csv"
    report = args.report or run_dir / "report.json"
    checkpoint_dir = run_dir / "checkpoints"
    if "horizon_ridge" in args.versions and len(args.versions) != 1:
        raise ValueError("horizon_ridge 只能单独运行，避免与旧版使用不同特征配置混合")
    horizon_ridge_only = tuple(args.versions) == ("horizon_ridge",)
    config = horizon_ridge_forecast_config() if horizon_ridge_only else ForecastConfig()
    dataset = align_tables(args.data_dir, config.feature.frequency)
    prices = sorted(args.data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    if horizon_ridge_only:
        features = build_causal_features(dataset.frame, config.feature, price)
    else:
        # M1 只复现旧 V1/V2/V2.5/V3 报告；增强特征由 run_experiment 单独登记。
        legacy_feature = replace(
            config.feature,
            lags=(1, 2, 4, 8, 16, 32, 96),
            enable_anomaly_features=False,
            enable_physical_features=False,
            enable_long_cycle_features=False,
        )
        features = build_causal_features(dataset.frame, legacy_feature, price)
    result = build_legacy_oof(
        dataset.frame,
        features,
        versions=args.versions,
        config=config,
        max_folds=args.max_folds,
        n_jobs=args.jobs,
        checkpoint_dir=checkpoint_dir,
    )
    if args.online_modes:
        if args.online_base is None:
            raise ValueError("指定 --online-modes 时必须同时指定 --online-base")
        base_column = f"{args.online_base}_pred"
        if base_column not in result.rows.columns:
            raise ValueError(
                f"在线校准基础列不存在: {base_column}；请将对应版本加入 --versions"
            )
        online_rows = result.rows
        online_reports = {}
        for mode in args.online_modes:
            online_rows = apply_online_calibration_to_oof(
                online_rows,
                base_column,
                config.targets,
                config.feature.horizons,
                mode=mode,
                warmup_rows=args.online_warmup_rows,
                half_life=args.online_half_life,
                bias_clip=args.online_bias_clip,
                vintage_weight=args.online_vintage_weight,
            )
            output_column = (
                f"{args.online_base}_online_{mode}_pred"
            )
            warmup_column = f"{output_column}_is_warmup"
            fallback_column = f"{output_column}_is_fallback"
            scored = online_rows.loc[~online_rows[warmup_column]].copy()
            online_reports[output_column.removesuffix("_pred")] = {
                **score_oof_long(scored, output_column),
                "base_column": base_column,
                "mode": mode,
                "warmup_rows_per_fold": args.online_warmup_rows,
                "scored_rows": int(len(scored)),
                "fallback_rows": int(online_rows[fallback_column].sum()),
                "baseline_on_same_scored_rows": score_oof_long(scored, base_column),
            }
        result.report["models"].update(online_reports)
        result.report["online_calibration"] = {
            "base_column": base_column,
            "modes": list(args.online_modes),
            "warmup_rows_per_fold": args.online_warmup_rows,
            "evaluation_mode": (
                "cold_start" if args.online_warmup_rows == 0 else "within_fold_warmup"
            ),
        }
        result = OOFResult(rows=online_rows, report=result.report)
    write_oof(result, output, report)
    write_json(report, result.report)
    finalize_run(
        run_dir,
        {
            "run_type": "oof",
            "stage": "M1",
            "is_smoke": args.max_folds is not None and args.max_folds < 20,
            "outer_folds": len(result.report["folds"]),
            "pooled_mape": float(result.report.get("pooled_mape", 0.0)),
            "report": str(report.relative_to(run_dir)),
            "oof": str(output.relative_to(run_dir)),
        },
    )
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
