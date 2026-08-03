"""运行 RichResidual 的短长权重网格和严格前向四组 horizon 路由。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.horizon_blend import (
    build_two_band_blend_grid,
    time_ordered_four_band_router,
)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def _parse_weights(value: str) -> tuple[float, ...]:
    weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not weights:
        raise ValueError("权重列表不能为空")
    return weights


def _best_grid_candidate(report: dict[str, object]) -> dict[str, object] | None:
    models = report.get("models", {})
    if not isinstance(models, dict):
        return None
    eligible = [
        {
            "candidate": name,
            "pooled_difference": float(metric["pooled_difference"]),
            "generator_1_difference": float(metric["generator_1_difference"]),
            "fold_wins": int(metric["fold_wins"]),
            "recent5_wins": sum(
                float(value) < 0.0
                for value in metric["recent_5_folds_difference"].values()
            ),
        }
        for name, metric in models.items()
        if isinstance(metric, dict)
        and float(metric["pooled_difference"]) < 0.0
        and float(metric["generator_1_difference"]) <= 0.0
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            float(item["pooled_difference"]),
            float(item["generator_1_difference"]),
            str(item["candidate"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline-column", required=True)
    parser.add_argument("--branch-column", required=True)
    parser.add_argument(
        "--comparison-column",
        required=True,
        help="候选必须相对此列不退化；P3 传入已冻结的 P2 blend",
    )
    parser.add_argument("--mode", choices=("grid", "route"), required=True)
    parser.add_argument("--scope", choices=("screening", "development"), default="development")
    parser.add_argument("--short-weights", default="0.10,0.15,0.20")
    parser.add_argument("--long-weights", default="0.20,0.25,0.30,0.35")
    parser.add_argument("--short-weight", type=float)
    parser.add_argument("--long-weight", type=float)
    parser.add_argument("--min-history-rows", type=int, default=128)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_frame(args.input)
    run_dir = args.run_dir or new_run_dir("results", "experiment_target_horizon_blend")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "grid":
        result = build_two_band_blend_grid(
            rows,
            baseline_column=args.baseline_column,
            branch_column=args.branch_column,
            comparison_column=args.comparison_column,
            short_weights=_parse_weights(args.short_weights),
            long_weights=_parse_weights(args.long_weights),
            scope=args.scope,
        )
        report: dict[str, object] = dict(result.report)
        selection = _best_grid_candidate(report)
        report["selection"] = {
            "best_candidate": selection,
            "selection_used_blind": False,
            "rule": "仅从预注册 3×4 两段权重中选择，并要求相对 P2 不退化。",
        }
        write_json(run_dir / "selection.json", report["selection"])
    else:
        if args.short_weight is None or args.long_weight is None:
            raise ValueError("route 模式必须提供 --short-weight 和 --long-weight")
        result = time_ordered_four_band_router(
            rows,
            baseline_column=args.baseline_column,
            branch_column=args.branch_column,
            comparison_column=args.comparison_column,
            short_weight=args.short_weight,
            long_weight=args.long_weight,
            scope=args.scope,
            min_history_rows=args.min_history_rows,
        )
        report = dict(result.report)
        report["route_trace"] = result.route_trace
        write_json(run_dir / "route_trace.json", {"route_trace": result.route_trace})
    oof_path = run_dir / "oof.csv"
    report_path = run_dir / "report.json"
    result.rows.to_csv(oof_path, index=False, encoding="utf-8")
    write_json(report_path, report)
    metrics = report["models"]
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("horizon blend 没有产生候选指标")
    best_mape = min(float(item["pooled_mape"]) for item in metrics.values())
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": f"target_horizon_{args.mode}",
            "scope": args.scope,
            "is_smoke": args.scope == "screening",
            "blind_included": False,
            "formal_candidate": False,
            "pooled_mape": best_mape,
            "baseline": args.comparison_column,
            "input": str(args.input.resolve()),
            "report": "report.json",
            "oof": "oof.csv",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "mode": args.mode,
                "best_mape": best_mape,
                "selection": report.get("selection"),
            },
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
    )


if __name__ == "__main__":
    main()
