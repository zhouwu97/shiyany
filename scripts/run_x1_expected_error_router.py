"""运行 X1 Dynamic Expected-Error Router 并输出完整评估报告。

输入：P3 集成 OOF（58,368 行）+ 可选 X3（A57）CatBoost OOF。路由严格按
时间前向 cross-fit：held fold 的期望误差模型只用更早折训练。输出 routed
OOF、逐折选择轨迹、覆盖率与稳定门禁报告。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from gas_forecast.x1_expected_error_router import (  # noqa: E402
    DEFAULT_CONFIDENCE_MIN_PP,
    DEFAULT_MIN_HISTORY_FOLDS,
    build_x1_report,
    evaluate_x1_result,
    load_x1_oof,
    route_expected_error,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integration-oof",
        type=Path,
        required=True,
        help="P3 集成 OOF（integration/oof.csv 或 oof.parquet）",
    )
    parser.add_argument(
        "--x3-oof",
        type=Path,
        default=None,
        help="X3 CatBoost（A57）OOF；默认取 a57b_residual_a51_cat10_pred 列",
    )
    parser.add_argument(
        "--x3-column",
        default="a57b_residual_a51_cat10_pred",
        help="X3 OOF 中的预测列名",
    )
    parser.add_argument(
        "--current-value-oof",
        type=Path,
        default=None,
        help="P1 CausalRolling OOF（提供 current_value 单元格级因果特征）",
    )
    parser.add_argument("--mode", choices=("prior", "lightgbm"), default="prior")
    parser.add_argument("--confidence-min-pp", type=float, default=DEFAULT_CONFIDENCE_MIN_PP)
    parser.add_argument("--blend-top", type=int, default=2)
    parser.add_argument(
        "--min-history-folds",
        type=int,
        default=None,
        help="历史折下限；缺省时 prior=1、lightgbm=3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    start = time.monotonic()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    def progress(message: str, elapsed_from: float | None = None) -> float:
        suffix = f"  [{time.monotonic() - elapsed_from:.1f}s]" if elapsed_from else ""
        print(f"[X1 {time.strftime('%H:%M:%S')}] {message}{suffix}", flush=True)
        return time.monotonic()

    timer = progress("加载 P3 集成 OOF 与 X3 CatBoost OOF")
    rows = load_x1_oof(
        args.integration_oof,
        x3_oof=args.x3_oof,
        x3_column=args.x3_column,
        current_value_oof=args.current_value_oof,
    )
    timer = progress(f"OOF 就绪：{len(rows)} 行 × {len(rows.columns)} 列", timer)

    timer = progress(f"路由（mode={args.mode}, conf={args.confidence_min_pp}pp, top={args.blend_top}）")
    min_history_folds = (
        args.min_history_folds
        if args.min_history_folds is not None
        else (1 if args.mode == "prior" else DEFAULT_MIN_HISTORY_FOLDS)
    )
    result = route_expected_error(
        rows,
        confidence_min_pp=args.confidence_min_pp,
        min_history_folds=min_history_folds,
        seed=args.seed,
        mode=args.mode,
        blend_top=args.blend_top,
    )
    timer = progress("路由完成，计算评估与稳定门禁", timer)
    evaluation = evaluate_x1_result(result)
    report = build_x1_report(result, evaluation)
    report["settings"] = {
        "mode": args.mode,
        "confidence_min_pp": args.confidence_min_pp,
        "blend_top": args.blend_top,
        "min_history_folds": min_history_folds,
        "seed": args.seed,
        "x3_oof": None if args.x3_oof is None else str(args.x3_oof),
        "x3_column": args.x3_column,
        "current_value_oof": None if args.current_value_oof is None else str(args.current_value_oof),
        "integration_oof": str(args.integration_oof),
    }
    report["elapsed_seconds"] = float(time.monotonic() - start)

    result.rows.to_csv(run_dir / "routed_oof.csv", index=False, encoding="utf-8", lineterminator="\n")
    result.fold_selections.to_csv(
        run_dir / "fold_selections.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    result.coverage.to_csv(
        run_dir / "coverage.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    (run_dir / "x1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    timer = progress("产物写盘完成", timer)

    pooled = report["evaluation"]["pooled"]
    gates = report["evaluation"]["gates"]
    print()
    print(f"X1 {args.mode} routed pooled MAPE: {pooled['routed_mape']:.6f}%")
    print(f"  vs A61 parent: {pooled['improvement_vs_parent_pp']:+.5f}pp")
    print(f"  vs P3 static:  {pooled['improvement_vs_p3_pp']:+.5f}pp")
    print(f"  gates passed:  {gates['passed']}  {gates['checks']}")
    print(f"  coverage:      {report['evaluation']['coverage']}")
    print(f"  产物目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
