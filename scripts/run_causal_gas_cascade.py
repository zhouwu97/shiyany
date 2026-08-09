"""运行严格因果的两阶段煤气级联实验。

脚本只接受一个已按时间排序的训练 CSV（包含 ``datetime``），不读取评分集
或 blind 标签。所有产物写入独立 ``results/raw/runs/experiments`` 子目录。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# 允许从仓库根目录直接执行脚本。
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from gas_forecast.causal_gas_cascade import (  # noqa: E402
    CascadeConfig,
    CausalGasCascadeForecaster,
    future_perturbation_audit,
    write_cascade_run,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行因果煤气级联 Stage1/Stage2 实验")
    parser.add_argument("--input", type=Path, required=True, help="训练期 CSV，必须包含 datetime")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="独立实验目录；默认写入 results/raw/runs/experiments/<name>",
    )
    parser.add_argument("--name", default="causal_gas_cascade", help="默认 run 目录名")
    parser.add_argument("--max-folds", type=int, help="只运行前 N 个外层折（screening）")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--min-train-rows", type=int, default=64)
    parser.add_argument("--min-validation-rows", type=int, default=16)
    parser.add_argument(
        "--origins",
        type=int,
        default=5,
        help="未来扰动审计使用的最近 origin 数量",
    )
    return parser.parse_args()


def _read_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "datetime" not in frame.columns:
        raise ValueError("输入 CSV 缺少 datetime 列")
    timestamps = pd.to_datetime(frame.pop("datetime"), errors="raise")
    frame.index = pd.DatetimeIndex(timestamps, name="datetime")
    frame = frame.sort_index(kind="stable")
    if frame.index.has_duplicates:
        raise ValueError("输入 CSV 含重复 datetime")
    return frame


def main() -> int:
    args = _parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    frame = _read_input(input_path)
    config = CascadeConfig(
        inner_folds=args.inner_folds,
        outer_folds=args.outer_folds,
        ridge_alpha=args.ridge_alpha,
        min_train_rows=args.min_train_rows,
        min_validation_rows=args.min_validation_rows,
    )
    model = CausalGasCascadeForecaster(config)
    result = model.build_oof(frame, max_folds=args.max_folds)
    origins = frame.index[-max(1, min(args.origins, len(frame))):]
    perturbation = future_perturbation_audit(model.fit(frame), frame, origins)
    result.report["future_perturbation"] = perturbation
    run_dir = args.run_dir or (
        ROOT / "results" / "raw" / "runs" / "experiments" / args.name
    )
    write_cascade_run(
        result,
        run_dir,
        config=config,
        mapping=model.mapping_,
        screening={
            "status": "SCREENING_ONLY" if args.max_folds is not None else "FULL_DEVELOPMENT",
            "blind_used": False,
            "future_labels_used_for_training": False,
        },
    )
    summary = {
        "run_dir": str(Path(run_dir).resolve()),
        "rows": int(len(result.rows)),
        "stage1_pooled_mape": result.report["stage1"]["pooled_mape"],
        "stage2_pooled_mape": result.report["stage2"]["pooled_mape"],
        "future_perturbation_passed": perturbation["passed"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
