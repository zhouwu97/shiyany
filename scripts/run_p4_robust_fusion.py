"""在唯一 P3 development OOF 上运行 P4 稳健交叉拟合融合。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from gas_forecast.p4_robust_fusion import (
    load_and_validate_p3_inputs,
    robust_cross_fitted_fusion,
    write_robust_fusion_artifacts,
)


DEFAULT_INPUT = Path(
    r"E:\AI\shiyan\results\raw\runs\experiments\p3_rolling_training_20260809_190558"
)


def parse_args() -> argparse.Namespace:
    """解析只读输入和全新输出目录。"""

    parser = argparse.ArgumentParser(description="运行 P4 Robust Cross-Fitted Fusion")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """执行输入审计、19 折稳健选择、最终门禁和产物落盘。"""

    args = parse_args()
    started = time.perf_counter()
    rows, input_receipt = load_and_validate_p3_inputs(args.input_dir)
    result = robust_cross_fitted_fusion(rows, input_receipt=input_receipt)
    result.report["elapsed_seconds"] = float(time.perf_counter() - started)
    output = write_robust_fusion_artifacts(result, args.run_dir)
    final_gate = result.report["final_static_fusion_gate"]
    payload = {
        "status": result.report["status"],
        "run_dir": str(output.resolve()),
        "elapsed_seconds": result.report["elapsed_seconds"],
        "pooled_candidate_mape": final_gate["pooled_candidate_mape"],
        "pooled_parent_mape": final_gate["pooled_parent_mape"],
        "improvement_pp": final_gate["improvement_pp"],
        "fold_wins": final_gate["fold_wins"],
        "recent5_wins": final_gate["recent5_wins"],
        "worst_fold_regression_pp": final_gate["worst_fold_regression_pp"],
        "by_target_improvement_pp": final_gate["by_target_improvement_pp"],
        "selected_per_fold": result.report["selected_per_fold"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if final_gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
