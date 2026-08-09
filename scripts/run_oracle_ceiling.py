"""X0 — P3 Oracle Ceiling Audit CLI。

只读复用 P3 滚动训练的 development OOF（``integration/oof.csv``），不重训、
不读取 blind、不修改 ``results/best`` 与正式提交。输出目录必须全新。

用法示例：

.. code-block:: powershell

    python scripts/run_oracle_ceiling.py `
      --input <p3_run>/integration/oof.csv `
      --routes-dir <p3_run> `
      --expected-a61-mape 0.05195744922629299 `
      --expected-p3-mape 0.051591406939815905 `
      --out results/raw/runs/audits/x0_oracle_ceiling_20260809
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from gas_forecast.oracle_ceiling import (
    EXPECTED_FOLDS,
    EXPECTED_ORIGINS,
    EXPECTED_ROWS,
    build_trajectory_frame,
    cross_check_route_oof_keys,
    load_p3_integration_oof,
    run_oracle_ceiling,
    write_oracle_ceiling_artifacts,
)


def _print_report_summary(report: dict[str, object], hashes: dict[str, str]) -> None:
    """打印人类可读的审计摘要。"""

    keys = report["key_validation"]
    checks = keys["checks"]
    print("== X0 P3 Oracle Ceiling Audit ==")
    print(
        f"key contract: rows={keys['rows']} folds={keys['folds']} "
        f"origins={keys['origins']} per_fold={keys['origins_per_fold']} "
        f"all_passed={keys['all_passed']} ({json.dumps(checks)})"
    )
    current = report["current_mape"]
    print("\ncurrent pooled MAPE:")
    for name in (
        "a61_parent",
        "p3_fusion",
        "a64_direct_delta",
        "p1_causal_rolling",
        "p2_historical_analog",
        "p2_matured_residual",
    ):
        print(f"  {name:24s} {current[name]:.6f}")
    print("\noracle levels (mape / gap_vs_a61_pp / gap_vs_p3_pp / distinct):")
    oracle = report["oracle"]
    for level in ("row", "target", "horizon", "target_x_horizon", "origin", "fold"):
        item = oracle[level]
        print(
            f"  {level:16s} {item['mape']:.6f}  {item['gap_pp_vs_a61']:+.4f}  "
            f"{item['gap_pp_vs_p3']:+.4f}  {item['distinct_selected']}"
        )
    split = report["split_half_oracle"]
    print("\nsplit-half oracle:")
    for direction in ("first_to_second", "second_to_first"):
        item = split[direction]
        print(
            f"  {direction:16s} {item['mape']:.6f}  "
            f"gap_vs_a61 {item['gap_pp_vs_a61']:+.4f}pp  "
            f"gap_vs_p3 {item['gap_pp_vs_p3']:+.4f}pp"
        )
    print(f"  combined_mean    {split['combined_mean_mape']:.6f}")
    verdict = report["pre_registered_verdict"]
    print(
        f"\npre-registered verdict: {verdict['verdict']} "
        f"(row oracle {verdict['row_oracle_mape']:.6f} <= {verdict['threshold']})"
    )
    print(f"  {verdict['conclusion']}")
    hit = oracle["row"]["hit_rate"]
    print("\nrow-level candidate hit rate:")
    for name, frac in hit.items():
        print(f"  {name:24s} {frac:.4f}")
    print("\nartifacts (SHA-256):")
    for name, digest in hashes.items():
        print(f"  {name:40s} {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="P3 集成 OOF 文件（integration/oof.csv）")
    parser.add_argument(
        "--routes-dir",
        default=None,
        help="P3 实验目录（含四条候选的独立 OOF 文件），用于完整键一致性核对",
    )
    parser.add_argument("--out", required=True, help="全新输出目录（results/raw/runs/audits/...）")
    parser.add_argument(
        "--expected-a61-mape",
        type=float,
        default=None,
        help="冻结 A61 报告 MAPE，校验复现（可选）",
    )
    parser.add_argument(
        "--expected-p3-mape",
        type=float,
        default=None,
        help="冻结 P3 融合报告 MAPE，校验复现（可选）",
    )
    args = parser.parse_args()

    rows, summary = load_p3_integration_oof(
        args.input,
        expected_folds=EXPECTED_FOLDS,
        expected_origins=EXPECTED_ORIGINS,
        expected_rows=EXPECTED_ROWS,
    )
    report, winners = run_oracle_ceiling(rows)
    if args.routes_dir is not None:
        report["routes_cross_check"] = cross_check_route_oof_keys(rows, args.routes_dir)
    checks: dict[str, object] = {}
    if args.expected_a61_mape is not None:
        observed = report["current_mape"]["a61_parent"]
        checks["a61_parent"] = observed
        if abs(observed - args.expected_a61_mape) > 1e-9:
            raise SystemExit(
                f"A61 MAPE 复现失败: observed={observed!r} expected={args.expected_a61_mape!r}"
            )
    if args.expected_p3_mape is not None:
        observed = report["current_mape"]["p3_fusion"]
        checks["p3_fusion"] = observed
        if abs(observed - args.expected_p3_mape) > 1e-9:
            raise SystemExit(
                f"P3 MAPE 复现失败: observed={observed!r} expected={args.expected_p3_mape!r}"
            )
    if checks:
        report["reproducibility_check"] = checks

    trajectory = build_trajectory_frame(rows)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"输出目录必须全新且为空: {out}")
    hashes = write_oracle_ceiling_artifacts(report, winners, trajectory, out)
    _print_report_summary(report, hashes)


if __name__ == "__main__":
    main()
