"""运行严格因果历史轨迹相似样本的 screening 或完整开发实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.historical_analog import (
    PRE_REGISTERED_SPECS,
    audit_historical_analog_future_perturbation,
    build_historical_analog_oof,
)


def _resolve_data_dir(path: Path) -> Path:
    """兼容传入官方数据目录的父目录，但不读取任何测试标签。"""

    if (path / "Pre_gas.csv").exists():
        return path
    matches = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "Pre_gas.csv").exists()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"无法解析官方训练数据目录: {path}")
    return matches[0]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """以 UTF-8 持久化报告，保证实验目录可单独审计。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_result(run_dir: Path, name: str, result) -> None:
    """写入统一 OOF、trace 和报告，不触碰 results/best 或正式提交目录。"""

    result.rows.to_csv(run_dir / f"{name}_oof.csv", index=False, encoding="utf-8")
    result.trace.to_csv(run_dir / f"{name}_trace.csv", index=False, encoding="utf-8")
    _write_json(run_dir / f"{name}_report.json", result.report)
    # 同时保留统一文件名，便于下游工具按既有实验目录契约读取。
    result.rows.to_csv(run_dir / "oof.csv", index=False, encoding="utf-8")
    result.trace.to_csv(run_dir / "trace.csv", index=False, encoding="utf-8")
    _write_json(run_dir / "metrics.json", result.report.get("metrics", {}))


def parse_args() -> argparse.Namespace:
    """解析固定范围的实验参数。"""

    parser = argparse.ArgumentParser(description="运行严格因果历史轨迹相似样本实验")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--name",
        required=True,
        help="独立实验目录名称，写入 results/raw/runs/experiments/<name>/",
    )
    parser.add_argument(
        "--scope",
        choices=("screening", "development"),
        default="screening",
        help="development 会先强制运行 screening；screening 失败即 STOP。",
    )
    return parser.parse_args()


def main() -> None:
    """执行预注册的六配置实验，并将非 blind 审计材料归档。"""

    args = parse_args()
    if Path(args.name).name != args.name or args.name in {"", ".", ".."}:
        raise ValueError("--name 必须是单层实验目录名")
    run_dir = Path("results/raw/runs/experiments") / args.name
    if run_dir.exists():
        raise FileExistsError(f"实验目录已存在，拒绝覆盖: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    data_dir = _resolve_data_dir(args.data_dir)
    config = ForecastConfig()
    frame = align_tables(data_dir, config.feature.frequency).frame
    manifest: dict[str, object] = {
        "experiment": "historical_analog",
        "status": "running",
        "scope_requested": args.scope,
        "run_dir": str(run_dir),
        "data_dir": str(data_dir.resolve()),
        "blind_included": False,
        "blind_labels_used": False,
        "platform_reference_used": False,
        "pre_registered_search": [
            {"context": spec.context, "neighbors": spec.neighbors, "metric": spec.metric}
            for spec in PRE_REGISTERED_SPECS
        ],
    }
    _write_json(run_dir / "manifest.json", manifest)

    screening = build_historical_analog_oof(frame, config=config, scope="screening")
    _write_result(run_dir, "screening", screening)
    manifest["screening"] = screening.report["screening"]
    if not bool(screening.report["screening"]["passed"]):
        manifest["status"] = "STOP"
        manifest["stop_reason"] = "screening_gate_failed"
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "report.json",
            {
                "status": "STOP",
                "screening": screening.report,
                "blind_included": False,
                "full_development_run": False,
            },
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    if args.scope == "screening":
        manifest["status"] = "SCREENING_PASS"
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "report.json",
            {
                "status": "SCREENING_PASS",
                "screening": screening.report,
                "blind_included": False,
                "full_development_run": False,
            },
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    development = build_historical_analog_oof(frame, config=config, scope="development")
    _write_result(run_dir, "development", development)
    audits = [
        audit_historical_analog_future_perturbation(frame, spec=spec)
        for spec in PRE_REGISTERED_SPECS
    ]
    future_perturbation_passed = all(bool(audit["passed"]) for audit in audits)
    development_gate = development.report["full_development_gate"]
    development_gate_passed = bool(development_gate["passed"])
    report = {
        "status": (
            "DEVELOPMENT_PASS"
            if future_perturbation_passed and development_gate_passed
            else "STOP"
        ),
        "screening": screening.report,
        "development": development.report,
        "full_development_gate": development_gate,
        "future_perturbation": audits,
        "future_perturbation_passed": future_perturbation_passed,
        "full_development_gate_passed": development_gate_passed,
        "blind_included": False,
        "formal_promotion_attempted": False,
    }
    manifest.update(
        {
            "status": report["status"],
            "full_development_run": True,
            "future_perturbation_passed": future_perturbation_passed,
            "full_development_gate_passed": development_gate_passed,
            "report": "report.json",
            "oof": "development_oof.csv",
            "trace": "development_trace.csv",
        }
    )
    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
