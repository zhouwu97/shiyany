"""运行 A64 直接增量研究，产物只写入独立 experiments run 目录。"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.direct_delta import (
    DirectDeltaConfig,
    DirectDeltaForecaster,
    audit_direct_delta_future_perturbations,
    build_direct_delta_features,
    build_direct_delta_oof,
    build_direct_delta_targets,
    screen_direct_delta,
)
from gas_forecast.splits import make_outer_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 A64 直接绝对增量实验")
    parser.add_argument("--data-dir", type=Path, required=True, help="仅含训练生产观测的官方数据目录")
    parser.add_argument(
        "--scope",
        choices=("screening", "development"),
        default="screening",
        help="development 仍先重跑五折 screening；失败即停止，不读取 blind",
    )
    parser.add_argument(
        "--primary-model",
        choices=("ridge", "lightgbm"),
        default="ridge",
        help="预注册的开发候选；不会根据本次结果切换",
    )
    parser.add_argument("--run-name", default="a64_direct_delta", help="独立实验目录前缀")
    parser.add_argument("--run-dir", type=Path, help="results/raw/runs/experiments 下的新目录")
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--lgb-estimators", type=int, default=120)
    return parser.parse_args()


def _resolve_data_dir(path: Path) -> Path:
    """兼容传入官方数据父目录，但不会扫描或修改其他工作树。"""

    path = path.resolve()
    if (path / "Pre_gas.csv").exists():
        return path
    matches = sorted(child for child in path.iterdir() if child.is_dir() and (child / "Pre_gas.csv").exists())
    if len(matches) != 1:
        raise FileNotFoundError("data-dir 必须是训练数据目录或只含一个训练数据子目录的父目录")
    return matches[0]


def _new_run_dir(args: argparse.Namespace) -> Path:
    """强制实验输出留在唯一工作树的独立 raw run 区域。"""

    root = (ROOT / "results" / "raw" / "runs" / "experiments").resolve()
    if args.run_dir is not None:
        candidate = args.run_dir.resolve()
        if root not in candidate.parents:
            raise ValueError("--run-dir 必须位于 results/raw/runs/experiments 下")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = root / f"{args.run_name}_{stamp}"
    if candidate.exists():
        raise FileExistsError(f"run 目录已存在，拒绝覆盖: {candidate}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def default(value: object) -> object:
        if isinstance(value, (pd.Timestamp, Path)):
            return str(value)
        raise TypeError(f"无法序列化: {type(value).__name__}")

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=default) + "\n", encoding="utf-8")


def _merge_parent_and_candidates(
    parent_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> pd.DataFrame:
    """以 OOF 主键精确合并线性父模型与非线性候选。"""

    keys = ["fold", "origin_time", "train_end", "target", "horizon", "actual", "current_value", "actual_delta"]
    parent = parent_rows.loc[:, keys + ["ridge_prediction"]].rename(
        columns={"ridge_prediction": "parent_ridge_prediction"}
    )
    candidate = candidate_rows.loc[:, keys + ["ridge_prediction", "lightgbm_prediction"]]
    merged = candidate.merge(parent, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(candidate_rows) or len(merged) != len(parent_rows):
        raise RuntimeError("父模型与候选 OOF 主键不完整一致")
    return merged.sort_values(["origin_time", "target", "horizon", "fold"], kind="stable").reset_index(drop=True)


def _fold_summary(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for name in ("parent_ridge", "ridge", "lightgbm"):
        column = f"{name}_prediction"
        rows[f"{name}_ape"] = (rows["actual"] - rows[column]).abs() / rows["actual"].abs().clip(lower=1e-6)
    return (
        rows.groupby(["fold", "target", "horizon"], sort=True)[
            ["parent_ridge_ape", "ridge_ape", "lightgbm_ape"]
        ]
        .mean()
        .reset_index()
    )


def _run_oof(
    frame: pd.DataFrame,
    config: DirectDeltaConfig,
    folds,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """固定父特征和非线性候选特征，避免把 feature ablation 混入模型选择。"""

    parent_features = build_direct_delta_features(frame, include_nonlinear_state=False)
    candidate_features = build_direct_delta_features(frame, include_nonlinear_state=True)
    parent_rows, parent_report = build_direct_delta_oof(
        frame, parent_features, config=config, folds=folds, include_blind=False, nested=True
    )
    candidate_rows, candidate_report = build_direct_delta_oof(
        frame, candidate_features, config=config, folds=folds, include_blind=False, nested=True
    )
    rows = _merge_parent_and_candidates(parent_rows, candidate_rows)
    report = {
        "parent": parent_report,
        "candidate": candidate_report,
        "feature_ablation": {
            "parent_feature_count": int(parent_features.shape[1]),
            "candidate_feature_count": int(candidate_features.shape[1]),
            "nonlinear_state_features_enabled": True,
        },
    }
    return rows, report


def main() -> int:
    args = parse_args()
    run_dir = _new_run_dir(args)
    config = DirectDeltaConfig(
        ridge_alpha=args.ridge_alpha,
        lgb_n_estimators=args.lgb_estimators,
    )
    base_report: dict[str, Any] = {
        "experiment": "a64_direct_delta",
        "scope_requested": args.scope,
        "primary_model": args.primary_model,
        "config": asdict(config),
        "blind_labels_used": False,
        "platform_score_used": False,
        "future_generator_truth_used": False,
        "status": "RUNNING",
    }
    try:
        data_dir = _resolve_data_dir(args.data_dir)
        dataset = align_tables(data_dir)
        frame = dataset.frame
        all_folds = [
            fold for fold in make_outer_folds(frame.index, ForecastConfig()) if not fold.blind
        ]
        screening_folds = all_folds[:5]
        if len(screening_folds) < 5:
            raise ValueError("训练数据无法提供五个 development screening 折")

        screening_rows, screening_detail = _run_oof(frame, config, screening_folds)
        screening = screen_direct_delta(
            screening_rows,
            candidate=args.primary_model,
            parent="parent_ridge",
        )
        rows = screening_rows
        detail: dict[str, object] = {"screening": screening_detail}
        status = str(screening["status"])
        if args.scope == "development" and status == "PASS":
            rows, development_detail = _run_oof(frame, config, all_folds)
            detail["development"] = development_detail
            status = "DEVELOPMENT_COMPLETE"
        elif args.scope == "development":
            status = "STOP"

        rows["prediction_model"] = args.primary_model
        rows["prediction"] = rows[f"{args.primary_model}_prediction"]
        rows.to_csv(run_dir / "oof.csv", index=False, encoding="utf-8")
        _fold_summary(rows).to_csv(run_dir / "folds.csv", index=False, encoding="utf-8")
        trace = {
            "parent": detail["screening"]["parent"].get("trace", []),
            "candidate": detail["screening"]["candidate"].get("trace", []),
        }
        if "development" in detail:
            trace["development_parent"] = detail["development"]["parent"].get("trace", [])
            trace["development_candidate"] = detail["development"]["candidate"].get("trace", [])
        _write_json(run_dir / "trace.json", trace)

        perturbation: dict[str, object] | None = None
        if status == "DEVELOPMENT_COMPLETE":
            full_features = build_direct_delta_features(frame, include_nonlinear_state=True)
            final_model = DirectDeltaForecaster(config).fit(
                full_features,
                build_direct_delta_targets(frame),
                frame.loc[:, list(config.targets)],
            )
            perturbation = audit_direct_delta_future_perturbations(
                frame,
                full_features,
                final_model,
                model_name=args.primary_model,
            )
            if not perturbation["passed"]:
                status = "STOP_FUTURE_PERTURBATION"

        base_report.update(
            {
                "status": status,
                "data_dir": str(data_dir),
                "rows": int(len(rows)),
                "folds": sorted(rows["fold"].astype(str).unique().tolist()),
                "screening": screening,
                "details": detail,
                "future_perturbation": perturbation,
                "artifacts": {
                    "oof": "oof.csv",
                    "folds": "folds.csv",
                    "trace": "trace.json",
                    "report": "report.json",
                },
            }
        )
        _write_json(run_dir / "report.json", base_report)
        _write_json(
            run_dir / "manifest.json",
            {
                "run_type": "experiment",
                "experiment": "a64_direct_delta",
                "status": status,
                "blind_labels_used": False,
                "promoted": False,
                "oof": "oof.csv",
                "report": "report.json",
                "trace": "trace.json",
            },
        )
        print(f"A64 {status}: {run_dir}")
        return 0 if status in {"PASS", "DEVELOPMENT_COMPLETE"} else 2
    except Exception as error:
        base_report.update({"status": "ERROR", "error_type": type(error).__name__, "error": str(error)})
        _write_json(run_dir / "report.json", base_report)
        _write_json(
            run_dir / "manifest.json",
            {"run_type": "experiment", "experiment": "a64_direct_delta", "status": "ERROR", "report": "report.json"},
        )
        print(f"A64 ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
