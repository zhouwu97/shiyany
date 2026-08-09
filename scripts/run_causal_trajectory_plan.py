"""运行 A69 严格因果轨迹集成的 screening、development 与静态 cross-fit 融合。

该脚本只使用官方训练表和 development OOF。它不会读取评分集、blind 标签、
提交文件或平台参考。A62--A65 任一路线 screening STOP 后均保留原始收据，
且不会继续为该路线执行 development 或调参。
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from gas_forecast.causal_gas_cascade import screening_decision
from gas_forecast.causal_trajectory_ensemble import (
    PARENT_ROUTE,
    RouteReceipt,
    build_causal_trajectory_ensemble,
    collect_matching_oofs,
    read_oof,
    validate_oof_contract,
    write_ensemble_artifacts,
)
from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.recursive_arx import build_recursive_arx_diversity
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config


ROUTE_NAMES = ("a62_state_space", "a63_historical_analog", "a64_direct_delta", "a65_causal_gas_cascade")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="官方训练数据目录")
    parser.add_argument("--a61-oof", type=Path, required=True, help="A61 development OOF CSV/Parquet")
    parser.add_argument("--a61-report", type=Path, required=True, help="A61 development report.json")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="新的 results/raw/runs/experiments/<name> 目录",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--a62-dir", type=Path, help="复用已完成的 A62 独立实验目录")
    parser.add_argument("--a63-dir", type=Path, help="复用已完成的 A63 独立实验目录")
    parser.add_argument("--a64-dir", type=Path, help="复用已完成的 A64 独立实验目录")
    parser.add_argument("--a65-dir", type=Path, help="复用已完成的 A65 独立实验目录")
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """写入可独立审计的 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _experiment_root(run_dir: Path) -> Path:
    """拒绝把 A69 产物写到 results/best 或正式提交区域。"""

    resolved = run_dir.resolve()
    expected = (ROOT / "results" / "raw" / "runs" / "experiments").resolve()
    if resolved.parent != expected:
        raise ValueError("--run-dir 必须是 results/raw/runs/experiments 的直接子目录")
    if resolved.exists():
        raise FileExistsError(f"A69 run 目录已存在，拒绝覆盖: {resolved}")
    return expected


def _run_command(command: list[str], *, events: list[dict[str, object]]) -> int:
    """执行既有路线 CLI，并记录命令和退出状态而不吞掉终端输出。"""

    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(CODE) + (os.pathsep + current_pythonpath if current_pythonpath else "")
    completed = subprocess.run(command, cwd=ROOT, check=False, env=environment)
    events.append({"command": command, "returncode": int(completed.returncode)})
    return int(completed.returncode)


def _contains_true_blind(value: object) -> bool:
    """递归检查报告中是否出现任何明确的 blind=true 声明。"""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if "blind" in str(key).lower() and nested is True:
                return True
            if _contains_true_blind(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_true_blind(item) for item in value)
    return False


def _find_future_audit(value: object) -> bool | None:
    """从路线报告中递归提取未来扰动审计的 passed 收据。"""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if "future_perturbation" in str(key) and isinstance(nested, Mapping):
                passed = nested.get("passed")
                if isinstance(passed, bool):
                    return passed
            if "future_perturbation" in str(key) and isinstance(nested, bool):
                return nested
            found = _find_future_audit(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_future_audit(nested)
            if found is not None:
                return found
    return None


def _read_report(run_dir: Path) -> dict[str, Any]:
    """读取路线报告；缺失时返回空字典以保留拒绝收据。"""

    path = run_dir / "report.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _route_output_path(name: str, run_dir: Path) -> Path | None:
    """返回通过 development 后应存在的统一 OOF 文件。"""

    candidates = {
        "a62_state_space": ("oof.parquet",),
        "a63_historical_analog": ("development_oof.csv",),
        "a64_direct_delta": ("oof.csv",),
        "a65_causal_gas_cascade": ("oof.csv",),
    }[name]
    for filename in candidates:
        path = run_dir / filename
        if path.is_file():
            return path
    return None


def _status_passes(name: str, report: Mapping[str, object], run_dir: Path) -> tuple[bool, str]:
    """按各路线冻结的 development 状态判断，不根据 A69 标签改写路线结果。"""

    status = str(report.get("status", ""))
    expected = {
        "a62_state_space": "PROMOTE_CANDIDATE",
        "a63_historical_analog": "DEVELOPMENT_PASS",
        "a64_direct_delta": "DEVELOPMENT_COMPLETE",
    }
    if name in expected:
        return status == expected[name], status or "MISSING_REPORT_STATUS"
    screening_path = run_dir / "a69_screening_decision.json"
    if not screening_path.is_file():
        return False, "MISSING_A69_SCREENING_DECISION"
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    passed = bool(screening.get("passed", False))
    return passed and _route_output_path(name, run_dir) is not None, str(screening.get("status", ""))


def _stage2_only(rows: pd.DataFrame) -> pd.DataFrame:
    """A65 OOF 同时包含资源 Stage1，A69 只接收两个发电目标的 Stage2。"""

    if "stage" not in rows:
        return rows
    return rows.loc[
        rows["stage"].eq("stage2") & rows["target"].isin(["generator_1", "generator_all"])
    ].copy()


def _make_cascade_input(data_dir: Path, output: Path) -> None:
    """将官方四表对齐为 A65 CLI 所需的单一 datetime CSV，写入路线实验目录。"""

    frame = align_tables(data_dir, ForecastConfig().feature.frequency).frame
    frame.reset_index().to_csv(output, index=False, encoding="utf-8")


def _run_routes(args: argparse.Namespace, events: list[dict[str, object]]) -> dict[str, Path]:
    """执行四条既有路线，A65 先显式做五折 screening。"""

    root = args.run_dir.parent
    prefix = args.run_dir.name
    python = str(args.python.resolve())
    route_dirs = {
        "a62_state_space": root / f"{prefix}_a62",
        "a63_historical_analog": root / f"{prefix}_a63",
        "a64_direct_delta": root / f"{prefix}_a64",
        "a65_causal_gas_cascade": root / f"{prefix}_a65_screening",
    }
    _run_command(
        [
            python,
            str(ROOT / "scripts" / "run_state_space_diversity.py"),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--input",
            str(args.a61_oof.resolve()),
            "--scope",
            "development",
            "--run-dir",
            str(route_dirs["a62_state_space"]),
        ],
        events=events,
    )
    _run_command(
        [
            python,
            str(ROOT / "scripts" / "run_historical_analog.py"),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--scope",
            "development",
            "--name",
            route_dirs["a63_historical_analog"].name,
        ],
        events=events,
    )
    _run_command(
        [
            python,
            str(ROOT / "scripts" / "run_direct_delta.py"),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--scope",
            "development",
            "--primary-model",
            "ridge",
            "--run-dir",
            str(route_dirs["a64_direct_delta"]),
        ],
        events=events,
    )
    cascade_dir = route_dirs["a65_causal_gas_cascade"]
    cascade_dir.mkdir(parents=True, exist_ok=False)
    cascade_input = cascade_dir / "aligned_training.csv"
    _make_cascade_input(args.data_dir.resolve(), cascade_input)
    _run_command(
        [
            python,
            str(ROOT / "scripts" / "run_causal_gas_cascade.py"),
            "--input",
            str(cascade_input),
            "--run-dir",
            str(cascade_dir),
            "--max-folds",
            "5",
        ],
        events=events,
    )
    screening_rows = _stage2_only(read_oof(cascade_dir / "oof.csv"))
    screening_fold_count = int(screening_rows["fold"].nunique())
    if screening_fold_count < 5:
        decision = {
            "status": "STOP_INSUFFICIENT_SCREENING_FOLDS",
            "passed": False,
            "completed_folds": sorted(screening_rows["fold"].astype(str).unique().tolist()),
            "required_folds": 5,
            "reason": "A65 早期外折历史不足，不能以少于五折的结果替代既定 screening",
        }
    else:
        decision = screening_decision(screening_rows, first_folds=5)
    _write_json(cascade_dir / "a69_screening_decision.json", decision)
    if not bool(decision["passed"]):
        return route_dirs
    development_dir = root / f"{prefix}_a65_development"
    _run_command(
        [
            python,
            str(ROOT / "scripts" / "run_causal_gas_cascade.py"),
            "--input",
            str(cascade_input),
            "--run-dir",
            str(development_dir),
        ],
        events=events,
    )
    _write_json(development_dir / "a69_screening_decision.json", decision)
    route_dirs["a65_causal_gas_cascade"] = development_dir
    return route_dirs


def _reuse_routes(args: argparse.Namespace) -> dict[str, Path]:
    """解析显式提供的既有路线目录，避免在审计复跑时重新训练。"""

    supplied = {
        "a62_state_space": args.a62_dir,
        "a63_historical_analog": args.a63_dir,
        "a64_direct_delta": args.a64_dir,
        "a65_causal_gas_cascade": args.a65_dir,
    }
    if any(value is None for value in supplied.values()):
        missing = [name for name, value in supplied.items() if value is None]
        raise ValueError(f"复用模式必须提供全部路线目录: {missing}")
    return {name: Path(value).resolve() for name, value in supplied.items() if value is not None}


def _audit_a61_parent_future(
    data_dir: Path,
    parent_rows: pd.DataFrame,
    report: Mapping[str, object],
) -> dict[str, object]:
    """独立重建一个早期 development 折，审计 A61 预测的四种未来扰动。"""

    try:
        payload = report.get("effective_feature_config")
        if not isinstance(payload, Mapping):
            return {"passed": False, "reason": "A61 report 缺少 effective_feature_config"}
        config = forecast_config_from_dict(payload)
        effective = rich_feature_config(config, RICH_FEATURE_GROUPS, feature_profile="long_horizon")
        dataset = align_tables(data_dir, effective.feature.frequency)
        frame = dataset.frame
        price_paths = sorted(data_dir.glob("*price*.xlsx"))
        if len(price_paths) != 1:
            return {"passed": False, "reason": "官方 price 文件数量不是一"}
        price = load_price_schedule(price_paths[0])
        work = parent_rows.copy()
        work["origin_time"] = pd.to_datetime(work["origin_time"], errors="raise")
        work["train_end"] = pd.to_datetime(work["train_end"], errors="raise")
        fold_order = (
            work.groupby("fold", sort=False)["origin_time"].min().sort_values().index.astype(str).tolist()
        )
        fold = fold_order[0]
        origins = pd.DatetimeIndex(sorted(work.loc[work["fold"].eq(fold), "origin_time"].unique()))
        audit_origin = origins[len(origins) // 2]
        selected = work.loc[work["fold"].eq(fold) & work["origin_time"].le(audit_origin)].copy()
        baseline_frame = frame

        def predict(candidate_frame: pd.DataFrame) -> pd.DataFrame:
            features = build_causal_features(candidate_frame, effective.feature, price)
            result = build_recursive_arx_diversity(
                candidate_frame,
                features,
                selected,
                baseline_column="a60_gall_long_blend_30_pred",
            )
            return result.rows.sort_values(
                ["origin_time", "target", "horizon"], kind="stable"
            ).reset_index(drop=True)

        baseline = predict(baseline_frame)
        prediction_column = "a61_recursive_blend_05_pred"
        key = ["fold", "origin_time", "target", "horizon"]
        observed_series = baseline.set_index(key)[prediction_column]
        expected = selected.set_index(key)[prediction_column].reindex(observed_series.index).to_numpy(dtype=float)
        observed = observed_series.to_numpy(dtype=float)
        baseline_match = bool(
            len(expected) == len(observed)
            and np.allclose(expected, observed, rtol=0.0, atol=1e-12)
        )
        future_mask = frame.index > audit_origin
        numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
        extreme = frame.copy()
        extreme.loc[future_mask, numeric_columns] = (
            extreme.loc[future_mask, numeric_columns].to_numpy(dtype=float) * 17.0 + 123.0
        )
        shuffled = frame.copy()
        if int(future_mask.sum()) > 1 and numeric_columns:
            future_values = shuffled.loc[future_mask, numeric_columns].to_numpy(dtype=float)
            shuffled.loc[future_mask, numeric_columns] = future_values[
                np.random.default_rng(17).permutation(len(future_values))
            ]
        nulled = frame.copy()
        nulled.loc[future_mask, numeric_columns] = np.nan
        variants = {
            "extreme": extreme,
            "shuffle": shuffled,
            "null": nulled,
            "delete_future": frame.loc[~future_mask | (frame.index == audit_origin)],
        }
        cases: dict[str, dict[str, object]] = {}
        all_passed = baseline_match
        for name, candidate_frame in variants.items():
            candidate = predict(candidate_frame).set_index(key)[prediction_column]
            base_values = baseline.set_index(key)[prediction_column]
            aligned = candidate.reindex(base_values.index)
            difference = float(np.max(np.abs(base_values.to_numpy(dtype=float) - aligned.to_numpy(dtype=float))))
            passed = bool(difference == 0.0)
            all_passed = all_passed and passed
            cases[name] = {
                "passed": passed,
                "changed_future_rows": int(future_mask.sum()),
                "max_abs_difference": difference,
            }
        return {
            "passed": bool(all_passed),
            "fold": fold,
            "origin": str(audit_origin),
            "rows_checked": int(len(selected)),
            "prediction_column": prediction_column,
            "baseline_matches_frozen_oof": baseline_match,
            "cases": cases,
            "methods": list(variants),
        }
    except Exception as error:  # 审计失败必须转为可审计 STOP，而不是吞掉路线。
        return {
            "passed": False,
            "reason": f"{type(error).__name__}: {error}",
            "methods": ["extreme", "shuffle", "null", "delete_future"],
        }


def _parent_receipt(
    path: Path,
    report_path: Path,
    *,
    future_audit: Mapping[str, object],
) -> RouteReceipt:
    """将冻结 A61 development OOF 记录为基线收据。"""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    blind_used = _contains_true_blind(report)
    return RouteReceipt(
        name=PARENT_ROUTE,
        source=str(path.resolve()),
        status=str(report.get("status", "FROZEN_DEVELOPMENT_PARENT")),
        accepted=not blind_used,
        reason="冻结 A61 development 父模型",
        rows=int(len(read_oof(path))),
        blind_labels_used=blind_used,
        future_perturbation_passed=(True if future_audit.get("passed") is True else False),
    )


def _admit_routes(
    parent_rows: pd.DataFrame,
    route_dirs: Mapping[str, Path],
) -> tuple[dict[str, pd.DataFrame], list[RouteReceipt], dict[str, object]]:
    """执行状态、blind、未来审计、OOF 契约和完整键覆盖五层准入。"""

    accepted: dict[str, pd.DataFrame] = {}
    receipts: list[RouteReceipt] = []
    contracts: dict[str, object] = {}
    for name in ROUTE_NAMES:
        run_dir = route_dirs[name]
        report = _read_report(run_dir)
        passed_status, status = _status_passes(name, report, run_dir)
        blind_used = _contains_true_blind(report)
        future_passed = _find_future_audit(report)
        output_path = _route_output_path(name, run_dir)
        reason = ""
        rows = 0
        candidate: pd.DataFrame | None = None
        if not passed_status:
            reason = f"development 状态未通过: {status}"
        elif blind_used:
            reason = "报告声明使用了 blind 标签"
        elif future_passed is not True:
            reason = "缺少或未通过 future perturbation 收据"
        elif output_path is None:
            reason = "缺少 development OOF"
        else:
            try:
                candidate = _stage2_only(read_oof(output_path)) if name.startswith("a65") else read_oof(output_path)
                rows = int(len(candidate))
                contracts[name] = validate_oof_contract(candidate, source=name)
                # 单路线试合并即可强制验证完整主键、折边界和 19 折覆盖。
                collect_matching_oofs(parent_rows, {name: candidate})
            except (OSError, ValueError, pd.errors.ParserError) as error:
                reason = f"OOF 契约或覆盖不通过: {type(error).__name__}: {error}"
            else:
                accepted[name] = candidate
                reason = "通过 status、blind、future audit 与完整 OOF 主键检查"
        receipts.append(
            RouteReceipt(
                name=name,
                source=str(run_dir.resolve()),
                status=status,
                accepted=name in accepted,
                reason=reason,
                rows=rows,
                blind_labels_used=blind_used,
                future_perturbation_passed=future_passed,
            )
        )
    return accepted, receipts, contracts


def main() -> int:
    args = _parse_args()
    _experiment_root(args.run_dir)
    events: list[dict[str, object]] = []
    reuse_values = (args.a62_dir, args.a63_dir, args.a64_dir, args.a65_dir)
    route_dirs = _reuse_routes(args) if any(value is not None for value in reuse_values) else _run_routes(args, events)
    parent_rows = read_oof(args.a61_oof)
    parent_contract = validate_oof_contract(parent_rows, source=PARENT_ROUTE, prediction_column="a61_recursive_blend_05_pred")
    parent = parent_rows.rename(columns={"a61_recursive_blend_05_pred": "prediction"})
    a61_report = json.loads(args.a61_report.read_text(encoding="utf-8"))
    parent_future_audit = _audit_a61_parent_future(args.data_dir.resolve(), parent_rows, a61_report)
    parent_receipt = _parent_receipt(
        args.a61_oof,
        args.a61_report,
        future_audit=parent_future_audit,
    )
    accepted, receipts, contracts = _admit_routes(parent, route_dirs)
    result = build_causal_trajectory_ensemble(
        parent,
        accepted,
        route_receipts=[parent_receipt, *receipts],
    )
    result.report["input_contracts"] = {"a61_parent": parent_contract, **contracts}
    result.report["route_directories"] = {name: str(path) for name, path in route_dirs.items()}
    result.report["execution"] = {"commands": events, "reused_routes": not bool(events)}
    result.report["future_perturbation"]["a61_parent"] = parent_future_audit
    output = write_ensemble_artifacts(result, args.run_dir)
    print(
        json.dumps(
            {
                "run_dir": str(output.resolve()),
                "status": result.report["status"],
                "accepted_routes": result.report["routes"]["accepted"],
                "rejected_routes": result.report["routes"]["rejected"],
                "pooled_parent_mape": result.report["static_gate"]["pooled_parent_mape"],
                "pooled_static_mape": result.report["static_gate"]["pooled_candidate_mape"],
                "blind_labels_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
