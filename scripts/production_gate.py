"""执行正式候选的 OOF、泄漏、测试、提交和哈希生产门。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, promote_if_best, sha256, write_json
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.leakage import audit_origin_predictor
from gas_forecast.scoring import score_oof_long
from gas_forecast.submission import (
    validate_submission_archive,
    validate_submission_frame,
    validate_submission_input,
)
from gas_forecast.submission_quality import COMPETITION_QUALITY_POLICY
from gas_forecast.workflow import resolve_prediction_feature_config


def _resolve_data_dir(path: Path) -> Path:
    if (path / "Pre_gas.csv").exists():
        return path
    matches = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "Pre_gas.csv").exists()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"无法解析官方数据目录: {path}")
    return matches[0]


def _candidate_prediction_column(candidate: str) -> str:
    return "routed_pred" if candidate.startswith("lofo_") else f"{candidate}_pred"


def _run_pytest(repo_root: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "passed": result.returncode == 0,
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _zip_receipt(path: Path, input_path: Path, result_path: Path) -> dict[str, object]:
    try:
        return validate_submission_archive(
            path,
            expected_input_path=input_path,
            expected_result_path=result_path,
            quality_policy=COMPETITION_QUALITY_POLICY,
        )
    except (OSError, ValueError, AssertionError) as exc:
        return {"valid": False, "error": str(exc)}


def _required_false(name: str, *sources: dict[str, object]) -> bool:
    """只接受显式、无冲突的 false，避免缺失元数据被默认为安全。"""

    found = False
    for source in sources:
        if name in source:
            found = True
            if source[name] is not False:
                return False
    return found


def _expand_policy_sources(*sources: dict[str, object]) -> tuple[dict[str, object], ...]:
    """同时读取旧版顶层和新版 causal_prediction_audit 声明。"""

    expanded: list[dict[str, object]] = []
    for source in sources:
        expanded.append(source)
        nested = source.get("causal_prediction_audit")
        if isinstance(nested, dict):
            expanded.append(nested)
    return tuple(expanded)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行正式候选 Production Gate")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--best-dir", type=Path, default=Path("results/best"))
    parser.add_argument("--origins", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"候选运行缺少 manifest.json: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = str(manifest.get("candidate", "unknown"))
    model_path = run_dir / str(manifest.get("best_files", {}).get("model", "model.joblib"))
    input_path = run_dir / str(
        manifest.get("best_files", {}).get("input", "submission/input.csv")
    )
    result_path = run_dir / str(
        manifest.get("best_files", {}).get("result", "submission/s_result.csv")
    )
    archive_path = run_dir / str(
        manifest.get("best_files", {}).get("submission", "submission.zip")
    )
    oof_path = run_dir / str(manifest.get("oof", "oof.csv"))
    report_path = run_dir / str(manifest.get("report", "report.json"))
    for path in (model_path, input_path, result_path, archive_path, oof_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"Production Gate 缺少产物: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = pd.read_csv(oof_path, parse_dates=["origin_time"])
    prediction_column = _candidate_prediction_column(candidate)
    if prediction_column not in rows:
        prediction_column = "routed_pred" if "routed_pred" in rows else prediction_column
    oof_score = score_oof_long(rows, prediction_column)
    oof_receipt = {
        "passed": bool(report.get("strict_label_purge", False)),
        "pooled_mape": float(oof_score["pooled_mape"]),
        "prediction_column": prediction_column,
        "strict_label_purge": bool(report.get("strict_label_purge", False)),
        "report": str(report_path.relative_to(run_dir)),
    }
    if not oof_receipt["passed"]:
        raise RuntimeError("OOF 报告未声明 strict_label_purge")

    model = joblib.load(model_path)
    config = getattr(model, "config", ForecastConfig())
    data_dir = _resolve_data_dir(args.data_dir)
    dataset = align_tables(data_dir, config.feature.frequency)
    prices = sorted(data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    feature_config = resolve_prediction_feature_config(model)

    def predictor(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        features = build_causal_features(frame, feature_config, price)
        current = frame.loc[[origin], list(config.targets)]
        return model.predict(features.loc[[origin]], current)

    causal_prediction_audit = audit_origin_predictor(
        dataset.frame,
        predictor=predictor,
        origins=args.origins,
    )
    policy_sources = _expand_policy_sources(report, manifest)
    oracle_candidate_safe = _required_false("oracle_candidate", *policy_sources)
    blind_labels_unused = _required_false("blind_labels_used", *policy_sources)
    audit_passed = causal_prediction_audit.get("passed") is True
    audit_cases_checked = int(causal_prediction_audit.get("cases_checked", 0))
    policy_passed = bool(oracle_candidate_safe and blind_labels_unused)
    leakage = {
        "passed": bool(audit_passed and audit_cases_checked >= 250 and policy_passed),
        "causal_prediction_audit": causal_prediction_audit,
        "oracle_candidate": not oracle_candidate_safe,
        "blind_labels_used": not blind_labels_unused,
        "policy_declarations_valid": policy_passed,
    }
    write_json(run_dir / "oof_report.json", oof_receipt)
    write_json(run_dir / "leakage.json", leakage)

    pytest_receipt = _run_pytest(Path.cwd())
    write_json(run_dir / "pytest.json", pytest_receipt)
    input_frame = pd.read_csv(input_path)
    submission_frame = pd.read_csv(result_path)
    validation = validate_submission_frame(submission_frame, config)
    input_validation = validate_submission_input(
        input_frame,
        submission_frame,
        quality_policy=COMPETITION_QUALITY_POLICY,
        enforce_quality=True,
    )
    zip_receipt = _zip_receipt(archive_path, input_path, result_path)
    submission_receipt = {
        "valid": bool(zip_receipt["valid"]),
        "validation": validation,
        "input": input_validation,
        "archive": zip_receipt,
    }
    write_json(run_dir / "submission.json", submission_receipt)
    hashes = {
        "model": sha256(model_path),
        "input": sha256(input_path),
        "result": sha256(result_path),
        "submission": sha256(archive_path),
        "oof": sha256(oof_path),
        "report": sha256(report_path),
    }
    evidence = {
        "oof_report": "oof_report.json",
        "leakage_report": "leakage.json",
        "pytest_report": "pytest.json",
        "submission_report": "submission.json",
    }
    passed = bool(
        oof_receipt["passed"]
        and leakage["passed"] is True
        and pytest_receipt["passed"] is True
        and submission_receipt["valid"] is True
        and zip_receipt["valid"] is True
    )
    finalized = finalize_run(
        run_dir,
        {
            "candidate": candidate,
            "pooled_mape": float(oof_receipt["pooled_mape"]),
            "leakage_passed": bool(leakage.get("passed") is True),
            "causal_prediction_audit": causal_prediction_audit,
            "oracle_candidate": not oracle_candidate_safe,
            "blind_labels_used": not blind_labels_unused,
            "tests_passed": bool(pytest_receipt["passed"] is True),
            "submission_valid": bool(submission_receipt["valid"] and zip_receipt["valid"]),
            "promotion_evidence": evidence,
            "hashes": hashes,
            "production_gate_passed": passed,
        },
    )
    promoted = False
    if passed and not args.no_promote:
        promoted = promote_if_best(run_dir, args.best_dir)
    print(
        json.dumps(
            {
                "candidate": candidate,
                "pooled_mape": finalized["pooled_mape"],
                "production_gate_passed": passed,
                "promoted": promoted,
                "leakage": leakage,
                "pytest": pytest_receipt,
                "submission": submission_receipt,
                "hashes": hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
