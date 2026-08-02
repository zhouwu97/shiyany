"""不接触测试未来标签的初赛自动编排流程。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from gas_forecast.config import ForecastConfig
from gas_forecast.config import legacy_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.freeze import sha256_file
from gas_forecast.oof import build_legacy_oof, write_oof
from gas_forecast.routing import leave_one_fold_out_route
from gas_forecast.selection import choose_version
from gas_forecast.selection_competition import choose_competition_candidate
from gas_forecast.submission import package_submission, validate_submission_frame
from gas_forecast.validation import backtest_model
from gas_forecast.workflow import predict_rolling, train_model


SUPPORTED_VERSIONS = ("v1", "v2", "v25", "v3")


def _target_route() -> dict[str, object]:
    cells = {}
    for target in ("generator_1", "generator_all"):
        selected = "v2_pred" if target == "generator_1" else "v3_pred"
        for horizon in range(15, 121, 15):
            cells[f"{target}|{horizon}"] = {"selected": selected}
    return {
        "policy": "fixed_v2_v3_target_route",
        "global": {"selected": "v3_pred"},
        "targets": {
            "generator_1": {"selected": "v2_pred"},
            "generator_all": {"selected": "v3_pred"},
        },
        "cells": cells,
    }


def _single_model_route(model: str) -> dict[str, object]:
    selected = f"{model.removesuffix('_pred')}_pred"
    return {
        "policy": "single_model_route",
        "global": {"selected": selected},
        "targets": {},
        "cells": {},
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_price(data_dir: Path) -> Path | None:
    matches = sorted(data_dir.glob("*price*.xlsx"))
    return matches[0] if matches else None


def audit_future_perturbation(
    frame,
    config: ForecastConfig,
    *,
    price=None,
    baseline_features=None,
) -> dict[str, object]:
    """改写参考时刻后的生产数据，确认参考时刻特征严格不变。"""

    if len(frame) < 3:
        raise ValueError("未来扰动审计至少需要3个时间点")
    origin_position = min(len(frame) - 2, max(1, int(len(frame) * 0.75)))
    origin = frame.index[origin_position]
    baseline = (
        baseline_features
        if baseline_features is not None
        else build_causal_features(frame, config.feature, price)
    )

    perturbed = frame.copy()
    future_mask = perturbed.index > origin
    numeric_columns = list(perturbed.select_dtypes(include=[np.number]).columns)
    perturbed[numeric_columns] = perturbed[numeric_columns].astype(float)
    perturbed.loc[future_mask, numeric_columns] = -999_999.0
    changed = build_causal_features(perturbed, config.feature, price)

    before = baseline.loc[origin]
    after = changed.loc[origin]
    equal = before.eq(after) | (before.isna() & after.isna())
    changed_columns = [str(column) for column in equal.index[~equal]]
    return {
        "passed": not changed_columns,
        "origin": str(origin),
        "future_rows_perturbed": int(future_mask.sum()),
        "checked_feature_columns": int(len(before)),
        "changed_columns": changed_columns,
    }


def run_automated_pipeline(
    train_dir: str | Path,
    test_dir: str | Path,
    *,
    versions: Iterable[str] = SUPPORTED_VERSIONS,
    run_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    selection_path: str | Path | None = None,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    archive_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    jobs: int = 1,
    max_folds: int | None = None,
    minimum_folds: int = 15,
    expected_rows: int = 192,
) -> dict[str, object]:
    """执行数据审计、训练期选型、全量重训、滚动预测和提交打包。"""

    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    run_path = Path(run_dir) if run_dir is not None else new_run_dir(
        "results/raw/runs", "legacy_pipeline"
    )
    run_path.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(reports_dir) if reports_dir is not None else run_path / "backtests"
    selection_path = (
        Path(selection_path) if selection_path is not None else run_path / "selection.json"
    )
    model_path = Path(model_path) if model_path is not None else run_path / "model.joblib"
    output_dir = Path(output_dir) if output_dir is not None else run_path / "submission"
    archive_path = Path(archive_path) if archive_path is not None else run_path / "submission.zip"
    summary_path = Path(summary_path) if summary_path is not None else run_path / "summary.json"
    requested_versions = tuple(dict.fromkeys(versions))
    invalid_versions = sorted(set(requested_versions).difference(SUPPORTED_VERSIONS))
    if invalid_versions:
        raise ValueError(f"不支持的模型版本: {invalid_versions}")
    if "v1" not in requested_versions:
        raise ValueError("自动选择必须包含V1作为基础版本")
    if minimum_folds < 15:
        raise ValueError("正式自动流水线至少需要15个滚动折")
    if max_folds is not None and max_folds < minimum_folds:
        raise ValueError(f"--max-folds不能小于正式门槛{minimum_folds}")

    config = ForecastConfig()
    train_dataset = align_tables(train_dir, config.feature.frequency)
    test_dataset = align_tables(test_dir, config.feature.frequency)
    price_path = _find_price(train_dir)
    price = load_price_schedule(price_path) if price_path else None
    features = build_causal_features(train_dataset.frame, config.feature, price)
    perturbation = audit_future_perturbation(
        train_dataset.frame,
        config,
        price=price,
        baseline_features=features,
    )
    if not perturbation["passed"]:
        raise RuntimeError(f"未来扰动测试失败: {perturbation['changed_columns']}")

    reports: dict[str, dict[str, object]] = {}
    report_paths: dict[str, str] = {}
    for version in requested_versions:
        report = backtest_model(
            train_dataset.frame,
            features,
            version,
            config,
            max_folds=max_folds,
            n_jobs=jobs,
        )
        fold_count = len(report["folds"])
        if fold_count < minimum_folds:
            raise RuntimeError(
                f"{version}仅生成{fold_count}个滚动折，低于正式门槛{minimum_folds}"
            )
        report_path = reports_dir / f"backtest_{version}_full.json"
        _write_json(report_path, report)
        reports[version] = report
        report_paths[version] = str(report_path)

    decision = choose_version(reports)
    decision["future_perturbation"] = perturbation
    decision["test_labels_used"] = False
    _write_json(selection_path, decision)

    selected_version = str(decision["selected_version"])
    train_model(train_dir, model_path, selected_version, config=config)
    input_features, predictions = predict_rolling(train_dir, test_dir, model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.csv"
    result_path = output_dir / "s_result.csv"
    input_features.reset_index().to_csv(input_path, index=False, encoding="utf-8")
    result_frame = predictions.reset_index()
    result_frame.to_csv(result_path, index=False, encoding="utf-8")
    validation = validate_submission_frame(result_frame)
    if int(validation["rows"]) != expected_rows:
        raise RuntimeError(
            f"提交结果应有{expected_rows}行，实际为{validation['rows']}行"
        )
    archive = package_submission(input_path, result_path, archive_path)

    summary: dict[str, object] = {
        "run_dir": str(run_path),
        "selected_version": selected_version,
        "reason": decision["reason"],
        "train_audit": train_dataset.audit.to_dict(),
        "test_audit": test_dataset.audit.to_dict(),
        "future_perturbation": perturbation,
        "backtest_reports": report_paths,
        "selection": str(selection_path),
        "model": str(model_path),
        "input_csv": str(input_path),
        "result_csv": str(result_path),
        "archive": archive,
        "validation": validation,
        "sha256": {
            "model": sha256_file(model_path),
            "result_csv": sha256_file(result_path),
            "archive": sha256_file(archive_path),
        },
        "test_labels_used": False,
        "leaderboard_feedback_used": False,
        "manual_prediction_edits": False,
    }
    _write_json(summary_path, summary)
    selected_report = reports.get(selected_version, {})
    finalize_run(
        run_path,
        {
            "run_type": "experiment",
            "stage": "legacy_pipeline",
            "is_smoke": max_folds is not None and max_folds < minimum_folds,
            "candidate": selected_version,
            "pooled_mape": selected_report.get("mean_mape"),
            "leakage_passed": bool(perturbation["passed"]),
            "tests_passed": False,
            "submission_valid": bool(validation.get("valid", True)),
            "model": str(model_path.relative_to(run_path)),
            "result": str(result_path.relative_to(run_path)),
            "submission": str(archive_path.relative_to(run_path)),
        },
    )
    return summary


def run_competition_pipeline(
    train_dir: str | Path,
    test_dir: str | Path,
    *,
    versions: Iterable[str] = SUPPORTED_VERSIONS,
    run_dir: str | Path | None = None,
    jobs: int = 8,
    max_folds: int | None = None,
    minimum_folds: int = 15,
    expected_rows: int = 192,
) -> dict[str, object]:
    """按 pooled OOF 直接比较单模型、目标路由与稳定目标×步长路由。"""

    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    run_path = Path(run_dir) if run_dir is not None else new_run_dir(
        "results/raw/runs", "competition_pipeline"
    )
    run_path.mkdir(parents=True, exist_ok=True)
    config = legacy_forecast_config()
    train_dataset = align_tables(train_dir, config.feature.frequency)
    test_dataset = align_tables(test_dir, config.feature.frequency)
    price_path = _find_price(train_dir)
    price = load_price_schedule(price_path) if price_path else None
    features = build_causal_features(train_dataset.frame, config.feature, price)
    perturbation = audit_future_perturbation(
        train_dataset.frame, config, price=price, baseline_features=features
    )
    if not perturbation["passed"]:
        raise RuntimeError(f"未来扰动测试失败: {perturbation['changed_columns']}")
    result = build_legacy_oof(
        train_dataset.frame,
        features,
        versions=versions,
        config=config,
        max_folds=max_folds,
        n_jobs=jobs,
        checkpoint_dir=run_path / "checkpoints",
    )
    if len(result.report["folds"]) < minimum_folds:
        raise RuntimeError(f"滚动折少于正式门槛 {minimum_folds}")
    write_oof(result, run_path / "oof_legacy.csv", run_path / "oof_legacy_report.json")
    write_json(run_path / "oof_legacy_report.json", result.report)
    rows = result.rows.copy()
    if not {"v2_pred", "v3_pred"}.issubset(rows.columns):
        raise ValueError("pooled_oof 自动流程必须包含 v2 和 v3")
    rows["v2_v3_target_pred"] = rows["v3_pred"]
    generator_1 = rows["target"].eq("generator_1")
    rows.loc[generator_1, "v2_v3_target_pred"] = rows.loc[generator_1, "v2_pred"]
    candidate_columns = tuple(
        column for column in ("persistence_pred", "v1_pred", "v2_pred", "v25_pred", "v3_pred")
        if column in rows
    )
    routed, route_report = leave_one_fold_out_route(rows, candidate_columns)
    candidates = {
        column.removesuffix("_pred"): column for column in candidate_columns
    }
    candidates["v2_v3_target"] = "v2_v3_target_pred"
    candidates["stable_target_horizon_lofo"] = "routed_pred"
    selection = choose_competition_candidate(routed, candidates)
    routed.to_csv(run_path / "oof_candidates.csv", index=False, encoding="utf-8")
    decision = {
        "selection_policy": "pooled_oof",
        "selection": selection,
        "routing": route_report,
        "future_perturbation": perturbation,
        "test_labels_used": False,
    }
    write_json(run_path / "selection.json", decision)

    selected = str(selection["selected_candidate"])
    model_path = run_path / "model.joblib"
    if selected == "stable_target_horizon_lofo":
        route = route_report["final_route"]
        train_model(train_dir, model_path, "routed", route=route, n_jobs=jobs, config=config)
        trained_version = "routed"
    elif selected == "v2_v3_target":
        train_model(train_dir, model_path, "routed", route=_target_route(), n_jobs=jobs, config=config)
        trained_version = "routed"
    elif selected == "persistence":
        train_model(
            train_dir,
            model_path,
            "routed",
            route=_single_model_route("persistence"),
            n_jobs=jobs,
            config=config,
        )
        trained_version = "routed"
    else:
        train_model(train_dir, model_path, selected, config=config)
        trained_version = selected

    input_features, predictions = predict_rolling(train_dir, test_dir, model_path)
    submission_dir = run_path / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    input_path = submission_dir / "input.csv"
    result_path = submission_dir / "s_result.csv"
    input_features.reset_index().to_csv(input_path, index=False, encoding="utf-8")
    result_frame = predictions.reset_index()
    result_frame.to_csv(result_path, index=False, encoding="utf-8")
    validation = validate_submission_frame(result_frame)
    if int(validation["rows"]) != expected_rows:
        raise RuntimeError(f"提交结果应有 {expected_rows} 行，实际为 {validation['rows']} 行")
    archive_path = run_path / "submission.zip"
    package_submission(input_path, result_path, archive_path)
    summary = {
        "run_dir": str(run_path),
        "selected_candidate": selected,
        "trained_version": trained_version,
        "selection": str(run_path / "selection.json"),
        "model": str(model_path),
        "result_csv": str(result_path),
        "archive": str(archive_path),
        "validation": validation,
        "train_audit": train_dataset.audit.to_dict(),
        "test_audit": test_dataset.audit.to_dict(),
        "sha256": {
            "model": sha256_file(model_path),
            "result_csv": sha256_file(result_path),
            "archive": sha256_file(archive_path),
        },
        "test_labels_used": False,
        "leaderboard_feedback_used": False,
        "manual_prediction_edits": False,
    }
    write_json(run_path / "summary.json", summary)
    selected_report = selection.get("reports", {}).get(selected, {})
    finalize_run(
        run_path,
        {
            "run_type": "experiment",
            "stage": "M1",
            "is_smoke": max_folds is not None and max_folds < minimum_folds,
            "outer_folds": len(result.report["folds"]),
            "pooled_mape": selected_report.get("pooled_mape"),
            "candidate": selected,
            "leakage_passed": bool(perturbation["passed"]),
            "tests_passed": False,
            "submission_valid": bool(validation.get("valid", True)),
            "best_files": {
                "model": "model.joblib",
                "result": "submission/s_result.csv",
                "submission": "submission.zip",
                "report": "oof_legacy_report.json",
                "selection": "selection.json",
            },
        },
    )
    return summary
