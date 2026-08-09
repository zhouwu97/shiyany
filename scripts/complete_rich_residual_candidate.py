"""把通过 final 验收的 RichResidual 固化为独立、可审计的生产候选。"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import build_fingerprints, finalize_run, sha256, write_json
from gas_forecast.features import load_price_schedule
from gas_forecast.rich_residual import (
    RichResidualAggressiveForecaster,
    RichResidualSpec,
    fit_full_rich_residual_corrector,
    rich_feature_config,
)
from gas_forecast.scoring import score_oof_long
from gas_forecast.submission import package_submission, validate_submission_frame
from gas_forecast.submission_quality import (
    COMPETITION_QUALITY_POLICY,
    prepare_submission_input,
)
from gas_forecast.workflow import predict_rolling


BASE_CANDIDATE = "aggressive_r75_lgb20"
BASELINE_COLUMN = "aggressive_r75_lgb20_pred"
CANDIDATE = "rich_gas_blend_30"
PREDICTION_COLUMN = f"{CANDIDATE}_pred"
BLEND_WEIGHT = 0.30
RICH_SPEC = RichResidualSpec(
    name="rich_gas",
    feature_groups=frozenset({"gas"}),
    min_train_rows=256,
)


def _resolve_data_dir(path: Path, marker: str) -> Path:
    """允许调用方传入官方数据根目录或其直接父目录。"""

    if (path / marker).exists():
        return path
    matches = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / marker).exists()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"无法解析数据目录: {path}")
    return matches[0]


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    """读取并验证单个 JSON 对象，避免把半成品收据当作冻结依据。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取{description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description}必须是 JSON 对象: {path}")
    return payload


def _require_mapping(value: object, description: str) -> dict[str, object]:
    """将外部 JSON 的嵌套字段收敛为明确的对象契约。"""

    if not isinstance(value, dict):
        raise ValueError(f"{description}缺失或格式错误")
    return value


def _validate_base_artifacts(
    model_path: Path,
    baseline_oof: Path,
) -> tuple[object, dict[str, object]]:
    """确认模型、OOF 和严格基线报告来自同一已冻结 Champion 运行。"""

    if not model_path.is_file():
        raise FileNotFoundError(f"缺少冻结 Champion 模型: {model_path}")
    if not baseline_oof.is_file():
        raise FileNotFoundError(f"缺少冻结 Champion OOF: {baseline_oof}")
    manifest_path = model_path.parent / "manifest.json"
    report_path = model_path.parent / "report.json"
    manifest = _read_json_object(manifest_path, "冻结 Champion manifest")
    report = _read_json_object(report_path, "冻结 Champion OOF 报告")
    if manifest.get("candidate") != BASE_CANDIDATE:
        raise ValueError(f"生产基线必须是 {BASE_CANDIDATE}")
    if report.get("strict_label_purge") is not True:
        raise ValueError("冻结 Champion 报告未声明 strict_label_purge")
    expected_oof = model_path.parent / str(manifest.get("oof", "oof.csv"))
    if expected_oof.resolve() != baseline_oof.resolve():
        raise ValueError("--baseline-oof 与冻结 Champion manifest 不一致")
    model = joblib.load(model_path)
    if not hasattr(model, "config") or not hasattr(model, "predict"):
        raise TypeError("冻结 Champion 模型不具有 config 和 predict 接口")
    return model, manifest


def _validate_final_evidence(
    final_run: Path,
    baseline_oof: Path,
    base_config: ForecastConfig,
) -> tuple[pd.DataFrame, dict[str, object], Path, Path]:
    """验证 frozen final 收据、指标和 Rich 配置均可追溯到同一基线。"""

    report_path = final_run / "report.json"
    manifest_path = final_run / "manifest.json"
    oof_path = final_run / "oof.csv"
    report = _read_json_object(report_path, "RichResidual final 报告")
    manifest = _read_json_object(manifest_path, "RichResidual final manifest")
    if report.get("scope") != "final" or manifest.get("scope") != "final":
        raise ValueError("RichResidual 运行不是 final 验收")
    if report.get("strict_oof") is not True:
        raise ValueError("RichResidual final 报告未声明 strict_oof")
    if report.get("blind_used_for_selection") is not False:
        raise ValueError("RichResidual final 报告未证明 blind 未参与选型")
    if report.get("baseline_column") != BASELINE_COLUMN:
        raise ValueError("RichResidual final 基线列不匹配")
    source_oof = Path(str(report.get("baseline_oof", "")))
    if source_oof.resolve() != baseline_oof.resolve():
        raise ValueError("RichResidual final 报告引用的基线 OOF 不匹配")

    candidates = _require_mapping(report.get("candidates"), "RichResidual candidates")
    gas = _require_mapping(candidates.get("gas"), "RichResidual gas 候选")
    models = _require_mapping(gas.get("models"), "RichResidual gas 模型报告")
    selected = _require_mapping(models.get(PREDICTION_COLUMN), "冻结的 RichResidual 候选")
    if selected.get("formal_candidate") is not True:
        raise ValueError(f"{PREDICTION_COLUMN} 未通过 final 候选门槛")
    contract = _require_mapping(gas.get("strict_oof_contract"), "RichResidual 严格 OOF 契约")
    if contract.get("blind_labels_used") is not True:
        raise ValueError("RichResidual final 未使用一次性 blind 验收")
    if contract.get("residual_target") != "actual - same_fold_champion_prediction":
        raise ValueError("RichResidual 残差标签契约不匹配")

    effective_configs = _require_mapping(report.get("effective_configs"), "RichResidual 有效配置")
    final_config = forecast_config_from_dict(
        _require_mapping(effective_configs.get("rich_gas"), "RichResidual gas 有效配置")
    )
    expected_config = rich_feature_config(base_config, RICH_SPEC.feature_groups)
    if asdict(final_config) != asdict(expected_config):
        raise ValueError("RichResidual final 配置与冻结 Champion 或 gas 特征组不一致")

    if not oof_path.is_file():
        raise FileNotFoundError(f"RichResidual final 缺少 OOF: {oof_path}")
    rows = pd.read_csv(oof_path, parse_dates=["origin_time"])
    required = {"fold", "origin_time", "target", "horizon", "actual", PREDICTION_COLUMN}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"RichResidual final OOF 缺少字段: {missing}")
    if rows.duplicated(["fold", "origin_time", "target", "horizon"]).any():
        raise ValueError("RichResidual final OOF 存在重复评分键")
    if not rows["fold"].eq("blind").any():
        raise ValueError("RichResidual final OOF 缺少 blind 评分行")
    score = score_oof_long(rows, PREDICTION_COLUMN)
    reported_mape = selected.get("pooled_mape")
    if not isinstance(reported_mape, (int, float)):
        raise ValueError("冻结 RichResidual 候选缺少 pooled_mape")
    if abs(float(score["pooled_mape"]) - float(reported_mape)) > 1e-12:
        raise ValueError("RichResidual final OOF 与报告中的 pooled_mape 不一致")
    return rows, selected, report_path, oof_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument("--rich-final-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-confirmed-blind-oof",
        action="store_true",
        help="显式确认 final blind 已验收，允许其标签参与一次全量残差重训",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_confirmed_blind_oof:
        raise ValueError("必须显式传入 --allow-confirmed-blind-oof 才能进行全量重训")
    if args.run_dir.exists():
        raise FileExistsError(f"生产候选目录已存在，拒绝覆盖: {args.run_dir}")

    base_model, base_manifest = _validate_base_artifacts(args.base_model, args.baseline_oof)
    base_config = base_model.config
    final_rows, final_metrics, final_report_path, final_oof_path = _validate_final_evidence(
        args.rich_final_run,
        args.baseline_oof,
        base_config,
    )
    train_dir = _resolve_data_dir(args.data_dir, "Pre_gas.csv")
    test_dir = _resolve_data_dir(args.test_dir, "Pre_test_gas.csv")
    dataset = align_tables(train_dir, base_config.feature.frequency)
    price_paths = sorted(train_dir.glob("*price*.xlsx"))
    price_schedule = load_price_schedule(price_paths[0]) if price_paths else None
    champion_oof = pd.read_csv(args.baseline_oof, parse_dates=["origin_time"])

    corrector = fit_full_rich_residual_corrector(
        dataset.frame,
        champion_oof,
        config=base_config,
        spec=RICH_SPEC,
        baseline_column=BASELINE_COLUMN,
        price_schedule=price_schedule,
        allow_confirmed_blind_oof=True,
    )
    expected_horizons = {15 * horizon for horizon in corrector.config.feature.horizons}
    missing_horizons = sorted(expected_horizons.difference(corrector.states_))
    if missing_horizons:
        raise RuntimeError(f"全量 RichResidual 未训练所有预测步长: {missing_horizons}")
    model = RichResidualAggressiveForecaster(
        base_model,
        corrector,
        blend_weight=BLEND_WEIGHT,
    )

    args.run_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.run_dir / "model.joblib"
    input_path = args.run_dir / "submission" / "input.csv"
    result_path = args.run_dir / "submission" / "s_result.csv"
    archive_path = args.run_dir / "submission.zip"
    joblib.dump(model, model_path)

    input_features, predictions = predict_rolling(train_dir, test_dir, model_path)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_missing_cells = int(input_features.isna().sum().sum())
    input_export = input_features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    quality_input, quality_report = prepare_submission_input(
        input_export.reset_index(),
        COMPETITION_QUALITY_POLICY,
    )
    quality_input.to_csv(input_path, index=False, encoding="utf-8")
    result_frame = predictions.reset_index()
    result_frame.to_csv(result_path, index=False, encoding="utf-8")
    validation = validate_submission_frame(result_frame, model.config)
    package_submission(
        input_path,
        result_path,
        archive_path,
        quality_policy=COMPETITION_QUALITY_POLICY,
    )

    oof_path = args.run_dir / "oof.csv"
    shutil.copy2(final_oof_path, oof_path)
    shutil.copy2(final_report_path, args.run_dir / "final_rich_oof_report.json")
    blind_confirmation = {
        "candidate": CANDIDATE,
        "prediction_column": PREDICTION_COLUMN,
        "source_final_run": str(args.rich_final_run.resolve()),
        "source_final_report": "final_rich_oof_report.json",
        "source_final_report_sha256": sha256(final_report_path),
        "source_final_oof_sha256": sha256(final_oof_path),
        "pooled_mape": float(final_metrics["pooled_mape"]),
        "blind": final_metrics.get("blind"),
        "blind_difference": final_metrics.get("blind_difference"),
        "blind_used_for_selection": False,
        "confirmed_blind_oof_used_for_refit": True,
    }
    write_json(args.run_dir / "blind_confirmation.json", blind_confirmation)
    report_path = args.run_dir / "report.json"
    write_json(
        report_path,
        {
            "candidate": CANDIDATE,
            "strict_label_purge": True,
            "base_candidate": BASE_CANDIDATE,
            "baseline_column": BASELINE_COLUMN,
            "feature_groups": sorted(RICH_SPEC.feature_groups),
            "blend_weight": BLEND_WEIGHT,
            "capacity_projection": True,
            "confirmed_blind_oof_used_for_refit": True,
            "residual_training_rows": {
                f"t+{horizon}": state.training_rows
                for horizon, state in sorted(corrector.states_.items())
            },
            "final_oof_evaluation": final_metrics,
            "blind_confirmation": "blind_confirmation.json",
        },
    )
    write_json(args.run_dir / "config.json", asdict(model.config))

    fingerprints = build_fingerprints(
        config=model.config,
        dataset=dataset.frame,
        model_params={
            "candidate": CANDIDATE,
            "base_model": str(args.base_model.resolve()),
            "base_model_sha256": sha256(args.base_model),
            "baseline_oof": str(args.baseline_oof.resolve()),
            "baseline_oof_sha256": sha256(args.baseline_oof),
            "rich_final_run": str(args.rich_final_run.resolve()),
            "rich_final_report_sha256": sha256(final_report_path),
            "rich_final_oof_sha256": sha256(final_oof_path),
            "feature_groups": sorted(RICH_SPEC.feature_groups),
            "blend_weight": BLEND_WEIGHT,
            "confirmed_blind_oof_used_for_refit": True,
        },
        random_seed=model.config.model.random_state,
    )
    manifest = finalize_run(
        args.run_dir,
        {
            "run_type": "training",
            "stage": "rich_residual_candidate",
            "is_smoke": False,
            "candidate": CANDIDATE,
            "pooled_mape": float(final_metrics["pooled_mape"]),
            "config": asdict(model.config),
            "best_files": {
                "model": "model.joblib",
                "input": "submission/input.csv",
                "result": "submission/s_result.csv",
                "submission": "submission.zip",
                "report": "report.json",
            },
            "leakage_passed": False,
            "tests_passed": False,
            "submission_valid": True,
            "oof": "oof.csv",
            "report": "report.json",
            "blind_confirmation": "blind_confirmation.json",
            "source_base_manifest": str((args.base_model.parent / "manifest.json").resolve()),
            "source_base_run_id": base_manifest.get("run_id"),
            "source_final_run": str(args.rich_final_run.resolve()),
            "confirmed_blind_oof_used_for_refit": True,
            "input_export_forward_filled_cells": input_missing_cells,
            "submission_quality": quality_report,
            "validation": validation,
            **fingerprints,
        },
    )
    print(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "pooled_mape": manifest["pooled_mape"],
                "validation": validation,
                "run_dir": str(args.run_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
