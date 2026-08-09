"""P4 稳健交叉拟合静态融合。

本模块只消费 P3 已生成的 development OOF，不训练基础模型。每个 held fold
的权重只由其余折的标签决定，并先经过 P3 冻结的稳定性门槛。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from gas_forecast.causal_trajectory_ensemble import (
    HORIZONS,
    IDENTITY_COLUMNS,
    KEY_COLUMNS,
    PARENT_ROUTE,
    STATIC_MAX_TARGET_REGRESSION_PP,
    STATIC_MAX_WORST_FOLD_REGRESSION_PP,
    STATIC_MIN_IMPROVEMENT_PP,
    STATIC_MIN_RECENT5_WINS,
    TARGETS,
    _prediction_for_weights,
    pre_registered_weight_candidates,
    static_fusion_gate,
)


EXPECTED_FOLDS = 19
EXPECTED_ORIGINS = 3_648
EXPECTED_ROWS = 58_368
ROUTE_NAMES: tuple[str, ...] = (
    "a64_direct_delta",
    "p1_causal_rolling",
    "p2_historical_analog",
    "p2_matured_residual",
)
RAW_ROUTE_FILES: Mapping[str, str] = {
    "a64_direct_delta": "a64_direct_delta_oof.csv",
    "p1_causal_rolling": "p1_causal_rolling_oof.csv",
    "p2_historical_analog": "p2_historical_analog_oof.csv",
    "p2_matured_residual": "p2_matured_residual_oof.csv",
}


@dataclass(frozen=True)
class RobustFusionResult:
    """P4 OOF、候选级轨迹、逐折选择和最终报告。"""

    rows: pd.DataFrame
    trace: pd.DataFrame
    selections: pd.DataFrame
    report: dict[str, object]


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_key_frame(rows: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """解析完整 OOF 键并拒绝重复、空值、非有限标签和 blind 折。"""

    missing = sorted(set(KEY_COLUMNS).difference(rows.columns))
    if missing:
        raise ValueError(f"{source} OOF 缺少完整键字段: {missing}")
    result = rows.loc[:, list(KEY_COLUMNS)].copy()
    result["fold"] = result["fold"].astype(str)
    if result["fold"].str.lower().str.contains("blind").any():
        raise ValueError(f"{source} OOF 包含 blind 折")
    for column in ("origin_time", "train_end"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"{source} OOF 的 {column} 包含无法解析的时间")
    result["target"] = result["target"].astype(str)
    result["horizon"] = pd.to_numeric(result["horizon"], errors="coerce")
    result["actual"] = pd.to_numeric(result["actual"], errors="coerce")
    if result[["horizon", "actual"]].isna().any().any():
        raise ValueError(f"{source} OOF 含空 horizon 或 actual")
    if not np.equal(result["horizon"] % 1, 0).all():
        raise ValueError(f"{source} OOF 的 horizon 必须是整数")
    result["horizon"] = result["horizon"].astype(int)
    if not np.isfinite(result["actual"].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} OOF 的 actual 含 NaN/Inf")
    if result.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{source} OOF 的身份键不唯一")
    return result.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)


def validate_integration_rows(rows: pd.DataFrame) -> dict[str, object]:
    """对 P3 integration OOF 执行固定规模、覆盖和预测列审计。"""

    keys = _parse_key_frame(rows, source="P3 integration")
    required_predictions = [f"{PARENT_ROUTE}__prediction"] + [
        f"{route}__prediction" for route in ROUTE_NAMES
    ]
    missing = sorted(set(required_predictions).difference(rows.columns))
    if missing:
        raise ValueError(f"P3 integration OOF 缺少基础预测列: {missing}")
    predictions = rows.loc[:, required_predictions].apply(pd.to_numeric, errors="coerce")
    if predictions.isna().any().any() or not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("P3 integration OOF 的基础预测含 NaN/Inf")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"P3 integration OOF 行数应为 {EXPECTED_ROWS}，实际为 {len(rows)}")
    folds = _chronological_folds(keys)
    if len(folds) != EXPECTED_FOLDS:
        raise ValueError(f"P3 integration OOF 折数应为 {EXPECTED_FOLDS}，实际为 {len(folds)}")
    origin_columns = ["fold", "origin_time", "train_end"]
    origins = keys.loc[:, origin_columns].drop_duplicates()
    if len(origins) != EXPECTED_ORIGINS:
        raise ValueError(
            f"P3 integration OOF origin 数应为 {EXPECTED_ORIGINS}，实际为 {len(origins)}"
        )
    coverage = keys.groupby(origin_columns, sort=False).size()
    expected_cells = len(TARGETS) * len(HORIZONS)
    if not coverage.eq(expected_cells).all():
        raise ValueError("P3 integration OOF 每个 origin 未完整覆盖两个目标和八个 horizon")
    target_horizons = keys.groupby("target", sort=True)["horizon"].agg(
        lambda values: tuple(sorted(values.unique()))
    )
    if set(target_horizons.index) != set(TARGETS) or any(
        values != HORIZONS for values in target_horizons
    ):
        raise ValueError("P3 integration OOF 未完整覆盖两个目标和八个 horizon")
    return {
        "rows": int(len(rows)),
        "fold_count": len(folds),
        "folds_chronological": folds,
        "origin_count": int(len(origins)),
        "unique_identity_keys": True,
        "blind_labels_used": False,
        "targets": list(TARGETS),
        "horizons": list(HORIZONS),
    }


def validate_matching_keys(
    integration_rows: pd.DataFrame,
    route_rows: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, object]]:
    """机械验证每份原始路线 OOF 与 integration 的完整键完全一致。"""

    base = _parse_key_frame(integration_rows, source="P3 integration")
    checks: dict[str, dict[str, object]] = {}
    if set(route_rows) != set(ROUTE_NAMES):
        raise ValueError(f"原始路线集合必须精确为 {list(ROUTE_NAMES)}")
    for route in ROUTE_NAMES:
        candidate = _parse_key_frame(route_rows[route], source=route)
        compared = base.merge(candidate, on=list(KEY_COLUMNS), how="outer", indicator=True)
        counts = compared["_merge"].value_counts().to_dict()
        parent_only = int(counts.get("left_only", 0))
        route_only = int(counts.get("right_only", 0))
        shared = int(counts.get("both", 0))
        if parent_only or route_only or shared != EXPECTED_ROWS:
            raise ValueError(
                f"{route} 与 P3 integration 完整键不一致: "
                f"integration_only={parent_only}, route_only={route_only}, shared={shared}"
            )
        checks[route] = {
            "matching_keys": True,
            "integration_only": parent_only,
            "route_only": route_only,
            "shared": shared,
            "blind_labels_used": False,
        }
    return checks


def load_and_validate_p3_inputs(input_dir: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """只读加载唯一 P3 输入目录并返回完整准入收据。"""

    root = Path(input_dir).resolve()
    integration_path = root / "integration" / "oof.parquet"
    if not integration_path.is_file():
        raise ValueError(f"缺少唯一 integration 输入: {integration_path}")
    paths = {route: root / filename for route, filename in RAW_ROUTE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"缺少原始路线 OOF: {missing}")

    integration = pd.read_parquet(integration_path)
    summary = validate_integration_rows(integration)
    raw_rows = {route: pd.read_csv(path) for route, path in paths.items()}
    key_checks = validate_matching_keys(integration, raw_rows)
    input_paths = {"integration": integration_path, **paths}
    receipt = {
        **summary,
        "input_dir": str(root),
        "key_checks": key_checks,
        "input_sha256": {name: sha256_file(path) for name, path in sorted(input_paths.items())},
        "read_only_inputs": True,
    }
    return integration, receipt


def _chronological_folds(rows: pd.DataFrame) -> list[str]:
    """按每折最早 origin 排序；同一时刻以稳定折名破平局。"""

    fold_times = (
        rows.assign(fold=rows["fold"].astype(str))
        .groupby("fold", sort=False)["origin_time"]
        .min()
        .reset_index()
    )
    fold_times["origin_time"] = pd.to_datetime(fold_times["origin_time"], errors="coerce")
    if fold_times["origin_time"].isna().any():
        raise ValueError("OOF 折含无法解析的 origin_time")
    ordered = fold_times.sort_values(["origin_time", "fold"], kind="stable")
    return ordered["fold"].astype(str).tolist()


def evaluate_training_gate(
    rows: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    parent_prediction_column: str = f"{PARENT_ROUTE}__prediction",
) -> dict[str, object]:
    """按时间 recent5 复现 P3 冻结门槛，供 18 折候选准入。"""

    required = {
        "fold",
        "origin_time",
        "target",
        "actual",
        prediction_column,
        parent_prediction_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"训练侧稳定门槛缺少字段: {missing}")
    if rows["fold"].astype(str).str.lower().str.contains("blind").any():
        raise ValueError("训练侧稳定门槛包含 blind 折")
    values = rows.loc[:, ["actual", prediction_column, parent_prediction_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("训练侧稳定门槛含 NaN/Inf")
    denominator = np.maximum(np.abs(values["actual"].to_numpy(dtype=float)), 1e-6)
    scored = rows.loc[:, ["fold", "origin_time", "target"]].copy()
    scored["candidate_ape"] = (
        np.abs(
            values["actual"].to_numpy(dtype=float) - values[prediction_column].to_numpy(dtype=float)
        )
        / denominator
    )
    scored["parent_ape"] = (
        np.abs(
            values["actual"].to_numpy(dtype=float)
            - values[parent_prediction_column].to_numpy(dtype=float)
        )
        / denominator
    )

    by_fold = scored.groupby("fold", sort=False)[["candidate_ape", "parent_ape"]].mean()
    fold_improvement_pp = (by_fold["parent_ape"] - by_fold["candidate_ape"]) * 100.0
    by_target = scored.groupby("target", sort=True)[["candidate_ape", "parent_ape"]].mean()
    target_improvement_pp = (by_target["parent_ape"] - by_target["candidate_ape"]) * 100.0
    chronological_folds = _chronological_folds(scored)
    recent_folds = chronological_folds[-5:]
    recent = fold_improvement_pp.reindex(recent_folds)
    if recent.isna().any():
        raise ValueError("训练侧 recent5 折无法与折指标对齐")

    pooled_candidate = float(scored["candidate_ape"].mean())
    pooled_parent = float(scored["parent_ape"].mean())
    improvement_pp = float((pooled_parent - pooled_candidate) * 100.0)
    worst_regression_pp = float((-fold_improvement_pp).max())
    max_target_regression_pp = float((-target_improvement_pp).max())
    checks = {
        "pooled_improvement": improvement_pp >= STATIC_MIN_IMPROVEMENT_PP,
        "recent5_wins": int((recent > 0.0).sum()) >= STATIC_MIN_RECENT5_WINS,
        "worst_fold_regression": (worst_regression_pp <= STATIC_MAX_WORST_FOLD_REGRESSION_PP),
        "target_regression": max_target_regression_pp <= STATIC_MAX_TARGET_REGRESSION_PP,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "pooled_candidate_mape": pooled_candidate,
        "pooled_parent_mape": pooled_parent,
        "improvement_pp": improvement_pp,
        "fold_wins": int((fold_improvement_pp > 0.0).sum()),
        "recent5_folds": recent_folds,
        "recent5_wins": int((recent > 0.0).sum()),
        "worst_fold_regression_pp": worst_regression_pp,
        "max_target_regression_pp": max_target_regression_pp,
        "by_fold_improvement_pp": {
            str(fold): float(fold_improvement_pp.loc[fold]) for fold in chronological_folds
        },
        "by_target_improvement_pp": {
            str(target): float(value) for target, value in target_improvement_pp.items()
        },
    }


def robust_cross_fitted_fusion(
    rows: pd.DataFrame,
    *,
    route_names: Sequence[str] = ROUTE_NAMES,
    input_receipt: Mapping[str, object] | None = None,
) -> RobustFusionResult:
    """逐 held fold 先过滤不稳定候选，再冻结权重并评分 held fold。"""

    routes = tuple(sorted(route_names))
    required = (
        set(KEY_COLUMNS)
        | {f"{PARENT_ROUTE}__prediction"}
        | {f"{route}__prediction" for route in routes}
    )
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"P4 OOF 缺少字段: {missing}")
    if rows["fold"].astype(str).str.lower().str.contains("blind").any():
        raise ValueError("P4 OOF 包含 blind 折")
    if rows.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("P4 OOF 身份键不唯一")

    candidates = pre_registered_weight_candidates(routes)
    candidate_by_name = {name: weights for name, weights in candidates}
    if "parent_only" not in candidate_by_name:
        raise AssertionError("预注册候选缺少 parent_only 回退")
    folds = _chronological_folds(rows)
    output = rows.copy()
    output["prediction"] = np.nan
    output["selected_candidate"] = ""
    output["selection_reason"] = ""
    for route in (PARENT_ROUTE, *routes):
        output[f"weight_{route}"] = 0.0

    trace_records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []
    for held_fold in folds:
        held_mask = rows["fold"].astype(str).eq(held_fold)
        train_rows = rows.loc[~held_mask]
        held_rows = rows.loc[held_mask]
        if train_rows.empty or held_rows.empty:
            raise ValueError(f"fold {held_fold} 无法做 P4 cross-fit")
        train_folds = _chronological_folds(train_rows)
        evaluations: list[tuple[str, dict[str, float], dict[str, object]]] = []
        for candidate_name, weights in candidates:
            predicted_train = _prediction_for_weights(train_rows, weights)
            predicted_train[f"{PARENT_ROUTE}__prediction"] = train_rows[
                f"{PARENT_ROUTE}__prediction"
            ].to_numpy(dtype=float)
            gate = evaluate_training_gate(predicted_train)
            evaluations.append((candidate_name, weights, gate))

        eligible = [item for item in evaluations if bool(item[2]["passed"])]
        if eligible:
            selected_name, selected_weights, selected_gate = min(
                eligible,
                key=lambda item: (float(item[2]["pooled_candidate_mape"]), item[0]),
            )
            selected_reason = "训练侧全部稳定门槛通过，按 pooled MAPE 与稳定名称选中"
        else:
            selected_name = "parent_only"
            selected_weights = candidate_by_name[selected_name]
            selected_gate = next(item[2] for item in evaluations if item[0] == selected_name)
            selected_reason = "训练侧无候选通过全部稳定门槛，回退 A61 parent-only"

        predicted_held = _prediction_for_weights(held_rows, selected_weights)
        output.loc[held_mask, "prediction"] = predicted_held["prediction"].to_numpy(dtype=float)
        output.loc[held_mask, "selected_candidate"] = selected_name
        output.loc[held_mask, "selection_reason"] = selected_reason
        for route in (PARENT_ROUTE, *routes):
            output.loc[held_mask, f"weight_{route}"] = float(selected_weights.get(route, 0.0))

        selection_records.append(
            {
                "held_fold": held_fold,
                "training_fold_count": len(train_folds),
                "training_folds": json.dumps(train_folds, ensure_ascii=False),
                "held_fold_labels_used": False,
                "selected_candidate": selected_name,
                "selected_training_mape": float(selected_gate["pooled_candidate_mape"]),
                "selected_reason": selected_reason,
                "weights": json.dumps(selected_weights, ensure_ascii=False, sort_keys=True),
            }
        )
        for candidate_name, weights, gate in evaluations:
            is_selected = candidate_name == selected_name
            failed_checks = [name for name, passed in gate["checks"].items() if not passed]
            if is_selected:
                reason = selected_reason
            elif gate["passed"]:
                reason = "训练侧门槛通过，但 pooled MAPE/稳定名称排序未选中"
            else:
                reason = f"训练侧门槛未通过: {','.join(failed_checks)}"
            trace_records.append(
                {
                    "held_fold": held_fold,
                    "candidate": candidate_name,
                    "training_fold_count": len(train_folds),
                    "training_folds": json.dumps(train_folds, ensure_ascii=False),
                    "recent5_folds": json.dumps(gate["recent5_folds"], ensure_ascii=False),
                    "held_fold_labels_used": False,
                    "pooled_candidate_mape": float(gate["pooled_candidate_mape"]),
                    "pooled_parent_mape": float(gate["pooled_parent_mape"]),
                    "pooled_improvement_pp": float(gate["improvement_pp"]),
                    "recent5_wins": int(gate["recent5_wins"]),
                    "worst_fold_regression_pp": float(gate["worst_fold_regression_pp"]),
                    "max_target_regression_pp": float(gate["max_target_regression_pp"]),
                    "check_pooled_improvement": bool(gate["checks"]["pooled_improvement"]),
                    "check_recent5_wins": bool(gate["checks"]["recent5_wins"]),
                    "check_worst_fold_regression": bool(gate["checks"]["worst_fold_regression"]),
                    "check_target_regression": bool(gate["checks"]["target_regression"]),
                    "eligible": bool(gate["passed"]),
                    "selected": is_selected,
                    "selected_reason": reason,
                    "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                    "by_fold_improvement_pp": json.dumps(
                        gate["by_fold_improvement_pp"], ensure_ascii=False, sort_keys=True
                    ),
                    "by_target_improvement_pp": json.dumps(
                        gate["by_target_improvement_pp"], ensure_ascii=False, sort_keys=True
                    ),
                }
            )

    output = output.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    if output["prediction"].isna().any():
        raise AssertionError("P4 cross-fit 未覆盖全部 held 行")
    trace = pd.DataFrame(trace_records)
    selections = pd.DataFrame(selection_records)
    final_gate = static_fusion_gate(output)
    status = "ROBUST_STATIC_ELIGIBLE" if final_gate["passed"] else "STOP_STATIC_FUSION"
    report: dict[str, object] = {
        "experiment": "P4_robust_cross_fitted_fusion",
        "status": status,
        "robust_static_eligible": bool(final_gate["passed"]),
        "blind_labels_used": False,
        "platform_reference_used": False,
        "future_perturbation": "NOT_RUN",
        "base_models_retrained": False,
        "held_fold_labels_used_for_selection": False,
        "rows": int(len(output)),
        "fold_count": len(folds),
        "candidate_count_per_fold": len(candidates),
        "candidate_evaluations": int(len(trace)),
        "selection_policy": (
            "在训练侧 18 folds 先通过原 static_fusion_gate 四项门槛；"
            "通过者按 pooled MAPE、候选名称排序；无通过者回退 parent_only"
        ),
        "thresholds": {
            "min_improvement_pp": STATIC_MIN_IMPROVEMENT_PP,
            "min_recent5_wins": STATIC_MIN_RECENT5_WINS,
            "max_worst_fold_regression_pp": STATIC_MAX_WORST_FOLD_REGRESSION_PP,
            "max_target_regression_pp": STATIC_MAX_TARGET_REGRESSION_PP,
        },
        "final_static_fusion_gate": final_gate,
        "selected_per_fold": selections.loc[
            :, ["held_fold", "selected_candidate", "selected_reason", "weights"]
        ].to_dict(orient="records"),
        "input_validation": dict(input_receipt) if input_receipt is not None else None,
    }
    return RobustFusionResult(rows=output, trace=trace, selections=selections, report=report)


def write_robust_fusion_artifacts(result: RobustFusionResult, run_dir: str | Path) -> Path:
    """写入全新 P4 实验目录，不触碰 best 或正式提交。"""

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    result.rows.to_parquet(output / "oof.parquet", index=False)
    result.rows.to_csv(output / "oof.csv", index=False, encoding="utf-8")
    result.trace.to_parquet(output / "candidate_trace.parquet", index=False)
    result.trace.to_csv(output / "candidate_trace.csv", index=False, encoding="utf-8")
    result.selections.to_csv(output / "fold_selections.csv", index=False, encoding="utf-8")
    (output / "report.json").write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_paths = {
        name: output / name
        for name in (
            "oof.parquet",
            "oof.csv",
            "candidate_trace.parquet",
            "candidate_trace.csv",
            "fold_selections.csv",
            "report.json",
        )
    }
    hashes = {name: sha256_file(path) for name, path in artifact_paths.items()}
    (output / "hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


__all__ = [
    "EXPECTED_FOLDS",
    "EXPECTED_ORIGINS",
    "EXPECTED_ROWS",
    "ROUTE_NAMES",
    "RobustFusionResult",
    "evaluate_training_gate",
    "load_and_validate_p3_inputs",
    "robust_cross_fitted_fusion",
    "sha256_file",
    "validate_integration_rows",
    "validate_matching_keys",
    "write_robust_fusion_artifacts",
]
