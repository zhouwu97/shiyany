"""Strict C0 之后的初赛冲分实验基础设施。

本模块只处理已经严格 OOF 的长表预测，不负责改变外层折、purge 或特征语义。
所有可学习组合都按折时间顺序训练，当前折及未来折不会参与自身权重拟合。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gas_forecast.scoring import competition_mape, score_oof_long


STRICT_C0_POOLED_MAPE = 0.05297932227793573
STRICT_C0_TARGET_MAPE = {
    "generator_1": 0.06130328,
    "generator_all": 0.04465537,
}
STRICT_C0_BLIND_MAPE = 0.05790875
STRICT_C0_SCORING_ROWS = 62_858
DEFAULT_BRANCH_COLUMNS = (
    "persistence_pred",
    "ridge_pred",
    "recent_ridge_pred",
    "gas_ridge_pred",
    "lgb_residual_pred",
)
REGISTRY_COLUMNS = (
    "experiment_id",
    "parent",
    "date",
    "model",
    "target_scope",
    "horizon_scope",
    "n_params",
    "pooled_mape",
    "delta_vs_c0",
    "g1_mape",
    "gall_mape",
    "fold_wins",
    "recent5_wins",
    "max_fold_regression",
    "blind_used",
    "leakage_passed",
    "status",
    "next_action",
)


@dataclass(frozen=True)
class StackingConfig:
    """单个严格前向 stacking 候选的自由度配置。"""

    level: str
    lambda_global: float = 0.0
    lambda_target: float = 0.0
    smooth_horizon: bool = False
    max_correction_weight: float = 1.5
    regularization: float = 1e-7

    def __post_init__(self) -> None:
        if self.level not in {"global", "target", "horizon", "target_horizon"}:
            raise ValueError(f"不支持的 stacking level: {self.level}")
        if self.lambda_global < 0 or self.lambda_target < 0:
            raise ValueError("回缩系数不能为负数")
        if self.lambda_global + self.lambda_target > 0.9 + 1e-12:
            raise ValueError("lambda_global + lambda_target 必须 <= 0.9")
        if self.level != "target_horizon" and (
            self.lambda_global != 0 or self.lambda_target != 0 or self.smooth_horizon
        ):
            raise ValueError("三级回缩和平滑只适用于 target_horizon")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> dict[str, object]:
    """生成稳定的列、类型和内容指纹，供 Phase 0 防串档。"""

    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("datetime64[ns]").astype("int64")
    row_hash = pd.util.hash_pandas_object(normalized, index=True).to_numpy(np.uint64)
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "schema": schema,
        "content_sha256": _sha256_bytes(row_hash.tobytes()),
    }


def normalize_branch_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """把历史研究脚本的分支命名统一到 research_v2 契约。"""

    output = rows.copy()
    aliases = {
        "recent_pred": "recent_ridge_pred",
        "gas_pred": "gas_ridge_pred",
        "v2_pred": "c0_pred",
        "routed_pred": "c0_pred",
        "v2_v3_target_reconciled_pred": "c0_pred",
    }
    for source, destination in aliases.items():
        if destination not in output and source in output:
            output[destination] = output[source]
    if "target" not in output:
        output["target"] = "generator_1"
    if "origin_time" in output:
        output["origin_time"] = pd.to_datetime(output["origin_time"])
    if "horizon" in output and output["horizon"].max() <= 8:
        output["horizon"] = output["horizon"].astype(int) * 15
    return output


def validate_oof_contract(rows: pd.DataFrame, prediction_columns: Sequence[str]) -> None:
    required = {
        "fold",
        "origin_time",
        "target",
        "horizon",
        "actual",
        "persistence_pred",
        *prediction_columns,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"OOF 缓存缺少字段: {missing}")
    if rows.duplicated(["fold", "origin_time", "target", "horizon"]).any():
        raise ValueError("OOF 缓存存在重复 fold×origin×target×horizon")
    numeric = rows.loc[:, ["actual", "persistence_pred", *prediction_columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("OOF 真实值或预测值含缺失/非有限数")


def freeze_research_base(
    c0_rows: pd.DataFrame,
    branch_rows: pd.DataFrame,
    output_dir: str | Path,
    *,
    c0_column: str = "c0_pred",
    feature_frame: pd.DataFrame | None = None,
    split_payload: Mapping[str, object] | None = None,
    expected_pooled_mape: float | None = STRICT_C0_POOLED_MAPE,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    """冻结 Phase 0 的统一 Parquet 缓存并验证 Strict C0 可复现性。"""

    c0 = normalize_branch_frame(c0_rows)
    branch_source = normalize_branch_frame(branch_rows)
    if c0_column not in c0_rows.columns and c0_column not in c0.columns:
        raise ValueError(f"Strict C0 长表缺少 {c0_column}")
    if c0_column in c0_rows.columns:
        c0["c0_pred"] = pd.to_numeric(c0_rows[c0_column], errors="coerce")
    keys = ["fold", "origin_time", "target", "horizon"]
    source_columns = keys + [
        column
        for column in (*DEFAULT_BRANCH_COLUMNS[1:], "v2_pred", "v3_pred")
        if column in branch_source
    ]
    base_columns = keys + [
        column
        for column in ("actual", "current_value", "persistence_pred", "c0_pred")
        if column in c0
    ]
    branches = c0.loc[:, base_columns].merge(
        branch_source.loc[:, source_columns], on=keys, how="left", validate="one_to_one"
    )
    for column in DEFAULT_BRANCH_COLUMNS[1:]:
        if column not in branches:
            branches[column] = np.nan
    source_match = branches[list(DEFAULT_BRANCH_COLUMNS[1:])].notna().all(axis=1)
    branches["branch_available"] = source_match.astype("int8")
    for column in DEFAULT_BRANCH_COLUMNS[1:]:
        branches[column] = branches[column].fillna(branches["c0_pred"])
    validate_oof_contract(branches, DEFAULT_BRANCH_COLUMNS + ("c0_pred",))
    validate_oof_contract(c0, ("c0_pred",))

    score = score_oof_long(c0, "c0_pred")
    if expected_pooled_mape is not None:
        checks = {
            "pooled": (float(score["pooled_mape"]), expected_pooled_mape),
            "generator_1": (
                float(score["by_target"]["generator_1"]),
                STRICT_C0_TARGET_MAPE["generator_1"],
            ),
            "generator_all": (
                float(score["by_target"]["generator_all"]),
                STRICT_C0_TARGET_MAPE["generator_all"],
            ),
            "blind": (float(score["by_fold"]["blind"]), STRICT_C0_BLIND_MAPE),
        }
        failed = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in checks.items()
            if abs(actual - expected) >= tolerance
        }
        if int(score["rows"]) != STRICT_C0_SCORING_ROWS:
            failed["rows"] = {
                "actual": int(score["rows"]),
                "expected": STRICT_C0_SCORING_ROWS,
            }
        if failed:
            raise ValueError(f"Strict C0 指标无法精确复现: {failed}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    c0_columns = keys + [
        column
        for column in ("train_end", "actual", "current_value", "persistence_pred", "c0_pred")
        if column in c0
    ]
    branch_columns = keys + [
        column
        for column in (
            "actual",
            "current_value",
            *DEFAULT_BRANCH_COLUMNS,
            "c0_pred",
            "v2_pred",
            "v3_pred",
            "branch_available",
        )
        if column in branches
    ]
    c0.loc[:, c0_columns].to_parquet(destination / "strict_c0_oof.parquet", index=False)
    branches.loc[:, branch_columns].to_parquet(
        destination / "branch_predictions.parquet", index=False
    )
    c0.loc[:, keys].drop_duplicates().to_parquet(
        destination / "fold_assignment.parquet", index=False
    )
    c0.loc[:, keys + ["actual"]].to_parquet(destination / "targets.parquet", index=False)
    metadata_columns = keys + [
        column
        for column in c0.columns
        if column not in {*keys, "actual"} and not column.endswith("_pred")
    ]
    c0.loc[:, list(dict.fromkeys(metadata_columns))].to_parquet(
        destination / "origin_metadata.parquet", index=False
    )

    feature_fingerprint = _frame_fingerprint(feature_frame if feature_frame is not None else c0)
    split_fingerprint = {
        "folds": (
            c0.groupby("fold")["origin_time"]
            .agg(["min", "max", "count"])
            .astype({"count": int})
            .astype(str)
            .to_dict(orient="index")
        ),
        "payload": dict(split_payload or {}),
    }
    baseline_metrics = {
        "pooled_mape": float(score["pooled_mape"]),
        "by_target": score["by_target"],
        "blind_mape": score["by_fold"].get("blind"),
        "rows": int(score["rows"]),
        "tolerance": tolerance,
        "branch_source_rows": int(source_match.sum()),
        "branch_fallback_to_c0_rows": int((~source_match).sum()),
        "branch_source_by_target": {
            str(target): int(source_match.loc[branches["target"].eq(target)].sum())
            for target in sorted(branches["target"].unique())
        },
    }
    for name, payload in (
        ("feature_fingerprint.json", feature_fingerprint),
        ("split_fingerprint.json", split_fingerprint),
        ("baseline_metrics.json", baseline_metrics),
    ):
        (destination / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return baseline_metrics


def read_research_base(base_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(base_dir)
    c0 = pd.read_parquet(base / "strict_c0_oof.parquet")
    branches = pd.read_parquet(base / "branch_predictions.parquet")
    metrics = json.loads((base / "baseline_metrics.json").read_text(encoding="utf-8"))
    reproduced = score_oof_long(c0, "c0_pred")
    if abs(float(reproduced["pooled_mape"]) - float(metrics["pooled_mape"])) >= float(
        metrics.get("tolerance", 1e-8)
    ):
        raise ValueError("Phase 0 缓存重读后无法复现冻结指标")
    return c0, branches


def fit_mape_correction_weights(
    actual: np.ndarray,
    persistence: np.ndarray,
    branch_matrix: np.ndarray,
    *,
    max_weight: float = 1.5,
    regularization: float = 1e-7,
) -> np.ndarray:
    """直接最小化 MAPE，拟合各分支相对 persistence 的非负 correction。"""

    y = np.asarray(actual, dtype=float)
    anchor = np.asarray(persistence, dtype=float)
    matrix = np.asarray(branch_matrix, dtype=float)
    if matrix.ndim != 2 or len(y) != len(anchor) or len(y) != len(matrix):
        raise ValueError("correction 权重输入形状不一致")
    corrections = matrix - anchor[:, None]
    valid = np.isfinite(y) & np.isfinite(anchor) & np.isfinite(corrections).all(axis=1)
    if int(valid.sum()) < max(32, matrix.shape[1] * 8):
        return np.zeros(matrix.shape[1], dtype=float)
    y = y[valid]
    anchor = anchor[valid]
    corrections = corrections[valid]

    def objective(weights: np.ndarray) -> float:
        prediction = anchor + corrections @ weights
        return competition_mape(y, prediction) + regularization * float(weights @ weights)

    result = minimize(
        objective,
        np.full(matrix.shape[1], 0.2, dtype=float),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * matrix.shape[1],
        options={"maxiter": 300, "ftol": 1e-11},
    )
    if result.success and np.isfinite(result.x).all():
        return np.clip(result.x, 0.0, max_weight)
    seeds = np.eye(matrix.shape[1])
    scores = [objective(seed) for seed in seeds]
    return seeds[int(np.argmin(scores))]


def _fold_order(rows: pd.DataFrame) -> list[str]:
    return (
        rows.groupby("fold", sort=False)["origin_time"].min().sort_values().index.astype(str).tolist()
    )


def _fit_weight_map(
    rows: pd.DataFrame,
    branch_columns: Sequence[str],
    group_columns: Sequence[str],
    config: StackingConfig,
) -> dict[tuple[object, ...], np.ndarray]:
    grouped: Iterable[tuple[object, pd.DataFrame]]
    if group_columns:
        grouped = rows.groupby(list(group_columns), sort=True, dropna=False)
    else:
        grouped = [((), rows)]
    output: dict[tuple[object, ...], np.ndarray] = {}
    for key, part in grouped:
        normalized_key = key if isinstance(key, tuple) else (key,)
        output[normalized_key] = fit_mape_correction_weights(
            part["actual"].to_numpy(float),
            part["persistence_pred"].to_numpy(float),
            part.loc[:, list(branch_columns)].to_numpy(float),
            max_weight=config.max_correction_weight,
            regularization=config.regularization,
        )
    return output


def _smooth_target_horizon_weights(
    raw: dict[tuple[object, ...], np.ndarray],
) -> dict[tuple[object, ...], np.ndarray]:
    output = {key: value.copy() for key, value in raw.items()}
    targets = sorted({key[0] for key in raw})
    for target in targets:
        horizons = sorted(key[1] for key in raw if key[0] == target)
        for position, horizon in enumerate(horizons):
            neighbours = [raw[(target, horizon)]]
            coefficients = [0.5]
            if position > 0:
                neighbours.insert(0, raw[(target, horizons[position - 1])])
                coefficients.insert(0, 0.25)
            if position + 1 < len(horizons):
                neighbours.append(raw[(target, horizons[position + 1])])
                coefficients.append(0.25)
            coefficients_array = np.asarray(coefficients, dtype=float)
            coefficients_array /= coefficients_array.sum()
            output[(target, horizon)] = np.average(
                np.stack(neighbours), axis=0, weights=coefficients_array
            )
    return output


def time_ordered_persistence_stack(
    rows: pd.DataFrame,
    branch_columns: Sequence[str] = DEFAULT_BRANCH_COLUMNS[1:],
    *,
    config: StackingConfig,
    output_column: str = "stack_pred",
    baseline_column: str = "c0_pred",
    use_blind_for_reporting: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """按历史折拟合 persistence-centered 权重，并输出逐折权重轨迹。"""

    work = normalize_branch_frame(rows).reset_index(drop=True)
    validate_oof_contract(work, tuple(branch_columns) + (baseline_column,))
    folds = _fold_order(work)
    prediction = work[baseline_column].to_numpy(float).copy()
    trajectories: list[dict[str, object]] = []
    for position, fold in enumerate(folds):
        held = work["fold"].astype(str).eq(fold).to_numpy()
        if position == 0 or (fold == "blind" and not use_blind_for_reporting):
            trajectories.append({"fold": fold, "fallback": "strict_c0"})
            continue
        history_folds = [value for value in folds[:position] if value != "blind"]
        history_mask = work["fold"].astype(str).isin(history_folds)
        if "branch_available" in work:
            history_mask &= work["branch_available"].astype(bool)
        history = work.loc[history_mask]
        if history.empty:
            trajectories.append({"fold": fold, "fallback": "strict_c0"})
            continue

        global_map = _fit_weight_map(history, branch_columns, (), config)
        target_map = _fit_weight_map(history, branch_columns, ("target",), config)
        if config.level == "global":
            group_columns: tuple[str, ...] = ()
        elif config.level == "target":
            group_columns = ("target",)
        elif config.level == "horizon":
            group_columns = ("horizon",)
        else:
            group_columns = ("target", "horizon")
        raw_map = _fit_weight_map(history, branch_columns, group_columns, config)
        if config.level == "target_horizon" and config.smooth_horizon:
            raw_map = _smooth_target_horizon_weights(raw_map)

        eligible = held.copy()
        if "branch_available" in work:
            eligible &= work["branch_available"].astype(bool).to_numpy()
        held_rows = work.loc[eligible]
        if held_rows.empty:
            trajectories.append({"fold": fold, "fallback": "strict_c0_no_branch"})
            continue
        held_prediction = np.empty(len(held_rows), dtype=float)
        fold_weights: dict[str, list[float]] = {}
        for local_position, (_, row) in enumerate(held_rows.iterrows()):
            if not group_columns:
                raw_key: tuple[object, ...] = ()
            else:
                raw_key = tuple(row[column] for column in group_columns)
            raw = raw_map.get(raw_key, global_map[()])
            if config.level == "target_horizon":
                target_weight = target_map.get((row["target"],), global_map[()])
                residual_weight = 1.0 - config.lambda_global - config.lambda_target
                weights = (
                    config.lambda_global * global_map[()]
                    + config.lambda_target * target_weight
                    + residual_weight * raw
                )
            else:
                weights = raw
            anchor = float(row["persistence_pred"])
            branch = row.loc[list(branch_columns)].to_numpy(dtype=float)
            held_prediction[local_position] = anchor + (branch - anchor) @ weights
            fold_weights["|".join(map(str, raw_key)) or "global"] = weights.tolist()
        prediction[eligible] = held_prediction
        trajectories.append({"fold": fold, "weights": fold_weights})
    work[output_column] = prediction
    report = evaluate_candidate(
        work,
        output_column,
        baseline_column=baseline_column,
        include_blind=use_blind_for_reporting,
    )
    report.update(
        {
            "config": asdict(config),
            "branch_columns": list(branch_columns),
            "weight_trajectory": trajectories,
        }
    )
    return work, report


def stacking_parameter_grid() -> list[StackingConfig]:
    output = [
        StackingConfig("global"),
        StackingConfig("target"),
        StackingConfig("horizon"),
    ]
    for global_weight in (0.25, 0.50, 0.75):
        for target_weight in (0.00, 0.25, 0.50):
            if global_weight + target_weight <= 0.9:
                for smooth in (False, True):
                    output.append(
                        StackingConfig(
                            "target_horizon",
                            lambda_global=global_weight,
                            lambda_target=target_weight,
                            smooth_horizon=smooth,
                        )
                    )
    return output


def run_stacking_suite(
    rows: pd.DataFrame,
    branch_columns: Sequence[str] = DEFAULT_BRANCH_COLUMNS[1:],
    *,
    baseline_column: str = "c0_pred",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """运行 S0–S3 的冻结小网格；候选选择完全排除 blind。"""

    work = normalize_branch_frame(rows).reset_index(drop=True)
    validate_oof_contract(work, tuple(branch_columns) + (baseline_column,))
    folds = _fold_order(work)
    configs: list[tuple[str, StackingConfig]] = []
    for index, config in enumerate(stacking_parameter_grid()):
        name = f"S{index:02d}_{config.level}"
        if config.level == "target_horizon":
            name += (
                f"_lg{config.lambda_global:.2f}_lt{config.lambda_target:.2f}"
                f"_{'smooth' if config.smooth_horizon else 'raw'}"
            )
        configs.append((name, config))

    predictions = {
        name: work[baseline_column].to_numpy(float).copy() for name, _ in configs
    }
    trajectories: dict[str, list[dict[str, object]]] = {name: [] for name, _ in configs}
    fit_config = StackingConfig("global")
    for position, fold in enumerate(folds):
        held_mask = work["fold"].astype(str).eq(fold)
        if position == 0 or fold == "blind":
            for name, _ in configs:
                trajectories[name].append({"fold": fold, "fallback": "strict_c0"})
            continue
        history_folds = [value for value in folds[:position] if value != "blind"]
        history_mask = work["fold"].astype(str).isin(history_folds)
        if "branch_available" in work:
            history_mask &= work["branch_available"].astype(bool)
        history = work.loc[history_mask]
        global_map = _fit_weight_map(history, branch_columns, (), fit_config)
        target_map = _fit_weight_map(history, branch_columns, ("target",), fit_config)
        horizon_map = _fit_weight_map(history, branch_columns, ("horizon",), fit_config)
        target_horizon_raw = _fit_weight_map(
            history, branch_columns, ("target", "horizon"), fit_config
        )
        target_horizon_smooth = _smooth_target_horizon_weights(target_horizon_raw)
        eligible = held_mask.copy()
        if "branch_available" in work:
            eligible &= work["branch_available"].astype(bool)
        held_rows = work.loc[eligible]
        for name, config in configs:
            if held_rows.empty:
                trajectories[name].append(
                    {"fold": fold, "fallback": "strict_c0_no_branch"}
                )
                continue
            if config.level == "global":
                group_columns: tuple[str, ...] = ()
                raw_map = global_map
            elif config.level == "target":
                group_columns = ("target",)
                raw_map = target_map
            elif config.level == "horizon":
                group_columns = ("horizon",)
                raw_map = horizon_map
            else:
                group_columns = ("target", "horizon")
                raw_map = (
                    target_horizon_smooth if config.smooth_horizon else target_horizon_raw
                )
            if group_columns:
                grouped: Iterable[tuple[object, pd.DataFrame]] = held_rows.groupby(
                    list(group_columns), sort=True, dropna=False
                )
            else:
                grouped = [((), held_rows)]
            fold_weights: dict[str, list[float]] = {}
            for key, part in grouped:
                normalized_key = key if isinstance(key, tuple) else (key,)
                raw = raw_map.get(normalized_key, global_map[()])
                if config.level == "target_horizon":
                    target_weight = target_map.get((part.iloc[0]["target"],), global_map[()])
                    residual_weight = 1.0 - config.lambda_global - config.lambda_target
                    weights = (
                        config.lambda_global * global_map[()]
                        + config.lambda_target * target_weight
                        + residual_weight * raw
                    )
                else:
                    weights = raw
                anchor = part["persistence_pred"].to_numpy(float)
                matrix = part.loc[:, list(branch_columns)].to_numpy(float)
                predictions[name][part.index.to_numpy()] = anchor + (matrix - anchor[:, None]) @ weights
                fold_weights["|".join(map(str, normalized_key)) or "global"] = weights.tolist()
            trajectories[name].append({"fold": fold, "weights": fold_weights})

    reports: dict[str, dict[str, object]] = {}
    table: list[dict[str, object]] = []
    for name, config in configs:
        work[name] = predictions[name]
        report = evaluate_candidate(work, name, baseline_column=baseline_column)
        report.update(
            {
                "config": asdict(config),
                "branch_columns": list(branch_columns),
                "weight_trajectory": trajectories[name],
            }
        )
        reports[name] = report
        table.append({"candidate": name, **_flat_evaluation(report)})
    ranking = pd.DataFrame(table).sort_values("pooled_mape", kind="stable").reset_index(drop=True)
    best = str(ranking.iloc[0]["candidate"])
    return work, {
        "baseline": baseline_column,
        "selection_scope": "development_only",
        "ranking": ranking.to_dict(orient="records"),
        "best_candidate": best,
        "best_report": reports[best],
        "reports": reports,
    }


def _simplex_weights(predictions: np.ndarray, actual: np.ndarray) -> np.ndarray:
    matrix = np.asarray(predictions, dtype=float)
    y = np.asarray(actual, dtype=float)
    n_branches = matrix.shape[1]

    def objective(weights: np.ndarray) -> float:
        return competition_mape(y, matrix @ weights)

    result = minimize(
        objective,
        np.full(n_branches, 1.0 / n_branches),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_branches,
        constraints={"type": "eq", "fun": lambda values: values.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-11},
    )
    if result.success:
        return result.x
    scores = [objective(np.eye(n_branches)[index]) for index in range(n_branches)]
    return np.eye(n_branches)[int(np.argmin(scores))]


def oracle_gap_diagnostics(
    rows: pd.DataFrame,
    branch_columns: Sequence[str] = DEFAULT_BRANCH_COLUMNS,
    *,
    baseline_column: str = "c0_pred",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """计算 current、best branch、full oracle 和双向 split-half oracle。"""

    work = normalize_branch_frame(rows).reset_index(drop=True)
    validate_oof_contract(work, tuple(branch_columns) + (baseline_column,))
    full_prediction = np.full(len(work), np.nan)
    best_prediction = np.full(len(work), np.nan)
    split_prediction = np.full(len(work), np.nan)
    cells: list[dict[str, object]] = []
    group_columns = ["target", "horizon", "fold"]
    for keys, part in work.groupby(group_columns, sort=True):
        ordered = part.sort_values("origin_time")
        positions = ordered.index.to_numpy()
        has_branches = (
            bool(ordered["branch_available"].astype(bool).all())
            if "branch_available" in ordered
            else True
        )
        if str(keys[2]) == "blind" or not has_branches:
            baseline = ordered[baseline_column].to_numpy(float)
            full_prediction[positions] = baseline
            best_prediction[positions] = baseline
            split_prediction[positions] = baseline
            continue
        matrix = ordered.loc[:, list(branch_columns)].to_numpy(float)
        actual = ordered["actual"].to_numpy(float)
        weights = _simplex_weights(matrix, actual)
        full_prediction[positions] = matrix @ weights
        branch_scores = np.array(
            [competition_mape(actual, matrix[:, index]) for index in range(matrix.shape[1])]
        )
        best_index = int(np.argmin(branch_scores))
        best_prediction[positions] = matrix[:, best_index]
        half = len(ordered) // 2
        if half < max(16, matrix.shape[1] * 4) or len(ordered) - half < max(
            16, matrix.shape[1] * 4
        ):
            split_prediction[positions] = ordered[baseline_column].to_numpy(float)
            split_weights = {"first_to_second": None, "second_to_first": None}
        else:
            first_weights = _simplex_weights(matrix[:half], actual[:half])
            second_weights = _simplex_weights(matrix[half:], actual[half:])
            local = np.empty(len(ordered), dtype=float)
            local[:half] = matrix[:half] @ second_weights
            local[half:] = matrix[half:] @ first_weights
            split_prediction[positions] = local
            split_weights = {
                "first_to_second": first_weights.tolist(),
                "second_to_first": second_weights.tolist(),
            }
        cells.append(
            {
                "target": keys[0],
                "horizon": int(keys[1]),
                "fold": keys[2],
                "rows": int(len(ordered)),
                "current_c0_mape": competition_mape(actual, ordered[baseline_column]),
                "best_single_branch": branch_columns[best_index],
                "best_single_mape": float(branch_scores[best_index]),
                "full_oracle_mape": competition_mape(actual, matrix @ weights),
                "split_half_oracle_mape": competition_mape(
                    actual, split_prediction[positions]
                ),
                "full_oracle_weights": weights.tolist(),
                "split_half_weights": split_weights,
            }
        )
    work["best_single_pred"] = best_prediction
    work["full_oracle_pred"] = full_prediction
    work["split_half_oracle_pred"] = split_prediction
    development = work.loc[~work["fold"].astype(str).eq("blind")]
    current = competition_mape(development["actual"], development[baseline_column])
    split_score = competition_mape(development["actual"], development["split_half_oracle_pred"])
    gap_pp = (current - split_score) * 100.0
    if gap_pp >= 0.030:
        verdict = "A"
    elif gap_pp >= 0.015:
        verdict = "B"
    elif gap_pp < 0.010:
        verdict = "C"
    else:
        verdict = "BORDERLINE"
    report = {
        "scope": "development_only",
        "rows": int(len(development)),
        "current_c0_mape": current,
        "best_single_mape": competition_mape(
            development["actual"], development["best_single_pred"]
        ),
        "full_oracle_mape": competition_mape(
            development["actual"], development["full_oracle_pred"]
        ),
        "split_half_oracle_mape": split_score,
        "oracle_gap_pp": gap_pp,
        "verdict": verdict,
        "cells": cells,
    }
    return work, report


def evaluate_candidate(
    rows: pd.DataFrame,
    candidate_column: str,
    *,
    baseline_column: str = "c0_pred",
    include_blind: bool = False,
) -> dict[str, object]:
    work = rows if include_blind else rows.loc[~rows["fold"].astype(str).eq("blind")]
    candidate = score_oof_long(work, candidate_column)
    baseline = score_oof_long(work, baseline_column)
    fold_order = _fold_order(work)
    fold_delta = {
        fold: (candidate["by_fold"][fold] - baseline["by_fold"][fold]) * 100.0
        for fold in fold_order
    }
    recent = fold_order[-5:]
    return {
        "scope": "all" if include_blind else "development_only",
        "candidate": candidate,
        "baseline": baseline,
        "delta_pp": (candidate["pooled_mape"] - baseline["pooled_mape"]) * 100.0,
        "fold_wins": int(sum(value < 0 for value in fold_delta.values())),
        "recent5_wins": int(sum(fold_delta[fold] < 0 for fold in recent)),
        "max_fold_regression_pp": float(max(fold_delta.values())),
        "fold_delta_pp": fold_delta,
    }


def _flat_evaluation(report: Mapping[str, object]) -> dict[str, object]:
    candidate = report["candidate"]
    by_target = candidate["by_target"]
    return {
        "pooled_mape": candidate["pooled_mape"],
        "delta_pp": report["delta_pp"],
        "generator_1_mape": by_target.get("generator_1"),
        "generator_all_mape": by_target.get("generator_all"),
        "fold_wins": report["fold_wins"],
        "recent5_wins": report["recent5_wins"],
        "max_fold_regression_pp": report["max_fold_regression_pp"],
    }


def e21_crossing_routes(
    rows: pd.DataFrame,
    *,
    c0_column: str = "c0_pred",
    e21_column: str = "e21_pred",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """只评估冻结的 R75/R90/R105，generator_all 始终保持 C0。"""

    work = normalize_branch_frame(rows).reset_index(drop=True)
    validate_oof_contract(work, (c0_column, e21_column))
    routes = {"all_c0": 135, "R75": 75, "R90": 90, "R105": 105}
    reports: dict[str, object] = {}
    for name, threshold in routes.items():
        column = f"e21_{name}_pred"
        use_e21 = work["target"].eq("generator_1") & work["horizon"].ge(threshold)
        use_e21 &= ~work["fold"].astype(str).eq("blind")
        if name == "all_c0":
            use_e21[:] = False
        work[column] = np.where(use_e21, work[e21_column], work[c0_column])
        reports[name] = evaluate_candidate(work, column, baseline_column=c0_column)
    ranking = sorted(reports, key=lambda name: reports[name]["candidate"]["pooled_mape"])
    return work, {
        "routes": reports,
        "ranking": ranking,
        "best_route": ranking[0],
    }


def diversity_sweep(
    rows: pd.DataFrame,
    challenger_columns: Sequence[str],
    *,
    baseline_column: str = "c0_pred",
    weights: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """以固定小权重重新审查 standalone 较弱但可能互补的候选。"""

    work = normalize_branch_frame(rows).reset_index(drop=True)
    validate_oof_contract(work, tuple(challenger_columns) + (baseline_column,))
    dev = ~work["fold"].astype(str).eq("blind")
    actual = work.loc[dev, "actual"].to_numpy(float)
    baseline = work.loc[dev, baseline_column].to_numpy(float)
    baseline_residual = actual - baseline
    baseline_mape = competition_mape(actual, baseline)
    records: list[dict[str, object]] = []
    for challenger in challenger_columns:
        challenger_values = work[challenger].to_numpy(float)
        standalone = challenger_values[dev]
        correlation = float(np.corrcoef(baseline_residual, actual - standalone)[0, 1])
        best_weight = 0.0
        best_mape = baseline_mape
        for weight in weights:
            column = f"blend_{challenger}_{int(weight * 100):02d}"
            work[column] = work[baseline_column]
            work.loc[dev, column] = (
                (1.0 - weight) * work.loc[dev, baseline_column]
                + weight * work.loc[dev, challenger]
            )
            score = competition_mape(actual, work.loc[dev, column])
            if score < best_mape:
                best_mape = score
                best_weight = float(weight)
        improvement_pp = (baseline_mape - best_mape) * 100.0
        status = "STACKING_POOL" if improvement_pp >= 0.005 else (
            "CANDIDATE_POOL" if improvement_pp >= 0.003 else "STOP"
        )
        records.append(
            {
                "challenger": challenger,
                "standalone_mape": competition_mape(actual, standalone),
                "residual_correlation": correlation,
                "best_weight": best_weight,
                "best_blend_mape": best_mape,
                "improvement_pp": improvement_pp,
                "status": status,
            }
        )
    return work, pd.DataFrame(records).sort_values(
        "best_blend_mape", kind="stable"
    ).reset_index(drop=True)


def confirm_frozen_blend_on_blind(
    rows: pd.DataFrame,
    *,
    challenger_column: str,
    baseline_column: str,
    weight: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """一次性确认已冻结 blend；开发折公式不一致时拒绝读取 blind 指标。"""

    if weight not in {0.05, 0.10, 0.15, 0.20, 0.30}:
        raise ValueError("blind 确认权重必须来自冻结 Diversity 网格")
    work = normalize_branch_frame(rows).reset_index(drop=True)
    validate_oof_contract(work, (challenger_column, baseline_column))
    frozen_column = f"blend_{challenger_column}_{int(weight * 100):02d}"
    if frozen_column not in work.columns:
        raise ValueError(f"缺少冻结开发候选列: {frozen_column}")
    blind = work["fold"].astype(str).eq("blind")
    if not blind.any():
        raise ValueError("没有可供一次性确认的 blind 行")
    dev = ~blind
    expected_dev = (
        (1.0 - weight) * work.loc[dev, baseline_column]
        + weight * work.loc[dev, challenger_column]
    )
    if not np.allclose(
        work.loc[dev, frozen_column].to_numpy(float),
        expected_dev.to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("开发集冻结候选与登记的 blend 公式不一致")
    confirmed_column = f"confirmed_{frozen_column}"
    work[confirmed_column] = work[frozen_column]
    work.loc[blind, confirmed_column] = (
        (1.0 - weight) * work.loc[blind, baseline_column]
        + weight * work.loc[blind, challenger_column]
    )
    actual = work.loc[blind, "actual"].to_numpy(float)
    baseline_mape = competition_mape(actual, work.loc[blind, baseline_column])
    candidate_mape = competition_mape(actual, work.loc[blind, confirmed_column])
    delta_pp = (candidate_mape - baseline_mape) * 100.0
    verdict = "BETTER" if delta_pp < 0.0 else "WORSE"
    return work, {
        "candidate": confirmed_column,
        "baseline": baseline_column,
        "challenger": challenger_column,
        "weight": weight,
        "blind_rows": int(blind.sum()),
        "blind_baseline_mape": baseline_mape,
        "blind_candidate_mape": candidate_mape,
        "blind_delta_pp": delta_pp,
        "verdict": verdict,
        "selection_used_blind": False,
    }


def project_production_predictions(
    rows: pd.DataFrame,
    prediction_column: str,
    *,
    output_column: str,
    max_generator_rest: float = 240.0,
) -> pd.DataFrame:
    """对生产预测长表执行与 OOF 一致的容量投影；不要求 actual/persistence。

    与 :func:`project_long_candidate` 共享同一可行域规则
    （g1 [0,200]、gall [0,440]、gall in [g1, g1+240]），但生产帧没有 actual /
    persistence_pred，因此不调用 OOF 契约校验。要求每个 fold×origin×horizon
    同时含两个 target。
    """

    keys = ["fold", "origin_time", "horizon"]
    counts = rows.groupby(keys, observed=True)["target"].nunique()
    if not counts.eq(2).all():
        raise ValueError("容量投影要求每个 fold×origin×horizon 同时包含两个目标")
    wide = rows.pivot(index=keys, columns="target", values=prediction_column)
    if not {"generator_1", "generator_all"}.issubset(wide.columns):
        raise ValueError("容量投影缺少 generator_1 或 generator_all")
    wide["generator_1"] = wide["generator_1"].clip(0.0, 200.0)
    wide["generator_all"] = wide["generator_all"].clip(0.0, 440.0)
    wide["generator_all"] = np.maximum(wide["generator_all"], wide["generator_1"])
    wide["generator_all"] = np.minimum(
        wide["generator_all"], wide["generator_1"] + max_generator_rest
    )
    lookup = wide.stack()
    row_index = pd.MultiIndex.from_frame(rows.loc[:, keys + ["target"]])
    out = rows.copy()
    out[output_column] = lookup.reindex(row_index).to_numpy(dtype=float)
    if out[output_column].isna().any():
        raise ValueError("容量投影后的生产长表存在缺失值")
    return out


def project_long_candidate(
    rows: pd.DataFrame,
    prediction_column: str,
    *,
    output_column: str,
    max_generator_rest: float = 240.0,
) -> pd.DataFrame:
    """把长表候选投影到与生产推理一致的两目标容量可行域。"""

    work = normalize_branch_frame(rows).reset_index(drop=True)
    validate_oof_contract(work, (prediction_column,))
    keys = ["fold", "origin_time", "horizon"]
    counts = work.groupby(keys, observed=True)["target"].nunique()
    if not counts.eq(2).all():
        raise ValueError("容量投影要求每个 fold/origin/horizon 同时包含两个目标")
    wide = work.pivot(index=keys, columns="target", values=prediction_column)
    required = {"generator_1", "generator_all"}
    if not required.issubset(wide.columns):
        raise ValueError("容量投影缺少 generator_1 或 generator_all")
    wide["generator_1"] = wide["generator_1"].clip(0.0, 200.0)
    wide["generator_all"] = wide["generator_all"].clip(0.0, 440.0)
    wide["generator_all"] = np.maximum(wide["generator_all"], wide["generator_1"])
    wide["generator_all"] = np.minimum(
        wide["generator_all"], wide["generator_1"] + max_generator_rest
    )
    lookup = wide.stack()
    row_index = pd.MultiIndex.from_frame(work.loc[:, keys + ["target"]])
    work[output_column] = lookup.reindex(row_index).to_numpy(float)
    if work[output_column].isna().any():
        raise ValueError("容量投影后的长表存在缺失值")
    return work


def decide_experiment_status(
    *,
    delta_pp: float,
    fold_wins: int,
    total_folds: int,
    recent5_wins: int,
    screening: bool = False,
) -> tuple[str, str]:
    """把统一 STOP/KEEP/PROMOTE 规则转成机械状态。"""

    improvement = -delta_pp
    if screening and delta_pp > 0.015:
        return "STOP", "screening 退化超过 0.015pp"
    if improvement < 0.002:
        return "STOP", "完整开发改善不足 0.002pp"
    if fold_wins <= 3 and fold_wins < total_folds / 2:
        return "STOP", "收益集中在不超过 3 个折"
    if improvement >= 0.005 and (fold_wins >= 11 or recent5_wins >= 3):
        return "PROMOTE", "达到 0.005pp 且时间稳定性通过"
    if improvement >= 0.003:
        return "KEEP", "达到候选池保留门槛"
    return "SCREEN", "弱信号，仅保留代码和结果"


class ExperimentRegistry:
    """统一 CSV 实验登记；写入前执行固定字段和状态校验。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: Mapping[str, object]) -> pd.DataFrame:
        payload = {column: record.get(column) for column in REGISTRY_COLUMNS}
        payload["date"] = payload["date"] or datetime.now().astimezone().isoformat()
        if payload["status"] not in {"SCREEN", "KEEP", "PROMOTE", "STOP"}:
            raise ValueError("registry status 只允许 SCREEN/KEEP/PROMOTE/STOP")
        existing = (
            pd.read_csv(self.path)
            if self.path.exists()
            else pd.DataFrame(columns=REGISTRY_COLUMNS)
        )
        if payload["experiment_id"] in set(existing["experiment_id"].astype(str)):
            raise ValueError(f"experiment_id 已存在: {payload['experiment_id']}")
        new_record = pd.DataFrame([payload], columns=REGISTRY_COLUMNS)
        output = new_record if existing.empty else pd.concat(
            [existing, new_record], ignore_index=True
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        output.to_csv(temporary, index=False)
        temporary.replace(self.path)
        return output
