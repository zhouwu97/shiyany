"""实验登记与外层 OOF 候选编排。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from gas_forecast.branches import CrossFittedBranchForecaster
from gas_forecast.candidates import CatBoostDeltaForecaster
from gas_forecast.config import ForecastConfig
from gas_forecast.gas_stage import GasTrajectoryForecaster
from gas_forecast.oof import _base_fold_rows
from gas_forecast.regimes import attach_regimes
from gas_forecast.scoring import ScoreSpec, score_oof_long
from gas_forecast.splits import TimeFold, make_outer_folds
from gas_forecast.targets import add_generator_rest, build_delta_targets


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    git_commit: str
    feature_config: dict[str, object]
    model_config: dict[str, object]
    outer_folds: int
    inner_folds: int
    purge_steps: int
    random_seed: int
    training_command: str
    pooled_mape: dict[str, float]
    by_target: dict[str, object]
    by_horizon: dict[str, object]
    by_regime: dict[str, object]
    worst_day: dict[str, object]
    training_time: float
    prediction_time: float
    artifact_hash: str
    python_version: str


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def new_run_dir(root: str | Path, prefix: str) -> Path:
    """为每次实验创建日期优先、便于浏览的独立结果目录。"""

    labels = {
        "legacy_oof": "OOF结果",
        "compare_candidates": "候选比较结果",
        "compare_experiments": "实验比较结果",
        "train_model": "模型训练结果",
        "leakage_audit": "泄漏审计结果",
        "data_audit": "数据审计结果",
        "preprocess": "数据预处理结果",
        "fit_stacker": "融合训练结果",
        "select_candidate": "候选选择结果",
        "select_model": "模型选择结果",
        "compare_reproduction": "复现比较结果",
        "evaluate_frozen": "冻结评估结果",
        "legacy_pipeline": "旧流水线结果",
        "competition_pipeline": "竞赛流水线结果",
        "prelim_pipeline": "初赛流水线结果",
    }
    if prefix.startswith("backtest_"):
        label = f"{prefix.removeprefix('backtest_').upper()}回测结果"
    elif prefix.startswith("experiment_"):
        label = "增强实验结果"
    else:
        label = labels.get(prefix, "实验结果")
    timestamp = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    output = Path(root) / f"{timestamp}_{label}"
    if output.exists():
        output = Path(root) / f"{timestamp}_{os.getpid()}_{label}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unavailable"


def _prediction_long(
    prediction: pd.DataFrame,
    validation_index: pd.DatetimeIndex,
    config: ForecastConfig,
) -> pd.Series:
    records = []
    for target in config.targets:
        for horizon in config.feature.horizons:
            records.append(
                pd.DataFrame(
                    {
                        "origin_time": validation_index,
                        "target": target,
                        "horizon": 15 * horizon,
                        "prediction": prediction[f"{target}_t+{15 * horizon}_pred"].to_numpy(),
                    }
                )
            )
    return pd.concat(records, ignore_index=True).set_index(
        ["origin_time", "target", "horizon"]
    )["prediction"]


def _experimental_fold(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    fold: TimeFold,
    config: ForecastConfig,
    include_catboost: bool,
    include_gas_trajectory: bool,
) -> pd.DataFrame:
    structural_frame = add_generator_rest(frame)
    structural_config = replace(
        config, targets=("generator_1", "generator_rest", "generator_all")
    )
    deltas = build_delta_targets(
        structural_frame, structural_config.targets, structural_config.feature.horizons
    )
    train_mask, validation_mask = fold.masks(frame.index)
    validation_index = frame.index[validation_mask]
    rows = _base_fold_rows(frame, fold, validation_mask, config)
    rows = attach_regimes(rows, features.loc[validation_mask])
    keys = pd.MultiIndex.from_frame(rows[["origin_time", "target", "horizon"]])
    current_columns = list(structural_config.targets)

    crossfit = CrossFittedBranchForecaster(structural_config).fit(
        features.loc[train_mask],
        deltas.loc[train_mask],
        structural_frame.loc[train_mask, current_columns],
    )
    candidate_predictions = crossfit.predict_candidates(
        features.loc[validation_mask],
        structural_frame.loc[validation_mask, current_columns],
    )
    for name, prediction in candidate_predictions.items():
        long = _prediction_long(prediction, validation_index, config)
        rows[f"crossfit_{name}_pred"] = long.loc[keys].to_numpy()

    if include_catboost:
        catboost = CatBoostDeltaForecaster(structural_config).fit(
            features.loc[train_mask],
            deltas.loc[train_mask],
            structural_frame.loc[train_mask, current_columns],
        )
        prediction = catboost.predict(
            features.loc[validation_mask],
            structural_frame.loc[validation_mask, current_columns],
        )
        rows["catboost_pred"] = _prediction_long(
            prediction, validation_index, config
        ).loc[keys].to_numpy()

    if include_gas_trajectory:
        gas_stage = GasTrajectoryForecaster(structural_config).fit(
            structural_frame.loc[train_mask],
            features.loc[train_mask],
            deltas.loc[train_mask],
            structural_frame.loc[train_mask, current_columns],
        )
        prediction = gas_stage.predict(
            features.loc[validation_mask],
            structural_frame.loc[validation_mask, current_columns],
        )
        rows["gas_trajectory_pred"] = _prediction_long(
            prediction, validation_index, config
        ).loc[keys].to_numpy()
    return rows


def _experimental_named_fold(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    fold: TimeFold,
    config: ForecastConfig,
    include_catboost: bool,
    include_gas_trajectory: bool,
) -> tuple[str, pd.DataFrame]:
    return fold.name, _experimental_fold(
        frame,
        features,
        fold,
        config,
        include_catboost,
        include_gas_trajectory,
    )


def build_experimental_oof(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    config: ForecastConfig | None = None,
    max_folds: int | None = None,
    n_jobs: int = 1,
    include_catboost: bool = False,
    include_gas_trajectory: bool = False,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, float]:
    """生成 cross-fitting/协调及可选增强分支的外层 OOF。"""

    config = config or ForecastConfig()
    folds = make_outer_folds(frame.index, config)
    if max_folds is not None:
        folds = folds[-max_folds:]
    started = time.perf_counter()
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    required = {"crossfit_simplex_regularized_pred"}
    if include_catboost:
        required.add("catboost_pred")
    if include_gas_trajectory:
        required.add("gas_trajectory_pred")
    parts: list[pd.DataFrame] = []
    pending: list[TimeFold] = []
    for fold in folds:
        cached = checkpoint_path / f"fold_{fold.name}.csv" if checkpoint_path else None
        if cached is not None and cached.exists():
            rows = pd.read_csv(cached, parse_dates=["origin_time", "train_end"])
            if required.issubset(rows.columns):
                parts.append(rows)
                continue
        pending.append(fold)
    if n_jobs == 1:
        completed = (
            _experimental_named_fold(
                frame,
                features,
                fold,
                config,
                include_catboost,
                include_gas_trajectory,
            )
            for fold in pending
        )
    else:
        completed = Parallel(
            n_jobs=n_jobs, verbose=10, return_as="generator_unordered"
        )(
            delayed(_experimental_named_fold)(
                frame,
                features,
                fold,
                config,
                include_catboost,
                include_gas_trajectory,
            )
            for fold in pending
        )
    for fold_name, rows in completed:
        parts.append(rows)
        if checkpoint_path is not None:
            rows.to_csv(
                checkpoint_path / f"fold_{fold_name}.csv", index=False, encoding="utf-8"
            )
            write_json(
                checkpoint_path / "manifest.json",
                {
                    "completed_folds": sorted(
                        path.stem.removeprefix("fold_")
                        for path in checkpoint_path.glob("fold_*.csv")
                    ),
                    "fold_count": len(folds),
                    "include_catboost": include_catboost,
                    "include_gas_trajectory": include_gas_trajectory,
                },
            )
    return pd.concat(parts, ignore_index=True), time.perf_counter() - started


def register_experiment(
    rows_path: str | Path,
    report_path: str | Path,
    *,
    experiment_id: str,
    config: ForecastConfig,
    training_command: str,
    training_time: float,
    prediction_time: float = 0.0,
) -> dict[str, object]:
    rows = pd.read_csv(rows_path, parse_dates=["origin_time", "train_end"])
    spec = ScoreSpec()
    reports = {
        column.removesuffix("_pred"): score_oof_long(rows, column, spec=spec)
        for column in rows.columns
        if column.endswith("_pred")
    }
    best = min(reports, key=lambda name: reports[name]["pooled_mape"])
    best_report = reports[best]
    worst_day_name, worst_day_value = max(
        best_report["by_day"].items(), key=lambda item: item[1]
    )
    record = ExperimentRecord(
        experiment_id=experiment_id,
        git_commit=current_commit(),
        feature_config=asdict(config.feature),
        model_config=asdict(config.model),
        outer_folds=int(rows["fold"].nunique()),
        inner_folds=config.model.inner_folds,
        purge_steps=max(config.feature.horizons),
        random_seed=config.model.random_state,
        training_command=training_command,
        pooled_mape={name: float(report["pooled_mape"]) for name, report in reports.items()},
        by_target={name: report["by_target"] for name, report in reports.items()},
        by_horizon={name: report["by_horizon"] for name, report in reports.items()},
        by_regime={name: report.get("by_regime", {}) for name, report in reports.items()},
        worst_day={"candidate": best, "day": worst_day_name, "mape": worst_day_value},
        training_time=float(training_time),
        prediction_time=float(prediction_time),
        artifact_hash=sha256(rows_path),
        python_version=platform.python_version(),
    )
    payload = {"record": asdict(record), "reports": reports, "best_candidate": best}
    write_json(report_path, payload)
    return payload
