"""实验登记与外层 OOF 候选编排。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
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


RESULTS_ROOT = Path("results")
DEFAULT_RANDOM_SEED = 20250731


def stable_hash(value: object) -> str:
    """对 JSON 可序列化对象生成稳定 SHA256。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """同时覆盖字段结构、索引和数值内容的数据指纹。"""

    metadata = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_name": str(frame.index.name),
        "index_dtype": str(frame.index.dtype),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64")
    digest.update(values.tobytes())
    return digest.hexdigest()


def feature_schema_fingerprint(features: pd.DataFrame) -> str:
    """仅对特征 schema 生成指纹，便于 checkpoint 拒绝错列恢复。"""

    return stable_hash(
        {
            "columns": [str(column) for column in features.columns],
            "dtypes": [str(dtype) for dtype in features.dtypes],
            "index_name": str(features.index.name),
            "index_dtype": str(features.index.dtype),
            "shape": [int(features.shape[0]), int(features.shape[1])],
        }
    )


def config_fingerprint(config: object) -> str:
    """对 dataclass 或普通配置对象生成稳定指纹。"""

    try:
        value = asdict(config)
    except TypeError:
        value = config
    return stable_hash(value)


def build_fingerprints(
    *,
    config: object | None = None,
    dataset: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    model_params: object | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    """构造运行和 checkpoint 共同使用的来源指纹。"""

    requirements = Path("requirements-lock.txt")
    return {
        "config_hash": config_fingerprint(config) if config is not None else "unavailable",
        "feature_hash": (
            feature_schema_fingerprint(features) if features is not None else "unavailable"
        ),
        "dataset_hash": dataframe_fingerprint(dataset) if dataset is not None else "unavailable",
        "requirements_hash": sha256(requirements) if requirements.exists() else "unavailable",
        "model_params_hash": stable_hash(model_params) if model_params is not None else "unavailable",
        "random_seed": int(random_seed),
        "git_commit": current_commit(),
    }


def _run_partition(prefix: str) -> tuple[tuple[str, ...], str]:
    """根据任务前缀决定历史归档分区和 latest 文件名。"""

    if prefix in {
        "legacy_oof",
        "online_oof",
        "backtest_v1",
        "backtest_v2",
        "backtest_v25",
        "backtest_v3",
    }:
        return ("raw", "runs", "oof"), "oof"
    if prefix.startswith("compare_") or prefix in {"select_model", "select_candidate"}:
        return ("raw", "runs", "comparisons"), "comparison"
    if prefix in {"train_model", "prelim_pipeline"}:
        return ("raw", "runs", "training"), "training"
    if prefix.startswith("experiment_"):
        return ("raw", "runs", "experiments"), "experiment"
    if prefix == "fit_stacker":
        return ("raw", "runs", "experiments"), "experiment"
    if prefix == "leakage_audit":
        return ("raw", "runs", "audits", "leakage"), "audit"
    if prefix in {"data_audit", "preprocess"}:
        return ("raw", "runs", "audits", "data"), "audit"
    if prefix in {"evaluate_frozen", "compare_reproduction"}:
        return ("raw", "runs", "audits"), "audit"
    if prefix in {"legacy_pipeline", "competition_pipeline"}:
        return ("raw", "runs", "experiments"), "experiment"
    return ("raw", "runs", "other"), "experiment"


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def new_run_dir(root: str | Path, prefix: str) -> Path:
    """为每次运行创建分类归档目录，并登记初始 manifest。"""

    partition, run_type = _run_partition(prefix)
    root_path = Path(root)
    if root_path.name == "runs" and root_path.parent.name == "raw":
        base = root_path.parent.parent
    else:
        base = root_path
    timestamp = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output = base.joinpath(*partition, timestamp)
    if output.exists():
        output = output.with_name(f"{timestamp}_{os.getpid()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "manifest.json",
        {
            "run_id": output.name,
            "run_type": run_type,
            "run_prefix": prefix,
            "status": "running",
            "is_smoke": False,
            "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "git_commit": current_commit(),
            "requirements_hash": (
                sha256(Path("requirements-lock.txt"))
                if Path("requirements-lock.txt").exists()
                else "unavailable"
            ),
            "random_seed": DEFAULT_RANDOM_SEED,
        },
    )
    return output


def _latest_name(run_type: str) -> str:
    return {
        "oof": "oof.json",
        "comparison": "comparison.json",
        "training": "training.json",
        "experiment": "experiment.json",
        "audit": "audit.json",
    }.get(run_type, "run.json")


def finalize_run(run_dir: str | Path, manifest: dict[str, object]) -> dict[str, object]:
    """写入完成状态，并更新 results/latest 下的分类指针。

    先合并初始 manifest，避免结束时覆盖 ``created_at``、``run_prefix`` 和
    初始 git SHA；调用方提供的字段优先级更高。
    """

    run_path = Path(run_dir)
    existing: dict[str, object] = {}
    existing_path = run_path / "manifest.json"
    if existing_path.exists():
        try:
            loaded = json.loads(existing_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = {}
    payload = {**existing, **manifest}
    payload.setdefault("run_id", run_path.name)
    payload.setdefault("run_prefix", existing.get("run_prefix", "unknown"))
    payload.setdefault("created_at", existing.get("created_at", pd.Timestamp.now(tz="Asia/Shanghai").isoformat()))
    payload.setdefault("git_commit", existing.get("git_commit", current_commit()))
    payload.setdefault(
        "requirements_hash",
        sha256(Path("requirements-lock.txt"))
        if Path("requirements-lock.txt").exists()
        else "unavailable",
    )
    payload.setdefault("config_hash", config_fingerprint(payload.get("config", {})))
    payload.setdefault("feature_hash", payload.get("feature_schema_hash", "unavailable"))
    payload.setdefault("dataset_hash", payload.get("data_hash", "unavailable"))
    payload.setdefault("model_params_hash", payload.get("model_params_hash", "unavailable"))
    payload.setdefault("random_seed", DEFAULT_RANDOM_SEED)
    payload.setdefault("status", "completed")
    payload["status"] = "completed"
    payload["finished_at"] = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    payload["run_dir"] = str(run_path.resolve())
    write_json(run_path / "manifest.json", payload)
    run_type = str(payload.get("run_type", "experiment"))
    latest_path = RESULTS_ROOT / "latest" / _latest_name(run_type)
    write_json(latest_path, payload)
    return payload


def is_eligible_for_best(manifest: dict[str, object]) -> bool:
    """判断运行是否满足正式 best 候选的机械门槛。"""

    required = (
        manifest.get("status") == "completed"
        and not bool(manifest.get("is_smoke", False))
        and bool(manifest.get("leakage_passed", False))
        and bool(manifest.get("tests_passed", False))
        and bool(manifest.get("submission_valid", False))
    )
    files = manifest.get("best_files", {})
    required_files = isinstance(files, dict) and all(
        isinstance(files.get(key), str) for key in ("model", "result", "submission")
    )
    mape = manifest.get("pooled_mape")
    evidence = manifest.get("promotion_evidence")
    evidence_ok = isinstance(evidence, dict) and all(
        isinstance(evidence.get(key), str)
        for key in ("oof_report", "leakage_report", "pytest_report", "submission_report")
    )
    return bool(
        required
        and required_files
        and evidence_ok
        and isinstance(mape, (int, float))
        and float(mape) >= 0
    )


def _read_json_object(path: Path) -> dict[str, object] | None:
    """读取机械收据；格式错误时返回 ``None`` 而不是信任人工标志。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _bool_receipt(payload: dict[str, object], keys: tuple[str, ...]) -> bool:
    """从不同脚本约定的字段名中读取通过状态。"""

    return any(payload.get(key) is True for key in keys)


def _find_numeric(payload: object, keys: tuple[str, ...]) -> float | None:
    """在有限层级的 JSON 报告中查找数值指标。"""

    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        for value in payload.values():
            found = _find_numeric(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_numeric(value, keys)
            if found is not None:
                return found
    return None


def promotion_evidence_passes(run_dir: str | Path, manifest: dict[str, object]) -> bool:
    """读取四类收据并确认其内容与 manifest 一致。"""

    evidence = manifest.get("promotion_evidence")
    if not isinstance(evidence, dict):
        return False
    paths = {
        key: Path(run_dir) / str(evidence.get(key, ""))
        for key in ("oof_report", "leakage_report", "pytest_report", "submission_report")
    }
    receipts = {key: _read_json_object(path) for key, path in paths.items()}
    if any(payload is None for payload in receipts.values()):
        return False
    oof = receipts["oof_report"]
    leakage = receipts["leakage_report"]
    pytest_report = receipts["pytest_report"]
    submission = receipts["submission_report"]
    assert oof is not None
    assert leakage is not None
    assert pytest_report is not None
    assert submission is not None
    pooled = _find_numeric(oof, ("pooled_mape", "score_oof_long"))
    manifest_mape = manifest.get("pooled_mape")
    if pooled is None or not isinstance(manifest_mape, (int, float)):
        return False
    if abs(float(pooled) - float(manifest_mape)) > 1e-12:
        return False
    if not _bool_receipt(leakage, ("passed", "leakage_passed")):
        return False
    if not _bool_receipt(pytest_report, ("passed", "tests_passed")):
        return False
    if pytest_report.get("returncode") not in (None, 0):
        return False
    return _bool_receipt(submission, ("valid", "submission_valid", "passed"))


def promote_if_best(run_dir: str | Path, best_dir: str | Path) -> bool:
    """按 pooled MAPE 比较并原子更新 best 目录，不静默覆盖较优结果。"""

    run_path = Path(run_dir)
    best_path = Path(best_dir)
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not is_eligible_for_best(manifest):
        return False
    evidence = manifest.get("promotion_evidence", {})
    if isinstance(evidence, dict):
        for key in ("oof_report", "leakage_report", "pytest_report", "submission_report"):
            path = run_path / str(evidence[key])
            if not path.is_file() or path.stat().st_size == 0:
                return False
    if not promotion_evidence_passes(run_path, manifest):
        return False
    files = manifest.get("best_files", {})
    if isinstance(files, dict):
        for key in ("model", "result", "submission"):
            path = run_path / str(files[key])
            if not path.is_file() or path.stat().st_size == 0:
                return False
    current_summary = best_path / "summary.json"
    if current_summary.exists():
        current = json.loads(current_summary.read_text(encoding="utf-8"))
        current_mape = current.get("pooled_mape")
        if (
            is_eligible_for_best(current)
            and isinstance(current_mape, (int, float))
            and float(current_mape) <= float(manifest["pooled_mape"])
        ):
            return False
    temp_path = best_path.parent / ".best_tmp"
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=False)
    files = manifest.get("best_files", {})
    if not isinstance(files, dict):
        files = {}
    summary = dict(manifest)
    summary["source_run"] = str(run_path.resolve())
    file_map = {
        "model.joblib": files.get("model", "model.joblib"),
        "input.csv": files.get("input", "input.csv"),
        "result.csv": files.get("result", "result.csv"),
        "submission.zip": files.get("submission", "submission.zip"),
        "report.json": files.get("report", "report.json"),
        "selection.json": files.get("selection", "selection.json"),
        "oof.csv": manifest.get("oof", "oof.csv"),
        "config.json": manifest.get("config_file", "config.json"),
    }
    for target_name, source_name in file_map.items():
        if not isinstance(source_name, str):
            continue
        source = run_path / source_name
        if source.exists() and source.is_file():
            shutil.copy2(source, temp_path / target_name)
    evidence_targets = {
        "oof_report": "oof_report.json",
        "leakage_report": "leakage.json",
        "pytest_report": "pytest.json",
        "submission_report": "submission.json",
    }
    source_evidence = manifest.get("promotion_evidence", {})
    if isinstance(source_evidence, dict):
        copied_evidence: dict[str, str] = {}
        for key, target_name in evidence_targets.items():
            source_name = source_evidence.get(key)
            if not isinstance(source_name, str):
                continue
            source = run_path / source_name
            if source.exists() and source.is_file():
                shutil.copy2(source, temp_path / target_name)
                copied_evidence[key] = target_name
        summary["promotion_evidence"] = copied_evidence
    summary_path = temp_path / "summary.json"
    write_json(summary_path, summary)
    write_json(temp_path / "manifest.json", summary)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = best_path.parent / ".best_previous"
    if backup_path.exists():
        shutil.rmtree(backup_path)
    if best_path.exists():
        best_path.replace(backup_path)
    temp_path.replace(best_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)
    return True


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
