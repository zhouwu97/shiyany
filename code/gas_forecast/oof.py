"""统一生成和持久化逐行外层 OOF 预测。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gas_forecast.config import ForecastConfig
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster
from gas_forecast.model_horizon import HorizonSpecificRidgeForecaster
from gas_forecast.model_v1 import RidgeDeltaForecaster
from gas_forecast.regimes import attach_regimes
from gas_forecast.scoring import ScoreSpec, absolute_percentage_error, score_oof_long
from gas_forecast.splits import TimeFold, make_outer_folds
from gas_forecast.targets import build_delta_targets


SUPPORTED_LEGACY_MODELS = ("v1", "v2", "v25", "v3")
SUPPORTED_OOF_MODELS = SUPPORTED_LEGACY_MODELS + ("horizon_ridge",)


@dataclass(frozen=True)
class OOFResult:
    rows: pd.DataFrame
    report: dict[str, object]


def _make_model(version: str, config: ForecastConfig):
    if version == "v1":
        return RidgeDeltaForecaster(config)
    if version in {"v2", "v25", "v3"}:
        return GasAwareEnsembleForecaster(version, config)
    if version == "horizon_ridge":
        return HorizonSpecificRidgeForecaster(config)
    raise ValueError(f"不支持的 OOF 模型: {version}")


def _base_fold_rows(
    frame: pd.DataFrame,
    fold: TimeFold,
    validation_mask: np.ndarray,
    config: ForecastConfig,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    validation_index = frame.index[validation_mask]
    for target in config.targets:
        current = frame.loc[validation_index, target]
        for horizon in config.feature.horizons:
            actual = frame[target].shift(-horizon).loc[validation_index]
            valid = actual.notna() & current.notna()
            part = pd.DataFrame(
                {
                    "fold": fold.name,
                    "origin_time": validation_index[valid],
                    "train_end": fold.train_end,
                    "target": target,
                    "horizon": 15 * horizon,
                    "actual": actual.loc[valid].to_numpy(dtype=float),
                    "current_value": current.loc[valid].to_numpy(dtype=float),
                    "persistence_pred": current.loc[valid].to_numpy(dtype=float),
                }
            )
            records.append(part)
    return pd.concat(records, ignore_index=True)


def _evaluate_fold(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    fold: TimeFold,
    versions: tuple[str, ...],
    config: ForecastConfig,
) -> pd.DataFrame:
    train_mask, validation_mask = fold.masks(frame.index)
    if not train_mask.any() or not validation_mask.any():
        raise ValueError(f"折 {fold.name} 的训练集或验证集为空")
    rows = _base_fold_rows(frame, fold, validation_mask, config)
    rows = attach_regimes(rows, features.loc[validation_mask])
    validation_index = frame.index[validation_mask]
    keys = pd.MultiIndex.from_frame(rows[["origin_time", "target", "horizon"]])

    for version in versions:
        model = _make_model(version, config).fit(
            features.loc[train_mask],
            deltas.loc[train_mask],
            frame.loc[train_mask, list(config.targets)],
        )
        predicted = model.predict(
            features.loc[validation_mask],
            features.loc[validation_mask, list(config.targets)],
        )
        prediction_records: list[pd.DataFrame] = []
        for target in config.targets:
            for horizon in config.feature.horizons:
                prediction_records.append(
                    pd.DataFrame(
                        {
                            "origin_time": validation_index,
                            "target": target,
                            "horizon": 15 * horizon,
                            "prediction": predicted[f"{target}_t+{15 * horizon}_pred"].to_numpy(),
                        }
                    )
                )
        prediction_long = pd.concat(prediction_records, ignore_index=True).set_index(
            ["origin_time", "target", "horizon"]
        )
        rows[f"{version}_pred"] = prediction_long.loc[keys, "prediction"].to_numpy(dtype=float)

    if not (pd.to_datetime(rows["train_end"]) <= pd.to_datetime(rows["origin_time"]) - pd.Timedelta(minutes=120)).all():
        raise RuntimeError(f"折 {fold.name} 未满足 120 分钟 purge")
    return rows


def _evaluate_named_fold(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    fold: TimeFold,
    versions: tuple[str, ...],
    config: ForecastConfig,
) -> tuple[str, pd.DataFrame]:
    return fold.name, _evaluate_fold(frame, features, deltas, fold, versions, config)


def build_legacy_oof(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    versions: Iterable[str] = SUPPORTED_LEGACY_MODELS,
    config: ForecastConfig | None = None,
    max_folds: int | None = None,
    n_jobs: int = 1,
    score_spec: ScoreSpec | None = None,
    checkpoint_dir: str | Path | None = None,
) -> OOFResult:
    """在完全相同的外层折上生成 Persistence 与 V1/V2/V2.5/V3 OOF。"""

    config = config or ForecastConfig()
    versions_tuple = tuple(dict.fromkeys(versions))
    invalid = sorted(set(versions_tuple).difference(SUPPORTED_OOF_MODELS))
    if invalid:
        raise ValueError(f"不支持的版本: {invalid}")
    deltas = build_delta_targets(frame, config.targets, config.feature.horizons)
    folds = make_outer_folds(frame.index, config)
    if max_folds is not None:
        folds = folds[-max_folds:]
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    required_columns = {f"{version}_pred" for version in versions_tuple}
    parts: list[pd.DataFrame] = []
    pending: list[TimeFold] = []
    for fold in folds:
        cached = checkpoint_path / f"fold_{fold.name}.csv" if checkpoint_path else None
        if cached is not None and cached.exists():
            cached_rows = pd.read_csv(cached, parse_dates=["origin_time", "train_end"])
            if required_columns.issubset(cached_rows.columns):
                parts.append(cached_rows)
                continue
        pending.append(fold)
    # 无序结果流会在任意折完成时立即落盘，无需等待同批最慢任务。
    if n_jobs == 1:
        completed_results = (
            _evaluate_named_fold(frame, features, deltas, fold, versions_tuple, config)
            for fold in pending
        )
    else:
        completed_results = Parallel(
            n_jobs=n_jobs,
            verbose=10,
            return_as="generator_unordered",
        )(
            delayed(_evaluate_named_fold)(
                frame, features, deltas, fold, versions_tuple, config
            )
            for fold in pending
        )
    for fold_name, completed_rows in completed_results:
        parts.append(completed_rows)
        if checkpoint_path is not None:
            completed_rows.to_csv(
                checkpoint_path / f"fold_{fold_name}.csv", index=False, encoding="utf-8"
            )
            manifest = {
                "completed_folds": sorted(
                    path.stem.removeprefix("fold_")
                    for path in checkpoint_path.glob("fold_*.csv")
                ),
                "requested_versions": list(versions_tuple),
                "fold_count": len(folds),
            }
            (checkpoint_path / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    rows = pd.concat(parts, ignore_index=True).sort_values(
        ["origin_time", "target", "horizon", "fold"], kind="stable"
    )
    rows["persistence_ape"] = absolute_percentage_error(
        rows["actual"], rows["persistence_pred"], epsilon=(score_spec or ScoreSpec()).epsilon
    )
    reports = {
        column.removesuffix("_pred"): score_oof_long(rows, column, spec=score_spec)
        for column in rows.columns
        if column.endswith("_pred")
    }
    return OOFResult(
        rows=rows.reset_index(drop=True),
        report={
            "folds": [fold.name for fold in folds],
            "models": reports,
            "shared_outer_folds": True,
            "purge_minutes": 120,
        },
    )


def write_oof(result: OOFResult, rows_path: str | Path, report_path: str | Path) -> None:
    """以 CSV 保存逐行预测，以 JSON 由调用脚本登记汇总。"""

    rows_path = Path(rows_path)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    result.rows.to_csv(rows_path, index=False, encoding="utf-8")
    # JSON 写入由 experiments.write_json 统一完成，避免这里引入多个登记口径。
