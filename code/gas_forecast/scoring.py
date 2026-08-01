"""统一的竞赛评分与 OOF 诊断口径。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreSpec:
    """显式记录尚待官方细则确认的评分假设。"""

    epsilon: float = 1e-6
    target_aggregation: str = "pooled_cells"
    horizon_aggregation: str = "pooled_cells"
    missing_policy: str = "drop_pairwise"


def competition_mape(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    *,
    epsilon: float = 1e-6,
) -> float:
    """按显式 epsilon 计算逐单元格 MAPE，不冒充未公开的官方零值规则。"""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("真实值与预测值形状不一致")
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if not valid.any():
        return float("nan")
    denominator = np.maximum(np.abs(actual_array[valid]), epsilon)
    return float(np.mean(np.abs(actual_array[valid] - predicted_array[valid]) / denominator))


def absolute_percentage_error(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    epsilon: float = 1e-6,
) -> pd.Series:
    """返回与输入索引对齐的逐行绝对百分比误差。"""

    denominator = actual.abs().clip(lower=epsilon)
    result = (actual - predicted).abs() / denominator
    return result.where(actual.notna() & predicted.notna())


def score_oof_long(
    rows: pd.DataFrame,
    prediction_column: str,
    *,
    spec: ScoreSpec | None = None,
) -> dict[str, object]:
    """汇总统一 OOF 长表，兼容 pooled 与旧折均值等多种诊断口径。"""

    spec = spec or ScoreSpec()
    required = {"fold", "origin_time", "target", "horizon", "actual", prediction_column}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"OOF 长表缺少字段: {missing}")

    scored = rows.loc[:, list(required)].copy()
    scored["origin_time"] = pd.to_datetime(scored["origin_time"])
    scored["ape"] = absolute_percentage_error(
        scored["actual"], scored[prediction_column], epsilon=spec.epsilon
    )
    scored = scored.dropna(subset=["ape"])
    if scored.empty:
        raise ValueError("OOF 长表没有可评分预测")

    by_target = scored.groupby("target", sort=True)["ape"].mean()
    by_horizon = scored.groupby("horizon", sort=True)["ape"].mean()
    by_cell = scored.groupby(["target", "horizon"], sort=True)["ape"].mean()
    by_fold = scored.groupby("fold", sort=True)["ape"].mean()
    by_day = scored.assign(day=scored["origin_time"].dt.date).groupby("day")["ape"].mean()
    by_origin = scored.groupby(["fold", "origin_time"], sort=True)["ape"].mean()

    report: dict[str, object] = {
        "score_spec": asdict(spec),
        "rows": int(len(scored)),
        "pooled_mape": float(scored["ape"].mean()),
        "equal_target_mape": float(by_target.mean()),
        "equal_target_horizon_mape": float(by_cell.mean()),
        "legacy_fold_mean_mape": float(by_fold.mean()),
        "by_target": {str(key): float(value) for key, value in by_target.items()},
        "by_horizon": {f"t+{int(key)}": float(value) for key, value in by_horizon.items()},
        "by_fold": {str(key): float(value) for key, value in by_fold.items()},
        "by_day": {str(key): float(value) for key, value in by_day.items()},
        "origin_error_p90": float(by_origin.quantile(0.90)),
        "origin_error_p95": float(by_origin.quantile(0.95)),
    }
    if "regime" in rows.columns:
        regime = rows.loc[scored.index, "regime"].fillna("unknown")
        report["by_regime"] = {
            str(key): float(value)
            for key, value in scored.assign(regime=regime).groupby("regime")["ape"].mean().items()
        }
    return report


def block_bootstrap_improvement_probability(
    rows: pd.DataFrame,
    candidate_column: str,
    baseline_column: str,
    *,
    spec: ScoreSpec | None = None,
    block: str = "day",
    samples: int = 2000,
    random_state: int = 20250731,
) -> dict[str, float | int | str]:
    """按日期或连续折块重采样，估计候选优于基线的概率。"""

    spec = spec or ScoreSpec()
    required = {"origin_time", "actual", candidate_column, baseline_column}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"bootstrap 输入缺少字段: {missing}")
    work = rows.copy()
    work["origin_time"] = pd.to_datetime(work["origin_time"])
    if block == "day":
        work["_block"] = work["origin_time"].dt.floor("D")
    elif block == "fold":
        if "fold" not in work:
            raise ValueError("按折 bootstrap 需要 fold 字段")
        work["_block"] = work["fold"].astype(str)
    else:
        raise ValueError("block 仅支持 day 或 fold")

    grouped = []
    for _, part in work.groupby("_block", sort=True):
        candidate = competition_mape(part["actual"], part[candidate_column], epsilon=spec.epsilon)
        baseline = competition_mape(part["actual"], part[baseline_column], epsilon=spec.epsilon)
        grouped.append(candidate - baseline)
    differences = np.asarray(grouped, dtype=float)
    rng = np.random.default_rng(random_state)
    sampled = rng.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
    return {
        "block": block,
        "blocks": int(len(differences)),
        "samples": int(samples),
        "mean_difference": float(differences.mean()),
        "probability_candidate_better": float(
            np.mean(sampled < 0.0) + 0.5 * np.mean(np.isclose(sampled, 0.0))
        ),
        "difference_ci_low": float(np.quantile(sampled, 0.025)),
        "difference_ci_high": float(np.quantile(sampled, 0.975)),
    }
