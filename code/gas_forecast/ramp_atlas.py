"""严格 OOF 的 Ramp Error Atlas。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


RAMP_BANDS: tuple[tuple[str, float, float], ...] = (
    ("stable", 0.0, 3.0),
    ("mild", 3.0, 7.0),
    ("medium", 7.0, 15.0),
    ("large", 15.0, float("inf")),
)


@dataclass(frozen=True)
class RampAtlasResult:
    """带单元标注、分组统计和严格范围声明的 ramp 诊断结果。"""

    cells: pd.DataFrame
    table: pd.DataFrame
    report: dict[str, object]


def assign_ramp_band(move: pd.Series) -> pd.Categorical:
    """按绝对真实增量映射到预注册的四档运行状态。"""

    labels = [name for name, _, _ in RAMP_BANDS]
    boundaries = [upper for _, _, upper in RAMP_BANDS[:-1]]
    return pd.cut(
        move.abs(),
        bins=[-np.inf, *boundaries, np.inf],
        labels=labels,
        right=False,
        ordered=True,
    )


def _validate_rows(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    candidate_column: str,
) -> pd.DataFrame:
    """验证 Atlas 使用的 OOF 键、标签与候选预测均完整且有限。"""

    required = {
        "fold",
        "origin_time",
        "target",
        "horizon",
        "actual",
        "current_value",
        baseline_column,
        candidate_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Ramp Atlas 输入缺少字段: {missing}")
    output = rows.copy()
    output["origin_time"] = pd.to_datetime(output["origin_time"], errors="coerce")
    if output["origin_time"].isna().any():
        raise ValueError("Ramp Atlas 输入含非法 origin_time")
    keys = ["fold", "origin_time", "target", "horizon"]
    if output.duplicated(keys).any():
        raise ValueError("Ramp Atlas 输入存在重复 fold×origin×target×horizon")
    numeric_columns = ["actual", "current_value", baseline_column, candidate_column]
    numeric = output.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Ramp Atlas 输入含缺失或非有限真实值/预测")
    output.loc[:, numeric.columns] = numeric
    output["horizon"] = pd.to_numeric(output["horizon"], errors="raise").astype(int)
    return output


def _direction_accuracy(actual_delta: pd.Series, predicted_delta: pd.Series) -> float:
    """统计预测增减方向与真实增减方向完全一致的比例。"""

    actual_direction = np.sign(actual_delta.to_numpy(dtype=float))
    predicted_direction = np.sign(predicted_delta.to_numpy(dtype=float))
    return float(np.mean(actual_direction == predicted_direction))


def _summarize_group(
    group: pd.DataFrame,
    *,
    baseline_column: str,
    candidate_column: str,
    epsilon: float,
) -> dict[str, float | int]:
    """计算单个目标×步长×ramp 档的成对误差、偏差和方向指标。"""

    actual = group["actual"].to_numpy(dtype=float)
    denominator = np.maximum(np.abs(actual), epsilon)
    baseline_error = group[baseline_column].to_numpy(dtype=float) - actual
    candidate_error = group[candidate_column].to_numpy(dtype=float) - actual
    baseline_delta = group["baseline_delta"].astype(float)
    candidate_delta = group["candidate_delta"].astype(float)
    actual_delta = group["actual_delta"].astype(float)
    baseline_mape = float(np.mean(np.abs(baseline_error) / denominator))
    candidate_mape = float(np.mean(np.abs(candidate_error) / denominator))
    return {
        "rows": int(len(group)),
        "baseline_mape": baseline_mape,
        "candidate_mape": candidate_mape,
        "mape_difference": candidate_mape - baseline_mape,
        "baseline_residual_bias": float(np.mean(baseline_error)),
        "candidate_residual_bias": float(np.mean(candidate_error)),
        "baseline_direction_accuracy": _direction_accuracy(actual_delta, baseline_delta),
        "candidate_direction_accuracy": _direction_accuracy(actual_delta, candidate_delta),
        "baseline_delta_mae": float(np.mean(np.abs(baseline_delta - actual_delta))),
        "candidate_delta_mae": float(np.mean(np.abs(candidate_delta - actual_delta))),
        "actual_delta_mae": float(np.mean(np.abs(actual_delta))),
        "candidate_better_fraction": float(
            np.mean(np.abs(candidate_error) < np.abs(baseline_error))
        ),
    }


def build_ramp_error_atlas(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    candidate_column: str,
    target: str = "generator_1",
    scope: str = "development",
    epsilon: float = 1e-6,
) -> RampAtlasResult:
    """按真实 ramp 档汇总 Champion 与候选的严格 OOF 表现。

    该函数不训练、不选参，也不读取测试答案；``development`` 范围会剔除 blind
    折，供后续候选是否值得继续研究的错误结构诊断使用。
    """

    if scope not in {"development", "final"}:
        raise ValueError("Ramp Atlas scope 只能是 development 或 final")
    if epsilon <= 0.0:
        raise ValueError("Ramp Atlas epsilon 必须为正数")
    work = _validate_rows(
        rows,
        baseline_column=baseline_column,
        candidate_column=candidate_column,
    )
    if scope == "development":
        work = work.loc[work["fold"].ne("blind")].copy()
    work = work.loc[work["target"].eq(target)].copy()
    if work.empty:
        raise ValueError(f"Ramp Atlas 没有目标 {target} 的可评分 OOF 行")

    work["actual_delta"] = work["actual"] - work["current_value"]
    work["baseline_delta"] = work[baseline_column] - work["current_value"]
    work["candidate_delta"] = work[candidate_column] - work["current_value"]
    work["ramp_band"] = assign_ramp_band(work["actual_delta"])
    work["baseline_absolute_error"] = (work[baseline_column] - work["actual"]).abs()
    work["candidate_absolute_error"] = (work[candidate_column] - work["actual"]).abs()
    work["baseline_ape"] = work["baseline_absolute_error"] / np.maximum(
        work["actual"].abs(), epsilon
    )
    work["candidate_ape"] = work["candidate_absolute_error"] / np.maximum(
        work["actual"].abs(), epsilon
    )

    summaries: list[dict[str, object]] = []
    group_columns = ["target", "horizon", "ramp_band"]
    for keys, group in work.groupby(group_columns, observed=False, sort=True):
        if group.empty:
            continue
        target_name, horizon, band = keys
        summaries.append(
            {
                "target": str(target_name),
                "horizon": int(horizon),
                "ramp_band": str(band),
                **_summarize_group(
                    group,
                    baseline_column=baseline_column,
                    candidate_column=candidate_column,
                    epsilon=epsilon,
                ),
            }
        )
    table = pd.DataFrame(summaries)
    if table.empty:
        raise ValueError("Ramp Atlas 没有可汇总的分组")
    table["ramp_band"] = pd.Categorical(
        table["ramp_band"],
        categories=[name for name, _, _ in RAMP_BANDS],
        ordered=True,
    )
    table = table.sort_values(["target", "horizon", "ramp_band"]).reset_index(drop=True)
    overall = _summarize_group(
        work,
        baseline_column=baseline_column,
        candidate_column=candidate_column,
        epsilon=epsilon,
    )
    band_summary = (
        table.groupby("ramp_band", observed=False, sort=False)["rows"]
        .sum()
        .astype(int)
        .to_dict()
    )
    return RampAtlasResult(
        cells=work.reset_index(drop=True),
        table=table,
        report={
            "scope": scope,
            "target": target,
            "baseline_column": baseline_column,
            "candidate_column": candidate_column,
            "epsilon": float(epsilon),
            "ramp_bands": [
                {"name": name, "lower_inclusive": lower, "upper_exclusive": upper}
                for name, lower, upper in RAMP_BANDS
            ],
            "rows_by_ramp_band": {str(key): int(value) for key, value in band_summary.items()},
            "overall": overall,
            "strict_oof_contract": (
                "仅汇总现有严格 OOF；development 范围不包含 blind，也不使用任何测试答案。"
            ),
        },
    )
