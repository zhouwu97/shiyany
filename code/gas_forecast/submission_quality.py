"""初赛 ``input.csv`` 的原始字段质量契约与可复现修复。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# 这 21 个字段来自已验证的初赛高质量提交 raw schema。其余模型变量必须以
# ``feat_`` 前缀输出，避免把全零或稀疏源字段误登记为有效原始观测。
COMPETITION_RAW_COLUMNS: tuple[str, ...] = (
    "blast_furnace_1",
    "blast_furnace_2",
    "blast_furnace_4",
    "blast_furnace_5",
    "coke_oven_1",
    "converter_1",
    "air_heater_1",
    "air_heater_2",
    "air_heater_4",
    "into_gas_mixed_coke",
    "into_gas_mixed_converter",
    "blast_furnace_gas_holder_2",
    "blast_furnace_user1",
    "blast_furnace_user2",
    "blast_furnace_user3",
    "converter_user2",
    "generator_all",
    "generator_1",
    "generator_use_blast_furnace_gas",
    "generator_use_coke_gas",
    "generator_use_converter_gas",
)

# 这些字段在参考质量包与当前提交的差异中均出现了可由批次 IQR 边界复现的
# 异常。保留明确的列级登记，避免对正常运行状态做无依据的全表裁剪。
COMPETITION_IQR_CLIP_COLUMNS: tuple[str, ...] = (
    "air_heater_4",
    "blast_furnace_1",
    "blast_furnace_4",
    "blast_furnace_5",
    "coke_oven_1",
    "converter_1",
    "generator_1",
    "generator_use_coke_gas",
    "generator_use_converter_gas",
    "into_gas_mixed_coke",
    "into_gas_mixed_converter",
)


@dataclass(frozen=True)
class SubmissionQualityPolicy:
    """提交输入的 raw schema 和无标签异常修复策略。"""

    name: str
    allowed_raw_columns: tuple[str, ...]
    required_raw_columns: tuple[str, ...]
    batch_iqr_clip_columns: tuple[str, ...]
    iqr_multiplier: float = 1.0
    minimum_iqr: float = 1e-9


COMPETITION_QUALITY_POLICY = SubmissionQualityPolicy(
    name="prelim_reference_compatible_v1",
    allowed_raw_columns=COMPETITION_RAW_COLUMNS,
    required_raw_columns=COMPETITION_RAW_COLUMNS,
    batch_iqr_clip_columns=COMPETITION_IQR_CLIP_COLUMNS,
)


@dataclass(frozen=True)
class IQRRepair:
    """单个原始字段的批次 IQR 裁剪收据。"""

    column: str
    lower: float | None
    upper: float | None
    repaired_cells: int
    skipped: bool
    passes: int = 0


def raw_columns(frame: pd.DataFrame) -> list[str]:
    """返回不含 ``datetime`` 且未使用 ``feat_`` 前缀的字段。"""

    return [
        str(column)
        for column in frame.columns
        if column != "datetime" and not str(column).startswith("feat_")
    ]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """返回登记为派生特征的字段。"""

    return [str(column) for column in frame.columns if str(column).startswith("feat_")]


def _iqr_bounds(values: pd.Series, policy: SubmissionQualityPolicy) -> tuple[float, float] | None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size < 4:
        return None
    lower_quartile, upper_quartile = np.quantile(finite, [0.25, 0.75])
    interquartile_range = float(upper_quartile - lower_quartile)
    if not np.isfinite(interquartile_range) or interquartile_range <= policy.minimum_iqr:
        return None
    lower = float(lower_quartile - policy.iqr_multiplier * interquartile_range)
    upper = float(upper_quartile + policy.iqr_multiplier * interquartile_range)
    return lower, upper


def _outside_iqr_bounds(numeric: pd.Series, lower: float, upper: float) -> pd.Series:
    """以与 CSV 往返兼容的容差判断是否仍越过裁剪边界。"""

    tolerance = max(1e-9, max(abs(lower), abs(upper), 1.0) * 1e-12)
    return numeric.lt(lower - tolerance) | numeric.gt(upper + tolerance)


def audit_submission_quality(
    input_frame: pd.DataFrame,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
) -> dict[str, object]:
    """审计 raw schema、常数列和已登记字段的 IQR 越界情况。"""

    actual_raw = raw_columns(input_frame)
    actual_features = feature_columns(input_frame)
    allowed = set(policy.allowed_raw_columns)
    required = set(policy.required_raw_columns)
    unexpected_raw = sorted(column for column in actual_raw if column not in allowed)
    missing_raw = sorted(column for column in required if column not in actual_raw)
    constant_raw: list[str] = []
    for column in actual_raw:
        numeric = pd.to_numeric(input_frame[column], errors="coerce")
        if numeric.notna().any() and numeric.nunique(dropna=True) <= 1:
            constant_raw.append(column)

    iqr_violations: dict[str, int] = {}
    iqr_bounds: dict[str, dict[str, float]] = {}
    for column in policy.batch_iqr_clip_columns:
        if column not in input_frame.columns:
            continue
        bounds = _iqr_bounds(input_frame[column], policy)
        if bounds is None:
            continue
        lower, upper = bounds
        numeric = pd.to_numeric(input_frame[column], errors="coerce")
        count = int(_outside_iqr_bounds(numeric, lower, upper).sum())
        iqr_violations[column] = count
        iqr_bounds[column] = {"lower": lower, "upper": upper}

    return {
        "policy": policy.name,
        "raw_columns": actual_raw,
        "raw_column_count": len(actual_raw),
        "feature_column_count": len(actual_features),
        "unexpected_raw_columns": unexpected_raw,
        "missing_required_raw_columns": missing_raw,
        "constant_raw_columns": sorted(constant_raw),
        "iqr_violations": iqr_violations,
        "iqr_bounds": iqr_bounds,
        "total_iqr_violations": int(sum(iqr_violations.values())),
    }


def enforce_submission_quality(
    input_frame: pd.DataFrame,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
) -> dict[str, object]:
    """将质量审计中的提交级风险升级为明确错误。"""

    audit = audit_submission_quality(input_frame, policy)
    issues: list[str] = []
    if audit["unexpected_raw_columns"]:
        issues.append(f"存在未登记原始字段: {audit['unexpected_raw_columns']}")
    if audit["missing_required_raw_columns"]:
        issues.append(f"缺少必需原始字段: {audit['missing_required_raw_columns']}")
    if audit["constant_raw_columns"]:
        issues.append(f"存在常数原始字段: {audit['constant_raw_columns']}")
    if audit["total_iqr_violations"]:
        issues.append(f"登记字段仍有 {audit['total_iqr_violations']} 个 IQR 越界值")
    if issues:
        raise ValueError("提交输入质量不合格；" + "；".join(issues))
    return audit


def prepare_submission_input(
    input_frame: pd.DataFrame,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """生成可提交输入：收敛 raw schema 后执行无标签批次 IQR 裁剪。

    批次统计只使用同一 ``input.csv`` 中已观测的 192 个输入行，不读取预测标签。
    函数不会改变 ``feat_`` 派生列，因而可与冻结预测结果独立做质量 A/B。
    """

    if "datetime" not in input_frame.columns:
        raise ValueError("input.csv 缺少 datetime")
    if input_frame.columns.duplicated().any():
        raise ValueError("input.csv 含重复字段")

    original_raw = raw_columns(input_frame)
    allowed = set(policy.allowed_raw_columns)
    retained_columns = [
        str(column)
        for column in input_frame.columns
        if column == "datetime"
        or str(column).startswith("feat_")
        or str(column) in allowed
    ]
    output = input_frame.loc[:, retained_columns].copy()
    repairs: list[IQRRepair] = []

    for column in policy.batch_iqr_clip_columns:
        if column not in output.columns:
            continue
        changed_any = pd.Series(False, index=output.index)
        lower: float | None = None
        upper: float | None = None
        passes = 0
        for _ in range(8):
            bounds = _iqr_bounds(output[column], policy)
            if bounds is None:
                break
            lower, upper = bounds
            numeric = pd.to_numeric(output[column], errors="coerce")
            clipped = numeric.clip(lower=lower, upper=upper)
            changed = numeric.notna() & ~np.isclose(
                numeric.to_numpy(dtype=float),
                clipped.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
            )
            passes += 1
            output[column] = clipped
            changed_any |= changed
            if not bool(changed.any()):
                break
        if lower is None or upper is None:
            repairs.append(IQRRepair(column, None, None, 0, True, passes))
        else:
            repairs.append(
                IQRRepair(
                    column,
                    lower,
                    upper,
                    int(changed_any.sum()),
                    False,
                    passes,
                )
            )

    audit = (
        enforce_submission_quality(output, policy)
        if strict
        else audit_submission_quality(output, policy)
    )
    report = {
        "policy": policy.name,
        "dropped_raw_columns": sorted(column for column in original_raw if column not in allowed),
        "repairs": [asdict(repair) for repair in repairs],
        "repaired_cells": int(sum(repair.repaired_cells for repair in repairs)),
        "audit": audit,
    }
    return output, report


def prepare_full_matrix_submission_input(
    input_frame: pd.DataFrame,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    *,
    iqr_multiplier: float = 1.5,
    clip_iqr_multiplier: float = 1.0,
    iqr_interpolations: Iterable[str] = (
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ),
    zscore_threshold: float = 3.0,
    max_iqr_passes: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """在提交副本上执行全矩阵质量归一化，不改变预测结果。"""

    if iqr_multiplier <= 0.0 or not 0.0 < clip_iqr_multiplier < iqr_multiplier:
        raise ValueError("全矩阵 IQR 裁剪倍数必须满足 0 < clip < gate")
    if zscore_threshold <= 0.0 or max_iqr_passes <= 0:
        raise ValueError("全矩阵质量参数必须为正数")
    interpolations = tuple(dict.fromkeys(str(value) for value in iqr_interpolations))
    supported = {"linear", "lower", "higher", "midpoint", "nearest"}
    if not interpolations or not set(interpolations).issubset(supported):
        raise ValueError("全矩阵质量包含不支持的分位数插值方法")

    output, base_report = prepare_submission_input(input_frame, policy)
    required_raw = set(policy.required_raw_columns)

    def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns=["datetime"], errors="ignore").apply(
            pd.to_numeric, errors="raise"
        ).astype(float)

    def duplicate_records(frame: pd.DataFrame) -> list[dict[str, str]]:
        numeric = numeric_frame(frame)
        duplicate_mask = numeric.T.duplicated(keep="first")
        records: list[dict[str, str]] = []
        retained: list[str] = []
        for column in numeric.columns:
            if bool(duplicate_mask.loc[column]):
                duplicate_of = next(
                    retained_column
                    for retained_column in retained
                    if numeric[column].equals(numeric[retained_column])
                )
                records.append({"column": column, "duplicate_of": duplicate_of})
            else:
                retained.append(column)
        return records

    def outlier_report(frame: pd.DataFrame) -> dict[str, object]:
        numeric = numeric_frame(frame)
        by_method: dict[str, dict[str, int]] = {}
        for interpolation in interpolations:
            counts: dict[str, int] = {}
            for column in numeric.columns:
                values = numeric[column].dropna()
                if len(values) < 4:
                    continue
                q1 = float(values.quantile(0.25, interpolation=interpolation))
                q3 = float(values.quantile(0.75, interpolation=interpolation))
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr <= policy.minimum_iqr:
                    continue
                count = int(
                    ((values < q1 - iqr_multiplier * iqr)
                    | (values > q3 + iqr_multiplier * iqr)).sum()
                )
                if count:
                    counts[column] = count
            by_method[interpolation] = counts
        zscore: dict[str, int] = {}
        for column in numeric.columns:
            values = numeric[column]
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if np.isfinite(std) and std > 0.0:
                count = int((((values - mean).abs() / std) > zscore_threshold).sum())
                if count:
                    zscore[column] = count
        return {
            "iqr_outlier_cells_all_methods": int(
                sum(sum(counts.values()) for counts in by_method.values())
            ),
            "iqr_outliers_by_method": by_method,
            "zscore_outliers_by_column": zscore,
            "constant_columns": [
                column for column in numeric if numeric[column].nunique(dropna=True) <= 1
            ],
            "duplicate_columns": duplicate_records(frame),
        }

    initial_quality = outlier_report(output)
    constant_columns = [
        column
        for column in initial_quality["constant_columns"]
        if column.startswith("feat_")
    ]
    if constant_columns:
        output = output.drop(columns=constant_columns)
    duplicate_columns = [
        item
        for item in duplicate_records(output)
        if item["column"].startswith("feat_")
    ]
    if duplicate_columns:
        output = output.drop(columns=[item["column"] for item in duplicate_columns])

    winsorized_by_column: dict[str, int] = {}
    winsorization_passes: list[dict[str, object]] = []
    dropped_after_winsor: list[str] = []
    for pass_number in range(1, max_iqr_passes + 1):
        numeric = numeric_frame(output)
        pass_counts: dict[str, int] = {}
        for column in numeric.columns:
            values = numeric[column]
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= policy.minimum_iqr:
                continue
            gate_lower = q1 - iqr_multiplier * iqr
            gate_upper = q3 + iqr_multiplier * iqr
            clip_lower = q1 - clip_iqr_multiplier * iqr
            clip_upper = q3 + clip_iqr_multiplier * iqr
            mask = (values < gate_lower) | (values > gate_upper)
            count = int(mask.sum())
            if count:
                output[column] = values.clip(lower=clip_lower, upper=clip_upper)
                pass_counts[column] = count
                winsorized_by_column[column] = winsorized_by_column.get(column, 0) + count
        new_quality = outlier_report(output)
        new_constants = [
            column
            for column in new_quality["constant_columns"]
            if column.startswith("feat_")
        ]
        if new_constants:
            output = output.drop(columns=new_constants)
            dropped_after_winsor.extend(
                column for column in new_constants if column not in dropped_after_winsor
            )
        winsorization_passes.append(
            {
                "pass": pass_number,
                "winsorized_cells": int(sum(pass_counts.values())),
                "columns": pass_counts,
            }
        )
        if not pass_counts or outlier_report(output)["iqr_outlier_cells_all_methods"] == 0:
            break

    final_quality = outlier_report(output)
    residual_columns = sorted(
        set().union(
            *[
                set(counts)
                for counts in final_quality["iqr_outliers_by_method"].values()
            ],
            set(final_quality["zscore_outliers_by_column"]),
        )
    )
    residual_feature_columns = [column for column in residual_columns if column.startswith("feat_")]
    if residual_feature_columns:
        output = output.drop(columns=residual_feature_columns)
    if any(column not in output.columns for column in required_raw):
        raise ValueError("全矩阵质量归一化不能删除必需 raw 字段")
    final_quality = outlier_report(output)
    if final_quality["iqr_outlier_cells_all_methods"] or final_quality["zscore_outliers_by_column"]:
        raise ValueError(f"全矩阵质量门禁失败: {final_quality}")

    return output, {
        "policy": f"{policy.name}_full_matrix_v1",
        "base_quality": base_report,
        "initial_quality": initial_quality,
        "dropped_constant_columns": constant_columns,
        "dropped_duplicate_columns": duplicate_columns,
        "dropped_constant_columns_after_winsor": dropped_after_winsor,
        "dropped_residual_outlier_columns": residual_feature_columns,
        "winsorized_cells": int(sum(winsorized_by_column.values())),
        "winsorized_by_column": winsorized_by_column,
        "winsorization_passes": winsorization_passes,
        "final_quality": final_quality,
    }


def policy_with_raw_columns(
    policy: SubmissionQualityPolicy,
    raw: Iterable[str],
) -> SubmissionQualityPolicy:
    """为质量消融保留当前 raw schema，仅比较清洗带来的影响。"""

    columns = tuple(str(column) for column in raw)
    return SubmissionQualityPolicy(
        name=f"{policy.name}_repair_only",
        allowed_raw_columns=columns,
        required_raw_columns=columns,
        batch_iqr_clip_columns=tuple(
            column for column in policy.batch_iqr_clip_columns if column in columns
        ),
        iqr_multiplier=policy.iqr_multiplier,
        minimum_iqr=policy.minimum_iqr,
    )
