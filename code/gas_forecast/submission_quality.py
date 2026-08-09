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


@dataclass(frozen=True)
class FrozenColumnQuality:
    """训练期冻结的单列质量统计，不在评分期重新估计。"""

    column: str
    kind: str
    retained: bool
    valid: bool
    constant: bool
    duplicate_of: str | None
    median: float | None
    lower: float | None
    upper: float | None
    apply_iqr_clip: bool


@dataclass(frozen=True)
class FittedSubmissionQualityPolicy:
    """只由 ``train_end`` 之前生产观测拟合的提交质量策略。"""

    policy: SubmissionQualityPolicy
    training_rows: int
    training_start: str
    training_end: str
    retained_columns: tuple[str, ...]
    missing_required_raw_columns: tuple[str, ...]
    dropped_raw_columns: tuple[str, ...]
    dropped_invalid_columns: tuple[str, ...]
    dropped_constant_columns: tuple[str, ...]
    dropped_duplicate_columns: tuple[tuple[str, str], ...]
    columns: tuple[FrozenColumnQuality, ...]

    def column_map(self) -> dict[str, FrozenColumnQuality]:
        """返回冻结列统计的副本，调用方不能修改策略本身。"""

        return {item.column: item for item in self.columns}

    def to_dict(self) -> dict[str, object]:
        """导出可审计收据，避免把 pandas 对象写进运行结果。"""

        return {
            "policy": asdict(self.policy),
            "training_rows": self.training_rows,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "retained_columns": list(self.retained_columns),
            "missing_required_raw_columns": list(self.missing_required_raw_columns),
            "dropped_raw_columns": list(self.dropped_raw_columns),
            "dropped_invalid_columns": list(self.dropped_invalid_columns),
            "dropped_constant_columns": list(self.dropped_constant_columns),
            "dropped_duplicate_columns": [list(item) for item in self.dropped_duplicate_columns],
            "columns": [asdict(item) for item in self.columns],
        }


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


def _training_slice(
    training_frame: pd.DataFrame,
    train_end: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """只截取 ``train_end`` 之前的生产观测，并返回解析后的时间列。"""

    if "datetime" not in training_frame.columns:
        raise ValueError("训练质量帧缺少 datetime")
    if training_frame.columns.duplicated().any():
        raise ValueError("训练质量帧含重复字段")
    timestamps = pd.to_datetime(training_frame["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("训练质量帧 datetime 含非法时间戳")
    if train_end is not None:
        cutoff = pd.Timestamp(train_end)
        if pd.isna(cutoff):
            raise ValueError("train_end 不是合法时间戳")
        mask = timestamps.le(cutoff)
        if not bool(mask.any()):
            raise ValueError("train_end 之前没有可用于拟合质量策略的生产观测")
        selected = training_frame.loc[mask].copy()
        selected_times = timestamps.loc[mask]
    else:
        selected = training_frame.copy()
        selected_times = timestamps
    if selected.empty:
        raise ValueError("训练质量帧为空")
    return selected, selected_times


def _finite_numeric(values: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """把一列转成数值，并显式区分有限值与无效值。"""

    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(array)
    return numeric, finite


def fit_quality_policy(
    training_frame: pd.DataFrame,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    *,
    train_end: str | pd.Timestamp | None = None,
) -> FittedSubmissionQualityPolicy:
    """在生产训练期冻结质量规则。

    统计量、有效性、常数列和重复列均只从 ``training_frame`` 中
    ``datetime <= train_end`` 的行计算。评分期不得调用此函数重新拟合，
    而应把返回对象传给 :func:`transform_submission_input`。
    """

    train, timestamps = _training_slice(training_frame, train_end)
    allowed = set(policy.allowed_raw_columns)
    required = set(policy.required_raw_columns)
    actual_raw = raw_columns(train)
    dropped_raw = tuple(sorted(column for column in actual_raw if column not in allowed))
    missing_required = tuple(sorted(column for column in required if column not in actual_raw))

    # 训练期保留顺序，确保生成的 input.csv 与模型/平台契约稳定一致。
    candidates = [
        str(column)
        for column in train.columns
        if column == "datetime"
        or str(column) in allowed
        or str(column).startswith("feat_")
    ]
    feature_stats: list[FrozenColumnQuality] = []
    dropped_invalid: list[str] = []
    dropped_constant: list[str] = []
    dropped_duplicate: list[tuple[str, str]] = []
    retained_numeric: dict[str, pd.Series] = {}
    retained_columns: list[str] = []

    for column in candidates:
        if column == "datetime":
            continue
        numeric, finite = _finite_numeric(train[column])
        finite_values = numeric.iloc[np.flatnonzero(finite)].astype(float)
        is_feature = column.startswith("feat_")
        is_required_raw = column in required
        has_valid = bool(finite.any())
        constant = has_valid and int(finite_values.nunique(dropna=True)) <= 1
        duplicate_of: str | None = None
        if has_valid:
            for previous, previous_values in retained_numeric.items():
                if numeric.equals(previous_values):
                    duplicate_of = previous
                    break

        if not has_valid:
            dropped_invalid.append(column)
        if constant and is_feature:
            dropped_constant.append(column)
        if duplicate_of is not None and is_feature:
            dropped_duplicate.append((column, duplicate_of))

        # 必需 raw 列必须保留；派生/可选 raw 的坏列在训练期冻结时移除。
        retained = has_valid and not (
            (constant or duplicate_of is not None) and not is_required_raw
        )
        if not has_valid and is_required_raw:
            retained = True
        if retained:
            retained_columns.append(column)
            retained_numeric[column] = numeric

        median: float | None = None
        lower: float | None = None
        upper: float | None = None
        if has_valid:
            finite_array = finite_values.to_numpy(dtype=float)
            median = float(np.median(finite_array))
            bounds = _iqr_bounds(train[column], policy)
            if bounds is not None and (
                column in policy.batch_iqr_clip_columns or is_feature
            ):
                lower, upper = bounds
        kind = "feature" if is_feature else "raw"
        if not has_valid:
            kind = "invalid"
        elif constant:
            kind = "constant"
        elif duplicate_of is not None:
            kind = "duplicate"
        feature_stats.append(
            FrozenColumnQuality(
                column=column,
                kind=kind,
                retained=retained,
                valid=has_valid,
                constant=constant,
                duplicate_of=duplicate_of,
                median=median,
                lower=lower,
                upper=upper,
                apply_iqr_clip=lower is not None and upper is not None,
            )
        )

    return FittedSubmissionQualityPolicy(
        policy=policy,
        training_rows=len(train),
        training_start=str(timestamps.iloc[0]),
        training_end=str(timestamps.iloc[-1]),
        retained_columns=tuple(retained_columns),
        missing_required_raw_columns=missing_required,
        dropped_raw_columns=dropped_raw,
        dropped_invalid_columns=tuple(sorted(dropped_invalid)),
        dropped_constant_columns=tuple(sorted(dropped_constant)),
        dropped_duplicate_columns=tuple(dropped_duplicate),
        columns=tuple(feature_stats),
    )


def _audit_fitted_quality(
    input_frame: pd.DataFrame,
    fitted_policy: FittedSubmissionQualityPolicy,
) -> dict[str, object]:
    """用冻结边界审计评分输入；此函数不从评分行估计任何统计量。"""

    policy = fitted_policy.policy
    actual_raw = raw_columns(input_frame)
    actual_features = feature_columns(input_frame)
    allowed = set(policy.allowed_raw_columns)
    required = set(policy.required_raw_columns)
    unexpected_raw = sorted(column for column in actual_raw if column not in allowed)
    missing_raw = sorted(column for column in required if column not in actual_raw)
    stats = fitted_policy.column_map()
    invalid_cells: dict[str, int] = {}
    iqr_violations: dict[str, int] = {}
    for column in fitted_policy.retained_columns:
        if column not in input_frame.columns:
            continue
        numeric, finite = _finite_numeric(input_frame[column])
        invalid_cells[column] = int((~finite).sum())
        item = stats[column]
        if item.apply_iqr_clip and item.lower is not None and item.upper is not None:
            iqr_violations[column] = int(
                _outside_iqr_bounds(numeric, item.lower, item.upper).fillna(False).sum()
            )
    missing_columns = sorted(
        column for column in fitted_policy.retained_columns if column not in input_frame.columns
    )
    return {
        "policy": policy.name,
        "frozen": True,
        "training_end": fitted_policy.training_end,
        "raw_columns": actual_raw,
        "raw_column_count": len(actual_raw),
        "feature_column_count": len(actual_features),
        "unexpected_raw_columns": unexpected_raw,
        "missing_required_raw_columns": missing_raw,
        "missing_frozen_columns": missing_columns,
        "constant_raw_columns": [],
        "invalid_cells_by_column": invalid_cells,
        "total_invalid_cells": int(sum(invalid_cells.values())),
        "iqr_violations": iqr_violations,
        "iqr_bounds": {
            column: {"lower": stats[column].lower, "upper": stats[column].upper}
            for column in iqr_violations
            if stats[column].lower is not None and stats[column].upper is not None
        },
        "total_iqr_violations": int(sum(iqr_violations.values())),
    }


def transform_submission_input(
    scoring_origin_rows: pd.DataFrame,
    fitted_policy: FittedSubmissionQualityPolicy,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """将评分起点按训练期冻结策略逐元素变换。

    任何缺失/裁剪决策都只读取冻结的中位数与 IQR 边界，
    不读取评分未来分布或标签。
    """

    if not isinstance(scoring_origin_rows, pd.DataFrame) or not isinstance(
        fitted_policy, FittedSubmissionQualityPolicy
    ):
        raise TypeError("需要 scoring_origin_rows 和 fitted_policy")
    if "datetime" not in scoring_origin_rows.columns:
        raise ValueError("评分质量帧缺少 datetime")
    if scoring_origin_rows.columns.duplicated().any():
        raise ValueError("评分质量帧含重复字段")

    policy = fitted_policy.policy
    stats = fitted_policy.column_map()
    output = pd.DataFrame({"datetime": scoring_origin_rows["datetime"].copy()})
    invalid_cells: dict[str, int] = {}
    clipped_cells: dict[str, int] = {}
    added_columns: list[str] = []

    for column in fitted_policy.retained_columns:
        item = stats[column]
        if column in scoring_origin_rows.columns:
            values = scoring_origin_rows[column]
            numeric, finite = _finite_numeric(values)
        else:
            values = pd.Series(np.nan, index=scoring_origin_rows.index)
            numeric, finite = _finite_numeric(values)
            added_columns.append(column)
        if item.median is None:
            if not finite.all():
                if column in policy.required_raw_columns and strict:
                    raise ValueError(f"必需 raw 字段 {column} 没有可用训练期中位数")
                replacement = 0.0
            else:
                replacement = None
        else:
            replacement = item.median
        transformed = numeric.astype(float).copy()
        invalid_mask = ~finite
        invalid_cells[column] = int(invalid_mask.sum())
        if bool(invalid_mask.any()):
            if replacement is None:
                if strict:
                    raise ValueError(f"字段 {column} 含训练期无法修复的无效值")
            else:
                transformed.loc[invalid_mask] = replacement
        clipped = pd.Series(False, index=transformed.index)
        if item.apply_iqr_clip and item.lower is not None and item.upper is not None:
            clipped = transformed.lt(item.lower) | transformed.gt(item.upper)
            transformed = transformed.clip(lower=item.lower, upper=item.upper)
        clipped_cells[column] = int(clipped.fillna(False).sum())
        output[column] = transformed

    if strict:
        missing_required = [
            column for column in policy.required_raw_columns if column not in output.columns
        ]
        if missing_required:
            raise ValueError(f"评分输入缺少必需 raw 字段: {missing_required}")
        values = output.drop(columns=["datetime"]).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("质量变换后仍含无效输入值")

    audit = _audit_fitted_quality(output, fitted_policy)
    report = {
        "policy": policy.name,
        "frozen": True,
        "training_end": fitted_policy.training_end,
        "scoring_rows": len(scoring_origin_rows),
        "dropped_raw_columns": list(fitted_policy.dropped_raw_columns),
        "dropped_invalid_columns": list(fitted_policy.dropped_invalid_columns),
        "dropped_constant_columns": list(fitted_policy.dropped_constant_columns),
        "dropped_duplicate_columns": [list(item) for item in fitted_policy.dropped_duplicate_columns],
        "added_columns": added_columns,
        "invalid_cells_by_column": invalid_cells,
        "clipped_cells_by_column": clipped_cells,
        "repaired_cells": int(sum(invalid_cells.values()) + sum(clipped_cells.values())),
        "audit": audit,
    }
    return output, report


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

    # 兼容旧调用方：旧接口在传入的当前帧上拟合一次，然后立即做纯变换。
    # 生产代码若已有训练期，应显式调用 fit_quality_policy 并复用其结果。
    fitted = fit_quality_policy(input_frame, policy)
    output, report = transform_submission_input(input_frame, fitted, strict=strict)
    report["legacy_fit_on_input"] = True
    return output, report


def _prepare_submission_input_legacy_batch(
    input_frame: pd.DataFrame,
    policy: SubmissionQualityPolicy,
    *,
    strict: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """保留历史批次算法，仅供旧报告复现；正式入口不再使用。"""

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
    """历史质量消融：评分矩阵统计只用于非生产 Q3 对照。"""

    if iqr_multiplier <= 0.0 or not 0.0 < clip_iqr_multiplier < iqr_multiplier:
        raise ValueError("全矩阵 IQR 裁剪倍数必须满足 0 < clip < gate")
    if zscore_threshold <= 0.0 or max_iqr_passes <= 0:
        raise ValueError("全矩阵质量参数必须为正数")
    interpolations = tuple(dict.fromkeys(str(value) for value in iqr_interpolations))
    supported = {"linear", "lower", "higher", "midpoint", "nearest"}
    if not interpolations or not set(interpolations).issubset(supported):
        raise ValueError("全矩阵质量包含不支持的分位数插值方法")

    fitted = fit_quality_policy(input_frame, policy)
    output, base_report = transform_submission_input(input_frame, fitted)
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

    initial_quality = outlier_report(input_frame.loc[:, [
        column for column in input_frame.columns if column != "datetime"
    ]].assign(datetime=input_frame["datetime"]))
    constant_columns = list(fitted.dropped_constant_columns)
    if constant_columns:
        output = output.drop(columns=constant_columns, errors="ignore")
    duplicate_columns = [
        {"column": column, "duplicate_of": duplicate_of}
        for column, duplicate_of in fitted.dropped_duplicate_columns
    ]
    if duplicate_columns:
        output = output.drop(
            columns=[item["column"] for item in duplicate_columns],
            errors="ignore",
        )

    winsorized_by_column: dict[str, int] = {
        column: int(count)
        for column, count in base_report.get("clipped_cells_by_column", {}).items()
        if int(count) > 0
    }
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
        "production_eligible": False,
        "reason": "全矩阵统计依赖评分期分布，仅允许固定结果的历史质量消融",
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
