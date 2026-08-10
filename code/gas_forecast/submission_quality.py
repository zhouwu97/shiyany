"""初赛 ``input.csv`` 的原始字段质量契约与可复现修复。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class SubmissionQualityMode(str, Enum):
    """提交质量链的作用域。

    ``Q_CAUSAL`` 只使用训练期和当前时点之前的评分观测，能够参与构造模型输入。
    ``Q_REFERENCE`` 只针对已经生成的最终 ``input.csv`` 副本做矩阵归一化，禁止将
    其结果写回预测模型。
    """

    Q_CAUSAL = "Q_CAUSAL"
    Q_REFERENCE = "Q_REFERENCE"


Q_CAUSAL = SubmissionQualityMode.Q_CAUSAL.value
"""因果质量链模式名。"""

Q_REFERENCE = SubmissionQualityMode.Q_REFERENCE.value
"""仅限最终提交副本的参考质量链模式名。"""

REFERENCE_IQR_INTERPOLATIONS: tuple[str, ...] = (
    "linear",
    "lower",
    "higher",
    "midpoint",
    "nearest",
)


@dataclass(frozen=True)
class CausalSourceSettings:
    """评分原始字段的历史修复参数。

    默认值固定为参考仓库 ``official_preliminary.yaml`` 的提交预处理配置：
    672 点历史窗口、Hampel 最小历史 96 点和 MAD 阈值 6。训练期中位数仅在
    这段历史无法提供前向值时作为最后兜底。
    """

    history_points: int = 672
    hampel_window: int = 672
    hampel_mad_threshold: float = 6.0
    hampel_min_periods: int = 96

    def to_dict(self) -> dict[str, object]:
        """返回可直接 JSON 序列化的稳定参数收据。"""

        return {
            "history_points": int(self.history_points),
            "hampel_window": int(self.hampel_window),
            "hampel_mad_threshold": float(self.hampel_mad_threshold),
            "hampel_min_periods": int(self.hampel_min_periods),
        }


@dataclass(frozen=True)
class FittedSubmissionFeatureSchema:
    """只由训练段拟合的工程特征 schema 与回退统计。"""

    training_rows: int
    retained_columns: tuple[str, ...]
    all_nonfinite_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    duplicate_columns: tuple[tuple[str, str], ...]
    medians: tuple[tuple[str, float], ...]

    def median_map(self) -> dict[str, float]:
        """返回训练期中位数的独立映射。"""

        return dict(self.medians)

    def to_dict(self) -> dict[str, object]:
        """导出稳定、可序列化的训练 schema。"""

        return {
            "training_rows": int(self.training_rows),
            "retained_columns": list(self.retained_columns),
            "all_nonfinite_columns": list(self.all_nonfinite_columns),
            "constant_columns": list(self.constant_columns),
            "duplicate_columns": [
                {"column": column, "duplicate_of": duplicate_of}
                for column, duplicate_of in self.duplicate_columns
            ],
            "medians": {column: float(value) for column, value in self.medians},
        }


@dataclass(frozen=True)
class ReferenceNormalizationSettings:
    """最终提交副本的固定全矩阵质量门禁参数。"""

    drop_constant_columns: bool = True
    drop_duplicate_columns: bool = True
    iqr_multiplier: float = 1.5
    clip_iqr_multiplier: float = 1.0
    iqr_interpolations: tuple[str, ...] = REFERENCE_IQR_INTERPOLATIONS
    zscore_threshold: float = 3.0
    max_iqr_passes: int = 10

    def to_dict(self) -> dict[str, object]:
        """返回不含 pandas 或 NumPy 对象的配置快照。"""

        return {
            "drop_constant_columns": bool(self.drop_constant_columns),
            "drop_duplicate_columns": bool(self.drop_duplicate_columns),
            "iqr_multiplier": float(self.iqr_multiplier),
            "clip_iqr_multiplier": float(self.clip_iqr_multiplier),
            "iqr_interpolations": list(self.iqr_interpolations),
            "zscore_threshold": float(self.zscore_threshold),
            "max_iqr_passes": int(self.max_iqr_passes),
        }


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
            "mode": Q_CAUSAL,
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


def _string_column_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """复制并验证字段名，避免数值字段名在报告中产生不稳定表示。"""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} 必须是 pandas.DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError(f"{label} 含重复字段")
    output = frame.copy()
    output.columns = [str(column) for column in output.columns]
    if output.columns.duplicated().any():
        raise ValueError(f"{label} 的字段转为字符串后重复")
    return output


def _time_indexed_frame(
    frame: pd.DataFrame,
    label: str,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """取得唯一、有序的时间索引和不含 ``datetime`` 的值矩阵。"""

    source = _string_column_frame(frame, label)
    if "datetime" in source.columns:
        timestamps = pd.DatetimeIndex(pd.to_datetime(source["datetime"], errors="coerce"))
        values = source.drop(columns="datetime")
    elif isinstance(source.index, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(source.index)
        values = source
    else:
        raise ValueError(f"{label} 需要 DatetimeIndex 或 datetime 字段")
    if bool(timestamps.isna().any()):
        raise ValueError(f"{label} 含非法时间戳")
    if timestamps.has_duplicates:
        raise ValueError(f"{label} 的时间戳必须唯一")

    # 使用稳定排序使输入顺序不会影响因果计算，同时不会打乱同一时点之外的字段顺序。
    order = np.argsort(timestamps.asi8, kind="stable")
    ordered_index = timestamps.take(order)
    ordered_values = values.iloc[order].copy()
    ordered_values.index = ordered_index
    ordered_values.index.name = "datetime"
    return ordered_index, ordered_values


def _coerce_origins(
    origins: pd.DatetimeIndex | Sequence[object] | None,
    scoring_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """验证预测起点，保留调用方提供的顺序以对应最终提交行。"""

    if origins is None:
        selected = scoring_index
    else:
        selected = pd.DatetimeIndex(pd.to_datetime(origins, errors="coerce"))
    if bool(selected.isna().any()):
        raise ValueError("预测起点含非法时间戳")
    if selected.has_duplicates:
        raise ValueError("预测起点必须唯一")
    missing = selected.difference(scoring_index)
    if len(missing):
        raise ValueError(f"评分原始输入未覆盖预测起点: {missing[:3].tolist()}")
    selected.name = "datetime"
    return selected


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """统一把非数值和正负无穷值表示为 ``NaN``。"""

    return (
        frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    )


def _feature_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """获得特征矩阵；若包含时间列，只把它作为行标识而不参与统计。"""

    source = _string_column_frame(frame, label)
    return source.drop(columns="datetime", errors="ignore")


def _duplicate_column_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    """按列顺序返回精确重复列及其首次出现列。"""

    records: list[dict[str, str]] = []
    retained: list[str] = []
    for column in frame.columns:
        name = str(column)
        duplicate_of = next(
            (previous for previous in retained if frame[name].equals(frame[previous])),
            None,
        )
        if duplicate_of is None:
            retained.append(name)
        else:
            records.append({"column": name, "duplicate_of": duplicate_of})
    return records


def _coerce_reference_settings(
    settings: ReferenceNormalizationSettings | Mapping[str, Any] | None,
) -> ReferenceNormalizationSettings:
    """解析且固定参考矩阵门禁的必需五种分位数口径。"""

    if settings is None:
        values: Mapping[str, Any] = {}
    elif isinstance(settings, ReferenceNormalizationSettings):
        values = settings.to_dict()
    elif isinstance(settings, Mapping):
        values = settings
    else:
        raise TypeError("参考质量归一化 settings 必须是映射或 ReferenceNormalizationSettings")

    interpolation_values = values.get("iqr_interpolations", REFERENCE_IQR_INTERPOLATIONS)
    if isinstance(interpolation_values, str):
        raise ValueError("iqr_interpolations 必须按顺序包含五种分位数插值方法")
    interpolations = tuple(str(value) for value in interpolation_values)
    if len(interpolations) != len(REFERENCE_IQR_INTERPOLATIONS) or set(interpolations) != set(
        REFERENCE_IQR_INTERPOLATIONS
    ):
        raise ValueError("参考质量归一化必须覆盖五种分位数插值方法")

    parsed = ReferenceNormalizationSettings(
        drop_constant_columns=bool(values.get("drop_constant_columns", True)),
        drop_duplicate_columns=bool(values.get("drop_duplicate_columns", True)),
        iqr_multiplier=float(values.get("iqr_multiplier", 1.5)),
        clip_iqr_multiplier=float(values.get("clip_iqr_multiplier", 1.0)),
        # 固定顺序是报告和复现契约的一部分，不由调用方的列表顺序改变。
        iqr_interpolations=REFERENCE_IQR_INTERPOLATIONS,
        zscore_threshold=float(values.get("zscore_threshold", 3.0)),
        max_iqr_passes=int(values.get("max_iqr_passes", 10)),
    )
    if not parsed.drop_constant_columns or not parsed.drop_duplicate_columns:
        raise ValueError("参考质量归一化必须删除常量列和重复列")
    if parsed.iqr_multiplier <= 0.0 or not 0.0 < parsed.clip_iqr_multiplier < parsed.iqr_multiplier:
        raise ValueError("全矩阵 IQR 裁剪倍数必须满足 0 < clip < gate")
    if parsed.zscore_threshold <= 0.0:
        raise ValueError("Z-score 阈值必须为正数")
    if not 0 < parsed.max_iqr_passes <= 10:
        raise ValueError("参考质量归一化最多允许 10 轮 IQR 裁剪")
    return parsed


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
        if column == "datetime" or str(column) in allowed or str(column).startswith("feat_")
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
            if bounds is not None and (column in policy.batch_iqr_clip_columns or is_feature):
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
        "mode": Q_CAUSAL,
        "reference_only": False,
        "policy": policy.name,
        "frozen": True,
        "training_end": fitted_policy.training_end,
        "scoring_rows": len(scoring_origin_rows),
        "dropped_raw_columns": list(fitted_policy.dropped_raw_columns),
        "dropped_invalid_columns": list(fitted_policy.dropped_invalid_columns),
        "dropped_constant_columns": list(fitted_policy.dropped_constant_columns),
        "dropped_duplicate_columns": [
            list(item) for item in fitted_policy.dropped_duplicate_columns
        ],
        "added_columns": added_columns,
        "invalid_cells_by_column": invalid_cells,
        "clipped_cells_by_column": clipped_cells,
        "repaired_cells": int(sum(invalid_cells.values()) + sum(clipped_cells.values())),
        "audit": audit,
    }
    return output, report


def fit_causal_quality_policy(
    training_frame: pd.DataFrame,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    *,
    train_end: str | pd.Timestamp | None = None,
) -> FittedSubmissionQualityPolicy:
    """正式 ``Q_CAUSAL`` 拟合入口。

    保留 ``fit_quality_policy`` 作为兼容名称，但正式提交链通过这个显式名称
    表达“训练期拟合一次、评分期只变换”的职责边界。
    """

    return fit_quality_policy(training_frame, policy=policy, train_end=train_end)


def transform_causal_model_input(
    scoring_origin_rows: pd.DataFrame,
    fitted_policy: FittedSubmissionQualityPolicy,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """正式 ``Q_CAUSAL`` 评分输入变换入口，不会重新估计统计量。"""

    return transform_submission_input(scoring_origin_rows, fitted_policy, strict=strict)


def _hampel_replacement(
    value: float,
    history: Sequence[float],
    settings: CausalSourceSettings,
) -> tuple[float, bool]:
    """仅用当前点之前的有限历史判断并修复单个 Hampel 异常。"""

    if len(history) < settings.hampel_min_periods:
        return value, False
    window = np.asarray(history[-settings.hampel_window :], dtype=float)
    finite = window[np.isfinite(window)]
    if finite.size < settings.hampel_min_periods:
        return value, False
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        return value, False
    if abs(value - median) > settings.hampel_mad_threshold * scale:
        return median, True
    return value, False


def prepare_submission_sources(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    preprocessing: Mapping[str, Any] | pd.DatetimeIndex | Sequence[object] | None = None,
    origins: pd.DatetimeIndex | Sequence[object] | None = None,
    *,
    history_points: int = 672,
    hampel_window: int = 672,
    hampel_mad_threshold: float = 6.0,
    hampel_min_periods: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """因果地准备提交原始字段，不读取评分期未来行。

    为兼容旧调用点，第三个位置参数可保留预处理映射；该映射不改变此函数固定的
    672 点历史、Hampel 窗口 672 / 最小历史 96 / MAD 6 规则。若第三个参数不是
    映射，则将其视为 ``origins``。
    """

    if preprocessing is not None and not isinstance(preprocessing, Mapping):
        if origins is not None:
            raise TypeError("origins 不能同时作为第三和第四个位置参数传入")
        origins = preprocessing
        preprocessing = None
    if history_points <= 0 or hampel_window <= 0:
        raise ValueError("提交原始字段历史窗口必须为正数")
    if hampel_mad_threshold <= 0.0:
        raise ValueError("Hampel MAD 阈值必须为正数")
    min_periods = (
        96 if hampel_min_periods is None else int(hampel_min_periods)
    )
    if min_periods <= 0 or min_periods > hampel_window:
        raise ValueError("Hampel 最小历史点数必须位于窗口范围内")
    settings = CausalSourceSettings(
        history_points=int(history_points),
        hampel_window=int(hampel_window),
        hampel_mad_threshold=float(hampel_mad_threshold),
        hampel_min_periods=min_periods,
    )

    training_index, training_values = _time_indexed_frame(training, "训练原始输入")
    scoring_index, scoring_values = _time_indexed_frame(scoring, "评分原始输入")
    selected_origins = _coerce_origins(origins, scoring_index)
    scoring_start = scoring_index.min()
    historical_training = training_values.loc[training_index < scoring_start]
    if historical_training.empty:
        raise ValueError("评分期之前没有可用于因果修复的训练历史")

    source_columns = [str(column) for column in scoring_values.columns]
    numeric_training = _numeric_frame(historical_training.reindex(columns=source_columns))
    numeric_scoring = _numeric_frame(scoring_values.reindex(columns=source_columns))
    invalid_columns = [
        column
        for column in source_columns
        if not bool(np.isfinite(numeric_training[column].to_numpy(dtype=float)).any())
    ]
    retained_columns = [column for column in source_columns if column not in invalid_columns]
    if not retained_columns:
        raise ValueError("训练期没有可用于提交的有效原始字段")

    repaired_all = pd.DataFrame(index=scoring_index)
    missing_repairs: dict[str, int] = {}
    outlier_repairs: dict[str, int] = {}
    median_fallbacks: dict[str, int] = {}
    history_rows = min(len(numeric_training), settings.history_points)
    for column in retained_columns:
        training_series = numeric_training[column]
        median = float(training_series.median(skipna=True))
        if not np.isfinite(median):
            # ``invalid_columns`` 已经保证这里不可能出现，保留防御性检查便于排障。
            raise ValueError(f"训练原始字段 {column} 没有有限中位数")

        history: list[float] = []
        for raw_value in training_series.tail(settings.history_points).to_numpy(dtype=float):
            if np.isfinite(raw_value):
                repaired_value, _ = _hampel_replacement(float(raw_value), history, settings)
            else:
                repaired_value = history[-1] if history else median
            history.append(float(repaired_value))
            if len(history) > settings.hampel_window:
                history.pop(0)

        values: list[float] = []
        missing_count = 0
        outlier_count = 0
        fallback_count = 0
        for raw_value in numeric_scoring[column].to_numpy(dtype=float):
            if np.isfinite(raw_value):
                repaired_value, was_outlier = _hampel_replacement(
                    float(raw_value), history, settings
                )
                outlier_count += int(was_outlier)
            else:
                missing_count += 1
                if history:
                    repaired_value = history[-1]
                else:
                    repaired_value = median
                    fallback_count += 1
            if not np.isfinite(repaired_value):
                raise ValueError(f"原始提交字段 {column} 因果修复后仍包含非有限值")
            values.append(float(repaired_value))
            history.append(float(repaired_value))
            if len(history) > settings.hampel_window:
                history.pop(0)

        repaired_all[column] = values
        missing_repairs[column] = int(missing_count)
        outlier_repairs[column] = int(outlier_count)
        median_fallbacks[column] = int(fallback_count)

    repaired = repaired_all.reindex(selected_origins)
    repaired.index.name = "datetime"
    if not np.isfinite(repaired.to_numpy(dtype=float)).all():
        raise ValueError("提交原始字段修复后仍包含 NaN/Inf")
    repaired.attrs["submission_quality_mode"] = Q_CAUSAL
    report: dict[str, object] = {
        "mode": Q_CAUSAL,
        "reference_only": False,
        "settings": settings.to_dict(),
        "training_rows": int(len(historical_training)),
        "scoring_rows": int(len(scoring_values)),
        "origin_rows": int(len(selected_origins)),
        "history_rows": int(history_rows),
        "retained_columns": retained_columns,
        "invalid_columns": invalid_columns,
        "all_nonfinite_columns": invalid_columns,
        "missing_repairs": missing_repairs,
        "outlier_repairs": outlier_repairs,
        "median_fallbacks": median_fallbacks,
        "nonfinite_after": 0,
        "legacy_preprocessing_supplied": preprocessing is not None,
    }
    return repaired, report


def fit_submission_feature_schema(
    training_features: pd.DataFrame,
) -> FittedSubmissionFeatureSchema:
    """只从训练段拟合工程特征的可提交 schema 和中位数。"""

    training = _numeric_frame(_feature_frame(training_features, "训练工程特征"))
    retained: list[str] = []
    invalid: list[str] = []
    constant: list[str] = []
    medians: dict[str, float] = {}
    for column in training.columns:
        name = str(column)
        finite_values = training[name].dropna()
        if finite_values.empty:
            invalid.append(name)
        elif int(finite_values.nunique(dropna=True)) <= 1:
            constant.append(name)
        else:
            retained.append(name)
            medians[name] = float(finite_values.median())

    duplicates: list[tuple[str, str]] = []
    final_retained: list[str] = []
    for column in retained:
        duplicate_of = next(
            (
                previous
                for previous in final_retained
                if training[column].equals(training[previous])
            ),
            None,
        )
        if duplicate_of is None:
            final_retained.append(column)
        else:
            duplicates.append((column, duplicate_of))

    return FittedSubmissionFeatureSchema(
        training_rows=int(len(training)),
        retained_columns=tuple(final_retained),
        all_nonfinite_columns=tuple(invalid),
        constant_columns=tuple(constant),
        duplicate_columns=tuple(duplicates),
        medians=tuple((column, medians[column]) for column in final_retained),
    )


def sanitize_submission_features(
    training_features: pd.DataFrame,
    scoring_features: pd.DataFrame,
    *,
    schema: FittedSubmissionFeatureSchema | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """按训练冻结 schema 删除坏特征，并仅用训练中位数填评分 NaN/Inf。"""

    fitted = schema or fit_submission_feature_schema(training_features)
    if not isinstance(fitted, FittedSubmissionFeatureSchema):
        raise TypeError("schema 必须是 FittedSubmissionFeatureSchema")
    if not fitted.retained_columns:
        raise ValueError("训练期清理后没有可提交的工程特征")

    scoring = _numeric_frame(_feature_frame(scoring_features, "评分工程特征"))
    medians = fitted.median_map()
    output = pd.DataFrame(index=scoring.index)
    missing_scoring_columns: list[str] = []
    filled_with_training_median: dict[str, int] = {}
    for column in fitted.retained_columns:
        if column in scoring.columns:
            values = scoring[column].copy()
        else:
            values = pd.Series(np.nan, index=scoring.index, dtype=float)
            missing_scoring_columns.append(column)
        invalid_mask = ~np.isfinite(values.to_numpy(dtype=float))
        filled_with_training_median[column] = int(invalid_mask.sum())
        if bool(invalid_mask.any()):
            values.iloc[np.flatnonzero(invalid_mask)] = medians[column]
        output[column] = values.astype(float)

    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError("清理后的提交工程特征仍包含 NaN/Inf")
    output.attrs["submission_quality_mode"] = Q_CAUSAL
    ignored_scoring_columns = [
        str(column) for column in scoring.columns if str(column) not in fitted.retained_columns
    ]
    report: dict[str, object] = {
        "mode": Q_CAUSAL,
        "reference_only": False,
        "schema_frozen": True,
        "schema": fitted.to_dict(),
        "retained_columns": list(fitted.retained_columns),
        "all_nonfinite": list(fitted.all_nonfinite_columns),
        "constant": list(fitted.constant_columns),
        "duplicate": [column for column, _ in fitted.duplicate_columns],
        "duplicate_of": [
            {"column": column, "duplicate_of": duplicate_of}
            for column, duplicate_of in fitted.duplicate_columns
        ],
        "missing_scoring_columns": missing_scoring_columns,
        "ignored_scoring_columns": ignored_scoring_columns,
        "filled_with_training_median": filled_with_training_median,
        "nonfinite_after": 0,
    }
    return output, report


def inspect_submission_input_quality(
    frame: pd.DataFrame,
    *,
    iqr_multiplier: float = 1.5,
    iqr_interpolations: Sequence[str] = REFERENCE_IQR_INTERPOLATIONS,
    zscore_threshold: float | None = 3.0,
) -> dict[str, object]:
    """按全部指定分位数口径审计提交特征矩阵的终态质量。"""

    if iqr_multiplier <= 0.0:
        raise ValueError("提交矩阵 IQR 倍数必须大于 0")
    supported = set(REFERENCE_IQR_INTERPOLATIONS)
    interpolations = tuple(str(value) for value in iqr_interpolations)
    if not interpolations or len(set(interpolations)) != len(interpolations):
        raise ValueError("提交矩阵分位数插值方法不能为空或重复")
    if not set(interpolations).issubset(supported):
        raise ValueError("提交矩阵包含不支持的分位数插值方法")
    if zscore_threshold is not None and zscore_threshold <= 0.0:
        raise ValueError("提交矩阵 Z-score 阈值必须大于 0")

    numeric = _numeric_frame(_feature_frame(frame, "提交矩阵"))
    nonfinite_by_column = {
        str(column): int((~np.isfinite(numeric[column].to_numpy(dtype=float))).sum())
        for column in numeric.columns
        if int((~np.isfinite(numeric[column].to_numpy(dtype=float))).sum()) > 0
    }
    constant_columns = [
        str(column) for column in numeric.columns if int(numeric[column].nunique(dropna=True)) <= 1
    ]
    duplicate_records = _duplicate_column_records(numeric)
    duplicate_columns = [record["column"] for record in duplicate_records]

    union_masks: dict[str, np.ndarray] = {
        str(column): np.zeros(len(numeric), dtype=bool) for column in numeric.columns
    }
    outliers_by_method: dict[str, dict[str, int]] = {}
    for interpolation in interpolations:
        method_counts: dict[str, int] = {}
        for column in numeric.columns:
            name = str(column)
            values = numeric[name].to_numpy(dtype=float)
            finite_mask = np.isfinite(values)
            finite_values = values[finite_mask]
            if finite_values.size == 0:
                continue
            series = pd.Series(finite_values)
            q1 = float(series.quantile(0.25, interpolation=interpolation))
            q3 = float(series.quantile(0.75, interpolation=interpolation))
            iqr = q3 - q1
            if not np.isfinite(iqr):
                continue
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask = finite_mask & ((values < lower) | (values > upper))
            count = int(mask.sum())
            if count:
                method_counts[name] = count
                union_masks[name] |= mask
        outliers_by_method[interpolation] = method_counts

    zscore_outliers: dict[str, int] = {}
    if zscore_threshold is not None:
        for column in numeric.columns:
            name = str(column)
            values = numeric[name].to_numpy(dtype=float)
            finite_mask = np.isfinite(values)
            finite_values = values[finite_mask]
            if finite_values.size == 0:
                continue
            mean = float(finite_values.mean())
            standard_deviation = float(finite_values.std(ddof=0))
            if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
                continue
            count = int(
                (np.abs(finite_values - mean) / standard_deviation > zscore_threshold).sum()
            )
            if count:
                zscore_outliers[name] = count

    linear_outliers = outliers_by_method.get("linear", {})
    all_method_outliers = {
        column: int(mask.sum()) for column, mask in union_masks.items() if int(mask.sum()) > 0
    }
    return {
        "rows": int(len(numeric)),
        "columns": int(len(numeric.columns)),
        "nonfinite_cells": int(sum(nonfinite_by_column.values())),
        "nonfinite_by_column": nonfinite_by_column,
        "constant_columns": constant_columns,
        "duplicate_columns": duplicate_columns,
        "duplicate_column_pairs": duplicate_records,
        "iqr_multiplier": float(iqr_multiplier),
        "iqr_interpolations": list(interpolations),
        "iqr_outlier_cells": int(sum(linear_outliers.values())),
        "iqr_outliers_by_column": linear_outliers,
        "iqr_outlier_cells_all_methods": int(sum(all_method_outliers.values())),
        "iqr_outliers_by_column_all_methods": all_method_outliers,
        "iqr_outliers_by_method": outliers_by_method,
        "iqr_method_summary": {
            method: {"cells": int(sum(counts.values())), "columns": int(len(counts))}
            for method, counts in outliers_by_method.items()
        },
        "zscore_threshold": None if zscore_threshold is None else float(zscore_threshold),
        "zscore_outlier_cells": int(sum(zscore_outliers.values())),
        "zscore_outliers_by_column": zscore_outliers,
    }


def _reference_iqr_winsorization_pass(
    frame: pd.DataFrame,
    settings: ReferenceNormalizationSettings,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """按五种分位数口径各自的内侧边界执行一次联合 IQR 截尾。"""

    changed_by_column: dict[str, int] = {}
    changed_by_method: dict[str, dict[str, int]] = {
        interpolation: {} for interpolation in settings.iqr_interpolations
    }
    for column in frame.columns:
        name = str(column)
        values = frame[name].to_numpy(dtype=float)
        lower_targets = np.full(len(values), -np.inf, dtype=float)
        upper_targets = np.full(len(values), np.inf, dtype=float)
        lower_mask = np.zeros(len(values), dtype=bool)
        upper_mask = np.zeros(len(values), dtype=bool)
        for interpolation in settings.iqr_interpolations:
            series = pd.Series(values)
            q1 = float(series.quantile(0.25, interpolation=interpolation))
            q3 = float(series.quantile(0.75, interpolation=interpolation))
            iqr = q3 - q1
            if not np.isfinite(iqr):
                continue
            gate_lower = q1 - settings.iqr_multiplier * iqr
            gate_upper = q3 + settings.iqr_multiplier * iqr
            clip_lower = q1 - settings.clip_iqr_multiplier * iqr
            clip_upper = q3 + settings.clip_iqr_multiplier * iqr
            method_lower_mask = values < gate_lower
            method_upper_mask = values > gate_upper
            method_count = int(method_lower_mask.sum() + method_upper_mask.sum())
            if method_count:
                changed_by_method[interpolation][name] = method_count
                lower_mask |= method_lower_mask
                upper_mask |= method_upper_mask
                lower_targets[method_lower_mask] = np.maximum(
                    lower_targets[method_lower_mask], clip_lower
                )
                upper_targets[method_upper_mask] = np.minimum(
                    upper_targets[method_upper_mask], clip_upper
                )
        if not bool((lower_mask | upper_mask).any()):
            continue
        updated = values.copy()
        updated[lower_mask] = np.maximum(updated[lower_mask], lower_targets[lower_mask])
        updated[upper_mask] = np.minimum(updated[upper_mask], upper_targets[upper_mask])
        changed = ~np.isclose(updated, values, rtol=1e-12, atol=1e-12)
        changed_count = int(changed.sum())
        if changed_count:
            frame[name] = updated
            changed_by_column[name] = changed_count
    return changed_by_column, changed_by_method


def normalize_submission_input_frame(
    frame: pd.DataFrame,
    settings: ReferenceNormalizationSettings | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """在最终提交副本上执行 ``Q_REFERENCE`` 全矩阵归一化。

    本函数总是深复制输入，并在报告中标记其仅限最终提交副本。它不拟合或更新任何
    模型统计，调用方不得将结果作为模型预测的反馈输入。
    """

    parsed = _coerce_reference_settings(settings)
    source = _string_column_frame(frame, "提交矩阵")
    datetime_values = source["datetime"].copy() if "datetime" in source.columns else None
    numeric = _numeric_frame(source.drop(columns="datetime", errors="ignore"))
    if numeric.empty:
        raise ValueError("提交矩阵不含可归一化特征")

    initial_quality = inspect_submission_input_quality(
        source,
        iqr_multiplier=parsed.iqr_multiplier,
        iqr_interpolations=parsed.iqr_interpolations,
        zscore_threshold=parsed.zscore_threshold,
    )
    output = numeric.copy(deep=True)
    dropped_nonfinite_columns = list(initial_quality["nonfinite_by_column"])
    output = output.drop(columns=dropped_nonfinite_columns, errors="ignore")
    constant_before = [
        column for column in initial_quality["constant_columns"] if column in output.columns
    ]
    output = output.drop(columns=constant_before, errors="ignore")

    duplicate_before_records = _duplicate_column_records(output)
    duplicate_before = [record["column"] for record in duplicate_before_records]
    output = output.drop(columns=duplicate_before, errors="ignore")
    if output.empty:
        raise ValueError("提交矩阵质量归一化后没有可用特征")

    winsorized_by_column: dict[str, int] = {}
    winsorization_passes: list[dict[str, object]] = []
    constant_after: list[str] = []
    duplicate_after: list[str] = []
    duplicate_after_records: list[dict[str, str]] = []
    for pass_number in range(1, parsed.max_iqr_passes + 1):
        changed_by_column, changed_by_method = _reference_iqr_winsorization_pass(output, parsed)
        for column, count in changed_by_column.items():
            winsorized_by_column[column] = winsorized_by_column.get(column, 0) + int(count)

        new_constants = [
            str(column)
            for column in output.columns
            if int(output[str(column)].nunique(dropna=True)) <= 1
        ]
        if new_constants:
            output = output.drop(columns=new_constants, errors="ignore")
            constant_after.extend(
                column for column in new_constants if column not in constant_after
            )

        new_duplicate_records = _duplicate_column_records(output)
        new_duplicates = [record["column"] for record in new_duplicate_records]
        if new_duplicates:
            output = output.drop(columns=new_duplicates, errors="ignore")
            duplicate_after.extend(
                column for column in new_duplicates if column not in duplicate_after
            )
            duplicate_after_records.extend(
                record for record in new_duplicate_records if record not in duplicate_after_records
            )

        winsorization_passes.append(
            {
                "pass": int(pass_number),
                "winsorized_cells": int(sum(changed_by_column.values())),
                "columns": changed_by_column,
                "iqr_outliers_by_method": changed_by_method,
                "dropped_constant_columns": new_constants,
                "dropped_duplicate_columns": new_duplicates,
            }
        )
        if output.empty:
            break
        pass_quality = inspect_submission_input_quality(
            output,
            iqr_multiplier=parsed.iqr_multiplier,
            iqr_interpolations=parsed.iqr_interpolations,
            zscore_threshold=None,
        )
        if int(pass_quality["iqr_outlier_cells_all_methods"]) == 0:
            break

    if output.empty:
        raise ValueError("提交矩阵 IQR 归一化后没有可用特征")
    residual_quality = inspect_submission_input_quality(
        output,
        iqr_multiplier=parsed.iqr_multiplier,
        iqr_interpolations=parsed.iqr_interpolations,
        zscore_threshold=parsed.zscore_threshold,
    )
    residual_outlier_columns = set(residual_quality["iqr_outliers_by_column_all_methods"])
    residual_outlier_columns.update(residual_quality["zscore_outliers_by_column"])
    residual_bad_columns = set(residual_quality["nonfinite_by_column"])
    residual_bad_columns.update(residual_quality["constant_columns"])
    residual_bad_columns.update(residual_quality["duplicate_columns"])
    residual_bad_columns.update(residual_outlier_columns)
    ordered_residual_bad_columns = [
        str(column) for column in output.columns if str(column) in residual_bad_columns
    ]
    ordered_residual_outlier_columns = [
        str(column) for column in output.columns if str(column) in residual_outlier_columns
    ]
    if ordered_residual_bad_columns:
        output = output.drop(columns=ordered_residual_bad_columns, errors="ignore")
    if output.empty:
        raise ValueError("提交矩阵终态质量门禁删除了全部特征")

    final_quality = inspect_submission_input_quality(
        output,
        iqr_multiplier=parsed.iqr_multiplier,
        iqr_interpolations=parsed.iqr_interpolations,
        zscore_threshold=parsed.zscore_threshold,
    )
    gate_passed = (
        int(final_quality["nonfinite_cells"]) == 0
        and not final_quality["constant_columns"]
        and not final_quality["duplicate_columns"]
        and int(final_quality["iqr_outlier_cells_all_methods"]) == 0
        and int(final_quality["zscore_outlier_cells"]) == 0
    )
    if not gate_passed:
        raise ValueError(f"提交矩阵参考质量门禁失败: {final_quality}")

    if datetime_values is not None:
        normalized = pd.DataFrame({"datetime": datetime_values}, index=source.index)
        for column in output.columns:
            normalized[str(column)] = output[str(column)].to_numpy(dtype=float)
    else:
        normalized = output.copy()
    normalized.attrs["submission_quality_mode"] = Q_REFERENCE
    normalized.attrs["reference_only"] = True
    terminal_gate = {
        "nonfinite_cells": 0,
        "constant_columns": 0,
        "duplicate_columns": 0,
        "iqr_outlier_cells_all_methods": 0,
        "zscore_outlier_cells": 0,
    }
    report: dict[str, object] = {
        "mode": Q_REFERENCE,
        "reference_only": True,
        "model_input_mutated": False,
        "settings": parsed.to_dict(),
        "enabled": True,
        "clip_iqr_multiplier": float(parsed.clip_iqr_multiplier),
        "iqr_interpolations": list(parsed.iqr_interpolations),
        "zscore_threshold": float(parsed.zscore_threshold),
        "initial_quality": initial_quality,
        "dropped_nonfinite_columns": dropped_nonfinite_columns,
        "dropped_constant_columns_before_winsor": constant_before,
        "dropped_duplicate_columns_before_winsor": duplicate_before,
        "dropped_duplicate_column_pairs_before_winsor": duplicate_before_records,
        "winsorized_cells": int(sum(winsorized_by_column.values())),
        "winsorized_by_column": winsorized_by_column,
        "winsorization_passes": winsorization_passes,
        "dropped_constant_columns_after_winsor": constant_after,
        "dropped_duplicate_columns_after_winsor": duplicate_after,
        "dropped_duplicate_column_pairs_after_winsor": duplicate_after_records,
        "residual_quality": residual_quality,
        "dropped_residual_bad_columns": ordered_residual_bad_columns,
        "dropped_residual_outlier_columns": ordered_residual_outlier_columns,
        "final_quality": final_quality,
        "terminal_gate": terminal_gate,
        "passed": True,
    }
    return normalized, report


def prepare_reference_submission_input(
    causal_model_input: pd.DataFrame,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    *,
    settings: ReferenceNormalizationSettings | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """对已冻结的 ``Q_CAUSAL`` 输入创建 ``Q_REFERENCE`` 提交副本。

    ``policy`` 仅为旧调用方保留形参，绝不能在这里调用
    :func:`fit_quality_policy` 或 :func:`transform_submission_input`。参考阶段
    允许读取完整提交矩阵，但输出只用于 ``input.csv``，不会回流模型预测。
    """

    del policy
    normalized, report = normalize_submission_input_frame(causal_model_input, settings)
    report = dict(report)
    report["production_eligible"] = False
    report["reason"] = "Q_REFERENCE 仅归一化冻结输入副本，不拟合或改写模型输入"
    return normalized, report


def prepare_exact_reference_input(
    training_raw: pd.DataFrame,
    scoring_raw: pd.DataFrame,
    origins: pd.DatetimeIndex | Sequence[object],
    training_features: pd.DataFrame,
    scoring_features: pd.DataFrame,
    *,
    history_points: int = 672,
    hampel_window: int = 672,
    hampel_mad_threshold: float = 6.0,
    hampel_min_periods: int | None = None,
    settings: ReferenceNormalizationSettings | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """按参考仓库完整语义构造 ``input.csv``（R1 Exact Reference Input Clone）。

    顺序与 ``diaofenyuan/aic-gangtie`` 的 ``predict_pipeline`` 一致：

    1. :func:`prepare_submission_sources`：从官方评分原始字段出发，训练期判定
       invalid 列，Hampel 672/96/6 + median 兜底 + 无限制 ffill 修复；
    2. :func:`sanitize_submission_features`：训练期拟合 all-nonfinite /
       constant / duplicate / median schema，评分期只套用；
    3. ``pd.concat([repaired_raw, sanitized_features], axis=1)``；
    4. :func:`normalize_submission_input_frame`：全矩阵 IQR/Z-score 归一化。

    本链只构造提交副本，绝不回流模型预测；``training_raw``/``scoring_raw``
    应来自 :func:`gas_forecast.data.load_original_input_frame`，特征帧应只保留
    ``feat_`` 前缀列（与参考仓库的纯工程特征语义一致）。
    """

    repaired_raw, raw_report = prepare_submission_sources(
        training_raw,
        scoring_raw,
        origins=origins,
        history_points=history_points,
        hampel_window=hampel_window,
        hampel_mad_threshold=hampel_mad_threshold,
        hampel_min_periods=hampel_min_periods,
    )
    sanitized, feature_report = sanitize_submission_features(
        training_features,
        scoring_features,
    )
    # 参考语义要求评分特征与修复后的 raw sources 都对齐到 origins。
    sanitized = sanitized.reindex(repaired_raw.index)
    if sanitized.isna().any().any():
        raise ValueError("评分工程特征未覆盖全部预测起点")
    combined = pd.concat([repaired_raw, sanitized], axis=1)
    if combined.index.has_duplicates:
        raise ValueError("R1 拼接后的提交帧时间戳不唯一")
    combined.index.name = "datetime"
    combined = combined.reset_index(names="datetime")
    if not np.isfinite(combined.drop(columns="datetime").to_numpy(dtype=float)).all():
        raise ValueError("R1 拼接后的提交帧仍包含 NaN/Inf")

    normalized, matrix_report = normalize_submission_input_frame(combined, settings)

    report: dict[str, object] = {
        "mode": Q_REFERENCE,
        "reference_only": True,
        "feeds_model": False,
        "pipeline": "exact_reference_clone_v1",
        "production_eligible": False,
        "reason": "R1 仅构造最终 input.csv 提交副本，不回流模型预测",
        "raw_sources": raw_report,
        "feature_sanitize": feature_report,
        "concat": {
            "rows": int(len(combined)),
            "raw_columns": list(repaired_raw.columns),
            "raw_column_count": int(len(repaired_raw.columns)),
            "feature_columns": list(sanitized.columns),
            "feature_column_count": int(len(sanitized.columns)),
            "input_columns_excluding_datetime": int(len(combined.columns) - 1),
        },
        "matrix_normalization": matrix_report,
        "final_quality": matrix_report["final_quality"],
    }
    return normalized, report


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


def prepare_full_matrix_submission_input(
    input_frame: pd.DataFrame,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    *,
    iqr_multiplier: float = 1.5,
    clip_iqr_multiplier: float = 1.0,
    iqr_interpolations: Iterable[str] = REFERENCE_IQR_INTERPOLATIONS,
    zscore_threshold: float = 3.0,
    max_iqr_passes: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """兼容旧入口，直接生成仅供提交的 ``Q_REFERENCE`` 矩阵。

    历史版本在这里先对传入帧再次拟合 ``Q_CAUSAL``，会让参考归一化意外
    消费评分期分布。现在调用方若需要因果清洗，必须先显式调用
    :func:`fit_causal_quality_policy` / :func:`transform_causal_model_input`；本
    入口只保留全矩阵归一化的旧函数名。
    """

    interpolations = tuple(str(value) for value in iqr_interpolations)
    settings = {
        "drop_constant_columns": True,
        "drop_duplicate_columns": True,
        "iqr_multiplier": float(iqr_multiplier),
        "clip_iqr_multiplier": float(clip_iqr_multiplier),
        "iqr_interpolations": interpolations,
        "zscore_threshold": float(zscore_threshold),
        "max_iqr_passes": int(max_iqr_passes),
    }
    normalized, reference_report = prepare_reference_submission_input(
        input_frame,
        policy,
        settings=settings,
    )
    missing_required = [
        column for column in policy.required_raw_columns if column not in normalized.columns
    ]
    if missing_required:
        raise ValueError(f"全矩阵质量归一化不能删除必需 raw 字段: {missing_required}")

    dropped_constants = list(reference_report["dropped_constant_columns_before_winsor"])
    dropped_constants.extend(
        column
        for column in reference_report["dropped_constant_columns_after_winsor"]
        if column not in dropped_constants
    )
    dropped_duplicates = list(reference_report["dropped_duplicate_column_pairs_before_winsor"])
    dropped_duplicates.extend(
        record
        for record in reference_report["dropped_duplicate_column_pairs_after_winsor"]
        if record not in dropped_duplicates
    )
    combined_winsorized = dict(reference_report["winsorized_by_column"])

    return normalized, {
        "mode": Q_REFERENCE,
        "reference_only": True,
        "model_input_mutated": False,
        "policy": f"{policy.name}_full_matrix_v2",
        "production_eligible": False,
        "reason": "Q_REFERENCE 仅作用于已生成的最终 input.csv 副本，不回流模型预测",
        "base_quality": None,
        "initial_quality": reference_report["initial_quality"],
        "dropped_constant_columns": dropped_constants,
        "dropped_duplicate_columns": dropped_duplicates,
        "dropped_constant_columns_after_winsor": reference_report[
            "dropped_constant_columns_after_winsor"
        ],
        "dropped_residual_outlier_columns": reference_report["dropped_residual_outlier_columns"],
        "winsorized_cells": int(sum(combined_winsorized.values())),
        "winsorized_by_column": combined_winsorized,
        "winsorization_passes": reference_report["winsorization_passes"],
        "final_quality": reference_report["final_quality"],
        "reference_quality": reference_report,
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
