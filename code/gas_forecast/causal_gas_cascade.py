"""严格因果的煤气级联预测。

本模块是独立于历史 ``gas_stage.py`` 的研究实现。它把未来煤气轨迹预测
（Stage1）和发电量预测（Stage2）分成两个明确的契约：Stage2 的未来煤气
字段只能来自 held-fold OOF 预测，不能直接从原始数据读取未来观测。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from gas_forecast.model_v1 import make_ridge_pipeline
from gas_forecast.scoring import competition_mape


HORIZONS: tuple[int, ...] = tuple(range(1, 9))
RESOURCE_NAMES: tuple[str, ...] = (
    "blast_furnace_gas",
    "coke_gas",
    "converter_gas",
    "gas_holder",
    "major_users",
    "generator_gas_consumption_proxy",
)
GENERATOR_TARGETS: tuple[str, ...] = ("generator_1", "generator_all")


# 列名映射只依赖输入 schema，不依赖数值、时间范围或未来标签。
_EXACT_ALIASES: dict[str, tuple[str, ...]] = {
    "blast_furnace_gas": (
        "blast_furnace_gas",
        "blast_furnace_gas_production",
        "bf_gas",
    ),
    "coke_gas": ("coke_gas", "coke_gas_production"),
    "converter_gas": ("converter_gas", "converter_gas_production"),
    "gas_holder": (
        "gas_holder",
        "gas_holder_level",
        "blast_furnace_gas_holder_2",
    ),
    "major_users": ("major_users", "major_user", "gas_users", "gas_user_total"),
    "generator_gas_consumption_proxy": (
        "generator_gas_consumption_proxy",
        "generator_gas_use_total",
    ),
}


@dataclass(frozen=True)
class CascadeConfig:
    """级联模型配置。

    ``min_train_rows`` 和 ``min_validation_rows`` 比项目旧入口更小，便于
    在单元测试和短实验数据上复现同一套严格时间语义；真实运行可显式增大。
    """

    horizons: tuple[int, ...] = HORIZONS
    inner_folds: int = 5
    outer_folds: int = 5
    purge_steps: int | None = None
    ridge_alpha: float = 20.0
    min_train_rows: int = 64
    min_validation_rows: int = 16
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999
    random_state: int = 20250809
    strict_resources: bool = True

    def __post_init__(self) -> None:
        horizons = tuple(int(value) for value in self.horizons)
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("horizons 必须为正整数且非空")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons 必须严格递增且不能重复")
        if self.inner_folds < 2 or self.outer_folds < 1:
            raise ValueError("inner_folds 至少为2，outer_folds 至少为1")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha 不能为负数")
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("分位数边界无效")

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def purge(self) -> int:
        # 严格规则：最长标签结束时刻必须早于验证起点，故多留一格。
        return self.purge_steps if self.purge_steps is not None else self.max_horizon + 1


@dataclass(frozen=True)
class ResourceMapping:
    """规范资源名到实际输入列的稳定映射。"""

    columns: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(values) for name, values in self.columns.items()}


@dataclass
class Stage1PredictionBundle:
    """Stage1 预测及其来源收据。"""

    values: pd.DataFrame
    source: str
    is_oof: bool
    resource_names: tuple[str, ...]
    horizons: tuple[int, ...]

    @property
    def complete(self) -> pd.Series:
        return self.values.notna().all(axis=1)


@dataclass
class CascadeFitTrace:
    """训练过程的可序列化追踪信息。"""

    stage1_inner_folds: list[dict[str, object]] = field(default_factory=list)
    stage2_rows: int = 0
    stage1_rows: int = 0
    stage2_source: str = ""
    feature_columns: list[str] = field(default_factory=list)
    resource_mapping: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalise_column(name: object) -> str:
    """将实际列名标准化为可比较的 ASCII token。"""

    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _choose_exact(columns: Sequence[str], aliases: Iterable[str]) -> list[str]:
    by_normalised = {_normalise_column(column): column for column in columns}
    for alias in aliases:
        found = by_normalised.get(_normalise_column(alias))
        if found is not None:
            return [found]
    return []


def resolve_resource_mapping(
    columns: Iterable[str], *, strict: bool = True
) -> ResourceMapping:
    """按固定优先级解析六类煤气资源的实际列名。

    聚合类资源（高炉/焦炉/转炉产气和主要用户）按原始列名排序后求和，
    从而不受 CSV 字段顺序影响。发电机耗气 proxy 优先使用显式总量列，
    否则对 ``generator_use_*_gas`` 列求和。
    """

    actual = [str(column) for column in columns]
    normalised = {column: _normalise_column(column) for column in actual}
    result: dict[str, tuple[str, ...]] = {}

    # 显式 canonical 列优先，避免同时存在总量和分项时重复计数。
    for resource in RESOURCE_NAMES:
        exact = _choose_exact(actual, _EXACT_ALIASES[resource])
        if exact:
            result[resource] = tuple(exact)
            continue
        candidates: list[str] = []
        for column in actual:
            token = normalised[column]
            if resource == "blast_furnace_gas":
                if (
                    "blast_furnace" in token
                    and "holder" not in token
                    and "user" not in token
                    and "heater" not in token
                    and "into_gas" not in token
                    and not token.startswith("generator_use")
                ):
                    candidates.append(column)
            elif resource == "coke_gas":
                if (
                    ("coke_oven" in token or token.startswith("coke"))
                    and "user" not in token
                    and "into_gas" not in token
                    and not token.startswith("generator_use")
                ):
                    candidates.append(column)
            elif resource == "converter_gas":
                if (
                    "converter" in token
                    and "user" not in token
                    and "into_gas" not in token
                    and not token.startswith("generator_use")
                ):
                    candidates.append(column)
            elif resource == "gas_holder":
                if "holder" in token and not token.startswith("generator_use"):
                    candidates.append(column)
            elif resource == "major_users":
                if (
                    ("user" in token or "air_heater" in token or "into_gas_mixed" in token)
                    and not token.startswith("generator_use")
                ):
                    candidates.append(column)
            elif resource == "generator_gas_consumption_proxy":
                if token.startswith("generator_use") and "gas" in token:
                    candidates.append(column)
        result[resource] = tuple(sorted(set(candidates)))

    missing = [name for name in RESOURCE_NAMES if not result.get(name)]
    if missing and strict:
        raise ValueError(f"无法按稳定列名映射资源: {', '.join(missing)}")
    return ResourceMapping(columns=result)


def build_resource_frame(
    frame: pd.DataFrame,
    mapping: ResourceMapping | Mapping[str, Sequence[str]] | None = None,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """根据映射生成六列规范资源表，不读取任何未来行之外的信息。"""

    if mapping is None:
        resolved = resolve_resource_mapping(frame.columns, strict=strict)
    elif isinstance(mapping, ResourceMapping):
        resolved = mapping
    else:
        resolved = ResourceMapping(
            {name: tuple(values) for name, values in mapping.items()}
        )
    output = pd.DataFrame(index=frame.index)
    for resource in RESOURCE_NAMES:
        columns = tuple(resolved.columns.get(resource, ()))
        if not columns:
            output[resource] = np.nan
            continue
        absent = sorted(set(columns).difference(frame.columns))
        if absent:
            raise ValueError(f"资源 {resource} 的映射列不存在: {absent}")
        values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        output[resource] = values.sum(axis=1, min_count=1)
    return output


def _ensure_frame(frame: pd.DataFrame, *, allow_sort: bool = True) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("级联输入必须使用 DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError("级联输入时间轴必须唯一")
    if not frame.index.is_monotonic_increasing:
        if not allow_sort:
            raise ValueError("级联输入时间轴必须严格递增")
        # 预测接口接受生产数据的任意行顺序；按时间排序后再做因果截断。
        return frame.sort_index(kind="stable")
    return frame


def _state_features(frame: pd.DataFrame, resources: pd.DataFrame) -> pd.DataFrame:
    """构造 state_t 及其历史统计；所有窗口先 shift(1)。"""

    required = [target for target in GENERATOR_TARGETS if target in frame.columns]
    if len(required) != len(GENERATOR_TARGETS):
        missing = sorted(set(GENERATOR_TARGETS).difference(frame.columns))
        raise ValueError(f"缺少 Stage2 当前状态列: {missing}")
    base = pd.concat(
        [frame.loc[:, GENERATOR_TARGETS].apply(pd.to_numeric, errors="coerce"), resources],
        axis=1,
    )
    values: dict[str, pd.Series] = {}
    for column in base.columns:
        series = base[column].astype(float)
        values[f"state_{column}_t"] = series
        for lag in (1, 2, 4, 8, 16):
            values[f"state_{column}_lag_{lag}"] = series.shift(lag)
        history = series.shift(1)
        for window in (4, 8, 16):
            rolling = history.rolling(window, min_periods=max(2, window // 2))
            values[f"state_{column}_mean_{window}"] = rolling.mean()
            values[f"state_{column}_std_{window}"] = rolling.std()
        values[f"state_{column}_diff_1"] = series.diff()
    return pd.DataFrame(values, index=frame.index)


def _future_targets(series: pd.Series, horizons: tuple[int, ...], prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{prefix}_t+{15 * horizon}": series.shift(-horizon)
            for horizon in horizons
        },
        index=series.index,
    )


def _make_folds(
    index: pd.DatetimeIndex,
    *,
    folds: int,
    purge_steps: int,
    min_train_rows: int,
    min_validation_rows: int,
    prefix: str,
) -> list[dict[str, object]]:
    """在时间轴上生成 expanding、purged 折。"""

    n = len(index)
    if n < min_train_rows + purge_steps + 2 * min_validation_rows:
        raise ValueError(
            f"时间样本不足以生成折: n={n}, min_train_rows={min_train_rows}, "
            f"purge={purge_steps}, min_validation_rows={min_validation_rows}"
        )
    max_folds = min(int(folds), max(1, (n - min_train_rows - purge_steps) // min_validation_rows))
    validation_rows = max(min_validation_rows, (n - min_train_rows - purge_steps) // max_folds)
    first_start = n - max_folds * validation_rows
    result: list[dict[str, object]] = []
    for position in range(max_folds):
        start = first_start + position * validation_rows
        end = min(n, start + validation_rows)
        train_end_position = start - purge_steps - 1
        if train_end_position < min_train_rows - 1 or end <= start:
            continue
        result.append(
            {
                "name": f"{prefix}_{len(result) + 1:02d}",
                "train_start": pd.Timestamp(index[0]),
                "train_end": pd.Timestamp(index[train_end_position]),
                "validation_start": pd.Timestamp(index[start]),
                "validation_end": (
                    pd.Timestamp(index[end])
                    if end < n
                    else pd.Timestamp(index[-1]) + (index[-1] - index[-2])
                ),
                "train_positions": np.arange(0, train_end_position + 1),
                "validation_positions": np.arange(start, end),
            }
        )
    if not result:
        raise ValueError("没有生成有效时间折")
    return result


def _target_matrix(resources: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for resource in RESOURCE_NAMES:
        for horizon in horizons:
            values[f"{resource}_t+{15 * horizon}"] = resources[resource].shift(-horizon)
    return pd.DataFrame(values, index=resources.index)


def _fit_multioutput_models(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    alpha: float,
) -> dict[str, object]:
    models: dict[str, object] = {}
    for column in y.columns:
        valid = y[column].notna() & x.index.to_series().notna()
        if int(valid.sum()) < 2:
            raise ValueError(f"目标 {column} 有效训练行不足")
        models[column] = make_ridge_pipeline(alpha).fit(x.loc[valid], y.loc[valid, column])
    return models


def _predict_models(models: Mapping[str, object], x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {column: model.predict(x) for column, model in models.items()}, index=x.index
    )


def _fit_stage1_oof(
    state: pd.DataFrame,
    targets: pd.DataFrame,
    config: CascadeConfig,
) -> tuple[Stage1PredictionBundle, list[dict[str, object]]]:
    valid = targets.notna().all(axis=1)
    valid_index = state.index[valid]
    folds = _make_folds(
        valid_index,
        folds=config.inner_folds,
        purge_steps=config.purge,
        min_train_rows=config.min_train_rows,
        min_validation_rows=config.min_validation_rows,
        prefix="inner",
    )
    oof = pd.DataFrame(np.nan, index=state.index, columns=targets.columns)
    trace: list[dict[str, object]] = []
    # 折位置基于 valid_index，避免缺失标签行改变边界语义。
    for fold in folds:
        train_index = valid_index[fold["train_positions"]]
        validation_index = valid_index[fold["validation_positions"]]
        models = _fit_multioutput_models(
            state.loc[train_index], targets.loc[train_index], alpha=config.ridge_alpha
        )
        oof.loc[validation_index] = _predict_models(models, state.loc[validation_index])
        trace.append(
            {
                "fold": fold["name"],
                "train_end": str(fold["train_end"]),
                "validation_start": str(fold["validation_start"]),
                "validation_end": str(fold["validation_end"]),
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "source": "stage1_inner_held_fold_oof",
            }
        )
    oof.columns = [f"stage1_pred_{column}" for column in oof.columns]
    return (
        Stage1PredictionBundle(
            values=oof,
            source="stage1_inner_held_fold_oof",
            is_oof=True,
            resource_names=RESOURCE_NAMES,
            horizons=config.horizons,
        ),
        trace,
    )


def _fit_stage1_final(
    state: pd.DataFrame,
    targets: pd.DataFrame,
    config: CascadeConfig,
) -> dict[str, object]:
    valid = targets.notna().all(axis=1)
    return _fit_multioutput_models(state.loc[valid], targets.loc[valid], alpha=config.ridge_alpha)


def _stage1_columns(horizons: tuple[int, ...]) -> list[str]:
    return [
        f"stage1_pred_{resource}_t+{15 * horizon}"
        for resource in RESOURCE_NAMES
        for horizon in horizons
    ]


def _rename_stage1_columns(columns: Iterable[str]) -> list[str]:
    return [str(column).replace("_t+", "_tplus_") for column in columns]


def _stage2_features(
    state: pd.DataFrame,
    stage1: Stage1PredictionBundle,
    config: CascadeConfig,
) -> pd.DataFrame:
    if not stage1.is_oof and stage1.source != "stage1_final_fit":
        raise ValueError("Stage2 只接受登记过的 Stage1 OOF 或 final-fit 预测")
    if stage1.is_oof and stage1.source != "stage1_inner_held_fold_oof":
        raise ValueError("Stage1 OOF 来源不符合 nested cross-fitting 契约")
    future = stage1.values.copy()
    expected = _stage1_columns(config.horizons)
    if list(future.columns) != expected:
        raise ValueError("Stage1 预测列顺序或集合不符合契约")
    future.columns = _rename_stage1_columns(future.columns)
    return pd.concat([state, future], axis=1)


def _fit_stage2(
    stage2_x: pd.DataFrame,
    frame: pd.DataFrame,
    config: CascadeConfig,
) -> tuple[dict[str, object], dict[str, tuple[np.ndarray, np.ndarray]], int]:
    models: dict[str, object] = {}
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    valid_rows = stage2_x.notna().all(axis=1)
    for target in GENERATOR_TARGETS:
        labels = _future_targets(frame[target], config.horizons, target)
        valid = valid_rows & labels.notna().all(axis=1) & frame[target].notna()
        if int(valid.sum()) < 2:
            raise ValueError(f"Stage2 目标 {target} 有效训练行不足")
        models[target] = _fit_multioutput_models(
            stage2_x.loc[valid], labels.loc[valid], alpha=config.ridge_alpha
        )
        y = labels.loc[valid]
        bounds[target] = (
            y.quantile(config.lower_quantile).to_numpy(dtype=float),
            y.quantile(config.upper_quantile).to_numpy(dtype=float),
        )
    return models, bounds, int(valid_rows.sum())


@dataclass
class _FittedBundle:
    mapping: ResourceMapping
    resources: pd.DataFrame
    state: pd.DataFrame
    stage1_models: dict[str, object]
    stage2_models: dict[str, object]
    stage2_bounds: dict[str, tuple[np.ndarray, np.ndarray]]
    trace: CascadeFitTrace


class CausalGasCascadeForecaster:
    """Stage1 气体轨迹 + Stage2 发电目标的 nested cross-fitted 级联。"""

    version = "causal_gas_cascade_v1"

    def __init__(self, config: CascadeConfig | None = None) -> None:
        self.config = config or CascadeConfig()
        self._bundle: _FittedBundle | None = None
        self.mapping_: ResourceMapping | None = None
        self.stage1_oof_: pd.DataFrame | None = None
        self.stage2_training_features_: pd.DataFrame | None = None
        self.trace_: CascadeFitTrace | None = None

    def fit(self, frame: pd.DataFrame) -> "CausalGasCascadeForecaster":
        """在历史训练帧上拟合，并以 nested Stage1 OOF 训练 Stage2。"""

        frame = _ensure_frame(frame)
        mapping = resolve_resource_mapping(frame.columns, strict=self.config.strict_resources)
        resources = build_resource_frame(frame, mapping, strict=self.config.strict_resources)
        state = _state_features(frame, resources)
        stage1_targets = _target_matrix(resources, self.config.horizons)
        stage1_oof, inner_trace = _fit_stage1_oof(state, stage1_targets, self.config)
        # 只有完整的 held-fold OOF 行才可进入 Stage2；不允许用 actual future 补洞。
        stage2_x = _stage2_features(state, stage1_oof, self.config)
        stage2_models, bounds, stage2_rows = _fit_stage2(stage2_x, frame, self.config)
        stage1_models = _fit_stage1_final(state, stage1_targets, self.config)
        trace = CascadeFitTrace(
            stage1_inner_folds=inner_trace,
            stage1_rows=int(stage1_targets.notna().all(axis=1).sum()),
            stage2_rows=stage2_rows,
            stage2_source=stage1_oof.source,
            feature_columns=list(state.columns),
            resource_mapping=mapping.to_dict(),
        )
        self._bundle = _FittedBundle(
            mapping=mapping,
            resources=resources,
            state=state,
            stage1_models=stage1_models,
            stage2_models=stage2_models,
            stage2_bounds=bounds,
            trace=trace,
        )
        self.mapping_ = mapping
        self.stage1_oof_ = stage1_oof.values
        self.stage2_training_features_ = stage2_x
        self.trace_ = trace
        return self

    def _predict_stage1(self, frame: pd.DataFrame) -> Stage1PredictionBundle:
        if self._bundle is None:
            raise RuntimeError("级联模型尚未训练")
        resources = build_resource_frame(frame, self._bundle.mapping, strict=False)
        state = _state_features(frame, resources).reindex(columns=self._bundle.state.columns)
        values = _predict_models(self._bundle.stage1_models, state)
        values.columns = [f"stage1_pred_{column}" for column in values.columns]
        return Stage1PredictionBundle(
            values=values,
            source="stage1_final_fit",
            is_oof=False,
            resource_names=RESOURCE_NAMES,
            horizons=self.config.horizons,
        )

    def predict_stage1(self, frame: pd.DataFrame) -> pd.DataFrame:
        """返回六类资源的未来八步预测。"""

        return self._predict_stage1(_ensure_frame(frame)).values

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """在每个输入 origin 生成两目标×八步绝对预测。"""

        frame = _ensure_frame(frame)
        if self._bundle is None:
            raise RuntimeError("级联模型尚未训练")
        resources = build_resource_frame(frame, self._bundle.mapping, strict=False)
        state = _state_features(frame, resources).reindex(columns=self._bundle.state.columns)
        stage1 = self._predict_stage1(frame)
        stage2_x = _stage2_features(state, stage1, self.config)
        result: dict[str, np.ndarray] = {}
        for target in GENERATOR_TARGETS:
            values = _predict_models(self._bundle.stage2_models[target], stage2_x)
            lower, upper = self._bundle.stage2_bounds[target]
            clipped = np.clip(values.to_numpy(dtype=float), lower, upper)
            for position, horizon in enumerate(self.config.horizons):
                result[f"{target}_t+{15 * horizon}_pred"] = clipped[:, position]
        output = pd.DataFrame(result, index=frame.index)
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("Stage2 预测包含非有限值")
        return output

    def predict_at_origins(
        self,
        frame: pd.DataFrame,
        origins: Iterable[pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        """逐 origin 截断生产帧，显式保证 ``timestamp <= origin``。"""

        frame = _ensure_frame(frame)
        selected = list(frame.index if origins is None else pd.to_datetime(list(origins)))
        rows: list[pd.DataFrame] = []
        for origin in selected:
            if origin not in frame.index:
                raise KeyError(f"origin 不在生产时间轴: {origin}")
            truncated = frame.loc[frame.index <= origin]
            prediction = self.predict(truncated).iloc[[-1]].copy()
            prediction.insert(0, "origin_time", origin)
            rows.append(prediction.reset_index(drop=True))
        if not rows:
            return pd.DataFrame(columns=["origin_time", *self.prediction_columns()])
        return pd.concat(rows, ignore_index=True).set_index("origin_time")

    def prediction_columns(self) -> list[str]:
        return [
            f"{target}_t+{15 * horizon}_pred"
            for target in GENERATOR_TARGETS
            for horizon in self.config.horizons
        ]

    def build_oof(
        self,
        frame: pd.DataFrame,
        *,
        max_folds: int | None = None,
    ) -> "CascadeOOFResult":
        """外层时间 OOF；每个外层训练集内部重新执行 Stage1 inner OOF。"""

        frame = _ensure_frame(frame)
        mapping = resolve_resource_mapping(frame.columns, strict=self.config.strict_resources)
        resources = build_resource_frame(frame, mapping, strict=self.config.strict_resources)
        outer = _make_folds(
            frame.index,
            folds=self.config.outer_folds,
            purge_steps=self.config.purge,
            min_train_rows=self.config.min_train_rows,
            min_validation_rows=self.config.min_validation_rows,
            prefix="outer",
        )
        if max_folds is not None:
            outer = outer[: int(max_folds)]
        parts: list[pd.DataFrame] = []
        trace: list[dict[str, object]] = []
        for fold in outer:
            train_end = pd.Timestamp(fold["train_end"])
            validation_index = frame.index[fold["validation_positions"]]
            train_frame = frame.loc[frame.index <= train_end]
            # 外层训练集映射固定于全 schema，但数值只来自 train_frame。
            try:
                local_model = CausalGasCascadeForecaster(self.config).fit(train_frame)
            except ValueError as error:
                # 早期外折可能因历史缺失使 Stage1 有效标签不足。跳过该折而非
                # 改写模型参数；后续调用方必须把不足的 screening 折显式 STOP。
                if "时间样本不足以生成折" not in str(error):
                    raise
                trace.append(
                    {
                        "fold": fold["name"],
                        "train_end": str(train_end),
                        "validation_start": str(fold["validation_start"]),
                        "validation_end": str(fold["validation_end"]),
                        "train_rows": int(len(train_frame)),
                        "validation_rows": int(len(validation_index)),
                        "status": "SKIPPED_INSUFFICIENT_INNER_HISTORY",
                        "skip_reason": str(error),
                    }
                )
                continue
            # 批量推理也必须带上 origin 之前的历史；否则 slice 首行的 lag
            # 会被错误地当作缺失，与线上逐 origin 语义不一致。
            prediction_context = frame.loc[frame.index <= validation_index[-1]]
            predicted_context = local_model.predict(prediction_context)
            stage1_context = local_model.predict_stage1(prediction_context)
            predicted = predicted_context.loc[validation_index]
            stage1_pred = stage1_context.loc[validation_index]
            # 评分的 Stage1 actual 必须定位到 origin+h，而不是当前资源值。
            stage1_actual_wide = _target_matrix(resources, self.config.horizons).loc[validation_index]
            for resource in RESOURCE_NAMES:
                for horizon in self.config.horizons:
                    pcol = f"stage1_pred_{resource}_t+{15 * horizon}"
                    acol = f"{resource}_t+{15 * horizon}"
                    rows = pd.DataFrame(
                        {
                            "fold": fold["name"],
                            "origin_time": validation_index,
                            "train_end": train_end,
                            "stage": "stage1",
                            "target": resource,
                            "horizon": 15 * horizon,
                            "actual": stage1_actual_wide[acol].to_numpy(),
                            "prediction": stage1_pred[pcol].to_numpy(),
                        }
                    ).dropna(subset=["actual", "prediction"])
                    parts.append(rows)
            for target in GENERATOR_TARGETS:
                actual = _future_targets(frame[target], self.config.horizons, target).loc[validation_index]
                for horizon in self.config.horizons:
                    acol = f"{target}_t+{15 * horizon}"
                    pcol = f"{target}_t+{15 * horizon}_pred"
                    rows = pd.DataFrame(
                        {
                            "fold": fold["name"],
                            "origin_time": validation_index,
                            "train_end": train_end,
                            "stage": "stage2",
                            "target": target,
                            "horizon": 15 * horizon,
                            "actual": actual[acol].to_numpy(),
                            "baseline_prediction": frame.loc[
                                validation_index, target
                            ].to_numpy(),
                            "prediction": predicted[pcol].to_numpy(),
                        }
                    ).dropna(subset=["actual", "prediction"])
                    parts.append(rows)
            trace.append(
                {
                    "fold": fold["name"],
                    "train_end": str(train_end),
                    "validation_start": str(fold["validation_start"]),
                    "validation_end": str(fold["validation_end"]),
                    "nested_stage1_source": "stage1_inner_held_fold_oof",
                    "stage2_inference_source": "stage1_final_fit_on_outer_train",
                    "train_rows": int(len(train_frame)),
                    "validation_rows": int(len(validation_index)),
                    "status": "COMPLETED",
                }
            )
        if not parts:
            raise ValueError("没有生成可评分 OOF 行")
        rows = pd.concat(parts, ignore_index=True).sort_values(
            ["origin_time", "stage", "target", "horizon"], kind="stable"
        )
        report = make_cascade_report(rows)
        report["folds"] = [str(fold["name"]) for fold in outer]
        report["completed_folds"] = sorted(rows["fold"].astype(str).unique().tolist())
        report["skipped_folds"] = [
            str(item["fold"])
            for item in trace
            if item.get("status") == "SKIPPED_INSUFFICIENT_INNER_HISTORY"
        ]
        report["nested_cross_fitting"] = True
        report["resource_mapping"] = mapping.to_dict()
        return CascadeOOFResult(rows=rows.reset_index(drop=True), report=report, trace=trace)


@dataclass
class CascadeOOFResult:
    rows: pd.DataFrame
    report: dict[str, object]
    trace: list[dict[str, object]]


def make_cascade_report(rows: pd.DataFrame) -> dict[str, object]:
    """分别汇总 Stage1 future-gas 和 Stage2 generator 误差。"""

    required = {"fold", "origin_time", "train_end", "stage", "target", "horizon", "actual", "prediction"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"级联 OOF 缺少字段: {missing}")
    reports: dict[str, object] = {}
    for stage in ("stage1", "stage2"):
        part = rows.loc[rows["stage"].eq(stage)].copy()
        if part.empty:
            reports[stage] = {"rows": 0, "pooled_mape": float("nan"), "by_target": {}}
            continue
        part["ape"] = np.abs(part["actual"] - part["prediction"]) / np.maximum(
            np.abs(part["actual"]), 1e-6
        )
        reports[stage] = {
            "rows": int(len(part)),
            "pooled_mape": float(part["ape"].mean()),
            "by_target": {
                str(key): float(value)
                for key, value in part.groupby("target")["ape"].mean().items()
            },
            "by_horizon": {
                f"t+{int(key)}": float(value)
                for key, value in part.groupby("horizon")["ape"].mean().items()
            },
            "by_fold": {
                str(key): float(value)
                for key, value in part.groupby("fold")["ape"].mean().items()
            },
        }
    reports["stage1_future_gas"] = reports["stage1"]
    reports["stage2_generator"] = reports["stage2"]
    reports["pooled_mape"] = float(
        np.mean(
            [
                reports[stage]["pooled_mape"]
                for stage in ("stage1", "stage2")
                if reports[stage]["rows"]
            ]
        )
    )
    return reports


def screening_decision(
    rows: pd.DataFrame,
    *,
    baseline_column: str | None = "baseline_prediction",
    candidate_column: str = "prediction",
    first_folds: int = 5,
    min_improvement_pp: float = 0.02,
    min_wins: int = 3,
) -> dict[str, object]:
    """执行预注册 screening 门槛；失败时明确 STOP。"""

    if baseline_column is None or baseline_column not in rows.columns:
        return {
            "status": "STOP_NO_BASELINE",
            "passed": False,
            "reason": "screening 必须提供同口径 baseline prediction",
        }
    work = rows.loc[rows["stage"].eq("stage2")].copy()
    folds = list(dict.fromkeys(work["fold"].tolist()))[:first_folds]
    work = work.loc[work["fold"].isin(folds)]
    if work.empty:
        return {"status": "STOP_NO_ROWS", "passed": False, "folds": folds}
    candidate_mape = competition_mape(work["actual"], work[candidate_column])
    baseline_mape = competition_mape(work["actual"], work[baseline_column])
    by_fold: list[dict[str, object]] = []
    wins = 0
    for fold, part in work.groupby("fold", sort=False):
        candidate = competition_mape(part["actual"], part[candidate_column])
        baseline = competition_mape(part["actual"], part[baseline_column])
        win = candidate < baseline
        wins += int(win)
        by_fold.append(
            {
                "fold": str(fold),
                "candidate_mape": float(candidate),
                "baseline_mape": float(baseline),
                "improvement_pp": float((baseline - candidate) * 100.0),
                "win": bool(win),
            }
        )
    improvement_pp = float((baseline_mape - candidate_mape) * 100.0)
    passed = improvement_pp >= min_improvement_pp and wins >= min_wins
    return {
        "status": "PASS_SCREENING" if passed else "STOP_SCREENING",
        "passed": bool(passed),
        "folds": folds,
        "pooled_candidate_mape": float(candidate_mape),
        "pooled_baseline_mape": float(baseline_mape),
        "pooled_improvement_pp": improvement_pp,
        "wins": int(wins),
        "required_improvement_pp": float(min_improvement_pp),
        "required_wins": int(min_wins),
        "by_fold": by_fold,
    }


def future_perturbation_audit(
    model: CausalGasCascadeForecaster,
    frame: pd.DataFrame,
    origins: Iterable[pd.Timestamp],
) -> dict[str, object]:
    """修改各 origin 之后的生产值，验证 16 个预测逐元素不变。"""

    frame = _ensure_frame(frame)
    origins_list = list(pd.to_datetime(list(origins)))
    if not origins_list:
        raise ValueError("未来扰动审计至少需要一个 origin")
    prediction_columns = model.prediction_columns()
    cases: dict[str, dict[str, object]] = {}
    max_difference = 0.0
    all_passed = True
    for origin in origins_list:
        baseline = model.predict_at_origins(frame, [origin])
        future_mask = frame.index > origin
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        perturbations: dict[str, pd.DataFrame] = {}
        extreme = frame.copy()
        if len(numeric_columns):
            extreme.loc[future_mask, numeric_columns] = (
                extreme.loc[future_mask, numeric_columns].to_numpy(dtype=float) * 17.0
                + 123.0
            )
        perturbations["extreme"] = extreme
        shuffled = frame.copy()
        if int(future_mask.sum()) > 1 and len(numeric_columns):
            # 只打乱 origin 之后的数值而保留时间轴；整行 sample 后再排序会
            # 还原原始时间和值的配对，不能构成有效的未来数据扰动。
            future_values = shuffled.loc[future_mask, numeric_columns].to_numpy(dtype=float)
            order = np.random.default_rng(17).permutation(len(future_values))
            shuffled.loc[future_mask, numeric_columns] = future_values[order]
        perturbations["shuffle"] = shuffled
        nulled = frame.copy()
        if len(numeric_columns):
            nulled.loc[future_mask, numeric_columns] = np.nan
        perturbations["null"] = nulled
        perturbations["delete_future"] = frame.loc[~future_mask | (frame.index == origin)]
        for name, candidate_frame in perturbations.items():
            candidate = model.predict_at_origins(candidate_frame, [origin])
            difference = float(
                np.max(
                    np.abs(
                        baseline.loc[:, prediction_columns].to_numpy(dtype=float)
                        - candidate.loc[:, prediction_columns].to_numpy(dtype=float)
                    )
                )
            )
            passed = bool(difference == 0.0)
            all_passed = all_passed and passed
            max_difference = max(max_difference, difference)
            cases[f"{origin.isoformat()}::{name}"] = {
                "passed": passed,
                "changed_future_rows": int(future_mask.sum()),
                "max_abs_difference": difference,
            }
    return {
        "passed": bool(all_passed),
        "origins": [str(value) for value in origins_list],
        "prediction_columns": prediction_columns,
        "cases": cases,
        "max_abs_difference": max_difference,
    }


def write_cascade_run(
    result: CascadeOOFResult,
    run_dir: str | Path,
    *,
    config: CascadeConfig,
    mapping: ResourceMapping | None = None,
    screening: Mapping[str, object] | None = None,
) -> Path:
    """将 OOF、指标、trace 和统一报告写入独立实验目录。"""

    path = Path(run_dir)
    resolved = path.resolve()
    if "results\\best" in str(resolved).lower() or "results/best" in str(resolved).lower():
        raise ValueError("级联实验禁止写入 results/best")
    path.mkdir(parents=True, exist_ok=True)
    result.rows.to_csv(path / "oof.csv", index=False, encoding="utf-8")
    (path / "metrics.json").write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (path / "trace.json").write_text(
        json.dumps(result.trace, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    report = {
        "version": CausalGasCascadeForecaster.version,
        "config": asdict(config),
        "resource_mapping": mapping.to_dict() if mapping else result.report.get("resource_mapping", {}),
        "nested_cross_fitting": True,
        "stage2_future_input": "stage1_held_fold_oof_prediction_only",
        "screening": dict(screening or {}),
        "metrics": result.report,
        "files": {"oof": "oof.csv", "metrics": "metrics.json", "trace": "trace.json"},
    }
    (path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (path / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return path


# 兼容更短的调用名，便于实验脚本和外部 notebook 使用。
GasCascadeForecaster = CausalGasCascadeForecaster


__all__ = [
    "CascadeConfig",
    "CascadeFitTrace",
    "CascadeOOFResult",
    "CausalGasCascadeForecaster",
    "GasCascadeForecaster",
    "GENERATOR_TARGETS",
    "HORIZONS",
    "RESOURCE_NAMES",
    "ResourceMapping",
    "Stage1PredictionBundle",
    "build_resource_frame",
    "future_perturbation_audit",
    "make_cascade_report",
    "resolve_resource_mapping",
    "screening_decision",
    "write_cascade_run",
]
