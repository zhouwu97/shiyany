"""基于严格 Champion OOF 的 generator_1 Rich Residual 研究分支。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer

from gas_forecast.aggressive import project_long_candidate
from gas_forecast.config import ForecastConfig
from gas_forecast.features import PriceSchedule, build_causal_features
from gas_forecast.research import compare_research_candidate, select_research_folds
from gas_forecast.research_models import apply_capacity_projection


RICH_FEATURE_GROUPS = frozenset({"quantile", "ramp", "gas"})
RICH_FEATURE_PROFILES = frozenset({"all", "long_horizon"})
DEFAULT_BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
CHAMPION_PREDICTION_FEATURE = "feat_champion_prediction"
LONG_HORIZON_MAX_FEATURES = 250
LONG_HORIZON_ABLATION_GROUP_ORDER: Final[tuple[str, ...]] = (
    "generation_dynamics",
    "gas_production",
    "gas_consumption",
    "holder_balance",
    "quantile_ramp_state",
    "branch_prediction_disagreement",
    "time_price",
)
LONG_HORIZON_ABLATION_GROUPS: Final[frozenset[str]] = frozenset(
    LONG_HORIZON_ABLATION_GROUP_ORDER
)


# 长步长专模只保留与机组爬坡、煤气供需及已知电价直接相关的因果字段。
_LONG_HORIZON_RAW_COLUMNS = (
    "generator_1",
    "generator_all",
    "generator_use_blast_furnace_gas",
    "generator_use_coke_gas",
    "generator_use_converter_gas",
    "blast_furnace_gas_holder_2",
    "blast_furnace_1",
    "blast_furnace_2",
    "blast_furnace_4",
    "blast_furnace_5",
    "coke_oven_1",
    "converter_1",
    "air_heater_1",
    "air_heater_2",
    "air_heater_4",
    "air_heater_5",
    "blast_furnace_user1",
    "blast_furnace_user2",
    "blast_furnace_user3",
    "blast_furnace_user4",
    "converter_user1",
    "converter_user2",
    "into_gas_mixed_coke",
    "into_gas_mixed_blast_furnace",
    "into_gas_mixed_converter",
)

_LONG_HORIZON_GAS_PRODUCTION_RAW_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "blast_furnace_1",
        "blast_furnace_2",
        "blast_furnace_4",
        "blast_furnace_5",
        "coke_oven_1",
        "converter_1",
        "into_gas_mixed_coke",
        "into_gas_mixed_blast_furnace",
        "into_gas_mixed_converter",
    }
)
_LONG_HORIZON_GAS_CONSUMPTION_DERIVED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "feat_air_heater_use",
        "feat_bf_user_use",
        "feat_generator_gas_total",
        "feat_rich_gas_priority_demand",
        "feat_rich_gas_available_for_generation",
        "feat_rich_gas_production_demand_ratio",
        "feat_rich_gas_available_production_ratio",
    }
)
_LONG_HORIZON_GAS_PRODUCTION_DERIVED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "feat_bf_production",
        "feat_rich_gas_total_production",
        "feat_rich_gas_blast_production_share",
        "feat_rich_gas_coke_production_share",
        "feat_rich_gas_converter_production_share",
        "feat_gas_mix_entropy",
        "feat_dominant_gas_type",
        "feat_dominant_gas_changed",
        "feat_steps_since_gas_switch",
        "feat_coke_down_blast_up",
        "feat_coke_up_blast_down",
        "feat_converter_down_blast_up",
    }
)


def _normalize_long_horizon_ablation_groups(groups: Iterable[str]) -> frozenset[str]:
    """规范化 A56 的固定长步长特征组名，拒绝未登记的排除项。"""

    normalized = frozenset(str(group).strip().lower() for group in groups if str(group).strip())
    invalid = sorted(normalized.difference(LONG_HORIZON_ABLATION_GROUPS))
    if invalid:
        raise ValueError(f"未知 long_horizon 消融特征组: {invalid}")
    return normalized


def long_horizon_feature_group(column: str) -> str:
    """将 A51 静态白名单字段归入唯一预注册 A56 消融组。"""

    if column == CHAMPION_PREDICTION_FEATURE:
        return "branch_prediction_disagreement"
    if column.startswith("feat_target_price_"):
        return "time_price"
    if column.startswith("feat_rich_quantile_") or column.startswith("feat_rich_ramp_"):
        return "quantile_ramp_state"
    if (
        column == "blast_furnace_gas_holder_2"
        or column.startswith("feat_blast_furnace_gas_holder_2_")
        or column.startswith("feat_gas_holder")
        or column.startswith("feat_rich_gas_holder")
        or column.startswith("feat_blast_balance")
        or column.startswith("feat_coke_balance")
        or column.startswith("feat_converter_balance")
        or column == "feat_bf_surplus_proxy"
    ):
        return "holder_balance"
    if (
        column.startswith("generator_use_")
        or column.startswith("feat_generator_use_")
        or column.startswith("air_heater_")
        or column.startswith("blast_furnace_user")
        or column.startswith("converter_user")
        or column in _LONG_HORIZON_GAS_CONSUMPTION_DERIVED_COLUMNS
    ):
        return "gas_consumption"
    if (
        column in {"generator_1", "generator_all"}
        or column.startswith("feat_generator_1_")
        or column.startswith("feat_generator_all_")
        or column.startswith("feat_generator_rest")
    ):
        return "generation_dynamics"
    if (
        column in _LONG_HORIZON_GAS_PRODUCTION_RAW_COLUMNS
        or column in _LONG_HORIZON_GAS_PRODUCTION_DERIVED_COLUMNS
    ):
        return "gas_production"
    raise ValueError(f"A51 long_horizon 字段未归入消融组: {column}")


def _validate_feature_profile(profile: str) -> str:
    """验证并规范化 RichResidual 的特征白名单配置。"""

    normalized = str(profile).strip().lower()
    if normalized not in RICH_FEATURE_PROFILES:
        raise ValueError(f"未知 RichResidual 特征 profile: {profile}")
    return normalized


def _history_feature_names(
    prefix: str,
    *,
    lags: tuple[int, ...],
    diffs: tuple[int, ...],
    windows: tuple[int, ...],
    include_slope: bool = True,
) -> list[str]:
    """返回指定信号的固定历史统计字段名，不依赖列顺序或数据值。"""

    names = [f"feat_{prefix}_lag_{lag}" for lag in lags]
    names.extend(f"feat_{prefix}_diff_{lag}" for lag in diffs)
    for window in windows:
        names.extend(
            (
                f"feat_{prefix}_mean_{window}",
                f"feat_{prefix}_std_{window}",
                f"feat_{prefix}_vs_mean_{window}",
            )
        )
    if include_slope:
        names.extend((f"feat_{prefix}_slope_4", f"feat_{prefix}_slope_8"))
    return names


def _long_horizon_feature_candidates() -> list[str]:
    """构造 A51 的显式因果特征白名单，顺序即进入模型的优先级。"""

    candidates = list(_LONG_HORIZON_RAW_COLUMNS)
    for prefix in ("generator_1", "generator_all", "generator_rest"):
        candidates.extend(
            _history_feature_names(
                prefix,
                lags=(1, 2, 4, 8, 16, 32, 96),
                diffs=(1, 2, 4, 8, 16),
                windows=(4, 16, 96),
            )
        )
    candidates.extend(
        _history_feature_names(
            "blast_furnace_gas_holder_2",
            lags=(1, 2, 4, 8, 16, 32, 96),
            diffs=(1, 2, 4, 8, 16),
            windows=(4, 16, 96),
        )
    )
    for prefix in (
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
    ):
        candidates.extend(
            _history_feature_names(
                prefix,
                lags=(1, 4, 16),
                diffs=(1, 4, 16),
                windows=(16,),
            )
        )
    candidates.extend(
        (
            "feat_generator_rest",
            "feat_bf_production",
            "feat_air_heater_use",
            "feat_bf_user_use",
            "feat_bf_surplus_proxy",
            "feat_generator_gas_total",
            "feat_rich_gas_total_production",
            "feat_rich_gas_priority_demand",
            "feat_rich_gas_available_for_generation",
            "feat_rich_gas_production_demand_ratio",
            "feat_rich_gas_available_production_ratio",
            "feat_rich_gas_blast_production_share",
            "feat_rich_gas_coke_production_share",
            "feat_rich_gas_converter_production_share",
            "feat_rich_gas_holder_buffer",
            "feat_rich_gas_holder_to_production_ratio",
            "feat_rich_gas_holder_to_available_ratio",
            "feat_generator_use_blast_furnace_gas_share",
            "feat_generator_use_coke_gas_share",
            "feat_generator_use_converter_gas_share",
            "feat_gas_mix_entropy",
            "feat_dominant_gas_type",
            "feat_dominant_gas_changed",
            "feat_steps_since_gas_switch",
            "feat_coke_down_blast_up",
            "feat_coke_up_blast_down",
            "feat_converter_down_blast_up",
        )
    )
    for prefix in ("blast_balance", "coke_balance", "converter_balance"):
        candidates.append(f"feat_{prefix}")
        candidates.extend(f"feat_{prefix}_diff_{lag}" for lag in (1, 4, 16))
        candidates.extend(f"feat_{prefix}_mean_{window}" for window in (4, 16, 96))
        candidates.extend(f"feat_{prefix}_slope_{window}" for window in (4, 8))
    for prefix in ("generator_1", "generator_all", "generator_rest", "gas_holder"):
        for window in (32, 96):
            candidates.extend(
                (
                    f"feat_rich_quantile_{prefix}_q10_{window}",
                    f"feat_rich_quantile_{prefix}_q90_{window}",
                )
            )
    for prefix in ("generator_1", "generator_all", "generator_rest"):
        candidates.extend(
            (
                f"feat_rich_ramp_{prefix}_rate",
                f"feat_rich_ramp_{prefix}_threshold",
                f"feat_rich_ramp_{prefix}_up",
                f"feat_rich_ramp_{prefix}_down",
                f"feat_rich_ramp_{prefix}_stable",
                f"feat_rich_ramp_{prefix}_time_since_event",
                f"feat_rich_ramp_{prefix}_up_run_length",
                f"feat_rich_ramp_{prefix}_down_run_length",
            )
        )
    candidates.extend(
        (
            "feat_target_price_tplus_75",
            "feat_target_price_tplus_90",
            "feat_target_price_tplus_105",
            "feat_target_price_tplus_120",
        )
    )
    return list(dict.fromkeys(candidates))


def select_rich_feature_columns(
    features: pd.DataFrame,
    profile: str,
    *,
    exclude_long_feature_groups: Iterable[str] = (),
) -> list[str]:
    """按固定 profile 选列，并只在 A56 中排除预注册 long-horizon 特征组。"""

    normalized = _validate_feature_profile(profile)
    excluded = _normalize_long_horizon_ablation_groups(exclude_long_feature_groups)
    if excluded and normalized != "long_horizon":
        raise ValueError("long_horizon 特征组消融只能用于 long_horizon profile")
    numeric_columns = list(features.select_dtypes(include=[np.number]).columns)
    if normalized == "all":
        return numeric_columns
    available = set(numeric_columns)
    selected = [column for column in _long_horizon_feature_candidates() if column in available]
    if not selected:
        raise ValueError("long_horizon profile 没有匹配到可用因果特征")
    if len(selected) > LONG_HORIZON_MAX_FEATURES:
        raise RuntimeError("long_horizon profile 超出预注册的 250 个特征上限")
    if excluded:
        selected = [
            column
            for column in selected
            if long_horizon_feature_group(column) not in excluded
        ]
        if not selected:
            raise ValueError("A56 排除后没有可用的 long_horizon 因果特征")
    return selected


def long_horizon_feature_group_counts(columns: Iterable[str]) -> dict[str, int]:
    """统计 A51/A56 选中字段的固定分组覆盖，供运行收据独立核验。"""

    counts = {group: 0 for group in LONG_HORIZON_ABLATION_GROUP_ORDER}
    for column in columns:
        counts[long_horizon_feature_group(column)] += 1
    return counts


@dataclass(frozen=True)
class RichResidualSpec:
    """一个冻结的 RichResidual 特征组和残差模型容量配置。"""

    name: str
    target: str = "generator_1"
    feature_groups: frozenset[str] = frozenset()
    feature_profile: str = "all"
    active_horizons: tuple[int, ...] | None = None
    include_champion_prediction: bool = False
    exclude_long_feature_groups: frozenset[str] = frozenset()
    min_train_rows: int = 256
    n_estimators: int | None = None
    blend_weights: tuple[float, ...] = DEFAULT_BLEND_WEIGHTS

    def __post_init__(self) -> None:
        target = str(self.target).strip()
        if not target:
            raise ValueError("RichResidual target 不能为空")
        object.__setattr__(self, "target", target)
        invalid = sorted(self.feature_groups.difference(RICH_FEATURE_GROUPS))
        if invalid:
            raise ValueError(f"未知 Rich 特征组: {invalid}")
        profile = _validate_feature_profile(self.feature_profile)
        object.__setattr__(self, "feature_profile", profile)
        excluded_groups = _normalize_long_horizon_ablation_groups(
            self.exclude_long_feature_groups
        )
        if excluded_groups and profile != "long_horizon":
            raise ValueError("long_horizon 特征组消融只能用于 long_horizon profile")
        object.__setattr__(self, "exclude_long_feature_groups", excluded_groups)
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("RichResidual 名称只能包含字母、数字和下划线")
        if self.active_horizons is not None:
            horizons = tuple(self.active_horizons)
            if not horizons:
                raise ValueError("active_horizons 不能为空；使用 None 表示全部步长")
            if any(not isinstance(horizon, (int, np.integer)) for horizon in horizons):
                raise ValueError("active_horizons 必须为整数分钟")
            normalized_horizons = tuple(sorted(int(horizon) for horizon in horizons))
            if any(horizon <= 0 or horizon % 15 != 0 for horizon in normalized_horizons):
                raise ValueError("active_horizons 必须为正的 15 分钟整数倍")
            if len(set(normalized_horizons)) != len(normalized_horizons):
                raise ValueError("active_horizons 不能重复")
            object.__setattr__(self, "active_horizons", normalized_horizons)
        if self.min_train_rows < 16:
            raise ValueError("RichResidual 每个步长至少需要 16 条历史 OOF 行")
        if self.n_estimators is not None and self.n_estimators < 1:
            raise ValueError("n_estimators 必须为正数")
        if not self.blend_weights:
            raise ValueError("RichResidual 至少需要一个固定 blend 权重")
        if any(weight <= 0.0 or weight > 1.0 for weight in self.blend_weights):
            raise ValueError("RichResidual blend 权重必须位于 (0, 1]")
        if len(set(self.blend_weights)) != len(self.blend_weights):
            raise ValueError("RichResidual blend 权重不能重复")


def _residual_target(spec: RichResidualSpec) -> str:
    """兼容旧版 joblib spec，并统一读取本次残差修正的唯一目标。"""

    target = str(getattr(spec, "target", "generator_1")).strip()
    if not target:
        raise ValueError("RichResidual target 不能为空")
    return target


@dataclass
class RichResidualHorizonState:
    """单个预测步长的严格 OOF 残差模型和训练期裁剪边界。"""

    imputer: SimpleImputer
    model: LGBMRegressor
    residual_lower: float
    residual_upper: float
    training_rows: int


@dataclass(frozen=True)
class RichResidualOOFResult:
    """RichResidual 的严格外层 OOF、报告和训练期特征配置。"""

    rows: pd.DataFrame
    report: dict[str, object]
    feature_config: ForecastConfig


def rich_feature_config(
    config: ForecastConfig,
    groups: Iterable[str],
    *,
    feature_profile: str = "all",
) -> ForecastConfig:
    """只按登记的组启用 Rich 特征，不隐式改变既有 Champion 开关。"""

    selected = frozenset(groups)
    invalid = sorted(selected.difference(RICH_FEATURE_GROUPS))
    if invalid:
        raise ValueError(f"未知 Rich 特征组: {invalid}")
    _validate_feature_profile(feature_profile)
    return replace(
        config,
        feature=replace(
            config.feature,
            enable_rich_quantile_features="quantile" in selected,
            enable_rich_ramp_state_features="ramp" in selected,
            enable_rich_gas_resource_features="gas" in selected,
        ),
    )


def _active_horizons(config: ForecastConfig, spec: RichResidualSpec) -> tuple[int, ...]:
    """将 A51 的分钟步长约束映射到当前 Champion 的已配置步长。"""

    configured = tuple(15 * horizon for horizon in config.feature.horizons)
    requested = getattr(spec, "active_horizons", None)
    if requested is None:
        return configured
    missing = sorted(set(requested).difference(configured))
    if missing:
        raise ValueError(f"active_horizons 不在冻结 Champion 配置中: {missing}")
    return tuple(requested)


def _mape_weights(actual: np.ndarray) -> np.ndarray:
    """以绝对预测量倒数近似 MAPE，避免极小负荷主导树模型。"""

    magnitude = np.abs(np.asarray(actual, dtype=float))
    lower = max(float(np.nanquantile(magnitude, 0.05)), 1e-6)
    upper = max(float(np.nanquantile(magnitude, 0.95)), lower)
    weights = 1.0 / np.clip(magnitude, lower, upper)
    return weights / weights.mean()


def _finite_feature_matrix(features: pd.DataFrame, origins: pd.Series) -> pd.DataFrame:
    """按 origin 对齐数值特征，并把无穷值交给训练期插补器处理。"""

    aligned = features.reindex(pd.DatetimeIndex(origins)).copy()
    if aligned.empty:
        return aligned
    numeric = aligned.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("RichResidual 没有可用数值特征")
    return numeric.replace([np.inf, -np.inf], np.nan)


def _validate_oof_rows(rows: pd.DataFrame, baseline_column: str) -> pd.DataFrame:
    """验证外部 Champion OOF 的键、数值和基线预测契约。"""

    required = {
        "fold",
        "origin_time",
        "target",
        "horizon",
        "actual",
        "persistence_pred",
        baseline_column,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Champion OOF 缺少字段: {missing}")
    output = rows.copy()
    output["origin_time"] = pd.to_datetime(output["origin_time"], errors="coerce")
    if output["origin_time"].isna().any():
        raise ValueError("Champion OOF 含非法 origin_time")
    keys = ["fold", "origin_time", "target", "horizon"]
    if output.duplicated(keys).any():
        raise ValueError("Champion OOF 存在重复 fold×origin×target×horizon")
    numeric = output.loc[:, ["actual", "persistence_pred", baseline_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Champion OOF 的真实值或基线预测含缺失/非有限数")
    output.loc[:, numeric.columns] = numeric
    output["horizon"] = pd.to_numeric(output["horizon"], errors="raise").astype(int)
    return output


class RichResidualCorrector:
    """只从已完成的 Champion OOF 学习一个冻结目标的残差。"""

    version = "generator1_rich_residual_corrector"

    def __init__(self, config: ForecastConfig, spec: RichResidualSpec) -> None:
        self.config = config
        self.spec = spec
        self.states_: dict[int, RichResidualHorizonState] = {}
        self.feature_columns_: list[str] = []
        self.static_feature_columns_: list[str] = []
        self.include_champion_prediction_ = bool(
            getattr(spec, "include_champion_prediction", False)
        )

    def _make_model(self, horizon: int) -> LGBMRegressor:
        estimators = self.spec.n_estimators or self.config.model.lgb_n_estimators
        return LGBMRegressor(
            objective=self.config.model.lgb_objective,
            n_estimators=estimators,
            learning_rate=self.config.model.lgb_learning_rate,
            num_leaves=self.config.model.lgb_num_leaves,
            max_depth=self.config.model.lgb_max_depth,
            min_child_samples=self.config.model.lgb_min_child_samples,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=self.config.model.random_state + horizon,
            n_jobs=self.config.model.tree_threads_per_worker,
            verbosity=-1,
        )

    def _model_matrix(
        self,
        features: pd.DataFrame,
        origins: pd.Series,
        *,
        baseline: np.ndarray | pd.Series | None,
    ) -> pd.DataFrame:
        """组装训练和推理共用的因果特征矩阵，并兼容旧版 joblib。"""

        static_columns = getattr(self, "static_feature_columns_", None)
        if not static_columns:
            static_columns = [
                column
                for column in self.feature_columns_
                if column != CHAMPION_PREDICTION_FEATURE
            ]
        matrix = _finite_feature_matrix(features, origins).reindex(columns=static_columns)
        include_champion = bool(getattr(self, "include_champion_prediction_", False))
        if include_champion:
            if baseline is None:
                raise ValueError("启用 Champion 预测特征时必须提供同一预测步长的基线")
            values = np.asarray(baseline, dtype=float)
            if len(values) != len(matrix):
                raise ValueError("Champion 预测特征长度与 origin 行数不一致")
            matrix[CHAMPION_PREDICTION_FEATURE] = values
        return matrix.reindex(columns=self.feature_columns_)

    def fit(
        self,
        features: pd.DataFrame,
        oof_rows: pd.DataFrame,
        *,
        baseline_column: str,
    ) -> "RichResidualCorrector":
        """用历史折的同折 Champion 预测构造残差标签。"""

        rows = _validate_oof_rows(oof_rows, baseline_column)
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("RichResidual 特征必须使用 DatetimeIndex")
        target = _residual_target(self.spec)
        if target not in self.config.targets:
            raise ValueError(f"RichResidual target 不在冻结配置中: {target}")
        target_rows = rows.loc[rows["target"].eq(target)].copy()
        if target_rows.empty:
            raise ValueError(f"Champion OOF 没有 {target} 行")
        self.states_.clear()
        profile = _validate_feature_profile(getattr(self.spec, "feature_profile", "all"))
        excluded_groups = _normalize_long_horizon_ablation_groups(
            getattr(self.spec, "exclude_long_feature_groups", frozenset())
        )
        self.static_feature_columns_ = select_rich_feature_columns(
            features,
            profile,
            exclude_long_feature_groups=excluded_groups,
        )
        self.feature_columns_ = list(self.static_feature_columns_)
        self.include_champion_prediction_ = bool(
            getattr(self.spec, "include_champion_prediction", False)
            and "branch_prediction_disagreement" not in excluded_groups
        )
        if self.include_champion_prediction_:
            self.feature_columns_.append(CHAMPION_PREDICTION_FEATURE)
        if profile == "long_horizon" and len(self.feature_columns_) > LONG_HORIZON_MAX_FEATURES:
            raise RuntimeError("long_horizon profile 含 Champion 特征后超出 250 个特征上限")
        if not self.static_feature_columns_:
            raise ValueError("RichResidual 没有可用数值特征")
        for horizon in _active_horizons(self.config, self.spec):
            part = target_rows.loc[target_rows["horizon"].eq(horizon)].sort_values(
                "origin_time"
            )
            if part.empty:
                continue
            baseline = part[baseline_column].to_numpy(dtype=float)
            matrix = self._model_matrix(
                features,
                part["origin_time"],
                baseline=baseline,
            )
            actual = part["actual"].to_numpy(dtype=float)
            residual = actual - baseline
            valid = np.isfinite(residual) & np.isfinite(matrix.to_numpy(dtype=float)).any(axis=1)
            if int(valid.sum()) < self.spec.min_train_rows:
                continue
            x = matrix.loc[valid]
            y = residual[valid]
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            transformed = imputer.fit_transform(x)
            model = self._make_model(horizon)
            weights = _mape_weights(actual[valid]) if self.config.model.lgb_use_mape_weights else None
            model.fit(transformed, y, sample_weight=weights)
            self.states_[horizon] = RichResidualHorizonState(
                imputer=imputer,
                model=model,
                residual_lower=float(np.quantile(y, self.config.model.lower_quantile)),
                residual_upper=float(np.quantile(y, self.config.model.upper_quantile)),
                training_rows=int(valid.sum()),
            )
        return self

    def predict_long(
        self,
        features: pd.DataFrame,
        rows: pd.DataFrame,
        *,
        baseline_column: str,
    ) -> pd.Series:
        """对长表登记 target 行修正基线；未训练步长严格回退到基线。"""

        required = {"origin_time", "target", "horizon", baseline_column}
        missing = sorted(required.difference(rows.columns))
        if missing:
            raise ValueError(f"RichResidual 预测行缺少字段: {missing}")
        prediction = pd.to_numeric(rows[baseline_column], errors="coerce").astype(float).copy()
        if not np.isfinite(prediction.to_numpy()).all():
            raise ValueError("RichResidual 预测基线含非有限数")
        target_rows = rows["target"].eq(_residual_target(self.spec))
        for horizon, state in self.states_.items():
            mask = target_rows & rows["horizon"].eq(horizon)
            if not mask.any():
                continue
            part = rows.loc[mask]
            matrix = self._model_matrix(
                features,
                pd.to_datetime(part["origin_time"]),
                baseline=part[baseline_column].to_numpy(dtype=float),
            )
            correction = state.model.predict(state.imputer.transform(matrix))
            correction = np.clip(correction, state.residual_lower, state.residual_upper)
            prediction.loc[mask] = part[baseline_column].to_numpy(dtype=float) + correction
        return prediction

    def predict_wide(self, features: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
        """在生产宽表上只修正登记 target，其他目标留给容量投影协调。"""

        output = baseline.copy()
        target = _residual_target(self.spec)
        for horizon, state in self.states_.items():
            column = f"{target}_t+{horizon}_pred"
            if column not in output:
                raise ValueError(f"生产基线缺少预测列: {column}")
            matrix = self._model_matrix(
                features,
                pd.Series(features.index),
                baseline=baseline[column].to_numpy(dtype=float),
            )
            correction = state.model.predict(state.imputer.transform(matrix))
            correction = np.clip(correction, state.residual_lower, state.residual_upper)
            output[column] = output[column].to_numpy(dtype=float) + correction
        return output


class RichResidualAggressiveForecaster:
    """将已冻结 target 的 RichResidual 以固定小权重叠加到 Aggressive Champion。"""

    version = "generator1_rich_residual"

    def __init__(
        self,
        base_model: object,
        corrector: RichResidualCorrector,
        *,
        blend_weight: float,
    ) -> None:
        if blend_weight <= 0.0 or blend_weight > 1.0:
            raise ValueError("RichResidual 生产 blend 权重必须位于 (0, 1]")
        if not hasattr(base_model, "predict") or not hasattr(base_model, "config"):
            raise TypeError("RichResidual 需要具有 config 和 predict 的冻结基线模型")
        if tuple(base_model.config.feature.horizons) != tuple(corrector.config.feature.horizons):
            raise ValueError("RichResidual 与冻结基线的 horizon 配置不一致")
        self.base_model = base_model
        self.corrector = corrector
        self.blend_weight = blend_weight
        self.config = corrector.config

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        baseline = self.base_model.predict(features, current)
        corrected = self.corrector.predict_wide(features, baseline)
        output = baseline.copy()
        target = _residual_target(self.corrector.spec)
        for horizon in self.config.feature.horizons:
            column = f"{target}_t+{15 * horizon}_pred"
            output[column] = (1.0 - self.blend_weight) * baseline[column] + self.blend_weight * corrected[
                column
            ]
        return apply_capacity_projection(output)


def fit_full_rich_residual_corrector(
    frame: pd.DataFrame,
    champion_oof: pd.DataFrame,
    *,
    config: ForecastConfig,
    spec: RichResidualSpec,
    baseline_column: str = "aggressive_r75_lgb20_pred",
    price_schedule: PriceSchedule | None = None,
    allow_confirmed_blind_oof: bool = False,
) -> RichResidualCorrector:
    """拟合可部署残差校正器，默认不使用 blind 标签。

    ``allow_confirmed_blind_oof`` 仅用于已经完成一次正式 blind 验收后的全量重训。
    调用方必须显式声明该授权，避免研究阶段将 blind 标签无意带入模型。
    """

    rich_config = rich_feature_config(
        config,
        spec.feature_groups,
        feature_profile=getattr(spec, "feature_profile", "all"),
    )
    features = build_causal_features(frame, rich_config.feature, price_schedule)
    rows = _validate_oof_rows(champion_oof, baseline_column)
    history = rows.copy() if allow_confirmed_blind_oof else rows.loc[rows["fold"].ne("blind")].copy()
    return RichResidualCorrector(rich_config, spec).fit(
        features,
        history,
        baseline_column=baseline_column,
    )


def build_rich_residual_oof(
    frame: pd.DataFrame,
    champion_oof: pd.DataFrame,
    *,
    config: ForecastConfig,
    spec: RichResidualSpec,
    baseline_column: str = "aggressive_r75_lgb20_pred",
    scope: str = "development",
    price_schedule: PriceSchedule | None = None,
) -> RichResidualOOFResult:
    """以历史 Champion OOF 为唯一残差标签来源，生成严格外层折预测。"""

    if scope not in {"screening", "development", "final"}:
        raise ValueError("RichResidual scope 只能是 screening、development 或 final")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("RichResidual 输入数据必须使用 DatetimeIndex")
    rich_config = rich_feature_config(
        config,
        spec.feature_groups,
        feature_profile=getattr(spec, "feature_profile", "all"),
    )
    target = _residual_target(spec)
    if target not in rich_config.targets:
        raise ValueError(f"RichResidual target 不在冻结配置中: {target}")
    active_horizons = _active_horizons(rich_config, spec)
    rows = _validate_oof_rows(champion_oof, baseline_column)
    folds = select_research_folds(frame.index, rich_config, scope=scope)
    if not folds:
        raise ValueError("RichResidual 没有可用外层折")
    fold_names = [fold.name for fold in folds]
    missing_folds = sorted(set(fold_names).difference(rows["fold"].astype(str)))
    if missing_folds:
        raise ValueError(f"Champion OOF 未覆盖 RichResidual 外层折: {missing_folds}")
    features = build_causal_features(frame, rich_config.feature, price_schedule)
    excluded_groups = _normalize_long_horizon_ablation_groups(
        getattr(spec, "exclude_long_feature_groups", frozenset())
    )
    selected_feature_columns = select_rich_feature_columns(
        features,
        getattr(spec, "feature_profile", "all"),
        exclude_long_feature_groups=excluded_groups,
    )
    champion_prediction_feature = bool(
        getattr(spec, "include_champion_prediction", False)
        and "branch_prediction_disagreement" not in excluded_groups
    )
    if champion_prediction_feature:
        selected_feature_columns.append(CHAMPION_PREDICTION_FEATURE)
    if (
        getattr(spec, "feature_profile", "all") == "long_horizon"
        and len(selected_feature_columns) > LONG_HORIZON_MAX_FEATURES
    ):
        raise RuntimeError("long_horizon profile 含 Champion 特征后超出 250 个特征上限")
    validation = rows.loc[rows["fold"].astype(str).isin(fold_names)].copy()
    validation = validation.sort_values(["origin_time", "target", "horizon"]).reset_index(drop=True)
    raw_column = f"{spec.name}_residual_raw_pred"
    projected_column = f"{spec.name}_residual_pred"
    pieces: list[pd.DataFrame] = []
    fold_training_rows: dict[str, int] = {}
    trained_horizons: dict[str, list[int]] = {}
    history_source = rows if scope == "final" else rows.loc[rows["fold"].ne("blind")]
    for fold in folds:
        held = validation.loc[validation["fold"].eq(fold.name)].copy()
        history = history_source.loc[
            history_source["target"].eq(target)
            & history_source["origin_time"].le(fold.train_end)
        ].copy()
        fold_training_rows[fold.name] = int(len(history))
        if history.empty:
            # 第一个可评分折没有已完成的 Champion OOF，只能严格回退到基线。
            held[raw_column] = held[baseline_column].to_numpy(dtype=float)
            trained_horizons[fold.name] = []
        else:
            corrector = RichResidualCorrector(rich_config, spec).fit(
                features,
                history,
                baseline_column=baseline_column,
            )
            held[raw_column] = corrector.predict_long(
                features,
                held,
                baseline_column=baseline_column,
            )
            trained_horizons[fold.name] = sorted(corrector.states_)
        pieces.append(held)
    raw = pd.concat(pieces, ignore_index=True)
    projected = project_long_candidate(raw, raw_column, output_column=projected_column)
    candidate_columns = [projected_column]
    for weight in spec.blend_weights:
        label = f"{int(round(weight * 100)):02d}"
        raw_blend_column = f"{spec.name}_blend_{label}_raw_pred"
        blend_column = f"{spec.name}_blend_{label}_pred"
        projected[raw_blend_column] = (
            (1.0 - weight) * projected[baseline_column] + weight * projected[projected_column]
        )
        projected = project_long_candidate(projected, raw_blend_column, output_column=blend_column)
        candidate_columns.append(blend_column)
    reports = {
        column: compare_research_candidate(projected, column, baseline_column, scope=scope)
        for column in candidate_columns
    }
    return RichResidualOOFResult(
        rows=projected,
        report={
            "scope": scope,
            "baseline_column": baseline_column,
            "target_scope": target,
            "feature_groups": sorted(spec.feature_groups),
            "feature_profile": getattr(spec, "feature_profile", "all"),
            "excluded_long_feature_groups": sorted(excluded_groups),
            "folds": fold_names,
            "feature_columns": int(len(selected_feature_columns)),
            "selected_feature_columns": selected_feature_columns,
            "long_horizon_feature_group_counts": (
                long_horizon_feature_group_counts(selected_feature_columns)
                if getattr(spec, "feature_profile", "all") == "long_horizon"
                else None
            ),
            "active_horizons": list(active_horizons),
            "champion_prediction_feature": champion_prediction_feature,
            "fold_training_rows": fold_training_rows,
            "trained_horizons": trained_horizons,
            "models": reports,
            "strict_oof_contract": {
                "residual_target": "actual - same_fold_champion_prediction",
                "history_rule": "origin_time <= outer_fold.train_end",
                "blind_labels_used": scope == "final",
                "target_scope": target,
                "active_horizons": list(active_horizons),
                "feature_profile": getattr(spec, "feature_profile", "all"),
                "excluded_long_feature_groups": sorted(excluded_groups),
                "champion_prediction_is_production_available": champion_prediction_feature,
            },
        },
        feature_config=rich_config,
    )
