"""按目标路由的研究候选模型。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gas_forecast.config import ForecastConfig
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster
from gas_forecast.model_horizon import HorizonSpecificRidgeForecaster
from gas_forecast.model_v1 import RidgeDeltaForecaster, make_ridge_pipeline
from gas_forecast.splits import make_inner_folds
from gas_forecast.targets import target_columns


def select_generator1_features(
    features: pd.DataFrame,
    profile: str,
    config: ForecastConfig | None = None,
) -> pd.DataFrame:
    """选择 ``generator_1`` 低风险核心特征，避免无关字段扩大线性模型方差。"""

    if profile == "all":
        return features
    if profile != "core":
        raise ValueError(f"不支持的 generator_1 特征配置: {profile}")

    direct = {
        "generator_1",
        "feat_generator_rest",
        "feat_generator_gas_total",
        "feat_bf_surplus_proxy",
        "feat_hour",
        "feat_minute_slot",
        "feat_day_of_week",
        "feat_month",
        "feat_time_sin",
        "feat_time_cos",
        "feat_is_weekend",
    }
    prefixes = (
        "feat_generator_1_lag_",
        "feat_generator_1_diff_",
        "feat_generator_1_slope_",
        "feat_generator_1_ramp_",
        "feat_generator_1_acceleration",
        "feat_generator_1_ewma_",
        "feat_generator_1_ramp_volatility_",
        "feat_generator_1_ramp_range_",
        "feat_generator_1_ramp_q",
        "feat_relation_",
        "feat_generator_1_aligned_",
        "feat_generator_1_same_slot_",
        "feat_generator_rest_lag_",
        "feat_generator_rest_diff_",
        "feat_generator_rest_slope_",
        "feat_generator_use_",
        "feat_target_price_",
        "feat_price_",
        "feat_generator1_price_",
        "feat_generator1_slope_price_",
        "feat_generator_gas_total_price_",
        "feat_rich_",
        "feat_slot_",
        "feat_weekday_",
        "feat_day_fourier_",
        "feat_week_fourier_",
        "feat_dynamic_",
    )
    columns = [
        column
        for column in features.columns
        if column in direct
        or column.startswith(prefixes)
        or column.startswith("generator_use_")
        or "gas_holder" in column
    ]
    if config is not None:
        feature = config.feature

        def enabled(column: str) -> bool:
            if column.startswith("feat_rich_quantile_"):
                return feature.enable_rich_quantile_features
            if column.startswith("feat_rich_ramp_"):
                return feature.enable_rich_ramp_state_features
            if column.startswith("feat_rich_gas_"):
                return feature.enable_rich_gas_resource_features
            if not feature.enable_target_aligned_features and "_aligned_" in column:
                return False
            if not feature.enable_long_cycle_features and "_same_slot_" in column:
                return False
            if not feature.enable_slot_one_hot and (
                column.startswith("feat_slot_")
                or column.startswith("feat_weekday_")
                or column == "feat_is_weekend"
            ):
                return False
            if not feature.enable_time_fourier and "_fourier_" in column:
                return False
            if not feature.enable_price_delta_features and (
                "_price_delta_" in column
                or "_price_ratio_" in column
                or "_price_direction_" in column
                or column.startswith("feat_next_2h_price_")
            ):
                return False
            if not feature.enable_price_interactions and (
                column.startswith("feat_generator1_price_")
                or column.startswith("feat_generator1_slope_price_")
                or column.startswith("feat_generator_gas_total_price_")
                or column.startswith("feat_gas_holder_price_")
            ):
                return False
            if not feature.enable_ramp_features and (
                column.startswith("feat_generator_1_ramp_")
                or column.startswith("feat_generator_1_acceleration")
                or column.startswith("feat_generator_1_ewma_")
            ):
                return False
            if feature.relation_features and column.startswith("feat_relation_"):
                return True
            if not feature.relation_features and column.startswith("feat_relation_"):
                return False
            if not column.startswith("feat_dynamic_"):
                return True
            if feature.dynamic_feature_scope == "none":
                return False
            if feature.dynamic_feature_scope == "all":
                return True
            return (
                "generator_" in column
                or "generator_use_" in column
                or "gas_holder" in column
                or "blast_furnace" in column
                or "coke" in column
                or "converter" in column
            )

        columns = [column for column in columns if enabled(column)]
    if not columns:
        raise ValueError("generator_1 核心特征配置没有匹配到任何字段")
    return features.loc[:, columns]


def select_generator1_tree_features(
    features: pd.DataFrame,
    profile: str,
    config: ForecastConfig | None = None,
) -> pd.DataFrame:
    """为 direct LightGBM 保留至多 80 个核心字段，控制树模型搜索自由度。"""

    core = select_generator1_features(features, profile, config)
    return core.iloc[:, :80]


def select_generator_all_features(features: pd.DataFrame) -> pd.DataFrame:
    """隔离 generator_1 后续研究字段，保持 generator_all 的既有 V3 输入不变。"""

    excluded_prefixes = (
        "feat_generator1_price_",
        "feat_generator1_slope_price_",
        "feat_generator_gas_total_price_",
        "feat_gas_holder_price_",
        "feat_slot_",
        "feat_weekday_",
        "feat_day_fourier_",
        "feat_week_fourier_",
        "feat_dynamic_",
        "feat_next_2h_price_",
    )
    columns = [
        column
        for column in features.columns
        if "_aligned_" not in column
        and not column.startswith(excluded_prefixes)
        and "_price_delta_" not in column
        and "_price_ratio_" not in column
        and "_price_direction_" not in column
        and column != "feat_is_weekend"
        and column != "feat_current_price"
    ]
    if not columns:
        raise ValueError("generator_all 既有 V3 没有可用特征")
    return features.loc[:, columns]


def apply_capacity_projection(prediction: pd.DataFrame) -> pd.DataFrame:
    """投影到两台机组的容量可行域，且不改变输入对象。"""

    output = prediction.copy()
    horizons = sorted(
        {
            int(column.rsplit("_t+", 1)[1].removesuffix("_pred"))
            for column in output.columns
            if column.startswith("generator_1_t+") or column.startswith("generator_all_t+")
        }
    )
    for minutes in horizons:
        generator_1 = f"generator_1_t+{minutes}_pred"
        generator_all = f"generator_all_t+{minutes}_pred"
        if generator_1 in output:
            output[generator_1] = output[generator_1].clip(0.0, 200.0)
        if generator_all in output:
            output[generator_all] = output[generator_all].clip(0.0, 440.0)
        if generator_1 in output and generator_all in output:
            output[generator_all] = np.maximum(output[generator_all], output[generator_1])
            output[generator_all] = np.minimum(
                output[generator_all], output[generator_1] + 240.0
            )
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError("容量投影后包含非有限预测")
    return output


def smooth_prediction_paths(
    prediction: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    penalty: float,
) -> pd.DataFrame:
    """在每个预测起点上做轻量二阶差分平滑，不假设路径单调。"""

    if penalty < 0:
        raise ValueError("路径平滑惩罚不能为负数")
    output = prediction.copy()
    if penalty == 0.0 or len(horizons) < 3:
        return output
    difference = np.zeros((len(horizons) - 2, len(horizons)), dtype=float)
    for position in range(len(horizons) - 2):
        difference[position, position : position + 3] = (1.0, -2.0, 1.0)
    system = np.eye(len(horizons)) + penalty * difference.T @ difference
    for target in ("generator_1", "generator_all"):
        columns = [f"{target}_t+{15 * horizon}_pred" for horizon in horizons]
        available = [column for column in columns if column in output]
        if not available:
            continue
        if len(available) != len(columns):
            raise ValueError(f"{target} 的路径平滑缺少部分预测步长")
        values = output.loc[:, columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("路径平滑输入包含非有限预测")
        output.loc[:, columns] = np.linalg.solve(system, values.T).T
    return output


def _make_generator_all_baseline(
    config: ForecastConfig,
) -> HorizonSpecificRidgeForecaster | RidgeDeltaForecaster | GasAwareEnsembleForecaster:
    """构造保持独立的 generator_all 基线。"""

    all_config = replace(config, targets=("generator_all",))
    version = config.model.generator_all_route_model
    if version == "horizon_ridge":
        return HorizonSpecificRidgeForecaster(all_config)
    if version == "v1":
        return RidgeDeltaForecaster(all_config)
    if version == "v3":
        return GasAwareEnsembleForecaster("v3", all_config)
    raise ValueError(f"不支持的 generator_all 路由基线: {version}")


def _fit_generator_all_baseline(
    config: ForecastConfig,
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    current: pd.DataFrame,
) -> HorizonSpecificRidgeForecaster | RidgeDeltaForecaster | GasAwareEnsembleForecaster:
    """以冻结的 V3 特征子集拟合 generator_all，隔离 generator_1 研究变量。"""

    return _make_generator_all_baseline(config).fit(
        select_generator_all_features(features), deltas, current
    )


def _predict_generator_all_baseline(
    model: object,
    features: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    """以与训练完全相同的冻结特征子集推理 generator_all。"""

    return model.predict(select_generator_all_features(features), current)


def merge_target_route_predictions(
    config: ForecastConfig,
    generator_1: pd.DataFrame,
    generator_all: pd.DataFrame,
) -> pd.DataFrame:
    """合并两个单目标模型，并恢复提交要求的固定列顺序。"""

    output = pd.concat([generator_1, generator_all], axis=1)
    output = output.loc[:, ~output.columns.duplicated()].reindex(
        columns=[
            f"{target}_t+{15 * horizon}_pred"
            for target in config.targets
            for horizon in config.feature.horizons
        ]
    )
    if config.model.apply_capacity_projection:
        return apply_capacity_projection(output)
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError("未投影的目标路由预测包含非有限值")
    return output


class Generator1HorizonRidgeForecaster:
    """仅替换 ``generator_1``，并保持 ``generator_all`` 独立基线的目标路由。"""

    version = "generator1_horizon_ridge"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.generator1_model_: HorizonSpecificRidgeForecaster | None = None
        self.generator_all_model_: (
            HorizonSpecificRidgeForecaster | RidgeDeltaForecaster | GasAwareEnsembleForecaster | None
        ) = None

    def _generator1_config(self) -> ForecastConfig:
        return replace(self.config, targets=("generator_1",))

    def _generator_all_config(self) -> ForecastConfig:
        return replace(self.config, targets=("generator_all",))

    def _make_generator_all_model(
        self,
    ) -> HorizonSpecificRidgeForecaster | RidgeDeltaForecaster | GasAwareEnsembleForecaster:
        return _make_generator_all_baseline(self.config)

    def fit_generator1_only(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "Generator1HorizonRidgeForecaster":
        """只拟合 generator_1，用于复用已冻结的 generator_all OOF 路由。"""

        required = {"generator_1"}
        missing = sorted(required.difference(current.columns))
        if missing:
            raise ValueError(f"generator_1 训练缺少当前目标: {missing}")
        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        self.generator1_model_ = HorizonSpecificRidgeForecaster(self._generator1_config()).fit(
            generator1_features,
            deltas,
            current[["generator_1"]],
        )
        return self

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "Generator1HorizonRidgeForecaster":
        required = {"generator_1", "generator_all"}
        missing = sorted(required.difference(current.columns))
        if missing:
            raise ValueError(f"目标路由训练缺少当前目标: {missing}")
        self.fit_generator1_only(features, deltas, current)
        self.generator_all_model_ = _fit_generator_all_baseline(
            self.config, features, deltas, current
        )
        return self

    def predict_generator1_only(
        self, features: pd.DataFrame, current: pd.DataFrame
    ) -> pd.DataFrame:
        """仅产生 generator_1 的预测，供 OOF 路由复用调用。"""

        if self.generator1_model_ is None:
            raise RuntimeError("generator_1 专项模型尚未训练")
        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        return self.generator1_model_.predict(generator1_features, current[["generator_1"]])

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if self.generator_all_model_ is None:
            raise RuntimeError("generator_1 目标路由尚未训练")
        generator_1 = self.predict_generator1_only(features, current)
        generator_all = _predict_generator_all_baseline(self.generator_all_model_, features, current)
        return merge_target_route_predictions(self.config, generator_1, generator_all)


@dataclass
class IncrementalTargetState:
    """一个目标的累计增量 Ridge 状态。"""

    model: object
    increment_lower: np.ndarray
    increment_upper: np.ndarray


class IncrementalPathRidgeForecaster:
    """预测相邻步长增量并累加回绝对路径的低相关线性候选。"""

    version = "incremental_path_ridge"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.feature_columns_: list[str] = []
        self.states_: dict[str, IncrementalTargetState] = {}

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "IncrementalPathRidgeForecaster":
        self.feature_columns_ = list(features.columns)
        for target in self.config.targets:
            columns = target_columns(target, self.config.feature.horizons)
            valid = current[target].notna() & deltas[columns].notna().all(axis=1)
            x = features.loc[valid]
            target_deltas = deltas.loc[valid, columns].to_numpy(dtype=float)
            if len(x) < 200:
                raise ValueError(f"{target} 的累计增量 Ridge 有效训练样本不足 200 行")
            increments = np.column_stack(
                [target_deltas[:, 0], np.diff(target_deltas, axis=1)]
            )
            model = make_ridge_pipeline(self.config.model.ridge_alpha)
            model.fit(x, increments)
            self.states_[target] = IncrementalTargetState(
                model=model,
                increment_lower=np.quantile(
                    increments, self.config.model.lower_quantile, axis=0
                ),
                increment_upper=np.quantile(
                    increments, self.config.model.upper_quantile, axis=0
                ),
            )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if not self.states_:
            raise RuntimeError("累计增量 Ridge 尚未训练")
        x = features.reindex(columns=self.feature_columns_)
        output: dict[str, np.ndarray] = {}
        for target in self.config.targets:
            state = self.states_[target]
            increments = np.asarray(state.model.predict(x), dtype=float)
            increments = np.clip(increments, state.increment_lower, state.increment_upper)
            absolute = current[target].ffill().to_numpy(dtype=float)[:, None] + np.cumsum(
                increments, axis=1
            )
            for position, horizon in enumerate(self.config.feature.horizons):
                output[f"{target}_t+{15 * horizon}_pred"] = absolute[:, position]
        return apply_capacity_projection(pd.DataFrame(output, index=features.index))


class Generator1IncrementalPathForecaster:
    """将 generator_1 累计路径候选与冻结的 generator_all 基线组合。"""

    version = "generator1_incremental_path"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.generator1_model_: IncrementalPathRidgeForecaster | None = None
        self.generator_all_model_: (
            HorizonSpecificRidgeForecaster | RidgeDeltaForecaster | GasAwareEnsembleForecaster | None
        ) = None

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "Generator1IncrementalPathForecaster":
        required = {"generator_1", "generator_all"}
        missing = sorted(required.difference(current.columns))
        if missing:
            raise ValueError(f"累计路径目标路由训练缺少当前目标: {missing}")
        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        self.generator1_model_ = IncrementalPathRidgeForecaster(
            replace(self.config, targets=("generator_1",))
        ).fit(generator1_features, deltas, current[["generator_1"]])
        self.generator_all_model_ = _fit_generator_all_baseline(
            self.config, features, deltas, current
        )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if self.generator1_model_ is None or self.generator_all_model_ is None:
            raise RuntimeError("累计路径目标路由尚未训练")
        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        generator_1 = self.generator1_model_.predict(generator1_features, current[["generator_1"]])
        generator_all = _predict_generator_all_baseline(self.generator_all_model_, features, current)
        return merge_target_route_predictions(self.config, generator_1, generator_all)


class DirectIncrementalBlendForecaster:
    """以固定低自由度权重融合直接 Ridge 与累计路径 Ridge。"""

    version = "generator1_direct_incremental_blend"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.direct_model_: HorizonSpecificRidgeForecaster | None = None
        self.incremental_model_: IncrementalPathRidgeForecaster | None = None
        self.generator_all_model_: object | None = None

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "DirectIncrementalBlendForecaster":
        weight = self.config.model.incremental_blend_weight
        if not 0.0 <= weight <= 1.0:
            raise ValueError("累计路径融合权重必须位于 [0, 1]")
        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        generator1_config = replace(self.config, targets=("generator_1",))
        self.direct_model_ = HorizonSpecificRidgeForecaster(generator1_config).fit(
            generator1_features, deltas, current[["generator_1"]]
        )
        self.incremental_model_ = IncrementalPathRidgeForecaster(generator1_config).fit(
            generator1_features, deltas, current[["generator_1"]]
        )
        self.generator_all_model_ = _fit_generator_all_baseline(
            self.config, features, deltas, current
        )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if (
            self.direct_model_ is None
            or self.incremental_model_ is None
            or self.generator_all_model_ is None
        ):
            raise RuntimeError("直接/累计路径融合尚未训练")
        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        direct = self.direct_model_.predict(generator1_features, current[["generator_1"]])
        incremental = self.incremental_model_.predict(
            generator1_features, current[["generator_1"]]
        )
        weight = self.config.model.incremental_blend_weight
        generator_1 = direct * (1.0 - weight) + incremental * weight
        generator_all = _predict_generator_all_baseline(self.generator_all_model_, features, current)
        return merge_target_route_predictions(self.config, generator_1, generator_all)


class PathSmoothedGenerator1HorizonRidgeForecaster(Generator1HorizonRidgeForecaster):
    """在目标路由预测后施加二阶差分路径平滑的候选。"""

    version = "generator1_horizon_ridge_path_smoothed"

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        base = super().predict(features, current)
        smoothed = smooth_prediction_paths(
            base,
            self.config.feature.horizons,
            penalty=self.config.model.path_smoothing_lambda,
        )
        return apply_capacity_projection(smoothed)


class Generator1CatBoostForecaster:
    """只给 generator_1 的浅层 CatBoost 复验，generator_all 维持冻结基线。"""

    version = "generator1_catboost"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.generator1_model_: object | None = None
        self.generator_all_model_: object | None = None

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "Generator1CatBoostForecaster":
        from gas_forecast.candidates import CatBoostDeltaForecaster

        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        self.generator1_model_ = CatBoostDeltaForecaster(
            replace(self.config, targets=("generator_1",))
        ).fit(generator1_features, deltas, current[["generator_1"]])
        self.generator_all_model_ = _fit_generator_all_baseline(
            self.config, features, deltas, current
        )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if self.generator1_model_ is None or self.generator_all_model_ is None:
            raise RuntimeError("generator_1 CatBoost 路由尚未训练")
        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        generator_1 = self.generator1_model_.predict(generator1_features, current[["generator_1"]])
        generator_all = _predict_generator_all_baseline(self.generator_all_model_, features, current)
        return merge_target_route_predictions(self.config, generator_1, generator_all)


class Generator1LightGBMForecaster:
    """只给 generator_1 的直接 LightGBM 复验，避免再次进入 residual 主线。"""

    version = "generator1_lgb_direct"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.generator1_model_: object | None = None
        self.generator_all_model_: object | None = None

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "Generator1LightGBMForecaster":
        from gas_forecast.candidates import LightGBMDirectDeltaForecaster

        generator1_features = select_generator1_tree_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        self.generator1_model_ = LightGBMDirectDeltaForecaster(
            replace(self.config, targets=("generator_1",))
        ).fit(generator1_features, deltas, current[["generator_1"]])
        self.generator_all_model_ = _fit_generator_all_baseline(
            self.config, features, deltas, current
        )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if self.generator1_model_ is None or self.generator_all_model_ is None:
            raise RuntimeError("generator_1 直接 LightGBM 路由尚未训练")
        generator1_features = select_generator1_tree_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        generator_1 = self.generator1_model_.predict(generator1_features, current[["generator_1"]])
        generator_all = _predict_generator_all_baseline(self.generator_all_model_, features, current)
        return merge_target_route_predictions(self.config, generator_1, generator_all)


@dataclass
class StateClassifier:
    """三状态分类器或小样本时的常量回退。"""

    model: Pipeline | None
    constant_class: int | None = None

    def probabilities(self, features: pd.DataFrame) -> np.ndarray:
        output = np.zeros((len(features), 3), dtype=float)
        if self.model is None:
            if self.constant_class is None:
                raise RuntimeError("状态分类器没有可用类别")
            output[:, self.constant_class] = 1.0
            return output
        probabilities = self.model.predict_proba(features)
        classes = self.model.named_steps["logistic"].classes_
        for position, state in enumerate(classes):
            output[:, int(state)] = probabilities[:, position]
        return output


@dataclass
class StateExpertTargetState:
    """状态专家的最终分类器、专家回归器和边界。"""

    feature_columns: list[str]
    classifiers: list[StateClassifier]
    experts: list[Pipeline]
    delta_lower: np.ndarray
    delta_upper: np.ndarray


def select_generator1_state_features(features: pd.DataFrame) -> pd.DataFrame:
    """提取状态专家允许使用的当前工况特征。"""

    direct = {
        "generator_1",
        "feat_generator_gas_total",
        "feat_generator_1_slope_4",
        "feat_generator_1_slope_8",
        "feat_generator_1_diff_1",
        "feat_generator_1_diff_2",
        "feat_generator_1_diff_4",
    }
    columns = [
        column
        for column in features.columns
        if column in direct
        or column.startswith("generator_use_")
        or column.startswith("feat_generator_use_")
        or "gas_holder" in column
    ]
    if not columns:
        raise ValueError("状态专家没有匹配到可用的当前工况特征")
    return features.loc[:, columns]


class Generator1StateExpertRidgeForecaster:
    """用时间交叉拟合状态概率训练的三专家 Ridge，不做硬分类。"""

    version = "generator1_state_expert_ridge"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.state_: StateExpertTargetState | None = None
        self.oof_probability_rows_: int = 0

    def _fit_classifier(self, features: pd.DataFrame, labels: np.ndarray) -> StateClassifier:
        classes = np.unique(labels)
        if len(classes) == 1:
            return StateClassifier(model=None, constant_class=int(classes[0]))
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=0.2,
                        class_weight="balanced",
                        max_iter=400,
                        random_state=self.config.model.random_state,
                    ),
                ),
            ]
        )
        model.fit(features, labels)
        return StateClassifier(model=model)

    def _state_labels(self, deltas: np.ndarray) -> np.ndarray:
        split = len(self.config.feature.horizons) // 2
        grouped = (deltas[:, :split].mean(axis=1), deltas[:, split:].mean(axis=1))
        threshold = self.config.model.state_transition_threshold
        if threshold <= 0:
            raise ValueError("状态切换阈值必须为正数")
        return np.column_stack(
            [np.where(value < -threshold, 0, np.where(value > threshold, 2, 1)) for value in grouped]
        )

    def _cross_fitted_probabilities(
        self,
        state_features: pd.DataFrame,
        labels: np.ndarray,
    ) -> np.ndarray:
        horizons = self.config.feature.horizons
        folds = make_inner_folds(
            state_features.index,
            folds=self.config.model.state_expert_inner_folds,
            purge_steps=max(horizons),
            min_train_rows=200,
            min_validation_rows=48,
        )
        output = np.full((len(state_features), labels.shape[1], 3), np.nan, dtype=float)
        for fold in folds:
            train_mask, validation_mask = fold.masks(state_features.index)
            for group in range(labels.shape[1]):
                classifier = self._fit_classifier(
                    state_features.loc[train_mask], labels[train_mask, group]
                )
                output[validation_mask, group, :] = classifier.probabilities(
                    state_features.loc[validation_mask]
                )
        return output

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "Generator1StateExpertRidgeForecaster":
        target = "generator_1"
        columns = target_columns(target, self.config.feature.horizons)
        valid = current[target].notna() & deltas[columns].notna().all(axis=1)
        x = features.loc[valid]
        state_features = select_generator1_state_features(x)
        y = deltas.loc[valid, columns].to_numpy(dtype=float)
        if len(x) < 600:
            raise ValueError("generator_1 状态专家有效训练样本不足 600 行")
        labels = self._state_labels(y)
        oof_probabilities = self._cross_fitted_probabilities(state_features, labels)
        expert_mask = np.isfinite(oof_probabilities).all(axis=(1, 2))
        self.oof_probability_rows_ = int(expert_mask.sum())
        if self.oof_probability_rows_ < 200:
            raise ValueError("状态专家没有足够的 OOF 状态概率训练专家")
        experts: list[Pipeline] = []
        for state in range(3):
            weights = oof_probabilities[expert_mask, :, state].mean(axis=1)
            weights = np.clip(weights, 0.02, None)
            model = make_ridge_pipeline(self.config.model.ridge_alpha)
            model.fit(x.loc[expert_mask], y[expert_mask], ridge__sample_weight=weights)
            experts.append(model)
        classifiers = [
            self._fit_classifier(state_features, labels[:, group])
            for group in range(labels.shape[1])
        ]
        self.state_ = StateExpertTargetState(
            feature_columns=list(state_features.columns),
            classifiers=classifiers,
            experts=experts,
            delta_lower=np.quantile(y, self.config.model.lower_quantile, axis=0),
            delta_upper=np.quantile(y, self.config.model.upper_quantile, axis=0),
        )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if self.state_ is None:
            raise RuntimeError("generator_1 状态专家尚未训练")
        state_features = select_generator1_state_features(features).reindex(
            columns=self.state_.feature_columns
        )
        group_probabilities = np.stack(
            [classifier.probabilities(state_features) for classifier in self.state_.classifiers],
            axis=1,
        )
        expert_delta = np.stack(
            [model.predict(features) for model in self.state_.experts], axis=1
        )
        split = len(self.config.feature.horizons) // 2
        delta = np.empty((len(features), len(self.config.feature.horizons)), dtype=float)
        for step in range(delta.shape[1]):
            group = 0 if step < split else 1
            delta[:, step] = np.sum(
                group_probabilities[:, group, :] * expert_delta[:, :, step], axis=1
            )
        delta = np.clip(delta, self.state_.delta_lower, self.state_.delta_upper)
        anchor = current["generator_1"].ffill().to_numpy(dtype=float)[:, None]
        output = pd.DataFrame(index=features.index)
        for step, horizon in enumerate(self.config.feature.horizons):
            output[f"generator_1_t+{15 * horizon}_pred"] = anchor[:, 0] + delta[:, step]
        return apply_capacity_projection(output)


class Generator1StateExpertForecaster:
    """将 OOF 状态专家作为 generator_1 候选，并保留 generator_all 独立基线。"""

    version = "generator1_state_expert"

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.generator1_model_: Generator1StateExpertRidgeForecaster | None = None
        self.generator_all_model_: object | None = None

    def fit(
        self,
        features: pd.DataFrame,
        deltas: pd.DataFrame,
        current: pd.DataFrame,
    ) -> "Generator1StateExpertForecaster":
        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        self.generator1_model_ = Generator1StateExpertRidgeForecaster(
            replace(self.config, targets=("generator_1",))
        ).fit(generator1_features, deltas, current[["generator_1"]])
        self.generator_all_model_ = _fit_generator_all_baseline(
            self.config, features, deltas, current
        )
        return self

    def predict(self, features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        if self.generator1_model_ is None or self.generator_all_model_ is None:
            raise RuntimeError("generator_1 状态专家路由尚未训练")
        generator1_features = select_generator1_features(
            features, self.config.model.generator1_feature_profile, self.config
        )
        generator_1 = self.generator1_model_.predict(generator1_features, current[["generator_1"]])
        generator_all = _predict_generator_all_baseline(self.generator_all_model_, features, current)
        return merge_target_route_predictions(self.config, generator_1, generator_all)
