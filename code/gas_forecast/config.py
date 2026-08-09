"""项目级配置。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping


@dataclass(frozen=True)
class FeatureConfig:
    """特征窗口配置，所有步数均以 15 分钟为单位。"""

    frequency: str = "15min"
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 96, 192, 288, 672)
    diff_lags: tuple[int, ...] = (1, 2, 4, 8, 16)
    rolling_windows: tuple[int, ...] = (4, 8, 16, 32, 96)
    max_forward_fill_steps: int = 8
    anomaly_window: int = 16
    anomaly_threshold: float = 4.5
    enable_anomaly_features: bool = True
    enable_physical_features: bool = True
    enable_long_cycle_features: bool = True
    enable_target_aligned_features: bool = False
    target_aligned_cycle_days: tuple[int, ...] = (1, 2, 3, 7)
    enable_slot_one_hot: bool = False
    enable_time_fourier: bool = False
    enable_price_delta_features: bool = False
    enable_price_interactions: bool = False
    enable_ramp_features: bool = False
    enable_rich_quantile_features: bool = False
    enable_rich_ramp_state_features: bool = False
    enable_rich_gas_resource_features: bool = False
    rich_quantile_windows: tuple[int, ...] = (8, 32, 96)
    relation_features: tuple[str, ...] = ()
    dynamic_feature_scope: str = "none"
    dynamic_lags: tuple[int, ...] = (1, 2, 4, 8)
    dynamic_rolling_windows: tuple[int, ...] = (4, 8)


@dataclass(frozen=True)
class ModelConfig:
    """模型和稳健化参数。"""

    ridge_alpha: float = 20.0
    recent_days: int = 60
    calibration_fraction: float = 0.15
    random_state: int = 20250731
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999
    lgb_n_estimators: int = 160
    lgb_max_estimators: int = 800
    lgb_learning_rate: float = 0.03
    lgb_num_leaves: int = 15
    lgb_max_depth: int = 5
    lgb_min_child_samples: int = 100
    lgb_objective: str = "regression_l1"
    lgb_early_stopping_rounds: int = 50
    lgb_use_mape_weights: bool = True
    lgb_use_early_stopping: bool = True
    tree_threads_per_worker: int = 1
    inner_folds: int = 5
    simplex_regularization: float = 0.002
    catboost_iterations: int = 600
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.03
    catboost_early_stopping_rounds: int = 50
    state_components: int = 5
    gate_min: float = 0.05
    gate_max: float = 0.70
    generator_1_max_shrink: float = 0.60
    generator_all_max_shrink: float = 0.40
    generator1_feature_profile: str = "all"
    generator_all_route_model: str = "v3"
    generator1_short_alpha: float | None = None
    generator1_long_alpha: float | None = None
    ridge_recency_mode: str = "all"
    ridge_hard_window_days: int | None = None
    ridge_half_life_days: float | None = None
    ridge_short_half_life_days: float | None = None
    ridge_long_half_life_days: float | None = None
    ridge_magnitude_weighting: str = "uniform"
    ridge_loss: str = "ridge"
    weighted_lad_alpha: float = 0.05
    state_transition_threshold: float = 3.0
    state_expert_inner_folds: int = 5
    path_smoothing_lambda: float = 0.0
    incremental_blend_weight: float = 0.25
    online_calibration_rows: int = 192
    online_refit_stride: int = 8
    online_half_life: float = 16.0
    online_bias_clip: float = 12.0
    online_vintage_weight: float = 0.25
    apply_capacity_projection: bool = True
    analog_k: int = 40
    analog_slot_penalty: float = 0.0
    analog_mode: str = "weighted_median"
    analog_local_ridge_alpha: float = 20.0
    damped_trend_window: int = 4
    damped_trend_damping: float = 0.85


@dataclass(frozen=True)
class ValidationConfig:
    """前向滚动验证参数。"""

    first_validation_date: str = "2025-03-20"
    fold_spacing_days: int = 2
    validation_days: int = 2
    blind_days: int = 4
    min_train_days: int = 45


@dataclass(frozen=True)
class ForecastConfig:
    """完整运行配置。"""

    targets: tuple[str, ...] = ("generator_1", "generator_all")
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def legacy_forecast_config() -> ForecastConfig:
    """返回与既有 V1/V2/V2.5/V3 报告一致的基础特征配置。"""

    config = ForecastConfig()
    return replace(
        config,
        feature=replace(
            config.feature,
            lags=(1, 2, 4, 8, 16, 32, 96),
            enable_anomaly_features=False,
            enable_physical_features=False,
            enable_long_cycle_features=False,
        ),
    )


def horizon_ridge_forecast_config() -> ForecastConfig:
    """返回目标时刻对齐 Ridge 使用的低自由度配置。"""

    config = legacy_forecast_config()
    return replace(
        config,
        feature=replace(
            config.feature,
            enable_target_aligned_features=True,
            enable_long_cycle_features=True,
        ),
    )


def research_forecast_config() -> ForecastConfig:
    """返回 Phase 1–14 目标路由候选的默认冻结前配置。"""

    config = ForecastConfig()
    return replace(
        config,
        feature=replace(
            config.feature,
            enable_target_aligned_features=True,
            enable_long_cycle_features=True,
        ),
        model=replace(config.model, generator1_feature_profile="core"),
    )


def research_feature_superset(feature: FeatureConfig) -> FeatureConfig:
    """返回同时覆盖既有 V3 与 generator_1 研究模块的因果特征超集。"""

    return replace(
        feature,
        enable_long_cycle_features=True,
    )


def forecast_config_from_dict(payload: Mapping[str, object]) -> ForecastConfig:
    """从实验报告中的 ``asdict(ForecastConfig)`` 恢复不可变运行配置。"""

    feature_payload = dict(payload.get("feature", {}))
    for key in (
        "horizons",
        "lags",
        "diff_lags",
        "rolling_windows",
        "target_aligned_cycle_days",
        "dynamic_lags",
        "dynamic_rolling_windows",
        "relation_features",
        "rich_quantile_windows",
    ):
        if key in feature_payload:
            feature_payload[key] = tuple(feature_payload[key])
    model_payload = dict(payload.get("model", {}))
    validation_payload = dict(payload.get("validation", {}))
    targets = tuple(payload.get("targets", ForecastConfig().targets))
    return ForecastConfig(
        targets=targets,
        feature=FeatureConfig(**feature_payload),
        model=ModelConfig(**model_payload),
        validation=ValidationConfig(**validation_payload),
    )
