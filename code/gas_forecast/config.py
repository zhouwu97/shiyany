"""项目级配置。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


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
