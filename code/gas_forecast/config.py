"""项目级配置。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureConfig:
    """特征窗口配置，所有步数均以 15 分钟为单位。"""

    frequency: str = "15min"
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 96)
    diff_lags: tuple[int, ...] = (1, 2, 4, 8, 16)
    rolling_windows: tuple[int, ...] = (4, 8, 16, 32, 96)
    max_forward_fill_steps: int = 8


@dataclass(frozen=True)
class ModelConfig:
    """模型和稳健化参数。"""

    ridge_alpha: float = 20.0
    recent_days: int = 60
    calibration_fraction: float = 0.15
    gate_margin: float = 0.25
    random_state: int = 20250731
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999
    lgb_n_estimators: int = 160
    lgb_num_leaves: int = 15
    lgb_max_depth: int = 5
    state_components: int = 5
    uncertainty_scale: float = 0.05


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
