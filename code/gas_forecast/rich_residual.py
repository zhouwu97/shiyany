"""基于严格 Champion OOF 的 generator_1 Rich Residual 研究分支。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

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
DEFAULT_BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass(frozen=True)
class RichResidualSpec:
    """一个冻结的 RichResidual 特征组和残差模型容量配置。"""

    name: str
    feature_groups: frozenset[str] = frozenset()
    min_train_rows: int = 256
    n_estimators: int | None = None
    blend_weights: tuple[float, ...] = DEFAULT_BLEND_WEIGHTS

    def __post_init__(self) -> None:
        invalid = sorted(self.feature_groups.difference(RICH_FEATURE_GROUPS))
        if invalid:
            raise ValueError(f"未知 Rich 特征组: {invalid}")
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("RichResidual 名称只能包含字母、数字和下划线")
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


def rich_feature_config(config: ForecastConfig, groups: Iterable[str]) -> ForecastConfig:
    """只按登记的组启用 Rich 特征，不隐式改变既有 Champion 开关。"""

    selected = frozenset(groups)
    invalid = sorted(selected.difference(RICH_FEATURE_GROUPS))
    if invalid:
        raise ValueError(f"未知 Rich 特征组: {invalid}")
    return replace(
        config,
        feature=replace(
            config.feature,
            enable_rich_quantile_features="quantile" in selected,
            enable_rich_ramp_state_features="ramp" in selected,
            enable_rich_gas_resource_features="gas" in selected,
        ),
    )


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
    """只从已完成的 Champion OOF 学习 generator_1 残差。"""

    version = "generator1_rich_residual_corrector"

    def __init__(self, config: ForecastConfig, spec: RichResidualSpec) -> None:
        self.config = config
        self.spec = spec
        self.states_: dict[int, RichResidualHorizonState] = {}
        self.feature_columns_: list[str] = []

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
        target_rows = rows.loc[rows["target"].eq("generator_1")].copy()
        if target_rows.empty:
            raise ValueError("Champion OOF 没有 generator_1 行")
        self.states_.clear()
        self.feature_columns_ = list(features.select_dtypes(include=[np.number]).columns)
        if not self.feature_columns_:
            raise ValueError("RichResidual 没有可用数值特征")
        horizons = tuple(15 * value for value in self.config.feature.horizons)
        for horizon in horizons:
            part = target_rows.loc[target_rows["horizon"].eq(horizon)].sort_values(
                "origin_time"
            )
            if part.empty:
                continue
            matrix = _finite_feature_matrix(features, part["origin_time"]).reindex(
                columns=self.feature_columns_
            )
            actual = part["actual"].to_numpy(dtype=float)
            baseline = part[baseline_column].to_numpy(dtype=float)
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
        """对长表 generator_1 行修正基线；未训练步长严格回退到基线。"""

        required = {"origin_time", "target", "horizon", baseline_column}
        missing = sorted(required.difference(rows.columns))
        if missing:
            raise ValueError(f"RichResidual 预测行缺少字段: {missing}")
        prediction = pd.to_numeric(rows[baseline_column], errors="coerce").astype(float).copy()
        if not np.isfinite(prediction.to_numpy()).all():
            raise ValueError("RichResidual 预测基线含非有限数")
        generator_1 = rows["target"].eq("generator_1")
        for horizon, state in self.states_.items():
            mask = generator_1 & rows["horizon"].eq(horizon)
            if not mask.any():
                continue
            part = rows.loc[mask]
            matrix = _finite_feature_matrix(features, pd.to_datetime(part["origin_time"]))
            matrix = matrix.reindex(columns=self.feature_columns_)
            correction = state.model.predict(state.imputer.transform(matrix))
            correction = np.clip(correction, state.residual_lower, state.residual_upper)
            prediction.loc[mask] = part[baseline_column].to_numpy(dtype=float) + correction
        return prediction

    def predict_wide(self, features: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
        """在生产宽表上只修正 generator_1，generator_all 留给容量投影协调。"""

        output = baseline.copy()
        for horizon, state in self.states_.items():
            column = f"generator_1_t+{horizon}_pred"
            if column not in output:
                raise ValueError(f"生产基线缺少预测列: {column}")
            matrix = _finite_feature_matrix(features, pd.Series(features.index))
            matrix = matrix.reindex(columns=self.feature_columns_)
            correction = state.model.predict(state.imputer.transform(matrix))
            correction = np.clip(correction, state.residual_lower, state.residual_upper)
            output[column] = output[column].to_numpy(dtype=float) + correction
        return output


class RichResidualAggressiveForecaster:
    """将已冻结的 RichResidual 以固定小权重叠加到 Aggressive Champion。"""

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
        for horizon in self.config.feature.horizons:
            column = f"generator_1_t+{15 * horizon}_pred"
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

    rich_config = rich_feature_config(config, spec.feature_groups)
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
    rich_config = rich_feature_config(config, spec.feature_groups)
    rows = _validate_oof_rows(champion_oof, baseline_column)
    folds = select_research_folds(frame.index, rich_config, scope=scope)
    if not folds:
        raise ValueError("RichResidual 没有可用外层折")
    fold_names = [fold.name for fold in folds]
    missing_folds = sorted(set(fold_names).difference(rows["fold"].astype(str)))
    if missing_folds:
        raise ValueError(f"Champion OOF 未覆盖 RichResidual 外层折: {missing_folds}")
    features = build_causal_features(frame, rich_config.feature, price_schedule)
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
            history_source["target"].eq("generator_1")
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
            "feature_groups": sorted(spec.feature_groups),
            "folds": fold_names,
            "feature_columns": int(features.shape[1]),
            "fold_training_rows": fold_training_rows,
            "trained_horizons": trained_horizons,
            "models": reports,
            "strict_oof_contract": {
                "residual_target": "actual - same_fold_champion_prediction",
                "history_rule": "origin_time <= outer_fold.train_end",
                "blind_labels_used": scope == "final",
                "target_scope": "generator_1",
            },
        },
        feature_config=rich_config,
    )
