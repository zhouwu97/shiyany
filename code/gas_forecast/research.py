"""Phase 1–14 研究候选登记、统一切分与 OOF 评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gas_forecast.config import ForecastConfig, research_feature_superset
from gas_forecast.features import PriceSchedule, build_causal_features
from gas_forecast.oof import _base_fold_rows
from gas_forecast.online import (
    apply_online_calibration_hot_start,
    apply_online_calibration_hot_start_pipeline,
)
from gas_forecast.regimes import attach_regimes
from gas_forecast.scoring import absolute_percentage_error, score_oof_long
from gas_forecast.splits import TimeFold, make_outer_folds
from gas_forecast.targets import build_delta_targets


@dataclass(frozen=True)
class ResearchCandidate:
    """一个参数已冻结、可用同一 OOF 口径比较的研究候选。"""

    experiment_id: str
    name: str
    kind: str
    config: ForecastConfig
    description: str
    online_mode: str | None = None
    online_modes: tuple[str, ...] = ()


def _base_research_config(config: ForecastConfig) -> ForecastConfig:
    """所有 generator_1 研究候选默认从低风险核心特征开始。"""

    return replace(
        config,
        model=replace(config.model, generator1_feature_profile="core"),
    )


def _with_feature(config: ForecastConfig, **changes: object) -> ForecastConfig:
    return replace(config, feature=replace(config.feature, **changes))


def _with_model(config: ForecastConfig, **changes: object) -> ForecastConfig:
    return replace(config, model=replace(config.model, **changes))


def _candidate(
    experiment_id: str,
    name: str,
    kind: str,
    config: ForecastConfig,
    description: str,
    *,
    online_mode: str | None = None,
    online_modes: tuple[str, ...] = (),
) -> ResearchCandidate:
    return ResearchCandidate(
        experiment_id=experiment_id,
        name=name,
        kind=kind,
        config=config,
        description=description,
        online_mode=online_mode,
        online_modes=online_modes,
    )


def _best_linear_seed(config: ForecastConfig) -> ForecastConfig:
    """组合实验的保守初值；最终取值仍由开发 OOF 冻结。"""

    return _with_model(
        _with_feature(
            config,
            enable_target_aligned_features=True,
            enable_long_cycle_features=True,
        ),
        ridge_recency_mode="exp",
        ridge_half_life_days=60.0,
    )


def make_research_candidates(
    experiment_id: str,
    config: ForecastConfig | None = None,
) -> list[ResearchCandidate]:
    """按路线中的实验 ID 生成有限、可追溯的候选集合。"""

    config = _base_research_config(config or ForecastConfig())
    if experiment_id == "E10_gen1_hridge_base":
        value = _with_feature(
            config,
            enable_target_aligned_features=False,
            enable_long_cycle_features=False,
        )
        return [_candidate(experiment_id, "e10_base", "horizon", value, "核心逐步长 Ridge")]
    if experiment_id == "E11_gen1_hridge_aligned":
        value = _with_feature(
            config,
            enable_target_aligned_features=True,
            enable_long_cycle_features=False,
            target_aligned_cycle_days=(1, 2),
        )
        return [_candidate(experiment_id, "e11_aligned", "horizon", value, "目标时刻对齐")]
    if experiment_id == "E12_gen1_hridge_aligned_longcycle":
        value = _with_feature(
            config,
            enable_target_aligned_features=True,
            enable_long_cycle_features=True,
            target_aligned_cycle_days=(1, 2, 3, 7),
        )
        return [_candidate(experiment_id, "e12_aligned_longcycle", "horizon", value, "长周期对齐")]
    if experiment_id == "E13_gen1_alpha_group":
        # 只调整短/长步长正则；特征开关必须继承上一阶段冻结的赢家。
        seed = config
        candidates = []
        for alpha in (5.0, 10.0, 20.0, 40.0, 80.0):
            candidates.append(
                _candidate(
                    experiment_id,
                    f"e13_short_alpha_{int(alpha)}",
                    "horizon",
                    _with_model(seed, generator1_short_alpha=alpha),
                    "仅调整短步长 alpha",
                )
            )
            candidates.append(
                _candidate(
                    experiment_id,
                    f"e13_long_alpha_{int(alpha)}",
                    "horizon",
                    _with_model(seed, generator1_long_alpha=alpha),
                    "仅调整长步长 alpha",
                )
            )
        return candidates
    if experiment_id == "E20_gen1_recency_hard":
        return [
            _candidate(
                experiment_id,
                f"e20_hard_{days}d",
                "horizon",
                _with_model(config, ridge_recency_mode="hard", ridge_hard_window_days=days),
                f"{days} 天硬窗口",
            )
            for days in (30, 60, 90)
        ]
    if experiment_id == "E21_gen1_recency_exp":
        return [
            _candidate(
                experiment_id,
                f"e21_exp_half_life_{days}d",
                "horizon",
                _with_model(config, ridge_recency_mode="exp", ridge_half_life_days=float(days)),
                f"{days} 天半衰期",
            )
            for days in (30, 60, 90)
        ]
    if experiment_id in {"E30_gen1_time_slot", "E31_gen1_fourier", "E32_gen1_slot_fourier"}:
        slot = experiment_id in {"E30_gen1_time_slot", "E32_gen1_slot_fourier"}
        fourier = experiment_id in {"E31_gen1_fourier", "E32_gen1_slot_fourier"}
        value = _with_feature(config, enable_slot_one_hot=slot, enable_time_fourier=fourier)
        return [_candidate(experiment_id, experiment_id.lower(), "horizon", value, "时间表达消融")]
    if experiment_id == "E40_price_delta":
        value = _with_feature(config, enable_price_delta_features=True)
        return [_candidate(experiment_id, "e40_price_delta", "horizon", value, "未来电价差")]
    if experiment_id == "E41_price_interactions":
        value = _with_feature(
            config,
            enable_price_delta_features=True,
            enable_price_interactions=True,
        )
        return [_candidate(experiment_id, "e41_price_interactions", "horizon", value, "电价交互")]
    if experiment_id == "E50_gen1_weighted_ridge":
        return [
            _candidate(
                experiment_id,
                f"e50_{mode}",
                "horizon",
                _with_model(config, ridge_magnitude_weighting=mode),
                "MagnitudeWeightedRidge",
            )
            for mode in ("inverse_absolute", "inverse_squared")
        ]
    if experiment_id == "E51_gen1_weighted_lad":
        value = _with_model(
            config,
            ridge_loss="weighted_lad",
            ridge_magnitude_weighting="inverse_absolute",
        )
        return [_candidate(experiment_id, "e51_weighted_lad", "horizon", value, "Weighted LAD")]
    if experiment_id in {"E60_aligned_recency", "E61_aligned_recency_time", "E62_aligned_recency_time_price", "E63_best_linear"}:
        value = _best_linear_seed(config)
        if experiment_id in {"E61_aligned_recency_time", "E62_aligned_recency_time_price", "E63_best_linear"}:
            value = _with_feature(value, enable_slot_one_hot=True)
        if experiment_id in {"E62_aligned_recency_time_price", "E63_best_linear"}:
            value = _with_feature(value, enable_price_delta_features=True, enable_price_interactions=True)
        if experiment_id == "E63_best_linear":
            value = _with_model(value, ridge_magnitude_weighting="inverse_absolute")
        return [_candidate(experiment_id, experiment_id.lower(), "horizon", value, "最佳线性组合")]
    if experiment_id == "E70_catboost_gen1_fixed_metric":
        seed = _best_linear_seed(config)
        return [
            _candidate(
                experiment_id,
                f"e70_catboost_d{depth}_lr{rate:.2f}",
                "catboost",
                _with_model(seed, catboost_depth=depth, catboost_learning_rate=rate),
                "固定加权 MAE CatBoost",
            )
            for depth in (4, 6)
            for rate in (0.03, 0.06)
        ]
    if experiment_id == "E80_lgb_direct_gen1":
        value = _best_linear_seed(config)
        return [_candidate(experiment_id, "e80_lgb_direct", "lgb", value, "直接增量 LightGBM")]
    if experiment_id in {
        "E90_online_bias_true_hot",
        "E91_online_gain_true_hot",
        "E92_online_vintage_true_hot",
    }:
        mode = experiment_id.split("_")[2]
        vintage_weights = (0.15, 0.25) if mode == "vintage" else (0.25,)
        return [
            _candidate(
                experiment_id,
                f"{experiment_id.lower()}_hl{half_life:g}_vw{vintage_weight:g}",
                "online_hot_start",
                _with_model(
                    _best_linear_seed(config),
                    online_half_life=half_life,
                    online_vintage_weight=vintage_weight,
                ),
                "真正 OOF hot start 在线校准",
                online_mode=mode,
            )
            for half_life in (8.0, 16.0, 32.0)
            for vintage_weight in vintage_weights
        ]
    if experiment_id == "E100_dynamic_core":
        value = _with_feature(_best_linear_seed(config), dynamic_feature_scope="core")
        return [_candidate(experiment_id, "e100_dynamic_core", "horizon", value, "核心动态字段")]
    if experiment_id == "E101_dynamic_all":
        value = _with_feature(_best_linear_seed(config), dynamic_feature_scope="all")
        return [_candidate(experiment_id, "e101_dynamic_all", "horizon", value, "全字段动态特征")]
    if experiment_id == "E110_gen1_moe":
        return [_candidate(experiment_id, "e110_state_expert", "state_expert", config, "三状态软专家")]
    if experiment_id == "E120_capacity_projection":
        return [
            _candidate(
                experiment_id,
                "e120_unprojected",
                "capacity",
                _with_model(config, apply_capacity_projection=False),
                "容量投影前的研究对照，仅用于 OOF 量化",
            ),
            _candidate(
                experiment_id,
                "e120_capacity_projection",
                "capacity",
                _with_model(config, apply_capacity_projection=True),
                "容量可行域投影",
            ),
        ]
    if experiment_id == "E121_path_smoothing":
        return [
            _candidate(
                experiment_id,
                f"e121_path_lambda_{penalty:g}",
                "path",
                _with_model(config, path_smoothing_lambda=penalty),
                "二阶差分路径平滑",
            )
            for penalty in (0.0, 0.1, 0.3, 1.0)
        ]
    if experiment_id == "E130_incremental_path":
        return [_candidate(experiment_id, "e130_incremental", "incremental", config, "累计增量路径")]
    if experiment_id == "E131_direct_incremental_blend":
        value = _with_model(config, incremental_blend_weight=0.25)
        return [_candidate(experiment_id, "e131_direct_incremental", "direct_incremental", value, "低权重路径融合")]
    raise ValueError(f"未知研究实验 ID: {experiment_id}")


def filter_research_candidate_names(
    candidates: list[ResearchCandidate],
    names: list[str] | None,
) -> list[ResearchCandidate]:
    """按冻结名称限制候选，避免 blind 阶段读取未登记的参数。"""

    if names is None:
        return candidates
    expected = set(names)
    selected = [candidate for candidate in candidates if candidate.name in expected]
    found = {candidate.name for candidate in selected}
    missing = sorted(expected.difference(found))
    if missing:
        raise ValueError(f"实验中不存在指定候选: {missing}")
    return selected


def make_online_combination_candidate(
    config: ForecastConfig,
    modes: tuple[str, ...],
) -> ResearchCandidate:
    """构造 Phase 10 的最多双模块 online 组合，调用方须先冻结单模块赢家。"""

    if not modes or len(modes) > 2 or len(set(modes)) != len(modes):
        raise ValueError("online 组合必须是一个或两个不重复模块")
    invalid = sorted(set(modes).difference({"bias", "gain", "vintage"}))
    if invalid:
        raise ValueError(f"未知 online 模块: {invalid}")
    value = _best_linear_seed(_base_research_config(config))
    name = "e93_online_" + "_".join(modes)
    return _candidate(
        "E93_online_combo",
        name,
        "online_hot_start",
        value,
        "冻结单模块赢家后的在线组合",
        online_modes=modes,
    )


def select_research_folds(
    index: pd.DatetimeIndex,
    config: ForecastConfig,
    *,
    scope: str,
) -> list[TimeFold]:
    """按快筛、完整开发和最终验收三层规则选择外层折。"""

    if scope not in {"screening", "development", "final"}:
        raise ValueError("研究 scope 只能是 screening、development 或 final")
    all_folds = make_outer_folds(index, config)
    development = [fold for fold in all_folds if not fold.blind]
    if scope == "development":
        return development
    if scope == "final":
        return all_folds
    if len(development) <= 5:
        return development
    positions = {0, len(development) // 2, len(development) - 1}
    switch_date = pd.Timestamp("2025-04-18")
    for position, fold in enumerate(development):
        if fold.validation_start <= switch_date < fold.validation_end:
            positions.add(position)
            break
    for position in (round((len(development) - 1) * ratio) for ratio in (0.25, 0.75)):
        if len(positions) >= 5:
            break
        positions.add(position)
    return [development[position] for position in sorted(positions)[:5]]


@dataclass(frozen=True)
class ResearchOOFResult:
    """研究实验的逐行 OOF 及统一评估报告。"""

    rows: pd.DataFrame
    report: dict[str, object]
    duration_seconds: float


def make_research_model(candidate: ResearchCandidate):
    """将注册候选解析为可训练模型；在线候选先返回其静态基础模型。"""

    from gas_forecast.research_models import (
        DirectIncrementalBlendForecaster,
        Generator1CatBoostForecaster,
        Generator1HorizonRidgeForecaster,
        Generator1IncrementalPathForecaster,
        Generator1LightGBMForecaster,
        Generator1StateExpertForecaster,
        PathSmoothedGenerator1HorizonRidgeForecaster,
    )

    if candidate.kind in {"horizon", "capacity", "online_hot_start"}:
        return Generator1HorizonRidgeForecaster(candidate.config)
    if candidate.kind == "catboost":
        return Generator1CatBoostForecaster(candidate.config)
    if candidate.kind == "lgb":
        return Generator1LightGBMForecaster(candidate.config)
    if candidate.kind == "state_expert":
        return Generator1StateExpertForecaster(candidate.config)
    if candidate.kind == "path":
        return PathSmoothedGenerator1HorizonRidgeForecaster(candidate.config)
    if candidate.kind == "incremental":
        return Generator1IncrementalPathForecaster(candidate.config)
    if candidate.kind == "direct_incremental":
        return DirectIncrementalBlendForecaster(candidate.config)
    raise ValueError(f"不支持的研究候选类型: {candidate.kind}")


def _prediction_long(
    prediction: pd.DataFrame,
    validation_index: pd.DatetimeIndex,
    config: ForecastConfig,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for target in config.targets:
        for horizon in config.feature.horizons:
            column = f"{target}_t+{15 * horizon}_pred"
            if column not in prediction:
                raise ValueError(f"研究候选缺少预测列: {column}")
            parts.append(
                pd.DataFrame(
                    {
                        "origin_time": validation_index,
                        "target": target,
                        "horizon": 15 * horizon,
                        "prediction": prediction[column].to_numpy(dtype=float),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def _calibration_history_predictions(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    fold: TimeFold,
    candidate: ResearchCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对训练尾部逐块重训，产生验证前真正 OOF 的 online 初始化历史。"""

    train_mask, _ = fold.masks(frame.index)
    train_index = frame.index[train_mask]
    history_rows = candidate.config.model.online_calibration_rows
    stride = candidate.config.model.online_refit_stride
    if history_rows < 1 or stride < 1:
        raise ValueError("online calibration rows 和 refit stride 必须为正数")
    origins = train_index[-min(history_rows, len(train_index)) :]
    if len(origins) < max(32, max(candidate.config.feature.horizons) * 2):
        raise ValueError("训练尾部没有足够的 OOF calibration history")
    purge = pd.Timedelta(minutes=15 * max(candidate.config.feature.horizons))
    parts: list[pd.DataFrame] = []
    for start in range(0, len(origins), stride):
        block = origins[start : start + stride]
        cutoff = block[0] - purge
        fit_mask = (frame.index >= fold.train_start) & (frame.index <= cutoff)
        if int(fit_mask.sum()) < 200:
            raise ValueError("真正 hot start 的前置 OOF 历史训练样本不足 200 行")
        model = make_research_model(candidate).fit(
            features.loc[fit_mask],
            deltas.loc[fit_mask],
            frame.loc[fit_mask, list(candidate.config.targets)],
        )
        parts.append(
            model.predict(
                features.loc[block], frame.loc[block, list(candidate.config.targets)]
            )
        )
    history_predictions = pd.concat(parts, axis=0).reindex(origins)
    history_current = frame.loc[origins, list(candidate.config.targets)]
    return history_predictions, history_current


def _evaluate_research_fold(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    fold: TimeFold,
    candidate: ResearchCandidate,
    cached_generator_all: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """训练一个候选并返回一个外层折的长表预测。"""

    train_mask, validation_mask = fold.masks(frame.index)
    validation_index = frame.index[validation_mask]
    train_current = frame.loc[train_mask, list(candidate.config.targets)]
    validation_current = frame.loc[validation_mask, list(candidate.config.targets)]
    if cached_generator_all is None:
        model = make_research_model(candidate).fit(
            features.loc[train_mask],
            deltas.loc[train_mask],
            train_current,
        )
        base_prediction = model.predict(features.loc[validation_mask], validation_current)
    else:
        # E13 仅改 generator_1 alpha；复用相同 V3 的 OOF 预测，避免重复拟合。
        from gas_forecast.research_models import (
            Generator1HorizonRidgeForecaster,
            merge_target_route_predictions,
        )

        model = make_research_model(candidate)
        if not isinstance(model, Generator1HorizonRidgeForecaster):
            raise ValueError("冻结 generator_all OOF 仅支持逐步长 Ridge 路由")
        model.fit_generator1_only(features.loc[train_mask], deltas.loc[train_mask], train_current)
        generator_1 = model.predict_generator1_only(
            features.loc[validation_mask], validation_current
        )
        base_prediction = merge_target_route_predictions(
            candidate.config,
            generator_1,
            cached_generator_all.reindex(validation_index),
        )
    if candidate.kind == "online_hot_start":
        history_prediction, history_current = _calibration_history_predictions(
            frame, features, deltas, fold, candidate
        )
        online_modes = candidate.online_modes or (candidate.online_mode or "bias",)
        online_kwargs = {
            "half_life": candidate.config.model.online_half_life,
            "bias_clip": candidate.config.model.online_bias_clip,
            "vintage_weight": candidate.config.model.online_vintage_weight,
        }
        if len(online_modes) == 1:
            prediction = apply_online_calibration_hot_start(
                base_prediction,
                frame.loc[validation_mask, list(candidate.config.targets)],
                candidate.config.targets,
                candidate.config.feature.horizons,
                calibration_predictions=history_prediction,
                calibration_current=history_current,
                mode=online_modes[0],
                **online_kwargs,
            )
        else:
            prediction = apply_online_calibration_hot_start_pipeline(
                base_prediction,
                frame.loc[validation_mask, list(candidate.config.targets)],
                candidate.config.targets,
                candidate.config.feature.horizons,
                calibration_predictions=history_prediction,
                calibration_current=history_current,
                modes=online_modes,
                **online_kwargs,
            )
    else:
        prediction = base_prediction
    result = _prediction_long(prediction, validation_index, candidate.config)
    result["fold"] = fold.name
    return result.loc[:, ["fold", "origin_time", "target", "horizon", "prediction"]]


def _load_or_evaluate_fold(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    fold: TimeFold,
    candidate: ResearchCandidate,
    checkpoint_dir: Path | None,
    cached_generator_all: pd.DataFrame | None = None,
) -> pd.DataFrame:
    path = (
        checkpoint_dir / f"{candidate.name}__{fold.name}.csv" if checkpoint_dir is not None else None
    )
    if path is not None and path.exists():
        cached = pd.read_csv(path, parse_dates=["origin_time"])
        required = {"fold", "origin_time", "target", "horizon", "prediction"}
        if required.issubset(cached.columns):
            return cached.loc[:, sorted(required)]
    result = _evaluate_research_fold(
        frame,
        features,
        deltas,
        fold,
        candidate,
        cached_generator_all,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False, encoding="utf-8")
    return result


def _can_reuse_frozen_generator_all(
    source: ResearchCandidate,
    candidate: ResearchCandidate,
) -> bool:
    """判定 E13 alpha 变体能否安全复用基线的 generator_all V3 OOF。"""

    if source.kind != "horizon" or candidate.kind != "horizon":
        return False
    if source.config.model.generator_all_route_model != "v3":
        return False
    if candidate.config.model.generator_all_route_model != "v3":
        return False

    def v3_route_config(config: ForecastConfig) -> ForecastConfig:
        """删除 V3 generator_all 路由不会读取的 generator_1 专项开关。"""

        feature = replace(
            config.feature,
            enable_long_cycle_features=True,
            enable_target_aligned_features=False,
            target_aligned_cycle_days=(),
            enable_slot_one_hot=False,
            enable_time_fourier=False,
            enable_price_delta_features=False,
            enable_price_interactions=False,
            dynamic_feature_scope="none",
            dynamic_lags=(),
            dynamic_rolling_windows=(),
        )
        model = replace(
            config.model,
            generator1_feature_profile="core",
            generator1_short_alpha=None,
            generator1_long_alpha=None,
            ridge_recency_mode="all",
            ridge_hard_window_days=None,
            ridge_half_life_days=None,
            ridge_magnitude_weighting="uniform",
            ridge_loss="ridge",
            weighted_lad_alpha=0.05,
        )
        return replace(config, feature=feature, model=model)

    return v3_route_config(source.config) == v3_route_config(candidate.config)


def _generator_all_predictions_by_fold(
    rows: pd.DataFrame,
    source_name: str,
    folds: list[TimeFold],
    horizons: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    """从已评分的基线长表恢复每个验证折的 generator_all 宽表预测。"""

    prediction_column = f"{source_name}_pred"
    if prediction_column not in rows:
        raise ValueError(f"缺少可复用的 generator_all 基线列: {prediction_column}")
    expected_horizons = [15 * horizon for horizon in horizons]
    output: dict[str, pd.DataFrame] = {}
    for fold in folds:
        subset = rows.loc[
            (rows["fold"] == fold.name) & (rows["target"] == "generator_all"),
            ["origin_time", "horizon", prediction_column],
        ]
        wide = subset.pivot(
            index="origin_time",
            columns="horizon",
            values=prediction_column,
        ).reindex(columns=expected_horizons)
        if len(wide) == 0 or not np.isfinite(wide.to_numpy(dtype=float)).all():
            # blind 没有可评分真值时长表不含该折；该折改为独立拟合 V3。
            continue
        wide.columns = [
            f"generator_all_t+{minutes}_pred" for minutes in expected_horizons
        ]
        output[fold.name] = wide
    return output


def _candidate_comparison(
    rows: pd.DataFrame,
    prediction_column: str,
    baseline_column: str,
    *,
    scope: str,
) -> dict[str, object]:
    """返回计划要求的 pooled、目标、折、日块和近期稳定性诊断。"""

    report = score_oof_long(rows, prediction_column)
    if prediction_column == baseline_column:
        comparison = rows.loc[:, ["fold", "origin_time", "actual", prediction_column]].copy()
        comparison["difference"] = 0.0
    else:
        comparison = rows.loc[
            :, ["fold", "origin_time", "actual", prediction_column, baseline_column]
        ].copy()
        candidate_error = absolute_percentage_error(comparison["actual"], comparison[prediction_column])
        baseline_error = absolute_percentage_error(comparison["actual"], comparison[baseline_column])
        comparison["difference"] = candidate_error - baseline_error
    by_fold = comparison.groupby("fold", sort=True)["difference"].mean()
    by_day = comparison.assign(day=pd.to_datetime(comparison["origin_time"]).dt.date).groupby(
        "day", sort=True
    )["difference"].mean()
    recent = by_fold.tail(5)
    generator_rows = rows.loc[rows["target"] == "generator_1"]
    generator_difference = absolute_percentage_error(
        generator_rows["actual"], generator_rows[prediction_column]
    ) - absolute_percentage_error(generator_rows["actual"], generator_rows[baseline_column])
    pooled_difference = float(comparison["difference"].mean())
    formal = (
        scope == "final"
        and pooled_difference < 0.0
        and float(generator_difference.mean()) <= 0.0
        and int((by_fold < 0.0).sum()) >= int(np.ceil(len(by_fold) / 2))
        and float(by_fold.max()) <= 0.001
    )
    research = pooled_difference <= -0.00005 and float(generator_difference.mean()) <= 0.0
    return {
        **report,
        "baseline_column": baseline_column,
        "pooled_difference": pooled_difference,
        "generator_1_difference": float(generator_difference.mean()),
        "fold_wins": int((by_fold < 0.0).sum()),
        "day_block_wins": int((by_day < 0.0).sum()),
        "recent_5_folds_difference": {
            str(name): float(value) for name, value in recent.items()
        },
        "worst_fold_regression": float(by_fold.max()),
        "research_candidate": bool(research),
        "formal_candidate": bool(formal),
        "next_action": (
            "可以进入正式候选审计"
            if formal
            else "保留为研究候选，等待完整开发/盲测验收"
            if research and scope != "final"
            else "拒绝或保持当前基线"
        ),
    }


def _capacity_projection_audit(
    rows: pd.DataFrame,
    raw_column: str,
    projected_column: str,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    """量化容量投影修改的单元数及其对统一 MAPE 的影响。"""

    wide_raw = rows.pivot(
        index=["fold", "origin_time"], columns=["target", "horizon"], values=raw_column
    )
    violations: dict[str, int] = {
        "generator_1_below_zero": 0,
        "generator_1_above_200": 0,
        "generator_all_below_zero": 0,
        "generator_all_above_440": 0,
        "generator_all_below_generator_1": 0,
        "generator_rest_above_240": 0,
    }
    for horizon in horizons:
        minutes = 15 * horizon
        generator_1 = wide_raw[("generator_1", minutes)]
        generator_all = wide_raw[("generator_all", minutes)]
        violations["generator_1_below_zero"] += int((generator_1 < 0.0).sum())
        violations["generator_1_above_200"] += int((generator_1 > 200.0).sum())
        violations["generator_all_below_zero"] += int((generator_all < 0.0).sum())
        violations["generator_all_above_440"] += int((generator_all > 440.0).sum())
        violations["generator_all_below_generator_1"] += int((generator_all < generator_1).sum())
        violations["generator_rest_above_240"] += int((generator_all - generator_1 > 240.0).sum())
    changed = ~np.isclose(rows[raw_column], rows[projected_column], equal_nan=True)
    return {
        "raw_violations": violations,
        "raw_violation_cells": int(sum(violations.values())),
        "modified_cells": int(changed.sum()),
        "raw_pooled_mape": score_oof_long(rows, raw_column)["pooled_mape"],
        "projected_pooled_mape": score_oof_long(rows, projected_column)["pooled_mape"],
    }


def build_research_oof(
    frame: pd.DataFrame,
    price_schedule: PriceSchedule | None,
    candidates: list[ResearchCandidate],
    *,
    scope: str = "screening",
    n_jobs: int = 1,
    checkpoint_dir: str | Path | None = None,
    baseline_name: str | None = None,
) -> ResearchOOFResult:
    """以相同外层折运行研究候选，默认不接触 blind。"""

    if not candidates:
        raise ValueError("研究 OOF 至少需要一个候选")
    if n_jobs < 1:
        raise ValueError("n_jobs 必须大于等于 1")
    reference = candidates[0].config
    for candidate in candidates:
        if candidate.config.targets != reference.targets:
            raise ValueError("同一研究 OOF 的候选必须使用相同目标集合")
        if candidate.config.feature.horizons != reference.feature.horizons:
            raise ValueError("同一研究 OOF 的候选必须使用相同步长集合")
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("研究候选名称必须唯一")
    baseline_name = baseline_name or candidates[0].name
    baseline_candidate = next(
        (candidate for candidate in candidates if candidate.name == baseline_name),
        None,
    )
    if baseline_candidate is None:
        raise ValueError(f"指定的基线候选不存在: {baseline_name}")
    folds = select_research_folds(frame.index, reference, scope=scope)
    if not folds:
        raise ValueError("研究 OOF 没有可用外层折")
    started = time.perf_counter()
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    # Alpha、损失权重等消融会共享同一特征配置；缓存可避免重复计算长窗口特征。
    feature_cache: dict[object, pd.DataFrame] = {}

    def resolve_features(config: ForecastConfig) -> pd.DataFrame:
        feature_config = research_feature_superset(config.feature)
        cached = feature_cache.get(feature_config)
        if cached is None:
            cached = build_causal_features(frame, feature_config, price_schedule)
            feature_cache[feature_config] = cached
        return cached

    base_features = resolve_features(reference)
    row_parts = []
    for fold in folds:
        _, validation_mask = fold.masks(frame.index)
        rows = _base_fold_rows(frame, fold, validation_mask, reference)
        row_parts.append(attach_regimes(rows, base_features.loc[validation_mask]))
    rows = pd.concat(row_parts, ignore_index=True).sort_values(
        ["fold", "origin_time", "target", "horizon"], kind="stable"
    )
    keys = ["fold", "origin_time", "target", "horizon"]
    row_index = pd.MultiIndex.from_frame(rows[keys])
    for candidate in candidates:
        features = resolve_features(candidate.config)
        deltas = build_delta_targets(
            frame, candidate.config.targets, candidate.config.feature.horizons
        )
        reusable_generator_all = (
            _generator_all_predictions_by_fold(
                rows,
                baseline_name,
                folds,
                candidate.config.feature.horizons,
            )
            if f"{baseline_name}_pred" in rows
            and _can_reuse_frozen_generator_all(baseline_candidate, candidate)
            else None
        )
        if n_jobs == 1:
            parts = [
                _load_or_evaluate_fold(
                    frame,
                    features,
                    deltas,
                    fold,
                    candidate,
                    checkpoint_path,
                    reusable_generator_all.get(fold.name)
                    if reusable_generator_all is not None
                    else None,
                )
                for fold in folds
            ]
        else:
            parts = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_load_or_evaluate_fold)(
                    frame,
                    features,
                    deltas,
                    fold,
                    candidate,
                    checkpoint_path,
                    reusable_generator_all.get(fold.name)
                    if reusable_generator_all is not None
                    else None,
                )
                for fold in folds
            )
        predicted = pd.concat(parts, ignore_index=True).set_index(keys)["prediction"]
        values = predicted.reindex(row_index).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"候选 {candidate.name} 未覆盖全部 OOF 行")
        rows[f"{candidate.name}_pred"] = values
    baseline_column = f"{baseline_name}_pred"
    if baseline_column not in rows:
        raise ValueError(f"指定的基线候选不存在: {baseline_name}")
    reports = {
        candidate.name: _candidate_comparison(
            rows,
            f"{candidate.name}_pred",
            baseline_column,
            scope=scope,
        )
        for candidate in candidates
    }
    capacity_candidates = [candidate for candidate in candidates if candidate.kind == "capacity"]
    capacity_audit: dict[str, object] | None = None
    raw = next(
        (
            candidate
            for candidate in capacity_candidates
            if not candidate.config.model.apply_capacity_projection
        ),
        None,
    )
    projected = next(
        (
            candidate
            for candidate in capacity_candidates
            if candidate.config.model.apply_capacity_projection
        ),
        None,
    )
    if raw is not None and projected is not None:
        capacity_audit = _capacity_projection_audit(
            rows,
            f"{raw.name}_pred",
            f"{projected.name}_pred",
            reference.feature.horizons,
        )
    duration = time.perf_counter() - started
    report: dict[str, object] = {
        "scope": scope,
        "folds": [fold.name for fold in folds],
        "blind_included": any(fold.blind for fold in folds),
        "baseline": baseline_name,
        "candidates": {
            candidate.name: {
                "experiment_id": candidate.experiment_id,
                "kind": candidate.kind,
                "description": candidate.description,
                "config": asdict(candidate.config),
            }
            for candidate in candidates
        },
        "models": reports,
        "capacity_projection_audit": capacity_audit,
        "duration_seconds": duration,
    }
    return ResearchOOFResult(rows=rows.reset_index(drop=True), report=report, duration_seconds=duration)
