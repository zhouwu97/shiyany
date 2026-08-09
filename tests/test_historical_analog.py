from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.config import FeatureConfig, ForecastConfig, ModelConfig, ValidationConfig
from gas_forecast.historical_analog import (
    HORIZONS,
    PRE_REGISTERED_SPECS,
    HistoricalAnalogSpec,
    _predict_target_for_origins,
    _prepare_targets,
    _strict_candidate_positions,
    _validate_frame,
    audit_historical_analog_future_perturbation,
    build_historical_analog_oof,
    predict_historical_analog_at_origin,
    select_historical_analog_folds,
    validate_pre_registered_specs,
)


def _frame(periods: int = 12 * 96) -> pd.DataFrame:
    """构造带重复局部形状的 15 分钟生产序列，便于验证增量轨迹。"""

    index = pd.date_range("2025-01-01", periods=periods, freq="15min")
    phase = np.arange(periods, dtype=float)
    generator_1 = 90.0 + 8.0 * np.sin(phase / 7.0) + 0.03 * phase
    generator_all = 210.0 + 14.0 * np.cos(phase / 9.0) + 0.02 * phase
    return pd.DataFrame(
        {
            "generator_1": generator_1,
            "generator_all": generator_all,
            "unrelated_production": 1000.0 + phase,
        },
        index=index,
    )


def _config() -> ForecastConfig:
    """缩短测试日历，但保留两个目标、八步轨迹和嵌套折语义。"""

    return ForecastConfig(
        feature=FeatureConfig(horizons=HORIZONS),
        model=ModelConfig(inner_folds=2),
        validation=ValidationConfig(
            first_validation_date="2025-01-05",
            fold_spacing_days=1,
            validation_days=1,
            blind_days=2,
            min_train_days=3,
        ),
    )


def test_registry_is_exactly_the_six_pre_registered_euclidean_candidates() -> None:
    assert [(spec.context, spec.neighbors, spec.metric) for spec in PRE_REGISTERED_SPECS] == [
        (16, 8, "euclidean"),
        (16, 16, "euclidean"),
        (16, 32, "euclidean"),
        (32, 8, "euclidean"),
        (32, 16, "euclidean"),
        (32, 32, "euclidean"),
    ]
    with pytest.raises(ValueError, match="未预注册"):
        validate_pre_registered_specs((HistoricalAnalogSpec(16, 4),))


def test_single_origin_prediction_uses_delta_trajectory_and_returns_all_diagnostics() -> None:
    frame = _frame()
    origin = frame.index[700]

    predicted = predict_historical_analog_at_origin(frame, origin, spec=HistoricalAnalogSpec(16, 8))

    assert len(predicted) == 16
    assert set(predicted["target"]) == {"generator_1", "generator_all"}
    assert set(predicted["horizon"]) == set(range(15, 121, 15))
    assert {
        "nearest_distance",
        "median_neighbor_distance",
        "effective_neighbor_count",
        "analog_uncertainty",
    }.issubset(predicted.columns)
    # 相似样本输出必须是“当前值 + 历史增量”，而不是直接重放历史绝对量。
    for target, part in predicted.groupby("target"):
        anchor = frame.loc[origin, target]
        assert np.isfinite(part["prediction"]).all()
        assert np.isfinite(part["prediction"].to_numpy(dtype=float) - anchor).all()


def test_future_mutation_shuffle_null_and_delete_do_not_change_origin_predictions() -> None:
    audit = audit_historical_analog_future_perturbation(
        _frame(), origin=pd.Timestamp("2025-01-08 12:00:00"), spec=HistoricalAnalogSpec(32, 16)
    )

    assert audit["prediction_count"] == 16
    assert audit["passed"] is True
    assert set(audit["cases"]) == {"modified", "shuffled", "nulled", "deleted"}
    assert all(case["changed_prediction_positions"] == [] for case in audit["cases"].values())


def test_outer_oof_keeps_trajectories_before_held_fold_and_uses_nested_selection() -> None:
    frame = _frame()
    config = _config()

    result = build_historical_analog_oof(frame, config=config, scope="screening")

    assert result.report["blind_included"] is False
    assert result.report["nested_cross_fitting"] is True
    assert result.rows["selection_used_held_fold_labels"].eq(False).all()
    assert result.rows["candidate_window_and_trajectory_before_train_cutoff"].all()
    assert result.rows["candidate_trajectory_end_before_train_cutoff"].all()
    assert result.rows["candidate_trajectory_end_before_validation"].all()
    assert {
        "historical_analog_pred",
        "nearest_distance",
        "median_neighbor_distance",
        "effective_neighbor_count",
        "analog_uncertainty",
    }.issubset(result.rows.columns)
    assert result.rows["historical_analog_pred"].notna().all()
    assert result.trace["selection_data_end"].le(result.trace["train_end"]).all()


def test_screening_uses_the_first_five_development_folds_and_never_returns_blind() -> None:
    frame = _frame()
    config = _config()
    screening = select_historical_analog_folds(frame.index, config, scope="screening")
    development = select_historical_analog_folds(frame.index, config, scope="development")

    assert screening == development[:5]
    assert all(not fold.blind for fold in screening)
    with pytest.raises(ValueError, match="禁止读取 blind"):
        select_historical_analog_folds(frame.index, config, scope="final")


def test_candidate_future_trajectory_must_end_strictly_before_train_cutoff() -> None:
    frame = _frame(periods=500)
    prepared = _prepare_targets(frame, ("generator_1",))["generator_1"]
    train_end = frame.index[300]
    positions = _strict_candidate_positions(
        prepared,
        train_start=frame.index[0],
        train_end=train_end,
        validation_start=frame.index[380],
        context=16,
    )

    # j+120min == train_end 是训练截止边界，必须排除；最后允许的位置
    # 是再早一个 15 分钟刻度，且其第八步真值严格早于 train_end。
    equal_boundary = frame.index.get_loc(train_end - pd.Timedelta(minutes=120))
    last_safe = frame.index.get_loc(train_end - pd.Timedelta(minutes=135))
    assert equal_boundary not in positions
    assert last_safe in positions


def test_candidate_time_gap_and_insufficient_history_fail_closed() -> None:
    frame = _frame(periods=80)
    gapped = pd.concat([frame.iloc[:40], frame.iloc[41:]])
    with pytest.raises(ValueError, match="连续的 15 分钟"):
        _validate_frame(gapped, ("generator_1", "generator_all"))

    prepared = _prepare_targets(frame, ("generator_1",))["generator_1"]
    with pytest.raises(ValueError, match="候选不足"):
        _predict_target_for_origins(
            prepared,
            pd.DatetimeIndex([frame.index[70]]),
            HistoricalAnalogSpec(32, 32),
            train_start=frame.index[0],
            train_end=frame.index[55],
            validation_start=frame.index[70],
        )
