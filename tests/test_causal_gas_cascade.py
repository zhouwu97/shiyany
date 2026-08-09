from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.causal_gas_cascade import (
    CascadeConfig,
    CausalGasCascadeForecaster,
    RESOURCE_NAMES,
    build_resource_frame,
    future_perturbation_audit,
    resolve_resource_mapping,
)


def _frame(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    # 使用真实项目中出现过的分项列名，验证映射不依赖字段顺序。
    values: dict[str, np.ndarray] = {
        "generator_1": 100.0 + 0.02 * phase + 2.0 * np.sin(phase / 11),
        "generator_all": 220.0 + 0.03 * phase + 3.0 * np.sin(phase / 13),
        "blast_furnace_1": 500_000.0 + 20.0 * phase,
        "blast_furnace_2": 300_000.0 + 10.0 * phase,
        "coke_oven_1": 50_000.0 + 8.0 * phase,
        "converter_1": 30_000.0 + 4.0 * phase,
        "blast_furnace_gas_holder_2": 100_000.0 + 5.0 * phase,
        "blast_furnace_user1": 40_000.0 + phase,
        "blast_furnace_user2": 20_000.0 + phase,
        "converter_user1": 4_000.0 + phase,
        "air_heater_1": 10_000.0 + phase,
        "into_gas_mixed_blast_furnace": 2_000.0 + phase,
        "into_gas_mixed_converter": 1_000.0 + phase,
        "generator_use_blast_furnace_gas": 100_000.0 + 3.0 * phase,
        "generator_use_coke_gas": 15_000.0 + 2.0 * phase,
        "generator_use_converter_gas": 8_000.0 + phase,
    }
    return pd.DataFrame(values, index=index)


def _config() -> CascadeConfig:
    return CascadeConfig(
        horizons=(1, 2, 3, 4),
        inner_folds=2,
        outer_folds=2,
        purge_steps=5,
        min_train_rows=48,
        min_validation_rows=20,
        ridge_alpha=5.0,
    )


def test_resource_mapping_is_stable_and_aggregates_components() -> None:
    frame = _frame(40)
    shuffled = frame.loc[:, list(reversed(frame.columns))]
    mapping = resolve_resource_mapping(shuffled.columns)
    assert mapping.columns["blast_furnace_gas"] == ("blast_furnace_1", "blast_furnace_2")
    assert mapping.columns["major_users"] == (
        "air_heater_1",
        "blast_furnace_user1",
        "blast_furnace_user2",
        "converter_user1",
        "into_gas_mixed_blast_furnace",
        "into_gas_mixed_converter",
    )
    resources = build_resource_frame(shuffled, mapping)
    np.testing.assert_allclose(
        resources["blast_furnace_gas"].to_numpy(),
        (frame["blast_furnace_1"] + frame["blast_furnace_2"]).to_numpy(),
    )
    assert tuple(resources.columns) == RESOURCE_NAMES


def test_cascade_nested_oof_and_inference_semantics() -> None:
    frame = _frame()
    model = CausalGasCascadeForecaster(_config()).fit(frame)
    assert model.trace_ is not None
    assert model.trace_.stage2_source == "stage1_inner_held_fold_oof"
    assert model.stage1_oof_ is not None
    assert model.stage1_oof_.columns[0].startswith("stage1_pred_")
    # Stage2 训练矩阵中的 future gas 输入必须来自 OOF 列，不是 actual 资源表。
    assert model.stage2_training_features_ is not None
    future_columns = [column for column in model.stage2_training_features_ if "tplus" in column]
    assert future_columns
    assert model.stage2_training_features_.loc[:, future_columns].equals(
        model.stage1_oof_.rename(columns=lambda value: value.replace("_t+", "_tplus_"))
    )
    prediction = model.predict(frame.iloc[-12:])
    assert list(prediction.columns) == model.prediction_columns()
    assert prediction.shape == (12, 8)
    assert np.isfinite(prediction.to_numpy()).all()


def test_batch_and_origin_prediction_match() -> None:
    frame = _frame()
    model = CausalGasCascadeForecaster(_config()).fit(frame)
    origins = frame.index[-3:]
    # 批量推理必须保留 origin 以前的历史状态，再截取目标行。
    batch = model.predict(frame.loc[: origins[-1]]).loc[origins]
    by_origin = model.predict_at_origins(frame, origins)
    np.testing.assert_allclose(
        batch.to_numpy(dtype=float),
        by_origin.loc[:, model.prediction_columns()].to_numpy(dtype=float),
    )


def test_future_perturbation_keeps_all_sixteen_predictions() -> None:
    frame = _frame()
    model = CausalGasCascadeForecaster(_config()).fit(frame)
    origins = frame.index[-5:-2]
    audit = future_perturbation_audit(model, frame, origins)
    assert audit["passed"] is True
    assert audit["max_abs_difference"] == 0.0
    assert len(audit["prediction_columns"]) == 8


def test_stage2_rejects_unregistered_stage1_source() -> None:
    frame = _frame()
    model = CausalGasCascadeForecaster(_config()).fit(frame)
    assert model.stage2_training_features_ is not None
    with pytest.raises(ValueError, match="Stage1 OOF 来源"):
        # 通过内部契约测试，防止调用者伪造 OOF 来源。
        from gas_forecast.causal_gas_cascade import Stage1PredictionBundle, _stage2_features

        bad = Stage1PredictionBundle(
            values=model.stage1_oof_.copy(),
            source="actual_future_values",
            is_oof=True,
            resource_names=RESOURCE_NAMES,
            horizons=_config().horizons,
        )
        _stage2_features(model._bundle.state, bad, _config())
