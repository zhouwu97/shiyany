from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.matured_residual import (
    MaturedResidualConfig,
    MaturedResidualState,
    audit_matured_residual_future_perturbation,
    build_matured_residual_oof,
    predict_matured_residual_at_origin,
)


def _ledger() -> pd.DataFrame:
    """构造一个每 15 分钟一条 t+15 OOF 预测的人工台账。"""

    origin = pd.date_range("2025-02-01", periods=5, freq="15min")
    rows = pd.DataFrame(
        {
            "fold": ["dev_01"] * len(origin),
            "origin_time": origin,
            "train_end": origin - pd.Timedelta(minutes=15),
            "target": ["generator_1"] * len(origin),
            "horizon": [15] * len(origin),
            "prediction": [100.0] * len(origin),
            # 依次在下一 origin 成熟为 +10、-10、-20、0；最后一条尚未成熟。
            "actual": [110.0, 90.0, 80.0, 100.0, 777.0],
            "future_feature": np.arange(len(origin), dtype=float),
        }
    )
    rows["target_datetime"] = rows["origin_time"] + pd.Timedelta(minutes=15)
    return rows


def _config() -> MaturedResidualConfig:
    return MaturedResidualConfig(horizons=(15,), ewma_alpha=0.5, slope_window=3)


def test_matured_ledger_updates_at_exact_target_datetime_with_expected_state() -> None:
    rows = _ledger()
    result = build_matured_residual_oof(rows, config=_config())

    observed = result.rows.loc[
        :,
        [
            "matured_error_count",
            "latest_matured_error",
            "error_ewma",
            "error_slope",
            "consecutive_overestimate_count",
            "consecutive_underestimate_count",
            "matured_residual_pred",
        ],
    ]
    expected = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
            [1.0, 10.0, 10.0, 0.0, 0.0, 1.0, 110.0],
            [2.0, -10.0, 0.0, -20.0, 1.0, 0.0, 100.0],
            [3.0, -20.0, -10.0, -15.0, 2.0, 0.0, 90.0],
            [4.0, 0.0, -5.0, 5.0, 0.0, 0.0, 95.0],
        ]
    )
    np.testing.assert_allclose(observed.to_numpy(dtype=float), expected, rtol=0.0, atol=0.0)
    assert result.report["maturity_rule"] == (
        "earlier_origin and target_datetime == current_origin"
    )
    assert result.report["in_sample_residual_used"] is False


def test_state_only_settles_exactly_due_earlier_oof_predictions() -> None:
    rows = _ledger()
    origin = rows["origin_time"]
    delayed = rows.loc[[0]].copy()
    delayed["horizon"] = 30
    delayed["target_datetime"] = delayed["origin_time"] + pd.Timedelta(minutes=30)
    # 同一 target_datetime 的实际值在不同 horizon 台账行中必须相同。
    delayed["actual"] = 90.0
    ledger = pd.concat([rows, delayed], ignore_index=True)
    config = MaturedResidualConfig(horizons=(15, 30), ewma_alpha=0.5, slope_window=3)
    state = MaturedResidualState(config)

    assert state.advance(origin.iloc[0], ledger) == 0
    # t+30 标签尚未到达，不能提前影响当前 horizon 的状态。
    assert state.correction("generator_1", 15) == 0.0
    assert state.correction("generator_1", 30) == 0.0
    assert state.advance(origin.iloc[1], ledger) == 1
    assert state.correction("generator_1", 15) == 10.0
    assert state.correction("generator_1", 30) == 0.0
    assert state.advance(origin.iloc[2], ledger) == 2
    assert state.correction("generator_1", 30) == -10.0


def test_future_actuals_features_and_rows_do_not_change_current_origin_output() -> None:
    rows = _ledger()
    origin = rows.loc[3, "origin_time"]
    expected = predict_matured_residual_at_origin(rows, origin, config=_config())
    assert "actual" not in expected.columns

    changed = rows.copy()
    changed.loc[changed["target_datetime"] > origin, "actual"] = -999_999.0
    changed.loc[changed["origin_time"] > origin, "future_feature"] = -999_999.0
    observed = predict_matured_residual_at_origin(changed, origin, config=_config())
    pd.testing.assert_frame_equal(expected, observed)

    deleted = rows.loc[rows["origin_time"] <= origin].copy()
    after_delete = predict_matured_residual_at_origin(deleted, origin, config=_config())
    pd.testing.assert_frame_equal(expected, after_delete)

    audit = audit_matured_residual_future_perturbation(rows, origin=origin, config=_config())
    assert audit["passed"] is True
    assert set(audit["cases"]) == {"modified", "nulled", "deleted"}


def test_oof_builder_rejects_in_sample_residuals() -> None:
    rows = _ledger()
    rows["train_end"] = rows["origin_time"]

    with pytest.raises(ValueError, match="in-sample"):
        build_matured_residual_oof(rows, config=_config())


def test_oof_builder_can_emit_generic_prediction_for_later_fusion() -> None:
    result = build_matured_residual_oof(
        _ledger(), config=_config(), output_column="prediction"
    )

    assert {
        "fold",
        "origin_time",
        "train_end",
        "target",
        "horizon",
        "actual",
        "prediction",
        "matured_residual_base_prediction",
    }.issubset(result.rows.columns)
    assert result.rows.loc[1, "matured_residual_base_prediction"] == 100.0
    assert result.rows.loc[1, "prediction"] == 110.0
