from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.online import apply_online_calibration, apply_online_calibration_to_oof


def _inputs(rows: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    current = pd.DataFrame(
        {"generator_1": np.full(rows, 100.0), "generator_all": np.full(rows, 220.0)},
        index=index,
    )
    predictions = {}
    for target, value in (("generator_1", 110.0), ("generator_all", 230.0)):
        for horizon in (1, 2):
            predictions[f"{target}_t+{15 * horizon}_pred"] = np.full(rows, value)
    return pd.DataFrame(predictions, index=index), current


def test_bias_calibration_uses_only_matured_predictions() -> None:
    base, current = _inputs()
    calibrated = apply_online_calibration(
        base, current, ("generator_1", "generator_all"), (1, 2), mode="bias", half_life=1
    )

    # 第一个时刻没有任何成熟预测；第二个时刻才揭晓第一步预测误差。
    assert calibrated.iloc[0, 0] == 110.0
    assert calibrated.iloc[1, 0] < 110.0


def test_gain_calibration_is_grouped_by_prediction_horizon() -> None:
    base, current = _inputs()
    calibrated = apply_online_calibration(
        base,
        current,
        ("generator_1", "generator_all"),
        (1, 2),
        mode="gain",
        half_life=1,
    )

    # t+15 已在第二行成熟并学习到“预测增量应收缩为 0”；t+30 尚未成熟。
    assert calibrated.loc[base.index[1], "generator_1_t+15_pred"] == 100.0
    assert calibrated.loc[base.index[1], "generator_1_t+30_pred"] == 110.0


def test_online_calibration_is_causal_under_future_actual_perturbation() -> None:
    base, current = _inputs()
    changed = current.copy()
    changed.iloc[5:, :] = -999_999.0
    baseline = apply_online_calibration(
        base, current, ("generator_1", "generator_all"), (1, 2), mode="bias"
    )
    perturbed = apply_online_calibration(
        base, changed, ("generator_1", "generator_all"), (1, 2), mode="bias"
    )
    pd.testing.assert_frame_equal(baseline.iloc[:5], perturbed.iloc[:5])


def test_vintage_mode_fuses_previous_prediction_for_same_target_time() -> None:
    base, current = _inputs(rows=3)
    base.loc[base.index[0], "generator_1_t+30_pred"] = 100.0
    base.loc[base.index[1], "generator_1_t+15_pred"] = 120.0
    calibrated = apply_online_calibration(
        base,
        current,
        ("generator_1", "generator_all"),
        (1, 2),
        mode="vintage",
        vintage_weight=0.5,
    )

    assert calibrated.loc[base.index[1], "generator_1_t+15_pred"] == 110.0


def test_oof_adapter_cold_starts_each_outer_fold_and_marks_hot_warmup() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="15min")
    rows = pd.DataFrame(
        {
            "fold": ["fold_a", "fold_a", "fold_b", "fold_b"],
            "origin_time": index,
            "target": ["generator_1"] * 4,
            "horizon": [15] * 4,
            "actual": [100.0] * 4,
            "current_value": [100.0] * 4,
            "v1_pred": [110.0] * 4,
        }
    )

    cold = apply_online_calibration_to_oof(
        rows,
        "v1_pred",
        ("generator_1",),
        (1,),
        mode="bias",
        half_life=1,
    )
    assert cold.loc[2, "v1_online_bias_pred"] == 110.0
    assert cold["v1_online_bias_pred_is_warmup"].eq(False).all()
    assert cold["v1_online_bias_pred_is_fallback"].eq(False).all()

    hot = apply_online_calibration_to_oof(
        rows,
        "v1_pred",
        ("generator_1",),
        (1,),
        mode="bias",
        half_life=1,
        warmup_rows=1,
    )
    assert hot["v1_online_bias_pred_is_warmup"].tolist() == [True, False, True, False]
