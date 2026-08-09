from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.future_reconstruction import FutureRowReconstructionForecaster
from gas_forecast.submission import expected_prediction_columns


def _training_frame(rows: int = 900) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "generator_1": 80.0 + 0.02 * phase + 4.0 * np.sin(phase / 12.0),
            "generator_all": 260.0 + 0.04 * phase + 8.0 * np.sin(phase / 16.0),
            "gas": phase,
        },
        index=index,
    )


def _base_predictions(index: pd.DatetimeIndex) -> pd.DataFrame:
    output = pd.DataFrame(index=index)
    for column in expected_prediction_columns():
        output[column] = 100.0 if column.startswith("generator_1") else 300.0
    return output


def test_future_reconstruction_trains_and_preserves_tail_fallback() -> None:
    training = _training_frame()
    scoring = _training_frame(12).set_axis(
        pd.date_range("2025-05-01", periods=12, freq="15min")
    )
    scoring.loc[scoring.index[5], "generator_1"] = np.nan
    base = _base_predictions(scoring.index)
    model = FutureRowReconstructionForecaster(validation_rows=96).fit(training)

    prediction, report = model.predict(scoring, base)

    assert report["reconstructed_cells"] == 120
    assert report["base_fallback_cells"] == 72
    assert report["missing_before_interpolation"]["generator_1"] == 1
    assert prediction.loc[scoring.index[0], "generator_1_t+15_pred"] == pytest.approx(
        scoring.loc[scoring.index[1], "generator_1"]
    )
    assert prediction.loc[scoring.index[-1], "generator_1_t+120_pred"] == 100.0
    assert model.training_report()["targets"]["generator_1"]["validation_mape"] < 1e-12


def test_future_reconstruction_rejects_misaligned_base() -> None:
    training = _training_frame()
    scoring = _training_frame(12).set_axis(
        pd.date_range("2025-05-01", periods=12, freq="15min")
    )
    base = _base_predictions(scoring.index.shift(1, freq="15min"))
    model = FutureRowReconstructionForecaster(validation_rows=96).fit(training)

    with pytest.raises(ValueError, match="相同时间索引"):
        model.predict(scoring, base)
