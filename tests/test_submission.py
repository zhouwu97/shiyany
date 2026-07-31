from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gas_forecast.submission import (
    expected_prediction_columns,
    export_legacy_json,
    package_submission,
    validate_submission_frame,
)


def _valid_frame() -> pd.DataFrame:
    rows = 10
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-05-01", periods=rows, freq="15min").astype(str),
        }
    )
    for column in expected_prediction_columns():
        frame[column] = 100.0 if column.startswith("generator_1") else 250.0
    return frame


def test_submission_validation_and_archive_contract(tmp_path: Path) -> None:
    frame = _valid_frame()
    result_path = tmp_path / "s_result.csv"
    archive_path = tmp_path / "team_gas_predict_prelim.zip"
    frame.to_csv(result_path, index=False)

    summary = package_submission(result_path, archive_path)

    assert summary["rows"] == len(frame)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["result.csv"]

    json_path = tmp_path / "legacy.json"
    export_legacy_json(result_path, json_path)
    assert '"columns"' in json_path.read_text(encoding="utf-8")


def test_submission_rejects_non_finite_prediction() -> None:
    frame = _valid_frame()
    frame.loc[0, "generator_1_t+15_pred"] = np.nan
    with pytest.raises(ValueError, match="非有限值"):
        validate_submission_frame(frame)
