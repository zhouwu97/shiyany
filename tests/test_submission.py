from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gas_forecast.freeze import compare_reproductions
from gas_forecast.submission import (
    SUBMISSION_MEMBERS,
    expected_prediction_columns,
    export_legacy_json,
    package_submission,
    validate_submission_archive,
    validate_submission_frame,
    validate_submission_input,
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
    input_frame = pd.DataFrame(
        {
            "datetime": frame["datetime"],
            "feature_a": np.arange(len(frame), dtype=float),
            "feature_b": 1.0,
        }
    )
    input_path = tmp_path / "input.csv"
    result_path = tmp_path / "s_result.csv"
    archive_path = tmp_path / "team_gas_predict_prelim.zip"
    input_frame.to_csv(input_path, index=False)
    frame.to_csv(result_path, index=False)

    summary = package_submission(input_path, result_path, archive_path)

    assert summary["rows"] == len(frame)
    assert summary["input_columns"] == 2
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == SUBMISSION_MEMBERS
    assert validate_submission_archive(archive_path)["valid"] is True

    first_bytes = archive_path.read_bytes()
    package_submission(input_path, result_path, archive_path)
    assert archive_path.read_bytes() == first_bytes

    json_path = tmp_path / "legacy.json"
    export_legacy_json(result_path, json_path)
    assert '"columns"' in json_path.read_text(encoding="utf-8")


def test_submission_rejects_non_finite_prediction() -> None:
    frame = _valid_frame()
    frame.loc[0, "generator_1_t+15_pred"] = np.nan
    with pytest.raises(ValueError, match="非有限值"):
        validate_submission_frame(frame)


def test_submission_rejects_misaligned_input() -> None:
    result = _valid_frame()
    input_frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-05-02", periods=len(result), freq="15min"),
            "feature": 1.0,
        }
    )
    with pytest.raises(ValueError, match="行数或时间戳不一致"):
        validate_submission_input(input_frame, result)


def test_reproduction_comparison_requires_all_frozen_fields() -> None:
    manifest = {
        "model_version": "v2",
        "model_sha256": "a",
        "input_sha256": "input",
        "result_sha256": "b",
        "zip_sha256": "c",
        "zip_input_sha256": "zip-input",
        "zip_result_sha256": "d",
        "selection_sha256": "e",
        "requirements_lock_sha256": "f",
        "rows": 192,
        "prediction_columns": 16,
        "archive_members": list(SUBMISSION_MEMBERS),
    }
    assert compare_reproductions(manifest, manifest)["identical"] is True
    changed = {**manifest, "zip_sha256": "different"}
    assert compare_reproductions(manifest, changed)["identical"] is False
