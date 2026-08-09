from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gas_forecast.submission import (
    expected_prediction_columns,
    package_submission,
    validate_submission_archive,
    validate_submission_input,
)
from gas_forecast.submission_quality import (
    COMPETITION_QUALITY_POLICY,
    audit_submission_quality,
    enforce_submission_quality,
    prepare_full_matrix_submission_input,
    prepare_submission_input,
)


def _policy():
    return replace(
        COMPETITION_QUALITY_POLICY,
        name="test_quality_policy",
        allowed_raw_columns=("generator_1",),
        required_raw_columns=("generator_1",),
        batch_iqr_clip_columns=("generator_1",),
    )


def _input_frame(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-05-01", periods=rows, freq="15min").astype(str),
            "generator_1": [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 1000.0],
            "air_heater_5": 0.0,
            "feat_generator_1_lag_1": np.arange(rows, dtype=float),
        }
    )


def _result_frame(rows: int = 8) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"datetime": pd.date_range("2025-05-01", periods=rows, freq="15min").astype(str)}
    )
    for column in expected_prediction_columns():
        frame[column] = 100.0 if column.startswith("generator_1") else 250.0
    return frame


def test_quality_policy_removes_invalid_raw_and_clips_registered_outlier() -> None:
    policy = _policy()
    raw = _input_frame()

    before = audit_submission_quality(raw, policy)
    assert before["unexpected_raw_columns"] == ["air_heater_5"]
    assert before["constant_raw_columns"] == ["air_heater_5"]
    with pytest.raises(ValueError, match="未登记原始字段"):
        enforce_submission_quality(raw, policy)

    prepared, report = prepare_submission_input(raw, policy)

    values = raw["generator_1"].to_numpy(float)
    lower, upper = np.quantile(values, [0.25, 0.75])
    expected_upper = upper + (upper - lower)
    assert "air_heater_5" not in prepared.columns
    assert "feat_generator_1_lag_1" in prepared.columns
    assert prepared["generator_1"].iloc[-1] == pytest.approx(expected_upper)
    assert report["dropped_raw_columns"] == ["air_heater_5"]
    assert report["repaired_cells"] == 1
    assert report["audit"]["total_iqr_violations"] == 0


def test_quality_policy_is_enforced_by_submission_archive(tmp_path: Path) -> None:
    policy = _policy()
    input_path = tmp_path / "input.csv"
    result_path = tmp_path / "s_result.csv"
    archive_path = tmp_path / "quality.zip"
    raw = _input_frame()
    result = _result_frame()
    raw.to_csv(input_path, index=False)
    result.to_csv(result_path, index=False)

    with pytest.raises(ValueError, match="未登记原始字段"):
        validate_submission_input(raw, result, quality_policy=policy, enforce_quality=True)

    summary = package_submission(
        input_path,
        result_path,
        archive_path,
        quality_policy=policy,
    )

    assert summary["input_columns"] == 2
    assert summary["quality_repair"]["repaired_cells"] == 1
    archive = validate_submission_archive(
        archive_path,
        expected_input_path=input_path,
        expected_result_path=result_path,
        quality_policy=policy,
    )
    assert archive["valid"] is True


def test_full_matrix_quality_removes_constant_and_duplicate_features() -> None:
    source = _input_frame().drop(columns="air_heater_5")
    source["feat_constant"] = 1.0
    source["feat_duplicate"] = source["feat_generator_1_lag_1"]

    prepared, report = prepare_full_matrix_submission_input(source, _policy())

    assert list(prepared.columns) == [
        "datetime",
        "generator_1",
        "feat_generator_1_lag_1",
    ]
    assert report["dropped_constant_columns"] == ["feat_constant"]
    assert report["dropped_duplicate_columns"] == [
        {
            "column": "feat_duplicate",
            "duplicate_of": "feat_generator_1_lag_1",
        }
    ]


def test_full_matrix_quality_winsorizes_derived_features() -> None:
    source = _input_frame().drop(columns="air_heater_5")
    source["feat_outlier"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1000.0]

    prepared, report = prepare_full_matrix_submission_input(source, _policy())

    assert prepared["feat_outlier"].iloc[-1] < 1000.0
    assert report["winsorized_cells"] >= 1
    assert report["final_quality"]["iqr_outlier_cells_all_methods"] == 0
