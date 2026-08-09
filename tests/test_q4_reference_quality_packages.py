from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from gas_forecast.submission import SUBMISSION_MEMBERS, expected_prediction_columns
from gas_forecast.submission_quality import COMPETITION_RAW_COLUMNS
from scripts.run_q4_reference_quality_packages import run_q4


def _input_frame(start: str, rows: int) -> pd.DataFrame:
    positions = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {"datetime": pd.date_range(start, periods=rows, freq="15min").astype(str)}
    )
    for offset, column in enumerate(COMPETITION_RAW_COLUMNS, start=1):
        frame[column] = positions * (offset + 0.25) + offset**2
    frame["feat_keep"] = np.sin(positions / 7.0) + positions / 100.0
    frame["feat_duplicate"] = frame["feat_keep"]
    frame["feat_constant"] = 1.0
    return frame


def _result_frame(scoring: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"datetime": scoring["datetime"]})
    for offset, column in enumerate(expected_prediction_columns(), start=1):
        result[column] = 100.0 + offset + np.arange(len(scoring), dtype=float) / 100.0
    return result


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_q4_uses_formal_chain_and_keeps_shared_result_bytes(tmp_path: Path) -> None:
    training = _input_frame("2025-01-01", 256)
    scoring = _input_frame("2025-05-01", 192)
    scoring.loc[0, "feat_keep"] = 1_000_000.0
    result = _result_frame(scoring)
    training_path = tmp_path / "training.csv"
    scoring_path = tmp_path / "scoring.csv"
    source_zip = tmp_path / "source.zip"
    training.to_csv(training_path, index=False, encoding="utf-8")
    scoring.to_csv(scoring_path, index=False, encoding="utf-8")
    scoring_bytes_before = scoring_path.read_bytes()
    result_bytes = result.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("input.csv", scoring_path.read_bytes())
        archive.writestr("s_result.csv", result_bytes)

    output = tmp_path / "q4"
    report = run_q4(
        scoring_path,
        source_zip,
        output,
        training_input=training_path,
        train_end="2025-01-03 15:45:00",
    )

    assert report["formal_chain"]["api"] == (
        "gas_forecast.submission.prepare_submission_chain"
    )
    assert report["formal_chain"]["future_perturbation_passed"] is True
    assert report["formal_chain"]["q_reference_feeds_model"] is False
    assert report["s_result_freeze"]["all_byte_identical"] is True
    assert set(report["s_result_freeze"]["hashes"].values()) == {_sha256(result_bytes)}
    assert report["platform"]["submitted"] is False
    assert report["platform"]["quality_score"] is None

    for name in ("SUB_A_Q_CAUSAL", "SUB_B_Q_REFERENCE"):
        assert sorted(path.name for path in (output / name).iterdir()) == sorted(
            SUBMISSION_MEMBERS
        )
        with ZipFile(output / f"{name}.zip") as archive:
            assert archive.namelist() == list(SUBMISSION_MEMBERS)
            assert archive.read("s_result.csv") == result_bytes

    terminal = report["SUB_B"]["terminal_quality"]
    assert terminal["nonfinite_cells"] == 0
    assert terminal["constant_columns"] == []
    assert terminal["duplicate_columns"] == []
    assert terminal["iqr_outlier_cells_all_methods"] == 0
    assert terminal["zscore_outlier_cells"] == 0
    assert report["SUB_B"]["write_read_back"]["input.csv"][
        "numeric_values_match"
    ] is True
    assert scoring_path.read_bytes() == scoring_bytes_before
