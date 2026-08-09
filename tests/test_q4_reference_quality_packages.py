from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from gas_forecast.submission import SUBMISSION_MEMBERS, expected_prediction_columns
from gas_forecast.submission_quality import COMPETITION_RAW_COLUMNS
import pytest

from scripts import run_q4_reference_quality_packages as q4_module
from scripts.run_q4_reference_quality_packages import run_q4


def _input_frame(start: str, rows: int) -> pd.DataFrame:
    positions = np.arange(rows, dtype=float)
    frame = pd.DataFrame({"datetime": pd.date_range(start, periods=rows, freq="15min").astype(str)})
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


LEGAL_MANIFEST: dict[str, object] = {
    "candidate": "legal_test_model",
    "production_gate_passed": True,
    "leakage_passed": True,
    "tests_passed": True,
    "submission_valid": True,
    "hashes": {"submission": "archive", "result": "declared" + "0" * 56},
}


def _write_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def test_q4_uses_formal_chain_and_keeps_shared_result_bytes(tmp_path: Path) -> None:
    training = _input_frame("2025-01-01", 256)
    scoring = _input_frame("2025-05-01", 192)
    scoring.loc[0, "feat_keep"] = 1_000_000.0
    result = _result_frame(scoring)
    training_path = tmp_path / "training.csv"
    scoring_path = tmp_path / "scoring.csv"
    source_zip = tmp_path / "source.zip"
    manifest_path = tmp_path / "manifest.json"
    training.to_csv(training_path, index=False, encoding="utf-8")
    scoring.to_csv(scoring_path, index=False, encoding="utf-8")
    scoring_bytes_before = scoring_path.read_bytes()
    result_bytes = result.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("input.csv", scoring_path.read_bytes())
        archive.writestr("s_result.csv", result_bytes)
    manifest_path.write_text(
        json.dumps(
            {
                "candidate": "legal_test_model",
                "production_gate_passed": True,
                "leakage_passed": True,
                "tests_passed": True,
                "submission_valid": True,
                "hashes": {
                    "submission": _sha256(source_zip.read_bytes()),
                    "result": _sha256(result_bytes),
                },
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "q4"
    report = run_q4(
        scoring_path,
        source_zip,
        output,
        prediction_manifest=manifest_path,
        training_input=training_path,
        train_end="2025-01-03 15:45:00",
    )

    assert report["formal_chain"]["api"] == ("gas_forecast.submission.prepare_submission_chain")
    assert report["formal_chain"]["future_perturbation_passed"] is True
    assert report["formal_chain"]["q_reference_feeds_model"] is False
    assert report["s_result_freeze"]["all_byte_identical"] is True
    assert set(report["s_result_freeze"]["hashes"].values()) == {_sha256(result_bytes)}
    assert report["platform"]["submitted"] is False
    assert report["platform"]["quality_score"] is None

    for name in ("SUB_A_Q_CAUSAL", "SUB_B_Q_REFERENCE"):
        assert sorted(path.name for path in (output / name).iterdir()) == sorted(SUBMISSION_MEMBERS)
        with ZipFile(output / f"{name}.zip") as archive:
            assert archive.namelist() == list(SUBMISSION_MEMBERS)
            assert archive.read("s_result.csv") == result_bytes

    terminal = report["SUB_B"]["terminal_quality"]
    assert terminal["nonfinite_cells"] == 0
    assert terminal["constant_columns"] == []
    assert terminal["duplicate_columns"] == []
    assert terminal["iqr_outlier_cells_all_methods"] == 0
    assert terminal["zscore_outlier_cells"] == 0
    assert report["SUB_B"]["write_read_back"]["input.csv"]["numeric_values_match"] is True
    assert scoring_path.read_bytes() == scoring_bytes_before


def test_q4_rejects_future_reconstruction_metadata_before_writing(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate": "future_row_reconstruction",
                "production_gate_passed": True,
                "leakage_passed": True,
                "tests_passed": True,
                "submission_valid": True,
                "hashes": {"submission": "archive", "result": "result"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不是合法正式候选"):
        q4_module._validate_prediction_source(
            manifest,
            archive_sha256="archive",
            result_sha256="result",
        )


def test_q4_rejects_registered_oracle_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(q4_module, "FORBIDDEN_SOURCE_ARCHIVE_SHA256", frozenset({"blocked"}))

    with pytest.raises(ValueError, match="future_row_reconstruction"):
        q4_module._validate_prediction_source(
            manifest,
            archive_sha256="blocked",
            result_sha256="other",
        )


@pytest.mark.parametrize(
    "override",
    [
        {"oracle_candidate": True},
        {"oracle_only": True},
        {"diagnostic_only": True},
        {"causal": False},
        {"formal_candidate": False},
    ],
)
def test_q4_rejects_oracle_and_diagnostic_metadata(
    tmp_path: Path,
    override: dict[str, object],
) -> None:
    manifest = _write_manifest(tmp_path, {**LEGAL_MANIFEST, **override})
    with pytest.raises(ValueError, match="不是合法正式候选"):
        q4_module._validate_prediction_source(
            manifest,
            archive_sha256="archive",
            result_sha256="result" + "0" * 60,
        )


@pytest.mark.parametrize(
    "gate",
    ["production_gate_passed", "leakage_passed", "tests_passed", "submission_valid"],
)
def test_q4_rejects_failed_gates(tmp_path: Path, gate: str) -> None:
    manifest_value = dict(LEGAL_MANIFEST)
    manifest_value[gate] = False
    manifest = _write_manifest(tmp_path, manifest_value)
    with pytest.raises(ValueError, match="Production Gate"):
        q4_module._validate_prediction_source(
            manifest,
            archive_sha256="archive",
            result_sha256="result" + "0" * 60,
        )


def test_q4_rejects_submission_hash_mismatch(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, LEGAL_MANIFEST)
    with pytest.raises(ValueError, match="ZIP SHA256"):
        q4_module._validate_prediction_source(
            manifest,
            archive_sha256="different_archive",
            result_sha256="result" + "0" * 60,
        )


def test_q4_rejects_unreconcilable_result_hash(tmp_path: Path) -> None:
    manifest_value = dict(LEGAL_MANIFEST)
    manifest_value["hashes"] = {
        "submission": "archive",
        "result": "declared" + "0" * 56,
    }
    manifest = _write_manifest(tmp_path, manifest_value)
    with pytest.raises(ValueError, match="s_result SHA256"):
        q4_module._validate_prediction_source(
            manifest,
            archive_sha256="archive",
            result_sha256="frozen" + "0" * 57,
            frozen_result_bytes=b"",
        )


def test_q4_reconciles_precision_declared_result(tmp_path: Path) -> None:
    """更精细位数副本：字节不同、数值相同，且 manifest 指定副本证明一致。"""

    declared_file = tmp_path / "declared_s_result.csv"
    declared_bytes = (
        b"datetime,generator_1_t+15_pred\n"
        b"2025-05-01 00:00:00,103.12345678901234\n"
        b"2025-05-01 00:15:00,104.25000000000000\n"
    )
    declared_file.write_bytes(declared_bytes)
    frozen_bytes = (
        b"datetime,generator_1_t+15_pred\n"
        b"2025-05-01 00:00:00,103.123457\n"
        b"2025-05-01 00:15:00,104.250000\n"
    )
    manifest_value = {
        **LEGAL_MANIFEST,
        "hashes": {
            "submission": "archive",
            "result": _sha256(declared_bytes),
        },
    }
    manifest = _write_manifest(tmp_path, manifest_value)

    provenance = q4_module._validate_prediction_source(
        manifest,
        archive_sha256="archive",
        result_sha256=_sha256(frozen_bytes),
        frozen_result_bytes=frozen_bytes,
        declared_result_file=declared_file,
    )

    assert provenance["result_verified"]["mode"] == "reconciled_precision"
    assert provenance["result_verified"]["frozen_numeric_equal"] is True
    assert provenance["result_verified"]["frozen_max_abs_diff"] < 1e-4


def test_q5_rejects_forbidden_result_and_marks_readiness(
    tmp_path: Path,
) -> None:
    """Q5 必须同时满足冻结字节五处一致、双 CSV 契约和未提交状态。"""

    training = _input_frame("2025-01-01", 256)
    scoring = _input_frame("2025-05-01", 192)
    result = _result_frame(scoring)
    training_path = tmp_path / "training.csv"
    scoring_path = tmp_path / "scoring.csv"
    source_zip = tmp_path / "source.zip"
    manifest_path = tmp_path / "manifest.json"
    training.to_csv(training_path, index=False, encoding="utf-8")
    scoring.to_csv(scoring_path, index=False, encoding="utf-8")
    result_bytes = result.to_csv(index=False, lineterminator="\n").encode("utf-8")
    scoring_bytes = scoring.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("input.csv", scoring_bytes)
        archive.writestr("s_result.csv", result_bytes)
    manifest_value = {
        **LEGAL_MANIFEST,
        "hashes": {
            "submission": _sha256(source_zip.read_bytes()),
            "result": _sha256(result_bytes),
        },
    }
    manifest_path.write_text(json.dumps(manifest_value, ensure_ascii=False), encoding="utf-8")

    output = tmp_path / "q5"
    report = run_q4(
        scoring_path,
        source_zip,
        output,
        prediction_manifest=manifest_path,
        training_input=training_path,
        train_end="2025-01-03 15:45:00",
        experiment="q5",
    )

    assert report["experiment"] == "Q5_reference_quality_ab"
    assert report["status"] == "LEGAL_Q5_READY_FOR_PLATFORM"
    assert report["platform"]["submitted"] is False
    assert report["platform"]["quality_score"] is None
    for name in ("SUB_A_Q_CAUSAL", "SUB_B_Q_REFERENCE"):
        members = sorted(path.name for path in (output / name).iterdir())
        assert members == sorted(SUBMISSION_MEMBERS)
        with ZipFile(output / f"{name}.zip") as archive:
            assert archive.namelist() == list(SUBMISSION_MEMBERS)


def test_q5_fails_closed_when_result_hash_is_forbidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q5 遇到已登记 Oracle s_result 必须 fail closed，不生成任何报告。"""

    monkeypatch.setattr(q4_module, "FORBIDDEN_RESULT_SHA256", frozenset({"blocked_result"}))
    manifest = _write_manifest(tmp_path, LEGAL_MANIFEST)
    with pytest.raises(ValueError, match="future_row_reconstruction"):
        q4_module._validate_prediction_source(
            manifest,
            archive_sha256="archive",
            result_sha256="blocked_result",
        )
