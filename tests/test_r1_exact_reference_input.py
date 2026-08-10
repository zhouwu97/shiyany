from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables, combine_context
from gas_forecast.features import build_causal_features
from gas_forecast.submission import SUBMISSION_MEMBERS, expected_prediction_columns

from scripts import run_r1_exact_reference_input as r1_module
from scripts.run_r1_exact_reference_input import run_r1


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_table(directory: Path, name: str, start: str, rows: int) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=rows, freq="15min")
    frame = pd.DataFrame({"datetime": timestamps.astype(str)})
    positions = np.arange(rows, dtype=float)
    if name == "gas":
        frame["generator_1"] = 90.0 + np.sin(positions / 5.0) * 5.0
        frame["generator_all"] = 250.0 + np.sin(positions / 7.0) * 10.0
        frame["blast_furnace_1"] = 380_000.0 + np.cos(positions / 9.0) * 5_000.0
        frame["blast_furnace_2"] = 375_000.0 + np.sin(positions / 8.0) * 4_000.0
        frame["blast_furnace_4"] = 415_000.0 + np.cos(positions / 10.0) * 3_000.0
        frame["blast_furnace_5"] = 405_000.0 + np.sin(positions / 12.0) * 3_500.0
        frame["coke_oven_1"] = 68_000.0 + np.sin(positions / 11.0) * 1_000.0
        frame["converter_1"] = 17.0 + np.cos(positions / 3.0) * 1.5
        frame["air_heater_1"] = 95_000.0 + np.sin(positions / 6.0) * 2_000.0
        frame["air_heater_2"] = 88_000.0 + np.cos(positions / 7.0) * 1_800.0
        frame["air_heater_4"] = 102_000.0 + np.sin(positions / 5.0) * 2_500.0
        frame["into_gas_mixed_coke"] = 42_000.0 + np.cos(positions / 9.0) * 900.0
        frame["into_gas_mixed_converter"] = 83_000.0 + np.sin(positions / 10.0) * 1_500.0
        frame["generator_use_blast_furnace_gas"] = 100_000.0 + np.sin(positions / 4.0) * 2_000.0
        frame["generator_use_coke_gas"] = 1_300.0 + np.cos(positions / 8.0) * 120.0
        frame["generator_use_converter_gas"] = 32_000.0 + np.sin(positions / 6.0) * 900.0
        frame["blast_furnace_user4"] = 12_000.0 + np.cos(positions / 7.0) * 400.0
        frame["air_heater_5"] = 76_000.0 + np.sin(positions / 9.0) * 1_100.0
        frame["converter_user1"] = 6_500.0 + np.cos(positions / 5.0) * 250.0
        frame["into_gas_mixed_blast_furnace"] = 55_000.0 + np.sin(positions / 8.0) * 1_300.0
    elif name == "gas_holder":
        frame["blast_furnace_gas_holder_2"] = 150_000.0 + np.sin(positions / 13.0) * 3_000.0
        frame["blast_furnace_gas_holder_1"] = 80_000.0 + np.cos(positions / 11.0) * 2_000.0
    elif name == "gas_user":
        frame["blast_furnace_user1"] = 40_000.0 + np.cos(positions / 5.0) * 800.0
        frame["blast_furnace_user2"] = 25_000.0 + np.sin(positions / 6.0) * 600.0
        frame["blast_furnace_user3"] = 18_000.0 + np.cos(positions / 8.0) * 500.0
        frame["converter_user2"] = 5_000.0 + np.sin(positions / 6.0) * 200.0
    elif name == "load":
        frame["load"] = 350_000.0 + np.sin(positions / 7.0) * 15_000.0
    frame.to_csv(directory / f"Pre_{name}.csv", index=False, encoding="utf-8", lineterminator="\n")
    return frame


def _result_frame(scoring_rows: int) -> pd.DataFrame:
    result = pd.DataFrame(
        {"datetime": pd.date_range("2025-05-01", periods=scoring_rows, freq="15min").astype(str)}
    )
    for offset, column in enumerate(expected_prediction_columns(), start=1):
        result[column] = 100.0 + offset + np.arange(scoring_rows, dtype=float) / 100.0
    return result


def _setup_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_dir = tmp_path / "train"
    scoring_dir = tmp_path / "scoring"
    model_path = tmp_path / "model.joblib"
    train_dir.mkdir()
    scoring_dir.mkdir()
    for name in ("gas", "gas_holder", "gas_user", "load"):
        _write_table(train_dir, name, "2025-01-01", 700)
        _write_table(scoring_dir, name, "2025-05-01", 192)
    model_path.write_bytes(b"placeholder")
    return train_dir, scoring_dir, model_path


def test_run_r1_generates_r0_and_r1_with_shared_result_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_dir, scoring_dir, model_path = _setup_dirs(tmp_path)
    result = _result_frame(192)
    result_bytes = result.to_csv(index=False, lineterminator="\n").encode("utf-8")

    # 源 ZIP 的 input.csv 必须是真实特征格式（21 raw + feat_ 特征），
    # 否则 R0 的 Q_REFERENCE 归一化会因输入无有效特征而 fail closed。
    config = ForecastConfig()
    train_aligned = align_tables(train_dir, config.feature.frequency).frame
    test_aligned = align_tables(scoring_dir, config.feature.frequency).frame
    context = combine_context(train_aligned, test_aligned)
    features = build_causal_features(context, config.feature)
    origins = pd.date_range("2025-05-01", periods=192, freq="15min")
    from gas_forecast.submission_quality import COMPETITION_RAW_COLUMNS

    source_input = pd.DataFrame({"datetime": origins.astype(str)})
    # raw 列直接取真实评分表数值（与训练分布同量级，避免被训练 IQR clip 成常数）。
    raw_values = test_aligned.reindex(origins)
    for column in COMPETITION_RAW_COLUMNS:
        source_input[column] = raw_values[column].to_numpy(dtype=float)
    for column in features.columns:
        if str(column).startswith("feat_"):
            source_input[str(column)] = features.loc[origins, column].to_numpy(dtype=float)
    source_input_bytes = source_input.to_csv(index=False, lineterminator="\n").encode("utf-8")

    source_zip = tmp_path / "source.zip"
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("input.csv", source_input_bytes)
        archive.writestr("s_result.csv", result_bytes)
    manifest_path = tmp_path / "manifest.json"
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

    fake_model = SimpleNamespace(config=config)
    monkeypatch.setattr(r1_module.joblib, "load", lambda path: fake_model)
    monkeypatch.setattr(r1_module, "resolve_prediction_feature_config", lambda model: model.config.feature)

    output = tmp_path / "r1"
    report = run_r1(
        source_zip,
        output,
        prediction_manifest=manifest_path,
        training_data_dir=train_dir,
        scoring_data_dir=scoring_dir,
        model_path=model_path,
        train_end="2025-04-30 23:45:00",
    )

    assert report["status"] == "LEGAL_R1_READY_FOR_PLATFORM"
    assert report["s_result_freeze"]["all_byte_identical"] is True
    assert report["platform"]["submitted"] is False

    for name in ("R0_Q5_EQUIVALENT", "R1_EXACT_REFERENCE_CLONE"):
        assert name in report
    with ZipFile(output / "R0_Q5_EQUIVALENT.zip") as archive:
        assert archive.namelist() == list(SUBMISSION_MEMBERS)
        assert archive.read("s_result.csv") == result_bytes
    with ZipFile(output / "R1_EXACT_REFERENCE.zip") as archive:
        assert archive.namelist() == list(SUBMISSION_MEMBERS)
        assert archive.read("s_result.csv") == result_bytes

    r0_audit = report["comparison_table"]["R0"]
    r1_audit = report["comparison_table"]["R1"]
    # R0 走 21 列 allowlist；R1 动态 raw 列，至少保留四表全部有效列。
    assert r0_audit["raw_columns"] == 21
    assert r1_audit["raw_columns"] > r0_audit["raw_columns"]
    assert r1_audit["all_nonfinite_cells"] == 0
    assert r1_audit["constant_columns"] == []
    assert r1_audit["duplicate_columns"] == []
    assert r1_audit["iqr_outlier_cells_all_methods"] == 0
    assert r1_audit["abs_z_gt_3_cells"] == 0

    r1_chain = report["R1_EXACT_REFERENCE_CLONE"]
    assert r1_chain["r1_report"]["pipeline"] == "exact_reference_clone_v1"
    assert r1_chain["r1_report"]["feeds_model"] is False
    final_quality = r1_chain["r1_report"]["final_quality"]
    assert final_quality["nonfinite_cells"] == 0
    assert final_quality["constant_columns"] == []
    assert final_quality["duplicate_columns"] == []
    assert final_quality["iqr_outlier_cells_all_methods"] == 0
    assert final_quality["zscore_outlier_cells"] == 0

    # R0/R1 的 input.csv 必须不同，s_result 必须一致。
    hashes_r0 = report["R0_Q5_EQUIVALENT"]["archive_member_hashes"]
    hashes_r1 = r1_chain["archive_member_hashes"]
    assert hashes_r0["input.csv"] != hashes_r1["input.csv"]
    assert hashes_r0["s_result.csv"] == hashes_r1["s_result.csv"] == _sha256(result_bytes)
