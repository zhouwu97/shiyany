from __future__ import annotations

import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gas_forecast.submission as submission_module
from gas_forecast.freeze import compare_reproductions
from gas_forecast.submission import (
    CAUSAL_MODEL_INPUT_RECEIPT,
    Q_CAUSAL,
    Q_REFERENCE,
    SUBMISSION_MEMBERS,
    SUBMISSION_QUALITY_RECEIPT,
    expected_prediction_columns,
    export_legacy_json,
    package_submission,
    prepare_submission_chain,
    prepare_submission_chain_with_origin_predictor,
    validate_submission_archive,
    validate_submission_frame,
    validate_submission_input,
)
from gas_forecast.submission_quality import COMPETITION_QUALITY_POLICY


def _valid_frame(rows: int = 10, *, start: str = "2025-05-01") -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range(start, periods=rows, freq="15min").astype(str),
        }
    )
    for column in expected_prediction_columns():
        frame[column] = 100.0 if column.startswith("generator_1") else 250.0
    return frame


def _causal_policy():
    return replace(
        COMPETITION_QUALITY_POLICY,
        name="submission_chain_test_policy",
        allowed_raw_columns=("generator_1",),
        required_raw_columns=("generator_1",),
        batch_iqr_clip_columns=("generator_1",),
    )


def _training_input() -> pd.DataFrame:
    rows = 8
    signal = np.arange(1, rows + 1, dtype=float)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-04-30 22:00", periods=rows, freq="15min").astype(str),
            "generator_1": np.arange(90, 90 + 5 * rows, 5, dtype=float),
            "feat_signal": signal,
            "feat_constant": 1.0,
            "feat_duplicate": signal,
            "feat_invalid": ["bad"] * rows,
        }
    )


def _origin_input(rows: int = 10) -> pd.DataFrame:
    signal = np.arange(1, rows + 1, dtype=float)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-05-01", periods=rows, freq="15min").astype(str),
            "generator_1": np.arange(100, 100 + rows, dtype=float),
            "feat_signal": signal,
            "feat_constant": np.arange(1000, 1000 + rows, dtype=float),
            "feat_duplicate": np.arange(2000, 2000 + rows, dtype=float),
            "feat_invalid": ["future"] * rows,
        }
    )


class _StrictOriginPredictor:
    """记录输入边界，并拒绝 Q_REFERENCE 在预测阶段提前出现。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.origins: list[pd.Timestamp] = []

    def predict_at_origin(self, history_until_origin: pd.DataFrame) -> pd.DataFrame:
        assert not (self.output_dir / "input.csv").exists()
        assert not (self.output_dir / SUBMISSION_QUALITY_RECEIPT).exists()
        origin = pd.Timestamp(history_until_origin.index[-1])
        self.origins.append(origin)
        values: dict[str, float] = {}
        generator_1 = float(history_until_origin["generator_1"].iloc[-1])
        for column in expected_prediction_columns():
            values[column] = generator_1 if column.startswith("generator_1") else generator_1 + 120.0
        return pd.DataFrame([values], index=pd.DatetimeIndex([origin]))


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


def test_formal_submission_chain_freezes_result_and_writes_two_receipts(tmp_path: Path) -> None:
    """Q_CAUSAL 与 Q_REFERENCE 的边界必须可由落盘收据独立复核。"""

    policy = _causal_policy()
    training = _training_input()
    origins = _origin_input()
    result = _valid_frame(len(origins))
    training_path = tmp_path / "training.csv"
    origins_path = tmp_path / "origin_input.csv"
    result_path = tmp_path / "raw_result.csv"
    output_dir = tmp_path / "formal"
    training.to_csv(training_path, index=False)
    origins.to_csv(origins_path, index=False)
    result.to_csv(result_path, index=False)
    frozen_source_bytes = result_path.read_bytes()

    chain = prepare_submission_chain(
        training_path,
        origins_path,
        result_path,
        output_dir,
        train_end=training["datetime"].iloc[-1],
        policy=policy,
    )

    causal_receipt_path = output_dir / CAUSAL_MODEL_INPUT_RECEIPT
    quality_receipt_path = output_dir / SUBMISSION_QUALITY_RECEIPT
    causal_receipt = chain["causal_receipt"]
    quality_receipt = chain["quality_receipt"]
    assert causal_receipt_path.is_file()
    assert quality_receipt_path.is_file()
    assert causal_receipt["quality_mode"] == Q_CAUSAL
    assert causal_receipt["training_statistics_frozen"] is True
    assert causal_receipt["future_values_can_influence_policy"] is False
    assert causal_receipt["future_perturbation"]["passed"] is True
    assert "feat_constant" in causal_receipt["fitted_policy"]["dropped_constant_columns"]
    assert ["feat_duplicate", "feat_signal"] in causal_receipt["fitted_policy"][
        "dropped_duplicate_columns"
    ]
    assert quality_receipt["quality_mode"] == Q_REFERENCE
    assert quality_receipt["reference_only"] is True
    assert quality_receipt["feeds_model"] is False
    assert quality_receipt["write_read_back"]["input.csv"]["numeric_values_match"] is True
    assert quality_receipt["write_read_back"]["s_result.csv"]["bytes_match_frozen_source"] is True
    assert (output_dir / "s_result.csv").read_bytes() == frozen_source_bytes

    archive_path = output_dir / "submission.zip"
    summary = package_submission(
        chain["input_path"],
        chain["result_path"],
        archive_path,
        quality_receipt_path=quality_receipt_path,
        result_freeze=chain["result_freeze"],
    )
    assert summary["quality_receipt_verified"] is True
    assert summary["result_freeze_verified"] is True
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == SUBMISSION_MEMBERS
        assert archive.read("input.csv") == (output_dir / "input.csv").read_bytes()
        assert archive.read("s_result.csv") == frozen_source_bytes
    assert (
        validate_submission_archive(
            archive_path,
            expected_input_path=output_dir / "input.csv",
            expected_result_path=output_dir / "s_result.csv",
            quality_receipt_path=quality_receipt_path,
            result_freeze=chain["result_freeze"],
        )["valid"]
        is True
    )


def test_strict_origin_predictor_chain_runs_before_reference_normalization(tmp_path: Path) -> None:
    """正式链必须逐 origin 预测，且 Q_REFERENCE 不得提前存在或回流。"""

    policy = _causal_policy()
    training = _training_input()
    origins = _origin_input(rows=6)
    training_path = tmp_path / "training.csv"
    origins_path = tmp_path / "origins.csv"
    output_dir = tmp_path / "formal"
    training.to_csv(training_path, index=False)
    origins.to_csv(origins_path, index=False)
    predictor = _StrictOriginPredictor(output_dir)

    chain = prepare_submission_chain_with_origin_predictor(
        training_path,
        origins_path,
        output_dir,
        predictor=predictor,
        policy=policy,
        train_end=training["datetime"].iloc[-1],
    )

    assert predictor.origins == pd.to_datetime(origins["datetime"]).tolist()
    causal = chain["causal_receipt"]
    quality = chain["quality_receipt"]
    assert causal["future_perturbation"]["gate"] == "q_causal_future_perturbation_v2"
    assert causal["future_perturbation"]["passed"] is True
    assert causal["future_perturbation"]["max_abs_diff"] == 0.0
    assert quality["prediction_input"]["origin_only_predictor"] is True
    assert quality["prediction_input"]["prediction_origin_count"] == len(origins)
    assert quality["prediction_input"]["generated_after_q_causal"] is True
    assert quality["prediction_input"]["q_reference_available_during_prediction"] is False
    assert quality["causal_input_immutable_after_prediction"]["passed"] is True
    assert quality["s_result_freeze"]["verified_after_reference"] is True
    assert (output_dir / "causal_model_s_result.csv").read_bytes() == (
        output_dir / "s_result.csv"
    ).read_bytes()


def test_packaging_twice_never_refits_quality_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """打包是纯验证/封装，重复调用不能触发 Q_CAUSAL 或 Q_REFERENCE 拟合。"""

    policy = _causal_policy()
    training = _training_input()
    origins = _origin_input()
    result = _valid_frame(len(origins))
    training_path = tmp_path / "training.csv"
    origins_path = tmp_path / "origin_input.csv"
    result_path = tmp_path / "raw_result.csv"
    output_dir = tmp_path / "formal"
    training.to_csv(training_path, index=False)
    origins.to_csv(origins_path, index=False)
    result.to_csv(result_path, index=False)
    chain = prepare_submission_chain(
        training_path,
        origins_path,
        result_path,
        output_dir,
        policy=policy,
    )

    def fail_if_refit(*_args, **_kwargs):
        raise AssertionError("package_submission 不得重新拟合质量策略")

    monkeypatch.setattr(submission_module.quality_api, "fit_quality_policy", fail_if_refit)
    monkeypatch.setattr(
        submission_module.quality_api,
        "prepare_reference_submission_input",
        fail_if_refit,
    )
    archive_path = output_dir / "submission.zip"
    kwargs = {
        "quality_receipt_path": output_dir / SUBMISSION_QUALITY_RECEIPT,
        "result_freeze": chain["result_freeze"],
    }
    package_submission(chain["input_path"], chain["result_path"], archive_path, **kwargs)
    first_bytes = archive_path.read_bytes()
    package_submission(chain["input_path"], chain["result_path"], archive_path, **kwargs)
    assert archive_path.read_bytes() == first_bytes


def test_packaging_rejects_result_changed_after_freeze(tmp_path: Path) -> None:
    policy = _causal_policy()
    training = _training_input()
    origins = _origin_input()
    result = _valid_frame(len(origins))
    training_path = tmp_path / "training.csv"
    origins_path = tmp_path / "origin_input.csv"
    result_path = tmp_path / "raw_result.csv"
    output_dir = tmp_path / "formal"
    training.to_csv(training_path, index=False)
    origins.to_csv(origins_path, index=False)
    result.to_csv(result_path, index=False)
    chain = prepare_submission_chain(
        training_path,
        origins_path,
        result_path,
        output_dir,
        policy=policy,
    )

    changed = pd.read_csv(chain["result_path"])
    changed.loc[0, "generator_1_t+15_pred"] = 101.0
    changed.to_csv(chain["result_path"], index=False)
    with pytest.raises(ValueError, match="SHA256"):
        package_submission(
            chain["input_path"],
            chain["result_path"],
            output_dir / "changed.zip",
            quality_receipt_path=output_dir / SUBMISSION_QUALITY_RECEIPT,
            result_freeze=chain["result_freeze"],
        )


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
