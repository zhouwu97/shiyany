from __future__ import annotations

from dataclasses import replace
import json
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
    Q_CAUSAL,
    Q_REFERENCE,
    REFERENCE_IQR_INTERPOLATIONS,
    audit_submission_quality,
    enforce_submission_quality,
    fit_submission_feature_schema,
    fit_quality_policy,
    inspect_submission_input_quality,
    normalize_submission_input_frame,
    prepare_full_matrix_submission_input,
    prepare_submission_input,
    prepare_submission_sources,
    sanitize_submission_features,
    transform_submission_input,
)


def _timestamps(start: str, periods: int) -> pd.DatetimeIndex:
    """避免不同 pandas/NumPy 组合对字符串频率的兼容性差异。"""

    return pd.date_range(start, periods=periods, freq=pd.Timedelta(minutes=15))


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
            "datetime": _timestamps("2025-05-01", rows).astype(str),
            "generator_1": [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 1000.0],
            "air_heater_5": 0.0,
            "feat_generator_1_lag_1": np.arange(rows, dtype=float),
        }
    )


def _result_frame(rows: int = 8) -> pd.DataFrame:
    frame = pd.DataFrame({"datetime": _timestamps("2025-05-01", rows).astype(str)})
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


def test_package_submission_only_encloses_prepared_quality_input(tmp_path: Path) -> None:
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

    prepared, quality_report = prepare_submission_input(raw, policy)
    prepared.to_csv(input_path, index=False)
    summary = package_submission(
        input_path,
        result_path,
        archive_path,
        quality_policy=policy,
    )

    assert summary["input_columns"] == 2
    assert quality_report["repaired_cells"] == 1
    archive = validate_submission_archive(
        archive_path,
        expected_input_path=input_path,
        expected_result_path=result_path,
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
    assert report["production_eligible"] is False


def test_fitted_quality_is_unchanged_by_scoring_tail_perturbation() -> None:
    training = pd.DataFrame(
        {
            "datetime": _timestamps("2025-01-01", 8).astype(str),
            "generator_1": [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0],
            "feat_signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    policy = _policy()
    fitted = fit_quality_policy(training, policy, train_end="2025-01-01 01:15")
    scoring = training.iloc[-3:].copy()
    altered = scoring.copy()
    altered["generator_1"] = [10_000.0, -10_000.0, 50.0]
    transformed, _ = transform_submission_input(scoring, fitted)
    altered_transformed, altered_report = transform_submission_input(altered, fitted)

    assert fitted.training_rows == 6
    assert fitted.training_end == "2025-01-01 01:15:00"
    assert fitted.to_dict()["training_rows"] == 6
    assert transformed.columns.tolist() == altered_transformed.columns.tolist()
    # 规则相同；只有当前评分行被确定性地修复，不会重新估计边界。
    assert altered_report["training_end"] == fitted.training_end
    assert altered_report["audit"]["iqr_bounds"]["generator_1"] == {
        "lower": 83.75,
        "upper": 121.25,
    }
    assert altered_transformed["generator_1"].tolist() == [121.25, 83.75, 83.75]


def test_fitted_quality_freezes_invalid_constant_and_duplicate_features() -> None:
    training = pd.DataFrame(
        {
            "datetime": _timestamps("2025-01-01", 6).astype(str),
            "generator_1": [90.0, 95.0, 100.0, 105.0, 110.0, 115.0],
            "feat_constant": [1.0] * 6,
            "feat_duplicate": [90.0, 95.0, 100.0, 105.0, 110.0, 115.0],
            "feat_invalid": ["bad"] * 6,
        }
    )
    fitted = fit_quality_policy(training, _policy())
    assert "feat_constant" in fitted.dropped_constant_columns
    assert ("feat_duplicate", "generator_1") in fitted.dropped_duplicate_columns
    assert "feat_invalid" in fitted.dropped_invalid_columns

    scoring = training.iloc[:2].copy()
    scoring["feat_constant"] = [99.0, 100.0]
    scoring["feat_duplicate"] = [1_000.0, 2_000.0]
    scoring["feat_invalid"] = ["future", "future"]
    transformed, report = transform_submission_input(scoring, fitted)
    assert "feat_constant" not in transformed.columns
    assert "feat_duplicate" not in transformed.columns
    assert "feat_invalid" not in transformed.columns
    assert report["dropped_constant_columns"] == ["feat_constant"]
    assert report["dropped_invalid_columns"] == ["feat_invalid"]


def test_transform_does_not_require_result_labels_or_future_production_values() -> None:
    training = _input_frame().drop(columns="air_heater_5")
    fitted = fit_quality_policy(training, _policy(), train_end=training["datetime"].iloc[5])
    scoring = training.iloc[6:].copy()
    baseline, _ = transform_submission_input(scoring, fitted)
    future_changed = scoring.copy()
    future_changed.loc[:, "generator_1"] = -9999.0
    changed, _ = transform_submission_input(future_changed, fitted)
    assert baseline.columns.tolist() == changed.columns.tolist()
    assert np.isfinite(changed.iloc[:, 1:].to_numpy(dtype=float)).all()


def test_prepare_submission_sources_is_causal_and_uses_reference_history_policy() -> None:
    train_index = _timestamps("2025-01-01", 700)
    score_index = _timestamps("2025-01-08 07:00:00", 6)
    training = pd.DataFrame(
        {
            "signal": 100.0 + np.sin(np.arange(len(train_index))),
            "empty": np.nan,
        },
        index=train_index,
    )
    scoring = pd.DataFrame(
        {
            "signal": [np.nan, 10_000.0, 101.0, np.nan, 98.0, 102.0],
            "empty": np.nan,
        },
        index=score_index,
    )
    origins = score_index[:4]

    repaired, report = prepare_submission_sources(training, scoring, origins=origins)
    future_perturbed = scoring.copy()
    future_perturbed.loc[score_index[4] :, "signal"] = [-1_000_000.0, np.nan]
    perturbed, perturbed_report = prepare_submission_sources(
        training,
        future_perturbed,
        origins=origins,
    )

    pd.testing.assert_frame_equal(repaired, perturbed)
    assert list(repaired.columns) == ["signal"]
    assert repaired.iloc[0, 0] == pytest.approx(training.iloc[-1, 0])
    assert repaired.iloc[1, 0] < 200.0
    assert report["mode"] == Q_CAUSAL
    assert report["settings"] == {
        "history_points": 672,
        "hampel_window": 672,
        "hampel_mad_threshold": 6.0,
        "hampel_min_periods": 168,
    }
    assert report["history_rows"] == 672
    assert report["invalid_columns"] == ["empty"]
    assert report["missing_repairs"]["signal"] == 2
    assert perturbed_report["mode"] == Q_CAUSAL
    json.dumps(report, ensure_ascii=False)


def test_sanitize_submission_features_freezes_schema_and_uses_training_median_only() -> None:
    training = pd.DataFrame(
        {
            "feat_keep": [1.0, 2.0, np.nan, 4.0],
            "feat_duplicate": [1.0, 2.0, np.nan, 4.0],
            "feat_constant": 1.0,
            "feat_empty": np.nan,
        }
    )
    schema = fit_submission_feature_schema(training)
    scoring = pd.DataFrame(
        {
            "feat_keep": [np.nan, 50.0],
            "feat_duplicate": [-100.0, 100.0],
            "feat_new": [9.0, 10.0],
        }
    )

    cleaned, report = sanitize_submission_features(training, scoring, schema=schema)

    assert list(cleaned.columns) == ["feat_keep"]
    assert cleaned["feat_keep"].tolist() == [2.0, 50.0]
    assert report["mode"] == Q_CAUSAL
    assert report["schema_frozen"] is True
    assert report["all_nonfinite"] == ["feat_empty"]
    assert report["constant"] == ["feat_constant"]
    assert report["duplicate"] == ["feat_duplicate"]
    assert report["filled_with_training_median"] == {"feat_keep": 1}
    assert report["ignored_scoring_columns"] == ["feat_duplicate", "feat_new"]
    assert np.isfinite(cleaned.to_numpy(dtype=float)).all()
    json.dumps(report, ensure_ascii=False)


def test_reference_normalization_uses_all_quantiles_converges_and_proves_terminal_gate() -> None:
    values = [
        0.0,
        1.0002541350117526,
        105144.53137419577,
        105144.53137419577,
        105149.9729784665,
        107857.55786043475,
    ]
    source = pd.DataFrame(
        {
            "feature_keep": values,
            "feature_duplicate": values,
            "feature_constant": 1.0,
            "feature_nonfinite": [np.nan] * len(values),
        }
    )
    original = source.copy(deep=True)

    normalized, report = normalize_submission_input_frame(source)

    pd.testing.assert_frame_equal(source, original)
    assert list(normalized.columns) == ["feature_keep"]
    assert report["mode"] == Q_REFERENCE
    assert report["reference_only"] is True
    assert report["model_input_mutated"] is False
    assert report["settings"]["iqr_interpolations"] == list(REFERENCE_IQR_INTERPOLATIONS)
    assert list(report["initial_quality"]["iqr_outliers_by_method"]) == list(
        REFERENCE_IQR_INTERPOLATIONS
    )
    assert len(report["winsorization_passes"]) == 2
    assert [item["winsorized_cells"] for item in report["winsorization_passes"]] == [3, 1]
    assert report["dropped_nonfinite_columns"] == ["feature_nonfinite"]
    assert report["dropped_constant_columns_before_winsor"] == ["feature_constant"]
    assert report["dropped_duplicate_columns_before_winsor"] == ["feature_duplicate"]
    assert report["terminal_gate"] == {
        "nonfinite_cells": 0,
        "constant_columns": 0,
        "duplicate_columns": 0,
        "iqr_outlier_cells_all_methods": 0,
        "zscore_outlier_cells": 0,
    }
    final_quality = inspect_submission_input_quality(normalized)
    assert final_quality["nonfinite_cells"] == 0
    assert final_quality["constant_columns"] == []
    assert final_quality["duplicate_columns"] == []
    assert final_quality["iqr_outlier_cells_all_methods"] == 0
    assert final_quality["zscore_outlier_cells"] == 0
    json.dumps(report, ensure_ascii=False)
