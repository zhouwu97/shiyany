from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_forecast.orchestration import SUPPORTED_VERSIONS
from gas_forecast.future_reconstruction import FutureRowReconstructionForecaster
from gas_forecast.submission import expected_prediction_columns
from scripts.auto_pipeline import parse_args as parse_auto_pipeline_args
from scripts.prepare_submission import _reject_oracle_candidate
from scripts.production_gate import main as production_gate_main
from scripts.train_future_reconstruction import _oracle_run_dir, parse_args


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
    report = model.training_report()
    assert report["targets"]["generator_1"]["validation_mape"] < 1e-12
    assert report["oracle_candidate"] is True
    assert report["oracle_only"] is True
    assert report["diagnostic_only"] is True
    assert report["causal"] is False
    assert report["formal_candidate"] is False
    assert report["deployable"] is False
    assert report["production_candidate"] is False
    assert report["research_only"] is True


def test_future_reconstruction_rejects_misaligned_base() -> None:
    training = _training_frame()
    scoring = _training_frame(12).set_axis(
        pd.date_range("2025-05-01", periods=12, freq="15min")
    )
    base = _base_predictions(scoring.index.shift(1, freq="15min"))
    model = FutureRowReconstructionForecaster(validation_rows=96).fit(training)

    with pytest.raises(ValueError, match="相同时间索引"):
        model.predict(scoring, base)


def test_future_reconstruction_is_future_sensitive_and_not_causal() -> None:
    """修改 origin 之后的生产行必须改变 Oracle 输出，明确证明它不是因果模型。"""

    training = _training_frame()
    scoring = _training_frame(24).set_axis(
        pd.date_range("2025-05-01", periods=24, freq="15min")
    )
    base = _base_predictions(scoring.index)
    model = FutureRowReconstructionForecaster(validation_rows=96).fit(training)
    origin = scoring.index[2]

    baseline, _ = model.predict(scoring, base)
    perturbed = scoring.copy()
    perturbed.loc[perturbed.index > origin, "generator_1"] = 1_000.0
    perturbed.loc[perturbed.index > origin, "generator_all"] = 2_000.0
    changed, _ = model.predict(perturbed, base)

    assert not np.allclose(
        baseline.loc[origin].to_numpy(dtype=float),
        changed.loc[origin].to_numpy(dtype=float),
    )
    assert model.oracle_candidate is True
    assert model.causal is False


def test_future_reconstruction_cannot_enter_formal_model_path() -> None:
    """正式自动管线的版本白名单不包含未来行 Oracle。"""

    assert FutureRowReconstructionForecaster.version not in SUPPORTED_VERSIONS

    training = _training_frame()
    scoring = _training_frame(12).set_axis(
        pd.date_range("2025-05-01", periods=12, freq="15min")
    )
    model = FutureRowReconstructionForecaster(validation_rows=96).fit(training)
    causal_current = scoring.loc[:, ["generator_1", "generator_all"]]
    with pytest.raises(ValueError, match="ORACLE/DIAGNOSTIC ONLY"):
        model.predict(scoring, causal_current)


def test_auto_pipeline_cli_rejects_oracle_version(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "auto_pipeline.py",
            "--train-dir",
            "train",
            "--test-dir",
            "test",
            "--versions",
            FutureRowReconstructionForecaster.version,
        ],
    )
    with pytest.raises(SystemExit):
        parse_auto_pipeline_args()


def test_future_reconstruction_requires_explicit_research_switch(monkeypatch) -> None:
    """研究 CLI 缺少显式授权时必须由 argparse 拒绝。"""

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_future_reconstruction.py",
            "--train-dir",
            "train",
            "--test-dir",
            "test",
            "--base-model",
            "base.joblib",
            "--run-dir",
            "results/oracle/example",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_future_reconstruction.py",
            "--train-dir",
            "train",
            "--test-dir",
            "test",
            "--base-model",
            "base.joblib",
            "--run-dir",
            "results/oracle/example",
            "--allow-oracle-research",
        ],
    )
    assert parse_args().allow_oracle_research is True


def test_future_reconstruction_output_directory_is_restricted(monkeypatch, tmp_path) -> None:
    """Oracle 只能创建 results/oracle 下的新目录，拒绝正式目录和覆盖。"""

    monkeypatch.chdir(tmp_path)
    valid = _oracle_run_dir(tmp_path / "results" / "oracle" / "case_a")
    assert valid.is_dir()

    with pytest.raises(ValueError, match="只能写入"):
        _oracle_run_dir(tmp_path / "results" / "raw" / "runs" / "case_b")
    with pytest.raises(ValueError, match="best、submission"):
        _oracle_run_dir(tmp_path / "results" / "oracle" / "best")
    with pytest.raises(ValueError, match="best、submission"):
        _oracle_run_dir(tmp_path / "results" / "oracle" / "my_submission_case")
    with pytest.raises(ValueError, match="不存在的新目录"):
        _oracle_run_dir(valid)


def test_prepare_submission_rejects_oracle_manifest() -> None:
    with pytest.raises(SystemExit, match="ORACLE/DIAGNOSTIC ONLY"):
        _reject_oracle_candidate(
            {
                "candidate": "future_row_reconstruction",
                "oracle_candidate": True,
                "causal": False,
                "research_only": True,
            },
            context="source run",
        )


def test_production_gate_rejects_oracle_run_without_formal_artifacts(monkeypatch, tmp_path) -> None:
    """Production Gate 不会把 Oracle 的诊断 manifest 当成正式运行。"""

    run_dir = tmp_path / "results" / "oracle" / "case_a"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        '{"run_type":"oracle_research","candidate":"future_row_reconstruction",'
        '"oracle_candidate":true,"causal":false,"formal_candidate":false,'
        '"deployable":false,"best_files":{}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "production_gate.py",
            "--run-dir",
            str(run_dir),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )
    with pytest.raises(FileNotFoundError, match="缺少产物"):
        production_gate_main()
