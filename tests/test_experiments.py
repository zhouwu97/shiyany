from __future__ import annotations

import re

from gas_forecast.experiments import (
    is_eligible_for_best,
    new_run_dir,
    promote_if_best,
    write_json,
)


def test_new_run_dir_uses_date_first_readable_name(tmp_path) -> None:
    output = new_run_dir(tmp_path / "results" / "raw" / "runs", "data_audit")

    assert output.is_dir()
    assert output.parent.name == "data"
    assert re.fullmatch(r"\d{8}_\d{6}_\d{3}", output.name)
    assert (output / "manifest.json").exists()


def test_best_promotion_requires_complete_eligible_run(tmp_path) -> None:
    run = tmp_path / "results" / "raw" / "runs" / "training" / "run"
    run.mkdir(parents=True)
    for name in ("model.joblib", "input.csv", "result.csv", "submission.zip"):
        (run / name).write_bytes(name.encode())
    for name, payload in (
        ("oof.json", {"pooled_mape": 0.05}),
        ("leakage.json", {"passed": True}),
        ("pytest.json", {"passed": True}),
        ("submission.json", {"valid": True}),
    ):
        write_json(run / name, payload)
    write_json(
        run / "manifest.json",
        {
            "run_type": "training",
            "status": "completed",
            "is_smoke": False,
            "pooled_mape": 0.05,
            "leakage_passed": True,
            "tests_passed": True,
            "submission_valid": True,
            "best_files": {
                "model": "model.joblib",
                "input": "input.csv",
                "result": "result.csv",
                "submission": "submission.zip",
            },
            "promotion_evidence": {
                "oof_report": "oof.json",
                "leakage_report": "leakage.json",
                "pytest_report": "pytest.json",
                "submission_report": "submission.json",
            },
        },
    )
    assert is_eligible_for_best({"status": "completed", "is_smoke": True}) is False
    assert promote_if_best(run, tmp_path / "results" / "best") is True
    assert (tmp_path / "results" / "best" / "model.joblib").exists()
    assert (tmp_path / "results" / "best" / "input.csv").exists()
