"""提交产物冻结与复现比较。"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gas_forecast.submission import SUBMISSION_MEMBERS, validate_submission_archive, validate_submission_frame


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: str | Path = ".") -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def git_is_clean(repo: str | Path = ".") -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return not completed.stdout.strip()


def build_freeze_manifest(
    model_path: str | Path,
    input_path: str | Path,
    result_path: str | Path,
    archive_path: str | Path,
    selection_path: str | Path,
    lock_path: str | Path,
    *,
    repo: str | Path = ".",
) -> dict[str, object]:
    model_path = Path(model_path)
    input_path = Path(input_path)
    result_path = Path(result_path)
    archive_path = Path(archive_path)
    selection_path = Path(selection_path)
    lock_path = Path(lock_path)

    result = pd.read_csv(result_path)
    validation = validate_submission_frame(result)
    validate_submission_archive(
        archive_path,
        expected_input_path=input_path,
        expected_result_path=result_path,
    )
    with zipfile.ZipFile(archive_path) as archive:
        archived_input_bytes = archive.read("input.csv")
        archived_bytes = archive.read("s_result.csv")
    archived = pd.read_csv(io.BytesIO(archived_bytes))
    validate_submission_frame(archived)
    if list(result.columns) != list(archived.columns):
        raise ValueError("磁盘结果与 ZIP 内结果字段不一致")
    if not result["datetime"].equals(archived["datetime"]):
        raise ValueError("磁盘结果与 ZIP 内结果时间戳不一致")
    np.testing.assert_allclose(
        result.iloc[:, 1:].to_numpy(float),
        archived.iloc[:, 1:].to_numpy(float),
        rtol=0.0,
        atol=5e-7,
    )

    model = joblib.load(model_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if model.version != selection["selected_version"]:
        raise ValueError("模型版本与选择文件不一致")
    return {
        "git_commit": git_commit(repo),
        "git_clean": git_is_clean(repo),
        "model_version": model.version,
        "model_sha256": sha256_file(model_path),
        "input_sha256": sha256_file(input_path),
        "result_sha256": sha256_file(result_path),
        "zip_sha256": sha256_file(archive_path),
        "zip_input_sha256": hashlib.sha256(archived_input_bytes).hexdigest(),
        "zip_result_sha256": hashlib.sha256(archived_bytes).hexdigest(),
        "selection_sha256": sha256_file(selection_path),
        "requirements_lock_sha256": sha256_file(lock_path),
        "rows": validation["rows"],
        "prediction_columns": validation["prediction_columns"],
        "archive_members": list(SUBMISSION_MEMBERS),
    }


def compare_reproductions(
    reference: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    keys = (
        "model_version",
        "model_sha256",
        "input_sha256",
        "result_sha256",
        "zip_sha256",
        "zip_input_sha256",
        "zip_result_sha256",
        "selection_sha256",
        "requirements_lock_sha256",
        "rows",
        "prediction_columns",
        "archive_members",
    )
    comparisons = {key: reference[key] == candidate[key] for key in keys}
    return {"identical": all(comparisons.values()), "comparisons": comparisons}
