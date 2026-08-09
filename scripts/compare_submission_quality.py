"""对照两个提交包的 input.csv raw schema 与质量修复差异。"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.submission_quality import (
    COMPETITION_QUALITY_POLICY,
    audit_submission_quality,
    raw_columns,
)


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path)
    with zipfile.ZipFile(path) as archive:
        try:
            payload = archive.read("input.csv")
        except KeyError as exc:
            raise ValueError(f"{path} 不含 input.csv") from exc
    return pd.read_csv(io.BytesIO(payload))


def _difference(left: pd.Series, right: pd.Series) -> dict[str, object]:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    equal = np.isclose(left_values, right_values, rtol=0.0, atol=1e-9, equal_nan=True)
    delta = np.abs(left_values - right_values)
    return {
        "different_rows": int((~equal).sum()),
        "max_abs_delta": float(np.nanmax(delta)) if np.isfinite(delta).any() else None,
    }


def compare(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, object]:
    """生成可作为质量回归基线的结构化差异报告。"""

    candidate_raw = raw_columns(candidate)
    reference_raw = raw_columns(reference)
    common = sorted(set(candidate_raw).intersection(reference_raw))
    differences = {
        column: _difference(candidate[column], reference[column]) for column in common
    }
    differences = {
        column: item for column, item in differences.items() if item["different_rows"]
    }
    return {
        "candidate_shape": list(candidate.shape),
        "reference_shape": list(reference.shape),
        "candidate_quality": audit_submission_quality(candidate, COMPETITION_QUALITY_POLICY),
        "reference_quality": audit_submission_quality(reference, COMPETITION_QUALITY_POLICY),
        "raw_only_candidate": sorted(set(candidate_raw) - set(reference_raw)),
        "raw_only_reference": sorted(set(reference_raw) - set(candidate_raw)),
        "common_raw_columns": common,
        "raw_value_differences": differences,
        "raw_value_difference_cells": int(
            sum(int(item["different_rows"]) for item in differences.values())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="待审计 ZIP 或 input.csv")
    parser.add_argument("--reference", type=Path, required=True, help="参考 ZIP 或 input.csv")
    parser.add_argument("--output", type=Path, help="可选 JSON 报告路径")
    args = parser.parse_args()

    report = compare(_read_input(args.candidate), _read_input(args.reference))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
