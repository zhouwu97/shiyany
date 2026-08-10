"""Gate E6：冻结 SAFE60 s_result。

把 SAFE60 评分长表转成平台 s_result 格式（192 rows × datetime + 16 pred 列），
记录 SHA256，并验证与 R1 冻结 s_result 的 schema 一致（值不同：SAFE60 替代
aggressive）。此步骤不改 R1 input；只生成新的 s_result 供后续 R1 打包。

用法：
  python scripts/pred1_e6_freeze_safe60.py --input <safe60_scoring_oof.csv> --output <s_result_safe60.csv>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SAFE60_REFERENCE = Path("results/raw/runs/experiments/r1_exact_reference_input_20260810_final/R1_EXACT_REFERENCE_CLONE/s_result.csv")
HORIZONS = (15, 30, 45, 60, 75, 90, 105, 120)
TARGETS = ("generator_1", "generator_all")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    long = pd.read_csv(args.input, parse_dates=["origin_time"])
    if "safe60_pred" not in long.columns:
        raise ValueError("输入缺少 safe60_pred 列")
    if len(long) != 3072 or long["origin_time"].nunique() != 192:
        raise ValueError(f"SAFE60 长表结构异常: {len(long)} rows / {long['origin_time'].nunique()} origins")

    wide = pd.pivot_table(
        long, index="origin_time", columns=["target", "horizon"],
        values="safe60_pred").reset_index()
    wide.columns = ["datetime"] + [f"{t}_t+{h}_pred" for t, h in wide.columns[1:]]
    cols = ["datetime"]
    for t in TARGETS:
        for h in HORIZONS:
            cols.append(f"{t}_t+{h}_pred")
    out = wide[cols].copy()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # schema 校验 vs R1 冻结 s_result
    ref = pd.read_csv(SAFE60_REFERENCE)
    if list(ref.columns) != list(out.columns):
        raise ValueError(f"s_result schema 不一致: \n ref={list(ref.columns)} \n new={list(out.columns)}")
    if len(ref) != len(out):
        raise ValueError(f"s_result 行数不一致: ref={len(ref)} new={len(out)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")
    receipt = {
        "schema_matches_r1": True,
        "rows": int(len(out)),
        "columns": list(out.columns),
        "s_result_sha256": _sha256(args.output),
        "source": str(args.input.resolve()),
        "note": "SAFE60 = 0.60*X3 + 0.40*A61；值替代 aggressive baseline；R1 input 未改",
    }
    (args.output.with_suffix(".json")).write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(out), "s_result_sha256": receipt["s_result_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
