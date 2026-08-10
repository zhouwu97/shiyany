"""PRED-1 上传打包：冻结 R1 input + SAFE60 s_result → ZIP。

- input.csv：直接字节级使用平台已验证 50/50 的冻结 R1 input，不重新生成。
- s_result.csv：冻结 SAFE60 s_result（SHA a73ded18...）。
- ZIP 只含 input.csv + s_result.csv，记录 input SHA / s_result SHA / ZIP SHA。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

FROZEN_R1_INPUT = Path("results/raw/runs/experiments/r1_exact_reference_input_20260810_final/R1_EXACT_REFERENCE_CLONE/input.csv")
FROZEN_SAFE60_S_RESULT = Path("results/raw/runs/audits/pred1_e34_scoring_20260810/s_result_safe60.csv")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="ZIP 输出路径")
    args = parser.parse_args()

    for p in (FROZEN_R1_INPUT, FROZEN_SAFE60_S_RESULT):
        if not p.exists():
            raise FileNotFoundError(f"冻结文件缺失: {p}")

    input_sha = _sha256(FROZEN_R1_INPUT)
    result_sha = _sha256(FROZEN_SAFE60_S_RESULT)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(FROZEN_R1_INPUT, arcname="input.csv")
        zf.write(FROZEN_SAFE60_S_RESULT, arcname="s_result.csv")

    zip_sha = _sha256(args.output)
    receipt = {
        "package": str(args.output.resolve()),
        "members": ["input.csv", "s_result.csv"],
        "input_sha256": input_sha,
        "s_result_sha256": result_sha,
        "zip_sha256": zip_sha,
        "input_source": str(FROZEN_R1_INPUT.resolve()),
        "s_result_source": str(FROZEN_SAFE60_S_RESULT.resolve()),
        "note": "input.csv 为平台已验证 50/50 的冻结 R1 输入，字节级未改",
    }
    (args.output.with_suffix(".json")).write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
