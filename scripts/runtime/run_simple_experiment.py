"""执行实验并登记命令、输入、输出和指标收据。"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行并登记可复现实验")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--result-id", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--metrics-from", type=Path, required=True)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        args.command,
        shell=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    (args.run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (args.run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(f"实验命令失败，退出码 {completed.returncode}")
    if not args.expect.is_file() or args.expect.stat().st_size == 0:
        raise SystemExit(f"预期输出不存在或为空: {args.expect}")
    metrics = json.loads(args.metrics_from.read_text(encoding="utf-8"))
    manifest = {
        "question": args.question,
        "kind": args.kind,
        "result_id": args.result_id,
        "command": args.command,
        "inputs": [str(path) for path in args.input],
        "expected_output": str(args.expect),
        "metrics_file": str(args.metrics_from),
        "metrics": metrics,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "return_code": completed.returncode,
    }
    (args.run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

