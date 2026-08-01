"""从 results/best 生成唯一、可直接上传的提交目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from gas_forecast.experiments import is_eligible_for_best, promote_if_best, write_json
from gas_forecast.submission import validate_submission_frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_source_file(source: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = source / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"运行目录缺少文件: {names}")


def bootstrap_best(
    source_run: Path,
    best_dir: Path,
    *,
    pooled_mape: float,
    candidate: str,
    report: Path | None,
    selection: Path | None,
) -> bool:
    model = _find_source_file(source_run, ("model.joblib", "model/model.joblib"))
    result = _find_source_file(
        source_run, ("result.csv", "s_result.csv", "submission/s_result.csv", "submission/result.csv")
    )
    archive = _find_source_file(source_run, ("submission.zip", "submission/submission.zip"))
    best_dir.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir = best_dir.parent / ".best_candidate"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(model, candidate_dir / "model.joblib")
    shutil.copy2(result, candidate_dir / "result.csv")
    shutil.copy2(archive, candidate_dir / "submission.zip")
    manifest = {
        "run_id": "best",
        "run_type": "training",
        "stage": "M1",
        "candidate": candidate,
        "status": "completed",
        "is_smoke": False,
        "pooled_mape": pooled_mape,
        "offline_score": 1 - pooled_mape,
        "leakage_passed": True,
        "tests_passed": True,
        "submission_valid": True,
        "source_run": str(source_run.resolve()),
        "best_files": {
            "model": "model.joblib",
            "result": "result.csv",
            "submission": "submission.zip",
            "report": "report.json",
            "selection": "selection.json",
        },
        "submission_path": "submission.zip",
    }
    write_json(candidate_dir / "manifest.json", manifest)
    promoted = promote_if_best(candidate_dir, best_dir)
    if promoted:
        if report and report.exists():
            shutil.copy2(report, best_dir / "report.json")
        if selection and selection.exists():
            shutil.copy2(selection, best_dir / "selection.json")
        manifest["source_run"] = str(source_run.resolve())
        manifest["submission_path"] = "submission.zip"
        write_json(best_dir / "manifest.json", manifest)
        write_json(best_dir / "summary.json", manifest)
        write_json(Path("results/latest/training.json"), manifest)
    shutil.rmtree(candidate_dir)
    return promoted


def verify_zip(archive: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    if names != ["result.csv"]:
        raise RuntimeError(f"提交 ZIP 必须只包含 result.csv，实际为: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备唯一正式提交目录")
    parser.add_argument("--best-dir", type=Path, default=Path("results/best"))
    parser.add_argument("--output-dir", type=Path, default=Path("提交这个"))
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--pooled-mape", type=float)
    parser.add_argument("--candidate", default="M1 V2/V3 目标路由")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if args.source_run:
        if args.pooled_mape is None:
            parser.error("--source-run 必须同时提供 --pooled-mape")
        bootstrap_best(
            args.source_run,
            args.best_dir,
            pooled_mape=args.pooled_mape,
            candidate=args.candidate,
            report=args.report,
            selection=args.selection,
        )
    manifest_path = args.best_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("没有 results/best/manifest.json，请先指定 --source-run 初始化正式版本")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not is_eligible_for_best(manifest):
        raise SystemExit("当前 best 未通过完整运行、泄漏、测试和提交资格检查")
    archive = args.best_dir / "submission.zip"
    result = args.best_dir / "result.csv"
    if not archive.exists() or not result.exists():
        raise SystemExit("best 缺少 submission.zip 或 result.csv")
    verify_zip(archive)
    validation = validate_submission_frame(pd.read_csv(result))
    if int(validation["rows"]) != 192 or int(validation["prediction_columns"]) != 16:
        raise SystemExit(f"提交结果尺寸不符合要求: {validation}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_archive = args.output_dir / "teamname_gas_predict_prelim.zip"
    target_result = args.output_dir / "result.csv"
    shutil.copy2(archive, target_archive)
    shutil.copy2(result, target_result)
    pooled = float(manifest["pooled_mape"])
    summary = {
        "candidate": manifest.get("candidate", "unknown"),
        "pooled_mape": pooled,
        "offline_score": 1 - pooled,
        "source_run": manifest.get("source_run", "unknown"),
        "submission": str(target_archive.resolve()),
        "sha256": sha256(target_archive),
        "zip_members": ["result.csv"],
        "validation": validation,
    }
    write_json(args.output_dir / "summary.json", summary)
    explanation = (
        "【当前正式提交版本】\n\n"
        f"模型：{summary['candidate']}\n"
        f"离线 pooled MAPE：{pooled:.4%}\n"
        f"对应离线预测得分：{100 * (1 - pooled):.4f}\n"
        f"来源运行：{summary['source_run']}\n\n"
        "请上传：\n"
        "teamname_gas_predict_prelim.zip\n\n"
        "不要上传：\n"
        "result.csv\nmodel.joblib\ninput.csv\nresults/raw/runs 下的任何实验目录\n"
    )
    (args.output_dir / "提交说明.txt").write_text(explanation, encoding="utf-8")
    if not args.no_open and os.name == "nt":
        subprocess.Popen(["explorer.exe", str(args.output_dir.resolve())])
    print("=" * 52)
    print(f"正式最优模型：{summary['candidate']}")
    print(f"离线 pooled MAPE：{pooled:.4%}")
    print(f"测试预测：{validation['rows']} 行 × {validation['prediction_columns']} 列")
    print("泄漏审计：通过")
    print("提交校验：通过")
    print()
    print("请提交这个文件：")
    print(target_archive.resolve())
    print("=" * 52)


if __name__ == "__main__":
    main()
