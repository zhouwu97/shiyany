"""从 results/best 生成唯一、可直接上传的提交目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from gas_forecast.experiments import (
    is_eligible_for_best,
    promotion_evidence_passes,
    promote_if_best,
    write_json,
)
from gas_forecast.submission import (
    SUBMISSION_MEMBERS,
    package_submission,
    validate_submission_archive,
    validate_submission_frame,
    validate_submission_input,
)
from gas_forecast.submission_quality import (
    COMPETITION_QUALITY_POLICY,
    prepare_submission_input,
)


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
    report: Path | None = None,
    selection: Path | None = None,
) -> bool:
    """仅从已有正式运行的机械报告初始化 best。

    不再接受 CLI 传入的 pooled MAPE、候选名或通过标志；这些字段必须来自
    source run 的 manifest 和四类验证收据。
    """

    source_manifest_path = source_run / "manifest.json"
    if not source_manifest_path.exists():
        raise SystemExit(f"正式运行缺少 manifest.json: {source_run}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not is_eligible_for_best(source_manifest):
        raise SystemExit("source run 未通过完整 OOF、泄漏、测试和提交机械门槛")
    if not promotion_evidence_passes(source_run, source_manifest):
        raise SystemExit("source run 的验证收据内容与 manifest 不一致")
    model = _find_source_file(source_run, ("model.joblib", "model/model.joblib"))
    input_file = _find_source_file(source_run, ("input.csv", "submission/input.csv"))
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
    shutil.copy2(input_file, candidate_dir / "input.csv")
    shutil.copy2(result, candidate_dir / "result.csv")
    shutil.copy2(archive, candidate_dir / "submission.zip")
    manifest = dict(source_manifest)
    manifest.update(
        {
            "run_id": "best",
            "source_run": str(source_run.resolve()),
            "best_files": {
                "model": "model.joblib",
                "input": "input.csv",
                "result": "result.csv",
                "submission": "submission.zip",
                "report": "report.json",
                "selection": "selection.json",
            },
            "submission_path": "submission.zip",
        }
    )
    evidence = source_manifest.get("promotion_evidence", {})
    if not isinstance(evidence, dict):
        raise SystemExit("source run 缺少 promotion_evidence")
    copied_evidence: dict[str, str] = {}
    for key in ("oof_report", "leakage_report", "pytest_report", "submission_report"):
        source_evidence = source_run / str(evidence[key])
        if not source_evidence.is_file():
            raise SystemExit(f"source run 缺少验证收据: {source_evidence}")
        target_name = f"promotion_{key}.json"
        shutil.copy2(source_evidence, candidate_dir / target_name)
        copied_evidence[key] = target_name
    manifest["promotion_evidence"] = copied_evidence
    write_json(candidate_dir / "manifest.json", manifest)
    promoted = promote_if_best(candidate_dir, best_dir)
    if promoted:
        if report and report.exists():
            shutil.copy2(report, best_dir / "report.json")
        if selection and selection.exists():
            shutil.copy2(selection, best_dir / "selection.json")
        promoted_manifest = json.loads(
            (best_dir / "manifest.json").read_text(encoding="utf-8")
        )
        promoted_manifest.update(
            {
                "run_id": "best",
                "source_run": str(source_run.resolve()),
                "submission_path": "submission.zip",
            }
        )
        write_json(best_dir / "manifest.json", promoted_manifest)
        write_json(best_dir / "summary.json", promoted_manifest)
        write_json(Path("results/latest/training.json"), promoted_manifest)
    else:
        # 上一版本可能已完成原子替换，但被中间 manifest 的 evidence 名称覆盖。
        current_path = best_dir / "manifest.json"
        if current_path.exists():
            current = json.loads(current_path.read_text(encoding="utf-8"))
            same_source = current.get("source_run") == str(source_run.resolve())
            standard_evidence = {
                "oof_report": "oof_report.json",
                "leakage_report": "leakage.json",
                "pytest_report": "pytest.json",
                "submission_report": "submission.json",
            }
            repaired = dict(current)
            repaired["promotion_evidence"] = standard_evidence
            if same_source and promotion_evidence_passes(best_dir, repaired):
                write_json(best_dir / "manifest.json", repaired)
                write_json(best_dir / "summary.json", repaired)
    shutil.rmtree(candidate_dir)
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser(description="准备唯一正式提交目录")
    parser.add_argument("--best-dir", type=Path, default=Path("results/best"))
    parser.add_argument("--output-dir", type=Path, default=Path("提交这个"))
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--input-file",
        type=Path,
        help="仅用于给旧 best 补充由同一冻结模型生成的 input.csv",
    )
    parser.add_argument("--team-name", default="咕咕嘎嘎")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if args.source_run:
        bootstrap_best(
            args.source_run,
            args.best_dir,
            report=args.report,
            selection=args.selection,
        )
    manifest_path = args.best_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("没有 results/best/manifest.json，请先提供通过机械收据的正式运行")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not is_eligible_for_best(manifest):
        raise SystemExit("当前 best 未通过完整运行、泄漏、测试和提交资格检查")
    if not promotion_evidence_passes(args.best_dir, manifest):
        raise SystemExit("当前 best 的验证收据内容未通过机械复核")
    archive = args.best_dir / "submission.zip"
    input_file = args.best_dir / "input.csv"
    result = args.best_dir / "result.csv"
    if not result.exists():
        raise SystemExit("best 缺少 result.csv")
    result_frame = pd.read_csv(result)
    validation = validate_submission_frame(result_frame)
    if args.input_file:
        source_input = pd.read_csv(args.input_file)
        quality_input, _ = prepare_submission_input(
            source_input,
            COMPETITION_QUALITY_POLICY,
        )
        validate_submission_input(
            quality_input,
            result_frame,
            quality_policy=COMPETITION_QUALITY_POLICY,
            enforce_quality=True,
        )
        quality_input.to_csv(input_file, index=False, encoding="utf-8")
    if not input_file.exists():
        raise SystemExit("best 缺少 input.csv；请用 --input-file 提供同一冻结模型的推理输入")
    quality_input, quality_report = prepare_submission_input(
        pd.read_csv(input_file),
        COMPETITION_QUALITY_POLICY,
    )
    quality_input.to_csv(input_file, index=False, encoding="utf-8")
    input_validation = validate_submission_input(
        quality_input,
        result_frame,
        quality_policy=COMPETITION_QUALITY_POLICY,
        enforce_quality=True,
    )
    if int(validation["rows"]) != 192 or int(validation["prediction_columns"]) != 16:
        raise SystemExit(f"提交结果尺寸不符合要求: {validation}")
    package_submission(
        input_file,
        result,
        archive,
        quality_policy=COMPETITION_QUALITY_POLICY,
    )
    archive_validation = validate_submission_archive(
        archive,
        expected_input_path=input_file,
        expected_result_path=result,
        quality_policy=COMPETITION_QUALITY_POLICY,
    )
    if any(character in args.team_name for character in '<>:"/\\|?*') or not args.team_name.strip():
        raise SystemExit("队伍名称为空或含 Windows 文件名非法字符")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_archive = args.output_dir / f"{args.team_name}_gas_predict_prelim.zip"
    target_input = args.output_dir / "input.csv"
    target_result = args.output_dir / "s_result.csv"
    shutil.copy2(archive, target_archive)
    shutil.copy2(input_file, target_input)
    shutil.copy2(result, target_result)
    pooled = float(manifest["pooled_mape"])
    hashes = manifest.get("hashes", {})
    if not isinstance(hashes, dict):
        hashes = {}
    hashes.update(
        {
            "input": sha256(input_file),
            "result": sha256(result),
            "submission": sha256(archive),
        }
    )
    manifest["hashes"] = hashes
    best_files = manifest.get("best_files", {})
    if not isinstance(best_files, dict):
        best_files = {}
    best_files.update({"input": "input.csv", "result": "result.csv", "submission": "submission.zip"})
    manifest["best_files"] = best_files
    submission_receipt = {
        "valid": True,
        "validation": validation,
        "input": input_validation,
        "archive": archive_validation,
        "quality_repair": quality_report,
    }
    write_json(args.best_dir / "submission.json", submission_receipt)
    write_json(args.best_dir / "manifest.json", manifest)
    write_json(args.best_dir / "summary.json", manifest)
    summary = {
        "candidate": manifest.get("candidate", "unknown"),
        "pooled_mape": pooled,
        "offline_score": 1 - pooled,
        "source_run": manifest.get("source_run", "unknown"),
        "submission": str(target_archive.resolve()),
        "sha256": sha256(target_archive),
        "zip_members": list(SUBMISSION_MEMBERS),
        "input_validation": input_validation,
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
        f"{target_archive.name}\n\n"
        "不要上传：\n"
        "单独的 input.csv 或 s_result.csv\nmodel.joblib\nresults/raw/runs 下的任何实验目录\n"
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
