"""从 results/best 生成唯一、可直接上传的提交目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 直接执行脚本时优先使用当前工作树，避免导入其他 worktree 的已安装包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CODE = PROJECT_ROOT / "code"
if str(LOCAL_CODE) not in sys.path:
    sys.path.insert(0, str(LOCAL_CODE))

from gas_forecast.experiments import (  # noqa: E402
    is_eligible_for_best,
    promotion_evidence_passes,
    promote_if_best,
    write_json,
)
from gas_forecast.submission import (  # noqa: E402
    CAUSAL_MODEL_INPUT_RECEIPT,
    SUBMISSION_MEMBERS,
    SUBMISSION_QUALITY_RECEIPT,
    package_submission,
    prepare_submission_chain,
    prepare_submission_from_frozen_causal_input,
    validate_submission_archive,
)
from gas_forecast.submission_quality import COMPETITION_QUALITY_POLICY  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_if_distinct(source: Path, destination: Path) -> None:
    """复制正式工件，避免 --output-dir 与来源相同时触发 SameFileError。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def _training_input_from_manifest(manifest: dict[str, object], best_dir: Path) -> Path | None:
    """兼容已登记训练输入的旧 best；绝不回退到评分 input.csv。"""

    candidates: list[Path] = [best_dir / "training_input.csv"]
    for key in ("training_input", "training_input_csv", "causal_training_input"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            candidates.append(candidate if candidate.is_absolute() else best_dir / candidate)
    source_run = manifest.get("source_run")
    if isinstance(source_run, str) and source_run.strip():
        root = Path(source_run)
        candidates.extend((root / "training_input.csv", root / "submission" / "training_input.csv"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _find_source_file(source: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = source / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"运行目录缺少文件: {names}")


def _reject_oracle_candidate(manifest: dict[str, object], *, context: str) -> None:
    """正式提交入口拒绝所有明确标记为 Oracle 的产物。"""

    candidate = str(manifest.get("candidate", ""))
    oracle = manifest.get("oracle_candidate") is True
    oracle_only = manifest.get("oracle_only") is True
    diagnostic_only = manifest.get("diagnostic_only") is True
    non_causal = manifest.get("causal") is False
    research_only = manifest.get("research_only") is True
    if (
        candidate == "future_row_reconstruction"
        or oracle
        or oracle_only
        or diagnostic_only
        or (non_causal and research_only)
    ):
        raise SystemExit(
            f"{context} 是 ORACLE/DIAGNOSTIC ONLY（oracle_candidate=true, causal=false），"
            "禁止进入 results/best 或正式提交目录"
        )


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
    _reject_oracle_candidate(source_manifest, context="source run")
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
        help="由同一模型逐 origin 生成的原始评分输入；与 --training-input 一起生成 Q_CAUSAL 输入",
    )
    parser.add_argument(
        "--training-input",
        type=Path,
        help="仅含训练期生产观测的输入，用于冻结 Q_CAUSAL 中位数、无效列和 schema",
    )
    parser.add_argument(
        "--train-end",
        help="训练期最后一个可用 origin；未给出时由训练输入的最后一行确定",
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
    _reject_oracle_candidate(manifest, context="当前 best")
    if not is_eligible_for_best(manifest):
        raise SystemExit("当前 best 未通过完整运行、泄漏、测试和提交资格检查")
    if not promotion_evidence_passes(args.best_dir, manifest):
        raise SystemExit("当前 best 的验证收据内容未通过机械复核")
    archive = args.best_dir / "submission.zip"
    try:
        source_result = _find_source_file(args.best_dir, ("result.csv", "s_result.csv"))
    except FileNotFoundError as exc:
        raise SystemExit("best 缺少 result.csv 或 s_result.csv") from exc

    training_input = args.training_input or _training_input_from_manifest(manifest, args.best_dir)
    causal_input = args.best_dir / "causal_model_input.csv"
    causal_receipt = args.best_dir / CAUSAL_MODEL_INPUT_RECEIPT
    if args.input_file is not None and training_input is None:
        raise SystemExit(
            "提供 --input-file 时必须同时提供 --training-input；"
            "禁止以评分 input.csv 重新拟合 Q_CAUSAL 训练统计"
        )
    if training_input is not None:
        source_input = args.input_file or args.best_dir / "input.csv"
        if not source_input.is_file():
            raise SystemExit(f"缺少评分 origin 输入: {source_input}")
        chain = prepare_submission_chain(
            training_input,
            source_input,
            source_result,
            args.best_dir,
            train_end=args.train_end,
            policy=COMPETITION_QUALITY_POLICY,
        )
    elif causal_input.is_file() and causal_receipt.is_file():
        # 第二次执行复用首轮冻结的 Q_CAUSAL 输入，不重新拟合训练期统计。
        chain = prepare_submission_from_frozen_causal_input(
            causal_input,
            causal_receipt,
            source_result,
            args.best_dir,
            policy=COMPETITION_QUALITY_POLICY,
        )
    else:
        raise SystemExit(
            "缺少训练期质量输入；请提供 --training-input，或保留已验证的 "
            "causal_model_input.csv 与 causal_model_input_receipt.json。"
            "正式链禁止以提交输入重新拟合 Q_CAUSAL。"
        )

    input_file = Path(chain["input_path"])
    result = Path(chain["result_path"])
    causal_input = Path(chain["causal_input_path"])
    causal_receipt = Path(chain["causal_receipt_path"])
    quality_receipt = Path(chain["quality_receipt_path"])
    result_freeze = chain["result_freeze"]
    validation = chain["result_validation"]
    input_validation = chain["input_validation"]
    if int(validation["rows"]) != 192 or int(validation["prediction_columns"]) != 16:
        raise SystemExit(f"提交结果尺寸不符合要求: {validation}")
    archive_report = package_submission(
        input_file,
        result,
        archive,
        quality_receipt_path=quality_receipt,
        result_freeze=result_freeze,
    )
    archive_validation = validate_submission_archive(
        archive,
        expected_input_path=input_file,
        expected_result_path=result,
        quality_receipt_path=quality_receipt,
        result_freeze=result_freeze,
    )
    if any(character in args.team_name for character in '<>:"/\\|?*') or not args.team_name.strip():
        raise SystemExit("队伍名称为空或含 Windows 文件名非法字符")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_archive = args.output_dir / f"{args.team_name}_gas_predict_prelim.zip"
    target_input = args.output_dir / "input.csv"
    target_result = args.output_dir / "s_result.csv"
    target_causal_input = args.output_dir / causal_input.name
    target_causal_receipt = args.output_dir / CAUSAL_MODEL_INPUT_RECEIPT
    target_quality_receipt = args.output_dir / SUBMISSION_QUALITY_RECEIPT
    _copy_if_distinct(archive, target_archive)
    _copy_if_distinct(input_file, target_input)
    _copy_if_distinct(result, target_result)
    _copy_if_distinct(causal_input, target_causal_input)
    _copy_if_distinct(causal_receipt, target_causal_receipt)
    _copy_if_distinct(quality_receipt, target_quality_receipt)
    validate_submission_archive(
        target_archive,
        expected_input_path=target_input,
        expected_result_path=target_result,
        quality_receipt_path=target_quality_receipt,
        result_freeze=result_freeze,
    )
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
        "archive_package": archive_report,
        "causal_model_input_receipt": CAUSAL_MODEL_INPUT_RECEIPT,
        "submission_quality_receipt": SUBMISSION_QUALITY_RECEIPT,
        "result_freeze": result_freeze,
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
        "causal_model_input_receipt": str(target_causal_receipt.resolve()),
        "submission_quality_receipt": str(target_quality_receipt.resolve()),
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
        "单独的 input.csv 或 s_result.csv\nmodel.joblib\nresults/raw/runs 下的任何实验目录\n\n"
        "同目录的 causal_model_input_receipt.json 与 submission_quality_receipt.json "
        "用于审计，不需要上传到平台。\n"
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
