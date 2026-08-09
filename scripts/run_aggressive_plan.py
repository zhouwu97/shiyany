"""执行 Strict C0 之后的初赛冲分阶段，不触碰 blind 选参和正式 best。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from gas_forecast.aggressive import (
    DEFAULT_BRANCH_COLUMNS,
    ExperimentRegistry,
    confirm_frozen_blend_on_blind,
    decide_experiment_status,
    diversity_sweep,
    e21_crossing_routes,
    freeze_research_base,
    oracle_gap_diagnostics,
    read_research_base,
    run_stacking_suite,
)
from gas_forecast.config import FeatureConfig
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.physical_rest import run_x1_blend_grid
from gas_forecast.price_specialist import price_error_atlas, run_price_specialist_grid


def _read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"无法序列化 {type(value).__name__}")


def _write_report(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_weight_diagnostics(report: dict[str, object], output: Path) -> None:
    """把最佳 S0-S3 权重轨迹写成明细和稳定性图。"""

    import matplotlib.pyplot as plt

    best_report = report.get("formal_report", report["best_report"])
    branches = best_report["branch_columns"]
    records: list[dict[str, object]] = []
    for fold_position, item in enumerate(best_report["weight_trajectory"]):
        for group, weights in item.get("weights", {}).items():
            for branch, weight in zip(branches, weights, strict=True):
                records.append(
                    {
                        "fold_position": fold_position,
                        "fold": item["fold"],
                        "group": group,
                        "branch": branch,
                        "weight": weight,
                    }
                )
    trajectory = pd.DataFrame(records)
    trajectory.to_csv(output / "weight_trajectory.csv", index=False)
    if trajectory.empty:
        return
    figure, axes = plt.subplots(len(branches), 1, figsize=(11, 2.4 * len(branches)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, branch in zip(axes, branches, strict=True):
        part = trajectory.loc[trajectory["branch"].eq(branch)]
        summary = part.groupby("fold_position")["weight"].agg(["min", "median", "max"])
        x = summary.index.to_numpy(float)
        axis.fill_between(x, summary["min"], summary["max"], alpha=0.20, color="#4C78A8")
        axis.plot(x, summary["median"], color="#1F4E79", linewidth=1.5)
        axis.axhline(0.0, color="#666666", linewidth=0.6)
        axis.set_ylabel(branch.replace("_pred", ""))
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("forward fold position")
    figure.suptitle(f"{report['best_candidate']} correction weight stability")
    figure.tight_layout()
    figure.savefig(output / "weight_stability.png", dpi=160)
    plt.close(figure)


def _load_branch_directory(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("branches_*.csv"))
    if not files:
        raise FileNotFoundError(f"目录内没有 branches_*.csv: {path}")
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


def _merge_external_baseline(
    rows: pd.DataFrame,
    baseline_file: str | None,
    baseline_column: str,
) -> pd.DataFrame:
    if not baseline_file:
        return rows
    external = _read_frame(baseline_file)
    keys = ["fold", "origin_time", "target", "horizon"]
    rows = rows.drop(columns=[baseline_column], errors="ignore")
    rows["origin_time"] = pd.to_datetime(rows["origin_time"])
    external["origin_time"] = pd.to_datetime(external["origin_time"])
    merged = rows.merge(
        external.loc[:, keys + [baseline_column]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if merged[baseline_column].isna().any():
        raise ValueError(f"外部 baseline {baseline_column} 与研究 OOF 键不完整")
    return merged


def _merge_candidate_file(
    rows: pd.DataFrame,
    column: str,
    path: str | Path,
    *,
    fallback_column: str | None = None,
) -> pd.DataFrame:
    """按冻结 OOF 主键合并候选；分支预测缺失时可回退到当前基线。"""

    external = _read_frame(path)
    keys = ["fold", "origin_time", "target", "horizon"]
    missing = [name for name in keys + [column] if name not in external.columns]
    if missing:
        raise ValueError(f"候选文件 {path} 缺少字段: {missing}")
    output = rows.drop(columns=[column], errors="ignore").copy()
    output["origin_time"] = pd.to_datetime(output["origin_time"])
    external["origin_time"] = pd.to_datetime(external["origin_time"])
    output = output.merge(
        external.loc[:, keys + [column]],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator="_candidate_merge",
    )
    if output["_candidate_merge"].ne("both").any():
        raise ValueError(f"候选 {column} 与研究 OOF 键不完整")
    output = output.drop(columns="_candidate_merge")
    if output[column].isna().any():
        if fallback_column is None or fallback_column not in output.columns:
            raise ValueError(f"候选 {column} 含缺失预测且未配置有效 fallback")
        output[column] = output[column].fillna(output[fallback_column])
    return output


def command_enrich(args: argparse.Namespace) -> None:
    """只用预测起点及更早生产数据和官方未来电价扩充冻结 OOF。"""

    rows = _base_branches(args.base)
    data_dir = Path(args.data_dir)
    aligned = align_tables(data_dir).frame
    price = load_price_schedule(data_dir / "price.xlsx")
    feature_config = FeatureConfig(
        enable_price_delta_features=True,
        enable_price_interactions=False,
    )
    features = build_causal_features(aligned, feature_config, price)
    direct = {
        "generator_1",
        "generator_all",
        "feat_current_price",
        "feat_generator_gas_total",
        "feat_gas_balance",
        "feat_price_switch_within_120",
        "feat_steps_to_price_switch",
    }
    prefixes = (
        "feat_target_price_tplus_",
        "feat_price_delta_tplus_",
        "feat_next_2h_price_",
        "feat_generator_1_",
        "feat_generator_rest_",
        "feat_gas_holder",
    )
    selected = [
        column
        for column in features.columns
        if column in direct or any(column.startswith(prefix) for prefix in prefixes)
    ]
    origin_features = features.loc[:, selected].reset_index().rename(
        columns={features.index.name or "index": "origin_time"}
    )
    origin_features["origin_time"] = pd.to_datetime(origin_features["origin_time"])
    rows["origin_time"] = pd.to_datetime(rows["origin_time"])
    enriched = rows.merge(origin_features, on="origin_time", how="left", validate="many_to_one")
    if enriched[selected].isna().all(axis=1).any():
        raise ValueError("部分 OOF origin 无法匹配因果特征")
    output = Path(args.output)
    _write_frame(enriched, output)
    print(f"rows={len(enriched)}, causal_feature_columns={len(selected)}, output={output}")


def command_freeze(args: argparse.Namespace) -> None:
    c0 = _read_frame(args.c0)
    branches = (
        _load_branch_directory(Path(args.branches))
        if Path(args.branches).is_dir()
        else _read_frame(args.branches)
    )
    metrics = freeze_research_base(
        c0,
        branches,
        args.output,
        c0_column=args.c0_column,
        split_payload={"folds": 20, "purge_minutes": 135, "strict_causal_features": True},
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _base_branches(base: str | Path) -> pd.DataFrame:
    _, branches = read_research_base(base)
    return branches


def command_oracle(args: argparse.Namespace) -> None:
    rows, report = oracle_gap_diagnostics(_base_branches(args.base), DEFAULT_BRANCH_COLUMNS)
    output = Path(args.output)
    _write_frame(rows, output / "oracle_predictions.parquet")
    _write_report(report, output / "report.json")
    pd.DataFrame(report["cells"]).drop(columns=["split_half_weights"], errors="ignore").to_csv(
        output / "cells.csv", index=False
    )
    print(f"oracle_gap_pp={report['oracle_gap_pp']:+.6f}, verdict={report['verdict']}")


def command_stacking(args: argparse.Namespace) -> None:
    rows, report = run_stacking_suite(_base_branches(args.base))
    output = Path(args.output)
    oracle_path = output.parent / "oracle" / "report.json"
    if oracle_path.exists():
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        report["oracle_verdict"] = oracle.get("verdict")
        report["oracle_gap_pp"] = oracle.get("oracle_gap_pp")
        if oracle.get("verdict") == "C":
            simple = next(
                item for item in report["ranking"] if item["candidate"].startswith("S00_global")
            )
            report["formal_candidate"] = simple["candidate"]
            report["formal_report"] = report["reports"][simple["candidate"]]
            report["decision"] = (
                "Oracle gap < 0.010pp；按冻结规则只保留 global shrinkage 弱信号，"
                "S1-S3 仅作为实现验收结果，不参与晋级。"
            )
        else:
            report["formal_candidate"] = report["best_candidate"]
            report["formal_report"] = report["best_report"]
            report["decision"] = "Oracle gap 允许完整比较 S0-S3。"
    else:
        report["formal_candidate"] = report["best_candidate"]
        report["formal_report"] = report["best_report"]
        report["decision"] = "未找到 Oracle 报告；仅输出实现验收排名，不执行晋级。"
    _write_frame(rows, output / "stacking_predictions.parquet")
    _write_report(report, output / "report.json")
    pd.DataFrame(report["ranking"]).to_csv(output / "ranking.csv", index=False)
    _write_weight_diagnostics(report, output)
    print(
        f"best_candidate={report['best_candidate']}, "
        f"formal_candidate={report['formal_candidate']}"
    )


def command_e21(args: argparse.Namespace) -> None:
    if args.input:
        source = _read_frame(args.input)
    else:
        if not args.e21_oof:
            raise ValueError("e21 需要 --input 或 --e21-oof")
        c0, _ = read_research_base(args.base)
        challenger = _read_frame(args.e21_oof)
        keys = ["fold", "origin_time", "target", "horizon"]
        challenger["origin_time"] = pd.to_datetime(challenger["origin_time"])
        source = c0.merge(
            challenger.loc[:, keys + [args.e21_column]].rename(
                columns={args.e21_column: "e21_pred"}
            ),
            on=keys,
            how="left",
            validate="one_to_one",
        )
        if source["e21_pred"].isna().any():
            raise ValueError("E21 OOF 与 Strict C0 键不完整")
    rows, report = e21_crossing_routes(source)
    output = Path(args.output)
    _write_frame(rows, output / "e21_crossing.parquet")
    _write_report(report, output / "report.json")
    print(f"best_route={report['best_route']}")


def command_price(args: argparse.Namespace) -> None:
    source = _merge_external_baseline(
        _read_frame(args.input), args.baseline_file, args.baseline_column
    )
    rows, report = run_price_specialist_grid(
        source, baseline_column=args.baseline_column
    )
    output = Path(args.output)
    _write_frame(rows, output / "price_specialist.parquet")
    _write_report(report, output / "report.json")
    pd.DataFrame(report["ranking"]).to_csv(output / "ranking.csv", index=False)
    print(f"best_candidate={report['best_candidate']}")


def command_price_atlas(args: argparse.Namespace) -> None:
    rows, report = price_error_atlas(_read_frame(args.input))
    output = Path(args.output)
    _write_frame(rows, output / "price_atlas_rows.parquet")
    _write_report(report, output / "atlas.json")
    print(
        f"switch_coverage={report['switch_coverage']:.6f}, "
        f"mape_switch={report['mape_switch']}, mape_non_switch={report['mape_non_switch']}"
    )


def command_physical(args: argparse.Namespace) -> None:
    source = _merge_external_baseline(
        _read_frame(args.input), args.baseline_file, args.baseline_column
    )
    rows, report = run_x1_blend_grid(source, baseline_column=args.baseline_column)
    output = Path(args.output)
    _write_frame(rows, output / "physical_x1.parquet")
    _write_report(report, output / "report.json")
    pd.DataFrame(report["ranking"]).to_csv(output / "ranking.csv", index=False)
    pd.DataFrame(report["residual_correlation"]).to_csv(
        output / "residual_correlation.csv", index=False
    )
    print(f"best_candidate={report['best_candidate']}")


def command_diversity(args: argparse.Namespace) -> None:
    source = _merge_external_baseline(
        _read_frame(args.input), args.baseline_file, args.baseline_column
    )
    for column, path in args.candidate_file:
        source = _merge_candidate_file(
            source,
            column,
            path,
            fallback_column=args.baseline_column,
        )
    rows, ranking = diversity_sweep(
        source,
        tuple(args.challengers),
        baseline_column=args.baseline_column,
    )
    output = Path(args.output)
    _write_frame(rows, output / "diversity_predictions.parquet")
    ranking.to_csv(output / "ranking.csv", index=False)
    print(ranking.to_string(index=False))


def command_confirm_blind(args: argparse.Namespace) -> None:
    output = Path(args.output)
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(f"blind 已确认，拒绝重复运行: {report_path}")
    rows, report = confirm_frozen_blend_on_blind(
        _read_frame(args.input),
        challenger_column=args.challenger,
        baseline_column=args.baseline_column,
        weight=args.weight,
    )
    _write_frame(rows, output / "confirmed_predictions.parquet")
    _write_report(report, report_path)
    print(
        f"blind_delta_pp={report['blind_delta_pp']:+.6f}, "
        f"verdict={report['verdict']}"
    )


def command_register(args: argparse.Namespace) -> None:
    status, reason = decide_experiment_status(
        delta_pp=args.delta_pp,
        fold_wins=args.fold_wins,
        total_folds=args.total_folds,
        recent5_wins=args.recent5_wins,
        screening=args.screening,
    )
    registry = ExperimentRegistry(args.registry)
    registry.append(
        {
            "experiment_id": args.experiment_id,
            "parent": args.parent,
            "model": args.model,
            "target_scope": args.target_scope,
            "horizon_scope": args.horizon_scope,
            "n_params": args.n_params,
            "pooled_mape": args.pooled_mape,
            "delta_vs_c0": args.delta_pp,
            "g1_mape": args.g1_mape,
            "gall_mape": args.gall_mape,
            "fold_wins": args.fold_wins,
            "recent5_wins": args.recent5_wins,
            "max_fold_regression": args.max_fold_regression,
            "blind_used": args.blind_used,
            "leakage_passed": args.leakage_passed,
            "status": status,
            "next_action": reason,
        }
    )
    print(f"status={status}, reason={reason}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Phase 0 冻结 Strict C0 与分支缓存")
    freeze.add_argument("--c0", default="results/best/oof.csv")
    freeze.add_argument("--c0-column", default="v2_v3_target_reconciled_pred")
    freeze.add_argument("--branches", required=True, help="branch CSV 或含 branches_*.csv 的目录")
    freeze.add_argument("--output", default="results/research_v2/base")
    freeze.set_defaults(func=command_freeze)

    enrich = subparsers.add_parser("enrich", help="合并因果工况与官方 known-future price")
    enrich.add_argument("--base", default="results/research_v2/base")
    enrich.add_argument("--data-dir", default="data/raw/official/初赛-参赛者使用")
    enrich.add_argument("--output", default="results/research_v2/base/research_features.parquet")
    enrich.set_defaults(func=command_enrich)

    oracle = subparsers.add_parser("oracle", help="Phase 1 A2.1 双向 split-half Oracle")
    oracle.add_argument("--base", default="results/research_v2/base")
    oracle.add_argument("--output", default="results/research_v2/oracle")
    oracle.set_defaults(func=command_oracle)

    stacking = subparsers.add_parser("stacking", help="Phase 1 S0-S3 严格前向 stacking")
    stacking.add_argument("--base", default="results/research_v2/base")
    stacking.add_argument("--output", default="results/research_v2/stacking")
    stacking.set_defaults(func=command_stacking)

    e21 = subparsers.add_parser("e21", help="Phase 2 R75/R90/R105 crossing")
    e21.add_argument("--input", help="含 c0_pred/e21_pred 的 OOF 长表")
    e21.add_argument("--base", default="results/research_v2/base")
    e21.add_argument("--e21-oof", help="已有 E21 OOF；与冻结 Strict C0 按键合并")
    e21.add_argument("--e21-column", default="e21_exp_half_life_30d_pred")
    e21.add_argument("--output", default="results/research_v2/e21")
    e21.set_defaults(func=command_e21)

    price = subparsers.add_parser("price", help="Phase 3 Price Atlas 与 residual specialist")
    price.add_argument("--input", required=True, help="含 known-future price 特征的 OOF 长表")
    price.add_argument("--baseline-file", help="可选：含当前临时 champion 的 OOF 长表")
    price.add_argument("--baseline-column", default="c0_pred")
    price.add_argument("--output", default="results/research_v2/price")
    price.set_defaults(func=command_price)

    atlas = subparsers.add_parser("price-atlas", help="只运行 Price Error Atlas，不训练专家")
    atlas.add_argument("--input", required=True)
    atlas.add_argument("--output", default="results/research_v2/price")
    atlas.set_defaults(func=command_price_atlas)

    physical = subparsers.add_parser("physical", help="Phase 4 Physical Rest 与 X1")
    physical.add_argument("--input", required=True, help="含两目标和工况特征的 OOF 长表")
    physical.add_argument("--baseline-file", help="可选：含当前临时 champion 的 OOF 长表")
    physical.add_argument("--baseline-column", default="c0_pred")
    physical.add_argument("--output", default="results/research_v2/physical")
    physical.set_defaults(func=command_physical)

    diversity = subparsers.add_parser("diversity", help="Phase 5 固定小权重 sweep")
    diversity.add_argument("--input", required=True)
    diversity.add_argument("--baseline-file", help="可选：含当前临时 champion 的 OOF 长表")
    diversity.add_argument("--baseline-column", default="c0_pred")
    diversity.add_argument(
        "--candidate-file",
        nargs=2,
        action="append",
        default=[],
        metavar=("COLUMN", "PATH"),
        help="按冻结主键合并候选列；可重复传入",
    )
    diversity.add_argument("--challengers", nargs="+", required=True)
    diversity.add_argument("--output", default="results/research_v2/diversity")
    diversity.set_defaults(func=command_diversity)

    confirm = subparsers.add_parser("confirm-blind", help="一次性确认已冻结候选的 blind")
    confirm.add_argument("--input", required=True)
    confirm.add_argument("--baseline-column", required=True)
    confirm.add_argument("--challenger", required=True)
    confirm.add_argument("--weight", type=float, required=True)
    confirm.add_argument("--output", default="results/research_v2/blind_confirmation")
    confirm.set_defaults(func=command_confirm_blind)

    register = subparsers.add_parser("register", help="按冻结规则写入统一实验 Registry")
    register.add_argument("--registry", default="results/aggressive_registry.csv")
    register.add_argument("--experiment-id", required=True)
    register.add_argument("--parent", default="Strict C0")
    register.add_argument("--model", required=True)
    register.add_argument("--target-scope", default="all")
    register.add_argument("--horizon-scope", default="all")
    register.add_argument("--n-params", type=int, default=0)
    register.add_argument("--pooled-mape", type=float, required=True)
    register.add_argument("--delta-pp", type=float, required=True)
    register.add_argument("--g1-mape", type=float)
    register.add_argument("--gall-mape", type=float)
    register.add_argument("--fold-wins", type=int, required=True)
    register.add_argument("--total-folds", type=int, default=20)
    register.add_argument("--recent5-wins", type=int, required=True)
    register.add_argument("--max-fold-regression", type=float, required=True)
    register.add_argument("--screening", action="store_true")
    register.add_argument("--blind-used", action="store_true")
    register.add_argument("--leakage-passed", action="store_true")
    register.set_defaults(func=command_register)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
