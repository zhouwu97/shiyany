"""按统一三层 OOF 规则运行 Phase 1–14 研究实验。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.features import load_price_schedule
from gas_forecast.research import (
    _candidate_comparison,
    build_research_oof,
    filter_research_candidate_names,
    make_online_combination_candidate,
    make_research_candidates,
)


EXPERIMENT_IDS = (
    "E10_gen1_hridge_base",
    "E11_gen1_hridge_aligned",
    "E12_gen1_hridge_aligned_longcycle",
    "E13_gen1_alpha_group",
    "E20_gen1_recency_hard",
    "E21_gen1_recency_exp",
    "E22_damped_trend",
    "E23b_relation_ridge",
    "E24_ramp_features",
    "E25_analog_weighted_median",
    "E25b_analog_local_ridge",
    "E26_grouped_recency",
    "E30_gen1_time_slot",
    "E31_gen1_fourier",
    "E32_gen1_slot_fourier",
    "E40_price_delta",
    "E41_price_interactions",
    "E50_gen1_weighted_ridge",
    "E51_gen1_weighted_lad",
    "E60_aligned_recency",
    "E61_aligned_recency_time",
    "E62_aligned_recency_time_price",
    "E63_best_linear",
    "E70_catboost_gen1_fixed_metric",
    "E80_lgb_direct_gen1",
    "E90_online_bias_true_hot",
    "E91_online_gain_true_hot",
    "E92_online_vintage_true_hot",
    "E100_dynamic_core",
    "E101_dynamic_all",
    "E110_gen1_moe",
    "E120_capacity_projection",
    "E121_path_smoothing",
    "E130_incremental_path",
    "E131_direct_incremental_blend",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行统一研究 OOF 实验")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", choices=EXPERIMENT_IDS)
    parser.add_argument(
        "--include-experiment-id",
        nargs="+",
        choices=EXPERIMENT_IDS,
        help="在同一批外层折中并行登记的额外实验，用于严格消融比较",
    )
    parser.add_argument(
        "--only-candidate-name",
        nargs="+",
        help="只运行指定候选；用于 blind 对已冻结的单一配置做 accept/reject",
    )
    parser.add_argument(
        "--online-combination",
        nargs="+",
        choices=("bias", "gain", "vintage"),
        help="Phase 10：仅在单模块赢家已冻结后测试的一到两个模块组合",
    )
    parser.add_argument(
        "--scope",
        choices=("screening", "development", "final"),
        default="screening",
        help="screening 不读取 blind；final 才包含 blind 验收",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--baseline-experiment-id",
        choices=EXPERIMENT_IDS,
        help="可选：将另一个单候选实验一并作为对比基线",
    )
    parser.add_argument(
        "--baseline-name",
        help="当基线实验包含多个候选时，明确指定其中一个候选名称",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        help="上一个已冻结候选的 report.json 或纯 ForecastConfig JSON",
    )
    parser.add_argument(
        "--base-candidate-name",
        help="当 --base-config 指向研究 report.json 时指定要继承的候选名称",
    )
    parser.add_argument(
        "--relation-report",
        type=Path,
        help="E23 report.json；读取其中已冻结的 relation feature specs",
    )
    parser.add_argument(
        "--champion-oof",
        type=Path,
        help="C0 的严格 OOF 长表；提供后所有候选均与 C0 而不是 e10 比较",
    )
    parser.add_argument(
        "--champion-column",
        help="C0 OOF 中的预测列，例如 v2_v3_target_reconciled_pred",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resolve_data_dir(path: Path) -> Path:
    """兼容 Windows 执行器，允许传入 ASCII 官方数据父目录。"""

    if (path / "Pre_gas.csv").exists():
        return path
    matches = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "Pre_gas.csv").exists()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"无法解析官方数据目录: {path}")
    return matches[0]


def _load_base_config(path: Path, candidate_name: str | None) -> ForecastConfig:
    """从冻结研究报告恢复下一阶段使用的唯一配置。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "feature" in payload:
        raw = payload
    else:
        candidates = payload.get("candidates", {})
        if candidate_name is None:
            raise ValueError("研究 report.json 必须同时提供 --base-candidate-name")
        candidate = candidates.get(candidate_name)
        if not isinstance(candidate, dict) or "config" not in candidate:
            raise ValueError("研究报告中没有指定的 base candidate config")
        raw = candidate["config"]
    return forecast_config_from_dict(raw)


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs 必须大于等于 1")
    if args.experiment_id is None and not args.online_combination:
        raise ValueError("必须提供 --experiment-id 或 --online-combination")
    if args.experiment_id is not None and args.online_combination:
        raise ValueError("--experiment-id 和 --online-combination 不能同时使用")
    stage = args.experiment_id or "E93_online_combo"
    run_dir = args.run_dir or new_run_dir(
        "results/raw/runs", f"experiment_{stage}"
    )
    output = args.output or run_dir / "oof.csv"
    report_path = args.report or run_dir / "report.json"
    config = (
        _load_base_config(args.base_config, args.base_candidate_name)
        if args.base_config
        else ForecastConfig()
    )
    if args.relation_report:
        relation_payload = json.loads(args.relation_report.read_text(encoding="utf-8"))
        frozen = relation_payload.get("frozen_relation_features", [])
        if not isinstance(frozen, list) or not frozen:
            raise ValueError("relation report 没有可用的 frozen_relation_features")
        config = replace(
            config,
            feature=replace(
                config.feature,
                relation_features=tuple(str(item) for item in frozen),
            ),
        )
    candidates = (
        [make_online_combination_candidate(config, tuple(args.online_combination))]
        if args.online_combination
        else make_research_candidates(args.experiment_id, config)
    )
    if args.include_experiment_id:
        if args.online_combination:
            raise ValueError("online 组合不能与 --include-experiment-id 混用")
        included = [
            candidate
            for experiment_id in args.include_experiment_id
            for candidate in make_research_candidates(experiment_id, config)
        ]
        existing = {candidate.name for candidate in candidates}
        candidates.extend(candidate for candidate in included if candidate.name not in existing)
    candidates = filter_research_candidate_names(candidates, args.only_candidate_name)
    baseline_name = args.baseline_name
    if args.baseline_experiment_id:
        baseline_candidates = make_research_candidates(args.baseline_experiment_id, config)
        if baseline_name is None:
            if len(baseline_candidates) != 1:
                raise ValueError("多候选基线实验必须提供 --baseline-name")
            baseline_name = baseline_candidates[0].name
        candidates = baseline_candidates + [
            candidate for candidate in candidates if candidate.name not in {item.name for item in baseline_candidates}
        ]
    data_dir = _resolve_data_dir(args.data_dir)
    dataset = align_tables(data_dir, config.feature.frequency)
    prices = sorted(data_dir.glob("*price*.xlsx"))
    price_schedule = load_price_schedule(prices[0]) if prices else None
    result = build_research_oof(
        dataset.frame,
        price_schedule,
        candidates,
        scope=args.scope,
        n_jobs=args.jobs,
        checkpoint_dir=run_dir / "checkpoints",
        baseline_name=baseline_name,
    )
    if args.champion_oof:
        if not args.champion_column:
            raise ValueError("--champion-oof 必须同时提供 --champion-column")
        champion_rows = pd.read_csv(
            args.champion_oof,
            parse_dates=["origin_time"],
        )
        keys = ["fold", "origin_time", "target", "horizon"]
        champion = champion_rows.loc[
            :, keys + [args.champion_column]
        ].rename(columns={args.champion_column: "c0_champion_pred"})
        output_rows = result.rows.merge(
            champion, on=keys, how="left", validate="one_to_one"
        )
        if output_rows["c0_champion_pred"].isna().any():
            raise ValueError("C0 OOF 未覆盖研究候选的全部严格 OOF 行")
        result.report["models"] = {
            candidate.name: _candidate_comparison(
                output_rows,
                f"{candidate.name}_pred",
                "c0_champion_pred",
                scope=args.scope,
            )
            for candidate in candidates
        }
        result.report["baseline"] = "c0_champion"
        result.report["champion_oof"] = str(args.champion_oof.resolve())
        result.report["champion_column"] = args.champion_column
    else:
        output_rows = result.rows
    output.parent.mkdir(parents=True, exist_ok=True)
    output_rows.to_csv(output, index=False, encoding="utf-8")
    write_json(report_path, result.report)
    pooled = [
        float(value["pooled_mape"])
        for value in result.report["models"].values()
        if isinstance(value, dict)
        and isinstance(value.get("pooled_mape"), (int, float))
    ]
    fingerprints = result.report.get("checkpoint_fingerprint", {})
    if not isinstance(fingerprints, dict):
        fingerprints = {}
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": stage,
            "scope": args.scope,
            "is_smoke": args.scope == "screening",
            "blind_included": result.report["blind_included"],
            "outer_folds": len(result.report["folds"]),
            "pooled_mape": min(pooled) if pooled else None,
            "baseline": result.report["baseline"],
            "base_config": str(args.base_config) if args.base_config else None,
            "relation_report": str(args.relation_report) if args.relation_report else None,
            "champion_oof": str(args.champion_oof) if args.champion_oof else None,
            "champion_column": args.champion_column,
            "config": {
                "targets": list(config.targets),
                "feature": asdict(config.feature),
                "model": asdict(config.model),
                "validation": asdict(config.validation),
            },
            **fingerprints,
            "report": _relative_or_absolute(report_path, run_dir),
            "oof": _relative_or_absolute(output, run_dir),
        },
    )
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
