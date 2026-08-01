"""复核冻结 challenger 是否可替代当前正式 routed champion。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.config import forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.features import load_price_schedule
from gas_forecast.research import (
    ResearchCandidate,
    build_research_oof,
    make_formal_routed_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同口径比较正式 routed champion 与冻结 challenger")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        required=True,
        help="当前 results/best 的 selection.json，必须包含 final_route",
    )
    parser.add_argument(
        "--challenger-report",
        type=Path,
        required=True,
        help="冻结 challenger 的研究 report.json",
    )
    parser.add_argument("--challenger-name", required=True)
    parser.add_argument("--champion-name", default="formal_routed_champion")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def _load_route(path: Path) -> dict[str, object]:
    """从既有选型记录读取唯一的正式最终路由。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    route = payload.get("final_route")
    if route is None:
        routing = payload.get("routing", {})
        if isinstance(routing, dict):
            route = routing.get("final_route")
    if not isinstance(route, dict):
        raise ValueError("selection 文件不包含 final_route")
    return route


def _load_challenger(path: Path, name: str) -> ResearchCandidate:
    """从最终验收报告恢复已经冻结的单一 challenger 配置。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", {})
    candidate = candidates.get(name) if isinstance(candidates, dict) else None
    if not isinstance(candidate, dict):
        raise ValueError(f"challenger report 中不存在候选: {name}")
    raw_config = candidate.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("challenger report 缺少冻结 config")
    kind = candidate.get("kind")
    experiment_id = candidate.get("experiment_id")
    description = candidate.get("description")
    if not all(isinstance(value, str) for value in (kind, experiment_id, description)):
        raise ValueError("challenger report 的候选元数据不完整")
    return ResearchCandidate(
        experiment_id=experiment_id,
        name=name,
        kind=kind,
        config=forecast_config_from_dict(raw_config),
        description=description,
    )


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs 必须大于等于 1")
    run_dir = args.run_dir or new_run_dir(
        "results/raw/runs", "experiment_formal_promotion_comparison"
    )
    route = _load_route(args.selection)
    champion = make_formal_routed_candidate(route, name=args.champion_name)
    challenger = _load_challenger(args.challenger_report, args.challenger_name)
    if challenger.name == champion.name:
        raise ValueError("champion 与 challenger 名称不能相同")

    dataset = align_tables(args.data_dir, champion.config.feature.frequency)
    prices = sorted(args.data_dir.glob("*price*.xlsx"))
    price_schedule = load_price_schedule(prices[0]) if prices else None
    result = build_research_oof(
        dataset.frame,
        price_schedule,
        [champion, challenger],
        scope="final",
        n_jobs=args.jobs,
        checkpoint_dir=run_dir / "checkpoints",
        baseline_name=champion.name,
    )
    oof_path = run_dir / "oof.csv"
    report_path = run_dir / "report.json"
    decision_path = run_dir / "decision.json"
    result.rows.to_csv(oof_path, index=False, encoding="utf-8")
    write_json(report_path, result.report)
    challenger_report = result.report["models"][challenger.name]
    if not isinstance(challenger_report, dict):
        raise RuntimeError("最终比较没有生成 challenger 报告")
    decision = {
        "champion": champion.name,
        "challenger": challenger.name,
        "scope": "final",
        "formal_candidate": challenger_report["formal_candidate"],
        "next_action": challenger_report["next_action"],
        "selection": str(args.selection.resolve()),
        "challenger_report": str(args.challenger_report.resolve()),
        "report": "report.json",
        "oof": "oof.csv",
    }
    write_json(decision_path, decision)
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": "formal_promotion_comparison",
            "scope": "final",
            "is_smoke": False,
            "blind_included": result.report["blind_included"],
            "outer_folds": len(result.report["folds"]),
            "baseline": champion.name,
            "challenger": challenger.name,
            "pooled_mape": challenger_report["pooled_mape"],
            "formal_candidate": challenger_report["formal_candidate"],
            "selection": str(args.selection.resolve()),
            "challenger_report": str(args.challenger_report.resolve()),
            "report": "report.json",
            "oof": "oof.csv",
            "decision": "decision.json",
        },
    )
    print(json.dumps({"decision": decision, "report": result.report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
