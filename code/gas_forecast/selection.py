"""仅依据训练期滚动报告选择最终模型版本。"""

from __future__ import annotations

from typing import Mapping


def _fold_map(report: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {str(fold["name"]): fold for fold in report["folds"]}


def _beats(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
) -> tuple[bool, dict[str, object]]:
    candidate_folds = _fold_map(candidate)
    baseline_folds = _fold_map(baseline)
    common = sorted(set(candidate_folds).intersection(baseline_folds))
    if not common:
        return False, {"reason": "没有可比滚动折", "common_folds": 0}

    wins = sum(
        float(candidate_folds[name]["mape"]) < float(baseline_folds[name]["mape"])
        for name in common
    )
    mean_better = float(candidate["mean_mape"]) < float(baseline["mean_mape"])
    majority = wins > len(common) / 2
    blind_ok = True
    if "blind" in common:
        blind_ok = float(candidate_folds["blind"]["mape"]) <= float(
            baseline_folds["blind"]["mape"]
        )
    details = {
        "common_folds": len(common),
        "wins": wins,
        "mean_better": mean_better,
        "majority_wins": majority,
        "blind_not_worse": blind_ok,
    }
    return mean_better and majority and blind_ok, details


def choose_version(reports: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """按 V1 -> V2 -> V3 的逐级门槛选择，不允许越级。"""

    if "v1" not in reports:
        raise ValueError("版本选择至少需要 V1 滚动报告")
    selected = "v1"
    comparisons: dict[str, object] = {}

    if "v2" in reports:
        accepted, details = _beats(reports["v2"], reports["v1"])
        comparisons["v2_vs_v1"] = {"accepted": accepted, **details}
        if accepted:
            selected = "v2"

    if selected == "v2" and "v25" in reports:
        accepted, details = _beats(reports["v25"], reports["v2"])
        comparisons["v25_vs_v2"] = {"accepted": accepted, **details}
        if accepted:
            selected = "v25"

    if selected == "v25" and "v3" in reports:
        accepted, details = _beats(reports["v3"], reports["v25"])
        comparisons["v3_vs_v25"] = {"accepted": accepted, **details}
        if accepted:
            selected = "v3"

    return {
        "selected_version": selected,
        "policy": "mean_mape_better_and_majority_wins_and_blind_not_worse",
        "comparisons": comparisons,
    }
