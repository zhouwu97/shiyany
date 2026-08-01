"""基于外层 OOF 的稳定目标/步长路由。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gas_forecast.scoring import ScoreSpec, competition_mape, score_oof_long


@dataclass(frozen=True)
class RoutingConfig:
    min_relative_improvement: float = 0.002
    min_fold_win_rate: float = 0.55


def _score(rows: pd.DataFrame, column: str, spec: ScoreSpec) -> float:
    return competition_mape(rows["actual"], rows[column], epsilon=spec.epsilon)


def _stable_choice(
    rows: pd.DataFrame,
    candidates: tuple[str, ...],
    fallback: str,
    config: RoutingConfig,
    spec: ScoreSpec,
) -> tuple[str, dict[str, object]]:
    available = tuple(column for column in candidates if column in rows and rows[column].notna().any())
    if fallback not in available:
        raise ValueError(f"回退模型 {fallback} 不在候选中")
    scores = {column: _score(rows, column, spec) for column in available}
    best = min(scores, key=scores.get)
    fallback_score = scores[fallback]
    improvement = (fallback_score - scores[best]) / max(fallback_score, spec.epsilon)
    folds = sorted(rows["fold"].astype(str).unique())
    wins = 0
    comparable = 0
    for fold in folds:
        part = rows.loc[rows["fold"].astype(str).eq(fold)]
        best_score = _score(part, best, spec)
        fallback_fold_score = _score(part, fallback, spec)
        if np.isfinite(best_score) and np.isfinite(fallback_fold_score):
            comparable += 1
            wins += int(best_score < fallback_fold_score)
    win_rate = wins / comparable if comparable else 0.0
    accepted = (
        best == fallback
        or (
            improvement >= config.min_relative_improvement
            and win_rate >= config.min_fold_win_rate
        )
    )
    selected = best if accepted else fallback
    return selected, {
        "selected": selected,
        "raw_best": best,
        "fallback": fallback,
        "scores": scores,
        "relative_improvement": float(improvement),
        "fold_win_rate": float(win_rate),
        "accepted": bool(accepted),
    }


def learn_hierarchical_route(
    rows: pd.DataFrame,
    candidate_columns: tuple[str, ...],
    *,
    config: RoutingConfig | None = None,
    score_spec: ScoreSpec | None = None,
) -> dict[str, object]:
    """学习全局→目标→目标×步长三级回缩路由。"""

    config = config or RoutingConfig()
    spec = score_spec or ScoreSpec()
    global_scores = {column: _score(rows, column, spec) for column in candidate_columns}
    global_best = min(global_scores, key=global_scores.get)
    targets: dict[str, object] = {}
    cells: dict[str, object] = {}
    for target, target_rows in rows.groupby("target", sort=True):
        target_choice, target_detail = _stable_choice(
            target_rows, candidate_columns, global_best, config, spec
        )
        targets[str(target)] = target_detail
        for horizon, cell_rows in target_rows.groupby("horizon", sort=True):
            _, cell_detail = _stable_choice(
                cell_rows, candidate_columns, target_choice, config, spec
            )
            cells[f"{target}|{int(horizon)}"] = cell_detail
    return {
        "policy": "hierarchical_target_horizon_shrinkage",
        "config": asdict(config),
        "score_spec": asdict(spec),
        "global": {"selected": global_best, "scores": global_scores},
        "targets": targets,
        "cells": cells,
    }


def apply_route(
    rows: pd.DataFrame,
    route: dict[str, object],
    *,
    output_column: str = "routed_pred",
) -> pd.DataFrame:
    """将冻结路由应用到逐行候选预测。"""

    output = rows.copy()
    selected = []
    prediction = []
    cells = route["cells"]
    global_choice = str(route["global"]["selected"])
    for row in output.itertuples(index=False):
        key = f"{row.target}|{int(row.horizon)}"
        column = str(cells.get(key, {}).get("selected", global_choice))
        selected.append(column.removesuffix("_pred"))
        prediction.append(float(getattr(row, column)))
    output["selected_model"] = selected
    output[output_column] = prediction
    return output


def leave_one_fold_out_route(
    rows: pd.DataFrame,
    candidate_columns: tuple[str, ...],
    *,
    config: RoutingConfig | None = None,
    score_spec: ScoreSpec | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """逐折在其余外层折学习路由，形成无偏路由 OOF。"""

    config = config or RoutingConfig()
    spec = score_spec or ScoreSpec()
    parts: list[pd.DataFrame] = []
    fold_routes: dict[str, object] = {}
    folds = sorted(rows["fold"].astype(str).unique())
    if len(folds) < 2:
        raise ValueError("LOFO 路由至少需要2个外层折")
    for held_out in folds:
        train = rows.loc[~rows["fold"].astype(str).eq(held_out)]
        validation = rows.loc[rows["fold"].astype(str).eq(held_out)]
        route = learn_hierarchical_route(
            train, candidate_columns, config=config, score_spec=spec
        )
        parts.append(apply_route(validation, route))
        fold_routes[held_out] = route
    routed = pd.concat(parts, ignore_index=True).sort_values(
        ["origin_time", "target", "horizon", "fold"], kind="stable"
    )
    final_route = learn_hierarchical_route(
        rows, candidate_columns, config=config, score_spec=spec
    )
    return routed.reset_index(drop=True), {
        "unbiased_oof": score_oof_long(routed, "routed_pred", spec=spec),
        "fold_routes": fold_routes,
        "final_route": final_route,
    }
