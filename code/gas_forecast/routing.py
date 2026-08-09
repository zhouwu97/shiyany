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
    post_route_reconciliation: bool = True,
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
    return (
        reconcile_post_route(output, prediction_column=output_column)
        if post_route_reconciliation
        else output
    )


def reconcile_post_route(
    rows: pd.DataFrame,
    *,
    prediction_column: str = "routed_pred",
    max_generator_rest: float = 240.0,
) -> pd.DataFrame:
    """对不同目标路由后的长表执行低自由度结构协调。

    目标路由可能从不同模型拼出两个目标；因此单模型内部的约束不能保证
    最终组合仍满足 ``generator_all >= generator_1``。这里使用与生产
    ``RoutedLegacyForecaster`` 相同的确定性投影，并保留原始行顺序。
    """

    required = {"fold", "origin_time", "target", "horizon", prediction_column}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"路由后结构协调缺少字段: {missing}")
    output = rows.copy()
    keys = ["fold", "origin_time", "horizon"]
    gen1 = output.loc[output["target"].eq("generator_1"), keys + [prediction_column]]
    total_mask = output["target"].eq("generator_all")
    if gen1.empty or not total_mask.any():
        return output
    gen1_indexed = gen1.set_index(keys)[prediction_column]
    total = output.loc[total_mask, keys]
    total_index = pd.MultiIndex.from_frame(total)
    gen1_values = gen1_indexed.reindex(total_index).to_numpy(dtype=float)
    total_values = output.loc[total_mask, prediction_column].to_numpy(dtype=float)
    valid = np.isfinite(gen1_values) & np.isfinite(total_values)
    reconciled = total_values.copy()
    reconciled[valid] = np.maximum(total_values[valid], gen1_values[valid])
    if max_generator_rest is not None:
        reconciled[valid] = np.minimum(
            reconciled[valid], gen1_values[valid] + float(max_generator_rest)
        )
    output.loc[total_mask, prediction_column] = reconciled
    return output


def leave_one_fold_out_route(
    rows: pd.DataFrame,
    candidate_columns: tuple[str, ...],
    *,
    config: RoutingConfig | None = None,
    score_spec: ScoreSpec | None = None,
    post_route_reconciliation: bool = True,
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
        parts.append(
            apply_route(
                validation,
                route,
                post_route_reconciliation=post_route_reconciliation,
            )
        )
        fold_routes[held_out] = route
    routed = pd.concat(parts, ignore_index=True).sort_values(
        ["origin_time", "target", "horizon", "fold"], kind="stable"
    )
    final_route = learn_hierarchical_route(
        rows, candidate_columns, config=config, score_spec=spec
    )
    final_route["post_route_reconciliation"] = {
        "enabled": bool(post_route_reconciliation),
        "max_generator_rest": 240.0,
    }
    return routed.reset_index(drop=True), {
        "unbiased_oof": score_oof_long(routed, "routed_pred", spec=spec),
        "fold_routes": fold_routes,
        "final_route": final_route,
        "post_route_reconciliation": {
            "enabled": bool(post_route_reconciliation),
            "max_generator_rest": 240.0,
        },
    }
