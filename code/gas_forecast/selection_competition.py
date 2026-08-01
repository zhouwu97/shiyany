"""以 pooled OOF MAPE 为主指标的竞赛选择器。"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from gas_forecast.scoring import (
    ScoreSpec,
    block_bootstrap_improvement_probability,
    score_oof_long,
)


def choose_competition_candidate(
    rows: pd.DataFrame,
    candidates: Mapping[str, str],
    *,
    score_spec: ScoreSpec | None = None,
) -> dict[str, object]:
    """直接比较所有单模型、路由、融合与协调候选，不设置逐级资格。"""

    spec = score_spec or ScoreSpec()
    reports = {
        name: score_oof_long(rows, prediction_column, spec=spec)
        for name, prediction_column in candidates.items()
    }
    selected = min(reports, key=lambda name: float(reports[name]["pooled_mape"]))
    baseline = min(
        (name for name in reports if name != selected),
        key=lambda name: float(reports[name]["pooled_mape"]),
        default=selected,
    )
    bootstrap = None
    tied = baseline != selected and abs(
        float(reports[baseline]["pooled_mape"])
        - float(reports[selected]["pooled_mape"])
    ) <= 1e-12
    if baseline != selected:
        bootstrap = block_bootstrap_improvement_probability(
            rows,
            candidates[selected],
            candidates[baseline],
            spec=spec,
        )
    return {
        "selected_candidate": selected,
        "policy": "minimum_unbiased_pooled_competition_mape",
        "reports": reports,
        "runner_up": baseline,
        "tied_with_runner_up": tied,
        "block_bootstrap_vs_runner_up": bootstrap,
    }
