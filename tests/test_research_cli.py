from __future__ import annotations

import pytest

from gas_forecast.research import filter_research_candidate_names, make_research_candidates


def test_blind_candidate_filter_keeps_only_frozen_name() -> None:
    """blind 阶段只能保留预先冻结的候选参数。"""

    candidates = make_research_candidates("E21_gen1_recency_exp")

    selected = filter_research_candidate_names(candidates, ["e21_exp_half_life_30d"])

    assert [candidate.name for candidate in selected] == ["e21_exp_half_life_30d"]
    with pytest.raises(ValueError, match="不存在指定候选"):
        filter_research_candidate_names(candidates, ["not_frozen"])
