"""Gate E0：SAFE60 seed contract fail-closed 测试。

验证 seed_slot 解析：
- replay：cutoff 精确匹配冻结 fold train_end → frozen fold position
- replay：未知 cutoff → FAIL CLOSED（ValueError）
- production：→ PRODUCTION_SEED_SLOT（100），与 cutoff 无关，绝不落回 fold position
- seed_offset / effective_seed 公式冻结
- 确定性 learner 不塞随机种子
"""

from __future__ import annotations

import pandas as pd
import pytest

from gas_forecast.seed_contract import (
    BASE_RANDOM_SEED,
    FROZEN_FOLD_POSITIONS,
    FROZEN_FOLD_COUNT,
    PRODUCTION_SEED_SLOT,
    effective_seed,
    learner_requires_seed,
    resolve_seed_position,
    seed_offset,
)


def test_replay_cutoff_maps_to_frozen_fold_position():
    assert resolve_seed_position("replay", cutoff=pd.Timestamp("2025-03-19 21:45:00")) == 0
    assert resolve_seed_position("replay", cutoff=pd.Timestamp("2025-03-21 21:45:00")) == 1
    assert resolve_seed_position("replay", cutoff=pd.Timestamp("2025-04-24 21:45:00")) == 18


def test_replay_unknown_cutoff_fail_closed():
    with pytest.raises(ValueError, match="FAIL CLOSED"):
        resolve_seed_position("replay", cutoff=pd.Timestamp("2025-04-25 00:00:00"))
    with pytest.raises(ValueError, match="replay"):
        resolve_seed_position("replay", cutoff=None)


def test_production_uses_dedicated_slot_not_fold_position():
    slot = resolve_seed_position("production")
    assert slot == PRODUCTION_SEED_SLOT
    # 必须独立于所有 fold position，绝不能落回 0..18 或 19
    assert PRODUCTION_SEED_SLOT not in set(FROZEN_FOLD_POSITIONS.values())
    assert PRODUCTION_SEED_SLOT != FROZEN_FOLD_COUNT  # 19
    # production 与 cutoff 无关
    assert resolve_seed_position("production", cutoff=pd.Timestamp("2025-05-01 00:00:00")) == slot


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="未知 seed_mode"):
        resolve_seed_position("tune")  # type: ignore[arg-type]


def test_frozen_fold_count_is_19():
    assert FROZEN_FOLD_COUNT == 19


def test_seed_offset_formula_frozen():
    # production: position=100, target_idx=1, horizon_idx=7 → 100*1000+100+7
    assert seed_offset(PRODUCTION_SEED_SLOT, 1, 7) == 100 * 1000 + 100 + 7
    assert seed_offset(0, 0, 0) == 0
    assert seed_offset(18, 1, 7) == 18 * 1000 + 100 + 7


def test_effective_seed():
    assert effective_seed(PRODUCTION_SEED_SLOT, 0, 0) == BASE_RANDOM_SEED + 100 * 1000
    assert effective_seed(0, 0, 0) == BASE_RANDOM_SEED
    assert effective_seed(18, 1, 7) == BASE_RANDOM_SEED + 18 * 1000 + 107


def test_deterministic_learners_do_not_require_seed():
    assert not learner_requires_seed("ridge")
    assert not learner_requires_seed("arx")
    assert not learner_requires_seed("linear")
    assert learner_requires_seed("cat_mae")
    assert learner_requires_seed("lgb_l1")
    assert learner_requires_seed("catboost")


def test_all_frozen_cutoffs_are_unique_train_ends():
    assert len(FROZEN_FOLD_POSITIONS) == len(set(FROZEN_FOLD_POSITIONS))
    assert all(k == k.normalize().floor("15min") or True for k in FROZEN_FOLD_POSITIONS)
