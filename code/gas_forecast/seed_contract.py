"""PRED-1 Gate E0：SAFE60 seed contract（冻结，FAIL-CLOSED）。

seed 是工程确定性参数，不是超参。Gate E 内禁止任何 seed 扫描。

两种 mode：
- ``replay``：seed_slot = frozen_fold_position。cutoff 必须精确匹配某 development
  fold 的 train_end；未知 cutoff → FAIL CLOSED（抛错）。用途 = OOF reproduction。
- ``production``：seed_slot = PRODUCTION_SEED_SLOT（独立命名空间，不复用任何
  fold position 0–18）。用途 = final fit once。

对所有依赖 fold_position 的随机 learner 统一生效；确定性层（ARX / Ridge / 线性）
不人为塞随机种子（见 ``learner_requires_seed``）。
"""

from __future__ import annotations

from typing import Final

import pandas as pd

SEED_POLICY_VERSION: Final[str] = "safe60_seed_v1"
BASE_RANDOM_SEED: Final[int] = 20250731
PRODUCTION_SEED_SLOT: Final[int] = 100

# 冻结 fold position：chronological（按 held origin 起点排序），position = index。
# 来源：pred1_x3_replay_20260810 oof（train_end 每折唯一）。
FROZEN_FOLD_POSITIONS: Final[dict[pd.Timestamp, int]] = {
    pd.Timestamp("2025-03-19 21:45:00"): 0,  # dev_01
    pd.Timestamp("2025-03-21 21:45:00"): 1,  # dev_02
    pd.Timestamp("2025-03-23 21:45:00"): 2,  # dev_03
    pd.Timestamp("2025-03-25 21:45:00"): 3,  # dev_04
    pd.Timestamp("2025-03-27 21:45:00"): 4,  # dev_05
    pd.Timestamp("2025-03-29 21:45:00"): 5,  # dev_06
    pd.Timestamp("2025-03-31 21:45:00"): 6,  # dev_07
    pd.Timestamp("2025-04-02 21:45:00"): 7,  # dev_08
    pd.Timestamp("2025-04-04 21:45:00"): 8,  # dev_09
    pd.Timestamp("2025-04-06 21:45:00"): 9,  # dev_10
    pd.Timestamp("2025-04-08 21:45:00"): 10,  # dev_11
    pd.Timestamp("2025-04-10 21:45:00"): 11,  # dev_12
    pd.Timestamp("2025-04-12 21:45:00"): 12,  # dev_13
    pd.Timestamp("2025-04-14 21:45:00"): 13,  # dev_14
    pd.Timestamp("2025-04-16 21:45:00"): 14,  # dev_15
    pd.Timestamp("2025-04-18 21:45:00"): 15,  # dev_16
    pd.Timestamp("2025-04-20 21:45:00"): 16,  # dev_17
    pd.Timestamp("2025-04-22 21:45:00"): 17,  # dev_18
    pd.Timestamp("2025-04-24 21:45:00"): 18,  # dev_19
}
FROZEN_FOLD_COUNT: Final[int] = len(FROZEN_FOLD_POSITIONS)

# 冻结 target / horizon 索引顺序（与 mape_aligned.X3_TARGETS / X3_HORIZONS 一致）。
X3_TARGETS_ORDER: Final[tuple[str, ...]] = ("generator_1", "generator_all")
X3_HORIZONS_ORDER: Final[tuple[int, ...]] = (15, 30, 45, 60, 75, 90, 105, 120)

# 确定性 learner（不塞随机种子）。
DETERMINISTIC_LEARNERS: Final[tuple[str, ...]] = ("ridge", "arx", "linear", "pipeline")


def seed_mode() -> tuple[str, str]:
    """返回 (policy_version, mode) 审计标识；mode 由调用方显式给出。"""
    return SEED_POLICY_VERSION, "replay|production"


def resolve_seed_position(mode: str, *, cutoff: pd.Timestamp | None = None) -> int:
    """按 mode 解析 seed_slot；replay 模式对未知 cutoff 抛错（FAIL CLOSED）。

    Parameters
    ----------
    mode : {"replay", "production"}
    cutoff : 仅在 replay 模式使用，必须是某 frozen fold 的精确 train_end。

    Returns
    -------
    int
        seed_slot（replay = frozen fold position；production = PRODUCTION_SEED_SLOT）。
    """
    if mode == "production":
        return PRODUCTION_SEED_SLOT
    if mode != "replay":
        raise ValueError(f"未知 seed_mode: {mode!r}")

    if cutoff is None:
        raise ValueError("replay 模式必须提供 cutoff")
    ts = pd.Timestamp(cutoff)
    if ts not in FROZEN_FOLD_POSITIONS:
        raise ValueError(
            f"replay 模式 cutoff 不是冻结 development fold train_end: {ts} "
            f"(FAIL CLOSED；可用 train_end: {sorted(str(k) for k in FROZEN_FOLD_POSITIONS)})"
        )
    return FROZEN_FOLD_POSITIONS[ts]


def seed_offset(position: int, target_idx: int, horizon_idx: int) -> int:
    """冻结公式：position×1000 + target_idx×100 + horizon_idx。"""
    return position * 1000 + target_idx * 100 + horizon_idx


def effective_seed(position: int, target_idx: int, horizon_idx: int) -> int:
    """最终随机种子 = BASE_RANDOM_SEED + seed_offset。"""
    return BASE_RANDOM_SEED + seed_offset(position, target_idx, horizon_idx)


def learner_requires_seed(learner: str) -> bool:
    """随机 learner（CatBoost / LGB）需要 seed；确定性层（ARX/Ridge/线性）不需要。"""
    name = learner.lower()
    if any(d in name for d in DETERMINISTIC_LEARNERS):
        return False
    if name in {"cat_mae", "lgb_l1", "lgb_huber", "catboost", "lightgbm"}:
        return True
    # 未知 learner：默认保守要求 seed，避免静默确定性。
    return True
