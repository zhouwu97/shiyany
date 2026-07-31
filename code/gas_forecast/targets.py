"""训练标签构造；项目中只有本模块允许读取未来目标。"""

from __future__ import annotations

import pandas as pd


def build_delta_targets(
    frame: pd.DataFrame,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """构造直接多步绝对增量标签，仅供训练与离线评分使用。"""

    labels: dict[str, pd.Series] = {}
    for target in targets:
        current = frame[target]
        for horizon in horizons:
            labels[f"{target}_tplus_{15 * horizon}"] = current.shift(-horizon) - current
    return pd.DataFrame(labels, index=frame.index)


def target_columns(target: str, horizons: tuple[int, ...]) -> list[str]:
    return [f"{target}_tplus_{15 * horizon}" for horizon in horizons]
