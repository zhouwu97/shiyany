"""训练标签构造；项目中只有本模块允许读取未来目标。"""

from __future__ import annotations

import pandas as pd


def add_generator_rest(frame: pd.DataFrame) -> pd.DataFrame:
    """返回含 generator_rest 的副本，不覆盖两路官方原始目标。"""

    required = {"generator_1", "generator_all"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"构造 generator_rest 缺少字段: {missing}")
    output = frame.copy()
    output["generator_rest"] = output["generator_all"] - output["generator_1"]
    return output


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
