"""赛事四表的读取、时间对齐和数据审计。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


TABLE_ORDER = ("gas", "gas_holder", "gas_user", "load")


class DataContractError(ValueError):
    """输入数据不符合赛事数据契约。"""


@dataclass(frozen=True)
class TableAudit:
    """单表审计结果。"""

    name: str
    rows: int
    start: str
    end: str
    duplicate_timestamps: int
    invalid_timestamps: int
    missing_cells: int
    missing_grid_rows: int


@dataclass(frozen=True)
class DatasetAudit:
    """完整数据集审计结果。"""

    frequency: str
    grid_rows: int
    structural_empty_columns: tuple[str, ...]
    tables: tuple[TableAudit, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AlignedDataset:
    """对齐后的数据及其审计记录。"""

    frame: pd.DataFrame
    audit: DatasetAudit


def _classify_csv(path: Path) -> str | None:
    stem = path.stem.lower()
    for name in ("gas_holder", "gas_user", "load", "gas"):
        if stem.endswith(name):
            return name
    return None


def discover_tables(data_dir: str | Path) -> dict[str, Path]:
    """按稳定后缀识别四张表，兼容训练与测试文件名前缀。"""

    root = Path(data_dir)
    if not root.is_dir():
        raise DataContractError(f"数据目录不存在: {root}")

    discovered: dict[str, Path] = {}
    for path in sorted(root.glob("*.csv")):
        name = _classify_csv(path)
        if name is None:
            continue
        if name in discovered:
            raise DataContractError(f"发现多个 {name} 文件: {discovered[name]} 与 {path}")
        discovered[name] = path

    missing = [name for name in TABLE_ORDER if name not in discovered]
    if missing:
        raise DataContractError(f"缺少赛事数据表: {', '.join(missing)}")
    return discovered


def read_timeseries(path: str | Path) -> tuple[pd.DataFrame, int, int]:
    """读取单表并按时间排序；重复边界记录保留最后一条。"""

    frame = pd.read_csv(path)
    if "datetime" not in frame.columns:
        raise DataContractError(f"{path} 缺少 datetime 列")

    frame = frame.copy()
    parsed = pd.to_datetime(frame["datetime"], errors="coerce")
    invalid_count = int(parsed.isna().sum())
    if invalid_count:
        raise DataContractError(f"{path} 含 {invalid_count} 个非法时间戳")
    frame["datetime"] = parsed

    duplicate_count = int(frame["datetime"].duplicated(keep=False).sum())
    frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return frame.set_index("datetime"), duplicate_count, invalid_count


def align_tables(data_dir: str | Path, frequency: str = "15min") -> AlignedDataset:
    """建立完整时间网格并仅按时间戳连接四表。"""

    paths = discover_tables(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    duplicate_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}

    for name, path in paths.items():
        table, duplicates, invalid = read_timeseries(path)
        tables[name] = table
        duplicate_counts[name] = duplicates
        invalid_counts[name] = invalid

    starts = [table.index.min() for table in tables.values()]
    ends = [table.index.max() for table in tables.values()]
    if any(pd.isna(value) for value in starts + ends):
        raise DataContractError("至少一张赛事数据表为空")

    grid = pd.date_range(min(starts), max(ends), freq=frequency, name="datetime")
    aligned_parts: list[pd.DataFrame] = []
    audits: list[TableAudit] = []
    seen_columns: set[str] = set()

    for name in TABLE_ORDER:
        table = tables[name]
        collisions = seen_columns.intersection(table.columns)
        if collisions:
            raise DataContractError(f"跨表字段重名: {sorted(collisions)}")
        seen_columns.update(table.columns)

        present = pd.Series(True, index=table.index).reindex(grid, fill_value=False)
        reindexed = table.reindex(grid)
        reindexed[f"feat_missing_row_{name}"] = (~present).astype("int8")
        aligned_parts.append(reindexed)
        audits.append(
            TableAudit(
                name=name,
                rows=len(table),
                start=str(table.index.min()),
                end=str(table.index.max()),
                duplicate_timestamps=duplicate_counts[name],
                invalid_timestamps=invalid_counts[name],
                missing_cells=int(table.isna().sum().sum()),
                missing_grid_rows=int((~present).sum()),
            )
        )

    aligned = pd.concat(aligned_parts, axis=1)
    structural_empty = tuple(
        column
        for column in aligned.columns
        if not column.startswith("feat_") and aligned[column].isna().all()
    )
    if structural_empty:
        aligned = aligned.drop(columns=list(structural_empty))

    audit = DatasetAudit(
        frequency=frequency,
        grid_rows=len(grid),
        structural_empty_columns=structural_empty,
        tables=tuple(audits),
    )
    return AlignedDataset(frame=aligned, audit=audit)


def combine_context(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    prefer_test_at_overlap: bool = True,
) -> pd.DataFrame:
    """合并训练历史与评测输入，边界重叠只保留一条记录。"""

    keys = [train, test] if prefer_test_at_overlap else [test, train]
    combined = pd.concat(keys, axis=0).sort_index(kind="stable")
    return combined[~combined.index.duplicated(keep="last")]


def audit_summary(audit: DatasetAudit) -> Mapping[str, object]:
    """生成适合 JSON 序列化的审计摘要。"""

    return audit.to_dict()

