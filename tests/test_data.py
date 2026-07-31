from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gas_forecast.data import DataContractError, align_tables, combine_context, discover_tables


def _write_table(root: Path, suffix: str, values: dict[str, list[object]]) -> None:
    pd.DataFrame(values).to_csv(root / f"Pre_{suffix}.csv", index=False)


def _make_dataset(root: Path) -> None:
    times = ["2025-01-01 00:00:00", "2025-01-01 00:15:00", "2025-01-01 00:45:00"]
    _write_table(root, "gas", {"datetime": times, "blast_furnace_1": [1.0, 2.0, 4.0]})
    _write_table(
        root,
        "gas_holder",
        {"datetime": times, "blast_furnace_gas_holder_1": [None, None, None]},
    )
    _write_table(root, "gas_user", {"datetime": times, "blast_furnace_user1": [3, 4, 5]})
    _write_table(
        root,
        "load",
        {
            "datetime": times,
            "generator_1": [100, 101, 103],
            "generator_all": [200, 201, 203],
        },
    )


def test_discover_tables_requires_all_four_files(tmp_path: Path) -> None:
    _write_table(tmp_path, "gas", {"datetime": ["2025-01-01"], "x": [1]})
    with pytest.raises(DataContractError, match="缺少赛事数据表"):
        discover_tables(tmp_path)


def test_align_tables_builds_grid_and_drops_structural_empty_column(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    result = align_tables(tmp_path)

    assert len(result.frame) == 4
    assert result.frame.index[-1] == pd.Timestamp("2025-01-01 00:45:00")
    assert result.frame.loc["2025-01-01 00:30:00", "feat_missing_row_load"] == 1
    assert "blast_furnace_gas_holder_1" not in result.frame.columns
    assert result.audit.structural_empty_columns == ("blast_furnace_gas_holder_1",)


def test_combine_context_prefers_test_boundary_value() -> None:
    train = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:15"]),
    )
    test = pd.DataFrame(
        {"value": [20.0, 30.0]},
        index=pd.to_datetime(["2025-01-01 00:15", "2025-01-01 00:30"]),
    )

    combined = combine_context(train, test)

    assert combined.index.is_unique
    assert combined.loc["2025-01-01 00:15", "value"] == 20.0

