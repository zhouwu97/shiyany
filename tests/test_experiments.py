from __future__ import annotations

import re

from gas_forecast.experiments import new_run_dir


def test_new_run_dir_uses_date_first_readable_name(tmp_path) -> None:
    output = new_run_dir(tmp_path, "data_audit")

    assert output.is_dir()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{3}_数据审计结果",
        output.name,
    )
