"""E23 工业时延关系发现与确认。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_RELATION_SOURCES = (
    "generator_use_blast_furnace_gas",
    "generator_use_coke_gas",
    "generator_use_converter_gas",
    "blast_furnace_gas_holder_2",
    "generator_1",
    "generator_all",
    "generator_rest",
    "feat_bf_surplus_proxy",
    "feat_blast_balance",
    "feat_coke_balance",
    "feat_converter_balance",
)


def _corr(left: pd.Series, right: pd.Series) -> float:
    valid = np.isfinite(left.to_numpy(dtype=float)) & np.isfinite(right.to_numpy(dtype=float))
    if int(valid.sum()) < 32:
        return float("nan")
    x = left.to_numpy(dtype=float)[valid]
    y = right.to_numpy(dtype=float)[valid]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _source_series(frame: pd.DataFrame, source: str) -> pd.Series | None:
    if source in frame:
        return frame[source]
    if source == "generator_rest" and {"generator_all", "generator_1"}.issubset(frame.columns):
        return frame["generator_all"] - frame["generator_1"]
    if source.startswith("feat_"):
        return frame[source] if source in frame else None
    return None


def build_residual_relation_scan(
    frame: pd.DataFrame,
    oof_rows: pd.DataFrame,
    *,
    prediction_column: str,
    sources: Iterable[str] = DEFAULT_RELATION_SOURCES,
    max_lag: int = 16,
    max_horizon: int = 8,
) -> pd.DataFrame:
    """基于真正外层 OOF residual 计算 lag×horizon 三套相关性。"""

    required = {"origin_time", "target", "horizon", "actual", "current_value", prediction_column}
    missing = sorted(required.difference(oof_rows.columns))
    if missing:
        raise ValueError(f"E23 OOF 缺少字段: {missing}")
    work = oof_rows.copy()
    work["origin_time"] = pd.to_datetime(work["origin_time"])
    work["residual"] = work["actual"] - work[prediction_column]
    work["delta_actual"] = work["actual"] - work["current_value"]
    source_map: dict[str, pd.Series] = {}
    for source in sources:
        value = _source_series(frame, source)
        if value is not None:
            source_map[source] = value
    if not source_map:
        raise ValueError("E23 没有可用关系来源字段")
    records: list[dict[str, object]] = []
    for source, series in source_map.items():
        for lag in range(max_lag + 1):
            lagged = series.shift(lag).reindex(work["origin_time"]).set_axis(work.index)
            for horizon in range(1, max_horizon + 1):
                subset = work.loc[work["horizon"].eq(15 * horizon)]
                values = lagged.loc[subset.index]
                records.append(
                    {
                        "source": source,
                        "lag": lag,
                        "horizon": horizon,
                        "target": "all_targets",
                        "corr_actual": _corr(values, subset["actual"]),
                        "corr_delta": _corr(values, subset["delta_actual"]),
                        "corr_residual": _corr(values, subset["residual"]),
                        "rows": int(values.notna().sum()),
                    }
                )
    return pd.DataFrame.from_records(records)


def add_stability_diagnostics(
    scan: pd.DataFrame,
    oof_rows: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    month_column: str = "month",
) -> pd.DataFrame:
    """为候选关系补充月份和目标子空间的方向稳定性。"""

    if scan.empty:
        return scan
    work = oof_rows.copy()
    work["origin_time"] = pd.to_datetime(work["origin_time"])
    work["residual"] = work["actual"] - work[prediction_column]
    work["delta_actual"] = work["actual"] - work["current_value"]
    work["month"] = work["origin_time"].dt.month.astype(int)
    enriched = scan.copy()
    month_records: list[dict[str, object]] = []
    for (source, lag, horizon), group in scan.groupby(["source", "lag", "horizon"], sort=False):
        series = _source_series(frame, str(source))
        if series is None:
            continue
        lagged = series.shift(int(lag)).reindex(work["origin_time"]).set_axis(work.index)
        subset = work.loc[work["horizon"].eq(15 * int(horizon))]
        for month, month_rows in subset.groupby(month_column, sort=True):
            month_records.append(
                {
                    "source": source,
                    "lag": int(lag),
                    "horizon": int(horizon),
                    "month": int(month),
                    "corr_residual": _corr(lagged.loc[month_rows.index], month_rows["residual"]),
                    "corr_delta": _corr(lagged.loc[month_rows.index], month_rows["delta_actual"]),
                }
            )
    if not month_records:
        return enriched
    monthly = pd.DataFrame.from_records(month_records)
    summary = monthly.groupby(["source", "lag", "horizon"], sort=False).agg(
        month_direction_consistency=("corr_residual", lambda values: float(
            abs(np.sign(values.dropna().to_numpy()).sum()) / len(values.dropna())
            if len(values.dropna()) else np.nan
        )),
        month_count=("corr_residual", lambda values: int(values.notna().sum())),
        month_abs_corr_mean=("corr_residual", lambda values: float(values.abs().mean())),
    ).reset_index()
    return enriched.merge(summary, on=["source", "lag", "horizon"], how="left")


def freeze_relation_features(
    scan: pd.DataFrame,
    *,
    max_features: int = 20,
    min_month_count: int = 2,
) -> list[str]:
    """冻结少量连续且跨月稳定的 relation specs。"""

    if scan.empty:
        return []
    work = scan.copy()
    if "month_count" in work:
        work = work.loc[work["month_count"].fillna(0).ge(min_month_count)]
    work = work.loc[work["corr_residual"].notna()]
    if "month_direction_consistency" in work:
        work = work.loc[work["month_direction_consistency"].fillna(0.0).ge(0.6)]
    if work.empty:
        return []
    work["score"] = work["corr_residual"].abs()
    work["horizon_group"] = np.where(work["horizon"].astype(int).le(4), "short", "long")
    work = work.sort_values(
        ["score", "rows"], ascending=[False, False], kind="stable"
    )
    selected: list[str] = []
    seen_source_h: set[tuple[str, int]] = set()
    for row in work.itertuples(index=False):
        key = (str(row.source), int(row.horizon))
        if key in seen_source_h and len(selected) >= max_features // 2:
            continue
        selected.append(f"{row.source}|{int(row.lag)}|{int(row.horizon)}")
        seen_source_h.add(key)
        if len(selected) >= max_features:
            break
    return selected


def write_relation_scan(path: str | Path, scan: pd.DataFrame, frozen: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": scan.to_dict(orient="records"), "frozen_relation_features": frozen}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
