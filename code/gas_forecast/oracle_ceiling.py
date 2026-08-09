"""X0 — P3 Oracle Ceiling Audit（标签知情的诊断性路由上限）。

只读复用 P3 滚动训练的 development OOF（A61 anchor + P1/P2/A64 四条候选），
不训练任何模型，不读取 blind/final/平台参考。所有 oracle 指标均为
label-informed 诊断：``label_informed_diagnostic=true``、
``formal_candidate=false``，绝不进入 ``results/best`` 或正式提交。

Oracle 定义（统一 pooled cell MAPE，epsilon=1e-6，与 ``competition_mape``
一致；oracle 选择均为离散候选 argmin，平局取候选声明顺序的第一个）：
  row oracle         每行取绝对误差最小的候选（逐行选择轨迹）
  target oracle      每个 target 内取 MAPE 最小的候选
  horizon oracle     每个 horizon 内取 MAPE 最小的候选
  target×horizon     每个 (target, horizon) 单元内取 MAPE 最小的候选
  origin oracle      每个 (fold, origin_time) 内取 MAPE 最小的候选
  fold oracle        每个 fold 内取 MAPE 最小的候选
  split-half oracle  每折按时间把 origin 对半：前半选择候选、后半评估；
                     双向各评估一次，界定后见乐观

预注册判定（阈值固定为 ``ROW_ORACLE_THRESHOLD = 0.049``，不可修改）：
row oracle MAPE <= 4.9% 说明现有候选存在显著动态路由空间，否则优先新基模型。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from gas_forecast.causal_trajectory_ensemble import (
    HORIZONS,
    KEY_COLUMNS,
    TARGETS,
    canonicalize_oof,
)
from gas_forecast.scoring import competition_mape

ORACLE_CANDIDATES: tuple[str, ...] = (
    "a61_parent",
    "a64_direct_delta",
    "p1_causal_rolling",
    "p2_historical_analog",
    "p2_matured_residual",
)
P3_FUSION_COLUMN = "prediction"
SELECTED_CANDIDATE_COLUMN = "selected_candidate"
ROW_ORACLE_THRESHOLD = 0.049  # 预注册判定阈值，不可修改
EXPECTED_FOLDS = 19
EXPECTED_ORIGINS = 3648
EXPECTED_ROWS = 58368
CELLS_PER_ORIGIN = len(TARGETS) * len(HORIZONS)
IDENTITY_COLUMNS: tuple[str, ...] = tuple(KEY_COLUMNS[:-1])

ROUTE_OOF_FILES: Mapping[str, str] = {
    "a64_direct_delta": "a64_direct_delta_oof.csv",
    "p1_causal_rolling": "p1_causal_rolling_oof.csv",
    "p2_historical_analog": "p2_historical_analog_oof.csv",
    "p2_matured_residual": "p2_matured_residual_oof.csv",
}
ROUTE_PREDICTION_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "a64_direct_delta": ("ridge_prediction", "prediction"),
    "p1_causal_rolling": ("prediction", "causal_rolling_prediction"),
    "p2_historical_analog": ("prediction",),
    "p2_matured_residual": ("prediction",),
}

_GROUP_LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("target", ("target",)),
    ("horizon", ("horizon",)),
    ("target_x_horizon", ("target", "horizon")),
    ("origin", ("fold", "origin_time")),
    ("fold", ("fold",)),
)


def candidate_column(name: str) -> str:
    """返回候选在 P3 集成 OOF 中的预测列名。"""

    if name not in ORACLE_CANDIDATES:
        raise ValueError(f"未知候选: {name}")
    return f"{name}__prediction"


def _candidate_columns() -> tuple[str, ...]:
    return tuple(candidate_column(name) for name in ORACLE_CANDIDATES)


def _gap_pp(current: float, oracle: float) -> float:
    """oracle gap（百分点）；正数表示 oracle 相对当前参考更优。"""

    return (current - oracle) * 100.0


def validate_p3_integration_oof(
    rows: pd.DataFrame,
    *,
    source: str = "p3_integration",
    expected_folds: int = EXPECTED_FOLDS,
    expected_origins: int = EXPECTED_ORIGINS,
    expected_rows: int = EXPECTED_ROWS,
) -> dict[str, object]:
    """验证 P3 集成 OOF 的完整键契约：折数、origin 数、行数、无 blind。

    任一检查失败立即抛 ``ValueError``，不允许通过 inner join 静默丢行。
    """

    required = (
        set(KEY_COLUMNS) | set(_candidate_columns()) | {P3_FUSION_COLUMN, SELECTED_CANDIDATE_COLUMN}
    )
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"{source} OOF 缺少字段: {missing}")
    work = rows.copy()
    work["fold"] = work["fold"].astype(str)
    work["origin_time"] = pd.to_datetime(work["origin_time"], errors="coerce")
    work["train_end"] = pd.to_datetime(work["train_end"], errors="coerce")
    work["target"] = work["target"].astype(str)
    work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce")
    if work[["origin_time", "train_end"]].isna().any().any():
        raise ValueError(f"{source} OOF 含无法解析的时间键")
    if work["horizon"].isna().any() or not np.equal(work["horizon"] % 1, 0).all():
        raise ValueError(f"{source} OOF 的 horizon 必须是整数")
    work["horizon"] = work["horizon"].astype(int)
    numeric = ["actual", *_candidate_columns(), P3_FUSION_COLUMN]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[numeric].isna().any().any():
        raise ValueError(f"{source} OOF 含空标签或预测")
    if not np.isfinite(work[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} OOF 含 NaN/Inf")
    if work[list(IDENTITY_COLUMNS)].isna().any().any():
        raise ValueError(f"{source} OOF 主键含空值")
    if work.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{source} OOF 主键不唯一")
    if work["fold"].str.lower().str.contains("blind").any():
        raise ValueError(f"{source} OOF 包含 blind 折")
    if not work["train_end"].lt(work["origin_time"]).all():
        raise ValueError(f"{source} OOF 存在 train_end 不早于 origin_time 的行")

    by_origin = work.groupby(["fold", "origin_time"], sort=True)
    if by_origin.size().ne(CELLS_PER_ORIGIN).any():
        raise ValueError(f"{source} OOF 存在不完整 target×horizon origin 矩阵")
    expected_pairs = {(target, horizon) for target in TARGETS for horizon in HORIZONS}
    for (fold, origin), part in by_origin:
        pairs = set(zip(part["target"], part["horizon"], strict=True))
        if pairs != expected_pairs:
            raise ValueError(f"{source} OOF origin {fold}|{origin} 覆盖不完整")

    fold_train_ends = work.groupby("fold", sort=True)["train_end"].nunique()
    if fold_train_ends.gt(1).any():
        bad = fold_train_ends.loc[fold_train_ends.gt(1)].index.tolist()
        raise ValueError(f"{source} OOF 每个 fold 必须只有一个 train_end: {bad}")
    per_fold_origins = work.groupby("fold", sort=True)["origin_time"].nunique()
    if per_fold_origins.nunique() != 1:
        raise ValueError(f"{source} OOF 各 fold 的 origin 数量不一致")
    for fold, part in work.groupby("fold", sort=True):
        origins = pd.DatetimeIndex(sorted(part["origin_time"].unique()))
        gaps = origins.to_series().diff().dropna()
        if not gaps.empty and not gaps.eq(pd.Timedelta(minutes=15)).all():
            raise ValueError(f"{source} OOF fold {fold} 的 origin 不是连续 15 分钟网格")

    fold_count = int(work["fold"].nunique())
    origin_count = int(per_fold_origins.sum())
    row_count = int(len(work))
    if fold_count != expected_folds:
        raise ValueError(f"{source} OOF 折数不符: observed={fold_count} expected={expected_folds}")
    if origin_count != expected_origins:
        raise ValueError(
            f"{source} OOF origin 数不符: observed={origin_count} expected={expected_origins}"
        )
    if row_count != expected_rows:
        raise ValueError(f"{source} OOF 行数不符: observed={row_count} expected={expected_rows}")
    checks = {
        "folds": fold_count == expected_folds,
        "origins": origin_count == expected_origins,
        "rows": row_count == expected_rows,
        "no_blind": True,
        "unique_keys": True,
        "complete_origin_matrix": True,
        "single_train_end_per_fold": True,
        "contiguous_15min_origins": True,
    }
    return {
        "source": source,
        "rows": row_count,
        "folds": fold_count,
        "origins": origin_count,
        "origins_per_fold": int(per_fold_origins.iloc[0]),
        "targets": sorted(work["target"].unique().tolist()),
        "horizons": sorted(work["horizon"].unique().tolist()),
        "blind_folds": 0,
        "expected": {
            "folds": expected_folds,
            "origins": expected_origins,
            "rows": expected_rows,
        },
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }


def load_p3_integration_oof(
    path: str | Path,
    *,
    expected_folds: int = EXPECTED_FOLDS,
    expected_origins: int = EXPECTED_ORIGINS,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """读取并验证 P3 集成 OOF，返回规范化长表与键契约摘要。"""

    value = Path(path)
    if value.suffix.lower() == ".parquet":
        rows = pd.read_parquet(value)
    else:
        rows = pd.read_csv(value)
    summary = validate_p3_integration_oof(
        rows,
        expected_folds=expected_folds,
        expected_origins=expected_origins,
        expected_rows=expected_rows,
    )
    canonical = rows.copy()
    canonical["fold"] = canonical["fold"].astype(str)
    canonical["origin_time"] = pd.to_datetime(canonical["origin_time"])
    canonical["train_end"] = pd.to_datetime(canonical["train_end"])
    canonical["target"] = canonical["target"].astype(str)
    canonical["horizon"] = canonical["horizon"].astype(int)
    for column in ["actual", *_candidate_columns(), P3_FUSION_COLUMN]:
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce")
    canonical = canonical.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    return canonical, summary


def _row_oracle(rows: pd.DataFrame) -> dict[str, object]:
    """逐行绝对误差 argmin；返回 oracle MAPE、命中率与分组命中率。"""

    actual = rows["actual"].to_numpy(dtype=float)
    errors = np.column_stack(
        [
            np.abs(actual - rows[candidate_column(name)].to_numpy(dtype=float))
            for name in ORACLE_CANDIDATES
        ]
    )
    winner_index = np.argmin(errors, axis=1)
    oracle_pred = np.choose(
        winner_index,
        np.column_stack(
            [rows[candidate_column(name)].to_numpy(dtype=float) for name in ORACLE_CANDIDATES]
        ).T,
    )
    hit = np.bincount(winner_index, minlength=len(ORACLE_CANDIDATES)) / len(rows)
    hit_rate = {name: float(hit[i]) for i, name in enumerate(ORACLE_CANDIDATES)}

    def subset_hit(mask: np.ndarray) -> dict[str, float]:
        if not mask.any():
            return {name: 0.0 for name in ORACLE_CANDIDATES}
        sub = np.bincount(winner_index[mask], minlength=len(ORACLE_CANDIDATES)) / int(mask.sum())
        return {name: float(sub[i]) for i, name in enumerate(ORACLE_CANDIDATES)}

    by_target = {str(target): subset_hit(rows["target"].to_numpy() == target) for target in TARGETS}
    by_horizon = {
        int(horizon): subset_hit(rows["horizon"].to_numpy() == horizon) for horizon in HORIZONS
    }
    return {
        "mape": competition_mape(actual, oracle_pred),
        "distinct_selected": int(np.unique(winner_index).size),
        "hit_rate": hit_rate,
        "hit_rate_by_target": by_target,
        "hit_rate_by_horizon": by_horizon,
    }


def _group_oracle(
    rows: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    level: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """在 ``group_columns`` 单元内选 MAPE 最小候选并池化评估。"""

    actual = rows["actual"].to_numpy(dtype=float)
    pred_matrix = np.column_stack(
        [rows[candidate_column(name)].to_numpy(dtype=float) for name in ORACLE_CANDIDATES]
    )
    name_to_index = {name: i for i, name in enumerate(ORACLE_CANDIDATES)}
    selected_index = np.empty(len(rows), dtype=int)
    winners: list[dict[str, object]] = []
    for key, part in rows.groupby(list(group_columns), sort=True):
        part_actual = part["actual"].to_numpy(dtype=float)
        part_matrix = pred_matrix[part.index.to_numpy(), :]
        scores = {
            name: competition_mape(part_actual, part_matrix[:, i])
            for name, i in name_to_index.items()
        }
        winner = min(scores, key=scores.get)
        selected_index[part.index.to_numpy()] = name_to_index[winner]
        group = key if isinstance(key, tuple) else (key,)
        winners.append(
            {
                "level": level,
                "group": "|".join(str(item) for item in group),
                "rows": int(len(part)),
                "winner": winner,
                "winner_mape": float(scores[winner]),
            }
        )
    oracle_pred = np.choose(selected_index, pred_matrix.T)
    return (
        {
            "mape": competition_mape(actual, oracle_pred),
            "distinct_selected": int(np.unique(selected_index).size),
            "groups": int(len(winners)),
        },
        winners,
    )


def _best_candidate(part: pd.DataFrame) -> str:
    """返回单元内 MAPE 最小的候选。"""

    actual = part["actual"].to_numpy(dtype=float)
    scores = {
        name: competition_mape(actual, part[candidate_column(name)].to_numpy(dtype=float))
        for name in ORACLE_CANDIDATES
    }
    return min(scores, key=scores.get)


def _split_half_oracle(
    rows: pd.DataFrame,
    *,
    a61_mape: float,
    p3_mape: float,
) -> dict[str, object]:
    """每折按时间对半切分 origin：前半选候选、后半评估；双向各一次。"""

    work = rows.sort_values(["fold", "origin_time"], kind="stable")
    per_fold: list[dict[str, object]] = []
    first_actual: list[np.ndarray] = []
    first_pred: list[np.ndarray] = []
    second_actual: list[np.ndarray] = []
    second_pred: list[np.ndarray] = []
    for fold, part in work.groupby("fold", sort=True):
        origins = sorted(part["origin_time"].unique())
        half = len(origins) // 2
        first_origins = set(origins[:half])
        second_origins = set(origins[half:])
        first_rows = part.loc[part["origin_time"].isin(first_origins)]
        second_rows = part.loc[part["origin_time"].isin(second_origins)]
        first_winner = _best_candidate(first_rows)
        second_winner = _best_candidate(second_rows)
        f2s_mape = competition_mape(
            second_rows["actual"].to_numpy(dtype=float),
            second_rows[candidate_column(first_winner)].to_numpy(dtype=float),
        )
        s2f_mape = competition_mape(
            first_rows["actual"].to_numpy(dtype=float),
            first_rows[candidate_column(second_winner)].to_numpy(dtype=float),
        )
        per_fold.append(
            {
                "fold": str(fold),
                "origins": len(origins),
                "first_origins": len(first_origins),
                "second_origins": len(second_origins),
                "first_half_winner": first_winner,
                "second_half_winner": second_winner,
                "first_to_second_eval_mape": f2s_mape,
                "second_to_first_eval_mape": s2f_mape,
            }
        )
        second_actual.append(second_rows["actual"].to_numpy(dtype=float))
        second_pred.append(second_rows[candidate_column(first_winner)].to_numpy(dtype=float))
        first_actual.append(first_rows["actual"].to_numpy(dtype=float))
        first_pred.append(first_rows[candidate_column(second_winner)].to_numpy(dtype=float))
    f2s_mape = competition_mape(np.concatenate(second_actual), np.concatenate(second_pred))
    s2f_mape = competition_mape(np.concatenate(first_actual), np.concatenate(first_pred))
    return {
        "method": (
            "per fold: origins time-sorted; first half selects, "
            "second half evaluates; bidirectional"
        ),
        "first_to_second": {
            "mape": f2s_mape,
            "gap_pp_vs_a61": _gap_pp(a61_mape, f2s_mape),
            "gap_pp_vs_p3": _gap_pp(p3_mape, f2s_mape),
        },
        "second_to_first": {
            "mape": s2f_mape,
            "gap_pp_vs_a61": _gap_pp(a61_mape, s2f_mape),
            "gap_pp_vs_p3": _gap_pp(p3_mape, s2f_mape),
        },
        "combined_mean_mape": (f2s_mape + s2f_mape) / 2.0,
        "per_fold": per_fold,
    }


def pre_registered_verdict(row_oracle_mape: float) -> dict[str, object]:
    """预注册判定：row oracle <=4.9% 说明存在显著动态路由空间。"""

    if row_oracle_mape <= ROW_ORACLE_THRESHOLD:
        verdict = "DYNAMIC_ROUTING_SPACE_EXISTS"
        conclusion = (
            "row oracle 不高于 4.9%，现有候选存在显著动态路由空间，"
            "值得继续开发可部署的动态路由；本结论仅限诊断。"
        )
    else:
        verdict = "PREFER_NEW_BASE_MODEL"
        conclusion = (
            "row oracle 高于 4.9% 门槛，逐行选择的理论上限不足，优先开发新基模型而非动态路由。"
        )
    return {
        "rule": (
            "row oracle <= 4.9% 表示现有候选存在显著动态路由空间，"
            "否则优先新基模型；阈值固定为 0.049，不得修改"
        ),
        "threshold": ROW_ORACLE_THRESHOLD,
        "row_oracle_mape": float(row_oracle_mape),
        "verdict": verdict,
        "conclusion": conclusion,
    }


def run_oracle_ceiling(
    rows: pd.DataFrame,
    *,
    source: str = "p3_integration",
    expected_folds: int = EXPECTED_FOLDS,
    expected_origins: int = EXPECTED_ORIGINS,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """对规范化的 P3 集成 OOF 执行完整 oracle ceiling 审计。

    返回 (report, winners)；``winners`` 是各分组单元的最优候选明细，
    供 ``oracle_winners.csv`` 落盘。所有 oracle 均标记诊断专用。
    """

    summary = validate_p3_integration_oof(
        rows,
        source=source,
        expected_folds=expected_folds,
        expected_origins=expected_origins,
        expected_rows=expected_rows,
    )
    canonical = rows.copy()
    canonical["fold"] = canonical["fold"].astype(str)
    canonical["origin_time"] = pd.to_datetime(canonical["origin_time"])
    canonical["train_end"] = pd.to_datetime(canonical["train_end"])
    canonical["target"] = canonical["target"].astype(str)
    canonical["horizon"] = canonical["horizon"].astype(int)
    canonical = canonical.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    actual = canonical["actual"].to_numpy(dtype=float)
    current_mape = {
        name: competition_mape(actual, canonical[candidate_column(name)].to_numpy(dtype=float))
        for name in ORACLE_CANDIDATES
    }
    current_mape["p3_fusion"] = competition_mape(
        actual, canonical[P3_FUSION_COLUMN].to_numpy(dtype=float)
    )
    a61_mape = current_mape["a61_parent"]
    p3_mape = current_mape["p3_fusion"]

    oracle: dict[str, object] = {}
    winners: list[dict[str, object]] = []
    row = _row_oracle(canonical)
    oracle["row"] = {
        "mape": row["mape"],
        "gap_pp_vs_a61": _gap_pp(a61_mape, row["mape"]),
        "gap_pp_vs_p3": _gap_pp(p3_mape, row["mape"]),
        "distinct_selected": row["distinct_selected"],
        "hit_rate": row["hit_rate"],
        "hit_rate_by_target": row["hit_rate_by_target"],
        "hit_rate_by_horizon": row["hit_rate_by_horizon"],
    }
    for level, group_columns in _GROUP_LEVELS:
        summary_level, winners_level = _group_oracle(canonical, group_columns, level=level)
        oracle[level] = {
            "mape": summary_level["mape"],
            "gap_pp_vs_a61": _gap_pp(a61_mape, summary_level["mape"]),
            "gap_pp_vs_p3": _gap_pp(p3_mape, summary_level["mape"]),
            "distinct_selected": summary_level["distinct_selected"],
            "groups": summary_level["groups"],
        }
        winners.extend(winners_level)
    split = _split_half_oracle(canonical, a61_mape=a61_mape, p3_mape=p3_mape)
    verdict = pre_registered_verdict(row["mape"])
    report: dict[str, object] = {
        "audit": "x0_p3_oracle_ceiling",
        "label_informed_diagnostic": True,
        "formal_candidate": False,
        "score_definition": "pooled cell MAPE; epsilon=1e-6",
        "candidates": list(ORACLE_CANDIDATES),
        "key_validation": summary,
        "current_mape": current_mape,
        "oracle": oracle,
        "split_half_oracle": split,
        "pre_registered_verdict": verdict,
    }
    return report, winners


def build_trajectory_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """逐行选择轨迹：候选预测、绝对误差、oracle 胜者与当前选择。"""

    work = rows.copy()
    actual = work["actual"].to_numpy(dtype=float)
    for name in ORACLE_CANDIDATES:
        column = candidate_column(name)
        work[f"{name}__abs_err"] = np.abs(actual - work[column].to_numpy(dtype=float))
    errors = np.column_stack(
        [work[f"{name}__abs_err"].to_numpy(dtype=float) for name in ORACLE_CANDIDATES]
    )
    winner_index = np.argmin(errors, axis=1)
    winners = np.asarray(ORACLE_CANDIDATES)[winner_index]
    work["row_oracle_winner"] = winners
    work["current_is_row_oracle"] = work[SELECTED_CANDIDATE_COLUMN].astype(str).eq(winners)
    columns = [
        *IDENTITY_COLUMNS,
        "actual",
        P3_FUSION_COLUMN,
        SELECTED_CANDIDATE_COLUMN,
        *_candidate_columns(),
        *[f"{name}__abs_err" for name in ORACLE_CANDIDATES],
        "row_oracle_winner",
        "current_is_row_oracle",
    ]
    return work.loc[:, columns]


def cross_check_route_oof_keys(rows: pd.DataFrame, routes_dir: str | Path) -> dict[str, object]:
    """把四条候选的独立 OOF 文件与集成 OOF 做完整键一致性核对。"""

    directory = Path(routes_dir)
    if not directory.is_dir():
        raise ValueError(f"routes-dir 不是目录: {directory}")
    integration_keys = rows.loc[:, list(KEY_COLUMNS)]
    routes: dict[str, object] = {}
    for name, filename in ROUTE_OOF_FILES.items():
        path = directory / filename
        if not path.exists():
            routes[name] = {"status": "MISSING_FILE", "path": str(path)}
            continue
        route_rows = pd.read_csv(path)
        prediction_column = next(
            (col for col in ROUTE_PREDICTION_CANDIDATES[name] if col in route_rows.columns),
            None,
        )
        if prediction_column is None:
            routes[name] = {
                "status": "NO_PREDICTION_COLUMN",
                "columns": sorted(route_rows.columns.tolist()),
            }
            continue
        canonical = canonicalize_oof(route_rows, source=name, prediction_column=prediction_column)
        merged = integration_keys.merge(
            canonical.loc[:, list(KEY_COLUMNS)],
            on=list(KEY_COLUMNS),
            how="outer",
            indicator=True,
        )
        counts = merged["_merge"].value_counts().to_dict()
        routes[name] = {
            "status": "OK",
            "prediction_column": prediction_column,
            "rows": int(len(canonical)),
            "integration_only": int(counts.get("left_only", 0)),
            "route_only": int(counts.get("right_only", 0)),
            "shared": int(counts.get("both", 0)),
            "key_consistent": (
                int(counts.get("left_only", 0)) == 0 and int(counts.get("right_only", 0)) == 0
            ),
        }
    return {"routes_dir": str(directory), "routes": routes}


def _sha256(path: Path) -> str:
    """返回文件 SHA-256（大写十六进制，与仓库文档一致）。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _hit_rate_rows(report: Mapping[str, object]) -> pd.DataFrame:
    """从报告展开行级候选命中率。"""

    row = report["oracle"]["row"]
    rows_out: list[dict[str, object]] = []
    for name in ORACLE_CANDIDATES:
        rows_out.append(
            {
                "candidate": name,
                "hit_rate": row["hit_rate"][name],
                "hit_rate_generator_1": row["hit_rate_by_target"]["generator_1"][name],
                "hit_rate_generator_all": row["hit_rate_by_target"]["generator_all"][name],
            }
        )
    return pd.DataFrame(rows_out)


def _gap_rows(report: Mapping[str, object]) -> pd.DataFrame:
    """从报告展开各层 oracle MAPE 与 gap。"""

    rows_out: list[dict[str, object]] = []
    oracle = report["oracle"]
    for level in ("row", "target", "horizon", "target_x_horizon", "origin", "fold"):
        item = oracle[level]
        rows_out.append(
            {
                "level": level,
                "mape": item["mape"],
                "gap_pp_vs_a61": item["gap_pp_vs_a61"],
                "gap_pp_vs_p3": item["gap_pp_vs_p3"],
                "distinct_selected": item["distinct_selected"],
            }
        )
    split = report["split_half_oracle"]
    for direction in ("first_to_second", "second_to_first"):
        item = split[direction]
        rows_out.append(
            {
                "level": f"split_half_{direction}",
                "mape": item["mape"],
                "gap_pp_vs_a61": item["gap_pp_vs_a61"],
                "gap_pp_vs_p3": item["gap_pp_vs_p3"],
                "distinct_selected": None,
            }
        )
    return pd.DataFrame(rows_out)


def _split_half_rows(report: Mapping[str, object]) -> pd.DataFrame:
    """展开 split-half 每折明细。"""

    rows_out: list[dict[str, object]] = []
    for item in report["split_half_oracle"]["per_fold"]:
        rows_out.append(
            {
                "fold": item["fold"],
                "origins": item["origins"],
                "first_origins": item["first_origins"],
                "second_origins": item["second_origins"],
                "first_half_winner": item["first_half_winner"],
                "second_half_winner": item["second_half_winner"],
                "first_to_second_eval_mape": item["first_to_second_eval_mape"],
                "second_to_first_eval_mape": item["second_to_first_eval_mape"],
            }
        )
    return pd.DataFrame(rows_out)


def write_oracle_ceiling_artifacts(
    report: Mapping[str, object],
    winners: Sequence[Mapping[str, object]],
    trajectory: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, str]:
    """把审计报告、明细 CSV 与拒绝型 manifest 写入全新输出目录。

    返回 {文件名: SHA-256}。该目录与 ``results/best``、正式提交完全隔离。
    """

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "report.json": (json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"),
        "oracle_selection_trajectory.csv": trajectory.to_csv(index=False, encoding="utf-8"),
        "oracle_winners.csv": pd.DataFrame(winners).to_csv(index=False, encoding="utf-8"),
        "hit_rates.csv": _hit_rate_rows(report).to_csv(index=False, encoding="utf-8"),
        "oracle_gaps.csv": _gap_rows(report).to_csv(index=False, encoding="utf-8"),
        "split_half_detail.csv": _split_half_rows(report).to_csv(index=False, encoding="utf-8"),
    }
    for name, content in payloads.items():
        (output / name).write_text(content, encoding="utf-8")
    hashes = {name: _sha256(output / name) for name in payloads}
    manifest = {
        "audit": "x0_p3_oracle_ceiling",
        "label_informed_diagnostic": True,
        "formal_candidate": False,
        "causal": False,
        "deployable": False,
        "blind_labels_used": False,
        "production_usage": "FORBIDDEN",
        "results_best_modified": False,
        "formal_submission_modified": False,
        "files": hashes,
    }
    manifest_path = output / "oracle_ceiling_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hashes["oracle_ceiling_manifest.json"] = _sha256(manifest_path)
    return hashes


__all__ = [
    "EXPECTED_FOLDS",
    "EXPECTED_ORIGINS",
    "EXPECTED_ROWS",
    "ORACLE_CANDIDATES",
    "P3_FUSION_COLUMN",
    "ROW_ORACLE_THRESHOLD",
    "SELECTED_CANDIDATE_COLUMN",
    "build_trajectory_frame",
    "candidate_column",
    "cross_check_route_oof_keys",
    "load_p3_integration_oof",
    "pre_registered_verdict",
    "run_oracle_ceiling",
    "validate_p3_integration_oof",
    "write_oracle_ceiling_artifacts",
]
