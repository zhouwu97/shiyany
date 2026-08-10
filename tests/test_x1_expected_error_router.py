from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gas_forecast.causal_trajectory_ensemble import HORIZONS, TARGETS
from gas_forecast.x1_expected_error_router import (
    DEFAULT_CANDIDATE_COLUMNS,
    PARENT_CANDIDATE,
    ROUTER_CANDIDATES,
    build_x1_report,
    evaluate_x1_result,
    fit_expected_error_models,
    load_x1_oof,
    route_expected_error,
)


def _synthetic_oof(rows_per_fold: int = 96, folds: int = 6) -> pd.DataFrame:
    """合成 19 折结构的缩水 OOF：7 候选 + actual + P3 融合列。"""

    records: list[pd.DataFrame] = []
    for fold_index in range(folds):
        origins = pd.date_range(
            f"2025-01-{1 + fold_index:02d}",
            periods=rows_per_fold // (len(TARGETS) * len(HORIZONS)),
            freq="15min",
        )
        for target in TARGETS:
            for horizon in HORIZONS:
                for origin in origins:
                    base = 100.0 if target == "generator_1" else 250.0
                    actual = base + np.sin(fold_index + origin.hour) * 10.0
                    row: dict[str, object] = {
                        "fold": f"dev_{fold_index + 1:02d}",
                        "origin_time": origin,
                        "train_end": origin - pd.Timedelta(minutes=15),
                        "target": target,
                        "horizon": horizon,
                        "actual": actual,
                    }
                    for offset, name in enumerate(ROUTER_CANDIDATES):
                        column = DEFAULT_CANDIDATE_COLUMNS[name]
                        if name == "x3_catboost":
                            row[column] = actual + 1.0 + offset * 0.05
                        elif name == "p3_static":
                            row[column] = actual + 0.2
                        else:
                            row[column] = actual + (0.5 + offset * 0.3) + np.sin(origin.hour) * 0.1
                    records.append(pd.DataFrame([row]))
    return pd.concat(records, ignore_index=True)


def test_load_x1_oof_requires_x3_column() -> None:
    rows = _synthetic_oof(folds=4)
    rows = rows.drop(columns=DEFAULT_CANDIDATE_COLUMNS["x3_catboost"])
    tmp = Path(__import__("tempfile").mkdtemp()) / "oof.csv"
    rows.to_csv(tmp, index=False)
    with pytest.raises(ValueError, match="缺少 X3 候选列"):
        load_x1_oof(
            tmp,
            expected_folds=4,
            expected_origins=rows["origin_time"].nunique(),
            expected_rows=len(rows),
        )


def test_load_x1_oof_with_x3_merges_keys(tmp_path: Path) -> None:
    rows = _synthetic_oof(folds=4)
    x3 = rows[["fold", "origin_time", "train_end", "target", "horizon"]].copy()
    x3["a57b_residual_a51_cat10_pred"] = rows[DEFAULT_CANDIDATE_COLUMNS["x3_catboost"]]
    integration = rows.drop(columns=DEFAULT_CANDIDATE_COLUMNS["x3_catboost"])
    oof_path = tmp_path / "oof.csv"
    x3_path = tmp_path / "x3.csv"
    integration.to_csv(oof_path, index=False)
    x3.to_csv(x3_path, index=False)
    loaded = load_x1_oof(
        oof_path,
        x3_oof=x3_path,
        expected_folds=4,
        expected_origins=rows["origin_time"].nunique(),
        expected_rows=len(rows),
    )
    assert loaded["x3_catboost__prediction"].notna().all()
    assert len(loaded) == len(rows)


def test_route_prior_mode_is_causal_and_falls_back() -> None:
    rows = _synthetic_oof(folds=6)
    result = route_expected_error(
        rows,
        confidence_min_pp=0.0,
        min_history_folds=2,
        mode="prior",
    )
    routed = result.rows
    assert routed["x1_prediction"].notna().all()
    assert np.isfinite(routed["x1_prediction"].to_numpy(dtype=float)).all()
    # 早期折无足够历史 → 回退 A61。
    early = routed[routed["fold"].eq("dev_01")]
    assert (early["x1_selected"] == PARENT_CANDIDATE).all()
    assert (early["x1_reason"] == "insufficient_history").all()
    # 路由预测与任何候选列和实际都有限。
    assert routed["x1_prediction"].between(
        routed[DEFAULT_CANDIDATE_COLUMNS[PARENT_CANDIDATE]].min(),
        routed[DEFAULT_CANDIDATE_COLUMNS[PARENT_CANDIDATE]].max(),
    ).any()


def test_route_lightgbm_mode_matches_prior_structure() -> None:
    rows = _synthetic_oof(folds=6)
    result = route_expected_error(
        rows,
        confidence_min_pp=0.005,
        min_history_folds=2,
        mode="lightgbm",
    )
    routed = result.rows
    assert routed["x1_prediction"].notna().all()
    assert np.isfinite(routed["x1_prediction"].to_numpy(dtype=float)).all()
    assert set(routed["x1_reason"].unique()) <= {
        "insufficient_history",
        "best_is_parent",
        "confidence_below_0.005pp",
        "soft_blend_a61_parent_a64_direct_delta",
        "soft_blend_a61_parent_p3_static",
        "soft_blend_p3_static_a61_parent",
        "soft_blend_a61_parent_x3_catboost",
        "soft_blend_x3_catboost_a61_parent",
        "soft_blend_p3_static_x3_catboost",
        "soft_blend_x3_catboost_p3_static",
        "soft_blend_a61_parent_p2_historical_analog",
        "soft_blend_a64_direct_delta_a61_parent",
        "soft_blend_p3_static_a64_direct_delta",
        "soft_blend_a64_direct_delta_p3_static",
        "soft_blend_x3_catboost_a64_direct_delta",
        "soft_blend_a64_direct_delta_x3_catboost",
        "soft_blend_a61_parent_p1_causal_rolling",
        "soft_blend_p1_causal_rolling_a61_parent",
        "soft_blend_a61_parent_p2_matured_residual",
        "soft_blend_p2_matured_residual_a61_parent",
        "soft_blend_p3_static_p2_historical_analog",
        "soft_blend_p2_historical_analog_p3_static",
        "soft_blend_p3_static_p1_causal_rolling",
        "soft_blend_p1_causal_rolling_p3_static",
        "soft_blend_p3_static_p2_matured_residual",
        "soft_blend_p2_matured_residual_p3_static",
    }


def test_fit_expected_error_models_rejects_short_history() -> None:
    rows = _synthetic_oof(folds=2)
    with pytest.raises(ValueError, match="历史折不足"):
        fit_expected_error_models(rows, min_history_folds=3)


def test_evaluate_x1_result_reports_gates() -> None:
    rows = _synthetic_oof(folds=6)
    result = route_expected_error(rows, confidence_min_pp=0.0, min_history_folds=2, mode="prior")
    evaluation = evaluate_x1_result(result)
    assert set(evaluation["gates"]["checks"]) == {
        "pooled_improvement",
        "recent5_wins",
        "worst_fold_regression",
        "target_regression",
    }
    assert evaluation["coverage"]["total_cells"] == len(rows)
    report = build_x1_report(result, evaluation)
    assert report["held_labels_used"] is False
    assert report["future_folds_used"] is False
    json.dumps(report, ensure_ascii=False, default=str)
