from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gas_forecast.state_space import (
    STATE_SPACE_HORIZONS,
    STATE_SPACE_TARGETS,
    KalmanLocalLinearTrend,
    LocalLinearTrend,
    build_state_space_diversity,
    candidate_report,
    forecast_at_origin,
    future_perturbation_audit,
    full_development_gate,
    screening_gate,
)


def _frame(rows: int = 420) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "generator_1": 100.0 + 0.03 * phase + 2.0 * np.sin(phase / 14.0),
            "generator_all": 220.0 + 0.02 * phase + 3.0 * np.sin(phase / 19.0),
        },
        index=index,
    )


def _parent(frame: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    starts = [80 + 40 * i for i in range(folds)]
    for number, start in enumerate(starts, 1):
        train_end = frame.index[start - 10]
        for origin in frame.index[start : start + 12]:
            for target in STATE_SPACE_TARGETS:
                current = float(frame.loc[origin, target])
                for horizon in STATE_SPACE_HORIZONS:
                    actual = float(frame.loc[origin, target])
                    records.append(
                        {
                            "fold": f"dev_{number:02d}",
                            "origin_time": origin,
                            "train_end": train_end,
                            "target": target,
                            "horizon": horizon,
                            "actual": actual,
                            "current_value": current,
                            "persistence_pred": current,
                            "a61_recursive_blend_05_pred": current,
                        }
                    )
    return pd.DataFrame(records)


def test_local_and_kalman_have_two_state_eight_step_contract() -> None:
    values = np.linspace(10.0, 20.0, 50)
    local = LocalLinearTrend(0.85).fit(values).forecast()
    kalman = KalmanLocalLinearTrend(0.85).fit(values).forecast()
    assert local.shape == (8,)
    assert kalman.shape == (8,)
    assert np.isfinite(local).all()
    assert np.isfinite(kalman).all()


def test_origin_predictions_ignore_all_future_production_perturbations() -> None:
    frame = _frame()
    origin = frame.index[200]
    baseline = forecast_at_origin(
        frame,
        origin,
        "generator_1",
        model="kalman",
        damping=0.85,
    )
    for method in ("extreme", "shuffle", "null", "delete"):
        changed = frame.copy()
        if method == "extreme":
            changed.loc[changed.index > origin, :] = -999_999.0
        elif method == "shuffle":
            future = changed.loc[changed.index > origin].to_numpy(copy=True)
            changed.loc[changed.index > origin, :] = future[::-1]
        elif method == "null":
            changed.loc[changed.index > origin, :] = np.nan
        else:
            changed = changed.loc[:origin]
        np.testing.assert_allclose(
            baseline,
            forecast_at_origin(changed, origin, "generator_1", model="kalman", damping=0.85),
            rtol=0.0,
            atol=1e-12,
        )


def test_builder_writes_all_candidate_families_and_never_accepts_blind() -> None:
    frame = _frame()
    parent = _parent(frame)
    result = build_state_space_diversity(frame, parent, scope="screening")
    assert len(result.rows) == len(parent)
    for column in (
        "persistence_pred",
        "local_trend_pred",
        "kalman_pred",
        "parent_pred",
        "parent_local_trend_blend_05_pred",
        "parent_local_trend_blend_10_pred",
        "parent_local_trend_blend_20_pred",
        "parent_kalman_blend_05_pred",
        "parent_kalman_blend_10_pred",
        "parent_kalman_blend_20_pred",
    ):
        assert column in result.rows
    assert result.report["blind_used"] is False
    assert result.report["scope"] == "screening"
    assert {"fold", "target", "horizon"}.issubset(result.training_trace.columns)


def test_future_audit_passes_for_state_models() -> None:
    frame = _frame(180)
    parent = _parent(frame, folds=2)
    result = build_state_space_diversity(frame, parent, scope="screening")
    audit = future_perturbation_audit(
        frame,
        result.rows,
        phi_by_fold={
            fold: {
                key: 0.85
                for target in STATE_SPACE_TARGETS
                for key in (("local_trend", target), ("kalman", target))
            }
            for fold in result.rows["fold"].unique()
        },
        max_origins=2,
    )
    assert audit["passed"] is True


def test_fixed_gates_are_mechanical() -> None:
    report = {
        "improvement_pp": 0.021,
        "fold_wins": 3,
        "recent5_wins": 3,
        "worst_fold_regression_pp": 0.05,
        "target_metrics": {
            "generator_1": {"regression_pp": 0.01},
            "generator_all": {"regression_pp": -0.02},
        },
    }
    assert screening_gate(report)["passed"] is True
    assert full_development_gate(report, perturbation_passed=True)["passed"] is True
    assert full_development_gate(report, perturbation_passed=False)["passed"] is False


def test_report_is_json_serialisable() -> None:
    frame = _frame()
    result = build_state_space_diversity(frame, _parent(frame), scope="screening")
    json.dumps(result.report, ensure_ascii=False, default=str)
    assert candidate_report(result.rows, "parent_local_trend_blend_05_pred")["fold_count"] == 5
