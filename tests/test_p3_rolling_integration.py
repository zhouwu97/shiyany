from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gas_forecast.causal_rolling import CausalRollingConfig
from gas_forecast.direct_delta import DirectDeltaConfig
from gas_forecast.p3_rolling_integration import (
    FINAL_ROUTE,
    FORMAL_ROUTE_NAMES,
    HORIZONS_MINUTES,
    MaturedResidualOriginPredictor,
    P1_ROUTE,
    P3RouteOOF,
    StaticWideFusion,
    audit_future_perturbation_gate,
    build_p3_oof_integration,
    build_p3_route_oofs,
    freeze_p3_static_weights,
    runtime_budget_receipt,
    validate_future_perturbation_receipt,
    validate_p3_oof_keys,
    write_p3_integration_artifacts,
)
from gas_forecast.splits import TimeFold


def _frame(rows: int = 80) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    point = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "generator_1": 100.0 + point,
            "generator_all": 220.0 + point,
            "blast_furnace_1": 500_000.0 + point,
            "blast_furnace_gas_holder_2": 80_000.0 + point,
            "blast_furnace_user1": 90_000.0 + point,
            "air_heater_1": 20_000.0 + point,
            "into_gas_mixed_blast_furnace": 5_000.0 + point,
        },
        index=index,
    )


class _OriginOnlyPredictor:
    """测试替身：显式记录实际收到的历史截止点。"""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.history_endpoints: list[pd.Timestamp] = []

    def predict_at_origin(self, history_until_origin: pd.DataFrame) -> pd.DataFrame:
        assert history_until_origin.index.is_monotonic_increasing
        origin = pd.Timestamp(history_until_origin.index[-1])
        self.history_endpoints.append(origin)
        values: dict[str, float] = {}
        for minutes in HORIZONS_MINUTES:
            values[f"generator_1_t+{minutes}_pred"] = (
                float(history_until_origin["generator_1"].iloc[-1]) + self.offset
            )
            values[f"generator_all_t+{minutes}_pred"] = (
                float(history_until_origin["generator_all"].iloc[-1]) + self.offset
            )
        return pd.DataFrame([values], index=pd.DatetimeIndex([origin]))


class _BrokenPredictor:
    def predict_at_origin(self, history_until_origin: pd.DataFrame) -> pd.DataFrame:
        origin = pd.Timestamp(history_until_origin.index[-1])
        return pd.DataFrame({"bad": [np.nan]}, index=pd.DatetimeIndex([origin]))


def _oof_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-08")
    for fold_number in range(1, 6):
        train_end = start + pd.Timedelta(days=fold_number)
        for offset in range(2):
            origin = train_end + pd.Timedelta(minutes=135 + 15 * offset)
            for target, actual in (("generator_1", 100.0), ("generator_all", 220.0)):
                for horizon in HORIZONS_MINUTES:
                    records.append(
                        {
                            "fold": f"dev_{fold_number:02d}",
                            "origin_time": origin,
                            "train_end": train_end,
                            "target": target,
                            "horizon": horizon,
                            "actual": actual,
                            "prediction": actual + 4.0,
                        }
                    )
    return pd.DataFrame(records)


def _formal_future_gate() -> dict[str, object]:
    """构造已覆盖正式路线的最小、可验证零差异收据。"""

    groups = ("generator", "gas", "holder", "users", "all_features")
    operations = ("perturb", "delete")
    cases = [
        {
            "origin": "2025-01-01 00:00:00",
            "group": group,
            "operation": operation,
            "passed": True,
            "bitwise_identical": True,
            "max_abs_diff": 0.0,
            "reason": None,
        }
        for group in groups
        for operation in operations
    ]
    return {
        "gate": "p3_future_perturbation_v1",
        "passed": True,
        "groups": {group: [] for group in groups},
        "operations": list(operations),
        "candidates": {
            name: {"passed": True, "max_abs_diff": 0.0, "cases": cases}
            for name in FORMAL_ROUTE_NAMES
        },
    }


def _anchor_for_fold(frame: pd.DataFrame, fold: TimeFold) -> pd.DataFrame:
    """构造冻结 A61 OOF 形状，仅用于验证 P3 的键与 origin-only 编排。"""

    records: list[dict[str, object]] = []
    for origin in frame.index[
        (frame.index >= fold.validation_start) & (frame.index < fold.validation_end)
    ]:
        for target in ("generator_1", "generator_all"):
            for horizon in HORIZONS_MINUTES:
                target_time = origin + pd.Timedelta(minutes=horizon)
                records.append(
                    {
                        "fold": fold.name,
                        "origin_time": origin,
                        "train_end": fold.train_end,
                        "target": target,
                        "horizon": horizon,
                        "actual": float(frame.at[target_time, target]),
                        "prediction": float(frame.at[target_time, target] + 2.0),
                    }
                )
    return pd.DataFrame(records)


def test_unified_oof_key_rejects_blind_and_invalid_training_boundary() -> None:
    rows = _oof_rows()
    summary = validate_p3_oof_keys(rows, source="synthetic")
    assert summary["rows"] == len(rows)
    assert summary["blind_labels_used"] is False

    invalid = rows.copy()
    invalid.loc[0, "train_end"] = invalid.loc[0, "origin_time"]
    with pytest.raises(ValueError, match="train_end"):
        validate_p3_oof_keys(invalid, source="invalid")

    blind = rows.copy()
    blind.loc[0, "fold"] = "blind"
    with pytest.raises(ValueError, match="blind"):
        validate_p3_oof_keys(blind, source="blind")


def test_future_gate_checks_all_feature_groups_and_final_fusion_bitwise() -> None:
    frame = _frame()
    primary = _OriginOnlyPredictor()
    secondary = _OriginOnlyPredictor(offset=2.0)
    final = StaticWideFusion(
        {"primary": primary, "secondary": secondary},
        {"primary": 0.8, "secondary": 0.2},
    )
    origins = [frame.index[24], frame.index[48]]

    receipt = audit_future_perturbation_gate(
        frame,
        {P1_ROUTE: primary, FINAL_ROUTE: final},
        origins=origins,
    )

    assert receipt["passed"] is True
    assert receipt["max_abs_diff"] == 0.0
    assert set(receipt["groups"]) == {"generator", "gas", "holder", "users", "all_features"}
    for route in (P1_ROUTE, FINAL_ROUTE):
        route_receipt = receipt["candidates"][route]
        assert route_receipt["passed"] is True
        assert route_receipt["max_abs_diff"] == 0.0
        cases = route_receipt["cases"]
        assert len(cases) == len(origins) * 5 * 2
        assert {(case["group"], case["operation"]) for case in cases} == {
            (group, operation)
            for group in ("generator", "gas", "holder", "users", "all_features")
            for operation in ("perturb", "delete")
        }
        assert all(case["passed"] is True for case in cases)

    # 每一次调用都只收到当前 origin 的历史前缀，绝不包含其后的评分行。
    assert set(primary.history_endpoints).issubset(set(origins))
    assert set(secondary.history_endpoints).issubset(set(origins))


def test_unified_future_gate_requires_every_formal_route_and_zero_difference() -> None:
    frame = _frame()
    routes = {
        name: _OriginOnlyPredictor(offset=float(position))
        for position, name in enumerate(name for name in FORMAL_ROUTE_NAMES if name != FINAL_ROUTE)
    }
    final = StaticWideFusion(
        routes,
        {name: 1.0 / len(routes) for name in routes},
    )
    receipt = audit_future_perturbation_gate(
        frame,
        {**routes, FINAL_ROUTE: final},
        origins=[frame.index[36]],
    )

    summary = validate_future_perturbation_receipt(receipt)
    assert summary["passed"] is True
    assert set(summary["routes"]) == set(FORMAL_ROUTE_NAMES)
    assert receipt["runtime"]["passed"] is True


def test_future_gate_records_failure_closed_receipt_for_invalid_predictor_output() -> None:
    frame = _frame()
    receipt = audit_future_perturbation_gate(
        frame,
        {"broken": _BrokenPredictor()},
        origins=[frame.index[32]],
    )

    assert receipt["passed"] is False
    route = receipt["candidates"]["broken"]
    assert route["passed"] is False
    assert any(case["operation"] == "perturb" for case in route["cases"])
    assert any(case["max_abs_diff"] is None for case in route["cases"])


def test_matured_residual_wrapper_uses_only_exactly_matured_earlier_oof_errors() -> None:
    frame = _frame()
    origin = frame.index[48]
    rows: list[dict[str, object]] = []
    for target, actual, base in (
        ("generator_1", 105.0, 100.0),
        ("generator_all", 225.0, 220.0),
    ):
        for horizon in HORIZONS_MINUTES:
            earlier_origin = origin - pd.Timedelta(minutes=horizon)
            rows.append(
                {
                    "fold": "dev_01",
                    "origin_time": earlier_origin,
                    "train_end": earlier_origin - pd.Timedelta(minutes=15),
                    "target": target,
                    "horizon": horizon,
                    "actual": actual,
                    "prediction": base,
                }
            )
            rows.append(
                {
                    "fold": "dev_01",
                    "origin_time": origin + pd.Timedelta(minutes=15),
                    "train_end": origin,
                    "target": target,
                    "horizon": horizon,
                    "actual": -999_999.0,
                    "prediction": base,
                }
            )
    ledger = pd.DataFrame(rows)
    base = _OriginOnlyPredictor()
    baseline = MaturedResidualOriginPredictor(base, ledger).predict_at_origin(frame.loc[:origin])
    altered = ledger.copy()
    altered.loc[pd.to_datetime(altered["origin_time"]) > origin, "actual"] = 999_999.0
    observed = MaturedResidualOriginPredictor(_OriginOnlyPredictor(), altered).predict_at_origin(
        frame.loc[:origin]
    )

    np.testing.assert_array_equal(baseline.to_numpy(dtype=float), observed.to_numpy(dtype=float))


def test_p3_route_oofs_use_a61_fold_keys_and_origin_only_predictors() -> None:
    frame = _frame(rows=360)
    validation_start = frame.index[220]
    fold = TimeFold(
        name="dev_01",
        train_start=frame.index[0],
        train_end=validation_start - pd.Timedelta(minutes=135),
        validation_start=validation_start,
        validation_end=frame.index[223],
    )
    anchor = _anchor_for_fold(frame, fold)
    result = build_p3_route_oofs(
        frame,
        anchor,
        causal_config=CausalRollingConfig(min_train_rows=48, min_history_rows=17),
        direct_config=DirectDeltaConfig(
            min_train_rows=48,
            lgb_n_estimators=2,
            lgb_min_child_samples=8,
            inner_folds=2,
        ),
    )

    for name, rows, prediction_column in (
        ("p1", result.p1_causal_rolling, "prediction"),
        ("p2_matured", result.p2_matured_residual, "prediction"),
        ("p2_analog", result.p2_historical_analog, "prediction"),
        ("a64", result.a64_direct_delta, "ridge_prediction"),
    ):
        summary = validate_p3_oof_keys(rows, source=name, prediction_column=prediction_column)
        assert summary["rows"] == len(anchor)
    assert result.report["blind_labels_used"] is False
    assert result.report["origin_only_prediction"] is True
    assert result.report["p1"]["forward_refit"] is False
    assert result.report["p2_analog"]["origin_only_prediction"] is True
    assert result.report["a64"]["origin_only_prediction"] is True


def test_a61_anchor_is_never_replaced_until_oof_gate_and_all_receipts_pass() -> None:
    anchor = _oof_rows()
    improved = anchor.copy()
    improved["prediction"] = improved["actual"]
    direct = improved.rename(columns={"prediction": "ridge_prediction"})
    future_gate = _formal_future_gate()
    runtime = runtime_budget_receipt(elapsed_seconds=3.0, maximum_origin_seconds=0.2)
    route_oofs = P3RouteOOF(
        anchor=anchor,
        p1_causal_rolling=improved,
        p2_matured_residual=improved,
        p2_historical_analog=improved,
        a64_direct_delta=direct,
        report={"blind_labels_used": False},
    )

    result = build_p3_oof_integration(
        route_oofs,
        future_gate=future_gate,
        runtime=runtime,
        existing_promotion_passed=True,
    )

    assert result.report["candidate_eligible"] is True
    assert result.report["champion"]["name"] == "a61_parent"
    assert result.report["champion"]["replaced"] is False
    assert result.ensemble.report["static_gate"]["passed"] is True
    frozen = freeze_p3_static_weights(result)
    assert frozen.report["selection_source"] == "development_oof_only"
    assert frozen.report["blind_labels_used"] is False
    assert sum(frozen.weights.values()) == pytest.approx(1.0)

    stopped = build_p3_oof_integration(
        route_oofs,
        future_gate=None,
        runtime=runtime,
        existing_promotion_passed=True,
    )
    assert stopped.report["candidate_eligible"] is False
    assert stopped.report["status"] == "STOP_FAIL_CLOSED"
    with pytest.raises(ValueError, match="完整门禁"):
        freeze_p3_static_weights(stopped)


def test_p3_artifact_writer_persists_fail_closed_receipt(tmp_path: Path) -> None:
    anchor = _oof_rows()
    direct = anchor.rename(columns={"prediction": "ridge_prediction"})
    route_oofs = P3RouteOOF(
        anchor=anchor,
        p1_causal_rolling=anchor,
        p2_matured_residual=anchor,
        p2_historical_analog=anchor,
        a64_direct_delta=direct,
        report={"blind_labels_used": False},
    )
    result = build_p3_oof_integration(
        route_oofs,
        future_gate=None,
        runtime=runtime_budget_receipt(elapsed_seconds=1.0, maximum_origin_seconds=0.1),
        existing_promotion_passed=False,
    )

    output = write_p3_integration_artifacts(result, tmp_path / "p3_stop")
    receipt = json.loads((output / "p3_integration_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "STOP_FAIL_CLOSED"
    assert (output / "p3_oof.csv").is_file()
    assert (output / "route_receipts.csv").is_file()
