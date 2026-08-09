"""P3 滚动集成：严格 OOF、逐 origin 推理与 fail-closed 因果门禁。

该模块把冻结 A61 anchor、P1 CausalRolling、P2 成熟残差/历史类比和
A64 Direct Delta 放在同一条 development-only 审计链中。它不读取 blind、
final 或平台参考值，也不会更新 ``results/best``。正式预测器的唯一输入
形式是 ``predict_at_origin(history_until_origin)``。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from gas_forecast.causal_rolling import CausalRollingConfig, build_causal_rolling_oof
from gas_forecast.causal_trajectory_ensemble import (
    EnsembleResult,
    PARENT_ROUTE,
    RouteReceipt,
    build_causal_trajectory_ensemble,
    canonicalize_oof,
    pre_registered_weight_candidates,
    validate_oof_contract,
)
from gas_forecast.config import ForecastConfig
from gas_forecast.direct_delta import (
    DirectDeltaConfig,
    build_direct_delta_oof,
)
from gas_forecast.historical_analog import (
    StrictHistoricalAnalogForecaster,
    build_historical_analog_oof,
)
from gas_forecast.matured_residual import (
    MaturedResidualConfig,
    build_matured_residual_oof,
    predict_matured_residual_at_origin,
)
from gas_forecast.scoring import competition_mape
from gas_forecast.splits import TimeFold


P1_ROUTE = "p1_causal_rolling"
P2_MATURED_ROUTE = "p2_matured_residual"
P2_ANALOG_ROUTE = "p2_historical_analog"
A64_ROUTE = "a64_direct_delta"
FINAL_ROUTE = "p3_final_fusion"
FORMAL_ROUTE_NAMES: tuple[str, ...] = (
    PARENT_ROUTE,
    P1_ROUTE,
    P2_MATURED_ROUTE,
    P2_ANALOG_ROUTE,
    A64_ROUTE,
    FINAL_ROUTE,
)
HORIZONS_MINUTES: tuple[int, ...] = tuple(range(15, 121, 15))


class OriginPredictor(Protocol):
    """P3 允许的唯一线上预测协议。"""

    def predict_at_origin(self, history_until_origin: pd.DataFrame) -> pd.DataFrame:
        """使用当前 origin 及其历史返回一个宽表预测。"""


@dataclass(frozen=True)
class RuntimeBudget:
    """P3 运行预算；超时只会阻止候选资格，不会降级为无收据发布。"""

    max_total_seconds: float = 7_200.0
    max_origin_prediction_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_total_seconds <= 0.0 or self.max_origin_prediction_seconds <= 0.0:
            raise ValueError("P3 运行预算必须为正数")


@dataclass(frozen=True)
class P3RouteOOF:
    """P3 各路线 development OOF 与生成收据。"""

    anchor: pd.DataFrame
    p1_causal_rolling: pd.DataFrame
    p2_matured_residual: pd.DataFrame
    p2_historical_analog: pd.DataFrame
    a64_direct_delta: pd.DataFrame
    report: dict[str, object]


@dataclass(frozen=True)
class P3IntegrationResult:
    """静态 cross-fit 融合、P3 门禁及不可替换 A61 的结论。"""

    ensemble: EnsembleResult
    report: dict[str, object]


@dataclass(frozen=True)
class FrozenP3Fusion:
    """仅由 development OOF 选出的最终静态权重与可审计收据。"""

    weights: dict[str, float]
    report: dict[str, object]


def _require_history_until_origin(history_until_origin: pd.DataFrame) -> pd.Timestamp:
    """验证调用者只交付单一 origin 的历史前缀。"""

    if not isinstance(history_until_origin, pd.DataFrame):
        raise TypeError("history_until_origin 必须是 pandas.DataFrame")
    if not isinstance(history_until_origin.index, pd.DatetimeIndex):
        raise TypeError("history_until_origin 必须使用 DatetimeIndex")
    if (
        history_until_origin.empty
        or history_until_origin.index.has_duplicates
        or not history_until_origin.index.is_monotonic_increasing
    ):
        raise ValueError("history_until_origin 必须非空、唯一且严格递增")
    return pd.Timestamp(history_until_origin.index[-1])


def _wide_from_long_prediction(long_rows: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    """将严格 2×8 长表转成统一的单 origin 宽表。"""

    required = {"origin_time", "target", "horizon", value_column}
    missing = sorted(required.difference(long_rows.columns))
    if missing:
        raise ValueError(f"P3 长表预测缺少字段: {missing}")
    if long_rows.duplicated(["origin_time", "target", "horizon"]).any():
        raise ValueError("P3 长表预测存在重复 target×horizon")
    origins = pd.to_datetime(long_rows["origin_time"], errors="coerce")
    if origins.isna().any() or origins.nunique() != 1:
        raise ValueError("P3 长表预测必须只对应一个 origin")
    values: dict[str, float] = {}
    for row in long_rows.itertuples(index=False):
        target = str(getattr(row, "target"))
        horizon = int(getattr(row, "horizon"))
        value = float(getattr(row, value_column))
        if not np.isfinite(value):
            raise ValueError("P3 长表预测包含 NaN/Inf")
        values[f"{target}_t+{horizon}_pred"] = value
    expected = [
        f"{target}_t+{horizon}_pred"
        for target in ("generator_1", "generator_all")
        for horizon in HORIZONS_MINUTES
    ]
    if sorted(values) != sorted(expected):
        raise ValueError("P3 长表预测没有完整覆盖两个目标与八个步长")
    origin = pd.Timestamp(origins.iloc[0])
    return pd.DataFrame([values], index=pd.DatetimeIndex([origin])).reindex(columns=expected)


def _project_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    """对融合宽表施加与 A61 一致的确定性容量约束。"""

    result = frame.copy(deep=True)
    for minutes in HORIZONS_MINUTES:
        generator_1 = f"generator_1_t+{minutes}_pred"
        generator_all = f"generator_all_t+{minutes}_pred"
        if generator_1 not in result or generator_all not in result:
            continue
        result[generator_1] = result[generator_1].clip(0.0, 200.0)
        result[generator_all] = result[generator_all].clip(0.0, 440.0)
        result[generator_all] = np.maximum(result[generator_all], result[generator_1])
        result[generator_all] = np.minimum(result[generator_all], result[generator_1] + 240.0)
    return result


class HistoricalAnalogWideOriginPredictor:
    """将 P2 严格历史类比的单 origin 长表适配为融合宽表。"""

    def __init__(self, model: StrictHistoricalAnalogForecaster) -> None:
        self.model = model

    def predict_at_origin(self, history_until_origin: pd.DataFrame) -> pd.DataFrame:
        origin = _require_history_until_origin(history_until_origin)
        prediction = _wide_from_long_prediction(
            self.model.predict_at_origin(history_until_origin),
            value_column="prediction",
        )
        if prediction.index[0] != origin:
            raise RuntimeError("历史类比预测返回的 origin 与输入不一致")
        return _project_capacity(prediction)


class MaturedResidualOriginPredictor:
    """把历史 OOF 误差状态施加到当前基础预测，绝不读取当前 future label。"""

    def __init__(
        self,
        base_predictor: OriginPredictor,
        oof_ledger: pd.DataFrame,
        *,
        config: MaturedResidualConfig | None = None,
        base_prediction_column: str = "prediction",
    ) -> None:
        if not callable(getattr(base_predictor, "predict_at_origin", None)):
            raise TypeError("成熟残差基础预测器必须提供 predict_at_origin")
        required = {"origin_time", "train_end", "target", "horizon", "actual", base_prediction_column}
        missing = sorted(required.difference(oof_ledger.columns))
        if missing:
            raise ValueError(f"成熟残差 OOF 台账缺少字段: {missing}")
        self.base_predictor = base_predictor
        self.oof_ledger = oof_ledger.copy(deep=True)
        self.config = config or MaturedResidualConfig()
        self.base_prediction_column = base_prediction_column
        self.last_prediction_metadata_: dict[str, object] = {}

    def predict_at_origin(self, history_until_origin: pd.DataFrame) -> pd.DataFrame:
        origin = _require_history_until_origin(history_until_origin)
        base = self.base_predictor.predict_at_origin(history_until_origin)
        if len(base) != 1 or base.index[0] != origin:
            raise ValueError("成熟残差基础预测器必须返回当前 origin 的一行宽表")
        current_records: list[dict[str, object]] = []
        for target in ("generator_1", "generator_all"):
            for horizon in HORIZONS_MINUTES:
                column = f"{target}_t+{horizon}_pred"
                current_records.append(
                    {
                        "fold": "production_origin",
                        "origin_time": origin,
                        "train_end": origin - pd.Timedelta(minutes=15),
                        "target": target,
                        "horizon": horizon,
                        "actual": np.nan,
                        self.base_prediction_column: float(base.at[origin, column]),
                    }
                )
        visible = self.oof_ledger.loc[
            pd.to_datetime(self.oof_ledger["origin_time"], errors="raise").lt(origin)
        ].copy()
        ledger = pd.concat([visible, pd.DataFrame(current_records)], ignore_index=True)
        corrected = predict_matured_residual_at_origin(
            ledger,
            origin,
            config=self.config,
            prediction_column=self.base_prediction_column,
            output_column="matured_residual_pred",
            require_cross_fit=True,
        )
        result = _project_capacity(
            _wide_from_long_prediction(corrected, value_column="matured_residual_pred")
        )
        self.last_prediction_metadata_ = {
            "origin": origin,
            "base_prediction_source": type(self.base_predictor).__name__,
            "visible_oof_origins_strictly_before_origin": True,
            "maturity_rule": "earlier_origin and target_datetime == current_origin",
            "used_future_observations": False,
        }
        return result


class StaticWideFusion:
    """冻结的非负静态融合；每条输入路线仍只收到同一个历史前缀。"""

    def __init__(self, predictors: Mapping[str, OriginPredictor], weights: Mapping[str, float]) -> None:
        if not predictors or set(weights) != set(predictors):
            raise ValueError("融合预测器与权重必须一一对应")
        numeric_weights = {name: float(value) for name, value in weights.items()}
        if any(value < 0.0 for value in numeric_weights.values()) or not np.isclose(
            sum(numeric_weights.values()), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("融合权重必须是非负且和为 1")
        if any(not callable(getattr(item, "predict_at_origin", None)) for item in predictors.values()):
            raise TypeError("所有融合路线必须提供 predict_at_origin")
        self.predictors = dict(predictors)
        self.weights = numeric_weights

    def predict_at_origin(self, history_until_origin: pd.DataFrame) -> pd.DataFrame:
        origin = _require_history_until_origin(history_until_origin)
        outputs = {
            name: predictor.predict_at_origin(history_until_origin)
            for name, predictor in self.predictors.items()
        }
        first = next(iter(outputs.values()))
        if len(first) != 1 or first.index[0] != origin:
            raise ValueError("融合路线必须返回当前 origin 的单行宽表")
        prediction = np.zeros(first.shape, dtype=float)
        for name, output in outputs.items():
            if list(output.columns) != list(first.columns) or not output.index.equals(first.index):
                raise ValueError(f"融合路线 {name} 的预测 schema 不一致")
            values = output.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"融合路线 {name} 的预测含 NaN/Inf")
            prediction += self.weights[name] * values
        return _project_capacity(pd.DataFrame(prediction, index=first.index, columns=first.columns))


def future_feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    """按 P3 固定口径划分未来 generator/gas/holder/users/all-features。"""

    numeric = [str(column) for column in frame.select_dtypes(include=[np.number]).columns]
    groups: dict[str, list[str]] = {
        "generator": [],
        "gas": [],
        "holder": [],
        "users": [],
        "all_features": numeric,
    }
    for column in numeric:
        name = column.casefold()
        if name.startswith("generator"):
            groups["generator"].append(column)
        if any(token in name for token in ("gas", "blast_furnace", "coke", "converter")):
            groups["gas"].append(column)
        if "holder" in name:
            groups["holder"].append(column)
        if any(token in name for token in ("user", "air_heater", "demand", "mixed")):
            groups["users"].append(column)
    return groups


def _prediction_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[bool, float | None, str | None]:
    """比较两个预测宽表；schema 变化也视为 fail-closed。"""

    if list(baseline.columns) != list(candidate.columns) or not baseline.index.equals(candidate.index):
        return False, None, "预测 schema 或 origin 发生变化"
    try:
        left = baseline.to_numpy(dtype=float)
        right = candidate.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False, None, "预测输出不是纯数值宽表"
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return False, None, "预测输出包含 NaN/Inf"
    difference = np.abs(left - right)
    maximum = float(np.max(difference)) if difference.size else 0.0
    return bool(np.array_equal(left, right)), maximum, None


def _perturb_future(
    frame: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    columns: Sequence[str],
    operation: str,
) -> pd.DataFrame:
    """生成一个仅在 origin 后改变的输入副本。"""

    future = frame.index > origin
    if operation == "delete" and len(columns) == len(frame.select_dtypes(include=[np.number]).columns):
        return frame.loc[~future].copy(deep=True)
    variant = frame.copy(deep=True)
    if operation == "perturb":
        for position, column in enumerate(columns):
            variant.loc[future, column] = float(-9_999_991 - position)
    elif operation == "delete":
        variant.loc[future, list(columns)] = np.nan
    else:
        raise ValueError(f"不支持的未来扰动操作: {operation}")
    return variant


def audit_future_perturbation_gate(
    frame: pd.DataFrame,
    predictors: Mapping[str, OriginPredictor],
    *,
    origins: Sequence[pd.Timestamp | str] | None = None,
) -> dict[str, object]:
    """统一审计每条正式路线和最终融合对五组未来输入的 bitwise 不变性。

    对每一个 group 都执行 ``perturb`` 与 ``delete``。调用预测器时始终传入
    ``variant.loc[:origin]``；若预测器没有 ``predict_at_origin`` 或抛出异常，
    对应路线立即 fail-closed。
    """

    _require_history_until_origin(frame)
    started = time.perf_counter()
    maximum_prediction_seconds = 0.0
    if not predictors:
        raise ValueError("未来扰动门禁至少需要一个正式预测器")
    if any(not callable(getattr(item, "predict_at_origin", None)) for item in predictors.values()):
        raise TypeError("P3 未来门禁拒绝没有 predict_at_origin 的路线")
    if origins is None:
        positions = sorted(set(np.linspace(0, len(frame) - 2, num=min(3, len(frame) - 1), dtype=int)))
        selected = [pd.Timestamp(frame.index[position]) for position in positions]
    else:
        selected = [pd.Timestamp(value) for value in origins]
    if not selected or any(value not in frame.index or value >= frame.index[-1] for value in selected):
        raise ValueError("未来扰动门禁 origin 必须位于时间轴且其后仍有观测")

    groups = future_feature_groups(frame)
    candidate_receipts: dict[str, object] = {}
    all_passed = True
    total_maximum = 0.0
    for name, predictor in predictors.items():
        cases: list[dict[str, object]] = []
        route_passed = True
        route_maximum = 0.0
        for origin in selected:
            try:
                call_started = time.perf_counter()
                baseline = predictor.predict_at_origin(frame.loc[:origin].copy(deep=True))
                maximum_prediction_seconds = max(
                    maximum_prediction_seconds,
                    time.perf_counter() - call_started,
                )
            except Exception as error:  # 门禁应落盘失败原因而不是吞掉异常。
                cases.append(
                    {
                        "origin": str(origin),
                        "group": "baseline",
                        "operation": "predict",
                        "passed": False,
                        "bitwise_identical": False,
                        "max_abs_diff": None,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                route_passed = False
                continue
            for group, columns in groups.items():
                for operation in ("perturb", "delete"):
                    if not columns:
                        cases.append(
                            {
                                "origin": str(origin),
                                "group": group,
                                "operation": operation,
                                "passed": True,
                                "skipped": True,
                                "max_abs_diff": 0.0,
                                "reason": "输入不存在该类别的数值字段",
                            }
                        )
                        continue
                    try:
                        variant = _perturb_future(
                            frame,
                            origin=origin,
                            columns=columns,
                            operation=operation,
                        )
                        call_started = time.perf_counter()
                        observed = predictor.predict_at_origin(variant.loc[:origin].copy(deep=True))
                        maximum_prediction_seconds = max(
                            maximum_prediction_seconds,
                            time.perf_counter() - call_started,
                        )
                        identical, maximum, reason = _prediction_comparison(baseline, observed)
                    except Exception as error:  # 无法复算属于门禁失败。
                        identical, maximum, reason = (
                            False,
                            None,
                            f"{type(error).__name__}: {error}",
                        )
                    route_passed = route_passed and identical
                    if maximum is not None:
                        route_maximum = max(route_maximum, maximum)
                    cases.append(
                        {
                            "origin": str(origin),
                            "group": group,
                            "operation": operation,
                            "passed": identical,
                            "bitwise_identical": identical,
                            "max_abs_diff": maximum,
                            "reason": reason,
                        }
                    )
        all_passed = all_passed and route_passed
        total_maximum = max(total_maximum, route_maximum)
        candidate_receipts[name] = {
            "passed": route_passed,
            "max_abs_diff": route_maximum,
            "cases": cases,
        }
    return {
        "gate": "p3_future_perturbation_v1",
        "passed": all_passed,
        "origins": [str(value) for value in selected],
        "groups": {name: list(columns) for name, columns in groups.items()},
        "operations": ["perturb", "delete"],
        "max_abs_diff": total_maximum,
        "candidates": candidate_receipts,
        "runtime": runtime_budget_receipt(
            elapsed_seconds=time.perf_counter() - started,
            maximum_origin_seconds=maximum_prediction_seconds,
        ),
    }


def runtime_budget_receipt(
    *,
    elapsed_seconds: float,
    maximum_origin_seconds: float,
    budget: RuntimeBudget = RuntimeBudget(),
) -> dict[str, object]:
    """构造固定、可序列化的运行预算门禁收据。"""

    total = float(elapsed_seconds)
    maximum = float(maximum_origin_seconds)
    if not np.isfinite(total) or not np.isfinite(maximum) or total < 0.0 or maximum < 0.0:
        raise ValueError("运行耗时必须是非负有限数")
    checks = {
        "total_seconds": total <= budget.max_total_seconds,
        "origin_prediction_seconds": maximum <= budget.max_origin_prediction_seconds,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "elapsed_seconds": total,
        "maximum_origin_prediction_seconds": maximum,
        "budget": {
            "max_total_seconds": budget.max_total_seconds,
            "max_origin_prediction_seconds": budget.max_origin_prediction_seconds,
        },
    }


def validate_future_perturbation_receipt(
    receipt: Mapping[str, object],
    *,
    required_routes: Sequence[str] = FORMAL_ROUTE_NAMES,
) -> dict[str, object]:
    """严格验证统一未来门禁收据，缺项、失败或非零差异一律拒绝。

    该函数用于把测试收据与正式候选资格分开：单条路线的单元测试可以只
    审计自己，而 P3 正式融合必须同时提供 A61、四条候选和最终融合的完整
    收据。
    """

    if not isinstance(receipt, Mapping):
        raise TypeError("future perturbation 收据必须是映射")
    if receipt.get("gate") != "p3_future_perturbation_v1":
        raise ValueError("future perturbation 收据版本不匹配")
    if receipt.get("passed") is not True:
        raise ValueError("统一 future perturbation 门禁未通过")
    candidates = receipt.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("future perturbation 收据缺少 candidates")
    groups = receipt.get("groups")
    operations = receipt.get("operations")
    expected_groups = {"generator", "gas", "holder", "users", "all_features"}
    expected_operations = {"perturb", "delete"}
    if not isinstance(groups, Mapping) or set(groups) != expected_groups:
        raise ValueError("future perturbation 收据未覆盖五类未来字段")
    if not isinstance(operations, list) or set(operations) != expected_operations:
        raise ValueError("future perturbation 收据未覆盖 perturb/delete 两种操作")
    missing = sorted(set(required_routes).difference(candidates))
    if missing:
        raise ValueError(f"future perturbation 收据缺少正式路线: {missing}")
    checks: dict[str, object] = {}
    for route in required_routes:
        item = candidates.get(route)
        if not isinstance(item, Mapping) or item.get("passed") is not True:
            raise ValueError(f"路线 {route} 未通过 future perturbation 门禁")
        maximum = item.get("max_abs_diff")
        if not isinstance(maximum, (int, float)) or float(maximum) != 0.0:
            raise ValueError(f"路线 {route} 的 future perturbation 最大差异不是 0")
        cases = item.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"路线 {route} 缺少 future perturbation 明细")
        seen_pairs = {
            (str(case.get("group")), str(case.get("operation")))
            for case in cases
            if isinstance(case, Mapping)
        }
        expected_pairs = {
            (group, operation)
            for group in expected_groups
            for operation in expected_operations
        }
        if not expected_pairs.issubset(seen_pairs):
            raise ValueError(f"路线 {route} 未完整覆盖五组 × 两种 future 扰动")
        failures = [
            case
            for case in cases
            if not isinstance(case, Mapping)
            or case.get("passed") is not True
            or (
                case.get("skipped") is not True
                and (
                    case.get("bitwise_identical") is not True
                    or case.get("max_abs_diff") != 0.0
                )
            )
        ]
        if failures:
            raise ValueError(f"路线 {route} 的 future perturbation 明细不满足 bitwise=0")
        checks[route] = {"passed": True, "cases": len(cases), "max_abs_diff": 0.0}
    return {
        "passed": True,
        "routes": checks,
        "runtime": dict(receipt["runtime"])
        if isinstance(receipt.get("runtime"), Mapping)
        else None,
    }


def _canonical_route(rows: pd.DataFrame, *, prediction_column: str) -> pd.DataFrame:
    """将已有 OOF 的候选预测列统一为融合所需的 ``prediction``。"""

    if prediction_column not in rows:
        raise ValueError(f"P3 路线缺少预测列: {prediction_column}")
    output = rows.copy()
    output["prediction"] = pd.to_numeric(output[prediction_column], errors="coerce")
    return output


def validate_p3_oof_keys(
    rows: pd.DataFrame,
    *,
    source: str,
    prediction_column: str = "prediction",
) -> dict[str, object]:
    """验证 P3 统一 ``fold/origin/train_end/target/horizon/actual`` OOF 键。

    该公开入口不做 inner join 或补行；字段、blind、训练边界和两个目标八步
    覆盖任一不满足都会立即拒绝，从而避免多路线融合时静默错配。
    """

    normalized = _canonical_route(rows, prediction_column=prediction_column)
    summary = validate_oof_contract(normalized, source=source)
    canonical = canonicalize_oof(normalized, source=source)
    by_origin = canonical.groupby(["fold", "origin_time", "train_end"], sort=True)
    expected_pairs = {
        (target, horizon)
        for target in ("generator_1", "generator_all")
        for horizon in HORIZONS_MINUTES
    }
    invalid_origins: list[str] = []
    for key, part in by_origin:
        pairs = set(zip(part["target"], part["horizon"], strict=True))
        if len(part) != len(expected_pairs) or pairs != expected_pairs:
            invalid_origins.append(str(key))
    if invalid_origins:
        raise ValueError(f"{source} OOF 存在不完整 target×horizon origin 矩阵: {invalid_origins[:3]}")
    fold_train_end_counts = canonical.groupby("fold", sort=True)["train_end"].nunique()
    if fold_train_end_counts.gt(1).any():
        invalid_folds = fold_train_end_counts.loc[fold_train_end_counts.gt(1)].index.tolist()
        raise ValueError(f"{source} OOF 每个 fold 必须只有一个 train_end: {invalid_folds}")
    return {
        **summary,
        "complete_origin_matrix": True,
        "origin_count": int(by_origin.ngroups),
    }


def derive_anchor_folds(anchor_rows: pd.DataFrame, frame_index: pd.DatetimeIndex) -> list[TimeFold]:
    """从冻结 A61 development OOF 恢复完全相同的外层折边界。"""

    validate_p3_oof_keys(anchor_rows, source=PARENT_ROUTE)
    if not isinstance(frame_index, pd.DatetimeIndex) or frame_index.empty:
        raise ValueError("P3 原始生产时间轴必须是非空 DatetimeIndex")
    work = anchor_rows.copy()
    work["origin_time"] = pd.to_datetime(work["origin_time"], errors="raise")
    work["train_end"] = pd.to_datetime(work["train_end"], errors="raise")
    folds: list[TimeFold] = []
    for name, part in work.groupby("fold", sort=True):
        train_ends = part["train_end"].drop_duplicates()
        if len(train_ends) != 1:
            raise ValueError(f"A61 fold {name} 存在多个 train_end")
        origins = pd.DatetimeIndex(sorted(part["origin_time"].unique()))
        if len(origins) == 0:
            raise ValueError(f"A61 fold {name} 不含 origin")
        gaps = origins.to_series().diff().dropna()
        if not gaps.empty and not gaps.eq(pd.Timedelta(minutes=15)).all():
            raise ValueError(f"A61 fold {name} 的 origin 不是连续 15 分钟网格")
        folds.append(
            TimeFold(
                name=str(name),
                train_start=pd.Timestamp(frame_index.min()),
                train_end=pd.Timestamp(train_ends.iloc[0]),
                validation_start=pd.Timestamp(origins.min()),
                validation_end=pd.Timestamp(origins.max()) + pd.Timedelta(minutes=15),
            )
        )
    return folds


def build_p3_route_oofs(
    frame: pd.DataFrame,
    anchor_rows: pd.DataFrame,
    *,
    anchor_prediction_column: str = "prediction",
    causal_config: CausalRollingConfig | None = None,
    analog_config: ForecastConfig | None = None,
    direct_config: DirectDeltaConfig | None = None,
    matured_config: MaturedResidualConfig | None = None,
) -> P3RouteOOF:
    """以 A61 的固定 development 折逐 origin 构建 P1/P2/A64 OOF。

    基础路线的 held origin 预测分别调用 P1 的 ``predict_at_origin``、P2
    StrictHistoricalAnalog 的 ``predict_at_origin`` 和 A64 的
    ``predict_at_origin``；不会把整段 validation/scoring frame 交给模型。
    """

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("P3 原始 frame 必须使用 DatetimeIndex")
    anchor = _canonical_route(anchor_rows, prediction_column=anchor_prediction_column)
    folds = derive_anchor_folds(anchor, frame.index)
    started = time.perf_counter()
    p1_rows, p1_report = build_causal_rolling_oof(
        frame,
        config=causal_config,
        folds=folds,
        include_blind=False,
        forward_refit=False,
    )
    analog = build_historical_analog_oof(
        frame,
        config=analog_config,
        scope="development",
        folds=folds,
        origin_only=True,
    )
    # ``origin_only=True`` 不消费第二个参数的数值；传入空 schema 可防止误用
    # 预先构造的整段评分特征矩阵。
    empty_features = pd.DataFrame(index=frame.index)
    direct_rows, direct_report = build_direct_delta_oof(
        frame,
        empty_features,
        config=direct_config,
        folds=folds,
        include_blind=False,
        nested=False,
        origin_only=True,
    )
    matured = build_matured_residual_oof(
        anchor,
        config=matured_config,
        prediction_column="prediction",
        output_column="prediction",
    )
    elapsed = time.perf_counter() - started
    return P3RouteOOF(
        anchor=anchor,
        p1_causal_rolling=p1_rows,
        p2_matured_residual=matured.rows,
        p2_historical_analog=analog.rows,
        a64_direct_delta=direct_rows,
        report={
            "blind_labels_used": False,
            "origin_only_prediction": True,
            "elapsed_seconds": elapsed,
            "p1": p1_report,
            "p2_matured": matured.report,
            "p2_analog": analog.report,
            "a64": direct_report,
        },
    )


def _future_route_passed(future_gate: Mapping[str, object] | None, route: str) -> bool | None:
    if not isinstance(future_gate, Mapping):
        return None
    candidates = future_gate.get("candidates")
    if not isinstance(candidates, Mapping):
        return None
    receipt = candidates.get(route)
    return receipt.get("passed") if isinstance(receipt, Mapping) and isinstance(receipt.get("passed"), bool) else None


def _anchor_key_check(anchor: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, int]:
    """比较两条已校验 OOF 的完整键，不允许通过 inner join 静默丢行。"""

    parent = canonicalize_oof(anchor, source=PARENT_ROUTE)
    route = canonicalize_oof(candidate, source="p3_candidate")
    key_columns = ["fold", "origin_time", "train_end", "target", "horizon", "actual"]
    compared = parent.loc[:, key_columns].merge(
        route.loc[:, key_columns],
        on=key_columns,
        how="outer",
        indicator=True,
    )
    counts = compared["_merge"].value_counts().to_dict()
    return {
        "parent_only": int(counts.get("left_only", 0)),
        "route_only": int(counts.get("right_only", 0)),
        "shared": int(counts.get("both", 0)),
    }


def _route_receipt(
    name: str,
    rows: pd.DataFrame,
    *,
    anchor: pd.DataFrame,
    prediction_column: str,
    future_gate: Mapping[str, object] | None,
) -> tuple[pd.DataFrame | None, RouteReceipt]:
    """以完整 OOF 契约和可选因果门禁决定路线是否进入融合。"""

    future_passed = _future_route_passed(future_gate, name)
    try:
        canonical = _canonical_route(rows, prediction_column=prediction_column)
        contract = validate_p3_oof_keys(canonical, source=name)
    except (TypeError, ValueError) as error:
        return None, RouteReceipt(
            name=name,
            source="in_memory",
            status="OOF_REJECTED",
            accepted=False,
            reason=f"严格 OOF 契约失败: {type(error).__name__}: {error}",
            rows=int(len(rows)),
            blind_labels_used=False,
            future_perturbation_passed=future_passed,
        )
    key_check = _anchor_key_check(anchor, canonical)
    if key_check["parent_only"] or key_check["route_only"]:
        return None, RouteReceipt(
            name=name,
            source="in_memory",
            status="OOF_KEY_MISMATCH",
            accepted=False,
            reason=f"与冻结 A61 OOF 键不一致: {key_check}",
            rows=int(contract["rows"]),
            blind_labels_used=False,
            future_perturbation_passed=future_passed,
        )
    if future_passed is not True:
        return None, RouteReceipt(
            name=name,
            source="in_memory",
            status="CAUSAL_GATE_REJECTED",
            accepted=False,
            reason="缺少或未通过该路线的统一 future perturbation 收据",
            rows=int(contract["rows"]),
            blind_labels_used=False,
            future_perturbation_passed=future_passed,
        )
    return canonical, RouteReceipt(
        name=name,
        source="in_memory",
        status="DEVELOPMENT_OOF_ACCEPTED",
        accepted=True,
        reason="严格 development OOF 与统一未来门禁均通过",
        rows=int(contract["rows"]),
        blind_labels_used=False,
        future_perturbation_passed=True,
    )


def build_p3_oof_integration(
    route_oofs: P3RouteOOF,
    *,
    future_gate: Mapping[str, object] | None,
    runtime: Mapping[str, object] | None,
    existing_promotion_passed: bool = False,
) -> P3IntegrationResult:
    """将四条路线与 A61 anchor 做 OOF-only cross-fit，并默认不替换 champion。"""

    anchor = _canonical_route(route_oofs.anchor, prediction_column="prediction")
    validate_p3_oof_keys(anchor, source=PARENT_ROUTE)
    anchor_future = _future_route_passed(future_gate, PARENT_ROUTE)
    anchor_receipt = RouteReceipt(
        name=PARENT_ROUTE,
        source="frozen_a61_anchor",
        status="FROZEN_A61_ANCHOR",
        accepted=anchor_future is True,
        reason=(
            "A61 统一未来门禁通过" if anchor_future is True else "A61 缺少或未通过统一未来门禁"
        ),
        rows=int(len(anchor)),
        blind_labels_used=False,
        future_perturbation_passed=anchor_future,
    )
    routes = {
        P1_ROUTE: (route_oofs.p1_causal_rolling, "prediction"),
        P2_MATURED_ROUTE: (route_oofs.p2_matured_residual, "prediction"),
        P2_ANALOG_ROUTE: (route_oofs.p2_historical_analog, "prediction"),
        A64_ROUTE: (route_oofs.a64_direct_delta, "ridge_prediction"),
    }
    accepted: dict[str, pd.DataFrame] = {}
    receipts: list[RouteReceipt] = [anchor_receipt]
    for name, (rows, column) in routes.items():
        candidate, receipt = _route_receipt(
            name,
            rows,
            anchor=anchor,
            prediction_column=column,
            future_gate=future_gate,
        )
        receipts.append(receipt)
        if candidate is not None:
            accepted[name] = candidate

    # anchor 未通过统一门禁时仍输出 OOF 诊断，但 final candidate 一律 fail-closed。
    ensemble = build_causal_trajectory_ensemble(
        anchor,
        accepted,
        route_receipts=receipts,
    )
    strict_oof = bool(
        not ensemble.report.get("blind_labels_used", True)
        and all(not receipt.blind_labels_used for receipt in receipts)
    )
    try:
        future_summary = validate_future_perturbation_receipt(
            future_gate if isinstance(future_gate, Mapping) else {},
        )
        causal_passed = True
    except (TypeError, ValueError) as error:
        future_summary = {"passed": False, "reason": f"{type(error).__name__}: {error}"}
        causal_passed = False
    runtime_payload = runtime
    if runtime_payload is None and isinstance(future_summary.get("runtime"), Mapping):
        runtime_payload = future_summary["runtime"]
    runtime_passed = bool(
        isinstance(runtime_payload, Mapping) and runtime_payload.get("passed") is True
    )
    static_passed = bool(ensemble.report["static_gate"]["passed"])
    all_routes_accepted = bool(anchor_receipt.accepted and all(item.accepted for item in receipts[1:]))
    checks = {
        "strict_oof": strict_oof,
        "all_formal_routes_accepted": all_routes_accepted,
        "static_oof_promotion": static_passed,
        "future_perturbation": causal_passed,
        "runtime_budget": runtime_passed,
        "existing_promotion_rules": bool(existing_promotion_passed),
    }
    candidate_eligible = bool(all(checks.values()))
    report = {
        "experiment": "P3_rolling_integration",
        "status": "CANDIDATE_ELIGIBLE" if candidate_eligible else "STOP_FAIL_CLOSED",
        "checks": checks,
        "candidate_eligible": candidate_eligible,
        "champion": {
            "name": PARENT_ROUTE,
            "frozen": True,
            "replaced": False,
            "reason": "P3 只生成候选资格收据，绝不自动覆盖现有 A61 champion",
        },
        "future_perturbation": dict(future_gate) if isinstance(future_gate, Mapping) else None,
        "future_perturbation_validation": future_summary,
        "runtime": dict(runtime_payload) if isinstance(runtime_payload, Mapping) else None,
        "ensemble": ensemble.report,
    }
    return P3IntegrationResult(ensemble=ensemble, report=report)


def _project_long_capacity(rows: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    """对长表融合预测应用与线上宽表完全相同的容量投影。"""

    result = rows.loc[:, ["fold", "origin_time", "train_end", "horizon", "target"]].copy()
    result["prediction"] = np.asarray(prediction, dtype=float)
    generator_1_mask = result["target"].eq("generator_1")
    result.loc[generator_1_mask, "prediction"] = np.clip(
        result.loc[generator_1_mask, "prediction"].to_numpy(dtype=float),
        0.0,
        200.0,
    )
    generator_1 = result.loc[
        generator_1_mask,
        ["fold", "origin_time", "train_end", "horizon", "prediction"],
    ].rename(columns={"prediction": "generator_1_prediction"})
    generator_all_mask = result["target"].eq("generator_all")
    generator_all = result.loc[
        generator_all_mask,
        ["fold", "origin_time", "train_end", "horizon", "prediction"],
    ].merge(
        generator_1,
        on=["fold", "origin_time", "train_end", "horizon"],
        how="left",
        validate="one_to_one",
    )
    if generator_all["generator_1_prediction"].isna().any():
        raise ValueError("P3 融合缺少 generator_1 对应预测")
    all_values = np.clip(generator_all["prediction"].to_numpy(dtype=float), 0.0, 440.0)
    generator_1_values = generator_all["generator_1_prediction"].to_numpy(dtype=float)
    all_values = np.maximum(all_values, generator_1_values)
    all_values = np.minimum(all_values, generator_1_values + 240.0)
    result.loc[generator_all_mask, "prediction"] = all_values
    return result["prediction"].to_numpy(dtype=float)


def freeze_p3_static_weights(result: P3IntegrationResult) -> FrozenP3Fusion:
    """在通过 cross-fit 与全部门禁后，仅用 development OOF 冻结线上静态权重。

    ``build_p3_oof_integration`` 先以 leave-one-fold-out OOF 验证候选。只有
    它已经给出 ``CANDIDATE_ELIGIBLE`` 后，本函数才允许用完整 development
    OOF 选定一个预注册静态权重供 ``StaticWideFusion`` 使用。blind/final
    标签不在任何输入、选择或修正路径中。
    """

    if result.report.get("candidate_eligible") is not True:
        raise ValueError("P3 未通过完整门禁，禁止冻结线上融合权重")
    rows = result.ensemble.rows
    if result.ensemble.report.get("blind_labels_used") is not False:
        raise ValueError("P3 融合报告未证明 development OOF 不含 blind")
    routes_report = result.ensemble.report.get("routes")
    if not isinstance(routes_report, Mapping):
        raise ValueError("P3 融合报告缺少路线准入信息")
    accepted = routes_report.get("accepted", [])
    if not isinstance(accepted, list) or not accepted:
        raise ValueError("P3 没有已准入的辅助路线，禁止冻结融合权重")
    route_names = tuple(sorted(str(name) for name in accepted))
    required = {"actual", f"{PARENT_ROUTE}__prediction"}
    required.update(f"{name}__prediction" for name in route_names)
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"P3 OOF 缺少冻结权重所需列: {missing}")

    candidates: list[tuple[float, str, dict[str, float]]] = []
    for name, weights in pre_registered_weight_candidates(route_names):
        blended = np.zeros(len(rows), dtype=float)
        for route, weight in weights.items():
            blended += float(weight) * rows[f"{route}__prediction"].to_numpy(dtype=float)
        projected = _project_long_capacity(rows, blended)
        candidates.append((competition_mape(rows["actual"], projected), name, weights))
    score, selected_name, selected_weights = min(candidates, key=lambda item: (item[0], item[1]))
    frozen_weights = {name: float(weight) for name, weight in selected_weights.items()}
    return FrozenP3Fusion(
        weights=frozen_weights,
        report={
            "selection_source": "development_oof_only",
            "blind_labels_used": False,
            "final_labels_used": False,
            "pre_registered_candidate_count": len(candidates),
            "selected_candidate": selected_name,
            "selected_development_mape": float(score),
            "weights": frozen_weights,
            "capacity_projection": "generator_1 [0,200]; generator_all [0,440] and [g1,g1+240]",
        },
    )


def write_p3_integration_artifacts(result: P3IntegrationResult, run_dir: str | Path) -> Path:
    """写入独立 P3 OOF、融合和门禁收据，不触碰 ``results/best``。

    目录必须是新目录。无论候选资格最终为通过还是 STOP，都会落盘同一组
    可审计文件；STOP 不会被静默替换为 A61 的生产 champion。
    """

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    ensemble = result.ensemble
    ensemble.rows.to_csv(output / "p3_oof.csv", index=False, encoding="utf-8")
    ensemble.fold_metrics.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8")
    ensemble.target_metrics.to_csv(output / "target_metrics.csv", index=False, encoding="utf-8")
    ensemble.horizon_metrics.to_csv(output / "horizon_metrics.csv", index=False, encoding="utf-8")
    ensemble.residual_correlation.to_csv(
        output / "residual_correlation.csv", index=False, encoding="utf-8"
    )
    ensemble.training_trace.to_csv(output / "cross_fit_trace.csv", index=False, encoding="utf-8")
    ensemble.route_receipts.to_csv(output / "route_receipts.csv", index=False, encoding="utf-8")
    (output / "p3_integration_receipt.json").write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "A64_ROUTE",
    "FINAL_ROUTE",
    "FORMAL_ROUTE_NAMES",
    "FrozenP3Fusion",
    "HORIZONS_MINUTES",
    "HistoricalAnalogWideOriginPredictor",
    "MaturedResidualOriginPredictor",
    "OriginPredictor",
    "P1_ROUTE",
    "P2_ANALOG_ROUTE",
    "P2_MATURED_ROUTE",
    "P3IntegrationResult",
    "P3RouteOOF",
    "RuntimeBudget",
    "StaticWideFusion",
    "audit_future_perturbation_gate",
    "build_p3_oof_integration",
    "build_p3_route_oofs",
    "derive_anchor_folds",
    "future_feature_groups",
    "freeze_p3_static_weights",
    "runtime_budget_receipt",
    "validate_future_perturbation_receipt",
    "validate_p3_oof_keys",
    "write_p3_integration_artifacts",
]
