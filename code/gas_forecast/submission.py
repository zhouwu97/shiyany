"""正式提交链的校验、冻结、收据与 ZIP 封装。

本模块明确区分两类输入质量操作：

* :data:`Q_CAUSAL` 仅使用训练期统计量生成模型逐 origin 输入；
* :data:`Q_REFERENCE` 仅在 ``s_result.csv`` 已冻结后，对提交副本进行参考全矩阵归一化。

ZIP 阶段不拟合、不修复，也不重写任何 CSV 字节。这样可以把模型因果性、
最终提交质量和结果冻结分别审计。
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from gas_forecast import submission_quality as quality_api
from gas_forecast.config import ForecastConfig
from gas_forecast.submission_quality import COMPETITION_QUALITY_POLICY, SubmissionQualityPolicy

SUBMISSION_MEMBERS = ["input.csv", "s_result.csv"]
Q_CAUSAL = "Q_CAUSAL"
Q_REFERENCE = "Q_REFERENCE"
CAUSAL_MODEL_INPUT_RECEIPT = "causal_model_input_receipt.json"
SUBMISSION_QUALITY_RECEIPT = "submission_quality_receipt.json"


class OriginSubmissionPredictor(Protocol):
    """正式提交允许的唯一模型推理协议。"""

    def predict_at_origin(self, history_until_origin: pd.DataFrame) -> pd.DataFrame:
        """仅使用当前 origin 及其历史，返回当前 origin 的一行宽表预测。"""


def sha256_file(path: str | Path) -> str:
    """返回文件的 SHA-256；提交冻结统一使用该算法。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_default(value: object) -> object:
    """将 pandas/numpy 标量转换为收据可持久化的基础类型。"""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"收据包含不可序列化对象: {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path_or_payload: str | Path | Mapping[str, object], *, label: str) -> dict[str, object]:
    if isinstance(path_or_payload, Mapping):
        return dict(path_or_payload)
    path = Path(path_or_payload)
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label}不是合法 UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}必须是 JSON 对象: {path}")
    return payload


def _read_utf8_csv(path: str | Path, *, label: str) -> tuple[bytes, pd.DataFrame]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{label}不存在: {csv_path}")
    raw = csv_path.read_bytes()
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}必须为 UTF-8 编码: {csv_path}") from exc
    try:
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    except Exception as exc:  # pandas 会提供具体 CSV 解析上下文。
        raise ValueError(f"无法读取 {label}: {csv_path}") from exc
    return raw, frame


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _frame_time_summary(frame: pd.DataFrame, *, label: str) -> dict[str, object]:
    if frame.empty:
        raise ValueError(f"{label}为空")
    if "datetime" not in frame.columns:
        raise ValueError(f"{label}缺少 datetime")
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{label}的 datetime 含非法时间戳")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{label}的 datetime 必须唯一且严格递增")
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "start": str(timestamps.iloc[0]),
        "end": str(timestamps.iloc[-1]),
    }


def _validate_model_input_frame(frame: pd.DataFrame, *, label: str) -> dict[str, object]:
    summary = _frame_time_summary(frame, label=label)
    if frame.columns[0] != "datetime":
        raise ValueError(f"{label}第一列必须为 datetime")
    if frame.columns.duplicated().any():
        raise ValueError(f"{label}含重复字段")
    features = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if features.empty:
        raise ValueError(f"{label}不含模型输入字段")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(f"{label}的模型输入字段含缺失值、非数值或非有限值")
    summary["input_columns"] = int(len(features.columns))
    return summary


def _assert_same_frame(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    label: str,
) -> None:
    """验证 CSV 写回后的 schema、时间轴和数值与内存帧一致。"""

    if list(expected.columns) != list(actual.columns):
        raise ValueError(f"{label}写回后字段或顺序发生变化")
    if len(expected) != len(actual):
        raise ValueError(f"{label}写回后行数发生变化")
    expected_time = pd.to_datetime(expected["datetime"], errors="coerce")
    actual_time = pd.to_datetime(actual["datetime"], errors="coerce")
    if not expected_time.reset_index(drop=True).equals(actual_time.reset_index(drop=True)):
        raise ValueError(f"{label}写回后时间轴发生变化")
    try:
        np.testing.assert_allclose(
            expected.iloc[:, 1:].to_numpy(dtype=float),
            actual.iloc[:, 1:].to_numpy(dtype=float),
            # CSV 十进制往返在大数值上可能产生一个机器精度量级的解析差异。
            rtol=1e-15,
            atol=5e-12,
        )
    except AssertionError as exc:
        raise ValueError(f"{label}写回后数值发生变化") from exc


def _fitted_policy_dict(fitted_policy: object) -> dict[str, object]:
    exporter = getattr(fitted_policy, "to_dict", None)
    if not callable(exporter):
        raise TypeError("Q_CAUSAL 质量策略必须提供 to_dict() 以冻结训练期统计")
    payload = exporter()
    if not isinstance(payload, dict):
        raise TypeError("Q_CAUSAL 质量策略的 to_dict() 必须返回字典")
    return payload


def fit_causal_quality_policy(
    training_frame: pd.DataFrame,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    *,
    train_end: str | pd.Timestamp | None = None,
) -> object:
    """调用 Q1 的显式训练期策略拟合接口。"""

    return quality_api.fit_causal_quality_policy(
        training_frame,
        policy=policy,
        train_end=train_end,
    )


def _transform_causal_model_input(
    scoring_origin_rows: pd.DataFrame,
    fitted_policy: object,
) -> tuple[pd.DataFrame, dict[str, object]]:
    transformed, report = quality_api.transform_causal_model_input(
        scoring_origin_rows,
        fitted_policy,
    )
    if not isinstance(transformed, pd.DataFrame) or not isinstance(report, dict):
        raise TypeError("Q1 评分期质量变换必须返回 (DataFrame, dict)")
    return transformed, report


def _quality_future_feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    """按统一口径列出 Q_CAUSAL 需要隔离的未来字段组。"""

    numeric = [
        str(column)
        for column in frame.columns
        if column != "datetime" and pd.api.types.is_numeric_dtype(frame[column])
    ]
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


def _quality_gate_origins(scoring_origin_rows: pd.DataFrame) -> list[int]:
    """选取仍有未来行的稳定 origin 位置，避免只审计尾行。"""

    if len(scoring_origin_rows) < 2:
        return []
    positions = {0, (len(scoring_origin_rows) - 2) // 2, len(scoring_origin_rows) - 2}
    return sorted(positions)


def _quality_prefix_equal(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, float | None, str | None]:
    """以严格 schema、时间轴和 bitwise 数值比较 Q_CAUSAL 前缀。"""

    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False, None, "模型输入 schema 或行数发生变化"
    if not left["datetime"].equals(right["datetime"]):
        return False, None, "模型输入时间轴发生变化"
    try:
        baseline = left.iloc[:, 1:].to_numpy(dtype=float)
        candidate = right.iloc[:, 1:].to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False, None, "模型输入不是纯数值"
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        return False, None, "模型输入包含 NaN/Inf"
    maximum = float(np.max(np.abs(baseline - candidate))) if baseline.size else 0.0
    return bool(np.array_equal(baseline, candidate)), maximum, None


def _quality_future_variant(
    scoring_origin_rows: pd.DataFrame,
    *,
    origin_position: int,
    columns: Sequence[str],
    operation: str,
) -> pd.DataFrame:
    """只修改指定 origin 后的质量输入，不改变当期及此前任何单元格。"""

    future_index = scoring_origin_rows.index[origin_position + 1 :]
    if operation == "delete" and columns == _quality_future_feature_groups(scoring_origin_rows)["all_features"]:
        return scoring_origin_rows.iloc[: origin_position + 1].copy(deep=True)
    result = scoring_origin_rows.copy(deep=True)
    if operation == "perturb":
        for position, column in enumerate(columns):
            result.loc[future_index, column] = float(-9_999_991 - position)
    elif operation == "delete":
        result.loc[future_index, list(columns)] = np.nan
    else:
        raise ValueError(f"不支持的 Q_CAUSAL 未来扰动操作: {operation}")
    return result


def _future_perturbation_receipt(
    scoring_origin_rows: pd.DataFrame,
    fitted_policy: object,
    baseline: pd.DataFrame,
) -> dict[str, object]:
    """证明 Q_CAUSAL 的每个 origin 输入不读取后续评分块。"""

    groups = _quality_future_feature_groups(scoring_origin_rows)
    positions = _quality_gate_origins(scoring_origin_rows)
    if not positions:
        return {
            "gate": "q_causal_future_perturbation_v2",
            "passed": True,
            "origins": [],
            "groups": groups,
            "operations": ["perturb", "delete"],
            "max_abs_diff": 0.0,
            "cases": [],
            "method": "单行评分输入无可扰动的未来 origin；变换只读取冻结训练统计",
        }

    cases: list[dict[str, object]] = []
    passed = True
    maximum = 0.0
    for origin_position in positions:
        origin = str(scoring_origin_rows["datetime"].iloc[origin_position])
        expected_prefix = baseline.iloc[: origin_position + 1].reset_index(drop=True)
        for group, columns in groups.items():
            for operation in ("perturb", "delete"):
                if not columns:
                    cases.append(
                        {
                            "origin": origin,
                            "group": group,
                            "operation": operation,
                            "passed": True,
                            "skipped": True,
                            "bitwise_identical": True,
                            "max_abs_diff": 0.0,
                            "reason": "输入不存在该类别的数值字段",
                        }
                    )
                    continue
                try:
                    variant = _quality_future_variant(
                        scoring_origin_rows,
                        origin_position=origin_position,
                        columns=columns,
                        operation=operation,
                    )
                    transformed, _ = _transform_causal_model_input(variant, fitted_policy)
                    observed_prefix = transformed.iloc[: origin_position + 1].reset_index(drop=True)
                    identical, difference, reason = _quality_prefix_equal(
                        expected_prefix,
                        observed_prefix,
                    )
                except Exception as error:  # 质量因果门禁必须留下失败原因。
                    identical, difference, reason = (
                        False,
                        None,
                        f"{type(error).__name__}: {error}",
                    )
                passed = passed and identical
                if difference is not None:
                    maximum = max(maximum, difference)
                cases.append(
                    {
                        "origin": origin,
                        "group": group,
                        "operation": operation,
                        "passed": identical,
                        "bitwise_identical": identical,
                        "max_abs_diff": difference,
                        "reason": reason,
                    }
                )
    return {
        "gate": "q_causal_future_perturbation_v2",
        "passed": passed,
        "origins": [str(scoring_origin_rows["datetime"].iloc[position]) for position in positions],
        "groups": groups,
        "operations": ["perturb", "delete"],
        "max_abs_diff": maximum,
        "cases": cases,
        "method": "分别扰动或删除 origin 后五类字段，比较全部此前 Q_CAUSAL 输入",
    }


def prepare_causal_model_input(
    training_frame: pd.DataFrame,
    scoring_origin_rows: pd.DataFrame,
    *,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    train_end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """拟合一次训练期质量策略，并生成不读取未来分布的模型输入。"""

    fitted_policy = fit_causal_quality_policy(training_frame, policy, train_end=train_end)
    transformed, transform_report = _transform_causal_model_input(scoring_origin_rows, fitted_policy)
    input_summary = _validate_model_input_frame(transformed, label="Q_CAUSAL 模型输入")
    frozen_policy = _fitted_policy_dict(fitted_policy)
    future_check = _future_perturbation_receipt(scoring_origin_rows, fitted_policy, transformed)
    if future_check.get("passed") is not True:
        raise ValueError("Q_CAUSAL 未来扰动门禁失败，禁止继续生成模型输入")
    frozen_policy_sha256 = _sha256_bytes(
        json.dumps(
            frozen_policy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    )
    receipt: dict[str, object] = {
        "receipt_version": 1,
        "quality_mode": Q_CAUSAL,
        "purpose": "模型逐 origin 输入；仅使用训练期冻结统计",
        "training_statistics_frozen": True,
        "future_values_can_influence_policy": False,
        "q_reference_available_during_model_input": False,
        "fitted_policy": frozen_policy,
        "fitted_policy_sha256": frozen_policy_sha256,
        "transform_report": transform_report,
        "model_input": input_summary,
        "future_perturbation": future_check,
    }
    return transformed, receipt


def prepare_reference_submission_input(
    causal_model_input: pd.DataFrame,
    *,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """为冻结预测后的副本执行 :data:`Q_REFERENCE` 全矩阵归一化。

    该阶段允许参考整个提交输入矩阵，但其输出不得回流至模型预测。算法仍由
    Q1 提供；本模块只负责职责边界和收据。
    """

    normalized, report = quality_api.prepare_reference_submission_input(
        causal_model_input.copy(deep=True),
        policy=policy,
    )
    if not isinstance(normalized, pd.DataFrame) or not isinstance(report, dict):
        raise TypeError("Q1 参考归一化必须返回 (DataFrame, dict)")
    _validate_model_input_frame(normalized, label="Q_REFERENCE 提交输入")
    return normalized, report


def expected_prediction_columns(config: ForecastConfig | None = None) -> list[str]:
    config = config or ForecastConfig()
    return [
        f"{target}_t+{15 * horizon}_pred"
        for target in config.targets
        for horizon in config.feature.horizons
    ]


def predict_submission_by_origin(
    causal_model_input: pd.DataFrame,
    predictor: OriginSubmissionPredictor,
    *,
    config: ForecastConfig | None = None,
) -> pd.DataFrame:
    """逐 origin 从冻结 Q_CAUSAL 输入生成合法 ``s_result``。

    此入口故意不接收整段 ``scoring_frame`` 的批量推理方法。每次调用都仅把
    截至当前 origin 的前缀交给预测器，并要求其只返回一个同 origin 的宽表。
    这样结果文件的生成顺序可以由收据和测试共同审计。
    """

    if not callable(getattr(predictor, "predict_at_origin", None)):
        raise TypeError("正式提交预测器必须提供 predict_at_origin(history_until_origin)")
    _validate_model_input_frame(causal_model_input, label="逐 origin Q_CAUSAL 输入")
    indexed = causal_model_input.copy(deep=True)
    indexed["datetime"] = pd.to_datetime(indexed["datetime"], errors="raise")
    indexed = indexed.set_index("datetime", drop=True)
    expected = expected_prediction_columns(config)
    results: list[pd.DataFrame] = []
    for origin in indexed.index:
        history = indexed.loc[:origin].copy(deep=True)
        predicted = predictor.predict_at_origin(history)
        if not isinstance(predicted, pd.DataFrame):
            raise TypeError("predict_at_origin 必须返回 pandas.DataFrame")
        if len(predicted) != 1 or not isinstance(predicted.index, pd.DatetimeIndex):
            raise ValueError("predict_at_origin 必须返回带 DatetimeIndex 的单行宽表")
        if pd.Timestamp(predicted.index[0]) != pd.Timestamp(origin):
            raise ValueError("predict_at_origin 返回的 origin 与输入历史末端不一致")
        if list(predicted.columns) != expected:
            raise ValueError("predict_at_origin 返回的预测字段或顺序不合法")
        values = predicted.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("predict_at_origin 返回 NaN/Inf")
        result = predicted.copy(deep=True).reset_index(names="datetime")
        results.append(result)
    combined = pd.concat(results, ignore_index=True)
    validate_submission_frame(combined, config)
    return combined


def validate_submission_frame(
    frame: pd.DataFrame,
    config: ForecastConfig | None = None,
) -> dict[str, object]:
    config = config or ForecastConfig()
    expected = ["datetime", *expected_prediction_columns(config)]
    if list(frame.columns) != expected:
        missing = [column for column in expected if column not in frame.columns]
        extra = [column for column in frame.columns if column not in expected]
        raise ValueError(f"结果字段或顺序不合法，缺少={missing}，多余={extra}")
    if frame.empty:
        raise ValueError("结果文件为空")

    timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("datetime 含非法时间戳")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("datetime 必须唯一且严格递增")
    spacing = timestamps.diff().dropna()
    if not spacing.empty and not spacing.eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("滚动起点必须保持 15 分钟连续")

    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("预测字段含缺失值、非数值或非有限值")

    for horizon in config.feature.horizons:
        minutes = 15 * horizon
        generator_1 = values[f"generator_1_t+{minutes}_pred"]
        generator_all = values[f"generator_all_t+{minutes}_pred"]
        if not generator_1.between(0, 200).all():
            raise ValueError(f"generator_1 t+{minutes} 超出 [0, 200]")
        if not generator_all.between(0, 440).all():
            raise ValueError(f"generator_all t+{minutes} 超出 [0, 440]")
        if not (generator_all >= generator_1).all():
            raise ValueError(f"t+{minutes} 存在 generator_all < generator_1")

    return {
        "rows": int(len(frame)),
        "prediction_columns": int(len(values.columns)),
        "start": str(timestamps.iloc[0]),
        "end": str(timestamps.iloc[-1]),
    }


def validate_submission_input(
    input_frame: pd.DataFrame,
    result_frame: pd.DataFrame,
    *,
    quality_policy: SubmissionQualityPolicy | None = None,
    enforce_quality: bool = False,
) -> dict[str, object]:
    """校验提交输入特征，并确认它与预测结果逐行对应。

    ``quality_policy`` 是旧调用方的显式审计入口。正式 ZIP 封装不会传入它，
    因而不会从提交输入重新估计任何质量统计。
    """

    input_summary = _validate_model_input_frame(input_frame, label="input.csv")
    result_timestamps = pd.to_datetime(result_frame["datetime"], errors="coerce")
    input_timestamps = pd.to_datetime(input_frame["datetime"], errors="coerce")
    if len(input_frame) != len(result_frame) or not input_timestamps.reset_index(drop=True).equals(
        result_timestamps.reset_index(drop=True)
    ):
        raise ValueError("input.csv 与 s_result.csv 的行数或时间戳不一致")

    summary: dict[str, object] = {
        "rows": input_summary["rows"],
        "input_columns": input_summary["input_columns"],
        "start": input_summary["start"],
        "end": input_summary["end"],
    }
    if quality_policy is not None:
        auditor = (
            getattr(quality_api, "enforce_submission_quality")
            if enforce_quality
            else getattr(quality_api, "audit_submission_quality")
        )
        summary["quality"] = auditor(input_frame, quality_policy)
    return summary


def freeze_submission_result(result_path: str | Path) -> dict[str, object]:
    """在任何 Q_REFERENCE 输入变换前冻结合法 ``s_result.csv``。"""

    raw, frame = _read_utf8_csv(result_path, label="s_result.csv")
    validation = validate_submission_frame(frame)
    return {
        "algorithm": "sha256",
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "validation": validation,
    }


def verify_submission_result_freeze(
    result_path: str | Path,
    freeze: str | Path | Mapping[str, object],
) -> dict[str, object]:
    """逐字节与逐值复核结果仍等于冻结时的合法结果。"""

    payload = _read_json(freeze, label="s_result 冻结记录")
    if payload.get("algorithm") != "sha256":
        raise ValueError("s_result 冻结记录仅支持 sha256")
    expected_hash = payload.get("sha256")
    expected_size = payload.get("size_bytes")
    if not isinstance(expected_hash, str) or not isinstance(expected_size, int):
        raise ValueError("s_result 冻结记录缺少 sha256 或 size_bytes")
    raw, frame = _read_utf8_csv(result_path, label="待复核 s_result.csv")
    validation = validate_submission_frame(frame)
    if len(raw) != expected_size:
        raise ValueError("冻结后的 s_result.csv 字节数发生变化")
    actual_hash = _sha256_bytes(raw)
    if actual_hash != expected_hash:
        raise ValueError("冻结后的 s_result.csv SHA256 发生变化")
    return {"verified": True, "sha256": actual_hash, "size_bytes": len(raw), "validation": validation}


def _file_record(path: Path, *, relative_path: str) -> dict[str, object]:
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def validate_causal_model_input_receipt(
    causal_input_path: str | Path,
    receipt: str | Path | Mapping[str, object],
) -> dict[str, object]:
    """验证已有 Q_CAUSAL 文件与其冻结策略收据匹配，无需重新拟合。"""

    payload = _read_json(receipt, label="因果模型输入收据")
    if payload.get("quality_mode") != Q_CAUSAL:
        raise ValueError("因果模型输入收据的 quality_mode 必须为 Q_CAUSAL")
    if payload.get("training_statistics_frozen") is not True:
        raise ValueError("因果模型输入收据未声明训练统计已冻结")
    if payload.get("future_values_can_influence_policy") is not False:
        raise ValueError("因果模型输入收据未证明未来值不能影响策略")
    if payload.get("q_reference_available_during_model_input") is not False:
        raise ValueError("因果模型输入收据未证明 Q_REFERENCE 不参与模型输入")
    frozen_policy = payload.get("fitted_policy")
    frozen_policy_sha256 = payload.get("fitted_policy_sha256")
    if not isinstance(frozen_policy, dict) or not isinstance(frozen_policy_sha256, str):
        raise ValueError("因果模型输入收据缺少冻结策略")
    actual_policy_sha256 = _sha256_bytes(
        json.dumps(
            frozen_policy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    )
    if actual_policy_sha256 != frozen_policy_sha256:
        raise ValueError("因果模型输入收据中的冻结策略哈希不一致")
    future_check = payload.get("future_perturbation")
    if (
        not isinstance(future_check, dict)
        or future_check.get("gate") != "q_causal_future_perturbation_v2"
        or future_check.get("passed") is not True
        or future_check.get("max_abs_diff") != 0.0
    ):
        raise ValueError("因果模型输入收据缺少通过的未来扰动复核")
    files = payload.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("causal_model_input"), dict):
        raise ValueError("因果模型输入收据缺少 causal_model_input 文件记录")
    record = files["causal_model_input"]
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    path = Path(causal_input_path)
    if not path.is_file():
        raise FileNotFoundError(f"因果模型输入不存在: {path}")
    if not isinstance(expected_hash, str) or not isinstance(expected_size, int):
        raise ValueError("因果模型输入收据的文件哈希无效")
    if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
        raise ValueError("因果模型输入与冻结收据不一致，禁止重新使用")
    _, frame = _read_utf8_csv(path, label="因果模型输入")
    _validate_model_input_frame(frame, label="因果模型输入")
    return payload


def _write_reference_submission(
    *,
    causal_input_path: Path,
    causal_receipt: dict[str, object],
    result_source_path: Path,
    output_dir: Path,
    policy: SubmissionQualityPolicy,
    input_name: str,
    result_name: str,
    quality_receipt_name: str,
    prediction_input: Mapping[str, object],
) -> dict[str, object]:
    """执行冻结后的 Q_REFERENCE、副本写盘及 read-back 复检。"""

    _, causal_input = _read_utf8_csv(causal_input_path, label="Q_CAUSAL 模型输入")
    causal_summary = _validate_model_input_frame(causal_input, label="Q_CAUSAL 模型输入")
    causal_before_reference = _file_record(
        causal_input_path,
        relative_path=causal_input_path.name,
    )
    source_result_bytes, source_result = _read_utf8_csv(result_source_path, label="s_result.csv")
    result_validation = validate_submission_frame(source_result)
    validate_submission_input(causal_input, source_result)

    # 这是 Q_REFERENCE 之前唯一的结果冻结点；其后的任何文件操作都要复核该哈希。
    result_freeze = freeze_submission_result(result_source_path)
    result_destination = output_dir / result_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if result_destination.resolve() != result_source_path.resolve():
        shutil.copyfile(result_source_path, result_destination)

    # 从已落盘的因果输入创建独立副本，确保参考归一化绝不回写模型输入。
    reference_input, reference_report = prepare_reference_submission_input(causal_input, policy=policy)
    input_destination = output_dir / input_name
    _write_csv(reference_input, input_destination)

    input_bytes, read_back_input = _read_utf8_csv(input_destination, label="写回 input.csv")
    result_bytes, read_back_result = _read_utf8_csv(result_destination, label="写回 s_result.csv")
    input_validation = validate_submission_input(read_back_input, read_back_result)
    read_back_validation = validate_submission_frame(read_back_result)
    _assert_same_frame(reference_input, read_back_input, label="Q_REFERENCE input.csv")
    _assert_same_frame(source_result, read_back_result, label="冻结 s_result.csv")
    result_freeze_check = verify_submission_result_freeze(result_destination, result_freeze)
    if result_bytes != source_result_bytes:
        raise ValueError("Q_REFERENCE 阶段改变了 s_result.csv 的原始字节")
    causal_after_reference = _file_record(
        causal_input_path,
        relative_path=causal_input_path.name,
    )
    if causal_after_reference != causal_before_reference:
        raise ValueError("Q_REFERENCE 阶段改写了冻结 Q_CAUSAL 模型输入")

    quality_receipt: dict[str, object] = {
        "receipt_version": 1,
        "quality_mode": Q_REFERENCE,
        "purpose": "冻结预测后的提交输入副本参考全矩阵归一化",
        "reference_only": True,
        "feeds_model": False,
        "causal_model_input": {
            **_file_record(causal_input_path, relative_path=causal_input_path.name),
            "validation": causal_summary,
            "receipt_sha256": _sha256_bytes(
                json.dumps(
                    causal_receipt,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=_json_default,
                ).encode("utf-8")
            ),
        },
        "prediction_input": dict(prediction_input),
        "causal_input_immutable_after_prediction": {
            "passed": True,
            "before_reference": causal_before_reference,
            "after_reference": causal_after_reference,
        },
        "s_result_freeze": {
            **result_freeze,
            "source_path": result_source_path.name,
            "final_path": result_name,
            "verified_after_reference": result_freeze_check["verified"],
        },
        "reference_normalization": reference_report,
        "final_files": {
            "input.csv": _file_record(input_destination, relative_path=input_name),
            "s_result.csv": _file_record(result_destination, relative_path=result_name),
        },
        "write_read_back": {
            "input.csv": {
                "schema_matches": True,
                "rows_match": True,
                "time_axis_matches": True,
                "numeric_values_match": True,
                "sha256": _sha256_bytes(input_bytes),
                "validation": input_validation,
            },
            "s_result.csv": {
                "schema_matches": True,
                "rows_match": True,
                "time_axis_matches": True,
                "numeric_values_match": True,
                "bytes_match_frozen_source": True,
                "sha256": _sha256_bytes(result_bytes),
                "validation": read_back_validation,
            },
        },
        "result_validation_before_reference": result_validation,
    }
    quality_receipt_path = output_dir / quality_receipt_name
    _write_json(quality_receipt_path, quality_receipt)
    validate_submission_quality_receipt(
        input_destination,
        result_destination,
        quality_receipt,
    )
    return {
        "input_path": input_destination,
        "result_path": result_destination,
        "quality_receipt_path": quality_receipt_path,
        "quality_receipt": quality_receipt,
        "result_freeze": result_freeze,
        "input_validation": input_validation,
        "result_validation": read_back_validation,
    }


def prepare_submission_chain(
    training_input_path: str | Path,
    scoring_input_path: str | Path,
    result_path: str | Path,
    output_dir: str | Path,
    *,
    train_end: str | pd.Timestamp | None = None,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    causal_input_name: str = "causal_model_input.csv",
    input_name: str = "input.csv",
    result_name: str = "s_result.csv",
    causal_receipt_name: str = CAUSAL_MODEL_INPUT_RECEIPT,
    quality_receipt_name: str = SUBMISSION_QUALITY_RECEIPT,
    prediction_builder: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    prediction_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """执行提交链：训练统计冻结 → 因果输入 → 结果冻结 → 参考副本。

    传入 ``prediction_builder`` 时，该回调只能接收已落盘的 ``Q_CAUSAL``
    输入副本，并且必须返回合法的 ``s_result`` 宽表。这是正式新链路：
    ``training → Q_CAUSAL → per-origin model input → result freeze → Q_REFERENCE``。
    未传回调时保留既有已生成结果的兼容入口，但收据会明确标记为旧外部结果，
    不能据此证明模型预测使用了 ``Q_CAUSAL`` 输入。
    """

    training_path = Path(training_input_path)
    scoring_path = Path(scoring_input_path)
    source_result_path = Path(result_path)
    destination = Path(output_dir)
    training_bytes, training_frame = _read_utf8_csv(training_path, label="训练期质量输入")
    scoring_bytes, scoring_frame = _read_utf8_csv(scoring_path, label="评分 origin 输入")
    causal_input, causal_receipt = prepare_causal_model_input(
        training_frame,
        scoring_frame,
        policy=policy,
        train_end=train_end,
    )

    destination.mkdir(parents=True, exist_ok=True)
    causal_input_path = destination / causal_input_name
    _write_csv(causal_input, causal_input_path)
    _, read_back_causal = _read_utf8_csv(causal_input_path, label="写回 Q_CAUSAL 模型输入")
    _assert_same_frame(causal_input, read_back_causal, label="Q_CAUSAL 模型输入")
    causal_receipt["files"] = {
        "training_input": {
            "path": str(training_path),
            "sha256": _sha256_bytes(training_bytes),
            "size_bytes": len(training_bytes),
        },
        "scoring_origin_input": {
            "path": str(scoring_path),
            "sha256": _sha256_bytes(scoring_bytes),
            "size_bytes": len(scoring_bytes),
        },
        "causal_model_input": _file_record(causal_input_path, relative_path=causal_input_name),
    }
    causal_receipt["write_read_back"] = {
        "schema_matches": True,
        "rows_match": True,
        "time_axis_matches": True,
        "numeric_values_match": True,
        "sha256": sha256_file(causal_input_path),
    }
    causal_receipt_path = destination / causal_receipt_name
    _write_json(causal_receipt_path, causal_receipt)
    validate_causal_model_input_receipt(causal_input_path, causal_receipt)

    prediction_input: dict[str, object] = {
        "causal_model_input": _file_record(
            causal_input_path,
            relative_path=causal_input_path.name,
        ),
        "q_reference_available_during_prediction": False,
    }
    if prediction_builder is not None:
        generated = prediction_builder(read_back_causal.copy(deep=True))
        if not isinstance(generated, pd.DataFrame):
            raise TypeError("prediction_builder 必须返回 pandas.DataFrame")
        validate_submission_frame(generated)
        _write_csv(generated, source_result_path)
        prediction_input.update(
            {
                "generated_after_q_causal": True,
                "builder": getattr(prediction_builder, "__qualname__", type(prediction_builder).__name__),
            }
        )
        if prediction_evidence is not None:
            prediction_input.update(dict(prediction_evidence))
    else:
        prediction_input.update(
            {
                "generated_after_q_causal": False,
                "reason": "兼容旧入口：结果由外部预先生成，未声明 Q_CAUSAL 模型输入证明",
            }
        )

    reference = _write_reference_submission(
        causal_input_path=causal_input_path,
        causal_receipt=causal_receipt,
        result_source_path=source_result_path,
        output_dir=destination,
        policy=policy,
        input_name=input_name,
        result_name=result_name,
        quality_receipt_name=quality_receipt_name,
        prediction_input=prediction_input,
    )
    return {
        "causal_input_path": causal_input_path,
        "causal_receipt_path": causal_receipt_path,
        "causal_receipt": causal_receipt,
        **reference,
    }


def prepare_submission_chain_with_predictor(
    training_input_path: str | Path,
    scoring_input_path: str | Path,
    output_dir: str | Path,
    *,
    prediction_builder: Callable[[pd.DataFrame], pd.DataFrame],
    train_end: str | pd.Timestamp | None = None,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    causal_input_name: str = "causal_model_input.csv",
    input_name: str = "input.csv",
    result_name: str = "s_result.csv",
    causal_receipt_name: str = CAUSAL_MODEL_INPUT_RECEIPT,
    quality_receipt_name: str = SUBMISSION_QUALITY_RECEIPT,
) -> dict[str, object]:
    """执行强制 ``Q_CAUSAL`` 先于模型预测的正式提交链。

    与兼容入口不同，此函数没有预生成 ``result_path``：结果只能由回调在
    ``Q_REFERENCE`` 尚未出现时创建。它适合新的逐 origin 预测器接入正式
    冻结与 ZIP 流程。
    """

    destination = Path(output_dir)
    raw_result = destination / "causal_model_s_result.csv"
    stale_reference = [
        path
        for path in (destination / input_name, destination / quality_receipt_name)
        if path.exists()
    ]
    if stale_reference:
        raise FileExistsError(
            f"正式预测前发现已有 Q_REFERENCE 工件，拒绝其回流: {stale_reference}"
        )
    return prepare_submission_chain(
        training_input_path,
        scoring_input_path,
        raw_result,
        destination,
        train_end=train_end,
        policy=policy,
        causal_input_name=causal_input_name,
        input_name=input_name,
        result_name=result_name,
        causal_receipt_name=causal_receipt_name,
        quality_receipt_name=quality_receipt_name,
        prediction_builder=prediction_builder,
    )


def prepare_submission_chain_with_origin_predictor(
    training_input_path: str | Path,
    scoring_input_path: str | Path,
    output_dir: str | Path,
    *,
    predictor: OriginSubmissionPredictor,
    train_end: str | pd.Timestamp | None = None,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    config: ForecastConfig | None = None,
    causal_input_name: str = "causal_model_input.csv",
    input_name: str = "input.csv",
    result_name: str = "s_result.csv",
    causal_receipt_name: str = CAUSAL_MODEL_INPUT_RECEIPT,
    quality_receipt_name: str = SUBMISSION_QUALITY_RECEIPT,
) -> dict[str, object]:
    """执行强制逐 origin 推理的完整正式提交链。

    这是新代码应使用的入口：训练期拟合一次 Q_CAUSAL，随后每个评分 origin
    仅传入历史前缀给 ``predict_at_origin``，在结果合法且 SHA256 冻结后才
    生成 Q_REFERENCE ``input.csv``。已有外部结果的兼容入口不具备这条证明。
    """

    if not callable(getattr(predictor, "predict_at_origin", None)):
        raise TypeError("正式提交预测器必须提供 predict_at_origin(history_until_origin)")

    evidence: dict[str, object] = {
        "origin_only_predictor": True,
        "prediction_protocol": "predict_at_origin(history_until_origin)",
        "prediction_origin_count": 0,
        "predictor": type(predictor).__name__,
    }

    def origin_only_builder(causal_input: pd.DataFrame) -> pd.DataFrame:
        result = predict_submission_by_origin(causal_input, predictor, config=config)
        evidence["prediction_origin_count"] = int(len(result))
        return result

    destination = Path(output_dir)
    raw_result = destination / "causal_model_s_result.csv"
    stale_reference = [
        path
        for path in (destination / input_name, destination / quality_receipt_name, raw_result)
        if path.exists()
    ]
    if stale_reference:
        raise FileExistsError(
            f"正式逐 origin 预测前发现已有工件，拒绝覆盖或读取: {stale_reference}"
        )
    return prepare_submission_chain(
        training_input_path,
        scoring_input_path,
        raw_result,
        destination,
        train_end=train_end,
        policy=policy,
        causal_input_name=causal_input_name,
        input_name=input_name,
        result_name=result_name,
        causal_receipt_name=causal_receipt_name,
        quality_receipt_name=quality_receipt_name,
        prediction_builder=origin_only_builder,
        prediction_evidence=evidence,
    )


def prepare_submission_from_frozen_causal_input(
    causal_input_path: str | Path,
    causal_receipt_path: str | Path,
    result_path: str | Path,
    output_dir: str | Path,
    *,
    policy: SubmissionQualityPolicy = COMPETITION_QUALITY_POLICY,
    input_name: str = "input.csv",
    result_name: str = "s_result.csv",
    quality_receipt_name: str = SUBMISSION_QUALITY_RECEIPT,
) -> dict[str, object]:
    """复用已冻结 Q_CAUSAL 输入生成新提交副本，不重新拟合训练统计。"""

    source_causal_path = Path(causal_input_path)
    source_receipt = validate_causal_model_input_receipt(source_causal_path, causal_receipt_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    destination_causal_path = destination / source_causal_path.name
    if destination_causal_path.resolve() != source_causal_path.resolve():
        shutil.copyfile(source_causal_path, destination_causal_path)
    copied_receipt_path = destination / CAUSAL_MODEL_INPUT_RECEIPT
    source_receipt_path = Path(causal_receipt_path)
    if copied_receipt_path.resolve() != source_receipt_path.resolve():
        shutil.copyfile(source_receipt_path, copied_receipt_path)
    validate_causal_model_input_receipt(destination_causal_path, source_receipt)
    reference = _write_reference_submission(
        causal_input_path=destination_causal_path,
        causal_receipt=source_receipt,
        result_source_path=Path(result_path),
        output_dir=destination,
        policy=policy,
        input_name=input_name,
        result_name=result_name,
        quality_receipt_name=quality_receipt_name,
        prediction_input={
            "causal_model_input": _file_record(
                destination_causal_path,
                relative_path=destination_causal_path.name,
            ),
            "generated_after_q_causal": False,
            "q_reference_available_during_prediction": False,
            "reason": "复用冻结 Q_CAUSAL 输入；不重新拟合或重新预测",
        },
    )
    return {
        "causal_input_path": destination_causal_path,
        "causal_receipt_path": copied_receipt_path,
        "causal_receipt": source_receipt,
        **reference,
    }


def validate_submission_quality_receipt(
    input_path: str | Path,
    result_path: str | Path,
    receipt: str | Path | Mapping[str, object],
) -> dict[str, object]:
    """校验 Q_REFERENCE 收据、最终文件哈希及结果冻结不变式。"""

    payload = _read_json(receipt, label="提交质量收据")
    if payload.get("quality_mode") != Q_REFERENCE:
        raise ValueError("提交质量收据的 quality_mode 必须为 Q_REFERENCE")
    if payload.get("reference_only") is not True or payload.get("feeds_model") is not False:
        raise ValueError("提交质量收据未声明参考归一化与模型输入职责隔离")
    prediction_input = payload.get("prediction_input")
    if not isinstance(prediction_input, dict):
        raise ValueError("提交质量收据缺少预测输入链路记录")
    if prediction_input.get("q_reference_available_during_prediction") is not False:
        raise ValueError("预测输入链路未证明 Q_REFERENCE 不回流模型")
    if prediction_input.get("origin_only_predictor") is True:
        if prediction_input.get("prediction_protocol") != "predict_at_origin(history_until_origin)":
            raise ValueError("逐 origin 预测收据缺少固定协议声明")
        origin_count = prediction_input.get("prediction_origin_count")
        if not isinstance(origin_count, int) or origin_count <= 0:
            raise ValueError("逐 origin 预测收据缺少有效 origin 数")
    causal_record = payload.get("causal_model_input")
    prediction_causal = prediction_input.get("causal_model_input")
    if not isinstance(causal_record, dict) or not isinstance(prediction_causal, dict):
        raise ValueError("提交质量收据缺少 Q_CAUSAL 输入哈希关联")
    if causal_record.get("sha256") != prediction_causal.get("sha256"):
        raise ValueError("预测输入与 Q_CAUSAL 收据哈希不一致")
    causal_immutable = payload.get("causal_input_immutable_after_prediction")
    if not isinstance(causal_immutable, dict) or causal_immutable.get("passed") is not True:
        raise ValueError("提交质量收据未证明 Q_REFERENCE 未改写 Q_CAUSAL 输入")
    before_reference = causal_immutable.get("before_reference")
    after_reference = causal_immutable.get("after_reference")
    if (
        not isinstance(before_reference, dict)
        or not isinstance(after_reference, dict)
        or before_reference.get("sha256") != causal_record.get("sha256")
        or after_reference.get("sha256") != causal_record.get("sha256")
    ):
        raise ValueError("提交质量收据中的 Q_CAUSAL 不可变哈希不一致")
    final_files = payload.get("final_files")
    if not isinstance(final_files, dict):
        raise ValueError("提交质量收据缺少 final_files")
    for name, path in (("input.csv", Path(input_path)), ("s_result.csv", Path(result_path))):
        record = final_files.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"提交质量收据缺少 {name} 文件记录")
        expected_hash = record.get("sha256")
        expected_size = record.get("size_bytes")
        if not isinstance(expected_hash, str) or not isinstance(expected_size, int):
            raise ValueError(f"提交质量收据中的 {name} 哈希无效")
        if not path.is_file() or path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise ValueError(f"{name} 与提交质量收据不一致")
    freeze = payload.get("s_result_freeze")
    if not isinstance(freeze, dict):
        raise ValueError("提交质量收据缺少 s_result 冻结记录")
    verify_submission_result_freeze(result_path, freeze)
    read_back = payload.get("write_read_back")
    if not isinstance(read_back, dict):
        raise ValueError("提交质量收据缺少 write_read_back 复检记录")
    for name in ("input.csv", "s_result.csv"):
        check = read_back.get(name)
        if not isinstance(check, dict) or not all(
            check.get(key) is True
            for key in ("schema_matches", "rows_match", "time_axis_matches", "numeric_values_match")
        ):
            raise ValueError(f"提交质量收据中的 {name} 写回复检未通过")
    _, input_frame = _read_utf8_csv(input_path, label="收据 input.csv")
    _, result_frame = _read_utf8_csv(result_path, label="收据 s_result.csv")
    input_validation = validate_submission_input(input_frame, result_frame)
    result_validation = validate_submission_frame(result_frame)
    return {
        "valid": True,
        "quality_mode": Q_REFERENCE,
        "input": input_validation,
        "result": result_validation,
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def validate_submission_archive(
    archive_path: str | Path,
    *,
    expected_input_path: str | Path | None = None,
    expected_result_path: str | Path | None = None,
    quality_policy: SubmissionQualityPolicy | None = None,
    quality_receipt_path: str | Path | Mapping[str, object] | None = None,
    result_freeze: str | Path | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """校验 ZIP 成员、内部 CSV 及其与磁盘冻结文件的字节一致性。

    ``quality_policy`` 仅保留旧调用兼容性，不会在此处审计、拟合或修复输入。
    正式链应传入 ``quality_receipt_path`` 以复核 Q_REFERENCE 收据。
    """

    del quality_policy
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if names != SUBMISSION_MEMBERS:
            raise ValueError(f"提交 ZIP 必须依次包含 {SUBMISSION_MEMBERS}，实际为: {names}")
        archived_input_bytes = archive.read("input.csv")
        archived_result_bytes = archive.read("s_result.csv")
    try:
        archived_input_bytes.decode("utf-8", errors="strict")
        archived_result_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("ZIP 内 CSV 必须为 UTF-8 编码") from exc
    archived_input = pd.read_csv(io.BytesIO(archived_input_bytes), encoding="utf-8")
    archived_result = pd.read_csv(io.BytesIO(archived_result_bytes), encoding="utf-8")
    result_summary = validate_submission_frame(archived_result)
    input_summary = validate_submission_input(archived_input, archived_result)

    if expected_input_path is not None:
        expected_input_bytes, expected_input = _read_utf8_csv(
            expected_input_path,
            label="磁盘 input.csv",
        )
        validate_submission_input(expected_input, archived_result)
        if expected_input_bytes != archived_input_bytes:
            raise ValueError("磁盘 input.csv 与 ZIP 内 input.csv 字节不一致")
    if expected_result_path is not None:
        expected_result_bytes, expected_result = _read_utf8_csv(
            expected_result_path,
            label="磁盘 s_result.csv",
        )
        validate_submission_frame(expected_result)
        if expected_result_bytes != archived_result_bytes:
            raise ValueError("磁盘 s_result.csv 与 ZIP 内 s_result.csv 字节不一致")
        if result_freeze is not None:
            verify_submission_result_freeze(expected_result_path, result_freeze)
    elif result_freeze is not None:
        payload = _read_json(result_freeze, label="s_result 冻结记录")
        expected_hash = payload.get("sha256")
        if not isinstance(expected_hash, str) or _sha256_bytes(archived_result_bytes) != expected_hash:
            raise ValueError("ZIP 内 s_result.csv 与冻结 SHA256 不一致")
    if quality_receipt_path is not None:
        if expected_input_path is None or expected_result_path is None:
            raise ValueError("复核提交质量收据时必须同时提供磁盘 input.csv 和 s_result.csv")
        validate_submission_quality_receipt(
            expected_input_path,
            expected_result_path,
            quality_receipt_path,
        )
    return {
        "valid": True,
        "members": list(SUBMISSION_MEMBERS),
        "validation": result_summary,
        "input": input_summary,
        "input_sha256": _sha256_bytes(archived_input_bytes),
        "result_sha256": _sha256_bytes(archived_result_bytes),
    }


def package_submission(
    input_path: str | Path,
    result_path: str | Path,
    output_zip: str | Path,
    *,
    quality_policy: SubmissionQualityPolicy | None = None,
    quality_receipt_path: str | Path | Mapping[str, object] | None = None,
    result_freeze: str | Path | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """只验证并封装已冻结、已处理的 input/result；绝不调用质量变换。"""

    # 为既有 Python 调用方保留形参，但它不能再触发隐式质量处理。
    del quality_policy
    source_input = Path(input_path)
    source_result = Path(result_path)
    destination = Path(output_zip)
    if destination.resolve() in {source_input.resolve(), source_result.resolve()}:
        raise ValueError("ZIP 输出路径不能覆盖 input.csv 或 s_result.csv")

    input_bytes, input_frame = _read_utf8_csv(source_input, label="input.csv")
    result_bytes, result_frame = _read_utf8_csv(source_result, label="s_result.csv")
    result_summary = validate_submission_frame(result_frame)
    input_summary = validate_submission_input(input_frame, result_frame)
    freeze_check: dict[str, object] | None = None
    if result_freeze is not None:
        freeze_check = verify_submission_result_freeze(source_result, result_freeze)
    quality_check: dict[str, object] | None = None
    if quality_receipt_path is not None:
        quality_check = validate_submission_quality_receipt(
            source_input,
            source_result,
            quality_receipt_path,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            _zip_info("input.csv"),
            input_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
        archive.writestr(
            _zip_info("s_result.csv"),
            result_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    # 防止封装过程中源文件被并发修改而把无证据版本装入 ZIP。
    if sha256_file(source_input) != _sha256_bytes(input_bytes):
        raise ValueError("打包过程中 input.csv 发生变化")
    if sha256_file(source_result) != _sha256_bytes(result_bytes):
        raise ValueError("打包过程中 s_result.csv 发生变化")
    archive_summary = validate_submission_archive(
        destination,
        expected_input_path=source_input,
        expected_result_path=source_result,
        quality_receipt_path=quality_receipt_path,
        result_freeze=result_freeze,
    )
    summary: dict[str, object] = {
        **result_summary,
        **input_summary,
        "prediction_columns": result_summary["prediction_columns"],
        "archive_members": archive_summary["members"],
        "archive": str(destination),
        "input_sha256": _sha256_bytes(input_bytes),
        "result_sha256": _sha256_bytes(result_bytes),
        "quality_receipt_verified": quality_check is not None,
        "result_freeze_verified": freeze_check is not None,
    }
    return summary


def export_legacy_json(result_path: str | Path, output_path: str | Path) -> dict[str, object]:
    """导出数据字典旧版 columns/data JSON；不得与正式 CSV 同包提交。"""

    frame = pd.read_csv(result_path)
    summary = validate_submission_frame(frame)
    payload = {
        "columns": list(frame.columns),
        "data": frame.astype(object).where(pd.notna(frame), None).values.tolist(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    summary["json"] = str(destination)
    return summary
