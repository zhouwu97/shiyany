"""评分期未来行重建 Oracle。

此模块故意保留一个可用于诊断的“未来行”重建器。它读取评分期未来行，
因此不是因果模型，不能用于正式训练、选型、生产门禁或提交。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from gas_forecast.config import ForecastConfig
from gas_forecast.submission import expected_prediction_columns


def _linear_pipeline() -> Pipeline:
    """返回可序列化的单变量同刻重建管线。"""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("linear", LinearRegression()),
        ]
    )


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.maximum(np.abs(actual), 1e-6)
    return float(np.mean(np.abs(actual - predicted) / denominator))


@dataclass(frozen=True)
class ReconstructionState:
    """一个目标的全量模型和前向验证回执。"""

    model: Pipeline
    training_rows: int
    validation_rows: int
    validation_mape: float
    coefficient: float
    intercept: float


class FutureRowReconstructionForecaster:
    """把目标时刻评分输入映射为多步预测的非因果诊断 Oracle。

    该模型显式使用完整评分输入中的目标时刻行，绝不能被当作因果滚动模型。
    它仍然通过历史训练标签拟合并保存 sklearn 管线；外部参考答案不进入
    ``fit``。类级硬标识供通用生产代码和人工审计拒绝该候选。
    """

    version = "future_row_reconstruction"
    # 这些字段是持久化产物契约的一部分，不得改成“正式候选”语义。
    oracle_candidate = True
    oracle_only = True
    diagnostic_only = True
    causal = False
    formal_candidate = False
    production_candidate = False
    deployable = False
    research_only = True
    oracle_reason = "读取评分期未来行，违反 origin 时点因果边界"

    def __init__(
        self,
        config: ForecastConfig | None = None,
        *,
        validation_rows: int = 672,
    ) -> None:
        if validation_rows < 96:
            raise ValueError("前向验证至少需要 96 行")
        self.config = config or ForecastConfig()
        self.validation_rows = int(validation_rows)
        self.states_: dict[str, ReconstructionState] = {}

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, targets: tuple[str, ...]) -> None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("重建模型输入必须使用 DatetimeIndex")
        if frame.empty or frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
            raise ValueError("重建模型输入时间轴必须非空、唯一且递增")
        missing = sorted(set(targets).difference(frame.columns))
        if missing:
            raise ValueError(f"重建模型输入缺少目标字段: {missing}")

    def fit(self, training_frame: pd.DataFrame) -> "FutureRowReconstructionForecaster":
        """使用历史同刻输入和标签训练，每个目标保留末段前向验证。"""

        self._validate_frame(training_frame, self.config.targets)
        self.states_.clear()
        for target in self.config.targets:
            values = pd.to_numeric(training_frame[target], errors="coerce")
            valid = values.notna() & np.isfinite(values.to_numpy(dtype=float))
            positions = np.flatnonzero(valid.to_numpy())
            if len(positions) < self.validation_rows + 200:
                raise ValueError(f"{target} 的有效训练数据不足")
            split = len(positions) - self.validation_rows
            development_positions = positions[:split]
            validation_positions = positions[split:]

            development = _linear_pipeline()
            development.fit(
                training_frame.iloc[development_positions][[target]],
                values.iloc[development_positions],
            )
            validation_actual = values.iloc[validation_positions].to_numpy(dtype=float)
            validation_prediction = development.predict(
                training_frame.iloc[validation_positions][[target]]
            )

            final_model = _linear_pipeline()
            final_model.fit(training_frame.loc[valid, [target]], values.loc[valid])
            linear = final_model.named_steps["linear"]
            self.states_[target] = ReconstructionState(
                model=final_model,
                training_rows=int(valid.sum()),
                validation_rows=len(validation_positions),
                validation_mape=_mape(validation_actual, validation_prediction),
                coefficient=float(np.ravel(linear.coef_)[0]),
                intercept=float(linear.intercept_),
            )
        return self

    def training_report(self) -> dict[str, object]:
        if set(self.states_) != set(self.config.targets):
            raise RuntimeError("重建模型尚未完成训练")
        return {
            "version": self.version,
            "oracle_candidate": self.oracle_candidate,
            "oracle_only": self.oracle_only,
            "diagnostic_only": self.diagnostic_only,
            "causal": self.causal,
            "formal_candidate": self.formal_candidate,
            "deployable": self.deployable,
            "production_candidate": self.production_candidate,
            "research_only": self.research_only,
            "uses_future_rows": True,
            "production_path_allowed": False,
            "selection_allowed": False,
            "weights_allowed": False,
            "thresholds_allowed": False,
            "oracle_reason": self.oracle_reason,
            "validation_policy": "last_rows_forward_holdout",
            "validation_rows": self.validation_rows,
            "targets": {
                target: {
                    "training_rows": state.training_rows,
                    "validation_rows": state.validation_rows,
                    "validation_mape": state.validation_mape,
                    "coefficient": state.coefficient,
                    "intercept": state.intercept,
                }
                for target, state in self.states_.items()
            },
        }

    def reconstruct_rows(
        self,
        scoring_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """重建评分期每行目标；内部缺失按完整评分序列线性插值。"""

        self._validate_frame(scoring_frame, self.config.targets)
        if set(self.states_) != set(self.config.targets):
            raise RuntimeError("重建模型尚未完成训练")
        reconstructed = pd.DataFrame(index=scoring_frame.index)
        missing: dict[str, int] = {}
        for target in self.config.targets:
            observed = pd.to_numeric(scoring_frame[target], errors="coerce")
            missing[target] = int(observed.isna().sum())
            filled = observed.interpolate(method="linear", limit_direction="both")
            if filled.isna().any() or not np.isfinite(filled.to_numpy(dtype=float)).all():
                raise ValueError(f"评分期 {target} 无法完成线性插值")
            reconstructed[target] = self.states_[target].model.predict(
                filled.to_frame(name=target)
            )
        return reconstructed, {
            "scoring_rows": len(scoring_frame),
            "missing_before_interpolation": missing,
            "interpolation": "linear_both",
        }

    def predict(
        self,
        scoring_frame: pd.DataFrame,
        base_predictions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """覆盖评分输入可达的目标时刻，越界单元保留基础模型预测。

        该签名只接受 ``scoring_frame`` 和完整的基础预测表。生产模型通用接口
        通常传入因果特征与当前值表（第二个表只有目标列），会在字段校验处
        失败，从而避免该 Oracle 被生产门禁误当成正式模型。
        """

        if not base_predictions.index.equals(scoring_frame.index):
            raise ValueError("基础预测与评分输入必须使用相同时间索引")
        expected = expected_prediction_columns(self.config)
        if list(base_predictions.columns) != expected:
            raise ValueError(
                "ORACLE/DIAGNOSTIC ONLY：未来行重建不能接入因果生产预测接口；"
                "基础预测字段或顺序不符合研究调用契约"
            )
        output = base_predictions.copy()
        reconstructed, row_report = self.reconstruct_rows(scoring_frame)
        overwritten = 0
        fallback = 0
        for origin in scoring_frame.index:
            for horizon in self.config.feature.horizons:
                target_time = origin + pd.Timedelta(minutes=15 * horizon)
                for target in self.config.targets:
                    column = f"{target}_t+{15 * horizon}_pred"
                    if target_time in reconstructed.index:
                        output.loc[origin, column] = reconstructed.loc[target_time, target]
                        overwritten += 1
                    else:
                        fallback += 1

        generator_1_columns = [
            f"generator_1_t+{15 * horizon}_pred" for horizon in self.config.feature.horizons
        ]
        generator_all_columns = [
            f"generator_all_t+{15 * horizon}_pred" for horizon in self.config.feature.horizons
        ]
        output.loc[:, generator_1_columns] = output.loc[:, generator_1_columns].clip(0.0, 200.0)
        output.loc[:, generator_all_columns] = output.loc[:, generator_all_columns].clip(0.0, 440.0)
        for generator_1, generator_all in zip(
            generator_1_columns, generator_all_columns, strict=True
        ):
            output[generator_all] = np.maximum(output[generator_all], output[generator_1])
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("未来行重建结果含非有限值")
        total = overwritten + fallback
        return output, {
            **row_report,
            "prediction_cells": total,
            "reconstructed_cells": overwritten,
            "base_fallback_cells": fallback,
            "reconstruction_ratio": overwritten / total,
        }
