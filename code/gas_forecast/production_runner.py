"""PRED-1 Gate E production runner：SAFE60 六层生产推理链（fit-once）。

依赖序（自底向上）：
  RichGas(base) → A51 splice → A60 → A61 → X3 → SAFE60

每层 fit-once 语义：在冻结 cutoff 上拟合，对评分/历史 origins 预测，
并用冻结 OOF 在历史 cutoff 上做 layer replay 验证（E1）。

本模块先实现 corrector 族通用原语 + A60 层；RichGas/A51 splice 层随后接入。
确定性层（Ridge/ARX）无随机种子；随机层（CatBoost/LGB）走 seed_contract。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gas_forecast.aggressive import project_production_predictions
from gas_forecast.config import ForecastConfig
from gas_forecast.features import build_causal_features
from gas_forecast.rich_residual import (
    RichResidualCorrector,
    RichResidualSpec,
    rich_feature_config,
)

# --- RichGas 冻结 spec（与 rich_gas_blend_30_20260803 一致） ---
RICH_GAS_TARGET = "generator_1"
RICH_GAS_GROUPS = frozenset({"gas"})
RICH_GAS_PROFILE = "all"
RICH_GAS_BLEND_WEIGHT = 0.30
RICH_GAS_PARENT_COLUMN = "aggressive_r75_lgb20_pred"
RICH_GAS_OUTPUT_COLUMN = "rich_gas_blend_30_pred"
RICH_GAS_SPEC = RichResidualSpec(
    name="rich_gas",
    target=RICH_GAS_TARGET,
    feature_groups=RICH_GAS_GROUPS,
    feature_profile=RICH_GAS_PROFILE,
    include_champion_prediction=False,
    min_train_rows=256,
    blend_weights=(RICH_GAS_BLEND_WEIGHT,),
)

# --- A51 冻结 spec（RESULTS_REPORT A51 splice） ---
A51_TARGET = "generator_1"
A51_GROUPS = frozenset({"quantile", "ramp", "gas"})
A51_PROFILE = "long_horizon"
A51_ACTIVE_HORIZONS = (75, 90, 105, 120)
A51_BLEND_WEIGHT = 0.30
# A51 corrector 的残差 baseline 是 aggressive（candidate spec: baseline_column=aggressive_r75_lgb20_pred）；
# 其输出 rich_g1_long_blend_30_pred 是 aggressive-based，再被 splice 用于 long 步长。
A51_PARENT_COLUMN = "aggressive_r75_lgb20_pred"
A51_OUTPUT_COLUMN = "rich_g1_long_blend_30_pred"
A51_SPEC = RichResidualSpec(
    name="a51_g1_long",
    target=A51_TARGET,
    feature_groups=A51_GROUPS,
    feature_profile=A51_PROFILE,
    active_horizons=A51_ACTIVE_HORIZONS,
    include_champion_prediction=True,
    min_train_rows=256,
    blend_weights=(A51_BLEND_WEIGHT,),
)

# splice 输出列
SPLICE_OUTPUT_COLUMN = "rich_short00_long100_pred"
SHORT_HORIZONS = (15, 30, 45, 60)
LONG_HORIZONS = (75, 90, 105, 120)


def build_rich_gas_production_predictions(
    frame: pd.DataFrame,
    oof_rows: pd.DataFrame,
    *,
    config: ForecastConfig,
    cutoff: pd.Timestamp,
    scoring_rows: pd.DataFrame,
    price_schedule=None,
    fold_label: str = "production",
) -> CorrectedLayerResult:
    """RichGas fit-once production：g1 corrector + 30% blend with aggressive。

    scoring_rows 须含 RICH_GAS_PARENT_COLUMN（aggressive）在评分 origins 的预测。
    仅 g1 行被修正；gall 行保持 aggressive。
    """
    rich_config = rich_feature_config(
        config, RICH_GAS_SPEC.feature_groups, feature_profile=RICH_GAS_SPEC.feature_profile
    )
    features = build_causal_features(frame, rich_config.feature, price_schedule)
    corrector, fit_receipt = fit_rich_residual_corrector_production(
        features,
        oof_rows,
        config=rich_config,
        spec=RICH_GAS_SPEC,
        cutoff=cutoff,
        baseline_column=RICH_GAS_PARENT_COLUMN,
    )
    result = predict_rich_residual_blend_production(
        corrector,
        features,
        scoring_rows,
        baseline_column=RICH_GAS_PARENT_COLUMN,
        blend_weight=RICH_GAS_BLEND_WEIGHT,
        residual_column="rich_gas_residual_raw_pred",
        blend_raw_column="rich_gas_blend_30_raw_pred",
        blend_column=RICH_GAS_OUTPUT_COLUMN,
        fold_label=fold_label,
    )
    result.receipts.update(
        {"fit": fit_receipt, "spec": RICH_GAS_SPEC.name, "feature_columns": int(len(features.columns))}
    )
    return result


def build_a51_production_predictions(
    frame: pd.DataFrame,
    oof_rows: pd.DataFrame,
    *,
    config: ForecastConfig,
    cutoff: pd.Timestamp,
    scoring_rows: pd.DataFrame,
    price_schedule=None,
    fold_label: str = "production",
) -> CorrectedLayerResult:
    """A51 fit-once production：g1-long corrector + 30% blend with rich_gas_blend_30。

    scoring_rows 须含 A51_PARENT_COLUMN（rich_gas_blend_30_pred）在评分 origins。
    仅 g1-long 行被修正；其他行保持 parent。
    """
    rich_config = rich_feature_config(
        config, A51_SPEC.feature_groups, feature_profile=A51_SPEC.feature_profile
    )
    features = build_causal_features(frame, rich_config.feature, price_schedule)
    corrector, fit_receipt = fit_rich_residual_corrector_production(
        features,
        oof_rows,
        config=rich_config,
        spec=A51_SPEC,
        cutoff=cutoff,
        baseline_column=A51_PARENT_COLUMN,
    )
    result = predict_rich_residual_blend_production(
        corrector,
        features,
        scoring_rows,
        baseline_column=A51_PARENT_COLUMN,
        blend_weight=A51_BLEND_WEIGHT,
        residual_column="a51_g1_long_residual_raw_pred",
        blend_raw_column="a51_g1_long_blend_30_raw_pred",
        blend_column=A51_OUTPUT_COLUMN,
        fold_label=fold_label,
    )
    result.receipts.update(
        {"fit": fit_receipt, "spec": A51_SPEC.name, "feature_columns": int(len(features.columns))}
    )
    return result


def apply_short_long_splice(
    rows: pd.DataFrame,
    *,
    baseline_column: str,
    branch_column: str,
    short_weight: float = 0.0,
    long_weight: float = 1.0,
    output_column: str = SPLICE_OUTPUT_COLUMN,
) -> pd.DataFrame:
    """确定性拼接（与 horizon_blend._candidate_values 一致）。

    仅 generator_1 行参与：short(15-60) 用 baseline，long(75-120) 用 branch；
    generator_all 行全部保持 baseline。
    """
    work = rows.copy()
    baseline = work[baseline_column].to_numpy(dtype=float)
    branch = work[branch_column].to_numpy(dtype=float)
    g1 = work["target"].eq("generator_1").to_numpy(dtype=bool)
    short = work["horizon"].isin(SHORT_HORIZONS).to_numpy(dtype=bool)
    weights = np.where(short, short_weight, long_weight)
    weights = np.where(g1, weights, 0.0)
    work[output_column] = baseline + weights * (branch - baseline)
    # 与 OOF 一致：splice 输出经容量投影后落盘。
    work = project_production_predictions(work, output_column, output_column=output_column)
    return work
A60_TARGET = "generator_all"
A60_ACTIVE_HORIZONS = (75, 90, 105, 120)
A60_BLEND_WEIGHT = 0.30
A60_MIN_TRAIN_ROWS = 256
A60_FEATURE_GROUPS = frozenset({"quantile", "ramp", "gas"})
A60_PARENT_COLUMN = "rich_short00_long100_pred"
A60_RICH_GAS_COLUMN = "rich_gas_blend_30_pred"
A60_SPEC = RichResidualSpec(
    name="a60_gall_long",
    target=A60_TARGET,
    feature_groups=A60_FEATURE_GROUPS,
    feature_profile="long_horizon",
    active_horizons=A60_ACTIVE_HORIZONS,
    include_champion_prediction=True,
    min_train_rows=A60_MIN_TRAIN_ROWS,
    blend_weights=(A60_BLEND_WEIGHT,),
)


@dataclass
class CorrectedLayerResult:
    rows: pd.DataFrame
    receipts: dict[str, object]


def _mask_history(oof_rows: pd.DataFrame, *, target: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """corrector 训练 mask：只取目标行且 ``origin_time <= cutoff``。"""
    return oof_rows.loc[
        oof_rows["target"].eq(target) & (pd.to_datetime(oof_rows["origin_time"]) <= cutoff)
    ].copy()


def fit_rich_residual_corrector_production(
    features: pd.DataFrame,
    oof_rows: pd.DataFrame,
    *,
    config: ForecastConfig,
    spec: RichResidualSpec,
    cutoff: pd.Timestamp,
    baseline_column: str,
) -> tuple[RichResidualCorrector | None, dict[str, object]]:
    """在单个冻结 cutoff 上拟合一个 RichResidual corrector（fit-once / replay 共用）。

    与 OOF builder 一致：history 为空（无 ``origin <= cutoff`` 的目标行）时
    返回 ``None``（调用方回退到 baseline），不抛错。
    """
    history = _mask_history(oof_rows, target=spec.target, cutoff=cutoff)
    if history.empty:
        return None, {"target": spec.target, "cutoff": str(cutoff), "history_rows": 0, "fallback": True}
    corrector = RichResidualCorrector(config, spec).fit(
        features,
        history,
        baseline_column=baseline_column,
    )
    receipt = {
        "target": spec.target,
        "cutoff": str(cutoff),
        "history_rows": int(len(history)),
        "trained_horizons": sorted(corrector.states_),
        "fallback": False,
    }
    return corrector, receipt


def predict_rich_residual_blend_production(
    corrector: RichResidualCorrector,
    features: pd.DataFrame,
    scoring_rows: pd.DataFrame,
    *,
    baseline_column: str,
    blend_weight: float,
    residual_column: str,
    blend_raw_column: str,
    blend_column: str,
    fold_label: str = "production",
) -> CorrectedLayerResult:
    """对 scoring rows 预测 corrector 残差 + 固定 blend + 容量投影。

    与 OOF 一致：corrected_raw = parent + correction；blend_raw = (1-w)*parent +
    w*corrected_projected；最后再投影 blend。scoring_rows 需含 baseline_column。
    """
    work = scoring_rows.copy()
    work["fold"] = fold_label
    if corrector is None:
        # fallback（无 history）：blend = parent，与 OOF builder 一致。
        work[residual_column] = work[baseline_column].to_numpy(dtype=float)
    else:
        work[residual_column] = corrector.predict_long(
            features, work, baseline_column=baseline_column
        )
    work = project_production_predictions(
        work, residual_column, output_column=f"{residual_column}_proj"
    )
    parent_proj = project_production_predictions(
        work, baseline_column, output_column=f"{baseline_column}_proj"
    )
    work[blend_raw_column] = (
        (1.0 - blend_weight) * parent_proj[f"{baseline_column}_proj"].to_numpy(dtype=float)
        + blend_weight * work[f"{residual_column}_proj"].to_numpy(dtype=float)
    )
    work = project_production_predictions(work, blend_raw_column, output_column=blend_column)
    receipt = {
        "blend_weight": blend_weight,
        "baseline_column": baseline_column,
        "residual_column": residual_column,
        "blend_column": blend_column,
    }
    return CorrectedLayerResult(rows=work, receipts=receipt)


def build_a60_production_predictions(
    frame: pd.DataFrame,
    oof_rows: pd.DataFrame,
    *,
    config: ForecastConfig,
    cutoff: pd.Timestamp,
    scoring_rows: pd.DataFrame,
    price_schedule=None,
    fold_label: str = "production",
) -> CorrectedLayerResult:
    """A60 fit-once production：corrector + 30% blend，输出 a60_gall_long_blend_30_pred。

    ``scoring_rows`` 须含 A60_PARENT_COLUMN（rich_short00_long100_pred）在评分
    origins 的预测（由上游层自底向上计算；replay 用冻结 A60 OOF 该列）。
    """
    rich_config = rich_feature_config(
        config, A60_SPEC.feature_groups, feature_profile=A60_SPEC.feature_profile
    )
    features = build_causal_features(frame, rich_config.feature, price_schedule)
    corrector, fit_receipt = fit_rich_residual_corrector_production(
        features,
        oof_rows,
        config=rich_config,
        spec=A60_SPEC,
        cutoff=cutoff,
        baseline_column=A60_PARENT_COLUMN,
    )
    result = predict_rich_residual_blend_production(
        corrector,
        features,
        scoring_rows,
        baseline_column=A60_PARENT_COLUMN,
        blend_weight=A60_BLEND_WEIGHT,
        residual_column="a60_gall_long_residual_raw_pred",
        blend_raw_column="a60_gall_long_blend_30_raw_pred",
        blend_column="a60_gall_long_blend_30_pred",
        fold_label=fold_label,
    )
    result.receipts.update(
        {"fit": fit_receipt, "spec": A60_SPEC.name, "feature_columns": int(len(features.columns))}
    )
    return result
