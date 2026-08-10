"""X1 Dynamic Expected-Error Router：按历史 cross-fit 的期望误差动态路由。

候选池（7 个）：

* ``a61_parent``（A61 Recursive ARX parent）
* ``p3_static``（P3 静态融合 prediction 列）
* ``x3_catboost``（A57 全矩阵残差 CatBoost）
* ``a64_direct_delta``（A64 Direct Delta）
* ``p1_causal_rolling``（P1 Causal Rolling）
* ``p2_historical_analog``（P2 Historical Analog）
* ``p2_matured_residual``（P2 Matured Residual）

路由协议（严格因果，绝不读取 held fold 的 actual 或未来折）：

1. 对每个候选 ``c``、每个 held fold ``f``，只用 ``fold < f`` 的行训练期望误差
   模型 ``expected_error_c(X_t, target, horizon)``（LightGBM 小树，特征仅含
   各候选预测值、候选间分歧、target/horizon/时间编码）；
2. 对 held fold 的每个单元格预测 7 个期望误差；
3. 置信度不足（best 相对 A61 的期望改善低于阈值）或 best 就是 A61 →
   输出 A61；
4. 置信度足够 → soft blend top2（按期望误差反比加权）。

早期折（无足够历史训练数据）回落 A61 并登记 ``insufficient_history``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from gas_forecast.causal_trajectory_ensemble import (
    IDENTITY_COLUMNS,
    TARGETS,
)
from gas_forecast.scoring import competition_mape


ROUTER_CANDIDATES: tuple[str, ...] = (
    "a61_parent",
    "p3_static",
    "x3_catboost",
    "a64_direct_delta",
    "p1_causal_rolling",
    "p2_historical_analog",
    "p2_matured_residual",
)

DEFAULT_CANDIDATE_COLUMNS: dict[str, str] = {
    "a61_parent": "a61_parent__prediction",
    "p3_static": "prediction",
    "x3_catboost": "x3_catboost__prediction",
    "a64_direct_delta": "a64_direct_delta__prediction",
    "p1_causal_rolling": "p1_causal_rolling__prediction",
    "p2_historical_analog": "p2_historical_analog__prediction",
    "p2_matured_residual": "p2_matured_residual__prediction",
}

PARENT_CANDIDATE = "a61_parent"
DEFAULT_CONFIDENCE_MIN_PP = 0.01
DEFAULT_MIN_HISTORY_FOLDS = 3
EXPECTED_FOLDS = 19
EXPECTED_ORIGINS = 3_648
EXPECTED_ROWS = 58_368

# X1 晋级门禁（与 P4 一致的预注册稳定门槛）。
X1_MIN_IMPROVEMENT_PP = 0.02
X1_MIN_RECENT5_WINS = 3
X1_MAX_WORST_FOLD_REGRESSION_PP = 0.10
X1_MAX_TARGET_REGRESSION_PP = 0.10


@dataclass(frozen=True)
class ExpectedErrorModel:
    """单个候选的期望误差 LightGBM 回归器。"""

    candidate: str
    model: object
    trained_folds: tuple[str, ...]
    training_rows: int
    prior_error: float
    feature_columns: tuple[str, ...]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """返回剪裁到 [0, 1] 的期望绝对百分比误差。"""

        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"期望误差特征缺少字段: {missing}")
        values = self.model.predict(features.loc[:, list(self.feature_columns)])
        return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


@dataclass(frozen=True)
class X1RouterResult:
    """X1 路由后的 OOF 长表与收据。"""

    rows: pd.DataFrame
    report: dict[str, object]
    fold_selections: pd.DataFrame
    coverage: pd.DataFrame


def load_x1_oof(
    integration_oof: str | Path,
    *,
    x3_oof: str | Path | None = None,
    x3_column: str = "a57b_residual_a51_cat10_pred",
    candidate_columns: Mapping[str, str] = DEFAULT_CANDIDATE_COLUMNS,
    expected_folds: int = EXPECTED_FOLDS,
    expected_origins: int = EXPECTED_ORIGINS,
    expected_rows: int = EXPECTED_ROWS,
    current_value_oof: str | Path | None = None,
) -> pd.DataFrame:
    """读取 P3 集成 OOF，可选合并 X3（A57）CatBoost 列，返回规范化长表。

    X3 列默认取 A57 全矩阵残差 CatBoost（``a57b_residual_a51_cat10_pred``）；
    若调用方持有其他 X3 OOF，可通过 ``x3_oof`` + ``x3_column`` 传入。所有
    键（fold/origin_time/train_end/target/horizon）必须与 P3 OOF 完全对齐。

    ``current_value_oof`` 提供每个 (fold, origin_time, target) 的当前观测值
    （P1 CausalRolling OOF 的 ``current_value`` 列），作为单元格级路由的
    合法因果特征；缺失时用训练折中位数兜底。
    """

    source = Path(integration_oof)
    if source.suffix.lower() == ".parquet":
        rows = pd.read_parquet(source)
    else:
        rows = pd.read_csv(source)
    rows["fold"] = rows["fold"].astype(str)
    rows["origin_time"] = pd.to_datetime(rows["origin_time"])
    rows["train_end"] = pd.to_datetime(rows["train_end"])
    rows["target"] = rows["target"].astype(str)
    rows["horizon"] = rows["horizon"].astype(int)

    if rows.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("X1 OOF 身份键不唯一")
    actual_folds = rows["fold"].nunique()
    actual_origins = rows["origin_time"].nunique()
    if actual_folds != expected_folds or actual_origins != expected_origins or len(rows) != expected_rows:
        raise ValueError(
            f"X1 OOF 尺寸不符: folds={actual_folds} origins={actual_origins} rows={len(rows)}"
        )

    output = rows.copy()
    base_columns = [
        column for column in candidate_columns.values() if column != "x3_catboost__prediction"
    ]
    for column in base_columns:
        if column not in output.columns:
            raise ValueError(f"候选列 {column} 不在集成 OOF 中")
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if not np.isfinite(output.loc[:, base_columns].to_numpy(dtype=float)).all():
        raise ValueError("集成 OOF 候选预测含 NaN/Inf")

    if x3_oof is not None:
        x3_path = Path(x3_oof)
        x3 = pd.read_csv(x3_path)
        x3["fold"] = x3["fold"].astype(str)
        x3["origin_time"] = pd.to_datetime(x3["origin_time"])
        x3["train_end"] = pd.to_datetime(x3["train_end"])
        x3["target"] = x3["target"].astype(str)
        x3["horizon"] = x3["horizon"].astype(int)
        if x3_column not in x3.columns:
            raise ValueError(f"X3 OOF 缺少列 {x3_column}")
        x3_keys = list(IDENTITY_COLUMNS)
        x3 = x3.loc[:, [*x3_keys, x3_column]].rename(columns={x3_column: "x3_catboost__prediction"})
        merged = output.merge(x3, on=x3_keys, how="left", validate="one_to_one")
        if merged["x3_catboost__prediction"].isna().any():
            raise ValueError("X3 OOF 与集成 OOF 键未完全对齐")
        output = merged
        output["x3_catboost__prediction"] = pd.to_numeric(
            output["x3_catboost__prediction"], errors="coerce"
        )
    if "x3_catboost__prediction" not in output.columns:
        raise ValueError("缺少 X3 候选列；请提供 --x3-oof 或使用含 x3_catboost__prediction 的 OOF")

    if current_value_oof is not None:
        cv_path = Path(current_value_oof)
        cv = pd.read_csv(cv_path)
        cv["fold"] = cv["fold"].astype(str)
        cv["origin_time"] = pd.to_datetime(cv["origin_time"])
        cv["target"] = cv["target"].astype(str)
        if "current_value" not in cv.columns:
            raise ValueError("current_value OOF 缺少 current_value 列")
        cv = cv.loc[:, ["fold", "origin_time", "target", "current_value"]].drop_duplicates(
            ["fold", "origin_time", "target"], keep="first"
        )
        merged = output.merge(cv, on=["fold", "origin_time", "target"], how="left")
        output = merged
        output["current_value"] = pd.to_numeric(output["current_value"], errors="coerce")

    output = output.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    output["actual"] = pd.to_numeric(output["actual"], errors="coerce")
    if output["actual"].isna().any():
        raise ValueError("OOF 含缺失 actual")
    return output


def _chronological_folds(rows: pd.DataFrame) -> list[str]:
    """按每折最早 origin_time 排序的折名列表（与 P3/P4 一致）。"""

    order = rows.groupby("fold", sort=False)["origin_time"].min().sort_values()
    return [str(name) for name in order.index]


def _candidate_matrix(rows: pd.DataFrame, candidates: Sequence[str]) -> np.ndarray:
    columns = [DEFAULT_CANDIDATE_COLUMNS[name] for name in candidates]
    return rows.loc[:, columns].to_numpy(dtype=float)


def build_router_features(
    rows: pd.DataFrame,
    *,
    error_priors: Mapping[str, Mapping[str, Mapping[int, float]]] | None = None,
) -> pd.DataFrame:
    """构造因果特征矩阵。

    只含单元格预测值、分歧、时间/目标/步长编码，以及可选的候选历史误差
    先验（由训练折同 (target, horizon) 的 MAPE 提供）。本函数不读取任何
    ``actual`` 或未来行；期望误差目标在 :func:`fit_expected_error_models`
    内部单独构造。
    """

    candidates = list(ROUTER_CANDIDATES)
    columns = [DEFAULT_CANDIDATE_COLUMNS[name] for name in candidates]
    features = rows.loc[:, columns].copy()
    features.columns = [f"pred_{name}" for name in candidates]

    matrix = features.to_numpy(dtype=float)
    features["disagreement_range"] = np.ptp(matrix, axis=1)
    features["disagreement_std"] = matrix.std(axis=1)
    for name in candidates:
        index = candidates.index(name)
        features[f"disagreement_vs_{name}"] = np.abs(
            matrix - matrix[:, index, None]
        ).mean(axis=1)

    if error_priors is not None:
        for name in candidates:
            prior_by_target = error_priors[name]
            features[f"prior_error_{name}"] = [
                prior_by_target.get(str(target), {}).get(int(horizon), float("nan"))
                for target, horizon in zip(rows["target"], rows["horizon"])
            ]

    if "current_value" in rows.columns:
        current = pd.to_numeric(rows["current_value"], errors="coerce").to_numpy(dtype=float)
        features["current_value"] = np.where(np.isfinite(current), current, np.nanmedian(current))
        for name in candidates:
            predicted = rows[DEFAULT_CANDIDATE_COLUMNS[name]].to_numpy(dtype=float)
            features[f"pred_vs_current_{name}"] = predicted - features["current_value"]
        features["current_to_parent"] = (
            features["current_value"]
            - rows[DEFAULT_CANDIDATE_COLUMNS[PARENT_CANDIDATE]].to_numpy(dtype=float)
        )

    features["target_generator_1"] = (rows["target"] == "generator_1").astype("int8")
    features["horizon"] = rows["horizon"].astype(float)
    features["horizon_15"] = (rows["horizon"] == 15).astype("int8")
    features["horizon_30"] = (rows["horizon"] == 30).astype("int8")
    features["horizon_45"] = (rows["horizon"] == 45).astype("int8")
    features["horizon_60"] = (rows["horizon"] == 60).astype("int8")
    features["horizon_75"] = (rows["horizon"] == 75).astype("int8")
    features["horizon_90"] = (rows["horizon"] == 90).astype("int8")
    features["horizon_105"] = (rows["horizon"] == 105).astype("int8")
    features["horizon_120"] = (rows["horizon"] == 120).astype("int8")
    features["hour"] = rows["origin_time"].dt.hour.astype(float)
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24.0)
    features["day_of_week"] = rows["origin_time"].dt.dayofweek.astype(float)
    return features


def _error_prior_map(
    train_rows: pd.DataFrame,
    candidates: Sequence[str],
) -> dict[str, dict[str, dict[int, float]]]:
    """训练折上每 (target, horizon) 每候选的 MAPE 先验映射。"""

    priors: dict[str, dict[str, dict[int, float]]] = {}
    for candidate in candidates:
        column = DEFAULT_CANDIDATE_COLUMNS[candidate]
        by_target: dict[str, dict[int, float]] = {}
        for (target, horizon), part in train_rows.groupby(["target", "horizon"]):
            by_target.setdefault(str(target), {})[int(horizon)] = float(
                competition_mape(part["actual"], part[column])
            )
        priors[candidate] = by_target
    return priors


def _prior_error(train_rows: pd.DataFrame, candidate: str, target: str, horizon: int) -> float:
    """训练折中同 (target, horizon) 的候选平均绝对百分比误差。"""

    part = train_rows[(train_rows["target"] == target) & (train_rows["horizon"] == horizon)]
    if part.empty:
        return float("nan")
    column = DEFAULT_CANDIDATE_COLUMNS[candidate]
    actual = part["actual"].to_numpy(dtype=float)
    predicted = part[column].to_numpy(dtype=float)
    return float(competition_mape(actual, predicted))


def fit_expected_error_models(
    train_rows: pd.DataFrame,
    *,
    min_history_folds: int = DEFAULT_MIN_HISTORY_FOLDS,
    candidates: Sequence[str] = ROUTER_CANDIDATES,
    seed: int = 42,
) -> dict[str, ExpectedErrorModel]:
    """在历史折上为每个候选训练期望误差模型。

    ``train_rows`` 必须是 held fold 之前的全部行。目标为逐单元格
    ``|actual - pred| / (|actual| + 1e-6)``；使用 LightGBM 小树防止过拟合。
    若历史折数不足 ``min_history_folds``，返回带 ``prior_error`` 的
    ``insufficient_history`` 标记模型（由调用方回退 A61）。
    """

    from lightgbm import LGBMRegressor

    if train_rows.empty:
        raise ValueError("历史折为空，无法训练期望误差模型")
    train_folds = _chronological_folds(train_rows)
    if len(train_folds) < min_history_folds:
        raise ValueError(
            f"历史折不足 {min_history_folds}（实际 {len(train_folds)}），调用方应回退 A61"
        )

    features = build_router_features(
        train_rows,
        error_priors=_error_prior_map(train_rows, candidates),
    )
    feature_columns = tuple(features.columns)
    models: dict[str, ExpectedErrorModel] = {}
    for candidate in candidates:
        column = DEFAULT_CANDIDATE_COLUMNS[candidate]
        actual = train_rows["actual"].to_numpy(dtype=float)
        predicted = train_rows[column].to_numpy(dtype=float)
        target = np.abs(actual - predicted) / (np.abs(actual) + 1e-6)
        if not np.isfinite(target).all():
            raise ValueError(f"候选 {candidate} 期望误差目标含非有限值")
        model = LGBMRegressor(
            n_estimators=200,
            max_depth=4,
            num_leaves=31,
            learning_rate=0.04,
            min_child_samples=48,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(features, target)
        prior = float(np.mean(target))
        models[candidate] = ExpectedErrorModel(
            candidate=candidate,
            model=model,
            trained_folds=tuple(train_folds),
            training_rows=int(len(train_rows)),
            prior_error=prior,
            feature_columns=feature_columns,
        )
    return models


def _soft_blend_top2(
    expected: dict[str, float],
    candidates: Sequence[str],
) -> tuple[np.ndarray, list[str], list[float]]:
    """按期望误差反比加权 top2；返回归一化权重与对应候选。"""

    ordered = sorted(candidates, key=lambda name: expected[name])
    first, second = ordered[0], ordered[1]
    e1, e2 = float(expected[first]), float(expected[second])
    if e1 <= 0.0 or e2 <= 0.0 or not np.isfinite(e1) or not np.isfinite(e2):
        return np.array([1.0, 0.0]), [first, second], [1.0, 0.0]
    w1 = e2 / (e1 + e2)
    w2 = e1 / (e1 + e2)
    return np.array([w1, w2]), [first, second], [w1, w2]


def route_expected_error(
    rows: pd.DataFrame,
    *,
    confidence_min_pp: float = DEFAULT_CONFIDENCE_MIN_PP,
    min_history_folds: int = DEFAULT_MIN_HISTORY_FOLDS,
    candidates: Sequence[str] = ROUTER_CANDIDATES,
    seed: int = 42,
    mode: str = "prior",
    blend_top: int = 2,
) -> X1RouterResult:
    """逐 held fold 训练期望误差模型并路由整个 OOF。

    每个 held fold 的模型只由时间上更早的折训练（``fold < held``），
    绝不使用 held fold 的 actual 或未来折。历史折不足的早期折回退 A61。

    ``mode``：

    * ``prior``（默认）：期望误差 = 历史折同 (target, horizon) 的候选 MAPE
      先验，无 ML、零过拟合；
    * ``lightgbm``：单元格级 LightGBM 期望误差回归（特征含预测值、分歧、
      时间与先验）。

    ``blend_top`` 决定置信度足够时的 soft blend 候选数（默认 2）。
    """

    if confidence_min_pp < 0.0:
        raise ValueError("置信度阈值必须为非负")
    if blend_top < 1:
        raise ValueError("blend_top 必须至少为 1")
    folds = _chronological_folds(rows)
    output = rows.copy()
    output["x1_prediction"] = np.nan
    output["x1_selected"] = ""
    output["x1_reason"] = ""
    output["x1_expected_errors"] = None
    output["x1_weight_1"] = np.nan
    output["x1_weight_2"] = np.nan

    selection_records: list[dict[str, object]] = []
    coverage_records: list[dict[str, object]] = []
    for index, held_fold in enumerate(folds):
        held_mask = rows["fold"].astype(str).eq(held_fold)
        held_rows = rows.loc[held_mask]
        history_rows = rows.loc[rows["fold"].astype(str).isin(folds[:index])]
        if len(_chronological_folds(history_rows)) < min_history_folds:
            output.loc[held_mask, "x1_prediction"] = rows.loc[held_mask, DEFAULT_CANDIDATE_COLUMNS[PARENT_CANDIDATE]].to_numpy(dtype=float)
            output.loc[held_mask, "x1_selected"] = PARENT_CANDIDATE
            output.loc[held_mask, "x1_reason"] = "insufficient_history"
            output.loc[held_mask, "x1_weight_1"] = 1.0
            output.loc[held_mask, "x1_weight_2"] = 0.0
            coverage_records.append(
                {
                    "held_fold": held_fold,
                    "rows": int(len(held_rows)),
                    "routed": 0,
                    "fallback_insufficient_history": int(len(held_rows)),
                    "fallback_confidence": 0,
                }
            )
            continue

        priors = _error_prior_map(history_rows, candidates)
        if mode == "lightgbm":
            models = fit_expected_error_models(
                history_rows,
                min_history_folds=min_history_folds,
                candidates=candidates,
                seed=seed,
            )
            features = build_router_features(held_rows, error_priors=priors)
            expected_matrix = {
                name: model.predict(features) for name, model in models.items()
            }
        elif mode == "prior":
            expected_matrix = {
                name: np.asarray(
                    [
                        priors[name][str(target)][int(horizon)]
                        for target, horizon in zip(held_rows["target"], held_rows["horizon"])
                    ],
                    dtype=float,
                )
                for name in candidates
            }
        else:
            raise ValueError(f"未知路由模式: {mode}")

        pred_matrix = _candidate_matrix(held_rows, candidates)
        selected_first: list[str] = []
        selected_second: list[str] = []
        weights_first: list[float] = []
        weights_second: list[float] = []
        reasons: list[str] = []
        routed_count = 0
        confidence_fallback = 0

        for cell in range(len(held_rows)):
            expected = {name: float(expected_matrix[name][cell]) for name in candidates}
            ordered = sorted(candidates, key=lambda name: expected[name])
            best = ordered[0]
            if best == PARENT_CANDIDATE:
                selected_first.append(PARENT_CANDIDATE)
                selected_second.append("")
                weights_first.append(1.0)
                weights_second.append(0.0)
                reasons.append("best_is_parent")
                continue
            improvement_pp = (expected[PARENT_CANDIDATE] - expected[best]) * 100.0
            if improvement_pp < confidence_min_pp:
                selected_first.append(PARENT_CANDIDATE)
                selected_second.append("")
                weights_first.append(1.0)
                weights_second.append(0.0)
                reasons.append(f"confidence_below_{confidence_min_pp:.3f}pp")
                confidence_fallback += 1
                continue
            if blend_top == 1:
                selected_first.append(best)
                selected_second.append("")
                weights_first.append(1.0)
                weights_second.append(0.0)
                reasons.append(f"argmin_{best}")
                routed_count += 1
                continue
            blend_weights, top2, _ = _soft_blend_top2(expected, candidates)
            selected_first.append(top2[0])
            selected_second.append(top2[1])
            weights_first.append(float(blend_weights[0]))
            weights_second.append(float(blend_weights[1]))
            reasons.append(f"soft_blend_{top2[0]}_{top2[1]}")
            routed_count += 1

        routed_prediction = np.empty(len(held_rows), dtype=float)
        for cell in range(len(held_rows)):
            first = selected_first[cell]
            if not selected_second[cell]:
                routed_prediction[cell] = pred_matrix[cell, candidates.index(first)]
            else:
                second = selected_second[cell]
                routed_prediction[cell] = (
                    weights_first[cell] * pred_matrix[cell, candidates.index(first)]
                    + weights_second[cell] * pred_matrix[cell, candidates.index(second)]
                )
        output.loc[held_mask, "x1_prediction"] = routed_prediction
        output.loc[held_mask, "x1_selected"] = np.asarray(selected_first)
        output.loc[held_mask, "x1_reason"] = np.asarray(reasons)
        output.loc[held_mask, "x1_weight_1"] = np.asarray(weights_first)
        output.loc[held_mask, "x1_weight_2"] = np.asarray(weights_second)

        selection_records.append(
            {
                "held_fold": held_fold,
                "training_folds": json.dumps(
                    _chronological_folds(history_rows), ensure_ascii=False
                ),
                "training_rows": int(len(history_rows)),
                "held_rows": int(len(held_rows)),
                "routed_rows": routed_count,
                "confidence_fallback": confidence_fallback,
                "held_labels_used": False,
            }
        )
        coverage_records.append(
            {
                "held_fold": held_fold,
                "rows": int(len(held_rows)),
                "routed": routed_count,
                "fallback_insufficient_history": 0,
                "fallback_confidence": confidence_fallback,
            }
        )

    coverage = pd.DataFrame(coverage_records)
    fold_selections = pd.DataFrame(selection_records)
    return X1RouterResult(
        rows=output,
        report={},
        fold_selections=fold_selections,
        coverage=coverage,
    )


def evaluate_x1_result(
    result: X1RouterResult,
    *,
    candidates: Sequence[str] = ROUTER_CANDIDATES,
) -> dict[str, object]:
    """汇总 X1 pooled MAPE、相对 A61 改善、命中率与稳定门禁。"""

    rows = result.rows
    actual = rows["actual"].to_numpy(dtype=float)
    routed = rows["x1_prediction"].to_numpy(dtype=float)
    if not np.isfinite(routed).all():
        raise ValueError("X1 路由结果含 NaN/Inf")
    parent = rows[DEFAULT_CANDIDATE_COLUMNS[PARENT_CANDIDATE]].to_numpy(dtype=float)
    p3 = rows[DEFAULT_CANDIDATE_COLUMNS["p3_static"]].to_numpy(dtype=float)

    routed_mape = competition_mape(actual, routed)
    parent_mape = competition_mape(actual, parent)
    p3_mape = competition_mape(actual, p3)

    folds = _chronological_folds(rows)
    fold_improvement: dict[str, float] = {}
    fold_routed: dict[str, float] = {}
    for fold in folds:
        mask = rows["fold"].astype(str).eq(fold).to_numpy()
        fold_routed[fold] = competition_mape(actual[mask], routed[mask])
        fold_parent = competition_mape(actual[mask], parent[mask])
        fold_improvement[fold] = (fold_parent - fold_routed[fold]) * 100.0

    target_improvement: dict[str, float] = {}
    for target in TARGETS:
        mask = rows["target"].eq(target).to_numpy()
        target_improvement[target] = (
            competition_mape(actual[mask], parent[mask])
            - competition_mape(actual[mask], routed[mask])
        ) * 100.0

    recent_folds = folds[-5:]
    recent_improvement = [fold_improvement[fold] for fold in recent_folds]
    recent5_wins = int(sum(value > 0.0 for value in recent_improvement))
    worst_fold_regression = max(
        (0.0, *(-value for value in fold_improvement.values() if value < 0.0))
    )
    max_target_regression = max(
        (0.0, *(-value for value in target_improvement.values() if value < 0.0))
    )

    checks = {
        "pooled_improvement": (parent_mape - routed_mape) * 100.0 >= X1_MIN_IMPROVEMENT_PP,
        "recent5_wins": recent5_wins >= X1_MIN_RECENT5_WINS,
        "worst_fold_regression": worst_fold_regression <= X1_MAX_WORST_FOLD_REGRESSION_PP,
        "target_regression": max_target_regression <= X1_MAX_TARGET_REGRESSION_PP,
    }

    reason_counts = rows["x1_reason"].value_counts().to_dict()
    selected_counts = rows["x1_selected"].value_counts().to_dict()
    routed_mask = rows["x1_selected"].ne(PARENT_CANDIDATE).to_numpy()
    routed_mape_only = competition_mape(actual[routed_mask], routed[routed_mask])
    parent_mape_only = competition_mape(actual[routed_mask], parent[routed_mask])

    return {
        "pooled": {
            "routed_mape": float(routed_mape),
            "parent_mape": float(parent_mape),
            "p3_mape": float(p3_mape),
            "improvement_vs_parent_pp": float((parent_mape - routed_mape) * 100.0),
            "improvement_vs_p3_pp": float((p3_mape - routed_mape) * 100.0),
        },
        "coverage": {
            "routed_cells": int(routed_mask.sum()),
            "total_cells": int(len(rows)),
            "routed_share": float(routed_mask.mean()),
            "routed_mape_only": float(routed_mape_only),
            "parent_mape_only": float(parent_mape_only),
            "improvement_on_routed_cells_pp": float((parent_mape_only - routed_mape_only) * 100.0),
        },
        "reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
        "selected_counts": {str(key): int(value) for key, value in selected_counts.items()},
        "by_fold": {
            "routed_mape": fold_routed,
            "improvement_pp": fold_improvement,
            "recent5_folds": recent_folds,
            "recent5_wins": int(recent5_wins),
            "worst_fold_regression_pp": float(worst_fold_regression),
        },
        "by_target_improvement_pp": target_improvement,
        "max_target_regression_pp": float(max_target_regression),
        "gates": {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "min_improvement_pp": X1_MIN_IMPROVEMENT_PP,
            "min_recent5_wins": X1_MIN_RECENT5_WINS,
            "max_worst_fold_regression_pp": X1_MAX_WORST_FOLD_REGRESSION_PP,
            "max_target_regression_pp": X1_MAX_TARGET_REGRESSION_PP,
        },
    }


def build_x1_report(result: X1RouterResult, evaluation: dict[str, object]) -> dict[str, object]:
    """组装 X1 收据；result.report 现在被填充后返回。"""

    report: dict[str, object] = {
        "experiment": "X1_DYNAMIC_EXPECTED_ERROR_ROUTER",
        "candidates": list(ROUTER_CANDIDATES),
        "parent_candidate": PARENT_CANDIDATE,
        "held_labels_used": False,
        "future_folds_used": False,
        "evaluation": evaluation,
        "fold_selections": result.fold_selections.to_dict(orient="records"),
        "coverage": result.coverage.to_dict(orient="records"),
    }
    result.report.update(report)
    return result.report
