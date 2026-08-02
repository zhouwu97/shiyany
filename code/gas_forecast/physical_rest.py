"""Physical generator_rest 软状态模型与 X1 cross-target 路径。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gas_forecast.aggressive import _fold_order, evaluate_candidate, normalize_branch_frame


@dataclass
class _ClassifierState:
    model: Pipeline | None
    constant_class: int | None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        output = np.zeros((len(features), 4), dtype=float)
        if self.model is None:
            if self.constant_class is None:
                raise RuntimeError("物理状态分类器缺少模型和常量类别")
            output[:, self.constant_class] = 1.0
            return output
        probabilities = self.model.predict_proba(features)
        classes = self.model.named_steps["model"].classes_.astype(int)
        output[:, classes] = probabilities
        return output


def _fit_classifier(features: pd.DataFrame, labels: np.ndarray) -> _ClassifierState:
    classes = np.unique(labels)
    if len(classes) == 1:
        return _ClassifierState(None, int(classes[0]))
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.2,
                    class_weight="balanced",
                    max_iter=1500,
                    random_state=20250731,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return _ClassifierState(model, None)


def _fit_regressor(features: pd.DataFrame, target: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=20.0)),
        ]
    )
    model.fit(features, target)
    return model


def build_rest_training_frame(
    rows: pd.DataFrame,
    *,
    baseline_column: str = "c0_pred",
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把两目标 OOF 长表变成每个 origin×horizon 一行的 rest 训练表。"""

    work = normalize_branch_frame(rows).reset_index(drop=True)
    required = {
        "fold",
        "origin_time",
        "target",
        "horizon",
        "actual",
        "current_value",
        baseline_column,
    }
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"Physical Rest 输入缺少字段: {missing}")
    targets = set(work["target"].unique())
    if not {"generator_1", "generator_all"}.issubset(targets):
        raise ValueError("Physical Rest 同时需要 generator_1 和 generator_all")
    keys = ["fold", "origin_time", "horizon"]
    values = work.pivot(index=keys, columns="target", values=["actual", "current_value", baseline_column])
    values.columns = [f"{metric}_{target}" for metric, target in values.columns]
    frame = values.reset_index()
    frame["rest_actual"] = frame["actual_generator_all"] - frame["actual_generator_1"]
    frame["rest_current"] = (
        frame["current_value_generator_all"] - frame["current_value_generator_1"]
    )
    frame["direct_g1_pred"] = frame[f"{baseline_column}_generator_1"]
    frame["gall_c0_pred"] = frame[f"{baseline_column}_generator_all"]
    frame["rest_c0_pred"] = frame["gall_c0_pred"] - frame["direct_g1_pred"]

    source = work.loc[work["target"].eq("generator_1")].drop_duplicates(keys).set_index(keys)
    if feature_columns is None:
        direct = {
            "current_price",
            "feat_current_price",
            "feat_generator_gas_total",
            "feat_gas_balance",
            "feat_price_switch_within_120",
            "feat_steps_to_price_switch",
        }
        prefixes = (
            "feat_generator_rest_",
            "feat_generator_1_",
            "feat_gas_holder",
            "feat_target_price_",
            "future_price_",
            "price_delta_",
        )
        feature_columns = [
            column
            for column in source.columns
            if column in direct or any(column.startswith(prefix) for prefix in prefixes)
        ]
    extras = source.loc[:, list(feature_columns)].reset_index()
    frame = frame.merge(extras, on=keys, how="left", validate="one_to_one")
    numeric = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    physical = pd.DataFrame(
        {
            "generator_1_current": frame["current_value_generator_1"],
            "generator_all_current": frame["current_value_generator_all"],
            "generator_rest_current": frame["rest_current"],
            "horizon": frame["horizon"],
        }
    )
    horizon_one_hot = pd.get_dummies(frame["horizon"].astype(str), prefix="h", dtype=float)
    features = pd.concat(
        [physical.reset_index(drop=True), numeric.reset_index(drop=True), horizon_one_hot], axis=1
    )
    return frame, features


def _physical_labels(rest_actual: np.ndarray, rest_current: np.ndarray) -> np.ndarray:
    """应用软先验的训练标签；实际推理始终输出概率而非硬状态。"""

    state = np.where(rest_actual < 36.0, 0, np.where(rest_actual < 132.0, 1, 2))
    transition = np.abs(rest_actual - rest_current) >= 20.0
    return np.where(transition, 3, state).astype(int)


def time_ordered_physical_rest_oof(
    rows: pd.DataFrame,
    *,
    baseline_column: str = "c0_pred",
    feature_columns: Sequence[str] | None = None,
    use_blind_for_reporting: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """按历史折拟合四状态概率和条件回归器，生成 physical_rest_pred。"""

    frame, features = build_rest_training_frame(
        rows, baseline_column=baseline_column, feature_columns=feature_columns
    )
    frame = frame.reset_index(drop=True)
    features = features.reset_index(drop=True)
    labels = _physical_labels(
        frame["rest_actual"].to_numpy(float), frame["rest_current"].to_numpy(float)
    )
    prediction = frame["rest_c0_pred"].to_numpy(float).copy()
    probabilities = np.zeros((len(frame), 4), dtype=float)
    folds = _fold_order(frame)
    trajectory: list[dict[str, object]] = []
    for position, fold in enumerate(folds):
        held = frame["fold"].astype(str).eq(fold).to_numpy()
        if position == 0 or (fold == "blind" and not use_blind_for_reporting):
            probabilities[held, 3] = 1.0
            trajectory.append({"fold": fold, "fallback": "c0_implied_rest"})
            continue
        history_folds = [value for value in folds[:position] if value != "blind"]
        train = frame["fold"].astype(str).isin(history_folds).to_numpy()
        if int(train.sum()) < 200:
            probabilities[held, 3] = 1.0
            trajectory.append({"fold": fold, "fallback": "c0_implied_rest"})
            continue
        classifier = _fit_classifier(features.loc[train], labels[train])
        held_probability = classifier.predict_proba(features.loc[held])
        probabilities[held] = held_probability
        expert_prediction = np.zeros((int(held.sum()), 4), dtype=float)
        generic = _fit_regressor(features.loc[train], frame.loc[train, "rest_actual"].to_numpy(float))
        generic_prediction = generic.predict(features.loc[held])
        for state in range(4):
            state_train = train & (labels == state)
            if int(state_train.sum()) < 32:
                expert_prediction[:, state] = generic_prediction
            else:
                expert = _fit_regressor(
                    features.loc[state_train], frame.loc[state_train, "rest_actual"].to_numpy(float)
                )
                expert_prediction[:, state] = expert.predict(features.loc[held])
        prediction[held] = np.sum(held_probability * expert_prediction, axis=1)
        trajectory.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "held_rows": int(held.sum()),
                "class_counts": {
                    str(state): int(np.sum(labels[train] == state)) for state in range(4)
                },
            }
        )
    frame["physical_rest_pred"] = np.clip(prediction, 0.0, 240.0)
    frame["x1_indirect_g1_pred"] = frame["gall_c0_pred"] - frame["physical_rest_pred"]
    for state, name in enumerate(("state_0", "state_1", "state_2", "transition")):
        frame[f"prob_{name}"] = probabilities[:, state]
    return frame, {
        "feature_columns": list(features.columns),
        "fold_trajectory": trajectory,
        "rest_bounds_mw": [0.0, 240.0],
        "state_semantics": {
            "state_0": "rest < 36MW",
            "state_1": "36MW <= rest < 132MW",
            "state_2": "rest >= 132MW",
            "transition": "|future rest - current rest| >= 20MW",
        },
    }


def _residual_correlation_table(
    frame: pd.DataFrame, *, include_blind: bool = False
) -> list[dict[str, object]]:
    frame = frame if include_blind else frame.loc[~frame["fold"].astype(str).eq("blind")]
    records: list[dict[str, object]] = []

    def append(part: pd.DataFrame, scope: str, value: str) -> None:
        direct = part["actual_generator_1"] - part["direct_g1_pred"]
        indirect = part["actual_generator_1"] - part["x1_indirect_g1_pred"]
        correlation = float(np.corrcoef(direct, indirect)[0, 1]) if len(part) >= 2 else float("nan")
        records.append({"scope": scope, "value": value, "rows": int(len(part)), "correlation": correlation})

    append(frame, "overall", "all")
    for horizon, part in frame.groupby("horizon", sort=True):
        append(part, "horizon", str(horizon))
    for fold, part in frame.groupby("fold", sort=True):
        append(part, "fold", str(fold))
    return records


def run_x1_blend_grid(
    rows: pd.DataFrame,
    *,
    baseline_column: str = "c0_pred",
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """只对 generator_1 执行 5/10/15/20% 固定 X1 blend。"""

    rest, physical_report = time_ordered_physical_rest_oof(
        rows, baseline_column=baseline_column, feature_columns=feature_columns
    )
    work = normalize_branch_frame(rows).reset_index(drop=True)
    keys = ["fold", "origin_time", "horizon"]
    indirect = rest.loc[:, keys + ["x1_indirect_g1_pred"]]
    work = work.merge(indirect, on=keys, how="left", validate="many_to_one")
    generator_1 = work["target"].eq("generator_1")
    ranking: list[dict[str, object]] = []
    for weight in (0.0, 0.05, 0.10, 0.15, 0.20):
        column = f"x1_blend_{int(weight * 100):02d}_pred"
        work[column] = work[baseline_column]
        work.loc[generator_1, column] = (
            (1.0 - weight) * work.loc[generator_1, baseline_column]
            + weight * work.loc[generator_1, "x1_indirect_g1_pred"]
        )
        evaluation = evaluate_candidate(work, column, baseline_column=baseline_column)
        ranking.append(
            {
                "candidate": column,
                "weight": weight,
                "pooled_mape": evaluation["candidate"]["pooled_mape"],
                "delta_pp": evaluation["delta_pp"],
                "fold_wins": evaluation["fold_wins"],
                "recent5_wins": evaluation["recent5_wins"],
                "max_fold_regression_pp": evaluation["max_fold_regression_pp"],
            }
        )
    ranking.sort(key=lambda value: value["pooled_mape"])
    best = ranking[0]["candidate"]
    return work, {
        "physical": physical_report,
        "residual_correlation": _residual_correlation_table(rest),
        "ranking": ranking,
        "best_candidate": best,
        "best_evaluation": evaluate_candidate(work, best, baseline_column=baseline_column),
    }
