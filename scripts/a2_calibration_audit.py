"""A2 — simplex calibration 稳定性审计（只读仪器化，不修改冻结逻辑）。

问题：当前 simplex calibration 是否已成为 V2/generator_1 的主要瓶颈？

对每个外层折 (dev_01..dev_19, blind)，对 generator_1:
  1. 用 legacy_forecast_config() + 与 rebuild_clean_champion 完全相同的数据
     拟合 v2, 捕获 fit() 后的 state.blend_weights (5,8)。
  2. 在验证窗调用 _predict_branches 得到 (n,5,8) 分支绝对预测张量，
     应用与 predict() 相同的 [0,200] clip，落盘分支预测 + 真实值 + 权重。
  3. 离线按 fold x horizon 计算: blend regret / weight turnover /
     weight->next-fold advantage / oracle gap。

A2 规则: 只用 development folds 做相关/阈值/结论; blind 仅观察。
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import legacy_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.model_ensemble import GasAwareEnsembleForecaster
from gas_forecast.oof import _base_fold_rows
from gas_forecast.scoring import absolute_percentage_error
from gas_forecast.splits import make_outer_folds
from gas_forecast.targets import build_delta_targets

BRANCH_NAMES = ("persistence", "ridge", "recent", "gas", "lgb_residual")
HORIZONS_MIN = (15, 30, 45, 60, 75, 90, 105, 120)
TARGET = "generator_1"
LOW = 0.0
HIGH = 200.0


def clip_branch_abs(branch_abs: np.ndarray) -> np.ndarray:
    """镜像 model_v1._apply_weak_constraints 对 generator_1 的 clip (n,5,8)。"""
    return np.clip(branch_abs, LOW, HIGH)


def capture_fold(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    deltas: pd.DataFrame,
    fold,
    config,
) -> dict[str, object]:
    train_mask, validation_mask = fold.masks(frame.index)
    model = GasAwareEnsembleForecaster("v2", config).fit(
        features.loc[train_mask],
        deltas.loc[train_mask],
        frame.loc[train_mask, list(config.targets)],
    )
    state = model.ensemble_states_[TARGET]

    anchor = frame.loc[validation_mask, TARGET].ffill().to_numpy(dtype=float)
    x_val = features.loc[validation_mask].reindex(columns=model.feature_columns_)
    branch_abs = model._predict_branches(state.branches, x_val, anchor)

    base = _base_fold_rows(frame, fold, validation_mask, config)
    base = base.loc[base["target"].eq(TARGET)]
    # 与 predict() 一致的最终 v2 预测（blend + [0,200] clip）
    v2_abs = np.clip(
        np.einsum("nbh,bh->nh", branch_abs, state.blend_weights),
        LOW,
        HIGH,
    )

    # 关键对齐：_base_fold_rows 是 horizon-major（外层 loop horizon，内层 origin），
    # branch_abs/v2_abs 是 origin-major (n_origins, 8)。不能用 reshape(-1) 直拼，
    # 必须按 (origin_time, horizon) 显式对齐。
    records = base[["fold", "origin_time", "horizon", "actual", "current_value"]].copy()
    origin_times = pd.DatetimeIndex(frame.index[validation_mask])
    # 每分支列: (origin_time, horizon) 两层展开
    branch_long = pd.DataFrame(
        {
            "origin_time": np.repeat(origin_times.to_numpy(), len(HORIZONS_MIN)),
            "horizon": np.tile(HORIZONS_MIN, len(origin_times)),
        }
    )
    for b, name in enumerate(BRANCH_NAMES):
        branch_long[f"{name}_pred"] = branch_abs[:, b, :].reshape(-1)
    branch_long["v2_pred"] = v2_abs.reshape(-1)
    records = records.merge(branch_long, on=["origin_time", "horizon"], how="left")
    records["horizon_idx"] = records["horizon"].map(
        {minutes: step for step, minutes in enumerate(HORIZONS_MIN)}
    )
    records["ape"] = absolute_percentage_error(records["actual"], records["v2_pred"])

    # 自检: v2_pred 必须与 model.predict() 一致（防对齐回归）
    out = model.predict(
        features.loc[validation_mask],
        features.loc[validation_mask, list(config.targets)],
    )
    check_cols = [f"{TARGET}_t+{minutes}_pred" for minutes in HORIZONS_MIN]
    expected = out[check_cols].to_numpy(dtype=float).reshape(-1)
    merged_check = records.merge(
        pd.DataFrame(
            {
                "origin_time": np.repeat(origin_times.to_numpy(), len(HORIZONS_MIN)),
                "horizon": np.tile(HORIZONS_MIN, len(origin_times)),
                "expected": expected,
            }
        ),
        on=["origin_time", "horizon"],
        how="inner",
    )
    max_diff = float((merged_check["v2_pred"] - merged_check["expected"]).abs().max())
    if max_diff > 1e-6:
        raise RuntimeError(
            f"折 {fold.name} 对齐自检失败: v2_pred vs predict() max_diff={max_diff}"
        )

    return {
        "fold": fold.name,
        "blend_weights": state.blend_weights.tolist(),
        "correction_weights": state.correction_weights.tolist(),
        "rows": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/official/初赛-参赛者使用")
    parser.add_argument("--run-dir", default=None, help="写入分支/权重/评分的运行目录")
    parser.add_argument("--folds", default=None, help="逗号分隔的 fold 名白名单（默认全部）")
    parser.add_argument("--no-wipe", action="store_true", help="不清空已存在的 run-dir")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "Pre_gas.csv").exists():
        matches = sorted(
            child for child in data_dir.iterdir()
            if child.is_dir() and (child / "Pre_gas.csv").exists()
        )
        if len(matches) == 1:
            data_dir = matches[0]
        else:
            raise FileNotFoundError(f"无法从数据目录解析官方表格: {data_dir}")

    config = legacy_forecast_config()
    dataset = align_tables(data_dir, config.feature.frequency)
    prices = sorted(data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    features = build_causal_features(dataset.frame, config.feature, price)
    deltas = build_delta_targets(dataset.frame, config.targets, config.feature.horizons)
    folds = make_outer_folds(dataset.frame.index, config)
    if args.folds is not None:
        allowed = set(args.folds.split(","))
        folds = [fold for fold in folds if fold.name in allowed]

    run_dir = Path(args.run_dir or f"results/raw/runs/a2_calibration/{time.strftime('%Y%m%d_%H%M%S')}")
    if run_dir.exists() and not args.no_wipe:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {}
    for fold in folds:
        t0 = time.perf_counter()
        captured = capture_fold(dataset.frame, features, deltas, fold, config)
        captured["rows"].to_csv(run_dir / f"branches_{fold.name}.csv", index=False)
        (run_dir / f"weights_{fold.name}.json").write_text(
            json.dumps(captured["blend_weights"], ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / f"correction_weights_{fold.name}.json").write_text(
            json.dumps(captured["correction_weights"], ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            "fold": fold.name,
            "rows": len(captured["rows"]),
            "elapsed_seconds": round(time.perf_counter() - t0, 1),
        }
        summary[fold.name] = meta
        print(f"{fold.name}: {len(captured['rows'])} rows, {meta['elapsed_seconds']}s",
              flush=True)

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "experiment": "A2 calibration stability audit (generator_1, v2)",
                "target": TARGET,
                "git_commit": None,
                "git_commit_note": "HEAD=4c78efd fixes feat_price_switch_within_120; "
                                   "对照对象是同一 HEAD 特征上的 calibration 各环节 vs oracle",
                "folds": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
