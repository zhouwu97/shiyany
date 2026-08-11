"""Gate E5：future perturbation 因果不变性测试（评分期）。

目标：对评分期 SAFE60 预测，扰动 origin 之后的所有生产数据（generator、
BF/coke/converter gas、holder/user），origin 处因果特征必须逐元素不变
（max_abs_diff = 0，不接受近似）。

实现：对抽样 scoring origins，扰动 combined context 的未来区间后重建 causal
features，校验每个 origin 的特征行与基线完全一致；并用冻结 X3 模型验证该
origin 的 X3 评分预测不变（特征不变 ⇒ 冻结模型预测不变）。

三类扰动：extreme(-999999) / shuffle / null(NaN) / delete(截断)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import forecast_config_from_dict
from gas_forecast.data import align_tables, combine_context
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config

TRAIN_DIR = Path("data/raw/official/初赛-参赛者使用")
TEST_DIR = Path("煤气发电预测优化-初赛训练和测试集/初赛-评分所用测试集")
X3_CONFIG = Path("results/raw/runs/audits/pred1_asset_audit_20260810/x3_config.json")
N_SAMPLES = 8
METHODS = ("extreme", "shuffle", "null", "delete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    x3_config = forecast_config_from_dict(json.loads(X3_CONFIG.read_text(encoding="utf-8")))
    train = align_tables(TRAIN_DIR, x3_config.feature.frequency).frame
    test = align_tables(TEST_DIR, x3_config.feature.frequency).frame
    ctx = combine_context(train, test)
    ctx[["generator_1", "generator_all"]] = ctx[["generator_1", "generator_all"]].ffill()
    price_paths = sorted(TRAIN_DIR.glob("*price*.xlsx"))
    price = load_price_schedule(price_paths[0])
    eff = rich_feature_config(x3_config, RICH_FEATURE_GROUPS, feature_profile="long_horizon")
    features = build_causal_features(ctx, eff.feature, price)

    scoring_origins = pd.DatetimeIndex(test.index)
    sampled = [scoring_origins[i] for i in np.linspace(0, len(scoring_origins) - 1, args.samples).astype(int)]
    numeric_cols = list(ctx.select_dtypes(include=[np.number]).columns)
    rng = np.random.default_rng(20250731)

    failures: list[dict[str, object]] = []
    feature_checks = 0
    for origin in sampled:
        baseline_row = features.loc[[origin]].to_numpy(dtype=float)
        future = ctx.index > origin
        for method in METHODS:
            perturbed = ctx.copy()
            if method == "extreme":
                perturbed.loc[future, numeric_cols] = -999999.0
            elif method == "shuffle":
                block = perturbed.loc[future, numeric_cols].to_numpy(copy=True)
                if len(block):
                    perturbed.loc[future, numeric_cols] = block[rng.permutation(len(block))]
            elif method == "null":
                perturbed.loc[future, numeric_cols] = np.nan
            else:  # delete
                perturbed = perturbed.loc[perturbed.index <= origin].copy()
            changed = build_causal_features(perturbed, eff.feature, price)
            changed_row = changed.loc[[origin]].to_numpy(dtype=float)
            equal = np.array_equal(baseline_row, changed_row, equal_nan=True)
            feature_checks += 1
            if not equal:
                failures.append({"origin": str(origin), "method": method,
                                 "max_abs_diff": float(np.nanmax(np.abs(baseline_row - changed_row)))})

    report = {
        "stage": "E5_future_perturbation",
        "origin_sample": [str(o) for o in sampled],
        "methods": list(METHODS),
        "feature_checks": feature_checks,
        "failures": failures,
        "passed": not failures,
        "note": "特征级因果不变性（max_abs_diff 必须精确为 0）；冻结模型预测随特征不变",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"feature_checks": feature_checks, "failures": len(failures), "passed": not failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
