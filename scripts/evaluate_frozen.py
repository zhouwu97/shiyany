"""预测冻结后读取评测期未来真实值，独立计算本地 MAPE。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="只评估已冻结预测，不参与训练或版本选择")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-frozen-prediction", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/raw/frozen_test_score.json"))
    args = parser.parse_args()
    if not args.acknowledge_frozen_prediction:
        parser.error("必须确认预测已冻结：--acknowledge-frozen-prediction")

    raw = args.prediction.read_bytes()
    prediction_hash = hashlib.sha256(raw).hexdigest()
    predictions = pd.read_csv(args.prediction, parse_dates=["datetime"]).set_index("datetime")
    truth = align_tables(args.truth_dir).frame
    config = ForecastConfig()
    scores: dict[str, float] = {}
    for target in config.targets:
        for horizon in config.feature.horizons:
            minutes = 15 * horizon
            actual = truth[target].shift(-horizon).reindex(predictions.index)
            predicted = predictions[f"{target}_t+{minutes}_pred"]
            valid = actual.notna() & predicted.notna()
            denominator = np.maximum(np.abs(actual.loc[valid].to_numpy()), 1e-6)
            scores[f"{target}_t+{minutes}"] = float(
                np.mean(np.abs(actual.loc[valid].to_numpy() - predicted.loc[valid].to_numpy()) / denominator)
            )
    payload = {
        "prediction_sha256": prediction_hash,
        "rows": len(predictions),
        "mape": float(np.mean(list(scores.values()))),
        "by_target_horizon": scores,
        "warning": "该结果不得反馈到训练、阈值、融合权重或版本选择。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

