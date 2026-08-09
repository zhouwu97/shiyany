"""按测试期每个滚动起点生成宽表预测。"""

from __future__ import annotations

import argparse
from pathlib import Path

from gas_forecast.workflow import predict_rolling
from gas_forecast.submission_quality import (
    COMPETITION_QUALITY_POLICY,
    prepare_submission_input,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成短周期滚动预测")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("submissions"))
    args = parser.parse_args()

    features, predictions = predict_rolling(args.train_dir, args.test_dir, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_output, _ = prepare_submission_input(
        features.reset_index(),
        COMPETITION_QUALITY_POLICY,
    )
    result_output = predictions.reset_index()
    feature_output.to_csv(args.output_dir / "input.csv", index=False, encoding="utf-8")
    result_output.to_csv(args.output_dir / "s_result.csv", index=False, encoding="utf-8")
    print(f"已生成 {len(result_output)} 个滚动起点")


if __name__ == "__main__":
    main()
