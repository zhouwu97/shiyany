"""从训练期滚动报告生成可审计的版本选择文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_forecast.config import ForecastConfig
from gas_forecast.data import align_tables
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.orchestration import audit_future_perturbation
from gas_forecast.selection import choose_version


def main() -> None:
    parser = argparse.ArgumentParser(description="选择通过滚动与盲折门槛的最高模型版本")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path)
    parser.add_argument("--v25", type=Path)
    parser.add_argument("--v3", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/raw/model_selection.json"))
    args = parser.parse_args()

    reports = {}
    for version in ("v1", "v2", "v25", "v3"):
        path = getattr(args, version)
        if path:
            reports[version] = json.loads(path.read_text(encoding="utf-8"))

    config = ForecastConfig()
    dataset = align_tables(args.data_dir, config.feature.frequency)
    price_paths = sorted(args.data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(price_paths[0]) if price_paths else None
    features = build_causal_features(dataset.frame, config.feature, price)
    perturbation = audit_future_perturbation(
        dataset.frame,
        config,
        price=price,
        baseline_features=features,
    )
    if not perturbation["passed"]:
        raise RuntimeError(f"未来扰动测试失败: {perturbation['changed_columns']}")

    decision = choose_version(reports)
    decision["future_perturbation"] = perturbation
    decision["test_labels_used"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
