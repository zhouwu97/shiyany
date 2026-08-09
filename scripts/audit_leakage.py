"""执行至少 50 个起点、5 种未来扰动的因果审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import finalize_run, new_run_dir, write_json
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.leakage import audit_origin_predictor
from gas_forecast.workflow import resolve_prediction_feature_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行多起点未来扰动审计")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="冻结配置 JSON；未提供时使用模型配置或默认配置")
    parser.add_argument("--origins", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=8, help="保留兼容参数；模型级审计按起点顺序执行")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve_data_dir(path: Path) -> Path:
    if (path / "Pre_gas.csv").exists():
        return path
    matches = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "Pre_gas.csv").exists()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"无法解析官方数据目录: {path}")
    return matches[0]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "experiment_causal_prediction_audit")
    output = args.output or run_dir / "report.json"
    config = ForecastConfig()
    model = joblib.load(args.model)
    if hasattr(model, "config"):
        config = model.config
    if args.config:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        config = forecast_config_from_dict(payload)
    data_dir = _resolve_data_dir(args.data_dir)
    dataset = align_tables(data_dir, config.feature.frequency)
    prices = sorted(data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None

    feature_config = resolve_prediction_feature_config(model)

    def predictor(frame, origin):
        features = build_causal_features(frame, feature_config, price)
        current = frame.loc[[origin], list(config.targets)]
        return model.predict(features.loc[[origin]], current)

    causal_prediction_audit = audit_origin_predictor(
        dataset.frame,
        predictor,
        origins=args.origins,
    )
    result = {
        "passed": bool(causal_prediction_audit["passed"]),
        "causal_prediction_audit": causal_prediction_audit,
        "oracle_candidate": False,
        "blind_labels_used": False,
    }
    write_json(output, result)
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": "causal_prediction_audit",
            "passed": bool(causal_prediction_audit["passed"]),
            "causal_prediction_audit": "report.json",
            "oracle_candidate": False,
            "blind_labels_used": False,
            "report": str(output.relative_to(run_dir)),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not causal_prediction_audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
