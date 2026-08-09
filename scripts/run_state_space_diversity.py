"""运行 A62 state-space diversity 实验并保存独立、可复现的 development 产物。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import config_fingerprint, dataframe_fingerprint, write_json
from gas_forecast.state_space import build_state_space_diversity


def _read_parent(path: Path) -> pd.DataFrame:
    """读取父模型 OOF，兼容 CSV 和 Parquet。"""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def _read_config(path: Path | None) -> ForecastConfig:
    """读取父模型配置；没有配置时使用项目默认频率。"""

    if path is None:
        return ForecastConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config 必须是 ForecastConfig JSON 对象")
    return forecast_config_from_dict(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="只含生产历史表的数据目录")
    parser.add_argument("--input", type=Path, required=True, help="A61-5%% parent development OOF")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--parent-column", default="a61_recursive_blend_05_pred")
    parser.add_argument("--scope", choices=("screening", "development"), default="screening")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_config(args.config)
    dataset = align_tables(args.data_dir, config.feature.frequency)
    parent = _read_parent(args.input)
    result = build_state_space_diversity(
        dataset.frame,
        parent,
        parent_column=args.parent_column,
        scope=args.scope,
        horizons=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # 逐行 OOF 使用 Parquet 保留时间与数值类型，避免 CSV 往返改变审计键。
    result.rows.to_parquet(args.run_dir / "oof.parquet", index=False)
    result.fold_metrics.to_csv(args.run_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    result.target_metrics.to_csv(args.run_dir / "target_metrics.csv", index=False, encoding="utf-8")
    result.horizon_metrics.to_csv(args.run_dir / "horizon_metrics.csv", index=False, encoding="utf-8")
    result.training_trace.to_csv(args.run_dir / "training_trace.csv", index=False, encoding="utf-8")
    report = dict(result.report)
    report["parent_comparison"] = {
        candidate: report.get("screening_reports", {}).get(candidate)
        for candidate in report.get("comparison_candidates", [])
    }
    report.update(
        {
            "input": str(args.input.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "config": asdict(config),
            "config_hash": config_fingerprint(config),
            "data_hash": dataframe_fingerprint(dataset.frame),
            "outputs": {
                "oof": "oof.parquet",
                "fold_metrics": "fold_metrics.csv",
                "target_metrics": "target_metrics.csv",
                "horizon_metrics": "horizon_metrics.csv",
                "training_trace": "training_trace.csv",
                "report": "report.json",
            },
        }
    )
    write_json(args.run_dir / "report.json", report)
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "scope": args.scope,
                "status": report["status"],
                "selected_candidate": report["selected_candidate"],
                "blind_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
