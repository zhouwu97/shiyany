"""E23：基于 Clean Champion 真正 OOF residual 的工业时延扫描。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_forecast.config import legacy_forecast_config
from gas_forecast.data import align_tables
from gas_forecast.experiments import build_fingerprints, finalize_run, new_run_dir, write_json
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.relations import (
    add_stability_diagnostics,
    build_residual_relation_scan,
    freeze_relation_features,
)


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
    parser = argparse.ArgumentParser(description="运行 E23 lag×horizon residual scan")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-lag", type=int, default=16)
    parser.add_argument("--max-features", type=int, default=20)
    args = parser.parse_args()
    if args.max_lag < 0 or args.max_features < 1:
        raise ValueError("扫描范围必须为正")
    run_dir = args.run_dir or new_run_dir("results/raw/runs", "experiment_E23")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = legacy_forecast_config()
    data_dir = _resolve_data_dir(args.data_dir)
    dataset = align_tables(data_dir, config.feature.frequency)
    prices = sorted(data_dir.glob("*price*.xlsx"))
    price = load_price_schedule(prices[0]) if prices else None
    features = build_causal_features(dataset.frame, config.feature, price)
    scan_frame = pd.concat(
        [dataset.frame, features.drop(columns=dataset.frame.columns.intersection(features.columns))],
        axis=1,
    )
    rows = pd.read_csv(args.oof, parse_dates=["origin_time", "train_end"])
    development = rows.loc[rows["fold"].ne("blind")].copy()
    fold_order = (
        development.groupby("fold", sort=False)["origin_time"]
        .min()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    split = max(1, len(fold_order) // 2)
    discovery_folds = set(fold_order[:split])
    confirmation_folds = set(fold_order[split:])
    discovery = development.loc[development["fold"].astype(str).isin(discovery_folds)].copy()
    confirmation = development.loc[
        development["fold"].astype(str).isin(confirmation_folds)
    ].copy()
    scan = build_residual_relation_scan(
        scan_frame,
        discovery,
        prediction_column=args.prediction_column,
        max_lag=args.max_lag,
        max_horizon=max(config.feature.horizons),
    )
    scan = add_stability_diagnostics(
        scan,
        discovery,
        scan_frame,
        prediction_column=args.prediction_column,
    )
    frozen = freeze_relation_features(scan, max_features=args.max_features)
    confirmation_scan = build_residual_relation_scan(
        scan_frame,
        confirmation,
        prediction_column=args.prediction_column,
        max_lag=args.max_lag,
        max_horizon=max(config.feature.horizons),
    ) if not confirmation.empty else pd.DataFrame()
    frozen_keys = {
        tuple([source, int(lag), int(horizon)])
        for source, lag, horizon in (item.split("|") for item in frozen)
    }
    confirmation_rows = confirmation_scan.loc[
        confirmation_scan.apply(
            lambda row: (str(row["source"]), int(row["lag"]), int(row["horizon"])) in frozen_keys,
            axis=1,
        )
    ] if not confirmation_scan.empty else confirmation_scan
    top = scan.assign(abs_corr=scan["corr_residual"].abs()).sort_values(
        ["abs_corr", "rows"], ascending=[False, False], kind="stable"
    )
    report = {
        "experiment_id": "E23",
        "prediction_column": args.prediction_column,
        "scope": "development_only",
        "blind_used_for_selection": False,
        "rows_scanned": int(len(discovery)),
        "discovery_folds": sorted(discovery_folds),
        "confirmation_folds": sorted(confirmation_folds),
        "confirmation_rows_scanned": int(len(confirmation)),
        "scan_rows": int(len(scan)),
        "frozen_relation_features": frozen,
        "top_relations": top.head(100).to_dict(orient="records"),
        "scan": scan.to_dict(orient="records"),
        "confirmation_scan": confirmation_rows.to_dict(orient="records"),
        "fingerprints": build_fingerprints(
            config=config,
            dataset=dataset.frame,
            features=features,
            model_params={"prediction_column": args.prediction_column, "max_lag": args.max_lag},
        ),
    }
    write_json(run_dir / "report.json", report)
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": "E23",
            "scope": "development",
            "is_smoke": False,
            "pooled_mape": None,
            "candidate": "relation_discovery",
            **report["fingerprints"],
            "report": "report.json",
            "blind_used_for_selection": False,
        },
    )
    print(json.dumps({key: report[key] for key in ("experiment_id", "frozen_relation_features", "top_relations")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
