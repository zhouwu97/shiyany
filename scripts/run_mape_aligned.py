"""在严格 development OOF 上真实运行 X3 MAPE-aligned 基模型。

复用冻结 A61 verification 的 19 个 development 折、long_horizon causal
features 与生产一致的容量投影。三条固定分支（LightGBM regression_l1、
LightGBM Huber、CatBoost MAE）逐 target×horizon 训练，样本权重固定
``1 / max(abs(y_future), epsilon)``，epsilon 只由训练侧固定规则确定并写入
收据；随后执行 pre-registered ``A61 80% + branch 20%`` 融合。

本脚本只写独立 experiments run 目录，不读取 blind / 平台 / 生产覆盖，
不做上传或推送。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from gas_forecast.config import ForecastConfig, forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import (
    config_fingerprint,
    dataframe_fingerprint,
    feature_schema_fingerprint,
    finalize_run,
    new_run_dir,
    write_json,
)
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.mape_aligned import (
    build_mape_aligned_oof,
)
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config


def _read_rows(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["origin_time", "train_end"])


def _load_config(path: Path) -> ForecastConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config 必须是 ForecastConfig JSON 对象")
    return forecast_config_from_dict(payload)


def _price_schedule(data_dir: Path):
    paths = sorted(data_dir.glob("*price*.xlsx"))
    if len(paths) > 1:
        raise ValueError(f"X3 发现多个 price 文件: {paths}")
    return load_price_schedule(paths[0]) if paths else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_label_maturity(result) -> dict[str, object]:
    """复核所有训练记录：origin<=train_end 且标签结束<=train_end，held 不混入。"""

    trace = result.training_trace
    if "history_max_time" not in trace.columns:
        return {"skipped": True}
    trace = trace.copy()
    trace["history_max_time"] = pd.to_datetime(trace["history_max_time"], errors="coerce")
    trace["label_max_time"] = pd.to_datetime(trace["label_max_time"], errors="coerce")
    trace["train_end"] = pd.to_datetime(trace["train_end"], errors="coerce")
    maturity_1 = (trace["label_max_time"] <= trace["train_end"]).fillna(True)
    maturity_2 = (trace["history_max_time"] <= trace["train_end"]).fillna(True)
    return {
        "skipped": False,
        "trained_records": int(trace["status"].eq("trained").sum()),
        "origin_after_train_end": int((trace["history_max_time"] > trace["train_end"]).sum()),
        "label_end_after_train_end": int((trace["label_max_time"] > trace["train_end"]).sum()),
        "label_maturity_passed": bool(
            int((trace["history_max_time"] > trace["train_end"]).fillna(False).sum()) == 0
            and int((trace["label_max_time"] > trace["train_end"]).fillna(False).sum()) == 0
        ),
        "min_label_margin_minutes": (
            None
            if trace["status"].eq("trained").sum() == 0
            else float(
                (
                    pd.to_datetime(trace.loc[trace["status"].eq("trained"), "train_end"])
                    - pd.to_datetime(
                        trace.loc[trace["status"].eq("trained"), "label_max_time"]
                    )
                )
                .dt.total_seconds()
                .div(60)
                .min()
            )
        ),
    }


def _audit_future_perturbation(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    feature_config,
    price_schedule,
    *,
    origins,
    feature_columns,
) -> dict[str, object]:
    """防未来测试：扰动 origin 之后的生产行后，origin 处因果特征必须逐元素不变。

    该方法与 A64 的 future-perturbation 审计语义一致：只扰动未来区间，验证
    当前 origin 的特征输入不受影响。选取少量 origins 以控制重建耗时。
    """

    methods = ("extreme", "shuffle", "null", "delete")
    selected = [pd.Timestamp(value) for value in origins if value in frame.index]
    selected = selected[:6]
    failures: list[dict[str, object]] = []
    baseline = features.loc[selected, feature_columns].to_numpy(dtype=float)
    rng = np.random.default_rng(20250731)
    for position, origin in enumerate(selected):
        numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
        future = frame.index > origin
        for method in methods:
            perturbed = frame.copy()
            if method == "extreme":
                perturbed.loc[future, numeric_columns] = -999999.0
            elif method == "shuffle":
                block = perturbed.loc[future, numeric_columns].to_numpy(copy=True)
                if len(block):
                    perturbed.loc[future, numeric_columns] = block[rng.permutation(len(block))]
            elif method == "null":
                perturbed.loc[future, numeric_columns] = np.nan
            else:
                perturbed = perturbed.loc[perturbed.index <= origin].copy()
            changed_features = build_causal_features(
                perturbed, feature_config, price_schedule
            )
            changed_row = changed_features.loc[[origin], feature_columns].to_numpy(dtype=float)
            equal = np.allclose(
                baseline[position],
                changed_row[0],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
            if not equal:
                failures.append({"origin": str(origin), "method": method})
    return {
        "methods": list(methods),
        "origins_checked": int(len(selected)),
        "cases": int(len(selected) * len(methods)),
        "passes": int(len(selected) * len(methods) - len(failures)),
        "failures": failures,
        "passed": not failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-column", default="a61_recursive_blend_05_pred")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    dataset = align_tables(args.data_dir, config.feature.frequency)
    effective_config = rich_feature_config(
        config,
        RICH_FEATURE_GROUPS,
        feature_profile="long_horizon",
    )
    price_schedule_value = _price_schedule(args.data_dir)
    features = build_causal_features(
        dataset.frame,
        effective_config.feature,
        price_schedule_value,
    )
    parent_rows = _read_rows(args.input)
    result = build_mape_aligned_oof(
        dataset.frame,
        features,
        parent_rows,
        parent_column=args.parent_column,
        n_jobs=args.jobs,
    )
    run_dir = args.run_dir or new_run_dir("results", "experiment_x3_mape_aligned")
    run_dir.mkdir(parents=True, exist_ok=True)
    report = dict(result.report)
    report["effective_feature_config"] = asdict(effective_config)
    report["input"] = str(args.input.resolve())
    report["config"] = str(args.config.resolve())
    report["label_maturity_audit"] = _audit_label_maturity(result)
    # 只审计少量 origin 的未来扰动；帧内其余行在因果特征构造中天然不进入。
    audit_origins = pd.DatetimeIndex(
        parent_rows.drop_duplicates(["fold", "origin_time"])["origin_time"].sort_values()
    )[[0, len(parent_rows["origin_time"].unique()) // 2, -1]]
    report["future_perturbation_audit"] = _audit_future_perturbation(
        dataset.frame,
        features,
        effective_config.feature,
        price_schedule_value,
        origins=audit_origins,
        feature_columns=list(result.report["feature_columns"]),
    )
    report["dependency_versions"] = {
        "python": sys.version.split()[0],
        "lightgbm": _import_version("lightgbm"),
        "catboost": _import_version("catboost"),
        "pandas": _import_version("pandas"),
        "numpy": _import_version("numpy"),
    }
    result.rows.to_csv(run_dir / "oof.csv", index=False, encoding="utf-8")
    result.training_trace.to_csv(run_dir / "training_trace.csv", index=False, encoding="utf-8")
    result.weight_trace.to_csv(run_dir / "weight_trace.csv", index=False, encoding="utf-8")
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", asdict(effective_config))
    report["oof_file_sha256"] = _file_sha256(run_dir / "oof.csv")
    write_json(run_dir / "report.json", report)
    finalize_run(
        run_dir,
        {
            "run_type": "experiment",
            "stage": "X3_mape_aligned_baselines",
            "scope": "development",
            "is_smoke": False,
            "formal_candidate": False,
            "blind_used": False,
            "pooled_mape": None,
            "parent_column": args.parent_column,
            "input": str(args.input.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "config": asdict(effective_config),
            "config_file": "config.json",
            "data_hash": dataframe_fingerprint(dataset.frame),
            "feature_schema_hash": feature_schema_fingerprint(features),
            "config_hash": config_fingerprint(effective_config),
            "oof_hash": report["oof_hash_sha256"],
            "oof_file_hash": report["oof_file_sha256"],
            "report": "report.json",
            "oof": "oof.csv",
            "training_trace": "training_trace.csv",
            "weight_trace": "weight_trace.csv",
            "status": str(report["status"]),
            "retained_candidates": report["retained_candidates"],
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "status": report["status"],
                "retained_candidates": report["retained_candidates"],
                "rows": report["rows"],
                "feature_column_count": report["feature_column_count"],
                "blind_used": report["blind_labels_used"],
                "oof_hash": report["oof_hash_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _import_version(module_name: str) -> str:
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


if __name__ == "__main__":
    main()
