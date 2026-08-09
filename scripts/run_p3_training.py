"""运行 P3 严格 development OOF 训练并保存可恢复的阶段产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from gas_forecast.causal_rolling import build_causal_rolling_oof  # noqa: E402
from gas_forecast.causal_trajectory_ensemble import (  # noqa: E402
    PARENT_ROUTE,
    RouteReceipt,
    build_causal_trajectory_ensemble,
    write_ensemble_artifacts,
)
from gas_forecast.config import ForecastConfig  # noqa: E402
from gas_forecast.data import align_tables  # noqa: E402
from gas_forecast.direct_delta import build_direct_delta_oof  # noqa: E402
from gas_forecast.historical_analog import build_historical_analog_oof  # noqa: E402
from gas_forecast.matured_residual import build_matured_residual_oof  # noqa: E402
from gas_forecast.p3_rolling_integration import (  # noqa: E402
    A64_ROUTE,
    P1_ROUTE,
    P2_ANALOG_ROUTE,
    P2_MATURED_ROUTE,
    derive_anchor_folds,
    validate_p3_oof_keys,
)
from gas_forecast.scoring import competition_mape  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--a61-oof", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor-column",
        default="a61_recursive_blend_05_pred",
        help="冻结 A61 development OOF 预测列",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _phase(run_dir: Path, name: str, **extra: object) -> None:
    payload = {"phase": name, "updated_at": pd.Timestamp.now(tz="Asia/Shanghai"), **extra}
    _write_json(run_dir / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def _score(rows: pd.DataFrame, prediction_column: str) -> float:
    return float(competition_mape(rows["actual"], rows[prediction_column]))


def _receipt(name: str, rows: pd.DataFrame) -> RouteReceipt:
    return RouteReceipt(
        name=name,
        source="p3_training_origin_only_oof",
        status="OOF_PERFORMANCE_ONLY",
        accepted=True,
        reason="严格 development OOF 性能诊断；统一未来门禁仍待独立复跑",
        rows=int(len(rows)),
        blind_labels_used=False,
        future_perturbation_passed=None,
    )


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        _phase(run_dir, "load_inputs")
        frame = align_tables(
            args.data_dir.resolve(), ForecastConfig().feature.frequency
        ).frame
        anchor_raw = pd.read_csv(args.a61_oof.resolve())
        if args.anchor_column not in anchor_raw:
            raise ValueError(f"A61 OOF 缺少冻结列: {args.anchor_column}")
        anchor = anchor_raw.copy()
        anchor["prediction"] = pd.to_numeric(anchor[args.anchor_column], errors="raise")
        anchor_contract = validate_p3_oof_keys(anchor, source=PARENT_ROUTE)
        folds = derive_anchor_folds(anchor, frame.index)
        _write_json(run_dir / "anchor_contract.json", anchor_contract)

        _phase(run_dir, "train_p1", folds=len(folds))
        p1_rows, p1_report = build_causal_rolling_oof(
            frame,
            folds=folds,
            include_blind=False,
            forward_refit=False,
        )
        p1_rows.to_csv(run_dir / "p1_causal_rolling_oof.csv", index=False)
        _write_json(run_dir / "p1_report.json", p1_report)

        _phase(run_dir, "train_p2_historical_analog")
        analog = build_historical_analog_oof(
            frame,
            config=ForecastConfig(),
            scope="development",
            folds=folds,
            origin_only=True,
        )
        analog.rows.to_csv(run_dir / "p2_historical_analog_oof.csv", index=False)
        _write_json(run_dir / "p2_historical_analog_report.json", analog.report)

        _phase(run_dir, "train_a64_direct_delta")
        direct_rows, direct_report = build_direct_delta_oof(
            frame,
            pd.DataFrame(index=frame.index),
            folds=folds,
            include_blind=False,
            nested=False,
            origin_only=True,
        )
        direct_rows.to_csv(run_dir / "a64_direct_delta_oof.csv", index=False)
        _write_json(run_dir / "a64_direct_delta_report.json", direct_report)

        _phase(run_dir, "build_p2_matured_residual")
        matured = build_matured_residual_oof(
            anchor,
            prediction_column="prediction",
            output_column="prediction",
        )
        matured.rows.to_csv(run_dir / "p2_matured_residual_oof.csv", index=False)
        _write_json(run_dir / "p2_matured_residual_report.json", matured.report)

        _phase(run_dir, "cross_fit_performance_diagnostic")
        routes = {
            P1_ROUTE: p1_rows,
            P2_MATURED_ROUTE: matured.rows,
            P2_ANALOG_ROUTE: analog.rows,
            A64_ROUTE: direct_rows,
        }
        receipts = [_receipt(name, rows) for name, rows in routes.items()]
        ensemble = build_causal_trajectory_ensemble(
            anchor,
            routes,
            route_prediction_columns={A64_ROUTE: "ridge_prediction"},
            route_receipts=receipts,
        )
        integration_dir = write_ensemble_artifacts(ensemble, run_dir / "integration")
        scores = {
            PARENT_ROUTE: _score(anchor, "prediction"),
            P1_ROUTE: _score(p1_rows, "prediction"),
            P2_MATURED_ROUTE: _score(matured.rows, "prediction"),
            P2_ANALOG_ROUTE: _score(analog.rows, "prediction"),
            A64_ROUTE: _score(direct_rows, "ridge_prediction"),
            "p3_cross_fitted_static": _score(ensemble.rows, "prediction"),
        }
        report = {
            "experiment": "P3_rolling_training",
            "status": "OOF_PERFORMANCE_ONLY_FUTURE_GATE_PENDING",
            "candidate_promoted": False,
            "blind_labels_used": False,
            "rows": int(len(anchor)),
            "folds": int(len(folds)),
            "origin_count": int(anchor_contract["origin_count"]),
            "scores": scores,
            "static_gate": ensemble.report["static_gate"],
            "elapsed_seconds": time.perf_counter() - started,
            "integration_dir": str(integration_dir),
        }
        _write_json(run_dir / "report.json", report)
        _phase(run_dir, "complete", report="report.json", scores=scores)
        return 0
    except Exception as error:
        _phase(
            run_dir,
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.perf_counter() - started,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
