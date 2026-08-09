"""X0 Oracle Ceiling 审计模块测试。

覆盖：P3 集成 OOF 键契约、行/分组 oracle、split-half 双向、
预注册判定阈值（固定 0.049）、诊断标记与产物隔离、路线键一致性。
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pytest

from gas_forecast.oracle_ceiling import (
    ORACLE_CANDIDATES,
    ROW_ORACLE_THRESHOLD,
    build_trajectory_frame,
    cross_check_route_oof_keys,
    load_p3_integration_oof,
    pre_registered_verdict,
    run_oracle_ceiling,
    validate_p3_integration_oof,
    write_oracle_ceiling_artifacts,
)
from gas_forecast.scoring import competition_mape

HORIZONS = (15, 30, 45, 60, 75, 90, 105, 120)
TARGETS = ("generator_1", "generator_all")


def _make_oof(
    *,
    folds: int = 2,
    origins_per_fold: int = 4,
    start: str = "2025-03-20 00:00:00",
    biases: Mapping[str, tuple[float, float]] | None = None,
    prediction_mix: tuple[str, str, float] | None = None,
) -> pd.DataFrame:
    """构造完整 2×8 origin 矩阵的合成 P3 OOF（确定性，无噪声）。

    ``biases``：候选名 -> (generator_1 偏置, generator_all 偏置)；
    默认全部为 2.0。``prediction_mix``：P3 融合列 = w*A + (1-w)*B。
    """

    defaults = {name: (2.0, 2.0) for name in ORACLE_CANDIDATES}
    if biases:
        defaults.update(biases)
    rows: list[dict[str, object]] = []
    origin_index = 0
    for fold_number in range(1, folds + 1):
        fold = f"dev_{fold_number:02d}"
        train_end = pd.Timestamp(start) - pd.Timedelta(minutes=15)
        for origin in range(origins_per_fold):
            origin_time = pd.Timestamp(start) + pd.Timedelta(minutes=15 * origin)
            for target in TARGETS:
                for horizon in HORIZONS:
                    offset = 0.05 * origin_index + 0.02 * horizon
                    actual = 30.0 + offset + (5.0 if target == "generator_all" else 0.0)
                    record: dict[str, object] = {
                        "fold": fold,
                        "origin_time": origin_time,
                        "train_end": train_end,
                        "target": target,
                        "horizon": horizon,
                        "actual": actual,
                        "selected_candidate": "parent_80_a64_direct_delta_20",
                    }
                    for name in ORACLE_CANDIDATES:
                        bias = defaults[name][0 if target == "generator_1" else 1]
                        record[f"{name}__prediction"] = actual + bias
                    if prediction_mix is None:
                        record["prediction"] = actual + 1.0
                    else:
                        left, right, weight = prediction_mix
                        record["prediction"] = weight * float(record[f"{left}__prediction"]) + (
                            1.0 - weight
                        ) * float(record[f"{right}__prediction"])
                    rows.append(record)
            origin_index += 1
    return pd.DataFrame(rows)


def _small_report(frame: pd.DataFrame) -> dict[str, object]:
    """在合成帧上跑完整审计，返回报告。"""

    report, _ = run_oracle_ceiling(
        frame,
        expected_folds=int(frame["fold"].nunique()),
        expected_origins=int(frame.groupby(["fold", "origin_time"]).ngroups),
        expected_rows=len(frame),
    )
    return report


# ---------- 键契约 ----------


def test_validate_rejects_blind_fold() -> None:
    frame = _make_oof()
    frame["fold"] = np.where(frame["fold"].eq("dev_01"), "blind", frame["fold"])
    with pytest.raises(ValueError, match="blind"):
        validate_p3_integration_oof(frame)


def test_validate_rejects_duplicate_keys() -> None:
    frame = _make_oof()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="不唯一"):
        validate_p3_integration_oof(duplicated)


def test_validate_rejects_wrong_fold_count() -> None:
    frame = _make_oof(folds=2)
    with pytest.raises(ValueError, match="折数不符"):
        validate_p3_integration_oof(frame, expected_folds=19)


def test_validate_rejects_incomplete_origin_matrix() -> None:
    frame = _make_oof()
    frame = frame.loc[~((frame["target"].eq("generator_1")) & (frame["horizon"].eq(15)))]
    with pytest.raises(ValueError, match="origin"):
        validate_p3_integration_oof(frame)


def test_validate_rejects_nonfinite_prediction() -> None:
    frame = _make_oof()
    frame.loc[0, "a61_parent__prediction"] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_p3_integration_oof(frame)


def test_load_p3_integration_oof_round_trip(tmp_path: Path) -> None:
    frame = _make_oof(folds=2, origins_per_fold=4)
    path = tmp_path / "oof.csv"
    frame.to_csv(path, index=False)
    rows, summary = load_p3_integration_oof(
        path,
        expected_folds=2,
        expected_origins=8,
        expected_rows=len(frame),
    )
    assert summary["all_passed"] is True
    assert len(rows) == len(frame)
    assert rows["fold"].eq("dev_01").any()


# ---------- 行 oracle ----------


def test_row_oracle_picks_min_abs_error_each_row() -> None:
    frame = _make_oof(biases={"a61_parent": (0.5, 0.5), "a64_direct_delta": (3.0, 3.0)})
    report = _small_report(frame)
    row = report["oracle"]["row"]
    assert row["hit_rate"]["a61_parent"] == pytest.approx(1.0)
    assert row["hit_rate"]["a64_direct_delta"] == pytest.approx(0.0)
    expected_mape = competition_mape(frame["actual"], frame["a61_parent__prediction"])
    assert row["mape"] == pytest.approx(expected_mape)


def test_row_oracle_tie_breaks_by_candidate_order() -> None:
    frame = _make_oof(biases={"a61_parent": (1.0, 1.0), "a64_direct_delta": (1.0, 1.0)})
    report = _small_report(frame)
    row = report["oracle"]["row"]
    assert row["hit_rate"]["a61_parent"] == pytest.approx(1.0)
    assert sum(row["hit_rate"].values()) == pytest.approx(1.0)


def test_row_hit_rates_sum_to_one() -> None:
    frame = _make_oof(biases={"p1_causal_rolling": (0.3, 0.3)})
    report = _small_report(frame)
    assert sum(report["oracle"]["row"]["hit_rate"].values()) == pytest.approx(1.0)


# ---------- 分组 oracle ----------


def test_group_oracle_target_level() -> None:
    frame = _make_oof(biases={"a61_parent": (0.5, 5.0), "a64_direct_delta": (5.0, 0.5)})
    report, winners = run_oracle_ceiling(
        frame,
        expected_folds=int(frame["fold"].nunique()),
        expected_origins=int(frame.groupby(["fold", "origin_time"]).ngroups),
        expected_rows=len(frame),
    )
    target = report["oracle"]["target"]
    assert target["distinct_selected"] == 2
    by_group = {item["group"]: item["winner"] for item in winners if item["level"] == "target"}
    assert by_group["generator_1"] == "a61_parent"
    assert by_group["generator_all"] == "a64_direct_delta"


def test_group_oracle_fold_and_target_x_horizon_levels() -> None:
    frame = _make_oof(biases={"a61_parent": (0.5, 0.5)})
    # dev_02 整折改让 a64 胜出
    mask_fold = frame["fold"].eq("dev_02")
    frame.loc[mask_fold, "a61_parent__prediction"] = frame.loc[mask_fold, "actual"] + 5.0
    frame.loc[mask_fold, "a64_direct_delta__prediction"] = frame.loc[mask_fold, "actual"] + 0.5
    # horizon 120 的单元让 p1 胜出
    mask_h = frame["horizon"].eq(120)
    frame.loc[mask_h, "p1_causal_rolling__prediction"] = frame.loc[mask_h, "actual"] + 0.2
    frame.loc[mask_h, "a61_parent__prediction"] = frame.loc[mask_h, "actual"] + 3.0
    frame.loc[mask_h, "a64_direct_delta__prediction"] = frame.loc[mask_h, "actual"] + 3.0
    report, winners = run_oracle_ceiling(
        frame,
        expected_folds=int(frame["fold"].nunique()),
        expected_origins=int(frame.groupby(["fold", "origin_time"]).ngroups),
        expected_rows=len(frame),
    )
    fold_winner = {item["group"]: item["winner"] for item in winners if item["level"] == "fold"}
    assert fold_winner["dev_01"] == "a61_parent"
    assert fold_winner["dev_02"] == "a64_direct_delta"
    cell_winner = {
        item["group"]: item["winner"]
        for item in winners
        if item["level"] == "target_x_horizon" and item["group"].endswith("|120")
    }
    assert cell_winner
    assert set(cell_winner.values()) == {"p1_causal_rolling"}
    assert report["oracle"]["fold"]["distinct_selected"] == 2


# ---------- split-half ----------


def test_split_half_bidirectional_selects_half_winners() -> None:
    frame = _make_oof(folds=1, origins_per_fold=8)
    first_mask = frame["origin_time"].lt(
        pd.Timestamp("2025-03-20 00:00:00") + pd.Timedelta(minutes=60)
    )
    # 前半 a61 完美，后半 a64 完美
    frame.loc[first_mask, "a61_parent__prediction"] = frame.loc[first_mask, "actual"]
    frame.loc[~first_mask, "a64_direct_delta__prediction"] = frame.loc[~first_mask, "actual"]
    report = _small_report(frame)
    split = report["split_half_oracle"]
    detail = split["per_fold"][0]
    assert detail["first_half_winner"] == "a61_parent"
    assert detail["second_half_winner"] == "a64_direct_delta"
    assert split["first_to_second"]["mape"] > 0.01
    assert split["second_to_first"]["mape"] > 0.01
    assert split["combined_mean_mape"] == pytest.approx(
        (split["first_to_second"]["mape"] + split["second_to_first"]["mape"]) / 2.0
    )


# ---------- 预注册判定 ----------


def test_pre_registered_verdict_threshold_is_fixed() -> None:
    assert ROW_ORACLE_THRESHOLD == 0.049
    assert pre_registered_verdict(0.045)["verdict"] == "DYNAMIC_ROUTING_SPACE_EXISTS"
    assert pre_registered_verdict(0.049)["verdict"] == "DYNAMIC_ROUTING_SPACE_EXISTS"
    assert pre_registered_verdict(0.052)["verdict"] == "PREFER_NEW_BASE_MODEL"
    assert pre_registered_verdict(0.045)["threshold"] == 0.049


def test_verdict_and_gap_consistency() -> None:
    frame = _make_oof(biases={"a61_parent": (0.3, 5.0), "a64_direct_delta": (5.0, 0.3)})
    report = _small_report(frame)
    row = report["oracle"]["row"]
    a61 = report["current_mape"]["a61_parent"]
    assert row["gap_pp_vs_a61"] == pytest.approx((a61 - row["mape"]) * 100.0)
    assert row["gap_pp_vs_a61"] > 0.0
    verdict = report["pre_registered_verdict"]
    assert verdict["row_oracle_mape"] == pytest.approx(row["mape"])


# ---------- 当前参考与融合列 ----------


def test_current_mape_p3_fusion_blend() -> None:
    frame = _make_oof(
        biases={"a61_parent": (0.5, 0.5), "a64_direct_delta": (2.0, 2.0)},
        prediction_mix=("a61_parent", "a64_direct_delta", 0.8),
    )
    report = _small_report(frame)
    blended = 0.8 * frame["a61_parent__prediction"] + 0.2 * frame["a64_direct_delta__prediction"]
    assert report["current_mape"]["p3_fusion"] == pytest.approx(
        competition_mape(frame["actual"], blended)
    )


def test_current_mape_reports_all_candidates() -> None:
    frame = _make_oof()
    report = _small_report(frame)
    for name in ORACLE_CANDIDATES:
        assert name in report["current_mape"]
        assert report["current_mape"][name] == pytest.approx(
            competition_mape(frame["actual"], frame[f"{name}__prediction"])
        )


# ---------- 轨迹与产物 ----------


def test_build_trajectory_frame_columns() -> None:
    frame = _make_oof(biases={"a61_parent": (0.5, 0.5)})
    trajectory = build_trajectory_frame(frame)
    expected = {
        "fold",
        "origin_time",
        "train_end",
        "target",
        "horizon",
        "actual",
        "prediction",
        "selected_candidate",
        "row_oracle_winner",
        "current_is_row_oracle",
    }
    assert expected.issubset(trajectory.columns)
    assert trajectory["row_oracle_winner"].eq("a61_parent").all()


def test_write_artifacts_isolated_and_flagged(tmp_path: Path) -> None:
    frame = _make_oof()
    report, winners = run_oracle_ceiling(
        frame,
        expected_folds=int(frame["fold"].nunique()),
        expected_origins=int(frame.groupby(["fold", "origin_time"]).ngroups),
        expected_rows=len(frame),
    )
    assert report["label_informed_diagnostic"] is True
    assert report["formal_candidate"] is False
    out = tmp_path / "audit"
    hashes = write_oracle_ceiling_artifacts(report, winners, build_trajectory_frame(frame), out)
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [
            "report.json",
            "oracle_ceiling_manifest.json",
            "oracle_selection_trajectory.csv",
            "oracle_winners.csv",
            "hit_rates.csv",
            "oracle_gaps.csv",
            "split_half_detail.csv",
        ]
    )
    assert hashes["report.json"] == hashes["report.json"].upper()
    manifest = pd.read_json(out / "oracle_ceiling_manifest.json", typ="series").to_dict()
    assert manifest["label_informed_diagnostic"] is True
    assert manifest["formal_candidate"] is False
    assert manifest["production_usage"] == "FORBIDDEN"
    assert manifest["files"]["report.json"] == hashes["report.json"]


# ---------- 路线键一致性 ----------


def test_cross_check_route_oof_keys_consistent(tmp_path: Path) -> None:
    frame = _make_oof(folds=2, origins_per_fold=4)
    route = frame.loc[:, ["fold", "origin_time", "train_end", "target", "horizon", "actual"]]
    route["prediction"] = frame["p1_causal_rolling__prediction"]
    route.to_csv(tmp_path / "p1_causal_rolling_oof.csv", index=False)
    result = cross_check_route_oof_keys(frame, tmp_path)
    item = result["routes"]["p1_causal_rolling"]
    assert item["status"] == "OK"
    assert item["shared"] == len(frame)
    assert item["integration_only"] == 0
    assert item["route_only"] == 0
    assert item["key_consistent"] is True


def test_cross_check_route_oof_keys_detects_extra_row(tmp_path: Path) -> None:
    frame = _make_oof(folds=2, origins_per_fold=4)
    route = frame.loc[:, ["fold", "origin_time", "train_end", "target", "horizon", "actual"]]
    route["prediction"] = frame["p2_matured_residual__prediction"]
    extra = route.iloc[[0]].copy()
    extra["actual"] = extra["actual"] + 1.0
    route = pd.concat([route, extra], ignore_index=True)
    route.to_csv(tmp_path / "p2_matured_residual_oof.csv", index=False)
    result = cross_check_route_oof_keys(frame, tmp_path)
    item = result["routes"]["p2_matured_residual"]
    assert item["status"] == "OK"
    assert item["route_only"] == 1
    assert item["key_consistent"] is False
