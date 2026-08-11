"""Wave 0a: SAFE60-era horizon atlas（重新生成，不用旧 error atlas）。

在 SAFE60-era OOF（merged_safe60_eval.csv，19 折全量）上按
target × horizon × fold 生成 X3 / A61 / SAFE60 的 MAPE。

决定：SAFE60 是否仍在 75–120 呈现主要剩余误差结构 → 决定 P1 优先级。

产物：results/runs/<stamp>/（atlas 表格 + run_meta.json + 日志）。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from gas_forecast.scoring import competition_mape  # noqa: E402

SAFE60_OOF = Path(
    "results/raw/runs/audits/pred1_gate_c_20260810/merged_safe60_eval.csv"
)
EXPERTS = ("x3_cat_mae_pred", "a61_recursive_blend_05_pred", "safe60_pred")
LABELS = {"x3_cat_mae_pred": "X3", "a61_recursive_blend_05_pred": "A61", "safe60_pred": "SAFE60"}
HORIZONS = (15, 30, 45, 60, 75, 90, 105, 120)


def _pp(a: float, b: float) -> float:
    return (a - b) * 100.0


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{stamp}_wave0a_atlas")
    run_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(SAFE60_OOF, parse_dates=["origin_time", "train_end"])
    required = {"fold", "origin_time", "target", "horizon", "actual"} | set(EXPERTS)
    missing = sorted(required.difference(m.columns))
    if missing:
        raise ValueError(f"SAFE60 OOF 缺少字段: {missing}")
    m["fold"] = m["fold"].astype(str)
    folds = _fold_order(m)
    recent5 = folds[-5:]

    def mape_for(rows: pd.DataFrame, col: str) -> float:
        return float(competition_mape(rows["actual"], rows[col]))

    records: list[dict[str, object]] = []
    # 总体 pooled
    for col in EXPERTS:
        records.append({"scope": "pooled", "value": "all", "expert": LABELS[col],
                        "rows": int(len(m)), "mape": mape_for(m, col)})
    # 按 target
    for target, part in m.groupby("target", sort=True):
        for col in EXPERTS:
            records.append({"scope": "target", "value": str(target), "expert": LABELS[col],
                            "rows": int(len(part)), "mape": mape_for(part, col)})
    # 按 horizon
    for horizon, part in m.groupby("horizon", sort=True):
        for col in EXPERTS:
            records.append({"scope": "horizon", "value": str(horizon), "expert": LABELS[col],
                            "rows": int(len(part)), "mape": mape_for(part, col)})
    # 按 fold
    for fold in folds:
        part = m.loc[m["fold"].eq(fold)]
        for col in EXPERTS:
            records.append({"scope": "fold", "value": fold, "expert": LABELS[col],
                            "rows": int(len(part)), "mape": mape_for(part, col)})

    atlas = pd.DataFrame.from_records(records)

    # 关键判读表：horizon × target，SAFE60 MAPE + SAFE60-vs-A61 delta
    key_horizon: list[dict[str, object]] = []
    for target, part in m.groupby("target", sort=True):
        for horizon in HORIZONS:
            sub = part.loc[part["horizon"].eq(horizon)]
            if sub.empty:
                continue
            s = mape_for(sub, "safe60_pred")
            a = mape_for(sub, "a61_recursive_blend_05_pred")
            x = mape_for(sub, "x3_cat_mae_pred")
            key_horizon.append({"target": str(target), "horizon": horizon,
                                "rows": int(len(sub)),
                                "SAFE60_mape": s, "X3_mape": x, "A61_mape": a,
                                "SAFE60_vs_A61_pp": _pp(a, s)})
    horizon_table = pd.DataFrame.from_records(key_horizon)

    # 长 horizon(75-120) vs 短 horizon(15-60) 分解
    short = m.loc[m["horizon"].isin((15, 30, 45, 60))]
    long_ = m.loc[m["horizon"].isin((75, 90, 105, 120))]
    hsplit: list[dict[str, object]] = []
    for name, part in (("short_15_60", short), ("long_75_120", long_)):
        for col in EXPERTS:
            hsplit.append({"band": name, "expert": LABELS[col],
                           "rows": int(len(part)), "mape": mape_for(part, col)})

    # recent5 pooled
    r5 = m.loc[m["fold"].isin(recent5)]
    r5rows: list[dict[str, object]] = []
    for col in EXPERTS:
        r5rows.append({"band": "recent5", "expert": LABELS[col],
                       "rows": int(len(r5)), "mape": mape_for(r5, col)})

    summary = pd.concat([pd.DataFrame.from_records(hsplit), pd.DataFrame.from_records(r5rows)])

    atlas.to_csv(run_dir / "atlas_long.csv", index=False, encoding="utf-8")
    horizon_table.to_csv(run_dir / "horizon_by_target.csv", index=False, encoding="utf-8")
    summary.to_csv(run_dir / "short_long_split.csv", index=False, encoding="utf-8")

    # run_meta.json
    run_meta = {
        "run": f"wave0a_safe60_horizon_atlas",
        "stamp": stamp,
        "date": datetime.now().isoformat(),
        "python": sys.executable,
        "input": str(SAFE60_OOF.resolve()),
        "inputs_sha": {"note": "see logs"},
        "status": "complete",
        "outputs": ["atlas_long.csv", "horizon_by_target.csv", "short_long_split.csv"],
        "pre_registered": {
            "purpose": "决定 P1 target-aligned 75-120 是否 S 级",
            "decision_rule": "SAFE60 长 horizon MAPE 仍显著高于短 horizon -> P1 保持优先级",
        },
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"=== Wave 0a: SAFE60 horizon atlas ===")
    print(f"run_dir: {run_dir}")
    pooled = {r["expert"]: r["mape"] for r in records if r["scope"] == "pooled"}
    print("pooled:", {k: f"{v:.6f}" for k, v in pooled.items()})
    print("\n--- long(75-120) vs short(15-60) ---")
    print(summary[["band", "expert", "mape"]].to_string(index=False))
    print("\n--- SAFE60 by horizon×target (vs A61 pp) ---")
    pivot = horizon_table.pivot_table(index="horizon", columns="target",
                                      values="SAFE60_mape")
    print(pivot.round(4).to_string())


def _fold_order(rows: pd.DataFrame) -> list[str]:
    order = rows.groupby("fold", sort=False, observed=True)["origin_time"].min().sort_values()
    return order.index.astype(str).tolist()


if __name__ == "__main__":
    main()
