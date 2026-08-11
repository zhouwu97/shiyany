"""PRED-7：Small Causal TCN（最后的结构性赌注）。

目的：验证直接从严格因果历史序列学习的 temporal representation，能否产生
与 SAFE60 足够低相关的预测误差（corr(e_TCN, e_SAFE60)）。不是 standalone 冠军
竞赛，而是 diversity 探针。

模型：3 个 dilated causal Conv1d block（dilation 1/2/4，hidden 32），ReLU，
causal padding（左填充，保证第 t 步输出只用 <= t 数据），global-avg-pool →
dense → 16 输出（2 target × 8 delta）。参数 < 100k。L1 loss。

严格 forward：held fold 只用 origin<=train_end-120（16 标签全成熟）训练，
归一化也只 fit 训练侧。预测 held origins → absolute = current + delta。

评估：standalone MAPE、residual corr vs SAFE60、4 个预注册 blend（95/5,90/10,
85/15,80/20）。

用法：
  python scripts/run_pred7_tcn.py --output <report.json> [--folds dev_19] [--window 32]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gas_forecast.config import forecast_config_from_dict
from gas_forecast.data import align_tables

TRAIN_DIR = Path("data/raw/official/初赛-参赛者使用")
A60_CONFIG = Path("results/raw/runs/experiments/a60_generator_all_long_residual_verification_20260804/config.json")
SAFE60_OOF = Path("results/raw/runs/audits/pred1_gate_c_20260810/merged_safe60_eval.csv")
TARGETS = ("generator_1", "generator_all")
STEP = 15
HORIZONS = tuple(15 * k for k in range(1, 9))
BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20)


def _channels(frame: pd.DataFrame) -> pd.DataFrame:
    """构建时序通道（原始/轻微清洗的历史变量）。"""
    bf = frame[[c for c in frame.columns if c.startswith("blast_furnace_") and c != "blast_furnace_gas_holder_1"]].fillna(0)
    out = pd.DataFrame(index=frame.index)
    out["g1"] = pd.to_numeric(frame["generator_1"], errors="coerce")
    out["gall"] = pd.to_numeric(frame["generator_all"], errors="coerce")
    out["rest"] = out["gall"] - out["g1"]
    out["holder2"] = pd.to_numeric(frame["blast_furnace_gas_holder_2"], errors="coerce")
    out["bf_total"] = bf.sum(axis=1)
    out["coke"] = pd.to_numeric(frame["coke_oven_1"], errors="coerce")
    out["conv"] = pd.to_numeric(frame["converter_1"], errors="coerce")
    for c in ["generator_use_blast_furnace_gas", "generator_use_coke_gas", "generator_use_converter_gas"]:
        out[c] = pd.to_numeric(frame[c], errors="coerce")
    for c in ["into_gas_mixed_coke", "into_gas_mixed_blast_furnace", "into_gas_mixed_converter"]:
        out[c] = pd.to_numeric(frame[c], errors="coerce")
    out["user_total"] = frame[[c for c in frame.columns if c.startswith("blast_furnace_user") or c.startswith("converter_user")]].fillna(0).sum(axis=1)
    out["air_heater"] = frame[[c for c in frame.columns if c.startswith("air_heater_")]].fillna(0).sum(axis=1)
    return out


class SmallTCN(nn.Module):
    def __init__(self, n_channels: int, n_outputs: int, hidden: int = 32, blocks: int = 3):
        super().__init__()
        convs = []
        in_ch = n_channels
        for i in range(blocks):
            convs.append(nn.Conv1d(in_ch, hidden, kernel_size=3, dilation=2 ** i, padding=2 ** i))
            in_ch = hidden
        self.convs = nn.ModuleList(convs)
        self.head = nn.Linear(hidden, n_outputs)
        self.act = nn.ReLU()

    def forward(self, x):  # x: (B, C, W)
        for conv in self.convs:
            x = self.act(conv(x))  # causal padding keeps length W
        x = x.mean(dim=2)  # global avg pool over time
        return self.head(x)


def _windows(ch: pd.DataFrame, origins: pd.DatetimeIndex, window: int) -> np.ndarray:
    """每个 origin 取最近 window 步的历史通道窗口。"""
    ch_idx = ch.index
    out = np.full((len(origins), len(ch.columns), window), np.nan, dtype=np.float32)
    pos = {t: i for i, t in enumerate(ch_idx)}
    for j, o in enumerate(origins):
        o_pos = pos[o] if o in pos else -1
        if o_pos < 0:
            continue
        start = max(0, o_pos - window + 1)
        slice_ = ch.iloc[start:o_pos + 1].to_numpy(dtype=np.float32)
        out[j, :, window - len(slice_):] = slice_.T
    return out


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p) / np.maximum(np.abs(a), 1e-6)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="dev_10,dev_15,dev_19")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(20250731)
    np.random.seed(20250731)

    cfg = forecast_config_from_dict(json.loads(A60_CONFIG.read_text(encoding="utf-8")))
    frame = align_tables(TRAIN_DIR, cfg.feature.frequency).frame
    ch = _channels(frame)
    ch = ch.ffill().bfill()
    n_ch = ch.shape[1]

    safe60 = pd.read_csv(SAFE60_OOF, parse_dates=["origin_time", "train_end"])
    folds = [f.strip() for f in args.folds.split(",")]
    standalone_rows = []

    for fold in folds:
        part = safe60[safe60["fold"] == fold]
        train_end = part["train_end"].unique()[0]
        held_origins = pd.DatetimeIndex(sorted(part["origin_time"].unique()))
        train_origins = pd.DatetimeIndex(
            frame.index[frame.index <= train_end - pd.Timedelta(minutes=120)])
        # 归一化只 fit 训练侧
        train_ch = ch.reindex(train_origins).to_numpy(dtype=np.float32)
        mean = np.nanmean(train_ch, axis=0, keepdims=True)
        std = np.nanstd(train_ch, axis=0, keepdims=True) + 1e-6
        Xtr = (_windows(ch, train_origins, args.window) - mean[..., 0][None, :, None]) / std[..., 0][None, :, None]
        Xte = (_windows(ch, held_origins, args.window) - mean[..., 0][None, :, None]) / std[..., 0][None, :, None]
        # 训练标签：两 target 的 8 步 delta
        Ytr = np.full((len(train_origins), 16), np.nan, dtype=np.float32)
        for ti, tgt in enumerate(TARGETS):
            s = pd.to_numeric(frame[tgt], errors="coerce").reindex(train_origins)
            Ytr[:, ti * 8:(ti + 1) * 8] = _deltas(s)
        valid_tr = np.isfinite(Ytr).all(axis=1) & np.isfinite(Xtr.reshape(len(train_origins), -1)).all(axis=1)
        Xtr, Ytr = Xtr[valid_tr], Ytr[valid_tr]

        model = SmallTCN(n_ch, 16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        xt = torch.tensor(Xtr)
        yt = torch.tensor(Ytr)
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 512):
                idx = perm[i:i + 512]
                opt.zero_grad()
                loss = torch.nn.functional.l1_loss(model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred_delta = model(torch.tensor(Xte)).numpy()  # (n_held, 16)
        # 重构 absolute
        for ti, tgt in enumerate(TARGETS):
            cur = pd.to_numeric(frame[tgt], errors="coerce").reindex(held_origins).to_numpy(float)
            for k, h in enumerate(HORIZONS):
                mask = part["target"].eq(tgt) & part["horizon"].eq(h)
                idx = held_origins.get_indexer(part.loc[mask, "origin_time"])
                pred = cur[idx] + pred_delta[idx, ti * 8 + k]
                standalone_rows.append(pd.DataFrame({
                    "fold": fold, "target": tgt, "horizon": h,
                    "actual": part.loc[mask, "actual"].to_numpy(float),
                    "anchor": part.loc[mask, "safe60_pred"].to_numpy(float),
                    "pred": pred,
                }))

    r = pd.concat(standalone_rows, ignore_index=True)
    r["resid_a"] = r["actual"] - r["anchor"]
    r["resid_p"] = r["actual"] - r["pred"]
    corr = float(np.corrcoef(r["resid_a"], r["resid_p"])[0, 1])
    summary = {
        "tcn_standalone_mape": _mape(r["actual"].to_numpy(float), r["pred"].to_numpy(float)),
        "anchor_mape": _mape(r["actual"].to_numpy(float), r["anchor"].to_numpy(float)),
        "residual_corr_with_safe60": corr,
        "cells": len(r),
    }
    for w in BLEND_WEIGHTS:
        blend = (1 - w) * r["anchor"] + w * r["pred"]
        summary[f"blend{w:.2f}_mape"] = _mape(r["actual"].to_numpy(float), blend.to_numpy(float))
        print(f"blend {w:.2f}: {summary[f'blend{w:.2f}_mape']:.4f}")

    print(f"TCN standalone={summary['tcn_standalone_mape']:.4f} anchor={summary['anchor_mape']:.4f} corr={corr:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "folds": folds, "window": args.window},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True}))


def _deltas(s: pd.Series) -> np.ndarray:
    out = np.full((len(s), 8), np.nan)
    v = s.to_numpy(dtype=float)
    for k in range(1, 9):
        out[:-k, k - 1] = v[k:] - v[:-k]
    return out


if __name__ == "__main__":
    main()
