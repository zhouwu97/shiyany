# Wave 0b: P0 — Physical × SAFE60 Re-Anchored Audit — **STOP（诚实负结果）**

**日期**：2026-08-11
**运行**：`results/runs/20260811_145713_wave0b_p0_physical/`
**基线**：SAFE60 = 0.6×X3 + 0.4×A61（X3 replay OOF 重锚，g1 5.9736 / gall 4.2254 / pooled 5.0995，
与 Gate C 冻结数字逐位一致 ✓）

## 结论

**旧 physical（rest 软状态模型）不是可重开候选。P0 = STOP。**

## 四组数字

### 1) Residual corr(res_safe60, res_physical) — g1
| 范围 | Pearson | Spearman |
| --- | ---: | ---: |
| overall | **0.3240** | 0.3684 |
| h15 → h120 | 0.247 → 0.368 | 单调上升 |
| **recent5** | **0.5650** | — |
| fold mean | 0.436 | （min 0.028 / max 1.000） |

低相关主要在早期折；**近期折升到 0.565** —— diversity 不稳定，且正落在评分期代表分布上。

### 2) Physical standalone（g1 = gall_safe60 − rest_pred）
| 目标 | SAFE60 | Physical | Δ |
| --- | ---: | ---: | --- |
| g1 | 5.9736% | **25.9873%** | **+20.01pp** |
| gall | 4.2254% | = SAFE60 | — |
| pooled | 5.0995% | 15.1064% | +10.01pp |

机制：rest 预测本身尚可（rest MAPE 8.02%，corr(rest_pred,rest_actual)=0.68），但
g1=gall−rest 把 rest 误差放大 ~3.5× 到 g1 上（rest 均量 213 vs g1 均量 60）。
**低 corr 是"弱预测器买来的便宜低相关"** —— 预注册 guardrail 捕获。

### 3) 两专家 Oracle（g1）
- origin oracle raw **5.5524%**、dwell4(60min) **5.7209%** → headroom vs SAFE60 = **0.2527pp**（过了 ≥0.08 线）
- physical 胜 SAFE60 coverage **14.72%**，胜区平均 margin **+2.86pp**，败区平均 **−23.96pp**

hindsight 空间存在（0.25pp），但胜区只有 15%，败区灾难性。

### 4) **Causal-selective gate（决定性）**
- 严格更早折拟合的 origin 级 Logistic gate（features = 4 态概率 + rest/holder/balance/price，全部 origin 可得）
- 选择 coverage **39.8%**，因果选择 g1 **12.43%** vs SAFE60 5.97% → **−6.46pp**（recent5 −4.49pp）
- **因果利用失败**。gate 在 physical 只真胜 14.7% 的分布上选了 40%，每次选都输。
  与 PRED-8A 完全同形：oracle 大 + corr 低，但 origin 时刻无法识别谁赢。

## 预注册判定（写于 run 前）
- corr ≤ 0.70：**PASS**（0.324）
- standalone g1 差 ≤ 0.5pp：**FAIL**（+20.01pp）→ 便宜低相关
- dwell4 oracle headroom ≥ 0.08pp：**PASS**（0.253）
- causal gate ≥ 0.05pp & coverage ≥10% & recent5 稳定：**FAIL**（−6.46pp）

**verdict = STOP**（corr 低但无因果价值；且 standalone 灾难 + recent5 corr 反弹到 0.565）。

## 意义
- 旧 physical（直接 rest 分解）路线**永久关闭**，与其原 X1 判定一致。
- 但这次给出了"为什么"：不是 corr 高，而是 (a) standalone 灾难、(b) 低 corr 不跨近期折稳定、
  (c) 因果 gate 利用失败。**三重独立证据，比单看 0.324 强得多。**
- **SAFE60 收敛被强化**：又一个"hindsight 空间存在但因果不可用"的诚实负结果。
- 不排除"新物理专家"（直接 g1 + physical state 特征，非 rest 分解）——那是 Wave 2 Switching
  MoE / stock-flow 的 gated 范围，与 A51/A60/A61（已在 SAFE60 内）不同。仅当其能产生
  真正低 corr 且因果可选的 g1 residual 时才有资格。
