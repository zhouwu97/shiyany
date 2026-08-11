# Wave 1b: 价格 Hazard 消融 — **STOP_PRICE_PERMANENTLY_CLOSED（决定性负结果）**

**日期**：2026-08-11
**运行**：`results/runs/20260811_172734_wave1b_price_hazard/`
**目的**：已知未来电价是规则下**唯一合法未来信息通道**。若它在物理 state transition 识别
（MoE 的 transition 引擎）上无预测增益，价格线**永久关闭**，不再允许任何新包装重开。

## 数据有效性（先验证，防"价格恒定导致测试空转"）
- 训练期价格：4 个不同值（0.22–1.10），std 0.29。
- 未来 120min 内出现价格切换的 origin 占比 **49.1%**。→ 特征非退化，测试有意义。

## 三组物理 transition label（future-actual 定义，仅作监督标签）
1. `rest_transition`: |future rest delta| ≥ 20 MW（base 17.1%）
2. `holder_slope_flip`: holder 斜率方向反转（base 41.1%）
3. `avail_contraction`: 可用气未来缩减 > 5（base 49.9%）

## forward cross-fit 消融（state-only vs state+price）

| label | state-only AUC | +price AUC | ΔAUC | Δlogloss（正=改进） |
| --- | ---: | ---: | ---: | ---: |
| rest_transition | 0.6851 | 0.6734 | **−0.0117** | −0.0076 |
| holder_flip | 0.4811 | 0.4960 | **+0.0149** | −0.0029 |
| avail_contraction | 0.7126 | 0.7121 | **−0.0005** | −0.0002 |

max |ΔAUC| = **0.0149**（< 0.02）；max Δlogloss 改进 = **−0.0002**（无任何正增益）。

## 预注册 Gate（写于 run 前）
价格线关闭当且仅当：三组 label 的 |ΔAUC| 全部 < 0.02 且 Δlogloss 全部 < 0.01 改进。
→ **满足。verdict = STOP_PRICE_PERMANENTLY_CLOSED。**

## 判定
- **已知未来电价对物理 state transition 预测零增益**：rest_transition 上反而轻微有害
  （−0.0117 AUC），其余两组中性。
- 注意：state-only 本身在 rest_transition（0.685）和 avail_contraction（0.713）上
  有真实预测力——物理状态确实可识别。但**价格没有为此增加任何信息**。
- 这不是"价格信息太弱"，而是：价格 schedule 与物理 transition 在训练期不共线。
  价格决定的是电价时段（发电经济性），与 gas 物理状态（holder/rest/avail 变化）
  是不同过程。旧 price→residual 失败、新 price→transition 也失败 → **两条 framing 都关闭**。

## 意义
- **价格整条路线永久关闭**。未来任何"加价格"提案（hazard augmentation / price MoE prior）
  须先在本 gate 上证明价格能预测某个物理 state 变化，否则不再受理。
- Switching MoE 的 transition 引擎（Wave 2）失去其唯一的未来信息增强源。
- SAFE60 收敛再次强化。
