# Wave 0 + Wave 1 重开审计汇总 — 全部诚实负结果，SAFE60 收敛强化

**日期**：2026-08-11
**状态**：`PROJECT_FROZEN` 保持；未出现任何满足重开条件（FINAL_CONVERGENCE §4）的因果可用候选。

## 五枪结果一览

| Wave | 实验 | 关键数字 | verdict |
| --- | --- | --- | --- |
| **0a** | SAFE60 horizon atlas | long(75-120) **6.5715%** vs short(15-60) **3.6275%** | P1 曾判定 S 级 → 实际 1a 也失败 |
| **0b** | P0 physical × SAFE60 | corr **0.324** 但 recent5 **0.565**；standalone **+20.01pp**；causal gate **−6.46pp** | **STOP** |
| **0c** | stock-flow R² 证伪 | ceiling ΔR² **+0.141**；causal pred-future **−0.068** | **STOP**（内容在不可预测分量） |
| **1a** | P1 target-aligned 75-120 | 最佳 horizon_huber **+0.90pp**，corr **0.960** | **STOP** |
| **1b** | 价格 hazard 消融 | max \|ΔAUC\| **0.015**，Δlogloss 全负 | **STOP_PRICE_PERMANENTLY_CLOSED** |

## 每枪的机制性结论

- **P0（old physical）**：rest 专家本身不差（rest MAPE 8%），但 g1=gall−rest 把 rest 误差
  放大 ~3.5× → g1 +20pp。低 corr（0.324）是"弱预测器买来的便宜低相关"，recent5 反弹到
  0.565，因果 gate 选 40% 处 physical 只真胜 14.7%、每次选都输 −6.46pp。
- **stock-flow**：事后上限 +0.141 R² 说明 SAFE60 误差确实与"实际物理路径"共线，但该内容
  住在未来轨迹的**不可预测分量**里；可预测的 10–18% 部分与误差不共线，用预测反而有害（−0.068）。
- **P1（target-aligned）**：4 个变体全部 +0.9–1.65pp、corr 0.87–0.96。target-clock 表示的信息
  已被 SAFE60 的 long-horizon 组件（A51/A60/A61）吸收。`enable_target_aligned_features=false`
  是"吃过没有新东西"，不是"没吃到"。
- **价格**：训练期价格有真实变化（4 值，49% 切换），但对三组物理 transition 的预测增益
  全部 ~0（rest_transition 上反而 −0.012 AUC）。唯一合法未来信息通道 → 永久关闭。

## Wave 2/3 gate 判定（全部不触发）

| 路线 | gate 前提 | 结果 |
| --- | --- | --- |
| **Switching MoE** | Wave 0/1 出现真正低相关因果候选 + 其 transition 引擎（价格 hazard）存活 | **不触发**：无候选；价格已永久关闭 |
| **完整 stock-flow expert** | Wave 0c 证伪通过 | **不触发**：causal pred-future ΔR² = −0.068 |
| **Router resurrection** | 新低相关 expert 存在 | **不触发**：无新 expert |
| **TimeXer** | 便宜模型证明外生信息有新信号 | **不触发**：物理/价格/target-clock 三条外生线全负 |

## 对 SAFE60 收敛的影响

- 五枪全部是"事后内容存在（或有外表低相关）但因果利用失败 / 无内容"的同形结构，
  与 X1 / X0 oracle / PRED-3/5/6/7 / PRED-8A 的负证据链完全一致。
- **SAFE60 = 0.60×X3 + 0.40×A61（pooled 5.0995%，acc 42.3）在当前合法信息集合下仍是局部上限。**
  这次审计用一个更彻底的方式（分层因果 falsification + causal-selective gate）再次确认，
  而非仅靠 residual corr 高。
- **重开条件 §4 未满足**。模型搜索继续保持关闭。SAFE60 提交资产不动。

## 值得留档的信号（未来重开依据）
1. **stock-flow ceiling +0.141 R²（transition regime +0.168）**：物理链对误差确实有事后解释力。
   若未来出现**新的合法外部信息**（如更细的产耗分表、设备状态、气候）使未来物理轨迹可预测性
   显著提高，此上限可重新启用——但必须用专门动力学模型 + 新预注册。
2. **物理状态本身可识别**（state-only：rest_transition AUC 0.685 / avail_contraction 0.713）：
   物理 regime 识别是有信号的，只是价格帮不上、且无低相关专家可路由。
3. **P1 corr 0.87–0.96** 确认 target-clock 已被吸收——未来不必再试同表示。

**最终状态**：`FINAL_CHAMPION_SAFE60` / `PROJECT_FROZEN`（不变）。
合法上传资产唯一：`pred1_safe60_submission.zip`（SHA 3e8993d7…）。
