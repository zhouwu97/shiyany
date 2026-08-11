# Wave 0c: Stock-Flow residual-R² 证伪 — **STOP（内容存在但因果不可榨）**

**日期**：2026-08-11
**运行**：`results/runs/20260811_150346_wave0c_stockflow/`
**基线**：SAFE60（g1 residual，19 折 forward cross-fit）

## 结论

**stock-flow 物理链有事后内容（+0.14 R² 上限），但因果不可榨（预测未来物理量反而更差 −0.068）。
按预注册 gate → 线保持关闭。**

## 三层分解（signed residual，g1）

| 层 | 含义 | R² |
| --- | --- | ---: |
| **A** 因果 origin 状态 | holder/rest/balance/ramp/price（origin 可得） | −0.008 |
| **B** A + OOF 预测未来 delta | 预测 P̂/D̂/ΔĤ/B̂ 后作为特征 | **−0.075** |
| **C_cf** cross-fit 完美预见 | 真实未来物理轨迹（forward，上限） | 0.133 → **Δ+0.141** |
| C_insample | in-sample ORACLE（仅参考） | 0.215 |

- **abs residual**：A=0.410（主要由 |y| 水平驱动，非动力学信号，不可榨）；C_cf Δ+0.004 —— 误差幅度上没有物理内容。
- **未来物理量可预测性**（cross-fit R² from origin state）：
  ΔH 0.182 / available 0.163 / rest 0.104 / production 0.070 / gen-gas 0.021。
  未来物理量**部分可预测**（10–18%），不是完全不可知。

## 机制（为什么死）

Ceiling 内容 +0.14 主要住在未来物理轨迹的**不可预测分量**（shock）里：
- 未来 delta 只有 10–18% 可预测，而 B 用了这些预测反而 **−0.068**（比 A 更差）→
  **可预测部分与 SAFE60 误差不共线，噪声注入有害**；
- 真正解释误差的是"实际发生的物理路径"，那在 origin 时刻恰是不可知的。

**又一个 X1/PRED-8A 同形**：事后有内容（甚至给了机制），因果利用失败。

## Regime 分层（signed, cross-fit ceiling vs causal）
| regime | rows | A 因果 | C_cf 上限 | Δ |
| --- | ---: | ---: | ---: | ---: |
| all | 29184 | −0.008 | 0.133 | +0.141 |
| rest transition (\|Δrest\|≥20) | 4991 | 0.088 | 0.256 | **+0.168** |
| holder 快速 | 9728 | 0.069 | 0.193 | +0.124 |
| price switch | 13376 | −0.023 | 0.115 | +0.139 |

transition regime 上限最高（+0.168），但恰是物理 shock 主导、origin 最不可知的区域。

## 预注册判定（写于 run 前）
- ceiling content ≥ 0.05：**PASS**（+0.141）
- transition regime ceiling ≥ 0.10：**PASS**（+0.168）
- causal pred-future ΔR² ≥ 0.03：**FAIL**（−0.068）

**verdict = STOP_CONTENT_BUT_CAUSALLY_UNCAPTURABLE**。

## 保留的不确定性（诚实记录）
B 用的是 Ridge 预测 delta。一个更强的非线性动力学 delta 预测器理论上可能捞回部分
+0.14 ceiling。但机制证据（信息住在不可预测分量、线性捕获反向有害）表明这是低概率路线。
若未来重开，必须用专门的动力学模型 + 新预注册，不能沿用本 gate。

## 意义
- **stock-flow 作为"预测中间量再重构"的专家路线，本轮关闭。**
- SAFE60 收敛再次被强化：物理链的内容全部落在因果边界之外。
- Wave 2 的完整 stock-flow expert **不启动**（gate 未过）。
