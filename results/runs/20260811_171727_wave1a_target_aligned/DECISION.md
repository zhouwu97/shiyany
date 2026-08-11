# Wave 1a: P1 — Target-Aligned Long-Horizon Expert — **STOP（诚实负结果）**

**日期**：2026-08-11
**运行**：`results/runs/20260811_171727_wave1a_target_aligned/`
**基线**：SAFE60 on g1 × {75,90,105,120}，long-horizon MAPE = **7.4835%**

## 设计（严格按 P1 规格）
- 只预测 g1 × {75,90,105,120}；gall 保持 SAFE60。
- 预测器：ŷ_{t+h} = y_t + median_{d=1,2,3,7}(y_{t+h−96d} − y_{t−96d})
  - anchor_median（4 锚点中位数，零训练）
  - anchor_wmedian（日均权重中位数 [4,3,2,1]，零训练）
  - horizon_ridge / horizon_huber（每 horizon 独立，5 特征 = 4 锚点 + current；**h120 只看 h120 对齐特征**）
- 禁 CatBoost / 大网格。低自由度，任何结论都能归因到 target-clock alignment。

## 结果（standalone long-horizon g1）

| 专家 | MAPE | Δ vs SAFE60 | residual corr (Pearson) | dwell4 oracle headroom | recent5 corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchor_median | 8.7366% | +1.25pp | 0.909 | 0.637pp | 0.936 |
| anchor_wmedian | 9.1321% | +1.65pp | 0.868 | 0.613pp | 0.893 |
| horizon_ridge | 9.0988% | +1.62pp | 0.894 | 0.563pp | 0.896 |
| **horizon_huber** | **8.3851%** | **+0.90pp** | **0.960** | 0.395pp | 0.971 |

最佳 = horizon_huber：+0.90pp，corr **0.9595**（>0.70 → 无多样性），headroom 0.39pp。

## 预注册 Gate（写于 run 前）
- standalone 超 SAFE60：**FAIL**（全正，+0.90~1.65pp）
- residual corr ≤ 0.70：**FAIL**（0.868–0.960）
- oracle headroom ≥ 0.10/0.15pp：PASS 但 corr 不过

**verdict = STOP**。

## 判定
- **target-clock 对齐没有产生低相关或竞争性的 long-horizon 专家**。四个变体 error structure
  与 SAFE60 高度重合（corr 0.87–0.96），standalone 全部更差。
- 这解释了为何 champion 配置 `enable_target_aligned_features=false`：该表示的信息已被 SAFE60
  （及其 long-horizon 组件 A51/A60/A61）吸收。A51 的 `long_horizon` 特征已含 same-slot/holder/
  balance 周期信息，target-aligned 无增量。
- 不是"没吃到"，是"吃过了，没有新东西"。诚实负结果。
- recent5 corr 更高（0.89–0.97）→ 即便换评估分布，多样性也不存在。

## 意义
- P1 路线关闭。P1 不再有资格作为新 candidate。
- Wave 2 Switching MoE 的第二个前提（出现真正低相关候选）进一步削弱。
