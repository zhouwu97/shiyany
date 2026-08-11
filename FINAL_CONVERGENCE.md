# FINAL CONVERGENCE

> 为什么停在 SAFE60，而不是"没想到更多模型"。

**日期**：2026-08-10

---

## 1. 合法冠军

```
SAFE60 = 0.60 × X3 + 0.40 × A61
```

- **Platform**：92.3 = Quality 50.0 + Accuracy 42.3（Success A `PLATFORM_IMPROVEMENT_CONFIRMED`）
- g1 1-MAPE **0.9457**（MAPE 5.43%）、gall 1-MAPE **0.9581**（MAPE 4.19%）
- development pooled **5.099520%**，相对 A61 +0.0962pp，19/19 fold 胜，bootstrap P=1.0
- 正式提交：`pred1_safe60_submission.zip`（SHA `3e8993d7…`）+ 冻结 R1 input（`23330d3c…`）

## 2. 正证据（为何相信 SAFE60 是真的）

1. **逐字节复现**：X3/A61 在主仓重放 OOF 与冻结 OOF 完全一致（SHA 逐位匹配）。
2. **生产链无语义漂移**：六层 production runner（RichGas→A51→splice→A60→A61→X3）
   在 19 个历史 cutoff 端到端复现 SAFE60 到机器精度（corr=1.0）。
3. **未来扰动零失败**：32/32 特征检查对 origin 后数据完全不变。
4. **平台验证**：acc 42.0→42.3，两个 target 1-MAPE 同时提升；gall 提升 > g1，
   与 PRED-R0 transfer 分析预测一致。
5. **合法因果纪律贯穿**：forward-fill 只读 `<origin`、blind 单向冻结、R1 input
   平台验证 50/50 未动。

## 3. 负证据（为何停止搜索）

五条独立结构性路线全部诚实失败，共同指向"残差相关 0.77–0.97"：

| 路线 | 停止原因 |
| --- | --- |
| X1 router | 动态路由（oracle 大但不可学）|
| PRED-3 残差校准 | bias 可校正量 <0.01pp 且 recent5 反转 |
| PRED-5 共享 trajectory | 与 SAFE60 残差相关 0.77–0.97，无新多样性 |
| PRED-6 target joint | rest=gall−g1 分解不敌直接预测 |
| PRED-7 small TCN | 因果时序表示 residual corr 0.925 |
| PRED-8A 动态机会审计 | Oracle 4.7032% + regret AC 0.637，但 causal tracker 5.1061% 不敌 5.0995% |

**结论**：现有合法信息集合下，tabular/时序/结构/动态表示均已高度相关于 SAFE60。
SAFE60 捕获了绝大多数可利用信号，是当前信息边界下的局部最优。

## 4. 重新开放条件（仅三者之一）

1. **新增合法外部信息**：真实的新业务变量/数据源。
2. **真正低相关的新 candidate**：与 SAFE60 residual corr 显著 <0.7（例如 <0.6）。
   `regret lag1 autocorr=0.637` 是此类 candidate 出现后重研究 routing 的依据，
   不代表当前 router 可用。
3. **比赛数据/评分协议变化**：平台规则或测试分布发生实质改变。

除此之外，不为 CatBoost/Ridge 权重、TCN 参数、<0.02–0.03pp OOF 抖动重开搜索。

---

**最终提交状态**：`FINAL_CHAMPION_SAFE60` / `PROJECT_FROZEN`。唯一合法上传资产是
`pred1_safe60_submission.zip`；`89.9 / accuracy 49.9` 的 future-row Oracle 已永久隔离。
