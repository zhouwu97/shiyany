# FINAL CONVERGENCE

> 为什么停在 SAFE60，而不是"没想到更多模型"。

**日期**：2026-08-10；**更新**：2026-08-11（Wave 0/1 Reopen Audit 作为第二层收敛证据，§4 重开条件升级）

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

## 4. 重新开放条件（2026-08-11 升级，Wave 0/1 审计后）

> **旧版"residual corr <0.6 即可重开"已作废**：P0 用教科书反例证明其过弱——
> physical 专家 corr 0.324（诱人）、但 standalone g1 +20pp、causal-selective gate −6.46pp。
> 低残差相关完全可能只是弱模型造成的"便宜多样性"。以后只允许三类情况，均需因果可兑现：

1. **Reopen A — 新增真正的前瞻业务信息**（可提前解释未来 physical innovation 的变量）：
   已确认生产计划、机组启停计划、高炉/转炉操作计划、煤气用户计划、气柜控制计划、
   设备状态/检修、调度指令等。关键不是"更多变量"，而是**能提高未来物理轨迹可预测性**。
   `stock-flow ceiling ΔR²=+0.141`（transition +0.168）是此类信息出现后重新研究的依据。
2. **Reopen B — 新 candidate 必须同时过三关**（不能只低相关）：
   ① residual corr ≤0.7；② standalone skill 不得严重崩坏（不得靠"便宜低相关"）；
   ③ **strict causal conditional gate 兑现正收益**（forward cross-fit 选择能真实拿到头寸，
   recent folds 不反转）。P0 physical 作为此条件的反例入档。
3. **Reopen C — 数据分布或比赛协议实质改变**：测试期变长、新月份进入、新输入字段、
   known-future 信息范围改变、评分函数改变等。

除此之外，不为 CatBoost/Ridge 权重、TCN 参数、<0.02–0.03pp OOF 抖动重开搜索。
不为 price 的任一新包装重开（price→transition hazard 已测，max|ΔAUC|=0.015，永久关闭）。

---

## 5. Wave 0/1 Reopen Audit（2026-08-11，第二层收敛证据）

五枪全部诚实负结果（全独立 run 目录 + 预注册 gate）：

| Wave | 实验 | 关键数字 | 判定 |
| --- | --- | --- | --- |
| 0a | SAFE60 horizon atlas | long(75-120) 6.5715% vs short 3.6275% | 长 horizon 仍主导误差 |
| 0b | P0 physical×SAFE60 | corr 0.324 但 recent5 0.565；standalone +20.01pp；causal gate −6.46pp | **STOP** |
| 0c | stock-flow R² 证伪 | 完美预见 ceiling ΔR²+0.141；预测未来 ΔR²−0.068 | **STOP**（内容在不可预测分量） |
| 1a | P1 target-aligned 75-120 | 最佳 +0.90pp、corr 0.960 | **STOP** |
| 1b | 价格 hazard 消融 | 价格非退化(4值/49%切换)但 max|ΔAUC|=0.015、logloss 全负 | **价格永久关闭** |

### 三个机制性结论（本轮比五个 STOP 更有价值）

1. **长 horizon 难 ≠ 存在遗漏周期信号**。P1 钉死：target-clock 对齐不但没解决长 horizon
   （最佳 +0.90pp），残差 corr 还到 0.960。长 horizon 困难主要来自未来状态本身越来越不可知，
   不是"没对齐到昨天的目标时刻"。
2. **低 residual corr 单独不足**。P0 是教科书反例：corr 0.324、standalone +20pp、
   causal gate −6.46pp、recent5 corr 升到 0.565。低相关来自"错得和 SAFE60 不一样"，非新信息。
3. **stock-flow 统一解释了历史 Oracle 为何漂亮、因果实现为何失败**。未来物理路径对 SAFE60
   residual 确实有解释力（ceiling +0.141），但该内容住在预测时刻后才产生的 innovation 里；
   `F_t` 预测不到。origin Oracle 4.19%、dwell 4.7032%、regret AC 0.637、delayed tracker 失败——
   全部同因：**未来会进入不同动力学状态且持续，但触发状态变化的创新在 origin 尚未进入信息集。**

### 升级后的科学结论

**SAFE60 是当前已验证的合法信息、特征和模型假设集合下的强局部性能上限；其剩余误差主要与
未来不可预知的工况创新有关，而非尚未拟合好的常规历史信号。** 对工程决策而言，
**没有新的合法信息源或真正不同的信息结构，就不应再投入模型搜索。**

### 保留的研究触发器（比保存失败模型更有价值）

- `stock-flow ceiling ΔR² +0.141`（transition +0.168）：新增数据时第一件事是测试它是否
  提高 future physical trajectory predictability。
- `rest_transition AUC 0.685`：regime 可识别，只是当前缺乏足够领先指标。

### 本轮不触发（与 2026-08-10 结论一致，且理由更充分）

Switching MoE（无"低相关+有预测力"的专家，且价格 transition 引擎已关）、完整 stock-flow
expert（0c 证伪败）、Router（无新 expert）、TimeXer（外生信息已在 stock-flow/price 两条
机制探针上失败，无依据用更大网络重新编码同一批变量）。

---

**最终提交状态**：`FINAL_CHAMPION_SAFE60` / `PROJECT_FROZEN`。唯一合法上传资产是
`pred1_safe60_submission.zip`（SHA 3e8993d7…）。platform Accuracy 42.3 / Quality 50。
`89.9 / accuracy 49.9` 的 future-row Oracle 已永久隔离。
