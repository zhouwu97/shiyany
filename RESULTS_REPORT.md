# 实验结果

## 2026-07-31 M1 共享逐行 OOF

使用完全相同的 20 个外层滚动折，保存 62,858 条目标×步长预测单元；每行登记 `train_end` 并满足 120 分钟 purge。8 worker 完整回算耗时 33.7 分钟，折级结果即时保存。

| 候选 | pooled MAPE | 旧折均值 MAPE |
| --- | ---: | ---: |
| Persistence | 5.5604% | 5.5474% |
| V1 | 5.4073% | 5.3958% |
| V2 | 5.3798% | 5.3703% |
| V2.5 | 5.3435% | 5.3333% |
| V3 | 5.3167% | 5.3055% |
| **V2/V3 按目标路由** | **5.3062%** | **5.2947%** |
| 稳定目标×步长 LOFO | 5.3062% | 5.2947% |

稳定目标×步长路由回缩后等价于 `generator_1 -> V2`、`generator_all -> V3`。相对同口径 V3 改善约 0.0106 个百分点；旧折均值低于原报告 V3 的 5.3028%。

## 2026-07-31 M2 cross-fitting blind smoke

真实 blind 折使用 5 个内层 expanding 折、8 步 OOF 残差 LightGBM、三种 simplex、动态门控与三目标结构协调。运行耗时 469 秒。

| 候选 | pooled MAPE |
| --- | ---: |
| Persistence | 6.1258% |
| Ridge | 6.1428% |
| OOF residual LightGBM | 8.4022% |
| Simplex target | 6.0734% |
| Simplex regularized | **6.0724%** |
| Dynamic gate | 6.0924% |
| Diagonal reconciliation | 6.0967% |

该 smoke 中新体系未超过 M1 路由；完整 20 折正在验证，未据单折提前删除模块。

## 外部方案报告中的参考结果

下列数值仅用于确定首轮实现优先级，尚未由本仓库复现：

| 方法 | 报告滚动 MAPE | 当前状态 |
| --- | ---: | --- |
| 持续性预测 | 约 5.56% | 待复现 |
| 绝对增量 Ridge | 约 5.27% | 待复现 |
| 动态软门控 | 约 5.11% | 待复现 |

正式结论必须来自 `results/raw/` 中登记的真实命令、配置、折明细和指标。测试期得分只在预测冻结后用于最终评估，不用于继续调参。

## 2026-07-31 V1 真实盲折 smoke

命令：

```powershell
python scripts/backtest.py --data-dir "data/raw/official/初赛-参赛者使用" --version v1 --max-folds 1 --output results/raw/backtest_v1_smoke.json
```

| 指标 | 结果 |
| --- | ---: |
| 最后 4 天盲折持续性 MAPE | 6.1437% |
| 最后 4 天盲折 V1 MAPE | 5.9333% |
| 相对持续性改善 | 3.42% |

结论：V1 在该盲折通过最低可行性门槛。这里只是单盲折 smoke，不替代完整滚动验证。端到端推理已生成 192 个评测滚动起点、16 个预测字段且无缺失；未读取评测期未来标签进行调参。

## 2026-07-31 V2/V3 同盲折对比

| 版本 | 最后 4 天盲折 MAPE | 相对持续性改善 |
| --- | ---: | ---: |
| 持续性 | 6.1437% | - |
| V1 | 5.9333% | 3.42% |
| V2 | **5.8145%** | **5.36%** |
| V3 | 5.8765% | 4.35% |

结论：煤气增强、近期分支与 LightGBM 残差在该折继续改善 V1，V2 获得当前默认资格。V3 状态迁移和动态门控虽然优于 V1，但未超过 V2，按预设规则回退到 V2。上述比较只使用训练期最后 4 天盲折；完整滚动折仍需在最终选择前执行。

## 2026-07-31 补充方案修复后复验

修复内容包括：煤气切换事件、`generator_rest` BIC 状态、OOF 最优融合系数连续门控、目标×步长 MAD 分歧回缩、特征可用时间审计，以及未来标签与特征模块分离。

单盲折结果：

| 版本 | MAPE |
| --- | ---: |
| 持续性 | 6.1437% |
| V1 | 5.9219% |
| V2 | 5.8112% |
| V2.5 | **5.8055%** |
| V3 | 5.8202% |

最近两个开发折加盲折的复核：

| 折 | V2 MAPE | V2.5 MAPE | 胜者 |
| --- | ---: | ---: | --- |
| dev_18 | **4.2538%** | 4.3210% | V2 |
| dev_19 | **4.8228%** | 4.8709% | V2 |
| blind | 5.8112% | **5.8055%** | V2.5 |
| 三折平均 | **4.9626%** | 4.9991% | V2 |

同三折中，持续性平均 MAPE 为 5.2520%，V1 为 5.0092%，V2 为 4.9626%；V1 在 3/3 折优于持续性，V2 在 3/3 折优于 V1。V2.5 只赢 V2 的 1/3 个共同折且平均 MAPE 更差，未通过逐级门槛。正式默认保持 V2；V2.5 与 V3 继续作为候选，不因单盲折的微小优势晋级。完整 15+ 折仍是赛前最终复核要求。

## 2026-07-31 完整 20 折冻结验证

验证使用 19 个连续但不重叠的两天开发块，以及最后 3 天盲折。所有折保留 120 分钟标签隔离并重新拟合全部处理器和模型。

| 方案 | 平均 MAPE | 优于持续性折数 | `generator_1` | `generator_all` | 最差折 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 持续性 | 5.5483% | - | - | - | - |
| V1 | 5.3974% | 13/20 | 6.1534% | 4.6413% | 7.6782% |
| **V2** | **5.3697%** | **13/20** | **6.1350%** | **4.6043%** | **7.6338%** |

V2 相对 V1：

- 20 个共同折中赢 11 折，满足多数非重叠折获胜；
- 最大单折退化为 0.0483 个百分点，低于预先固定的 0.3 个百分点上限；
- 两个目标平均 MAPE 均改善；
- 15 至 120 分钟的 8 个步长平均 MAPE 均改善；
- 盲折不劣于 V1。

4 月 18 日切换块 `dev_15`：持续性 3.7908%，V1 3.9582%，V2 3.9864%。V2 在该块比 V1 差 0.0282 个百分点，且两种学习模型都不及持续性。这是明确残余风险，但未违反冻结前设定的晋级条件。根据机械选择器，V2 正式通过完整验证并冻结；不再修改模型架构或参数。

## 2026-07-31 四版本自动编排验收

`scripts/auto_pipeline.py`在相同特征矩阵和相同20个非重叠时间折上完整比较V1、V2、V2.5与V3，随后自动选择版本、全量重训、滚动预测、校验并打包。首次运行时，未来扰动守卫发现“是否创建缺失标记列”依赖全时段未来缺失情况；修复为按输入字段无条件创建缺失标记后，390个参考时刻特征全部通过未来扰动测试。

| 方案 | 平均 MAPE | 优于持续性折数 | `generator_1` | `generator_all` | 盲折 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 持续性 | 5.5483% | - | - | - | - |
| V1 | 5.3974% | 13/20 | 6.1534% | 4.6413% | 5.9219% |
| **V2（正式选择）** | **5.3696%** | **13/20** | **6.1349%** | 4.6043% | **5.8112%** |
| V2.5 | 5.3302% | 15/20 | 6.1642% | 4.4963% | 5.8055% |
| V3 | 5.3028% | 16/20 | 6.1576% | 4.4480% | 5.8202% |

高阶版本虽然平均MAPE更低，但没有通过预先冻结的逐级规则：

- V2.5对V2赢13/20折且盲折略优，但`generator_1`从6.1349%退化到6.1642%，违反双目标均不得持续变差的门槛。
- V3对V2.5赢17/20折且两个目标平均值均改善，但盲折从5.8055%退化到5.8202%，违反盲折不得退化的门槛；同时V2.5没有取得晋级资格。
- 自动选择器因此保持V2，没有按平均MAPE单指标改选V2.5或V3。

最终自动产物包含192个滚动起点、16个预测字段，时间范围为2025-05-01 00:00至2025-05-02 23:45；ZIP内仅有`result.csv`。整个自动流程明确登记`test_labels_used=false`、`leaderboard_feedback_used=false`和`manual_prediction_edits=false`。

## 2026-08-01 M2/M3 完整 OOF 验收

使用与 M1 相同的 20 个外层时间折、5 个 expanding 内层折和 120 分钟 purge，8 个 worker 并行，结果保存在 `results/raw/runs/experiments/m2_m3/2026-08-01_00-05-06`。完整运行耗时 29.7 分钟，所有折均有 checkpoint。

| 候选 | pooled MAPE |
| --- | ---: |
| Persistence | 5.5604% |
| crossfit simplex horizon | 5.5195% |
| simplex regularized | 5.5196% |
| struct blended | 5.5196% |
| dynamic gate | 5.5332% |
| struct diagonal | 5.5354% |
| OOF residual LightGBM | 7.4808% |

M1 的 `m1_v2_v3_target` 为 5.3062%，因此 M2/M3 没有达到晋级门槛，正式选择器保持 M1；机械比较记录在 `results/raw/runs/comparisons/2026-08-01_00-35-21`。

## 2026-08-01 M4 候选 smoke

在最终 blind 折执行 CatBoost 和第一阶段 OOF 煤气轨迹两候选，运行目录为 `results/raw/runs/experiments/m4/2026-08-01_00-38-03`，耗时 233.5 秒。

| 候选 | blind pooled MAPE | 结论 |
| --- | ---: | --- |
| M1 目标路由 | 5.8039% | 当前正式模型 |
| CatBoost | 5.9211% | 拒绝，不启动 20 折 |
| Gas trajectory | 6.3353% | 拒绝 |

CatBoost 虽优于新 cross-fitting 候选，但未超过冻结的 M1；煤气轨迹两阶段分支存在误差传播且本折退化。两者均保留为独立 OOF 代码分支，不进入提交模型。

## 2026-08-01 最终模型泄漏审计

对冻结模型 `results/raw/runs/training/2026-07-31_23-53-00/model.joblib` 执行 50 个起点、5 种未来扰动（extreme、shuffle、null、single_field、delete_future），共 250 个案例，8 worker。结果目录为 `results/raw/runs/audits/leakage/2026-08-01_00-43-00`，`passed=true`、失败数为 0。

## 正式提交入口

当前正式最优模型已复制到 `results/best/`，唯一提交目录为 `提交这个/`。上传 `提交这个/teamname_gas_predict_prelim.zip`，该 ZIP 已校验为只包含 `result.csv`；`result.csv`、模型文件和所有历史运行目录均不应上传。

## 并行与复现记录

实验脚本新增 `tree_threads_per_worker`：默认按逻辑核心数和实际外层 worker 数自动分配，避免外层 8 折同时运行时每个 CatBoost/LightGBM 进程都占满全部线程。一折 smoke 使用 16 线程；多折运行自动降到每折 2 线程。每次运行仍写入独立时间戳目录和逐折 checkpoint，可从中断位置恢复。

## 2026-08-01 P1/P2 目标对齐 Ridge 与在线校准完整 OOF

本轮使用同一组 20 个外层折、62,858 个目标×步长单元和 120 分钟 purge。V1 基线来自 `results/raw/runs/oof/2026-07-31_23-17-51/report.json`；目标对齐 Ridge 来自 `results/raw/runs/oof/20260801_145625_722/report.json`；在线校准 cold-start 和 within-fold warm-up 分别来自 `results/raw/runs/oof/20260801_online_full/report.json` 与 `results/raw/runs/oof/20260801_online_full_hot/report.json`。

### Cold-start OOF

| 候选 | pooled MAPE | `generator_1` | `generator_all` | blind 折 | 胜 V1 折数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 基线 | 5.4073% | 6.1518% | 4.6629% | 5.9047% | — |
| Horizon-specific Ridge | 5.4475% | 6.1664% | 4.7287% | **5.9004%** | 4/20 |
| EMA bias | 5.6204% | 6.5065% | 4.7342% | 6.3088% | 5/20 |
| EMA correction gain | **5.3543%** | 6.1992% | **4.5095%** | 5.9484% | 6/20 |
| Forecast vintage | 5.5934% | 6.2561% | 4.9306% | 6.0870% | 1/20 |

`gain` 的 pooled MAPE 比 V1 低 0.0530 个百分点，但 blind 折退化 0.0437 个百分点；Horizon-specific Ridge 仅在 blind 折微降，完整 pooled 和 `generator_1` 均退化。因此二者都不替换正式 M1 路由。

### Within-fold warm-up OOF

每个外层折前 96 个 origin 仅用于填充状态，不计入评分；这不是使用外部历史 OOF 的生产等价外部热启动。相同评分子集上的 V1 基线为 pooled 5.1769%、blind 6.2366%。

| 候选 | pooled MAPE | `generator_1` | `generator_all` | blind 折 | 胜同子集 V1 折数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMA bias | 5.3752% | 6.0674% | 4.6830% | 6.6087% | 4/20 |
| EMA correction gain | **5.0921%** | **5.7633%** | **4.4208%** | 6.3061% | 12/20 |
| Forecast vintage | 5.3724% | 5.8337% | 4.9111% | 6.4595% | 0/20 |

within-fold warm-up 结果同样未通过 blind 折不退化门槛，所有在线候选均保留为可复用实验框架，不进入正式推理和提交模型。

### 固定参数与可复现口径

- Horizon-specific Ridge：目标 `generator_1/generator_all`，步长 `1..8`，`ridge_alpha=20`，校准区比例 `0.15`，增量分位裁剪 `0.001/0.999`；启用目标时刻对齐周期特征，周期日为 `1/2/3/7`。
- 在线校准：`half_life=16`，`bias_clip=12`，`gain_clip=[0, 1.3]`，`vintage_weight=0.25`；每个 outer fold 独立 cold start。
- 评价名称固定为 `cold_start` 与 `within_fold_warmup`；不将 within-fold warm-up 称为外部热启动。

### 冻结提交与路由约束审计

- 收尾时执行 `scripts/prepare_submission.py --no-open`，提交校验通过；`results/best/submission.zip` 与 `提交这个/teamname_gas_predict_prelim.zip` 均只有 `result.csv`，SHA256 都为 `0ca59bb6e66004ae95efa64000ecbf81a86bcc988f0559cae33dc0b7e0d7fb27`。
- 两份 `result.csv` 的字节内容和逐元素内容均一致，SHA256 都为 `a1124f9fe1991c06d4b4d81e841e6296e86c206a424ba78fdb1a48a2dd656e77`，形状为 `192×17`；因此本 PR 不会改变当前待提交 ZIP。
- 对既有 `results/raw/runs/comparisons/2026-07-31_23-51-56/routed.csv` 做与 `RoutedLegacyForecaster.predict()` 相同的确定性后处理审计：无新增上下界裁剪单元；`generator_all > generator_1 + 240` 的 796 个 OOF 单元被收缩，`generator_all < generator_1` 为 0 个，其中 blind 为 48 个。总 MAPE 从 5.3062% 降到 5.3020%，`generator_all` 从 4.4812% 降到 4.4728%，blind 从 5.8039% 降到 5.7997%。该项是对已有 OOF 的后处理影响量化，不重训、不替换冻结提交。

## 2026-08-01 Phase 1 30 天指数时间衰减验收

本轮先在固定 5 折快筛拒绝了目标时刻对齐扩展（E11/E12），并发现长步长 `alpha=5` 的收益仅 0.002 个百分点，未冻结。随后只在 E10 核心 `generator_1` Horizon Ridge 上比较时间漂移。E21 的完整 development OOF 与 final blind 验收均使用冻结的 `generator_all -> V3` 路由、120 分钟 purge 和同一批外层折。

| final 候选 | pooled MAPE | `generator_1` | `generator_all` | 优于基线折数 |
| --- | ---: | ---: | ---: | ---: |
| E10 核心基线 | 5.3160% | 6.1603% | 4.4717% | — |
| **E21，指数半衰期 30d** | **5.3017%** | **6.1319%** | **4.4715%** | **14/20** |

- pooled 改善 0.0143 个百分点，`generator_1` 改善 0.0283 个百分点；最大单折退化为 0.0362 个百分点，低于 0.1 个百分点正式门槛。
- 独立 blind 折从 5.7504% 降至 5.7243%，改善 0.0262 个百分点；`generator_1` 从 6.08% 降至 6.03%。因此没有出现 blind 反转。
- 完整开发 OOF、最终验收报告和逐折预测分别登记在 `results/raw/runs/experiments/e21_recency_development_20260801_1641/` 与 `results/raw/runs/experiments/e21_recency_final_20260801_1657/`。
- 已按 final 报告全量训练 `generator1_horizon`，模型为 `results/raw/runs/training/e21_recency_30d_full_20260801_1714/model.joblib`；泄漏审计覆盖 50 个 origin、5 类未来扰动、250 个案例，全部通过，报告位于 `results/raw/runs/audit/e21_recency_30d_20260801_1717/report.json`。
- 该训练产物尚未覆盖 `results/best/` 或现有提交 ZIP；是否晋级为正式提交由后续明确发布动作决定。

## 2026-08-01 E21 对正式 routed champion 的同口径晋级复核（拒绝）

E10→E21 的 14/20 胜折结论仅证明 30 天指数衰减优于 E10 核心 Ridge，不等价于优于当前正式模型。为作最终晋级判定，使用 `results/best/selection.json` 中冻结的 `generator_1 -> V2`、`generator_all -> V3` 路由，在当前代码与容量约束下重新运行 20 个外层折、62,858 个目标×步长单元；E21 只从其最终验收报告恢复冻结的 30 天半衰期配置。完整可追溯产物位于 `results/raw/runs/experiments/20260801_175745_257/`。

| 同口径模型 | pooled MAPE | `generator_1` | `generator_all` | 相对 formal champion |
| --- | ---: | ---: | ---: | ---: |
| 当前 formal routed champion | 5.301877% | **6.131131%** | 4.472623% | — |
| E21，指数半衰期 30d | **5.301845%** | 6.131938% | **4.471752%** | pooled -0.000032pp |

| 步长 | formal champion | E21 | E21 − champion |
| --- | ---: | ---: | ---: |
| t+15 | **2.285030%** | 2.304443% | +0.019413pp |
| t+30 | **3.353161%** | 3.383004% | +0.029842pp |
| t+45 | **4.225739%** | 4.243979% | +0.018240pp |
| t+60 | **5.116907%** | 5.128239% | +0.011332pp |
| t+75 | 5.898396% | **5.895906%** | -0.002490pp |
| t+90 | 6.608002% | **6.586113%** | -0.021889pp |
| t+105 | 7.213513% | **7.187835%** | -0.025678pp |
| t+120 | 7.723348% | **7.694219%** | -0.029128pp |

- E21 仅赢 8/20 折；最近 5 个开发折仅 `dev_16` 获胜（1/5），其余差异分别为 +0.096792、-0.026963、+0.084512、+0.020258、+0.009475 个百分点。
- blind 折确实从 5.799374% 降至 5.724263%（-0.075110pp），但不能覆盖开发折中的不稳定性和 `generator_1` 的 +0.000807pp 退化。
- 日块 bootstrap（41 日块、2,000 次）中 E21 优于 champion 的概率仅为 48.45%，95% CI 为 [-0.033913pp, +0.031830pp]；仅开发折的概率更低，为 34.68%（38 日块）。
- 因此 `formal_candidate=false`：尽管 pooled 有极小数值改善且 blind 不反转，E21 未达到多数折获胜、`generator_1` 不退化与 bootstrap 支持三道门槛。`results/best/`、现有 192×16 `result.csv` 与提交 ZIP 均保持不变；没有执行 promotion、预测手工编辑或任何测试标签/榜单反馈操作。

## 2026-08-02 严格 Clean Champion C0 与 Phase 2 受控实验收尾

本轮先修复标签边界、OOF checkpoint fingerprint、候选 promotion evidence 和路由后协调，再重新构建严格 C0。C0 使用 20 个严格外层折、62,858 个评分单元和 135 分钟 purge；选中 `v2_v3_target_reconciled`，LOFO 复核与固定路由一致。

| 指标 | 严格 C0 |
| --- | ---: |
| pooled MAPE | **5.297932%** |
| `generator_1` | 6.130328% |
| `generator_all` | 4.465537% |
| blind 折 | 5.790875% |
| 路由后协调前/后 | 5.298231% → 5.297932% |

C0 OOF、选择、逐折 checkpoint 和收据位于 `results/raw/runs/oof/clean_c0_strict_20260801_v2/` 与 `results/raw/runs/receipts/clean_c0_20260801_v2/`。真正 hot-start 实验的历史拟合曾出现重复计算，已增加跨在线候选/外层折缓存；缓存不改变模型语义，并通过研究测试后重跑 E90–E92。

### 受控研究实验结果

- E23 relation scan 仅作为残差关系诊断；E23b screening 相对 C0 退化 0.036307pp，E24 ramp 退化 0.027008pp，E26 grouped recency 的全部短筛配置均退化至少 0.027533pp，均停止。
- E90 true-hot bias 的最佳 hl32 相对 C0 退化 0.068476pp，`generator_1` 退化 0.074147pp，只胜 2/5 折；E91 gain 最佳 hl4 退化 0.032717pp，只胜 1/5 折；E92 vintage 最佳权重 0.15 退化 0.089368pp，只胜 1/5 折。三组均不进入 development。
- E22 damped trend 的 4 个配置结果相同，均退化 0.037341pp；E50/E51 weighted Ridge/LAD 均退化 0.032755–0.047945pp，停止。
- E25 analog 在 5 折 screening 中 k40/k80 一度改善 0.022196/0.024028pp，因此执行完整 development；但 development 反转为退化 0.031618/0.021643pp，日块 bootstrap 支持仅 6.30%/11.45%，不做 blind，也不启动 E25b 或专家路由。

### 正式训练与 Production Gate

严格 C0 已完成全量 routed 训练和滚动预测，产物位于 `results/raw/runs/training/c0_formal_20260801/`。Production Gate 收据确认：

- pooled OOF MAPE：5.297932%；
- 50 个起点 × 5 类未来扰动，共 250 个案例，全部通过；
- pytest：83 passed；
- 提交：192 行、16 个预测列，ZIP 内唯一文件为 `result.csv`；
- `production_gate_passed=true`，已自动晋级 `results/best/`。

当前正式提交仍由严格 C0 路由产生，不包含任何研究候选或测试标签反馈。
