# 实验结果

## 2026-08-09 X0 P3 Oracle Ceiling Audit（标签知情诊断上限）

只读复用 `p3_rolling_training_20260809_190558` 的 development OOF，不重训。
键契约：19 折、3,648 origin、58,368 行、无 blind、完整键一致（四条候选独立
OOF 与集成 OOF shared=58,368 / integration_only=0 / route_only=0）。

| 候选/参考 | pooled MAPE |
| --- | ---: |
| A61 parent | 5.195745% |
| P3 静态融合（80% A61 + 20% A64） | 5.159141% |
| A64 DirectDelta | 5.328928% |
| P1 CausalRolling | 5.543435% |
| P2 HistoricalAnalog | 5.504724% |
| P2 MaturedResidual | 6.575605% |

| Oracle 层级 | MAPE | gap vs A61 | gap vs P3 | 选中候选数 |
| --- | ---: | ---: | ---: | ---: |
| row | 3.376515% | +1.819230pp | +1.782626pp | 5 |
| origin | 4.191370% | +1.004375pp | +0.967770pp | 5 |
| fold | 5.111658% | +0.084087pp | +0.047482pp | 4 |
| target | 5.195745% | +0.000000pp | -0.036604pp | 1 |
| horizon | 5.195745% | +0.000000pp | -0.036604pp | 1 |
| target×horizon | 5.195745% | +0.000000pp | -0.036604pp | 1 |
| split-half 前半→后半 | 4.994948% | +0.200797pp | +0.164193pp | - |
| split-half 后半→前半 | 5.621315% | -0.425570pp | -0.462174pp | - |
| split-half 双向均值 | 5.308113% | - | - | - |

行级候选命中率：a61_parent 16.79%、a64_direct_delta 21.75%、p1_causal_rolling
17.92%、p2_historical_analog 22.54%、p2_matured_residual 20.99%。

预注册判定：row oracle `3.376515% <= 4.9%` → `DYNAMIC_ROUTING_SPACE_EXISTS`
（阈值固定 0.049，未修改）。粗粒度（target/horizon/单元）无路由增益；
动态路由头寸集中在 origin/row 粒度，fold 粒度 split-half 双向均值
`5.308113%` 对 A61 无稳定正收益，反向方向为负。全部产物
`results/raw/runs/audits/x0_oracle_ceiling_20260809/`，标记
`label_informed_diagnostic=true`、`formal_candidate=false`、`production_usage=FORBIDDEN`，
不进 `results/best`、不产生提交。report.json SHA-256
`25CF238C3F58896733002B428DF6967FE09B8BA76C1F713015E2BBB83DEC9BC2`；
oracle_selection_trajectory.csv SHA-256
`AB888C8E1A4BF69291C5B9E4888412A4240AEE0B5EC0708CE81120C366A55709`；
oracle_ceiling_manifest.json SHA-256
`5C4724ECB008387FBD3CDB9A88BBC05FBDE7966F6F1395AEC4E397B088E15644`。

## 2026-08-09 平台真实评分

提交：`提交这个_训练优化_复跑/咕咕嘎嘎_gas_predict_prelim.zip`

平台返回：`split=gas_power`，`horizons=[15, 30, 45, 60, 75, 90, 105, 120]`。

| 项目 | 得分 | 满分 |
| --- | ---: | ---: |
| 质量 | 40.0 | 50 |
| 准确率 | 49.9 | 50 |
| **总分** | **89.9** | **100** |

质量细项：`miss=10.0`、`dup=5.0`、`out=0.0`、`intv=5.0`、`invalid_col=0.0`、`feat=5.0`、`comp=15.0`。

准确率细项：`1mape_1=0.9993`、`1mape_all=0.9993`，对应每个目标 MAPE 约 `0.07%`。

该记录是平台实际返回结果；本地使用参考标签计算的 `98.5668` 分仅作为离线估算，不能替代平台分数。

## 2026-08-09 Q4 reference-quality input A/B 本地审计

实际运行目录：`.tmp/q4_reference_quality_packages_20260809_run3/`。运行只读取正式评分 input、
官方训练期生产数据、冻结模型和已合法提交 ZIP；未读取 blind、平台答案或未来真实标签，也未上传。

- 正式 scoring input：192 行、699 列，SHA256
  `7629a0d4c65ed4a39e5dbefe1748100238b92502ba32d834d04226e463d2781e`。
- 正式特征 API 重建 training input：11,521 行、703 列，冻结统计使用截至
  `2025-04-30 23:45:00` 的 11,520 行；文件 SHA256
  `97c101e38631679739c03e7daa891696a10e5aaea4008de32b8fffeda276e933`。
- 冻结模型 SHA256：`90be24067dbfd67d677b6d03ba7d3ce1f0b5613ed7fa2c55aebc4e829e9413de`。
- 源 ZIP 内冻结 `s_result.csv` SHA256：
  `2dfe7f29cbde9faf846e4a03be292a61eceb93469b199963c565bba2a8c37efe`。

| 本地机械量 | SUB_A Q_CAUSAL | SUB_B Q_REFERENCE |
| --- | ---: | ---: |
| `miss`：非有限单元 | 0 | 0 |
| `dup`：重复时间戳 / 重复列 | 0 / 18 | 0 / 0 |
| `out`：五口径 IQR / `abs(z)>3` | 2293 / 395 | 0 / 0 |
| `intv`：非 15 分钟间隔 | 0 | 0 |
| `invalid_col`：非法列 | 0 | 0 |
| `feat`：`feat_` 字段 | 622 | 534 |
| `comp`：行数 / 时间轴对齐 / 缺必需 raw | 192 / 是 / 0 | 192 / 是 / 0 |

SUB_B 写回重读的 schema、行数、时间轴和数值均一致；终态为 0 nonfinite、0 constant、
0 duplicate、0 五口径 IQR outlier、0 `abs(z)>3` residual。Q_REFERENCE 收据确认
`feeds_model=false`，Q_CAUSAL 输入在参考阶段前后 SHA256 均为
`27fb44af8191979410ed2480f1ec8100d9ce840629ca86afcab3a2632f305e9b`。

SUB_A ZIP SHA256 为 `def9a256b569d4efc1ff5053d51c254cb2073b7f5bd4c7a8cb214e0c304ded83`；
SUB_B ZIP SHA256 为 `0bc5cf66e8c5adbf2dece21452bbcd0710d6fb841c9dc45905a43f5411f64b08`。
源、A/B 文件及 A/B ZIP 解压成员共五处 `s_result.csv` SHA256 完全相同。平台 A/B 状态为
`submitted=false`，质量与准确率分数均未填写；这些本地机械量不能推导平台 50/50。

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

## 2026-08-02 初赛提交格式纠正

依据已成功提交的初赛样例，ZIP 契约纠正为根目录依次包含 `input.csv` 与 `s_result.csv`。其中 `input.csv` 是冻结模型在 192 个滚动起点实际使用的因果特征，`s_result.csv` 是原 192×16 预测宽表；模型、预测数值和离线成绩不变。旧的“ZIP 仅含 `result.csv`”约定停止使用。

## 2026-08-02 Strict C0 后完整冲分计划结果

| 路线 | development MAPE | 相对父基线 | 胜折 / 最近 5 折 | 决策 |
| --- | ---: | ---: | ---: | --- |
| S00 global stacking | 5.257643% | -0.002370pp vs C0 | 10/19，4/5 | SCREEN |
| E21 R75 | 5.253764% | -0.006248pp vs C0 | 10/19，3/5 | PROMOTE 临时基线 |
| Price Ridge | 5.288436% | +0.034672pp vs R75 | 9/19，4/5 | STOP |
| Physical X1 5% | 5.420318% | +0.166554pp vs R75 | 4/19，2/5 | STOP |
| **R75 + LGB residual 20% + capacity projection** | **5.229437%** | **-0.030575pp vs C0** | **14/19，4/5** | **PROMOTE** |

最终候选只修改 `generator_1`：t+75 至 t+120 以 E21 作为 R75 基线，其他步长保留 Strict C0，然后统一融入 20% 的冻结 V2 LGB residual 分支；`generator_all` 保持 C0，并执行与生产推理相同的 `0 <= generator_1 <= 200`、`generator_1 <= generator_all <= min(440, generator_1 + 240)` 投影。投影影响 605 个 OOF 单元，但指标没有反转。

冻结后只查看一次 blind，改善 `0.040860pp`。正式运行位于 `results/raw/runs/training/aggressive_r75_lgb20_20260802/`；Production Gate 通过 250/250 个未来扰动案例、92 项测试、192×16 提交校验和确定性 ZIP。全 OOF（含 blind）门禁口径为 `5.266622%`，优于原 Strict C0 的 `5.297932%`，现已晋级 `results/best/`。

## 2026-08-03 RichResidual gas + 30% final 验收

RichResidual 仅学习 `generator_1` 的 Champion 残差，训练标签始终是同一外层折 Champion 的 `actual - prediction`。外层预测只能访问 `origin_time <= train_end` 的历史 OOF；`generator_all` 保留 Champion 路径，最终与 `generator_1` 一起通过正式容量投影。筛选阶段只让 `quantile` 与 `gas` 进入 development，固定 30% 权重优于两段融合和时间前向四段路由后，才运行一次 final。

| 口径 | Champion | `rich_gas_blend_30` | 候选 - Champion |
| --- | ---: | ---: | ---: |
| pooled MAPE（62,858 单元） | 5.266622% | **5.254319%** | **-0.012304pp** |
| `generator_1` | 6.068289% | **6.044234%** | **-0.024056pp** |
| blind MAPE | 5.750015% | **5.729442%** | **-0.020573pp** |

- final 候选赢 12/20 折，最近 5 个开发折赢 3；全样本日块 bootstrap 的候选更优概率为 94.80%，开发期为 91.75%。最终 OOF 与报告位于 `results/raw/runs/experiments/rich_residual_final_gas_20260803/`。
- 生产包装器 `RichResidualAggressiveForecaster` 保存冻结 Champion 和 full-fit gas corrector。`fit_full_rich_residual_corrector()` 的默认行为不使用 blind；只有 final 已确认后的生产命令显式传入 `--allow-confirmed-blind-oof` 才将 blind 标签加入全量重训，收据明确记录该事实。
- 生产运行 `results/raw/runs/training/rich_gas_blend_30_20260803/` 通过 50×5=250 个未来扰动、103 项 pytest、192×16 预测、21 个 raw 字段与无遗留 IQR 越界的 ZIP 校验。Production Gate 使用 `--no-promote`，所以这是可审计候选而不是已发布替代品；`results/best/` 和 `提交这个/` 保持原状。

## 2026-08-03 A50 Ramp Error Atlas、A51 长步长残差与 A52 两段权重

三项实验只读取 `results/raw/runs/experiments/rich_residual_development_b_20260803/oof.csv` 的 development 折；没有打开或读取 98 分答案包，也没有把 blind 用于权重、特征或步长选择。

### A50：真实 ramp 条件下的诊断

对每个 OOF cell 定义 `delta = actual - current_value`，按 `|delta|` 分为 stable `<3`、mild `3–<7`、medium `7–<15`、large `>=15 MW`。这一定义包含未来真实值，只能用于离线误差图谱，不能作为生产 gate。

| ramp 档 | 单元数 | Champion MAPE | RichGas MAPE | RichGas − Champion |
| --- | ---: | ---: | ---: | ---: |
| stable | 19,037 | 2.371463% | 2.445320% | +0.073857pp |
| mild | 5,568 | 8.905123% | 8.766240% | -0.138883pp |
| medium | 3,540 | 17.434962% | 17.136824% | -0.298138pp |
| large | 1,039 | 20.279278% | 20.046415% | -0.232863pp |

`generator_1` 总体从 6.082758% 降到 6.059983%。图谱产物位于 `results/raw/runs/experiments/a50_ramp_error_atlas_development_20260803/`，其中 `cells.csv` 保留逐单元标签，`ramp_atlas.csv` 保留目标×步长×档位统计。

### A52：预注册六组短长权重（拒绝）

短步长为 15/30/45/60，长步长为 75/90/105/120；所有候选相对固定 `rich_gas_blend_30` 比较。

| short / long | pooled 差值 | g1 差值 | 结论 |
| --- | ---: | ---: | --- |
| 20% / 30% | +0.000951pp | +0.001631pp | 拒绝 |
| 20% / 40% | +0.001628pp | +0.003089pp | 拒绝 |
| 20% / 50% | +0.003702pp | +0.007340pp | 拒绝 |
| 30% / 40% | +0.000793pp | +0.001458pp | 拒绝 |
| 30% / 50% | +0.002866pp | +0.005709pp | 拒绝 |
| 40% / 50% | +0.002334pp | +0.004684pp | 拒绝 |

没有候选满足 pooled 和 g1 均不退化的规则，`selection.json` 的 `best_candidate` 为 `null`。结果位于 `results/raw/runs/experiments/a52_two_band_pairs_development_20260803/`。

### A51：g1 长步长 RichResidual（保留）

`RichResidualSpec` 新增 `active_horizons`、`feature_profile` 与可选 Champion 预测特征。A51 固定 `active_horizons=[75,90,105,120]`、`feature_profile=long_horizon`、`feature_groups=quantile,ramp,gas`、`blend_weight=30%`，从 773 个完整因果特征中按静态白名单选择 249 个（包括同步长 Champion 预测）；未训练的步长和所有 `generator_all` 行严格回退到 Champion。

为避免把短步长已有收益丢失，最终比较使用确定性拼接：短步长保留 RichGas，长步长使用 A51。它不是权重搜索。

| 指标 | RichGas | A51 拼接 | 差值 |
| --- | ---: | ---: | ---: |
| pooled MAPE | 5.217769% | **5.211443%** | **-0.006326pp** |
| generator_1 MAPE | 6.059983% | **6.047402%** | **-0.012582pp** |
| t+75 | 5.796659% | **5.787409%** | **-0.009250pp** |
| t+90 | 6.480150% | **6.466507%** | **-0.013643pp** |
| t+105 | 7.073033% | **7.052156%** | **-0.020877pp** |
| t+120 | 7.581496% | **7.574659%** | **-0.006837pp** |

拼接候选赢 10/19 折、最近 5 折赢 2，最差折退化 0.026437pp。日块 bootstrap（38 块、2,000 次）候选更优概率为 95.30%，差值 95% CI 为 `[-0.014459pp, +0.000885pp]`。它达到了 0.005–0.010pp 的保留线，但没有达到 0.010pp 强晋级线，因此不查看 blind、不生产重训、不执行 Promotion。A51 模型 OOF 位于 `results/raw/runs/experiments/a51_g1_long_rich_residual_development_20260803/`，固定拼接 OOF 位于 `results/raw/runs/experiments/a51_g1_long_splice_development_20260803/`。

## 2026-08-03 A53 Perfect Ramp Router 与 A54 前向因果图谱

两项均只读取 A51 的 development OOF 和官方训练数据；A53 明确使用真实未来标签，因此标记为 `oracle_only=true`、`deployable=false`，不能成为候选模型。A54 的真实 ramp 标签只用于结果分层，分位数边界不使用标签。

### A53：真实 ramp 理论上限

| 指标 | RichGas | A53 Oracle | Oracle - RichGas |
| --- | ---: | ---: | ---: |
| pooled MAPE | 5.217769% | 5.213322% | -0.004447pp |
| generator_1 MAPE | 6.059983% | 6.051122% | -0.008862pp |

只允许 `generator_1 × {75,90,105,120}` 改写原始路由值；14,592 个可路由单元中真实 ramp 切换 6,336 个（43.4211%），原始路由审计确认非目标单元修改数为 0，之后统一经过容量投影。Oracle 赢 12/19 折，日块 bootstrap 更优概率为 97.45%，但其 headroom `0.004447pp < 0.005pp`，不满足 A55 的预注册启动条件。

| 真实 ramp 档 | 覆盖 | A51 - RichGas | Oracle - RichGas |
| --- | ---: | ---: | ---: |
| stable | 56.579% | -0.013149pp | 0.000000pp |
| mild | 21.683% | -0.030186pp | -0.030186pp |
| medium | 16.283% | -0.054147pp | -0.054147pp |
| large | 5.455% | -0.043287pp | -0.043287pp |

稳定段也受益，故真实-ramp 硬路由不是合理的上线替代：它主动删除 stable 的 A51 收益，整体甚至弱于 A51 固定长步长拼接。

### A54：每折前向 Q1--Q5 因果信号

图谱限制到 g1 75--120 分钟。10 个因果特征信号使用 `features.index <= train_end`，2 个模型分歧信号使用同一 horizon 的历史 OOF 预测；最早折没有历史 OOF，因此 8 个模型分歧阈值被记录为 `insufficient_history`。其余 334 个阈值均有 `history_max_time <= train_end`，审计违规数为 0。

最清晰的条件关系来自 `|A51-RichGas|`：可分位覆盖 94.737%，Q1 的 A51 - RichGas 为 `+0.000644pp`，而 Q5 覆盖 33.285%，为 `-0.063592pp`，五个分位的差值单调下降。该结果是研究线索而不是阈值选择或拟合结果；由于 A53 未过硬门槛，A55 Logistic Ramp Gate 状态为 `STOP_PRE_REGISTERED_HEADROOM`，没有训练、blind、生产重训或提交改动。

A53 产物位于 `results/raw/runs/experiments/a53_oracle_ramp_router_development_20260803/`，A54 产物位于 `results/raw/runs/experiments/a54_causal_disagreement_ramp_atlas_development_20260803/`。

## 2026-08-03 A58 Strict Forward Disagreement Specialist

A58 是在 A54 线索上重新预注册的最后一个 router 实验。对每个 development fold 的四个 g1 长步长，阈值只由此前折的 `D=|rich_g1_long_blend_30_pred-rich_gas_blend_30_pred|` 历史 OOF 的固定 q80 产生，并且额外要求历史起点不晚于当前折 `train_end`。首折四个步长因历史不足回退；没有使用当前折标签、当前折预测分布、blind 或候选阈值网格。

| 指标 | RichGas | A58 | A58 - RichGas |
| --- | ---: | ---: | ---: |
| pooled MAPE | 5.217769% | 5.212459% | **-0.005310pp** |
| `generator_1` | 6.059983% | 6.049400% | **-0.010583pp** |
| g1-long `t+75--120` | 7.593314% | 7.572148% | **-0.021167pp** |

A58 在可路由 g1-long 单元上切换覆盖 `33.285%`，最近 5 折胜 `3/5`；全体最差折退化 `0.026977pp`，但 g1-long 子集最差折退化 `0.107941pp`，超过固定 `0.1pp` 极端折门槛。故机械验收为 `STOP_ROUTER_SERIES`，`blind_eligible=false`。不执行 blind 确认、不生产重训、不更新正式提交；后续主线切换到 A57 Long-horizon CatBoost Diversity。

## 2026-08-04 A57 Long-horizon CatBoost Diversity

A57 仅使用 A51 splice 的 development OOF（58,368 行、19 折），并且只允许改写 `generator_1` 的 75--120 分钟四个步长。A57a 用原始训练区的绝对 `y[t+h]` 标签；A57b 仅用此前 development folds、同一 horizon 且 `origin_time <= train_end` 的 RichGas OOF 残差。两条分支都固定为 CatBoost MAE / 600 iterations / depth 6 / learning rate 0.03 / 单线程，使用 248 个 A51 long-horizon 静态因果原始字段；没有 blind、Optuna、early stopping 或融合权重搜索。

训练轨迹完整覆盖 152 个 fold x horizon x variant 组合：A57a 训练 76 个模型；A57b 在无历史的首折四个步长回退 RichGas，并训练其余 72 个模型。独立复核确认残差历史越过 `train_end`、held fold 混入和 held 标签使用均为 0；OOF 没有 blind 行，原始路由在两个单体和 14 个固定融合中均未改动非 g1-long 单元。

| 候选 | pooled MAPE | 父模型 pooled MAPE | 差值 | 最近 5 折 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| A57a absolute 单体 | 5.607097% | 5.217769% | +0.389327pp | 2/5 | 拒绝 |
| A57b residual 单体 | 5.321458% | 5.217769% | +0.103689pp | 2/5 | 单体退化 |
| A57b residual + 15% Cat / RichGas | **5.212509%** | 5.217769% | **-0.005260pp** | 3/5 | `RETAIN_DIVERSITY` |
| A57b residual + 20% Cat / RichGas | **5.212731%** | 5.217769% | **-0.005039pp** | 3/5 | `RETAIN_DIVERSITY` |

15% 项在 g1-long 的改善为 `0.020695pp`，其 t+75/t+90/t+105/t+120 分别为 `0.021096pp`、`0.025627pp`、`0.025076pp`、`0.010981pp`，最近 5 个 g1-long 折赢 3，最差折退化 `0.071655pp`。20% 项也达到门槛，因此二者都按原始预注册权重保留，不能以此次结果再选择一个新权重；全部 A51 + Cat 固定融合均未达到 `0.005pp` 保留门槛。

残差相关性显示异构误差来源确实存在：RichGas--A51 为 `0.999510`，A57a Cat--RichGas 为 `0.656488`，但 A57a 校准退化；A57b Cat--RichGas 为 `0.917409`，因而只有低权重残差融合留下约 `0.005pp` 的收益。A57a 停止，A57b 的两个固定候选仅作为后续独立确认的 diversity 分支；不读 blind、不生产重训，也不更新 `results/best/` 或 `提交这个/`。完整收据位于 `results/raw/runs/experiments/a57_long_catboost_diversity_development_20260803/`。

## 2026-08-04 A56 A51 Feature Group Ablation

A56 对 A51 的 249 个冻结字段做七次完整删除，不做逐列删除、子组重排或参数/权重搜索。每个消融都复用 A51 的 `quantile,ramp,gas` 配置、long-horizon profile、同步长 Champion 预测和固定 30% 融合，只允许改写 g1 的 75--120 分钟四个步长。固定保留规则为：相对 RichGas pooled 改善至少 `0.005pp`、相对 A51 pooled 回退不超过 `0.001pp`、最近 5 折相对 RichGas 至少赢 3。

| 删除组 | 删除字段数 | 相对 RichGas pooled 改善 | 相对 A51 pooled 回退 | recent5 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| generation dynamics | 72 | 0.002446pp | 0.003880pp | 3/5 | 拒绝 |
| gas production | 21 | 0.001197pp | 0.005129pp | 3/5 | 拒绝 |
| gas consumption | 56 | 0.003372pp | 0.002954pp | 3/5 | 拒绝 |
| holder/balance | 55 | -0.001170pp | 0.007496pp | 3/5 | 拒绝 |
| quantile/ramp state | 40 | 0.001712pp | 0.004614pp | 3/5 | 拒绝 |
| branch prediction/disagreement | 1 | 0.002083pp | 0.004243pp | 3/5 | 拒绝 |
| time/price | 4 | 0.003194pp | 0.003132pp | 3/5 | 拒绝 |

所有七个组都改善了 A51 的 recent5 表象，却同时失去足够的 pooled 收益；最接近的 gas consumption 仍达不到 `0.005pp`，并超过 A51 回退上限。每组 OOF 都验证 58,368 行唯一 development 键、无 blind，且每个原始候选的非 g1-long 改写数为 0。`retained_stability_candidates=[]`，所以 A56 是停止信号而不是特征筛选结果：不继续逐列、半组或阈值微调，不读 blind、不生产重训、不更新提交。完整收据位于 `results/raw/runs/experiments/a56_a51_group_ablation_development_20260804/`；后续新主线转为 A60 `generator_all` 长步长残差。

## 2026-08-04 A60 generator_all Long Residual

A60 是与 A51 g1-long 独立的第二目标专项：只训练 `generator_all` 的 `t+75--120` Champion 残差，其他 12 个预测 cell 保持 A51 splice。模型规格预注册并冻结为 A51 的 `long_horizon` 因果 profile、`quantile,ramp,gas`、同步长父预测特征、最少 256 条历史 OOF 和 30% 固定融合；没有参数、特征组、权重或 router 搜索。首两折无足够残差历史而回退，剩余 17 折的四个长步长分别独立训练。

| 指标 | A51 splice | A60 | A60 - A51 splice |
| --- | ---: | ---: | ---: |
| pooled MAPE | 5.211443% | **5.203700%** | **-0.007743pp** |
| `generator_all` MAPE | 4.375485% | **4.359999%** | **-0.015486pp** |
| 最近 5 折 | - | 4/5 胜 | - |

全部 58,368 个 OOF 键都属于 development，且没有 blind 行或重复键。原始路由审计确认非 `generator_all × {75,90,105,120}` 修改数为 0；19 个 fold 都保存了训练行数和训练步长收据，严格边界为 `origin_time <= train_end`。同配置验证重跑得到逐字节一致的 OOF（SHA-256 `B76605AC38E4CE99A8B72DB62402B7DC577F23FFF95C2ABB2C146245D96BD7F8`）。

相对 RichGas 的 pooled 改善为 `0.014069pp`，其中包含 A51 已有的 g1-long 增益，故 A60 的正确独立收益口径是表中的 `0.007743pp`。它通过 `0.005pp` 研究保留线和 recent5 条件，状态为 `RETAIN_GALL_DIVERSITY`；但未达到 `0.010pp` 强晋级线，不读取 blind、不生产重训、不更新 `results/best/` 或 `提交这个/`。初次运行位于 `results/raw/runs/experiments/a60_generator_all_long_residual_development_20260804/`，带训练收据的可复现验证位于 `results/raw/runs/experiments/a60_generator_all_long_residual_verification_20260804/`。

## 2026-08-04 A61 Recursive ARX Diversity

A61 使用 A60 verification OOF 的 `a60_gall_long_blend_30_pred` 作为冻结父模型。每个 development fold 对 `generator_1` 和 `generator_all` 各训练一个一步 Ridge ARX（`alpha=20`），递归输出八个 `t+15--120` 预测。每个模型固定使用本目标当前值、lag1/lag2、另一目标当前值、六个已登记煤气状态和八个官方已知未来价格；预测期的唯一反馈是自身已预测值。没有使用 held 实际值、blind、价格路由、特征搜索或参数搜索。

38 个 fold×target 模型均完成训练。所有历史起点满足 `origin_time <= train_end`，每条 `t+15` 标签结束时间都早于 held 起点，越界和 held 标签计数均为 0。候选覆盖两个目标的全部八个步长，并对 standalone 和每个固定融合统一施加生产一致的容量投影。

| 候选 | pooled MAPE | 相对 A60 splice | g1 差值 | gall 差值 | recent5 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ARX 单体 | 5.852764% | +0.649064pp | +0.775654pp | +0.522474pp | 0/5 | 拒绝 |
| 5% ARX | **5.195745%** | **-0.007955pp** | **-0.001819pp** | **-0.014092pp** | 3/5 | `RETAIN_RECURSIVE_DIVERSITY` |
| 10% ARX | 5.193177% | -0.010524pp | +0.001340pp | -0.022387pp | 1/5 | 拒绝 |
| 20% ARX | 5.203187% | -0.000513pp | +0.022902pp | -0.023929pp | 1/5 | 拒绝 |

10% 不能因为总体分数最低而事后成为赢家：预注册保留条件要求 pooled 改善至少 `0.005pp`、recent5 至少 `3/5`、最差折退化至多 `0.100pp`，只有 5% 同时满足（最差折 `0.008925pp`）。ARX 单体虽退化，但与父模型的误差相关性在 g1/gall/pooled 上分别为 `0.904788/0.856765/0.859856`，因此低权重融合有合理的独立收益来源。

两次同输入运行的 58,368 行 OOF 无 blind、无重复键，且 SHA-256 一致：`A5887C57EE4930F452D66FD6A8F231E125AF8B6DA86BA93083C3F8F06EA7C8ED`。5% 项仅作为固定研究分支保留；不继续尝试中间权重、ARX 特征组合或 Price gate，不读 blind、不生产重训、不更新 `results/best/` 或 `提交这个/`。初次和验证产物分别位于 `results/raw/runs/experiments/a61_recursive_arx_diversity_development_20260804/` 与 `results/raw/runs/experiments/a61_recursive_arx_diversity_verification_20260804/`。

## 2026-08-09 P3 Reference-quality Input + Legal Causal Rolling 真实训练

P3 复用 A61 verification 的 19 个 development 折，实际覆盖 3,648 个逐起点预测和 58,368 个目标×步长评分单元；没有使用 blind、平台分数或未来真实特征。完整运行耗时约 39.34 分钟。

| 路线 | pooled MAPE | `1-MAPE` | 相对 A61 |
| --- | ---: | ---: | ---: |
| A61 冻结父模型 | 5.195745% | 94.804255% | — |
| P1 CausalRolling | 5.543435% | 94.456565% | +0.347690pp |
| P2 Matured Residual | 6.575605% | 93.424395% | +1.379860pp |
| P2 Strict Historical Analog | 5.504724% | 94.495276% | +0.308979pp |
| A64 Direct Delta Ridge | 5.328928% | 94.671072% | +0.133183pp |
| P3 cross-fitted static | **5.159141%** | **94.840859%** | **-0.036604pp** |

P3 静态融合在 pooled、目标和 recent5 上有效：赢 `16/19` 折、recent5 赢 `4/5`，`generator_1` 与 `generator_all` 分别改善 `0.029906pp` 和 `0.043303pp`。但 `dev_15` 退化 `0.125153pp`，超过预注册 `0.100pp` 最差折上限，因此 `static_gate.passed=false`、状态为 `STOP_STATIC_FUSION`。该结果不触发未来门禁、生产重训或 Champion 覆盖。

## 2026-08-09 P4 Robust Cross-Fitted Fusion

P4 只读取 `p3_rolling_training_20260809_190558` 的冻结 OOF，不重训任何基础模型。输入验证通过：19 折、3,648 个 origin、58,368 行、无 blind；四份原始路线 OOF 与 integration 的完整键均为 58,368 行完全一致。每个 held fold 的 15 个候选只在其余 18 折上计算原稳定门槛，recent5 按训练侧时间顺序确定，held actual 只参与最终评分。

| 指标 | A61 | P4 robust cross-fit | 改善 |
| --- | ---: | ---: | ---: |
| pooled MAPE | 5.195745% | 5.188403% | 0.007342pp |
| `generator_1` | - | - | 0.005825pp |
| `generator_all` | - | - | 0.008859pp |
| 折胜数 | - | 13/19 | - |
| recent5 | - | 4/5 | - |
| 最差折退化 | - | 0.125153pp | - |

| held fold | 训练侧冻结权重 |
| --- | --- |
| dev_01 | 80% A61 + 10% A64 + 10% Analog |
| dev_02 | 80% A61 + 10% A64 + 10% Analog |
| dev_03 | 80% A61 + 10% A64 + 10% Analog |
| dev_04 | 80% A61 + 10% A64 + 10% Analog |
| dev_05 | 80% A61 + 10% A64 + 10% Analog |
| dev_06 | 90% A61 + 10% A64 |
| dev_07 | 80% A61 + 10% A64 + 10% Analog |
| dev_08 | 80% A61 + 10% A64 + 10% Analog |
| dev_09 | 80% A61 + 10% A64 + 10% Analog |
| dev_10 | 80% A61 + 10% A64 + 10% Matured |
| dev_11 | 80% A61 + 10% A64 + 10% Analog |
| dev_12 | 80% A61 + 10% A64 + 10% Analog |
| dev_13 | 90% A61 + 10% A64 |
| dev_14 | 80% A61 + 10% A64 + 10% Analog |
| dev_15 | 80% A61 + 20% A64 |
| dev_16 | 80% A61 + 10% A64 + 10% Analog |
| dev_17 | 80% A61 + 10% A64 + 10% Analog |
| dev_18 | 80% A61 + 10% A64 + 10% Analog |
| dev_19 | 80% A61 + 10% A64 + 10% Analog |

最终复用原 `static_fusion_gate`，其中 recent5 与目标回归通过，但 pooled 改善未达到 `0.020pp`，最差折退化也超过固定 `0.100pp`。机械结论为 `STOP_STATIC_FUSION`，不是 `ROBUST_STATIC_ELIGIBLE`。运行耗时 `19.435s`；未运行 future perturbation 或 blind，未重训基础模型，未修改 `results/best` 或正式提交。完整 OOF、285 行候选轨迹、逐折选择、报告及哈希位于 `results/raw/runs/experiments/p4_robust_cross_fit_20260809_203500/`。其中 `oof.parquet`、`candidate_trace.parquet`、`report.json` 的 SHA-256 分别为 `1118332ABD201B363DCAAACA00342AF6CA6381F81D6A7A7D554EE08C844998C6`、`51C5771D8B5EF06F17DA5FF45E8A9EF93AABB5686D45FC4B5C95DF26468273B0`、`96C27C733BD0030B8F96A51EBA3216DA4B3CA2CE34F78581ABA48506A6615CD7`。
