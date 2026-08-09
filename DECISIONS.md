# 决策记录

## 2026-08-09 Q4 reference-quality input A/B 包

- 预测来源审计确认：A61 目录只有 development/verification 的 `oof.csv`、报告与训练收据，
  两份 manifest 均为 `stage=A61_recursive_arx_diversity`、`formal_candidate=false`，不存在可
  独立声明为 A61 的生产 `s_result.csv`。Q4 因此使用已经合法提交、且仓库已有平台准确率
  `49.9/50` 记录的 `提交这个_训练优化_复跑/咕咕嘎嘎_gas_predict_prelim.zip` 冻结预测；
  源 ZIP SHA256 为 `65039ac7fd38a23c75a76dcacff79b1230efee07ee201d35ce146c65c7ee1561`，
  其中 `s_result.csv` SHA256 为
  `2dfe7f29cbde9faf846e4a03be292a61eceb93469b199963c565bba2a8c37efe`。不得把该预测
  描述为 A61，也不得改用 aggressive 历史结果冒充 A61。
- `scripts/run_q4_reference_quality_packages.py` 必须调用正式
  `gas_forecast.submission.prepare_submission_chain`。训练 input 由冻结正式模型配置通过
  `align_tables + build_causal_features` 重建，质量统计截止于 `2025-04-30 23:45:00`；评分
  input 读取上述正式 ZIP 的 `input.csv`。SUB_A 使用链内冻结的 Q_CAUSAL input，SUB_B 使用
  其独立副本经过 Q_REFERENCE 后的 input；两个包共享完全相同的冻结预测字节。
- A/B 目录和 ZIP 根目录都只允许 `input.csv`、`s_result.csv`，两阶段收据统一放在包外
  `receipts/formal_chain/`。本地报告明确记录平台分数为 `null`、`submitted=false`；没有明确
  外部授权前不上传，不能由本地零异常门禁推断平台 50/50。
- 正式宽表首次运行暴露 CSV 写回复核的真实缺陷：约 30 万量级数值的一 ULP 十进制解析差异
  （最大绝对差 `2.32830644e-10`、相对差约 `1.94e-16`）被 `rtol=0` 误判为内容改变。
  `_assert_same_frame` 最小调整为 `rtol=1e-15, atol=5e-12`，仍严格检查 schema、时间轴、
  非有限值和冻结文件字节，并增加对应回归测试。

## 2026-07-31 pooled OOF 重构

- 旧逐级选择器保留在 `selection_legacy.py`，默认自动入口切换为 `selection_policy=pooled_oof`。
- 评分统一命名为 `competition_mape`，默认 epsilon 为 `1e-6`，同时报告 pooled、等目标、等目标×步长与旧折均值。
- 每次实验使用独立 run 目录，逐折 checkpoint，默认 8 个外层 worker；中断后可从同一目录恢复。
- 20 折 62,858 个单元格上，V3 pooled MAPE 为 5.3167%，V2/V3 目标路由为 5.3062%；后者成为 M1 冻结胜者。
- 稳定目标×步长 LOFO 经胜率和最小改善回缩后与目标路由完全相同，不采用更自由的 16 单元切换。
- M2 blind smoke 中正则化 simplex 为 6.0724%，OOF 残差 LightGBM 为 8.4022%；完整 20 折完成前不晋级。

## 2026-07-31 初赛路线冻结

- 预测目标采用未来绝对增量，不采用比例变化作为主目标。
- 当前负荷持续性是安全锚点，任何增强模型必须通过前向滚动验证才能偏离锚点。
- 煤气信号作为未来变化的提前量，不进行固定热值的硬换算。
- LightGBM 只修正 Ridge 残差，不直接预测绝对负荷。
- 不预测“停机后输出 0”，改用升档、稳定、降档软概率。
- 初赛不使用 Transformer、LSTM 或 TFT。
- 测试期数据按滚动起点因果使用；测试未来标签不参与模型、阈值、权重或版本选择。

## 2026-07-31 首轮真实盲折版本选择

- V2 在训练期最后 4 天盲折优于 V1，保留为当前默认版本。
- V3 在同一盲折弱于 V2，因此只保留为实验分支，不作为默认提交。
- 最终版本选择必须在更多开发滚动折与盲折上重复该结论；若 V2/V3 未达到多数折胜率，继续回退到前一版本。

## 2026-07-31 补充审查修复

- 新增 V2.5：`generator_rest` 折内 BIC 状态、煤气切换事件和低参数连续软门控。
- 门控训练目标改为严格 OOF 的最优融合系数，不再使用噪声较高的二分类胜负标签。
- 不确定性使用 OOF 分支预测的 MAD，并按目标、按步长取 70%/95% 分位进行连续回缩。
- 未来标签迁移到独立 `targets.py`；`features.py` 增加源码守卫与可用时间审计。
- 三折复核中 V2.5 只赢 1 折且平均 MAPE 弱于 V2，因此默认版本仍为 V2。
- 旧版 JSON 只作为单独导出能力保留，正式 ZIP 仍严格只包含最新 PDF 要求的 `result.csv`。

## 2026-07-31 完整验证冻结

- 使用 20 个非重叠时间块完成 V1/V2 全量滚动验证。
- 晋级条件冻结为：平均更优、共同折多数获胜、盲折不差、最大单折退化不超过 0.3 个百分点、两个目标均不持续恶化。
- V2 在 11/20 个共同折优于 V1，最大退化 0.0483 个百分点，两个目标和全部步长平均值均改善，因此正式冻结 V2。
- 4 月 18 日切换块中 V2 略差于 V1，且两者均弱于持续性；保留为已知风险，不再据此事后修改模型。

## 2026-07-31 自动编排冻结

- 正式入口固定为`python scripts/auto_pipeline.py`，自动执行数据审计、四版本20折验证、未来扰动守卫、机械选型、全量重训、滚动预测、结果校验和确定性ZIP打包。
- 缺失标记列无条件按输入字段创建，字段模式不得依赖未来区间是否出现缺失；实际训练数据的390个特征已通过未来扰动审计。
- V2.5虽有更低总体平均MAPE，但`generator_1`持续退化；V3虽进一步降低平均MAPE，但盲折退化。两者均按预设规则拒绝，正式版本保持V2。
- 自动调参不进入默认提交入口。任何参数搜索必须在独立训练期实验中使用固定滚动折和有限候选集合，测试标签及排行榜反馈不得进入搜索闭环。

## 2026-08-01 M2/M3/M4 收尾

- M2/M3 使用完整 20 折 cross-fitting 后，最佳 `crossfit_simplex_horizon` 为 5.5195%，显著差于 M1 的 5.3062%；OOF 残差 LightGBM 为 7.4808%。按冻结门槛拒绝 M2/M3，不能因单折结果替换正式模型。
- M4 一折真实 smoke 中 CatBoost 为 5.9211%，Gas trajectory 为 6.3353%，均不超过 M1 blind 目标路由 5.8039%；不投入完整 20 折，候选代码保留但不进入提交。
- 最终冻结模型完成 50 起点×5 扰动、250 案例模型级泄漏审计，`passed=true` 且无失败。
- 训练并行改为外层 worker 与树模型线程显式配额，默认按逻辑核心数自动均分；每次实验保留独立 run 目录和 fold checkpoint。
- 当前状态：M1 正式冻结；M2、M3、M4 完成验收并拒绝晋级；泄漏审计和全量测试完成。

## 2026-08-01 结果归档与提交入口

- 历史运行按 `oof`、`comparisons`、`training`、`experiments`、`audits` 分区归档；`results/latest/` 只保存各类型最近一次运行指针。
- 正式最优模型固定在 `results/best/`，只有完整运行、非 smoke、泄漏通过、测试通过、提交校验通过且 pooled MAPE 更低的候选才允许晋级。
- 用户唯一提交入口为 `提交这个/teamname_gas_predict_prelim.zip`；`scripts/show_best.py` 查询当前 best，`scripts/prepare_submission.py` 负责校验和复制，不允许从历史 run 目录手工挑选 ZIP。

## 2026-08-01 P1/P2 目标对齐与在线校准收尾

- Horizon-specific Ridge 完整 20 折 pooled MAPE 为 5.4475%，blind 为 5.9004%；虽然 blind 略优于 V1 的 5.9047%，但 pooled、`generator_1` 和胜折数均不足以替换正式 M1。
- Cold-start 在线 correction gain pooled MAPE 为 5.3543%，但 blind 从 V1 的 5.9047% 退化到 5.9484%；bias 和 vintage 也在 blind 退化，全部拒绝晋级。
- 每折前 96 个 origin 的 within-fold warm-up gain 在同评分子集上从 5.1769% 降到 5.0921%，但 blind 从 6.2366% 退化到 6.3061%；该结果不能称为外部热启动，也不进入正式推理。
- 在线代码保留为严格因果、可复用的 OOF 校准框架；正式模型仍只使用冻结的持久化 M1 路由。
- 收尾审计确认正式提交未变：best ZIP 与用户提交 ZIP SHA256 均为 `0ca59bb6e66004ae95efa64000ecbf81a86bcc988f0559cae33dc0b7e0d7fb27`，`result.csv` 逐字节一致。现有 routed OOF 上的新增 `generator_all <= generator_1 + 240` 约束影响 796/62,858 个单元，MAPE 从 5.3062% 降至 5.3020%，不改变冻结提交。

## 2026-08-01 E21 正式晋级判定

- E21 的 30 天指数衰减相对 E10 核心 Ridge 是成功的低复杂度研究结论：pooled 改善 0.0143pp、`generator_1` 改善 0.0283pp，且对 E10 赢 14/20 折。
- 正式晋级必须比较当前 champion，而非 E10。使用 `results/best/selection.json` 的冻结 V2/V3 路由、当前容量约束、相同 20 折和 62,858 个单元重新训练后，formal champion 为 5.301877%，E21 为 5.301845%，表面 pooled 优势仅 0.000032pp。
- E21 的 `generator_1` 从 6.131131% 退化到 6.131938%，只赢 8/20 折，最近 5 个开发折仅赢 1 折；尽管 blind 改善 0.075110pp，日块 bootstrap 的全样本支持概率仅 48.45%，开发折为 34.68%。
- 决定：`formal_candidate=false`，拒绝将 E21 覆盖到 `results/best/`。保留已训练模型、OOF、泄漏审计和研究代码作为后续时间编码、价格交互、weighted Ridge 的受控起点；正式提交继续使用 M1 V2/V3 routed champion。

## 2026-08-02 严格 C0 重建与研究停止规则执行

- 修复并冻结严格标签边界：最大步长 8 的训练末端距验证起点保留 135 分钟；所有 OOF/research checkpoint 必须通过数据、特征、配置、依赖和切分语义 fingerprint。
- 严格 C0 在 20 折、62,858 个评分单元上得到 pooled 5.297932%，blind 5.790875%；`v2_v3_target_reconciled` 与 LOFO 复核一致，成为当前正式候选。
- E23b、E24、E26、E22、E50、E51 均在 screening 阶段按 pooled/目标/胜折门槛停止。E90–E92 共完成 20 个真实 hot-start 配置，最佳候选仍相对 C0 退化，全部停止，不做 development。
- E25 k40/k80 虽通过 5 折 screening，但完整 development 分别退化 0.031618pp/0.021643pp，bootstrap 支持仅 6.30%/11.45%；拒绝 blind、E25b 和专家路由，避免把筛选噪声带入正式训练。
- Production Gate 通过 250 个未来扰动案例、83 项测试和提交 ZIP 校验；`production_gate_passed=true`，严格 C0 已覆盖 `results/best/`。正式提交只保留 `result.csv`，没有读取测试未来标签或排行榜反馈。

## 2026-08-02 初赛提交格式纠正

- 以已成功上传的初赛 ZIP 为准，正式压缩包根目录固定为 `input.csv` 和 `s_result.csv`，不再使用只含 `result.csv` 的旧约定。
- `input.csv` 必须来自同一冻结模型的实际滚动推理特征，并与 `s_result.csv` 的 192 个时间戳逐行一致；格式纠正不重训、不改预测值，也不使用测试未来标签。

## 2026-08-02 Strict C0 后冲分计划最终决策

- A2.1 双向 split-half Oracle gap 为 `-0.003840pp`，判为 C；因此 S1-S3 只做实现验收，正式 stacking 仅保留 `S00_global` 弱信号（`SCREEN`）。
- E21 只比较 R75/R90/R105；R75 相对同口径 C0 改善 `0.006248pp`，10/19 折获胜、最近 5 折赢 3，按冻结规则晋级为临时基线。
- Price Ridge 相对 R75 退化 `0.034672pp`；Huber 存在未收敛折并被标为无效；Physical X1 的首个 5% blend 退化 `0.166554pp`。两条路线均 `STOP`，禁止继续调参。
- 统一 Diversity 的 `R75 + 20% lgb_residual` 经生产容量投影后，development MAPE 为 `5.229437%`，相对同口径 C0 改善 `0.030575pp`，14/19 折获胜、最近 5 折赢 4；冻结后唯一一次 blind 确认改善 `0.040860pp`，状态为 `PROMOTE`。
- 新候选通过 50 起点×5 扰动、92 项 pytest、提交校验和确定性 ZIP；`production_gate_passed=true`。正式 best 更新为 `aggressive_r75_lgb20`，不再启动第二梯队的大规模 OOF 消耗。
- 成功 ZIP 样板与正式 ZIP 均通过同一契约：根目录 `input.csv`、`s_result.csv`，UTF-8，192 个一致时间戳，结果 16 个预测列。样板特征属于旧模型，不能复制替换当前模型 input。

## 2026-08-03 初赛提交输入质量修复

- 以 574KB 满质量参考包作为回归 oracle 重新审计后，确认原 81 分包的 25 个 raw 字段逐值保留了官方测试原始读数；其中多出 `air_heater_5`、`into_gas_mixed_blast_furnace`、`blast_furnace_user4`、`converter_user1`，且 `air_heater_5`、`converter_user1` 为全零、`into_gas_mixed_blast_furnace` 稀疏，均会造成 raw schema 风险。
- 新增 `submission_quality.py`：固定初赛 21 列 raw schema，保留所有 `feat_` 派生字段，并对已登记高风险字段执行无标签、收敛式批次 `Q1±IQR` 裁剪。参考包只用于验证规则，不进入运行时输入，也不复制其 192 行数值。
- 新增 Q0/Q1/Q2 消融和 ZIP 对照工具；Q2 相比原包移除 4 个未登记 raw 字段、修复 145 个登记异常，raw schema 与参考包一致，预测宽表逐值未变。参考 raw 值差异由 186 个单元降至 107 个单元；剩余连续故障段需要平台质量 A/B 结果后才可安全指定更强的替代估计器。
- 正式 `提交这个/咕咕嘎嘎_gas_predict_prelim.zip` 已重建为 `fd707590aaa54cb3e964e70d1255f2ffa5277a7c399c58aea4068163dced6b41`，通过 192×16、双文件、raw schema、常数列和 IQR 质量校验；本地完整测试为 95 passed。

## 2026-08-03 RichResidual 严格 OOF 验收与生产候选

- 在 Champion `aggressive_r75_lgb20` 的同折 OOF 上，残差标签固定为 `actual - same_fold_champion_prediction`；每个外层折只使用 `origin_time <= train_end` 的历史 OOF。筛选后冻结的唯一配置为 `gas` 特征组与 30% 固定融合，blind 不参与特征组或融合权重选择。
- final 共 62,858 个评分单元：候选 pooled MAPE 为 `5.254319%`，相对 Champion 的 `5.266622%` 改善 `0.012304pp`；`generator_1` 改善 `0.024056pp`，12/20 折获胜、最近 5 个开发折赢 3。一次性 blind 从 `5.750015%` 降至 `5.729442%`，改善 `0.020573pp`；全样本日块 bootstrap 支持概率为 94.80%，开发期为 91.75%。
- `fit_full_rich_residual_corrector()` 默认仍剔除 blind OOF；只有生产脚本显式传入 `--allow-confirmed-blind-oof`，且 final 收据已验证后，才允许将已确认 blind 标签用于一次全量残差重训。该授权会写入 `blind_confirmation.json`、`report.json` 和 manifest。
- 生产候选位于 `results/raw/runs/training/rich_gas_blend_30_20260803/`：50 起点 × 5 类未来扰动全部通过、pytest 为 103 passed、21 raw schema/IQR/ZIP 校验通过，`production_gate_passed=true`。门禁使用 `--no-promote`，因此 `results/best/` 和 `提交这个/` 未被修改；平台尚未上传验证输入质量分，不对平台分数作提升声明。

## 2026-08-03 A50–A52 RichResidual 长步长路线

- A50 在不含 blind 的 `rich_residual_development_b_20260803` OOF 上完成 `generator_1 × horizon × ramp_band` 误差图谱。RichGas 相对 Champion 在 stable 档退化 `0.073857pp`，但在 mild/medium/large 档分别改善 `0.138883pp`、`0.298138pp`、`0.232863pp`。ramp 档定义使用 `|actual - current_value|`，因此结论只用于误差诊断，不能将真实未来 ramp 直接带入生产门控。
- A52 只比较六个预注册短长权重对；最佳 `short30/long40` 仍相对 `rich_gas_blend_30` 退化 `0.000793pp`，且 `generator_1` 退化 `0.001458pp`。全部六对不满足不退化规则，停止 31%/32% 等连续权重微调。
- A51 新增 `long_horizon` 显式因果 profile：249 个字段（含同步长 Champion 预测）、特征来源固定为 g1/gall/rest 历史、holder、煤气产销与余额、可供发电量、分位数、ramp 状态及已知未来电价；仅训练/修正 `generator_1` 的 `75/90/105/120` 分钟残差模型。默认 RichResidual 仍使用全步长/全特征路径，旧 joblib 推理可回退到原有字段行为。
- 固定拼接为短步长保留 `rich_gas_blend_30`、长步长替换为 A51，development pooled MAPE 为 `5.211443%`，相对 RichGas 改善 `0.006326pp`，`generator_1` 改善 `0.012582pp`。四个长步长均改善，分别为 `0.009250pp`、`0.013643pp`、`0.020877pp`、`0.006837pp`。
- 风险门槛未完全通过：A51 仅赢 10/19 折、最近 5 折赢 2，最差折退化 `0.026437pp`；日块 bootstrap 候选更优概率为 95.30%，但 95% CI 为 `[-0.014459pp, +0.000885pp]`。它落在预注册 `0.005–0.010pp` 的“保留”区间，不触发 blind、生产重训或 `results/best/` promotion；下一步仅可在新的预注册实验中研究可观测 ramp-risk/disagreement，不恢复 stacking、全局 Price 或 Physical 路线。

## 2026-08-03 A53–A55 Ramp Router 停止判定

- A53 使用 `a51_g1_long_rich_residual_development_20260803/oof.csv` 的 development 58,368 个单元，只有 `generator_1 × {75,90,105,120}` 的 14,592 个单元允许路由。真实 `|actual-current_value| >= 3MW` 时切换 A51，其余保持 RichGas；真实 ramp 仅作为理论上限，运行记录为 `oracle_only=true`、`actual_ramp_used=true`、`deployable=false` 和 `formal_candidate=false`。
- Oracle 在 6,336 个长步长 g1 单元切换（可路由单元覆盖 43.4211%），经同一容量投影后 pooled MAPE 为 `5.213322%`，相对 RichGas `5.217769%` 改善 `0.004447pp`；g1 改善 `0.008862pp`、12/19 折获胜、日块 bootstrap 候选更优概率 97.45%。这低于启动 A55 的预注册下限 `0.005pp`，因此不允许以该 Oracle 结果启动生产或 blind 路径。
- A53 的分档表还否定了“stable 必须回退 RichGas”的前提：A51 相对 RichGas 在 stable/mild/medium/large 分别为 `-0.013149pp`、`-0.030186pp`、`-0.054147pp`、`-0.043287pp`。真实-ramp 路由丢弃了 stable 的正收益，故其理论上限反而弱于 A51 的全长步长固定拼接。
- A54 仅用官方训练期因果特征与历史 OOF 预测作每折前向 Q1--Q5 分位数。190 个特征历史阈值和 144 个历史 OOF 阈值均已就绪，最早一折的 8 个模型分歧阈值因无历史被显式标为 `insufficient_history`；审计中没有任何 `history_max_time > train_end`，并且真实标签不参与阈值计算。
- A54 的 `|A51-RichGas|` 在可分位的 94.74% 长步长 g1 单元中呈现单调条件关系：Q1 相对 RichGas 为 `+0.000644pp`，Q5（覆盖 33.285%）为 `-0.063592pp`。该信号保留为后续重新预注册的 feature-group ablation 或异构分支诊断线索；尽管它有信息量，也不能绕过 A53 的硬门槛。
- 决定：A55 Logistic Ramp Gate 为 `STOP_PRE_REGISTERED_HEADROOM`。不训练分类器、不选择阈值、不读 blind、不改动 `results/best/`、`提交这个/` 或 98 分参考包。若未来开启新路线，必须重新预注册一个“差异度而非真实 ramp”的 gate 假设，并从 development-only OOF 重新做独立门槛验证。

## 2026-08-03 A58 Strict Forward Disagreement Specialist（router 收尾）

- A58 重新预注册为唯一固定规则：`D=|A51-RichGas|`；对每个 held development fold 和每个 `75/90/105/120` 步长，只从此前 development folds 且 `origin_time <= train_end` 的历史 OOF 计算 q80，当前折不参与阈值。无分类器、无 soft gate、无阈值网格和 blend weight 搜索；历史不足的首折显式回退 RichGas。
- 仅 `generator_1 × {75,90,105,120}` 的原始路由允许切换 A51，其他 12 个 cell 原始值保持 RichGas；所有原始结果统一通过容量投影。阈值收据、逐行 OOF 和 manifest 位于 `results/raw/runs/experiments/a58_forward_disagreement_specialist_development_20260803/`。

| 指标 | A58 相对 RichGas |
| --- | ---: |
| pooled 改善 | `0.005310pp` |
| generator_1 长步长改善 | `0.021167pp` |
| 最近 5 折胜数 | `3/5` |
| g1-long 最差折退化 | `0.107941pp` |
| 切换覆盖（可路由 g1-long） | `33.285%` |

- A58 只有前三项通过；g1-long 最差折超过固定 `0.1pp` 极端折门槛，故 `blind_eligible=false`、`router_series_status=STOP_ROUTER_SERIES`。没有读取 blind、没有生产重训，也没有修改 `results/best/` 或 `提交这个/`。A50–A55/A58 router 线至此永久停止，后续算力转入异构 CatBoost 分支。

## 2026-08-04 A57 Long-horizon CatBoost Diversity

- A57 只在 19 个 development 折的 `generator_1 × {75,90,105,120}` 上运行。A57a 预测绝对目标 `y[t+h]`，A57b 预测同步长 `actual - RichGas OOF` 残差；均固定为 MAE、600 棵、depth 6、learning rate 0.03，使用 A51 `long_horizon` 的 248 个静态因果原始字段，不含 A51 的 Champion 预测合成特征。没有 Optuna、早停、阈值搜索或连续权重搜索。
- 完整训练轨迹有 152 条记录：A57a 的 76 个模型全部训练；A57b 的首折四个步长因无历史 OOF 显式回退 RichGas，其余 72 个模型训练完成。残差历史越过 `train_end`、混入 held fold、使用 held 标签的记录均为 0；58,368 行 OOF 不含 blind，且所有 16 个候选的非 g1-long 原始改写数均为 0。所有候选随后采用同一容量投影。

| 候选 | 相对父模型 pooled | 最近 5 折 | 结论 |
| --- | ---: | ---: | --- |
| A57a absolute 单体 | `+0.389327pp` | `2/5` | 拒绝 |
| A57b residual 单体 | `+0.103689pp` | `2/5` | 单体不作为停止规则，但不保留 |
| A57b residual + 15% Cat / RichGas | `-0.005260pp` | `3/5` | `RETAIN_DIVERSITY` |
| A57b residual + 20% Cat / RichGas | `-0.005039pp` | `3/5` | `RETAIN_DIVERSITY` |

- 两个保留项都只是预注册的固定候选，不能在结果后从 15%/20% 中挑一个新“最佳权重”，也不能与 A51 再做新权重搜索；A51 + Cat 的固定项均未达到 `0.005pp` 保留线。15% 项的 g1-long 改善为 `0.020695pp`，四个步长均改善，最近 5 个 g1-long 折赢 `3/5`，最差折退化 `0.071655pp`。
- 误差相关性审计验证了异构性，但也解释了收益量级：RichGas 与 A51 为 `0.999510`，A57a 与 RichGas 为 `0.656488` 但校准严重失效；A57b 与 RichGas 为 `0.917409`，可用低权重融合带来约 `0.005pp` 的增益。A57a 路线停止；A57b 的两个固定权重只保留给未来独立确认，不读取 blind、不生产重训、不修改 `results/best/` 或 `提交这个/`。

## 2026-08-04 A56 A51 Group Ablation（停止）

- A56 固定删除 A51 249 个字段中的完整预注册组，不做逐列删除、子组重组、权重或参数搜索。七组为 generation dynamics（72）、gas production（21）、gas consumption（56）、holder/balance（55）、quantile/ramp state（40）、branch prediction/disagreement（1）和 time/price（4）；每个候选仍只改写 `generator_1 × {75,90,105,120}`，并走同一容量投影。
- 验收条件预先固定为：相对 RichGas pooled 改善至少 `0.005pp`、相对 A51 pooled 回退至多 `0.001pp`、最近 5 折相对 RichGas 至少赢 3。七项删除均实现 `3/5` recent5，但没有一项保留 A51 的总收益；最接近的删除 gas consumption 仅改善 `0.003372pp`，且相对 A51 回退 `0.002954pp`。删除 holder/balance 还相对 RichGas 退化 `0.001170pp`。
- 运行覆盖 58,368 个 development OOF 单元；7 个组收据均验证无 blind、无重复键、非 g1-long 原始改写数为 0。`retained_stability_candidates=[]`，故 A56 状态为 `STOP_GROUP_ABLATION`：不继续单列、半组或阈值微调，不读 blind、不生产重训、不修改 `results/best/` 或 `提交这个/`。产物位于 `results/raw/runs/experiments/a56_a51_group_ablation_development_20260804/`；下一条独立主线为 A60 `generator_all` long residual。

## 2026-08-04 A60 generator_all Long Residual（研究保留）

- A60 固定复用 A51 的 `long_horizon` 因果字段和容量投影，只将残差目标改为 `generator_all`，并且只允许改写 `75/90/105/120` 分钟。规格冻结为 `quantile,ramp,gas`、同步长 A51 splice 预测特征、256 最小历史行和 30% 固定融合；不做参数、特征、权重或路由搜索。首两折历史不足时显式回退父模型，其余 17 折各训练四个步长。
- 在 58,368 个无 blind development OOF 单元上，A60 相对冻结 A51 splice 的 pooled MAPE 从 `5.211443%` 降至 `5.203700%`，独立增量为 `0.007743pp`；`generator_all` 从 `4.375485%` 降至 `4.359999%`，改善 `0.015486pp`，最近 5 折赢 `4/5`。相对 RichGas 的 `0.014069pp` pooled 改善包含 A51 已有的 g1 收益，不能全部归因于 A60。
- 原始候选审计确认非 `generator_all` 长步长改写数为 0；19 个 fold 训练行数与训练步长均写入 report，strict OOF 规则仍为 `origin_time <= train_end`，且 `blind_labels_used=false`。同配置验证重跑的 OOF SHA-256 与初次运行完全一致：`B76605AC38E4CE99A8B72DB62402B7DC577F23FFF95C2ABB2C146245D96BD7F8`。
- 结论为 `RETAIN_GALL_DIVERSITY`，但独立增量未达到 `0.010pp` 强晋级线，`formal_candidate=false`。因此不读 blind、不生产重训、不修改 `results/best/` 或 `提交这个/`；初次与带训练收据的验证产物分别位于 `results/raw/runs/experiments/a60_generator_all_long_residual_development_20260804/` 和 `results/raw/runs/experiments/a60_generator_all_long_residual_verification_20260804/`。

## 2026-08-04 A61 Recursive ARX Diversity（研究保留）

- A61 以带收据的 A60 splice 为冻结父模型，对 `generator_1` 与 `generator_all` 的全部八个步长各训练一阶 Ridge ARX（`alpha=20`）。输入只含本目标当前值、lag1/lag2、另一目标当前值、六个固定煤气系统状态和对应步长的官方已知未来价格；递归期只回填模型自身预测。固定比较 standalone 与 `5%/10%/20%` 三条融合，不启动 Price route、Optuna、特征搜索或连续权重调节。
- 每折原始一步标签只取 `origin_time <= train_end`，且 `actual[t+15]` 的结束时间必须严格早于 held 起点。38 个 target×fold 模型全部训练，`history_after_train_end=0`、`labels_from_held_fold=0`；58,368 个 OOF 键无 blind、无重复，原始预测未越出两个目标×八个步长的登记范围。

| 候选 | 相对 A60 splice pooled 改善 | 最近 5 折 | 结论 |
| --- | ---: | ---: | --- |
| ARX 单体 | `-0.649064pp` | `0/5` | 拒绝 |
| 5% ARX 融合 | `+0.007955pp` | `3/5` | `RETAIN_RECURSIVE_DIVERSITY` |
| 10% ARX 融合 | `+0.010524pp` | `1/5` | 拒绝 |
| 20% ARX 融合 | `+0.000513pp` | `1/5` | 拒绝 |

- 5% 项将 pooled MAPE 从 `5.203700%` 降至 `5.195745%`；`generator_1` 改善 `0.001819pp`，`generator_all` 改善 `0.014092pp`，最差折退化仅 `0.008925pp`。10% 的总体改善更大但 recent5 不满足预注册 `3/5` 门槛，不能事后改选；禁止继续试 7%/8%/12% 或新 ARX 状态组合。
- ARX 与父模型误差相关性为 g1 `0.904788`、gall `0.856765`、pooled `0.859856`，证明它是不同于树残差的误差结构，但当前只保留研究分支。两次同配置运行的 OOF SHA-256 完全一致：`A5887C57EE4930F452D66FD6A8F231E125AF8B6DA86BA93083C3F8F06EA7C8ED`。`formal_candidate=false`，不读 blind、不生产重训、不修改 `results/best/` 或 `提交这个/`；产物位于 `results/raw/runs/experiments/a61_recursive_arx_diversity_development_20260804/` 和 `results/raw/runs/experiments/a61_recursive_arx_diversity_verification_20260804/`。

## 2026-08-09 FutureRowReconstruction Oracle 隔离

- `FutureRowReconstructionForecaster` 明确冻结为 `ORACLE / DIAGNOSTIC ONLY`：它按
  `origin + horizon` 读取评分期未来生产行，故对未来扰动敏感，`oracle_candidate=true`、
  `causal=false`、`formal_candidate=false`、`deployable=false`。不得将它包装成合法模型，
  不得把评分期未来生产量、未来 generator 真值、blind 标签或平台参考成绩用于训练、
  特征、选择、权重或阈值。
- 正式 `auto_pipeline` 版本白名单不包含该模型；`prepare_submission` 对候选名和元数据
  执行硬拒绝，不能从 Oracle 运行初始化 `results/best` 或写入正式提交目录。Production
  Gate 文件保持不变；即使外部伪造通用收据，Oracle 产物也没有正式文件契约并会被提交
  入口拒绝。
- 研究脚本必须显式传入 `--allow-oracle-research`，且只能写全新的
  `results/oracle/<name>/`；只生成模型、诊断输入/预测、报告和拒绝型 manifest，不生成
  `submission.zip`、不复制到 `results/raw/runs`、`results/best` 或 `提交这个*`。
- 未来扰动测试要求：修改、shuffle、null、删除 origin 之后的全部生产数据时，Oracle
  输出允许变化并必须标注非因果；正式因果模型的 16 个预测仍须逐元素不变，不能用该
  Oracle 结果替代正式泄漏审计。

## 2026-08-09 P3 Legal Causal Rolling Reconstruction（静态融合停止）

- 使用 A61 verification 的 19 个冻结 development 折，对 3,648 个 origin、58,368 个目标×步长单元实际训练 P1 CausalRolling、P2 Matured Residual、P2 Strict Historical Analog 与 A64 Direct Delta；全程不读取 blind 或平台反馈。
- A61 pooled MAPE 为 `5.195745%`；P1、Matured、Analog、A64 分别为 `5.543435%`、`6.575605%`、`5.504724%`、`5.328928%`。leave-one-fold-out 静态融合得到 `5.159141%`，相对 A61 改善 `0.036604pp`，赢 `16/19` 折、recent5 赢 `4/5`，两个目标均改善。
- 预注册稳定性门槛未全部通过：`dev_15` 退化 `0.125153pp`，超过最差折退化上限 `0.100pp`。因此决定为 `STOP_STATIC_FUSION`；不运行后续统一 future perturbation 晋级门禁、不生产重训、不读取 blind、不修改 `results/best/` 或正式提交。
- 可追溯产物位于 `results/raw/runs/experiments/p3_rolling_training_20260809_190558/`，状态为 `OOF_PERFORMANCE_ONLY_FUTURE_GATE_PENDING`；该成绩是研究诊断，不是已发布 Champion。
