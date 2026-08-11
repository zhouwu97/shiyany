# 决策记录

## 2026-08-11 Wave 0+1 重开审计：五枪全部诚实负结果，SAFE60 收敛强化（PROJECT_FROZEN 保持）

- **触发**：FINAL_CONVERGENCE §4 重开条件之"真正低相关新 candidate"检查。重开一条缝，
  只做便宜审计 + 低 DF 专家，SAFE60 冻结不动。全独立 run 目录 + 预注册 gate。
- **五枪结果**：
  1. **Wave 0a horizon atlas**：SAFE60 long(75-120) 6.5715% vs short(15-60) 3.6275%，长 horizon 仍主导误差。
  2. **Wave 0b P0 physical×SAFE60 STOP**：corr 0.324 但 recent5 0.565；standalone g1 +20.01pp
     （rest 分解放大误差）；dwell4 oracle headroom 0.253pp 存在，但 **causal-selective gate −6.46pp**（选 40% 处 physical 只真胜 14.7%）。
  3. **Wave 0c stock-flow R² STOP_CONTENT_BUT_CAUSALLY_UNCAPTURABLE**：完美预见 ceiling ΔR² +0.141
     （transition +0.168），但 OOF 预测未来物理量 ΔR² **−0.068** —— 内容住在不可预测分量。
  4. **Wave 1a P1 target-aligned STOP**：四变体（median/wmedian/Ridge/Huber）g1 long-horizon
     全部 +0.9~1.65pp、residual corr 0.868–0.960。target-clock 已被 SAFE60 long-horizon 组件吸收。
  5. **Wave 1b 价格 hazard 消融 STOP_PRICE_PERMANENTLY_CLOSED**：价格在训练期有真实变化
     （4 值/49% 切换），但对 rest_transition/holder_flip/avail_contraction 三组物理 transition
     预测增益全 ~0（max |ΔAUC| 0.015，Δlogloss 无正）。唯一合法未来信息通道 → 永久关闭。
- **Wave 2/3 全部不触发**：Switching MoE（无候选 + 价格已关）、stock-flow expert（0c 证伪败）、
  Router（无新 expert）、TimeXer（外生线全负）。
- **判定**：重开条件未满足。**SAFE60 收敛被强化**。模型搜索保持关闭，提交资产不动。
- 留档信号：stock-flow ceiling +0.141（transition +0.168）若未来出现新合法外部信息可重新启用；
  物理状态本身可识别（rest_transition AUC 0.685）但价格不增信息。
- 产物：`results/runs/20260811_wave01_summary/SUMMARY.md` 及五个 wave run 目录。

## FINAL CONVERGENCE / PROJECT FROZEN（2026-08-10）

- **Champion = SAFE60**（平台 92.3 = Quality 50 + Accuracy 42.3；g1 .9457 / gall .9581；
  development 5.099520%）。正式提交 `pred1_safe60_submission.zip`（SHA 3e8993d7…）。
- **已关闭路线**：X1（router）、PRED-3（残差校准）、PRED-5（trajectory）、
  PRED-6（target joint）、PRED-7（TCN）、PRED-8A（动态专家）。全部诚实负结果。
- **不再重开搜索**：CatBoost/Ridge 权重、TCN 参数、小于 0.02–0.03pp 的 OOF 抖动、
  动态路由参数、残差校准变体。研究自由度锁定。
- **仅三种情况允许重新开线**：① 新增合法外部信息；② 出现真正低相关的新
  candidate（与 SAFE60 residual corr 显著 <0.7）；③ 比赛数据/评分协议变化。
- **保留依据**：`regret lag1 autocorr = 0.637` 是未来出现低相关专家后重新研究
  routing 的依据，**不代表当前 router 可用**（PRED-8A causal tracker 失败）。
- 非法 Oracle（future-row 89.9 / accuracy 49.9）永久隔离，禁止引用。

## 2026-08-10 PRED-8A Dynamic Opportunity Audit：STOP（因果测试失败，动态专家路线关闭）

- 纯诊断零 ML，四专家（SAFE60/X3/A61/A64），严格 chronological matured-loss replay。
- 三数字：
  1) **Constrained oracle（dwell4/60min）= 4.7032%**（gap 0.396pp vs SAFE60 5.0995%）；
      raw origin oracle 4.5825、dwell2 4.6441、dwell8 4.7778。
  2) **Regret 自相关 lag1 均值 0.637**（X3 .615/A61 .621/A64 .676），lag4 衰减到 ~0.2；
      winner dwell 中位数 1、均值 2.2、max 24；X3/A61 自转移 0.54/0.58。
  3) **Delayed tracker（trailing 30/60/120/240min + EMA0.25）全部不敌 SAFE60**：
      best 5.1061%（+0.0066pp worse）。
- **判定：STOP**。按 §3.12，Strong/Weak GO 均要求 tracker 正收益（≥0.03 / 0.01–0.03pp），
  tracker 反而更差 → 未达任何 GO。regret 持续性强（0.637）是真实信号，但 trailing/EMA
  loss 选 expert 噪声太大（16-cell APE 方差、winner dwell 中位 1），因果上不可靠预判
  下一 origin 赢家。**X1 模式再现：Oracle 大 + 持续性有，合法因果利用失败。**
- 结论：动态专家路线（PRED-8/9/10）关闭，不进入复杂 router。与 X1、PRED-3/5/6/7
  共同证据：SAFE60 在合法信息集合下的信号空间已基本榨尽。
- 保留：regret 持续性数字入档，若未来出现真正低相关新 candidate 再考虑；否则不再
  为 0.02–0.03pp 潜在收益重开动态线。

## 2026-08-10 战略收敛：SAFE60 为最终方案，模型搜索结束

- **PRED-7 Small TCN：STOP**（全 19 折）。严格 causal dilated conv（3 block、
  hidden 32、~10k 参数、L1、严格 forward）：standalone 0.0591（+0.81pp vs anchor
  0.0510）、**residual corr vs SAFE60 = 0.9253**（高）、4 个预注册 blend 全不改善
  （0.0511~0.0515）。按策略 stop 条件（"TCN 与 SAFE60 residual correlation 再次
  达到非常高的水平 → 深度模型方向基本结束"），深度方向判死。
- **PRED-6 Target Joint：STOP**（rest=gall−g1 分解不敌直接 gall，前条记录）。
- **收敛触发**：PRED-6 与 PRED-7 双败，符合战略 §十二 停止规则。
- **四条结构性路线全部诚实负结果**：PRED-3（残差校准）、PRED-5（trajectory）、
  PRED-6（target joint）、PRED-7（TCN）。四者共同证据：**现有合法信息集合下的
  tabular/时序/结构表示均已高度相关（residual corr 0.77–0.97），SAFE60 已捕获
  绝大多数可利用信号**。
- **正式决策**：SAFE60 = `acc 42.3 / g1 .9457 / gall .9581 / quality 50` 为当前
  数据、验证框架和合法信息集合下的**局部性能上限**，作为最终提交方案。**结束
  模型搜索**；不重启旧模型族微调、不为 0.005pp 重开无限搜索。
- 剩余可做（非模型）：真正的新合法外部信息/业务变量、完全不同的任务建模；
  或等待平台出现新的外生证据后按预注册重开单一路线。
- 提交资产冻结：`pred1_safe60_submission.zip`（SHA 3e8993d7...），与 R1
  （input SHA 23330d3c...）构成最终提交。

## 2026-08-10 PRED-6 Target Joint Structure：STOP（rest 分解不敌直接 gall）

- 关系验证：`rest=gall−g1` 恒正、占 gall 75.7%，`corr(g1, rest)=−0.21`（联合结构
  存在），但 rest persistence MAPE 极差（h15 80%，近零值主导）。
- PRED-6A（SAFE60_g1 冻结 + 独立 rest delta 模型重构 gall）严格 forward 19 折：
  ridge pooled **−0.0045pp**、et **−0.0010pp**，gall 均更差（0.0512/0.0443 vs
  anchor 0.0423）。SAFE60_g1 + rest 重构不敌 SAFE60 直接预测 gall。
- **判定：STOP**。rest 分解未提供超过 SAFE60 直接建模的新信息（SAFE60 已隐式
  捕获 g1-rest 交互）。不拆 20 种 target transformation。
- 连续负结果：PRED-3（残差校准）、PRED-5（trajectory）、PRED-6（target joint）
  —— 现有 tabular 模型族信号接近饱和。
- 下一步：**PRED-7 Small TCN**（最后的结构性赌注，严格 causal conv、<100k 参数、
  评价核心 = residual correlation + 4 个预注册 blend）。

## 2026-08-10 后 SAFE60 阶段战略冻结（PRED-6 → PRED-7 → 收敛）

- **Champion 冻结**：SAFE60 = 平台确认 `acc 42.3 / g1 .9457 / gall .9581 / quality 50`，
  永久作为生产锚点。禁止再调 0.60/0.40、残差比例或小范围参数。
- **问题转型**：从"精度"转向"残差相关性"。PRED-3/5 负结果（残差相关 0.77–0.97）
  证明现有模型族共享同一误差结构；唯一出路是**真正解相关的信息源**。
- **两条主实验（≤2）**：
  1. **PRED-6 Target Joint Structure**（立即）：`rest = gall − g1`，研究 rest 是否
     比 gall 更稳/更可预测/更低相关；6A 独立 rest 模型、6B rest delta。baseline
     必须是"隐式 rest"（`SAFE60_gall − SAFE60_g1`），否则只是换表达复现。
  2. **PRED-7 Small TCN**（PRED-6 后）：严格 causal conv、2–4 block、参数 <100k、
     输出 16 cells；评价核心是 `corr(e_TCN, e_SAFE60)` 而非 standalone；只测
     95/5、90/10、85/15、80/20 四个预注册 blend。
- **冻结路线**：PRED-R0/R1（regime，重启需跨 forward fold 稳定的可提前识别 regime）、
  PRED-2/X1-v2（候选池无低相关新模型则不重启）、PRED-3/5 保持 STOP。
- **晋级门禁（5 层）**：严格合法性 → 相对 SAFE60 pooled ≥0.02pp 或明显低相关
  residual + 保守 blend 同级改善 → recent5 ≥3/5 → 双 target 安全 → 复杂度惩罚
  （模型越复杂 margin 要求越高）。Tier S≥0.02 / A 0.01–0.02 / B 0.005–0.01（需低
  相关或单 target 大幅或 recent 极稳）/ C<0.005 默认 STOP。
- **acc 期望修正**：平台曲线斜率 ≈1.36 分/1%MAPE；0.02pp OOF ≈ +0.03 acc。
  PRED-6/7 现实预期 42.3→42.4~42.5，**不是 42.6**。OOF gate ≥0.02pp 是触发条件，
  不把平台分锚到 42.6。
- **停止规则**：若 PRED-6 与 PRED-7 均诚实失败 → 正式承认 SAFE60 为当前信息边界
  下的局部最优，结束模型搜索，不重启旧模型族微调。

## 2026-08-10 PRED-5 PCA/MultiOutput Trajectory：STOP（核心假设未兑现，诚实负结果）

- 8 步 delta 轨迹，严格 forward 19 折，6 主实验（≤6 网格）：MultiOutput
  Ridge/ExtraTrees + PCA(2/3/4)×Ridge/CatBoost。SAFE60 为锚（pooled 5.0995）。

| kind | standalone | residual corr | blend05/10 |
| --- | --- | --- | --- |
| ridge | +1.32pp | 0.768 | 0.0510/0.0511 |
| ExtraTrees | +0.13pp | 0.965 | 0.0509/0.0509 |
| pca_2/3/4_ridge | ~+1.3pp | ~0.77 | 0.0510/0.0511 |
| pca_3_cat | +0.16pp | 0.970 | 0.0510/0.0509 |

- **判定：STOP**。三条保留规则全不满足：
  1) standalone 超 +0.15pp（ridge/pca 远超；et +0.13pp 勉强但 corr 不过）；
  2) 残差相关 0.77–0.97 **太高**（trajectory 与 SAFE60 误差结构重合，无新多样性）；
  3) 5–15% blend 无稳定改善（净 ≈ 0）。
- **结论**：共享低维 trajectory 假设未产生新残差结构——SAFE60（=0.6X3+0.4A61）
  已捕获 delta 轨迹的绝大部分信号。与 A57a（horizon 独立）不同，这次是
  "shared trajectory" 方向也失败，进一步确认现有模型族对轨迹信号的覆盖接近饱和。
- 不扩大网格（PCA 上限 6 主实验已守）。不保留 specialist（corr 过高无多样价值）。
- 下一步候选：PRED-6（target joint structure）或 PRED-7（小型 TCN），或回归
  PRED-R0/R1（regime-conditional）——均为低优先级/投机，需重新预注册。

## 2026-08-10 PRED-3 Matured Residual Calibration：STOP（未达 gate，诚实负结果）

- SAFE60 为锚，严格 forward 评估多路残差校准：
  - EWMA 全量修正（λ=1）：pooled **−1.41pp**（EWMA 短期噪声不可榨）。
  - λ 收缩（0.10/0.20）：**−0.01pp / −0.055pp**（仍负）。
  - 每 (target×horizon) 常数偏差修正（λ=0.5，全历史）：pooled **+0.008pp**、
    HIGH_STABLE hard +0.0104pp、regime-weighted **+0.0178pp**，但 **recent5 2/5、
    fold 8/19**（近期折反转）。
  - per-target bias / 近 5/3 折 bias 变体：+0.004pp ~ −0.010pp，均更弱。
- **系统性偏差存在**（SAFE60 低估：g1 +0.26、gall +1.02，随 horizon 增大），
  但校正量级 < 0.01pp 且 fold 不稳定。
- **判定：STOP**。pooled +0.008pp 低于生产 bar（≥0.01pp），recent5 2/5 明确反转
  （用户规则：recent folds 不反转才生产化）。不生产、不调 λ/窗口继续磨。
- 残差 bias 特征（每 target×horizon 常数偏差）保留为 PRED-4 的候选特征（PRED-4
  需 PRED-2 probe 通过后才启动）。
- 下一步：转 **PRED-5 PCA/MultiOutput trajectory**（第二增益源）。

## 2026-08-10 PRED-1 平台验证：Success A PLATFORM_IMPROVEMENT_CONFIRMED

- 平台对 `pred1_safe60_submission.zip` 真实打分：**quality 50.0/50（全细项满分）**、
  **acc 42.3/50**、`1mape_1=0.9457`、`1mape_all=0.9581`。
- 对照 PRED1_PLATFORM_BASELINE_V1（acc 42、g1 0.9448、gall 0.9546）：
  - quality 保持 50/50（R1 input 未动，字节一致）；
  - acc **42.0 → 42.3**（+0.3）；
  - g1 1-MAPE **0.9448 → 0.9457**（MAPE 5.52% → 5.43%）；
  - gall 1-MAPE **0.9546 → 0.9581**（MAPE 4.54% → 4.19%）。
- **判定：Success A `PLATFORM_IMPROVEMENT_CONFIRMED`**（quality=50 且 acc>42）。
- **转移验证**：gall 改善 0.35pp > g1 0.09pp，与 PRED-R0 transfer 分析一致
  （SAFE60 优势集中在 gall）；OOF 0.096pp 优势真实迁移到平台。
- **纪律**：平台结果只记录，不反向调权重/参数/阈值。SAFE60 0.60/0.40 保持冻结。
- 状态：`PRED1_PLATFORM_VERIFIED_SAFE60_CONFIRMED`。
- 下一步：按 Priority，PRED-3（matured residual, gall 主线）/ PRED-5（PCA trajectory）。

## 2026-08-10 PRED-1 上传包冻结 + 3-origin 定向复核全绿（待平台上传）

- 3-origin 定向复核（2025-05-02 10:15 g1、13:15 gall、21:45 g1）全 PASS：
  1) 前向填充 provenance：3 个 origin 的 current 全部 = `< origin` 最近可见值
     （10:00/13:00/21:30）；R1 重建值与之相同系 R1 当初亦用 LOCF，本链直接由
     测试数据 ffill 计算、不读 R1。
  2) 截断重放：full-context 全链预测 vs 冻结 s_result，repro_max = 5.7e-14。
  3) 扰动不变：truncated/perturbed 三上下文 6/6 checks = 0.0~1.4e-14（全部 ≤1e-10）。
  4) blind manifest：`blind_used_for_selection=false, confirmed_blind_oof_used_for_refit=true,
     post_blind_tuning=false`（在 E3/E4 receipt）。
  5) R1 input 冻结：SHA `23330d3c...`（平台已验证 50/50，字节级未改）。
- **上传包冻结**：`results/raw/runs/audits/pred1_e34_scoring_20260810/pred1_safe60_submission.zip`
  - input SHA `23330d3cfdb68618e2f878cb4325e3e121ea7d741941ca73f8d6b363ba230aef`
  - s_result SHA `a73ded1812eb223a156870eee62affbdc2b25df0335443b08ac838bf24a32284`
  - ZIP SHA `3e8993d76baa16e6a62515ddb556170fcf93bfb43651c6da5a543c71df0a6fb6`
  - 两成员与冻结源字节一致；ZIP 只含 input.csv + s_result.csv。
- **上传后只看四数**：quality 保持 50/50、1-MAPE_g1 > 0.9448、1-MAPE_gall > 0.9546、
  acc > 42。平台结果只记录，不反向调权重/参数。

## 2026-08-10 Gate E 完整通过：SAFE60 production runner 全门禁绿（PRE-PRODUCTION → PRODUCTION-ELIGIBLE 就绪）

- **E0 seed contract**：9/9 测试 PASS（replay=fold_position 0-18，production=slot 100，
  未知 cutoff FAIL CLOSED，seed 是工程确定性参数禁止扫描）。
- **E1 六层 production**：RichGas/A51/splice/A60/A61/X3 全部 fit-once + predict，
  dev 折逐位复现（max|diff|=5.7e-14）。抓出 3 个真实 bug：X3 imputer 一致性、
  A51 corrector baseline=aggressive（非 rich_gas_blend_30）、splice 仅 g1 参与
  （gall 保持 baseline）+ 输出投影。
- **E2 全链端到端 replay：19/19 折 PASS**，max|diff| ≤ 1.1e-13、correlation=1.0、
  pooled 与冻结 OOF 完全一致。空 history 折回退 baseline（dev_01）。
- **E3/E4 最终 fit + 评分推理**：final cutoff 2025-05-01 00:00、seed slot 100；
  A61 final cutoff=first_held-30min（one-step 标签成熟边界）；RichGas 用 final OOF
  （含 confirmed blind，31429 history 行），A51/A60 用 dev OOF（无 final OOF 存在）；
  missing_current=causal_forward_fill（3 个隐藏 origin）。输出 3072 cells × 192
  origins 全 finite。
- **E5 future perturbation**：32/32 特征检查零失败（8 origins × extreme/shuffle/
  null/delete），证明评分特征对 origin 后数据完全不变。
- **E6 s_result 冻结**：SHA256 `a73ded1812eb223a156870eee62affbdc2b25df0335443b08ac838bf24a32284`，
  192 rows、schema 与 R1 冻结 s_result 一致。
- **结论**：SAFE60 production 链无语义漂移（逐位复现），状态从 DEV-ELIGIBLE 升级为
  PRODUCTION-ELIGIBLE（待平台上传验证）。产物：
  `results/raw/runs/audits/pred1_e34_scoring_20260810/`。
- 下一步：R1 打包（冻结 s_result + R1 input → ZIP）→ 平台一次上传，比较 acc/1-MAPE。

## 2026-08-10 Gate E3/E4 政策拍板（Pre-Production 生产边界）

### E3 Blind 政策：允许纳入最终 refit（单向冻结）

- SAFE60 六层（RichGas/A51/A60/A61/X3）最终生产 fit 允许
  `confirmed_blind_oof_used_for_refit = true`，前提是 confirmed blind 是训练期
  内部已知标签块，**不是**官方评分期/测试期标签。
- **单向冻结约束**：模型结构/特征/超参/路由/fallback/blend 权重先全部冻结 →
  blind 仅做最后一次确认 → 全训练区间 refit（含 confirmed blind）→ scoring。
  refit 后禁止再根据 blind 指标改任何配置。
- **报告语义**：blind/OOF 性能必须保留 pre-refit 冻结模型的版本；不得拿 refit
  后模型重算同一批标签冒充 blind 性能。
- **stacking 语义**：元模型需要的输入中，blind 只能以其严格 OOF 预测进入，不能
  喂 in-sample base prediction（本链各层 parent 列即 OOF 预测，天然满足）。
- manifest 字段：
  `confirmed_blind_labels_used_for_selection = false`
  `confirmed_blind_oof_used_for_refit = true`
  `post_blind_tuning_allowed = false`

### E4 隐藏 current 政策：causal forward-fill（禁止 R1 重建值）

- 3 个隐藏评分 origin（2025-05-02 10:15 g1、21:45 g1、13:15 gall）的 generator
  current 缺失。测试集 `generator_use_*` 是煤气消耗量（不同量纲），**无确定性
  恒等式**可恢复（gall≠g1+分量和，gall/g1 比值 1.0–8.7 无固定关系）→ 确定性
  恢复优先级不适用。
- 正式 scoring 默认 `missing_current_policy = causal_forward_fill`：只读取
  `< origin` 最近可见观测 LOCF，不使用未来值、不使用 R1 重建值。
- **R1 reconstructed current 仅作 diagnostic/reference channel，禁止进入正式
  scoring inference**（即使历史 aggressive baseline 用过，不为极小增益增加合规耦合）。
- E5 perturbation 增加专属 case：删除/改掉 origin 后全部 generator 值，验证
  3 个隐藏 origin 的 current reconstruction 与最终预测完全不变（证明 E4 无未来偷看）。
- audit 记录：`affected_origins=3, uses_future_data=false, uses_reference_reconstruction=false`。

### 冻结边界

- fallback 修复（空 history → baseline）已并入 production_runner.py；dev_01（空
  history）+ dev_19 已 PASS（机器精度）。**完整 19 折 E2 重跑 = 新的冻结边界**，
  若 19 折结果改变，先据 19 折重新决定 SAFE60 是否成立，再进 blind/E3。

## 2026-08-10 SAFE60 development 定义永久冻结（DEV-ELIGIBLE / PRE-PRODUCTION）

- 基于 B1/B2/C/D 四重 PASS（X3/A61 逐字节 replay、SAFE60 5.099520 精确复现、
  P=1.0、19/19 fold、regime-weighted +0.0828pp/P=0.9998），SAFE60 development 定义
  **永久冻结**：`safe60 = 0.60×X3_cat_mae + 0.40×A61_recursive_blend_05`，
  状态升级为 `DEV-ELIGIBLE`（等待 Production Gate E），**不是** production champion。
- **Gate E 失败归因规则（写死）**：Gate E 失败只能说明 production implementation /
  replay semantics 有问题，不是 development model 有问题。Gate E 失败时**禁止**：
  `0.60→0.65` 调权重、换 seed 重试、重新调 CatBoost 参数、改 A61 参数。
- **Seed contract（Gate E 预注册）**：
  - `mode=replay`：`seed_offset = frozen_fold_position×1000 + target_idx×100 + horizon_idx`，
    cutoff 必须精确匹配某 dev fold 的 train_end；用途 = OOF reproduction。
  - `mode=production`：冻结 `PRODUCTION_SEED_SLOT = 100`（独立命名空间，不复用任何
    fold position 0–18），`seed_offset = 100×1000 + target_idx×100 + horizon_idx`。
  - **seed 是工程确定性参数，不是超参：Gate E 内禁止任何 seed 扫描（含跑
    18/19/100 看哪个好）。**
  - 确定性层（ARX / Ridge / 线性）不人为塞随机种子。
  - 该 contract 对所有依赖 fold_position 的随机 learner 统一生效。
- 提交切点：本 commit 冻结 dev-side 全部证据；Gate E 从干净 HEAD 单独形成第二个
  milestone。

## 2026-08-10 PRED-1 v4.1 regime 三口径报告：SAFE60 全口径胜出（Gate C/D 完整通过）

- PRED-R0 harness 在重放 SAFE60 合并帧上输出三口径（阈值只在 `< dev_01` 冻结：
  bf_total_hi 1,682,755 / ramp_lo 0.0433；density 模型也只 fit `< dev_01`）。
  temporal_support OK（matched_days=37、day_ESS=23.6≥5、cell_ESS 15,522）。

| 候选 | full dev | regime hard | regime weighted |
| --- | --- | --- | --- |
| SAFE60 | 5.0995 | 4.9306 | **4.9110** |
| A61 | 5.1957 | 5.0002 | 4.9932 |
| aggressive | 5.2294 | 5.0533 | 5.0338 |

- 主口径 weighted bootstrap：SAFE60 vs A61 = **+0.0828pp（P=0.9998）**；vs aggressive =
  **+0.1236pp（P=1.0）**。三个口径均无结构退化（无回归，不触发 robustness cap）。
- **结论：SAFE60 5.099520% 在 full / regime-hard / regime-weighted 三个口径全部为正且
  bootstrap P≥0.95，满足 v4.1 §4 主 gate（full dev）+ robustness（regime）双重判据。
  Gate C/D 完整通过。** 之前担心的"评分子分布上收益消失"未发生——proper 密度加权下
  收益从 0.096pp 缩至 0.083pp（vs A61），仍然稳健正。
- 产物：`results/raw/runs/audits/pred1_gate_c_20260810/regime_report/`。
- 下一步：Gate E final fit once + production replay + future perturb。

## 2026-08-10 PRED-1 Gate C/D SAFE60 Replay：PASS（三基线全绿）

- 用重放 X3 OOF + 重放 A61 OOF 实时构造 `safe60 = 0.60×X3 + 0.40×A61`（线性恒等
  max_abs_diff=0.0），不读外部 blend 列。结果与冻结 `.tmp/pred1_gate_c_frozen_validation.json`
  **完全一致**：

| 判据 | 要求 | 实测 |
| --- | --- | --- |
| SAFE60 pooled | 5.099520 ±0.005pp | **5.099520** |
| vs A61 | ≥0.050pp | +0.0962pp |
| vs aggressive | ≥0.020pp | +0.1299pp |
| bootstrap P | ≥0.95 | **1.0 / 1.0** |
| fold 胜 A61 / aggressive | — | 19/19, 17/19 |
| recent5 vs A61 | ≥3/5 | 5/5 |
| worst fold regr vs A61 | ≤0.100pp | **−0.0016pp**（无折退化） |
| 两 target | 均改善 | g1 5.9736 vs 6.0456, gall 4.2254 vs 4.3459 |

- **结论：SAFE60 5.099520% 在 main-repo replay 链上成立，Gate C/D 通过**。研究基线
  （A61）与 promotion 基线（aggressive dev-contract 5.229437）双口径均显著胜出。
- 产物：`results/raw/runs/audits/pred1_gate_c_20260810/`（safe60_gate_report.json、
  merged_safe60_eval.csv）。
- 下一步：v4.1 regime 三口径报告（PRED-R0 harness，运行中）→ Gate E final fit once +
  production replay + future perturb。

## 2026-08-10 PRED-1 Gate B2 A61 Fold Replay：PASS（逐字节复现）

- 主仓 `scripts/run_recursive_arx_diversity.py` + `code/gas_forecast/recursive_arx.py`
  以 A60 verification OOF 为父输入、冻结 config（hash 160fa9f4…）重跑 A61，输出与
  冻结 `a61_recursive_arx_diversity_verification_20260804` OOF **逐字节一致**：
  pooled `5.195745%`、g1 `6.045583%`、gall `4.345907%`、19 折全部 diff =
  0.000000pp；OOF SHA256 `A5887C57…` 与 DECISIONS 记录完全一致。
- **结论：A61 production-capable runner 语义成立**（此前 manifest git_commit 不可靠
  问题不影响 OOF 复现；HEAD 代码即最终判据）。Gate B2 通过。
- 产物：`results/raw/runs/experiments/pred1_a61_replay_20260810/`。
- 下一步：Gate C SAFE60 Replay + 双 baseline + bootstrap（运行中）。

## 2026-08-10 PRED-1 Gate B1 X3 Fold Replay：PASS（逐字节复现）

- 主仓 `scripts/run_mape_aligned.py` + `code/gas_forecast/mape_aligned.py` 在冻结
  `x3_config.json`（CatBoost MAE / 100 iter / depth 6 / lr 0.05 / has_time / seed
  20250731+offset）上重跑 19 折，输出与 worktree
  `20260809_233810_677` 冻结 OOF **逐字节一致**：pooled `5.119696%`、
  g1 `6.005004%`、gall `4.234387%`、19 折全部 diff = 0.000000pp；OOF 文件
  SHA256 `bc2718e4…` 完全相同。replay 状态 `RETAIN_MAPE_ALIGNED`、
  248 列、912 训练记录、label_maturity PASS、future perturb 12/12。
- **结论：X3 5.119696% 可复现性彻底确认**（此前 asset audit 标记的无模型文件、
  manifest git_commit 不可靠问题均不影响 OOF 复现）。Gate B1 通过，容差远优于
  ±0.005pp 要求。
- 产物：`results/raw/runs/experiments/pred1_x3_replay_20260810/`。
- 下一步：Gate B2 A61 Fold Replay（运行中）。

## 2026-08-10 平台真实反馈：Input 50/50 + Prediction 42/50 基线登记

- 平台对 `R1_EXACT_REFERENCE.zip` 真实打分：Input Quality **50/50**（miss
  10/10、dup 5/5、out 5/5、intv 5/5、invalid_col 5/5、feat 5/5、comp 15/15），
  Prediction Acc **42/50**。`s_result.csv` SHA256 `e0f471d873d67c1894...`
  （= R1 final 链中 R1_EXACT_REFERENCE_CLONE/s_result.csv，字节级等于
  aggressive_r75_lgb20 生产冻结预测）。→ R1 正式封版为
  `PLATFORM_VERIFIED_INPUT_50_OF_50`，此后禁止再改 input。
- 平台：g1 `1-MAPE=0.9448`（MAPE 5.52%）、gall `1-MAPE=0.9546`（MAPE 4.54%），
  两目标简单平均 5.03%。本地 3000-cell diagnostic 与平台 g1 差 0.043pp /
  gall 差 0.038pp → 本地诊断可作 blind-independent sanity check，但禁止反推
  最后 72 单元标签。
- **取消任何 MAPE→50 分线性换算**（50×0.95≈47.5 与平台 42/50 不符）。以后只
  有两类数字：本地研发指标（MAPE/ΔMAPE/fold wins/bootstrap）与平台验证指标
  （acc/1mape_1/1mape_all），绝不自行转换。
- PRED-1 新增 `results/raw/runs/audits/pred1_asset_audit_20260810/PRED1_PLATFORM_BASELINE.json`
  作为唯一 platform baseline receipt；只记录，禁止用于调 X3 权重 / CatBoost
  参数 / router threshold / PCA components。PRED-1 成功标准：门禁全过后一次
  上传，Case A acc>42 → PLATFORM_CHAMPION；Case B acc=42 但 1-MAPE 提升 → 仍
  为预测 champion；Case C 1-MAPE 不变 → 停止调 SAFE60；Case D 明显变差 → 调查
  链路，禁止改权重重试。
- 优先级进一步明确：PRED-1 > PRED-3（matured residual，重点 g1 长周期）>
  PRED-5（PCA trajectory，重点 g1）> PRED-2（廉价 probe）> PRED-4。

## 2026-08-10 PRED-1 Gate A 资产核验完成（Asset Audit PASS）

- 完成三项锁定并产出 `results/raw/runs/audits/pred1_asset_audit_20260810/`：
  `PRED1_ASSET_AUDIT.json`、`x3_feature_schema.json`（248 列冻结）、
  `PRED1_PLATFORM_BASELINE.json`、`x3_config.json`（X3 冻结配置）、
  `scripts/pred1_check_x3_schema.py`（fail-closed 校验）。
- **248 vs 249 已定案**：差异恰为一列 `feat_champion_prediction`（A51 同步长
  Champion 预测，branch_prediction/disagreement 组）。X3 的 248 =
  `_long_horizon_feature_candidates()`（不含 champion 列），A51 的 249 = 248 +
  champion 列（仅 `include_champion_prediction=True` 时追加）。无需删列。
- **X3 冻结身份已锁定**：experiment `20260809_233810_677`，git_commit 334318a，
  19 folds × 192 origins × 16 cells = 58368 行；CatBoost MAE **100 iter / depth 6 /
  lr 0.05 / has_time / seed 20250731+offset**（注意：计划正文写的 600 iter/lr 0.03
  是通用 ForecastConfig 默认，非 X3 冻结参数）；parent = `a61_recursive_blend_05_pred`；
  248 列 long_horizon；数据/特征/OOF 哈希全部落盘。**无任何模型文件**。
- **aggressive 契约已定案**：aggressive 训练 OOF 是 blind+19dev 的 20 折混合，
  results/best pooled 5.2666% 含 blind；但 dev 折预测与 X3 OOF 中
  `aggressive_r75_lgb20_pred` 逐位一致（max diff 8.5e-14，0/58368 行不同）。
  故 promotion 可直接在 SAFE60 19 折契约上比较：aggressive 5.229437% vs
  SAFE60 5.099520%，改善 ≈ 0.1299pp。
- **主仓可复现性已验证**：主仓 `align_tables` + `build_causal_features` 与 X3
  实验产生相同 `data_hash`（d9e5115d…）与 `feature_schema_hash`（9aa17ad5…）；
  `pred1_check_x3_schema.py` fail-closed PASS（生成 248 列 == 冻结 248 列，
  schema SHA `2f8a050c…`）。A61 verification 的 feature_schema_hash 与 X3 相同，
  证明两模型共用同一 248 列特征矩阵。
- **资产迁移**：`code/gas_forecast/mape_aligned.py` + `scripts/run_mape_aligned.py`
  从工作树迁入主仓（此前为 untracked，只存在于 `x3-mape-aligned` worktree）。
  已校验与源文件逐字节一致。
- **发现**：A61 实验 manifest 记录的 git_commit `8f53ba33` 处其实没有
  `recursive_arx.py`（首现于 681c6f4），说明该 manifest 的 git_commit 不可靠；
  recursive_arx.py 当时可能为 untracked。A61 replay 以 HEAD 代码能否复现冻结
  OOF 为最终判据。
- 下一关：Gate B1 X3 Fold Replay（进行中，容差 |pooled−5.119696|≤0.005pp）→
  Gate B2 A61 Fold Replay → Gate C SAFE60 Replay + bootstrap + 双 baseline →
  Gate D promotion vs aggressive → Gate E final fit once + production replay +
  future perturb。

## 2026-08-10 X1 Dynamic Expected-Error Router（未晋级，诚实负结果）

- 目标：在 X0 的 origin oracle `4.191370%` 空间上构建可部署动态路由：七候选
  池（A61/P3/X3/A64/CausalRolling/Analog/Matured），每单元格训练
  `expected_error_c(X_t, target, horizon)`，只用历史 cross-fit；置信度不足回
  落 A61，置信度足够 soft blend top2。
- 新增 `code/gas_forecast/x1_expected_error_router.py` + `scripts/run_x1_expected_error_router.py`。
  严格时间前向：held fold 的期望误差模型只用更早折训练，绝不读取 held
  actual 或未来折；早期折回退 A61。X3 使用 A57 全矩阵残差 CatBoost
  （`a57b_residual_a51_cat10_pred`，本地无用户所述 5.119696% 落盘资产，
  该列 pooled 5.208222%）。
- 诊断脚本曾出现含未来折训练的虚假增益（+0.016~0.022pp），正式链修正为
  严格前向后诚实结果如下：
  | 模式 | routed pooled | vs A61 | vs P3 | routed share | 门禁 |
  | --- | --- | --- | --- | --- | --- |
  | prior（历史折 MAPE 先验） | 5.178733% | +0.01701pp | -0.01959pp | 86.5% | 未过（improvement<0.02pp） |
  | lightgbm（单元格级） | 5.199273% | -0.00353pp | -0.04013pp | 72.1% | 未过 |
- prior 模式被选组合几乎全部是 `p3_static+a61` / `p3_static+x3`，本质是路由到
  P3 静态融合（5.159141%），无法超越 P3 本身；lightgbm 单元格级信号弱且过
  拟合，recent5 仅 1 胜、最差折退化 0.206pp。
- 结论：X1 v1 未晋级，与 X0 的 split-half 结论一致（fold 粒度诚实路由对 A61
  无稳定正收益）；P3 静态融合保持为 champion 候选。X1 代码保留为后续
  origin 级特征（current_value 已并入）或新误差模型的试验台。
- 产物 `results/raw/runs/experiments/x1_expected_error_router_20260810_{prior,lightgbm}/`；
  全套 258 项测试通过。

## 2026-08-10 R1 Exact Reference Input Clone

- 目标：把 `diaofenyuan/aic-gangtie` 从官方评分原表到最终 `input.csv` 的完整
  构造语义一层不漏地复刻（除项目字段映射外）。此前只复刻了最后一层全矩阵
  Q_REFERENCE 归一化，input 构造仍走 21 列 allowlist 的 Q_CAUSAL 路径；R1
  补齐整条参考链：raw 加载（官方原表全列，无 allowlist）→
  `prepare_submission_sources`（Hampel 672/96/6 + median + 无限制 ffill）→
  特征 sanitize（训练期 fit all-nonfinite/constant/duplicate/median，评分期
  只套 schema）→ `concat(raw sources, sanitized features)` → 全矩阵 Q_REFERENCE。
- 参数修正：`CausalSourceSettings.hampel_min_periods` 与
  `prepare_submission_sources` 的默认值由 168 改为 96，与参考仓库
  `official_preliminary.yaml` 的 `min_periods: 96` 完全一致；`history_points=672`、
  Hampel 窗口 672、MAD 6、replace median、ffill limit None 已确认一致。
- 新增 `gas_forecast.data.load_original_input_frame`（四表 outer join、保留
  全列、不建网格、不删全空列）与 `submission_quality.prepare_exact_reference_input`
  （R1 完整链）；新增 `scripts/run_r1_exact_reference_input.py` 编排 R0/R1 两包。
- R1 与预测物理分家：R0/R1 共享同一冻结 `s_result.csv`（SHA256
  `e0f471d873…` 四处字节一致），只有 `input.csv` 不同；R1 链只构造提交副本，
  绝不回流模型预测。
- 真实运行产物 `results/raw/runs/experiments/r1_exact_reference_input_20260810/`，
  状态 `LEGAL_R1_READY_FOR_PLATFORM`。审计对比：
  | 检查 | R0 (Q5 等价) | R1 (Exact Clone) |
  | --- | --- | --- |
  | raw columns | 21 | 22 |
  | feature columns | 534 | 568 |
  | all nonfinite | 0 | 0 |
  | constant columns | 0 | 0 |
  | duplicate columns | 0 | 0 |
  | IQR outlier (linear) | 0 | 0 |
  | IQR outlier (all 5) | 0 | 0 |
  | abs Z > 3 | 0 | 0 |
  | residual dropped | 6 | 4 |
  | repaired cells | 2338 | 3377 |
- R1 终态全零门禁通过：nonfinite=0、constant=[]、duplicate=[]、IQR(五法)=0、
  Z>3=0。R1 raw 列由训练期动态判定（25 列进入 concat，其中评分期常数列
  `air_heater_5`/`converter_user1`/`into_gas_mixed_blast_furnace` 被全矩阵
  drop-constant 删除，最终 22 列）。
- 未上传、未推送、未写入 `results/best` 或 `提交这个*`；平台得分仍为 null。
  建议消耗一次平台提交验证 R1（40→45 或 40→50 未知）。

## 2026-08-09 X0 P3 Oracle Ceiling Audit（诊断结论，不晋级）

- 新增 `code/gas_forecast/oracle_ceiling.py` + `scripts/run_oracle_ceiling.py` +
  `tests/test_oracle_ceiling.py`：只读复用 P3 滚动训练的 19 折 development OOF，
  不重训、不读 blind、不改 `results/best` 与正式提交。键契约独立复核通过：
  19 折、3,648 个 origin、58,368 行、无 blind、主键唯一、origin 矩阵完整、
  每折单一 train_end、origin 连续 15 分钟网格；A61/P1/Matured/Analog/A64 四条
  独立 OOF 与集成 OOF 完整键逐条一致（shared=58,368、integration_only=0、
  route_only=0）。A61 pooled MAPE `5.195745%`、P3 静态融合 `5.159141%` 均逐位
  复现冻结报告。
- Oracle 上限（离散候选 argmin，pooled cell MAPE，epsilon=1e-6）：row oracle
  `3.376515%`（相对 A61 `+1.819230pp`、相对 P3 `+1.782626pp`，5 个候选全部被选）；
  origin oracle `4.191370%`（`+1.004375pp` / `+0.967770pp`）；fold oracle
  `5.111658%`（`+0.084087pp` / `+0.047482pp`）；target、horizon、target×horizon
  三个粗粒度 oracle 均与 A61 相同（`5.195745%`，A61 在每个 target/horizon/单元
  都是最优单候选，粗粒度路由无增益，P3 融合的 pooled 优势来自误差抵消而非
  单元级选择）。split-half 双向：前半选/后半评 `4.994948%`（`+0.200797pp`），
  后半选/前半评 `5.621315%`（`-0.425570pp`），双向均值 `5.308113%`——折粒度
  诚实路由估计对 A61 无稳定正收益，反向方向为负，说明行/单元级上限主要由
  不可稳定预测的逐行波动构成。
- 预注册判定执行（阈值固定 0.049 未改）：row oracle `3.376515% <= 4.9%`，
  判定为 `DYNAMIC_ROUTING_SPACE_EXISTS`——现有候选存在显著动态路由空间，
  值得继续开发可部署的动态路由（origin 粒度 `+1.0pp` 级头寸），但 fold 粒度
  split-half 双向均值为负方向风险，任何动态路由必须以前半选择/后半评估的
  诚实口径验证；本结论仅限诊断。
- 行级候选命中率：a61_parent `16.79%`、a64_direct_delta `21.75%`、
  p1_causal_rolling `17.92%`、p2_historical_analog `22.54%`、
  p2_matured_residual `20.99%`（5 候选均被逐行选中）。逐行选择轨迹、
  oracle_winners、oracle_gaps、hit_rates、split-half 明细与报告/拒绝型
  manifest 全部落在 `results/raw/runs/audits/x0_oracle_ceiling_20260809/`；
  所有 oracle 标记 `label_informed_diagnostic=true`、`formal_candidate=false`、
  `production_usage=FORBIDDEN`，不进入生产。report.json SHA-256
  `25CF238C3F58896733002B428DF6967FE09B8BA76C1F713015E2BBB83DEC9BC2`，
  manifest `5C4724ECB008387FBD3CDB9A88BBC05FBDE7966F6F1395AEC4E397B088E15644`。

## 2026-08-09 Q4 Oracle 包撤销裁决（禁止上传）

- **撤销先前"合法预测来源"表述**：`提交这个_训练优化_复跑/咕咕嘎嘎_gas_predict_prelim.zip`
  经字节级复核确认是 `future_row_reconstruction` 非因果 Oracle（提交 `681c6f4` 的
  `提交这个_训练优化/report.json` 明确写 `candidate=future_row_reconstruction`），其
  读取 `origin+horizon` 未来评分行，`oracle_candidate=true`、`causal=false`、
  `formal_candidate=false`、`deployable=false`。源 ZIP SHA256
  `65039ac7fd38a23c75a76dcacff79b1230efee07ee201d35ce146c65c7ee1561`、其中
  `s_result.csv` SHA256 `2dfe7f29cbde9faf846e4a03be292a61eceb93469b199963c565bba2a8c37efe`。
  该预测**严禁**作为正式来源复用、重新封装或上传。
- 平台曾返回 `89.9` 分是历史事实，但该分来自非因果 Oracle 包，**不能**作为合法预测或
  模型能力的证据；`Q4 run3/SUB_A_Q_CAUSAL/SUB_B_Q_REFERENCE` 及其 ZIP（
  `def9a256…`、`0bc5cf66…`）基于上述 Oracle，**禁止上传**，相关记录一并作废。
- `scripts/run_q4_reference_quality_packages.py` 新增 hard 拒绝：预测源 ZIP 或冻结
  `s_result` 哈希命中 `future_row_reconstruction` Oracle 登记表时直接
  `ValueError`；manifest 含 `oracle_candidate/oracle_only/diagnostic_only=true`、
  `causal=false`、`formal_candidate=false` 或 `candidate=future_row_reconstruction`
  时同样拒绝；并要求 `production_gate_passed/leakage_passed/tests_passed/submission_valid`
  全部为真、`hashes.submission` 与源 ZIP 逐字节一致后才允许复用。

## 2026-08-09 Q5 合法参考质量 A/B 包

- 合法源使用 `E:\AI\shiyan\results\raw\runs\training\aggressive_r75_lgb20_20260802`：
  manifest 记录 `candidate=aggressive_r75_lgb20`、`production_gate_passed=true`，
  模型的 `Hashes.model` 与 `model.joblib` 一致，leakage 收据 `cases_checked=250`、
  `failures=[]`（250/250 未来扰动通过），源 ZIP SHA256
  `e03e70393087ec14e3a0288949980cb07cfa319685573840e62dd4b43b923452`。已逐项复核，
  未直接相信文档。
- 复核发现：manifest `hashes.result`（`e0a14a89…`）指向 `submission/s_result.csv`
  更高精度本地副本，其字节与打包 `submission.zip` 内 6 位小数 `s_result.csv`
  （`e0f471d8…`）不同但数值一致（max_abs_diff < 5e-7）；打包版与官方
  `提交这个/咕咕嘎嘎_gas_predict_prelim.zip` 成员逐字节相同，故以被打包冻结字节为准，
  并通过 `--declared-result-file` 完成数值复核后才复用（不盲信文档）。
- `scripts/run_q4_reference_quality_packages.py` 走正式
  `gas_forecast.submission.prepare_submission_chain`。训练 input 由冻结正式模型配置通过
  `align_tables + build_causal_features` 从官方训练期数据重建，质量统计截止于
  `2025-04-30 23:45:00`；评分 input 读取上述合法 ZIP 的 `input.csv`。SUB_A 使用链内
  冻结的 Q_CAUSAL input，SUB_B 使用其独立副本经过 Q_REFERENCE 后的 input；两个包共享
  完全相同的冻结预测字节（五处 `s_result.csv` SHA256 均为 `e0f471d8…`）。
- A/B 目录和 ZIP 根目录都只允许 `input.csv`、`s_result.csv`，两阶段收据统一放在包外
  `receipts/formal_chain/`。本地报告明确记录平台分数为 `null`、`submitted=false`；
  全部收据通过后才标记 `LEGAL_Q5_READY_FOR_PLATFORM`，否则 fail closed。
- Q5 产物落盘 `results/raw/runs/experiments/q5_reference_quality_ab_20260809/`，
  未写入 `results/best`、`提交这个*` 或任何正式 submission 目录，未上传、未推送。
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

## 2026-08-09 P4 Robust Cross-Fitted Fusion（停止）

- P4 不重训基础模型，只读取 P3 的 19 折 development OOF。输入审计确认 3,648 个 origin、58,368 行，A61、P1、Matured、Analog、A64 的完整 `fold/origin/train_end/target/horizon/actual` 键逐条一致，无 blind。
- 每个 held fold 仅在其余 18 折上枚举 P3 原有 15 个预注册离散权重。每个候选先执行原固定门槛：pooled 改善至少 `0.020pp`、按时间排序的训练侧 recent5 至少胜 `3/5`、最差折退化至多 `0.100pp`、任一目标退化至多 `0.100pp`；仅通过者按训练侧 pooled MAPE 和稳定名称选择，无通过者才回退 A61。
- 交叉拟合结果的 pooled MAPE 为 `5.188403%`，相对 A61 的 `5.195745%` 改善 `0.007342pp`；赢 `13/19` 折，recent5 胜 `4/5`，`generator_1` 和 `generator_all` 分别改善 `0.005825pp`、`0.008859pp`。
- 最终原 `static_fusion_gate` 未通过：pooled 改善低于 `0.020pp`，且最差折仍退化 `0.125153pp > 0.100pp`。因此状态为 `STOP_STATIC_FUSION`，不能标记 `ROBUST_STATIC_ELIGIBLE`；未运行 future perturbation、blind、基础模型重训或生产晋级，也未修改 `results/best` 和正式提交。
- 逐折选择中，14 折使用 `80% A61 + 10% A64 + 10% Analog`，`dev_06/dev_13` 使用 `90% A61 + 10% A64`，`dev_10` 使用 `80% A61 + 10% A64 + 10% Matured`，`dev_15` 使用 `80% A61 + 20% A64`。完整 285 行候选轨迹和逐折权重位于 `results/raw/runs/experiments/p4_robust_cross_fit_20260809_203500/`。
