# Phase 1–14 研究工作流

PR #1 收尾完成后，后续候选统一通过 `scripts/run_research_experiment.py` 执行。该入口将每个候选登记为固定配置，并强制使用同一批外层 OOF 折、同一 pooled MAPE 口径和相同的目标/步长诊断。

## 三层验证

- `--scope screening`：固定最多 5 个开发折，包含早期、中期、近期和可用时的 2025-04-18 切换风险块；绝不读取 blind。
- `--scope development`：所有开发折；仍不读取 blind。
- `--scope final`：只在参数已经冻结后运行，才包含 blind 并产生正式候选判定。

每个报告都给出 pooled MAPE、两个目标、8 个步长、折胜数、日块胜数、最近 5 折、最差折退化和候选判定。`results/best/` 不会被该研究入口自动改写。

## 按阶段运行

```powershell
# Phase 1：generator_1 专项逐步长 Ridge 的三项消融
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E10_gen1_hridge_base --scope screening --jobs 4
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E11_gen1_hridge_aligned --scope screening --jobs 4
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E12_gen1_hridge_aligned_longcycle --scope screening --jobs 4

# Phase 2–6：时间漂移、时间表达、电价、加权损失和线性组合
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E13_gen1_alpha_group --scope screening --jobs 4
python scripts/run_research_experiment.py `
  --data-dir "data/raw/official/初赛-参赛者使用" `
  --experiment-id E21_gen1_recency_exp `
  --base-config <上一阶段运行目录>/report.json `
  --base-candidate-name <已冻结的候选名称> `
  --scope screening --jobs 4
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E63_best_linear --scope development --jobs 4

# Phase 7–8：有限预算的非线性复验
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E70_catboost_gen1_fixed_metric --scope screening --jobs 4
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E80_lgb_direct_gen1 --scope screening --jobs 4

# Phase 9：训练尾部真正 OOF 历史初始化，而不是 within-fold warm-up
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E90_online_bias_true_hot --scope development --jobs 2

# Phase 10：只有单模块赢家冻结后才显式组合，最多两个模块
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --online-combination bias vintage --scope development --jobs 2

# Phase 11–14：动态字段、状态专家、路径后处理和累计路径
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E100_dynamic_core --scope screening --jobs 4
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E110_gen1_moe --scope screening --jobs 2
python scripts/run_research_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id E131_direct_incremental_blend --scope screening --jobs 4
```

`E90`–`E92` 会为每个外层验证折在训练尾部生成仅使用更早历史训练的 calibration OOF 预测，然后逐时刻结算成熟误差再进入验证第一行。它不使用训练集 fitted residual，也不把折内 warm-up 误称为 hot start。

`--base-config` 与 `--base-candidate-name` 用于把上一阶段已冻结的候选配置原样带入下一阶段；它避免把后续实验悄悄退回默认参数。报告中的 blind 结果不应用于选择这个配置。

## 正式冻结

当一个候选通过完整开发 OOF 后，用完全相同的配置运行最终 blind 验收：

```powershell
python scripts/run_research_experiment.py `
  --data-dir "data/raw/official/初赛-参赛者使用" `
  --experiment-id E63_best_linear `
  --scope final `
  --jobs 4
```

通过后，从该运行的 `report.json` 读取候选配置，以相同版本重训；不要手工重写参数：

```powershell
python scripts/train.py `
  --data-dir "data/raw/official/初赛-参赛者使用" `
  --version generator1_horizon `
  --config <研究运行目录>/report.json `
  --candidate-name e63_best_linear
```

随后运行现有 `scripts/audit_leakage.py`、预测、提交校验和打包链路。只有完整 OOF、blind、泄漏审计、测试和提交校验都通过时，才允许按既有流程晋级 `results/best/`。
