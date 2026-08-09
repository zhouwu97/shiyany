# 煤气发电短周期预测

面向“煤气发电量预测与发电优化”初赛的可复现实现。项目严格遵守预测起点之后的生产数据不可见这一约束，采用以下路线：

> 合规预处理 -> 共享外层逐行 OOF -> 内层时间 cross-fitting -> 路由/融合/协调 -> 全量重训与因果滚动推理。

> **范围注意：本仓库只实现初赛内容。复赛、决赛、发电优化调度及其他赛段需求暂不实现，也不应复用初赛验证结果直接推断这些赛段的方案。**

## 当前范围

- 初赛短周期：每个 15 分钟滚动起点，直接预测未来 15 至 120 分钟共 8 步。
- 官方目标：`generator_1` 与 `generator_all`；结构分支额外预测 `generator_rest`。
- 交付：数据审计、滚动验证、训练、因果滚动推理、宽表结果和提交压缩包。
- 原始赛事数据、测试数据、模型产物和提交结果均被 Git 忽略，不上传公共仓库。

## 当前正式结果

Strict C0 冻结基线为 pooled MAPE `5.297932%`（含最终 blind 的全 OOF 口径）。冲分计划完成 Oracle、严格前向 stacking、E21 crossing、Price、Physical/X1 和 Diversity 后，冻结候选为 `R75 + 20% lgb_residual`：

- development（不含 blind）经生产容量投影后 MAPE `5.229437%`，相对同口径 C0 改善 `0.030575pp`；
- 19 个 development folds 中赢 14 个，最近 5 折赢 4 个；
- 冻结后唯一一次 blind 确认改善 `0.040860pp`；
- Production Gate 的 250 个未来扰动、历史 92 项 pytest、192×16 提交校验和确定性 ZIP 全部通过；后续研究分支回归后当前完整测试套件为 128 项。

正式初赛提交为 `提交这个/咕咕嘎嘎_gas_predict_prelim.zip`，ZIP 根目录只包含 `input.csv` 与 `s_result.csv`。第二梯队的固定 CatBoost 与 Recursive ARX 已在独立、无 blind 的 development OOF 上复验；A61 仅保留预注册的 5% Recursive ARX 融合，尚未生产重训、查看 blind 或替换正式提交。复赛、决赛和优化调度仍不实现。

## 环境

推荐 Python 3.10，兼容 Python 3.10 至 3.12。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

最终复现使用锁定依赖：

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps .
```

## 数据目录

将官方文件解压到下列目录，文件名前缀可以是 `Pre_` 或 `Pre_test_`：

```text
data/raw/official/初赛-参赛者使用/
  Pre_gas.csv
  Pre_gas_holder.csv
  Pre_gas_user.csv
  Pre_load.csv
  price.xlsx

data/raw/scoring/初赛-评分所用测试集/
  Pre_test_gas.csv
  Pre_test_gas_holder.csv
  Pre_test_gas_user.csv
  Pre_test_load.csv
```

## 质量检查

```powershell
pytest -q
python scripts/audit_data.py --data-dir "data/raw/official/初赛-参赛者使用"
```

## 提交输入质量

正式打包会执行 `submission_quality.py` 中的初赛 raw schema 与无标签 IQR 修复：原始字段收敛为 21 列，额外原始字段不会伪装成有效观测；派生字段必须保留 `feat_` 前缀。该步骤不改动 `s_result.csv`，因此可在冻结预测不变的前提下独立修复数据质量链路。

`scripts/prepare_submission.py` 和 `scripts/package_submission.py` 默认启用该质量策略；`scripts/production_gate.py` 会将额外 raw 字段、常数 raw 字段和登记字段的未修复 IQR 越界视为失败。

需要保留 Q0/Q1/Q2/Q3 上传消融包时，可执行：

```powershell
python scripts/run_quality_ablation.py `
  --input "提交这个/input.csv" `
  --result "提交这个/s_result.csv" `
  --output-dir results/quality/<run-name>

python scripts/compare_submission_quality.py `
  --candidate results/quality/<run-name>/Q3_full_matrix_quality/submission.zip `
  --reference <满质量参考包.zip>
```

Q3 在 Q2 的 21 列 raw schema 基础上，对整个提交矩阵删除常数/重复派生列，
迭代修复 IQR 异常，并删除仍无法通过多分位数与 Z-score 门禁的派生列。
Q3 只用于固定 `s_result.csv` 的平台质量 A/B，不会自动覆盖正式提交。
参考包仅作为 schema 与回归 oracle；质量策略不会读取、复制或按日期替换参考包中的 192 行值。

## 已验收 RichResidual 候选

`rich_gas_blend_30` 是对冻结 `aggressive_r75_lgb20` 的 `generator_1` 小权重残差修正候选。它先用严格、时间前向的 Champion OOF 选择 `gas + 30%`，然后只在一次 final/blind 验收已经完成后，显式允许 blind OOF 参与全量残差重训。该候选与当前正式 `results/best/` 和 `提交这个/` 相互独立；生成和门禁均不自动晋级：

```powershell
python scripts/complete_rich_residual_candidate.py `
  --base-model "results/raw/runs/training/aggressive_r75_lgb20_20260802/model.joblib" `
  --baseline-oof "results/raw/runs/training/aggressive_r75_lgb20_20260802/oof.csv" `
  --rich-final-run "results/raw/runs/experiments/rich_residual_final_gas_20260803" `
  --data-dir "data/raw/official/初赛-参赛者使用" `
  --test-dir "data/raw/scoring/初赛-评分所用测试集" `
  --run-dir "results/raw/runs/training/rich_gas_blend_30_20260803" `
  --allow-confirmed-blind-oof

python scripts/production_gate.py `
  --run-dir "results/raw/runs/training/rich_gas_blend_30_20260803" `
  --data-dir "data/raw/official/初赛-参赛者使用" `
  --origins 50 --jobs 8 --no-promote
```

脚本会校验 final 收据、基线 OOF、特征配置和固定 30% 融合权重；未传入 `--allow-confirmed-blind-oof` 时，blind 标签默认不能用于重训。

### A50–A52 后续研究状态

- A50 Ramp Error Atlas 已确认 RichGas 在 `generator_1` 的 mild/medium/large ramp 上改善，而 stable 段退化；ramp 档由真实未来增量定义，因此只用于 OOF 诊断，不能直接作为线上门控输入。
- A52 的六组预注册短长权重全部不及固定 `rich_gas_blend_30`，已停止权重微调。
- A51 的 `rich_g1_long` 仅训练 `t+75/90/105/120` 四个残差模型，使用 249 个固定因果特征（含同步长 Champion 预测），短步长保持 RichGas。development 拼接候选为 `5.211443%`，比 RichGas 改善 `0.006326pp`，属于保留研究候选，未进行 blind 验收或生产重训。
- A53 已将真实未来 ramp 写成显式 `oracle_only` 诊断：相对 RichGas 只能改善 `0.004447pp`，低于启动 A55 的 `0.005pp` 门槛；其输出不可部署。A54 的严格前向分位数图谱确认 `|A51-RichGas|` 存在条件信号，但不改变这一停止规则，因此不会启动 Logistic gate、blind 或生产重训。

可复现 A51 的严格 development 命令如下；`--comparison-column` 只复制已冻结 RichGas OOF 供固定短长拼接，不参与训练：

```powershell
python scripts/run_rich_residual.py `
  --data-dir "data/raw/official/初赛-参赛者使用" `
  --baseline-oof "results/raw/runs/experiments/rich_residual_development_b_20260803/oof.csv" `
  --baseline-column aggressive_r75_lgb20_pred `
  --config "results/raw/runs/experiments/rich_residual_development_b_20260803/config.json" `
  --scope development --group-set quantile,ramp,gas `
  --candidate-name rich_g1_long --feature-profile long_horizon `
  --active-horizons 75,90,105,120 --blend-weights 0.30 `
  --include-champion-prediction --comparison-column rich_gas_blend_30_pred
```

可复现 A53/A54 的 development-only 诊断如下。A53 使用真实标签定义 ramp，仅用于测理论上限；A54 的每个分位数边界只取该折 `train_end` 以前的因果特征或已完成 OOF 预测：

```powershell
python scripts/run_oracle_ramp_router.py `
  --input "results/raw/runs/experiments/a51_g1_long_rich_residual_development_20260803/oof.csv" `
  --baseline-column rich_gas_blend_30_pred `
  --specialist-column rich_g1_long_blend_30_pred `
  --run-dir "results/raw/runs/experiments/a53_oracle_ramp_router_development_20260803"

python scripts/run_causal_ramp_atlas.py `
  --data-dir "data/raw/official/初赛-参赛者使用" `
  --input "results/raw/runs/experiments/a51_g1_long_rich_residual_development_20260803/oof.csv" `
  --config "results/raw/runs/experiments/a51_g1_long_rich_residual_development_20260803/config.json" `
  --champion-column aggressive_r75_lgb20_pred `
  --rich-gas-column rich_gas_blend_30_pred `
  --specialist-column rich_g1_long_blend_30_pred `
  --run-dir "results/raw/runs/experiments/a54_causal_disagreement_ramp_atlas_development_20260803"
```

## 训练、预测与提交

### 未来行重建 Oracle（仅诊断研究）

`FutureRowReconstructionForecaster` 会按 `origin + horizon` 读取评分期未来行。
因此它对未来生产观测敏感，硬标识为 `oracle_candidate=true`、`causal=false`、
`formal_candidate=false`、`deployable=false`，绝不是合法的因果模型。它不能参与
训练特征、标签、模型选择、融合权重、阈值、`auto_pipeline`、Production Gate
或正式提交。自动管线的版本白名单也不会枚举它。

如需开展仅用于诊断的复现，必须显式提供研究开关，并把运行目录放在全新的
`results/oracle/<name>/` 下：

```powershell
python scripts/train_future_reconstruction.py `
  --train-dir "data/raw/official/初赛-参赛者使用" `
  --test-dir "data/raw/scoring/初赛-评分所用测试集" `
  --base-model "results/raw/runs/training/aggressive_r75_lgb20_20260802/model.joblib" `
  --run-dir "results/oracle/future_row_reconstruction_20260809" `
  --allow-oracle-research `
  --reference "E:/qq/submission.zip"
```

脚本只写 `model.joblib`、`base_model.joblib`、`oracle_input.csv`、
`oracle_predictions.csv`、`report.json` 和带硬拒绝标识的 `manifest.json`。
它不会生成 `submission.zip`，不会写 `results/best`、`results/raw`、
`提交这个*` 或任何正式 submission 目录。`--reference` 仅在模型与预测写盘后
用于独立诊断评分，不进入 `fit`、选择或任何生产决策。

正式自动编排入口默认按 pooled OOF 直接比较单模型、目标路由与稳定目标×步长路由。每次运行创建独立目录，逐折 checkpoint 可恢复：

```powershell
python scripts/auto_pipeline.py `
  --train-dir "data/raw/official/初赛-参赛者使用" `
  --test-dir "data/raw/scoring/初赛-评分所用测试集" `
  --jobs 8
```

默认比较 `v1`、`v2`、`v25`、`v3`、V2/V3 目标路由及带三级回缩的目标×步长 LOFO 路由。主指标为逐单元格 pooled `competition_mape`；同时报告等目标、等目标×步长和旧折均值。旧逐级选择器仍可通过 `--selection-policy legacy` 使用。

分阶段实验入口：

```powershell
python scripts/prepare_data.py --data-dir "data/raw/official/初赛-参赛者使用"
python scripts/build_oof.py --data-dir "data/raw/official/初赛-参赛者使用" --jobs 8
python scripts/compare_candidates.py --input <run-dir>/oof.csv
python scripts/run_experiment.py --data-dir "data/raw/official/初赛-参赛者使用" --experiment-id m2 --jobs 8
python scripts/audit_leakage.py --data-dir "data/raw/official/初赛-参赛者使用" --model <model> --jobs 8
```

在线校准只使用已经成熟的历史预测误差，按外层折独立冷启动。可以在生成 OOF 时直接比较，也可以复用已有 OOF 长表：

```powershell
python scripts/build_oof.py `
  --data-dir "data/raw/official/初赛-参赛者使用" `
  --versions v1 `
  --online-base v1 `
  --online-modes bias gain vintage `
  --online-warmup-rows 0

python scripts/apply_online_oof.py `
  --input <run-dir>/oof.csv `
  --base-column v1_pred `
  --modes bias gain vintage `
  --warmup-rows 0
```

`--warmup-rows 0` 是 cold-start OOF；设置为正数时，是 within-fold warm-up OOF：每个外层折的前若干个 origin 只用于填充状态、不计入在线候选评分，不代表跨折外部热启动。数据末端若缺少完整目标×步长组合，候选会保留基础预测并在报告中登记 fallback 行数。

所有命令默认写入 `results/raw/runs/{oof,comparisons,training,experiments,audits}/<运行时间>/` 分类目录。每个运行目录包含报告、manifest、日志和 checkpoint；指定同一 `--run-dir` 可从已完成折继续。最近一次各类运行的指针在 `results/latest/`。

正式提交只执行：

```powershell
python scripts/show_best.py
python scripts/prepare_submission.py
```

第二条命令会校验 `results/best`、输入特征、192×16 预测及 ZIP 内容，并打开 `提交这个/`。平台只上传 `提交这个/咕咕嘎嘎_gas_predict_prelim.zip`；ZIP 根目录固定包含 `input.csv` 和 `s_result.csv`。

自动调参不属于正式默认入口。需要开展参数搜索时，应使用独立训练期实验目录、同一套滚动折和有限候选集合，完成后再把冻结配置交给本入口验收。

也可以分步执行已有命令：

```powershell
# 使用冻结路由训练多模型包装器
python scripts/train.py --data-dir "data/raw/official/初赛-参赛者使用" --version routed --selection <comparison-report>

# 使用上一条命令输出的实际模型路径
python scripts/predict.py `
  --train-dir "data/raw/official/初赛-参赛者使用" `
  --test-dir "data/raw/scoring/初赛-评分所用测试集" `
  --model <train_run>/model.joblib `
  --output-dir submissions/final

python scripts/validate_submission.py --input submissions/final/s_result.csv
python scripts/package_submission.py `
  --input submissions/final/input.csv `
  --result submissions/final/s_result.csv `
  --output submissions/咕咕嘎嘎_gas_predict_prelim.zip
```

完整 20 折验证与冻结：

```powershell
python scripts/backtest.py --data-dir "data/raw/official/初赛-参赛者使用" --version v1 --jobs 4
python scripts/backtest.py --data-dir "data/raw/official/初赛-参赛者使用" --version v2 --jobs 4
python scripts/select_model.py --data-dir "data/raw/official/初赛-参赛者使用" --v1 <v1_run>/report.json --v2 <v2_run>/report.json
```

正式模型只认 `results/best/`。运行 `python scripts/show_best.py` 查看当前最优版本，运行 `python scripts/prepare_submission.py` 生成唯一提交目录 `提交这个/`；上传其中的 `咕咕嘎嘎_gas_predict_prelim.zip`，不要从历史 run 目录自行挑选文件。

若平台仍要求数据字典中的旧版 JSON，可单独执行：

```powershell
python scripts/export_json.py --input submissions/final/s_result.csv --output submissions/final/result_legacy.json
```

正式提交包只放 `input.csv` 和 `s_result.csv`，不要加入 JSON、模型或目录层级；该契约以已成功提交的初赛样例为准。

## 测试集隔离

测试期每一行只能在该行对应的滚动起点作为当前输入使用，后续行不能进入当前预测。不得根据测试集未来真实值反推模型、阈值、融合权重或版本。若需要本地查看测试得分，必须先冻结预测文件，再运行独立的 `scripts/evaluate_frozen.py`；该结果只用于最终评估，不得反馈到训练过程。

完整阶段、验收标准和分段提交设计见 [实施计划](docs/IMPLEMENTATION_PLAN.md)。实验结论与路线裁决分别记录在 [RESULTS_REPORT.md](RESULTS_REPORT.md) 和 [DECISIONS.md](DECISIONS.md)。

PR #1 之后的 `generator_1` 专项 Ridge、时间漂移、价格、加权损失、受限树模型、真正 OOF hot start、状态专家和路径候选，统一通过[研究工作流](docs/RESEARCH_WORKFLOW.md)运行；该入口默认不触碰 blind，也不会自动覆盖 `results/best/`。

## Strict C0 后的冲分主线

该阶段已执行完成；以下入口用于复验 Strict C0（pooled MAPE `5.297932%`）之后的冻结实验。研究子命令不会覆盖 `results/best/`，blind 默认不参与选参：

```powershell
# Phase 0：冻结 Strict C0、五分支、折和指纹
python scripts/run_aggressive_plan.py freeze `
  --branches results/raw/runs/a2_calibration/<run-id>

# Phase 1：双向 split-half Oracle 与 S0-S3 严格前向 stacking
python scripts/run_aggressive_plan.py oracle
python scripts/run_aggressive_plan.py stacking

# Phase 2-5：输入均须是严格 OOF 长表
python scripts/run_aggressive_plan.py e21 --input <e21-oof>
python scripts/run_aggressive_plan.py price --input <price-feature-oof>
python scripts/run_aggressive_plan.py physical --input <physical-feature-oof>
python scripts/run_aggressive_plan.py diversity --input <candidate-oof> `
  --baseline-column e21_R75_pred `
  --challengers lgb_residual_pred ridge_pred x1_indirect_g1_pred
```

缓存统一写入 `results/research_v2/`，实验登记写入 `results/aggressive_registry.csv`。完整输入契约、停止规则和各阶段产物见[冲分执行说明](docs/AGGRESSIVE_PLAN.md)。Absolute CatBoost-MAPE 与 Recursive ARX 只允许在 stacking、Price、Physical/X1 完成后的独立预注册 OOF 中启用；A61 已完成一次固定 5/10/20% Recursive ARX 比较，不执行 Optuna、大范围参数搜索或连续权重微调。
