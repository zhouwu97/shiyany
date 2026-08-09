# Strict C0 后初赛冲分执行说明

## 范围

本说明及相关代码只服务初赛。复赛、决赛、发电优化调度和其他赛段暂不实现。所有研究默认只读取严格 OOF 预测；任何候选晋级前都不能覆盖 `results/best/`。

## 冻结约束

- 唯一基线：Strict C0，pooled MAPE `5.297932%`。
- 20 个 forward OOF folds，purge `135min`。
- 当前折权重只能由时间更早的 folds 训练。
- blind 默认保持 C0，不参与模型、路由、alpha、clip 或 blend 选择。
- 单次实验只改变一个变量，结果统一登记到 `results/aggressive_registry.csv`。

## Phase 0

`run_aggressive_plan.py freeze` 生成：

```text
results/research_v2/base/
  strict_c0_oof.parquet
  branch_predictions.parquet
  fold_assignment.parquet
  targets.parquet
  origin_metadata.parquet
  feature_fingerprint.json
  split_fingerprint.json
  baseline_metrics.json
```

重读后 pooled 指标与冻结值误差必须小于 `1e-8`。分支缓存统一命名为 `persistence_pred`、`ridge_pred`、`recent_ridge_pred`、`gas_ridge_pred`、`lgb_residual_pred` 和 `c0_pred`。

## Phase 1

Oracle 同时计算 Current C0、best single、同 cell full oracle 和双向 split-half oracle。双向规则为前半训练后半预测、后半训练前半预测，避免只评估单一时间方向造成偶然偏差。

Stacking 使用 persistence-centered MAPE：

```text
prediction = persistence + sum(weight_b * (branch_b - persistence))
```

Correction 权重非负，直接以 SLSQP 最小化 competition MAPE。候选固定为 S0 global、S1 target、S2 horizon，以及带 global/target partial pooling 的 S3 target×horizon。S3 只扫描计划规定的回缩系数，并只比较无平滑和 `0.25/0.50/0.25` 轻平滑。

## Phase 2-5

- E21：只评估 all C0、R75、R90 和 R105；`generator_all` 始终使用 C0。
- Price：先输出 Error Atlas，再以严格历史折训练 Ridge、Huber、低结点样条 GAM 的 C0 residual；仅 switch gate 激活，alpha 与 clip 固定为 `4×3` 小网格。
- Physical Rest：输出 `P(state0/1/2/transition)` 和条件 rest 回归，只把间接 `generator_1 = generator_all_C0 - rest_physical` 作为 X1 分支；正式比较固定 `5/10/15/20%` blend。
- Diversity：每个 challenger 只扫描 `5/10/15/20/30%`；改善 `0.003pp` 进入候选池，`0.005pp` 进入 stacking pool。

第二梯队的 Absolute CatBoost-MAPE 只有 small/medium/more-nonlinear 三套固定参数。Recursive ARX 只使用历史状态、自身递归预测和官方 known-future price，固定比较 `5/10/20%` 融合。两者均不得在前三条主线完成前消耗主要算力。

## 机械决策

- Screening 退化超过 `0.015pp`：`STOP`。
- Full development 改善不足 `0.002pp`：`STOP`。
- 收益集中在不超过 3 个 folds 且多数 folds 退化：`STOP`。
- 改善达到 `0.003pp`：至少 `KEEP`。
- 改善达到 `0.005pp`，且赢折数或最近折稳定性通过：`PROMOTE`。

只有 `PROMOTE` 且参数、route、blend 全部冻结后才能单次查看 blind。通过 blind 后仍须运行 250 个 future perturbation cases、完整 pytest、提交校验和确定性 ZIP，才允许进入 Production Gate。

## 执行状态（2026-08-02）

Phase 0-5 已完成，统一结果登记于 `results/aggressive_registry.csv`。最终冻结的初赛模型为 `aggressive_r75_lgb20`，development 经容量投影后为 `5.229437%`，唯一一次 blind 确认为正向，Production Gate 全部通过。正式 ZIP 根目录为 `input.csv` 和 `s_result.csv`。

第二梯队 Absolute CatBoost-MAPE 与 Recursive ARX 的固定实现和测试已保留。2026-08-04 在 A51/A60 严格 development 路线完成后，A61 以新的独立预注册执行了 Recursive ARX 的固定 `5/10/20%` 融合；只有 5% 满足 pooled 与 recent5 门槛，仍不读取 blind 或覆盖正式模型。本文档不扩展到复赛、决赛或优化调度。
