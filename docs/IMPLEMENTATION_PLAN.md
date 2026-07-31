# 完整实施计划

## 1. 目标与不可突破的边界

初赛按 15 分钟滚动预测未来 2 小时，共 8 个直接步长，同时输出 `generator_1` 和 `generator_all`。任一预测起点 `t` 的特征只能由 `t` 及之前的生产数据和预先公布的目标时刻电价构成。

原始赛事数据受使用协议限制，只保存在 `data/raw/`；模型、验证明细和提交文件分别进入被 Git 忽略的 `artifacts/`、`results/raw/` 和 `submissions/`。评分测试表只可作为逐时刻滚动输入，未来行不得参与当前起点预测。测试期真实值仅由独立评分命令在预测文件冻结后读取。

## 2. 阶段与分段提交

| 阶段 | 实现内容 | 验收门槛 | 提交边界 |
| --- | --- | --- | --- |
| P0 仓库与契约 | Python 包、依赖、四表发现、时间对齐、审计 | 能识别真实四表；补齐 15 分钟网格；结构性空列可追溯 | `chore: initialize repository and data contract` |
| P1 V1 基线 | 因果清洗、负荷/煤气时序特征、绝对增量 Ridge、持续性融合 | 合成测试通过；真实数据可训练；预测列和行数合法 | `feat: add causal ridge forecasting baseline` |
| P2 V2 增强 | 煤气供需代理、气柜、煤气隐含负荷、近期模型、LightGBM 残差 | 多数滚动折不劣于 V1；失败时自动回退 V1 | `feat: add gas-aware residual ensemble` |
| P2.5 稳健候选 | `generator_rest` 状态、煤气切换事件、连续线性门控 | 平均与多数折优于 V2；失败时回退 V2 | `feat: add low-complexity continuous gating` |
| P3 V3 冲分 | GMM 状态、升/稳/降概率、动态软门控、分歧收缩 | 盲折及多数开发折优于 V2.5 才可选为提交模型 | `feat: add state-aware dynamic gating` |
| P4 交付 | 滚动验证、训练/推理 CLI、提交校验、说明文档 | 一条命令产出合法宽表；未来扰动测试严格不变 | `docs: finalize reproducible competition workflow` |

每个阶段执行 `pytest -q` 和对应真实数据 smoke run 后再提交，并立即推送 `main`。后一阶段出现问题时，前一提交仍是可运行保底版本。

## 3. 数据处理设计

1. 四表只按 `datetime` 连接，不按行号连接。
2. 建立从最早到最晚时间的完整 15 分钟网格，记录每张表的缺行指示器。
3. 单表重复时间保留最后一条；训练与测试在 `2025-05-01 00:00` 的重叠边界优先使用测试输入行。
4. 仅删除训练期结构性全空列，真实零值保持原样。
5. 缺失值使用有限步数前向填充、缺失标记及模型流水线内训练折中位数，不使用后向插值。
6. 标签为 `y(t+h)-y(t)`，训练起点必须保证全部 8 步标签位于训练边界内。
7. 生成特征时允许当前行；所有历史滚动统计先 `shift(1)`，保证窗口不含当前点之后的信息。

## 4. 模型版本

### V1

- `P0`：8 步均为当前负荷。
- `P1`：两个多输出 Ridge 分别预测两个目标的 8 步绝对增量。
- 在训练尾部校准区按步长学习受约束融合权重，短步长优先保留持续性。
- 输出执行非负、装机上限、`generator_all >= generator_1` 和训练增量分位截断。

### V2

- 增加发电煤气、炉端产耗、粗供需平衡、气柜趋势和零值状态特征。
- 训练全历史与近 60 天 Ridge 分支。
- 煤气分支只预测增量，不通过固定热值换算直接决定发电量。
- LightGBM 预测 Ridge 残差；只有校准区证明确有增益才获得融合权重。

### V3

- 每折训练 GMM 状态中心，构造软概率和中心距离。
- 按训练折增量分位确定升档、稳定、降档事件阈值，事件概率进入门控。
- 逻辑回归门控学习“修正是否显著优于持续性”，输出软概率而非硬切换。
- 模型分歧越大，最终预测越向持续性锚点收缩。

### V2.5

- 对 `generator_rest = generator_all - generator_1` 在每折开发段用 BIC 从 2 至 5 个 GMM 成分中选择状态结构。
- 增加主导气种变化、切换时距、煤气份额熵和交叉升降事件；所有事件只由当前及历史构造。
- 门控目标为严格 OOF 的最优融合系数，采用强正则 Ridge 回归并限制到 `[0.05, 0.70]`。
- 分支分歧使用 MAD，在校准段按目标与步长估计 70%/95% 分位并连续回缩。

## 5. 验证与防泄漏

- 3 月下旬至 4 月底每 2 天一个验证起点，每折覆盖 2 天；最后 4 天为盲折。
- 训练起点与验证起点之间至少保留 8 个 15 分钟标签隔离步。
- 每折重新拟合填补器、标准化器、Ridge、GMM、LightGBM、融合和门控。
- 同时报告持续性、V1、V2、V3 的总 MAPE、目标维度 MAPE、步长 MAPE 和折间胜率。
- V2 晋级还要求最大单折退化不超过 0.3 个百分点，且两个目标的跨折平均 MAPE 均不高于 V1。
- 未来扰动测试会修改起点后的所有生产数据，并断言该起点预测逐位完全一致。
- 测试期评分脚本与训练/预测脚本物理分离，且只接受已存在、内容冻结的预测文件。

## 6. 最终命令链

```powershell
python scripts/audit_data.py --data-dir <训练目录> --output results/raw/data_audit.json
python scripts/backtest.py --data-dir <训练目录> --version v1
python scripts/backtest.py --data-dir <训练目录> --version v2
python scripts/backtest.py --data-dir <训练目录> --version v3 --blind
python scripts/train.py --data-dir <训练目录> --version auto --output artifacts/model.joblib
python scripts/predict.py --train-dir <训练目录> --test-dir <评测输入目录> --model artifacts/model.joblib
python scripts/validate_submission.py --input submissions/s_result.csv
```

`auto` 只依据训练期滚动验证选择版本，绝不读取测试期未来真实值。V3 未同时通过盲折与多数折胜率时，自动提交 V2；V2 未通过时回退 V1。
