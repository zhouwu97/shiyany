# 煤气发电短周期预测

面向“煤气发电量预测与发电优化”初赛的可复现实现。项目严格遵守预测起点之后的生产数据不可见这一约束，采用以下路线：

> 因果时序清洗 -> 当前负荷持续性锚点 -> 未来绝对增量 Ridge -> 煤气/气柜增强 -> LightGBM 残差修正 -> 状态迁移与动态软门控。

## 当前范围

- 初赛短周期：每个 15 分钟滚动起点，直接预测未来 15 至 120 分钟共 8 步。
- 目标：`generator_1` 与 `generator_all`。
- 交付：数据审计、滚动验证、训练、因果滚动推理、宽表结果和提交压缩包。
- 原始赛事数据、测试数据、模型产物和提交结果均被 Git 忽略，不上传公共仓库。

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

## 训练、预测与提交

```powershell
# 先用训练期滚动报告选择版本；当前真实盲折默认选择 V2
python scripts/train.py --data-dir "data/raw/official/初赛-参赛者使用" --version v2 --output artifacts/model.joblib

python scripts/predict.py `
  --train-dir "data/raw/official/初赛-参赛者使用" `
  --test-dir "data/raw/scoring/初赛-评分所用测试集" `
  --model artifacts/model.joblib `
  --output-dir submissions/final

python scripts/validate_submission.py --input submissions/final/s_result.csv
python scripts/package_submission.py `
  --input submissions/final/s_result.csv `
  --output submissions/teamname_gas_predict_prelim.zip
```

完整 20 折验证与冻结：

```powershell
python scripts/backtest.py --data-dir "data/raw/official/初赛-参赛者使用" --version v1 --jobs 4 --output results/raw/backtest_v1_20fold.json
python scripts/backtest.py --data-dir "data/raw/official/初赛-参赛者使用" --version v2 --jobs 4 --output results/raw/backtest_v2_20fold.json
python scripts/select_model.py --v1 results/raw/backtest_v1_20fold.json --v2 results/raw/backtest_v2_20fold.json --output results/raw/model_selection_20fold.json
```

若平台仍要求数据字典中的旧版 JSON，可单独执行：

```powershell
python scripts/export_json.py --input submissions/final/s_result.csv --output submissions/final/result_legacy.json
```

不要把 CSV 和 JSON 同时放进正式提交包；优先级为平台当前模板、最新官方答疑、PDF 提交规范、数据包旧说明。

## 测试集隔离

测试期每一行只能在该行对应的滚动起点作为当前输入使用，后续行不能进入当前预测。不得根据测试集未来真实值反推模型、阈值、融合权重或版本。若需要本地查看测试得分，必须先冻结预测文件，再运行独立的 `scripts/evaluate_frozen.py`；该结果只用于最终评估，不得反馈到训练过程。

完整阶段、验收标准和分段提交设计见 [实施计划](docs/IMPLEMENTATION_PLAN.md)。实验结论与路线裁决分别记录在 [RESULTS_REPORT.md](RESULTS_REPORT.md) 和 [DECISIONS.md](DECISIONS.md)。
