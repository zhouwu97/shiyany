# 煤气发电短周期预测

面向“煤气发电量预测与发电优化”初赛的可复现实现。项目严格遵守预测起点之后的生产数据不可见这一约束，采用以下路线：

> 合规预处理 -> 共享外层逐行 OOF -> 内层时间 cross-fitting -> 路由/融合/协调 -> 全量重训与因果滚动推理。

## 当前范围

- 初赛短周期：每个 15 分钟滚动起点，直接预测未来 15 至 120 分钟共 8 步。
- 官方目标：`generator_1` 与 `generator_all`；结构分支额外预测 `generator_rest`。
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

第二条命令会校验 `results/best`、ZIP 内容和 192×16 预测，并打开 `提交这个/`。平台只上传 `提交这个/teamname_gas_predict_prelim.zip`。

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
  --input submissions/final/s_result.csv `
  --output submissions/teamname_gas_predict_prelim.zip
```

完整 20 折验证与冻结：

```powershell
python scripts/backtest.py --data-dir "data/raw/official/初赛-参赛者使用" --version v1 --jobs 4
python scripts/backtest.py --data-dir "data/raw/official/初赛-参赛者使用" --version v2 --jobs 4
python scripts/select_model.py --data-dir "data/raw/official/初赛-参赛者使用" --v1 <v1_run>/report.json --v2 <v2_run>/report.json
```

正式模型只认 `results/best/`。运行 `python scripts/show_best.py` 查看当前最优版本，运行 `python scripts/prepare_submission.py` 生成唯一提交目录 `提交这个/`；上传其中唯一的 `teamname_gas_predict_prelim.zip`，不要从历史 run 目录自行挑选文件。

若平台仍要求数据字典中的旧版 JSON，可单独执行：

```powershell
python scripts/export_json.py --input submissions/final/s_result.csv --output submissions/final/result_legacy.json
```

不要把 CSV 和 JSON 同时放进正式提交包；优先级为平台当前模板、最新官方答疑、PDF 提交规范、数据包旧说明。

## 测试集隔离

测试期每一行只能在该行对应的滚动起点作为当前输入使用，后续行不能进入当前预测。不得根据测试集未来真实值反推模型、阈值、融合权重或版本。若需要本地查看测试得分，必须先冻结预测文件，再运行独立的 `scripts/evaluate_frozen.py`；该结果只用于最终评估，不得反馈到训练过程。

完整阶段、验收标准和分段提交设计见 [实施计划](docs/IMPLEMENTATION_PLAN.md)。实验结论与路线裁决分别记录在 [RESULTS_REPORT.md](RESULTS_REPORT.md) 和 [DECISIONS.md](DECISIONS.md)。
