# RSI 确认与后续诊断厂内交接

先运行确认。已选定 tool_compare，脚本将条件和新种子固定。沿用诊断阶段的本地 Qwen2.5-7B-Instruct、tokenizer 和 vLLM 0.8.5.post1 服务。使用相同的 Python 环境。

## 1 拉取并运行 60 对确认

在已有的工作三仓库根目录执行。把 MODEL_DIR 改成上次使用的本地模型目录。

~~~bash
git switch main
git pull --ff-only origin main
git rev-parse HEAD
export MODEL_DIR=/absolute/path/to/Qwen2.5-7B-Instruct
curl -fsS http://127.0.0.1:8187/v1/models
bash scripts/run_rsi_followup.sh confirmation
~~~

服务应报告模型名 rsi-qwen25-7b。服务已关闭时，按[校准运行指南](RSI_oracle_calibration_guide.md)第 1 节启动。确认阶段使用种子 20261118，每个模板 30 对。完成应显示 Completed 60/60 pairs。回传输出目录中的完整 rsi_calibration_summary.json 和 rsi_calibration_cases.jsonl，以及提交和 vLLM 版本。

每个模板独立验收：有效输出至少 57/60，双世界均对至少 21/30，B 世界正确至少 24/30。两个模板都通过，才按下面的顺序执行后续诊断。未通过时回传原结果，由家服复盘。

## 2 单查询和批量查询对照

确认通过后，在同一终端和 Python 环境运行。

~~~bash
bash scripts/run_rsi_followup.sh batch
~~~

预期 Completed 54/54 pairs。含原诊断套件模板的 12 个目标对和 4 个对照对，以及新种子中 30 个目标对和 8 个对照对。保留同样的模型提示和四次查询预算。批量工具返回只有实际可见内容发生变化才分叉。回传该输出目录中的完整 rsi_failure_summary.json 和 rsi_failure_cases.jsonl。

## 3 同轨迹补证据诊断

完成批量对照后，在同一环境运行。

~~~bash
bash scripts/run_rsi_followup.sh evidence
~~~

预期 Completed 60/60 pairs。脚本在两个新的合成任务结构上检测获取失败和使用失败，每个结构 30 对。它对有有效错误答案且缺少充分证据的世界，从同一决策前历史分别补入缺失证据和无关但同结构的记录。回传该输出目录中的完整 rsi_failure_summary.json 和 rsi_failure_cases.jsonl。请在文件名外标注运行阶段，两个阶段的文件名相同。

全部阶段都用冻结模型推理。输出路径由脚本打印。若命令失败，回传原命令、提交、vLLM 版本和报错。需要重跑时直接再次运行，默认产生新的输出目录。

[后续实验协议](../../research/protocols/RSI_failure_followup.md)
