# RSI 订单引用定位厂内指南

**历史合成诊断，已停止厂内执行。** 当前工作三使用原 Wikipedia 离线索引，厂内执行入口见 [Wikipedia RSI 指南](RSI_wikipedia_guide.md)。

本轮用冻结模型验证，纠正一条订单查询能否恢复后续取证，以及低成本的 ID 约束是否已经足够。使用上轮同一 Qwen2.5-7B-Instruct、tokenizer、Python 环境和 vLLM 0.8.5.post1。若服务已停止，按[校准运行指南](RSI_oracle_calibration_guide.md)第 1 节恢复相同模型和端口。保活和推理的切换仍按厂内安排。

在工作三仓库根目录执行，将 MODEL_DIR 设为上轮使用的模型目录。

~~~bash
git switch main
git pull --ff-only origin main
git rev-parse HEAD
vllm --version
export MODEL_DIR=/absolute/path/to/Qwen2.5-7B-Instruct
curl -fsS http://127.0.0.1:8187/v1/models
bash scripts/run_rsi_followup.sh pointer
~~~

模型别名为 rsi-qwen25-7b。完成应显示 Completed 150/150 pairs。三个家族各跑四个固定条件，程序自动汇总。回传终端打印目录下完整的 rsi_pointer_summary.json 和 rsi_pointer_cases.jsonl，并附 Git 提交和 vLLM 版本。回传 Word 时可按完整 JSONL 行分块。失败时回传原命令及报错，重试使用新的默认输出目录。

[订单引用定位实验协议](../../research/protocols/RSI_pointer_probe.md)
