# RSI 获取与使用失败复验厂内指南

本轮重复验证两类失败。使用上轮同一 Qwen2.5-7B-Instruct、tokenizer 和 vLLM 版本。与上轮保持同一生成和查询预算，采用允许批量查询的接口，每条查询计入四次总预算。

## 执行

准备好上轮 vLLM 服务，端口 8187，模型别名 rsi-qwen25-7b。服务启动参数见[校准运行指南](RSI_oracle_calibration_guide.md)。在工作三仓库根目录执行，把 MODEL_DIR 改为同一个本地 7B 权重目录。

~~~bash
git switch main
git pull --ff-only origin main
git rev-parse HEAD
vllm --version
export MODEL_DIR=/absolute/path/to/Qwen2.5-7B-Instruct
curl -fsS http://127.0.0.1:8187/v1/models
bash scripts/run_rsi_followup.sh terminal
~~~

完成应显示 Completed 120/120 pairs。订单改版和合并库存各 60 个目标对，种子 20261120。脚本自动完成全部续跑条件，执行端沿用默认参数。

回传终端打印目录内完整的 rsi_terminal_summary.json、rsi_terminal_cases.jsonl，附提交和 vLLM 版本。JSONL 含拒绝的原始输出与必要上下文，Word 回传按完整 JSONL 行分块。如有错误，回传执行命令和原始日志。再次执行会使用新的输出目录。

[实验协议](../../research/protocols/RSI_terminal_probe.md)
