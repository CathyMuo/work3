# RSI 配对环境校准厂内运行指南

本轮先执行 24 个目标对与 8 个对照对的诊断，使用上轮同一 Qwen2.5-7B-Instruct 权重、tokenizer 和 vLLM 文本补全端口。输入由脚本生成。厂内按固定配置执行并回传，条件选择由家服负责。

## 1 更新仓库和核对服务

在原工作三仓库根目录和上次推理所用的 Python 环境中执行。

~~~bash
git switch main
git pull --ff-only origin main
git rev-parse HEAD
python -m work3.rsi_calibrate --help
vllm --version
curl -fsS http://127.0.0.1:8187/v1/models
~~~

将 MODEL_DIR 设为上一轮同一个本地 7B 权重目录。若上轮服务仍在运行，核对返回的模型名是 rsi-qwen25-7b，即可复用。若服务已关闭，在独立终端按下列命令启动，使用空闲 GPU。服务只监听本机。

~~~bash
MODEL_DIR=/absolute/path/to/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name rsi-qwen25-7b --host 127.0.0.1 --port 8187 \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.65 \
  --enforce-eager --generation-config vllm
~~~

## 2 运行本轮诊断

在原工作三仓库根目录执行。若 MODEL_DIR 已设为真实路径，第一行可省略。

~~~bash
MODEL_DIR=/absolute/path/to/Qwen2.5-7B-Instruct
RUN_DIR="$PWD/results/rsi-calibration-diagnostic-$(date +%Y%m%d-%H%M%S)"
python -m work3.rsi_calibrate --phase diagnostic \
  --endpoint http://127.0.0.1:8187 --model rsi-qwen25-7b \
  --tokenizer "$MODEL_DIR" --concurrency 4 --out-dir "$RUN_DIR"
~~~

程序完成后应打印 Completed 32/32 pairs。汇总文件应显示 target_pairs 为 24、control_pairs 为 8。两种模板的每个 oracle 条件均应有 12 对。诊断需要原样保留无效输出、答错及本次未重现的案例。以本轮实际运行结果为准。

回传以下两份完整文件以及 git rev-parse HEAD 和 vllm --version 的输出。

- $RUN_DIR/rsi_calibration_summary.json，含每模板四种证据条件的指标和自主拒绝原因计数。
- $RUN_DIR/rsi_calibration_cases.jsonl，诊断阶段共 32 行，含被工具解析器拒绝的原始输出及当时历史。用 Word 回传时可按完整 JSONL 行分块，保留全部字符。
- 如果命令失败，回传执行命令、提交、模型名和错误日志。重新运行使用新的输出路径，保留成功运行的文件完整。

## 3 新种子确认命令

收到家服选定的条件与确认通知后执行。模型、服务及环境沿用诊断阶段。在终端把 RSI_CONDITION 设为通知中的条件名，程序会校验该值。先回传诊断结果，由家服选择条件。

~~~bash
RUN_DIR="$PWD/results/rsi-calibration-confirmation-$(date +%Y%m%d-%H%M%S)"
python -m work3.rsi_calibrate --phase confirmation \
  --condition "${RSI_CONDITION:?Set RSI_CONDITION to the condition supplied by the home team}" \
  --endpoint http://127.0.0.1:8187 --model rsi-qwen25-7b \
  --tokenizer "$MODEL_DIR" --concurrency 4 --out-dir "$RUN_DIR"
~~~

确认阶段应打印 Completed 60/60 pairs，回传同名的两个完整文件。汇总报告各模板的有效答案率、双世界正确率、B 世界正确率及工程门槛结果。

样本、实验条件及判读规则见 [RSI 配对环境 oracle 校准协议](../../research/protocols/RSI_oracle_calibration.md)。
