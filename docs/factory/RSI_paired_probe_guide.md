# RSI 配对环境预实验运行指南

本轮使用合成订单和库存。厂内在收到执行通知后运行已发布代码，并回传两个结果文件。

## 1. 更新工作三仓库

在既有工作三仓库根目录执行。

~~~bash
git switch main
git pull --ff-only origin main
git rev-parse HEAD
~~~

单机推理使用 vLLM 及本地 Qwen2.5-7B-Instruct 权重，运行脚本需要 Python 3.10 或更新版本、`transformers` 和 `httpx`。脚本读取 OpenAI 兼容的原始文本 `/v1/completions` 接口，并使用本地 tokenizer 生成工具对话模板。将下列变量改成所在机器上的真实路径和空闲 GPU。保持下列服务在独立终端运行，确保该端口只供本机使用。若已按同样设置启动服务，可以直接使用。

~~~bash
MODEL_DIR=/absolute/path/to/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name rsi-qwen25-7b --host 127.0.0.1 --port 8187 \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.65 \
  --enforce-eager --generation-config vllm
~~~

## 2. 试跑与正式运行

另开终端，在工作三仓库根目录激活有上述依赖的 Python 环境，并设定本地模型路径。试跑用 4 个目标对、2 个对照对。

~~~bash
MODEL_DIR=/absolute/path/to/Qwen2.5-7B-Instruct
curl -fsS http://127.0.0.1:8187/v1/models
python -m work3.rsi_probe \
  --model rsi-qwen25-7b --tokenizer "$MODEL_DIR" \
  --split development --cases-per-family 2 --controls-per-family 1 \
  --seed 20261117 --out-dir "$PWD/results/rsi-paired-smoke-$(date +%Y%m%d-%H%M%S)"
~~~

试跑打印 6/6 并生成两个文件后，打开 `paired_rsi_summary.json` 检查模型名是 `rsi-qwen25-7b`，且 `overall.oracle.worlds_with_answer` 为 8、`overall.autonomous.worlds_with_answer` 大于 0。查看自主条件和对照的 `statuses`，并确认逐对文件记录了查询与分叉状态。通过标准检查的是结果可判读，试跑答错的样本仍应原样保留。

若输出未达到上述条件，回传两个试跑结果及启动命令，由研发复核工具格式和推理服务。试跑通过且收到正式运行通知后，按下列命令运行 120 个目标对和 30 个对照对。保留同一模型、种子和预算。

~~~bash
python -m work3.rsi_probe \
  --model rsi-qwen25-7b --tokenizer "$MODEL_DIR" \
  --split heldout --cases-per-family 60 --controls-per-family 15 \
  --seed 20261117 --max-queries 4 --max-tokens 256 --concurrency 4 \
  --out-dir "$PWD/results/rsi-paired-heldout-$(date +%Y%m%d-%H%M%S)"
~~~

## 3. 回传

- `paired_rsi_summary.json` 包含模型、参数、Git 提交、总体及分模板汇总。
- `paired_rsi_cases.jsonl` 包含逐对结果、查询轨迹和诊断。回传两个完整文件。使用项目约定的回传通道。
- 若服务或脚本异常，附上失败命令、当前 Git 提交、模型名、`vllm --version` 和报错文本。一次重跑使用新输出目录。执行期间出现接口错误时脚本以非零状态结束，退出后可核对是否生成了汇总文件。
- 本批任务为冻结模型预实验，排除训练，使用本指南的正式参数，调参与后续实验安排由研发端负责。

实验设计与指标定义见 [RSI 配对环境推理预实验](../../research/protocols/RSI_paired_feasibility.md)。
