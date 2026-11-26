# 厂内运行：放大差距 P0a

> **已结束（2026-11-26，反馈 #64）。** 主模型与副模型的 P0a 在 `1cd4447` 下完整跑完，均判 NO-GO，该方向停止，本指南无需再执行。四卡实测用时：主模型 974 秒，副模型 2,353 秒。四个 `amp-gap-*` 运行目录请保留（含 `rank-*.jsonl` 与 `texts-*.jsonl.gz`）。结论见 `research/DECISION_LOG.md`。

P0a 检验拟方法的根基：在有标准答案的数学训练题上，模型「多想后答得一致」的答案是否大多正确，按放大差距选出的题标签是否比 R-Zero 式选题更准。

- 只做推理，不训练。冒烟和正式运行默认都用四卡（`--gpus 4`），每卡一个 vLLM 进程，题目按卡轮流分配。
- 预计主模型约 30–60 分钟，副模型约 1–1.5 小时；首次实测后再更新。
- 门槛见[冻结协议](../../research/protocols/amp_gap_p0a.md)。

**反馈 #63 之后（协议修订 2）：** `force` 臂接回 `short` 的 token 前，先丢弃分词器范围外的保留 id，修复 `Token id … is out of vocabulary`。采样设定不变。上一版 `046cea4` 的采样掩码会拖慢生成，已撤回，请直接用最新提交。同时做了提速：每批题的独立臂合并成一次生成调用，程序执行与后续生成并行。worker 日志每批打印一次进度和越界计数。旧冒烟 `amp-gap-base-smoke-v1` 属于旧提交，不能再引用，请按第三节用 `-v2` 的运行名重跑冒烟和正式 P0a。

反馈 #62 的沙箱超时已在上一提交修复：外层期限改为「启动预算 + 执行限额 + 兜底」。执行限额仍由 launcher 在命名空间建好后单独计时；启动预算在 `run_amp_gap.sh` 中默认 120 秒，可用 `WORK3_SANDBOX_STARTUP_SECONDS` 调整。P0a 每卡只开 2 路程序执行，四卡共 8 路，对应 #62 实测的并发水平。`feedback_pair` 已停止，不需要再跑。

## 一、需要带入厂内的模型与数据

**P0a 必需**

| 类型 | Hugging Face ID | 用途 | 约大小 |
| --- | --- | --- | --- |
| 模型 | `Qwen/Qwen3-4B-Base` | 主模型，R-Zero 解题器起点 | 8 GB |
| 模型 | `Qwen/Qwen3-4B` | 副模型，带思考开关，加 `--hybrid-thinking` 运行 | 8 GB |
| 数据集 | `EleutherAI/hendrycks_math`（7 个子集全部，含 train 与 test 的 parquet） | P0a 只读 train 文件 | 约 20 MB |

在能联网的机器上下载，再整目录拷入厂内：

```bash
huggingface-cli download Qwen/Qwen3-4B-Base --local-dir models/Qwen3-4B-Base
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
huggingface-cli download --repo-type dataset EleutherAI/hendrycks_math --local-dir datasets/hendrycks_math
```

**后续阶段（P0b、P1）建议一并带入，P0a 不使用**

| 类型 | 名称 | 用途 |
| --- | --- | --- |
| 模型 | `Qwen/Qwen3-8B-Base` | 第二个模型规模 |
| 模型 | `Qwen/Qwen3-32B`（或沿用厂内已有的 `Qwen2.5-32B-Instruct`） | 生成题的代理真值判官 |
| 代码 | R-Zero 官方仓库（GitHub `Chengsong-Huang/R-Zero`），记下提交号 | P1 训练基线，基于 verl |
| 数据集 | `HuggingFaceH4/MATH-500`、`openai/gsm8k`（main） | 主评测 |
| 数据集 | AMC23、AIME24、AIME25、Minerva Math、OlympiadBench、`TIGER-Lab/MMLU-Pro`、`m-a-p/SuperGPQA`、BBEH | R-Zero 协议的其余评测；具体仓库 ID 以 R-Zero 评测脚本所用版本为准，下载时核对 |

## 二、运行环境

需要一个 Python 环境，满足以下条件：

- `vllm >= 0.8.5`、`transformers >= 4.51.0`：Qwen3 的最低支持版本（厂内 `rsi` 环境的 4.51.1 可用，无需升级）；
- `pyarrow`、`torch`；
- 系统自带的 `/usr/bin/python3`：沙箱用。

旧的 `searchr1` 环境里，vLLM 多半不支持 Qwen3，请先核对：

```bash
$AMP_PYTHON -c "import vllm, transformers, pyarrow, torch; print(vllm.__version__, transformers.__version__, torch.__version__)"
```

版本不够时，建议单独建一个环境，例如按 R-Zero 的依赖建，P1 也会用到。冒烟和正式运行都占满四张卡，运行期间请暂停保活训练，跑完再拉起。

## 三、执行

在 Work3 仓库根目录，拉取本指南所在提交后执行：

```bash
export AMP_PYTHON=/path/to/env/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
DATA=/path/to/datasets/hendrycks_math
BASE=/path/to/models/Qwen3-4B-Base
HYBRID=/path/to/models/Qwen3-4B

# 1. 沙箱预检（约 15 秒，使用 P0a 的代码执行限额）
bash scripts/run_amp_gap.sh probe-sandbox

# 2. 数据核对（CPU，几秒）：应看到约 7,500 道可用题、selected_p0a = 1000、五个等级各 200
bash scripts/run_amp_gap.sh verify-data --data "$DATA"

# 3. 主模型冒烟（四卡，约 5 分钟，主要是加载模型），verdict 应为 PASS
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_amp_gap.sh smoke \
  --id amp-gap-base-smoke-v2 --data "$DATA" --model "$BASE"

# 4. 主模型正式 P0a（四卡，预计 30–60 分钟）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_amp_gap.sh p0a \
  --id amp-gap-base-p0a-v2 --data "$DATA" --model "$BASE" --smoke-run amp-gap-base-smoke-v2

# 5. 副模型（带思考开关）冒烟与正式 P0a（四卡，预计 1–1.5 小时）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_amp_gap.sh smoke \
  --id amp-gap-hybrid-smoke-v1 --data "$DATA" --model "$HYBRID" --hybrid-thinking
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_amp_gap.sh p0a \
  --id amp-gap-hybrid-p0a-v1 --data "$DATA" --model "$HYBRID" --hybrid-thinking \
  --smoke-run amp-gap-hybrid-smoke-v1
```

- 每个运行名只能用一次，重跑请换新名字，例如 `-v3`。`-v1` 的主模型运行属于旧提交，保留原目录即可。
- 看板会显示「放大差距 P0a」各阶段和分卡进度。
- 主模型的结果决定去留。时间紧时，先跑完第 1–4 步并回传，第 5 步可以随后补跑。

**常见问题**

| 现象 | 处理 |
| --- | --- |
| 显存不足 | 加 `--gpu-memory 0.75` |
| 汇总里 `code` 臂的 `infra_rate` 超过 1%，或报 startup budget | 设 `export WORK3_SANDBOX_STARTUP_SECONDS=300` 后换新运行名重跑 |
| 报错信息里的退出码含 `None` | 表示该卡当时仍在正常运行，因其他卡失败被一并停止，看其他卡的日志 |
| 不要做的事 | 不要修改预算、种子和题目数，不要跳过冒烟 |

## 四、回传

每个运行目录在 `$WIKI_RSI_ROOT/<运行名>/` 下。

- **完成时回传：**
  - `wiki_rsi_amp_gap_summary.json`（含判定、各臂校准、放大差距和 `token_range` 越界计数）；
  - `inputs.json`；
  - `rank-*.jsonl`：只有答案键、token 数和结束原因，几 MB；
  - `status-*.json`（各卡耗时与越界计数）。
- **中断时回传：**
  - `incomplete.json`；
  - 每个 `worker-*.log` 的最后 100 行。
- **留在厂内：** `texts-*.jsonl.gz` 是完整生成文本，后续诊断如有需要再取。
- 同时回传 `probe-sandbox` 和 `verify-data` 的完整输出，以及 `git rev-parse HEAD`。
