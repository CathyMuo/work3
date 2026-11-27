# 厂内运行：答案先验去留检验（方案 A 第一周）

本检验回答方案 A 的前提：稠密检索器是否按"模型已经相信的答案"给段落打分，以及这种偏差在模型记错或不知道的地方是否造成损害。三项读数中至少两项达标才继续方案 A，否则转方案 B，判定日 12 月 4 日。门槛见[冻结协议](../../research/protocols/answer_prior_p0.md)。

- 共五步：单元测试 → 核对输入 → 语料准备（CPU）→ 四卡冒烟 → 四卡正式运行。
- 只推理，不训练。运行期间请暂停保活训练，跑完再拉起。
- 预计用时：语料准备 10–20 分钟；冒烟 15–25 分钟；正式运行 2–3 小时，其中 Qwen3-Embedding 编码 2,100 万段约占一半。首次实测后再更新。

## 一、需要带入厂内的模型与数据

**新带入**

| 类型 | Hugging Face ID | 用途 | 约大小 |
| --- | --- | --- | --- |
| 模型 | `Qwen/Qwen3-Embedding-0.6B` | 第二个检索器 | 1.2 GB |
| 数据集 | `akariasai/PopQA`（只有 `test.tsv`） | 问题集，带关系、主语页标题和热度 | 5 MB |

在能联网的机器上下载，再整目录拷入厂内：

```bash
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B
huggingface-cli download --repo-type dataset akariasai/PopQA --local-dir datasets/PopQA
```

**厂内已有，直接用**

| 名称 | 位置（按以往指南） |
| --- | --- |
| wiki-18 语料 | `$DRZERO_DIR/corpus/wiki-18.jsonl`（21,015,324 行） |
| `intfloat/e5-base-v2` | P0 检索服务用的本地 E5 路径 |
| `Qwen/Qwen3-4B` | 放大差距 P0a 的副模型 |
| 原 E5 索引（可选） | `$DRZERO_DIR/corpus/e5_Flat.index`，只用来核对编码一致性 |

本检验不需要 EntityQuestions、T-REx 和 Qwen3-Embedding-4B，后续主实验再带入。

## 二、运行环境与磁盘

- **Python 环境。** 直接用放大差距 P0a 的环境（`vllm >= 0.8.5`、`transformers >= 4.51`、`torch`、`numpy`），不需要 FAISS。
- **磁盘。**
  - `--index-root` 所在磁盘需要约 80 GB 空闲，用来保存全量语料向量：E5 约 32 GB，Qwen3-Embedding 约 43 GB。这些向量请保留，方案 A 后续只训练查询侧时直接复用。
  - 语料准备产物约 3–5 GB，在 `$WIKI_RSI_ROOT/answer-prior-prepare/`。

## 三、执行

在 Work3 仓库根目录拉取本指南所在提交，然后执行：

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export AP_PYTHON=/path/to/amp-env/bin/python        # 放大差距 P0a 用的环境
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
CORPUS=$DRZERO_DIR/corpus/wiki-18.jsonl
POPQA=/path/to/datasets/PopQA
GEN=/path/to/models/Qwen3-4B
E5=/path/to/models/e5-base-v2
QE=/path/to/models/Qwen3-Embedding-0.6B
E5_INDEX=$DRZERO_DIR/corpus/e5_Flat.index              # 可选；没有就去掉下面的 --e5-index
INDEX_ROOT=/path/with/80GB/answer-prior-index
MODELS=(--corpus "$CORPUS" --generator "$GEN" --e5 "$E5" --qwen3-embed "$QE" --e5-index "$E5_INDEX")

# 0. 单元测试（CPU，几秒）：应全部 OK
$AP_PYTHON -m unittest discover -s tests -p test_answer_prior.py -v

# 1. 核对输入（CPU，几秒）：popqa_rows 应为 14267，relations 16 种，e5_index 显示 21015324 行
bash scripts/run_answer_prior.sh verify --popqa "$POPQA" "${MODELS[@]}"

# 2. 语料准备（CPU，10–20 分钟，只跑一次）：verdict 应为 READY，rows 21015324
bash scripts/run_answer_prior.sh prepare --corpus "$CORPUS" --popqa "$POPQA"

# 3. 冒烟（四卡，15–25 分钟）：verdict 应为 PASS
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_answer_prior.sh smoke --id answer-prior-smoke-v1 "${MODELS[@]}"

# 4. 正式运行（四卡，2–3 小时）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_answer_prior.sh run --id answer-prior-run-v1 \
  --smoke-run answer-prior-smoke-v1 --index-root "$INDEX_ROOT" "${MODELS[@]}"
```

正式运行依次做五件事：

1. 四卡闭卷作答，划分认知层；
2. 四卡编码全量语料并检索，先 E5 后 Qwen3-Embedding；
3. 同时在 CPU 上跑 BM25；
4. 构造改写段落，并用 fp32 重新打分；
5. 计算三项读数和判定。

**注意**

- 每个运行名只能用一次，重跑请换新名字（如 `-v2`）。`prepare` 只跑一次；它已完成时再跑会直接报错，属正常。
- 正式运行中断后重跑，已经编码完整的语料分片按清单自动复用，不会重新编码。
- 看板（`bash scripts/run_board.sh`）的「研究说明」页显示当前研究问题与动机。「最近一次实验」显示总进度和各阶段进度条，每段标出已用时间和预计剩余时间，并给出预计完成时刻；BM25 与 GPU 阶段并行，单独显示。阶段开始报告处理量前按先验估计剩余时间，之后改用实测速度。

**常见问题**

| 现象 | 处理 |
| --- | --- |
| 闭卷作答阶段显存不足 | 加 `--gpu-memory 0.75` |
| 编码阶段显存不足 | 先把其他占卡进程停掉；仍不足请回传日志 |
| 报 `flash_attn` 或注意力实现相关错误 | 默认是 `sdpa`，不要改；如已改，换回默认值后用新运行名重跑冒烟 |
| 报错信息里的退出码含 `None` | 表示该卡当时仍在正常运行，因其他卡失败被一并停止，看其他卡的日志 |
| 不要做的事 | 不要修改门槛、题目、种子，不要跳过冒烟 |

## 四、回传

运行目录在 `$WIKI_RSI_ROOT/<运行名>/` 下。

- **完成时回传：**
  - `wiki_rsi_answer_prior_summary.json`（判定、三项读数、各层计数、BM25 与混合检索记录、编码一致性）；
  - `inputs.json`、`knownness.json`、`records.jsonl`（逐题分层、各系统命中和 T1 名次，不含段落原文，约 10 MB）；
  - `encode-*.json`、`status-score-*.json`；
  - `$WIKI_RSI_ROOT/answer-prior-prepare/wiki_rsi_answer_prior_prepare_summary.json`。
- **中断时回传：**
  - `incomplete.json`；
  - 每个 `worker-*.log` 的最后 100 行。
- **留在厂内：**
  - `items-*.json`（改写段落原文）、`search-*.npz`、`known-*.jsonl`（生成原文），后续诊断如有需要再取；
  - `--index-root` 下的语料向量请长期保留。
- **同时回传：** 第 0–2 步的完整输出，以及 `git rev-parse HEAD`。
