# 厂内运行：方案 B 首个实验（补检索发证）

本实验检验方案 B 的卖点：在同样的错误率保证、同样的检索开销下，"先读少量段落，拿不准的再读更多"能否比"所有论断统一多读"多保留至少 10 个点的正确论断。同时测判官的误判是否集中在模型自信的错误上。门槛见[冻结协议](../../research/protocols/claim_cert_b0.md)。

- **不需要带入新资产。** 全部复用方案 A 留在厂内的产物：
  - Qwen3-4B 的闭卷答案；
  - wiki-18 的 E5 全量向量（不会重新编码）；
  - 语料准备产物。
- **只推理，不训练。** 四卡全开，批量按上限设置；运行期间请暂停保活训练。
- **预计用时。** 按修订 1 带外部判官（Qwen3-32B）运行：冒烟 15–20 分钟，正式运行约 1–1.5 小时；不带判官时为 10–15 分钟和 30–45 分钟。首次实测后再更新。

## 一、需要在厂内已有的东西

| 名称 | 位置（按方案 A 的回执） |
| --- | --- |
| 方案 A 正式运行目录 | `$WIKI_RSI_ROOT/answer-prior-run-v1/`（要用其中的 `knownness.json`、`inputs.json` 和汇总） |
| 语料准备产物 | `$WIKI_RSI_ROOT/answer-prior-prepare/` |
| 已存语料向量 | `/root/develop/workspace/answer-prior-index`（其中 `e5/` 下的 4 个分片和清单） |
| wiki-18 语料 | `$DRZERO_DIR/corpus/wiki-18.jsonl` |
| 模型 | Qwen3-4B（核验器）、e5-base-v2（与已存向量同一份） |
| 外部判官（修订 1） | `/root/develop/workspace/models/Qwen3-32B` |

Python 环境与方案 A 相同（`AP_PYTHON`），需要 vLLM。

## 二、执行

在 Work3 仓库根目录执行。如果方案 A 时的本地修改还在，先撤掉：

```bash
git checkout -- work3/answer_prior_text.py 2>/dev/null || true
git fetch origin refs/heads/main && git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export AP_PYTHON=/path/to/amp-env/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
COMMON=(--corpus "$DRZERO_DIR/corpus/wiki-18.jsonl" --verifier /path/to/models/Qwen3-4B
        --e5 /path/to/models/e5-base-v2 --index-root /root/develop/workspace/answer-prior-index
        --source-run answer-prior-run-v1)

JUDGE=(--judge /root/develop/workspace/models/Qwen3-32B --judge-tp 2)   # 修订 1：外部判官，每副本两卡、共两副本

# 0. 单元测试（CPU，十几秒）：应全部 OK
$AP_PYTHON -m unittest discover -s tests -p "test_claim_cert.py" -v
$AP_PYTHON -m unittest discover -s tests -p "test_answer_prior.py"

# 1. 冒烟（四卡，15–20 分钟，含两次加载 32B 和一次 64 GB 散列）：verdict 应为 PASS
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_claim_cert.sh smoke --id claim-cert-smoke-j1 "${COMMON[@]}" "${JUDGE[@]}"

# 2. 正式运行（四卡，约 1–1.5 小时）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_claim_cert.sh run --id claim-cert-run-j1 \
  --smoke-run claim-cert-smoke-j1 "${COMMON[@]}" "${JUDGE[@]}"
```

- 带外部判官（Qwen3-32B）的这组就是正式运行，不必另跑一组不带判官的。冒烟与正式运行的判官参数必须一致。
- 请不要用 `ReSearch-Qwen-32B-Instruct` 当判官。它是带思考和检索标签的搜索智能体，读不出第一个 token 的 Yes/No。
- **退路**：32B 判官报错且一时排查不了时，去掉 `"${JUDGE[@]}"`，改用运行名 `claim-cert-smoke-v1` 和 `claim-cert-run-v1` 重跑，并回传判官的报错日志。此时判官误差只用自判官。

**运行流程。** 依次为：核对输入 → 四卡检索（两种查询，复用已存向量）→ 四卡核验 8 种检索配置 →（可选）外部判官 → 50 次划分的分析与判定。

**看板。** 「最近一次实验」显示总进度和各阶段进度条，标出已用时间、预计剩余时间和预计完成时刻。

**批量设置。** 按要求设到上限，以拉满显卡利用率和功率：

| 参数 | 默认值 |
| --- | --- |
| `--gpu-memory` | 0.92 |
| `--max-num-seqs` | 1024 |
| `--max-batched-tokens` | 131072 |
| `--chunk`（每次调用的论断数，乘 8 个提示） | 1024 |

E5 检索每块 4,096 条查询 × 209 万段。

**常见问题**

| 现象 | 处理 |
| --- | --- |
| 核验或判官阶段显存不足（OOM） | 依次试 `--max-batched-tokens 65536`，再试 `--gpu-memory 0.88`；换新运行名重跑冒烟 |
| 判官阶段报张量并行或 NCCL 端口错误 | 回传 `worker-judge-*.log` 最后 100 行；可临时改用 `--judge-tp 4`（单副本四卡）并换新运行名重跑冒烟 |
| 报 "does not match the stored-index manifest" | `--e5` 或 `--index-root` 与方案 A 用的不是同一份；改回方案 A 的路径。程序不会重新编码 |
| 报 "is not a complete answer-prior run" | `--source-run` 指向了冒烟目录，应为 `answer-prior-run-v1` |
| 不要做的事 | 不要改门槛、划分次数和网格，不要跳过冒烟 |

## 三、回传

运行目录在 `$WIKI_RSI_ROOT/<运行名>/` 下。

- **完成时回传：**
  - `wiki_rsi_claim_cert_summary.json`（判定、各 α 与开销下的保留率和增益、判官误差、核验器 AUC、解析率）；
  - `inputs.json`；
  - `claims.jsonl`：逐条论断的得分与标签，不含段落，约 5 MB。
- **中断时回传：**
  - `incomplete.json`；
  - 各 `worker-*.log` 的最后 100 行。
- **留在厂内：** `search.npz`、`verify-*.jsonl`、`judge-*.jsonl`。
- **同时回传：** 第 0 步单元测试的输出，以及 `git rev-parse HEAD`。
- **方案 A 的补件。** 如果还没回传，请一并附上 `answer-prior-run-v1/wiki_rsi_answer_prior_summary.json` 全文。
