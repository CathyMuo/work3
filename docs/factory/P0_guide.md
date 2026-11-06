# P0 厂内指导：复现 Dr. Zero Qwen2.5-3B（4×H800）

版本：tag `p0-r1`（2026-11-06，已处理厂内反馈 #1）。
协议（已冻结，不要改）：[research/protocols/P0_drzero_reproduction.md](../../research/protocols/P0_drzero_reproduction.md)，执行以末尾的 **Addendum 2026-11-06** 为准。
判定：第 3 轮 7 个数据集平均 EM ≥ **0.306** 为 PASS（论文 0.326）。

**规则**：厂内不推送、不写代码、不改 `research/DECISION_LOG.md`、`research/STATE.json` 和协议文件。出现 INVALID / FAIL / 报错时，不要自己改门槛或超参数，按第 7 节发回家服。

---

## 0. 开跑前：停掉显卡保活程序

卡上跑的是保活程序，不是研究任务，**不用等卡空闲**：确认是保活程序 → 停掉 → 跑 P0。

- 停的时候连带清掉它的 ray worker 孤儿进程（`ray::WorkerDict...`），否则会残留显存。
- 工作2 常驻的检索服务（占 :8000、约 40 GB）可以留着：P0 用 8100 / 8101 端口，清理时只杀 P0 自己启动的进程，不会碰它。显存要紧时也可以停掉它。
- `bash $P0 check` 会列出卡上所有计算进程；不为空就报错，提醒你先清干净。如果确实要保留某个进程（比如上面的检索服务），设 `ALLOW_BUSY_GPU=1` 再运行。
- P0 跑完后保活程序由厂内自行拉起，家服不需要为此做任何事。

## 1. 代码

```bash
cd <工作目录>
git clone https://github.com/CathyMuo/work3.git          # 已有就：git fetch --tags && git checkout p0-r1
cd work3 && git checkout p0-r1 && cd ..
git clone https://github.com/facebookresearch/drzero.git
cd drzero && git checkout 98c4a6dd2495e85d8b4d947ab4c030e540c922c5 && cd ..
```

Dr. Zero 的文件一个都不要改。所有适配由 `run_p0.sh patch` 生成到 `work3_p0/patched/`，或者在运行时由 `work3.p0_launch` 包一层完成。

## 2. 环境（新建 drzero，和 searchr1 分开）

Dr. Zero 自带 verl 0.5.0 源码（`drzero/verl/`），**不要 `pip install verl`**。

```bash
conda create -n drzero python=3.10 -y
conda activate drzero
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "sglang[all]==0.4.6.post5" "torch-memory-saver>=0.0.5"
pip install "tensordict>=0.8.0,<=0.9.1,!=0.9.0" "numpy<2.0.0" "pandas<3" "pyarrow>=19.0.0" \
    accelerate codetiming datasets dill hydra-core peft pybind11 pylatexenc "ray[default]>=2.10" \
    torchdata transformers wandb huggingface_hub liger-kernel math_verify uvicorn fastapi httpx
pip install flash-attn --no-build-isolation
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y      # 或沿用工作2环境里能用的 faiss-gpu 安装方式
```

- `pandas` 必须 < 3（厂内已实测：开 COW 会写坏题池，`pool_check` 会判 INVALID）。
- sglang 必须是 0.4.6.post5，torch 必须是 2.6.0；pip 冲突时以这两个为准，把冲突信息发回。

## 3. 变量（按厂内实际路径）

```bash
conda activate drzero
export DRZERO_DIR=<工作目录>/drzero
export MODEL_3B=<models>/Qwen2.5-3B-Instruct
export CORPUS_JSONL=<models>/wiki-18/wiki-18.jsonl
export INDEX_PATH=<models>/wiki-18/e5_Flat.index
export E5_MODEL=<models>/e5-base-v2
export TRAIN_PARQUET=<models>/nq_hotpotqa_train/train.parquet
export TEST_PARQUET=<models>/nq_hotpotqa_train/test.parquet
export NGPU=4
# 端口默认 RET_PORT=8100、REWARD_PORT=8101；被占用时改成别的空闲端口
P0=<工作目录>/work3/scripts/p0/run_p0.sh
```

设了 `TRAIN_PARQUET` / `TEST_PARQUET` 后，`prepare` 完全不联网。不需要 RUC-NLPIR 的 FlashRAG 数据，也不要从 jsonl 重建 test.parquet。

产物在 `$DRZERO_DIR/{data,checkpoints}`；日志、汇总、补丁后的脚本在 `$DRZERO_DIR/work3_p0/`。

## 4. 运行（tmux 里一步一步来）

```bash
bash $P0 check      # 版本、GPU 进程、端口、verl 导入路径；有问题先解决
bash $P0 prepare    # train / test / test_1200 三个 parquet，打印各数据集行数
bash $P0 patch      # 生成 4 卡版脚本，打印每处替换；有残留 8000/8001 会直接报错
bash $P0 iter1      # 出题者 50 步 → 生成题池 → 题池检查 → 解题者 50 步 → 转 HF
bash $P0 eval1
bash $P0 iter2
bash $P0 eval2
bash $P0 iter3
bash $P0 eval3
bash $P0 eval0      # 基座模型，只作参考
bash $P0 gate       # 写 work3_p0/p0_summary.json
```

- 每一步失败都会停下，打印 `FAILED <阶段>`，日志在 `work3_p0/logs/<阶段>.log`。修好后重跑这一步，前面的不用重来。
- 每步结束都会自动清理：只杀环境变量里带 `WORK3_P0_TAG=work3-p0` 的进程（P0 自己的检索、SGLang、ray），其他项目的进程不会动。中途手动 Ctrl-C 之后，运行 `bash $P0 cleanup`。
- 每次启动检索后，脚本会等检索能应答再继续（最多 30 分钟）。日志里的 `retriever ready on :8100 after N s` 是索引加载耗时，请一并发回。

**常见问题**
- **显存不够**：先 `export GEN_MEM=0.7`，重跑 `patch` 和失败的那一步；还不行就 `export FAISS_GPU=0`（检索改用 CPU，慢但不占显存），同样重跑 `patch`。
- **wandb**：默认离线。
- **磁盘**：每个 FSDP 检查点（含优化器）约 35 GB，总计约 450 GB（NAS 可用 1.9 TB，够用）。某一轮完成后可以删掉该轮的 `global_step_25/75/125`。**不要删** `global_step_50/100/150`、`solver_iter*_hf`、`data/zero_challenger_iter*.parquet`，P1 要用。

## 5. 耗时估计（第 1 轮跑完后按 timings.txt 重估）

| 步骤 | 估计 |
| --- | --- |
| 搭环境 | 半天到一天 |
| 每次检索启动（NAS 上 61 GB 索引） | 数分钟，每个阶段一次 |
| 出题者 50 步 | ~8 h（论文 8×H100 4.2 h） |
| 生成题池（33,923 个种子 × 5 次采样） | ~3–5 h |
| 解题者 50 步 | ~6 h |
| 每次评测（51,713 题，batch 256，多轮贪心） | **~2.5–4 h**（比上一版估计高） |
| 合计（3 轮 + 4 次评测） | **约 3.5–4.5 天** |

## 6. 并行度说明

只有 `tp` 和 `gpus` 决定并行度：出题者 / 解题者 `tp=2`，生成题池 `tp=1`。`dp` 只在出题者脚本里使用一次，用于奖励服务器的 `--dp-size`；在其他脚本里是不起作用的变量。

## 7. 发回家服的内容

```bash
cd $DRZERO_DIR/work3_p0
cat p0_summary.json
cat pool_check_iter*.json
cat timings.txt
grep -h "retriever ready" logs/*.log
grep -h "Final rewards" logs/iter*_challenger.log | awk 'NR%10==0' | tail -n 30
pip freeze | grep -iE "^(torch|sglang|transformers|flash|faiss|pandas|tensordict|ray|numpy)=="
nvidia-smi --query-gpu=name,memory.total --format=csv
cd <工作目录>/work3 && git describe --tags --always
```

失败时额外发失败阶段日志的最后 80 行：`tail -n 80 logs/<阶段>.log`。

检查点、题池和评测明细都不用发回，但要保留在厂内。
