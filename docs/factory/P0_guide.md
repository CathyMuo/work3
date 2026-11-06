# P0 厂内指导：复现 Dr. Zero Qwen2.5-3B（4×H800）

协议（已冻结，不要改）：[research/protocols/P0_drzero_reproduction.md](../../research/protocols/P0_drzero_reproduction.md)
判定：第 3 轮 7 个数据集平均 EM ≥ **0.306** 为 PASS（论文 0.326）。

**预计耗时**：搭环境约半天到一天；训练 3 轮每轮约 15–18 小时；评测每次约 1–2 小时。合计约 3–4 天。

**规则**：厂内不推送、不改 `research/DECISION_LOG.md`、`research/STATE.json` 和协议文件。出现 INVALID / FAIL / 报错时不要自己改门槛或超参数，把下面第 6 节的内容发回家服。

---

## 1. 代码

```bash
cd <工作目录>
git clone https://github.com/CathyMuo/work3.git          # 旧的 work3_byd 克隆可改 remote：git remote set-url origin https://github.com/CathyMuo/work3.git
git clone https://github.com/facebookresearch/drzero.git
cd drzero && git checkout 98c4a6dd2495e85d8b4d947ab4c030e540c922c5 && cd ..
```

## 2. 环境（新建，和工作2的 searchr1 环境分开）

Dr. Zero 自带 verl 0.5.0 源码（`drzero/verl/`），**不要 `pip install verl`**，否则会导入错版本（`check` 会报错）。

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

要点：
- `pandas` 必须 < 3。上游写训练池时依赖 pandas 的视图语义，pandas 3 会静默写坏（`pool_check` 会拦下）。
- sglang 必须是 0.4.6.post5，torch 必须是 2.6.0；如果 pip 解析冲突，以这两个为准，把冲突信息发回。

## 3. 数据与模型

沿用工作2已有的文件即可：
- `wiki-18.jsonl`、`e5_Flat.index`（Search-R1 那份 e5 Flat 索引）
- 本地 `Qwen2.5-3B-Instruct` 目录（**目录名必须是 `Qwen2.5-3B-Instruct`**，实验名依赖它）
- 本地 `e5-base-v2`（离线时用）

`prepare` 会从 HuggingFace 下载 `PeterJinGo/nq_hotpotqa_train` 的 `train.parquet` / `test.parquet`。厂内不能联网的话，设 `HF_ENDPOINT` 镜像，或者把这两个文件放进 HF 缓存后设 `HF_HUB_OFFLINE=1`。

## 4. 设置变量

```bash
conda activate drzero
export DRZERO_DIR=<工作目录>/drzero
export MODEL_3B=<路径>/Qwen2.5-3B-Instruct
export CORPUS_JSONL=<路径>/wiki-18.jsonl
export INDEX_PATH=<路径>/e5_Flat.index
export E5_MODEL=<路径>/e5-base-v2          # 能联网可以不设
export NGPU=4
P0=<工作目录>/work3/scripts/p0/run_p0.sh
```

所有产物在 `$DRZERO_DIR/{data,checkpoints}`，日志和汇总在 `$DRZERO_DIR/work3_p0/`。

## 5. 运行（建议在 tmux 里，一步一步来）

```bash
bash $P0 check      # 版本、GPU、verl 导入路径；有 PROBLEMS 先解决
bash $P0 prepare    # 生成 data/zero_ratio4321.parquet 和 data/test.parquet，打印各数据集行数
bash $P0 patch      # 生成 4 卡版脚本到 work3_p0/patched/，打印替换结果
bash $P0 iter1      # 出题者 50 步 → 生成题池 → 题池检查 → 解题者 50 步 → 转 HF
bash $P0 eval1
bash $P0 iter2
bash $P0 eval2
bash $P0 iter3
bash $P0 eval3
bash $P0 eval0      # 基座模型，只作参考
bash $P0 gate       # 写 work3_p0/p0_summary.json
```

每一步失败都会停下并打印 `FAILED <阶段>`，日志在 `work3_p0/logs/<阶段>.log`。修好后重跑这一步即可，前面的步骤不用重来。

**常见问题**
- **检索服务没起来**（搜索调用报连接错误）：上游脚本启动检索后只 `sleep 30`，索引加载可能更慢。先单独启动 `python search/retrieval_server.py ...` 测一下加载时间，发回家服再决定。
- **显存不够**：先 `export GEN_MEM=0.7` 后重跑 `patch` 和失败的那一步；还不行就 `export FAISS_GPU=0`（检索放 CPU，慢但不占显存）。这两项都只影响显存，协议允许。
- **wandb**：默认离线（`WANDB_MODE=offline`）。
- **`lsof: command not found` / `kill: usage`**：上游脚本清端口时的报错，可以忽略。
- **磁盘**：每个 FSDP 检查点（含优化器）约 35 GB，每次训练保存 25 步和 50 步两个，总计约 450 GB，请预留 600 GB。某一轮完成后，可以删掉该轮的 `global_step_25`、`global_step_75`、`global_step_125`。**不要删** `global_step_50/100/150`、`solver_iter*_hf` 和 `data/zero_challenger_iter*.parquet`，P1 要用。

## 6. 发回家服的内容

全部跑完（或中途失败）后，把下面这些贴回来：

```bash
cd $DRZERO_DIR/work3_p0
cat p0_summary.json
cat pool_check_iter*.json
cat timings.txt
grep -h "Final rewards" logs/iter*_challenger.log | awk 'NR%10==0' | tail -n 30
pip freeze | grep -iE "^(torch|sglang|transformers|flash|faiss|pandas|tensordict|ray|numpy)=="
nvidia-smi --query-gpu=name,memory.total --format=csv
```

失败时额外发：失败阶段日志的最后 80 行（`tail -n 80 logs/<阶段>.log`）。

不用发回检查点、题池和评测明细，但都要保留在厂内。
