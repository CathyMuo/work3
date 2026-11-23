# 厂内运行指南 · 方案 C 离线持续强化学习

**先运行预检和四卡冒烟，再运行 C1。只有 C1 的最终判定为 GO 时，才运行 C2。** 工厂只执行与回传，所有实验条件由 [冻结协议](../../research/protocols/continual_rlvr.md) 定义。每个正式运行目录用新 ID，旧输出保留。运行脚本自动打开 Hugging Face 离线模式。

## 厂外准备

- 在联网处从本仓库的公开 `main` 取得发布提交。通过已有厂内镜像或离线 Git bundle 将该提交交接到厂内。
- 使用原版 `Qwen/Qwen2.5-3B-Instruct` 的完整本地 Hugging Face 目录，含 tokenizer、generation config、全部 safetensors 和分片索引。已有原版快照可以复用。
- 用同版本的 Linux Python 和 CUDA PyTorch 环境打包所需 wheels。运行依赖为 Python 3.10 或更新版本、可用 CUDA 与 NCCL 的 PyTorch、`transformers==4.56.2` 和 `peft==0.17.1` 及其依赖。数据打包另需 PyArrow。预检会打印环境版本和本地模型 SHA-256。厂内用 `pip install --no-index --find-links /path/to/wheelhouse transformers==4.56.2 peft==0.17.1` 从本地 wheelhouse 安装，或使用已准备的同版本环境。

数据 bundle 可直接从厂外拷入。需要重建时，在联网处从两份固定版本各取一份训练 parquet，然后执行以下命令。输出有输入版本和 SHA-256，可在断网后台复验。源文件与生成的 bundle 都保存在数据目录。

```bash
mkdir -p data/continual
a_rev=bd2a93551b503a395fadd1a740d957559cfe6f3c
b_rev=47c03c73621c4aab2b824b7818681117d662770e
curl -fL "https://huggingface.co/datasets/allenai/RLVR-MATH/resolve/$a_rev/data/train-00000-of-00001.parquet" -o data/continual/math.parquet
curl -fL "https://huggingface.co/datasets/allenai/RLVR-IFeval/resolve/$b_rev/data/train-00000-of-00001.parquet" -o data/continual/if.parquet
python -m work3.continual_rlvr_data \
  --a data/continual/math.parquet --a-revision "$a_rev" \
  --b data/continual/if.parquet --b-revision "$b_rev" \
  --out data/continual/bundle.json
```

## 厂内第一步

如果用离线 Git bundle，把代码包带入已有厂内克隆，在 main 上运行以下命令。增量包会先校验厂内是否包含其基线提交。

```bash
git bundle verify /path/to/work3-c.bundle
git fetch /path/to/work3-c.bundle refs/heads/main
git merge --ff-only FETCH_HEAD
```

在发布提交的仓库根目录，把以下三个路径换成厂内对应位置。默认四卡用 GPU 0–3，若调度器分配其他四张卡，使用实际可见编号。

```bash
export CONTINUAL_PYTHON=/path/to/continual-env/bin/python
export C_MODEL=/path/to/Qwen2.5-3B-Instruct
export C_BUNDLE=/path/to/bundle.json
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"

bash scripts/run_continual_rlvr.sh preflight --model "$C_MODEL" --bundle "$C_BUNDLE"
bash scripts/run_continual_rlvr.sh smoke --model "$C_MODEL" --bundle "$C_BUNDLE" \
  --id continual-smoke-20261123
```

四卡冒烟要完成一次真实策略更新，并对 8 道 B 题各生成两次。冒烟只验程序和卡，不评价方法。冒烟报错时保留终端日志与 `incomplete.json`，先修复本地依赖、权重、数据或 CUDA 环境。

## C1 与自动准入的 C2

```bash
bash scripts/run_continual_rlvr.sh necessity --model "$C_MODEL" --bundle "$C_BUNDLE" \
  --id continual-c1-20261123
cat "$WIKI_RSI_ROOT/continual-c1-20261123/wiki_rsi_continual_necessity_summary.json"
```

预计 C1 数量为 A 前后各 512 条评估响应，B 前后各 8,192 条评估响应。A 在 C1 训练 48 步。48 小时为投入上限。C1 GO 要求 A 提升至少 5 个百分点、两组初始 B 覆盖至少 20%、B 两组覆盖均下降且平均下降至少 5 个百分点。

仅在终态为 GO 时，同一个提交和环境执行以下命令。入口读取 C1 原始评分、重算门槛，并校验模型和数据。

```bash
bash scripts/run_continual_rlvr.sh method --model "$C_MODEL" --bundle "$C_BUNDLE" \
  --necessity-run continual-c1-20261123 --id continual-c2-20261123
cat "$WIKI_RSI_ROOT/continual-c2-20261123/wiki_rsi_continual_method_summary.json"
```

C2 最多 12 小时。预期两种子各有六个方法记录，共 12 个完整的逐臂结果。每臂的 A 与 B 阶段分别为 48 和 32 个更新，B 有三次自由采样评估。最终门槛包含 B 增益、A 保留、成本账与代理有效性。若 C1 或 C2 NO-GO，按冻结条件停下。INCOMPLETE 时回传现有结果和日志，以便判别环境和预算。

## 进度与回传

同一工作目录和 `WIKI_RSI_ROOT` 下用 `bash scripts/run_board.sh` 启动原看板，实验页跟踪真实阶段、进度和心跳。若看板运行在另一台机器，把同一运行目录的 `progress.json` 同步到该看板读取的根目录，或跑在同一台机器并通过已有 SSH 转发查看。

用已约定的私有交接方式回传以下文件和终端日志。

- C1 的 `inputs.json`、`wiki_rsi_continual_necessity_summary.json`、完整 `necessity_rollouts.json`。
- C2 的 `inputs.json`、`admission.json`、`wiki_rsi_continual_method_summary.json` 及所有 `<seed>-<arm>.json`。
- 失败或超时的 `incomplete.json`、`progress.json` 和已经写出的全部评分文件。
