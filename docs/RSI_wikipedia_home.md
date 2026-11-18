# 家服运行 Wikipedia RSI 取证诊断

依次执行 W0 输入审计、W1 历史重建、GPU 冒烟和完整回放。使用原 P0 第 3 轮冻结解题器、保存的 Wikipedia 搜索轨迹，以及两个数据集原有的支持句。无需新增人工标注或启动检索服务。

本入口默认用修复后的公开支持句包运行 W0b。先核验厂内原轨迹和测试 Parquet 到家服后的 SHA-256。若两个数据集未达到候选门槛，只保留审计汇总等待决策。

## 1. 准备原 P0 的三项输入

如果原数据在另一台机器，先将三项文件复制到家服的私有数据目录。

| 变量 | 文件 |
| --- | --- |
| P0_EVAL_DUMP | 原 P0 的 OUT_ROOT/eval/iter3 中唯一的合并 JSONL，含 input、output、score、data_source |
| P0_TEST_PARQUET | 同次 P0 评测读取的 Dr. Zero/data/test.parquet，含 prompt 和 reward_model |
| P0_ITER3_MODEL | 原 P0 第 3 轮转出的完整 HF 目录，含权重、配置和 tokenizer |

原权重默认位置为 Dr. Zero checkout 下的 checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf。Dr. Zero 源码需处于仓库规定的 98c4a6d 提交。模型配置和权重会在准备和回放前分别计算 SHA-256。

## 2. 运行 CPU 审计与历史重建

在家服 Work3 仓库根目录执行，将三个 /absolute/path 路径换成真实位置。两项数据文件也可通过 DRZERO_DIR 和 OUT_ROOT 推断默认位置。

~~~bash
cd /mnt/wh/RAG/work3
git log -1 --oneline
export DRZERO_DIR=/mnt/wh/RAG/upstreams/drzero
export DIAG_PYTHON=/mnt/wh/miniconda3/envs/searchr1/bin/python
export REPLAY_PYTHON=/mnt/wh/miniconda3/envs/vllm/bin/python
export P0_EVAL_DUMP=/absolute/path/to/eval/iter3/0.jsonl
export P0_TEST_PARQUET=/absolute/path/to/data/test.parquet
export P0_ITER3_MODEL=/absolute/path/to/solver_iter3_hf
RUN="wiki-w1-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_wiki_rsi.sh audit --run "$RUN"
bash scripts/run_wiki_rsi.sh prepare --run "$RUN"
~~~

默认产物目录为 outputs/wiki_rsi/$RUN。四条命令使用同一个 RUN。自定义产物根目录时，在实验和看板进程中设置相同的 WIKI_RSI_ROOT。W0 要求每数据集 60 个缺证据候选，W1 准备要求每数据集至少 40 道可回放题，才打开完整回放。少于门槛时可对已有合格题做技术冒烟。

## 3. 单卡冒烟与完整回放

~~~bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi.sh smoke --run "$RUN"
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi.sh replay --run "$RUN"
~~~

默认一张可见 GPU，BF16，显存配额 70%。可选 --gpu-memory、--tp 2 或 --tp 4，同次回放的各分支共用配置。冒烟每数据集最多两道缺证据题，先检查模型加载、格式、token 数和延迟。完整回放最多 180 题，最多 600 次生成，每次最多 192 个新 token。GPU 阶段使用 Python 3.10 及以上与支持直接输入 token IDs 的 vLLM。

## 4. 看板和结果

现有看板研究说明页每 8 秒刷新进度。另起看板实例可使用 BOARD_PYTHON="$DIAG_PYTHON" bash scripts/run_board.sh。本地在单独终端代理后，打开[研究说明与实验进度](http://127.0.0.1:18766/#brief)。

~~~bash
ssh -N -L 127.0.0.1:18766:127.0.0.1:8766 xd-5090
~~~

- w0/wiki_rsi_input_summary.json 记录输入与流失分母。
- wiki_rsi_prepare_summary.json 记录排除原因、输入与模型散列、完整回放准入。
- wiki_rsi_smoke_summary.json 记录冒烟结果。
- wiki_rsi_replay_summary.json 记录配对改善与回退、区间、有效三臂、裸历史重放、吞吐和 GPU 小时。

逐题输出、prompt token、轨迹和权重留在运行机。阶段失败后可修正并重试，变更已完成阶段的输入或重复已完成阶段时使用新的 RUN。W2 等本轮读数，再决定是否执行预算内的真实查询。

[冻结协议与统计口径](../research/protocols/RSI_wikipedia.md)
