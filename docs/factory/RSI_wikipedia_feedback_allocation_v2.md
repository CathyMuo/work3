# 动态监督分配 v2 厂内运行

本次复用成功的 v1 输入冻结，将长引文生成改为对真实证据窗口选编号。保留 v1 的输出目录及 NO-GO 回执。首轮仍用原 3B 检查点。使用新运行名，阶段内的去留条件已写入[冻结协议](../../research/protocols/RSI_wikipedia_feedback_allocation_v2.md)。

## 拉取和准备

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
DRZERO_DIR=/root/develop/workspace/drzero
export P0_ITER1_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter1_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter1_hf"
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID=p0-e5-wiki18
export WIKI_RSI_FEEDBACK_GPUS=0,1,2,3

V1="$WIKI_RSI_ROOT/feedback-20261123-a"
RUN=feedback-20261123-v2
test -s "$V1/feedback_locked.json"
test -s "$V1/wiki_rsi_feedback_lock_summary.json"
test -s "$V1/wiki_rsi_feedback_prepare_summary.json"
```

沿用 v1 检索配置和 Python 环境。真实用卡阶段需四张已空闲 H800。已有保活训练按原安排继续，占用的卡空出来后再开此实验。可先单独执行 CPU 的 `lock`，取证核对和模型指纹校验会读取已冻结工件。

## 执行

```bash
bash scripts/run_wiki_rsi_feedback.sh lock --run "$RUN" --import-v1 "$V1" &&
bash scripts/run_wiki_rsi_feedback.sh prepare --run "$RUN" &&
bash scripts/run_wiki_rsi_feedback.sh mechanism --run "$RUN" &&
bash scripts/run_wiki_rsi_feedback.sh pilot --run "$RUN" &&
bash scripts/run_wiki_rsi_feedback.sh confirm --run "$RUN"
```

`prepare` 先在 512 源上做最多 30 分钟的四卡短筛。至少 80 对且三类各至少 10 对才自动处理剩余源页。完整产量门槛仍为 640 对及原类别要求。达到 NO-GO 时返回 2，后续阶段自动停止。

请回传新运行目录中实际生成的所有完整 `wiki_rsi_feedback_*_summary.json`，包括短筛和漏斗字段。逐题材料和权重留在运行主机。看板服务读取同一 `WIKI_RSI_ROOT`，用 `BOARD_PYTHON="$REPLAY_PYTHON" bash scripts/run_board.sh` 启动。已有服务需要使用当前提交重新启动，以显示 v2 的复用、短筛和停止状态。
