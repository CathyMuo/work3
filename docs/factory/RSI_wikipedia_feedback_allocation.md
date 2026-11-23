# 四卡动态监督分配实验

从原 Wikipedia 离线检索与已有 W7b 工件启动。家服负责协议、代码和决策。厂内负责检查环境、运行命令和回传完整汇总。没有额外标注步骤。所有五阶段和门槛见[冻结协议](../../research/protocols/RSI_wikipedia_feedback_allocation.md)。

## 准备

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export DRZERO_DIR=/root/develop/workspace/drzero
export P0_ITER1_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter1_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter1_hf"
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"

export WIKI_RSI_FEEDBACK_GPUS=0,1,2,3
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID=p0-e5-wiki18
RUN=feedback-20261123-a
SOURCE="$WIKI_RSI_ROOT/wiki-w0c-20261118-110954"

test -s "$P0_ITER1_MODEL/config.json"
test -s "$P0_ITER3_MODEL/config.json"
test -s "$SOURCE/w7_live_prepared.json"
test -s "$SOURCE/wiki_rsi_w7_live_prepare_summary.json"
```

卡号换成四张实际空闲 H800。R1 和 R3 指同次 P0 的第一、三轮 HF 转换模型。R3 必须和原 W7b 指纹一致。复用原 W0c 至 W7b 目录及原测试 parquet、参考包、Wikipedia 语料、Dr. Zero 固定检索。若 W7b 用过自定义参数，给 `lock` 传入同一 `--references`、`--test-parquet`、`--eval-dump` 或 `--corpus` 路径。工作过程只写本地机密输出目录。

## 执行顺序

```bash
bash scripts/run_wiki_rsi_feedback.sh lock --run "$RUN" --original-run "$SOURCE" --drzero "$DRZERO_DIR" &&
bash scripts/run_wiki_rsi_feedback.sh prepare --run "$RUN" &&
bash scripts/run_wiki_rsi_feedback.sh mechanism --run "$RUN" &&
bash scripts/run_wiki_rsi_feedback.sh pilot --run "$RUN" &&
bash scripts/run_wiki_rsi_feedback.sh confirm --run "$RUN"
```

也可逐条执行并在阶段间传回汇总。分阶段执行时保持相同 `RUN` 和环境变量。返回码 2 且汇总显示 `NO-GO` 时，后续阶段不会启动。环境错误时查看该阶段的终端和工作目录下 `worker.log`。修复环境问题后保留原配置，只重跑未写出终态的阶段。阶段时间上限是研究停止线，不作为重试理由。

预期依次生成 `wiki_rsi_feedback_lock_summary.json`、`wiki_rsi_feedback_prepare_summary.json`、`wiki_rsi_feedback_mechanism_summary.json`、`wiki_rsi_feedback_pilot_summary.json`、`wiki_rsi_feedback_confirm_summary.json`。请回传本次实际产生的所有完整汇总，包括首个 NO-GO。工件、模型、逐题打分和日志保留在 `$WIKI_RSI_ROOT/$RUN`。

## 实时进度

在运行所在主机上，用同一 `WIKI_RSI_ROOT` 启动：

```bash
BOARD_PYTHON="$REPLAY_PYTHON" bash scripts/run_board.sh
```

从现有 SSH 端口转发访问 `http://127.0.0.1:18766/#wiki`，或直接在运行主机查看 `http://127.0.0.1:8766/#wiki`。看板读取原子写入的 `progress.json`，运行时每两秒刷新，显示阶段、四卡各自的处理量、当前试更新或评估动作、心跳和 GO / NO-GO。看板只读。同一台机器上文件实时可见，跨主机可用已有的 SSH 转发指向运行主机。
