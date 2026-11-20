# 证据状态训练与自然题回放

反馈 #43 中原冻结答题器筛选只保留六对训练数据，四卡阶段未启动。新阶段直接复用原 720 个真实检索核验过的配对，训练仍以两状态补回不同缺失事实为监督。答题结果进入评估。整个阶段不重新扫描语料、不发新查询、不重复跑答题器。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
RUN=wiki-w0c-20261118-110954

bash scripts/run_wiki_rsi_w2a.sh switch-transfer-prepare --run "$RUN"
```

CPU 准备需要原 `switch_prepared.json`、原验证工作目录中的 `switch_encoded.json`、`w7_live_prepared.json`、`w7_live_cases.json` 及各阶段汇总。校验页面隔离和上游哈希。保留所有可编码的取证反转训练样本。构造题留出复用原 144 对的答题矩阵，并报告按「补查缺失年份」选择查询的无训练规则。

准备完成且四张 H800 空闲后执行。

```bash
WIKI_RSI_SWITCH_GPUS=0,1,2,3 bash scripts/run_wiki_rsi_w2a.sh switch-transfer --run "$RUN"
```

四组分别跑普通偏好和联合目标，各两个种子，固定六遍和相同训练预算。自然题回放用 W7b 的 384 道题，共享相同首查与候选池。候选池来自冻结续查、固定规则与两个旧正例策略，同文本只保留先前顺序中的第一个记录。各排序器按首查后状态对候选打分，最多选一条续查。逐题结果来自该候选已实现的返回及冻结作答，无候选时沿用冻结首查。汇总同时列冻结排序器和原冻结续查，用实际记录统计证据、严格 EM 和查询数。

本轮自然题是已观察过的开发集，作用是检查排序能力有没有跨出日期模板。只有联合目标在自然题上的收益可跨种子复现，才用未见题做独立确认。请回传 `wiki_rsi_switch_transfer_prepare_summary.json` 与 `wiki_rsi_switch_transfer_summary.json`，逐题结果及权重留厂。

反馈 #44 已跑完本轮，结论为 `no_go`。汇总新增的 `research_verdict` 单独显示继续条件，`decision` 中的 `natural_replay_complete` 只表示程序成功跑完。旧 `oracle_answer_count` 是现有候选中必须补查的事后上界，新增 `oracle_with_stop_answer_count` 和 `oracle_with_stop_evidence_count` 同时允许保留首查。后续按 [缺失事实补查](RSI_wikipedia_gap_follow_up.md) 执行。
