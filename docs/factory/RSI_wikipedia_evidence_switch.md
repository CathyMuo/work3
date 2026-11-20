# 正式方法首轮查询偏好训练

使用相同的固定 Wikipedia 检索器和冻结 R3。在同一道题上保留相同两条候选查询，交换已展示的事实，验证最佳查询是否反转。训练数据由语料自动构造，不读取评估集的答案。W7b 已完成，可使用原 RUN 和检索服务。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export WIKI_RSI_CORPUS="$DRZERO_DIR/corpus/wiki-18.jsonl"
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
RUN=wiki-w0c-20261118-110954

bash scripts/run_wiki_rsi_w2a.sh switch-prepare --run "$RUN"
```

准备阶段在 CPU 挖掘机构成立、场所开放和作品发表时间，使用原索引实时核验至多 12,288 条候选查询，另有锚点。训练和评估按页面隔离，两个状态都显示两个标题，仅交换哪项年份证据已知。所有页面来自原语料，基准支持页不纳入自动训练。准备汇总记录实际产量，上限是 720 个训练候选及 144 个留出配对。

准备摘要中 `ready_for_reader` 为 true 且四卡空闲后执行。

```bash
WIKI_RSI_SWITCH_GPUS=0,1,2,3 bash scripts/run_wiki_rsi_w2a.sh switch --run "$RUN"
```

GPU 阶段先在一张卡运行冻结答题器，对每个配对的四个实际证据条件分别作答。训练保留四格都符合格式且两条最佳查询都能答对、两条冗余查询都答不对的样本。所有可编码的留出题都评分，包括读者不能利用已取证据的题。至少 64 个训练配对、训练覆盖至少两种关系且至少 48 个留出配对才启动四卡。

四卡分别跑普通逐状态偏好与联合目标，各两个种子。每组 6 遍，样本、token、更新次数和 LoRA 配置一致。GPU 阶段共用 12 小时上限。冻结基线和四个训练策略各自依据可见状态选一条候选，留出评估记录两状态均选对、证据补齐、以及实际返回下的独立严格答案匹配。GPU 阶段读取 CPU 已真实执行的查询结果。本轮留出是自动构造的新页面，真实自然题最终效果在后续隔离测试中确认。

回传完整的 `wiki_rsi_switch_prepare_summary.json`、`wiki_rsi_switch_reader_gate_summary.json` 和 `wiki_rsi_switch_summary.json`。若 CPU 准备未达到读者验证的最低规模，只回传准备摘要。若程序报错，附控制台末尾及 `switch_workers_*` 中对应日志的末尾。模型、执行返回和权重留厂。
