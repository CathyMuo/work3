# W7b 新支持页补查确认

W7a 通过工程门槛。本轮确认相同两查预算下的证据覆盖和最终作答，使用新支持页，不重复 W7a。只复用已训练的两个仅正例适配器，GPU 阶段为推理。CPU 准备可以在已有训练继续运行时执行。

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

bash scripts/run_wiki_rsi_w2a.sh two-query-live-prepare --run "$RUN"
```

沿用原 P0 的 `P0_EVAL_DUMP`、`P0_TEST_PARQUET` 或 `OUT_ROOT` 自定义路径（如有）。同一 RUN 需保留 W7a 原汇总、W6b `w6_overnight_workers_*` 目录、第八遍两个 `positive_only` 权重和训练对文件。每个数据集最多锁定 192 道与既有诊断和训练支持页隔离的自然失败题，按实际准备规模评估。

准备成功且有空闲 H800 后，使用实际空闲卡号替换示例中的 0。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi_w2a.sh two-query-live --run "$RUN"
```

所有配置共享实际执行的冻结首查。第二查分别使用 W7a 的强规则、看到首查返回后重新生成的冻结 R3，以及看到同一返回后的两个仅正例策略。冻结首查单独作为参照。固定排名前 3、每题最多两条新查询，重复查询只执行一次。所有配置用相同的冻结 R3 作答，分开记录原始覆盖、实际送入模型的证据覆盖、严格 EM、查询数和完整对照的耗时。仅使用首查之前的历史判定参考标题可见层。参考答案和支持句仅供评分。

回传完整的 `wiki_rsi_w7_live_prepare_summary.json` 和 `wiki_rsi_w7_live_summary.json`。原始逐题查询、实际返回及答案由脚本写入厂内 `w7_live_cases.json`。
