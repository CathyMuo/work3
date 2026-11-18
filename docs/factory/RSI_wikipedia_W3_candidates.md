# W3a 独立页面上的候选查询筛查

反馈 #29 的 W2b 显示，原索引标题诊断取得的真实段落能改善冻结 3B 作答。上一阶段的 115 道题、33 个标题臂正确样本和 24 个 W1 救回样本继续只用于诊断。W3a 用不同支持页测候选动作是否够用，并与一次原题查询对照。此轮没有训练。

从原第 3 轮答错且缺句、历史可重建、还剩一次查询及作答预算的状态抽题，每数据集最多 80。跳过与 W0c 全候选清单共享标注支持标题的题。入本轮的题之间也不复用支持标题。按冻结种子选择，条件不包括缓存可回溯性、标题可见性或检索结果。评估使用现有公开标注。

候选生成只看原题、可见检索段落和同一 Wikipedia 语料的页名。优先未返回的标题，其次按命中的段落数、标题长度、是否在问题中出现，最后按标题排序。最多取前三个标题查询，再加入原题查询；没有标题候选时退回原题。

## 执行

复用现有 RUN 和原索引。两个阶段均不加载 GPU 解题器。查询复用现存 E5 服务，先复核原 P0 八条锚点。

~~~bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD
export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export WIKI_RSI_CORPUS="$DRZERO_DIR/corpus/wiki-18.jsonl"
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
RUN=wiki-w0c-20261118-110954
bash scripts/run_wiki_rsi_w2a.sh candidates-prepare --run "$RUN"

export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
bash scripts/run_wiki_rsi_w2a.sh candidates --run "$RUN"
~~~

若资产在其他目录，设置实际的 WIKI_RSI_CORPUS、P0_EVAL_DUMP 和 P0_TEST_PARQUET。最多 160 道题，每题至多 4 个可见候选加一个标注标题探针，最多 800 次查询，另有 8 条锚点。按批去重计实际查询。标注标题只用于诊断。

## 回传

回传 wiki_rsi_w3_candidates_prepare_summary.json、wiki_rsi_w3_candidates_retriever_check.json、wiki_rsi_w3_candidates_summary.json 和提交号。逐题问题、候选、参考和检索结果留在厂内。若某个数据集样本不足，保留实际分母和流失汇总。

比较原题与候选排序第一的**一次查询**。另报候选中是否存在任意单条能取齐的上界，不能合并多条返回。直接标题探针用于判断参考可达性。目标标题在完整词表匹配集、截断列表的出现率分别报告。

重启看板后，「实验运行」页显示语料扫描、批次进度和心跳。可让训练继续运行，查询复用原有服务。

[冻结协议](../../research/protocols/RSI_wikipedia.md)
