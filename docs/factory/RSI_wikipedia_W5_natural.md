# W5a 独立自然题的单查询检验

本轮需要了解真实自然题上还有多少值得学习的取证机会。沿用 P0 的固定语料、E5 检索服务、每查 top-3 与失败历史。从 HotpotQA 和 2WikiMultiHopQA 的原始自然问题各取至多 192 道。每道是 R3 原模型答错、仍可追加一条查询、参考缺句集中在同一支持页的多跳问题。排除 W0c/W3 支持页重叠和含电影或出生词的题面及支持标题。本轮样本之间也按所有参考支持页隔离。

先按确定规则抽样，再扫描同一语料并执行检索。抽样不使用缓存出处、原始语料是否覆盖缺句、金标题查询或任何候选结果。公开支持句用于确定缺句分层和计算实际检索覆盖。

| 条件 | 查询规则 |
| --- | --- |
| original | 原题 |
| linked | 从原题和已见返回中找能在原语料确认的字面标题。优先未返回标题和词汇最相关的桥接段落。 |
| linked_context | linked 查询加最多五个原题内容词 |
| 可见候选诊断 | 枚举其余两个最高排序标题，加上三个固定规则的结果。逐题看是否存在一条就能取齐，不合并返回。 |
| 已知标题诊断 | 直接查询缺句所在的已知参考标题，用于测一次查询的参考可达性 |

每条固定条件各用一条查询。任何固定规则能取齐的并集单独报告为保守基线，代表事后逐题任选规则。主终点为同一条查询的 top-3 覆盖所有缺句。原语料严格覆盖与语义可达性含义不同。旧 74 道 W4 题及 W5a 自然题均用于诊断，正式训练与最终测试仍需新拆分。

## 执行

复用已完成 W4c 的 RUN，保留原 W0c/W3 准备产物与 P0 第三轮评测。准备阶段加载原模型 tokenizer 核对原日志可重建，两阶段都由 CPU 客户端执行。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export WIKI_RSI_CORPUS="$DRZERO_DIR/corpus/wiki-18.jsonl"
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
RUN=wiki-w0c-20261118-110954

bash scripts/run_wiki_rsi_w2a.sh natural-prepare --run "$RUN"
bash scripts/run_wiki_rsi_w2a.sh natural-check --run "$RUN"
```

如果原始 P0 文件走自定义路径，继续用 W3a 相同的 `P0_EVAL_DUMP`、`P0_TEST_PARQUET` 和 `OUT_ROOT`。读取 21,015,324 条原语料，比较 SHA256。检索前按 W3a 锚点校验，每数据集四条、至少三条有序复现。查询去重后上限为 384 道乘 7，另有 8 条锚点。

## 回传

回传版本、退出码和以下三个完整 JSON。题目、标注支持和逐题返回继续仅保留厂内。

- `wiki_rsi_w5_natural_prepare_summary.json`
- `wiki_rsi_w5_natural_retriever_check.json`
- `wiki_rsi_w5_natural_summary.json`

## 决策条件

两个数据集各至少 100 道。如果各有至少 10 道由可见候选取齐、所有三个固定规则都失败，就做一次冻结动作选择对照。如果这项未通过，但两边各有至少 20 道已知标题取齐，并且各至少 10 道超过固定规则，就做一次冻结模型自由查询生成检查。其余结果记录为该设定下训练准入不足。门槛用于决定试点投入，正式方法结果需要等预算对照和独立评估。
