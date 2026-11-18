# W3c 原语料覆盖与逐缺失页探针

W3b 在 160 道题中确认 95 道至少有一条缺句能在原语料按原规则找到，37 道能找到全部。缺句集中在一页时全匹配 31/63，跨多页时是 6/97。同一语料上检查有多少缺口能由标题和切段口径解释，并量化增加已知标题查询能否在严格口径下取齐更多证据。

## 执行

在原 Work3 仓库复用已完成 W3b 的运行目录。保留 W3b 准备、汇总、逐题结果。两阶段均为 CPU 客户端，复用原索引与 top-3，不加载解题器或重启检索服务。

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
RUN=wiki-w0c-20261118-110954

bash scripts/run_wiki_rsi_w2a.sh coverage-audit-prepare --run "$RUN"
bash scripts/run_wiki_rsi_w2a.sh coverage-audit --run "$RUN"
```

准备会逐行扫描 21,015,324 条语料并验证 W3b 散列和严格匹配。此阶段需规范化全文，比 W3b 的标题扫描耗时长。进度每 100,000 条更新。在原规则未找到的句子中，分别记录在同标题相邻两行拼接后精确出现、在不同标题单个片段内精确出现的情况。两类是匹配口径敏感性读数。相邻两行未核实原页面偏移，异标题相同句也可能是不同实体或重复文本。它们均保留在独立统计中，原规则分母和命中数照旧。

检索阶段先检查旧锚点，每数据集 4 条至少 3 条按序复现。每题对每个不同缺失页已知标题执行一次查询。比较原主标题、本题所有已知标题中最佳单条、所有标题返回的并集。三种条件都按原严格定义评分，逐题记录原主标题有序复现。并集的查询预算随缺失页数变化，代表已知目标下的诊断上界。实际请求与逻辑查询数分别报告。

## 回传

回传实际提交号、退出码和原运行目录的三个完整 JSON。两数据集分母应各为 80。请连同按缺失页数的分层一起回传，便于区分覆盖和多查询收益。逐题文本、标题、定位行号及检索返回保留在厂内。

- `wiki_rsi_w3_coverage_audit_prepare_summary.json`
- `wiki_rsi_w3_coverage_audit_retriever_check.json`
- `wiki_rsi_w3_coverage_audit_summary.json`

家服先确认严格覆盖是否复现为 17/80 和 20/80，原主标题有序结果是否复现。若替代匹配恢复大量参考句，先分析匹配口径敏感性。若并集在有覆盖的多页题上补齐了最佳单查遗漏的证据，再考虑等预算的多步取证验证。若两项解释空间都小，下一方向是用固定索引中的可验证动作构造训练状态，并在独立页面评测，同时保留这 160 道的完整诊断分母。

[冻结协议](../../research/protocols/RSI_wikipedia.md)
