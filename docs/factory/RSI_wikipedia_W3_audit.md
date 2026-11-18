# W3b 语料覆盖与单查询归因

反馈 #30 的 W3a 在两数据集各测 80 道答错且缺证据的题。候选第一与原题均未取齐缺句，候选池中最佳单条只取齐 1/160。已知参考标题的单查询命中任一缺句 55/160、取齐全部 25/160。W3b 复用这 160 道题，检查缺句是否存在、可见标题是否进入动作列表、同一标题加问题语境能否取得正确段落。

## 固定条件

- 复用 W3a 运行目录、原 P0 第 3 轮轨迹、原测试 parquet 和同一份 `wiki-18.jsonl`。确认原始输入和 W3a 准备文件的散列，逐题复现旧候选前三。无需指定模型目录或加载解题器。
- 查找每条缺失支持句是否在对应标题的任意单个原始语料片段内精确匹配。报告目标页在语料中的可见性，以及任一和全部缺句的覆盖。匹配给出本快照与当前规范化口径下的覆盖，句子跨分段、版本差异和释义都会影响这个读数。
- 从同一份可见标题列表分别使用旧排序和问题重排。新排序依次优先标题完整出现在问题中、标题与问题的非停用词交集更大、原排名靠前。只比较执行一次的首选，以及枚举前三和前八后可取得的最佳单查询。枚举从不合并返回段落。
- 对每个候选，分别查标题和 `标题. 原问题`。两种已知参考标题探针只用于诊断，单列是否返回目标页、是否取到该页所需句子、是否取齐整题缺句。原题查询作为同预算基线与各离线列表的共同后备动作。
- 两阶段均不占解题器 GPU，复用正在运行的原检索接口及 top-3。训练和检索服务可继续运行。旧 RUN 的数据保持在厂内。

## 执行

在已完成 W3a 的 Work3 仓库运行以下命令。两个阶段沿用同一终端中的 `DIAG_PYTHON`。

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

bash scripts/run_wiki_rsi_w2a.sh candidate-audit-prepare --run "$RUN"
bash scripts/run_wiki_rsi_w2a.sh candidate-audit --run "$RUN"
```

若原始资产另存，沿用 W3a 的 `P0_EVAL_DUMP`、`P0_TEST_PARQUET` 和语料实际路径。准备应得到每个数据集 80 道，并读满 21,015,324 份原语料片段。运行前原检索锚点每个数据集应有 4 条，至少 3 条有序匹配才继续。真实查询最多 5,600 次，另有 8 条锚点，实际以去重后的计数为准。同题每个查询条件仍只取一次 top-3。摘要同时报告重跑原题和参考标题与 W3a 的逐题有序复现数。家服以两类复现数和本轮配对结果一起判断。

## 回传

回传以下三个完整 JSON、实际提交号和退出码。逐题问题、参考句、候选和真实返回保留在原运行目录。

- `wiki_rsi_w3_candidate_audit_prepare_summary.json`
- `wiki_rsi_w3_candidate_audit_retriever_check.json`
- `wiki_rsi_w3_candidate_audit_summary.json`

每个数据集分别报告全样本、缺句集中在一页和分散在多页的题。家服根据以下读数决定后续路线。

| 观察 | 下一步 |
| --- | --- |
| 原语料找不到大量参考缺句 | 先核对快照、分段和匹配口径，估计固定环境内严格取证目标的上界 |
| 目标页内有句，返回的却是同页其他片段，或没返回目标页 | 看同标题加问题语境能否稳定改善，优先解决段落级取证 |
| 标题探针能取得证据，目标多在旧前三之后 | 看问题重排能覆盖多少，再决定是否值得研究可见候选排序 |
| 标题探针能取得证据，目标多不在可见列表 | 优先验证需要中间关系线索的动作构造 |

W3b 是同题诊断。后续正式方法对照另冻结隔离支持页的测试集。新状态只有产生实际可执行的取证收益后，才安排冻结作答与训练试点。

[冻结协议](../../research/protocols/RSI_wikipedia.md)
