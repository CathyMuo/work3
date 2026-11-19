# W4b 审计修复状态的流失原因

复用 W4a 的冻结源列表和同一份原语料。每个源重做一次原查询，再查可见的同批人物名。用完整原语料中同规范化标题的片段对照原检索器 top-3，统计人物标题是否能精确对齐、是否能按规则核验出生年，以及有证据却未返回目标页或出生段落的情况。

同一份返回比较四种条件。所有条件都要求目标题的年份来自实际 top-3 的同标题片段，题面和首轮返回中没有答案年份。干扰查询均来自同一影片的另一位可见署名人物，其返回不能含目标页或目标年份。

| 条件 | 正动作解析 | 干扰动作要求 |
| --- | --- | --- |
| strict_paired | W4a 原规则 | 同标题内验证到不同的出生年 |
| strict_one_witness | W4a 原规则 | 返回确切的干扰人物页 |
| normalized_paired | 规范化 HTML、空白与标点，接受同标题页导语里包含查询姓名的全名 | 同标题内验证到不同的出生年 |
| normalized_one_witness | 同上 | 返回确切的干扰人物页 |

规范化继续要求明确的出生格式或生卒括号，多年冲突时排除。完整语料中没有被这两种规则核验到的句子仍可能包含其他表述。表中第二因素用于判断干扰查询是否有必要满足与正动作相同的属性要求。

## 执行

复用已完成 W4a 的 RUN 和产物，不重跑 W4a。审计先核验旧检索锚点，再复查同批源页，逐行扫描原语料，最后对通过规范化单证据条件的页面隔离候选重查原题。全程是 CPU 客户端，复用原检索服务。

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

bash scripts/run_wiki_rsi_w2a.sh repair-audit --run "$RUN"
```

应读取 768 个原源和 21,015,324 条语料。原锚点各数据集 4 条，至少 3 条有序复现才继续。原始源及人物查询量与新读数同报。新条件按原拆分轮转冻结影片、目标人物、干扰人物，训练、校准、测试各最多 60、20、20 道。只对这些页面隔离样本执行一次原题查询，入池条件不看原题查询结果。

## 回传

回传提交号、退出码与以下两个完整 JSON。逐源检索返回、逐人物诊断和候选存于 `w4_repair_audit_cases.json`，留在厂内。

- `wiki_rsi_w4_repair_audit_retriever_check.json`
- `wiki_rsi_w4_repair_audit_summary.json`

若可验证出生年主要留在原语料、没有进入原 top-3，优先研究证据段落可达性。若规范化或单证据干扰恢复了足够候选，再考虑冻结作答回放。若全语料可核验的对应属性仍少，下一轮转向可实际检索验证的其他属性。
