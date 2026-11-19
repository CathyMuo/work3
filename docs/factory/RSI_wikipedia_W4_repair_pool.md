# W4a 构造并验证索引内的修复状态

反馈 #32 确认 W3c 的严格语料覆盖为 37/160，合并两种替代匹配后为 39/160。逐缺失页标题查询的并集相对最佳单条从 25/160 提高到 29/160。开始从固定语料构造新候选，验证能否自动得到有真实动作和同预算对照的搜索状态。

## 本轮实验

扫描与 W3c 散列相同的 Wikipedia 语料。从电影介绍中的导演、编剧、制片人三类明确署名提议关系。用原检索器实际查询影片标题，只使用返回里确实读到的署名。随后让已见的两位不同人物各查一次。在对应人物标题的传记首段核验确定、不同的出生年。问题询问其中一个角色的出生年。接受条件包括原问题和整份首轮返回中都没有该年份，问题不直接给出目标人名，且干扰人物的真实返回不含目标人物或答案年份。

同一题保留三个各用一次追加查询的条件。正确署名人物，另一位可见署名人物，原题重查。两个人名查询的证据来自实际 top-3。原题条件按相同口径计分，并分别报告其中出现年份及取到目标人物出生证据的数量。题目采用规则提取和自动校验，可直接读署名的规则查询在这一题型天然很强。此轮只验证数据和动作，训练迁移还需要独立任务分布。

输入支持页排除采用原评估引用包中可识别的题目标题。训练、校准、测试候选的影片、目标人物和干扰人物按规范化标题互斥，最高各 60、20、20 道。运行完整保留筛选分母。若产量低或原题重查已有很高覆盖，回传实际数据即可，家服据读数裁决试点是否继续。当前 115 道旧诊断和 160 道 W3 题保留原诊断用途。

## 执行

复用已经完成 W3c 的运行目录。两阶段均为 CPU 客户端，沿用原语料及检索服务。

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

bash scripts/run_wiki_rsi_w2a.sh repair-pool-prepare --run "$RUN"
bash scripts/run_wiki_rsi_w2a.sh repair-pool --run "$RUN"
```

准备阶段逐行读取 21,015,324 条原语料，按固定散列顺序选最多 768 个影片源。检索阶段按原锚点检查每个数据集 4 条，至少 3 条有序复现才继续。一个源最多 1 次影片查询和 3 次人物查询。每个入池状态另做 1 次原题重查。总计最多 3,172 次请求内逻辑查询，另有 8 条锚点。对相同人名复用真实返回，报告实际各类查询数。

## 回传

请回传提交号、两个阶段退出码和以下三个完整 JSON。逐题问题、答案、检索段落和支持页拆分留在厂内。阶段开始后可在看板查看语料行数、已检查源数和入池状态数。

- `wiki_rsi_w4_repair_pool_prepare_summary.json`
- `wiki_rsi_w4_repair_pool_retriever_check.json`
- `wiki_rsi_w4_repair_pool_summary.json`

[Wikipedia 通用实验协议](../../research/protocols/RSI_wikipedia.md)
