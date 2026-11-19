# W6a 独立动作训练小试的数据准备

冻结模型已通过自然题单查询准入。本轮固定训练数据的可验证构造和独立评估，再按产量决定是否排期等预算短训。

训练候选来自同一 Wikipedia 索引中新影片页的真实返回。复用 W4b 的规范化和单证据规则，在相同问题和可见前缀下获取一条能查到目标人物出生年的查询，并验证查同一影片中另一位相关人物的返回不带目标页或答案。每条查询的预算相同。训练候选与既有标注支持页、W4b 扫过的人物及源页、独立评估首轮可见页隔离。影片源、目标和干扰三类页在接受的训练对之间不复用。

评估选取原 HotpotQA 和 2WikiMultiHopQA 的新支持页失败状态，保持还可加一查、缺句在一页，排除旧诊断支持页及电影和出生关键词。各数据集先选至多 64 道参考标题不逐字可见的题，再选至多 128 道逐字可见的题。按已有轨迹冻结分层后再查索引，报告原题、桥接标题、标题加词、可见候选上界和已知标题上界。支持标注只用于分层和评估。分层改变了抽样比例，须分别报告两层结果。

## 执行

复用已经完成 W5b 的 RUN、同一 P0 语料、原模型 tokenizer 和检索服务。两个阶段都是 CPU 客户端。准备分别复用自然题与影片源的扫描，核验两次语料散列。查验阶段先复查八条锚点，再执行训练正负动作和自然题基线。

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

bash scripts/run_wiki_rsi_w2a.sh action-pilot-prepare --run "$RUN"
bash scripts/run_wiki_rsi_w2a.sh action-pilot --run "$RUN"
```

P0 文件如走自定义路径，沿用 W5a 的 `P0_EVAL_DUMP`、`P0_TEST_PARQUET` 和 `OUT_ROOT`。计划至多检查 2,048 个影片源，收集至多 160 个独立训练对。自然题两边各至多 192 道。查验阶段新增物理查询上限 10,880 条，另有 8 条锚点，重复查询批内去重。

## 回传与准入

回传提交号、退出码和三个完整 JSON。逐题训练对和检索原文保存于厂内。

- `wiki_rsi_w6_action_pilot_prepare_summary.json`
- `wiki_rsi_w6_action_pilot_retriever_check.json`
- `wiki_rsi_w6_action_pilot_summary.json`

至少 80 个训练对，且两个评估数据集各至少 80 道标题逐字可见、32 道不逐字可见，逐字可见层各至少 10 个已知标题单查见证，才安排 GPU 小试。小试需使用同一出发权重，比较冻结生成、正例训练、真实动作配对训练，并对齐实际更新预算。作用是否体现在自然题上按分层检验。
