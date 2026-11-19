# W5b 独立自然题的冻结自由查询

复用 W5a 原 384 道题、原 P0 R3 权重、原 Wikipedia 检索器和每查 top-3。模型看到原题及此前真实检索段落，按照同一提示生成一条自然语言查询。已知支持标题、缺句、W5a 基线结果都只用于输入之外的对照与评分。此轮为固定追加一查的检查，检验查询生成能力。

CPU 准备逐题复算 W5a 缓存基线，对照回传摘要，冻结模型输入。GPU 阶段贪心、每题最多 96 新 token，只执行一条非空单行。与正常结束的单行等价的 query 标签、前缀或引号允许去壳。多行、截断、格式错误及上下文超限留在总分母，按取证失败计。固定规则、可见候选诊断和已知标题使用上轮同题的原返回，新增检索前再验原服务锚点。

预设通过要求为每个数据集至少 100 道，且模型相对 `any_fixed` 在每个数据集分别净胜至少 10 道。净胜为模型取齐、三个固定规则都未取齐的数量减去反向数量。`any_fixed` 逐题事后合并三个规则的成功标记，是比单条固定规则更保守的对照。通过后可考虑在新页面上做独立的动作训练试点。否则当前固定索引、单查、3B 的训练路线结束准入，保留本轮及 W4 的诊断结果。

## 执行

复用 W5a 已完成的 RUN。准备是 CPU，执行占一张空闲卡，最多 384 次生成及 384 条新增查询，另有八个锚点。无空闲卡时可先完成 CPU 准备。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_EXPECT_VLLM=0.8.5.post1
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
RUN=wiki-w0c-20261118-110954

bash scripts/run_wiki_rsi_w2a.sh natural-query-prepare --run "$RUN"
read -r -p '空闲 H800 的设备编号: ' IDLE_GPU
CUDA_VISIBLE_DEVICES="$IDLE_GPU" bash scripts/run_wiki_rsi_w2a.sh natural-query --run "$RUN"
```

单卡 `--gpu-memory` 默认 0.70。每数据集应保留 192 道，准备摘要会报告超长数与预计生成量。锚点每数据集四条，至少三条有序复现才执行。看板显示模型载入、生成及累计检索。

## 回传

请回传提交号、实际用卡、两阶段退出码和三个完整 JSON。逐题输入、查询及返回留在厂内。

- `wiki_rsi_w5_natural_query_prepare_summary.json`
- `wiki_rsi_w5_natural_query_retriever_check.json`
- `wiki_rsi_w5_natural_query_summary.json`

报告两套数据上全部缺句取齐、相对各对照的配对胜负与区间、执行比例，以及参考页标题是否在可见历史中逐字出现的分层。对可见候选之外的成功单列。该轮只验证取证，整个诊断批次继续隔离在正式训练和最终测试之外。
