# 缺失事实补查同预算对照

反馈 #44 已完成日期配对四卡训练。自然题结果未通过扩大条件。此轮复用 W7b 已观察的 384 道自然题，直接测试明确提出尚缺事实能否生成更有用的第二条查询。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
export P0_ITER3_MODEL=/root/develop/workspace/drzero/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf
RUN=wiki-w0c-20261118-110954

CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi_w2a.sh gap-follow-up --run "$RUN"
```

卡号使用实际空闲的一张 H800，无训练。RUN 需保留 W7b 的完整逐题结果和已完成的迁移汇总。两个新策略共享相同真实首查历史、冻结 R3、生成长度、查询格式和最终答题器。一个围绕问题主题直接写查询，另一个在写查询前明确当前证据尚不支持的事实。格式失败或查询重复时使用相同的既有规则回退。每种策略每题至多新增一次查询。新检索总数最多 768，另有锚点。

比较完整证据、实际送入读者的证据、最终严格 EM 和真实执行预算。保留首查即可答对的题，统计加入新候选后事后上界能扩大多少。只作开发比较。若最终答案收益在相同实际查询预算下可见，再在未见题上冻结比较；开发集汇总不用于投稿的独立确认。

请回传完整的 `wiki_rsi_w7_gap_summary.json`，逐题结果留厂内。
