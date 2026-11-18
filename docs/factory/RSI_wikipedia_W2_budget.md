# W2 查询生成预算与目标线索对照

反馈 #28 已完成。当前运行入口为 [W2b 真实检索返回的冻结作答](RSI_wikipedia_W2b.md)。

反馈 #27 已完成查询输出诊断。同输出放宽解析没有增加执行或命中。原生提示中 52/115 条生成在 64 token 达到长度上限。下一轮将生成长度和目标线索分开操纵，继续保持每个条件只查询一次、固定索引和 top-3。

| 条件 | 生成输入 | 最多新 token |
| --- | --- | ---: |
| 64 | 原生提示，原状态 | 64 |
| 192 | 相同原生提示，原状态 | 192 |
| 目标线索 | 原生提示加目标支持页标题 | 192 |
| 对照线索 | 同格式加入长度匹配的其他标题 | 192 |

首要比较为 192 相对 64 的全部缺句命中。64 条件重新生成，并报告与 #27 原始输出的一致性。线索条件同时比较目标标题、其他标题和同题无提示 192，区分增加有用线索与干扰标题。标注标题只进入诊断提示，作答和训练均不在本轮。CPU 检查不截断历史，分别报告无法容纳 192 token 或缺少可匹配的其他标题。历史是否已字面出现目标分层汇报。

## 执行

复用已完成查询诊断的同一个 RUN。环境和服务沿用上轮，默认单 H800。

~~~bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD
export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_EXPECT_VLLM=0.8.5.post1
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
RUN=wiki-w0c-20261118-110954
bash scripts/run_wiki_rsi_w2a.sh budget-prepare --run "$RUN"
~~~

~~~bash
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi_w2a.sh budget --run "$RUN"
~~~

如果其他三张 H800 已空闲，可追加 --native-gpus 1,2,3，只并行线索条件。64 和 192 的预算对照都留在原卡。最大 460 次生成。参考 #27 的单卡用时，本轮默认避免为并行专门停用全卡训练。

## 回传与看板

回传 wiki_rsi_w2_budget_prepare_summary.json、wiki_rsi_w2_budget_retriever_check.json、wiki_rsi_w2_budget_summary.json 和提交号。原始生成、查询与检索返回由厂内保留。

拉取后重启看板以加载进度读取器，使用原 DRZERO_DIR 和与实验一致的 WIKI_RSI_ROOT。实验运行页在本机 /#wiki。各页签都继续轮询，运行中每 2 秒，空闲每 8 秒。进程每 3 秒写心跳，实际批次写入已完成题数和当前数据集，模型加载显示已持续时间。W0 老审计从 DRZERO_DIR/work3_p0 下的原目录只读显示，若走自定义目录可设置 BOARD_WIKI_LEGACY_ROOT。

[冻结协议](../../research/protocols/RSI_wikipedia.md)
