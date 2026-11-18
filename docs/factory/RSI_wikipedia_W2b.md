# W2b 真实检索返回的冻结作答

反馈 #28 已完成生成预算与标题线索对照。192 token 消除了截断，取证没有改善。目标标题进入模型提示后，只在 HotpotQA 产生一个净新增取齐，在 2Wiki 损失一个。本轮直接回放 W2a 原索引实际返回的段落，验证检索动作取得的证据是否转化为答案收益。

| 作答条件 | 追加材料 |
| --- | --- |
| 空对照 | 复现 W1 的同一空提示 |
| 原题查询 | W2a 原题原样查询的返回 |
| 标题诊断查询 | W2a 将已知支持页标题直接交给索引的返回 |
| 长度匹配无关段落 | 来自同数据集其他 P0 搜索的真实段落，按标题诊断臂匹配数量和长度 |

检索臂按原排名在上下文内放入完整段落，最高 top-3。不会按支持标注挑段落、重排或用金句替代检索返回。无关条件排除支持页、当前历史、标题臂已经使用的页面和答案别名。匹配失败保留该题前三个条件，匹配差另报分母。标题诊断使用已知页名，只用于隔离已取回证据的作用。

## 执行

复用 W2a 已完成的 RUN 与原 P0 3B 模型。读取本地留下的真实返回，本轮无需运行或访问检索服务。

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
bash scripts/run_wiki_rsi_w2a.sh answer-prepare --run "$RUN"
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi_w2a.sh answer --run "$RUN"
~~~

CPU 准备核对 W1 历史、空提示、权重和 W2a 取证口径。正式作答最多 115 题和 460 次生成，保持 W1 的每次 192 新 token。默认单卡；有其他空闲 H800 时可给 answer 追加 --native-gpus 1,2,3，整批题会分到独立副本，同一题的各条件在同一张卡。

## 回传

回传 wiki_rsi_w2_answer_prepare_summary.json、wiki_rsi_w2_answer_summary.json 和提交号，逐题材料留在厂内。报告两数据集分别对等长无关、空对照和原题查询的配对 EM，以及全部缺句实际进入提示的题数。格式无效记错，另报两臂均有效的结果。保留全部 115 道作为诊断集，包括未成功取齐的题。

重启看板读取更新代码，「实验运行」页显示准备、生成和评分。前一轮历史始终保留。

[冻结协议](../../research/protocols/RSI_wikipedia.md)
