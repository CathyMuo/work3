# Wikipedia W2a 单查询取证实验

反馈 #25 的 W1 已通过。复用 wiki-w0c-20261118-110954 下的冻结准备文件和逐题 W1 回放结果。此轮考察一次实际原索引查询能否取回仍缺的支持句。候选框为 W1 的 115 道缺证据配对题，另分层报告 W1 中的 24 道证据特异救回。查询与答案的训练均待后续决策。

每个可回放状态运行三个检索条件。原题原样查询为基线。第三轮冻结 3B 只从原题和当时可见的搜索历史生成一次查询。已知支持页标题只进入诊断条件，用于识别即使提供页名也难以取回的情况。三个条件都只查一次、取 top-3。生成查询不合规计作无命中且无实际查询，逐臂报告执行率。

## 厂内运行

CPU 准备可在训练运行时执行，复用原评测和模型目录。此阶段不联系检索服务。

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
bash scripts/run_wiki_rsi_w2a.sh prepare --run "$RUN"
~~~

如果 P0 原评测和测试文件用了其他目录，沿用原 P0_EVAL_DUMP 与 P0_TEST_PARQUET。GPU 阶段确认原 Dr. Zero 使用的 wiki-18.jsonl、e5_Flat.index、E5 checkpoint 与 P0 评测一致，复用该检索服务的实际端口。若实际端口是 8100 或其他值，改下方 URL。WIKI_RSI_RETRIEVER_ID 记录实际索引、语料和编码器版本。程序仅请求检索，不重启或修改服务。用一张空闲 H800，并按厂内流程释放这张卡所需显存。

~~~bash
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi_w2a.sh run --run "$RUN"
~~~

启动模型前，每个数据集复查 4 条原轨迹中有 3 段返回的查询，要求每边至少 3 条有序段落与日志匹配。校验失败时回传检查汇总并核查服务配置。实际检索最多 345 次，另有 8 次校验查询。生成查询使用 BF16 贪心解码，最多 64 个新 token。

## 回传

回传同一 RUN 下的 wiki_rsi_w2a_prepare_summary.json、wiki_rsi_w2a_retriever_check.json、wiki_rsi_w2a_summary.json，以及提交号。若失败，提供已有汇总和报错。w2a_cases.json 中的逐题查询和真实检索返回留在厂内，供后续冻结作答。

主结果用所有可回放状态的全部缺句命中率，按题比较历史查询与原题查询。同步报告任一缺句命中、查询执行率、标题诊断及 24 道证据特异救回的可执行子集。任一条件命中只说明取得所需段落，接下来仍要用冻结作答对照验证检索结果能否救回答案。

[实验协议](../../research/protocols/RSI_wikipedia.md)
