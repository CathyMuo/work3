# W2 查询输出与证据取向诊断

反馈 #26 的历史查询只执行 25/115 次。此轮固定 W2a 原题、历史、原 P0 第 3 轮 3B、索引、top-3 和每臂一条查询。用同一批题区分生成格式和查询内容造成的损失。

| 条件 | 生成与解析 |
| --- | --- |
| 原严格重放 | 重放 W2a 原提示，沿用其严格的单行 query 标签解析 |
| 同输出兼容解析 | 复用上一条件的生成，允许一个完整的 think 段以及一条合法原生 search 调用或独立 query 标签 |
| 原生工具提示 | 从同一历史生成一条原生 search 调用，用同一个兼容解析器 |

三个条件均不把答案或标注支持页标题提供给查询模型。提前终止、截断、多条动作或歧义仍计未执行。标题字面是否出现在原题或已见段落中，只用于自动分层。输出同时保留历史严格查询能否复现、各种拒绝原因、执行率、任一/全部缺句命中和配对差。

## 运行

复用已完成 W2a 的现有 RUN。CPU 启动阶段校验原 W2a 文件、模型、评测轨迹与输入 token，可在 GPU 继续训练时进行。

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
bash scripts/run_wiki_rsi_w2a.sh contract-prepare --run "$RUN"
~~~

旧提示对照继续在 W2a 的原卡、同一环境和原批次上运行。索引端口保持一致，启动前复查原先的八条检索锚点。单卡模式适合本轮这样的小批量，CPU 准备结束后只需释放一张 H800。

~~~bash
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi_w2a.sh contract --run "$RUN"
~~~

若四张同规格 H800 已空闲，可以并行跑原生提示。原卡用环境变量指定，另外三卡用 --native-gpus 指定。示例沿用反馈中的原卡 0，也可只填一到两张其他空闲卡。每张卡各载入一份同模型，TP 均为 1，保持原显存配额和输入/解码预算。

~~~bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi_w2a.sh contract --run "$RUN" --native-gpus 1,2,3
~~~

任选一种方式执行。最多 230 次生成，每次最多 64 个新 token。同一批次相同查询共用检索返回，汇总分别记录逻辑执行和实际请求中的查询数，同时分开记录模型加载、查询生成、实际检索与总耗时。四卡模式的收益取决于生成在总耗时中的比重，以及多副本加载开销。若有副本失败，日志在运行目录的 w2_contract_native_gpuN.log，修复后可重试未完成阶段，或改用单卡。

## 回传

回传 wiki_rsi_w2_contract_prepare_summary.json、wiki_rsi_w2_contract_retriever_check.json、wiki_rsi_w2_contract_summary.json 和提交号。若失败，提供已完成汇总和报错。原始生成、实际查询和检索返回保存于厂内 w2_contract_cases.json，用于回看未执行的原因和后续作答验证。

[冻结协议](../../research/protocols/RSI_wikipedia.md)
