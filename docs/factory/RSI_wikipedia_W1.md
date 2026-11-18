# Wikipedia W1 单卡冻结回放

反馈 #25 已完成 W1。当前厂内入口为 [W2a 单查询取证](RSI_wikipedia_W2a.md)。

反馈 #24 的 W0c 与 W1 CPU 历史重建均已通过。复用厂内 wiki-w0c-20261118-110954 的 prepared.json 和准备汇总。HotpotQA 有 56 道缺证据配对题和 30 道证据齐全对照题，2Wiki 有 59 道配对题和 29 道对照题。共 174 道题，完整回放 578 次生成。

使用原 P0 第 3 轮冻结的 Qwen2.5-3B HF 解题器。脚本校验模型、tokenizer 与准备时的全部文件哈希。无需下载其他模型、迁移轨迹、访问索引或启动检索服务。

## CPU 启动检查

在当前厂内 Work3 checkout 运行。REPLAY_PYTHON 指向已有 vLLM 0.8.5.post1 环境的 Python。若目录不同，换成实际绝对路径。启动检查可在训练仍运行时做。

~~~bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD
export DRZERO_DIR=/root/develop/workspace/drzero
export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_EXPECT_VLLM=0.8.5.post1
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
RUN=wiki-w0c-20261118-110954
bash scripts/run_wiki_rsi.sh preflight --run "$RUN"
~~~

预期 prepared_cases=174、planned_completions=578。P0_ITER3_MODEL 若存于其他位置，指向 CPU 准备时的同一份完整目录。预检查失败时回传报错，勿为凑数重做准备。

## 单卡冒烟与正式回放

任选一张空闲 H800，保持两阶段使用同一张卡、同一 Python 环境、TP=1 和默认显存配额 70%。如果所有卡都有任务，先按厂内原流程释放一张卡。下例使用第 0 张，按实际情况修改。

~~~bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_wiki_rsi.sh smoke --run "$RUN" &&
bash scripts/run_wiki_rsi.sh replay --run "$RUN"
~~~

冒烟为 4 道题和 16 次生成，核验模型版本、token IDs、完成数与终止状态。冒烟准确率和证据收益不决定能否启动完整回放。脚本会校验完整回放使用同一批已准备样本和相同运行配置。两阶段均用 BF16、温度 0、最多 192 个新 token、3,072 上下文。若只做完冒烟，可在同一终端环境单独执行 replay 续跑。如果 GPU 任务失败，请保留终端日志和现有产物，按原配置修复运行问题后重试未完成阶段。

## 回传

同一个 RUN 下回传 wiki_rsi_preflight_summary.json、wiki_rsi_smoke_summary.json、wiki_rsi_replay_summary.json 和提交号。若失败，仅回传已完成的汇总和失败阶段错误日志。逐题模型输出、准备文件和检查点留在厂内。

家服将先看两个数据集各自的「真证据对等长无关段落」配对 EM，再核对空对照、三臂均有效的配对结果和证据来源分层。看板读取相同 WIKI_RSI_ROOT，显示启动检查、生成请求数和完成量。

[冻结协议](../../research/protocols/RSI_wikipedia.md) · [vLLM 0.8.5.post1 离线接口](https://docs.vllm.ai/en/v0.8.5.post1/api/offline_inference/llm.html)
