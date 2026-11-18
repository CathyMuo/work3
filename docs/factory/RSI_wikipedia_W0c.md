# Wikipedia W0c 跨轮缓存审计

反馈 #24 已完成 W0c 和 CPU 历史重建。当前执行入口为 [W1 单卡冻结回放](RSI_wikipedia_W1.md)。

反馈 #23 确认 HotpotQA 的同轮合格池为 43 道，2Wiki 为 393 道。W0b 仍未通过两个数据集各 60 道的门槛。W0c 固定第 3 轮的判错题、历史和支持句，只把第 0–2 轮保存的真实 Wikipedia 返回纳入来源查找。原索引和第 3 轮待回放状态保持一致。

从最近的已有缓存开始查，每条缺句必须在同数据集的有效轨迹中找到同标题且包含原句的段落。原第 3 轮的候选数和所有固定分母会与 W0b 原汇总核对。未匹配全部缺句的题，另外统计只找到部分来源与完全找不到来源。候选清单逐段记录源轮次、行号、查询序号和段落序号。

## 命令

在当前厂内 Work3 checkout 执行。若 W0b 产物不在默认位置，把 BASELINE 调整为本次反馈对应的原始汇总。W0c 只读已有输入，不会启动 GPU 或检索服务。

~~~bash
git pull --ff-only origin main
git rev-parse HEAD
export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export P0_SOURCE_CACHE_ROOT="$DRZERO_DIR/work3_p0/eval"
BASELINE="$WIKI_RSI_ROOT/wiki-w0b-20261118-104916/w0/wiki_rsi_input_summary.json"
RUN="wiki-w0c-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_wiki_rsi.sh audit --run "$RUN" --baseline-summary "$BASELINE"
~~~

P0_SOURCE_CACHE_ROOT 下的 iter0、iter1、iter2 各应有一份原始合并 JSONL。目标仍读取 iter3，可用 P0_EVAL_DUMP 指向原文件。P0_TEST_PARQUET 指向同次测试 Parquet，默认使用原 checkout 下的 data/test.parquet。看板读取相同 WIKI_RSI_ROOT，审计列显示已检查的轮次数和扩展得到的新候选数。

## 回传与准入

回传提交号与 WIKI_RSI_ROOT/$RUN/w0/wiki_rsi_input_summary.json。字典中的 missing_with_source_floor_iter3 应为 HotpotQA 43、2Wiki 393。较早轮次的分桶与该值相加得到扩展后的总候选数。逐题坐标留在厂内。若脚本拒绝输入，请带原报错一起反馈。

只有汇总的 ready_for_replay_preparation 为 true，才在同一个 RUN 下接着执行 CPU 历史重建，复用上述环境变量。默认 P0 第 3 轮 HF 目录按原运行配置取值，若存放位置不同就改为实际绝对路径。

~~~bash
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
bash scripts/run_wiki_rsi.sh prepare --run "$RUN"
~~~

同时回传 wiki_rsi_prepare_summary.json。两边各至少 40 道有效配对题才允许完整 W1。此轮到 CPU 准备结束，家服依据输出安排 GPU 冒烟和完整回放。

[冻结协议](../../research/protocols/RSI_wikipedia.md) · [家服后续入口](../RSI_wikipedia_home.md)
