# Wikipedia W0b 厂内 CPU 复审

反馈 #23 已完成 W0b。当前执行入口为 [W0c 跨轮缓存审计](RSI_wikipedia_W0c.md)。

反馈 #22 中的 W0 判定为未过门槛。家服已用 HotpotQA 官方 dev 的两个版本按题目 ID 核对问题、答案和支持句坐标。旧引用包从 fullwiki 的段落候选读取支持句，官方 distractor dev 可补齐 5,315 道题的支持句。W0b 只更换引用包。候选的实际段落仍须来自原第 3 轮评测轨迹，两个数据集每个 60 道候选的门槛保持不变。引用包已随代码提供，无需下载数据或新增人工标注。

## 运行

在已完成 P0 的厂内 Work3 checkout 执行。沿用同次的原始第 3 轮评测 JSONL 与测试 Parquet，下面使用默认 P0 目录。

~~~bash
git pull --ff-only origin main
git rev-parse HEAD
export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
RUN="wiki-w0b-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_wiki_rsi.sh audit --run "$RUN"
~~~

如果使用自定义 P0 目录，请显式设置 P0_EVAL_DUMP 和 P0_TEST_PARQUET。看板使用相同 WIKI_RSI_ROOT 时会显示本轮；在默认位置运行双方无需另配根目录。

请回传 outputs/wiki_rsi/$RUN/w0/wiki_rsi_input_summary.json 和提交号。即使退出码为 0，仍以 ready_for_replay_preparation 判定。读取两个数据集的 missing_with_source 总数与 selected_missing_with_source，分清池规模和抽样上限。2Wiki 的完整统计与 #22 应相同，它是这次输入变更的对照。逐题坐标留在运行机。本轮只执行 CPU 审计，后续历史重建与 GPU 回放由家服安排。

家服运行需要的最小集合是原第 3 轮合并 JSONL、同轮测试 Parquet 与该轮完整 HF 解题器目录。资产继续保存在厂内；迁移时通过约定的私有路径，并校验 SHA-256。复审本身不要求上传模型或轨迹。

[冻结协议](../../research/protocols/RSI_wikipedia.md) · [家服后续入口](../RSI_wikipedia_home.md)
