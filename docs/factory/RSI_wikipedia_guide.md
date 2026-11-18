# Wikipedia RSI 首轮输入审计厂内指南

当前厂内执行入口为 [W3a 候选查询筛查](RSI_wikipedia_W3_candidates.md)。此页保留首次 W0 的复现方式。家服执行后续阶段时使用[家服启动指南](../RSI_wikipedia_home.md)。

当前只运行 W0。沿用原 P0 的 Dr. Zero 工作目录、评测轨迹、数据文件和原 Python 环境。脚本用 CPU 读取原文件，不启动 vLLM 或检索服务，也无需变动现有训练。

在已有工作三仓库根目录执行。DRZERO_DIR 设成上轮 P0 运行时的 Dr. Zero checkout，包含 data/test.parquet。若 P0 使用自定义输出根目录，请沿用原 OUT_ROOT，默认读取 checkout 下的 work3_p0/eval/iter3。

~~~bash
git switch main
git pull --ff-only origin main
git rev-parse HEAD
export DRZERO_DIR=/absolute/path/to/Dr.Zero
bash scripts/run_wiki_rsi_audit.sh
~~~

若不在原 Python 环境，可将 DIAG_PYTHON 设为原环境 Python 的绝对路径。默认结果在原 OUT_ROOT 下的新建 wiki_rsi_audit 时间戳目录。若迭代目录存在多个不同 worker 文件，沿用原 P0 门禁认定的合并文件命名，脚本通过同一个解析入口读取。

请回传完整的 wiki_rsi_input_summary.json。wiki_rsi_manifest.jsonl 只保存在运行机，包含候选和真实检索结果的坐标。完整回传提交号和错误日志即可诊断失败。两个数据集候选数量不足时也回传汇总。

[研究与实验协议](../../research/protocols/RSI_wikipedia.md)
