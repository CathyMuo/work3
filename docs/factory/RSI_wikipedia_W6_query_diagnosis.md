# W6c 离线查询诊断

从 W6b 第 2、8 遍保存的逐题记录重算 384 道自然题的取证情况。读取冻结 R3 和四个训练结果的已生成查询、实际返回，以及 W6a 已核验的已知标题探针。整个阶段在 CPU 上执行，读取原有文件，不发新检索请求。四卡任务可以继续保活训练。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
bash scripts/run_wiki_rsi_w2a.sh action-diagnosis --run wiki-w0c-20261118-110954
```

需要原 RUN 的 W6a 与 W6b 产物和 W6b 原 worker 目录。脚本复核准备文件指纹和逐题命中计数。请回传完整的 `wiki_rsi_w6_query_diagnosis_summary.json`。本机留存 `w6_query_diagnosis_cases.json`，用来复核逐题丢失与新增，包含实际生成的查询。如果文件缺失，请提供报错末尾和原目录是否仍在。

每个数据集和标题可见层分别对照冻结基线，并统计几类情况。任何策略曾取齐证据表示记录中有有效候选。所有策略漏检而已知标题探针成功表示当前候选覆盖仍有缺口。实际返回了目标页却缺严格证据句的情况单独计数。查询是否逐字出现在输入中只作为行为描述。五个策略的成功并集和已知标题探针都用事后信息，既有训练或单查成绩仍按原预算判定。
