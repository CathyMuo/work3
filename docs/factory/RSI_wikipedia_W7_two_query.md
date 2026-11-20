# W7a 两次查询同预算复核

比较已有训练查询作为第二查时是否比规则查询更能补回证据。使用 W6a、W6b 和 W6c 在原 RUN 保存的结果。CPU 读取缓存，本阶段没有模型推理和新增检索。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
bash scripts/run_wiki_rsi_w2a.sh two-query --run wiki-w0c-20261118-110954
```

请回传完整的 `wiki_rsi_w7_two_query_summary.json`。保留 W6b 的原 worker 目录和 W6c 汇总。

所有组合共享第 8 遍记录中的冻结 R3 查询与实际 top-3 返回。第二查固定使用原问句、规则链接标题、带上下文词的规则链接标题，或一个既有训练策略的查询。每个组合每题最多两查，重复查询按一次计。原支持句用于评分，任何策略分配都不读取参考标题、支持句或上一查是否取齐。评分以两次实际返回合起来是否取齐全部缺句为准。分开报告两个数据集和标题可见层，也记录去重后的调用数。保留冻结一查作为解释性基线。

两个仅正例种子分别在两个数据集上至少超过最佳固定规则组合 5 道，才在全新隔离页面上确认学习到的补查价值。配对臂按原配置报告。通过此工程门槛仍属于同一诊断集的离线发现，正式确认需关注新页面、实际成本和最终作答。
