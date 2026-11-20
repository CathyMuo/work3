# 固定检索后的证据呈现对照

反馈 #45 的缺失事实补查有 340/384 条来自规则回退，暂停该两行提示路线。本轮比较对同一实际检索结果的三种证据呈现。只用一张空闲 H800 进行作答推理，新增检索为零。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export P0_ITER3_MODEL=/root/develop/workspace/drzero/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf
RUN=wiki-w0c-20261118-110954

CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi_w2a.sh reader-evidence --run "$RUN"
```

用实际空闲卡替换 0。原 RUN 需保留 W7b 和 W7-gap 的汇总及 W7b 逐题结果。若先前用过自定义参考包，则沿用原 `--references`。固定 W7b 种子 20261119 的仅正例臂的首查和补查返回，所有策略使用相同冻结 R3 和答案格式。

- 完整证据复现 W7b 送入读者的上下文。
- 无标注压缩按原问题和补查词从已呈现原文中选短片段，无需知道参考支持句或答案。
- 参考支持条件仅从上述实际已呈现材料截取能验证的参考支持句，附邻近原文。它不加入漏检的任何证据，仅作事后诊断。

完整与参考支持条件都保留全部参考事实的题另报。报告严格 EM、配对胜负、证据覆盖、格式有效性和提示 token。参考支持条件明显改善而无标注压缩未改善，支持优先改证据选择。无标注压缩收益在两数据集上都可见，再到未见题确认。若参考支持在同证据子集也无收益，当前结果不足以支撑扩大补查或压缩训练。开发集结果留作决策。

请回传 `wiki_rsi_w7_reader_summary.json`，逐题数据保留在厂内 `w7_reader_cases.json`。
