# 同预算跨页证据选择对照

反馈 #46 在 2Wiki 的 40 道同证据题上发现，参考支持摘录让冻结读者从答对 6 道增至 21 道。已有无标注压缩只保留了 14 道的全部参考事实。当前比较同一片段池下的独立相关性与跨页衔接打分。使用一张空闲 H800 做推理，无训练及新增检索。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export P0_ITER3_MODEL=/root/develop/workspace/drzero/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf
RUN=wiki-w0c-20261118-110954

CUDA_VISIBLE_DEVICES=0 bash scripts/run_wiki_rsi_w2a.sh reader-bridge --run "$RUN"
```

卡号使用实际空闲卡。需保留原 RUN 的 W7b、W7-gap、W7-reader 结果和 `w7_reader_cases.json` 完整逐题文件。若原运行用了自定义参考包，沿用 `--references`。参考句只用于评分，两种新策略在选证据时只能看到问题、实际补查词、实际文本和来源阶段。

两组使用同样的 64 词窗口、相同的早期与补查候选对、包含问题和格式的 640 输入 token 上限、冻结 R3 和答案格式。两组都优先保留来自不同页的早期与补查片段。独立组只按各片段与问题及补查词的相关性打分，衔接组额外考虑跨页标题提及与稀有词重合。选定一对后，两组用相同的片段顺序和补充规则。原完整、无标注压缩及参考支持读数从已核验缓存复用。

两个数据集分开报告，另列完整材料含全部参考事实的小组。若 2Wiki 的衔接组相对完整和同预算独立组分别净增至少 5 道严格答对，且在该小组保留了更多完整参考证据，才启动未见 2Wiki 题的独立确认。HotpotQA 用于核对适用范围。开发结果不用于独立确认。

请回传完整的 `wiki_rsi_w7_bridge_summary.json`。逐题输出 `w7_bridge_cases.json` 留厂。
