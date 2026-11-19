# W4c 索引内修复状态的冻结作答

复用 W4b 页面隔离后的 74 道题。问题来自电影署名关系，目标人物出生年有可核验的真实检索见证。本轮测解题器是否利用这个年份来作答。每题保持原问题、完整初始返回、同一作答提示与解码设置，固定五组追加材料。

| 条件 | 追加材料 |
| --- | --- |
| prefix_only | 空 |
| original_question | W4b 原题重查的实际 top-3 |
| related_foil | W4b 另一位相关署名人物查询的实际 top-3 |
| target_year_masked | 目标人物返回的同一组段落，仅遮掉目标年份的独立数字文本 |
| target | 目标人物查询的实际 top-3 |

检索返回保持原顺序，不截去任何段落。序列化前统一 HTML 与 Unicode 表示。主比较是完整返回对自身遮年版本，以减少人物、主题和段落差异。另报完整返回对初始历史、原题、相关干扰三组的效果。由影片和角色选出目标人物的规则正是构造时使用的动作，该规则在这批状态上是强查询基线。实验测作答机制。

## 执行

复用 W4b 的案例与汇总、本次 RUN 的 W1 权重记录以及原 R3 模型。CPU 阶段核验拆分、证据、权重及 370 条提示。GPU 阶段最多 370 次生成，同题五臂都在一张空闲卡，温度 0、每条最多 192 新 token。直接读取旧检索返回，无需请求检索器。无空闲卡时先完成 CPU 准备。

```bash
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

bash scripts/run_wiki_rsi_w2a.sh repair-answer-prepare --run "$RUN"
read -r -p '空闲 H800 的设备编号: ' IDLE_GPU
CUDA_VISIBLE_DEVICES="$IDLE_GPU" bash scripts/run_wiki_rsi_w2a.sh repair-answer --run "$RUN"
```

`--gpu-memory` 可为这张单卡设置 vLLM 的显存占用比例，默认 0.70。模型必须指向 W1 核验过的原 3B 权重。看板显示准备、模型加载和生成进度。

## 回传

回传运行提交、实际用卡、两个退出码与以下两个完整 JSON。逐题材料、提示 token 与生成只在厂内保留。

- `wiki_rsi_w4_repair_answer_prepare_summary.json`
- `wiki_rsi_w4_repair_answer_summary.json`

确认总数为 74 道及 370 次生成，保留原 44/15/15 拆分。填写整体和各拆分严格 EM、配对胜负、双方格式均有效的敏感性读数及弃答数。格式错误在主比较按答错计，`<answer>unknown</answer>` 计有效弃答。15 道测试题只提供先导读数。若完整返回稳定超过遮年和相关干扰，优先建立另一套独立题型与强规则基线，再决定短训练。
