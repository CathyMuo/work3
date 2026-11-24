# 厂内运行指南 · 预算受限 RLVR 同前缀试点

本轮只跑冻结模型推理。流程为离线准备题目、预检、四卡冒烟、四卡正式试点。正式上限两小时。 [冻结协议](../../research/protocols/budget_rlvr_paired_pilot.md)

## 资产和打包

复用 C1 的原版 Qwen2.5-3B-Instruct、Python 环境和原实验 `bundle_r2.json`。准备时还需要 C1 使用过的 RLVR-MATH train parquet。本地环境使用厂内既有的 transformers 4.56.2、PEFT 0.17.1、PyTorch、配套 tokenizer 和 PyArrow。运行期间离线。

在发布提交的仓库目录里替换前三个本地文件路径和 Python 路径。可见 GPU 编号按实际分配填写。题目包可从外部直接带入，已带入时从预检开始。

```bash
export BUDGET_PYTHON=/path/to/continual-env/bin/python
export BUDGET_MODEL=/path/to/Qwen2.5-3B-Instruct
export BUDGET_BUNDLE=/path/to/bundle_r2.json
export BUDGET_MATH=/path/to/original-math.parquet
export BUDGET_PANEL="$PWD/data/budget_rlvr/panel.json"
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/run_budget_rlvr.sh prepare \
  --model "$BUDGET_MODEL" --bundle "$BUDGET_BUNDLE" \
  --math "$BUDGET_MATH" --panel "$BUDGET_PANEL"
bash scripts/run_budget_rlvr.sh preflight \
  --model "$BUDGET_MODEL" --bundle "$BUDGET_BUNDLE" --panel "$BUDGET_PANEL"
```

准备应生成 256 道独立数学题，预检打印模型和面板哈希。

## 四卡冒烟与正式试点

```bash
bash scripts/run_budget_rlvr.sh smoke \
  --model "$BUDGET_MODEL" --bundle "$BUDGET_BUNDLE" --panel "$BUDGET_PANEL" \
  --id budget-rlvr-smoke-20261124

bash scripts/run_budget_rlvr.sh pilot \
  --model "$BUDGET_MODEL" --bundle "$BUDGET_BUNDLE" --panel "$BUDGET_PANEL" \
  --smoke-run budget-rlvr-smoke-20261124 --id budget-rlvr-pilot-20261124
```

冒烟应完成 16 条响应，且至少一条实际从同一前缀分出短答和长续写。正式入口校验冒烟 PASS 和输入一致性。正式预期 2,048 条，每卡 512 条。重复执行须使用新的运行 ID。若无空闲四卡，按厂内排程等待资源。

看板使用与实验相同的 `WIKI_RSI_ROOT`，显示已完成轨迹数、长短对照数、采样 token 和终态。对照条件及 GO、NO-GO 判法见冻结协议。

## 反馈

回传 `wiki_rsi_budget_rlvr_summary.json` 及 `inputs.json`，并通过既有私有通道回传四份 `rank-*.jsonl`。如遇 INCOMPLETE，同时回传 `incomplete.json`、已保存分卡数据和对应日志。

使用离线增量包时，在厂内已有的 `main` 上验证后快进合并。

```bash
git bundle verify /path/to/work3-budget-rlvr.bundle
git fetch /path/to/work3-budget-rlvr.bundle refs/heads/main
git merge --ff-only FETCH_HEAD
```
