# 厂内运行 可验证反馈组合首筛

运行分三步。CPU 准备完成且为 `READY` 后启动四卡冒烟，冒烟 `PASS` 后启动正式首筛。使用 [冻结协议](../../research/protocols/feedback_pair.md) 的门槛。模型权重全程冻结。

## 离线输入

- 在可联网机器上准备固定数据快照和本地模型目录，按现有流程把文件转入厂内。CodeContests 原训练分片约 8 GB。保留 `data/train-*-of-00039-*.parquet` 的文件名及 39 个实体文件。数据由 [DeepMind CodeContests](https://huggingface.co/datasets/deepmind/code_contests) 发放。
- Linux 需要 `/usr/bin/python3`、系统 `bwrap` 和允许隔离命名空间的执行环境。`prepare` 实际探测隔离，确认通过才运行候选代码。实验 Python 需有本地 `pyarrow`、`numpy`、`torch`、`transformers`、`tokenizers`、`safetensors`。使用上一轮已可离线加载的 Qwen2.5-3B-Instruct 本地目录，冒烟冻结其实际文件指纹。

联网机器上下载数据的命令如下。

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="deepmind/code_contests", repo_type="dataset", revision="802411c3010cb00d1b05bad57ca77365a3c699d6", allow_patterns=["data/train-*.parquet"], local_dir="code_contests")'
```

## 厂内执行

在 Work3 仓库根目录执行。按实际目录设置下列变量；每阶段用全新的运行名。板和脚本共用 `WIKI_RSI_ROOT`。

```bash
export PAIR_PYTHON=/mnt/wh/miniconda3/envs/searchr1/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
DATA=/path/to/code_contests
MODEL=/path/to/Qwen2.5-3B-Instruct

bash scripts/run_feedback_pair.sh prepare \
  --id pair-repair-prep-v1 --source "$DATA"

MANIFEST="$WIKI_RSI_ROOT/pair-repair-prep-v1/manifest.json"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_feedback_pair.sh smoke \
  --id pair-repair-smoke-v1 --manifest "$MANIFEST" --model "$MODEL"

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_feedback_pair.sh pilot \
  --id pair-repair-pilot-v1 --manifest "$MANIFEST" --model "$MODEL" \
  --smoke-run pair-repair-smoke-v1
```

`prepare` 可通过 `--cpu-jobs` 配置并行度，默认 12。正式实验上限 12 小时，实际耗时由冒烟和合格任务数决定。四卡分别运行同一模型副本。看板显示 CPU 参考检查、首答采集、修复采样、分卡进度及最终判定。分阶段百分比以当前工作量为分母。

## 回传

每步回传 `wiki_rsi_pair_repair_summary.json`。遇到 `INCOMPLETE` 时回传 `incomplete.json` 及对应 rank 日志的错误段。正式结束后保留 `inputs.json`、`selectors.json`、`sealed_policy_scores.json` 和各 rank JSONL 供复核。题目和代码保存在机内实验目录，公开提交包含脚本、协议和指南。
