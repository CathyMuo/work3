# W6b 四卡过夜训练

使用四张空闲 H800 做两组等预算对照，每组使用不同的预定种子。每个种子和训练方式独占一张卡。

| 卡位 | 种子 | 条件 |
| --- | --- | --- |
| 第 1 张 | 20261119 | 仅正例 LoRA，同时测共同冻结基线 |
| 第 2 张 | 20261119 | 同正例加真实负查询偏好损失 |
| 第 3 张 | 20261120 | 仅正例 LoRA |
| 第 4 张 | 20261120 | 同正例加真实负查询偏好损失 |

从 W6a 已验证的 155 对出发，扫描同一份 Wikipedia 语料，补入影片、书籍、歌曲和专辑的署名关系。新训练状态要求每个正查询从实际 top-3 核验出生年，相关干扰查询不能返回目标页或年份。排除旧支持页、既有源、训练三页互斥冲突和超出输入预算的候选。训练集目标上限 768 对，实际扩大规模按核验读数报告。原 384 道自然题及标题逐字可见分层直接复用。训练只读工具见证标签及训练页。

两训练臂沿用 W6b 的 LoRA、损失配比、学习率与梯度累积。每个种子内同初始化和样本顺序，各卡使用同一训练集。计划 8 遍，在第 2、8 遍分别保存适配器及独立自然题结果。四组达到相同遍数和更新预算后才发布相应配对读数。GPU 窗口最多 12 小时，触顶保存已配对完成的检查点和 worker 日志。第 8 遍为约定主结果，第 2 遍用于提前查验。

## 执行

在厂内 Work3 仓库，四卡已空闲后使用一条后台命令。CPU 扩充拉起在 GPU 训练前，使用原索引。GPU 的 12 小时从四个训练任务起跑计。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export WIKI_RSI_CORPUS="$DRZERO_DIR/corpus/wiki-18.jsonl"
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
export WIKI_RSI_OVERNIGHT_GPUS=0,1,2,3
RUN=wiki-w0c-20261118-110954

nohup bash scripts/run_wiki_rsi_overnight.sh --run "$RUN" \
  > "$WIKI_RSI_ROOT/$RUN/w6_overnight_console.log" 2>&1 </dev/null &
echo "W6b PID: $!"
```

同一 RUN 已有实验运行时先检查其状态。若 CPU 扩充已经成功且只需启动 GPU 阶段，可单独执行 `bash scripts/run_wiki_rsi_w2a.sh overnight --run "$RUN"`。可以从看板或者 `w6_overnight_console.log` 查看准备状态和已齐的检查点。

## 回传

优先回传以下 JSON。逐 worker 日志、原始查询和适配器位于该 RUN 的 `w6_overnight_workers_*`，在厂内留存。

- `wiki_rsi_w6_overnight_prepare_summary.json`
- `wiki_rsi_w6_overnight_summary.json`
- 如完整预算尚未完成，附 `wiki_rsi_w6_overnight_interim.json` 和失败 worker 日志的末尾

各数据集分别报告标题逐字可见和未逐字可见两层。只有两种子在第 8 遍均满足原先的配对门槛，才安排独立确认。届时仍需正式评估分布和最终作答读数。
