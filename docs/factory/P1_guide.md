# P1 厂内指导：噪声诊断（P0 跑完后运行）

版本：tag `p1-r16`（2026-11-16：P1 已完成，判定 NO-GO；微批次消融已完成）。协议（已冻结，不要改）：[research/protocols/P1_noise_diagnosis.md](../../research/protocols/P1_noise_diagnosis.md)，执行以末尾的 **Addendum 2026-11-06** 为准。

**前提**：P0 全部跑完（包括 `gate`），且 `p0_summary.json` 已经发回家服。只要 P0 不是 INVALID 就可以跑 P1；P0 是否 PASS 由家服判断，P1 的产物照样有用。

**规则同 P0**：厂内不推送、不写代码、不改协议与决策文件；出错时不要自行修改，按第 4 节发回。

---

## 0. P1 和微批次消融均已完成

消融结果见 [反馈 #15 回复](REPLY_20261116_fb15.md)。以下消融命令留作运行记录。

P1 判定 **NO-GO**，结果存在 `research/results/P1_summary_20261116.json`。按冻结规则，NCP 方向停止，**P2 取消**。家服回复见 `docs/factory/REPLY_20261116_fb14.md`。

GPU 空出来后立即运行（约 9–10 小时，可以无人值守）：

```bash
cd <工作目录>/work3
git fetch --tags && git checkout p0-r16
tmux new -s ablate
bash scripts/p0/run_p0.sh ablate 2>&1 | tee -a $DRZERO_DIR/work3_p0/ablate.log
```

- 消融把第 3 轮的解题者训练重跑两遍：从 `solver_iter2_hf` 出发、用第 3 轮题池，**微批次 2 一次、微批次 1 一次**（后者用来衡量重复训练本身的波动）。每遍之后都评测，最后自动生成 `work3_p0/p0_diag_ablate.json` / `.txt`。
- 输出都放在单独的目录，P0 的结果不动；中断后原样重跑即可续上。
- 微批次 2 如果 OOM，在命令前加 `EXPANDABLE_SEGMENTS=1` 重跑；还 OOM 就把日志最后 80 行发回。
- 跑完把 `p0_diag_ablate.json` 和 `.txt` 原样发回（不要手抄）。在新计划下达之前，不跑其他实验。

下面第 1–5 节是 P1 当时的运行说明，留作记录。

## 1. 准备

```bash
cd <工作目录>/work3 && git fetch --tags && git checkout p1-r15
conda activate drzero
# 与 P0 相同的变量：DRZERO_DIR、MODEL_3B、E5_MODEL、NGPU；端口默认 8100 / 8101
P1=<工作目录>/work3/scripts/p1/run_p1.sh
```

P1 用到的 P0 产物（都在 `$DRZERO_DIR` 下，不要删）：
- `data/zero_challenger_iter1_*.parquet`、`data/zero_challenger_iter3_*.parquet`
- `checkpoints/dr-zero/solver_iter2_*/solver_iter2_hf`
- `work3_p0/pool_check_all.json`

## 2. 运行

开跑前同样先停掉显卡保活程序（见 P0 指导第 0 节）。

```bash
bash $P1 check     # 产物、端口、GPU、sglang 和 verl 的导入
bash $P1 pool1     # 题池 1，解题模型 = 基座模型
bash $P1 pool3     # 题池 3，解题模型 = solver_iter2_hf
bash $P1 parity    # 只用 CPU：P1 拼出的提示与上游 verl 的逐项比对
bash $P1 summary   # 写 work3_p1/p1_summary.json
```

- `pool1` / `pool3` 会先起检索服务和 SGLang（4 卡数据并行），等两者都就绪后开始采样。每题 5 次原始采样，每次再做 2 次重放；如果 k=1 或 k∈{3,4} 的题少于 60 道，会自动再抽 500 题，最多 1500 题。最后对 k=1 的题各补 16 次采样。
- **中断可续跑**：已完成的题目记在 `work3_p1/pool*_questions.jsonl`，重跑同一步会跳过它们。手动 Ctrl-C 之后先运行 `bash $P1 cleanup`。
- **显存不够**：`export P1_MEM=0.5`，或者 `export FAISS_GPU=0`。
- **预计耗时**：每个题池 1–3 小时（含索引和模型加载），`parity` 不到 1 小时，合计半天左右。

## 3. 产物

`$DRZERO_DIR/work3_p1/` 下：
- `pool{1,3}_questions.jsonl`：完整轨迹，每池约 100–300 MB。保留在厂内，不发回。
- `pool{1,3}_lucky.jsonl`、`pool{1,3}_state.json`
- `p1_parity.json`、`p1_summary.json`、`timings.txt`、`logs/`

## 4. 发回家服的内容

```bash
cd $DRZERO_DIR/work3_p1
cat p1_summary.json
python -c "import json;d=json.load(open('p1_parity.json'));print({k:d[k] for k in ('checked','text_equal','ids_equal','errors','tool_schema_source')});[print(f['examples'][:2]) for f in d['files']]"
cat pool1_state.json pool3_state.json
cat timings.txt
cd <工作目录>/work3 && git describe --tags --always
```

失败时额外发失败阶段日志的最后 80 行：`tail -n 80 logs/<阶段>.log`。

## 5. 看板

`bash scripts/run_board.sh`（同 P0 指导第 6 节）会自动显示 `work3_p1/` 的进度：两个题池各自的阶段状态、检索延迟和最新日志。
