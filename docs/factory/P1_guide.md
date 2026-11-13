# P1 厂内指导：噪声诊断（P0 跑完后运行）

版本：tag `p1-r14`（2026-11-13：诊断补上检索到的查询条数和“标准答案是否被检索到”；新增微批次消融；P1 运行脚本与 r12 相同）。协议（已冻结，不要改）：[research/protocols/P1_noise_diagnosis.md](../../research/protocols/P1_noise_diagnosis.md)，执行以末尾的 **Addendum 2026-11-06** 为准。

**前提**：P0 全部跑完（包括 `gate`），且 `p0_summary.json` 已经发回家服。只要 P0 不是 INVALID 就可以跑 P1；P0 是否 PASS 由家服判断，P1 的产物照样有用。

**规则同 P0**：厂内不推送、不写代码、不改协议与决策文件；出错时不要自行修改，按第 4 节发回。

---

## 0. 反馈 #11 之后：先补诊断，再跑 P1，最后跑微批次消融（按顺序）

`p0_diag` 的第一版证实了一件事：解题者每题的**搜索调用次数**在减少，单跳题和多跳题都在减少。但它还回答不了两个问题：

- **一次调用可以带多条查询**（`query_list`）。调用次数变少，不等于检索变少。
- **多跳题答错，是因为没搜到答案就停了，还是搜到了却没答对？**

还有一个和我们自己有关的疑点（详见 `research/DECISION_LOG.md` 2026-11-13 (b)）。p0-r8 为了避免 OOM，把微批次从 2 改成了 1。verl 的 loss 是在每个微批次里按 token 求平均的，微批次为 1 时就变成每条回答各自做长度归一化。这种加权会偏爱短的正确回答（Dr. GRPO 论文指出过这一点）。它可能就是“少搜索”的部分原因，必须用实验排除。

**第 1 步：诊断第二版**（只用 CPU，几分钟）。完整的输出文件 **`p0_diag.json` 请直接作为附件发回，不要手抄表格**。上次表格里出现了 `0.000`，可脚本只会输出两位小数，说明是转录时出的错。

```bash
cd <工作目录>/work3
git fetch --tags && git checkout p1-r14
python -m work3.p0_diag --eval-root $DRZERO_DIR/work3_p0/eval --test-parquet $DRZERO_DIR/data/test.parquet --out $DRZERO_DIR/work3_p0/p0_diag.json
```

**第 2 步：P1**，按第 2 节运行（pool1 → pool3 → parity → summary），约半天。P1 的运行脚本没有改动，判定标准不变。

**第 3 步：微批次消融**，P1 结束后运行，约 9–10 小时，可以无人值守过夜：

```bash
cd <工作目录>/work3   # 同一个版本，p0-r14 与 p1-r14 指向同一个提交
tmux new -s ablate
bash scripts/p0/run_p0.sh ablate 2>&1 | tee -a $DRZERO_DIR/work3_p0/ablate.log
```

- 它把第 3 轮的解题者训练重跑两遍：从 `solver_iter2_hf` 出发，用第 3 轮题池，**微批次 2 一次、微批次 1 一次**（后者用来衡量重复训练本身的波动）。每遍之后转 HF 并完整评测，最后自动生成 `work3_p0/p0_diag_ablate.json` 和 `.txt`。
- 所有输出都放在单独的目录（`checkpoints/dr-zero-ablate/`、`work3_p0/eval_ablate/`），P0 的检查点和评测结果完全不动。
- 中断后重跑同一条命令即可，已完成的部分会自动跳过。
- **微批次 2 有可能 OOM**（反馈 #6 的 OOM 发生在出题者训练，解题者在微批次 2 下还没跑过）。如果 OOM，就在命令前加 `EXPANDABLE_SEGMENTS=1` 重跑。还 OOM 的话，把日志最后 80 行发回，不要改其他参数。

## 1. 准备

```bash
cd <工作目录>/work3 && git fetch --tags && git checkout p1-r12
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
