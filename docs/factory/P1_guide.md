# P1 厂内指导：噪声诊断（P0 跑完后运行）

版本：tag `p1-r13`（2026-11-13：P0 已 PASS，现在启动；汇总新增按 hop 分组的探索性统计，不影响判定；运行脚本与 r12 相同）。协议（已冻结，不要改）：[research/protocols/P1_noise_diagnosis.md](../../research/protocols/P1_noise_diagnosis.md)，执行以末尾的 **Addendum 2026-11-06** 为准。

**前提**：P0 全部跑完（包括 `gate`），且 `p0_summary.json` 已经发回家服。只要 P0 不是 INVALID 就可以跑 P1；P0 是否 PASS 由家服判断，P1 的产物照样有用。

**规则同 P0**：厂内不推送、不写代码、不改协议与决策文件；出错时不要自行修改，按第 4 节发回。

---

## 0. 反馈 #10 之后：先做 P0 行为诊断（只用 CPU，几分钟），再启动 P1

P0 判定 **PASS**（第 3 轮 0.3071 ≥ 0.306，门槛已冻结，不再讨论）。三轮的走势和论文不同：单跳题持续进步，多跳题持续退化。家服的判断见 `research/DECISION_LOG.md` 2026-11-13 一节。

先在 P0 的评测结果上跑一个只读诊断。它看的是解题者自己的轨迹：每题搜了几次、不搜就答的比例、搜索次数和对错的关系，以及同一道题在第 1 轮和第 3 轮之间的对错翻转。

```bash
cd <工作目录>/work3
git fetch --tags && git checkout p1-r13
python -m work3.p0_diag --eval-root $DRZERO_DIR/work3_p0/eval --out $DRZERO_DIR/work3_p0/p0_diag.json
```

终端会打印一张表，请把整张表连同 `p0_diag.json` 发回。然后接着按第 2 节启动 P1（pool1 → pool3 → parity → summary）。P1 的运行脚本与 r12 完全相同，`p1_summary.json` 里多出一节 `exploratory_by_hop`，只作探索性参考，不参与 GO / NO-GO 判定。

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
