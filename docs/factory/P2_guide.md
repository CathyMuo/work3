# P2 厂内指导：出题奖励对比（Beta 平滑臂、NCP 臂）

版本：tag `p2-r6`（2026-11-10，含反馈 #2–#4 的修复、提速与崩溃续跑；环境同 P0 指导第 3 节）。协议（已冻结，不要改）：[research/protocols/P2_ncp_pilot.md](../../research/protocols/P2_ncp_pilot.md)。

**前提**：家服确认 **P0 = PASS 且 P1 = GO** 并通知开跑后才运行。没有通知不要跑。

**规则同 P0**：厂内不推送、不写代码、不改协议与决策文件；出错时不要自行修改，按第 4 节发回。

Dr. Zero 臂直接复用 P0 的结果，不重跑。P2 只新跑两个臂，每个臂流程和 P0 完全一样（3 轮训练 + 3 次评测），只把出题奖励换掉。

---

## 1. 准备

```bash
cd <工作目录>/work3 && git fetch --tags && git checkout p2-r6
conda activate drzero
# 与 P0 相同的变量：MODEL_3B、E5_MODEL、NGPU、端口……（不需要 TRAIN_PARQUET / TEST_PARQUET）
export P0_DIR=<工作目录>/drzero               # P0 的 DRZERO_DIR
export ARM_ROOT=<工作目录>/drzero_p2          # 两个臂的工作树放这里（默认就是这个位置）
P2=<工作目录>/work3/scripts/p2/run_p2.sh
```

`setup` 会用 `git worktree` 从 P0 的 Dr. Zero 目录建出 `$ARM_ROOT/beta`、`$ARM_ROOT/ncp`，固定在同一个 commit。训练种子文件、测试集、验证集和检索文件都用软链接指向 P0 的那一份。**不要在臂目录里跑 `prepare`**（脚本也会拒绝）：上游生成种子文件时 hop 抽取没有固定 seed，重新生成会让三个臂的题源不一致。

## 2. 运行（每个臂都按这个顺序；先 beta 后 ncp，或反过来都可以）

开跑前同样先停掉显卡保活程序（见 P0 指导第 0 节）。

```bash
bash $P2 setup beta
bash $P2 beta check
bash $P2 beta patch     # 会打印 "proposer reward arm: beta"
bash $P2 beta iter1
bash $P2 beta eval1
bash $P2 beta iter2
bash $P2 beta eval2
bash $P2 beta iter3
bash $P2 beta eval3
bash $P2 beta gate      # 写 $ARM_ROOT/beta/work3_p0/p0_summary.json（这里的 PASS/FAIL 不用管）
bash $P2 p1 beta        # 次要指标：在该臂第 3 轮题池上跑 P1 诊断（只跑 pool3）

# 把上面所有的 beta 换成 ncp，再跑一遍

bash $P2 gate           # 三个臂对比，写 $ARM_ROOT/p2_summary.json
```

- 每一步失败都会停下，日志在 `$ARM_ROOT/<arm>/work3_p0/logs/`。修好后重跑这一步即可。
- 出题者训练日志里会多出 `[work3 reward <arm>] call N: ...` 行，每步一行；`$ARM_ROOT/<arm>/work3_p0/reward_log.jsonl` 记录每步的 k 分布、平均难度分和生成用量。
- **耗时**：出题者阶段比 P0 慢。Beta 每题要采 7 次而不是 5 次；NCP 每题 5 次采样之后，还要再生成 10 次最终回答。预计每个臂 4–5 天，两个臂合计 8–10 天。第 1 轮出题者跑完后，请把 `timings.txt` 发回，以便重估。
- **显存不够**：处理方法和 P0 相同（`GEN_MEM`、`FAISS_GPU=0`），改完重跑 `patch`。
- **磁盘**：每个臂和 P0 一样约 450 GB。

## 3. 产物

- `$ARM_ROOT/<arm>/work3_p0/`：`p0_summary.json`、`reward_log.jsonl`、`timings.txt`、日志、评测明细
- `$ARM_ROOT/<arm>/work3_p1/`：`p1_summary.json`（只有 pool3）
- `$ARM_ROOT/p2_summary.json`

检查点、题池都留在厂内，不发回。

## 4. 发回家服的内容

```bash
cat $ARM_ROOT/p2_summary.json
for a in beta ncp; do
  echo "== $a"; cat $ARM_ROOT/$a/work3_p0/p0_summary.json; cat $ARM_ROOT/$a/work3_p0/timings.txt
  tail -n 3 $ARM_ROOT/$a/work3_p0/reward_log.jsonl
  python -c "import json;d=json.load(open('$ARM_ROOT/$a/work3_p1/p1_summary.json'));print(d['pools'].get('3'))"
done
cd <工作目录>/work3 && git describe --tags --always
```

失败时额外发失败阶段日志的最后 80 行。

## 5. 看板

启动看板时加上 `ARM_ROOT=$ARM_ROOT`，会同时显示两个臂（各自的训练进度、评测 EM、奖励曲线）、两个臂的题池 3 诊断，以及 `p2_summary.json` 的判定结果。
