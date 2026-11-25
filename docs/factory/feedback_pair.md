# 厂内运行 可验证反馈组合首筛

先执行 4 秒的沙箱预检，再确认离线数据并执行 CPU 准备。CPU 准备完成且为 `READY` 后启动四卡冒烟，冒烟 `PASS` 后启动正式首筛。使用 [冻结协议](../../research/protocols/feedback_pair.md) 的门槛。模型权重全程冻结。

## 离线输入与执行环境

- CodeContests 原训练分片固定为修订 `802411c3010cb00d1b05bad57ca77365a3c699d6` 的 39 个文件、共 7,509,753,086 字节，通过[官方分片清单](../../research/protocols/feedback_pair_codecontests_shards.json)核对。厂内若已通过 `verify-source`（39/39，SHA-256 全对），直接复用该目录；不要从会产出尾部填零文件的镜像源拉取。 [DeepMind CodeContests](https://huggingface.co/datasets/deepmind/code_contests)
- **不再需要 `bwrap`、外部 shim、特权容器或修改 AppArmor。** 执行前请确认未 `source` 旧的外部 shim（如 `/root/develop/workspace/sbx/env.sh`），也不要保留 `/tmp/sbx2` 等临时原型在路径中。
- 内置执行器 `work3-userns-chroot-v1` 在 root 容器下**完全不调用 `mount(2)`**，仅使用 `/usr/bin/python3` 与容器允许的命名空间原语：
  1. 切换为非 root 外部 UID/GID（默认 `65534:65534`，可用 `WORK3_SANDBOX_UID` / `WORK3_SANDBOX_GID` 指定空闲非零 ID），使 `RLIMIT_NPROC=1` 真正生效且无法修改 root 所有的只读根目录。
  2. 显式创建独立 `user + pid + net + ipc + uts` 命名空间，由 supervisor 担任私有 PID 命名空间的 PID 1，任何子进程随命名空间原子回收，不泄漏到宿主机。
  3. 仅复制 `/usr/bin/python3`、标准库与 `ldd` 动态库到内容寻址只读根目录（约 50 MB，默认放 `/tmp/work3-pair-sandbox-<uid>`，可用 `WORK3_SANDBOX_ROOT` 指定其他允许执行的目录），提交代码通过只读写密封 `memfd` 传入。
  4. 安装内核 `seccomp-BPF` 过滤（阻止 `fork`/`vfork`/`clone`/`clone3`/`execveat`/`unshare`/`setns`/`mount`/`umount2`/`ptrace` 等），并叠加 `RLIMIT_CPU=1`、`RLIMIT_AS=256MB`、`RLIMIT_FSIZE=64KB`、`RLIMIT_NOFILE=32`、`RLIMIT_NPROC=1`。
  5. launcher 自身计时 2.5s 到点 `SIGKILL` 私有 PID 1 并回收，外层 `Popen(start_new_session=True)` 配 `killpg` 兜底。
- `probe-sandbox` 仅做安全有界探测（单次 `os.fork()` 验证 seccomp 返回 `EPERM` 且关闭 seccomp 时 `RLIMIT_NPROC=1` 独立返回 `EAGAIN`、单次线程拦截、私有网络与只读目录校验、sleep 超时与 CPU 死循环拦截、新增残留进程数必须为 0），**不执行任何无上限 fork 炸弹测试**。
- `prepare`、`smoke`、`pilot` 必须使用同一套通过预检的执行器和 `/usr/bin/python3` 指纹。实验 Python 需有本地 `pyarrow`、`numpy`、`torch`、`transformers`、`tokenizers`、`safetensors`，模型使用已离线校验的 `Qwen2.5-3B-Instruct`。

## 厂内执行

在 Work3 仓库根目录执行。按实际目录设置变量；每阶段使用全新的运行名。看板与脚本共用 `WIKI_RSI_ROOT`。

```bash
export PAIR_PYTHON=/mnt/wh/miniconda3/envs/searchr1/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
DATA=models/datasets/code_contests
MODEL=/path/to/Qwen2.5-3B-Instruct

# 1. 先跑内置无 mount 沙箱预检（约 4 秒，零风险）
bash scripts/run_feedback_pair.sh probe-sandbox

# 2. 复核数据分片（如刚验收过可快速确认 39/39）
bash scripts/run_feedback_pair.sh verify-source --source "$DATA"

# 3. CPU 参考执行准入（默认 --cpu-jobs 8，单任务内部串行跑 case，最多同时 8 个沙箱；
#    如宿主机仍在恢复可先传 --cpu-jobs 4）
bash scripts/run_feedback_pair.sh prepare \
  --id pair-repair-prep-v2 --source "$DATA" --cpu-jobs 8

# 4. prepare 为 READY 后，启动四卡冒烟（上限 30 分钟）
MANIFEST="$WIKI_RSI_ROOT/pair-repair-prep-v2/manifest.json"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_feedback_pair.sh smoke \
  --id pair-repair-smoke-v1 --manifest "$MANIFEST" --model "$MODEL"

# 5. smoke 为 PASS 后，启动四卡正式首筛（上限 12 小时）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_feedback_pair.sh pilot \
  --id pair-repair-pilot-v1 --manifest "$MANIFEST" --model "$MODEL" \
  --smoke-run pair-repair-smoke-v1
```

`prepare`、`smoke`、`pilot` 运行期间均在看板显示 CPU 参考检查、首答采集、修复采样、分卡进度及最终判定。若 `probe-sandbox` 报错，请直接回传完整报错 JSON/日志，不要绕过或自行替换沙箱。

## 回传

先回传 `probe-sandbox` 的完整 JSON 输出。之后每步回传 `wiki_rsi_pair_repair_summary.json`。遇到 `INCOMPLETE` 时回传 `incomplete.json` 及对应 rank 日志的错误段。正式结束后保留 `inputs.json`、`selectors.json`、`sealed_policy_scores.json` 和各 rank JSONL 供复核。题目和代码保存在机内实验目录，公开提交包含脚本、协议和指南。
