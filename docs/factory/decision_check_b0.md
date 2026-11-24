# 厂内运行指南 · 方案 B0 环境预检

**2026 年 11 月 24 日决定收束 B0。** 旧进程停在 221/240，合格 9 局。按执行状态归档 INCOMPLETE。即使剩余 19 局全部合格，总数也只到 28，低于 100 局门槛。当前厂内任务仅为停止旧 B0 进程并归档。 [冻结协议](../../research/protocols/decision_check_b0.md)

## 关闭旧运行

先在原厂内机器确认旧进程的完整命令、运行 ID 和 PID。目标必须是 `python -m work3.decision_check_preflight` 且运行 ID 为 `decision-check-b0-20261123`。使用单一 PID，勿使用按名称批量停止的命令。若它已退出，可直接归档。

```bash
ps -eo pid,ppid,stat,etimes,args | grep '[w]ork3\.decision_check_preflight'
```

从上面的结果录入该 PID，再核对一次命令。若旧进程已退出，跳到归档。

```bash
read -r -p '只输入旧 B0 运行的 PID: ' B0_PID
ps -p "$B0_PID" -o pid,ppid,stat,etimes,args
```

确认这条命令属于旧 B0 后再执行停止命令。如果同一 PID 仍存在，重新核对后才考虑 KILL。保活训练属于其他进程。

```bash
if [[ "$B0_PID" =` ^[1-9][0-9]*$ ]] && (( B0_PID > 1 )); then
  kill -TERM -- "$B0_PID"
fi
```

切到发布提交并使用**这次旧运行的原输出根目录**。进程退出后运行以下命令。归档入口只用 Python 标准库，会核对时间、进程和已保存记录。它写入 `incomplete.json` 并把看板改为 `INCOMPLETE 221/240`，原 JSONL、留出清单与运行提交保留。

```bash
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
python3 -m work3.decision_check_archive --id decision-check-b0-20261123
```

成功时预期 `tested_games=221`、`eligible_games=9`、`eligible_upper_bound=28`、`eligible_type_upper_bound=1`、`research_gate=UNREACHABLE`。回传归档 JSON 和已有 `decision_checks.jsonl`。看板用新版代码读取同一目录。

以下为历史复现参考。今后的同类 CPU 回放采用父进程监督，每局最多 90 秒，并受两小时总限时。发生超时会停止整个运行并保存进度。

## 联网处准备

用与厂内相同的 Linux x86-64 和 Python 3.10 准备 wheelhouse。构建机应已有 `make`、C/C++ 编译器及 Python 开发头文件，文本依赖 Jericho 需要原生构建。执行 `python -m pip wheel -w wheelhouse 'alfworld==0.4.2' 'textworld[pddl]==1.6.2'`。核对 wheelhouse 含 Jericho 和 Fast Downward 的兼容轮子。用 `sha256sum wheelhouse/* > wheelhouse.sha256` 留存依赖清单。 [TextWorld 原生依赖](https://pypi.org/project/textworld/1.6.2/)

下载官方文本数据包。原 zip SHA-256 为 `5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf`，解压根目录名为 `json_2.1.1`，任务在各切分的 `game.tw-pddl`。将数据、依赖、代码及其散列清单通过既有交接方式带入厂内。 [官方文本数据发布](https://github.com/alfworld/alfworld/releases/tag/0.4.2) · [数据包](https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip)

## 厂内断网执行

在发布提交的工作目录，用独立环境本地安装 wheelhouse。

```bash
export DECISION_PYTHON=/path/to/alfworld-env/bin/python
"$DECISION_PYTHON" -m pip install --no-index --find-links /path/to/wheelhouse \
  'alfworld==0.4.2' 'textworld[pddl]==1.6.2'
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export ALFWORLD_GAMES=/path/to/json_2.1.1
CUDA_VISIBLE_DEVICES='' timeout --signal=INT --kill-after=30s 2h \
  "$DECISION_PYTHON" -m work3.decision_check_preflight \
  --train "$ALFWORLD_GAMES/train" \
  --seen "$ALFWORLD_GAMES/valid_seen" \
  --unseen "$ALFWORLD_GAMES/valid_unseen" \
  --id decision-check-b0-20261123
```

将上面的数据路径指到实际包含三份切分的目录，每次使用新的运行 ID。离线镜像可直接更新至发布提交。若用本次增量 Git bundle，在厂内 `main` 上先 `git bundle verify /path/to/work3-b0-close.bundle`，再 `git fetch /path/to/work3-b0-close.bundle refs/heads/main` 和 `git merge --ff-only FETCH_HEAD`。

看板的「实验运行」页会显示已检查游戏数、合格数和 GO、NO-GO 或执行异常。用新版代码运行看板，并让看板与脚本读取同一 `WIKI_RSI_ROOT`。

## 回传

在约定的私有通道回传该运行目录的 `wiki_rsi_decision_check_summary.json`、`decision_checks.jsonl`、`heldout_manifest.json` 和 `progress.json`。若中途停止，回传 `incomplete.json`、已有输出和终端日志。C1 的完整 `necessity_rollouts.json` 仍可一并归档，用于逐题复核。
