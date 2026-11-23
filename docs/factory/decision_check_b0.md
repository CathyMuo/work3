# 厂内运行指南 · 方案 B0 环境预检

**C1 已按冻结门槛收束。目前只执行 B0 的 CPU 预检。** 工厂全程离线，使用独立 Python 环境与官方文本游戏。 [冻结协议](../../research/protocols/decision_check_b0.md)

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

将上面的数据路径指到实际包含三份切分的目录，每次使用新的运行 ID。离线镜像可直接更新至发布提交。若用本次增量 Git bundle，在厂内 `main` 上先 `git bundle verify /path/to/work3-b0.bundle`，再 `git fetch /path/to/work3-b0.bundle refs/heads/main` 和 `git merge --ff-only FETCH_HEAD`。

看板的「实验运行」页会显示已检查游戏数、合格数和 GO、NO-GO 或执行异常。用新版代码运行看板，并让看板与脚本读取同一 `WIKI_RSI_ROOT`。

## 回传

在约定的私有通道回传该运行目录的 `wiki_rsi_decision_check_summary.json`、`decision_checks.jsonl`、`heldout_manifest.json` 和 `progress.json`。若中途停止，回传 `incomplete.json`、已有输出和终端日志。C1 的完整 `necessity_rollouts.json` 仍可一并归档，用于逐题复核。
