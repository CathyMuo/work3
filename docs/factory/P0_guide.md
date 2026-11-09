# P0 厂内指导：复现 Dr. Zero Qwen2.5-3B（4×H800）

版本：tag `p0-r4`（2026-11-09：处理厂内反馈 #2；看板重做：研究路线、分轮进度、研究说明图文页。运行脚本与 p0-r2 完全相同）。
协议（已冻结，不要改）：[research/protocols/P0_drzero_reproduction.md](../../research/protocols/P0_drzero_reproduction.md)，执行以末尾两节 **Addendum** 为准。
判定：第 3 轮 7 个数据集平均 EM ≥ **0.306** 为 PASS（论文 0.326）。

**规则**：厂内不推送、不写代码、不改 `research/DECISION_LOG.md`、`research/STATE.json` 和协议文件。出现 INVALID / FAIL / 报错时，不要自己改门槛或超参数，按第 8 节发回。

---

## 0. 正在用 p0-r1 跑的话：怎样切到 p0-r2

**不要在 `run_p0.sh` 运行时切换代码**：bash 是边执行边读脚本的，中途改文件会让正在跑的进程出错。等当前这一步（比如 `bash $P0 iter1`）整步结束后再切：

```bash
cd <工作目录>/work3
git checkout -- scripts/p0/wait_retriever.sh scripts/p0/run_p0.sh   # 丢掉两处临时修改（p0-r2 已合入）
git fetch --tags && git checkout p0-r4
bash $P0 patch      # 重新生成补丁脚本；不要重跑 prepare
```

已完成的阶段不用重跑。第 1 轮出题者用的是旧的 0.25 显存比例，从下一个出题者阶段起改为 0.45。两者只影响显存和速度，协议补记里已写明。

## 1. 开跑前：停掉显卡保活程序

卡上跑的是保活程序，不是研究任务，**不用等卡空闲**：确认是保活程序 → 停掉 → 跑 P0。停的时候连带清掉它留下的 ray worker 孤儿进程。工作2 的检索服务可以留着：P0 用 8100 / 8101 端口，清理时只杀 P0 自己启动的进程。`bash $P0 check` 会列出卡上所有计算进程，不为空就报错；确实要保留的话，设 `ALLOW_BUSY_GPU=1`。

## 2. 代码

```bash
cd <工作目录>
git clone https://github.com/CathyMuo/work3.git && cd work3 && git checkout p0-r4 && cd ..
git clone https://github.com/facebookresearch/drzero.git
cd drzero && git checkout 98c4a6dd2495e85d8b4d947ab4c030e540c922c5 && cd ..
```

Dr. Zero 的文件一个都不要改。所有适配都由 `run_p0.sh patch` 生成到 `work3_p0/patched/`，或者在运行时由 `work3.p0_launch` 包一层完成。

## 3. 环境（drzero，Python 3.10）—— 按厂内实测跑通的版本钉死

| 包 | 版本 | 说明 |
| --- | --- | --- |
| torch | 2.6.0 | 内网 PyPI 默认就是 cu124 构建；`download.pytorch.org` 在厂内不可达 |
| sglang | 0.4.6.post5 | |
| transformers | 4.51.1 | 由 sglang 决定 |
| compressed-tensors | **0.10.2** | sglang[all] 默认装 0.15，会导入 4.53 才有的 `transformers.masking_utils`，导致奖励模块加载失败 |
| uvloop | **0.21.0** | 0.23 在主线程没有事件循环时直接报错，verl 的 `fsdp_sglang.py` 会崩 |
| numpy | 1.26.4（<2） | |
| faiss | `faiss-gpu-cu12==1.8.0.2` | conda 没有 py3.10 的 1.8.0；pip 最新版要求 numpy≥2，会把 numpy 静默升级 |
| flash-attn | 2.7.4.post1，**ABI 与 torch 一致** | 硬依赖（verl `dp_actor.py` 在 CUDA 下无条件导入）；内网没有 wheel，需从 GitHub release 下载 |
| pandas | 2.3.3（<3） | |
| tensordict | 0.9.1 | |

```bash
conda create -n drzero python=3.10 -y && conda activate drzero
pip install torch==2.6.0 torchvision
pip install "sglang[all]==0.4.6.post5" "torch-memory-saver>=0.0.5"
pip install "compressed-tensors==0.10.2" "uvloop==0.21.0" "transformers==4.51.1"
pip install "faiss-gpu-cu12==1.8.0.2" "numpy<2.0.0" "pandas<3" "pyarrow>=19.0.0" \
    "tensordict>=0.8.0,<=0.9.1,!=0.9.0" accelerate codetiming datasets dill hydra-core peft pybind11 \
    pylatexenc "ray[default]>=2.10" torchdata wandb huggingface_hub liger-kernel math_verify uvicorn fastapi httpx
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"     # False → 下载 abiFALSE 的 wheel
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

`bash $P0 check` 会逐项检查上述版本，并实际执行三类导入：奖励模块、flash-attn，以及 uvloop 策略下的 `get_event_loop()`。只要有一项不过就会报错，**不会等到训练时才崩**。

## 4. 变量

```bash
conda activate drzero
export DRZERO_DIR=<工作目录>/drzero
export MODEL_3B=<models>/Qwen2.5-3B-Instruct
export CORPUS_JSONL=<models>/wiki-18/wiki-18.jsonl
export INDEX_PATH=<models>/wiki-18/e5_Flat.index
export E5_MODEL=<models>/e5-base-v2
export TRAIN_PARQUET=<models>/nq_hotpotqa_train/train.parquet
export TEST_PARQUET=<models>/nq_hotpotqa_train/test.parquet
export NGPU=4
P0=<工作目录>/work3/scripts/p0/run_p0.sh
```

可选变量：
- `RET_PORT` / `REWARD_PORT`：检索服务和奖励服务的端口，默认 8100 / 8101。
- `CHAL_ROLLOUT_MEM`：出题者训练内 SGLang 的显存比例，默认 0.45。
- `EVAL_ROLLOUT_MEM`：评测时的显存比例，默认 0.6。
- `GEN_MEM`、`FAISS_GPU=0`：显存不够时用，见第 6 节。

## 5. 运行（tmux 里一步一步来）

```bash
bash $P0 check
bash $P0 prepare    # 已完成的话跳过
bash $P0 patch
bash $P0 iter1 ; bash $P0 eval1
bash $P0 iter2 ; bash $P0 eval2
bash $P0 iter3 ; bash $P0 eval3
bash $P0 eval0      # 基座模型，只作参考
bash $P0 gate       # 写 work3_p0/p0_summary.json
```

每一步失败都会停下，日志在 `work3_p0/logs/<阶段>.log`。`timings.txt` 现在每行是“阶段 秒数 退出码 结束时间”。

**上游的三个坑，`p0-r2` 已经替你处理**：
1. 检索服务对不带 `"return_scores": true` 的请求一律返回 500：上游代码无条件解包两个返回值。凡是自己手动测检索服务，请求里都必须带这个字段。
2. `--config-path` 必须用绝对路径：Hydra 把相对路径解析到 `verl/trainer/main_ppo.py` 所在目录，而不是当前目录。`patch` 会把 `CONFIG_PATH` 改成 `$DRZERO_DIR/config`，并检查 9 个脚本都已改到。
3. uvloop / compressed-tensors 的版本问题：见第 3 节。

## 6. 速度、显存与功率：怎样判断是否正常

**看板**（推荐一直开着）：

```bash
DRZERO_DIR=<工作目录>/drzero BOARD_PYTHON=$(which python) bash <工作目录>/work3/scripts/run_board.sh
# 浏览器打开 http://127.0.0.1:8766/ ；在别的机器上看：ssh -L 8766:127.0.0.1:8766 <厂内机器>
```

看板是只读服务：所有内容都从运行目录和仓库文件读取，不写任何东西，每 8 秒刷新一次。页面分四块：

- **实验进度**：
  - 顶部是研究路线：P0 → P1 → P2 → 主实验 → 投稿，每一站显示状态、冻结门槛和关键结果。
  - 每个运行有一个总进度条，旁边是还需多久、预计几点完成。
  - “正在进行”框用一句话说明当前这一步在做什么，并给出本轮训练第几步、本步内采样进度。
  - 三轮训练各占一条泳道，每条只分 4 段：出题者训练、生成题池、解题者训练、评测。段宽按耗时比例，评测段上直接显示 EM；末尾是“收尾”（基座评测、门槛判定）。
  - 评测完成后，各轮 EM 会和论文值、0.306 门槛并排显示。
  - 原始子步骤、检索服务、告警和日志都收在“详情”里。
  - 页面底部是每张卡的利用率、功率和显存条。
- **训练曲线**：
  - 可以选任意运行、任意训练或评测阶段。verl 每一步打印的全部数值指标（`critic/`、`actor/`、`response_length/`、`num_turns/`、`timing_s/`、`perf/`、训练中的验证 EM `val-core/…` 等）按组列出，点选即可画图；所选指标会记住。
  - 提供“常用指标 / 验证 EM / 耗时·吞吐”三个一键组合。
  - 可以叠加三轮对比，横轴换成本轮内步数；也可以做平滑。
  - 鼠标悬停显示每一步的数值；出题者奖励单独成图。
  - P2 各臂另有出题奖励日志的曲线：平均难度分、信号占比 w、噪声占比 η、k 分布、每步生成 token 和耗时。
  - GPU 功率、利用率、显存，以及检索延迟和请求数的历史曲线：看板启动后每 10 秒采样一次，只存在内存里，保留 6 小时。
- **研究说明**：一份可以直接给导师讲的图文页（`research/briefing.html`），依次讲背景、问题、核心思路、NCP 方法（带两张示意图和三类题的对比表）、和已有工作的区别、实验设计与门槛、预期贡献、风险和时间线。
- **项目记录**：STATE、实验计划、冻结的门槛和决策记录，都直接从仓库读取。

看板只读日志、不碰训练，**正在跑实验时也可以随时更新看板版本**：从 `p0-r2` 起，后续版本的运行脚本都没有改动，git 不会动到正在执行的文件。浏览器地址后面加 `#brief` 可以直接打开研究说明页，加 `#cur` 直接打开训练曲线页。

**各阶段的正常表现**（用于判断，不是硬性指标）：

| 阶段 | 功率 / 利用率 | 说明 |
| --- | --- | --- |
| 反向传播 | 450–560 W，>90% | 已接近满载 |
| rollout 前段（并发多） | 应明显高于空闲；利用率长时间 <30% 说明并发不足 | `CHAL_ROLLOUT_MEM` 从 0.25 提到 0.45 就是为这一段 |
| rollout 尾段 | 100–150 W，接近 0% | **预期现象**：少数轨迹要做满 5 轮“生成→检索”，每轮都得等上一轮结束，属于延迟瓶颈，加显存解决不了 |

如果尾段时间特别长，请对照看板上的检索延迟：p95 在百毫秒级说明检索不是瓶颈；超过 1 秒，或者“进行中”请求长期堆积，请截图发回。

**关于反馈 #2 里的几个问题**：
- `undefined function: query / answer`：出题者偶尔会写错工具名，这是模型的采样错误。提示词示例和工具描述里写的都是 `search`，两者是匹配的。工具名写错的调用会被丢弃，这条 rollout 通常拿不到或只拿到很少的格式分；这正是出题者要从奖励里学会避免的错误。上游复现时一样会出现，而且 P0 不能改提示词。所占算力很小：每 50 步约几十次，相对几万条 rollout。看板会统计次数；如果比例持续上升，请反馈。
- `max_parallel_calls=64`：指一个回合里最多并行执行几次工具调用，跟请求并发数无关，调大没有用。
- HRPO 是“按跳数分组计算优势”的训练算法，不负责调度，对 rollout 尾段的耗时没有影响。

**显存不够时**：先 `export CHAL_ROLLOUT_MEM=0.35`（或者退回上游的 0.25），生成题池阶段用 `GEN_MEM=0.7`，还不行就 `FAISS_GPU=0`。改完重跑 `patch` 和失败的那一步。

## 7. 耗时估计（第 1 轮跑完后，看板会按实测重估）

| 步骤 | 估计 |
| --- | --- |
| 出题者 50 步 | ~8 h |
| 生成题池 | ~3–5 h |
| 解题者 50 步 | ~6 h |
| 每次评测（51,713 题） | ~2–4 h |
| 合计（3 轮 + 4 次评测） | 约 3.5–4.5 天 |

**磁盘**：约 450 GB。不要删 `global_step_50/100/150`、`solver_iter*_hf` 和 `data/zero_challenger_iter*.parquet`；`global_step_25/75/125` 可以删。

## 8. 发回家服的内容

```bash
cd $DRZERO_DIR/work3_p0
cat p0_summary.json; cat pool_check_iter*.json; cat timings.txt
cat logs/retriever_stats.json 2>/dev/null
grep -h "retriever ready" logs/*.log
grep -h "Final rewards" logs/iter*_challenger.log | awk 'NR%10==0' | tail -n 30
pip freeze | grep -iE "^(torch|sglang|transformers|flash|faiss|pandas|tensordict|ray|numpy|uvloop|compressed)"
cd <工作目录>/work3 && git describe --tags --always
```

失败时额外发失败阶段日志的最后 80 行。看板截图也可以直接发。
