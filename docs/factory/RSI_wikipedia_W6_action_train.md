# W6b 独立动作对的同预算短训

复用 W6a 已核验的 155 个训练对和 384 道独立自然题。比较冻结 R3、仅正例更新、真实动作配对更新。两个训练臂均从相同初始化的 LoRA 开始。输入顺序、成对序列、非填充 token 数、优化器步数和搜索预算相同。仅正例臂只对正查询反传有效梯度，配对臂在同一正例损失上增加相对于冻结模型的正负偏好项。后者使用查询平均 token 对数概率。

固定 query/value 投影 LoRA，rank 8，缩放 2。AdamW 学习率 5e-5，训练两遍，四个训练对累积一次更新，共 78 步。配对项权重 0.5，种子 20261119。使用本机 Transformers 和 PyTorch，三个评估臂都走同一个生成后端。评估按 W5b 提示，每题最多 96 新 token，贪心生成，只执行一查并保留 top-3。

主评估分别在 HotpotQA、2Wiki 标题逐字可见的 128 道上检验取齐全部缺句。配对臂相对仅正例和冻结臂在两个数据集均至少净胜 5 道，才安排扩大后的独立确认。两个未逐字可见层各 64 道，独立报告配对、覆盖和执行，作为扩展迁移诊断。分层刻意改变了抽样比例。格式失败和超长历史留在各自分母。主指标是实际取证，这一轮没有追加作答生成。

## 执行

复用已完成 W6a 的 RUN。准备阶段为 CPU，核查训练和评估 SHA、重新验证正负查询证据及页面隔离，再固定三组共享的提示。执行阶段用一张空闲 H800 顺序运行三个臂，另验八条真实检索锚点。准备阶段可先于 GPU 时间窗口完成。

```bash
git fetch origin refs/heads/main
git merge --ff-only FETCH_HEAD
git rev-parse HEAD

export DRZERO_DIR=/root/develop/workspace/drzero
export DIAG_PYTHON=/root/develop/persistent-miniconda3/envs/drzero/bin/python
export REPLAY_PYTHON=/root/develop/persistent-miniconda3/envs/rsi/bin/python
export WIKI_RSI_ROOT="$PWD/outputs/wiki_rsi"
export WIKI_RSI_RETRIEVER=http://127.0.0.1:8000/retrieve
export WIKI_RSI_RETRIEVER_ID='P0 wiki-18.jsonl / e5_Flat.index / intfloat/e5-base-v2'
export P0_ITER3_MODEL="$DRZERO_DIR/checkpoints/dr-zero/solver_iter3_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter3_hf"
RUN=wiki-w0c-20261118-110954

bash scripts/run_wiki_rsi_w2a.sh action-train-prepare --run "$RUN"
read -r -p '空闲 H800 的设备编号: ' IDLE_GPU
CUDA_VISIBLE_DEVICES="$IDLE_GPU" bash scripts/run_wiki_rsi_w2a.sh action-train --run "$RUN"
```

每组最多 384 次实际单查，三组合计至多 1,152 次，另有八条锚点。GPU 阶段不加载 vLLM。看板会显示加载权重、参考评分、当前更新臂和实际取证。用原 P0 权重目录，短训的适配器在本次进程内评估，结果文件包含逐题输出及运行配置。

## 回传

回传版本、GPU 编号、两阶段退出码和以下三个完整 JSON。逐题材料保留在厂内。

- `wiki_rsi_w6_action_train_prepare_summary.json`
- `wiki_rsi_w6_action_train_retriever_check.json`
- `wiki_rsi_w6_action_train_summary.json`

逐层复核样本分母和训练预算一致性。若配对未过门槛，保留该小试结果，后续先确认负例信号和跨题型分布的限制。通过后仍须用独立样本、强查询基线及作答指标确认。
