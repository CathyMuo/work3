# Work 3：零数据自我进化的搜索智能体

**2026-11-06 状态：方向已确认，P0 / P1 门槛已冻结，等待厂内运行 P0。**

- 方向：递归自我改进（RSI）中的零数据自博弈搜索智能体（出题者 / 解题者共用同一模型，只靠检索环境提供监督）。P1 诊断协议见 [research/protocols/P1_noise_diagnosis.md](research/protocols/P1_noise_diagnosis.md)。
- 基线：Dr. Zero（facebookresearch/drzero @ `98c4a6d`，verl 0.5 + SGLang）。
- 当前任务：[P0 厂内指导](docs/factory/P0_guide.md)（tag `p0-r6`），之后 [P1 厂内指导](docs/factory/P1_guide.md)（tag `p1-r6`）、[P2 厂内指导](docs/factory/P2_guide.md)（tag `p2-r6`）；计划见 [实验计划](research/EXPERIMENT_PLAN.md)。
- 看板：`bash scripts/run_board.sh` → http://127.0.0.1:8766/（只读服务：研究说明 + 实时进度）
- 目标：ICML 2027（主）/ SIGIR 2027（备），12-15 决定。

仓库：https://github.com/CathyMuo/work3（公开，原名 work3_byd）。协作方式与工作2相同：家服开发并推送，厂内 4×H800 拉取运行、回传汇总。
