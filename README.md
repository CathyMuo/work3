# Work 3

**2026-11-24 当前为预算受限 RLVR 同前缀试点。** [厂内运行指南](docs/factory/budget_rlvr_paired_pilot.md) · [冻结协议](research/protocols/budget_rlvr_paired_pilot.md) · [研究计划](research/EXPERIMENT_PLAN.md)

历史 Wikipedia 研究已经收束。P0 通过，P1 NO-GO，NCP 已停止，P2 已取消。微批次消融完成，判定 NO EVIDENCE。

- 历史方向为零数据自博弈搜索智能体。 [P1 诊断协议](research/protocols/P1_noise_diagnosis.md)
- 基线：Dr. Zero（facebookresearch/drzero @ `98c4a6d`，verl 0.5 + SGLang）。
- 历史多跳证据诊断见 [决策记录](research/DECISION_LOG.md)和[厂内回复 #15](docs/factory/REPLY_20261116_fb15.md)。
- 家服实验：[Wikipedia 输入审计与冻结回放](docs/RSI_wikipedia_home.md)
- 看板：`bash scripts/run_board.sh` → http://127.0.0.1:8766/（只读服务：研究说明 + 实时进度）
- 目标：ICML 2027（主）/ SIGIR 2027（备），12-15 决定。

仓库：https://github.com/CathyMuo/work3（公开，原名 work3_byd）。协作方式与工作2相同：家服开发并推送，厂内 4×H800 拉取运行、回传汇总。
