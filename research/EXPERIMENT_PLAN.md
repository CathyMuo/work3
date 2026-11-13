# Work 3 实验计划

更新：2026-11-06。原 PBRO 计划已由用户放弃（2026-11-05）。

方向（用户 11-05 确认）：零数据自我进化（自博弈）搜索智能体。工作名 NCP：给自博弈出题奖励去掉与搜索行为无关的噪声。

| 阶段 | 内容 | 门槛 | 状态 |
| --- | --- | --- | --- |
| P0 | 厂内复现 Dr. Zero Qwen2.5-3B（3 轮），保存题池与解题检查点 | 第 3 轮 7 数据集平均 EM ≥ 0.306（[协议](protocols/P0_drzero_reproduction.md)，已冻结） | 厂内运行中（[指导](../docs/factory/P0_guide.md)，tag `p0-r14`） |
| P1 | 基于 P0 产物的离线诊断 | [协议](protocols/P1_noise_diagnosis.md)，已冻结，SHA-256 见 DECISION_LOG | 代码已就绪（[指导](../docs/factory/P1_guide.md)，tag `p1-r14`），等 P0 |
| P2 | 3B，三个臂（Dr. Zero = 复用 P0；Beta 平滑 n=7；NCP），各 3 轮 | NCP 第 3 轮平均 EM 比 Dr. Zero 高 ≥ 2.0 pp、比 Beta 高 ≥ 1.0 pp，且第 2→3 轮不下降（[协议](protocols/P2_ncp_pilot.md)，已冻结） | 代码已就绪（[指导](../docs/factory/P2_guide.md)，tag `p2-r12`）；需 P0 PASS 且 P1 GO |
| M | 主实验 | P2 之后登记 | — |

GO / NO-GO：P1 约 11-19，P2 2026-12-03。门槛冻结后不改。

## 不做
GMST 相关、停止策略、查询选择、证伪检索、干预式 E 作奖励、反思 / 重试信用分配（PBRO）。
