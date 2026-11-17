# Work 3 实验计划

更新：2026-11-17。原 PBRO 计划已由用户放弃（2026-11-05）。

方向（用户 11-05 确认）：零数据自我进化（自博弈）搜索智能体。工作名 NCP：给自博弈出题奖励去掉与搜索行为无关的噪声。

| 阶段 | 内容 | 门槛 | 状态 |
| --- | --- | --- | --- |
| P0 | 厂内复现 Dr. Zero Qwen2.5-3B（3 轮），保存题池与解题检查点 | 第 3 轮 7 数据集平均 EM ≥ 0.306（[协议](protocols/P0_drzero_reproduction.md)，已冻结） | 通过（[指导](../docs/factory/P0_guide.md)，tag `p0-r16`） |
| P1 | 基于 P0 产物的离线诊断 | [协议](protocols/P1_noise_diagnosis.md)，已冻结，SHA-256 见 DECISION_LOG | 已完成，NO-GO（[数据](results/P1_summary_20261116.json)） |
| P2 | 3B，三个臂（Dr. Zero = 复用 P0；Beta 平滑 n=7；NCP），各 3 轮 | NCP 第 3 轮平均 EM 比 Dr. Zero 高 ≥ 2.0 pp、比 Beta 高 ≥ 1.0 pp，且第 2→3 轮不下降（[协议](protocols/P2_ncp_pilot.md)，已冻结） | 取消（P1 NO-GO） |
| P0 消融 | 第 3 轮解题者微批次 1/2 重训 | 决策记录 2026-11-13 (b) | 已完成，NO EVIDENCE |
| P0 离线检索诊断 | 读取第 1、2、3 轮评测轨迹，分析固定查询预算和支持证据覆盖 | [已冻结协议](protocols/P0_retrieval_diagnosis.md) | 脚本就绪，待厂内 CPU 执行 |

厂内按[反馈 #16 交接](../docs/factory/REPLY_20261117_fb16.md)回传离线汇总，家服据此决定是否开展同前缀回放。

## 不做
GMST 相关、停止策略、查询选择、证伪检索、干预式 E 作奖励、反思 / 重试信用分配（PBRO）。
