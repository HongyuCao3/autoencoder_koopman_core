# 归档（只读）

这里放**已经执行完毕、或规格已被后续文档取代**的计划文档。它们保留下来是为了让人能追溯
"当时为什么这么决定"，**不是当前要执行的任务**。

> **给接手的会话**：不要执行这里的任何规格、不要修改这里的任何文件、不要把这里的判定
> 当成当前状态。当前的活计划只有一条（2026-09-10 更新）：
> `constraint` 线 → [`../constraint_retention_plan.md`](../constraint_retention_plan.md)（S0–S3，S1 已提交）。
> ERGO 线（[`../ergo_fidelity_restoration_plan.md`](../ergo_fidelity_restoration_plan.md)）与防御线的两份计划
> （[`../koopman_fast_track_plan.md`](../koopman_fast_track_plan.md) 与
> [`../defense_line_redesign_plan.md`](../defense_line_redesign_plan.md)）按用户指令**收尾挂起**——
> 挂起不等于归档，它们仍在 `../` 下，重启时从各自顶部的横幅进入。

**归档规则**：一份计划的全部任务执行完毕、或它的规格被后续文档取代之后，就移进来，并在
下表加一行。**结果不跟着进来**——结果留在对应的结果档案或运行记录里，否则会出现
"找不到数字"。

两种情况**不归档**（2026-09-10 复核 `../` 下 28 份文档后写死的判据）：**挂起的线**
（挂起不等于归档，见上）；**未执行且未被取代的设计稿**（例如
[`../reminder_history_interaction_plan.md`](../reminder_history_interaction_plan.md)——它既没跑完
也没被取代，归档规则的两个条件都不满足，留在 `../` 下带 📝 横幅）。含带数字结果节的文档
一律**原地冻结**（只加状态横幅，不拆分），拆一份 1598 行的结果档案等于重写历史。

| 文档 | 它是什么 | 被谁取代 | 结果落在哪 |
|---|---|---|---|
| `adaptive_vs_fixed_claim_plan_handoff.md` | 2026-09-06 会话中断时的交接记录（T0–T3 哪些跑完、接下来敲什么命令） | 无（T0–T3 已全部执行完毕，交接对象已不存在） | `../adaptive_vs_fixed_claim_plan.md` 第十一、十三节 |
| `readout_controllability_gate_plan.md` | 2026-09-07 联合审计立的 RC-gate 计划（D1/D2/D3 + E0/E1/E2） | `../signal_resolution_plan.md`：RC-gate 从三条判据升级为四条（新增 RC-0、RC-1 改多 split + aux 真值覆盖、RC-3 加方向声明）；其 §6（ERGO Phase C）迁移到新计划第四节并做了三处修订 | D1 → `../adaptive_vs_fixed_claim_plan.md` 第十三节；D3 → 已落在 README / 各结果档案；E0/E1 → `../ergo_multiturn_reliability_pilot.md`（**其闸门判定已被新计划 F0 修正**）；D2 因 RC-A 不过而作废，从未执行 |
| `two_task_success_plan.md` | 2026-09-07 立的两线计划：ERGO 线 E0–E6（append 执行器主攻）+ 防御线 D0–D3（两个 GPU-天的时间盒） | **ERGO 半边**由 `../ergo_fidelity_restoration_plan.md` 取代：E2/E3 跑完后 ERGO 定格 S3，失效被归因到本 harness 自己的 prompt 结构（每轮重复的答案格式指令 + 无 system prompt），新计划 R0–R5 修协议保真度。**防御半边**（第四节 D0–D3、12.8 的 D1 规格）**未被取代**，只是执行停在盲标；从 `../defense_line_redesign_plan.md` 第十三节进入 | ERGO → `../ergo_multiturn_reliability_pilot.md` 的「E2–E3」节；D1 采集 → `outputs/d1_screen_*`，判定未出 |
| `ergo_koopman_mpc_opus_design_questions.md` | ERGO Phase C 的两个待裁决设计问题（`shard_frac` 展望 / reset 预算） | `../signal_resolution_plan.md` 4.1（真值覆盖，已从"建议"升级为"必须"）与 4.2（实测 token 代价 + `k=1` 预算） | 裁决本身就是结果，见上 |
| `ergo_R_phase_closed_specs.md` | ERGO R 相位里被关闭/挂起的三步原规格（R2 attempt 口径 / R3 模型重写 reset / R5 熵触发臂） | `../ergo_fidelity_restoration_plan.md` 第零节的 Koopman 贡献过滤器——这三步都填不出 RQ3 的任何一格（2026-09-08 用户指令） | G-R2-1 的盲标结果在 `../ergo_multiturn_reliability_pilot.md` 的「R 相位的记录与裁决」节；R3/R5 从未执行，无结果 |

**已知的、留在归档里没有改的错误**（不要照抄这些数字）：

- `readout_controllability_gate_plan.md` 第 0.4 节把 ERGO 的 `ARX 0.0893 vs stateless 0.0792`
  当成 RC-1 不过的硬证据。那个数字本身没算错，但两个模型的信息集不对等（null 拿到真值
  `shard_frac`，ARX 要自己外推一个自系数 1.005 的确定性斜坡）。修正口径与 20-split 结果见
  `../signal_resolution_plan.md` 第 0.2 节。
- 同文档第四节 E0 的 RC-3 规格被实现成 `u_next = u_{t+2}`（差一位），见新计划第 0.1 节。

- `two_task_success_plan.md` 第七节"为什么 append 能恢复状态"的机制推理**只对了一半**：append
  确实让历史进入终点，但它同时让模型照抄自己上一轮的旧答案，reset 不再触发重新求解——那次
  "从头解"才是 overwrite 下 reset 权威的真正来源。见
  `../ergo_multiturn_reliability_pilot.md`「R 相位的记录与裁决」节的 0.1
  （原在 `../ergo_fidelity_restoration_plan.md`，2026-09-08 迁入结果档案）。
- 同文档 12.4 预判"模式 A 近乎预定只能到 S2"；实际两个模式都没走到，G-E3-2 先关门。
- 同文档第五节把论文 §1.3 第 5 句的填空写成"E5 的结论无论正负都填这里"；E5 未跑，填进去的是
  E2/E3 的双向结果。
