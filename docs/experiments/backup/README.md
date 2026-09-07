# 归档（只读）

这里放**已经执行完毕、或规格已被后续文档取代**的计划文档。它们保留下来是为了让人能追溯
"当时为什么这么决定"，**不是当前要执行的任务**。

> **给接手的会话**：不要执行这里的任何规格、不要修改这里的任何文件、不要把这里的判定
> 当成当前状态。当前唯一的活计划是
> [`../signal_resolution_plan.md`](../signal_resolution_plan.md)。

**归档规则**：一份计划的全部任务执行完毕、或它的规格被后续文档取代之后，就移进来，并在
下表加一行。**结果不跟着进来**——结果留在对应的结果档案或运行记录里，否则会出现
"找不到数字"。

| 文档 | 它是什么 | 被谁取代 | 结果落在哪 |
|---|---|---|---|
| `adaptive_vs_fixed_claim_plan_handoff.md` | 2026-09-06 会话中断时的交接记录（T0–T3 哪些跑完、接下来敲什么命令） | 无（T0–T3 已全部执行完毕，交接对象已不存在） | `../adaptive_vs_fixed_claim_plan.md` 第十一、十三节 |
| `readout_controllability_gate_plan.md` | 2026-09-07 联合审计立的 RC-gate 计划（D1/D2/D3 + E0/E1/E2） | `../signal_resolution_plan.md`：RC-gate 从三条判据升级为四条（新增 RC-0、RC-1 改多 split + aux 真值覆盖、RC-3 加方向声明）；其 §6（ERGO Phase C）迁移到新计划第四节并做了三处修订 | D1 → `../adaptive_vs_fixed_claim_plan.md` 第十三节；D3 → 已落在 README / 各结果档案；E0/E1 → `../ergo_multiturn_reliability_pilot.md`（**其闸门判定已被新计划 F0 修正**）；D2 因 RC-A 不过而作废，从未执行 |
| `ergo_koopman_mpc_opus_design_questions.md` | ERGO Phase C 的两个待裁决设计问题（`shard_frac` 展望 / reset 预算） | `../signal_resolution_plan.md` 4.1（真值覆盖，已从"建议"升级为"必须"）与 4.2（实测 token 代价 + `k=1` 预算） | 裁决本身就是结果，见上 |

**已知的、留在归档里没有改的错误**（不要照抄这些数字）：

- `readout_controllability_gate_plan.md` 第 0.4 节把 ERGO 的 `ARX 0.0893 vs stateless 0.0792`
  当成 RC-1 不过的硬证据。那个数字本身没算错，但两个模型的信息集不对等（null 拿到真值
  `shard_frac`，ARX 要自己外推一个自系数 1.005 的确定性斜坡）。修正口径与 20-split 结果见
  `../signal_resolution_plan.md` 第 0.2 节。
- 同文档第四节 E0 的 RC-3 规格被实现成 `u_next = u_{t+2}`（差一位），见新计划第 0.1 节。
