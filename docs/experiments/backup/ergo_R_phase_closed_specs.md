# 归档：R 相位中被关闭 / 挂起的三个步骤的原规格（R2 / R3 / R5）

**移入日期**：2026-09-08 · **来源**：`../ergo_fidelity_restoration_plan.md` 第四、五、七节
**为什么移出来**：按 2026-09-08 的用户指令，ERGO 线的剩余步骤按「它对 Koopman 主张有没有
实质贡献」重新筛过一遍。这三步都填不出 RQ3 的那个落点（辨识 + 闭环），因此从活计划里移出。
**处置见活计划的「已关闭 / 挂起」表**——那里写了每一步为什么关、以及重开需要什么条件。

> **不要执行本文件里的任何规格。** 它保留在这里只是为了将来若某一步重开，规格不必重写。
> 这些规格的判定与结果（G-R2-1 不通过、盲标数字）在
> [`../ergo_multiturn_reliability_pilot.md`](../ergo_multiturn_reliability_pilot.md)
> 的「R 相位的记录与裁决」一节，**不在本文件里**。

---

## 四、R2 · answer-attempt 打分口径（CPU 为主 + 少量 GPU）

上游只给 answer attempt 打分。我们目前给每一轮的正则抽取打分，于是在 append 下大量把
「非作答」记成 0 分并计入指标——这既污染主指标，也污染 `closeness` 读出（RC-3 就是在这些行上算的）。

### 4.1 实现

新增 `src/persona_drift/ergo_answer_attempt.py`：`classify_attempt(agent_message) -> bool`。
**两级**：
1. **规则层**（零成本，先做）：回复去掉 `Current answer: X` 那一行后，剩余非空白字符数 < 阈值
   → 判为非作答。阈值**预注册为 80 字符**（与第零节诊断同一刻度，不许在看到结果后调）。
2. **LLM 层**（仅当 G-R2-1 不过时才启用）：复用管线里已有的判官模型，prompt 照上游 7 类
   strategy classifier 的形状，只取 `answer attempt` 与否的二值输出。

新增 `scripts/analyze_ergo_dual_metric.py`：所有指标**同时**报两个口径——
`legacy`（全轮打分，与全部历史数字可比）与 `upstream`（只给 attempt 打分）。
**任何单口径的汇报都算偏离**（失败模式 23）。

**规格补丁 D1（Opus，2026-09-08）· "只给 attempt 打分"在变长轨迹上是歧义的，此处写死。**
上游按对话记分、并跟踪最后一次 answer attempt；我们的主指标是**每条轨迹自己的末轮**。
若末轮不是 attempt，有两种读法，**必须选一种并写死**：

| 读法 | 做法 | 裁决 |
|---|---|---|
| **(a) 保分母** | 回退到**该轨迹最后一个被判为 attempt 的轮次**的分数；全程无 attempt → 记 **0** | ✅ **采用** |
| (b) 缩分母 | 把末轮非 attempt 的轨迹整条丢掉 | ❌ **不采用** |

**理由**：(b) 会让每个臂的 n 不同，而"产生 attempt 的比例"本身就是臂间差异最大的量
（`always_reset_append` 末轮 91% 是 `Current answer: 12`）——按它筛样本，等于用一个与结果强相关的
量做选择，配对也当场断掉。(a) 保住 58 个配对 item、保住 bootstrap 的配对结构，且与上游"跟踪最后
一次 answer attempt"的做法同构。

因此 `analyze_ergo_dual_metric.py` 必须输出三列而不是两列：
`final_turn_success`（legacy 主指标，不变）、`final_attempt_success`（upstream 口径，按 (a)）、
以及 `attempt_rate`（该臂末轮是 attempt 的轨迹占比）。**第三列是解释前两列差异的必需品**，
不是可选诊断：两个口径的差全部来自它。

### 4.2 闸门

| 闸门 | 判据 | 不过 → |
|---|---|---|
| **G-R2-1 分类器有效性** | 在 60 行分层抽样上做**盲标**（P 协议，`general-purpose` subagent，每批全新实例），规则层与人判的 agreement **≥ 0.85**，且两类各自 ≥ 20 行 | 启用 LLM 层，重测；仍不过则**不换口径**，R2 终止并报告 |
| **G-R2-2 回归锁** | 在既有 legacy overwrite Phase C 数据上按新口径重算，`always_reset ≡ fixed_last` 的 **116/116 恒等式必须保持** | 分类器引入了臂间不对称，是 bug，修 |
| **G-R2-3 读出重算** | 在 `outputs/ergo_appendB_random_excite` 上按 upstream 口径重算 RC-0..3 与交互回归的 b0/b2 | 只记录；**这是诊断，不是把 G-E3-1/G-E3-2 的判定翻过来**（失败模式 21） |

**用一个没验过的分类器换掉主指标，就是用一个未验证的测量替换另一个**——G-R2-1 存在的理由就是这个。

---

---
## 五、R3 · 忠实的 ERGO reset：模型重写的 consolidation prompt

ERGO 的 reset 不是机械拼接，是**让模型把累积输入重写成一个优化的单轮 prompt**，它 56.6% 的增益
是在重写版上测的。同时这里能拆开上游自己没拆的混淆。

### 5.1 实现与臂

- `_consolidated_stimulus` 增加 `consolidation: str = "bullet" | "rewrite"`。`"rewrite"` 用同一个
  agent 模型（**不是判官**，避免把判官拉进被控回路）把已揭示 shard 重写成单轮 prompt。
- **消融对（上游没做的那个）**：
  - `fixed_last_append_up_bullet`（= R1 的 `fixed_last_append_up`，复用）
  - `fixed_last_append_up_rewrite`
  二者只差 consolidation 形式，**同一时机、同一代价** → 差值就是「重写 prompt」的净贡献。

### 5.2 闸门

| 闸门 | 判据 | 不过 → |
|---|---|---|
| **G-R3-1 信息保全** | 重写产物中，gold answer 推导所需的**全部数值**出现率 ≥ 0.95（对每个 item 用 shard 原文里的数字集合做覆盖检查） | 重写在删信息，它的效应与 reset 效应混淆，**修 prompt 再跑**；不许带着信息损失往下走（失败模式 24） |
| **G-R3-2 消融** | `rewrite − bullet` 的配对差与 CI | 只记录，不判定 |

**R1 未过不做 R3**：R3 工程量最大，且它的结论建立在 R1 的前提上（失败模式 25）。

---

---
## 七、R5 ·（**已推迟**，Q9 裁决）熵触发臂

上游 ERGO 相对 RECAP 的 +14.1 / +9.3 分来自**熵触发的自适应时机**。这正是本项目 P1/P2 要证的东西。
工程尾巴：要 `chat_model.generate` 在线吐 per-token 熵，`analyze_ergo_entropy_readout.py` 已经
离线做过一次熵读出，可复用其定义。触发规则照上游：ΔH̄(t) = H̄(t) − H̄(t−1) > τ，τ 按模型标定
（上游范围 0.03–0.3）。

**Q4 当初裁决"不做"的理由是「压在关键路径上、工程尾巴无界、按计划自己的表属可选」。那个裁决是在
这条线看着快死的时候做的；现在上游告诉我们它就是产生 +14 分的那个部件。**

**R0 裁决（Q9）：推迟，不否决——等 ERGO 的 Koopman 工作跑完之后再试。** 本节保留为待办的完整规格，
但 **R5 不在本轮范围内**，R1–R4 的任何闸门结果都不自动触发它。要开工需要一次新的签字。

---

