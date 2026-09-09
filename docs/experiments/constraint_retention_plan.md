# 执行计划：`constraint` 线（约束保持）S0–S3

**状态**：▶ **活跃（2026-09-08 起）**。ERGO 的 EK 相位已出结果并挂起
（[`ergo_fidelity_restoration_plan.md`](ergo_fidelity_restoration_plan.md)：算子辨识成立，但输入通道与状态解耦，
闭环退化为固定日程），触发条件"ERGO 出较明确结果"已满足，本线开工。
**开工前筛查与死亡条件见 [`constraint_signal_screening.md`](constraint_signal_screening.md)（S0-0）。**
**适用**：Opus 5（裁决）/ Sonnet 5（执行）。**规则**：`.claude/global.md` → `.claude/experiments.md` → `.claude/code.md`，**本文件不重抄规则**。
术语见 [`../NAMING.md`](../NAMING.md) 的 `constraint` 行。

---

## 零、它在论文里占的三格（不填这三格的不跑）

| 论文位置 | 本线要交的东西 |
|---|---|
| 前提 (i) 执行器权威 | 提醒再注入是否推得动 `y`（S1a） |
| 前提 (ii)(iii) + Koopman 拟合表 `constraint` 行 | `y_{t+1}=Ay_t+Bu_t+c` 的 held-out 一步误差 vs 三个平凡 null（S1b→S2） |
| RQ3 闭环结果表 | Koopman-MPC vs 等代价随机分配 vs 同代价最优固定日程（S3） |

**论文里的三句话与那张表的空壳**（*开一条新任务线* 第 3 步）见
[`constraint_signal_screening.md`](constraint_signal_screening.md) 第七节。

**为什么是这个 task**：`defense` 卡在读出没量程（独立 judge ceiling 0.91、turn-1 `sd=0`）、
ERGO 卡在主模态是确定性斜坡 + 激励功效不足（MDE 0.49 vs 效应 0.181）。这里 `y` 是
**k 条并行验证通道的计数**（分辨率来自任务结构，不是把二值事件软化），且闭环可选的不只是
时机、还有**方向**（提醒哪一条）。

## 一、任务与读出

数据 vendor 自 `deep-spin/SEQUOR`（arXiv 2605.06353，COLM 2026）到 `resources/sequor/`，
写 `resources/PROVENANCE.md`（照 `ergo_gsm8k_sharded.jsonl` 的先例）：
`data/testsets/tuples/3/*.jsonl`（200 个对话，每行一轮 = `{prompt, constraints_data}`，
对话长 120–210 轮，**T 由我们截**）、`data/constraints/3.jsonl`（948 个三元组）、
`data/gold_responses/*/*.jsonl`（500 follow + 500 violate × 2 个生成模型）。

- **读出** `y_t = (满足的活跃约束数)/k`，k=3 → 取值 {0, ⅓, ⅔, 1}。逐约束判决是上游已有的
  （`multi_if/evaluation.py:20` `ConstraintEvaluation.followed`），上游把它 `all()` 成
  turn-success（`:515`）——我们**只换聚合方式**，不造新仪器。
- **副读出**：3 维逐约束判决向量（哪一条掉了）。它是阶段 2 动作空间的依据，也可作多变量
  Koopman 的 lift 目标。
- 量程先验（上游 Qwen3-4B-Instruct-2507，读图近似）：tuples ≈50% → <12% @turn50，single 74% → 48%，**全程中量程**；判分 prompt 只吃「答案 + 单条约束」，不吃历史。

## 二、动作与代价

| 阶段 | 动作空间 | 复用 | 填哪格 |
|---|---|---|---|
| 阶段 1 | `u ∈ {0,1}`：重述全部 k 条约束，拼进用户轮 | `control.Controller` 的 0/1 决策源，**不改 `control.py`** | 前提 (i)、拟合表 |
| 阶段 2 | `u ∈ {0,…,k}`：提醒哪一条 | 新模块（不动既有复现路径） | RQ3 表 |

代价两项，逐行记：`inserted_tokens`（沿用 ERGO 字段）与**任务完成度**（上游自带 `SYSTEM_PROMPT_TASK` / `submit_generate_task_evals.py`，良性代价对照内置，不再开 MT-Bench 臂）。

再加一个记录项（同上文件 C5 的同构物）：**相邻两轮回复的逐字重合率**——提醒会诱发模型照抄约束原文，
这个量把 "echo"（抄回来但没执行）与真实恢复分开。零成本，纯文本统计。

**已知解释性风险，写进报告不回避**：judge 只看答案，而提醒会让回复更格式化 →
**judge 的错误率可能与 `u` 相关**，会给 `B` 灌水。gold 集里没有 `u`，答不了这一问；
在 S2 报告时对「提醒过 / 未提醒过」的行做强 judge 抽样对照。`B` 不许写成纯因果状态增益。

## 二·五、S0-0 开工前信号筛查（S0 之后、S1 之前，1 个小 GPU 臂）

**12 个对话 × T=20 的反事实分叉臂 = 240 个同前缀对**（S1 是 3200 行），三道闸门 K1 读出量程 / K2 状态超出轮次 / K3 执行器权威。**规格、阈值、MDE 口径、死亡条件与非主张台账见
[`constraint_signal_screening.md`](constraint_signal_screening.md)。** K1 或 K2 不过即关线。
**把 EK-A 的分叉设计放到最前面而不是最后**，是 ERGO 那 18 个作业的直接教训。

## 三、S0 判分校准（零轨迹生成，纯离线）

在 2000 条 gold（标签平衡，含 follow / violate 两侧）上选两个 judge：
**in-loop 选择信号** = 与 agent 同一个服务模型（自判，只用于选动作）；
**报告口径** = 独立的更强模型（候选 Qwen3-14B / Qwen3-30B-A3B），双向准确率与 κ 都报。

| 闸门 | 判据（阈值取文献常用值） | 过了 → | 不过 → | 填哪格 |
|---|---|---|---|---|
| **G-S0-1** | 报告 judge 在 gold 上原始一致率 **≥0.80** 且 Cohen's κ **≥0.60**（对标上游 GPT-oss-120B 的 Gold-Yes 88.30% / Gold-No 98.80%） | 定下两个 judge，进 S1 | 见下 | 判分口径段 |
| **G-S0-2 不阻塞下限** | 平衡准确率 **≥0.75** 且 κ **≥0.40** | 用它但在每处标注"judge 弱 → MDE 抬高" | **换候选一次**；两轮都够不上 → **关线** | 同上 |

judge 弱只抬噪声、不偏移臂间差——**前提是错误率与 `u` 无关**，所以 G-S0 不能替代第二节那条抽样对照。

**S0 已完成（2026-09-09）**：报告口径 judge = **Qwen3-14B**（G-S0-1 过），in-loop 自判 judge = **Qwen3-4B-Instruct-2507**（= agent）。数与谱系见 `../LEDGER.md` §三·五 的 15739195 / 15739196 两行；判分 cap 的例外见 `constraint_signal_screening.md` §十 第 4 条。

## 四、S1 采集（两组配对臂，T=20，N=40 item）

**这一组同时是**：前提 (i) 的检验、Koopman 的训练集、S3 的 zero-control 下界臂。
不设代理前置闸门（`.claude/experiments.md` 的依据段：拟合本身是更强的仪器检验）。

- **S1a 权威臂**：`zero_control` vs `constant_remind`，**按 item 配对**、满剂量对比（0 vs 1）。
  n=40 的配对 t 检验，**提交前先算 MDE**：设每题差值 sd ≈ 0.30 → MDE@80% ≈ **0.13**（y 的分数单位）。
  对标 [`../task/MULTITURN_RESET_INTERVENTION_EVIDENCE.md`](../task/MULTITURN_RESET_INTERVENTION_EVIDENCE.md)
  §5 的 C1（**MDE ≤ 0.07 为目标、0.15 为底线**）：0.13 在底线内、够不上目标。
  **提交前用 smoke test 实测的 sd 重算**——sd > 0.30 就把 N 提到 80（≈0.094）或 145（≈0.07）；
  本线的 y 是 3 条通道的均值，sd 有理由小于 ERGO 的二值成功率，但这要测不要假设。
  **不附 MDE 的"不显著"不许写成证否**（ERGO 失败模式 29）。
- **S1b 激励臂**：同一批 item，**反相配对**——轨迹 A 的 `u_t ~ Bernoulli(0.5)`，轨迹 B 取 `1−u_t`。
  每个 (item, turn) 格子恰好一条被提醒 → `B` 的一步系数是匹配对估计，功效远高于 i.i.d. pooled。
  （反相设计**故意**把终点剂量对比压成 0，所以终点权威必须由 S1a 出——两组不可互相替代。）

**闸门 G-S1**：`COMPLETED`；行数 = 臂数 × 40 × 20；`item_id` 集合四臂完全相同；
S1b 的 `u` 均值 ∈ [0.45, 0.55] 且每个 (turn) 上两种动作都出现；**`y` 每一轮的跨轨迹 sd > 0 且 ≥3 个取值**。
过了 → 进 S2。不过 → 报告，不自行重采。

## 五、S2 Koopman 拟合（纯 CPU，分钟级）

照 `scripts/fit_koopman_defense_model.py` 的既有口径（`--contemporaneous-v` 必传，
`u_t` 的索引按 `koopman_fast_track_plan.md` §2.3），输入 S1b，输出目录全新。

| 闸门 | 判据 | 过了 → | 不过 → | 填哪格 |
|---|---|---|---|---|
| **G-S2-1 状态** | 一步 MSE 在 **≥14/20 折**上打过 `const` / `turn_mean` / `stateless` 三个 null 里最好的（三个 null 都拿到 `turn`） | 进 S3 | 走一轮闭合前审查后按判定关 | 拟合表 + 前提 (ii)(iii) |
| **G-S2-2 可控** | `B` 的 95% CI 不含 0，符号 = 提醒使 `y` 上升 | 同上 | 记录并交裁决 | 前提 (i) |
| **G-S2-3 非退化** | 谱半径 ∈ (0.1, 1.05)；可控性矩阵满秩；Gramian 条件数 < 1e12 | 只记录 | — | 诊断段 |
| **G-S2-4 样本量** | 用 `B` 与 late-`y` 配对差 sd 反推 S3 每臂轨迹数（80% 功效 / α=0.05）；**> 150 条/臂就停下来报告** | 定 S3 的 seed 数 | — | — |

**不过就不许调 `nu`/`mu`/`ridge`/换特征重跑**；要扫参只能在看结果之前声明范围与选择规则。

## 六、S3 闭环（GPU，臂表由 S2 签字后才写）

三臂等预算：`koopman_mpc` / 等代价随机分配 / 同代价最优固定日程；**≥3 个 seed**（2 seed 分不开臂）。
主量：late 窗口 `y` 的轨迹均值，按 `(item_id, seed)` 配对 bootstrap 10000 次。
`koopman_mpc − 固定日程` 的 95% CI 为正 → RQ3 正面；CI 含 0 但对 zero-control 显著为正 →
干净负结果（"闭环额外价值为零"），同样可发。**不预测方向。**

## 七、死亡条件（写死）

以下任一成立即**关闭本线**，转"安全"叙事（2026-09-08 用户给的备选路径），
且**不第二次修同一个仪器**：

1. S0 两轮都选不出满足 G-S0-2 的报告 judge；
2. S1 的 `u` 均值合格、但 `y` 有任一轮跨轨迹 `sd = 0`；
3. S2 的 `B` 的 CI 含 0，**且** S1a 的 MDE 小于上游报的效应量级（即这个设计根本分不开）。

## 八、工程与环境约束

- **vLLM 走独立 env**（用户裁决：取 GPU-hour 少的方案）。新建 `/scratch/hcao2/envs/sequor_vllm`，
  **不动共享的 `persona_drift_pilot`**——2026-09-08 死掉两个作业的根因就在那个 env 的并发 pip 竞态。
  生成走本地 OpenAI 兼容 server（上游 `conversation.py` 的做法），**并发跨轨迹**而非跨轮。
- 新模块 `sequor_bank.py` / `sequor_constraint_judge.py` / `sequor_trajectory.py`，
  行 schema 照 `ergo_math_trajectory.py:198`；`control.py` / `controller_cli.py` / `modeling/evaluate.py` 不动。
- 每个 run 目录写 `run_summary.json`（`docs/LEDGER.md` §4 的 T-2 schema），额外带
  `mode: debug|canonical|analysis`——**debug 产物永不作为证据**。
- S3 的 in-loop judge 与 agent **同一个服务模型**（自判 = 选择信号），报告口径另跑独立重判：
  本线因此**不需要** `.claude/global.md` 那条只覆盖 `defense` 的具名例外。

## 九、运行时估计（假设写明；第一个动作是把它测出来）

假设：A100×1、Qwen3-4B、vLLM、并发 40–80、回复约 600 token（上游 gold 集中位 420 词）、
聚合吞吐 **500–1500 output tok/s**（单流 8B 约 142 tok/s 的并发外推，**未实测**）、开自动前缀缓存。

| 阶段 | 规模 | 估计 |
|---|---|---|
| S1a + S1b | 160 轨迹 × T=20 = 3200 行 | 生成 **20–60 min**；离线判分（3 次/轮，共 9600 次短调用）**15–45 min** |
| S3 | 3 臂 × 3 seed × 40 item × T=20 = 360 轨迹 | **5–8 h**（含 in-loop 判分） |
| **全线到 RQ3 有答案** | | **≈ 8–12 GPU-h**（HF 逐轨迹方案是 ~100 GPU-h） |

**估不准的量**：Qwen3-4B 在这个题面下的实际回复长度（上游未报生成超参，只说 default）与并发下的聚合吞吐。
**所以第一个动作是 2 条轨迹的 smoke test**（基建，不计入仪器配额），把上表从猜变成测，再谈提交。

## 十、分工

| 谁 | 干什么 |
|---|---|
| Opus 5 | `y` 的聚合口径与 k；S0 阈值的最终确认；S1 的 T/N/配对设计签字；S2 阈值（跑前定、跑后不改）；S3 臂表与 seed 数 |
| Sonnet 5 | vendor + PROVENANCE；三个新模块 + 单测；独立 vLLM env；sbatch 用真实 argparse 干跑；smoke test |
| 脚本 / Haiku 4.5 | S0 的 gold 统计与 κ 表；行数与 `item_id` 集合核对；`docs/LEDGER.md` §3 回填 |
| 新会话冷审（Opus） | 只读 handoff 与产物 JSON，独立重算 S2 每道闸门 |
