# 多轮可靠性侵蚀（ERGO / sharded-instruction）任务能否构成 Koopman 闭环（前置条件核对）

**写于 2026-09-06。** 这是一份**分析文档，不是实验记录**——除了明确标注出处的数字/文献引用之外
没有跑任何实验、没有写任何采集代码。目的和
[`SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md`](SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md) 完全一样：
在往这条新候选线投入 GPU 之前，把五个前置条件 + "平凡最优解"陷阱一次性核对清楚。

依据的既有结果/文献：

- `../task/KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md` 第五节——这条候选任务最初被提出的地方
  （"同一现象、不同执行器"）。**本文档第 3 节会指出该节一条具体判断已经过时，需要更正**。
- Laban et al. 2025, *LLMs Get Lost In Multi-Turn Conversation*（arXiv:2505.06120，
  Microsoft，CC BY 4.0，代码/数据集已开源：`github.com/microsoft/lost_in_conversation`，
  `huggingface.co/datasets/Microsoft/lost_in_conversation`）——"sharded instruction"多轮
  退化基准，本文档称之为 **Laban 基准**。
- ERGO 论文（arXiv:2510.14077，2nd Workshop on Uncertainty Aware NLP 2025）——"entropy-guided
  resetting"，直接建在 Laban 基准之上。
- 本项目 `docs/experiments/koopman_defense_pilot.md`（Phase A 执行器权威检验的方法模板）、
  `docs/task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md`（第七节验证顺序模板）、
  `docs/experiments/mc_sycophancy_screening_pilot.md`（这次踩坑的直接教训来源）。
- 本文档的文献细节均来自 arXiv HTML 全文抓取（WebFetch），**不是从摘要推断**，但仍是二手转述，
  实现前应对照原文/开源代码复核一次（尤其是评分脚本的确切调用方式）。

## 0. 一句话版本

Laban 基准把"任务指令拆成多个 shard、每轮只放出一个"这件事,和本项目"逐轮升级反驳/攻击"是
同一种多轮退化范式,而且**执行器（entropy 触发的重置）已经在外部论文里被证明有效**——这一点
比 sycophancy 线更强,因为 sycophancy 依赖的是"提醒对反驳压力有效"这个跨执行器/跨模型的类推,
而这里 ERGO 论文测的重置**就是**我们想用的同一个执行器形状（`ThresholdController` 的字面
类比）。但**读出结构和本项目现有管线不匹配**：Laban/ERGO 的评分只在对话结束时算一次
（"一次性通过/失败"),不是本项目"每轮都有一个 y_t"的设计,这是比 sycophancy 更根本的一个
结构性问题,需要在动手前做一次设计决策,不能直接套用 `trajectory_runner.py`。

## 1. 五个前置条件核对（初步、基于文献推断，均未在本项目管线上实测）

| # | 前置条件 | 状态 | 依据 |
|---|---|---|---|
| 1 | 惯性（跨轮记忆） | **未知，需要用 entropy 信号自己测一次** | Laban 论文没有报告逐轮相关性；ERGO 的触发逻辑（"熵较上一轮跳变超过阈值 τ 才重置"）隐含假设熵本身逐轮有意义地变化，但没有报告 lag-1 自相关系数。**不能照抄 sycophancy/防御线"已确认"的结论——这是全新维度，需要独立验证** |
| 2 | 执行器权威 | **文献里已证实（外部数据），本项目管线上未测** | ERGO 论文：相对 SHARDED（无干预）基线，平均性能 +56.6%，Aptitude(A⁹⁰) +24.7%，Unreliability(U¹⁰₉₀) −35.3%（5 个模型：Phi-4、Llama 3.1-8B、GPT-4o、GPT-4.1、GPT-4o-mini，每任务 100-325 条 prompt × 3 次独立模拟）。**但和 sycophancy 线学到的教训一致**：别人的模型/管线上有效不能直接当保证，本项目自己的执行器实现（未必是"熵触发的 LLM 改写"这么复杂，见第 4 节）必须重新测一次 |
| 3 | 读出可辨识性 | **看用哪个读出——entropy 大概率优于 sycophancy，task-success 大概率劣于 sycophancy** | ERGO 自己用的熵信号 `H̄=(1/n)Σ Hᵢ` 是连续实数，不是离散标签——结构上天然避开了 sycophancy 撞到的"三取值、84.5% 饱和"那堵墙。但 Laban 基准的**任务正确性**只在对话结束时判一次（见第 3 节），不是逐轮读出,不能直接套用 `y_consistency`/`y_safety` 那种"每轮一个数"的设计 |
| 4 | rollout horizon | **结构上比 sycophancy 宽松，但需要选任务/shard 数** | Laban 基准的 shard 数量是可变的（论文 6.3 节的"渐进分片"实验测了 2-8 个 shard），不像 SYCON-Bench 那样被写死在 T=5——可以选 shard 数更多的任务/设置来给 `lag=3` 留出真正的多步 rollout,这是相对 sycophancy 的一个结构性改善 |
| 5 | 数据量 | **好，且是真开源数据** | 六个任务（code/SQL/actions/math/data-to-text/summary）各来自成熟公开基准（HumanEval+LiveCodeBench、Spider、BFCL、GSM8K、ToTTo、Summary-of-a-Haystack），每任务 100-325 条 prompt，CC BY 4.0，代码和数据都已在 GitHub/HuggingFace 开源，vendor 成本和 SYCON-Bench/MMLU 是同一量级 |

## 2. 和本项目现有假设不一致的地方（如实标注，这是本文档最重要的发现）

### 2.1 "从没跟固定周期基线比过"这条判断已经过时，需要更正

`KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md` 第五节写的是"ERGO 目前只跟无干预基线比,还没
跟'固定周期重置'比过——正是本项目已经走过的路,可直接贡献新知识的空档"。**这次抓取 ERGO 全文
后确认这个判断不成立**：ERGO 论文本身就有 "random resets"、"fixed interval (每 5 个 shard)
resets" 这两个基线,SNOWBALL/RECAP（Laban 论文自己提的两种非重置式缓解）也在对比里。也就是说
——**"周期性 vs. 自适应对比"这个具体空档，外部论文自己已经填过了，不是可以直接拿来贡献新知识
的地方**。这条判断需要在 `KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md` 里更正（本文档不改那份
文件，留给协调会话决定怎么改）。

### 2.2 那么这条线还剩下什么可以贡献的东西？

外部论文做的是**臂间经验对比**（比谁的最终分数高）,没有人在这个现象上做**动力系统辨识**——
没有人拟合过 A/B/C 矩阵、没有人算过可控性 Gramian、没有人给出"熵的动力学是均值回归还是接近
临界"这种结构性诊断。这才是本项目方法论（`KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md` 第三节
"Koopman 为什么 work"那套机制分解）能加的增量——**贡献点从"我们发现了一个新的干预有效"
（已经不成立）收窄成"我们用同一套辨识+可控性分析方法看这个已知有效的干预，机制上是不是同一种
均值回归动力学，还是不同的动力学形状"**,和第五节里"hallucination snowball 反例场景"是同一
类贡献（诊断动力学形状，不是发现新干预）,只是这里执行器已知有效，重点是辨识而非发现。

### 2.3 读出结构不匹配：任务分数只在对话结束时判一次

这是比 sycophancy 更根本的问题。sycophancy/防御线的 `y_t` 都是"每轮独立可判分"（每轮都是一个
完整问题/独立安全判断）；Laban 基准的评分逻辑是"模型判断自己是否已经在给出最终答案尝试,
是则终止对话并打分,否则继续"——**没有逐轮/逐 shard 的中间分数**,六个任务的评分器还高度异构
（代码执行、SQL 执行、API 语义匹配、数学精确匹配、BLEU、LLM-judge 联合分）。

**两条可选设计路径，本文档不替之后的实现决策拍板，只把利弊列清楚**：

- **路径 A（熵作为 `y_t`，推荐用于第一步的执行器权威检验）**：直接复用 ERGO 已经算好的
  逐轮熵信号 `H̄` 作为 Koopman 状态/读出,完全避开"给六个任务分别写评分器"这个工程量,且熵天生
  连续、不撞饱和墙。代价：熵本身不是我们真正关心的下游目标（任务是否做对），只是一个和它相关的
  代理指标，最终还是要在闭环验证阶段（对应防御线 Phase E/F）汇报任务成功率作为"这个代理指标
  管不管用"的验证,不能只看熵本身降没降。
- **路径 B（自建逐轮部分打分器，更贴近本项目 sycophancy/防御线的既有设计）**：只选客观、自动化
  评分成本低的 1-2 个任务（数学/GSM8K 精确数字匹配、SQL 执行匹配，两者的评分逻辑本来就已经在
  Microsoft 开源代码里,照抄即可,不需要自己写）,在每个 shard 揭示后都问模型"给出当前最佳答案
  尝试"并现场打分,构造出和 `y_consistency`/`y_safety` 同构的"每轮一个数"序列。跳过 code/actions
  （需要执行沙箱,工程量更大）和 data-to-text/summary（LLM-judge,噪声特性接近本项目已经在
  sycophancy/防御线上吃过苦头的那种连续 rubric）。

**建议**：第一步（执行器权威最小验证）用路径 A（零额外工程,ERGO 代码本来就算熵）;只有权威
确认之后,再决定要不要为路径 B 投入评分器工程——完全镜像 sycophancy 线"先测权威,再决定要不要
建立完整 readout 管线"这个（这次才真正吸取到的）教训。

## 3. 一个必须提前想清楚的设计陷阱：平凡最优解（类比 sycophancy 第 4 节）

如果"重置"本身没有代价，那"每轮都重置"大概率是平凡最优解（不断清空历史、只保留最新一个 shard
的改写版本，等价于把多轮任务强行拆成一堆几乎独立的单轮任务）。这**和 sycophancy 的"永不改变
立场"是同一形状的陷阱**，但这里更容易量化代价，不需要单独构造 benign 对照数据集：

1. **重置本身有直接可测的代价**：ERGO 的"adaptive prompt consolidation"是拿一次额外的 LLM
   调用去改写整个历史（不是简单截断），每次重置= 一次额外前向 + token 开销，天然可以用本项目
   `BudgetLimitedController`/Phase J 的预算约束框架建模（"每条对话最多重置几次"），不需要像
   sycophancy 那样另外造一批"用户反驳其实是对的"数据。
2. 但仍需要确认**过度重置是否会丢失跨 shard 才需要的上下文**（比如某个 shard 里的约束要到
   后面才用得上，提前改写可能把它改丢）——这是"重置代价"里比 token 开销更隐蔽的一种，Laban
   论文的"gradual sharding"（2-8 shard）实验可能已经间接触及，需要细读原文而非只读摘要确认。

## 4. 建议的执行顺序（严格把"最小执行器权威检验"放在第一步，不是最后一步）

这是本文档相对 sycophancy 线两份文档的**关键流程差异**：sycophancy 线的执行顺序原本把执行器
权威检验放在第 2 步（数据核实之后、扩样本之前），但实际执行中批准提醒设计（channel A 文案+
插入时机）走了两轮才做完权威检验，且是在核心构建块（bank/judge/trajectory/analysis）已经
全部实现之后才跑的——GPU 成本不算失控,但"设计-实现-测权威"这个顺序本身值得倒过来。这次直接
把顺序倒过来：

1. **（零 GPU 成本）vendor 数据 + 核实评分脚本**：clone `microsoft/lost_in_conversation`，
   选 1 个客观评分任务（推荐 GSM8K 数学分片，评分逻辑最简单、和 MMLU sycophancy 的"数值/字母
   精确匹配"同一量级），核实 shard 数量分布、原始评分脚本的确切调用方式、CC BY 4.0 许可细节，
   写入 `resources/PROVENANCE.md`（沿用既有 vendoring 先例）。**不写 bank/judge/trajectory
   代码**。
2. **（最小 GPU 成本，本文档第 4 节路径 A）执行器权威最小验证**：~15-20 条对话 × 2 seed，
   `--controller zero_control` vs 一个**结构简化版**的重置执行器（不追求逐字复现 ERGO 的
   LLM-改写式 consolidation,先用最简单的"截断到原始完整指令+当前 shard"作为重置动作,类比
   `FixedScheduleController`/`ThresholdController` 已有的通用控制器,能否验证会不会移动逐轮
   熵和/或最终任务成功率。**只有这一步显示出可测的效应,才进入第 3 步**——这正是这次 sycophancy
   Phase A 应该更早做、而没有更早做的那一步。
3. **惯性检验**：用同一批零控制轨迹的熵序列做 `new_q3_autocorrelation` 同款 lag-1 自相关
   检验，确认前置条件 1。
4. 1-3 都过 → 决定是否要为路径 B（任务级逐轮部分打分）投入工程,再进入开环激励采集/建模阶段
   （镜像 `ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md` 第七节 Phase B 起的流程）。
5. 任一步不过 → 回到 `KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md` 第五节，重议
   context equilibria（当前排序里机制最相似但尚未评估执行器权威的候选）或其他任务。

## 5. 尚未验证的假设清单（如实标注，别当结论用）

1. **熵是否真的呈现均值回归/惯性动力学**：完全未知，ERGO 论文没有报告这个数字，是本文档
   排在第一位、必须独立验证的假设。
2. **重置在本项目自己的模型（大概率仍是 Qwen3-4B，待确认）/简化版重置动作上是否仍有权威**：
   ERGO 测的 5 个模型不含 Qwen 系列，且用的是更复杂的"LLM 改写式"重置，本文档建议先测的
   "简单截断式"重置权威可能更弱，需要实测，不能假定和 ERGO 报告的 56.6% 效应量同等强度。
3. **shard 数量与本项目 `lag=3` 的兼容性**：论文里明确的是"2-8 shard"这个范围来自一个单独的
   "渐进分片"消融实验，不确定六个主任务默认配置下典型 shard 数是多少——需要读 GitHub 仓库的
   任务定义文件而非只读论文正文确认，这是第 4 节第 1 步要做的事。
4. **"周期性基线已被外部论文测过"这条本身是否会削弱论文里这条线的可写性**：如果第 4 节的
   执行顺序走到底、且验证顺利，产出的贡献点是"动力系统辨识/机制诊断"而不是"新发现的干预"，
   这个框定需要和 `docs/article/PAPER_EXECUTION_PLAN.md` 的整体叙事弧（见该文档 §1.4，
   sycophancy 线目前的定位是 RQ5 可选跨任务迁移证据）对齐，是否值得占用页面预算是编排层面的
   决定，不是本文档能回答的。
