# 任务选型：Koopman 闭环控制抵抗多轮 prompt 攻击（草案 v0.1，2026-08-31）

> **状态：Phase A→I 已完整收尾**（`koopman_defense_pilot.md` + `koopman_case_study_design.md`，
> 2026-09-02），2026-09-02 起 `SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md` 成为新任务线。
> **但这条任务线本身没有关闭**：Phase A→I 收尾时留下的问题是"那套评测设定里根本不存在
> 分配问题"，2026-09-03 起的 Phase J（`../experiments/budget_constrained_defense_plan.md`）
> 在预算约束（每轮最多 k=1 次提醒）的新设定下重开对照，仍然用本文件定义的任务/通道/判据。
> 本文件是该线的原始任务定义，继续有效。本文件是继
> `STOLFO_ACTIVATION_STEERING_FEASIBILITY.md`（要素移植分析）、
> `CONTROL_THEORETIC_LLM_RELATED_WORK.md`（领域调研）之后的任务选型分析第三篇，
> 也是三篇里唯一给出"推荐往下走"结论的一篇。**仅分析记录，未修改代码与协议。**

## 0. 一句话版本

良性 self-chat 没有可辨识的动力学（`experiments/drift_confirmation_pilot.md` 的空结果 +
Drift No More 的平衡点结论），有持续外部压力的对话有（NBF-LLM 的 plant）；把压力源
（多轮 jailbreak 攻击者）纳入系统作为扰动通道，用连续 steering 做软调节防御，
Koopman 结构分析恰好落在 NBF-LLM（ICML 2025）和 GenCtrl 都没占的位置上。

## 1. 任务定义（拟）

被控对象：Qwen3-4B（沿用，自身有安全对齐 = 有能力可调度）在多轮 jailbreak 攻击
（ActorAttack / Crescendo 类，攻击轮次用公开预生成序列**固定回放**）下的对话过程。
控制目标：对话不中断的前提下，用连续防御输入把每轮安全 readout 维持在阈值之上，
同时最小化累计干预代价与 helpfulness 损失。交付的结构分析：升级速率（A 的谱）、
防御成本（B 的可控性 Gramian）、可防御性（E 与 B 的相对强度）。

## 2. 为什么它修复了 pilot 卡住的三个关卡

| 关卡 | 人格漂移任务的实测 | 本任务的先验 |
|---|---|---|
| Q1 有可测演化 | 10 prompt 斜率 5:5，p=0.99（无自发漂移） | 演化由攻击者主动制造：多轮攻击机制本身就是逐轮渐进升级（NBF 图 4 屏障值逐轮恶化）；不依赖系统"自己变坏" |
| Q2 输入有效 | u_remind 对 10 个 readout 全平坦 | 安全是激活空间线性结构证据最强的行为之一（refusal 单方向文献）；沿安全/拒绝方向的 steering 效应大且被反复验证；且防御是调度已训练进去的能力，不是无中生有 |
| Q3 有惯性 | 未测出（但当时 b≈0，无信息量） | 多轮攻击 ASR 显著高于单轮这一事实本身就是跨轮惯性的实证——若逐轮独立，Crescendo 与单轮攻击等价 |

## 3. 建模形态：扰动通道的语义替换

直觉：状态的下一步走向由三股力量合成——系统自发演化、我方防御输入、攻击方推力。
把修订稿 surrogate 里 r 的通道换成扰动语义，数学几乎不动：

    ψ(z⁺) ≈ A ψ(z) + B u + E d

式中 z 为当前对话状态、z⁺ 为下一轮状态、ψ 为提升观测函数、u 为我方连续防御输入
（steering 强度等）、d 为攻击者本轮输入的编码（**记录下来的外生扰动，不由我们控制**）、
A 为自发演化、B 为防御通道、E 为攻击通道。由此结构分析获得明确含义：

- A 的谱 → 攻击压力下的升级/恢复速率（对照 Drift No More 的恢复力叙事）；
- B 的 Gramian → "把状态按在安全区内"的最小控制能量；
- ‖E‖ 与 ‖B‖ 的相对大小 → 该模型对该攻击族的**可防御性**——GenCtrl 说"可控性脆弱"
  但给不出的结构解释，正是这一项。

辨识注意：d 是对抗性序列而非随机激励，作为记录的外生输入进入回归没有问题；
u 仍按协议做独立随机电平激励。泛化到未见攻击族是 OOD 问题（A-LQR 附录 D 的
退化警告适用），实验设计里要留 held-out 攻击族。

## 4. 相对已有工作的 delta

- **NBF-LLM（ICML 2025，最近邻）**：二值过滤 query（阻断输入），黑盒 3 层 MLP 动力学
  （预测精度全文未报告），无谱/可控性分析，强阈值下 MMLU 66→46.65。我们：连续软调节
  （对话继续）、提升线性 surrogate + 辨识质量验证、结构分析、以 Pareto 曲线为卖点
  （同等 ASR 下保住更多 helpfulness）。
- **GenCtrl（既定对标）**：Monte Carlo 可达集估计，无结构模型。我们给出其"脆弱性"
  结论的结构性解释，对标关系不变、且比人格漂移版本更贴合其安全语境。
- **Drift No More**：良性设定平衡点结论与我们空结果互证；本任务是其"受迫"对偶面。
- 主论文修订稿的数学框架（三者分离、移位状态、8 节验证协议）**全部保留**，只换应用语义。

## 5. 工程复用与数据来源

- **攻击轮次固定回放**：ActorAttack 的多轮 query 序列有公开预生成数据集（SafeMTData 等），
  攻击脚本不需要根据回复自适应——确定性扰动序列是辨识实验的标准做法，可复现性好，
  且整个绕开了 scripted-user 退化问题（那次失败的根因是"自生成脚本+无反馈闭环"，
  这里的脚本是现成的、且本来就该固定）。
- **readout**：① 免费连续量——每轮回复激活在 refusal/安全方向上的投影（零额外生成成本）；
  ② 外部校验——LLM judge 安全分 1–5（NBF 用 GPT-4o；可用本地更大模型控成本）；
  ③ 二值 ASR 只做最终评价指标，不做辨识用 y。
- **状态**：Sentence-BERT 句嵌入（照抄 NBF，不自创）或隐状态（`LLM_LATENT_STATE_FEASIBILITY.md`
  的备选升级）。
- **执行器**：通道 C（安全方向 steering，强度连续）；通道 A（安全提醒）可保留作对照——
  在安全语境下 reminder 未必像人格语境那样无效，本身就是可报告的对比点。
- **代码骨架**：screening/断点续跑/表结构全部沿用，改动集中在 user_sim → 攻击序列回放、
  readout 模块、steering hook。

## 6. 风险与未验证前提

1. **Qwen3-4B 上攻击的"逐轮渐进性"未亲测**：NBF 测的是 GPT-3.5/4o/Claude/Llama。小模型
   ASR 通常更高（先验有利），但需要的形态是渐进升级而非某轮突然跳变——跳变式失守对
   连续动力学建模不友好。这是新 screening 的第一问。
2. **Q3 需在本任务 readout 上重测**：无控制攻击轨迹里 y_t 的跨轮自相关。这次是在有真实
   扰动的系统上测，与人格漂移的空结果语境不同，但仍必须实测。
3. **剂量-响应倒 U**（When is Your LLM Steerable?）：防御 steering 电平须先扫描、
   取近线性段；过强 steering 的 helpfulness 损伤要进代价项。
4. **竞争密度与时间窗**：安全赛道比人格漂移拥挤（NBF 之后必有后续）；ICLR 2027 摘要
   截止 2026-09-18，全文 09-25——三周内要完成新 screening + 至少一轮辨识实验，
   时间是最紧约束。
5. 伦理/合规：全部使用公开攻击基准与公开数据集做防御研究，与 NBF/HarmBench 等已发表
   工作同一实践标准；不产生新攻击方法。

## 7. 验证顺序（低成本优先，供排期讨论）

1. **零/小 GPU screening（对应协议第 7 节，换语境重做）**：~20 条攻击轨迹（Qwen3-4B ×
   固定回放攻击序列 × 无控制），逐轮记录连续安全 readout。判定：(a) 是否逐轮渐进升级
   （新 Q1）；(b) y_t 自相关是否非零（新 Q3 前半）；顺带在同批数据上确认 refusal 方向
   投影与 judge 分的相关性（readout 有效性）。
2. **单轮剂量-响应**：安全方向 steering 的 α 扫描（新 Q2 + 倒 U 定位）。
3. 1、2 都过 → 带 u 激励的小规模辨识采集（沿用协议表结构 + 第 3 节扰动通道），
   拟合 (A,B,E)，先看谱和 Gramian 是否给出可解释的结构，再谈 MPC 闭环。
4. 任一步不过 → 回到 `CONTROL_THEORETIC_LLM_RELATED_WORK.md` 第 3 节重议（层轴/token 轴
   是仍开放的退路）。

## 8. 与其他文档的关系

- 取代 `DATA_COLLECTION_PROTOCOL.md` 的**应用语义**（人格漂移 → 攻击防御），保留其
  方法骨架；正式修订协议前以本文件为讨论稿。
- `STOLFO_ACTIVATION_STEERING_FEASIBILITY.md` 的执行器结论（通道 C、剂量扫描）在本任务
  中原样适用；其"长度调节"备选降为次优先。
- `KV_INJECTION_MONITORING.md` 的通道 D 是本任务的备选执行器（持续输入源性质对防御
  尤其合适），暂不启用。

## 参考

NBF-LLM（Hu, Robey, Liu，ICML 2025，arXiv 2503.00187，github.com/HanjiangHu/NBF-LLM）·
GenCtrl（arXiv 2601.05637）· Drift No More（arXiv 2510.07777）· A-LQR（arXiv 2604.19018）·
When is Your LLM Steerable?（arXiv 2606.11599）· Arditi et al., Refusal in Language Models
Is Mediated by a Single Direction（arXiv 2406.11717）· ActorAttack/SafeMTData
（arXiv 2410.10700）· Crescendo（Russinovich et al., arXiv 2404.01833）.
