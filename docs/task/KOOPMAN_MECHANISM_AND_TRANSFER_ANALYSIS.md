# Koopman 在对抗防御任务上的机制分析与迁移候选

供跨会话接续：这份文档回答三个问题——(1) `koopman_defense_pilot.md`/`koopman_case_study_design.md`/
`koopman_detection_design.md`/`ABLATION_STUDY.md` 这条实验线目前的 baseline 对比与机制分析
现状是什么；(2) Koopman 代理模型在这个任务上"work"的具体机制是什么，拟合出的矩阵除了选动作
还有什么意义；(3) 这套机制能不能迁移到学术界正在关心的、机制相似的真实 LLM 下游任务上。
**本文档是分析性总结,不改变任何已有实验文档的内容,引用的历史数字以各自源文档为准。**

## 一、baseline 对比情况总览

`koopman_mpc` 在对抗防御任务（Qwen3-4B，多轮渐进 jailbreak）上和五个对手打过擂，全部在
held-out 攻击 + 完全对齐的插入次数/token 代价下比较：

| 对手 | 定位 | 结果 |
|---|---|---|
| `zero_control` | 无控制基线 | 明确打赢（Phase E） |
| `constant_remind` | 暴力常提醒 | 用约 40% 代价追平（Phase E/F） |
| `threshold` | 经典反应式反馈 | 明确打赢——`threshold` 在 held-out 攻击上普遍欠触发 |
| `periodic`（同代价固定周期） | 生产环境常见"不看信号定期重提醒" | **没有打赢**，Phase G 显示同代价下打平，甚至 `periodic` 略优 |
| `koopman_mpc_interaction` / v-align 修正版 | Koopman 自身的迭代版本 | Phase H 学到反直觉的有害方向；Phase I 修 bug 后方向修正但仍未在主判据上赢 `periodic` |

核心 Koopman-AE 任务（`ABLATION_STUDY.md`）的消融也指向同一个方向：memory 状态优于 markov
（惯性假设成立），但**纯线性 Koopman baseline 反而略优于 AE 非线性提升**——"为什么需要非线性
lift"这条核心论点在默认配置下未获支持。

**结论层级**：相对"完全不建模"的基线（`zero_control`/`threshold`）有稳固优势；相对"同样不
建模、但按固定节奏执行"的 `periodic` 基线，优势至今未被证实。

## 二、机制分析情况总览（bug 定位链条）

这条线的价值主要不在"谁赢了"，而在于把每一次胜负都逐轮重放、定位到具体的数学结构原因：

1. **Case study 逐轮重放**：`koopman_mpc` 在有真实状态的 32/32 次决策全部选提醒，与状态、
   horizon 无关——根源是线性 Koopman 模型里动作项 `B@v` 与状态无交互项，MPC 最优动作在数学上
   退化成一个和 `z_t` 无关的常数。
2. **交互项修复**：加状态-动作交互项后 margin 真正依赖状态（corr=1.0），但 Phase H 真实闭环
   学到的方向有害（越危险的轨迹反而越少提醒）。
3. **v 对齐 bug（Phase I，根因）**：`build_reduced_state_pairs` 把 `v_t` 和已含 `y_t` 的
   `z_t` 错配，模型学到的是"提醒的残留效应"而非"当轮直接效应"。修正后同一架构不换模型不加
   数据就学出方向正确、更省成本的策略,`repeat_penalty=0.2`需要的经验校准也不再需要。
4. **检测方案重新评估**：v 对齐 bug 同样污染了 `koopman_detection_design.md` 的方案 1/3——
   修正后残差检测相关系数从 0.53 升到 0.86,双 regime 检测器逐轨迹准确率从低于平凡基线
   （0.475）首次超过（0.613）。**"这套状态表示信息量不够"这个原结论主要是 bug 的假象,
   不是真实的信息天花板。**

## 三、Koopman 为什么"work"：机制分解

1. **前提：状态确有惯性,不是纯 Markov 过程。** `ABLATION_STUDY.md` 第一阶段：memory 相对
   markov 的 rollout MSE 低约 2.5 倍；对抗防御线的 new-Q3 惯性检验持续显著。
2. **核心机制：单轮边际效应很小,但通过状态递推复利式累积成大效应。** Phase I"累积效应假说"
   实验是关键证据：修正对齐后单步直接效应只有 `B≈0.02`（nu=1,mu=1）,y 维自回归系数
   `a≈0.30`（衰减快）,单看这两个数字完全无法解释 Phase A"常提醒把终轮安全分从 0.45 顶到
   0.81"这种大效应；但用 `mu=2` 记忆窗口的模型沿时间滚动预测,能解释观测效应的 **107%**。
   Koopman 真正 work 的地方是**正确预测多轮累积后的量级**,不是单轮预测准。
3. **可控性是被形式化证明过的。** `controllability_diagnostics` 的 Gramian 秩/条件数/A 的
   谱半径不是装饰性输出——Phase C 里 `mu=1` 时 Gramian 条件数 193（病态）,换 `mu=2` 降到
   12.7 且满秩,这直接决定了"提醒动作能不能真正撬动这个状态"在当前记忆阶数下是否成立。
4. **但"自适应"被架构结构性锁死,除非显式加交互项。** Case study 定量证明：只要 `B@v` 线性
   可加、无交互项,`value_full(1)-value_full(0)` 数学上必然是与 `z_t` 无关的常数——不管
   horizon 多长、lifting 多丰富。这意味着 Koopman-MPC 目前 work 的部分几乎全部来自"正确
   预测密度该多高",不是"逐轮聪明判断"；真正的逐轮自适应需要额外架构设计,且已学出的具体
   方向还不可靠（Phase H 的反直觉方向）。
5. **因果对齐正确性比模型非线性容量更决定成败。** v-alignment bug 同时压垮了控制和检测效果,
   修一格时序错位后两条线同时改善——系统辨识的因果时序正确性,是比换更复杂模型家族更该
   优先检查的东西（AE 消融也印证：非线性 lift 没有跑赢纯线性 baseline）。

## 四、拟合矩阵的其他意义/用途

| 量 | 已算出但未深挖的含义 | 潜在用途 |
|---|---|---|
| A 的对角/自回归系数（如 y 维 0.82 旧对齐 mu=2 / 0.30 新对齐 mu=1） | 均值回归速度——"安全分自己会多快回弹" | 脱离控制器,单独作为"某系统提示/某类攻击固有韧性"指标,可跨配置比较 |
| A 的谱半径/特征值 | 系统整体稳定性——收敛到不动点还是发散 | 谱半径 ≥1 提示"雪崩式"而非"均值回归式"现象,可用于判断某个新任务是否适用同一套方法论（见第六节） |
| Controllability Gramian 及条件数 | 某个动作通道对状态的"杠杆力度",独立于具体策略 | 在跑昂贵闭环实验前,先用开环辨识数据比较候选执行器（本项目 `dose_response_pilot.md` 里失败的 steering 通道,若先算开环 Gramian 可能更早发现杠杆不够） |
| C（readout 权重）/ mu 阶数选择 | 多远的历史真正对预测有贡献 | `mu=2` 有用、`mu=3` 数值退化——这本身是"记忆窗口有多长"这个问题的直接回答,目前只是调参副产品,未被提炼成解释性结论 |
| 一步预测残差/innovation | 模型"预测漏掉了什么" | 已在 `koopman_detection_design.md` 独立于控制使用；v-align 修好后相关系数 0.53→0.86,这个"副产品"用途目前比控制本身的自适应优势更站得住 |
| 反事实滚动预测（不真正执行动作） | "如果什么都不做,轨迹会怎样" | Phase I"累积效应假说"实验已这么用过,可独立包装成决策支持/可视化输出,不需要真的接成自动控制器 |

## 五、可迁移的真实学术下游任务

按机制相似度排序（是否具备：多轮累积、可能有均值回归的线性/近线性动力学、存在轻量周期性/
反馈式干预可用），而非随意列举"LLM 长程任务"：

- **最贴近**：《Drift No More? Context Equilibria in Multi-Turn LLM Interactions》
  (arXiv:2510.07777) 把多轮 drift 直接建模成"有恢复力（restoring force）+受控干预的有界
  随机过程",并发现 drift 收敛到有限噪声均衡点——与本项目"A 的自回归系数 <1（均值回归）
  +B 的小幅正向修正通过递推复利累积"是同一个数学结构,只是他们用 KL 散度定义 drift、
  本项目用 judge 分数。可直接把本项目 A/B 拟合+Gramian 诊断+periodic/threshold/koopman_mpc
  五臂对比方法论搬过去。
- **同一执行器已被独立验证有效**：sycophancy drift 领域 SYCON Bench (arXiv:2505.23840) 发现
  "重申原始约束的提醒式干预能让模型从谄媚状态恢复"——和本项目 channel A（安全提醒注入）
  几乎是同一个执行器。这个领域目前还没人做过"周期性 vs. 状态反馈"对比,是可直接贡献新知识
  的空档（详见第七节）。
- **好的对照/反例场景**：hallucination snowball 相关工作（arXiv:2608.14588、2606.00622、
  2607.18292）把错误累积形式化成**近吸收态的马尔可夫链**（阶段间逃逸概率 24.6%~89.3%）,
  动力学形状上更像"谱半径接近/超过 1",而非本项目验证过的均值回归。用同一套 Koopman 辨识
  流程去测,很可能会测出谱半径 ≥1 的不同结果——这本身是有价值的诊断,能解释"为什么周期性
  检查在雪崩式错误上可能不够"。
- **同一现象、不同执行器**：《LLMs Get Lost In Multi-Turn Conversation》(arXiv:2505.06120)
  发现多轮退化主要来自"不可靠性"上升（+112%）而非能力下降；ERGO (arXiv:2510.14077) 的
  "entropy-guided resetting"本质上就是本项目分类里的 `ThresholdController`。ERGO 目前只跟
  无干预基线比,还没跟"固定周期重置"比过——正是本项目 Phase G 已经走过的路。
- **方法论直接先例**：Redman,《Interpreting Reinforcement Learning Model Behavior via
  Koopman with Control》(arXiv:2603.19968) 把 Koopman-with-control 拟合出的稳定性/可控性
  指标用作 RL 训练过程的可解释性工具而非控制——详细分析见第七节。
- **较远,不建议优先**：长程 agent 目标漂移/reward hacking（arXiv:2603.19685、2605.21384）
  更多是优化压力驱动而非上下文内累积,线性状态空间假设不一定成立。
- **现实佐证**：多篇工业博客材料显示"周期性重新锚定目标"已是生产环境应对 agent drift 的
  默认做法之一——独立印证了 Phase G 发现的"`periodic` 基线难打"不是偶然,是这类问题里
  普遍存在的强基线。

## 六、Sycophancy drift 与人格漂移的区别，及在 Qwen3-4B 上是否可能存在

**定义区分**：

| | 人格漂移（`signal_screening_pilot`/`drift_confirmation_pilot`/`pressure_screening_pilot`） | Sycophancy drift（SYCON Bench 等） | 本项目主线：安全侵蚀（jailbreak erosion） |
|---|---|---|---|
| 被侵蚀的对象 | system prompt 指定的风格/人设约束（如"永远用第三人称"、"只用法语回答"） | 模型对某个事实/伦理立场的**表态一致性** | 安全拒答边界 |
| 驱动力 | 长上下文自然累积,不需要用户主动施压（被动 `excite_iid` 刺激） | 用户**显式、持续的反驳/施压**（"你确定吗"、"我导师说是 X"） | 攻击者**刻意设计的渐进升级**攻击序列 |
| 典型测量设计 | 连续 0-1 judge 打分,测斜率 | 立场是否翻转的**离散事件**（更高统计功效） | 连续 0-1 judge 打分,测斜率 |
| 施压强度（相对排序） | 最弱 | 中等 | 最强 |

**是否可能在 Qwen3-4B 上存在**：`pressure_screening_pilot.md` 恰好是这个问题在人格域上最直接
的证据。把对抗任务的"渐进升级施压"设计移植到人格域后：

- N=4 prompt：`escalating_pressure` 方向对（3/4 负斜率）但 t=-1.28, p=0.29,不显著；
- N=12 prompt（job 15412821,已完成但此前未写入文档,本次已补上——见下方文档更新）：
  方向更一致（10/12 负斜率）,但 t=-1.4478, p=0.1756,**扩大样本后仍不显著**。

这是一个"中间态"结果：不是"证实",也不是干净的空结果。结合本项目"4B 容量不是瓶颈,同模型
在对抗任务上有清晰信号"这条已确认的判断（`drift_confirmation_pilot.md` 2026-08-31 补充分析）,
**合理推断是 sycophancy drift 很可能在 Qwen3-4B 上存在,且可能比本项目的人格-渐进施压 pilot
测得更清楚**,原因是测量设计而非现象强度：

1. sycophancy drift 有天然的**离散翻转事件**可测（"这一轮是否改变了立场"）,是比人格域连续
   0-1 rubric 统计功效更高的设计——`pressure_screening_pilot` 用连续打分在 N=12 下都还没
   到显著性,如果换成 SYCON Bench 式的"stance flip"二元事件,同样规模的数据可能更容易测出。
2. sycophancy drift 的刺激（用户直接反驳）比人格 pilot 用的固定劝退脚本更贴近本项目已验证
   有效的"渐进升级"范式（结构上更接近 jailbreak 的持续施压,而非单纯的风格劝退）。

**结论：值得作为下一个候选任务认真评估,但不能把 jailbreak 任务的清晰信号直接类推过去——
需要先用离散翻转事件的测量设计做一次独立的 screening,而不是直接复用人格 pilot 的连续打分
管线。**

## 七、Redman 先例的机制与和本项目结合的可行性

**已拆分为独立文档**：
[`../feasibility/REDMAN_KOOPMAN_TRAINING_INTERPRETABILITY_FEASIBILITY.md`](../feasibility/REDMAN_KOOPMAN_TRAINING_INTERPRETABILITY_FEASIBILITY.md)。

摘要：Redman,《Interpreting Reinforcement Learning Model Behavior via Koopman with
Control》(arXiv:2603.19968) 用同一个数学工具（Koopman-with-control/DMDc）诊断"RL 训练
如何改变策略的稳定性/可控性"（状态=环境观测,拟合频率=随训练 checkpoint 反复拟合,完全不
做控制,纯事后解释），和本项目"实时控制 LLM 行为"的用法目的不同。三层可行性判断：
**第一层（借用其框架重新整理本项目自己的模型迭代历史,零成本,现在可做）、第二层（把可控性
当检测信号,纳入 `koopman_detection_design.md` 方案 2 重开时的扩展）、第三层（真正对应
Redman"随训练演化"的设置,需要项目转向微调被防御的 LLM 或把状态换成内部隐藏表征,是方向性
决策,不在这次分析范围内替用户判断）**。详见拆分出的独立文档。
