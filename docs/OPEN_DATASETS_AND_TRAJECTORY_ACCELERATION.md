# 开源数据集候选与长文本轨迹采集加速：分析记录（草案 v0.1，2026-08-30）

本文件是 `DATA_SOURCES.md`、`ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md`、
`SCRIPTED_USER_TURNS_FEASIBILITY.md` 之后的第四份同类分析记录，回答两个问题：①除了已调研过的
候选，还有哪些开源数据集可能适用于本项目；②如果轨迹变长（比如几百上千轮），采集本身在工程上
怎么加速。**本文件仅做分析记录，不修改 `persona_drift_control/` 已有代码。**

## 1. 结论摘要

- **没有一个新调研到的开源数据集能替代主协议的开环激励采集**——跟此前 Alhafni 那次的结论
  是同一个模式：都缺"每轮独立随机施加的控制输入"这个 Koopman 辨识 B 矩阵的必要前提，价值都
  在参照对比或零件复用，不是直接可用的主数据源。
- 其中最值得记录的一个意外发现：`DATA_COLLECTION_PROTOCOL.md`/`method/trajectory_generation.md`
  里"探针分叉、不污染主线"的做法命名叫"ContextEcho 做法"，这次查到了具体出处——
  [ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions](https://arxiv.org/abs/2605.24279)
  （GitHub: [`Accenture/ContextEcho`](https://github.com/Accenture/ContextEcho)），且这篇论文
  公开发布了真实的、几千轮量级的长会话数据，可以作为"长轨迹 + 能力维度"两个方向的参照数据。
- **长文本轨迹加速**：在我们自己的 `chat_model.py`/`selfchat.py` 里找到两个具体的、和"轨迹
  多长"直接相关的工程浪费（跨轮无 KV cache 复用、探针间共享前缀未复用），换成带前缀缓存的
  推理引擎（vLLM/SGLang）能同时解决，且不改变协议/科学结论，是风险最低的一类加速手段；此外
  横向并行（Slurm array job）是与之正交的另一条杠杆。若真要做千轮级长轨迹，还有两个新增取舍
  （稀疏探针、上下文压缩不是免费加速）在当前16轮规模下不需要考虑。

## 2. 开源数据集候选

### 2.1 ContextEcho（真实长 agentic-coding 会话，已公开）— 候选（仅参照）

- 数据：3 段脱敏后的真实 Claude Code 会话（445–1,242 用户轮，**3,746–9,643 agent 步**），
  用 25 探针身份组合、同一套 snapshot-then-probe 分叉协议，在 **23 个前沿模型（10 家机构）**
  上测出的 ~42K 条逐格 JSON 响应，作者已把"会话原文 + 测量结果"分开发布。
- **对本项目的价值**：
  1. 直接触达本次讨论最早提出的问题——"模型能力越强，人格漂移是否减弱"：这份数据已经
     横向测了23个模型，可以先看现成结果的方向和量级，不必自己从头跑能力阶梯实验。
  2. 千轮级长会话下漂移的量级/模式，可以作为"我们要不要把16轮的协议扩展到更长"这个决策的
     先验参照。
- **局限**（决定了它不能替代主协议）：领域是 agentic coding（工具调用、调试），探针语义
  （"助手身份"）和我们的 `character_traits`/`language_constraints` 不通用；同样没有设计过的
  随机输入序列，无法用来辨识 B 矩阵；三段会话样本量小（n=3），不适合做统计推断，只适合定性
  参照。

### 2.2 多会话人格对话数据集（MSC / LoCoMo / Conversation Chronicles / GapChat / SHARE / ImplexConv）— 候选（脚本来源，非主数据）

这类数据集是为"长期人格一致性/记忆"研究专门构造的，人工撰写或经过质量把关的多会话对话：

| 数据集 | 特点 |
|---|---|
| MSC (Multi-Session Chat) | 首个人格感知的多会话对话数据集，人工撰写 persona 信息 |
| LoCoMo | 多模态长期对话（long conversation memory） |
| Conversation Chronicles | 机器生成，100万条多会话对话，含说话人关系与时间间隔 |
| GapChat | 含会话间时间间隔标注 |
| SHARE | 从电影剧本构造的共享记忆长期对话 |
| ImplexConv | 2,500 个样本，每个约100个会话，用于长期隐式推理研究 |

- **对本项目的意外价值**：这类数据可以直接解决 `SCRIPTED_USER_TURNS_FEASIBILITY.md` 里
  scripted-user 方案失败的具体问题——那次失败的根因是自己用 LLM 自聊离线生成脚本，退化成
  逐字重复或语气失控（`drift_confirmation_pilot.md`"历史"一节记录）；这类数据集是人工撰写
  或已经过质量把关的真实多轮对话，如果以后还想做脚本化用户，从这里挑选/改写现成对话，可能
  比自己用 LLM 生成更不容易碰到同样的退化问题。
- **仍然不能替代主协议**：即使脚本来源换成这些数据，`SCRIPTED_USER_TURNS_FEASIBILITY.md`
  第1节"边际收益约17%"和第6节"打断反馈闭环是真实的简化"这两条结论不变，只是解决了"脚本
  从哪来"这一个子问题，没有解决"值不值得做"这个更大的问题。

### 2.3 likenneth/persona_drift 官方数据 — 已在 `DATA_SOURCES.md` 记录，本次未发现新信息

已作为候选参照数据记录在案（`DATA_SOURCES.md` 3.1），本次调研未发现需要更新的新信息，仅在
此处标注一下与 2.1 的关系：两者都是"参照对比用的漂移数据"，likenneth 数据是短程（8轮量级）
纯聊天场景，ContextEcho 是长程（千轮量级）agentic coding 场景，二者互补，覆盖了"短/长"和
"聊天/agentic"两个维度的对角。

### 2.4 LMSYS-Chat-1M / WildChat — 候选（仅话题池/统计参照）

大规模真实用户与 LLM 的对话日志，规模大、免费，但没有受控的人格设定，也没有随机激励，价值
仅限于两处：`DATA_SOURCES.md` 3.2 里还悬而未决的"模拟用户话题池来源"问题（可以从真实对话
里提取话题分布，替代现在写死的12个候选池），或者统计"真实使用场景里人格漂移的自然发生率"
作为背景参照，都不是可以直接接入协议的数据。

### 2.5 ANCHOR（Best Friends, Not Forever, arXiv 2607.28818）— 候选，数据是否公开未确认

2,008 段对话、27 个人格、9 种交互 schedule、3 种记忆设置、4 个模型，用 Identity
Probe（102题问卷+逐轮判断）和 Trajectory Probe（35个对话库、110个反事实校准问题）分别测
"人格坍塌"和"行为漂移"两种失效模式，设计思路和我们的协议高度相似（分离测量身份/轨迹两个
维度）。但这次调研没能确认数据集/代码是否公开发布，**在确认之前不建议依赖它**，仅记录论文
本身的方法论参考价值（"分两个探针分别测身份保持与轨迹一致性"这个设计思路本身值得读一下原文）。

## 3. 长文本轨迹采集加速

### 3.1 现有代码里两个和"轨迹多长"直接相关的浪费点

核实自 `persona_drift_control/src/persona_drift/chat_model.py` 与 `selfchat.py`：

1. **每轮都从头重新编码整段历史，没有跨轮 KV cache 复用**：`ChatModel.generate()` 每次调用
   对传入的完整 `messages` 列表重新 `apply_chat_template` + tokenize + 前向，不保留、不复用
   上一轮已经算过的 KV cache。轨迹越长，越晚的轮次要重新编码的前缀越长，这个浪费随轮数增长
   （不是常数开销）——16轮规模下还不明显，若真要做 ContextEcho 那种千轮量级，这会成为主导
   成本项，比"要不要脚本化用户"这类边际优化量级大得多。
2. **`probe_repeats=4` 次重复共享同一前缀，却完全串行、各自重新计算**：`selfchat.py::
   run_trajectory` 里探针循环对 `agent_history + probe_question` 这段相同前缀调用了4次
   独立的 `generate()`，前缀部分的前向计算被重复了4遍，只有后面采样出的续写不同。

### 3.2 解法：带前缀缓存的推理引擎（不改变协议本身）

把 `ChatModel` 底层从裸 `transformers.AutoModelForCausalLM.generate()` 换成 vLLM/SGLang
这类支持 continuous batching + 前缀缓存（SGLang 的 radix attention 或 vLLM 的
`--enable-prefix-caching`）的推理引擎，能同时解决 3.1 的两个问题：跨轮场景下，未变化的
历史前缀的 KV cache 被引擎自动复用；探针场景下，4次重复共享的前缀只需算一次，只有各自的
采样续写单独算。**这类改动完全不涉及协议/测量方法本身，只是推理层的实现替换**，风险明显
低于此前讨论过的"改测量方式"（隐状态/免探针/scripted-user）那几类方案，是这次调研里唯一
一个"纯工程收益、无科学有效性风险"的选项。

### 3.3 正交的另一条杠杆：横向并行

现在 `screening.py` 是单进程严格串行跑完 `prompts × seeds × conditions` 的全部组合。由于
trajectory 之间本来就相互独立（各自维护自己的 `agent_history`/`user_history`，互不共享状态），
可以拆成多个 Slurm array job 并行提交，直接用更多 GPU 换墙钟时间，和 3.2 的单卡吞吐优化正交，
可以叠加。

### 3.4 若真做千轮级长轨迹，还有两个当前16轮规模下不需要考虑的取舍

- **要不要稀疏探针**：每轮都 fork 一次探针，在千轮量级下探针成本会主导总成本；可以考虑隔
  N 轮探测一次，用时间分辨率换成本。代价是可能测不到快速动态——这和我们现在16轮"每轮都要
  看"的设计目标（关注 turn-to-turn 的惯性）相反，是否值得取决于研究的是快漂移还是慢漂移，
  需要先明确这一点再决定探针密度，不能默认稀疏化不影响结论。
- **上下文压缩不是免费的加速手段**：ContextEcho 自己的发现是"in-session context compaction
  fails to restore the trained register"——如果为了省成本对长历史做摘要/压缩，压缩动作
  本身会改变被测系统的状态（这正是 ContextEcho 测出来的一个失效模式，不是假设），压缩不能
  当成纯粹的工程优化来用，必须作为实验设计的一部分显式声明和控制，否则会污染漂移测量本身。

## 4. 与现有文档的关系

- 与 `DATA_SOURCES.md` 的关系：第2节的数据集候选是对该文档"评价用外部数据"部分的补充，
  未来若正式采纳某一项，应该同步更新 `DATA_SOURCES.md` 的候选表格，而不是只留在本文件里。
- 与 `SCRIPTED_USER_TURNS_FEASIBILITY.md` 的关系：2.2 节的多会话数据集为该文档"脚本内容
  从哪来"的问题提供了一个新的候选来源，但不改变该文档已有的成本/风险结论。
- 与 `method/trajectory_generation.md` 的关系：3.1 节的两个浪费点是对该文档"随机性控制"/
  "代码索引"部分的补充说明，该文档本身不需要因此改动（描述的是现状实现，不是待办）。
