# Stolfo et al. ICLR 2025《Improving Instruction-Following in Language Models Through Activation Steering》：任务适配性分析（草案 v0.1，2026-08-31）

本文件是继 `LLM_LATENT_STATE_FEASIBILITY.md`、`SCRIPTED_USER_TURNS_FEASIBILITY.md`（已尝试后放弃）、
`ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md` 之后的第四份同类分析记录。触发背景：
`experiments/drift_confirmation_pilot.md`（job 15393201）给出了相当干净的空结果——10 个独立
prompt 的漂移斜率 5:5 对半开（p=0.99），u_remind 对 y_probe 及全部 9 个表层特征
（`experiments/surface_features_backfill.md`）都测不出效应。本文件回答的问题是：
[arXiv 2410.12877](https://arxiv.org/abs/2410.12877)（Stolfo, Balachandran, Yousefi, Horvitz,
Nushi；ETH+Microsoft；ICLR 2025）里的指令遵循任务，是否比人格漂移更适合我们的 Koopman 框架。
**本文件仅做分析记录，不修改 `persona_drift_control/` 已有代码与协议。**

## 1. 结论摘要

- **论文任务本身（原样照搬）不适合，且比人格漂移更不适合**：它是单轮条件生成（IFEval 查询 +
  附加约束指令 → 一次生成），没有时间轴、没有轨迹，协议第 7 节 Q3 要求的"惯性"在任务定义
  层面就不存在。这和 `ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md` 对 Alhafni 任务的结构性判定
  完全同类。
- **但 pilot 确认的失败点不是"任务缺轨迹结构"**——我们的 plant 本来就是多轮的。确认的失败是
  Q1（16 轮内无可测漂移）和 Q2（u_remind 对 10 个不同构造的 readout 一致无效应）。这篇论文
  恰好正面回应 Q2：它验证了协议第 3 节早已列为扩展通道的 **通道 C（u_steer，激活转向）** 在
  格式 / 长度 / 词汇包含三类约束上有大的、随系数连续可调的效应——正是
  `surface_features_backfill.md`"下一步"里"问题可能在输入太弱，换更强通道"的现成外部证据。
- **因此值得考虑的不是"换成他们的任务"，而是"把他们的要素移植进现有多轮 plant"**：
  (a) readout 换成 IFEval 式可程序判定的连续量（首选长度）；(b) 输入通道换成 u_steer；
  (c) 控制问题从"纠正自发漂移"改述为"存在持续违反下的调节/跟踪"。
- **长度（num_tokens / num_sents）是最有希望的单一 y**，三个独立理由：① 连续且零成本
  （`surface_features.py` 已实现）；② 是本项目自己数据里唯一测到自发动态迹象的特征族
  （zero_control 下 num_tokens 5/5 prompt 同向变长，avg_word_len 5/5 同向下降 p=0.041）；
  ③ Stolfo 证明了长度可被 steering 系数推动，且论文做了格式+长度的组合 steering。
- **最大未解风险仍是 Q3（跨轮惯性）**，换任务/换通道都不自动解决它；但它有零 GPU 的预检办法
  （见第 6 节），应该先做预检再谈任何协议修改。

## 2. 论文核实（2026-08-31，arXiv abs + HTML v2）

- 方法：对每条指令，用"带指令 / 不带指令"的成对输入在选定层 ℓ 的激活差的均值作为该指令的
  转向向量；推理时把向量加到残差流上，系数 c 经校准使激活在向量方向上的投影匹配"有指令时"
  的均值投影——所以 c 天然是个**连续标量执行器**，不只是开/关。
- 任务：IFEval 基础查询 + 三类约束：输出格式（如 JSON 合法性）、长度（如"最多三句话"）、
  特定词包含/排除。判定全部是程序化 checker（多为二值）。
- 模型：Phi-3、Gemma-2 2B/9B、Mistral 7B；指令微调模型上算的向量可迁移改善同家族 base 模型。
- 另有多指令组合 steering（同时控制格式+长度）与回复质量评估（附录 F）。
- **全部单轮**；论文不涉及多轮对话，也不研究遵循度随上下文/轮次衰减。

## 3. 为什么"任务本身"不满足 Koopman 前提

我们的 surrogate 的直觉是：对话过程本身是非线性的，但把状态提升到一组观测函数上之后，
下一步的观测值可以近似用线性关系预测——控制设计全部建立在这个线性近似上：

    ψ(z⁺) ≈ A ψ(z) + B v + E r + b

其中 z 是当前对话状态（修订稿的移位定义，只含过去的输入与输出），z⁺ 是下一轮状态，ψ 是提升
用的观测函数（字典或编码器），v = ρ(u) 是编码后的控制输入，r 是参考/设定值，A 描述无输入时
的自发演化，B 描述输入如何进入动态（可控性矩阵/Gramian 只用 B），E 是 r 进入动态的通道，
b 是常数偏置。**这个模型的前提是存在"下一步"**：辨识 A 需要跨步的自发演化，辨识 B 需要
输入对后续步的影响。Stolfo 的任务一次生成即结束，t 只有一个取值，A 无从谈起，"B" 退化成
一个静态的输入-输出映射——那是剂量-响应曲线，不是动力学。原样照搬等于把 Koopman 用在
一个没有时间的系统上。

## 4. 逐关卡对照：pilot 失败点 vs 论文能补什么

| 协议关卡 | pilot 实测 | 换成论文任务本身 | 移植论文要素进现有多轮 plant |
|---|---|---|---|
| Q1 漂移存在 | 10 prompt 斜率 5:5，p=0.99，无漂移 | 不解决（无多轮概念） | 不直接解决；但改述成调节问题后 Q1 不再必需（见下） |
| Q2 输入有效 | u_remind 对 10 个 readout 全平坦 | 论文核心贡献：steering 效应大且连续可调 | **很可能转为通过**；需在 Qwen3-4B 上复算向量验证 |
| Q3 有惯性 | 未测出（p≈0.95），且当时输入本身无效，测不出惯性不奇怪 | 不解决（单轮无此概念） | **仍是主要开放风险**，但有零 GPU 预检办法 |

两点展开：

**Q1 的改述。** 调节/跟踪问题需要的不是"系统自发变坏"，而是"不控制时 y 偏离目标"。这个条件
我们的数据已经满足：language_constraints 类（pattern prompts，本质上就是 IFEval 式格式约束）
的 y_probe 均值只有 0.56（sd 0.38）——不是"慢慢漂走"，是**从第一轮起就持续违反**。控制任务
于是变成"用最小 steering 代价把遵循度推到 r 以上并保持"，这不需要 Q1 意义上的漂移。顺带，
"Qwen3-4B 16 轮内很稳定"这个空结果在新叙事里不再是坏消息，而是把问题从"抗漂移"聚焦到
"消除稳态误差"。与 GenCtrl 的对照叙事（可达性/可控性的结构解释）也不依赖自发漂移——但主线
论文里"漂移纠正"的动机段落需要相应改写，这是叙事成本，要认账。

**Q2 的机制差异。** u_remind 是往上下文里插提示文本，模型可以（事实上看起来就是）忽略它；
u_steer 直接改残差流，绕过了"模型对 system prompt 注意力不够"这个 Harvard split-softmax
论文指出的瓶颈。通道 C 失败的先验概率明显低于通道 A 已失败十次的延长线。

## 5. 若移植，具体形态（仅分析，不实施）

- plant 不变：Qwen3-4B self-chat，16 轮，live user_sim，现有断点续跑/预筛机制全部沿用。
- y_t：主线回复的程序化约束度量，首选 num_sents/num_tokens 相对目标的偏差（连续），
  辅以格式符合率、约束词密度；与 `surface_features.py` 现有 9 特征直接兼容。
  探针分叉（每轮 4 次副本生成）可以整个省掉——readout 从"每轮 5 次生成"降到"每轮 1 次
  生成 + 免费计数"，同样本量下 GPU 成本约打 2 折，或同成本下独立 prompt 数 ×5。
- u_t：通道 C，α ∈ {−1, −0.5, 0, 0.5, 1} × α₀（协议表原样），每轮独立抽取；代价度量 ‖α‖。
- system prompt：pattern 类与 IFEval 约束同构，`prompt_bank.py` 的映射成本很低；也可直接用
  论文的 IFEval 指令模板。
- 工程增量：(a) 在 Qwen3-4B 上算各约束的转向向量（几百个成对样本的前向，单卡分钟级）；
  (b) 生成时 residual stream hook（transformers 标准做法）。与原计划的 split-softmax
  （通道 B）工程量同级，不是新增负担而是替代。

## 6. 风险与未验证假设（按重要性排序）

1. **Q3 跨轮惯性（核心）**：steering 只直接作用于当轮生成，对 t+1 的影响只能经由"被改变的
   回复文本留在上下文里"传导（自条件化/self-conditioning）。若这条反馈路径太弱，系统就是
   逐轮静态映射，每轮独立调 α 即可，Koopman/MPC 没有增量价值——协议 Q3 的判定标准本来
   就是为此设的，必须保留。**零 GPU 预检**：现有 40 条轨迹（drift_confirmation_pilot）上做
   跨轮记忆分析——控制 prompt 固定效应后，y_t（num_tokens、y_probe）对 y_{t+1} 的偏自相关
   是否显著非零。这直接回答"这个 plant 究竟有没有 A"，比任何新采集都便宜，和
   `backfill_surface_features.py`（见 `../experiments/surface_features_backfill.md`） 同一类型的回填分析。
2. **单轮向量在长上下文中的有效性**：论文的向量在短单轮输入上算得；16 轮、数千 token 的
   对话中同层同向量是否仍有同样方向的效应，需要小规模 dose-response pilot（单轮内 α 扫描，
   看 num_sents 是否单调）先行验证。
3. **二值 checker 不宜直接当 y**：JSON 合法性、词包含是 0/1，辨识用的 y 要用连续代理
   （计数偏差、符合行比例）。长度天然连续，所以排第一。
4. **模型不匹配**：论文没测 Qwen；向量不跨模型家族迁移，必须自算——方法本身模型无关，
   成本低，但"论文效应量"不能直接假设搬到 Qwen3-4B 上。
5. **质量损伤**：过强 steering 会伤文本质量（论文附录 F 自己评了）；协议里 ‖α‖ 的代价项
   和 refusal/parse 诊断要保留，必要时加一个质量 readout 做约束。

## 7. 与已有分析记录的关系

- 对"任务原样照搬"的判定与 `ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md` 相同（单轮、无轨迹、
  不满足前提）；差别在可移植要素：Alhafni 给的是 readout 抽取代码，Stolfo 给的是**执行器**
  ——而执行器恰好是 pilot 证据链指认的最薄弱环节。
- 本文件是 `surface_features_backfill.md` 综合解读（"怀疑重心推回输入强度"）的直接后续：
  它建议"换通道 B 或更强输入"，本文件论证通道 C 有更强的外部证据支撑。
- `DATA_COLLECTION_PROTOCOL.md` 不需要推倒重来：plant、激励设计、表结构、切分全部可沿用，
  改动集中在 §2 readout、§3 通道选择、§4 r 的语义和 §7 三问的 Q1 表述。

## 8. 建议的证据顺序（低成本优先，供讨论，本次未做）

1. 零 GPU：现有轨迹的跨轮记忆回填分析（风险 1 的预检）。若无记忆 → 整个动力学框架对这个
   plant 不成立，换 readout/通道也救不了，应该在这里停下来重新讨论方向。
2. 小 GPU：Qwen3-4B 上复算长度约束转向向量，单轮 α 扫描看 dose-response（风险 2、4）。
3. 若 1、2 都过：小规模多轮 lag 试验（Q3 with u_steer，沿用 screening 框架，只改通道），
   再决定是否正式修订协议进入 320 条采集。

## 参考

- Stolfo et al., Improving Instruction-Following in Language Models Through Activation Steering,
  ICLR 2025, arXiv:2410.12877。
- IFEval: Zhou et al., Instruction-Following Evaluation for Large Language Models, arXiv:2311.07911。
- 多轮遵循度衰减的外部证据（支持"多轮化后约束遵循存在动态"的方向性依据，未在本项目模型上
  验证）：Laban et al., LLMs Get Lost in Multi-Turn Conversation, arXiv:2505.06120。
