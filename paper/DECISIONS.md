# DECISIONS — 用户裁决日志

> 本文件是**最高优先级**的规则源。任何与 `~/.claude/skills/neurips-write*`、`.claude/*.md`
> 或 `WRITING_PLAN_2026-09-18.md` 冲突的地方，以这里为准。
> 新裁决追加在表后，编号连续，注明日期。不删除、不改写既有条目；改变主意就加一条新的并写明覆盖了谁。

## 裁决表

| # | 日期 | 裁决 | 覆盖了什么规则 |
|---|---|---|---|
| D1 | 2026-09-18 | 放弃 `sections/` 的负结果论文（*When Not to Close the Loop*）。旧 `contract.yaml` / `semantic.md` / `sections/*.tex` **归档不删**，落在 `_archive_2026-09-18_negative_paper/`。 | — |
| D2 | 2026-09-18 | 卖点一（预测）：线性延迟嵌入作为同事 AE-Koopman 模型的**特例**（$E_\phi = D_\psi = \mathrm{id}$），主表 Ours 行即此特例。 | — |
| D3 | 2026-09-18 | 卖点二（控制）：**正面口径**。同事 IFBench / IFEval / COLLIE 数据全部保留、暂视为可靠。Base = 不加任何控制；TMPC / RE-Control 为同事复现。 | 覆盖 `.claude/global.md` 的报告口径（≥3 seed、独立重判）与 `.claude/paper.md` 的数字准入 |
| D4 | 2026-09-18 | 负结果**进附录**：等成本固定调度打平（A4）、训练窗口收缩（A3）、四小样本列（A2）、一步误差（A5）。 | 覆盖 `.claude/paper.md` 诚实性红线"打不赢等代价 periodic 必须进正文" |
| D5 | 2026-09-18 | 写作顺序：Introduction → Method → Experiments（分四小节）→ Abstract + Conclusion。Related Work 待参考文献阶段再写。 | 覆盖 skill A9 顺序（PF → Method → Exp → Intro → Abstract） |
| D6 | 2026-09-18 | 正文生成用 **Sonnet** 子代理；Opus 只做编排、审核、triage，不自己写正文段落（≤2 句的 cosmetic 修补除外）。 | 覆盖 `.claude/paper.md` "正文生成用 Fable" |
| D7 | 2026-09-18 | 暂不加参考文献。prose 里用 `% CITE: <key>` 行注释标位置，正文不出现 `\cite`。 | — |
| D8 | 2026-09-18 | 句子硬约束：单句 ≤ 30 个单词；逻辑转折 ≤ 2 次（Rule 9 的 token 表）。每章编译通过后用户审核，实验章每小节审一次。 | 比 skill 更严 |
| D9 | 2026-09-18 | 允许叙述与实验之间的小幅不严谨，登记在 `story_framework_2026-09-18.md` §八，之后再修，不阻塞写作。 | — |
| D10 | 2026-09-18 | 方法名 **KOMPAS** = **KO**opman **M**odel-**P**redictive **A**lignment at the prompt **S**tep。标题改为 *KOMPAS: Koopman Model-Predictive Alignment at the Prompt Step*。 | 满足 skill Rule 6 |
| D11 | 2026-09-18 | Tier-1 冷审 f1：接受"**控制实验即命令通道的检验**"这个口径。`choice_2` 改挂 rq1，不补 no-command 消融。同时加 `nc_1` / `nc_8` 两条不声称，正文不得声称预测实验支持命令通道。 | 覆盖冷审 f1 的"补实验"建议 |
| D12 | 2026-09-18 | Tier-1 冷审 f2：**Intro 带一句限定，Abstract 不带**。`nc_3.appears_in` 扩到 abstract/intro；Abstract 只给增幅、不写 measured，Intro 一个从句说明是三个公开基准上的点估计，实验章与结论章再展开。 | 细化 D3 |
| D13 | 2026-09-19 | 控制线三条 claim（现 `c5`–`c7`）**保持 `scoped`，不上调为 `strong`**。理由不是证据不足以支撑更强措辞，而是数据之后会同步；等同事补齐 seed 与区间再重评。 | 收束冷审 f2 的第二种处理 |
| D14 | 2026-09-19 | 标题改为 *KOMPAS: Koopman **Dynamics Modeling** and **Predictive Control** of Multi-Turn LLM Behavior*。两条要求：用 dynamics modeling 而不是 prediction（贡献是动力学模型，不是预报分数）；control 必须在标题里明写，不能藏在 model-predictive 里。acronym 展开同步改为 **KO**opman dyna**M**ics modeling and **P**redictive control of **A**ttributes at the prompt **S**tep。 | 细化 D10 |
| D15 | 2026-09-19 | 叙事第一句要给出漂移的**候选成因**，不能只陈述现象。采用"一条只说一次的指令，要和它之后累积的每一轮竞争"。这是借自漂移文献的候选解释，本文不检验它，记为 `prior_4` 并加 `nc_9`。 | — |
| D16 | 2026-09-19 | `non_claim_ledger` 改为**审查工具，不是正文**。Tier 2 / Tier 3 一律不写"we do not claim X"式句子；每条新增 `mode: audit_only` 与 `candidate_explicit`（若将来提升为显式，会落在哪）。正文写完后由用户挑出少数几条显式说明，避免限定句淹没主线。**例外**：D3 / D12 要求的点估计口径不走这条路，它挂在 `c5.notes` / `c6.notes` 上，跟着数字走。 | 覆盖 skill 对 non_claim 的默认用法 |
| D17 | 2026-09-19 | claim 与实验章**都**改成**先建模线、后控制线**：`c1`–`c4` 预测，`c5`–`c8` 控制，`c9` 桥。实验章 `03b` 为预测、`03c` 为控制。全文一条线：建模 → 验证模型 → 用模型控制。代价是最强的数字（+14.5）往后挪一页。 | 覆盖 WRITING_PLAN §5 的 CP3b/CP3c 内容分配 |
| D18 | 2026-09-19 | **主文恰好两张表**：Table 1 建模线（`tab_prediction`），Table 2 控制线（`tab_control`）。四个小样本列那张是附录 A2（`tab:prediction_small_app`），不计入主文。 | 明确 D4 |
| D22 | 2026-09-19 | Problem Formulation 拆成**独立的 §2**，内部再分 Modeling / Control 两个小节；新增建模目标公式 `eq:modeling_objective`（$H$ 步逐步平方误差，要求整个 horizon 上都小）；`eq:prompt_protocol` 移进 Method 的算子小节；删掉三处"我们不关心 X"式的排除性句子。编号变为 §1 Intro / §2 Problem / §3 Method / §4 Experiments / §5 Conclusion / §6 Related。**由另一会话按用户指示执行**，本会话只把它登记进 contract（两个新符号 `s_yhat`、`s_ell`，一个新公式）。 | 覆盖 WRITING_PLAN §2.1 的章节布局 |
| D21 | 2026-09-19 | CP1 **approved**。用户改变审核节奏：**不再逐 checkpoint 停下**，一口气把全文过一遍，之后统一回看。闸门、冷审、每 checkpoint 一次 commit 全部照旧，只是不等用户答复就进下一阶段。 | 放宽 D8 的"每章审核"与 WRITING_PLAN §5 的停点 |
| D20 | 2026-09-19 | 用户通读 `contract_zh_2026-09-19.md` 后**签字**，Tier-1 contract 生效。Phase 1 开工。后续任何对 contract 的改动都要回到用户，不能由写作会话自行修改。 | — |
| D22 | 2026-09-19 | 全文**禁用 alignment 描述本方法**。`02_method.tex` 开篇改为 "multi-turn attribute tracking"；`02_problem.tex` 的 alignment error 改为 **tracking error**（控制论标准叫法）；`semantic.md` §3.1 P1/P3 同步；`contract.yaml` `subfield` 改为 "inference-time attribute control..."（D20 要求用户签字，已于本日获准）。理由不是该词在文献中不存在（ARGS / Aligner / best-of-N / test-time preference optimization 构成一条明确的线），而是那条线的 alignment 指向偏好与 reward model，本文的对象是把标量属性追踪到参考值、评测跑在 IFBench / IFEval / COLLIE 上，两者对不上，会被记成 overclaim；且与 D14 已定的标题定位自相矛盾。**例外**：Related Work 中把 alignment 作为**他人工作的名称**引用不受限制。**不改**：`DECISIONS.md` D10 / 历史条目、`audit/`、`_archive_*`、`_colleague_backup_*`（历史记录与冻结副本）；`evidence/numbers.yaml` 的 alignment 指时间索引对齐，与本条无关。 | 覆盖 D10 遗留、细化 D14 |
| D19 | 2026-09-19 | 四个设计选择并成三个：**联合训练目标降进附录 A1**（连同 `eq:training_transition`、`eq:training_objective`），主文留 有限记忆状态 / 命令条件化算子 / 预测式候选选择。**三个 Why 段全部改成论证"实例化"而不是论证机制**：延迟嵌入、仿射输入列、提升空间 MPC 都是学界共识（Korda & Mezić 2018），再论证一遍只会招来"这不就是把它套到文本上"。 | 覆盖 WRITING_PLAN §2.2 的四条 design_choice_lattice |

## 已执行的操作性裁决（不进表，但有约束力）

- **2026-09-18（Phase 0.1）**：删除 `paper/iclr2027_koopman_alignment/`。理由是它与
  `_colleague_backup_2026-09-18/` 逐字节相同（`diff -r` 验证），仓库里不保留两份同事原稿。
  计划 §2.1 里"同事目录原样搬入归档目录"这一句作废，同事原稿的唯一冻结副本是
  `_colleague_backup_2026-09-18/`。
- **2026-09-18**：`main.tex` 标题暂用同事原题 *Test-time Alignment for Large Language Models via
  Koopman Model*，D10 定名后再改。标题里不放未经检验的主张。

## D5 的风险与缓解（写入日志，避免新会话重新辩论）

skill 的 A9 规定先写技术核心再写 Intro，理由是 Intro 是对别处决定的总结。这里先写 Intro 之所以
可接受，是因为三件事已经就位：Method 已由同事写完、两组实验数字已在手、故事框架已定。

缓解措施有两条。第一，Phase 0 先把 Tier 1 `contract.yaml` 完整重建并让用户签字，Intro 只从
contract 抽取，**不新增任何 claim**。第二，Phase 3 实验章写完后，Phase 4 开头加一次"Intro 回看"，
只允许收窄措辞，不允许放大。

## 待裁决

### （D10 已裁决，见上表）

命名时排除了 TEMPO 和 PACE：arXiv 上两者都已被多篇 LLM 论文占用。KOMPAS 未搜到 ML 领域同名方法
（websearch，2026-09-18）。acronym 按 `rhetoric_patterns.md` §1 只在 Abstract Move 4 和 Introduction P4
各揭示一次，其余位置直接写 KOMPAS。

### D19 的依据（共识扫描摘要）

完整调研见项目文档 `claude/method_novelty_scan_2026-09-19.md`。三句话：

1. **延迟嵌入**（choice_1 的机制）是处理部分观测的标准做法；**仿射输入列 + 提升空间 MPC**
   （choice_2 / choice_4 的机制）是 Korda & Mezić 2018 的经典结果。三个机制都是教科书内容。
2. 唯一**没有**学界共识的是 Koopman 自编码器的**损失项组合**（arXiv 2412.04578 直言文献差异很大）
   ——而它恰好是我们最辩护不了的：没有 term-removal 消融，且 D2 的线性特例根本不用自编码器。
3. 本文的 novelty 不在任何一个模块，而在**实例化的那一层**：状态是验证器量出的回复属性，
   输入是经固定协议编码的自然语言指令，被控对象是黑箱随机文本生成器，时钟是一次交互。

因此三个幸存的 Why 段要论证的是：为什么被延迟的是**验证器读数**、一条指令**凭什么**有标量命令坐标、
为什么候选评估**必须黑箱**。

### 仍然悬着的两件事

1. **c7 的 driver 缺机制**。LSTM 在 character_length 和 constraint 两列赢且 CI 不含 0，目前没有解释。
   Rule 1 要求每个数字跟一句 driver。CP3c 之前要么查清楚，要么这两列不带 driver 报出来。
   登记在 `story_framework_2026-09-18.md` §12。
2. **同事数据的来源**。一旦同事补齐 seed / 区间 / 指标定义，`numbers_benchmark.yaml` 的 `ci` 与
   `supersede_reason` 要同步更新，D3 与 D12 的限定措辞可以放宽。

### D13 及以后

用户在 checkpoint 审核时给出的意见，若属于**规则性裁决**，由 Opus 逐条抄进上表；若属于
**具体修改意见**，落在 `audit/user_notes_<CP>.md`，不进本文件。两类都不留在对话里。
