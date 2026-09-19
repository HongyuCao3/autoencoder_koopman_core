# 论文初稿写作执行计划（Opus 编排 · Sonnet 执行）

> 版本 2026-09-18 v1。目标读者：**执行本计划的 Opus 会话**。用户随时可能开新对话，所以每一步都以磁盘上的文件为唯一状态源，不依赖对话记忆。
> 故事来源：`paper/story_framework_2026-09-18.md`（以下简称 **故事框架**）。证据来源：`paper/evidence/numbers.yaml`（自建任务）与 `paper/evidence/numbers_benchmark.yaml`（同事基准数据，88 条）。
> 写作方法：`~/.claude/skills/neurips-write-tiered`（层次化：Tier 1 contract → Tier 2 semantic → Tier 3 prose），语音规则委托给 `~/.claude/skills/neurips-write`。两个 skill 的源码副本在 `E:\claude-skill\`。

---

## 0. 每次开新对话，Opus 先做这五件事

1. 读本文件。
2. 读 `paper/WRITING_STATE.md`（进度账本，格式见 §7）。它告诉你当前在哪个 Phase、哪个 checkpoint、用户上一次审核的结论。
3. 读 `paper/DECISIONS.md`（用户裁决日志），任何与 skill 或仓库规则冲突的地方以它为准。
4. `git status paper/` 确认没有未提交的半成品；有则先读 diff 再决定继续或回退。
5. 只做 `WRITING_STATE.md` 里 `next_action` 指向的那一步，做完更新账本、提交、停下等用户。

**不要**跳过 checkpoint 连写两章。**不要**在用户未审核的章节上继续。

---

## 1. 已锁定的用户裁决（写进 `paper/DECISIONS.md`，本计划只是摘要）

| # | 裁决 | 覆盖了什么规则 |
|---|---|---|
| D1 | 放弃 `sections/` 的负结果论文（*When Not to Close the Loop*）；旧 `contract.yaml` / `semantic.md` / `sections/*.tex` 归档不删 | — |
| D2 | 卖点一（预测）：线性延迟嵌入作为同事 AE-Koopman 模型的特例（$E_\phi = D_\psi = \mathrm{id}$），主表 Ours 行即此特例 | — |
| D3 | 卖点二（控制）：正面口径，同事 IFBench/IFEval/COLLIE 数据全部保留、**暂视为可靠**；Base = 不加任何控制；TMPC / RE-Control 为同事复现 | 覆盖 `.claude/global.md` 报告口径（≥3 seed、独立重判）与 `.claude/paper.md` 数字准入 |
| D4 | 负结果进附录（等成本固定调度打平、训练窗口收缩、四小样本列、一步误差） | 覆盖 `.claude/paper.md` 诚实性红线"打不赢等代价 periodic 必须进正文" |
| D5 | 写作顺序：Introduction → Method → Experiments（分四小节）→ Abstract + Conclusion；Related Work 待参考文献阶段再写 | 覆盖 skill A9 顺序（PF → Method → Exp → Intro → Abstract） |
| D6 | 正文生成用 **Sonnet** 子代理，Opus 只做编排、审核、triage | 覆盖 `.claude/paper.md`"正文生成用 Fable" |
| D7 | 暂不加参考文献；prose 里用 `% CITE: <key>` 行注释标位置，正文不出现 `\cite` | — |
| D8 | 句子硬约束：单句 ≤ 30 个单词；逻辑转折 ≤ 2 次（Rule 9 的 token 表）；每章编译通过后用户审核，实验章每小节审一次 | 比 skill 更严 |
| D9 | 允许叙述与实验之间的小幅不严谨，登记在 `story_framework` §八，之后修 | — |

**D5 的风险与缓解。** skill 的 A9 规定先写技术核心再写 Intro，理由是 Intro 是对别处决定的总结。这里先写 Intro 之所以可接受，是因为 Method 已由同事写完、两组实验数字已在手、故事框架已定。缓解措施：**Phase 0 先把 Tier 1 contract 完整重建并让用户签字**，Intro 只从 contract 抽取，不新增任何 claim；Phase 3 实验章写完后，Phase 4 开头加一次"Intro 回看"，只允许收窄，不允许放大。

---

## 2. 目录布局与 Phase 0 的起点动作

### 2.1 目标布局（`autoencoder_koopman_core/paper/`）

```
paper/
├── main.tex                      # 唯一主文件（已有 ulem/booktabs/multirow；作者块留 TODO）
├── iclr2027_conference.{sty,bst}, math_commands.tex, fancyhdr.sty, natbib.sty
├── contract.yaml                 # Tier 1（Phase 0 重建）
├── semantic.md                   # Tier 2（逐章构建）
├── sections/
│   ├── abstract.tex              # Phase 4
│   ├── 01_introduction.tex       # Phase 1
│   ├── 02_method.tex             # Phase 2（同事 3-method.tex 迁入 + 改造）
│   ├── 03_experiments.tex        # Phase 3，\input 四个子文件
│   ├── 03a_setup.tex / 03b_control.tex / 03c_prediction.tex / 03d_analysis.tex
│   ├── 04_conclusion.tex         # Phase 4
│   ├── 05_related_work.tex       # 空壳，待引用阶段
│   └── appendix/A1_estimator.tex, A2_small_columns.tex, A3_window.tex, A4_equal_cost.tex, A5_onestep.tex
├── tables/tab_control.tex, tab_prediction.tex   # 表格独立文件，由脚本从 evidence 生成
├── figs/method.png（从同事目录复制）, fig_memory_forest.pdf（待从 PPT 源脚本重出）
├── tools/sentence_gate.py, build_tables.py, mech_audit.sh
├── evidence/numbers.yaml, numbers_benchmark.yaml
├── story_framework_2026-09-18.md
├── WRITING_PLAN_2026-09-18.md（本文件）, WRITING_STATE.md, DECISIONS.md
└── _archive_2026-09-18_negative_paper/   # 旧 contract.yaml, semantic.md, sections/, iclr2027_koopman_alignment/ 原样搬入
```

### 2.2 Phase 0 步骤（Opus 亲自做，不下放；预计一次会话）

| 步 | 动作 | 产出 / 闸门 |
|---|---|---|
| 0.1 | `git mv` 旧 `contract.yaml` `semantic.md` `sections/` 及同事的 `iclr2027_koopman_alignment/` 到 `_archive_2026-09-18_negative_paper/`；复制（不是移动）同事 `3-method.tex`、`figs/method.png` 到新位置 | commit `paper: archive negative-result draft, start positive draft` |
| 0.2 | 新建 `DECISIONS.md`（§1 表格全文）、`WRITING_STATE.md`（§7 模板） | — |
| 0.3 | 写 `tools/sentence_gate.py`（§4.2 规格）、`tools/mech_audit.sh`（§4.3）、`tools/build_tables.py`（从两个 evidence yaml 生成 `tables/*.tex`，Table 1 加 Access 列与 Avg. 列） | 脚本跑通；生成的两张表先编译一次 |
| 0.4 | **重建 Tier 1 `contract.yaml`**：以 `neurips-write-tiered/templates/tier1.yaml` 为骨架，内容全部来自故事框架。必填：`narrative_arc`（框架 §一 展开为 5 句）、`design_choice_lattice`（4 条：finite-memory state / command-conditioned Koopman operator with linear special case / joint AE training objective / predictive candidate selection）、`claim_ledger`（框架 §四 三贡献拆成 6–8 条，每条带 `evidence_ids` 指向 `n_main_*` / `n_bench_*`、`strength`）、`symbol_lattice`（同事记号）、`equation_lattice`（同事 3-method 的 8 个式子，`operation_name` 从式子读出，A5）、`intuition_lattice`（anchor example = tsar_cefr 过冲；named assumptions 三条，`tested_by` 按 A10 填）、`non_claim_ledger`（至少：不声称在自建四任务上闭环赢固定调度；不声称基准数字有 CI） | 跑 skill `references/reviewer.md` 的 **Tier-1 cold review**（Sonnet 子代理，persona 见 §6.4）→ Opus triage → **用户签字** |
| 0.5 | 决定方法名。skill Rule 6 要求 acronym；故事框架未命名。Opus 提 3 个候选（例：**KOMPAS** = KOopman Model-Predictive Alignment at the prompt Step；**PRISM** = PRompt-level Interaction State Model；或不取名沿用 "Koopman predictive control"）交用户选 | 写入 `contract.yaml: meta.method_name` 与 `DECISIONS.md D10` |

Phase 0 结束条件：contract 签字、脚本可用、`main.tex` 空章节能编译出 PDF。

---

## 3. 角色分工

| 角色 | 模型 | 干什么 | 不干什么 |
|---|---|---|---|
| 编排者 | **Opus**（当前会话） | 读状态、切分任务、组装子代理上下文包、跑闸门、Tier-1/2/3 审稿结论的 triage、写 `WRITING_STATE.md`、git commit、向用户汇报 | 不自己写正文段落（除 ≤2 句的 cosmetic 修补） |
| Tier-2 构建 | Sonnet | 为一章写 `semantic.md` 的该章块（无完整句子，A2） | 不写 prose |
| Tier-3 生成 | Sonnet | 按 §6 上下文包生成一章 / 一小节的 `.tex` | 不读其他章 prose（A7）；不改 contract |
| 冷审 | Sonnet（独立实例） | Tier-1 / Tier-2 / Tier-3 审稿 persona | 不改文件，只出 findings yaml |
| 机械审计 | 脚本（Haiku 可选） | 句长、转折、adverb、em-dash、Rule 21/22 grep | — |

一个原则：**同一个 Sonnet 实例不同时做生成和审稿**。

---

## 4. 硬约束与机械闸门

### 4.1 写作约束（塞进每个 Tier-3 子代理的 prompt）

1. 单句 ≤ **30 个单词**（LaTeX 命令、`$...$` 内容、`\ref{}` 不计词；按空格切分计数）。
2. 单句逻辑转折 ≤ **2**（Rule 9 token 表：however / while / yet / whereas / rather / `;` / because / so / once / when / if / although / but）。
3. 每段先直觉再公式；公式后每个符号在同一句或下一句给出含义（Rule 14a）。
4. 无 em-dash（Rule 19）、无副词表内副词（Rule 20）、Method 无结果数字（Rule 21）、Why-X 无公式（Rule 22）、代词有先行词（Rule 23）。
5. 不写 `\cite`，需要引用处单独一行 `% CITE: li2024instability`。
6. 每段 ≤ 6 句；每小节开头一句话说这一小节要回答什么。
7. 数字只能来自两个 evidence yaml，引用时在 `.tex` 行尾加 `% n_bench_037` 之类注释，便于审计。
8. 用 `\uline{}` 收尾问答对（Rule 18）；`main.tex` 已加载 `ulem`。

### 4.2 `tools/sentence_gate.py` 规格

输入：一个或多个 `.tex`。处理：去掉 `%` 注释、`\begin{equation}...\end{equation}`、`\begin{table}...`、`$...$`、`\cmd{...}` 的命令名（保留花括号里的文字）；按 `. ! ?` 后跟空格+大写切句。输出每条违规：文件、行号、句子前 60 字、词数、转折 token 列表。阈值：词数 > 30 或转折 ≥ 3 即 FAIL。退出码非零表示有违规。**这是每次 Tier-3 生成后的第一道闸门，不过不进入审稿。**

### 4.3 `tools/mech_audit.sh`

顺序执行：`sentence_gate.py` → em-dash grep（`---`）→ adverb grep（Rule 20 列表）→ 若目标是 `02_method.tex` 再跑 Rule 21 grep（`Table~\ref|sec:exp|outperform|baseline|\+[0-9.]+`）与 Rule 22 Why-X 段内 `$` grep → `\cite` 出现即 FAIL（D7）。全部通过才允许编译。

### 4.4 编译

`cd paper && latexmk -pdf -interaction=nonstopmode main.tex`（设备上有 `/usr/bin/latexmk`）。编译失败先修编译，再进用户审核。把 PDF 复制为 `paper/build/main_<phase>_<date>.pdf` 留档，方便用户对照不同 checkpoint。

---

## 5. 阶段、检查点与每步产出

每个 checkpoint 的固定六步：**B 建 Tier 2 → 审 Tier 2 → C 生成 Tier 3 → 机械闸门 → 编译 → 用户审核**。用户审核结论三种：`approved` / `revise:<cosmetic|structural>` / `redo`。structural 必须回 Tier 1/2 再重生成（A1）。

### Phase 1 — Introduction（checkpoint CP1）

- Tier 2 分配（5 段，页预算 1.0–1.2 页）：P1 现象（框架 §2.1）；P2 三个 gap，用 Rule 18 的"prior-work label ↔ However 句"结构，三个 gap 各一对（框架 §2.2）；P3 为什么交互层是正确时间步（§2.3，含"室温 / 分子"类比，Rule 22 级别无公式）；P4 为什么 Koopman，五项需求压成三句（§3.1，Rule 3 三元并列）；P5 三贡献编号列表（§四）。
- 子代理上下文包：contract 全文 + semantic §Intro + `neurips-write/SKILL.md` + `references/introduction.md` + `phrase_bank.md` + 故事框架 §二、§三、§四 + §4.1 约束。
- 冷审重点：Intro 是否承诺了 contract 里没有的东西（对照 `claim_ledger`）；三贡献是否和 `design_choice_lattice.intro_bullet_keyword` 一一对应。
- 产出：`sections/01_introduction.tex`、编译 PDF、`WRITING_STATE.md` 更新为 `CP1: awaiting_user_review`。

### Phase 2 — Method（checkpoint CP2）

同事的 `3-method.tex` 已是成稿，这一章**以改造为主、不重写**（A1 允许的范围之外的改动要走 Tier 2）。具体改造清单：

1. 首段总览保留；图注从 "Caption" 改为描述四个阶段的一句话。
2. §3.1 Problem Formulation 保留；加入故事框架 §二的 anchor example（tsar_cefr 过冲）一段，A11。
3. §3.2 加一段 **linear special case**：$E_\phi = D_\psi = \mathrm{id}$，此时式 (koopman_model) 退化为对记忆向量的线性回归；说明这是消融中的一档，不放任何数字（Rule 21）。
4. 每个 subsection 补一个 `\textbf{Why ...}` 段（Rule 24：恰好一个），并以 `\uline{}` 收尾（Rule 18）。同事原稿没有 Why-X，这是主要新增。
5. 全文过 `sentence_gate.py`：同事原稿多处超 30 词，需要拆句（cosmetic，允许直接改）。
6. §3.5 误差界保留，作为 Phase 3d 的桥。
7. 记号：删除旧稿的 $z_t, u_t, v_t$；全篇只用同事记号。

- 上下文包：contract + semantic §Method + `references/method.md` + `alignment_checklist.md` + 同事原文。
- 闸门额外项：Rule 21 / 22 grep 必须零命中。
- 产出：`sections/02_method.tex`；`figs/method.png` 引用正确；编译。

### Phase 3 — Experiments（四个 checkpoint，CP3a–CP3d，每个都停）

统一形式：Rule 7 tight-page form（`\textbf{Question.}` / `\textbf{Setup.}` / `\textbf{Findings.}`），RQ 问题 yes/no 可证伪、无公式（Rule 17）。每小节的 Findings 每个数字后跟 driver 句（Rule 1、5）。

| CP | 小节 | 内容与数据源 | 特别注意 |
|---|---|---|---|
| **3a** | `03a_setup.tex` 实验设置 | 两组任务：(i) 11 个多轮任务（7 属性追踪 + 4 行为任务，只在此处给数量与轮数）；(ii) 三个基准的 11 个子任务。模型 Qwen3-4B / 8B（基准）与 Qwen3-4B-Instruct 等（自建，按 numbers.yaml 记录）。基线定义：Stateless、Last-turn、Memory w/o action、LSTM、Koopman nonlinear、Koopman linear（预测）；Base、TMPC、RE-Control（控制），注明后两者为复现且需白盒访问。指标：满足率（↑）与 H 步 rollout MSE（↓），H 的取值。 | Rule 10：所有 setup 细节只能出现在这里和附录 |
| **3b** | `03b_control.tex` 控制结果（headline） | RQ1："Does predictive control built on the learned operator raise constraint satisfaction over an uncontrolled model and over white-box latent controllers?" Table 1 = `tables/tab_control.tex`（脚本生成，含 Access、Avg. 列）。Findings 三条：相对 Base 的均值提升（4B +14.5，8B +14.8）；相对 RE-Control（+5.2 / +5.7，最小 +2.4）；22/22 格最优。每条一个 driver（例：候选评估在隐空间完成，允许在每轮选择不等于目标的命令）。 | 明写"数字为点估计，区间待补"一句（D3 的诚实边界，不加脚注） |
| **3c** | `03c_prediction.tex` 预测结果 | RQ2(a)："Does the operator need several turns of history?" (b)："Does a nonlinear lifting or a sequence model predict better than the linear special case?" Table 2 = `tables/tab_prediction.tex`（7 列 × 6 行）+ Fig. 2 森林图（Claim A）。Findings：Ours 4/7 最优、其余第二（按 7 列重算，脚本给出）；记忆必要性五列 CI 不含 0；非线性/LSTM 差 ≤ 0.031 且符号翻转 → 写成"线性特例够用"。 | 列名用论文用语，不用代码名；Skill 的定义在 3a 或此处首次出现时给公式并解释每个符号 |
| **3d** | `03d_analysis.tex` 分析与成本 | RQ3："Does prediction quality translate into control quality, and at what inference cost?" 内容：(i) 用 §3.5 的界解释"预测准 → 选择近优"；(ii) 成本：候选评估零额外 LLM 调用，TMPC/RE-Control 需隐层访问；(iii) 一段 scope：在四个自建任务上闭环与等成本固定调度打平，指向附录 A4，措辞为"planning headroom"边界而非失败。 | 这是全文最容易过度声称的地方，冷审用 `paper-audit/references/claims.md` D4 |

每个 CP 的产出：对应 `.tex` + 编译 PDF + 账本更新。3a 之前先跑 `build_tables.py` 生成两张表并单独编译检查排版（`\resizebox` 到 `\textwidth`，字号 `\footnotesize`）。

### Phase 4 — Abstract + Conclusion（checkpoint CP4）

1. **Intro 回看**（Opus 亲自）：对照 CP3 的 Findings 逐条核对 Intro 的三贡献，只允许收窄措辞；有改动走 cosmetic 路径，超出 cosmetic 则回 Tier 1。
2. Conclusion：skill Rule 16 的 takeaway-first 6-move 形式（问题 / 要点 / 方法 / 一句实证 / 限制列表 / 一条 future work），≤ 150 词。限制列表必须包含：基准数字无区间；自建任务上闭环无余量；TMPC/RE-Control 为白盒复现。
3. Abstract：按 `references/abstract.md` 的抽取法，从 Intro P1、Method Why-X、CP3b headline、Conclusion takeaway 各取一句改写；≤ 180 词；不带 setup 细节（Rule 10）；可以带一个 headline 数字（约 +15 点）。
4. 两者一起编译、一起审。

### Phase 5（本轮不做）— Related Work 与参考文献

等用户开放引用后：收集全文 `% CITE:` 标记 → 建 `main.bib` → 写 `05_related_work.tex`（Rule 15：第三人称、只 bold 分类标签、放 Conclusion 前）。

---

## 6. 子代理 prompt 模板（英文，Opus 每次填空后发出）

### 6.1 Tier-2 builder（Sonnet）

```
You are building the Tier-2 semantic outline for §<SECTION> of an ICLR paper.
Read, in this order: paper/contract.yaml (full); ~/.claude/skills/neurips-write-tiered/references/semantic.md;
the §<SECTION> allocation notes below; paper/story_framework_2026-09-18.md §<relevant parts>.
Write ONLY the `## §<SECTION>` block of paper/semantic.md. Rules: no complete sentences (A2);
every bullet is `P<n> (<role>): ...` and references contract IDs in brackets; every contract item
that belongs to this section must be cited by at least one bullet (A3); no dangling IDs (A4);
intuition bullets precede the first equation bullet (A11). Page budget: <X> pages, <N> paragraphs.
Allocation notes: <paste from §5 of the plan>.
Return the block as text; do not touch any other file.
```

### 6.2 Tier-3 prose generator（Sonnet）

```
You are writing §<SECTION> (file paper/sections/<file>.tex) of an ICLR 2027 paper.
Context to read, in order, and nothing else about other sections' prose (A7):
1. paper/contract.yaml (full)   2. the `## §<SECTION>` block of paper/semantic.md
3. ~/.claude/skills/neurips-write/SKILL.md (all 24 rules)   4. ~/.claude/skills/neurips-write/references/<section>.md
5. ~/.claude/skills/neurips-write/references/phrase_bank.md   6. <evidence yaml(s) if numbers are needed>
7. <existing file to transform, if any>
Hard constraints on top of the 24 rules:
- Every sentence <= 30 words (LaTeX commands and math not counted). Every sentence <= 2 logical turns.
- Intuition sentence before every displayed equation; every symbol glossed in the same or next sentence.
- No \cite. Where a citation belongs, insert a separate line `% CITE: <key>`.
- Every number must come from the evidence yaml; append `% <evidence id>` at the end of that line.
- No em-dashes, no adverbs from the Rule 20 list, no results in §Method, no math in Why-X paragraphs.
- Q&A bookend: bold question at the start, \uline{} answer as the last sentence of the paragraph.
Walk the Tier-2 bullets in order, one paragraph per bullet, honoring each bullet's constraints.
Write the file. Then run `python paper/tools/sentence_gate.py paper/sections/<file>.tex` and fix every
violation before returning. Return: the file path, the gate output (must be clean), and a 5-line summary
of which contract claims each paragraph delivers.
```

### 6.3 Mechanical auditor（脚本优先；Haiku 备用）

```
Run paper/tools/mech_audit.sh paper/sections/<file>.tex and report the raw output. Do not fix anything.
```

### 6.4 Cold reviewer（Sonnet，独立实例）

```
Adopt the persona in ~/.claude/skills/neurips-write-tiered/references/reviewer.md for Tier <1|2|3>.
For Tier 3 additionally load ~/.claude/skills/paper-audit/references/{confusing,claims,numbers}.md.
Baseline knowledge whitelist: a NeurIPS/ICLR reviewer who knows MPC, Koopman operators, instruction-following
benchmarks, and activation steering, but has NOT read this project's docs.
Input: <artifact path(s)>. Output: paper/audit/<section>_<tier>_<date>.yaml using the finding schema in
paper-audit/references/report.md, each finding with severity and recommended_fix_tier (T1/T2/T3/data).
Also answer in one line: is every sentence easy to follow on a single read? Name the three hardest.
Do not edit any file.
```

Opus 收到 findings 后 triage：T3 cosmetic 自己改或让同一生成器改；T1/T2 回上层再重生成；`data` 类记到 `story_framework` §八，不阻塞。

---

## 7. 续写协议：`paper/WRITING_STATE.md`

固定格式，每个 checkpoint 更新一次并提交：

```markdown
# WRITING_STATE
updated: 2026-09-18 23:10   by: opus session <short id>
phase: 3   checkpoint: CP3b   status: awaiting_user_review | in_progress | approved | revise
last_commit: <hash> <message>
artifacts_ready: sections/01_introduction.tex (approved CP1, 2026-09-19), sections/02_method.tex (approved CP2), ...
current_target: sections/03b_control.tex
gates: sentence_gate=pass  mech_audit=pass  compile=pass (build/main_CP3b_2026-09-20.pdf)
cold_review: paper/audit/03b_control_T3_2026-09-20.yaml  (2 major, 3 minor; majors triaged -> T3 cosmetic, fixed)
user_verdict_on_previous: CP3a approved with note "shorten baseline paragraph"
next_action: wait for user verdict on CP3b; on approve -> Phase 3c (build Tier 2 for 03c_prediction first)
open_questions_for_user: (1) method name (D10) still pending? (2) ...
```

补充规则：

- 每个 checkpoint 一次 commit，message 形如 `paper CP3b: control results drafted, awaiting review`；用户 approve 后再 commit 一次 `paper CP3b: approved`。这样任何新会话用 `git log --oneline -- paper` 就能重建时间线。
- 用户的审核意见由 Opus 逐条抄进 `DECISIONS.md`（若是裁决）或 `paper/audit/user_notes_<CP>.md`（若是修改意见），不留在对话里。
- `semantic.md` 每章块顶部写一行 `<!-- status: approved CP1 -->`，避免新会话重建已批准的 Tier 2。
- 若新会话发现 `status: in_progress` 且有未提交改动：先 `git stash` 保存，读 `WRITING_STATE.md` 判断是继续还是丢弃，不要直接覆盖。

---

## 8. 与 skill / 仓库规则的偏离一览（便于审计）

| 偏离 | 来源 | 处理 |
|---|---|---|
| 写作顺序 Intro 先行 | D5 | Phase 0 先锁 contract；Phase 4 Intro 回看只收窄 |
| Sonnet 写正文 | D6 | 加 30 词硬闸门与独立冷审补偿 |
| 未满足报告口径的数字进正文 | D3 | 3b 里明写"点估计、区间待补"；`numbers_benchmark.yaml` 头部已记 |
| 负结果不进正文 | D4 | 3d 保留一段 scope 指向附录 A4，不省略事实 |
| 无参考文献 | D7 | `% CITE:` 标记，Phase 5 统一补 |
| Related Work 延后 | D5 | `05_related_work.tex` 空壳，`main.tex` 保留 `\input` |

---

## 9. 首个会话的具体待办（Phase 0，按序）

1. 读本文件与故事框架；确认 `git status` 干净。
2. 0.1 归档与迁移；commit。
3. 0.2 建 `DECISIONS.md`、`WRITING_STATE.md`。
4. 0.3 写三个脚本并跑通；`build_tables.py` 生成两张表，编译空壳 `main.tex`。
5. 0.4 重建 `contract.yaml`；派 Sonnet 做 Tier-1 冷审；triage。
6. 0.5 提三个方法名候选。
7. 更新 `WRITING_STATE.md`：`phase: 0, status: awaiting_user_review, next_action: user signs contract + picks method name → Phase 1`。停。
