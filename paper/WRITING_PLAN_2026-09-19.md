# 论文写作执行计划 v2.1（2026-09-19）：Method 重写 + 建模/控制双主线

> 版本 2026-09-19 **v2.1**，**取代** `WRITING_PLAN_2026-09-18.md`（旧文件保留作历史，不再作为状态源）。目标读者：**执行本计划的 Opus 会话**。每一步以磁盘文件为唯一状态源，不依赖对话记忆。
> v2.1 相对 v2 的改动（用户裁决 2026-09-19 晚）：Method 小节**按组件重切**（模型 / 学习 / 控制器 + Algorithm 1 / Proposition），不再按设计选择数切；3.4 保留独立短节；learned lifting 留在主文，但**首次出现必须有面向非 Koopman 读者的白话解释**；Rule 24 改为每节至多一个 Why。依据：`claude/method_subsection_allocation_iclr_2026-09-19.md`。
> 主线（用户 2026-09-19）：展示 Koopman 方法在 LLM 提示词控制 / 约束满足任务上的 **modeling** 与 **controlling** 能力。同事数据全部保留；同事 Method 的**骨架与论证**不再复用，**公式**可复用。
> 诊断依据：项目文档 `claude/method_rewrite_plan_2026-09-19.md`（为什么要重写）。本文件只讲怎么做。
> 起点状态：CP1 approved；CP2 done 但要 redo；§2 Problem 已独立成章（D23）；Experiments 四小节 Tier 2 已建（`semantic.md` §Experiments, draft CP3）；Fig. 1 TikZ 重绘到 step 3 rev 4 待审；契约 D1–D24（`DECISIONS.md` 里 D22 重号，见 §2.3）。

---

## 0. 每次开新对话，Opus 先做这五件事

1. 读本文件。
2. 读 `paper/WRITING_STATE.md`：当前 Phase、checkpoint、`next_action`。
3. 读 `paper/DECISIONS.md`：任何冲突以它为准。
4. 先调一次 `device_request_delete_permission`（git `index.lock` 会卡，见 `claude/repo_git_hygiene_notes_2026-09-18.md`），再 `git status paper/`；有未提交半成品先读 diff。
5. 只做 `next_action` 指向的那一步；做完更新账本、commit、按 §4 的停点规则决定停不停。

---

## 1. v2.1 相对 v1 改了什么（一张表）

| 项 | v1（09-18） | v2.1（09-19） | 为什么 |
|---|---|---|---|
| Phase 2 Method | "改造不重写"同事 `3-method.tex` | **按新骨架重写**（§5），同事公式可复用，骨架/开篇/Why 段/符号携带不复用 | 同事骨架是控制器论文，建模线在 Method 里无落点 |
| Method 结构 | 3.1 状态 / 3.2 算子 / 3.3 选择 / 3.4 误差界（按设计选择切） | **按组件切**：3.1 模型（状态窗口作段落）/ **3.2 从交互数据学模型**（辨识式 + 数据构造）/ 3.3 带重规划的预测式指令选择（**Algorithm 1**）/ 3.4 预测精度买到什么（**Proposition 1**，独立短节） | ICLR 读者按组件读 Method；训练必须在主文；定义与推导不独占小节 |
| Why 段规则 | Rule 24：每节**恰好**一个 | 每节**至多**一个；小节边界不为 Why 段而开；"验证器读数"论证前移到 §2.1 作可观测性事实 | v1 的 3.1/3.2 拆分是被 Rule 24 逼出来的 |
| learned lifting | 主文全程携带符号，无解释 | 主文保留（3.1 一句推广 + 3.2 一句指向 A1 + Table 1 一行），**首次出现给白话解释 + 类比**（§5.3） | 用户要求：其他领域审稿人要能看懂 |
| 设计选择的证据 | choice_1→rq2a，choice_2→rq1，choice_3→rq1（2 控制 : 1 建模） | choice_1→rq2a，**choice_2→rq2b（主）+ rq1（次）**，choice_3→rq1（**2 建模 : 1 控制**） | 与标题 D14 / claim 顺序 D17 一致 |
| 一般 lifting 的**公式** | 主文全程携带 $E_\phi, D_\psi, g_\psi, \xi$ | 主文公式只用 $s_t, K, B, b$；$E_\phi$ 只在解释 lifting 的那两句里出现，AE 训练目标在 A1 | D2：汇报实例是恒等 lifting |
| 术语 | D22 禁 alignment | 加禁用表进机械闸门：test-time、intervention、alignment cost（§6.1） | 残留词汇 |
| Intro 贡献 | 贡献 2 = 建模 + 控制数字 | 贡献 2 只挂 c2/c4，控制数字并入贡献 3 | 三贡献 = 状态 / 模型 / 控制器 |
| 停点 | D21：不停，一口气过完 | **CP2-redo 停一次**（结构性重写），其后恢复 D21 | 重写后实验章 Tier 2 要 relabel，先审再走 |
| Fig. 1 | 四模块 A 记忆 / B 算子 / C 训练 / D 控制 | 模块 C 待裁决：改为"辨识 + 线性特例"或降为附录图 | 训练目标已在附录，主图不应画它 |

其余（角色分工、闸门、子代理模板、账本协议）沿用 v1，本文件 §6–§8 只写变化。

---

## 2. 裁决

### 2.1 仍然有效的旧裁决
D1–D4、D6–D18、D20、D23、D24 全部有效。D5 的写作顺序保留（Intro 已写完）。D19 **部分修正**：AE 联合训练目标（三项损失）仍在附录 A1，但**"模型怎么从数据得到"回主文**——线性特例的最小二乘辨识式与三元组构造放在新的 3.2（D19 砍的是"损失项组合是卖点"，不是"模型怎么得到"）。D21 **暂停一次**：CP2-redo 后停下等用户，之后恢复。Rule 24 由 D32 放宽。

### 2.2 待用户签字的新裁决（提案，编号接 D24）

| # | 提案 | 覆盖 |
|---|---|---|
| D25 | Method 章**重写**而非改造，小节**按组件切**（§5.2：模型 / 学习 / 控制器 / 保证），小节数不等于设计选择数；同事公式可复用，骨架不复用 | v1 §5 Phase 2 |
| D26 | `choice_2` 改名 "Koopman Dynamics Model on Delay Coordinates"，主 `rq_id` 改为 rq2b，rq1 降为次要；`why_keyword` 改为建模论证（§5.2） | D11 的"choice_2 挂 rq1"（D11 的口径本身不变：控制实验仍是命令通道的检验，只是不再做 choice_2 的主证据） |
| D27 | 新小节 3.2 "Learning the Model from Interaction Data"：三元组构造 + 新公式 `eq_identification`（线性特例闭式最小二乘）+ 覆盖性一句；A1 只留 AE 三项损失 | D19 部分 |
| D28 | c9 保留，3.4 为**独立短节**（用户 2026-09-19 定），内容是 `Proposition 1`（$s$-空间陈述）+ 一句证明思路 + 边界一句；`eq_residual_assumptions` / `eq_cost_bound` 与完整证明迁入**新附录 A6** | — |
| D29 | 术语禁用表进 `mech_audit.sh`：`test-time`、`intervention`、`alignment cost`、`align`（Related Work 引他人工作名除外，用 `% GATE-EXEMPT: term` 豁免）。替换：test-time → inference-time / during control；intervention → instruction（正文）/ input（控制论语境） | 细化 D22 |
| D30 | Problem §2.2 加"满足 ⇔ $y_t\in\mathcal{Y}$，$r$ 取在 $\mathcal{Y}$ 内"的桥；**具体措辞等同事的度量定义**，之前放 `TODO(data)` | 补 D23 |
| D31 | Fig. 1 模块 C 的去向（二选一，用户定）：(a) 改画"辨识 + 线性特例是恒等 lifting"；(b) 降为附录图，主图三模块 + 闭环 | FIG_STATE Q1 |
| D32 | Rule 24 放宽：每小节 Why 段**至多**一个，可以没有；小节边界按组件切，不为安放 Why 段而开节。原 3.1 的"被窗口化的是验证器读数"论证**前移到 §2.1 Problem / Modeling**，写成可观测性事实（黑箱调用者只拿得到 $V$ 的读数），不再是 Why 段。`choice_1.why_keyword` 相应改为 "stated as an observability fact in Problem Formulation" | skill Rule 24 |
| D33 | **learned lifting 留主文**（用户 2026-09-19 定）：3.1 一句推广、3.2 一句指向 A1、Table 1 "KOMPAS (learned lifting)" 一行保留。条件：**首次出现处给一段白话解释 + 一个类比**（规格见 §5.3），目标读者是不熟 Koopman 的 ML 审稿人；冷审加定向问题检验 | — |
| D34 | 3.3 加 **Algorithm 1**（滚动时域循环）；`main.tex` 加 `algorithm` + `algorithmic` 包；伪代码受 Rule 21 约束（无结果数字）；`equation_gate.py` 跳过 algorithm 环境、识别 Proposition 环境内公式 | — |

### 2.3 `DECISIONS.md` 的编号修复
表里 **D22 出现两次**（Problem Formulation 拆分那条与 D23 内容重复；alignment 禁用那条是真正的 D22）。处理：把重复的那条标注 `（重号，内容由 D23 承载，作废）`，不删行（D 表不删不改的规则），新裁决从 D25 起，本轮到 D34。

---

## 3. 目标结构与当前状态盘点

```
paper/
├── main.tex                          §1 Intro / §2 Problem / §3 Method / §4 Exp / §5 Concl / §6 Related / 附录 A1–A6；加 algorithm/algorithmic 包（D34）
├── contract.yaml                     Tier 1 —— Phase R0 修订后再签字
├── semantic.md                       Tier 2 —— §Intro approved；§Method 作废重建；§Experiments draft 待 relabel
├── sections/
│   ├── abstract.tex                  空 → Phase 4
│   ├── 01_introduction.tex           CP1 approved → Phase R2 拆贡献 2（cosmetic+）
│   ├── 02_problem.tex                D23 已拆 → Phase R2 补桥（D30）、加可观测性两句（D32）、改前向引用
│   ├── 02_method.tex                 CP2 done → **Phase R1 重写**（CP2-redo）
│   ├── 03_experiments.tex + 03a–03d  骨架 → Phase 3 生成（Tier 2 已有，先 relabel）
│   ├── 04_conclusion.tex, abstract   空 → Phase 4
│   ├── 05_related_work.tex           空壳 → Phase 5
│   └── appendix/A1_estimator.tex（保留，改开头指向新 3.2）, A2–A5（占位）, **A6_selection_bound.tex（新）**
├── tables/tab_prediction.tex, tab_control.tex, tab_prediction_small_app.tex   已生成，不动
├── figs/method_fig/                  step 3 rev 4 待审 → Phase R3 对齐 D31
├── tools/sentence_gate.py, equation_gate.py, mech_audit.sh（加术语关）, build_tables.py, fig_symbol_audit.py
├── evidence/numbers.yaml, numbers_benchmark.yaml   不动
└── DECISIONS.md, WRITING_STATE.md, WRITING_PLAN_2026-09-19.md（本文件）
```

文件名前缀与章节号不对应（`02_method.tex` 是 §3）：不改名，v1 已登记为 cosmetic 遗留。

---

## 4. Phase 分解与停点

| Phase | 内容 | 谁做 | 停点 |
|---|---|---|---|
| **R0** 契约修订 | §2.2 的 D25–D31 提案写进 `DECISIONS.md`（待签字状态）；`contract.yaml` 按 §5.4 改；`mech_audit.sh` 加术语关；Sonnet Tier-1 冷审只审改动部分；重生成 `contract_zh_2026-09-19b.md` | Opus 亲自 | **停，用户签字**（D20） |
| **R1** Method 重写 | Tier 2 §Method 重建 → 冷审 → Sonnet 生成 `02_method.tex` + `appendix/A6` + A1 开头 → 闸门 → 编译 → 冷审 | Sonnet 生成，Opus 编排 | **停，用户审**（CP2-redo，D21 暂停一次） |
| **R2** 邻接修补 | `02_problem.tex` 补桥（D30，若度量定义未到则 `TODO(data)`）+ §2.1 加可观测性两句（D32）+ 前向引用改到新 3.1/3.2；`01_introduction.tex` 贡献 2 拆分；A1 开头改指向 3.2 | 可观测性两句与贡献拆分 → Sonnet（同一 R1 生成器，附 §2.1 上下文）；其余 Opus cosmetic | 不停，随 R1 一起审 |
| **R3** Fig. 1 对齐 | 按 D31 处理模块 C；`fig_symbol_audit.py` 对新 `02_method.tex` 重跑；图注改为"左建模右控制" | 按 `figs/method_fig/FIG_PLAN` 的会话 | 图有自己的 FIG_STATE 停点 |
| **3** Experiments | 先做 **relabel pass**（§5.5），再逐节生成 03a–03d；每节闸门 + 冷审 + commit | Sonnet / Opus | 不停（D21） |
| **4** Abstract + Conclusion | Intro 回看（只收窄）；Conclusion 6-move ≤150 词；Abstract ≤180 词 | Sonnet / Opus | **停，全文审** |
| **5** Related Work + 参考文献 | 收集 `% CITE:`；`main.bib`；含 §5.2 的 ARX / Hankel-DMD / Korda–Mezić 一组 | 等用户开放引用 | — |

R0 → R1 → R2 的依赖是硬的：Tier 2 从 contract 派生（A1），实验章引用 `sec:` 标签与 m1–m5 依赖 Method 定稿。**不要**先写 03a–03d 再回来重写 Method。

---

## 5. Method 重写规格（Phase R1 的核心）

### 5.1 开篇（1 段，≤ 4 句）
先建模再控制：一次提示—回复交互是动力系统的一步 → 在验证器读数的延迟坐标上辨识一个命令条件化的 Koopman 动力学模型 → 用该模型对候选指令做预测式选择 → LLM 参数不动，模型离线辨识。图 1 引用：左栏建模、右栏控制。**禁止**以 "We formulate … as feedback control" 开头。节标题：`\section{KOMPAS: Koopman Dynamics Modeling and Predictive Control at the Prompt Step}`（若与 D10 的"acronym 只揭示两次"冲突，则用 `\section{Method}`，首段第一句给 KOMPAS）。

### 5.2 四个小节的 Tier 2 分配（v2.1，按组件切；D25 / D28 / D32 / D33 / D34）

原则：**小节 = 读者要实现的组件**（模型 → 怎么学 → 怎么用 → 保证），标题是功能短语；定义作段落；训练在主文；Why 段每节至多一个。

| 小节 | 段落（页预算） | 公式（contract id） | Why 段（至多一个）| 机制 / 假设 / claim |
|---|---|---|---|---|
| **3.1 A Koopman Dynamics Model of Prompt–Response Interaction**（0.85 页） | P1 直觉：下一轮读数 = 历史带来的 + 指令推来的 + 常数；P2 **状态窗口作段落**：主张先行（单个读数分不清两个相反趋势，**不能**以 "For example" 开头）→ 式 memory_state → 两句 gloss；P3 式 koopman_operator + **紧接一句**：延迟坐标 $s_t$ 就是观测量字典，下面的线性模型是该算子在这个字典上的有限维近似；P4 式 prompt_protocol（命令坐标从哪来）；P5 式 koopman_model 的**线性形式**，只含 $s_t,K,B,b$（$\widehat s_{t+1}=Ks_t+Bc_t+b$，$\widehat y=C\widehat s$）；P6 **lifting 解释段**（规格见 §5.3，D33）：白话 + 类比 + 一句"本文汇报恒等 lifting，学习的 lifting 作为对照在 Table 1"；P7 Why | eq_memory_state, eq_koopman_operator, eq_prompt_protocol, eq_koopman_model（线性形式为主；一般形式只在 P6 以文字提及，公式在 A1） | **建模论证**：有限维线性算子凭什么近似多轮属性序列。正面回答"这不是 ARX 吗"：数值上等价于带外生输入的延迟回归；Koopman 视角多给三样东西——算子作用于观测量所以字典可换（lifting 是同一模型的推广而不是另一个模型）、多步 rollout 有闭式、成熟的 MPC 工具链。Why 段无公式（Rule 22），句式不得以 "X is standard, so…" 开头 | m1, as_1（由 §4.2 检验）, as_2（改为"延迟坐标上仿射"，由 rq2b 检验）, m4（只说"为什么 lifting 可能不帮忙"，数字留 §4.2）；choice_1 与 choice_2 同住本节 |
| **3.2 Learning the Model from Interaction Data**（0.4 页） | P1 三元组 $(s_t,c_t,s_{t+1})$ 构造：重叠窗口；固定目标轨迹的命令标注规则（从 A1 现有第一段搬来）；P2 **式 identification** + gloss + 一句"参数量以十计，几十到几百条轨迹够用"（兑现小样本可辨识）；P3 覆盖性一句：训练轨迹中命令—历史对的覆盖决定推理时可靠范围（从现 3.3 末段搬来）；P4 一句：学习 lifting 的变体用编码器代替恒等映射、三项损失联合训练，见 A1 | **eq_identification（新）** | 无 | c8 的前置（模型离线得到、在线不更新）；nc_7（无 term-removal 消融，audit_only） |
| **3.3 Predictive Instruction Selection with Replanning**（0.65 页） | P1 Why；P2 候选滚动（式 rollout，$s$-空间：$\widehat s_{t+h\mid t}(c)=K\widehat s_{t+h-1\mid t}(c)+Bc+b$）；P3 闭式（式 closed_form_rollout）+ "评估一个候选是一次矩阵乘与一次读出，不再调用 LLM"；P4 代价与选择（式 selection）；P5 **Algorithm 1**：观察 $y_t$ → 滑窗得 $s_t$ → 对每个 $c\in\mathcal{C}_t$ 闭式滚动并算 $\widehat J_t(c)$ → $c_t^\star=\arg\min$ → $u_t=\mathcal{P}(x,\mathcal{H}_t,c_t^\star)$ → 执行、读 $y_{t+1}$ → 重复；P6 执行只用第一步、重规划从真实读数出发；常命令假设只在假想滚动内 | eq_rollout, eq_closed_form_rollout, eq_selection | 打分不开模型（D19 版保留，句式与 3.1 Why 不同） | m2, m3；c8 |
| **3.4 What Prediction Accuracy Buys**（0.25 页，独立短节，用户定） | P1 直觉一句：预测越准，选出的命令越接近事后最优；P2 `\begin{proposition}` **Proposition 1**：假设一步残差与读出残差一致有界（文字陈述，符号 $\varepsilon_{\mathrm{dyn}},\varepsilon_{\mathrm{rec}}$），则 $J_t(c_t^\star)\le\min_{c}J_t(c)+2\delta_J$，其中 $\delta_J$ 由 $\Delta_h$ 沿时域加权累积；式 prediction_bound 与 selection_bound 合并为 Proposition 内的两行；P3 一句证明思路（代价界围绕 argmin 用两次）+ "完整假设与证明见 A6"；P4 边界一句：不保证候选集含可达 $r$ 的命令，不保证闭环收敛 | eq_prediction_bound, eq_selection_bound（合并入 Proposition）；eq_residual_assumptions, eq_cost_bound → A6 | 无 | c9（strong / theoretical） |

页预算合计约 2.15 页 + 图 1 约 0.35 页。**"被窗口化的是验证器读数"不再是 Why 段**，前移到 §2.1（D32，Phase R2 执行）。

**eq_identification 的规格**（进 `equation_lattice`）。直觉句：线性特例下未知量只有 $K, B, b$，且都线性进入下一步预测，所以拿记录的三元组最小化一步误差是普通最小二乘。式子：

$$
(\widehat K,\widehat B,\widehat b)=\arg\min_{K,B,b}\sum_{t}\bigl\|\,s_{t+1}-(K s_t+B c_t+b)\,\bigr\|_2^2
$$

gloss：$s_t$ 记忆状态、$c_t$ 该步命令、$s_{t+1}$ 记录的下一步记忆状态；求和跑遍训练轨迹的重叠窗口三元组（三元组构造见 A1）。`operation_name`: "least-squares fit of the transition, command column and bias to recorded one-step transitions"。它兑现故事框架 §3.1 的"小样本可辨识"需求。

### 5.3 术语与符号规则（塞进 Tier-3 prompt）
- 主文公式符号：$x, p_\theta, \mathcal{H}_t, u_t, o_t, y_t, V, r, e_t, c_t, \mathcal{C}, \mathcal{C}_t, \mathcal{P}, s_t, L, C, K, B, b, H, h, w_h, \widehat J_t, c_t^\star, \Delta_h, \delta_J, \varepsilon_{\mathrm{dyn}}, \varepsilon_{\mathrm{rec}}$。**主文公式里不出现** $D_\psi, g_\psi, d_\xi, \xi_t, L_g$；$E_\phi$ 只在 3.1 P6 的 lifting 解释段以行内符号出现一次。
- 3.4 的 Proposition 以 $s$-空间陈述；原 $\xi$-空间版本、Lipschitz 常数 $L_g$ 与 $\kappa$ 进 A6。
- 术语：instruction（$u_t$）、command（$c_t$）、reading / attribute（$y_t$）、inference-time；禁 test-time、intervention、alignment。
- Why 段（3.1、3.3 各一个）句式不得相同；每个 Why 段以 `\uline{}` 答句收尾（Rule 18）。3.2、3.4 无 Why 段。

**lifting 解释段的规格（D33，3.1 P6，4–6 句，无公式）。** 目标读者：没读过 Koopman 文献的 ML 审稿人。必须依次包含：
1. **它是什么**（一句）：lifting 是一次坐标变换——不直接在原始窗口 $s_t$ 上写线性模型，而是先把 $s_t$ 映到一组学习出来的特征 $E_\phi(s_t)$，在特征空间里写同样的仿射一步。
2. **为什么有人这么做**（一句）：非线性的一步演化，换到合适的特征坐标后可以变得接近线性，就像单摆的角度演化是非线性的、但换到正弦/余弦坐标后接近线性。类比可换成"给线性回归选特征，让直线能拟合弯曲关系"，二选一，不要两个都写。
3. **本文的位置**（一到两句）：本文汇报的实例是**恒等 lifting**，即不做坐标变换、模型直接写在读数窗口上；带学习 lifting 的变体作为对照出现在 Table 1，训练方式在 A1。
4. **为什么可以不做**（一句，m4 的机制，不带数字）：读数本身是一维标量，窗口已经很低维，学习出来的特征没有多少额外结构可暴露。

这段之后，全文再提 "lifting" 不再解释；A1 开头用一句话回指 3.1 P6。冷审用 §7.3 的定向问题 (3) 检验这段是否达标。

### 5.4 `contract.yaml` 改动清单（Phase R0）
1. `narrative_arc` beat 4 拆两句（建模一句 / 控制一句），原 beat 5 顺延。
2. `design_choice_lattice.choice_1`：`why_keyword` 改为 "stated as an observability fact in Problem Formulation §2.1; no Why paragraph"（D32）；`home: method_3.1_P2`。
3. `design_choice_lattice.choice_2`：改名、`rq_id: rq2b`（加 `secondary_rq: rq1`）、`why_keyword` 换为 §5.2 建模论证、`finding_keyword` 换为 c4 + c2、`intro_bullet_keyword` 换为 "a Koopman dynamics model on delay coordinates whose linear instance ranks first on four of seven tasks and is not beaten by a learned lifting"、`conclusion_takeaway_keyword` 换为 "a linear operator on a short window of readings is enough; the lifting can stay off"；`home: method_3.1`。
4. `design_choice_lattice.choice_3`：`home: method_3.3`，`notes` 记 Algorithm 1 归此节。
5. `equation_lattice`：新增 `eq_identification`（`in_section: method`，`home: 3.2`，`rhs_symbols` 填全：s_t, c_t, s_{t+1}, K, B, b）；`eq_residual_assumptions`、`eq_cost_bound` 改 `in_section: appendix`；`eq_prediction_bound`、`eq_selection_bound` 加 `env: proposition`；`eq_koopman_model` 的 `operation_name` 改为线性形式表述，`notes` 记一般形式在 A1。
6. `symbol_lattice`：$D_\psi, g_\psi, d_\xi, \xi_t, L_g, \kappa$ 的 `appears_in` 改为 appendix；$E_\phi$ 的 `appears_in: [method_inline_once, appendix]` 并加 `gloss_required: true`（D33）；新增可行集 $\mathcal{Y}$（`status: pending_metric_definition`）。
7. `claim_ledger.c9`：`home` 改为 method_3.4 Proposition 1 + appendix_A6 证明。
8. 新增 `terminology_bans`（D29）与 `terminology_glosses`：`lifting` → 3.1 P6 一次性解释（D33）。
9. `intuition_lattice.as_1.tested_by: rq2a` 不变；`as_2`（隐空间仿射）改为"在延迟坐标上仿射"，`tested_by: rq2b`；`priors` 加一条 `prior_5`：lifting 的单摆 / 特征选择类比（D33 的类比来源，标 `analogy`）。
10. `meta` 加 `plan_version: 2026-09-19 v2.1`；`structure.method_subsections: [model, learning, controller, guarantee]`。

Tier-1 冷审只审 diff（persona 同 v1 §6.4），重点三问：三个设计选择的证据分布是否 2 建模 : 1 控制；Method 四个小节名是否读起来像"模型 / 学习 / 控制器 / 保证"而不是控制器论文；lifting 的 gloss 是否在 contract 里有落点。

### 5.5 实验章 relabel pass（Phase 3 开工前，Opus 亲自）
`semantic.md` §Experiments 里所有 `§3.5`、`sec:analysis`、`sec:koopman_operator`、`sec:memory_state` 等引用改到新标签（`sec:model`, `sec:learning`, `sec:controller`, `sec:guarantee`）；4.4 P2 引用改为 "Proposition 1（证明在 A6）"；4.2 P3 的 m4 与新 3.1 P6 的 lifting 解释段对齐，不重复解释 lifting 是什么，只报数字（Rule 10）；4.1 P2 基线行名 "KOMPAS (learned lifting)" 保留（D33），其 setup 句回指 3.1 P6 与 A1；4.4 P3 的成本论证回指 Algorithm 1 的 for 循环。跑一遍 `grep -n "sec:\|eq:\|Proposition\|Algorithm" semantic.md` 核对。

---

## 6. 闸门变化

### 6.1 `mech_audit.sh` 新增第 3 关：术语
grep（大小写不敏感）：`test-time|test time|intervention|alignment cost|\balign(ed|ment)?\b`。命中即 FAIL，除非同一行有 `% GATE-EXEMPT: term <词> -- <理由>`。`05_related_work.tex` 引他人工作名时用豁免。`\begin{aligned}` 排除。

### 6.2 `equation_gate.py`
符号清单来自 contract，所以 §5.4 第 6 条改完后它会自动把 $\xi_t$ 等出现在主文视为违规。新式 `eq_identification` 的 `rhs_symbols` 要填全，否则 gate 查不到。两处扩展（D34）：(a) 识别 `\begin{proposition}…\end{proposition}` 内的公式，直觉句检查放宽为"Proposition 前一段有直觉句"；(b) `\begin{algorithmic}…\end{algorithmic}` 整块跳过符号引入检查，但 Rule 21 的结果数字 grep 仍覆盖它。

### 6.2b lifting 解释段的机械检查
`mech_audit.sh` 对 `02_method.tex` 加一条：首次出现 `lifting` 的段落内必须同时命中 `identity` 与（`pendulum` 或 `feature`）之一，且不含 `$`（Rule 22 级别无公式）。命中失败 → FAIL，提示 "D33 lifting gloss missing"。这是粗筛，达标与否由冷审定向问题 (3) 判。

### 6.3 `fig_symbol_audit.py`
FIG_STATE 已记模块 C 的 7 个未匹配符号（$\mathcal{L}_{rec/lin/pred}$ 等）。D31 定案后：选 (a) 则模块 C 重画并重跑；选 (b) 则脚本对附录图另跑一次（加 `--target appendix/A1_estimator.tex` 参数）。

其余（sentence_gate ≤30 词 / ≤2 转折、Rule 20/21/22 grep、`\cite` 禁用、编译留档 `build/main_<CP>_<date>.pdf`）沿用 v1 §4。

---

## 7. 子代理上下文包（只写与 v1 不同的）

### 7.1 Tier-2 builder（§Method）
v1 §6.1 模板 + 追加：
```
Structure is fixed by WRITING_PLAN_2026-09-19.md §5.2 (v2.1): four subsections named by
component (model / learning / controller / guarantee), paragraph roles and equation ids as listed
there. Subsections are NOT one-per-design-choice: choice_1 and choice_2 both live in 3.1, and the
state window is a paragraph, not a subsection. Do not reproduce the colleague's outline (state /
operator / learning / test-time control / bound). 3.1 carries the only Why paragraph that argues
modeling capacity and must answer the ARX objection in keywords; 3.1 P6 is the lifting gloss
(D33, four moves: what / why / our instance / why it can stay off). 3.2 carries the
identification equation and the data-construction bullets moved from appendix A1. 3.3 carries an
Algorithm 1 bullet listing the loop steps. 3.4 is a Proposition bullet. Every bullet cites
contract ids; as_1 and as_2 must be marked as tested_by rq2a / rq2b.
```

### 7.2 Tier-3 generator（`02_method.tex`）
v1 §6.2 模板 + 追加：
```
Reusable material: the displayed equations in paper/_colleague_backup_2026-09-18/3-method.tex
and in the current paper/sections/02_method.tex may be copied; the "Transition tuples" paragraph
of appendix/A1_estimator.tex moves into 3.2. Their surrounding prose, section order, opening
sentence and Why paragraphs may NOT be copied. Write in the s-space notation of §5.3; the lifted
notation appears only as inline $E_\phi$ once, inside the lifting gloss paragraph (3.1 P6), and
otherwise only in appendix/A1 and A6.
The lifting gloss paragraph (3.1 P6) is written for an ML reviewer who has never read a Koopman
paper: 4-6 sentences, no formulas, in this order: what lifting is (a change of coordinates to
learned features in which the one-step map is closer to affine); one analogy (pendulum angle vs
sine/cosine coordinates, OR choosing features so a straight line fits a curved relation; pick
one); that this paper reports the identity lifting and keeps the learned one as a Table 1
comparison trained as in A1; one sentence on why a learned lifting has little to expose when the
reading is a scalar and the window is short (no numbers).
3.3 must contain an Algorithm 1 (algorithmic environment) with the receding-horizon loop; no
result numbers inside it. 3.4 must state Proposition 1 in a proposition environment, in s-space,
followed by a one-sentence proof idea and a pointer to appendix A6.
Also write paper/sections/appendix/A6_selection_bound.tex (assumptions in xi-space with L_g and
kappa, both bounds, full proof) and rewrite the first paragraph of appendix/A1_estimator.tex to
point at 3.1 P6 (gloss) and 3.2 (linear identification), keeping only the learned-lifting
objective there.
Run paper/tools/mech_audit.sh on all three files (term gate and lifting-gloss check included)
before returning.
```

### 7.3 冷审（Tier 3，`02_method.tex`）
v1 §6.4 persona + 两道定向问题：
```
(1) After reading only §3, would you describe this paper as "a controller" or as "a dynamics
model and a controller built on it"? Quote the sentence that decided it.
(2) Does §3.1 answer "isn't this just ARX with a Koopman label?" If yes, in which sentence; if
no, what is missing.
(3) Adopt, for this question only, the persona of an ML reviewer who has never read a Koopman
paper. After reading §3.1's lifting paragraph once, explain in two sentences what "learned
lifting" means and why this paper does not use it. If you cannot, name the missing piece.
(4) Reading only the four subsection titles of §3, list what you take to be the paper's
contributions. (Expected: model / learning / controller / guarantee. Any answer containing
"stacking readings" or "delay embedding" as a contribution is a finding.)
(5) Can Algorithm 1 be implemented from §3 alone, without the appendix? Name any undefined step.
```

---

## 8. 外部依赖（会阻塞的）

| 依赖 | 阻塞什么 | 处理 |
|---|---|---|
| 同事的**约束满足率定义** | D30 的桥、4.1 P5、Problem §2.2 的闭合 | 用户去要。到位前正文放 `TODO(data)`，不转述 |
| 同事的 seed / 区间 | D3 / D12 / D13 的限定措辞 | 不阻塞写作；到位后更新 `numbers_benchmark.yaml` 的 `ci`，放宽措辞 |
| D31 图 1 模块 C 去向 | Phase R3 | 用户在 R0 签字时一并定 |
| 用户签字 D25–D31 | Phase R1 起步 | R0 结束时停 |

---

## 9. `WRITING_STATE.md` 模板（v2.1 字段）

```markdown
# WRITING_STATE
updated: <date>   by: opus session <id>
plan: WRITING_PLAN_2026-09-19.md (v2.1)
phase: R0 | R1 | R2 | R3 | 3 | 4 | 5    checkpoint: <CP>    status: in_progress | awaiting_user_review | approved | revise
last_commit: <hash> <msg>
artifacts_ready: contract.yaml (signed D20; v2 revision <signed|pending>), semantic.md §Intro (approved CP1), §Method (<status>), §Experiments (draft, relabel <done|pending>), sections/01 (CP1 approved; contribution 2 split <done|pending>), sections/02_problem (D23; bridge <done|TODO(data)>), sections/02_method (<CP2-redo status>), appendix/A6 (<status>)
gates: sentence=<> equation=<> term=<> mech=<> compile=<> (build/main_<CP>_<date>.pdf)
cold_review: <path> (<blocker/major/minor>; triage summary)
user_verdict_on_previous: <...>
next_action: <one step>
open_questions_for_user: (1) metric definition from colleague (2) D31 figure module C (3) ...
```

每 checkpoint 一次 commit，message 形如 `paper R1 (CP2-redo): Method rewritten, awaiting review`。

---

## 10. 首个会话待办（Phase R0，按序）

1. 读本文件、`DECISIONS.md`、`WRITING_STATE.md`；申请删除权限；`git status` 干净（`figs/method_fig/preview/_dbg*.png` 这些未跟踪调试图先加进 `.gitignore` 或让图会话自己处理，不要 `git add -A`）。
2. `DECISIONS.md`：修 D22 重号；追加 D25–D31（状态 `待签字`）。
3. `contract.yaml` 按 §5.4 改；`equation_gate.py` 对当前 `02_method.tex` 跑一次，确认 $\xi_t$ 等现在会被报出来（说明 gate 生效，重写时守得住）；加 proposition / algorithmic 环境处理（§6.2）。
4. `mech_audit.sh` 加术语关与 lifting gloss 粗筛（§6.1、§6.2b）；对现有 sections 跑一次，记录命中（预期：`02_method.tex` 的 test-time、intervention；`sec:analysis` 标题；lifting gloss missing）。
4b. `main.tex` 加 `\usepackage{algorithm,algorithmic}` 与 `\newtheorem{proposition}{Proposition}`（先查 `iclr2027_conference.sty` 是否已定义 theorem 环境，避免重复定义）；空壳编译一次。
5. Sonnet Tier-1 冷审 contract diff；Opus triage。
6. 重生成中文对照 `contract_zh_2026-09-19b.md`。
7. `WRITING_STATE.md`：`phase: R0, status: awaiting_user_review, next_action: user signs D25–D31 and picks D31 → Phase R1 (rebuild Tier 2 §Method)`。commit。**停。**
