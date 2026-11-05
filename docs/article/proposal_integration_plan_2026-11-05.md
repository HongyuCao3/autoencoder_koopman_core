# Proposal 整合规划：把本项目结果并入 NSF MFAI 草稿（#2 的落地版）

> **状态**：▶ 待与合作者商议（2026-11-05）：本项目侧的图与文字可先做，**动合作者文字的条目一律等 §六 商议结果**。#2 子规格 [`nsf_proposal_revision_plan_2026-10-25.md`](nsf_proposal_revision_plan_2026-10-25.md) 是按旧版 proposal 写的（"Task 1–3"、式 (1.25)/(2.12)、Fig 3 在本版都不存在）；
> 本文件按 **`proposal/Proposal_NSF_MFAI_2026.pdf`（2026-11-05 14:28 UTC 编译，17 页）** 重新对位，替代 #2 的「二、改动清单」。#2 的写作约束（图在前、文字 ≤6 行、首现术语括号白话）照旧。
> 范围：只管**本项目贡献的内容**（§3.2、Thrust A/B/C 的 Prelim 段、§4 一句）。tex 语法/坏引用（"?"、"Section ??"）、§2.2 与 §8 空节**不在本文件范围**。
> 位置记法：「p.N ¶k」= 第 N 页从上往下第 k 个正文段（标题、公式、RQ 行不计）；y 坐标为 PDF 点（页高 792），方便在 PDF 里核对。

## 一、总表

| # | 位置 | 动作 | 图 | 净增页数（估） |
|---|---|---|---|---|
| E1 | p.6 §3.2 ¶4 末（式 3.3 后 "estimated by least squares…" 一句，y≈565） | 加一句记号对照 | — | +0.02 |
| F1 | **p.2 ¶6 的箭头式**（"interaction history → behavioral state → controlled FM dynamics → future behavior"）**替换为 Fig. 1**；p.6 §3.2 ¶5 压到 2 行并引 Fig. 1 | Fig. 1（推荐方案 B，见 §二·Fig. 1） | +0.08 |
| E2 | p.7 §3.2 ¶3（"The experiments also expose limitations…"，y=210–275） | 改写为三条有出处的局限 | 引 Fig. 2b、4a | +0.04 |
| E3 | p.7 §4 ¶1 末（y≈486） | 加一句已做工作 | — | +0.03 |
| E4 | p.9 Thrust A Prelim ¶1（"Our preliminary controlled sentence-generation…"，y=332–425） | **整段替换**，删 MAE 3.778→2.007 | **Fig. 2**（两面板） | +0.18 |
| E5 | p.11 Thrust B Prelim ¶1（"Our preliminary results provide evidence…"，y=177–296） | **整段替换**，删 MAE≈0.0273 / R²≈0.942 | **Fig. 3** | +0.12 |
| E6 | p.11 Thrust B Prelim ¶1 与 ¶2 之间（y≈297） | 新增 4 行 "Evaluation protocol" | — | +0.08 |
| E7 | p.13 Thrust C Prelim ¶1（"Our sentence-length experiments already expose…"，y=96–161） | **合作者原段保留**，段末追加 3 句（2026-11-05 改为只追加不替换） | **Fig. 4**（两面板） | +0.15 |
| E8 | p.13 Thrust C Prelim ¶2（"Thrust 3 will deliver…"，y=163–256）末 | 加一句 future work | — | +0.04 |
| | | | **合计** | **≈ +0.75 页** |

**页数风险（列入 §六.1，与合作者商议）**：正文 p.2–16，p.16 只用到 y=353/792，现约 **14.45 页**，距 15 页上限余 ≈0.55 页，且 §2.2、§8 仍空。
本表 +0.8 页**超支 ≈0.25 页 + 两个空节的篇幅**。可选抵扣：(a) Fig. 1 不加（−0.08）；(b) Fig. 2–4 全用 `wrapfigure` 半栏（每张省 ≈0.05）；(c) 压 p.6 §3.1 ¶4（"We validate these findings…"，y=237–302，≈0.12 页）——不是本项目内容，需你或合作者同意。

## 二、四张图（白底；配色过 `dataviz` 验证；输出 `proposal/figs/*.pdf`）

数据一律取自已落盘、已登记的产物（下列路径），**不重算、不新增数字**；`paper/` 只读。出图脚本放 `docs/article/proposal_figs/make_proposal_figs.py`（新建），不依赖 Windows 端 `make_charts.py`。

**Fig. 1 加不加（2026-11-05 分析）。**

| | 利 | 弊 |
|---|---|---|
| 加 | ① Umesh 的明确要求（多图少字、外行可读）最直接的落实；② 全文 15 页**零张图**，评审略读时没有任何视觉锚点；③ Fig. 2–4 里的 "reading / state / instruction" 都要靠它先定义，否则每个 caption 都得重新解释；④ 能把 p.6 ¶5 的 4 行控制器文字压到 2 行 | ① 页数最紧时再占空间；② 若只画 KOMPAS 闭环（方案 A，放 p.6），图描述的是**已有方法**而非本 proposal 的框架，评审会把项目读成「KOMPAS 续作」，与 Overview 的「行为状态 + 两类干预通道」定位错位；③ §3.1 没有图，两条 prelim 线视觉上失衡 |
| 不加 | 省 ≈0.1 页 | Fig. 2–4 的 caption 被迫各自解释术语；Umesh 的核心要求没落在开篇 |

**建议：加，用方案 B。** 不画 KOMPAS 闭环，而是把 p.2 ¶6 那行文字箭头式换成一张图：interaction history → behavioral state $s_k$ → controlled dynamics (2.5) → future behavior，回路上两个输入口——**prompt 通道 $c^p_k$（实线，标 "realized in preliminary work, §3.2"）**、**activation 通道 $u^{(\ell)}_k$（虚线，标 "proposed, Thrusts A.3/C.2"）**。
这样一张图同时是 proposal 框架图和 KOMPAS 示意（KOMPAS = 只用实线那条路的特例，与 E1 记号对照呼应），替掉一行 display 后净增 ≈0.08 页；§3.2 只需一句 "KOMPAS instantiates the solid path of Figure 1"。
画法：全宽、高 ≈1.1 in，按 `paper-explainer-figure` 的视觉规范；不出现 Koopman/lifting 等术语，框内用白话（"last few measured values of the reply"）。
- Caption: *Figure 1: The proposed behavioral realization. Interaction history is compressed into a behavioral state; a controlled linear model predicts how future behavior responds to two kinds of intervention—natural-language prompts (solid, realized in our preliminary work, Section 3.2) and internal activation changes (dashed, proposed). Look at the two input arrows: the project asks which behaviors each one can reach.*

**Fig. 2（p.9，Thrust A，两面板）记忆买到什么。** 数据：`paper/tables/tab_prediction.tex`（n_main_* 行：Last-turn vs KOMPAS linear）；`paper/figs/_fig_report.json` → `fig_memory.ids_depth`（matched window）。
- (a) 七任务分组柱：Last-turn vs Memory（KOMPAS linear）的 $\mathrm{Skill}_H$，区间不含 0 的实心。(b) 四任务深度折线：$d=1..4$ 个过去读数相对 $d=1$ 的增益。
- Caption: *Figure 2: What interaction memory buys. (a) Multi-step prediction skill (1 = perfect, 0 = best trivial predictor) using only the current reading versus the last few readings, on seven multi-turn tasks; filled bars: 95% paired bootstrap interval excludes zero. (b) Gain from adding past readings when every depth is fit on the same data. Look at (b): on three of four tasks the second reading captures all of the gain.*

**Fig. 3（p.11，Thrust B，单面板，可 `wrapfigure`）学到的 lifting 不加分。** 数据：`fig_mechanism` 面板 (c) 同源（证据 id 见 `_fig_report.json`）。点 + 95% 区间，形状区分 1/2/3 维读数。
- Caption: *Figure 3: Paired difference in prediction skill, linear model minus the same model with a learned nonlinear lifting, on seven tasks (circle: one attribute; square/diamond: two/three attributes tracked jointly). Look at the zero line: the learned lifting is never better with an interval excluding zero, so added representation complexity bought no fidelity.*

**Fig. 4（p.13，Thrust C，两面板）命令效应：要先辨识，辨识了就能算可达范围。** 数据：(a) `results/core_excite_reachability/summary.json`（三 seed，按请求分四档：实测交付均值 ± std、train 拟合模型对 test 题目的一步预测、稳态可达上界 73.0 词水平虚线；对角线 = 完全照做）；(b) `results/core_excite_b_identifiability/summary.json` + `results/b_identifiability/{tsar_cefr,constraint}.json`：三个任务的一步命令效应 $CB$ 点估计 + 轨迹级 95% 区间（`core` 三 seed 各一点）。原 (a)「指令 vs 追踪误差」散点改为正文一句。
- Caption: *Figure 4: Prompt authority, measured. (a) Sentence length: requested versus delivered words when requests are randomized (three seeds); the line is a linear model fitted on training prompts and evaluated on held-out prompts, and the dashed level is its predicted reachable average under requests of at most 100 words. (b) One-step effect of the instruction on the tracked attribute, with 95% intervals, on three tasks where instructions were randomized. Look at (a): the model predicts what the prompt can and cannot reach, and errs on the conservative side at the top.*

## 三、逐句修改（英文为拟写入 proposal 的文本）

**E1（p.6 ¶4 末，追加）** *In the notation of (2.5), this is the special case $\Phi=I$, $A=K$, $B_p=B$, $c^p_k=c_k$ (scalar), $d_k=b$, with no internal channel $u^{(\ell)}_k$.*

**F1（p.6 ¶5 末，追加引用）** 在 "…Candidate evaluation requires no additional queries to the target FM." 后加 *(Figure 1)*。

**E2（p.7 ¶3，整段替换）** 原：*"The experiments also expose limitations… Much of the observed benefit of memory arises from persistent prompt-specific effects rather than temporal dynamics. Moreover, the collected trajectories do not consistently identify independent command effects. Thus, …"*
新：*The experiments also expose limitations that directly motivate the proposed research. First, most of the benefit of memory reflects a persistent prompt-specific level rather than turn-to-turn dynamics: a per-prompt intercept recovers 81% and 67% of the gain on the two tasks where a remainder is detected; elsewhere the remainder is indistinguishable from zero. Second, on the attribute-tracking tasks the instruction was computed from the tracking error, so the input matrix is not identifiable from those data (Figure 4a). Thus, accurate prediction and successful prompt control do not establish that the learned state is dynamically sufficient or that its estimated input matrix characterizes control authority.*
出处：`paper/sections/04b_ablation.tex`（c11，n_mech_061/060=0.807、n_mech_058/057=0.673）；`docs/experiments/command_channel_identifiability_results.md`。

**E3（p.7 §4 ¶1 末，追加）** *Preliminary work has tested interaction-level Koopman modeling on seven multi-turn tasks and three public instruction-following benchmarks, and has located where it falls short of dynamical sufficiency.*
（#2 原稿用 "validated"；KOMPAS 在 7 列中 4 列第一，"validated" 过强，改 "tested … and located where it falls short"。）

**E4（p.9 Thrust A Prelim ¶1，整段替换）** 删整段（MAE 3.778→2.007、46.9%、"Increasing latent dimension…"：合作者结果；删是按 #2 原稿「趋势不带数字」，不是因出处）。新：
*Figure 2 summarizes what interaction memory buys in our preliminary study, where the behavioral state is the last few measured values of a reply attribute (e.g., sentence length) placed side by side—the simplest instance of $\Psi$ in A.1. Memory improves multi-step prediction over the current reading on four of seven tasks and is indistinguishable from zero on the other three (Fig. 2a). When every depth is fit on the same data, a second past reading captures the whole gain on three of four tasks (Fig. 2b). Two limitations remain: memory is set by a fixed delay window rather than the information criterion (5.2), and most of the gain is a persistent per-prompt level rather than turn-to-turn dynamics—the gap between predictive and dynamical sufficiency that RQ1 targets.*
¶2（"Thrust 1 will deliver…"）不动。

**E5（p.11 Thrust B Prelim ¶1，整段替换）** 删 *"normalized rollout MAE ≈0.0273 and R²≈0.942"*（合作者结果；按 #2 原稿「少数字」删）。新：
*Our preliminary results support a restricted form of shared dynamics. Sentence length, character length, and average word length were tracked jointly by one linear Koopman model whose state stacks all three readings; with memory, it is the most accurate of six predictors on that task. Replacing the identity lifting with a learned nonlinear $\Phi$ improved prediction on none of seven tasks (Figure 3), so added representation complexity bought no fidelity—the trade-off that (5.9) makes explicit. These results establish feasibility, not generality: the tasks are closely related and use numerical targets rather than the semantic and internal coordinates of Thrust 1, and better compression need not preserve decision-relevant structure (Section 3.1).*
出处：`tab_prediction.tex` Joint-3 列（linear 0.826 最高）；`04b_prediction.tex` ¶3。

**E6（p.11，¶1 后新增小段）** *Evaluation protocol. We score an $H$-step forecast made without feeding back observed outputs by $\mathrm{Skill}_H = 1-\mathrm{MSE}_H(\text{model})/\mathrm{MSE}_H(\text{best trivial predictor})$, so 0 means no better than a constant or last-value guess. Comparisons are paired over the same held-out conversations with a 2000-resample conversation-level bootstrap (95% intervals), and models with different memory depths are fit on the same transitions so that depth is not confounded with training-set size.*

**E7（p.13 Thrust C Prelim ¶1，合作者原段保留，段末追加）**（2026-11-05 改：不替换；先写好，带成稿去商议）
*In the original collection, however, the instruction was computed from the tracking error, so its effect cannot be separated from the state. When requests are instead randomized, the effect is identified: each additional requested word yields about 0.73 delivered words (three seeds, every 95% interval within 0.69–0.78), and a linear model fitted on training prompts places the reachable average length near 73 words for requests of at most 100 words, within two words of held-out prompts below 80 words and conservative by about seven words above (Figure 4a). The same identification holds on two other tasks with randomized actions (Figure 4b); Thrust 3 therefore begins with persistently exciting data collection before Gramian-based actuator selection (5.13).*
出处：`../experiments/core_excite_b_identifiability_results.md`（门控过：$CB$ = 0.734 ± 0.009 (n=3)；可达性为探索性、无闸门，脚本 `scripts/core_excite_reachability.py`）；`command_channel_identifiability_results.md`。原 E7 稿里的 CEFR Gramian 条件数 1286 一句删去：「秩满但条件数大」需要读者懂 Gramian，且 CEFR 的 $B$ 由 MPC 自身选择的动作辨识，不如 `core` 这例直观。
口径注意：0.73 与 73 词是拟合诊断/模型推算，不是跨臂比较的头条数；只有一个任务、一个模型（Qwen3-4B）、线性拟合——段内不外推。

**E8（p.13 ¶2 末，追加，future work 口径）** *Because predicted outputs are affine in a held command, output-box constraints reduce to linear inequalities with a closed-form feasible set; extending this to commands that switch within the horizon, which a single held command cannot express, is part of C.3.*
（#3 判 future work：只写方向，**不写** 87.6% 这个数，除非你另裁。）

## 四、明确不加的内容

| 内容 | 理由 |
|---|---|
| 谱图 / 特征值叙述（#1） | 闸门判「不过」，事前签「不写进 proposal」；p.3 ¶3 "Spectral analysis can identify dominant behavioral modes" 不动 |
| IFBench/IFEval/COLLIE 柱状图（#2 原稿 `c3`） | `numbers_benchmark.yaml`：同事提供，**无原始日志、无 seed、无区间**；做成图会放大一组无法核对的点估计。p.6–7 现有文字（已注明 point estimates）保留 |
| #4b 激励采集 | 2026-11-05 闸门**过**（$CB$≈0.73，三 seed 区间均不含 0）；是否在 E7 加一句、Fig. 4b 是否加入 `core` 这一列 → §六.9 |
| 负面结果（论文 Fig 2(b)(c)、App. F） | #2 原稿已定本轮不写 |

## 五、用户裁决记录

1. ~~三处无出处数字~~ → 2026-11-05 用户答：合作者结果，来源正常。E4/E5 仍按 #2 原稿删数字留趋势；p.13 那句保留（见 E7）。
2. 页数超支 → 不自行取舍，转 §六.1 与合作者商议（用户 2026-11-05）。
3. Fig. 1 → **采纳方案 B**（用户 2026-11-05）；但它替换的是合作者写的 p.2 文字，替换本身仍列入 §六.2。
4. ~~产物放哪~~ → `proposal/*.pdf` 进 `.gitignore`（用户 2026-11-05）。

## 六、须与合作者商议（agent 不自作主张；本项目只准备材料和备选稿）

| # | 事项 | 涉及谁的文字 | 我们带去的材料 |
|---|---|---|---|
| 1 | **页数**：本计划 +≈0.75 页，余量 ≈0.55 页，§2.2、§8 也要篇幅。候选抵扣：Fig. 2–4 用 `wrapfigure`、压 §3.1 ¶4（p.6，≈0.12 页）、或砍图 | §3.1 是时序编码器线；排版是全文 | 每项的页数估计（§一表）；可先只做 wrapfigure 版看实际页数 |
| 2 | **F1**：用 Fig. 1 替换 p.2 ¶6 的箭头式，并把 p.6 §3.2 ¶5 压到 2 行 | p.2 是 proposal 总述（PI 文字） | Fig. 1 草图 + caption（§二） |
| 3 | **E4/E5 删数字**：p.9 "MAE 3.778→2.007 / 46.9%"、p.11 "MAE≈0.0273, R²≈0.942" | 合作者结果 | 依据是 Umesh「趋势不带数字」；备选：数字保留、只加图 |
| 4 | **E4/E5 整段替换** Thrust A/B 的 Prelim ¶1（E7 已改为只追加，见第 9 条） | 两段原稿均非本仓库所写 | 替换稿（§三）；备选：原段保留，只在段末追加我们的 1–2 句 + 图引用 |
| 5 | **E2** 改写 p.7 §3.2 局限段（加 81%/67% 与 B 不可辨识） | §3.2 是 KOMPAS 段，原稿作者需确认 | 出处已列于 E2 |
| 6 | **E3** 在 §4 Intellectual Merit 加一句；原 #2 稿要求的 Overview 一句不加（p.1 已满） | PI 文字 | 措辞已从 "validated" 降为 "tested … located where it falls short" |
| 7 | **E6** Evaluation protocol 放 Thrust B 末还是 §2.2（现为空节） | §2.2 的归属 | 4 行稿（§三） |
| 8 | 记号：§3.2 用 $(K,B,c_k,b)$、Thrust A 原段用 $\xi,K,B,r,c$，与 (2.5) 的 $(A,B_p,c^p_k,d_k)$ 并存——全文统一用哪套 | 全文 | E1 的对照句 |
| 9 | **#4b 新结果**（E7 追加稿 + Fig. 4 已按成稿设计好）：只在合作者原句后追加、不删字；它是 p.13 合作者「长请求跟不上」的定量版 | Thrust C ¶1 位置、Fig. 4 占位 | E7 成稿 + Fig. 4 设计（§二、§三）；[`../experiments/core_excite_b_identifiability_results.md`](../experiments/core_excite_b_identifiability_results.md) |

商议前可独立做、不碰合作者文字的：出 Fig. 1–4（`proposal/figs/`）与出图脚本；E6、E8 的独立句子稿。
