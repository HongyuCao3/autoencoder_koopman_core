# WRITING_STATE

updated: 2026-09-19   by: opus session 01UHUsmU
phase: 2   checkpoint: CP2   status: done (D21：不停下，直接进 Phase 3)
last_commit: (见本次提交)
artifacts_ready: contract.yaml (签字 D20), semantic.md §Introduction + §Method (both approved), sections/01_introduction.tex (CP1 approved), sections/02_method.tex + sections/appendix/A1_estimator.tex (CP2)
current_target: sections/03a_setup.tex (Phase 3 起点)
gates: sentence_gate=pass  equation_gate=pass (12 公式)  mech_audit=pass  compile=pass (build/main_CP2_2026-09-19.pdf, 7 页)
cold_review: audit/method_T2_2026-09-19.yaml (0/3/3) 与 audit/method_T3_2026-09-19.yaml (0/2/2)，全部 triage
user_verdict_on_previous: CP1 approved（D21：全文过完再统一回看）
next_action: Phase 3。先为整章 Experiments 建 Tier 2（四个小节一次建完），再逐节生成
open_questions_for_user: (1) c3 两列失利的机制解释是否要在 CP3b 之前查

## CP2 做了什么

方法章是**改造**不是重写：同事原稿五节重组为五节，加三个 Why-X，训练目标移进附录。

| 项 | 结果 |
|---|---|
| 结构 | 3.1 问题形式化 / 3.2 有限记忆状态 / 3.3 命令条件化算子 / 3.4 预测式候选选择 / 3.5 误差与选择。原"有限记忆 Koopman 动力学"拆成 3.2+3.3，因为它带两个设计选择而 Rule 24 只允许一个 Why-X |
| 三个 Why-X | 按 D19 论证实例化：为什么被窗口化的是验证器读数 / 一条指令凭什么有命令坐标 / 为什么打分不打开模型。每个都先承认底层机制是标准做法再转折 |
| 训练目标 | 连同 eq:training_transition、eq:training_objective 迁入 sections/appendix/A1_estimator.tex，含重叠窗口的元组构造与固定目标下的命令标注规则 |
| 修复 | 图注从 "Caption" 改成一句话点四个阶段；删 actually；补五处首次引入符号的解释 |

### 冷审 triage（两轮 0 blocker / 5 major / 5 minor）

Tier-2 五条已处理（训练数据构造没落点、3.5 超页、两处角色标签不在规范词表、Why-X 两条 bullet 实为一段、预测器冻结未写）。

Tier-3 两条 major 都是**符号没定义**，而且是 contract 的 symbol_lattice 里就缺：
- `$R$`（代价界里的那个 2R）从未定义，审稿人核不了 c9 的推导。
- `$\mathcal{C}$`（命令值域）从 3.1 用到 3.4 才在 3.4 定义它的子集。

两个都补进了 `symbol_lattice`，并挂到 `eq_prompt_protocol` / `eq_cost_bound` 的 `rhs_symbols` 上，
**所以 equation_gate 从此会自己守住它们**。另加两处：附录三个 λ 权重的解释、一句里 bounds 指两个东西。

### 工具改动

`mech_audit.sh` 的副词关加了声明式豁免：同一行写
`% GATE-EXEMPT: adverb <词> -- <理由>` 才放行，并在 PASS 时报出豁免条数。
之前脚本自己说 uniformly 是精度性白名单、正则里却照抓，PASS 名不副实。
现在全文只有一条豁免（"uniformly over the candidate rollouts"，数学量词）。
3.5 收尾那句以副词开头的答句改写掉了，不需要豁免。

## 新会话怎么用这个文件

读 `WRITING_PLAN_2026-09-18.md` §0 的五件事，然后**只做** `next_action` 指向的那一步。
做完更新本文件、commit、停下等用户。不要跳过 checkpoint 连写两章，不要在用户未审核的章节上继续。

## Phase 0 进度明细

| 步 | 内容 | 状态 |
|---|---|---|
| 0.1 | 归档旧负结果稿；建新目录布局；main.tex 重写；空壳编译 | ✅ d1bc477 |
| 0.2 | `DECISIONS.md`、`WRITING_STATE.md` | ✅ 4b130aa |
| 0.3 | 三个脚本；两张表生成并单独编译 | ✅ 4b130aa |
| 0.4 | 重建 Tier-1 `contract.yaml`（9 claims / **3** design choices / 31 symbols / 14 equations / 5 mechanisms / **9** non-claims） | ✅ 2026-09-18，09-19 二修 |
| 0.4b | Sonnet Tier-1 冷审 + Opus triage | ✅ 本次 |
| 0.5 | 方法名 KOMPAS（D10） | ✅ 本次 |
| 0.6 | 更新账本、commit、交用户签字 | ✅ 本次 |

## 冷审 triage 结果（audit/contract_T1_2026-09-18.yaml）

| 编号 | 严重度 | 处理 |
|---|---|---|
| f1 命令通道无实验支持 | blocker | D11：`choice_2` 从 rq2b 改挂 rq1；加 `nc_1`（说明五个 core 列上命令是状态的仿射函数，估不出效应）与 `nc_8` |
| f2 头条数字无区间却进 Abstract | blocker | D12：`nc_3.appears_in` 扩到 abstract/intro；Abstract 不带限定词，Intro 带一句 |
| f3 anchor_example 指向不存在的章节 | blocker | `home` 改为 `method_finite_memory_entrance`（本文无独立 Problem Formulation 章） |
| f4 as_3 未受检验却进贡献 | major | 从 `choice_4.intuition_refs` 移除 as_3 |
| f5 choice_3 声称三项损失缺一不可 | major | `why_keyword` 改为设计论证；必要性移入 `nc_7` |
| f6 arc_type 注释说三个 gap 但 arc 只写两个 | minor | 改注释，说明 beat 3 同时承载两个 gap |
| f7 strong 混淆经验与理论 | minor | c4 / c9 加 `strength_basis: theoretical` |
| f8 c3/c7/c8 的 driver 无机制支撑 | minor | c7 的 driver 改为 "no mechanism established"，登记为待查数据问题 |
| f9 operation_name 与 equation_operation 措辞不一致 | minor | 两处统一 |

## Phase 2 开工前要处理的存量问题

同事 `sections/02_method.tex` 跑 `mech_audit.sh` 的结果：句子闸门通过；Rule 20 命中三处副词
（`actually` 一处；`uniformly` 两处，后者是"在候选滚动上一致成立"的数学量词，属精度性白名单，
保留但要在 Phase 2 确认）。Rule 21 / 22 通过。
