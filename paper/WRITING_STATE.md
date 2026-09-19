# WRITING_STATE

updated: 2026-09-19   by: opus session 01UHUsmU
phase: 1   checkpoint: CP1   status: awaiting_user_review
last_commit: (见本次提交)
artifacts_ready: contract.yaml (Tier 1, 用户 2026-09-19 签字 D20), semantic.md §Introduction (approved CP1), sections/01_introduction.tex (CP1 草稿)
current_target: sections/01_introduction.tex
gates: sentence_gate=pass  equation_gate=pass (0 公式)  mech_audit=pass (全部五关)  compile=pass (build/main_CP1_2026-09-19.pdf, 6 页)
cold_review: audit/intro_T2_2026-09-19.yaml (0/2/2) 与 audit/intro_T3_2026-09-19.yaml (0/3/2)，全部已 triage
user_verdict_on_previous: contract 已签字（D20）
next_action: 等用户对 CP1 的审核结论（approved / revise:<cosmetic|structural> / redo）；approve 后进 Phase 2（Method 改造，先建 Tier 2）
open_questions_for_user: (1) CP1 是否通过；(2) c3 两列失利的机制解释是否要在 CP3b 之前查

## CP1 做了什么

六步走完：建 Tier 2 → 冷审 Tier 2 → 生成 Tier 3 → 机械闸门 → 编译 → 交用户审核。

| 步 | 产出 | 结果 |
|---|---|---|
| B | `semantic.md` 的 §Introduction 块，5 段（motivation / gap / intuition / bridge / contributions） | Sonnet 建，Opus 收 |
| 审 Tier 2 | `audit/intro_T2_2026-09-19.yaml` | 0 blocker / 2 major / 2 minor，全部 triage |
| C | `sections/01_introduction.tex` | Sonnet 写，两道闸门自查通过后交回 |
| 闸门 | `mech_audit.sh` 五关全过 | 句子、公式、em-dash、副词、`\cite` |
| 编译 | 6 页；Introduction 约 0.85 页，**低于 1.0–1.2 的预算** | `build/main_CP1_2026-09-19.pdf` |
| Tier-3 冷审 | `audit/intro_T3_2026-09-19.yaml` | 0 blocker / 3 major / 2 minor，全部 triage |

### 冷审 triage（两轮合计 0 blocker / 5 major / 4 minor）

| 编号 | 严重度 | 处理 |
|---|---|---|
| T2 f2 基准名没经过 contract | major | 把 IFBench/IFEval/COLLIE 及其 11 个子任务写进 `meta.benchmarks`，并在 `c5.notes` 注明；来源是 `numbers_benchmark.yaml` 的 `benchmark` 字段与同事表标题 |
| T2 f3 P5 有超页风险 | major | P5 压成"关键词 + 一个头条数字"，D12 的口径改为列表后**一条共享尾句**，不再每项重复 |
| T2 f1 P1 的对冲措辞像在预写 nc_9 | minor | 改为"用对冲动词写在同一从句里，不另起免责句" |
| T2 f4 角色标签偏离规范 | minor | 改为 motivation / gap / intuition / bridge / contributions |
| T3 f1 贡献 3 的 id 注释不自洽 | major | 该行补上 KOMPAS 侧的 11 个 id，使这一行能独立核算 |
| T3 f2 "超过两个白盒控制器 5.2 点"口径不准 | major | 5.2 是对 **RE-Control** 的差；TMPC 的差是 6.136。改为"超过两者中更强的那个 5.2 点"，与 contract 的 `finding_keyword` 一致 |
| T3 f3 "it" 有两个先行词 | major | 拆成两句，把 14.5 点明确挂在**线性特例**上（D2 的关键区分） |
| T3 f4 / f5 跨段回指与序数回指 | minor | 两处改写；三个要求不再用"第一/第二"回指，改成点名 |

### 冷审独立复算了两个头条数字

- 14.5 点：mean(n_bench_012–022) − mean(n_bench_001–011) = 41.373 − 26.900 = **14.473** ✓
- 5.2 点：41.373 − mean(n_bench_034–044) = 41.373 − 36.127 = **5.245** ✓

### 留给后面的

- Introduction 只用了约 0.85 页，比预算少。Phase 4 的"Intro 回看"只允许收窄，所以这点余量留给 Related Work。
- 标题在 ICLR 模板里折成三行且 "BEHAVIOR" 被断字。不影响评审，CP4 若要调只能收窄不能加主张。
- `c3` 两列失利仍无机制解释（Rule 1 挂账），CP3b 之前处理。

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
