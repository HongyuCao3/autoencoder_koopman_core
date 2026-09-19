# WRITING_STATE

updated: 2026-09-19   by: opus session 01UHUsmU
phase: 0   checkpoint: CP0   status: awaiting_user_review (contract 第二轮修订中)
last_commit: (见本次提交)
artifacts_ready: contract.yaml (Tier 1, 冷审后已 triage), tools/{sentence_gate.py,mech_audit.sh,build_tables.py}, tables/{tab_control,tab_prediction,tab_prediction_small}.tex
current_target: (等用户签 contract)
gates: sentence_gate=pass (02_method.tex)  mech_audit=fail 3 处副词 (留给 Phase 2)  compile=pass (build/main_phase0_2026-09-18.pdf)  evidence_check=pass (9 claims)
cold_review: audit/contract_T1_2026-09-18.yaml — 3 blocker / 2 major / 4 minor，全部已 triage，见下
user_verdict_on_previous: (Phase 0 是第一个阶段)
next_action: 用户裁决 D19（四个设计选择如何并成三个）→ 重生成中文对照 → 签字 → Phase 1
open_questions_for_user: (1) D19 合并方案；(2) c3 两列失利的机制解释是否要在 CP3b 之前查

## 2026-09-19 第二轮修订（用户反馈七条）

| 反馈 | 处理 | 裁决 |
|---|---|---|
| 标题只强调建模，要用 dynamics modeling 且必须提 control | 标题与 acronym 展开同步改写 | D14 |
| 控制线 claim 不需要更强，之后会同步数据 | 保持 scoped | D13 |
| 叙事第一句"轨迹会漂"要给候选成因 | 改为"指令要和之后累积的每一轮竞争"；加 `prior_4` 与 `nc_9` | D15 |
| 四个设计选择太多，只够讲三个 | **待裁决**，用户要求先做学界共识扫描 | D19 |
| claim 改成先预测线再控制线 | 重排并重编号：c1–c4 预测、c5–c8 控制、c9 桥；实验章 03b 预测 / 03c 控制 | D17 |
| 不声称的不要显式写进正文 | 全部改为 `mode: audit_only` + `candidate_explicit`；D3/D12 的点估计口径改挂 `c5.notes` / `c6.notes` | D16 |
| 主表要两张不是三张 | Table 1 建模、Table 2 控制；小样本表明确标为附录 A2 | D18 |
| 每个公式前要直觉、后要变量解释 | 新增 `tools/equation_gate.py`，接进 `mech_audit.sh` 第 2 关 | — |

`contract_zh_2026-09-18.md` 已过期（claim 编号、标题、不声称语义都变了），D19 定了之后一次性重生成。

## equation_gate 对同事 Method 的实测

14 个显示公式全部有"公式前直觉句"。5 处**首次引入却没有紧跟解释**的符号，留给 Phase 2：
`o_{t+1}`（eq:interaction）、`\mathcal{K}_{c}`（eq:koopman_operator）、`\xi_t`（eq:koopman_model）、
`w_h` 与 `\widehat{J}_t`（eq:selection）、`\varepsilon_{\mathrm{rec}}`（eq:residual_assumptions）。

## 新会话怎么用这个文件

读 `WRITING_PLAN_2026-09-18.md` §0 的五件事，然后**只做** `next_action` 指向的那一步。
做完更新本文件、commit、停下等用户。不要跳过 checkpoint 连写两章，不要在用户未审核的章节上继续。

## Phase 0 进度明细

| 步 | 内容 | 状态 |
|---|---|---|
| 0.1 | 归档旧负结果稿；建新目录布局；main.tex 重写；空壳编译 | ✅ d1bc477 |
| 0.2 | `DECISIONS.md`、`WRITING_STATE.md` | ✅ 4b130aa |
| 0.3 | 三个脚本；两张表生成并单独编译 | ✅ 4b130aa |
| 0.4 | 重建 Tier-1 `contract.yaml`（9 claims / 4 design choices / 31 symbols / 14 equations / 5 mechanisms / 8 non-claims） | ✅ 本次 |
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
