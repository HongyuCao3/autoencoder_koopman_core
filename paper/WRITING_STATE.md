# WRITING_STATE

updated: 2026-09-19   by: opus session 01UHUsmU
phase: 3   checkpoint: CP3a-CP3d   status: done (D21：不停下)
last_commit: (见本次提交)
artifacts_ready: 01_introduction / 02_problem / 02_method / 03a-03d / appendix A1 全部有正文
current_target: sections/abstract.tex 与 sections/04_conclusion.tex (Phase 4)
gates: sentence_gate=pass  equation_gate=pass  mech_audit=pass  compile=pass (build/main_CP3_2026-09-19.pdf, 9 页)
cold_review: audit/experiments_T3_2026-09-19.yaml — 0 blocker / 4 major / 5 minor，全部已 triage
user_verdict_on_previous: CP1 approved；D21 全文过完再统一回看
next_action: Phase 4。先做 Intro 回看（只许收窄），再写 Conclusion 与 Abstract
open_questions_for_user: (1) 约束满足率的指标定义仍缺，需向同事要；(2) 另一会话与本会话曾同时改 contract 与 Method，见下

## CP3 做了什么

四节一次生成（4.1 设置 / 4.2 预测 / 4.3 控制 / 4.4 分析），三道闸门全过，9 页。

### 冷审逐个复算了九组数字，全部与账本一致

控制两个增益、两个对 RE-Control 的差、两个最小单子任务边际（都落在 IFBench UWC）、22 格横扫、
4/7 排名及第二第三、5/7 记忆必要性及两个例外、CEFR 的 +0.599、LSTM 二比二的分列、
命令通道的 +0.0015 与 +0.0061。冷审还记下一个坑：拿表里**四舍五入后**的 Avg. 相减会得 5.3，
和正文的 5.2 对不上，那是舍入陷阱不是错误。

### 四条 major 的处理

| 编号 | 问题 | 处理 |
|---|---|---|
| f3 | **实质错误**：4.1 写成"Qwen3-4B-Instruct-2507 是两条线共用的模型"，但那只是建模线；控制线报的是 Qwen3-4B 与 Qwen3-8B 两个目标模型、且是同事跑的。这让读者无法判断 Table 2 一半的数字从哪来 | 拆成三句：建模线单模型、第二 backbone 未跑所以 Table 1 全部来自那一个模型、控制线另算且由同事提供 |
| f1 | 排名那条 claim 没有 driver 句（Rule 1） | 补上 contract 里 c4 已写好的 driver |
| f2 | 选择界的作用域没写：它只管单步保持命令的比较，不管重规划后的闭环 | 正文与收尾答句都补上这个边界 |
| f4 | 只给了加窗后的计分行数，没有轨迹数，也没写区间的重采样协议 | 都从账本的 `test` 字段追回：七列是 40–100 条轨迹，区间是按轨迹配对的 grouped bootstrap、2000 次重采样 |

五条 minor 也都改了（补证据 id、给 TMPC/RE-Control 留 `% CITE` 位、删一处重复免责、
改掉一个歧义习语和一个没解释的行话）。

### 顺手修的一件事

生成器为了绕过闸门，把 Skill_H 那个公式**不加 label**（因为 contract 里没有它的条目）。
这是拿掉温度计而不是退烧。正确做法是补 contract：新增 `eq_skill` 与符号 `s_ellnull`，
公式加上 `\label{eq:skill}`，并改写成 §2.1 建模目标里那个 $\ell_h$ 的求和——
这样"§2 声明的目标"和"§4 报告的指标"在纸面上就是同一个东西。

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
