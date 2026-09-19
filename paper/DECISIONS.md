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
| D10 | — | **待定**：方法名。见下方"待裁决"。 | skill Rule 6 要求 acronym |

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

### D10 方法名

skill Rule 6 要求给方法一个 acronym。故事框架未命名。Phase 0.5 由 Opus 提候选，用户选定后
写入 `contract.yaml: meta.acronym` 并在上表补一行。

### D11 及以后

用户在 checkpoint 审核时给出的意见，若属于**规则性裁决**，由 Opus 逐条抄进上表；若属于
**具体修改意见**，落在 `audit/user_notes_<CP>.md`，不进本文件。两类都不留在对话里。
