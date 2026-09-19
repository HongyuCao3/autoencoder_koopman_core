# WRITING_STATE

updated: 2026-09-18 (Phase 0 执行中)   by: opus session 01UHUsmU
phase: 0   checkpoint: CP0   status: in_progress
last_commit: d1bc477 paper: 归档负结果稿，切到正面口径的新骨架（Phase 0.1）
artifacts_ready: (none yet)
current_target: paper/contract.yaml (Tier 1 重建)
gates: sentence_gate=n/a  mech_audit=n/a  compile=pass (build/main_phase0_2026-09-18.pdf, 5 页空壳)
cold_review: (pending — Tier-1 冷审在 contract 写完后跑)
user_verdict_on_previous: (none — Phase 0 是第一个阶段)
next_action: 完成 0.3 脚本 → 0.4 contract → Tier-1 冷审 → 0.5 方法名候选 → 交用户签字
open_questions_for_user: (1) D10 方法名未定

## 新会话怎么用这个文件

读 `WRITING_PLAN_2026-09-18.md` §0 的五件事，然后**只做** `next_action` 指向的那一步。
做完更新本文件、commit、停下等用户。不要跳过 checkpoint 连写两章，不要在用户未审核的章节上继续。

## Phase 0 进度明细

| 步 | 内容 | 状态 |
|---|---|---|
| 0.1 | 归档旧负结果稿；建新目录布局；main.tex 重写；空壳编译 | ✅ d1bc477 |
| 0.2 | `DECISIONS.md`、`WRITING_STATE.md` | ✅ 本次 |
| 0.3 | `tools/sentence_gate.py`、`tools/mech_audit.sh`、`tools/build_tables.py`；两张表生成并编译 | ⏳ |
| 0.4 | 重建 Tier-1 `contract.yaml` | ⏳ |
| 0.4b | Sonnet Tier-1 冷审 + Opus triage | ⏳ |
| 0.5 | 方法名三候选 | ⏳ |
| 0.6 | 更新账本、commit、交用户签字 | ⏳ |

Phase 0 结束条件：contract 签字、三个脚本可用、`main.tex` 空章节能编译出 PDF。
