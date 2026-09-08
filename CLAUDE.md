# CLAUDE.md — 规则路由

本仓库用分层规则。**优先级从高到低**，冲突时服从更高层：

1. `.claude/global.md` — 硬约束（报告口径 / null 处置 / 环境）。**任何情况都先读。**
2. `.claude/experiments.md` — 开实验、开闸门、提交 GPU 作业
3. 模块级：`.claude/code.md`、`.claude/docs.md`、`.claude/paper.md`

## 按你要动什么读

| 你要做 | 读 |
|---|---|
| 提交 GPU 作业 / 设计闸门 / 开新任务线 | `.claude/experiments.md` |
| 改 `persona_drift_control/src/` 或 `src/koopman_ae/` | `.claude/code.md` |
| 写/改 `docs/**` | `.claude/docs.md` |
| 碰 `paper/` | `.claude/paper.md`（默认：**不要碰**） |
| 报告任何一个数字 | `.claude/global.md` → *报告口径* |

## 通用

- 计划文档是完整规格。**不要凭记忆复述、不要"优化"命令行、不要合并步骤。**
- 计划文档没写到的判断题，**停下来问用户**。不要从单个示例文件推断架构。
- 闸门不过就停下来报告，不要自己想办法绕过去。
- **长实验开工前先给运行时估计**，并写明估计依据的假设（题目数 / seed 数 / 轮数 /
  `max_new_tokens` / 显卡）。估不准就给数量级，并指出缺哪个信息。
- 修改文件前建议先 `git pull`。

术语基准 [`docs/NAMING.md`](docs/NAMING.md)；实验账本 [`docs/LEDGER.md`](docs/LEDGER.md)。
