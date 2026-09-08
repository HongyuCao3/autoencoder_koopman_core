# 论文规则（`paper/`）

先读 `.claude/global.md`。

## 默认：不要碰 `paper/`

除非用户在本次会话里明确要求，否则**不要新建、修改或删除 `paper/` 下任何文件**。
施工图是 `docs/article/PAPER_EXECUTION_PLAN.md`——**分工执行某一步时，先让那个会话读
本文档的对应 Step**，不要凭记忆复述。

## 进入论文的数字

只有满足 `.claude/global.md` → *报告口径* 的数才能进正文/图/表：
独立重判 × ≥3 seed × `mean ± std (n)`。**不满足就不进，不要加脚注解释。**

## 诚实性红线

`PAPER_EXECUTION_PLAN.md` 已锁定：**"打不赢等代价 periodic"必须进正文。**
负面/空结果不得从叙事里省略。

## 智能体分层

判断类用 **Opus**，正文生成用 **Fable**，抽取执行用 **Sonnet**，机械检查用 **Haiku/脚本**。
闸门准入判断（`.claude/experiments.md` 三行）属判断类，**不下放**。

## 论文空壳先行

新任务线开工前先写它在论文里那段的 3 句话 + 它要填的那张图/表的空壳。
**实验若不填那个空壳，不跑**（`.claude/experiments.md` → *开一条新任务线*）。
