# 命名基准表

后续所有文档、以及论文的 Tier 1 契约（`docs/article/PAPER_EXECUTION_PLAN.md` Step 4 的
keyword lattice）都以这份表为准。

| 代号 | 中文名 | 英文名 | 指什么 | 状态 |
|---|---|---|---|---|
| `core` | 核心 Koopman 建模 | controlled Koopman modeling of LLM output trajectories | 根目录 `src/koopman_ae/`，8 个标量/多变量轨迹任务，`ABLATION_STUDY.md` 八阶段消融 | 活跃 |
| `defense` | **抗攻击** | multi-turn attack resistance / safety-erosion defense | 多轮越狱攻击下的安全侵蚀与 channel-A 提醒注入防御；Phase A→J | 活跃（主线） |
| `stance` | **抗压力** | stance holding under sustained pushback / sycophancy resistance | 用户持续反驳下模型是否放弃正确立场；MMLU "Are You Sure?" 数据源 | 活跃 |
| `benign` | 良性代价对照 | benign helpfulness cost | Phase F 的 MT-Bench 良性会话代价对照 | 活跃（`defense` 的附属） |
| `detect` | 检测支线 | Koopman-based regime detection | 一步预测残差 / 双 regime 对比 / 内容相似度特征 | 收尾 |
| `persona_drift` | 人格漂移 | persona drift | 最初的任务线，screening 三问全挂后放弃 | **已放弃，仅作历史术语** |

## 三条消歧说明

1. **「抗压力」= `stance` 线（sycophancy），不是 `pressure_screening_pilot.md`。**
   `docs/experiments/pressure_screening_pilot.md` 记录的是**人格域**的渐进施压 pilot，
   属于已放弃的 `persona_drift` 线，保留其历史名称。两者都叫"施压"但不是同一件事。
2. **`persona_drift` 作为术语只用于两种场合**：指代那条已放弃的实验线本身，或指代
   目录/包名这个历史遗留标识符。**不得**用它描述项目当前在做什么。
3. **目录名与任务名已经脱钩**：`persona_drift_control/` 这个目录现在装的是 `defense` /
   `stance` / `benign` / `detect` 四条线的代码。名字没改的理由见
   [`DOC_CLEANUP_PLAN.md`](DOC_CLEANUP_PLAN.md) §五。
