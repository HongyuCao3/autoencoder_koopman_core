# 命名基准表

后续所有文档、以及论文的 Tier 1 契约（`docs/article/PAPER_EXECUTION_PLAN.md` Step 4 的
keyword lattice）都以这份表为准。

| 代号 | 中文名 | 英文名 | 指什么 | 状态 |
|---|---|---|---|---|
| `core` | 核心 Koopman 建模 | controlled Koopman modeling of LLM output trajectories | 根目录 `src/koopman_ae/`，8 个标量/多变量轨迹任务，`ABLATION_STUDY.md` 八阶段消融 | ✅ **收尾**（Table 1 的 core 七列已落盘，无待执行步骤）。**2026-09-15 补记**：`core` 的「控制量」是跟踪误差 $r-y_t$，它是状态的**精确仿射函数**（八任务全体 $R^2=1.0$，$[Z\|R\|1]$ 秩亏），且每步指令是上一步误差的确定性函数——**这批数据不能用于估计动作的独立效应或规划余量**，它记录的是一个手写逐维静态控制器的闭环轨迹。见 [`experiments/core_multiobjective_planning_headroom.md`](experiments/core_multiobjective_planning_headroom.md) |
| `defense` | **抗攻击** | multi-turn attack resistance / safety-erosion defense | 多轮越狱攻击下的安全侵蚀与 channel-A 提醒注入防御；Phase A→J | ⛔ **已关闭（2026-09-13）**——闸门 R1 判 FAIL（激活投影留一攻击 −0.0354），按事前签死的第二行关线 |
| `stance` | **抗压力** | stance holding under sustained pushback / sycophancy resistance | 用户持续反驳下模型是否放弃正确立场；MMLU "Are You Sure?" 数据源 | ⏸ **挂起**——执行器权威两次空结果（SYCON、MMLU） |
| `benign` | 良性代价对照 | benign helpfulness cost | Phase F 的 MT-Bench 良性会话代价对照 | ⛔ 随 `defense` 关闭 |
| `detect` | 检测支线 | Koopman-based regime detection | 一步预测残差 / 双 regime 对比 / 内容相似度特征 | 收尾 |
| `constraint` | **约束保持** | multi-turn constraint retention | **数据集：SEQUOR**（`resources/sequor/`，deep-spin/SEQUOR，COLM 2026，`tuples/3` regime）。多轮约束遵守下的预算内提醒再注入，k=3 分级读出 | ✅ **收尾**（2026-09-12：S3 闭环三臂跑完，判**干净负结果**——主量 +0.0014 ± 0.0145 (n=3)，0.05×MDE） |
| `gsm8k_sharded` | **分片指令可靠性侵蚀** | multi-turn reliability erosion under sharded instructions | **数据集：Laban et al. 2025 的 sharded instructions**（`microsoft/lost_in_conversation`，MIT；600 条里筛出的 103 条 GSM8K 子集，vendored 成 `resources/ergo_gsm8k_sharded.jsonl`）。执行器是 reset（重述已揭示的 shard）；相位 E0–E6 / R0–R5 / EK0–EK2 | ⏸ **挂起**（2026-09-08：算子辨识成立，但输入通道与状态解耦，闭环退化为固定日程） |
| `tsar_cefr` | **可读性定点调节** | readability-level regulation via iterative rewriting | **数据集：TSAR 2025 Shared Task**（Readability-Controlled Text Simplification），100 段落 × 目标级 {B1, A2}；读出 = CEFR 分类器**期望级别** $\ell=\sum_k k\,p_k$（非 argmax）+ MeaningBERT 保义分，FKGL 为副读出；执行器 = 逐步改写指令（降一级 / 降半级 / 同义改写 / 不改）；线性 Koopman-MPC 定点调节。死亡条件见 [`experiments/tsar_cefr_kill_criterion.md`](experiments/tsar_cefr_kill_criterion.md) | ▶ **活跃（唯一活线）**（2026-09-13 开线；Phase 1–4 已跑完：GPU-1 辨识 / GPU-2a 反事实树 / GPU-2 闭环五臂 + Table 1 第十一列。预注册的阳性对照未兑现，**2026-09-15 用户裁决「先继续」不关线**；新 GPU 作业仍需独立正面理由）。**2026-09-15 复盘定根因**：任务是**全观测一阶积分器**（每步精确观测 + 动作一步兑现 + `copy` 91% 恒等），贪心已收满 98%，规划余量 $G_{\text{plan}}$ 仅 **0.0130** 对 MDE 0.095——不是算子拟合坏。事前判据由 D-2.5「反馈 vs 开环」换成 $G_{\text{plan}}$「规划 vs 贪心」。现行计划 [`tsar_cefr_postmortem_and_next_tasks_2026-09-16.md`](tsar_cefr_postmortem_and_next_tasks_2026-09-16.md) |
| `backbone2` | **第二 backbone 组** | second-backbone robustness arm for Table 1 | Table 1 的 backbone 轴：`google/gemma-4-E4B-it` 重跑算子质量六行，回答「延迟嵌入有用 / 非线性不额外买东西」是不是 Qwen3-4B 一个模型上的现象。**不是新任务线，不重开任何一条线的科学问题**；`defense` 列按 2026-09-17 用户裁决不进本组。计划 [`experiments/second_backbone_plan_2026-09-17.md`](experiments/second_backbone_plan_2026-09-17.md) | ▶ **计划稿（2026-09-17）**，零 GPU 的 core 协议重建闸门 B0 已判，GPU 作业未提交 |
| `persona_drift` | 人格漂移 | persona drift | 最初的任务线，screening 三问全挂后放弃 | **已放弃，仅作历史术语** |

## 四条消歧说明

1. **「抗压力」= `stance` 线（sycophancy），不是 `pressure_screening_pilot.md`。**
   `docs/experiments/pressure_screening_pilot.md` 记录的是**人格域**的渐进施压 pilot，
   属于已放弃的 `persona_drift` 线，保留其历史名称。两者都叫"施压"但不是同一件事。
2. **`persona_drift` 作为术语只用于两种场合**：指代那条已放弃的实验线本身，或指代
   目录/包名这个历史遗留标识符。**不得**用它描述项目当前在做什么。
3. **目录名与任务名已经脱钩**：`persona_drift_control/` 这个目录现在装的是 `defense` /
   `stance` / `benign` / `detect` 四条线的代码。名字没改的理由见
   [`DOC_CLEANUP_PLAN.md`](DOC_CLEANUP_PLAN.md) §五。

4. **`ERGO` 是上游的方法名，不是本项目的线代号，也不是数据集名。** ERGO
   （arXiv:2510.14077，entropy-guided resetting）提供的是那条线借用的**执行器思路**；
   那条线的数据集是 Laban et al. 的 sharded instructions，所以代号按数据集取
   `gsm8k_sharded`（2026-09-10 用户裁定）。**`gsm8k_sharded` 与 `constraint` 的分界就是数据集**
   ——两条线共用"多轮 + 提醒/重置执行器 + Koopman 辨识"的同一套框架，区别在题库
   （sharded GSM8K vs SEQUOR `tuples/3`）与读出（二值/`closeness` vs 3 通道计数）。
   **既有文档正文里的"ERGO"不改**——那是当时的正确记述（历史陈述不改）；
   **顶层身份表述与新文档用 `gsm8k_sharded`**。代码与文件名里的 `ergo_*`
   （`ergo_math_trajectory.py`、`analyze_ergo_*.py`、`ergo_gsm8k_sharded.jsonl`、
   `run_ergo_*.sbatch`、`ergo_multiturn_reliability_pilot.md`）同样**不改**，
   理由与 `persona_drift` 包名同类，见 [`../.claude/code.md`](../.claude/code.md) *历史包名例外*。
