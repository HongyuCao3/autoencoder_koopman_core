# LLM 多轮行为闭环控制：文档索引

> **索引只作入口，不作内容权威。** 每条只给「状态 · 一句结论 · 结果在哪」，数字与论证一律
> 在目标文档里读。权威序见 [`../.claude/docs.md`](../.claude/docs.md)。新增文档前先读
> [`NAMING.md`](NAMING.md)，任务名以那份为准。

本目录是 Koopman 控制子项目（代码在 `../persona_drift_control/`）的设计/协议/实验文档集合。
纯 Markdown，不引入 Quarto 等构建工具。信息架构参照
[Pytorch-lightning-Hydra-Optuna-MLflow-Slurm-Project-Template-for-Scientific-Research](https://github.com/HongyuCao3/Pytorch-lightining-Hydra-Optuna-MLflow-Slurm-Project-Tempate-for-Scientific-Research)
的文档分类思路（任务定义/方法/实验/文献）。子目录：`task/`（任务选型）、`feasibility/`（方向可行性）、
`protocols/`（数据/通道协议）、`evaluation/`（指标与 baseline）、`method/`（实现细节）、
`experiments/`（实验记录）、`article/`（论文施工图）、`references/`（PDF 缓存，已 gitignore）。
跨文档的反引号引用（如 `` `BASELINES.md` ``）是文字引用，不是链接。

## 五条线的当前状态（2026-09-10）

| 线（代号以 [`NAMING.md`](NAMING.md) 为准） | 状态 | 一句话 | 入口文档（详见下面「实验」一节） |
|---|---|---|---|
| `constraint`（SEQUOR） | ▶ **活跃** | 唯一在跑的线；S1 辨识臂已提交（job 15772956/15772957） | `constraint_retention_plan.md` |
| `gsm8k_sharded`（旧称 ERGO） | ⏸ 挂起 | 算子辨识成立，但输入通道与状态解耦，闭环退化为固定日程 | `ergo_fidelity_restoration_plan.md` |
| `defense` | ⏸ 收尾 | 辨识成立时挂起；读出没量程（独立 judge 天花板 0.91） | `defense_line_redesign_plan.md` |
| `stance` | ⏸ 挂起 | 执行器权威两次空结果（SYCON、MMLU） | `mc_sycophancy_screening_pilot.md` |
| `persona_drift` | ⛔ 已放弃 | screening 三问全挂，仅作历史术语 | `NAMING.md` |

## 工作纪律（先读）

- **[`../CLAUDE.md`](../CLAUDE.md) — ★ 规则路由，每个会话自动加载。** 分层规则，按你要动什么读
  对应模块：`global.md`（报告口径 / null 默认关闭 / 产物谱系）、`experiments.md`（提交 GPU 作业、
  闸门、仪器配额）、`code.md`（复现路径与守卫）、`docs.md`（权威序、计划 ≤150 行）、`paper.md`。
  **计划文档不要重抄这些规则，引用路径即可。**
- **[LEDGER.md](LEDGER.md) — ★ 实验账本：一行一个 GPU 作业。** 含 2026-08-26→09-08 的 98 个作业
  回填基线（仪器 47 / 方法 48 / 基建 3）。配额规则：最近 10 个作业里"仪器"过半就停下报告。
- **[NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) — ★ 负结果归档：一行一条，含"是否可被新设定推翻"。**
  设计新设定前先读它，判哪些负结果自动失效、哪些仍成立。闸门判 FAIL 的同一次 commit 追加一行。
- [DOC_CLEANUP_PLAN.md](DOC_CLEANUP_PLAN.md) — 命名基准 + 残留修正 + 结构归位方案（Phase 0–3/5 已执行；
  Phase 4 包名重命名**明确不做**，理由见其 §五，2026-09-10 用户裁定为"历史包名例外"）。

## 仓库级文档（`../`）

根目录四份文档属于 `core` 任务（`src/koopman_ae/`：8 个轨迹任务的受控 Koopman 建模），
与本目录的子项目文档分属两套体系：

- [`../README.md`](../README.md) — 仓库总入口：`core` 怎么跑、数据集怎么来、子项目定位。
- [`../CODE_DESIGN.md`](../CODE_DESIGN.md) — `core` 的代码级设计。
- [`../ABLATION_STUDY.md`](../ABLATION_STUDY.md) — `core` 自己的八阶段消融记录。
- [`../DATASETS.md`](../DATASETS.md) — `core` 的数据集说明（清单 `../DATASET_MANIFEST.csv`）。

## 论文写作（`article/`）

- **[article/PAPER_EXECUTION_PLAN.md](article/PAPER_EXECUTION_PLAN.md) — ★ 跨会话施工图。**
  主线锁定 positive / `prior_injection`（Koopman 先验作归纳偏置，再用算子做控制与安全干预）；
  覆盖叙事弧、诚实性红线、Step 0–7 与 I1–I4 的目标/产物/出口闸门、智能体分层、启动语。
  **分工到新会话执行某一步时，先让那个会话读对应 Step。**

## 任务选型（`task/`）

- [task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md](task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md) —
  `defense` 线的任务定义、扰动通道、判据与验证顺序。Phase A→I 已收尾；Phase J 起的新设定沿用本文的任务/通道/判据。
- [task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md](task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md) —
  `stance` 线的任务定义、机制映射、执行器/判分/数据源设计（SYCON-Bench 已 vendor；结果见 `experiments/` 两份 screening）。
- [task/MULTITURN_RESET_INTERVENTION_EVIDENCE.md](task/MULTITURN_RESET_INTERVENTION_EVIDENCE.md) —
  ▶ 设计新激励臂前的文献核查：reset 类介入被独立验证有效（Qwen3-8B 66.0→72.8），
  文献效应量 +0.07~+0.18 → **新臂 MDE 必须 ≤0.07**。落地约束 C1–C6 见其第五节。
- [task/KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md](task/KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md) —
  `defense` 线 Phase A→I 的机制分析与迁移候选（含文献）。
- [task/CONTROL_THEORETIC_LLM_RELATED_WORK.md](task/CONTROL_THEORETIC_LLM_RELATED_WORK.md) —
  2025–2026 控制论×LLM 相关工作调研：时间轴分类、gap 确认、任务选型信号。

## 可行性分析（`feasibility/`）

- [feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md](feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md) —
  `gsm8k_sharded` 线的开线依据：执行器权威有外部论文支持，但读出结构与本项目管线不匹配；贡献点收窄为"诊断已知有效执行器的动力学形状"。
- [feasibility/SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md](feasibility/SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md) —
  `stance` 线的五个前置条件核对。**结论：AE 不需要**（离散立场下 Koopman 算子就是转移矩阵）；"惯性"一条已按 ground truth 审计收窄。
- [feasibility/SAFETY_SPEC_DECOMPOSITION_FEASIBILITY.md](feasibility/SAFETY_SPEC_DECOMPOSITION_FEASIBILITY.md) —
  把动作空间从时间轴扩到内容轴（注入规范的哪一部分）。方向对，但四道闸门未过；**仅分析，未实现**。
- [feasibility/LLM_LATENT_STATE_FEASIBILITY.md](feasibility/LLM_LATENT_STATE_FEASIBILITY.md) —
  用内部隐状态替代/增广 $z_t$（备选方向，暂不默认采用）。
- [feasibility/PROMPT_EMBEDDING_STATE_FEASIBILITY.md](feasibility/PROMPT_EMBEDDING_STATE_FEASIBILITY.md) —
  用文本 embedding 编码 prompt 替换/增广 $z_t$ 的简记。**未实现，用户表示还需再考虑。**
- [feasibility/STOLFO_ACTIVATION_STEERING_FEASIBILITY.md](feasibility/STOLFO_ACTIVATION_STEERING_FEASIBILITY.md) —
  Stolfo et al. ICLR 2025 激活转向的任务适配性：任务不照搬，通道 C 执行器与连续读出可移植。
- [feasibility/REDMAN_KOOPMAN_TRAINING_INTERPRETABILITY_FEASIBILITY.md](feasibility/REDMAN_KOOPMAN_TRAINING_INTERPRETABILITY_FEASIBILITY.md) —
  Koopman-with-control 解释训练动力学这条先例的三层评估（重新包装自身迭代史 / 可控性作检测信号 / 真正随训练演化）。
- [feasibility/OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md](feasibility/OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md) —
  其他开源数据集候选与长轨迹采集的工程加速（KV cache 复用、前缀共享、横向并行）。
- [feasibility/ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md](feasibility/ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md) —
  Alhafni et al. 2024 的数据集**不适用**，特征抽取代码可复用。
- [feasibility/SCRIPTED_USER_TURNS_FEASIBILITY.md](feasibility/SCRIPTED_USER_TURNS_FEASIBILITY.md) —
  用预生成脚本替代活的 user-simulator（尝试后放弃，记录见 `experiments/drift_confirmation_pilot.md`）。

## 数据/通道协议（`protocols/`）

- [protocols/DATA_COLLECTION_PROTOCOL.md](protocols/DATA_COLLECTION_PROTOCOL.md) — 被控对象 / readout /
  输入通道 A–D / 采集前 gate 的定义。**通道定义与 gate 原则被后续所有线沿用**；文中具体采集规模属已放弃的 `persona_drift` 线。
- [protocols/DATA_SOURCES.md](protocols/DATA_SOURCES.md) — 数据来源与候选清单（清单仍在用，框定已过时）。
- [protocols/KV_INJECTION_MONITORING.md](protocols/KV_INJECTION_MONITORING.md) — 通道 D（KV 注入）机制与监控协议。**从未实现。**

## 方法（实现细节，`method/`）

- [method/overview.md](method/overview.md) — 被控对象/控制器/测量/编排四层的代码位置索引。
- [method/trajectory_generation.md](method/trajectory_generation.md) — 每轮对话流程、长度/主题/随机性控制。
- [method/controllers.md](method/controllers.md) — Controller 可插拔接口、已实现的 baseline、Koopman-MPC 扩展点。
- [method/koopman_surrogate.md](method/koopman_surrogate.md) — 代理拟合/评测代码，ARX 作为同一套代码的特例。

## 评价与对比（`evaluation/`）

- [evaluation/BASELINES.md](evaluation/BASELINES.md) — 待对比 baseline 清单（控制器层、代理建模层）及对应论文。
- [evaluation/ASR_METRIC_DESIGN.md](evaluation/ASR_METRIC_DESIGN.md) — 二值 ASR 的判定口径草案，
  两个 judge 互相矛盾后升格为**第三方仲裁者**（per-attack 预注册证据规则，只覆盖 5/8 攻击）。**设计草案，未执行。**
- [evaluation/EVALUATION_METRICS.md](evaluation/EVALUATION_METRICS.md) — ⛔ 2026-08-28 为 `persona_drift` 线写的
  草案 v0.1，**不是现行判据清单**（现行判据在 `task/*_FEASIBILITY.md` 与各 `experiments/*`）；当指标设计方法论与文献出处读。

## 外部输入

- [next_step_diagnosis.md](next_step_diagnosis.md) — 用户 2026-09-02 提供的外部独立诊断（不是本项目实验记录）：
  三节建议均已执行（v 对齐已修 / 预算约束设定 = Phase J / 扩 seed 与效应量主指标）。
  **结论：扩 seed 没买到分辨力**——天花板是 judge 只有 5 个取值的读出，不是样本量。

## 实验（`experiments/`）

格式：`文档 — 状态 · 一句结论 · 结果在哪`。状态图标：▶ 活跃 / ⏸ 挂起或收尾 / ✅ 已执行完 /
📊 结果档案（冻结）/ 📝 设计稿未执行 / ⛔ 已放弃线。

### `constraint` 线（▶ 活跃）

- **[experiments/constraint_retention_plan.md](experiments/constraint_retention_plan.md) —
  ▶ **当前唯一的活计划**（2026-09-08 立项）· S0–S3：SEQUOR 多轮约束遵守 + 预算内提醒再注入，
  `y` 是 3 条验证通道的计数（4 档），动作有时机也有**方向** · 结果见 `constraint_results.md`。**
- [experiments/constraint_signal_screening.md](experiments/constraint_signal_screening.md) —
  ▶ 活跃 · 开线前信号筛查（S0-0）与**死亡条件**；§十 是判据修订记录 · 闸门数字见 `constraint_results.md`。
- [experiments/constraint_results.md](experiments/constraint_results.md) —
  📊 结果档案 · `constraint` 线全部结果（S0 / S0-0 三闸门 / 保真度臂 / S1 试点与辨识臂），从 LEDGER §三·五 拆出。

### `gsm8k_sharded` 线（分片指令，旧称 ERGO；⏸ 挂起）

- [experiments/ergo_fidelity_restoration_plan.md](experiments/ergo_fidelity_restoration_plan.md) —
  ⏸ 挂起 · Koopman 相位 EK0–EK2：算子辨识成立，但**输入通道与状态解耦**，闸门下闭环退化为固定日程；
  熵状态轴是轮次斜坡的代理（衰减 97%）。R 相位已结案（R1 通过，权威 +0.1810） · 结果见该文档与下一条。
- [experiments/ergo_multiturn_reliability_pilot.md](experiments/ergo_multiturn_reliability_pilot.md) —
  📊 结果档案 · 执行器权威确认且效应量大（60-item 0.3667→0.7917，p=6.39e-9），但惯性性质是"信息单调累积"
  不是"记忆黏滞"；Phase C 八臂三道闸门全不过 · 见其「F0–F3」与「G4」两节。

### `defense` 线（⏸ 收尾）

- [experiments/koopman_fast_track_plan.md](experiments/koopman_fast_track_plan.md) —
  ⏸ 收尾（2026-09-08 用户指令）· K1/K2 辨识闸门全过后挂起，**K3 闭环臂不提交**；K4 报告数字暂用自评，
  是 `global.md` 报告口径的**具名例外**（含到期条件） · 第八节。
- [experiments/defense_line_redesign_plan.md](experiments/defense_line_redesign_plan.md) —
  ⏸ 收尾 · **这条线的信息结构不支持闭环**：早期观测对 late 无预测力，而攻击身份留一法解释 57.1%/45.3%；
  D1 盲标基线率 9.8%（Wilson [3.9%, 22.5%]）；A1 留一 ridge R²<0 → 路线 A 出局；B1 排除截断混淆 · 第十、十三节。
- [experiments/koopman_defense_pilot.md](experiments/koopman_defense_pilot.md) —
  📊 结果档案 · Phase A→I：`koopman_mpc` 打赢 zero_control/threshold，但**等代价 `periodic` 不输甚至略优**；
  judge 分歧与 256-token 截断两项诊断 · 第七、八节。
- [experiments/koopman_case_study_design.md](experiments/koopman_case_study_design.md) —
  📊 结果档案 · 无状态-动作交互项时 MPC 最优动作结构上与 $z_t$ 无关；`dataset.py` 的"v 对齐"bug 修复后
  Phase I 方向正确、更省成本，但仍不赢 `periodic`（反应式策略无法在 turn1 抢跑）。
- [experiments/koopman_phaseI_policy_closed_form.md](experiments/koopman_phaseI_policy_closed_form.md) —
  📊 结果档案（纯离线）· Phase I 策略化简为一条阈值规则 `y_{t-1} ≤ 0.75 就插`，复算 112 次决策零失配；
  $y^*$ 夹在两个开环稳态之间 → 该策略**结构上必然抖动**。
- [experiments/koopman_detection_design.md](experiments/koopman_detection_design.md) —
  📊 结果档案 · 检测支线四方案：方案 1/3 在修 v 对齐 bug 后**翻正**（残差相关 0.53→0.86；准确率 0.475→0.613），
  方案 4（内容相似度）负结果仍成立，方案 2 未执行 · 文末"v 对齐修正后重新评估"节。
- [experiments/budget_constrained_defense_plan.md](experiments/budget_constrained_defense_plan.md) —
  📊 结果档案 · Phase J（k=1 预算，7 臂）：自适应臂没打赢最优固定臂，臂间只在花掉多少预算上可区分；
  5 seed 复核证明**扩样本必要但不充分**（补到够用约需 31 倍轨迹量） · 第十、十一节。
- [experiments/independent_judge_reactive_rerun_plan.md](experiments/independent_judge_reactive_rerun_plan.md) —
  ✅ 已执行（2026-09-06）· 反应式 5 臂独立 judge 真跑：new-Q1 自评全 `True`、独立全 `False`；
  原"省 86%"是控制器**停摆**不是效率（由 `adaptive_vs_fixed_claim_plan.md` §0.1 推翻） · 第七节。
- [experiments/adaptive_vs_fixed_claim_plan.md](experiments/adaptive_vs_fixed_claim_plan.md) —
  ✅ 已执行（2026-09-07 收尾）· **claim 不成立**：打得过"什么都不做"但打不过等代价随机分配；
  根因是状态变量就是 judge 分本身（`y_probe ≡ y_safety`，leaked metric）；T3 就地终止 · 第十一至十五节。
- [experiments/measurement_validity_plan.md](experiments/measurement_validity_plan.md) —
  ✅ 已收口（2026-09-07）· P 裁决**独立 judge 站得住**（$p_A$=0.25，Wilson [0.142, 0.402]）→ G0–G3 不做；
  G4 判 ERGO 设定退化（`fixed_last` 与 `always_reset` 116/116 逐字相同） · 其"P 已执行"与"G4"两节。
- [experiments/signal_resolution_plan.md](experiments/signal_resolution_plan.md) —
  ✅ 已收口（2026-09-07）· F0 修正 E0 两处测量错误后 RC-B 三条全过（ERGO 不该终止）；F1 固化 `closeness` 读出；
  F4 软 judge RC-1/RC-3 不过 → `defense` 读出族（硬标签/激活投影/软 judge）**封闭** · 第九节是文档状态地图。
- [experiments/continuous_readout_plan.md](experiments/continuous_readout_plan.md) —
  ✅ 已执行 · judge 标签 token 概率作连续读出：G0/G1 全过、**G2 不过**（96% 的行落在 {0,.5,1} 的 ±0.02 邻域），
  这条修法在此 judge/此 prompt 下不成立 · 第 8 节。
- [experiments/lstm_baseline_plan.md](experiments/lstm_baseline_plan.md) —
  📊 结果档案 · 同口径下 LSTM 0.082–0.095 差于 `richer_abs_sign` 0.0684，四个隐层全更差；
  但原记的"接近 2 倍"不成立，差距缩到 0.97–1.07 倍，剩下劣势全由早停口径撑着。
- [experiments/ae_baseline_plan.md](experiments/ae_baseline_plan.md) —
  📊 结果档案 · AE 与线性代理**打平**（held-out rollout MSE 0.0675±0.007 vs 0.068，3 seed）；
  加早停后 `latent_dim=1` 变差，诊断为验证集样本量不足，不是"AE 更差"的证据。
- [experiments/adversarial_screening_pilot.md](experiments/adversarial_screening_pilot.md) —
  ✅ 已执行 · `defense` 线 screening 步骤 1 **通过**（new-Q1/new-Q3 均 p<0.0001，20 攻击 18 个负斜率），
  与 `persona_drift` 线的完全空结果形成对照。
- [experiments/adversarial_screening_thinking_pilot.md](experiments/adversarial_screening_thinking_pilot.md) —
  ✅ 已执行 · 同一 screening 在 Qwen3 **thinking 模式**下的复现重跑（此前所有实验默认 non-thinking，该变量从未被检视）。
- [experiments/reminder_history_interaction_plan.md](experiments/reminder_history_interaction_plan.md) —
  📝 设计稿 · 把 `repeat_penalty` 从超参变成辨识量 $B_3(u\cdot u_{t-1})$。**未执行、无代码改动、无新产物。**
- [experiments/dose_response_pilot.md](experiments/dose_response_pilot.md) —
  ⏸ 已收尾 · 通道 C（安全方向 steering）单轮 α 剂量-响应：new-Q2 两次都不过（撞天花板 / 响应不单调），
  两次都不是代码问题。**用户已决定不再追加 channel C 实验。**

### `stance` 线（⏸ 挂起）

- [experiments/mc_sycophancy_screening_pilot.md](experiments/mc_sycophancy_screening_pilot.md) —
  ⏸ 挂起 · 换 MMLU "Are You Sure?" 数据源后 new-Q3（惯性）首次有干净数据支持（r=0.29–0.42，门槛后）；
  但 **Phase A 执行器权威两次空结果**（每轮插 p=0.704；只从 turn 2 插 差值恰好为 0） · 其"Phase A"节。
- [experiments/sycophancy_screening_pilot.md](experiments/sycophancy_screening_pilot.md) —
  📊 结果档案 · SYCON-Bench 数据源：干净空结果；**自评 judge 单向漏检压掉约 4× 效应量**（报告口径规则的来源）；
  ground truth 审计 8/20 有问题 → 换数据源。方法论教训仍适用。

### `persona_drift` 线（⛔ 已放弃）

- [experiments/signal_screening_pilot.md](experiments/signal_screening_pilot.md) — ⛔ · 采集前信号探针真实规模作业：**三问全挂**，排查结论见其"排查"节。
- [experiments/drift_confirmation_pilot.md](experiments/drift_confirmation_pilot.md) — ⛔ · 10-prompt 功效放大仍是干净空结果；含 scripted-user 方案失败的完整记录。
- [experiments/pressure_screening_pilot.md](experiments/pressure_screening_pilot.md) — ⛔ · 渐进施压移植到人格域：N=12 方向偏负（10/12 负斜率）但始终不显著（p=0.18），判为统计功效不够。
- [experiments/surface_features_backfill.md](experiments/surface_features_backfill.md) — ⛔ · CPU 回填表层特征重跑漂移检验：`avg_word_len` 显著下降，`y_probe` 未测到。

### 归档（只读）

- [experiments/backup/](experiments/backup/) — 🗄 已执行完或规格被取代的计划文档。**不要执行、不要修改、不要引用为当前状态**；
  归档规则与"已知留在里面没改的错误"见 [experiments/backup/README.md](experiments/backup/README.md)。
- [experiments/backup/two_task_success_plan.md](experiments/backup/two_task_success_plan.md) —
  ⛔ 已归档（2026-09-08）· ERGO 半边 E0–E6 结案于 S3、规格作废；防御半边 D0–D3 未被取代但执行停在盲标，
  入口改为 `defense_line_redesign_plan.md` 第十三节。
