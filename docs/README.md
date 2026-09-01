# 人格漂移闭环控制：文档索引

本目录是 persona-drift Koopman 控制子项目的设计/协议文档集合（代码在
`../persona_drift_control/`）。信息架构参照
[Pytorch-lightning-Hydra-Optuna-MLflow-Slurm-Project-Template-for-Scientific-Research](https://github.com/HongyuCao3/Pytorch-lightining-Hydra-Optuna-MLflow-Slurm-Project-Tempate-for-Scientific-Research)
的文档分类思路（任务定义/方法/实验/文献），用纯 Markdown 实现，不引入 Quarto 等构建工具。

## 任务与协议

- **[ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md](ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md) —
  ★ 当前优先级最高：任务选型候选"Koopman 闭环控制抵抗多轮 prompt 攻击"的分析
  （任务定义、扰动通道建模、相对 NBF-LLM/GenCtrl 的 delta、验证顺序）**
- [CONTROL_THEORETIC_LLM_RELATED_WORK.md](CONTROL_THEORETIC_LLM_RELATED_WORK.md) —
  2025–2026 控制论×LLM 相关工作调研：按时间轴分类、gap 确认、任务选型信号
- [STOLFO_ACTIVATION_STEERING_FEASIBILITY.md](STOLFO_ACTIVATION_STEERING_FEASIBILITY.md) —
  Stolfo et al. ICLR 2025 激活转向论文的任务适配性分析（任务不照搬、要素可移植：
  通道 C 执行器 + 连续约束 readout）

- [DATA_COLLECTION_PROTOCOL.md](DATA_COLLECTION_PROTOCOL.md) — 激励数据采集协议：被控对象、
  readout、输入通道、采集前信号探针（gate）
- [DATA_SOURCES.md](DATA_SOURCES.md) — 数据来源与候选清单：prompt 库、已有轨迹、待采集数据、
  评价用外部数据
- [KV_INJECTION_MONITORING.md](KV_INJECTION_MONITORING.md) — 通道 D（KV 注入）机制与监控协议
- [LLM_LATENT_STATE_FEASIBILITY.md](LLM_LATENT_STATE_FEASIBILITY.md) — 用内部隐状态替代/增广
  Koopman 状态 z_t 的可行性分析（备选方向，暂不默认采用）
- [SCRIPTED_USER_TURNS_FEASIBILITY.md](SCRIPTED_USER_TURNS_FEASIBILITY.md) — 用预生成脚本
  替代活的 user-simulator LLM 的可行性分析（尝试后放弃，见 `experiments/drift_confirmation_pilot.md`）
- [ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md](ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md) —
  Alhafni et al. 2024（Personalized Text Generation with Fine-Grained Linguistic Control）
  任务/数据集是否适合本项目 Koopman 框架的核实分析（数据集不适用，特征抽取代码候选可复用）
- [OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md](OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md) —
  其他开源数据集候选（ContextEcho 长会话数据、多会话人格对话数据集等）与长文本轨迹采集的
  工程加速分析（KV cache 复用、探针前缀共享、横向并行）

## 方法（实现细节）

- [method/overview.md](method/overview.md) — 流水线总览：被控对象/控制器/测量/编排四层的
  代码位置索引
- [method/trajectory_generation.md](method/trajectory_generation.md) — agent 与模拟用户
  具体如何对话：每轮流程、长度控制、主题控制、随机性控制
- [method/controllers.md](method/controllers.md) — 控制器（Controller）可插拔接口，已实现
  的 baseline 控制器，Koopman-MPC 与其他 baseline 的扩展点
- [method/koopman_surrogate.md](method/koopman_surrogate.md) — Koopman 代理拟合/评测代码，
  ARX baseline 作为同一套代码的特例，与 baseline 公平对比的具体设计

## 评价与对比

- [EVALUATION_METRICS.md](EVALUATION_METRICS.md) — 实验完成后用什么指标判断成败
- [BASELINES.md](BASELINES.md) — 待对比的 baseline 清单（控制器层、代理建模层）及对应论文

## 实验

- [experiments/signal_screening_pilot.md](experiments/signal_screening_pilot.md) — 采集前
  信号探针（协议第 7 节的 gate）真实规模作业的状态记录：job ID、怎么查进度、耗时预估、
  作业结束后该做什么。**新开一次对话想知道"之前那个作业现在怎么样了"，看这份文档。**
  该作业三问全挂，排查结论见文档内"排查"一节。
- [experiments/drift_confirmation_pilot.md](experiments/drift_confirmation_pilot.md) — 上面
  那次 screening 三问全挂之后，为判断"漂移到底存不存在还是样本太小测不出"而做的 10-prompt
  功效放大 pilot：job 状态、scripted-user 方案尝试失败的完整记录、结论（10-prompt 规模下
  仍是干净的空结果），以及 2026-08-31 补充分析——4B 容量不是瓶颈（同模型在对抗任务上有清晰
  信号），换 7B 前应先验证是否为刺激强度问题。
- [experiments/pressure_screening_pilot.md](experiments/pressure_screening_pilot.md) — ★
  正在做：上面那条补充分析提出的直接验证——把对抗任务的"渐进升级施压"设计移植到人格域，
  测 Qwen3-4B 是否会表现出可测漂移。**新开一次对话想知道"这个 pilot 现在进展到哪一步了"，
  看这份文档。**
- [experiments/surface_features_backfill.md](experiments/surface_features_backfill.md) — CPU-only
  分析：对 signal_screening_pilot 已完成的文本回填免费表层特征并重跑漂移检验，
  `avg_word_len` 测出统计显著的下降趋势（`y_probe` 未测到），`num_tokens` 方向一致但未显著。
- [experiments/adversarial_screening_pilot.md](experiments/adversarial_screening_pilot.md) —
  `ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md` 第 7 节步骤 1 的 screening（攻击序列回放 +
  LLM-judge 安全打分 + 渐进侵蚀/自相关检验）。状态：步骤 1 **通过**（new-Q1/new-Q3 均
  p<0.0001，20 攻击里 18 个负斜率，与人格漂移任务的完全空结果形成鲜明对比）。
- [experiments/adversarial_screening_thinking_pilot.md](experiments/adversarial_screening_thinking_pilot.md) —
  上面那次 screening 结果在 Qwen3 **thinking 模式**下的复现重跑（此前所有实验默认跑在
  non-thinking 模式，这个变量从未被检视过）。用 Hydra 管理 `enable_thinking`，避免
  thinking/non-thinking 两次跑的输出目录互相覆盖。
- [experiments/koopman_defense_pilot.md](experiments/koopman_defense_pilot.md) — ★ 当前进展
  最新：把人格漂移这条线的 `control.py`/`modeling/` 几乎零改动复用到对抗防御领域，设计并
  验证 Koopman-MPC 防御控制器。**新开一次对话想知道"Koopman 防御控制器现在做到哪一步了"，
  看这份文档。** Phase A→F 已完整闭环：`koopman_mpc` 打赢 zero_control/threshold 两个基线，
  以更低代价（约 40% 的提醒次数/token）追平 constant_remind，良性 helpfulness 代价也明显
  低于 constant_remind。**Phase G（补齐 `BASELINES.md` 的"周期性重提醒"基线，`PeriodicController`）
  修正了这个结论**：在完全对齐插入次数/token 代价的前提下,不看任何反馈信号的固定周期基线
  在攻击场景主判据和良性代价上都不输给、甚至略优于 `koopman_mpc`——"建模复杂度换来了收益"
  这个说法需要收窄为"koopman_mpc 相对 threshold 有明确优势,相对同代价的 periodic 基线的
  优势还未被证明,潜在的自适应性优势也还未被验证"。
- [experiments/koopman_detection_design.md](experiments/koopman_detection_design.md) — Phase E
  打赢后的支线：让 Koopman 代理模型具备显式"检测"能力（而不只是隐含在选动作过程里）。四个方案
  都已跑完：方案 1（一步预测残差）、方案 3（良性 vs 攻击双 regime 对比）、方案 4（状态里塞入
  攻击文本相似度特征，`ReducedStateConfig.aux_cols` + 新增 `modeling/content_similarity.py`）
  均为负面结果，方案 2（前瞻预警）因方案 1 负结果未执行。文档里说明了这不是 Koopman/线性代理
  方法本身的局限，是这套项目当前状态表示/数据规模能提供的信息量不够，与 `koopman_defense_pilot.md`
  主线分开接续。
- [experiments/koopman_case_study_design.md](experiments/koopman_case_study_design.md) — Phase G 打平
  `koopman_mpc`/`periodic` 后留下的开放问题（自适应性优势未证明也未证伪）的 case 分析：五类
  目标现象全部离线复现完毕，结论更负面——`koopman_mpc` 在有真实状态的每次决策（32/32）都选择
  提醒，与状态、horizon 无关；根源是这套线性 Koopman/ARX 模型里动作项和状态无交互项，MPC 的
  最优动作在结构上就是一个和 `z_t` 无关的常数。开放问题因此有了更明确的答案：不是数据不够看出
  自适应优势，是当前模型架构决定了它不可能自适应。状态：**已完成**。
- [experiments/lstm_baseline_plan.md](experiments/lstm_baseline_plan.md) — 补齐 `BASELINES.md`
  ③层"LSTM 一步-多步预测模型"这个 ablation 缺口的实现计划（只有 ARX vs `richer_abs_sign`
  两个同结构线性模型对比过,还没有真正的非线性/长记忆模型跑过对比）。记录了两种设计取舍
  （固定窗口 vs 隐状态跨轮次持续演化，推荐后者）、`Predictor` 接口怎么接、训练/评测口径怎么
  和现有 Koopman 两件套对齐、参数量/数据规模两个混杂因素怎么处理。**状态：仅有计划，尚未开始
  实现。**
- [experiments/dose_response_pilot.md](experiments/dose_response_pilot.md) —
  步骤 2，安全方向 steering（diff-in-means 方向 + 残差流 hook）的单轮 α 剂量-响应扫描。状态：
  工程全链路已验证跑通，但 new-Q2 **两次都不过**——v1 直问有害目标撞天花板（p=0.0563）；
  v2 换成步骤 1 里真实"已部分被攻破"的对话上下文重测，天花板问题解决了但响应噪声大、不单调
  （p=0.4535），根因假设指向校准点（短单轮 prompt）和应用点（数千 token 深层上下文）不匹配，
  或单层 steering 压不过已建立的多轮上下文；两次都不是代码问题。**用户已决定这一轮不再追加
  channel C 新实验**，改走 `koopman_defense_pilot.md` 的 channel A（提醒注入）路线，本文档
  不再是活跃开发线。
