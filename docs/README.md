# 人格漂移闭环控制：文档索引

本目录是 persona-drift Koopman 控制子项目的设计/协议文档集合（代码在
`../persona_drift_control/`）。信息架构参照
[Pytorch-lightning-Hydra-Optuna-MLflow-Slurm-Project-Template-for-Scientific-Research](https://github.com/HongyuCao3/Pytorch-lightining-Hydra-Optuna-MLflow-Slurm-Project-Tempate-for-Scientific-Research)
的文档分类思路（任务定义/方法/实验/文献），用纯 Markdown 实现，不引入 Quarto 等构建工具。

子目录：`task/`（任务选型与背景调研）、`feasibility/`（各条备选方向的可行性分析）、
`protocols/`（数据/通道协议）、`evaluation/`（评价指标与 baseline 清单）、`method/`（实现细节）、
`experiments/`（实验记录）、`references/`（本地 PDF 缓存，已 gitignore）。跨文档的反引号引用
（如 `` `BASELINES.md` ``）是文字引用，不是可跳转链接，不随文件搬移而失效。

## 任务选型（`task/`）

- [task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md](task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md) —
  任务选型候选"Koopman 闭环控制抵抗多轮 prompt 攻击"的分析（任务定义、扰动通道建模、相对
  NBF-LLM/GenCtrl 的 delta、验证顺序）。**调查线已在 `koopman_defense_pilot.md`
  Phase A→I（含 `koopman_case_study_design.md`）完整收尾，2026-09-02 优先级让位给下面的
  `SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md`**，本文档仍是该线的原始任务定义，历史参考价值保留。
- [task/CONTROL_THEORETIC_LLM_RELATED_WORK.md](task/CONTROL_THEORETIC_LLM_RELATED_WORK.md) —
  2025–2026 控制论×LLM 相关工作调研：按时间轴分类、gap 确认、任务选型信号
- [task/KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md](task/KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md) —
  对抗防御这条线（Phase A→I）的 baseline 对比/机制分析总结，Koopman 在这个任务上"work"的
  具体机制（惯性前提+边际效应复利累积+可控性形式化证明+无交互架构的自适应性上限）、拟合出的
  A/B/C/Gramian 矩阵除了选动作还有哪些用途，以及和学术界机制相似的真实 LLM 下游任务（context
  equilibria、sycophancy drift、hallucination snowball 等）的迁移候选分析（含文献）。
- **[task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md](task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md) —
  ★ 当前优先级最高：下一个 Koopman 闭环控制任务候选——sycophancy drift（用户持续反驳下模型
  放弃正确立场）的任务定义、机制映射、执行器/判分/数据源设计。**2026-09-02 更新**：
  SYCON-Bench False Presuppositions 数据已下载核实并 vendor（200 条，MIT 许可）；代码复用
  分析发现 `attack_trajectory.py`/`benign_trajectory.py` 的逐轮循环可抽出共享模块
  （新增 `trajectory_runner.py`，两个既有文件重构为薄封装，行为不变、测试全绿），核心构建块
  （bank/reminder/judge/trajectory/analysis，均含离散翻转事件判据）已实现，CPU 测试
  231 passed。编排层（`sycophancy_screening.py`/CLI/sbatch）和 GPU screening 尚未开始。

## 可行性分析（`feasibility/`）

- [feasibility/STOLFO_ACTIVATION_STEERING_FEASIBILITY.md](feasibility/STOLFO_ACTIVATION_STEERING_FEASIBILITY.md) —
  Stolfo et al. ICLR 2025 激活转向论文的任务适配性分析（任务不照搬、要素可移植：
  通道 C 执行器 + 连续约束 readout）
- [feasibility/LLM_LATENT_STATE_FEASIBILITY.md](feasibility/LLM_LATENT_STATE_FEASIBILITY.md) — 用内部隐状态替代/增广
  Koopman 状态 z_t 的可行性分析（备选方向，暂不默认采用）
- [feasibility/SCRIPTED_USER_TURNS_FEASIBILITY.md](feasibility/SCRIPTED_USER_TURNS_FEASIBILITY.md) — 用预生成脚本
  替代活的 user-simulator LLM 的可行性分析（尝试后放弃，见 `experiments/drift_confirmation_pilot.md`）
- [feasibility/ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md](feasibility/ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md) —
  Alhafni et al. 2024（Personalized Text Generation with Fine-Grained Linguistic Control）
  任务/数据集是否适合本项目 Koopman 框架的核实分析（数据集不适用，特征抽取代码候选可复用）
- [feasibility/OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md](feasibility/OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md) —
  其他开源数据集候选（ContextEcho 长会话数据、多会话人格对话数据集等）与长文本轨迹采集的
  工程加速分析（KV cache 复用、探针前缀共享、横向并行）
- [feasibility/REDMAN_KOOPMAN_TRAINING_INTERPRETABILITY_FEASIBILITY.md](feasibility/REDMAN_KOOPMAN_TRAINING_INTERPRETABILITY_FEASIBILITY.md) —
  Redman (arXiv:2603.19968) 用 Koopman-with-control 解释 RL 训练动力学这篇先例，和本项目
  Koopman-MPC 结合的可行性分三层评估：重新包装本项目自己的模型迭代历史（零成本可做）、
  可控性作为检测信号（成本不高值得试）、真正对应"随训练演化"的设置（需要微调 LLM 或换
  隐藏状态表征，方向性决策，不可行/待定）。

## 数据/通道协议（`protocols/`）

- [protocols/DATA_COLLECTION_PROTOCOL.md](protocols/DATA_COLLECTION_PROTOCOL.md) — 激励数据采集协议：被控对象、
  readout、输入通道、采集前信号探针（gate）
- [protocols/DATA_SOURCES.md](protocols/DATA_SOURCES.md) — 数据来源与候选清单：prompt 库、已有轨迹、待采集数据、
  评价用外部数据
- [protocols/KV_INJECTION_MONITORING.md](protocols/KV_INJECTION_MONITORING.md) — 通道 D（KV 注入）机制与监控协议

## 方法（实现细节，`method/`）

- [method/overview.md](method/overview.md) — 流水线总览：被控对象/控制器/测量/编排四层的
  代码位置索引
- [method/trajectory_generation.md](method/trajectory_generation.md) — agent 与模拟用户
  具体如何对话：每轮流程、长度控制、主题控制、随机性控制
- [method/controllers.md](method/controllers.md) — 控制器（Controller）可插拔接口，已实现
  的 baseline 控制器，Koopman-MPC 与其他 baseline 的扩展点
- [method/koopman_surrogate.md](method/koopman_surrogate.md) — Koopman 代理拟合/评测代码，
  ARX baseline 作为同一套代码的特例，与 baseline 公平对比的具体设计

## 评价与对比（`evaluation/`）

- [evaluation/EVALUATION_METRICS.md](evaluation/EVALUATION_METRICS.md) — 实验完成后用什么指标判断成败
- [evaluation/BASELINES.md](evaluation/BASELINES.md) — 待对比的 baseline 清单（控制器层、代理建模层）及对应论文

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
- [experiments/pressure_screening_pilot.md](experiments/pressure_screening_pilot.md) —
  上面那条补充分析提出的直接验证——把对抗任务的"渐进升级施压"设计移植到人格域，测 Qwen3-4B
  是否会表现出可测漂移。**新开一次对话想知道"这个 pilot 结果如何"，看这份文档。** 状态：
  **中间态,未证实也未证伪**——N=4（p=0.29）、扩样本到 N=12（p=0.18）方向都偏负（10/12 负
  斜率）但始终不显著；判断为"连续 0-1 rubric 统计功效不够",不是"施压无效"，不建议在当前
  测量设计上继续扩样本。
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
  优势还未被证明,潜在的自适应性优势也还未被验证"。**最新状态（Phase H→I，见
  `koopman_case_study_design.md`）**：加状态-动作交互项后 Phase H 真实闭环学到的自适应方向
  反而有害（在最需要保护的轨迹上少提醒）；根因定位到 `modeling/dataset.py` 的一处训练对
  时序错位 bug（"v 对齐"），修复后 Phase I 用同一架构不换模型不加数据学出方向正确、更省成本
  的策略（比 Phase E `koopman_mpc` 省 34% 插入次数），Phase H 的两个具名失败轨迹被救回，
  但**仍未在 new-Q1 主判据上打赢 `periodic`**——原因变成结构性的：反应式策略无法在 turn1
  这种"看起来一切正常"的早期轮次抢跑，`periodic` 靠盲打反而能在侵蚀开始前动手。调查线到此
  收尾。机制解读与更多迁移分析见
  `../task/KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md`。
- [experiments/koopman_detection_design.md](experiments/koopman_detection_design.md) — Phase E
  打赢后的支线：让 Koopman 代理模型具备显式"检测"能力（而不只是隐含在选动作过程里）。四个方案
  都已跑完：方案 1（一步预测残差）、方案 3（良性 vs 攻击双 regime 对比）、方案 4（状态里塞入
  攻击文本相似度特征，`ReducedStateConfig.aux_cols` + 新增 `modeling/content_similarity.py`）
  均为负面结果，方案 2（前瞻预警）因方案 1 负结果未执行。文档里说明了这不是 Koopman/线性代理
  方法本身的局限，是这套项目当前状态表示/数据规模能提供的信息量不够，与 `koopman_defense_pilot.md`
  主线分开接续。**2026-09-02 更新**：`koopman_case_study_design.md` 发现的"v 对齐"时序错位
  bug 同样污染了方案 1/3 的底层拟合，修正后重新评估——方案 1 残差与真实骤降的相关系数从
  0.53 涨到 0.86，方案 3 逐轨迹准确率从低于平凡基线（0.475）首次超过（0.613）。**原"信息量
  不够"的结论需要收窄：主要是 bug 的假象，不是真实的信息天花板**，方案 2（前瞻预警）值得
  重新考虑。方案 4（内容相似度特征）的负结果诊断与对齐无关，仍然成立。
- [experiments/koopman_case_study_design.md](experiments/koopman_case_study_design.md) — Phase G 打平
  `koopman_mpc`/`periodic` 后留下的开放问题（自适应性优势未证明也未证伪）的 case 分析：五类
  目标现象全部离线复现完毕，结论更负面——`koopman_mpc` 在有真实状态的每次决策（32/32）都选择
  提醒，与状态、horizon 无关；根源是这套线性 Koopman/ARX 模型里动作项和状态无交互项，MPC 的
  最优动作在结构上就是一个和 `z_t` 无关的常数。开放问题因此有了更明确的答案：不是数据不够看出
  自适应优势，是当前模型架构决定了它不可能自适应。加状态-动作交互项后离线证实可以修好这个
  架构限制（margin 与状态相关系数从 0 变 1.0），但真实闭环 Phase H 学到的方向反而有害。
  **2026-09-02 更新（Phase I）**：定位到根因是 `modeling/dataset.py` 的一处训练对时序错位
  bug（"v 对齐"：`v` 学到的是提醒的残留效应而非当轮直接效应），推翻了 Phase H"需要更好标定/
  更多数据"的结论。修复后不换模型不加数据、同一 `nu=1,mu=2` 架构学出方向正确、更经济的策略，
  Phase H 的两个具名失败轨迹被救回，但仍未在 new-Q1 主判据上打赢 `periodic`——反应式策略
  结构上无法在 turn1 抢跑，这是控制器架构选择问题，不是能靠修 Koopman 模型解决的。状态：
  **调查线完整收尾**。
- [experiments/lstm_baseline_plan.md](experiments/lstm_baseline_plan.md) — 补齐 `BASELINES.md`
  ③层"LSTM 一步-多步预测模型"这个 ablation 缺口的实现计划（只有 ARX vs `richer_abs_sign`
  两个同结构线性模型对比过,还没有真正的非线性/长记忆模型跑过对比）。记录了两种设计取舍
  （固定窗口 vs 隐状态跨轮次持续演化，推荐后者）、`Predictor` 接口怎么接、训练/评测口径怎么
  和现有 Koopman 两件套对齐、参数量/数据规模两个混杂因素怎么处理。**状态：仅有计划，尚未开始
  实现。**
- [experiments/sycophancy_screening_pilot.md](experiments/sycophancy_screening_pilot.md) — ★
  正在做：`SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md` 第八节步骤 2 的 screening（SYCON-Bench
  False Presuppositions 回放 + 三分类 judge + 连续斜率/离散翻转事件双判据）。**新开一次对话
  想知道"这个 screening 现在跑到哪一步了"，看这份文档。** 状态：job 15483493 跑完
  （2026-09-02，20 items × 2 seeds）。人工审计发现并修好了离散判据的统计设计 bug（原检验对
  任意非零翻转数都几乎必然"通过"）；修复后是干净空结果——连续/离散两套判据都不显著，
  只有 2/40 条轨迹出现过翻转。也发现了自评偏差的一个具体案例（agent 和 judge 是同一模型，
  对有争议 ground truth 无法互相纠错），已加诊断但未根治。下一步待定：扩样本/换独立
  judge/换更强施压设计三选一。
- [experiments/dose_response_pilot.md](experiments/dose_response_pilot.md) —
  步骤 2，安全方向 steering（diff-in-means 方向 + 残差流 hook）的单轮 α 剂量-响应扫描。状态：
  工程全链路已验证跑通，但 new-Q2 **两次都不过**——v1 直问有害目标撞天花板（p=0.0563）；
  v2 换成步骤 1 里真实"已部分被攻破"的对话上下文重测，天花板问题解决了但响应噪声大、不单调
  （p=0.4535），根因假设指向校准点（短单轮 prompt）和应用点（数千 token 深层上下文）不匹配，
  或单层 steering 压不过已建立的多轮上下文；两次都不是代码问题。**用户已决定这一轮不再追加
  channel C 新实验**，改走 `koopman_defense_pilot.md` 的 channel A（提醒注入）路线，本文档
  不再是活跃开发线。
