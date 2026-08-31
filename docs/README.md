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
  功效放大 pilot：job 状态、scripted-user 方案尝试失败的完整记录、结论待补。
- [experiments/surface_features_backfill.md](experiments/surface_features_backfill.md) — CPU-only
  分析：对 signal_screening_pilot 已完成的文本回填免费表层特征并重跑漂移检验，
  `avg_word_len` 测出统计显著的下降趋势（`y_probe` 未测到），`num_tokens` 方向一致但未显著。
- [experiments/adversarial_screening_pilot.md](experiments/adversarial_screening_pilot.md) —
  ★ 当前正在做：`ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md` 第 7 节步骤 1 的 screening
  （攻击序列回放 + LLM-judge 安全打分 + 渐进侵蚀/自相关检验）。**新开一次对话想知道"这个
  任务现在进展到哪一步了"，看这份文档。** 状态：步骤 1 **通过**（new-Q1/new-Q3 均 p<0.0001，
  20 攻击里 18 个负斜率，与人格漂移任务的完全空结果形成鲜明对比）；步骤 2（安全方向 steering
  剂量-响应扫描）进行中。
