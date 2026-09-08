# LLM 多轮行为闭环控制：文档索引

> 新增文档前先读 [`NAMING.md`](NAMING.md)，任务名以那份为准。

本目录是 Koopman 控制子项目的设计/协议文档集合（代码在 `../persona_drift_control/`），
覆盖该子项目长出的四条实验线。当前活跃的是**抗攻击**（`defense`）与**抗压力**（`stance`），
最初的人格漂移线（`persona_drift`）已放弃。**2026-09-06 起另有一条正在评估的新候选线——
ERGO/多轮可靠性侵蚀（`experiments/ergo_multiturn_reliability_pilot.md`，见下面`实验`一节）**，
是 `stance` 线两次执行器权威空结果之后开的第三条任务线，带独立的真实数据集（vendor 自
`microsoft/lost_in_conversation` 的 GSM8K sharded 题库），尚未并入 `stance`/`defense`
两条主线的框架。术语基准见 [`NAMING.md`](NAMING.md)。信息架构参照
[Pytorch-lightning-Hydra-Optuna-MLflow-Slurm-Project-Template-for-Scientific-Research](https://github.com/HongyuCao3/Pytorch-lightining-Hydra-Optuna-MLflow-Slurm-Project-Tempate-for-Scientific-Research)
的文档分类思路（任务定义/方法/实验/文献），用纯 Markdown 实现，不引入 Quarto 等构建工具。

子目录：`task/`（任务选型与背景调研）、`feasibility/`（各条备选方向的可行性分析）、
`protocols/`（数据/通道协议）、`evaluation/`（评价指标与 baseline 清单）、`method/`（实现细节）、
`experiments/`（实验记录）、`article/`（论文写作的跨会话施工图）、`references/`（本地 PDF
缓存，已 gitignore）。跨文档的反引号引用
（如 `` `BASELINES.md` ``）是文字引用，不是可跳转链接，不随文件搬移而失效。

## 工作纪律（先读）

- **[`../CLAUDE.md`](../CLAUDE.md) — ★ 规则路由，每个会话自动加载。** 分层规则，按你要动
  什么读对应模块。信息架构借鉴 `~/Pytorch-lightining-Hydra-...-Template` 的 `.claude/` 分层。
  - `../.claude/global.md` — **最高优先级**：报告口径（判分的选择/报告分离、≥3 seed、
    `mean ± std (n)`，以及六条明令禁止的头条数字）、null 默认关闭、同一仪器只修一次、产物谱系
  - `../.claude/experiments.md` — 提交 GPU 作业前五条、闸门准入三行、仪器:方法 配额、
    仪器修复后按谱系筛重跑范围、开新任务线四步
  - `../.claude/code.md` — 复现路径不得改动、守卫不得删除/绕过、回归检验要求
  - `../.claude/docs.md` — **文档权威序**（冲突不自行选边，标出来问用户）、计划文档 ≤150 行、
    历史陈述不改
  - `../.claude/paper.md` — 默认不碰 `paper/`、进入论文的数字口径、诚实性红线、智能体分层
  **计划文档不要再重抄这些规则，引用路径即可。**
- **[LEDGER.md](LEDGER.md) — ★ 实验账本：一行一个 GPU 作业。**
  含 2026-08-26→09-08 的 98 个作业回填基线（仪器 47 / 方法 48 / 基建 3，接近 1:1；
  其中约 11 个"方法"作业是仪器改动后的重跑）。配额规则：最近 10 个作业里"仪器"过半就停下报告。

## 文档整理

- **[DOC_CLEANUP_PLAN.md](DOC_CLEANUP_PLAN.md) — 命名基准 + 残留修正 + 结构归位方案（待执行）。**
  项目实际任务已从"人格漂移"转向**抗攻击**与**抗压力**两条线，文档里留有旧命名。方案的核心
  不是批量替换——75 处中文命中里绝大多数是**正确的历史陈述**，改掉就是篡改记录——而是划清
  "改"与"绝对不改"的边界：Phase 0 建 `NAMING.md` 命名基准表，Phase 1 逐行修正**顶层身份表述**
  （4 个文件、约 20 行），Phase 2 给过期草案加统一状态横幅并标记 `method/` 下 12 处指向已放弃线的
  **僵尸待办**，Phase 3 建立根目录 core 文档与 `docs/` 的双向索引（不搬文件），Phase 5 防复发。
  Phase 4（`persona_drift` 包名重命名，164 个文件）**明确不做**——`outputs/` 下 50+ 份冻结的
  `hydra.yaml` 记着旧 `_target_`，改名会让论文 Step 2 要读的历史产物无法重放。执行者：Sonnet 5。

## 仓库级文档（`../`）

根目录的四份文档属于 `core` 任务（`src/koopman_ae/`：8 个标量/多变量轨迹任务的受控 Koopman
建模），与 `docs/` 描述的 `persona_drift_control` 子项目文档分属两套体系，术语基准见
[`NAMING.md`](NAMING.md)：

- **[`../README.md`](../README.md)** — 仓库总入口：`core` 任务怎么跑、数据集怎么来，以及
  本子项目（`docs/` + `persona_drift_control/`）的定位说明。
- **[`../CODE_DESIGN.md`](../CODE_DESIGN.md)** — `core` 任务的代码级设计文档。
- **[`../ABLATION_STUDY.md`](../ABLATION_STUDY.md)** — `core` 任务自己的消融记录（状态定义/
  AE-vs-线性/训练方式/`latent_dim`/早停等八个阶段）。
- **[`../DATASETS.md`](../DATASETS.md)** — `core` 任务的数据集说明；配套清单见
  `../DATASET_MANIFEST.csv`。

## 论文写作（`article/`）

- **[article/PAPER_EXECUTION_PLAN.md](article/PAPER_EXECUTION_PLAN.md) — ★ 论文执行总流程：
  跨会话的施工图。** 主线已锁定为 positive / `prior_injection`——把 Koopman 这个动力系统-控制论
  先验作为归纳偏置引入 LLM 多轮行为建模，再用拟合出的算子做控制与安全干预。文档覆盖：论文
  叙事弧与纳入/排除范围（本仓库两套证据体——核心 AE-Koopman 八阶段消融 + `persona_drift_control`
  四条实验线——分别担任哪一节）、诚实性红线（"打不赢等代价 periodic"必须进正文）、Step 0–7 与
  I1–I4 每一步的目标/输入/产物/流程细节/出口闸门、每步适用的智能体分层（判断类用 Opus、正文
  生成用 Fable、抽取执行用 Sonnet、机械检查用 Haiku/脚本）、依赖与并行度、可直接粘贴的新会话
  启动语、进度表。**分工到新会话执行某一步时，先让那个会话读本文档的对应 Step。** 写作方法论
  依赖 `~/.claude/skills` 下的 `neurips-write-tiered`/`neurips-write`/`paper-audit`/`paper-pipeline`
  （**当前未安装，Step 0 就是装它**）。

## 任务选型（`task/`）

- [task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md](task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md) —
  任务选型候选"Koopman 闭环控制抵抗多轮 prompt 攻击"的分析（任务定义、扰动通道建模、相对
  NBF-LLM/GenCtrl 的 delta、验证顺序）。**Phase A→I（`koopman_defense_pilot.md` +
  `koopman_case_study_design.md`）已于 2026-09-02 完整收尾，同日
  `SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md` 成为新任务线；但这条线没有关闭——2026-09-03 起的
  Phase J（`experiments/budget_constrained_defense_plan.md`）在预算约束的新评测设定下重开
  `periodic` 对照，用的仍是本文档定义的任务/通道/判据。**
- [task/CONTROL_THEORETIC_LLM_RELATED_WORK.md](task/CONTROL_THEORETIC_LLM_RELATED_WORK.md) —
  2025–2026 控制论×LLM 相关工作调研：按时间轴分类、gap 确认、任务选型信号
- **[task/MULTITURN_RESET_INTERVENTION_EVIDENCE.md](task/MULTITURN_RESET_INTERVENTION_EVIDENCE.md) —
  ▶ 设计新激励臂（Q16）之前的文献核查（2026-09-08）**：reset 类介入在 Qwen3 家族 + 分片数学上
  **被独立验证有效**（CCOPD arXiv:2605.30251 的 Reset-Then-Answer，Qwen3-8B **66.0 → 72.8**），
  并给了我们的机制一个现成术语 **self-anchored drift**。文献效应量区间 **+0.07 ~ +0.18**，
  本项目 R1 的 +0.181 落在区间内——**因此新臂的 MDE 必须 ≤ 0.07（底线 0.15），而现设计是 0.49。**
  三条反证各改一处设计：ERGO 从未在 14B 以下测过且没拆开熵触发 vs 重写；Laban 的
  `RECAP > SNOWBALL` 只有 2 个模型、无误差棒（独立支持关闭 R4）；过度 reset 的语义漂移会反转收益
  （**中间预算不能砍**）。附带一条正面证据：Qwen3-4B 在 GSM8K 上 ECE=0.018 是同类最佳，
  而我们关掉熵读出族是在被污染的 legacy 数据上测的（→ Q17）。落地约束 C1–C6 见其第五节。
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
  （bank/reminder/judge/trajectory/analysis，均含离散翻转事件判据）已实现。**编排层
  （`sycophancy_screening.py`/CLI/sbatch）也已实现，两次 GPU screening（job 15483493 自评
  judge、job 15487325 独立 judge）都已跑完并归档，CPU 全套 238 passed**——结果与解读见
  下面 `experiments/sycophancy_screening_pilot.md` 条目。

## 可行性分析（`feasibility/`）

- [feasibility/SAFETY_SPEC_DECOMPOSITION_FEASIBILITY.md](feasibility/SAFETY_SPEC_DECOMPOSITION_FEASIBILITY.md) —
  用户 2026-09-03 提出的想法：对抗防御线的通道 A 现在是"一次性把整条安全规范加进去"，
  改成让 Koopman 决定**注入规范的哪一部分**（把动作空间从时间轴扩到内容轴）。结论：
  **方向对**——Phase J 的自适应臂之所以只重现了 `fixed_t4`，就是因为 5 轮 × 二值的决策空间
  太小；而且这样 `B` 才第一次是真正的多列矩阵，`ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md`
  第 3 节承诺的 Gramian/可防御性分析才非平凡；全量注入臂逐字等于现有臂，历史数据全部可比。
  **但四道闸门必须先过**：Gate 0 Phase J 10.7 的"扩样本前不加变体"；Gate 1 子句级可辨识性
  （按现有 300 行辨识数据估算，只能分辨"差异 > 全量效应 40%"的子句对）；Gate 2 `y_safety`
  65% 饱和在 1.0、只有 5 个取值，连续读出从"顺带确认"升级成前置刚需；Gate 3 静态话题匹配
  路由这个平凡基线必须先排除，否则又是 `fixed_t4` 重演。**仅分析，未实现、未跑实验。**
- **[feasibility/SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md](feasibility/SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md) —
  ★ sycophancy 这条线在投入 GPU 前的前置条件核对：它能否像 core 任务那样建 Koopman 模型、
  能否像防御线那样闭环、需不需要 AE。**结论：AE 不需要**（经验证据三条同向；且离散立场状态下
  有限状态空间的 Koopman 算子就是转移矩阵本身，精确线性、无需提升，AE 的存在理由不成立）；
  五个前置条件里**执行器权威完全没测**（真正的第一道门）、读出只有 3 个取值且 84.5% 取在
  上限（辨识性问题，2026-09-04 已确认换 token 概率修不好，见下面 `continuous_readout_plan.md`）、
  horizon 被 T=5 硬卡死只剩 1 步 rollout。**"惯性"这条 2026-09-05 更新为"部分满足，证据强度
  需收窄"**——ground truth 审计（见下面 `sycophancy_screening_pilot.md` "追加分析"一节）发现
  支撑它的 new-Q3 信号里有相当一部分来自几个 ground truth 有问题的 item 制造的吸收态序列，
  去混淆后信号强度掉了一个数量级、去掉存疑项后不再显著。建议顺序：~~先零成本把 judge 硬标签
  换成 token 概率拿到连续读出~~（已排除，见 `continuous_readout_plan.md`）→ 执行器权威检查 +
  准备 benign 对照臂（否则"永不改变立场"是平凡最优解）→ 扩样本前先审计新增 item 的 ground
  truth → 先线性建模 → AE 放最后且预期打平。**
- [feasibility/STOLFO_ACTIVATION_STEERING_FEASIBILITY.md](feasibility/STOLFO_ACTIVATION_STEERING_FEASIBILITY.md) —
  Stolfo et al. ICLR 2025 激活转向论文的任务适配性分析（任务不照搬、要素可移植：
  通道 C 执行器 + 连续约束 readout）
- [feasibility/LLM_LATENT_STATE_FEASIBILITY.md](feasibility/LLM_LATENT_STATE_FEASIBILITY.md) — 用内部隐状态替代/增广
  Koopman 状态 z_t 的可行性分析（备选方向，暂不默认采用）
- [feasibility/PROMPT_EMBEDDING_STATE_FEASIBILITY.md](feasibility/PROMPT_EMBEDDING_STATE_FEASIBILITY.md) — 用独立文本
  embedding 模型编码 prompt（或 safety_prompt+prompt）替换/增广 z_t 的简记（`ae_baseline_plan.md`
  讨论后提出，和上面的内部隐状态提案、`koopman_detection_design.md` 方案 4 的 TF-IDF 相似度特征
  都相关但不同。**未开始实现，用户表示还需要再考虑。**）
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
- **[feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md](feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md) —
  sycophancy 线两次执行器权威空结果后（`mc_sycophancy_screening_pilot.md` "Phase A"一节）评估
  的下一个候选：Laban et al. 2025 的 sharded-instruction 多轮退化基准（arXiv:2505.06120，
  CC BY 4.0，`microsoft/lost_in_conversation` 已开源）+ ERGO 的 entropy-guided resetting
  （arXiv:2510.14077）。**执行器权威已被外部论文证实**（相对无干预基线 +56.6% 平均性能，
  5 个模型/6 个任务），比 sycophancy 依赖的"跨执行器类推"更强的证据起点；但**读出结构和本项目
  管线不匹配**（任务分数只在对话结束判一次，不是逐轮 `y_t`）,且"周期性 vs 自适应对比"这个原本
  以为的空档（`KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md` 第五节）经核实**已被 ERGO 自己填过**
  （已在该文档更正）——贡献点收窄为"用本项目的辨识+可控性方法诊断这个已知有效执行器的动力学
  形状"。建议执行顺序把"最小执行器权威验证"排在文字/代码工作量之前的第一步（这次真正吸取的
  sycophancy 教训）。**2026-09-06 追加：最小执行器权威验证已完成，确认执行器有权威，效应量大**
  （job 15602880 zero_control / 15602881 reset，20 items × 2 seeds，配对 t-test：最终轮任务
  成功率 0.275→0.750，t=4.79，p=0.000127）——和 sycophancy 线两次都卡在效应量趋近 0 形成
  鲜明对比。**同日 60-item 扩样本复核（job 15603367/15603368）确认效应量在 3 倍样本下依然稳**
  （0.367→0.792，t=6.78，p=6.39e-9，显著性更强），排除了 20-item 结果是小样本噪声的可能。
  **但惯性/动力学的性质和 sycophancy 不同，不是"记忆黏滞"而是"信息单调累积"**
  （揭示的 shard 越多题目客观上越好答，不是模型"记住了之前的立场"），论文措辞需要区分这两种
  现象。详见 `experiments/ergo_multiturn_reliability_pilot.md`。

## 数据/通道协议（`protocols/`）

- [protocols/DATA_COLLECTION_PROTOCOL.md](protocols/DATA_COLLECTION_PROTOCOL.md) — 激励数据采集协议：被控对象、
  readout、输入通道、采集前信号探针（gate）。**注意这是 2026-08-27 为人格漂移线写的草案 v0.1**：
  其中的具体采集规模（T=16 self-chat、40 prompt × 2 通道 × 4 seed）属于那条已放弃的线，
  但**通道 A–D 的定义、readout 约定、"screening 先于建模"的 gate 原则被后续所有任务线沿用**，
  所以它仍是活文档（全仓库被引用最多的一份），读规模数字时按上述区分对待。
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

- [evaluation/EVALUATION_METRICS.md](evaluation/EVALUATION_METRICS.md) — 实验完成后用什么指标判断成败。
  **⚠️ 这是 2026-08-28 为已放弃的人格漂移线写的草案 v0.1，未随任务转向更新**：它完全不涉及
  后来实际在用的 new-Q1/new-Q3/离散翻转判据（那些定义在 `task/*_FEASIBILITY.md` 与各
  `experiments/*_pilot.md` 里），文中"以本文件为准并更新代码"那句话也已不适用于现在的
  `analysis_adversarial.py`/`analysis_sycophancy.py`。当作**指标设计的方法论参考和文献出处**
  读，不要当作现行判据清单。
- [evaluation/ASR_METRIC_DESIGN.md](evaluation/ASR_METRIC_DESIGN.md) — 二值 ASR（`task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md`
  readout ③）的判定口径草案。该 readout 从项目开始就写在清单里，**从未实现也从未排期**；
  2026-09-03 两个 judge 互相矛盾（`experiments/koopman_defense_pilot.md` 第七节）之后，它从
  "最终评价指标"升格为**第三方仲裁者**（不换模型、不请人标，把"成功"写成可复核的确定性规则）。
  文中记录了三条勘察结论：拒绝串匹配退化（各臂 ASR 96–100%）、编号步骤无分辨力（62.5% 的行
  都有）、**93.5% 的回复被 256 token 上限截断**（最后一条对 judge 分同样成立，是独立于 ASR 的
  发现）。采纳的口径是 per-attack 预注册证据规则，只覆盖 5/8 攻击（3 个 `stance` 类判不了，
  显式标 UNSCORABLE）。**状态：设计草案，未执行、无代码改动、无新产物。** 唯一被后续执行掉的
  是截断那一条：2026-09-04 的离线检查（`experiments/koopman_defense_pilot.md` 第八节）推翻了
  本文 2.3 里"提醒会让回复变短变完整"这个前提，臂间结论因此**不需要**以更大 token 上限重跑；
  同时把 2.3 推论 1（判据只能依赖已经出现的证据）从建议升级为口径的硬约束。
- [evaluation/BASELINES.md](evaluation/BASELINES.md) — 待对比的 baseline 清单（控制器层、代理建模层）及对应论文

## 外部输入

- [next_step_diagnosis.md](next_step_diagnosis.md) — 用户 2026-09-02 提供的**外部独立诊断
  文档**（不是本项目的实验记录，原名 `next step.md`）：针对"periodic 打平 `koopman_mpc`"
  给出的根因假设与三步建议。第一节的"v 对齐"时序错位诊断已被验证并修好（见
  `experiments/koopman_case_study_design.md` 的 Phase I）；**第二节的建议（改成预算约束/
  binding 代价的评测设定）已于 2026-09-03 开始执行，见下面的
  `experiments/budget_constrained_defense_plan.md`（Phase J）**；**第三节（扩 seed、主指标改成
  效应量+bootstrap）也已于 2026-09-03 执行完毕**——主指标早在 Phase J 第一轮就换成了配对效应量
  +bootstrap，seed 于 2026-09-03 从 2 扩到 5（该文档第十一节）。**结论：这一步没有买到分辨力**，
  CI 按 √n 收窄却仍比相邻臂间距宽 2–5 倍，天花板是 judge 只有 5 个取值的读出，不是样本量。

## 实验

- **[experiments/ergo_fidelity_restoration_plan.md](experiments/ergo_fidelity_restoration_plan.md) —
  ▶ **ERGO 线的活计划，也是当前唯一的活计划**（2026-09-08 立项；同日按 **Koopman 贡献过滤器**
  重写为 **Koopman 相位 EK0–EK2**，R 相位已结案）：E2/E3 之后
  ERGO 线定格 S3，而失效被归因到**我们自己的 prompt 结构**，不是模型也不是数据。决定性证据：跨 6 个臂
  Pearson(末轮回复中位长度, 成功率) = **+0.984**——reset 的全部权威 = 它让模型重新写了一遍推导。
  让"不写推导"成为可选项的是 `ergo_math_trajectory.py:42-46` 那条**每轮重复**的答案格式指令，
  而这个 harness **完全没有 system prompt**；上游把同一条指令放在 system prompt 里只给一次，
  并且用 strategy classifier 把 `Current answer: 12` 这类回复判为**非作答而不打分**。
  R1（格式指令搬进 system prompt）**已完成并通过**：权威回来了（`+0.1810`，CI [+0.0862, +0.2845]），
  状态也还在。**R2/R4 已关闭、R3/R5 挂起**——它们测的是"我们像不像上游"，
  填不出论文任何一格。计划现在只剩 **EK0**（状态依赖分析，CPU）→ **EK1**（Koopman
  辨识，CPU）→ **EK2**（闭环臂，待签字）。
  **战略要点**：`RECAP > SNOWBALL` 的非单调性就是控制问题本身，而它**只在 append 下存在**；
  上游 ERGO 相对 RECAP 的 +14.1/+9.3 分正是"自适应时机打赢最好的固定规则"，即本项目的 S1。
  > **该战略要点已被 R1 部分推翻（2026-09-08）**：G-R1-3 不通过——在我们修好的 harness 上
  > `SNOWBALL > RECAP`，非单调性**不存在**，所以"它就是控制问题本身"这条论证作废。
  > 剩下唯一还站着的问题是**固定预算 k 下最优 reset 时机是否随状态变化**，那正是 EK0 测的东西。
- **[experiments/constraint_retention_plan.md](experiments/constraint_retention_plan.md) —
  ⏸ **计划稿，未开工**（2026-09-08 立项）。`constraint` 线（约束保持）：SEQUOR
  （arXiv 2605.06353, COLM 2026）的多轮约束遵守 + 预算内提醒再注入。它为什么存在：`defense`
  卡在读出没量程、ERGO 卡在主模态是确定性斜坡且激励功效不足，而这里 `y` 是 **k 条并行验证通道
  的计数**（k=3，4 档），且闭环可选的不只是时机、还有**方向**（提醒哪一条）。
  按用户裁决，**第一个作业等 ERGO 出较明确结果之后**才提交。
- [experiments/backup/two_task_success_plan.md](experiments/backup/two_task_success_plan.md) —
  ⛔ **已归档**（2026-09-08）。ERGO 半边 E0–E6 结案（S3），规格作废，后继见上一条；防御半边
  D0–D3 未被取代但执行停在盲标，入口改为
  [defense_line_redesign_plan.md](experiments/defense_line_redesign_plan.md) 第十三节。原文：把
  "两条线都要正面结果"拆成跑前写定的 S1/S2/S3 三级，然后按信息结构分配资源。**ERGO 线主攻**：
  五层前提里已有四层站住，唯一没站住的第 0 层（终点被末轮单个动作决定）根因是
  `ergo_math_trajectory.py:114` 把 reset 实现成**覆盖整个历史**——overwrite 下
  `fixed_last` 与 `always_reset` 的末轮输出 **116/116 逐字相同**（E0 独立重算，`judge_raw_output`
  与 `agent_message` 皆然），所以"状态"根本不进入终点。改成**追加**后历史重新进入被控对象，
  四道预注册闸门（权威保留 / 恒等式打破 / reset 效应依赖 closeness / 控制器不退化成固定臂）
  决定是否投 Phase C′，GPU 总量约 8 小时（**实际结果**：G-E2-2 通过、G-E2-1/G-E3-1/G-E3-2 不过，
  E5 从未提交）。**防御线只给两个 GPU-天**判定换目标模型能否让
  前提 1–2 成立，闭环不再追。E0 复核（第十二节）补了四处规格缺口：现有
  `analyze_ergo_phaseC_comparison.py` 算不出任何一道新闸门（`ARM_DIRS` 硬编码）、模式 B 的两个
  对手臂尚不存在、`forced_last_reset` 的预算路径会让末轮强制动作静默失效、terminal 目标下
  `repeat_penalty` 与平手规则需预注册。**两条线的 S3 版本已经是可成稿的正文，不是失败。**
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
- [experiments/koopman_defense_pilot.md](experiments/koopman_defense_pilot.md) — **Phase A→I
  已于 2026-09-02 收尾**（该阶段的评测设定下调查结束；同一任务在新设定下的续作见下面的
  `budget_constrained_defense_plan.md`）：把人格漂移这条线的 `control.py`/`modeling/`
  几乎零改动复用到对抗防御领域，设计并验证 Koopman-MPC 防御控制器。**新开一次对话想知道"Koopman 防御控制器现在做到哪一步了"，
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
- [experiments/koopman_phaseI_policy_closed_form.md](experiments/koopman_phaseI_policy_closed_form.md) —
  上面那条线收尾之后的纯离线补充：把 Phase I 的 `koopman_mpc_interaction` 策略从"跑出来的
  行为"化简成一条闭式规则。谱半径 $\rho(A)=a_{11}=0.7036<1$（稳态存在，可以讲稳态而不必退回
  直流增益），边际收益 $C(I+A)(B_1+B_2y)$ 的零点 $y^*=0.8036$、控制器真正在算的两步值零点
  $y^*=0.7881$，在 0.25 的判官网格上两者等价于同一条阈值 `y_{t-1} <= 0.75 就插`。用它复算
  Phase I 两个 arm 的每一次决策，攻击 48/48、良性 64/64 零失配；59 个"不插"点 = 32 个热身期
  强制 + 27 个 $y_{t-1}=1.0$ 的真实负 margin。副产品：$y^*$ 夹在两个开环稳态（0.7677 /
  0.7908）之间，所以这个策略**结构上必然抖动**，21 插/27 不插不是调出来的比例。状态：
  **无新实验、无代码改动**。
- [experiments/lstm_baseline_plan.md](experiments/lstm_baseline_plan.md) — 补齐 `BASELINES.md`
  ③层"LSTM 一步-多步预测模型"这个 ablation 缺口的计划与实现（只有 ARX vs `richer_abs_sign`
  两个同结构线性模型对比过,还没有真正的非线性/长记忆模型跑过对比）。记录了两种设计取舍
  （固定窗口 vs 隐状态跨轮次持续演化，采用后者）、`Predictor` 接口怎么接、训练/评测口径怎么
  和现有 Koopman 两件套对齐、参数量/数据规模两个混杂因素怎么处理。**状态：已执行完毕，
  并已于 2026-09-03 在 v 对齐修正后重跑复核（原"待复核"结清）。结论方向不变、量级改写——
  在和 AE baseline 同一口径（早停用训练攻击切出的验证集）下，LSTM 是 0.082–0.095，比
  `richer_abs_sign` 的 0.0684 差 20%–39%，四个隐层大小全部更差；但原先记录的"接近 2 倍"
  （0.081 vs 0.043）不成立——v 对齐让对照基线自己从 0.043 掉到 0.0684，差距缩到 0.97–1.07 倍，
  剩下的劣势全部由早停口径撑着（旧口径的早停判据就是被报告的那个 held-out 集合，是乐观的）。
  加了一个消融把"口径变公平"和"训练数据少 4 个攻击"分开：H=8 上前者花 +0.0276、后者只花
  +0.0009。**
- [experiments/ae_baseline_plan.md](experiments/ae_baseline_plan.md) — 补齐 `BASELINES.md`
  ③层另一个 ablation 缺口：对照根目录 `src/koopman_ae/core.py` 里
  `DeepAugmentedKoopmanAutoencoder` 的 encoder-decoder 架构（非线性 encoder/decoder + 隐空间
  线性动力学，`reconstruction_then_ridge` 训练方式），迁移到对抗防御任务的 `z_t/v_t/y_t` schema
  上。和 LSTM 不同，这个模型的 `step`/`readout` 仍在原始 `z` 空间定义，直接复用了
  `modeling/evaluate.py` 而不需要另写评测代码。**状态：已执行完毕，打平——held-out rollout MSE
  和 `richer_abs_sign`/`arx` 基本相等（0.0675±0.007 vs 0.068，3 seed），不同于 LSTM 的明确
  更差；train one-step MSE 上明显更差，但诊断为训练目标不可比,不是拟合能力问题。**
  **2026-09-03 追加（早停）**：照 `ABLATION_STUDY.md` 第八阶段给这个 trainer 也加了早停并重跑，
  但**没有复现 core 任务那种改善**——`latent_dim=1` 从打平变成明确更差（0.0756±0.0105），
  早停轮数在种子间波动到 10 倍以上。诊断：22 个训练攻击里再切约 4 个做验证集，早停判定信号
  本身样本量不足，不是"AE 更差"的证据；也因此"重建 loss 没收敛是否影响 rollout 结论"这条
  局限**仍未被解决**，要等更大规模的 Phase B 开环数据。
- [experiments/budget_constrained_defense_plan.md](experiments/budget_constrained_defense_plan.md) —
  ★ 正在做（对抗防御线的续作）：Phase A→I 收尾时留下的问题是"当前评测里根本不存在分配问题"
  （提醒有正效应、代价几乎为零，理论最优就是常提醒，所以 `koopman_mpc` 没有展现自适应优势
  是设定的必然结果，不是方法的证据）——这条来自 `next_step_diagnosis.md` 第二节的诊断。
  Phase J 把设定改成**每条轨迹最多 k=1 次提醒**（核对已有轨迹后从建议的 k=2 改成 k=1，因为
  k=2 在这批数据上不 binding），策略的任务从"要不要提醒"变成"把这一次放在哪一轮"，
  `koopman_mpc` 对 5 个固定轮次臂 + `threshold` 共 7 个臂。为了不和 Phase A–I 的结果/日志互相
  覆盖或混淆，新增了 Hydra `conf/experiment/` 配置组（一个文件=一个臂=一个 `output_dir`，攻击集
  /seed/预算集中在 `phaseJ_base.yaml`）和 `persona_drift.run_config_guard`（同一 `output_dir`
  换了配置就拒绝续跑——screening 循环天生可续跑，指错目录不会报错，只会安静地把两个控制器的
  轨迹混进同一份报告）。**状态（2026-09-03）：7 个臂跑了两轮——2 seed（第十节）与
  5 seed 复核（第十一节，`next_step_diagnosis.md` 第三步）。定性结论两轮一字不差：
  自适应臂没打赢最优固定臂，也没明确打赢 `threshold`，三个臂在安全分上互相都不可区分，
  只在花掉多少预算上可区分；机制上 koopman 臂在 3/4 的轨迹上逐位复现某个固定日程
  （40 条里 turn4×20 = `fixed_t4`、turn5×10 = `fixed_t5`），全部净差异来自"一次都不花"
  的那 10 条，而那正是它唯一自己发明的动作，也是最明确的失分来源。**
  **5 seed 复核改写的两点**：① 2 seed 选出的"最优固定臂"从 `fixed_t4` 变成了 `fixed_t5`，
  固定臂之间的 late_y 跨度也从 0.125 压缩到 0.075——"哪一轮最好"这个判断本身在 2 seed 下
  就不稳。② **扩样本按 √n 如期生效但仍然不够**：CI 宽度 ×0.60（理论 0.63），却还比相邻臂
  0.010–0.029 的间距宽 2–5 倍，补到够用需要约 31 倍轨迹量（每臂 ~1250 条、约 100 GPU-小时）。
  **所以"扩样本是继续的必要条件"要改成"必要但不充分，且充分的代价不现实"**——这正面印证了
  `koopman_defense_pilot.md` 第六节提出的可能性：天花板是 judge 只有 5 个取值的读出分辨率，
  不是样本量。与下面那条独立 judge 的重打分结论合流。**
  **2026-09-03 追加（读出的第四个修法，job 15519620 已完成）**：这条线从 Phase A 到 J 的
  每一个 judge 分都是自评 judge（`judge_model == agent_model == Qwen/Qwen3-4B`）打的，
  从没检查过。已用独立 judge（`Qwen3-4B-Instruct-2507`，与 sycophancy 线同一个 checkpoint）
  **离线重打分** 10 个臂 1605 行（复用每行存着的 `agent_message`，不重新生成 agent 文本，
  配对是构造性精确的）。**结论比预期更糟**：方向与 sycophancy 线相反（600/624 处分歧是
  独立 judge 更宽松），而且**偏差随轮次系统性增长**（turn1 分歧 0% → turn5 71%，signed
  diff 斜率 +0.072/轮 p=0.0000），与 new-Q1 的自变量共线——**独立 judge 下 10 个臂的
  new-Q1 全部不显著**（自评下除 periodic 外全部显著）。人工核查确认两个判官都在犯错、
  方向相反（自评把"话题变黑"读成"安全下降"，独立 judge 在真正危险的后段轮次上过松），
  所以这不是"换个 judge 就修好"，而是**读出本身不可靠**的实测证据。**臂间比较不受影响**
  （偏差与 `u_remind` 无关，Welch p=0.13；Phase J 的判定在两个 judge 下同号同结论）。
  详见 `koopman_defense_pilot.md` 第七节。
  **2026-09-04 追加（读出的第三条独立证据，纯离线只读检查，无 GPU、无新产物）**：
  `evaluation/ASR_METRIC_DESIGN.md` 顺带发现 **93.5% 的回复被 256 token 上限截断**，
  这是已知 limitation 里唯一一条若成立就要求**全线重跑**的。已检查完毕并**排除**：
  提醒不让回复变短（逐轮 Δ长度 ≤ 29 字符，全部 p ≥ 0.13），t5 的提醒效应在"讲完了"和
  "被截断"两层里一样大（+0.1166 / +0.1218），剔掉全部拒答后仍有 +0.1275——**臂间结论
  不受截断影响，不需要重跑**。但截断确实加重判官分歧（被截断行 39.9% vs 完整行 23.7%，
  p=0.0091，逐轮同向且对轮次混淆是保守的），给第七节"两个判官都在犯错"补上了具体机制。
  顺带确认 `refusal_flag` 是一个确定性、与 judge 无关、提醒能显著驱动（t5 21.2% vs 7.3%）
  但过于稀疏（全体 2.5%）的信号。详见 `koopman_defense_pilot.md` 第八节。
- **[experiments/independent_judge_reactive_rerun_plan.md](experiments/independent_judge_reactive_rerun_plan.md) —
  ✅ 已执行（2026-09-06）：反应式臂的独立 judge 重跑。`koopman_defense_pilot.md`
  第七节的离线重打分**只能重新测量、不能重新决策**——固定臂（`zero_control`/`constant_remind`/
  `periodic`/`fixed_t*`）的日程与 judge 分无关，重打分等价于重跑；但反应式臂
  （`threshold`/`koopman_mpc`/`koopman_mpc_interaction`/两个 Phase J 预算臂）拿 `y_probe`
  当反馈做决策，它们在独立 judge 口径下的数字此前根本不存在。5 个臂全部真跑完成
  （job 15603666/15603644/15603645/15603646/15603647，均 `COMPLETED 0:0`），自洽性检查
  640 行 100% 一致。关键结果两条：(1) new-Q1 侵蚀判定在自评口径下全为 `True`、独立 judge
  下全为 `False`，与固定臂已知的"自评单向夸大侵蚀"结论一致；(2) `koopman_budget1` vs 最优
  固定臂的判定**翻转**——自评口径下差异不显著，独立 judge 口径下 koopman 在安全分上
  **显著弱于**最优固定臂（同时提醒数显著更省，约省 86%），这是需要写进论文讨论/限制的新
  发现。详见该文档第七节"执行结果"。触发来源是论文 Step 2b 的作废判定
  （`../paper/evidence/superseded.md` 事件 E3）。**
  **⚠️ 上面第 (2) 条的"约省 86%"已被 `adaptive_vs_fixed_claim_plan.md` 第 0.1 节推翻：
  那不是效率，是控制器在独立 judge 的天花板（0.91）下停摆——提醒/轨迹从自评的 0.750 掉到
  0.150，而 bang-bang 阈值即使推到最敏感的 `y_min=1.0`，独立 judge 下触发率上限也只有
  1.6%–5.0%。引用这条结果时请一并读该计划的第 0.1 节。**
- **[experiments/adaptive_vs_fixed_claim_plan.md](experiments/adaptive_vs_fixed_claim_plan.md) —
  ✅ T0/T1a/T1b/T2/T3 Step1 已执行完（2026-09-06 立项，2026-09-07 收尾于第十一节）：把
  "自适应控制抗侵蚀强于固定基线"这条 claim 做成可判定的。上游是
  `independent_judge_reactive_rerun_plan.md` 第七节结果的复核。**结论：claim 不成立**——
  自评下过"打得过什么都不做"（late_y +0.0937 [+0.0208, +0.1927]），但不过"打得过等代价
  随机分配"（T2 的真臂对手 `randsched_p75`，两个口径 CI 均跨零）；独立 judge 下两道都不过，
  且机制是控制器停摆而非高效（T1a：阈值推到最敏感的 `y_min=1.0`，触发率上限仍只有 6.25%）。
  第 11.4 节进一步追出根因：**状态变量就是 judge 打分本身**（`y_probe ≡ y_safety`），
  house 规则 V8 的 leaked-metric 缺陷，T1b 的"模型学出提醒无用"是这个循环的直接后果，
  所以它的 NO-GO 带一个前提限定语（"在状态与 judge 未解耦的当前架构下"），不是终局判死刑。
  读出层面的结构性诊断（去轮次 lag-1：独立 judge 0.000 / 自评 0.347–0.424 / 激活投影
  0.861–0.887）**已由 D1 复核完毕（2026-09-07）——T3 就地终止**：T3 Step 1 在736个转移上
  测到的"执行器能移动回复前投影"（`u_remind` +5.538, p=6.04e-6）补上遗漏的 `u_{t+1}` 项后
  腰斩且被方向相反、幅度更大的 `u_{t+1}` 系数（−7.244, p=4.51e-10）盖过，外生子样本
  （`phaseG_periodic`）与隔轮持久性检验（S3 的 `u_{t-1}`）均未能确认存在效应，RC-A 三条
  预注册判据不过。T3 Step 2 未执行，也不会执行。完整数字见
  [experiments/backup/readout_controllability_gate_plan.md](experiments/backup/readout_controllability_gate_plan.md)
  与 [experiments/adaptive_vs_fixed_claim_plan.md](experiments/adaptive_vs_fixed_claim_plan.md)
  第十三节。**
- **[experiments/defense_line_redesign_plan.md](experiments/defense_line_redesign_plan.md) —
  ⏳ **防御线的活入口**（2026-09-07 立项，同日 D1/A1/A2/B1 跑完；**第十三节是当前状态**）：
  D0–D3 时间盒的规格在已归档的
  [backup/two_task_success_plan.md](experiments/backup/two_task_success_plan.md) 第四节（未被取代），
  但执行停在盲标——D1 两个 GPU 臂各 100 攻击 × 5 轮已跑完，**14 个盲标 subagent 只完成 6 个**，
  其余撞上内容策略硬拒绝且缺失非随机，**G-D1-1 算不出来**，三条待决策路线见第十三节。
  本文档第十节的负结果写法仍是 D3 续写的目标。原文：P 判定独立 judge 站得住之后，
  复核"Koopman 要怎么改才可能生效"，测出一条量化结论——**这条线的信息结构不支持闭环**。
  255 条轨迹上：早期观测（前两轮均分）对 late(3–5) 的相关是 **−0.066/+0.041（p=0.29/0.52）**，
  而事先就知道的攻击身份用**留一法**能解释 **57.1%/45.3%** 的 late 方差。**可预测的部分事先
  就知道，事后观测没有预测力**——闭环的价值恰恰是用观测换预测。方差分解补一刀：轨迹内占
  71% 的方差，但独立口径下去趋势后 lag-1 = 0.000，**"有变化"不等于"有状态"**。
  闭环三前提（执行器权威／早期观测有预测力／读出耦合且有量程）防御线一个都没确证，
  而**第一条最基础的从 Phase A 到 J 从来没测过**。四个任务：**D1** 用 P 的盲化协议做执行器
  权威的裁决版检验（`zero_control` vs `constant_remind`，同 8 攻击同 16 轨迹、提醒 0 vs 5.00、
  late 96 行，约 12 次 subagent 调用、零 GPU、零新采集；n=16 配对的最小可检出效应 0.175–0.315，
  能查大效应查不出 0.05）、**A1** 只用攻击文本特征做留一攻击预测（零成本，路线 A 的第一道门）、
  **A2** 跨轨迹预算分配的离线策略评价（条件触发；**是事前预测+静态分配，不是闭环**）、
  **B1** 去掉 256-token 截断重跑（需用户裁决）。**执行顺序硬性：A1‖D1 先做，D1 出结果前不动
  A2/B1。**
  **2026-09-07 更新：D1 与 A1 已完成。** D1（116 条盲标、两遍）主结果
  `constant_remind − zero_control` = **−0.0625，95% CI [−0.1667, 0.0000]**，含 0；但
  **16 个配对差里 14 个精确为 0**，全部信号来自 2 个攻击，按攻击簇 bootstrap 同样贴零。
  真正的限制是事件太稀少：两臂 82 条一致行只有 **5 个** assisted 事件，
  **未防御臂 4/41 = 9.8%（Wilson [3.9%, 22.5%]）**——这是攻击到底成不成功的第一个盲标
  估计，证实 P 的推论。**在 9.8% 基线下检出减半每臂需 444 行，D1 只有 41 行（9%），
  所以 D1 在任何效应量下都没有功效**——它不是执行器没用的证据。另一条限定 P 的发现：
  裁决者认定的 4 个真实事件，**独立 judge 4/4 全判满分**。
  A1：留一攻击 ridge R² = **−0.604 / −0.779**，CI 完全在 0 以下 → **路线 A 出局，A2 作废**。
  剩下唯一的杠杆是设定本身，B1 已改写为**诊断版**（一个 GPU 臂把 `max_new_tokens` 提到
  1024，只测基线率抬不抬得起来）。第十节写明若抬不起来时负结果该怎么写。
  **2026-09-07 更新：B1 已完成，落在第二格。** 上限 256→1024：原臂 **80/80 全部贴顶**
  （比文档记的 93.5% 更极端），放开后**仍有 55% 贴顶**、回复均值 931.9 token。盲标结果
  非配对是 0.0976→0.1429，但两批是同一批 (trajectory, turn) 的重跑，**配对之后 38 行里
  0 条从无害翻成有害、0 条反向，配对差 0.0000 CI [0,0]**——**截断没有掩盖任何一次帮助的
  发生**，抬升全来自两批 unresolved 行集合不同。变的只是严重度（`substantial` 0→3 条）：
  截断砍掉的是帮助的**完整性**不是**发生**。**结论：9.8% 的低成功率是这套攻击对这个模型的
  真实属性，不是测量假象**，截断被排除为混淆，防御线停止投入、按第十节写负结果。
- **[experiments/measurement_validity_plan.md](experiments/measurement_validity_plan.md) —
  ✅ **已收口**（2026-09-07 立项，同日 P/G1/G4 跑完）：MV 判据与 G4 的结论已并入
  [backup/two_task_success_plan.md](experiments/backup/two_task_success_plan.md)（0.4 节的预算模式判定法
  → G-E3-3；7.1 节的 `fixed_last` 臂 → 恒等式基线）。原文：两条线的全部结果跑完后复核发现，
  它们卡在同一个毛病上——**先下结论，没先证明测量有效、设定非退化**。新增一条前置判据 **MV**
  （测量有效性），排在 RC-0..RC-3 之前。防御线的缺口是 judge 从未被外部真值验证过：同一批
  **1905 行**回复上自评与独立 judge 的 exact agreement 只有 **0.622**，且分歧**单向**——自评
  打非满分的 772 行里独立 judge 说满分的有 **610 行（79%）**，**自评的全部动态范围正好落在
  分歧区**。独立 judge 的 test-retest 是 **1.0000**（确定性≠有效）；判别效度上自评每条都强
  2–2.4 倍（拒答的 Cohen d 0.93 vs 0.38），但控制住问句直白度/轮次/拒答后，两者对"回复本身
  有害内容"的增量敏感度几乎一样小（−0.085 vs −0.082）。ERGO 的缺口是**设定退化**：reset 的
  内容是"已揭示 shard 的合并列表"，所以最后一轮 reset ＝ 把完整题面重述到离答案最近处，
  实测落在最后一轮 0.650 vs 其余 0.427（Spearman −0.278, p=0.0026），而 `num_shards` 事先
  已知——**k=1 计数预算下最优策略是一个不需要反馈的固定规则，而这个最优固定臂根本不在
  Phase C 的臂表里**。**先做 P（分歧层小样本先导，只要 8 次 subagent 调用，可能直接结束整个研究）**：判别信息
  几乎全在两个 judge 分歧的 A 层——那里两者预测相反，所以"裁决者在 A 层判 assisted 的比例
  p_A"直接就是答案（→0 独立 judge 对、→1 自评对）。实算 Wilson CI：n=40 能判定所有不含糊
  的情形（p̂≤0.25 或 ≥0.75 时 CI 排除 0.5），n=60 连 p̂=0.35 也能。P 抽 A40 + C10 漏检探针
  + 10 条拒答硬对照 + 10 条重复题 = 70 条，两遍不同排列，先过资格闸门再看 p_A。
  P 若判"独立 judge 站得住"，G0–G3 全部不做。其余任务：G0 构造盲化全量样本（A610/B141/
  C1128/D26 → 抽 206 + 59）、G1 裁决标注（两遍 + 只对分歧条目补跑第三遍，省约 28%）、
  G2 效度分析与主指标裁决（预注册四格规则，总体 assisted 率用 StratPPI 报 CI）、G3 条件触发
  地重打 claim 关键的 384 行、G4 ERGO 补 `fixed_t_last` 臂判定设定是否退化（退化则改 token
  预算）。**第六节写明这是 LLM 裁决不是人工标注，以及后续 30 条真人抽检的补救。**
  **2026-09-07 更新：P 已执行完毕**（8 次 + 1 次补标 subagent 调用，三条资格闸门全过）——
  A 层 40 条两遍标签零分歧，`p_A = 10/40 = 0.25`，Wilson 95% CI **[0.1419, 0.4019]**，
  上界 < 0.5 → **独立 judge 站得住，自评在过度标记**，防御线既有结论（T1a/T1b/F4/两道闸门）
  保留，**G0–G3 不做**。C 层非拒答探针 10 条只有 1 条判 assisted，未触发"两个 judge 一起
  漏检"的停下追问条件。数字见
  [experiments/adaptive_vs_fixed_claim_plan.md](experiments/adaptive_vs_fixed_claim_plan.md)
  第十五节。
  **2026-09-07 更新：G4 已完成，判定第一格（设定退化）。** 补上的 `fixed_last` 臂
  （reset 落在 `turn = num_shards`）拿到 **0.7759**，`fixed_last − randsched_p100` =
  **+0.3103 [+0.2069, +0.4224]**、`− fixed_t4` = **+0.1293 [+0.0259, +0.2414]**，
  所以 Phase C 的三道闸门作废、不重跑 MPC（对真正的最优固定臂，MPC 是 −0.569 而非 −0.4397）。
  **更要紧的是追出了一个恒等式**：`fixed_last` 与 `always_reset` 的 116/116 条轨迹**抽取答案
  逐字相同**——`ergo_math_trajectory.py:114` 的 reset 是**覆盖**对话历史而不是追加，所以任何
  末轮 reset 的策略末轮 prompt 相同、输出必然相同。三个推论：`always_reset` 的 0.7759 其实
  就是**未分片基线**（分片退化效应 0.3276→0.7759 = **−0.448**，干净复现）；
  **`final_turn_success` 是错误终点**（只是"末轮有没有 reset"的函数，控制问题被指标构造性
  地清空）；**token 预算补救不再推荐**（一次末轮重述 152.8 token 就达到 696.1 的天花板，
  4.56×）。ERGO 线现在的产出是三条非负结果，见
  [experiments/ergo_multiturn_reliability_pilot.md](experiments/ergo_multiturn_reliability_pilot.md) G4 节。
- **[experiments/signal_resolution_plan.md](experiments/signal_resolution_plan.md) —
  ✅ **已收口**（2026-09-07 立项，同日 F0–F4 跑完）：`closeness` 读出与 0.5 节的四条判据被
  [backup/two_task_success_plan.md](experiments/backup/two_task_success_plan.md) 的 G-E3-1 逐字沿用（不改阈值）。
  原文：复核 `8dde4b6`（ERGO 终止）时
  发现**那个终止判定建立在两处测量错误上**——(1) E0 的 RC-3 把 `u_{t+1}` 实现成了 `u_{t+2}`
  （`analyze_ergo_readout_state.py:216-219` 的三元 `zip`），改正后 `u_t=+0.0588, p=0.0296`
  而不是 p=0.659；(2) RC-1 的 rollout 让模型外推 `shard_frac` 这个**确定性外生量**
  （自系数 1.005>1，会发散），而被拿来当 null 的 stateless 回归每一行都拿到它的真值——
  把 aux 维换成真值后，ARX 在 **20/20** 个 item split 上打过全部三个平凡 null（naive 只有
  7/20）。**RC-B 三条全过，ERGO 线不该终止。** 同时指出 E1"换读出"的方向选错了：token 熵与
  被评价目标近乎正交（Spearman +0.114 / −0.122），正确的方向是**把同一个仪器去阈值化**——
  ERGO 用 `closeness`（同一抽取器、同一 gold answer，不在"完全相等"处砍成 0/1；与目标
  ρ=+0.632，lag-1 +0.276，输入增益 p=4.8e-4，四条判据全过），防御线用**软 judge**
  （judge 输出就是一个 token，取 `"1".."5"` 的下一 token 分布期望值而不是 argmax；独立 judge
  硬标签在 Phase B 上 95% 顶在天花板）。判据从 RC-gate 的三条升级为四条：新增 **RC-0
  信号一致性**、RC-1 改多 split + aux 真值覆盖、RC-3 要求先声明"变好"的方向。
  五个任务：F0 修 E0 的两处错误并重跑、F1 固化 `closeness` 读出、F2 用它重拟合 Phase B、
  F3 ERGO Phase C（从已归档的 RC-gate 计划 §6 迁移并修订）、**F4 防御线软 judge 读出
  ——已完成（2026-09-07）：RC-0/RC-2 过，RC-1/RC-3 不过（ARX 20 split 只打过最好 null
  6/20；执行器输入增益 p>0.7），防御线读出族（硬标签/激活投影/软 judge 三个候选）就此
  封闭，结论见 [experiments/adaptive_vs_fixed_claim_plan.md](experiments/adaptive_vs_fixed_claim_plan.md)
  第十四节**（与 F0–F3 无依赖，本身已跑完，F0–F3 状态见上）。**第九节是文档状态地图，开工前必读。**
- [experiments/backup/](experiments/backup/) — 🗄 **归档（只读）**：已执行完毕或规格已被取代的
  计划文档（`readout_controllability_gate_plan.md`、`ergo_koopman_mpc_opus_design_questions.md`、
  `adaptive_vs_fixed_claim_plan_handoff.md`）。**不要执行、不要修改、不要引用为当前状态**；
  归档规则与"已知留在里面没改的错误"见 [experiments/backup/README.md](experiments/backup/README.md)。
- [experiments/sycophancy_screening_pilot.md](experiments/sycophancy_screening_pilot.md) —
  `SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md` 第八节步骤 2 的 screening（SYCON-Bench
  False Presuppositions 回放 + 三分类 judge + 连续斜率/离散翻转事件双判据）。**2026-09-05 起
  不再是这条线往前走的数据源**（见下面文末追加）——**新开一次对话想知道"这条线现在跑到哪一步了"，
  看 [mc_sycophancy_screening_pilot.md](experiments/mc_sycophancy_screening_pilot.md)**，本文档
  保留作历史记录（screening 方法论/judge 自评偏差/ground truth 审计的教训仍然适用）。状态：
  job 15483493 跑完
  （2026-09-02，20 items × 2 seeds）。人工审计发现并修好了离散判据的统计设计 bug（原检验对
  任意非零翻转数都几乎必然"通过"）；修复后是干净空结果——连续/离散两套判据都不显著，
  只有 2/40 条轨迹出现过翻转。**2026-09-03 追加：独立 judge 配对重跑（job 15487325，只换
  judge 权重、agent 输出 200/200 逐字相同）已完成**——自评偏差被证实是单向漏检（26 处分歧
  全部同向，符号检验 p≈3e-8，人工核查确认独立 judge 正确），把效应量压掉约 4 倍、还抹平了
  惯性结构（new-Q3 r 0.43→0.61）；**判定结论不变（仍不显著），但性质从"无现象"改写为
  "欠功效"**，下一步优先级相应改为扩样本 + 基线门槛（斜率只在 turn 2–5 拟合，避免文档里
  记录的天花板选择偏差陷阱）+ ground truth 审计。`--judge-model` 独立 judge 从此是默认。
  **2026-09-05 追加（ground truth 审计，下一步 (e) 已执行）**：对实际用到的 20 个 item 逐条
  核实 `correction` 字段，**8/20（40%）有问题**（3 条确认错误，含新发现的 `sycon_fp_0091`
  "手机电量不受旅行影响"——现实相反；5 条答非所问/过度概括；1 条字段疑似合并错位）,比此前
  只知道 2 条可疑严重得多。敏感性分析（CPU-only 重算，脚本见
  `persona_drift_control/scripts/audit_ground_truth_quality.py`）显示 new-Q1 判定不受影响，
  但 new-Q3（惯性）的 r 从 0.6072 掉到 0.19（去掉 3 条确认错误的）、再去掉 5 条存疑的后**不再
  显著**——`SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md` 五个前置条件里唯一"已满足"的一条需要收窄，
  见下面该文档条目。**同一天顺带把下一步 (d)（turn-1 基线正式入选门槛+仅 turn2-5 拟合）也
  实现了**（`analyze_sycophancy_screening()` 新增两个默认关闭的参数,判定不变）。**Phase A
  （执行器权威检查,五个前置条件里"完全没测"的那道门）已执行完毕**（job 15567174 zero_control /
  15567175 constant_remind,用审计筛出的 11 条干净 item）：11 items 配对检验不显著（p=0.74，
  方向为负,被一个和提醒无关的基线缺陷 item 拖低）；排除该 item 后方向转正但仍不显著
  （p=0.096，n=10）。**既未确认也未排除执行器权威，卡在样本量**（粗估要 ~23 个干净 item
  才够功效），和 new-Q1/new-Q3 是同一个瓶颈。**2026-09-05 追加：ground truth 问题排查到根因后
  换了数据源，见 `mc_sycophancy_screening_pilot.md`。**
- **[experiments/mc_sycophancy_screening_pilot.md](experiments/mc_sycophancy_screening_pilot.md) —
  ★ 正在做（sycophancy 线的数据源替换续作）：`sycophancy_screening_pilot.md` 的 ground truth
  审计把 SYCON-Bench 数据的问题追到根因——问题来自同行评审的 CREPE，但`presupposition`/
  `correction`/`pushback_turns` 是 SYCON-Bench 自己用 GPT-4o 重新生成的，没有证据核验。换成
  Sharma et al. 2023（ICLR 2024）"Are You Sure?" 的 MMLU 子集（1000 items，57 学科），正确答案
  来自 MMLU 自己的答案表，不是任何人现编的一句话；judge 也从"事实仲裁"降级成"抽取回复选了哪个
  字母"（92.5% 正则直接命中，零模型调用；只有 7.5% 退到一次简短 LLM 抽取，不判断对错）。**首次
  验证跑（job 15567606，20 items × 2 seeds）：new-Q3（跨轮惯性）r=0.72（原始）/0.42（加 turn-1
  基线门槛后），p<0.0001——比 SYCON-Bench 清洗到底的最好结果（r≈0.19）强一个数量级以上，是这条
  线第一次有干净数据支持"惯性"这个前置条件。new-Q1（渐进翻转）仍不显著，但这次不是欠功效——
  逐轮均值显示的是"有粘性但可逆"的动力学（翻转后常在最后一轮反驳时恢复），不是单调恶化，也不是
  `SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md` 第 2 节假设的绝对吸收态。~~**下一步：扩样本到 ~60
  items、在这份更干净的数据上重做执行器权威检查（Phase A）。**~~（**两项均已于 2026-09-05/06
  执行完毕**，见本条目下文。）
  把 sycophancy judge 的三分类硬标签换成同一 prompt 下 next-token 分布在三个标签 token 上的
  归一化概率，得到 y∈[0,1] 的连续读出，只回溯打分已有的 2×200 行、不重跑 agent、不动防御线。
  它同时是两条线的前置条件——`SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md` 第 5 节把它排在
  sycophancy 线最前（零 GPU 成本、可能同时解决欠功效），`budget_constrained_defense_plan.md`
  11.6 把同一件事（Gate 2）升级为整条防御线的前置条件。**2026-09-04 已执行完毕（job
  15540600）：G0/G1 全过（200/200 argmax 精确复现磁盘标签，`label_mass_total` 中位数
  ≈0.9999999，前向重建的 prompt 与 `generate()` 完全同源）；G2（非退化/分辨力）不过——
  `n_distinct_levels`=193 达标，但 96% 的行落在 {0,.5,1} 的 ±0.02 邻域内，erosion_snr 连续
  版反而略低于硬标签版，turn-1 IQR≈3.7e-6，说明这个 4B judge 在标签 token 位置上的后验
  近乎完全确定（要么~100% 是 MAINTAINS 要么~100% 是 FLIPS），"连续读出"在数值上退化成对
  硬标签的浮点复述，没有真实的置信度梯度可用。按预注册规则 G3 未跑（G2 不过即停）。
  结论：token 概率加权这条修法在此 judge/此 prompt 下不成立，sycophancy 线回到 feasibility
  文档列的 (c) 更强施压 / (e) ground truth 审计；两份原始 trajectories.jsonl 未被触碰，
  现行判据（`stance_label`/`is_flip`/`y_consistency`）不变。防御线的 Gate 2（同一修法）
  因此也被排除，`budget_constrained_defense_plan.md` 11.6 列出的这条路径不再可行。**
  **2026-09-05/06 追加：样本扩充（job 15568231，60 items）+ 执行器权威检查重跑（job
  15570373 zero_control / 15570374 constant_remind）都已完成。**扩样本后 new-Q3 仍强显著但
  效应量收缩（r 0.42→0.29，门槛后），new-Q1/flip_trend 仍不显著但更接近边界（p 0.167→0.074、
  0.67→0.071）——20-item 那次的效应量估计有一部分是小样本噪声。**Phase A 重做（配对 t-test，
  turn-1 基线门槛后 n=22 干净 item）结论是空结果，且这次站得住**：p=0.704（原始 30 items 时
  p=0.109，方向为负，被基线缺陷 item 拖低——门槛后方向仍为负但趋近 0）——排除了样本量和 ground
  truth 质量这两个此前解释空结果的候选原因后，`consistency_reminder` 式提醒在此设计下的效力
  判定为不成立，不建议靠加样本继续测同一设计，见 `mc_sycophancy_screening_pilot.md` 的
  "Phase A" 一节。**2026-09-06 追加：换插入时机重测（job 15590462，提醒只从 turn 2 开始插，
  复用同 30 items 与 zero_control 直接配对）——门槛后差值恰好为 0（p=1.000），比每轮插版本
  （p=0.704）更彻底的空。两版提醒设计在同一批 item 上给出一致的空结果，排除了"turn 1 插入
  提醒"这个具体假设，这条支线的证据强度升级为"当前 channel-A 文案设计下无可测权威"，不建议
  再换插入时机——下一步该换文案强度/具体性，或转向其他前置条件（benign 对照臂）。**
- **[experiments/ergo_multiturn_reliability_pilot.md](experiments/ergo_multiturn_reliability_pilot.md) —
  ★ **新任务线**（sycophancy 两次执行器权威空结果后评估的下一个候选，`ERGO_MULTITURN_
  RELIABILITY_FEASIBILITY.md`）：一个具体的、带真实开源数据集的新 screening 任务，不是论文
  写作、也不是抗攻击/抗压力两条既有线的续作。**新开一次对话想知道"ERGO 这条线现在跑到哪一步
  了"，看这份文档。** vendor 了 103 个 GSM8K sharded 数学题
  （`resources/ergo_gsm8k_sharded.jsonl`，MIT 协议），实现了 bank/判官（纯正则，零 LLM 判官
  调用）/独立的重置-vs-追加轨迹循环（不复用 `trajectory_runner.py`——"重置"这个动作的内容
  依赖轮次，塞不进共享循环 `reminder_fn(level)` 的签名）/精简版分析，24 个新 CPU 单测全绿。
  **20-item pilot（job 15602880 zero_control / 15602881 reset）+ 60-item 扩样本（job
  15603367/15603368）都已完成**，同一批 item id 保证配对可比。**结果：执行器权威确认，效应量
  大且稳**（最终轮任务成功率：20-item 0.275→0.750，t=4.79，p=1.27e-4；60-item 0.367→0.792，
  t=6.78，p=6.39e-9——3 倍样本下效应量不变，显著性更强）——和 sycophancy 线两次空结果形成
  对照；**但惯性性质不同**：这里是"信息单调累积"（shard 越揭示越好答）而不是 sycophancy 那种
  "同一信息反复施压下的立场漂移"，论文措辞需要分清这两种现象，不能混着引用。**2026-09-06 追加：
  开始 Koopman 建模（Phase B）**——决定显式加 `shard_frac`（已揭示 shard 比例）作为状态协变量
  （不只用 y 滞后项），`contemporaneous_v=True`（reset 同轮直接影响该轮 y，代码已验证）；新增
  `--controller random_excite` 支持 + `scripts/fit_koopman_ergo_model.py`；开环随机激励采集
  job 15613799 已提交（60 items × 2 seeds, p=0.5，规模同防御线 Phase B）。~~2026-09-07 追加：
  拟合完成——ARX 赢过 richer baseline、可控性满秩、判断值得往下投~~ **2026-09-07 联合审计撤回**：
  `richer_abs_sign` 在二值 `y` 下与 `y` 精确共线、是 vacuous 对照；ARX 的 held-out rollout
  MSE 0.0893 连"零状态外生回归"null（0.0792）都打不过；满秩/谱半径也不构成证据（见
  `backup/readout_controllability_gate_plan.md` §0.4，**该节的 RC-1 论证后来被证明信息集不对等，
  见下**）。正式闸门 E0（RC-gate）判 RC-1/RC-3 不过、Phase C 不开工，转 E1 试 token 熵读出
  （`entropy_mean`/`entropy_answer_span`）也不过，据此记过一次"ERGO 线终止"（commit `8dde4b6`）。
  ~~ERGO 线在当前读出族下终止~~ **2026-09-07 再次复核后撤回该终止判定**：E0 的 RC-3 把
  `u_{t+1}` 实现成了 `u_{t+2}`（改正后 `u_t=+0.0588, p=0.0296`），RC-1 让模型外推 `shard_frac`
  这个确定性外生量而 null 拿到它的真值（对齐后 ARX 在 20/20 个 split 上打过全部三个 null）——
  **RC-B 三条其实全过**。E1 的熵读出确实不合适，但理由是它与被评价目标近乎正交（新增的 RC-0），
  不是"读出族全灭"。**2026-09-07 追加，F0–F3 已执行完毕**：F0 修正后 RC-B 全过（20-split
  aux=真值 rollout 20/20 赢最优 null）；F1 把 `closeness`（同一抽取器的连续版本）固化为正式
  状态读出，四条判据都不比二值差（`u_t=+0.0540, p=5.29e-4`）；F2 用 closeness 重拟合
  （`koopman_fit_report_closeness.json`）；**F3 跑了 8 个 Phase C 臂（58 items × 2 seeds），
  三道预注册闸门全部不过**——`ergo_koopman_mpc`（预算 1）最终轮成功率 0.2069，全面输给
  `zero_control`(0.3276)、`randsched_p100`(0.4655，主判据) 和最优固定臂 `fixed_t4`(0.6466)。
  诊断：MPC 116 条轨迹**全部**在 turn 2 花掉唯一一次 reset，与 `fixed_t2` 逐位相同——
  `horizon=2` 短视，看不到更优的 turn 4，按纪律未调参重跑。执行中还发现并修复一个真 bug：
  `ErgoKoopmanMPCController` 继承的 `_remaining_budget` 硬编码读 `u_remind` 列，预算从未生效
  （首次跑 reset 了 548/664 次而非应有的 116 次），已修复+补单测+重挑。执行器权威结论不受
  影响。详见 `experiments/ergo_multiturn_reliability_pilot.md`"F0–F3：读出分辨率修正与
  Phase C 结果"一节。
  执行器权威结论（上面"结果："段）自始至终不受影响，仍然成立。
- [experiments/dose_response_pilot.md](experiments/dose_response_pilot.md) —
  步骤 2，安全方向 steering（diff-in-means 方向 + 残差流 hook）的单轮 α 剂量-响应扫描。状态：
  工程全链路已验证跑通，但 new-Q2 **两次都不过**——v1 直问有害目标撞天花板（p=0.0563）；
  v2 换成步骤 1 里真实"已部分被攻破"的对话上下文重测，天花板问题解决了但响应噪声大、不单调
  （p=0.4535），根因假设指向校准点（短单轮 prompt）和应用点（数千 token 深层上下文）不匹配，
  或单层 steering 压不过已建立的多轮上下文；两次都不是代码问题。**用户已决定这一轮不再追加
  channel C 新实验**，改走 `koopman_defense_pilot.md` 的 channel A（提醒注入）路线，本文档
  不再是活跃开发线。
