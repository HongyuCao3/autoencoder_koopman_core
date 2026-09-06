# 论文执行总流程（Koopman × LLM 控制）

本文档是**跨会话的施工图**。每个 Step 都写清了目标、输入、产物、流程细节、出口闸门、
适用智能体，以及一句可直接粘贴到新会话的启动语。分工执行时，先让新会话读本文档对应
Step，再开工；不要凭记忆复述本文档的内容。

- 写作方法论来源：`/home/hcao2/claude-skill` 的 `neurips-write-tiered`（三层构建）、
  `neurips-write`（24 条文风规则）、`paper-audit`（D1–D8 审计）、`paper-pipeline`（收敛循环）。
- 证据来源：本仓库的 `ABLATION_STUDY.md`（核心 AE-Koopman 八阶段）、`results/`（106 个 run）、
  `docs/experiments/*.md`（persona_drift_control 四条实验线）、`docs/task/*.md`、
  `docs/feasibility/*.md`。
- 论文产物目录：仓库根下新建的 `paper/`（本文档写完时尚不存在）。

---

## 一、论文主线（已锁定，2026-09-06）

### 1.1 一句话主张

**Koopman 算子是 LLM 多轮行为的一个实用动力学基座**：把行为读出的多轮轨迹用带控制的线性
算子建模，拟合出的算子既能驱动闭环干预，本身也是可读的诊断量；这套做法的适用条件和天花板
在本文里被一并交付。

### 1.2 `arc_type`：`prior_injection`

`neurips-write-tiered` 的 `meta.arc_type` 三选一（`challenge_driven` / `diagnosis_driven` /
`prior_injection`）。本文核心动作是**把一个领域先验当作归纳偏置引进来**——Koopman 算子理论
来自动力系统与控制论，LLM 多轮行为漂移此前基本被当作无状态的逐轮分类问题处理。因此选
`prior_injection`。这个选择会决定 §Introduction 的骨架，并决定 `intuition_lattice` 哪些字段
是必填的（见 `neurips-write-tiered/references/contract.md` §arc_type）。

### 1.3 叙事弧（Tier 1 `narrative_arc` 的工作假设）

Phase A 先按下面五句填进 `contract.yaml`，**Phase B/C 写完 §Method + §Experiments 之后再回来
定稿**（skill 规则 A9 的硬性要求）。

1. LLM 多轮交互中的行为漂移（安全侵蚀、sycophancy 立场放弃）是一个**动力学**现象，但现有
   干预手段基本是无状态的：每轮独立判断、固定模板、或一次性 system prompt。
2. 我们引入 Koopman 先验：把标量行为读出的多轮轨迹建模为带控制的线性算子
   `ξ_{t+1} = K ξ_t + B u_t + c`，控制量 `u_t` 就是可下发的干预动作。
3. 这个先验成立得比预期强，而且**是往简单方向成立的**——8 个 LLM 输出轨迹任务上，线性
   Koopman 代理已经够用，非线性 AE 提升不带来增益（简单性是被测出来的结果，不是妥协）。
4. 拟合出的算子的价值不止于选动作：谱半径给稳态、有限时域可控性 Gramian 给可防御性、
   Phase I 学到的策略可化简成一条闭式阈值规则、一步预测残差能对齐真实安全骤降。
5. 边界同样被交付：在当前 5 值 LLM-judge 读出下，自适应控制器打不赢等代价的开环周期基线，
   瓶颈是**测量分辨率**而不是建模能力——这条界限本身是给社区的可执行结论。

### 1.4 纳入 / 排除范围

| 证据线 | 在论文中的角色 | 主要出处 |
|---|---|---|
| 核心 AE-Koopman 八阶段消融（8 任务、106 run） | §Method 基座 + §Experiments RQ1（先验是否成立、AE 是否必要） | `ABLATION_STUDY.md`、`results/` |
| 对抗防御 Koopman-MPC Phase A→J | §Experiments RQ2（闭环干预）+ RQ3（等代价开环对照） | `docs/experiments/koopman_defense_pilot.md`、`budget_constrained_defense_plan.md`、`koopman_case_study_design.md` |
| 算子的额外用途（谱/Gramian/闭式策略/残差检测） | §Experiments RQ4（算子可读性） | `koopman_phaseI_policy_closed_form.md`、`koopman_detection_design.md`、`docs/task/KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md` |
| sycophancy 惯性（MMLU 数据源，new-Q3 显著） | §Experiments RQ5（跨任务迁移证据，**可选**，取决于页数） | `mc_sycophancy_screening_pilot.md` |
| 读出分辨率天花板（判官分歧/token 概率退化/ground truth 审计/截断） | §Experiments 收尾 + §Conclusion limitations + `non_claim_ledger` | `koopman_defense_pilot.md` 七/八节、`continuous_readout_plan.md`、`sycophancy_screening_pilot.md` |
| LSTM / ARX / AE baseline 对照 | §Experiments 的代理建模层 baseline | `lstm_baseline_plan.md`、`ae_baseline_plan.md`、`docs/evaluation/BASELINES.md` |
| 人格漂移线（空结果）、steering 剂量响应（两次不过）、SYCON-Bench 旧数据 | **排除出正文**，只进 `non_claim_ledger` / 附录 | `signal_screening_pilot.md`、`dose_response_pilot.md`、`sycophancy_screening_pilot.md` |

### 1.5 诚实性红线（这条主线最大的风险）

主线是 positive framing，但证据里有真实的空结果与打平。**"打不赢等代价 periodic"必须写进
正文**，不能挪进附录或省略——它是 §1.3 第 5 句的支撑，也是 `paper-audit` D4（overclaim）
必然会探到的地方。判断标准：`claim_ledger` 里每一条 `strength` 必须能被 `paper/evidence/`
里的具体数字和 n / p 值撑住；撑不住的降级或移入 `non_claim_ledger`。这条在 Step 4 的 Tier-1
冷审里作为一票否决项。

### 1.6 待定字段

`meta.venue` / `meta.page_limit` 尚未定，Phase A 先按 **NeurIPS / 9 页** 填。这两个字段只影响
Tier 2 的段落预算算术，Phase B 开始之前改的代价很低；Phase B 之后改要重做段落分配。

---

## 二、智能体分层与选型原则

### 2.1 按任务形状选模型

| 任务形状 | 模型 | 理由 |
|---|---|---|
| 判断、取舍、强度校准、结构分配、审计、冲突裁决 | **Opus 5** | 这些是"选哪个"而不是"抄哪段"，错了会污染下游全部产物 |
| 长文正文生成、文风约束下的改写 | **Fable 5.1** | Tier 3 正文要在 24 条 voice 规则约束下连续产出；写作导向模型更合适。Opus 5 是等效备选 |
| 批量抽取、格式转换、读大量文件、跑脚本、查资料 | **Sonnet 5** | 有明确规格的执行任务，规格写在本文档里，模型只需照做 |
| grep 统计、文件搬运、机械格式检查 | **Haiku 4.5** | 可被脚本替代的部分，能用脚本就用脚本 |

### 2.2 三条硬规则

1. **冷审必须开新会话。** Tier-1 冷审、Tier-2 冷审、`paper-audit` 的 D1，都要求审计者没有
   参与过被审对象的生成。同一会话里"自己审自己"会把 skill 的 A8 规则架空，产出一份看起来
   通过、实际什么都没查出来的审计。
2. **抽取与判断分开派。** 凡是"读一堆文件得出结论"的活，拆成 Sonnet 抽取 + Opus 判断两步，
   不要让一个 agent 一次做完。Step 2 是这条原则的主战场。
3. **Tier 3 正文只能重新生成，不能结构性手改。** 段落增删、claim 强弱、设计选择改名，
   都要先落回 Tier 1 或 Tier 2 再重新生成该节（skill 规则 A1）。允许直接手改的只有
   词替换、断句、删破折号、删副词这类表层润色。

### 2.3 subagent 类型对照

| 用途 | `subagent_type` |
|---|---|
| 大范围搜文件、定位证据在哪 | `Explore` |
| 有明确规格的多步执行、WebSearch 取证 | `general-purpose` |
| 设计实现路径、权衡架构 | `Plan` |
| 其他 | `claude` |

---

## 三、产物总览

```text
paper/
├── evidence/
│   ├── numbers.yaml        # Step 2a：全部可引用数字 + 出处 + n/p/CI
│   └── superseded.md       # Step 2b：作废/修正关系与理由
├── equations.tex           # Step 3：定稿方程（A5 前置）
├── contract.yaml           # Step 4：Tier 1 契约
├── semantic.md             # Step 5：Tier 2 语义大纲
├── sections/               # Step 5：Tier 3 正文
│   ├── abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_problem_formulation.tex
│   ├── 03_method.tex
│   ├── 04_experiments.tex
│   ├── 05_conclusion.tex
│   └── 06_related_work.tex
├── main.tex / main.bib     # Step I2/I3：编译入口与参考文献
├── figures/                # Step I3
├── audit_findings.yaml     # Step 6：paper-audit 结构化发现
└── audit_report.md
```

`paper/` 进 git；不要把它放进 `.gitignore`。

---

## Step 0 · 安装 skill

**状态**：未完成（`~/.claude/skills` 不存在，`neurips-write-tiered` 等 skill 当前不可用）。
**阻塞**：全部后续步骤。

**流程细节**

`/home/hcao2/claude-skill` 已经是完整的 skill 仓库，但 Claude Code 只从 `~/.claude/skills`
发现 skill。两个做法：

```bash
# 做法 A（推荐）：软链，两处同源，git pull 一次两边都更新
mkdir -p ~/.claude
ln -s /home/hcao2/claude-skill ~/.claude/skills

# 做法 B：按 INSTALL.md 另 clone 一份
git clone git@github.com:HongyuCao3/claude-skill.git ~/.claude/skills
```

装完**重启 Claude Code 会话**，确认可用 skill 列表里出现 `neurips-write-tiered`、
`neurips-write`、`paper-audit`、`paper-pipeline`、`paperize-figure`。没出现就不要往下走——
后面每一步都依赖 skill 被自动加载，手工读 SKILL.md 模拟会漏掉 references 的路由。

**适用智能体**：无需 agent，直接在终端执行；三条命令。

**出口闸门**：新会话里 `neurips-write-tiered` 出现在 skill 列表中。

---

## Step 1 · 主线锁定

**状态**：已完成（本文档第一节）。arc_type = `prior_injection`，范围见 §1.4，红线见 §1.5。

后续会话如果想改主线，改的是 §1.3 和 §1.4，并且要**重做 Step 4**（Tier 1 契约）。Step 2/3
的产物与主线无关，不受影响。

---

## Step 2 · 跨线证据盘点

**目标**：把散在 `ABLATION_STUDY.md` / `docs/experiments/*.md` / `results/*/run.json` 里的
数字收敛成一份唯一事实源，并显式标出哪些数字**已被本项目自己推翻**。

**为什么需要这一步（skill 流程之外的加项）**：本项目的文档是中文叙事体，且同一份文档里
并存"被推翻前"和"修正后"的数字。直接喂给 Phase A，过期数字会进 `claim_ledger`；而
`paper-audit` 的 D2 只查节与节之间的一致性，查不出"论文引用了本项目自己已经作废的值"。

### Step 2a · 机械抽取

**产物**：`paper/evidence/numbers.yaml`

**schema**（每条一个数字）：

```yaml
- id: n_001
  value: 0.0684
  unit: "rollout MSE"
  what: "richer_abs_sign 线性代理在 held-out 攻击上的 rollout MSE"
  line: adversarial_defense          # core | adversarial_defense | detection | sycophancy | readout
  source_doc: docs/experiments/lstm_baseline_plan.md
  source_run: null                   # results/<run-name>/run.json 路径
  job_id: null                       # Slurm job id（若有）
  n: 3                               # seed 数或样本量
  stat: "mean±std"
  test: null                         # 检验方法（配对 t / bootstrap / 符号检验 ...）
  p: null
  ci: null
  date: 2026-09-03
  status: unknown                    # 2a 阶段一律填 unknown，由 2b 判定
  superseded_by: null
  supersede_reason: null
```

**流程细节**

1. 分片：按 §1.4 的六条证据线分片，一个 agent 一条线，可并行。
2. 每条数字都必须能追到 `source_doc` 的具体小节或 `source_run` 的 json 字段；追不到的
   **不要写进来**，另开 `numbers_unsourced.md` 记录待查。
3. `status` 一律填 `unknown`——2a 只抽取，不做任何"哪个更新"的判断。
4. `results/` 有 106 个 run 目录，用脚本批量读 `run.json`，不要逐个手读。

**适用智能体**：**Sonnet 5**，`general-purpose`，按证据线并行 4–6 个。
定位阶段可先派一个 `Explore` 摸清每条线的数字都藏在哪些小节。

**出口闸门**：`numbers.yaml` 条目数 ≥ 论文预期引用量的 2 倍；随机抽 10 条回查出处，10/10 对得上。

### Step 2b · 作废判定

**产物**：`paper/evidence/superseded.md`，并回填 `numbers.yaml` 的
`status` / `superseded_by` / `supersede_reason`。

**已知的五次改写事件**（必须逐一裁决其影响面，不要只处理这五个）：

1. **`modeling/dataset.py` 的 "v 对齐" 时序错位 bug**（2026-09-02 修复）——污染了防御线
   Phase H 之前的全部拟合，以及检测支线方案 1/3 的底层拟合。修复后方案 1 残差相关系数
   0.53→0.86，方案 3 逐轨迹准确率 0.475→0.613，`richer_abs_sign` 自身 0.043→0.0684。
2. **早停混淆**（`ABLATION_STUDY.md` 第八阶段）——`epochs=200` 对至少 2/8 任务不够收敛，
   `sentiment_t5` / `average_word_length_t5` 的重建 loss 差 67–100 倍，两个任务的
   AE-vs-线性效应量被完全改写。
3. **自评 judge → 独立 judge**（2026-09-03）——防御线全部 judge 分是自评（judge == agent ==
   Qwen3-4B）。独立 judge 离线重打分后，10 个臂的 new-Q1 全部不显著；sycophancy 线效应量
   被压掉约 4 倍。
4. **sycophancy 数据源更换 + 扩样本**——SYCON-Bench 20 items（ground truth 8/20 有问题）
   → MMLU 60 items。new-Q3 的 r 在这条链上依次是 0.6072 → 0.19（清洗后）→ 0.72（MMLU 原始）
   → 0.42（20 item 加门槛）→ 0.29（60 item 加门槛）。**只有最后一个可引用。**
5. **Phase J 2 seed → 5 seed**——"最优固定臂"从 `fixed_t4` 变成 `fixed_t5`，固定臂间 late_y
   跨度从 0.125 压到 0.075。

`superseded.md` 每条写：事件 / 触发日期 / 影响的 `n_xxx` id 列表 / 修正后应引用哪条 /
一句话理由。

**适用智能体**：**Opus 5**。这是判断题——要读懂"这次修正是否触及那个数字的底层拟合"，
不是字符串匹配。**不要**派 Sonnet。可以在同一会话里做，输入是 2a 的产物加原文档。

**出口闸门**：`numbers.yaml` 中 `status == unknown` 的条目数为 0；每条 `superseded` 都有
`superseded_by` 指向一个 `current` 条目。

**执行时对本闸门做了一处修订（2026-09-06）**：`status` 变成三值（`current` / `superseded` /
`caveated`），`superseded_by` 允许指向 `caveated`（即"非 superseded"）而不只是 `current`。
理由与影响见 `paper/evidence/superseded.md` 第一节——简言之，存在第三类事实（唯一可得、
但协议已知被混淆、且从未重跑），两值 schema 只能在"隐瞒"和"违反闸门"之间二选一。

---

## Step 3 · 方程定稿

**目标**：`paper/equations.tex`，满足 skill 规则 A5——`equation_operation` 字段必须**从方程
实际计算的东西推导出来**，不能先想好一个名字再把数学掰过去。所以方程必须先于契约定稿。

**需要定稿的方程**（至少）：

| 方程 | 代码出处 |
|---|---|
| 状态构造（Markov / Memory-L / Augmented-L） | `src/koopman_ae/core.py` |
| AE 提升与隐空间线性动力学 `ξ_{t+1}=Kξ_t+Br+c` | `src/koopman_ae/core.py` |
| 两段式 `reconstruction_then_ridge` 的闭式解（含非正则化截距） | `src/koopman_ae/core.py` |
| joint 的四项复合 loss | `src/koopman_ae/core.py` |
| 防御任务的 `z_t/v_t/y_t` schema 与带交互项的动力学 | `persona_drift_control/src/.../modeling/` |
| MPC 目标函数与预算约束（Phase J 的 k=1） | `persona_drift_control/src/.../control.py` |
| 边际收益 `C(I+A)(B_1+B_2 y)` 与闭式阈值 `y*` | `docs/experiments/koopman_phaseI_policy_closed_form.md` |
| 有限时域可控性矩阵与 Gramian | `src/koopman_ae/core.py` 的诊断部分 |

**流程细节**：先用 Sonnet 把每处代码里**实际发生的张量运算**抽成伪代码（不要抄注释、
不要抄论文里的标准写法，以代码为准），再由 Opus 写成 LaTeX 并逐条写一句
`equation_operation`——"这个方程实际算的是什么"。两者不一致时以代码为准，并记录差异。

**适用智能体**：Sonnet 5（`general-purpose`）抽取 → **Opus 5** 定稿与复核。

**出口闸门**：每个方程都有一句 `equation_operation`，且能指回具体代码行。

---

## Step 4 · Phase A：建 Tier 1 契约

**目标**：`paper/contract.yaml`，六个顶层 key 全部填满。
**skill 入口**：`neurips-write-tiered`，读 `references/contract.md`。
**模板**：`cp /home/hcao2/claude-skill/neurips-write-tiered/templates/tier1.yaml paper/contract.yaml`

**流程细节**

1. `meta`：`arc_type: prior_injection`，`venue: NeurIPS`，`page_limit: 9`，`subfield` 填
   LLM 多轮行为控制 / 安全。`acronym` 先留 null。
2. `narrative_arc`：抄 §1.3 的五句（英文化），标注为工作假设。
3. `design_choice_lattice`：一行 = §Method 的一个子节。按 §1.4 的角色划分推导子节。
4. `claim_ledger`：一行 = 论文里会出现的一条 claim。**每条 `evidence` 指向
   `paper/evidence/numbers.yaml` 的 `n_xxx` id**，且该条必须 `status: current`。
   `strength` 按 §1.5 的红线诚实校准。
5. `symbol_lattice` / `equation_lattice`：从 Step 3 的 `equations.tex` 填，`operation_name`
   直接用 Step 3 写好的 `equation_operation`。
6. `mechanism_lattice`：每条因果命题一行。`docs/task/KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md`
   是这一项的主要素材。
7. `intuition_lattice`：`prior_injection` 弧型下这一项是重头戏——先验从哪来、anchor example、
   failure instances、named assumptions。每条 `named_assumption` 必须标 `tested_by`：
   要么是一个真实验族 id，要么字面量 `motivational_only`，**没有第三种状态**（规则 A10）。
   本项目的 `docs/feasibility/*.md` 里全是"前置条件 + 是否被验证"的记录，直接对应。
8. `non_claim_ledger`：本项目的独有优势——`docs/` 里有大量被杀掉的假设和明确的范围边界
   （人格漂移空结果、steering 两次不过、检测方案 4 负结果、sycophancy Phase A 空结果、
   读出天花板）。这一项写厚，是 positive 主线的保命装置。
9. WebSearch 只用于：acronym 撞名检查、数据集/指标的规范命名、子领域符号惯例。**不得**用于
   决定本文 claim 的措辞或强度（规则 A6）。每条 WebSearch 来的值标
   `source: websearch:<query>:<top_url>`。
10. 跑 `references/contract.md` §Validation gates 的全部闸门。
11. 术语对齐：`design_choice_lattice`/`intro_bullet_keyword`/`why_keyword` 里凡是指代
    `defense`/`stance`/`benign`/`detect`/`persona_drift` 这几条实验线的措辞，直接用
    [`docs/NAMING.md`](../NAMING.md) 的英文名列，不要另造一套论文自己的叫法。

**适用智能体**：**Opus 5** 主控。WebSearch 取证可派 Sonnet 5 的 `general-purpose` 子 agent。

**出口闸门（两道）**

- 机械闸门：`contract.md` §Validation gates 全过；**并且
  `python paper/evidence/check_claim_evidence.py` 退出码为 0**——它查三件事：`claim_ledger`
  没有引用 `superseded` 数字、没有引用不存在的 id、引用 `caveated` 数字的 claim 其 `notes`
  字段非空。它会把每条 `caveated` 证据的限定条件打印在对应 claim 旁边；**"措辞里有没有真把
  限定写出来"是判断题，脚本查不了，由你逐条确认**（规则来源见
  `paper/evidence/superseded.md` 第 1.1 节与第五节第 3 条，2026-09-06 用户已拍板）。
- **Tier-1 冷审**：**新开一个 Opus 5 会话**，只给它 `contract.yaml` + `paper/evidence/`，
  按 `references/reviewer.md` §Tier-1 reviewer 出裁决——"这个 lattice 代表一个可发表的论证吗"。
  裁决为否就回来迭代 Phase A，**不许进 Step 5**。

---

## Step 5 · Phase B/C：逐节交错构建

**顺序（skill 规则 A9，不可改）**：

```
§Problem Formulation → §Method → §Experiments → §Conclusion
                    → §Related Work → §Introduction → §Abstract
```

**关键**：**不许批量**。每一节走完 B→C→D 四小步再进下一节。批量做 Tier 2 再批量做 Tier 3
会退化成一次性起草，正是这个 skill 要防的东西。

### 每节的四小步

**B.k · 建 Tier 2 大纲** → 往 `paper/semantic.md` 追加 `## §<section>` 块
- 一个 bullet = 一个段落，role 标注，且**每个 bullet 至少引用一个 Tier 1 id**。
- **禁止完整句子**（规则 A2）。自查：`grep -nE '\. [A-Z]' paper/semantic.md` 命中数应接近 0。
- 规则 A11：凡是契约里 intuition 关联到形式化对象的地方，Tier 2 必须把 intuition 排在
  formalism **前面**。
- 适用智能体：**Opus 5**（这是分配问题）。

**B.k 冷审** → **新开 Opus 5 会话**，按 `references/reviewer.md` §Tier-2 reviewer 审段落分配。

**C.k · 生成 Tier 3 正文** → `paper/sections/<NN>_<section>.tex`
- 上下文按规则 A7 **隔离**：Tier 1 全量 + Tier 2 **仅本节** + `neurips-write/SKILL.md` +
  `neurips-write/references/<section>.md` + `phrase_bank.md`。**不加载其他节的正文**。
- 适用智能体：**Fable 5.1**（长文 + 强文风约束）；Opus 5 等效备选。

**D.k · 机械审计** → 破折号禁令、副词禁令、Why-X 不含公式、Q&A 呼应配对、Method 不出结果、
符号定义覆盖。见 `neurips-write-tiered/references/prose.md` §Mechanical audits 与
`neurips-write/SKILL.md` 的 close-out gates。
- 适用智能体：**Haiku 4.5** 或直接写 grep 脚本；Sonnet 5 兜底。能脚本化的就脚本化。

### 各节的重点与素材

| 节 | 重点规则 | 主要素材 |
|---|---|---|
| §Problem Formulation | 符号引入纪律；A11 intuition 先于 formalism | Step 3 方程、`docs/protocols/DATA_COLLECTION_PROTOCOL.md` 的通道 A–D 定义与 readout 约定 |
| §Method | Rule 21 不放结果、Rule 22 Why-X 无公式、Rule 24 每子节一个 Why-X | `src/koopman_ae/core.py`、`docs/method/*.md`、`persona_drift_control/src/` |
| §Experiments | Rule 5 driver sentence、Rule 7/17 RQ 句式、Rule 18 Q&A 呼应 | §1.4 的 RQ 映射；数字**只能**取自 `numbers.yaml` 的 `current` 条目 |
| §Conclusion | Rule 16 压缩 | §1.3 第 5 句（边界交付）必须在这里落地 |
| §Related Work | Rule 15 第三人称、只加粗类别、置于 Conclusion 前 | `docs/task/CONTROL_THEORETIC_LLM_RELATED_WORK.md`（2025–2026 控制论×LLM 调研，已有 gap 确认） |
| §Introduction | contributions 段从 `intro_bullet_keyword` 抽取，写前先按 A9 重新确认 | 写这一节常会发现 `narrative_arc` 与 Method/Experiments 实际交付的不符——那就回 Tier 1 改 arc |
| §Abstract | 纯抽取，最后写 | §Intro P1 + §Method overview + §Experiments headline + §Conclusion takeaway 各一句 |

**常见回传**（预期会发生，不是异常）：写 §Method 的 Why-X 时发现 Tier 1 的 `why_keyword`
不完整 → 改 Tier 1，重新生成该节；写 §Experiments 的 Findings 时发现某条 claim 的
`strength` 校高了 → 在 Tier 1 降级，重新生成。

---

## Step 6 · Phase D/E/F：审计与回传

### E · 调 `paper-audit`

**产物**：`paper/audit_findings.yaml` + `paper/audit_report.md`

- **单节审计**（每写完一节跑）：D1 confusing + D2 numbers + D4 claims + D7 compliance。
- **全文审计**（全部节写完后跑一次）：D1–D8 全维度。

**按维度派模型**：

| 维度 | 内容 | 模型 |
|---|---|---|
| D1 confusing | NeurIPS reviewer 冷读探针 | **Opus 5，必须新会话** |
| D2 numbers | 跨节数字一致性 | Sonnet 5（比对 `numbers.yaml`） |
| D3 baselines | baseline 完备性/时效/公平 | Opus 5 |
| D4 claims | 四类过度声称检测 | **Opus 5**（§1.5 红线在这里被检验） |
| D5 citations | 引用真伪、错误归属 | Sonnet 5 抽取 + Opus 5 裁决 |
| D6 setup | 多 seed、公平对比、测试集调参 | Opus 5 |
| D7 compliance | 24 条规则机械合规 | Haiku 4.5 / 脚本 |
| D8 figures | 图的感知编码 | Opus 5 或 Fable 5.1 |

### F.1 · 分流

读 `audit_findings.yaml`，按 `recommended_fix_tier` 分组：

- `T1` → 改 `contract.yaml`，把 `appears_in` 涉及的节全部标记重生成
- `T2` → 改 `semantic.md` 对应 bullet，标记该节重生成
- `T3` → 直接改 `.tex`（A1 允许的表层润色范围内）
- `data` → **升给用户**，需要重跑实验或补测量，agent 不得自行决定
- `bib` → 改 `main.bib`，量大就攒一批给用户
- `external` → 升给用户（例如需要拿到某 baseline 的代码）

**适用智能体**：**Opus 5**（分流是判断题）。改完后按 Step 5 的 C.k 重新生成被标记的节，
再对这些节重跑 `paper-audit` 确认发现已消解。

---

## Step 7 · 可选：`paper-pipeline` 托管

Step 5 和 Step 6 手动跑通一轮之后，可以把"写→审→分流→再审"交给 `paper-pipeline` 自动
循环，直到收敛（零 blocker + 零可自动修复的 major）或触到迭代上限。它每轮自动 git commit
兜底，并做震荡/回归检测，`data` / `external` 类发现自动升给用户。

**适用智能体**：**Opus 5** 主控；子 agent 按 §2.1 分层。
**前提**：不要在第一轮就用——先手动走完至少一节的完整 B→C→D→E→F 闭环，确认各闸门的行为
符合预期，否则自动循环会把一个系统性错误复制到全文。

---

## 基础设施步骤（与 Step 2–5 并行，不阻塞 Phase A）

### Step I1 · LaTeX 编译链

集群上 `pdflatex` / `latexmk` / `xelatex` **都没有**。两条路：

- 在 `/scratch/$USER/envs/` 下 conda 装 texlive（与项目现有环境放置约定一致），本地可编译；
- 或走 Overleaf，`paper/` 目录同步过去，本地只管生成 `.tex`。

**建议**：先定这一条。Step 5 每写完一节都需要能编译验证，拖到最后会积压一堆编译错误。
**适用智能体**：Sonnet 5。

### Step I2 · `main.tex` 骨架

venue 的官方 style 文件 + `\input{sections/*}` 的骨架。Step 5 开始前就位。
**适用智能体**：Sonnet 5。

### Step I3 · `main.bib`

D5 要查引用真伪，所以需要真 bib 而不是占位。素材：
`docs/task/CONTROL_THEORETIC_LLM_RELATED_WORK.md`（已有 2025–2026 调研与 gap 确认）、
`docs/evaluation/BASELINES.md`（baseline 对应论文）、`docs/references/`（本地 PDF 缓存，
已 gitignore）。
**适用智能体**：Sonnet 5 抽取条目 → Opus 5 抽查真伪（D5 的假引用是强模型才查得出的问题）。

### Step I4 · 图

用 `paperize-figure` skill 把 PNG/JPG 示意图转成白底 + 隐形 OCR 层的 PDF。
数据图按 `dataviz` skill 的规范做。
**适用智能体**：Sonnet 5；图的设计取舍用 Opus 5 或 Fable 5.1。

---

## 依赖与并行度

```text
Step 0 ──┬─> Step 2a ──> Step 2b ──┐
         │                          ├──> Step 4 ──> Step 5 ──> Step 6 ──> Step 7
         └─> Step 3 ────────────────┘
              (Step 1 已完成)

I1/I2/I3/I4 全程可并行，I1+I2 需在 Step 5 之前就位
```

- **可并行**：Step 2a 的六条证据线之间；Step 2a 与 Step 3；全部 I 系列。
- **强串行**：Step 4 → Step 5 → Step 6；Step 5 内部各节之间（A9 顺序 + 逐节交错）。

---

## 新会话启动语（直接粘贴）

> **Step 2a**：读 `docs/article/PAPER_EXECUTION_PLAN.md` 的 Step 2a，按里面的 schema 把
> `<某条证据线>` 的数字抽成 `paper/evidence/numbers.yaml`。只抽取不判断，`status` 一律填
> `unknown`，追不到出处的另记 `numbers_unsourced.md`。

> **Step 2b**：读 `docs/article/PAPER_EXECUTION_PLAN.md` 的 Step 2b 和已有的
> `paper/evidence/numbers.yaml`，裁决每条数字的作废关系，产出 `paper/evidence/superseded.md`
> 并回填 `status` / `superseded_by` / `supersede_reason`。

> **Step 3**：读 `docs/article/PAPER_EXECUTION_PLAN.md` 的 Step 3，从代码里抽出实际张量运算，
> 定稿 `paper/equations.tex`，每个方程配一句 `equation_operation`（以代码为准，不以注释为准）。

> **Step 4**：读 `docs/article/PAPER_EXECUTION_PLAN.md` 的 Step 1 和 Step 4，调用
> `neurips-write-tiered` skill 的 Phase A，基于 `paper/evidence/` 和 `paper/equations.tex`
> 建 `paper/contract.yaml`。

> **Step 4 冷审**（新会话）：只读 `paper/contract.yaml` 和 `paper/evidence/`，按
> `neurips-write-tiered/references/reviewer.md` 的 Tier-1 reviewer persona 做冷审，出裁决。
> 不要读 `docs/article/PAPER_EXECUTION_PLAN.md` 的主线叙述——那会污染冷读。

> **Step 5（某一节）**：读 `docs/article/PAPER_EXECUTION_PLAN.md` 的 Step 5，调用
> `neurips-write-tiered`，为 `§<节名>` 走完 B→C→D 四小步。上下文按规则 A7 隔离，
> 不要加载其他节的正文。

> **Step 6**：读 `docs/article/PAPER_EXECUTION_PLAN.md` 的 Step 6，对
> `paper/sections/<节>` 调用 `paper-audit`，产出 `paper/audit_findings.yaml`，然后按
> F.1 分流。`data` / `external` 类发现列出来给我，不要自行处理。

---

## 进度表

| Step | 内容 | 状态 | 负责模型 |
|---|---|---|---|
| 0 | 安装 skill | ◐ 2026-09-06 软链已建，待新会话验证 skill 列表 | — |
| 1 | 主线锁定 | ☑ 2026-09-06 | Opus 5 |
| 2a | 证据抽取 | ☑ 2026-09-06 533 条（core 57/defense 227/operator 73/sycophancy 30/readout 92/baseline 54），10 条抽查全部对得上出处 | Sonnet 5 ×6 |
| 2b | 作废判定 | ☑ 2026-09-06 533 条全部裁决（current 244 / caveated 172 / superseded 117）；十次事件记于 `paper/evidence/superseded.md`（计划点名的 5 次 + 另查出 5 次，其中 E7 为本次新发现）；`status` 改为三值，见该文件第一节 | Opus 5 |
| 3 | 方程定稿 | ◐ 2026-09-06 Sonnet 半段（`paper/equations_pseudocode.md`，8 项全覆盖）已完成，待 Opus 定稿 `equations.tex` | Sonnet 5 → Opus 5 |
| 4 | Tier 1 契约 | ☐ | Opus 5 |
| 4' | Tier 1 冷审 | ☐ | Opus 5（新会话） |
| 5 | Tier 2/3 逐节 | ☐ | Opus 5 / Fable 5.1 |
| 6 | 审计与分流 | ☐ | Opus 5 为主 |
| 7 | pipeline 托管 | ☐ | Opus 5 |
| I1 | LaTeX 编译链 | ☐ 待拍板：本地 conda texlive vs. Overleaf | Sonnet 5 |
| I2 | main.tex 骨架 | ☑ 2026-09-06 占位 `\documentclass`，待 I1 定案后替换 | Sonnet 5 |
| I3 | main.bib | ◐ 2026-09-06 Sonnet 半段（条目抽取）已完成，Opus 真伪核查（D5）待办 | Sonnet 5 → Opus 5 |
| I4 | 图 | ☐ | Sonnet 5 |
