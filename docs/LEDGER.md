# 实验账本（LEDGER）

> 规则见 [`../.claude/experiments.md`](../.claude/experiments.md) → *提交 GPU 作业前* / *仪器 : 方法 配额*。
> **每个 GPU 作业提交后追加一行。"它改变了哪个决定"填不出来 → 这个作业不该跑。**
> **配额：最近 10 个作业里"仪器"超过 5 个 → 停止提交，向用户报告。**

分类口径：
- **仪器** = 测量装置本身：judge 有效性 / 读出量程与饱和 / 执行器权威 / base rate /
  退化与截断检查 / 重判 / 方向校准 / 信号 screening。
- **方法** = 主线：辨识数据（激励臂）、闭环评测、baseline 对比、消融。
- **基建** = smoke test / 单测 / 数据生成，不计入配额。

---

## 一、历史回填（2026-08-26 → 2026-09-08）

⚠️ **粗粒度**。本段按 campaign 回填，不是逐作业——历史作业的"改变了哪个决定"多数已无法
可靠重建。它的用途只有一个：**给配额规则一个基线**。2026-09-08 之后的行必须逐作业、细粒度。

| 期间 | Campaign | 作业数 | 类 | 结果 |
|---|---|---|---|---|
| 08-27 | smoke test / 单测 / 用户脚本生成 | 3 | 基建 | — |
| 08-29–30 | `persona_drift` signal screening + drift confirmation | 2 | 仪器 | screening 三问全挂 → 该线放弃 |
| 08-30 | surface features backfill + input effect | 2 | 仪器 | — |
| 08-31 | adversarial screening（含 thinking 消融） | 2 | 仪器 | 信号存在，`defense` 线成立 |
| 08-31 | dose-response screening（含 eroded / wide-alpha / 重分析） | 4 | 仪器 | — |
| 08-31 | pressure screening（含 wide） | 2 | 仪器 | 属已放弃的 `persona_drift` 线 |
| 08-31 | `defense` Phase A 执行器权威 | 1 | 仪器 | — |
| 08-31 | `defense` Phase B 随机激励（辨识数据） | 1 | 方法 | Koopman 拟合的输入 |
| 08-31–09-01 | `defense` Phase E / F 闭环四臂 ×2 | 8 | 方法 | — |
| 09-01 | Phase G periodic 基线（+benign） | 2 | 方法 | 等代价对照 |
| 09-01 | Phase H 状态-动作交互 MPC（+benign） | 2 | 方法 | 自适应为真，但学到有害方向 |
| 09-02 | Phase I v-aligned 重跑（+benign） | 2 | 方法 | — |
| 09-03 | v-alignment 修复验证 + 安全方向校准 | 2 | 仪器 | **抓到 bug**：有害方向源自 v-alignment |
| 09-03 | refusal-direction 读出（+扩展投影） | 2 | 仪器 | 读出跟得上侵蚀，但取代不了 judge |
| 09-03 | scripted consistency check | 1 | 仪器 | — |
| 09-03 | Phase J 预算约束七臂 | 7 | 方法 | 预算设定成立，2 seed 分不开臂 |
| 09-03 | Phase J 臂比较 + 重判 | 2 | 方法/仪器 各 1 | — |
| 09-03 | LSTM baseline（v-aligned） | 1 | 方法 | — |
| 09-02–05 | `stance` sycophancy screening（SYCON → MMLU，含独立 judge） | 4 | 仪器 | 换过一次题库 |
| 09-04 | 连续读出（token 概率） | 1 | 仪器 | **G2 挂**：token 概率饱和 |
| 09-05 | `stance` Phase A 执行器权威（SYCON + MMLU，含 turn-1-exempt） | 5 | 仪器 | 两次空结果 → 开第三条线 |
| 09-06 | 独立 judge 重跑电池（Phase B/E/I/J + 自一致性） | 8 | 仪器 | **抓到 bug**：self-judging bias 藏了 4× effect size |
| 09-06 | Phase J 等代价随机调度 p75/p100 | 2 | 方法 | — |
| 09-06 | ERGO 最小执行器权威检查（+60 题扩样） | 4 | 仪器 | 确认，效应大 |
| 09-06 | ERGO Phase B 随机激励 | 1 | 方法 | 辨识数据 |
| 09-07 | soft safety judge / entropy 读出 | 2 | 仪器 | RC-1/RC-3 挂，读出族关闭 |
| 09-07 | Phase E zero-control len1024（截断对照） | 1 | 仪器 | 排除截断 |
| 09-07 | **ERGO Phase C 闭环九臂** | 9 | 方法 | k=1 退化，`fixed_last == always_reset` 是恒等 |
| 09-07 | D1 base-rate screening（含 alt 模型） | 2 | 仪器 | 9.8% base rate → 权威问题不可答 |
| 09-07 | **ERGO append-mode 重跑（B + C 三臂）** | 4 | 方法 | append 修好 layer 0，打坏 layer 2 |
| 09-08 | **ERGO prompt_profile 重跑（upB + upC 六臂）** | 7 | 方法 | R1：权威回来了，状态还在——E2/E3 的机制主张是错的 |
| 09-08 | K1 激励臂 + K1.2 离线重判 | 2 | 方法/仪器 各 1 | 运行中（job 15696221 / 15696222） |

### 基线统计

| | 数 | 占比 |
|---|---|---|
| 仪器 | **47** | 48% |
| 方法 | **48** | 49% |
| 基建 | 3 | 3% |
| **合计** | **98** | |

---

## 二、回填暴露的两件事

**1. 仪器 : 方法 ≈ 1 : 1。** 一半的 GPU 预算花在测量装置上。这个比例本身不必然是错的——
其中至少三次抓到了会毁掉论文的东西（v-alignment 有害方向、self-judging bias 的 4× 效应、
早停缺失造成的 training-budget artifact）。**问题是它没有预算、没有终止条件、而且阻塞主线。**

**2. "方法"这 48 个里，同一个比较被重跑了三遍。**
ERGO 闭环比较：Phase C（9 臂，09-07）→ append 重跑（3 臂，09-07）→ prompt_profile 重跑
（6 臂，09-08）= **18 个作业回答同一个问题**，因为 harness 在底下改了两次。
加上 `defense` Phase I（v-alignment 修复后重跑 MPC 臂，2 个），**48 个"方法"作业里约 11 个
是已答问题在仪器改动后的重跑；真正推进新方法问题的约 37 个，占全部 98 个作业的 38%。**
（Phase F 的 4 个**不算**重跑：那是 MT-Bench 良性代价检查，是另一个问题。）

> 这条比第 1 条更该管：仪器修复触发主线全量重跑，是"仪器工作阻塞主线"的**代价放大器**。
> 对策见 [`../.claude/experiments.md`](../.claude/experiments.md)——主线先跑、仪器怀疑并行挂着不阻塞主线；
> 仪器修复后按产物谱系筛重跑范围，不默认全量重跑；以及 `../.claude/global.md`"同一个仪器只修一次"。

---

## 三、逐作业记录（2026-09-08 起，细粒度）

| 日期 | job id | sbatch / 作业名 | 仪器/方法 | 状态 | 它改变了哪个决定 |
|---|---|---|---|---|---|
| 09-08 10:23 | 15690582 | `pdc-ergoUP-zero` | 方法 | ✅ COMPLETED 37m | R1：prompt_profile 下 zero-control 基线 |
| 09-08 10:23 | 15690584 | `pdc-ergoUP-always` | 方法 | ✅ COMPLETED 41m | R1：always_reset 臂 |
| 09-08 10:23 | 15690586 | `pdc-ergoUP-fixed_last` | 方法 | ✅ COMPLETED 44m | R1：fixed_last 臂——权威回来了、状态还在，E2/E3 机制主张被否 |
| 09-08 11:42 | 15696222 | `run_defense_rejudge_d1_qwen4b.sbatch` | 仪器 | ❌ **FAILED 1m06** — 并发 pip 竞争，非科学失败 | — |
| 09-08 11:43 | 15696226 | `pdc-ergoUP-excite` | 方法 | 🏃 RUNNING | R4：prompt_profile 下的激励臂，供辨识 |
| 09-08 11:43 | 15696227 | `pdc-ergoALT-always` | 方法 | ❌ **FAILED 1m16** — 同一并发 pip 竞争。**⚠️ 未重提** | R4 的 ALT 三臂缺 always_reset，比较不完整 |
| 09-08 11:43 | 15696228 | `pdc-ergoALT-fixed_last` | 方法 | 🏃 RUNNING | R4：ALT fixed_last 臂 |
| 09-08 11:43 | 15696229 | `pdc-ergoALT-zero` | 方法 | 🏃 RUNNING | R4：ALT zero-control 基线 |
| 09-08 11:43 | 15696221 | `run_defense_excite_qwen4b.sbatch` | 方法 | 🏃 RUNNING（8h 上限） | K2 的 DMDc 辨识用哪份激励数据；`B` 的 CI 是否覆盖 0（前提 2 的判据） |
| 09-08 11:49 | 15696281 | `run_defense_rejudge_d1_qwen4b.sbatch`（15696222 重提） | 仪器 | 🏃 RUNNING | K4 的报告口径用自判还是重判分数 |

**近 10 个作业配额检查**：仪器 2 / 方法 8 → ✅ 通过（阈值：仪器 > 5 即停）。

**⚠️ 未决**：`pdc-ergoALT-always`(15696227) 失败后未重提，ALT 臂组不完整。
失败原因是两个作业并发对同一 conda env 做 editable install 的竞争
（`__editable__.persona_drift_control-0.1.0.pth` 缺失），与实验内容无关，重提即可。
**这是提交前应当避免的并发模式**，见 `.claude/experiments.md` → *提交 GPU 作业前*。

<!-- 追加新行时：先答"它改变了哪个决定"，答不出就不要提交这个作业。 -->

---

## 四、待完成项（阻塞于 K 系列结果）

> **触发条件**：K1(15696221) / K1.2(15696281) 落地并完成 K2 辨识之后。
> 在此之前**不要动**这两项——它们会改到复现路径，而 K 系列正在跑。
> 来源：2026-09-08 对 `~/Pytorch-lightining-Hydra-Optuna-MLflow-Slurm-Project-Tempate-for-Scientific-Research`
> 的 `.claude/` 规则体系的借鉴分析。

### T-1 报告口径守卫（`_validate_report` 的同构物）

**现状**：`.claude/global.md` 的*报告口径*目前是纯散文。模板的同名规则是**运行前 raise 的
代码守卫**（`src/train.py:140 _validate_report`，在 `run_bench` 顶部拒绝非 `test_*` 指标、
拒绝 <3 seeds）。本仓库已有同构先例——`persona_drift.run_config_guard` + 元测试
`tests/test_run_config_guard.py`，但只此一处。

**要做**：写一个报告口径守卫，在 analyzer 产出"可报告数字"之前拒绝不合格的基础：

- 拒绝以**自判分数**作为报告数（自判 = 选择信号；报告必须来自独立重判）
- 拒绝 **<3 个不同 seed** 的聚合
- 拒绝把拟合损失 / 一步预测误差当作任务结果
- 报错信息里指回 `.claude/global.md` → *报告口径*

**开放问题（需用户裁决）**：加独立守卫模块 + 在 analyzer 入口调用，还是改
`modeling/evaluate.py`？后者被 `.claude/code.md` 列为不得改动的复现路径。
**倾向前者**——新模块 + 新调用点，旧路径逐字节不变，可用回归检验证明（参照 K2.1 的
`--n-folds` 补丁做法）。

**验收**：守卫 + 元测试；元测试证明"守卫若静默通过，它要防的失败模式在报告里不可见"。
守卫与元测试此后不得删除或绕过。

### T-2 统一 run summary schema + 谱系字段

**现状**：99 个 outputs 目录 schema 各异，`docs/LEDGER.md` 只能手写回填；
且产物不记 harness 版本——这是 ERGO 那 18 个作业重跑同一比较的直接成因（§2）。

**要做**：每个 run 目录写一份固定 schema 的 `run_summary.json`，**异常路径也必须写**
（模板的硬约束：`status` 为 `completed`/`failed`，`traceback` 在失败时是完整栈、成功时为 `null`）。

字段至少含：

| 字段 | 用途 |
|---|---|
| `job_id` / `sbatch` / `submit_time` / `wall_time` | LEDGER §3 自动生成 |
| `status` / `traceback` | 静默失败可见化（今天 15696227 就是靠 `sacct` 才发现未重提） |
| `git_sha` | harness 指纹 |
| `harness_switches`（`reset_mode` / `prompt_profile` / judge 版本 / `controller`） | 仪器改动后**只重跑真正受影响的 run**，不全量重跑 |
| `judge_kind`（`self` / `independent`） | 让 T-1 的守卫可判定 |
| `seeds` | 让 T-1 的守卫可判定 |

**收益**：LEDGER §3 从手写变自动生成；重跑范围可计算；配额检查可脚本化。

**开放问题**：是否顺带把 `.claude/experiments.md` 的配额检查写成 `PreToolUse` hook
（提交 `sbatch` 前拦截未登记作业）。hook 依赖 T-2 的 schema，所以排在 T-2 之后。
