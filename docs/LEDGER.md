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
| 09-08 | K1 激励臂 + K1.2 离线重判 | 2 | 方法/仪器 各 1 | 均已落地：K1 过 G-K1 五条；K1.2 独立 judge 无量程（ceiling 0.91） |

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
| 09-08 11:43 | 15696226 | `pdc-ergoUP-excite` | 方法 | ✅ COMPLETED 43m | **R1.5**（本行原误记为 R4）：upstream profile 下的激励臂，666 行，供 EK0/EK1 辨识 |
| 09-08 11:43 | 15696227 | `pdc-ergoALT-always` | 方法 | ❌ **FAILED 1m16** — 同一并发 pip 竞争。**⚠️ 未重提** | R4 的 ALT 三臂缺 always_reset，比较不完整 |
| 09-08 11:43 | 15696228 | `pdc-ergoALT-fixed_last` | 方法 | ✅ COMPLETED 1h19m | R4：ALT fixed_last 臂。**R4 已关闭，本产物不进结果** |
| 09-08 11:43 | 15696229 | `pdc-ergoALT-zero` | 方法 | ✅ COMPLETED 1h18m | R4：ALT zero-control 基线。**R4 已关闭，本产物不进结果** |
| 09-08 11:43 | 15696221 | `run_defense_excite_qwen4b.sbatch` | 方法 | ✅ COMPLETED 19m17（先被抢占、`--requeue` 后重跑） | K2 的 DMDc 辨识用哪份激励数据；`B` 的 CI 是否覆盖 0（前提 2 的判据）。**G-K1 五条全过**：500 行 / 100 个 attack_id（与 `d1_screen_qwen4b` 同集）/ `u_remind` 均值 0.518 / reminded 259 行 / 每轮跨轨迹 sd > 0 |
| 09-08 11:49 | 15696281 | `run_defense_rejudge_d1_qwen4b.sbatch`（15696222 重提） | 仪器 | ✅ COMPLETED 6m37 | K4 的报告口径：独立 judge 无量程（ceiling 0.91）→ **暂用自判 + 局限句**，`defense` 线收尾 |
| 09-08 16:14 | 15708895 | `run_ergo_ekA_branch.sbatch` | 方法 | ✅ COMPLETED 1h11m（18:08，759 对，45 题 × 3 seed） | EK-A：`B` 与 `b₂` 的 CI 能不能收窄到可判定——**决定 EK-B 闭环三臂（约 6.1 GPU-h）跑不跑**，以及 ERGO 线是进 S1/S2 还是停在 S3。**已答**：`B` 收窄到 +0.0250 [+0.0095,+0.0418]（G-EKA-1/3 过，EK-B 准入成立）；`b₂` **没有**收窄（G-EKA-2 斜率 CI 跨 0，但 MDE ±0.126 是预期量级的 4.5 倍 → 判为不可判定，非无状态依赖）。EK-B 的 `k` 与是否提交待用户裁决 |
| 09-08 19:03 | 15716425 | `run_ergo_ekA_entropy.sbatch` | 仪器 | ✅ COMPLETED 2m41（759 行） | **ERGO 闭环还有没有状态轴**：反事实对已给出每轮 reset 有没有用的真值，本作业补上文献自己用的那个状态（熵）。过 → 用熵重问前提 (iii)（T2 形式）与策略可分性，决定 EK-B（6.1 GPU-h）跑不跑；不过 → ERGO 停在 S3，预算转 `constraint` 线。**用户 2026-09-08「先执行2」即为重开熵读出族的显式签字**（Q17 / `.claude/global.md` *同一个仪器只修一次*）。**已答：不过。** `entropy_mean` 与 `shard_frac` 相关 −0.5162，控住斜坡后斜率从 −0.2453 衰减到 −0.0070（97%）；唯一稳健的预测量仍是确定性外生的 `shard_frac`（+0.1161，CI [+0.0503,+0.1796]，效应大于 MDE）。算子层面 T2 仍跨 0，策略仍是「最后 k 轮」。→ **ERGO 停在 S3，预算转 `constraint` 线**（待用户确认）。副产物：`entropy_answer_span` 控住斜坡后仍有信号，方向与 ERGO 的熵触发**相反**（笃定时 reset 有用、犹豫时没用），但小于 MDE，只作讨论段观察 |

**近 10 个作业配额检查**（2026-09-08 16:14，含 15708895）：仪器 2 / 方法 8 → ✅ 通过（阈值：仪器 > 5 即停）。

**近 10 个作业配额检查**（2026-09-08 19:03，含 `pdc-ergoEKA-entropy`）：仪器 3 / 方法 7 → ✅ 通过。

**`pdc-ergoEKA-entropy` 提交前六条核查**：① 用户已裁决（「先执行2」）；② 运行时估计 5–10 分钟，依据：前一个熵作业 15644575 跑 666 行 legacy 行只用 1m29（含加载），本次 759 行、回复更长（均 53 词）、上下文最长 12 轮，`--time 00:45:00` ≥4× 余量；③ `outputs/ergo_ekA_branch/entropy_readout.json` 提交前不存在，脚本自身也拒绝覆盖；④ 见上一行；⑤ 配额通过；⑥ 队列里无其它本项目作业，且 pip 已用 `flock` 串行化。

**本次改了脚本**（计划 §4.1 说「不需要改代码」是错的，两处）：`analyze_ergo_entropy_readout.py` 的重放①在 upstream profile 下会丢掉 system message、②对 `reset_mode="append"` 的 reset 用的是 overwrite 语义。新增 `--prompt-profile`（默认 `legacy` = 旧行为逐字节不变，已发布的 phaseB 产物仍可由裸命令行复现）与两道拒绝式守卫；7 条单测，含 legacy 回归检验。反事实文件会被守卫挡下——这正是照计划做会掉进去的坑。

**15708895 提交前六条核查**：① 用户已裁决；② 运行时估计 95 分钟，依据写在 sbatch 注释里（45 题 × 3 seed = 135 轨迹，每轮生成两次，R1 实测单次 19–23 s/轨迹）；③ `outputs/ergo_ekA_branch` 提交前不存在，不覆盖任何产物；④ 本行；⑤ 配额通过；⑥ 队列里无其它本项目作业，且 pip 已用 `flock` 串行化。

**⚠️ 未决（原文保留）**：`pdc-ergoALT-always`(15696227) 失败后未重提，ALT 臂组不完整。
失败原因是两个作业并发对同一 conda env 做 editable install 的竞争
（`__editable__.persona_drift_control-0.1.0.pth` 缺失），与实验内容无关，重提即可。
**这是提交前应当避免的并发模式**，见 `.claude/experiments.md` → *提交 GPU 作业前*。

**⚠️ 未决 → 已裁决（2026-09-08）**：`pdc-ergoALT-always`(15696227) 失败后**不重提**。
理由：它服务的 G-R4-1 只回答"我们的排序像不像上游"，按
[`experiments/ergo_fidelity_restoration_plan.md`](experiments/ergo_fidelity_restoration_plan.md)
第零节的 **Koopman 贡献过滤器**填不出论文任何一格，R4 整步关闭。**G-R4-1 记为永远不可判定。**
（原失败根因仍成立且已修：并发 editable install 竞争 → `flock`；这是提交前应当避免的并发模式。）

**`defense` 线收尾（2026-09-08 用户指令）**：K1.2 的独立 rejudge 无量程（ceiling share 0.91），
论文暂用自判分数 + 局限句，见 `.claude/global.md` → *报告口径* → **具名例外**。
**K3 闭环臂不提交**；K1(15696221) 跑完归档、K2 拟合是纯 CPU 收尾动作。
此后 GPU 预算转到 ERGO 线的 EK 相位，且 EK0/EK1 两步都是零 GPU。

<!-- 追加新行时：先答"它改变了哪个决定"，答不出就不要提交这个作业。 -->

---

## 三·五、campaign：`constraint` 线（约束保持，2026-09-08 开线）

计划 [`experiments/constraint_retention_plan.md`](experiments/constraint_retention_plan.md) ·
开线前筛查与死亡条件 [`experiments/constraint_signal_screening.md`](experiments/constraint_signal_screening.md) ·
术语 [`NAMING.md`](NAMING.md) 的 `constraint` 行。

**开线四步核查**（`.claude/experiments.md` → *开一条新任务线*）：

| 步 | 要求 | 状态 |
|---|---|---|
| 1 | 跑 `/research-experiment-design`，产出 kill criterion | ✅ 2026-09-08，产物即 `constraint_signal_screening.md` |
| 2 | 写下死亡条件 | ✅ 计划 §7 三条 + 筛查文件 §5 追加两条（K1/K2 不过即关线） |
| 3 | 论文三句话 + 那张表的空壳 | ✅ 筛查文件 §7 |
| 4 | `NAMING.md` 登记 + LEDGER 开 campaign 段 | ✅ 代号已登记（commit 1fdab1a）；本段即 campaign 段 |

**数据**：`resources/sequor/` 三个文件，2026-09-08 vendor 自 `deep-spin/SEQUOR`
commit `60bcdaca`，结构在**全部 200 个对话上核过**（不是从单个文件推断）：
约束在第 1 轮说一次、之后 0 次重提；前言有 **6 种措辞**，不许按字面串解析；
`turns` **截断到前 30 轮**（上游 120–210）。详见 `resources/PROVENANCE.md`。

**尚未提交任何 GPU 作业。** 第一个动作是 S0（判分校准）；S0-0 筛查臂在 S0 之后。

| 日期 | job id | sbatch / 作业名 | 仪器/方法 | 状态 | 它改变了哪个决定 |
|---|---|---|---|---|---|
| 09-08 22:0x | 15719117 | `run_sequor_judge_smoke.sbatch` | **基建**（不入配额） | ✅ COMPLETED 4m46 | **定下 S0 全量的跑法**：实测 s/row < 3 → 直接在现有 env 上跑 2000 行（约 1.7 h/候选）；≥3 → 先建独立 vLLM env（计划 §8）。同时验 `--max-new-tokens 512` 够不够（解析失败率 / 触顶率）。24 行分层切片，`mode=debug`，**数字不作为证据**。**已答**：**7.42 s/row** → 全量 2000 行 **4.12 h/候选** → 按预注册规则（≥3 s/row）**先建独立 vLLM env**；`--max-new-tokens 512` **够用**（judge 输出中位 ~185 token、最长 ~340，解析失败 0/24、触顶 0/24） |
| 09-08 22:1x | 15719166 | `build_sequor_vllm_env.sbatch` | **基建**（CPU-only，不入配额） | ✅ COMPLETED 1h16m40（09-08 23:10） | 建 `/scratch/hcao2/envs/sequor_vllm`。**不碰共享 env**——vLLM 自带 torch，装进 `persona_drift_pilot` 就等于在 `defense`/`stance`/ERGO 已发布产物底下换数值栈。作业开头 `if [ -d ]` 拒绝重建；回滚 = 删目录重跑 |
| — | — | `run_sequor_judge_vllm_consistency.sbatch` | **基建**（不入配额） | ⏸ 已写好并干跑，待裁决 | **S0 全量在哪个后端出数**：vLLM 与 HF 在**同一批 24 行**（`--limit 24` 的确定性分层切片，单测证明与 smoke 报告逐行同集）上逐条比对判决。≥22/24 → S0 全量走 vLLM（约 0.5 h/候选）；≤21/24 → 回 HF 路径（4.12 h/候选），差异按工程 bug 修（chat template / `enable_thinking` 传递），**不作为科学结论** |
| — | — | `run_sequor_s0_report_judge_vllm.sbatch` | 仪器 | ⏸ 已写好并干跑，待裁决（consistency 过后才提） | **定下报告口径 judge**（`.claude/global.md` → *报告口径*：本线所有可报告数字的来源）。G-S0-1（一致率 ≥0.80 且 κ ≥0.60）过 → 进 S1；只过 G-S0-2 → 用但每处标注"judge 弱 → MDE 抬高"；都不过 → **换候选一次**（Qwen3-30B-A3B），第二次不过 → **关线**（计划 §7 死亡条件 1）。同址报 G-S0-0 长度地板（配对 64.8% / 不配对 59.7%） |
| — | — | `run_sequor_s0_inloop_judge_vllm.sbatch` | 仪器 | ⏸ 已写好并干跑，待裁决（**agent 模型待用户签字**） | **定下 in-loop 选择信号 judge**（= agent 同一个服务模型，自判，只用于选动作）。预注册判据：平衡准确率 ≥0.65 **且**高出 G-S0-0 配对长度地板 ≥10 点。过 → S0-0 筛查臂与 S3 的 in-loop 读出用它；不过 → **换仪器**：报告 judge 同时跑 in-loop（S3 变慢、无自判），**不许下调阈值**——否则闭环反馈的"状态"是回复长度的代理，控制器在调节啰嗦程度 |

**`run_sequor_judge_smoke` 提交前六条核查**：① 用户已裁决（「提交」）；② 运行时估计 5–15 分钟（约 1 分钟加载 + 24 行；s/row 未知正是它要买的量），`--time 00:30:00`；③ `outputs/sequor_s0_smoke/` 提交前不存在，脚本自身亦拒绝覆盖；④ 见上行；⑤ 近 10 个作业 仪器 3 / 方法 7 ✅（本作业记基建）；⑥ 队列无其它本项目作业，pip 已 `flock`。

**三个 S0 作业的提交前六条核查**（2026-09-09）：① **用户已裁决**（2026-09-09 三问三答：
「先提 consistency，过了我直接接全量」= consistency 过 ≥22/24 即**预授权**接着提两个全量、
≤21/24 则停下报告；agent 与 in-loop judge = **Qwen3-4B-Instruct-2507**（上游 tuples/3 用的就是它，
计划 §1 的量程先验读自它的曲线；计划 §9 写的 `Qwen3-4B` 是不一致处，以本裁决为准）；
S0-0 筛查臂**按 EK-A 先例记「方法」**）；
② 运行时估计与假设写在每个 sbatch 的注释里——consistency **5–10 min**（`--time 00:40`）、
报告 judge **15–30 min 中位 / ~1 h 悲观**（`--time 02:00`）、in-loop judge **8–20 min**（`--time 01:30`），
假设：A100-PCIE-40GB（smoke 实测落到的卡）/ 2000 行 / 提示均长 ~650 token（gold 回复 median 429 词、mean 427、p90 679、max 1936）/
judge 输出 median ~185 token（smoke 实测 741 字符中位、1351 最长）/ cap 512 / greedy / thinking off；
**未实测的量**：4B 模型在判分前的啰嗦程度（smoke 用的是 14B），它长则 decode 项变大；
③ 三个输出目录 `outputs/sequor_s0_vllm_consistency/` `outputs/sequor_s0_report_judge_qwen3_14b/`
`outputs/sequor_s0_inloop_judge_qwen3_4b_instruct_2507/` 提交前均不存在（干跑逐条核过），脚本自身亦拒绝覆盖，**一个作业一个目录**；
④ 见上表三行；⑤ 配额见下；⑥ **三个作业都不做任何 pip install**——vLLM env 里没装本项目，
runner 按文件路径加载共享模块（`load_shared_module`），因此结构上不可能重演 2026-09-08 的并发 editable install 竞态。

**近 10 个作业配额检查**（2026-09-09，含上表三个作业）：仪器 **4** / 方法 3 / 基建 3 → ✅ 通过（阈值：仪器 > 5 即停）。

**S0-0 筛查臂的分类（2026-09-09 用户裁决）：记「方法」**，按 EK-A（15708895）的先例——
同前缀反事实分叉臂产出的 240 对**就是后续拟合的辨识数据**，不是另造一个测量装置。
（分类口径里"信号 screening = 仪器"的字面读法会把它记成仪器，那会让本线的开线序列
S0 校准 ×2 + S0-0 共 3 个仪器、在 S1 之前就绑住配额；EK-A 当时记的是方法，同一设计同一归类。）
**后果**：若报告 judge 需要换候选（Qwen3-30B-A3B），那是第 5 个仪器，仍在阈值内但只剩 1 格。

**2026-09-09 干跑抓到的两件事**（都在申请 GPU 之前）：

1. **vLLM runner 在它自己的 env 里 import 就死**：`persona_drift/__init__.py` → `.analysis` → `pandas`，
   而 `sequor_vllm` env 按计划 §8 只有 vLLM。修法是 `load_shared_module()` 按文件路径加载
   `sequor_calibration_metrics` / `sequor_constraint_judge` / `run_provenance`，**不走包 `__init__`**——
   保住"两个后端读同一份 prompt/parser/metric 源码，不是两份拷贝"这个 `--compare-to` 赖以成立的性质。
   6 条单测，其中一条把这次的失败本身钉成回归检验（屏蔽 `pandas`/`torch`/`transformers` 后仍须加载成功）。
2. **本仓库没有任何产物记 harness 指纹**（`outputs/ergo_ekA_branch/run_config.json` 也没有），
   而这是 `.claude/global.md` → *产物与谱系* 的硬约束、也是 §2 里 ERGO 那 18 个作业的直接成因。
   新模块 `src/persona_drift/run_provenance.py`：`git_sha` / `git_branch` / `git_dirty` / `n_dirty_paths` /
   `job_id` / `node` / `python` / `argv` / `switches`，**取不到指纹就 raise**（不返回 `git_sha: null`——
   那会让"产物不可追溯"这个失败在产物里长得像已经追溯过了，参照 `run_config_guard` 元测试的论证）；
   9 条单测；已接进两个判分 runner 的报告。**这也是 §4 T-2 的第一块。**

---

## 四、待完成项（阻塞于 K 系列结果）

> **触发条件**：K1(15696221) / K1.2(15696281) 落地并完成 K2 辨识之后。
> 在此之前**不要动**这两项——它们会改到复现路径，而 K 系列正在跑。
> 来源：2026-09-08 对 `~/Pytorch-lightining-Hydra-Optuna-MLflow-Slurm-Project-Tempate-for-Scientific-Research`
> 的 `.claude/` 规则体系的借鉴分析。

**2026-09-08 更新（`defense` 线收尾后）**：触发条件放宽——K3 不跑了，T-1/T-2 只等 K1(15696221)
落地与 K2 拟合，之后即可开工，且**都不占 GPU**。T-1 的守卫必须能表达
`.claude/global.md` → *报告口径* 的那条**具名例外**（`defense` 线暂用自判）：
做法是让守卫**默认拒绝自判**，例外以显式参数传入并要求同时给出局限句字段，
**而不是**把自判那条判据整个去掉——去掉就等于这条例外悄悄外溢到其他线。

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

**2026-09-09 已落地第一块**：`src/persona_drift/run_provenance.py`（`git_sha` / `git_branch` / `git_dirty` /
`n_dirty_paths` / `job_id` / `job_name` / `node` / `python` / `started_at` / `argv` / `switches`；缺指纹即 raise；
9 条单测），并接进 `constraint` 线的两个 S0 判分 runner。**未动的部分**：统一 `run_summary.json` schema、
`status` / `traceback`（异常路径也必须写）、把它接到其余脚本、以及 99 个既有 outputs 目录的回填。

**收益**：LEDGER §3 从手写变自动生成；重跑范围可计算；配额检查可脚本化。

**开放问题**：是否顺带把 `.claude/experiments.md` 的配额检查写成 `PreToolUse` hook
（提交 `sbatch` 前拦截未登记作业）。hook 依赖 T-2 的 schema，所以排在 T-2 之后。
