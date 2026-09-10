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
| 09-09 09:21 | 15738603 | `run_sequor_judge_vllm_consistency.sbatch` | **基建**（不入配额） | ❌ **FAILED 3m25** — 环境失败，非科学失败：vLLM 的 sampler 走 FlashInfer，其采样 kernel 在第一个 token 时 JIT 编译、需要 nvcc/CUDA_HOME，计算节点上没有（本 env 的 torch 用的是 wheel 形式的 CUDA 库，不是 toolkit）。已加载权重、抓完 CUDA graph、KV cache 6.14 GiB / 40,256 token 都正常。修法：`VLLM_USE_FLASHINFER_SAMPLER=0`（greedy 解码根本不需要这些 kernel）；**不用 `module load cuda/13.0.2`**——那会把第二个 CUDA runtime 放到本 env 依赖的 wheel 之前，等于在仪器底下换数值栈。三个 sequor sbatch 同时打上此修。**已重提**（下一行） | **S0 全量在哪个后端出数**：vLLM 与 HF 在**同一批 24 行**（`--limit 24` 的确定性分层切片，单测证明与 smoke 报告逐行同集）上逐条比对判决。≥22/24 → S0 全量走 vLLM（约 0.5 h/候选）；≤21/24 → 回 HF 路径（4.12 h/候选），差异按工程 bug 修（chat template / `enable_thinking` 传递），**不作为科学结论** |
| 09-09 09:35 | 15738680 | `run_sequor_judge_vllm_consistency.sbatch`（15738603 重提，加 `VLLM_USE_FLASHINFER_SAMPLER=0`） | **基建**（不入配额） | ✅ COMPLETED 2m20 | 同 15738603：**S0 全量在哪个后端出数**。≥22/24 → 全量走 vLLM，并按用户预授权直接提两个全量；≤21/24 → 停下报告，回 HF 路径。**已答：24/24 判决逐条相同（100%）** → 后端核对通过，S0 全量走 vLLM。吞吐 **0.401 s/row**（HF 是 7.42，**18.5×**）→ 2000 行约 13 min 批处理 + 约 3 min 加载/编译。同一批 24 行上两后端的准确率、κ、长度地板逐位相同 |
| 09-09 09:47 | 15739195 | `run_sequor_s0_report_judge_vllm.sbatch` | 仪器 | ✅ COMPLETED 7m22（批处理 6m17，**0.185 s/row**） | **定下报告口径 judge**（`.claude/global.md` → *报告口径*：本线所有可报告数字的来源）。G-S0-1（一致率 ≥0.80 且 κ ≥0.60）过 → 进 S1；只过 G-S0-2 → 用但每处标注"judge 弱 → MDE 抬高"；都不过 → **换候选一次**（Qwen3-30B-A3B），第二次不过 → **关线**（计划 §7 死亡条件 1）。同址报 G-S0-0 长度地板（配对 64.8% / 不配对 59.7%）。**已答：G-S0-1 过**（2000 行，一致率 **0.8944** ≥0.80、κ **0.7886** ≥0.60）→ **报告口径 judge 定为 Qwen3-14B，不需要换候选**。Gold-Yes 0.8184 / Gold-No 0.9699（上游 GPT-oss-120B 是 0.883/0.988，本判分器是 14B）；G-S0-0：高出配对长度地板 0.6480 **+24.6 点**，即它确实在读约束而不是读长度；解析失败 12/2000（0.6%），与触顶 12 条同源。后端一致性同址复核：与 HF smoke 的 24 行 **24/24 相同** |
| 09-09 09:47 | 15739196 | `run_sequor_s0_inloop_judge_vllm.sbatch` | 仪器 | ✅ COMPLETED 5m29（批处理 3m31，0.099 s/row） | **定下 in-loop 选择信号 judge**（= agent 同一个服务模型，自判，只用于选动作）。预注册判据：平衡准确率 ≥0.65 **且**高出 G-S0-0 配对长度地板 ≥10 点。过 → S0-0 筛查臂与 S3 的 in-loop 读出用它；不过 → **换仪器**：报告 judge 同时跑 in-loop（S3 变慢、无自判），**不许下调阈值**——否则闭环反馈的"状态"是回复长度的代理，控制器在调节啰嗦程度。**已答：过。** 平衡准确率 **0.8855** ≥0.65，高出配对长度地板 **+23.8 点** ≥10；一致率 0.8885、κ 0.7754（比 14B 只低 1.3 点，自判信号比预想的干净）。**但抓到一个必须处理的东西**：解析失败 **162/2000 = 8.1%**，其中 **155 条是在 512 token 处触顶**（14B 只有 12 条 / 0.6%）。`--max-new-tokens 512` 这个上限是在 **14B** 上测的（smoke 15719117），**对 4B 不成立**——它在给出 `Final Verdict` 之前更啰嗦。后果不在校准上（8.1% 只抬噪声），而在 **S3 的闭环状态**上：`graded_readout` 对任一约束解析失败的整轮返回 None（防读出量程悄悄漂移），k=3 时整轮不可用概率 ≈ 1−(1−0.081)³ ≈ **22%**，控制器会有五分之一的轮次没有状态可反馈 |
| 09-09 10:00 | 15742083 | `run_sequor_s0_0_smoke.sbatch` | **基建**（不入配额） | ✅ COMPLETED 8m20（生成 4m1，**3.14 s/生成**，batch 宽 2） | **定下 canonical 臂的 cap 与运行时**：2 题 × 20 轮 = 78 次生成、38 对。**已答**：回复 token **中位 511 / 最长 866**，**0/78 触顶**（cap 1024）→ 过「触顶率 >5% 不合格」这条（筛查 §十 第 4 条）。**注意**：判分 smoke 当初在 14B 上认可的 cap 512，在这里会截掉**一半**回复。配对机制端到端跑通：`assert_pairs_share_prefix` 38 对全过、38 条 reminded 消息逐条等于「base 消息 + 约束块」、`inserted_tokens` 37/41、echo jaccard 中位 0.1477 最大 0.2426、逐字重复 0 条（ERGO 那边 echo 是主失败模式：逐字 0.2644 / jaccard 0.6051，这个任务上不是）。谱系干净：`git_dirty=false`，sha `e81c042` |
| 09-09 13:2x | 15756689 | `run_sequor_s0_0_arm.sbatch`（canonical 臂，N=12 / T=20 / **228 对**） | **方法** | ⚠️ **COMPLETED 11m50，但不合格**：触顶 **87/468 = 18.6%**，过不了预注册的「>5% 不合格」（筛查 §十 第 4 条）。468 行 / 228 对 / 前缀 228 对全过 / 1.19 s/生成（batch 12，比 smoke 的 3.14 快 2.6×）。**未判分、未自行抬 cap** | **这条线活不活**：K1 不过 → 关线（`defense` 的死法，且不许第二次修 judge）；K2 不过 → 关线（ERGO 的死法，衰减 97%）；K2 判 UNDECIDABLE → 报告并交还，不关线；K1+K2+K3 全过 → 开 S1。运行时估计 **12–30 min**（依据全部来自 15742083 实测，写在 sbatch 注释里；悲观界 = smoke 速率不变 × 468 次生成 = 24.5 min + 加载 4 min），`--time 02:00:00` ≥4× |
| 09-09 14:4x | 15758992 | `run_sequor_s0_0_arm_cap2048.sbatch`（15756689 重跑：cap 2048 / `max_model_len` 49152） | **方法** | ✅ COMPLETED 18m51（生成 15m7，2.01 s/生成） | 同 15756689（这条线活不活），但先把触顶这个混淆去掉。**预注册**：触顶判据**按题**算、阈值 5%；仍有单题超标 → **报告并交还**，不再抬 cap、不剔题。判据已做成守卫：runner 写 `token_cap_by_item`，analyzer 的 `refuse_if_the_cap_bound()` 拒绝不合格的臂，唯一通路是逐个 `--exclude-item` 并记进闸门报告。**已答：cap 判据全过——触顶 0/468，12 题逐题 0.0%。**回复 token 中位 603、最长 1864（< 2048，有余量）；两个长约束题现在的中位是 `tuple_294_1` 1653、`tuple_245_1` 1294——它们确实要 1024 的 1.3–1.6 倍，不是模型跑飞。228 对前缀全过。**这份臂产物是 S0-0 闸门的输入。** 估计 **20–45 min**（跨度来自 `--gpus a100:1` 会给 40GB 或 80GB：15756689 拿到 80GB、KV 62.6 GiB / 45.6 万 token；49k/请求时 80GB 并发 ~9.3×、40GB ~3.9×） |
| 09-09 15:1x | 15759226 | `run_sequor_s0_0_score_report_judge.sbatch`（14B，cap 512，independent） | 仪器 | ✅ COMPLETED 5m03（1404 次判分 2.3 min，0.10 s/次） | 三道闸门算在这份读出上（筛查 §十 第 3 条）。**已答：读出干净**——解析失败 **0/468**、判分触顶 **0**、`y` 取满四个值 {0, ⅓, ⅔, 1}、全体均值 0.8319（偏高，K1 的天花板份额留给闸门分析判） |
| 09-09 15:2x | 15759346 | `run_sequor_s0_0_score_inloop_judge.sbatch`（4B，cap **1024**，self） | 仪器 | ✅ COMPLETED 4m45（1404 次判分 2.2 min） | ① cap 512→1024 能不能把 8.1% 解析失败压下来（k=3 时 ~22% 的轮次没有状态）——**就在这个作业里量，不另开仪器作业**；② 两个 judge 按 `u` 拆分的系统性差距（计划 §2 那条风险的第一手弱答案）。**已答 ①：cap 抬到 1024 有效**——不可用轮次 **25/468 = 5.3%**（cap 512 时按 8.1% 单约束失败率推算的整轮损失是 ~22%），判分触顶 17 条。**已答 ②**：自判比独立判**更严**，u=0 上低 0.0430、u=1 上低 0.0275，逐轮精确一致 78.2% / 76.6%。**方向性差 0.0155**（自判在被提醒的行上相对更宽松），是 Δy=0.0643 的 24%——报告数字都出自独立判，但这条对 S3（in-loop 判分驱动控制器）是要写进局限的 |

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

结果见 [constraint_results.md](experiments/constraint_results.md) §S0-0 三道闸门、§⚠️ 上游保真度从未验证。

### 保真度臂（2026-09-09，用户「开始测试」）

| 日期 | job id | sbatch | 类 | 状态 | 它改变了哪个决定 |
|---|---|---|---|---|---|
| 09-09 16:4x | 15761176 | `run_sequor_fidelity_arm.sbatch`（2×2，zero-control，**一个作业两相**：生成 + 判分） | 仪器 | ✅ COMPLETED 39m33（生成 30m1 / 960 行，判分 4m9 / 2880 次；触顶 0/960、解析失败 0/960） | **1.7× 的服从度差距是不是我们自己选的两个旋钮造成的**。变体：{约束在 user turn 1 / 在 system message} × {greedy / **模型自带默认**（运行时从 `generation_config` 读，不硬编码：T=0.7 / top_p=0.8 / top_k=20）}。同样的 12 题，与 S0-0 臂逐题配对，只差 harness |
| 09-09 17:3x | 15761810 | `run_sequor_s0_0_arm_fidelity_harness.sbatch` | **方法** | ⚠️ **FAILED 54m08 — 我的代码 bug，但生成没有丢**（**此格已更正**：先前写成"52 分钟全部丢失"是错的）。三个 seed 各 468 行跑完（16.6 / 17.3 / 18.3 min），`trajectories.jsonl`（1404 行）与 `run_config.json` **都已落盘**，脚本崩在最后一步`arm_report.json` 的 `NameError: turn_clock`（多 seed 重构时删了定义、漏了这处引用）。**产物有效**：720 base + 684 reminded、3 seed、12 题、`constraints_in_system=true`、T=0.7/top_p=0.8/top_k=20、sha `47dfd4c`。缺的只有那个派生报告 | **K1/K2/K3 重算**——这次的失败或通过不再是 harness 造的。1404 次生成 / **684 对**；两处判据改动（K1 范围 t2..t20、3 seed）已签字，记在筛查 §十 第 5 条。运行时估计 **75–110 min**（依据：cap2048 臂 2.01 s/生成；保真臂 `system_default` 变体 2.10 s/生成、中位 804 token vs greedy 524 → 采样 + system 位置更费 token。三相：生成 50–90 min + 14B 判分 11 min + 4B 判分 8 min），`--time 03:00:00` ≥1.6× 悲观值 |
| 09-09 18:1x | 15762950 | `run_sequor_s0_0_arm_fidelity_harness.sbatch`（15761810 重提） | **方法** | ✅ **FAILED 2s — 守卫按设计拒绝**：`refusing to write into existing outputs/sequor_s0_0_branch_arm_fidelity_harness`。**这正是要的行为**——15761810 的产物在那里，重跑会覆盖 52 分钟的有效生成。**不需要重新生成**，改为从盘上的行重建派生报告 | 同 15761810：**K1/K2/K3 重算**，这次的判定不再是 harness 造的。1404 次生成 / 684 对 / 3 seed；判据两处改动见筛查 §十 第 5 条 |
| 09-09 18:4x | 15763577 | `run_sequor_s0_0_score_fidelity_harness.sbatch`（只判分两相，**不含生成**） | 仪器 | ✅ COMPLETED 19m16（14B 判分 8m / 4B 判分 8m，各 4212 次）。独立判解析失败 **0/1404**；自判 60/1404（4.3%，cap 1024） | 保真 harness 臂的两份读出。**用户裁决「执行a」**：`tuple_294_1` 以 `--exclude-item` 在**闸门这一步**具名剔除（筛查 §十 第 6 条），判分仍覆盖全部 1404 行 → 裁决可逆、不必再花一个 GPU 作业。闸门将在 **11 题 × 3 seed × 19 轮 = 627 对**上计算。估计 25–35 min |
| 09-09 21:0x | 15768372 | `run_sequor_response_length_probe.sbatch`（全库 200 题 × 3 轮 × 1 seed，zero-control，保真 harness，cap 2048） | **基建**（不入配额） | ✅ COMPLETED **4m55**（生成 2.3 min，**0.23 s/生成**，batch 200）。**结论有用但对照没过，见下** | **定下 S1 的 cap 与它跑不了的题**。S0-0 只有 12 题、可以逐题人看；S1 要 N=40 且必须出 gold 全覆盖的 16 题（筛查 §9.2），题集大到看不过来。按题的 cap 判据已经响过两次，两次都是**约束文本**造成的，不是模型：15756689 整体 18.6% 而 `tuple_294_1` 100%；保真 harness cap 2048 下同一题 21.4%、其余 11 题 0.0%。**先试了正则预筛，不够用**——它命中两个已实测的长题，但也命中 `tuple_245_1`（中位 1294、从未触顶），灵敏而不特异，拿它选题会误伤。**3 轮为什么能外推到 20 轮**：本任务的回复长度是约束集合的函数、不是上下文深度的函数（15756689 的触顶逐轮平坦，每轮约 4/24，而 12 题里 2 题吃掉 75/87）——这是**假设**，写在脚本 docstring 里：轮 3 贴着边界的题到轮 20 可能越界，所以它**定 cap、点名题**，不认证一条臂。运行时估计 **12–35 min**（依据：600 次生成 / cap 2048；cap2048 臂实测 2.01 s/生成 @batch 12、保真臂采样变体 2.10；本作业 batch 200 但 `--max-model-len 12288` 下 80GB 卡实际并发约 35–40 → 约 3× 的 batch-12 吞吐 → 0.7–2.1 s/生成 → 7–21 min，加载约 4 min），`--time 01:30:00` ≥2.5× 悲观值。**不是证据**：无闸门读它，数字不进论文 |
| 09-09 22:4x | 15768661 | `run_sequor_s1_pilot_arm.sbatch`（4 个日程臂 × 12 题 × 20 轮 × 3 seed = **2880 次生成**，保真 harness，一个作业三相：生成 + 14B 判分 + 4B 判分） | **方法**（EK-A 先例） | ✅ COMPLETED **2h15m35**（09-10 00:49；生成 99.4 min / 2.07 s/生成、14B 判分 15.4 min、4B 判分 15.7 min；落在 2.2–3.1 h 估计内） | **定下 S1 的 N，以及 S1a 到底能不能被定量**。计划 §4 按**假设**的每题差值 sd≈0.30 定出 MDE≈0.13@n=40，并在同一句里要求「提交前用 smoke 实测的 sd 重算」——**这个重算一直做不了**：本线从没跑过**持续臂**。S0-0 分叉臂量的是同前缀的**一步**增益，S1a 比的是「每轮都提醒」与「从不提醒」两条轨迹的**终点**，两个量、两个方差；ERGO 已经为「拿一个量的 MDE 去豁免另一个量的闸门」付过账（终点 MDE 0.49 vs 一步 se 0.013）。**臂表**：`zero_control` / `constant_remind`（S1a 剂量对比，按题配对）+ `bernoulli` / `antithetic`（S1b 反相配对，每个 (题,轮,seed) 格恰好一个被提醒侧）。**题集刻意不动**——N=40 需要一个尚未做出的选题裁决（200 题里只有 16 题 gold 全覆盖，筛查 §9.2），试点不该用默认值替用户做这个决定；选题的长度那一半由 15768372 并行在量。**它买到什么**：每题终点差值 sd → S1a 在 n=40 的真实 MDE；**持续**提醒下 `y` 在第 20 轮还有没有量程（分叉臂答不了，它从不走被提醒的轨迹）；G-S1 的 u 平衡与格覆盖在真行上的检查；S1 运行时估计的吞吐。**不是 S1、也不是判定**：n=12 对着一份为 n=40 写的计划，无闸门读它，终点数字是方差估计；**预注册后果：试点效应小 → 重新给 S1 定 N，绝不据此宣布执行器没有权威**（权威已由 K3 在一步上答过：+0.1233，CI [+0.0595,+0.1898]，1.30×MDE）。运行时估计 **2.2–3.1 h**（依据：2880 次生成 @batch 12；cap2048 臂实测 2.01 s/生成、保真 harness 臂 1404 次 / 52.2 min = 2.23；持续提醒每轮重注入约束块，prefill 随轨迹增长，这一项前两个臂都没测过 → 取 2.1–2.6 s/生成 = 100–125 min，悲观 150 min；二相 8640 次判分 × 0.10 s ≈ 15 min + 加载 4 min；三相同量级），`--time 06:00:00` ≈2× 悲观值。干跑：三个 seed 的 bernoulli u 均值 0.4825 / 0.5088 / 0.4561，**都在 G-S1 的 [0.45,0.55] 内**；轮 1 无动作、反相精确互补 |

结果见 [constraint_results.md](experiments/constraint_results.md) §保真度臂、§S1 试点。

### campaign 段：S1 辨识臂（2026-09-10 用户裁决「按照你的建议提交」）

| 日期 | job id | sbatch / 作业名 | 仪器/方法 | 状态 | 它改变了哪个决定 |
|---|---|---|---|---|---|
| 09-10 09:5x | 15772956 | `run_sequor_s1_arm.sbatch`（4 臂 × **40 题** × 20 轮 × 3 seed = **9600 次生成**，保真 harness，`--item-selection bank`，生成单相） | **方法** | ⏳ PENDING（09-10 09:56 提交，partition `work1`） | **它就是主线**：这批行同时是前提 (i) 的检验（S1a）、**S2 的 Koopman 训练集**（`y_{t+1}=Ay_t+Bu_t+c` 拟合 S1b）、以及 S3 的 zero-control 下界臂（计划 §4）。规模不再来自假设：试点（15768661）把每题差值 sd 从假设的 0.30 换成实测 0.144，拆成题间 0.1317 / 题内跨 seed 0.1000——**题间项占主导，所以 seed 加到 5 几乎不动 MDE，只有 N 能推动它** → N=40 / 3 seed，MDE(n=40,s=3) **0.0636** 对试点差值 +0.1397（2.20×）。**估计量在提交前签死**：t15..t20 窗口、独立判、按 (题,seed) 配对；t20 单轮只作次要量，事后不许提拔。**题集裁决的代价同址报**：3/40 题 gold 全覆盖，**78/120 条约束（65%）在 judge 校准集之外**；人工核验集**已裁决推迟**，随之定死——**G-S2-1 在这批行上的弱结果不构成关线结果**（弱系数有两个成因，S2 分不开）。**预期 1–3 题咬住 cap 2048**（`tuple_139_1` / `128_1` / `131_1` 落在越界 band，探针阳性对照失败只给下界）：不抬 cap、不自行剔题，只走具名 `--exclude-item`。运行时估计 **3.0–5.9 h**（9600 次生成 @batch 40；实测率全部来自 batch 12：试点 2.07 s/生成、cap2048 臂 2.01、保真臂 2.23；batch 40 宽 3.3× 且试点未打满 80GB 卡 → 取 1.1–2.2 s/生成。**未实测**：batch 40 在 20 轮上下文下的吞吐、以及 `--gpus a100:1` 给 40GB 还是 80GB），`--time 08:00:00` = 1.35× 悲观值；生成先写 `trajectories.jsonl` 再算汇总，超时丢尾不丢run |
| 09-10 09:5x | 15772957 | `run_sequor_s1_score.sbatch`（两份读出，57,600 次判分；`--dependency=afterok`） | 仪器 | ⏳ PENDING（`afterok:15772956`） | **S1 的所有数字从哪来**：独立判（14B，cap 512）是唯一可报告来源，自判（4B，cap 1024）是 S3 的 in-loop 选择信号、永不报告。**拆成第二个作业**的理由：判分再跑一次便宜、重新生成不便宜（15763577 救回 15761810 的先例）。估计 **1.6–2.4 h**（28,800 次/judge × 实测 0.11 s ≈ 53 min + 各 4 min 加载），`--time 04:00:00`。**预期而非失败**：自判在 cap 1024 上丢 ~4–5% 的行（试点 121/2880），那是 S3 控制器会瞎掉的比例，报告它、不在这里抬 cap |

**两个作业的提交前六条核查**（2026-09-10）：① **用户已裁决**（「按照你的建议提交」，估计量与人工核验集推迟同址签在筛查 §十 第 9 条）；② 运行时估计与假设见上表与两个 sbatch 头部，**未实测的量逐条列出**；③ `outputs/sequor_s1_arm/` 提交前不存在（干跑核过，runner 自身亦拒绝写入已存在目录），两份读出写在该目录内的新文件；④ 见上表两行；⑤ 配额见下；⑥ **两个作业都不做 pip install**（vLLM env 按文件路径加载共享模块），且 score 作业以 `afterok` 依赖排在 arm 之后，队列里无其它本项目作业 → 结构上不可能重演 2026-09-08 的并发 editable install 竞态。

**近 10 个作业配额检查**（2026-09-10，含上表两个作业）：提交前是 仪器 4 / 方法 5 / 基建 1；
加这两个后滑窗挤掉 15756689、15758992（都是方法）→ **仪器 5 / 方法 4 / 基建 1 → ✅ 通过，但零余量**
（阈值：仪器 > 5 即停）。**后果：下一个仪器作业会触发停止提交**，需要先向用户报告并要求裁决。

结果见 [constraint_results.md](experiments/constraint_results.md) §S1 辨识臂。

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
