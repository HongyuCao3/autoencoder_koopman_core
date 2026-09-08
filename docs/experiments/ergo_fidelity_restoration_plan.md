# 执行计划：ERGO 线的协议保真度修复（R0–R5）

**日期**：2026-09-08 · **起草**：Hongyu Cao（与 Claude Opus 5）· **适用**：Opus 5（裁决）/ Sonnet 5（执行）
**状态**：✅ **R0 已签字**（2026-09-08，用户，见第十三节）。R1 的三个臂已提交
（job 15690582 / 15690584 / 15690586）。**R5 推迟**，不在本轮范围内。
**取代**：[`backup/two_task_success_plan.md`](backup/two_task_success_plan.md) 第二节（ERGO 线 E0–E6）。
那份计划的 ERGO 半边已于 2026-09-08 结案（S3），**其 E1–E6 规格作废，不要执行**；防御线半边（D0–D3）
的去向见 [`defense_line_redesign_plan.md`](defense_line_redesign_plan.md) 第十三节。
**依据**：截至 commit `17da368` 的全部结果——`ergo_multiturn_reliability_pilot.md` 的 G4 与
「E2–E3」两节、`signal_resolution_plan.md`、`measurement_validity_plan.md`、以及 2026-09-08
对上游两篇论文与上游代码的核查（第零节 0.2）。
**时间假设**：ICLR 2027 摘要约 9/18、全文约 9/24；原合流点 9/13（**本计划可能要求后移，见 Q12**）。

> **开工前必读**（继承既有纪律，一条不减）
>
> 1. 本文档是完整规格。凭记忆复述、"优化"命令行、合并步骤，都算偏离。
> 2. 闸门不过就停下来报告；**不调参、不换臂、不扩样本重跑**去找想要的答案。
> 3. 文档没写到的判断题，停下来问用户或 Opus；Sonnet 不替他们判。
> 4. **不改** `control.py` / `controller_cli.py` / `modeling/evaluate.py` / `modeling/dataset.py`；
>    ERGO 侧一切改动放在 `ergo_*` 文件与 `run_ergo_math_screening.py` 本地分支里。**不动 `paper/`**。
> 5. **不覆盖任何已有产物**，所有输出目录都是新的；`prompt_profile="legacy"` 的既有行为逐字节不变
>    （这是本计划的 B 类补丁，见 3.1）。
> 6. 环境：`export PATH=/scratch/hcao2/envs/persona_drift_pilot/bin:$PATH`，
>    工作目录 `/home/hcao2/autoencoder_koopman_core/persona_drift_control`。
> 7. 每个任务完成后按第八节格式汇报，数字照抄带 CI；**Opus 独立重算后才算数**。
> 8. **禁止仓库级破坏性 git 操作**：`git stash`、`git clean`、`git checkout -- .`、`git reset --hard`
>    一律不许用。多个会话共享同一个工作树。只允许 `git add <明确列出的文件>`、`git diff`、
>    `git status`、`git show`。
> 9. **提交前后各跑一次 `git status`**，并核对 `git show --stat HEAD` 里确实有你要提交的文件。
>    2026-09-08 发生过一次：两个会话相隔 13 秒提交，先提交的那个把后者暂存的文件卷走，
>    后者的 commit 只剩一个文件而 message 描述了五个（`08fc875` / `17da368`）。记为失败模式 20。

---

## 零、这份计划为什么存在：E2/E3 的失效被归因到我们自己的 prompt 结构

### 0.1 决定性证据：我们测到的"reset 权威"其实是"模型有没有写出推导"

E2/E3 的结论是 append 下 reset 失去权威（G-E2-1、G-E3-1、G-E3-2 三道不过）。2026-09-08 的
事后诊断把这件事解释掉了：

| 臂 | 末轮回复中位字符 | 末轮 ≤80 字符占比 | `final_turn_success` |
|---|---:|---:|---:|
| `zero_control`（从不 reset） | 19 | 0.66 | 0.328 |
| `always_reset_append` | 18 | 0.91 | 0.233 |
| `fixed_t2_append` | 19 | 0.69 | 0.319 |
| `fixed_last_append` | 298 | 0.45 | 0.578 |
| `fixed_last`（overwrite） | 646 | 0.09 | 0.776 |
| `always_reset`（overwrite） | 646 | 0.09 | 0.776 |

跨 6 个臂 **Pearson(末轮回复中位长度, 成功率) = +0.984**。臂内同样成立：长回复（>80 字符）
成功率 0.67–1.00，短回复 0.15–0.36。**reset 的全部权威 = 它让模型重新写了一遍推导**，
而不是题面被重述这件事本身。

让"不写推导"成为可选项的是我们自己加的一行（`ergo_math_trajectory.py:42-46`）：

```python
_ANSWER_FORMAT_INSTRUCTION = (
    'Give your current best-guess final numeric answer to the math problem, even if you are '
    'not fully confident yet or do not have all the details, on its own line in this exact '
    'format: "Current answer: X" (a single number).'
)
```

它被拼在**每一轮** user message 末尾（`:90` 与 `:97`），而这个 harness **完全没有 system prompt**
（`agent_history` 从空列表开始）。overwrite 下无害——历史被清空，模型只看见一次指令、必须从头解；
append 下模型看见 k 轮「指令 + 自己回的 `Current answer: 12`」，这是一个 **k-shot 的简洁作答示范**。

### 0.2 与上游的三条偏离（ground truth，非论文措辞）

| | 上游 | 我们 |
|---|---|---|
| 格式指令位置 | **system prompt，只给一次**（`prompts/math/math_system_prompt.txt`，`generate_system_prompt()`）：*"The answer should be a single number..."* | **每轮 user message 末尾**，重复 4–12 次 |
| system prompt | 有，minimal，turn 1 之前给一次 | **完全没有** |
| 打分对象 | **strategy classifier** 把回复分成 7 类（澄清/拒答/对冲/追问/讨论/缺失/**answer attempt**），**只给 answer attempt 打分**，再由 answer extractor 抽 span | 每轮正则抽数字打分，**无 attempt 分类** |
| ERGO 的 reset 内容 | **让模型把累积输入重写成一个优化的单轮 prompt** | 机械拼接的 bullet list |
| ERGO 的 reset 语义 | **stateless**："passed into a new instance of the model, simulating a stateless chat environment with no memory of prior turns" | 我们的 `overwrite` **就是这个** |
| 模型 | Phi-4 / Llama-3.1-8B-Instruct / GPT-4o / GPT-4o-mini / GPT-4.1 | Qwen3-4B（**上游模型集里没有 Qwen**） |

第三行最要命：**上游会把 `Current answer: 12` 这类回复判为「非作答」而不打分，我们给它打 0 分
并计入指标。** 所以 append 下的 `y_task_success` 有相当一部分在测「模型愿不愿意写推导」，
而上游明确把这一维过滤掉了。

来源：
- Laban et al. 2025, *LLMs Get Lost In Multi-Turn Conversation*, arXiv:2505.06120（[abs](https://arxiv.org/abs/2505.06120) / [html](https://arxiv.org/html/2505.06120v1)）
- *ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models*, arXiv:2510.14077（[abs](https://arxiv.org/abs/2510.14077) / [html](https://arxiv.org/html/2510.14077v1) / [ACL Anthology](https://aclanthology.org/2025.uncertainlp-main.23/)）
- 上游代码与 prompt：[microsoft/lost_in_conversation](https://github.com/microsoft/lost_in_conversation)（本仓库 `resources/ergo_gsm8k_sharded.jsonl` 的来源）

### 0.3 我们的结果与上游同向，只是幅度更大

| | SHARDED（无 reset） | SNOWBALL（每轮 recap） | RECAP（末轮一次 recap） |
|---|---:|---:|---:|
| Laban GPT-4o-mini | 50.4 | 61.8 | **66.5** |
| Laban GPT-4o | 59.1 | 65.3 | **76.6** |
| ERGO 论文复现 4o-mini | 44.3 | 54.0 | **57.7** |
| **我们（append，Qwen3-4B）** | 0.328 | 0.233 | **0.578** |

三次独立测量都给出 **RECAP > SNOWBALL**。我们的排序一致，唯一差异是我们的 SNOWBALL 掉到
SHARDED 之下。Laban 的 root cause #4 是 *"overly rely on previous (incorrect) answer attempts
leading to lengthier 'bloated' answers"*——上游那里表现为**回复臃肿**，我们这里表现为**回复简短**，
因为我们把输出格式钉死了。**同一现象的两个表面形态。**

### 0.4 对 Koopman 的意义：非单调性就是控制问题，而它只在 append 下存在

overwrite 下 `always_reset ≡ fixed_last` 是 116/116 的恒等式——reset 次数完全不影响结果，
没有任何可权衡的东西（G4 的结论）。append 下「多 reset 反而更差」，上游两个模型 + 我们的数据
三次确认。**一个自适应控制器需要的正是这种非单调性。**

而上游还有一个数字直接就是本项目的 S1：

| | GPT-4o-mini | GPT-4o |
|---|---:|---:|
| RECAP（最优固定规则） | 57.7 | 66.3 |
| **ERGO（熵触发的自适应时机）** | **71.8** | **75.6** |

**+14.1 / +9.3 分——"自适应时机打赢最好的固定规则"。** 上游声称它存在。

同时上游留了一个**它自己没做的消融**：ERGO 的增益同时来自 (a) 熵触发的自适应时机与
(b) 模型重写的 consolidation prompt，论文没有报「固定日程 + 同样的重写 prompt」这一臂，
**两者贡献是混淆的**。拆开它是本计划 R3 的副产品，也是一个能独立成立的贡献。

### 0.5 三个不能忘的诚实前提

1. **R1/R2 可能恢复权威，但最优解仍停在「末轮 recap 一次」这个固定规则上。** 上游的 RECAP
   就是这个固定规则，Laban 从未声称自适应能赢它。真正能翻盘的是熵触发臂（R5），而它正是
   `backup/two_task_success_plan.md` 第九节 Q4 裁决掉的那一项。**Q4 值得重开，但那是用户的决定（Q9）。**
2. **`defense_line_redesign_plan.md` §12.3 曾预测「第 2 层需重测但大概率保留」。它被证伪了。**
   本计划的预测同样可能被证伪；G-R1-1 就是为了让证伪来得便宜且明确。
3. **改打分口径会让所有历史数字不可直接比较**（Q11）。R2 因此强制双口径报告。

---

## 一、成功分级（继承 `backup/two_task_success_plan.md` 第零节，按新证据修订）

| 等级 | ERGO 线的判据 |
|---|---|
| **S1 正面闭环** | 同代价下 Koopman-MPC 的最终轮成功率对**随机分配**配对差 CI>0，且对**同代价最优固定日程**不显著更差 |
| **S2 不劣于先知** | 对随机分配 CI>0，对最优固定日程打平 |
| **S3 方法论** | 前提检验程序 + 失效定位。**已在手且已成稿**：`ergo_multiturn_reliability_pilot.md` 的「E2–E3」节（append 修好第 0 层、塌掉第 2 层）+ G4 |

**当前状态**：S3 已经落袋，**且不会因为本计划的任何结果而失效**。本计划是在 S3 之上争取 S1/S2，
不是去救 S3。这一点决定了失败模式 21：**不许为了让 R 相位出正面结果而回头改写 E2/E3 的结论。**

---

## 二、R0 · 签字（零成本，用户 + Opus）

第十节的判断题 Q9–Q12 全部裁决，并确认第七节的日程与 9/13 合流点的处置。**R0 未签字前不得提交
任何 GPU 作业**；R1 的代码与单测（零 GPU）可以先做。

---

## 三、R1 · 把格式指令搬进 system prompt（代码 + 3 GPU 臂，约 1 GPU-小时）

**这是全计划信息量最高、最便宜的一步，它单独就能证实或证伪第零节的全部归因。**

### 3.1 改动范围（只这三处）

1. **`src/persona_drift/ergo_math_trajectory.py`**
   - `ErgoMathTrajectoryConfig` 加字段 `prompt_profile: str = "legacy"`，取值
     `"legacy" | "upstream"`，其他值在 `__post_init__` 抛 `ValueError`（与 `reset_mode` 同一形状）。
   - `"legacy"`：**逐字节不变**，格式指令仍拼在每轮 user message 末尾，`agent_history` 仍从空列表开始。
   - `"upstream"`：
     - `agent_history` 以一条 `{"role": "system", "content": _UPSTREAM_SYSTEM_PROMPT}` 开始，
       只出现一次；
     - `_incremental_stimulus(shard)` 返回 **shard 原文**，不附格式指令；
     - `_consolidated_stimulus(revealed_shards)` 返回 bullet list 引导句 + 列表，**不附格式指令**；
     - `_UPSTREAM_SYSTEM_PROMPT` 照上游 `prompts/math/math_system_prompt.txt` 的形状写：
       任务框架一句 + `The answer should be a single number (it could be decimal, or negative,
       or a fraction, etc.).` + 我们自己的可解析要求（末行 `Current answer: X`）。
       **可解析要求必须保留**——正则判官是我们唯一的打分手段，去掉它 R1 就没有指标了；
       它现在只出现在 system prompt 里一次，这正是与 legacy 的唯一实质差别。
   - `row` 加 `"prompt_profile": config.prompt_profile`。
2. **`scripts/run_ergo_math_screening.py`**：加 `--prompt-profile {legacy,upstream}`（默认 `legacy`），
   透传进 config；`excitation_design` 名字后缀 `_up`（例如 `constant_remind_append_up`），
   否则不同 profile 的产物会同名（与 `_append` 后缀同一处置）。
3. **`tests/test_ergo_math_trajectory.py`** 加 ≥ 5 条：`legacy` 下三处文本逐字不变（对既有常量做
   字符串断言）；`upstream` 下 `agent_history[0]["role"] == "system"` 且全轨迹只有一条 system 消息；
   `upstream` 的非 reset 轮 user message **不含** `"Current answer"`；`upstream` 的 consolidated
   stimulus 不含 `"Current answer"`；非法 `prompt_profile` 抛错。

**不改判官、不改 `closeness`、不改分析器。** R1 只动 prompt 结构这一个变量（一次只动一个执行器
变量，与 E0 的 Q1 裁决同一原则）。

### 3.2 GPU 臂（58 held-out item × seeds 0 1，`--item-ids` 与 Phase C 逐字相同，全部 `--reset-mode append --prompt-profile upstream`）

| 臂 | 控制器 | 输出目录 | 对应上游条件 |
|---|---|---|---|
| `zero_control_up` | `zero_control` | `outputs/ergo_upC_zero_control/` | SHARDED |
| `always_reset_append_up` | `constant_remind` | `outputs/ergo_upC_always_reset/` | SNOWBALL |
| `fixed_last_append_up` | `fixed_last` | `outputs/ergo_upC_fixed_last/` | RECAP |

**`zero_control` 必须重跑，不许复用**：它在 legacy profile 下跑的，跨 profile 复用就是拿两种
prompt 结构做比较（失败模式 22）。

sbatch 照抄 `environment/run_ergo_appendC_*.sbatch`，只改 job-name / 日志 / `--prompt-profile upstream`
/ `--output-dir`。**`--time` 按实测重估，不许照抄**（失败模式 14）：upstream profile 预期回复更长，
比 append legacy 慢；给 `06:00:00`。

### 3.3 预注册闸门（按 item 配对，58 对，bootstrap 10000，`default_rng(0)`，用 `analyze_ergo_append_comparison.py`）

| 闸门 | 判据 | 过 → | 不过 → |
|---|---|---|---|
| **G-R1-0 实现锁** | `legacy` profile 下重跑 1 个 item 的轨迹，与既有 `outputs/ergo_appendC_always_reset` 的对应行**逐字节相同**；`upstream` 下 system 消息恰好 1 条、非 reset 轮 user message 不含 `"Current answer"` | 继续 | 修实现，不跑臂 |
| **G-R1-1 机制闸门（主）** | `always_reset_append_up` 的**末轮回复中位字符数 ≥ 200** 且 **≤80 字符占比 ≤ 0.30**（legacy 为 18 / 0.91） | 归因成立，看 G-R1-2 | **归因证伪**：格式指令不是原因。**跳过 R2/R3，直接进 R4**，并把"简洁塌缩是 Qwen3-4B 的性质"作为待检验假设 |
| **G-R1-2 权威闸门** | `always_reset_append_up − zero_control_up` 的 95% CI 下界 **> 0** | 权威恢复 | 长度恢复但权威没恢复 → 还有第二个混淆，**进 R2 再判，不改 R1** |
| **G-R1-3 上游排序复现** | 三路排序 `fixed_last_append_up > always_reset_append_up > zero_control_up`，**两两配对差的 CI 下界都 > 0** | harness 判为忠实 | 记录并报告；排序错位是"我们仍与上游不同"的直接证据，交 Opus 判是进 R2 还是 R4 |
| **G-R1-4 记录项** | 三臂末轮回复长度分布；`fixed_last_append_up − always_reset_append_up` 的差与 CI（上游 4o-mini +4.7、4o +11.3）；prompt token 最大值（C2 护栏，用 `analyze_ergo_append_context_length.py`） | 只记录 | — |

**预注册预期**：G-R1-1 大概率过（这是本计划的核心归因）；G-R1-2 是真正的未知；G-R1-3 若过，
则我们第一次拥有一个与上游同构的 harness。

---

## 四、R2 · answer-attempt 打分口径（CPU 为主 + 少量 GPU）

上游只给 answer attempt 打分。我们目前给每一轮的正则抽取打分，于是在 append 下大量把
「非作答」记成 0 分并计入指标——这既污染主指标，也污染 `closeness` 读出（RC-3 就是在这些行上算的）。

### 4.1 实现

新增 `src/persona_drift/ergo_answer_attempt.py`：`classify_attempt(agent_message) -> bool`。
**两级**：
1. **规则层**（零成本，先做）：回复去掉 `Current answer: X` 那一行后，剩余非空白字符数 < 阈值
   → 判为非作答。阈值**预注册为 80 字符**（与第零节诊断同一刻度，不许在看到结果后调）。
2. **LLM 层**（仅当 G-R2-1 不过时才启用）：复用管线里已有的判官模型，prompt 照上游 7 类
   strategy classifier 的形状，只取 `answer attempt` 与否的二值输出。

新增 `scripts/analyze_ergo_dual_metric.py`：所有指标**同时**报两个口径——
`legacy`（全轮打分，与全部历史数字可比）与 `upstream`（只给 attempt 打分）。
**任何单口径的汇报都算偏离**（失败模式 23）。

**规格补丁 D1（Opus，2026-09-08）· "只给 attempt 打分"在变长轨迹上是歧义的，此处写死。**
上游按对话记分、并跟踪最后一次 answer attempt；我们的主指标是**每条轨迹自己的末轮**。
若末轮不是 attempt，有两种读法，**必须选一种并写死**：

| 读法 | 做法 | 裁决 |
|---|---|---|
| **(a) 保分母** | 回退到**该轨迹最后一个被判为 attempt 的轮次**的分数；全程无 attempt → 记 **0** | ✅ **采用** |
| (b) 缩分母 | 把末轮非 attempt 的轨迹整条丢掉 | ❌ **不采用** |

**理由**：(b) 会让每个臂的 n 不同，而"产生 attempt 的比例"本身就是臂间差异最大的量
（`always_reset_append` 末轮 91% 是 `Current answer: 12`）——按它筛样本，等于用一个与结果强相关的
量做选择，配对也当场断掉。(a) 保住 58 个配对 item、保住 bootstrap 的配对结构，且与上游"跟踪最后
一次 answer attempt"的做法同构。

因此 `analyze_ergo_dual_metric.py` 必须输出三列而不是两列：
`final_turn_success`（legacy 主指标，不变）、`final_attempt_success`（upstream 口径，按 (a)）、
以及 `attempt_rate`（该臂末轮是 attempt 的轨迹占比）。**第三列是解释前两列差异的必需品**，
不是可选诊断：两个口径的差全部来自它。

### 4.2 闸门

| 闸门 | 判据 | 不过 → |
|---|---|---|
| **G-R2-1 分类器有效性** | 在 60 行分层抽样上做**盲标**（P 协议，`general-purpose` subagent，每批全新实例），规则层与人判的 agreement **≥ 0.85**，且两类各自 ≥ 20 行 | 启用 LLM 层，重测；仍不过则**不换口径**，R2 终止并报告 |
| **G-R2-2 回归锁** | 在既有 legacy overwrite Phase C 数据上按新口径重算，`always_reset ≡ fixed_last` 的 **116/116 恒等式必须保持** | 分类器引入了臂间不对称，是 bug，修 |
| **G-R2-3 读出重算** | 在 `outputs/ergo_appendB_random_excite` 上按 upstream 口径重算 RC-0..3 与交互回归的 b0/b2 | 只记录；**这是诊断，不是把 G-E3-1/G-E3-2 的判定翻过来**（失败模式 21） |

**用一个没验过的分类器换掉主指标，就是用一个未验证的测量替换另一个**——G-R2-1 存在的理由就是这个。

---

## 五、R3 · 忠实的 ERGO reset：模型重写的 consolidation prompt

ERGO 的 reset 不是机械拼接，是**让模型把累积输入重写成一个优化的单轮 prompt**，它 56.6% 的增益
是在重写版上测的。同时这里能拆开上游自己没拆的混淆。

### 5.1 实现与臂

- `_consolidated_stimulus` 增加 `consolidation: str = "bullet" | "rewrite"`。`"rewrite"` 用同一个
  agent 模型（**不是判官**，避免把判官拉进被控回路）把已揭示 shard 重写成单轮 prompt。
- **消融对（上游没做的那个）**：
  - `fixed_last_append_up_bullet`（= R1 的 `fixed_last_append_up`，复用）
  - `fixed_last_append_up_rewrite`
  二者只差 consolidation 形式，**同一时机、同一代价** → 差值就是「重写 prompt」的净贡献。

### 5.2 闸门

| 闸门 | 判据 | 不过 → |
|---|---|---|
| **G-R3-1 信息保全** | 重写产物中，gold answer 推导所需的**全部数值**出现率 ≥ 0.95（对每个 item 用 shard 原文里的数字集合做覆盖检查） | 重写在删信息，它的效应与 reset 效应混淆，**修 prompt 再跑**；不许带着信息损失往下走（失败模式 24） |
| **G-R3-2 消融** | `rewrite − bullet` 的配对差与 CI | 只记录，不判定 |

**R1 未过不做 R3**：R3 工程量最大，且它的结论建立在 R1 的前提上（失败模式 25）。

---

## 六、R4 · 模型对照（约 1 GPU-小时）

上游模型集里**没有 Qwen**。本地 `/scratch/hcao2/hf_cache/hub/` 有 Qwen3-1.7B / Qwen3-4B /
Qwen3-4B-Instruct-2507（D1 刚用过后者）。

- 在 **R1 的三个臂**上换 `Qwen/Qwen3-4B-Instruct-2507` 重跑一遍。
- **G-R4-1（记录项，不判定）**：三路排序是否与 R1 一致；末轮回复长度分布是否一致。
- 目的只有一个：分离「4B 会照抄上下文模式」与「harness 教会了它」。**不进主结果**，只进归因段落。
- G-R1-1 不过时，R4 从"对照"升级为"主线"——那时问题就变成了模型选择。

---

## 七、R5 ·（**已推迟**，Q9 裁决）熵触发臂

上游 ERGO 相对 RECAP 的 +14.1 / +9.3 分来自**熵触发的自适应时机**。这正是本项目 P1/P2 要证的东西。
工程尾巴：要 `chat_model.generate` 在线吐 per-token 熵，`analyze_ergo_entropy_readout.py` 已经
离线做过一次熵读出，可复用其定义。触发规则照上游：ΔH̄(t) = H̄(t) − H̄(t−1) > τ，τ 按模型标定
（上游范围 0.03–0.3）。

**Q4 当初裁决"不做"的理由是「压在关键路径上、工程尾巴无界、按计划自己的表属可选」。那个裁决是在
这条线看着快死的时候做的；现在上游告诉我们它就是产生 +14 分的那个部件。**

**R0 裁决（Q9）：推迟，不否决——等 ERGO 的 Koopman 工作跑完之后再试。** 本节保留为待办的完整规格，
但 **R5 不在本轮范围内**，R1–R4 的任何闸门结果都不自动触发它。要开工需要一次新的签字。

---

## 八、成本、日程与汇报格式

| 任务 | 类型 | 预计 | 依赖 |
|---|---|---|---|
| R0 签字 | 用户 + Opus | 9/8 | — |
| R1 代码 + 单测 | CPU | 9/8–9/9 | R0（代码可先行） |
| R1 三臂 | **~1 GPU-小时** | 9/9 | G-R1-0 |
| R2 | CPU + 盲标 | 9/9–9/10 | G-R1-1 |
| R4 | ~1 GPU-小时 | 9/10（队列空档） | R1 臂表 |
| R3 | 代码 + 2 GPU 臂 | 9/10–9/12 | G-R1-1/2 |
| ~~R5~~ | **已推迟**（Q9） | ERGO Koopman 工作之后，需新签字 | Q9 |

GPU 总量约 2–4 小时（原 ERGO 的 8 小时预算与 E5 的 4 小时都没花掉）。

**汇报格式**（每个任务结束时）：任务编号；每道闸门的判据 / 实测值（带 CI）/ 过或不过；作业 id、
`COMPLETED` 状态、行数、`item_id` 集合断言结果；偏离逐条列出；产物路径（全部为新目录）；
留给 Opus 的判断题单独一节；**不写结论句**。

---

## 九、失败模式清单（前 19 条见 `backup/two_task_success_plan.md` 第六节，仍然有效）

20. **提交后不核对 `git show --stat HEAD`。** 2026-09-08 两个会话相隔 13 秒提交，先提交者卷走了
    后者暂存的文件，后者的 commit 只剩一个文件而 message 描述了五个。**提交前后各跑一次
    `git status`，并核对 `git show --stat HEAD`。**
21. **为了让 R 相位出正面结果而回头改写 E2/E3 的结论。** S3 已经落袋且不依赖 R 相位；
    R2 的读出重算是诊断，不是把 G-E3-1/G-E3-2 翻过来。
22. **跨 profile 复用臂。** `zero_control` 在 legacy 下跑过，upstream profile 必须重跑；
    复用就是拿两种 prompt 结构做比较。
23. **单口径汇报。** R2 之后所有指标必须 legacy / upstream 双报。
24. **带着信息损失的重写往下走。** G-R3-1 存在的理由。
25. **R1 未过就做 R3。** R3 工程量最大且依赖 R1 的前提。
26. **把 G-R1-3 排序不过读成"上游错了"。** 排序不过是"我们仍与上游不同"的证据，指向 R2/R4，
    不指向"上游的数字有问题"。

---

## 十、判断题与 R0 裁决

**全部已裁决**（用户，2026-09-08）。"裁决"列是最终规格；Sonnet 不再就这 4 项提问。

| # | 问题 | 起草建议 | **R0 裁决** |
|---|---|---|---|
| **Q9** | 是否重开 Q4（熵触发臂 R5）？ | 先看 G-R1-2 | **推迟，不否决。** 熵触发暂时不重做，**等 ERGO 的 Koopman 工作跑完之后再试**。R5 因此移出本轮范围（见 13.2） |
| **Q10** | 既有的 overwrite Phase B/C 结果是否按新口径重算？ | 重算，双口径并列 | **重算**，同意 |
| **Q11** | 主指标是否改为 attempt-only 分数？ | 不改，并列第二指标 | **不换主指标，新增指标并列**，同意 |
| **Q12** | 9/13 合流点是否后移？ | 保持 9/13 | **合流进度不变**，保持 9/13 |

---

## 十一、新会话启动语（直接粘贴）

> **R1**：读 `docs/experiments/ergo_fidelity_restoration_plan.md` 第零节与第三节，以及
> `src/persona_drift/ergo_math_trajectory.py`、`scripts/run_ergo_math_screening.py`。实现
> `prompt_profile`（≥5 条单测），报 G-R1-0；过后建 3 个 sbatch（照抄 `run_ergo_appendC_*.sbatch`，
> 只改四处，`--time 06:00:00`），提交，跑 `analyze_ergo_append_comparison.py` 与
> `analyze_ergo_append_context_length.py`，按第八节格式报 G-R1-1/2/3/4。**`zero_control` 必须
> 重跑，不许复用 legacy 的。不写结论句。**

> **R2**：读第零节与第四节。实现 `ergo_answer_attempt.py` 规则层与 `analyze_ergo_dual_metric.py`，
> 组织 60 行盲标（P 协议，每批全新 subagent 实例），报 G-R2-1/2/3。**所有指标双口径。**

> **R3**：读第零节与第五节，且 G-R1-1/2 已过。实现 `consolidation="rewrite"`，先报 G-R3-1，
> 过后再跑消融臂，报 G-R3-2。

> **R4**：读第六节。在 R1 的三个臂上换 `Qwen/Qwen3-4B-Instruct-2507` 重跑，报 G-R4-1（只记录）。

> **Opus 复核（新会话）**：只读 `<任务> handoff` 与产物 JSON，独立重算每个闸门数字，写"复核"段，
> 出裁决与下一步是否开工。不读 Sonnet 的过程叙述。

---

## 十二、与旧计划的关系

- [`backup/two_task_success_plan.md`](backup/two_task_success_plan.md) 第二节（E0–E6）**已作废**，
  其结果落在 `ergo_multiturn_reliability_pilot.md` 的「E2–E3」节。该文档第六节的失败模式 1–19、
  第一节的会话协议、第八节的汇报格式**继续有效**，本计划继承。
- 该文档第四节（防御线 D0–D3）的去向见
  [`defense_line_redesign_plan.md`](defense_line_redesign_plan.md) 第十三节。
- [`ergo_multiturn_reliability_pilot.md`](ergo_multiturn_reliability_pilot.md) 仍是 ERGO 线的
  **结果档案**；本计划的结果也回写到那里（新一节「R1–R4」）。


---

## 十三、R0 签字与 R1 提交记录（2026-09-08）

### 13.1 签字

用户于 2026-09-08 裁决 Q9–Q12（见第十节），**R0 签字完成**。R1 的三个 GPU 臂随即提交：

| 臂 | job | 输出目录 | 对应上游条件 |
|---|---|---|---|
| `zero_control_up` | **15690582** | `outputs/ergo_upC_zero_control/` | SHARDED |
| `always_reset_append_up` | **15690584** | `outputs/ergo_upC_always_reset/` | SNOWBALL |
| `fixed_last_append_up` | **15690586** | `outputs/ergo_upC_fixed_last/` | RECAP |

三臂全部 `--reset-mode append --prompt-profile upstream`，58 held-out item × seeds 0 1，
`--time 06:00:00`，`Qwen/Qwen3-4B`。

### 13.2 Q9 推迟 R5 的两个下游后果（记录，避免以后被读成遗漏）

1. **R1–R4 的任何闸门结果都不自动触发 R5。** 即使 G-R1-2 与 G-R1-3 全过、harness 判为忠实，
   熵触发臂也要等 ERGO 的 Koopman 工作跑完之后重新签字才开工。第七节保留完整规格，状态是待办。
2. **本轮能达到的上限因此是 S2，不是 S1。** 上游 ERGO 相对 RECAP 的 +14.1/+9.3 来自熵触发的
   自适应时机；不做 R5，本轮的被测臂就只有 Koopman-MPC 自己，S1 要靠它打赢同代价最优固定日程。
   这不是坏消息——**S1 本来就该由本项目自己的控制器去拿，用上游的熵触发去拿反而说明不了
   Koopman 的价值**——但要写清楚，免得以后把"没做 R5"记成"忘了做"。

### 13.3 Q10 与 Q11 合起来定义了 R2 的产物形状

- **主指标不换**（Q11）：`final_turn_success` 仍是主指标，attempt-only 分数**并列新增**，
  不是替换。论文里两个都报。
- **旧数据重算**（Q10）：既有的 overwrite Phase B/C 与 append Phase B′/C′ 全部按新口径重算，
  **旧数字不删**。全 CPU，用已有数据。
- 两条合起来 → 第四节 4.1 的 `analyze_ergo_dual_metric.py` 是 R2 的**唯一**汇报入口，
  任何单口径汇报算偏离（失败模式 23）。

### 13.4 Q12：合流点不变

9/13 合流点保持。R1（9/8 已提交）→ R2（9/9–9/10）→ R3（9/10–9/12）仍在窗口内；
R5 推迟本来就把日程压力去掉了。

---

## 十四、G-R2-1 不通过：规则层单向过判（2026-09-08）

**判定：不通过。** 按 4.2 节，下一步是**启用 LLM 层并重测**；若 LLM 层仍不过，则**不换口径**，
R2 终止并报告。在 LLM 层通过之前，`final_attempt_success` 与 `attempt_rate` **不得写进任何
对外结论**。

### 14.1 数字

60 行盲标样本（按规则层判定分层各 30，臂名/轮次/judge 分全部剥离，两遍独立全新 subagent 实例，
P 协议）：

| 比较 | agreement |
|---|---|
| 标注者间 A vs B | **0.9833**（59/60） |
| 规则层 vs A | 0.8333（50/60） |
| 规则层 vs B | 0.8167（49/60） |
| 规则层 vs 两遍共识（59 行） | 0.8305（49/59） |

判据要求 ≥ 0.85，**两遍都不到**。标注者间 0.983 说明分歧是真的，不是标注噪声。

### 14.2 失效是单向的，这一点决定了怎么读现有数字

混淆矩阵（规则层 × 两遍共识，59 行）：

| | 共识 = attempt | 共识 = 非 attempt |
|---|---:|---:|
| **规则 = attempt** | 19 | **10 ← 全部误差在这一格** |
| **规则 = 非 attempt** | **0** | 30 |

**规则层从不漏判**：它说不是 attempt 的，人判 30/30 全部同意。误差 100% 是过判。

过判的 10 行是同一种东西——**长篇的"拒绝求解"**：模型写一整段解释为什么信息不足、算不出来，
然后给一个兜底数字。例如 `row_33`「The percentage ... varies by school, district, and region.
Without specific data ...」、`row_36`「However, the total charge ...」、`row_57`「it is not
possible to calcula...」。这正是 Laban 七分类里的 hedging / discussion 类，不是 answer attempt。
**80 字符的规则分不出"80 个字符的推导"和"80 个字符的解释为什么推不出来"。**

### 14.3 对既有 Q10 数字的两条限定（必须一起引用）

1. **`attempt_rate` 是上界，不是估计。** 非 attempt 一侧 30/30 可靠，所以真实 attempt 率
   **不高于**已报的数。
2. **修正后臂间差距会缩小，不会重排。** 过判需要**长回复**才可能发生，而长回复正是
   overwrite 与 `fixed_last_append` 多、`always_reset_append`（末轮中位 18 字符）几乎没有的东西。
   在规则-attempt 那一层上过判率是 10/30 = 33%；若各臂同率，`always_reset_append` 的 0.086
   几乎不动而 overwrite 的 0.905 会明显下来。**过判率是否跨臂均匀是未知的，且没有理由假设它均匀**
   ——这正是必须先过 G-R2-1 的原因。

**主结论不受影响**：Q11 裁决主指标不换，E2/E3 与 Q10 的定性结论（append 下每轮 reset 不比
从不 reset 好）在 legacy 口径上独立成立，不依赖这个分类器。

### 14.4 重测时必须一并修的抽样缺陷

本次样本按**规则层**分层各 30，于是共识 attempt 类只有 **19 行 < 判据要求的 20**。
即使 agreement 达标，这一条也不满足。重测时改为：先用当前分类器抽一个更大的候选池
（例如各 45），标完后按**共识**类别核对两类各 ≥ 20；或直接过采样规则-attempt 层。
**这不是放宽判据，是修一个抽样设计缺陷**——原判据"两类各 ≥ 20"指的就是共识类别。

---

## 十五、R1 裁决（Opus，2026-09-08）

结果与数字见 [`ergo_multiturn_reliability_pilot.md`](ergo_multiturn_reliability_pilot.md) 的
「R1」一节。本节只出裁决。

### 15.1 G-R1-0/1/2 通过；第零节的归因成立

`always_reset_append − zero_control` 从 `−0.0948`（CI 跨 0）到 **`+0.1810`（CI [+0.0862, +0.2845]）**，
唯一改动是那条答案格式指令的位置。**E2/E3 的 G-E2-1 失效被解释并逆转。**

同时 `agent_message` 末轮逐字相同只有 2/116，所以**状态仍在终点里**。E2/E3 那句"两个性质互斥"
是对 legacy prompt 结构的描述，不是对 `reset_mode` 的描述——已在 pilot 文档里更正。

### 15.2 G-R1-3 不通过：裁决为**进 R4**，且**不许把它论证成通过**

上游三次测量都是 RECAP > SNOWBALL；我们得到 SNOWBALL > RECAP（`fixed_last − always_reset`
= −0.0776，CI [−0.1638, +0.0000]）。**判为不通过，记录在案。**

我现在认为这条判据当初写得不好：`always_reset` 花 T 次 reset、`fixed_last` 花 1 次，**二者不同代价**，
而本计划自己在 E5 臂表里就写明 `always_reset`"不是同代价对手，只作权威参照"。把一个代价不对称的
比较当作保真度判据，是把两件事混在了一起。**但这个事后认识不能把"不通过"变成"通过"**
（失败模式 26 的同构：判据写完就该认，改判据要在跑之前）。

**裁决：进 R4**（模型对照，约 1 GPU-小时）。上游模型集是 Phi-4 / Llama-3.1-8B / GPT-4o 系，
**没有 Qwen**；一个 4B 模型比 GPT-4o 更依赖反复的上下文合并，是这个排序差异最省事的解释，且可测。
**不进 R2 的 LLM 层**（理由见 15.3）。

### 15.3 R2 的剩余工作降级

三个 upstream 臂的 `attempt_rate` **皆为 1.0000**，`final_attempt_success` 与
`final_turn_success` 逐位相同。**在修好的 prompt 结构下，attempt / 非 attempt 的区分整个消失了**
——R2 这个口径本来就是为了处理 legacy harness 产生的非作答回复而存在的。

因此：
- **R1 及其之后的任何结论都不依赖 R2 的分类器**，G-R2-1 不过不阻塞 R 相位；
- R2 的 LLM 层仍然欠着，但它现在只影响 **Q10 对 legacy 旧数据的重算**这一件事，优先级降到
  R3/R4 之后；
- 14.3 的两条限定继续有效，且**只适用于 legacy 数据的 upstream 口径列**。

### 15.4 对 S1 前景的诚实修正：0.4 节的论证没了一半

第 0.4 节把"RECAP > SNOWBALL 的非单调性"当作"控制问题真实存在"的证据。**在我们的 harness 里
这个非单调性不存在**（G-R1-3），所以那条论证作废。

剩下的、也是唯一还站着的问题是：**在固定预算 k 下，最优的 reset 时机是否随 item / 状态变化。**
那是 b₂ 测的东西，而 b₂ 上一次是在 legacy prompt 结构的 Phase B′ 上测的，
**那份数据现在已知被同一个混淆污染**。

**因此下一步是 R1.5**：在 upstream profile 下重跑随机激励的 Phase B′（1 个 GPU 臂，约 1 小时），
重算 G-E3-1（读出闸门 RC-0..3）与 G-E3-2（b₂ 与它的 overwrite 负对照）。**这不是把旧闸门翻案**
——旧判定在 legacy 数据上仍然成立；这是在一个已修正的 harness 上**重新提出同一个问题**，
判据逐字沿用、不改阈值。R1.5 的结果决定 ERGO 线还有没有 S1/S2 可谈。

---

## 十六、R1.5 · upstream profile 下的辨识数据与状态依赖（1 GPU 臂，约 1 小时）

**为什么**：15.4 节。b₂ 上一次是在 legacy prompt 结构的 Phase B′ 上测的，那份数据现在已知被
同一个混淆污染（`always_reset_append` 末轮 91% 是裸的 `Current answer: X`）。**R1.5 是在修正后的
harness 上重新问同一个问题，判据逐字沿用、不改阈值。** 旧判定在 legacy 数据上仍然成立，不翻案。

### 16.1 臂

照抄 `environment/run_ergo_appendB_random_excite.sbatch`，只改 job-name / 日志 /
`--prompt-profile upstream` / `--output-dir`：60 item（`--item-rng-seed 2`，与 Phase B/B′ 同一批）
× seeds 0 1，`--controller random_excite --random-excite-p 0.5 --reset-mode append`，
输出 `outputs/ergo_upB_random_excite/`。**`--time 06:00:00`**（upstream 回复更长，G-R1-1 实测
末轮中位 563 字符 vs legacy 的 18，生成显著更慢——R1 三臂实际用了 37–44 分钟，legacy append 是
9–23 分钟）。

### 16.2 闸门（全部逐字沿用 E3，不改阈值）

| 闸门 | 判据 | 不过 → |
|---|---|---|
| **G-R15-0 采集** | `COMPLETED 0:0`；666 行；`item_id` 集合与 Phase B 相同；每个 `shard_frac` 十等分箱内两种动作都出现 | 报告，不自行重采 |
| **G-R15-1 读出** | `closeness` 的 RC-0..3 四条全过（判据沿用 `signal_resolution_plan.md` 0.5 节） | 报告；RC-3 是 legacy 下失效的那一条，它是否恢复是本步的主要看点 |
| **G-R15-2 状态依赖** | 交互回归 `c_{t+1} = a·c_t + b0·u_{t+1} + b2·u_{t+1}·c_t + g·shard_frac + const` 的 **b₂ 的 95% CI 不含 0 且符号为负**，**且** overwrite 负对照上 b₂ 不显著 | **ERGO 线停在 S3**，写"reset 效应与状态无关"，不调参不换臂 |
| **G-R15-3 记录项** | a / b0 / g 与 legacy Phase B′、overwrite Phase B 的三方对照；`attempt_rate` | 只记录 |

**负对照沿用 §12.6 的规定**：同一脚本先在已有的 overwrite Phase B（`outputs/ergo_math_phaseB_random_excite`，
666 行，legacy profile）上跑一遍，b₂ 必须不显著；显著则回归在捡伪影，G-R15-2 不能作为状态依赖的证据。
该负对照 2026-09-08 已跑过两次（45-item 与全 60-item），两次都通过，见
`backup/two_task_success_plan.md` 13.5——**不重跑，直接引用**。

### 16.3 与 R4 的关系

R1.5 与 R4（第六节，模型对照）**无依赖，可并行提交**。R4 用 `Qwen/Qwen3-4B-Instruct-2507`
重跑 R1 的三个臂，输出 `outputs/ergo_upC_alt_*`，**只出记录项 G-R4-1，不进主结果**
（失败模式 6 的同构：不做跨模型配对比较，只看排序与回复长度分布是否一致）。
