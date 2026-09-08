# 执行计划：ERGO 线的协议保真度修复（R0–R5）

**日期**：2026-09-08 · **起草**：Hongyu Cao（与 Claude Opus 5）· **适用**：Opus 5（裁决）/ Sonnet 5（执行）
**状态**：⏳ **待 R0 签字**。R0 之前不得提交任何 GPU 作业。
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

## 七、R5 ·（可选，需 Q9 先重开 Q4）熵触发臂

上游 ERGO 相对 RECAP 的 +14.1 / +9.3 分来自**熵触发的自适应时机**。这正是本项目 P1/P2 要证的东西。
工程尾巴：要 `chat_model.generate` 在线吐 per-token 熵，`analyze_ergo_entropy_readout.py` 已经
离线做过一次熵读出，可复用其定义。触发规则照上游：ΔH̄(t) = H̄(t) − H̄(t−1) > τ，τ 按模型标定
（上游范围 0.03–0.3）。

**Q4 当初裁决"不做"的理由是「压在关键路径上、工程尾巴无界、按计划自己的表属可选」。那个裁决是在
这条线看着快死的时候做的；现在上游告诉我们它就是产生 +14 分的那个部件。** 重开与否见 Q9。

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
| R5 | 未定 | 见 Q9 | Q9 |

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

## 十、判断题（待 R0 裁决）

| # | 问题 | 起草建议 |
|---|---|---|
| **Q9** | 是否重开 Q4（熵触发臂 R5）？上游说它是 +14.1/+9.3 的来源，也是唯一可能到 S1 的部件；代价是工程尾巴 + 关键路径 | **先看 G-R1-2**：权威恢复了再议；不恢复则 R5 无意义 |
| **Q10** | R2 改口径后，既有的 overwrite Phase B/C 结果是否按新口径全部重算？ | 重算（全 CPU，已有数据），但**双口径并列**，旧数字不删 |
| **Q11** | 主指标是否从 `final_turn_success`（全轮打分）改为上游口径的 attempt-only 分数？这会让所有历史数字不可直接比较 | **不改主指标**，upstream 口径作为并列的第二指标；论文里两个都报 |
| **Q12** | 9/13 合流点是否后移？R1+R2 能在 9/10 前完成，R3 压到 9/12，R5 放不下 | 合流点保持 9/13，**R5 若做则明确写成"摘要之后、全文之前"的追加实验** |

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
