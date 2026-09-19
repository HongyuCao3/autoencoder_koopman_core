# 论文故事框架 v1（2026-09-18）

> 定位：这是"大概的故事 + 已有事实能撑到哪"的草图，不是严谨稿。允许小幅矛盾，标在第七节里，之后再修。
> 决定（用户，2026-09-18）：放弃 `sections/` 的负结果论文；卖点一（预测）按线性特例并入同事的一般模型；卖点二（控制）走正面口径，同事数据全部保留并暂视为可靠（已录入 `paper/evidence/numbers_benchmark.yaml`，88 条，`n_bench_001–088`）；负结果进附录。

---

## 一、一句话故事

把一次"指令—回复"当作动力系统的一步，用 Koopman 算子在提示词层面预测 LLM 下一轮的行为属性，再用这个预测器做模型预测控制；预测器在 11 个多轮任务上优于无记忆、LSTM 和非线性变体，控制器在三个指令遵循基准上以零额外 LLM 调用、纯黑盒的方式把满足率提高约 15 个百分点，超过两种需要白盒访问的隐层控制方法。

候选标题（同事原题可用）：
- *Test-time Alignment for Large Language Models via Koopman Model*（现有）
- *Koopman Dynamics of Prompt–Response Interaction: Predicting and Controlling Multi-Turn LLM Behavior*
- *One Prompt, One Step: Koopman Predictive Control of LLM Behavior at the Interaction Level*

---

## 二、任务层动机：为什么在 prompt / interaction 层做预测和控制

### 2.1 现象：多轮交互里行为会走

- 事实：指令不稳定 / persona drift 已被多篇工作测到（Li et al. 2024 *Measuring and Controlling Instruction (In)Stability*，COLM；*Drift No More? Context Equilibria*；*Attractor States Emerge in Multi-Turn LLM Conversations*，均在 `main.bib`）。
- 我们自己的数据也支持"属性有惯性"：主表里"只看上一轮"（Last turn only）在 11 列中有 9 列好于"不看历史"（History-blind），说明属性不是每轮独立抽样的。

### 2.2 现有干预的三个 gap

**Gap 1：接口不对。** 现有控制方法几乎全在隐层上做——activation steering（Stolfo 2025）、PID steering（Nguyen 2026）、CBF-LLM（Miyaoka 2024/25）、BarrierSteer、RE-Control（Kong 2024）、TMPC（Wang 2026）。它们要求白盒访问、逐 token 干预。但部署里真正能动的杠杆是**提示词层**：每轮往对话里加一条指令，模型是黑盒 API。类比：用户不能改空调的压缩机电路，只能按遥控器。

**Gap 2：无状态、开环。** 一次性 system prompt、固定周期的提醒、逐轮分类器触发的模板，都不预测"这条指令发出去之后下一轮会变成什么"。后果是过冲或不足：同一个当前长度，可能处在"正在变长"或"正在变短"的趋势里，无记忆的规则给出同样的动作（同事 §3.2 的例子）。tsar_cefr 任务是个具体版本：把段落难度往 A2 推，推过头（overshoot）是有代价的。

**Gap 3：交互层没有可用的预测模型。** *What's the Magic Word*（Bhargava 2023）把 prompting 写成控制问题、证明了可达性，但没有给出"给一条指令、预测下一轮属性"的可操作模型；Li et al. 只做了测量和 split-softmax 这种解码器级干预，没有闭环。也就是说，控制论的语言已经有了，缺的是那个能拿来算的**动力学模型**。

### 2.3 为什么时间步是"一次交互"而不是"一个 token"

- 控制目标本身是回复级属性（词数、句数、段数、CEFR 等级、约束是否满足），验证器 $V$ 只能在整条回复上算。
- 逐 token 干预需要白盒，逐交互干预只需要能改 prompt。
- 类比：控制室温不需要建模每个空气分子，只需要房间尺度的动力学。token 是分子，交互是房间。

### 2.4 一段可以直接改写进 Intro 的英文骨架

> Multi-turn interaction turns the behavior of a frozen LLM into a trajectory: the length, style, or constraint-compliance of each reply depends on what was asked and answered before. Existing steering methods act inside the network—on activations, logits, or hidden states—and require white-box access at every token. In deployment the only lever is the prompt: one instruction per turn, against a black-box API. Interventions issued through this lever are today stateless: a system prompt written once, a reminder repeated on a schedule, a template fired by a per-turn classifier. None of them predicts what the next instruction will do, so they overshoot or undershoot. We ask whether prompt–response dynamics can be modeled well enough, at the interaction level, to plan the next instruction.

---

## 三、方法层动机：为什么用 Koopman

### 3.1 我们需要的预测器要同时满足五件事

| 需求 | 为什么 | Koopman 怎么满足 |
|---|---|---|
| 容纳非线性 | LLM 对指令的响应显然不是线性的 | 非线性编码器 $E_\phi$ 把状态提升到观测空间，非线性留在编码/解码里 |
| 隐空间线性 | 控制器要对每个候选指令做 $H$ 步 rollout；线性才有闭式 $K^h\xi + \sum K^j(Bc+b)$，不用再调 LLM | 同事 §3.4 式 (closed_form_rollout) |
| 显式输入通道 | "这条指令干了什么"要是一个可读的参数，而不是对文本的回归 | $B$ 列就是指令增益；tsar_cefr 上 `step_down` 增益 −0.0665（CI 排除 0）可当例子 |
| 小样本可辨识 | 轨迹只有几十到几百条（10 / 14 / 199 / 240） | 线性最小二乘 + 延迟嵌入，参数量以十计 |
| 可分析 | 审稿人会问"预测不准怎么办" | 同事 §3.5：预测误差界 $\Delta_h$ → 候选选择次优界 $2\delta_J$ |

一句话：LSTM 满足第 1 条、勉强第 4 条，不满足 2、3、5；逐轮 Markov 规则满足 2、4，不满足 1、3；Koopman 是同时满足五条的最小结构。

### 3.2 直觉引出公式

先说直觉：算子把"下一轮的属性"写成三部分之和——过去几轮属性带过来的那部分，这一轮指令推动的那部分，和一个常数。写成式子：

$$\hat{\xi}_{t+1} = K\,\xi_t + B\,c_t + b, \qquad \xi_t = E_\phi(s_t), \qquad \hat y_{t+1} = C\,D_\psi(\hat\xi_{t+1})$$

其中 $s_t$ 是最近 $L+1$ 轮属性拼成的记忆向量；$\xi_t$ 是它在观测空间的像；$K$ 描述惯性（过去怎样带到未来）；$B$ 描述指令 $c_t$ 的推力；$b$ 是偏置；$D_\psi$ 解码回记忆向量，$C$ 取出当前属性。**线性延迟嵌入是 $E_\phi = D_\psi = \mathrm{id}$ 的特例**，PPT 主表里的 Ours 就是这个特例。

### 3.3 和已有 Koopman-for-LLM 工作的区别

- Skifstad 2026（ALQR，*Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control*）和 RE-Control：线性/Koopman 结构用在**激活空间**，白盒。
- 我们：第一次把它用在**交互层的属性读数**上，黑盒，输入是离散的指令命令。这是方法层的 novelty 句。

---

## 四、三个贡献（Intro 结尾用）

1. **建模**：提出有限记忆、命令条件化的 Koopman 预测器，把一次 prompt–response 当作一步；给出从预测误差到候选选择误差的条件界。
2. **预测强**：在 11 个多轮任务（7 个属性追踪 + 4 个行为任务）上，4 步自由滚动误差在 11 列中 4 列最小、3 列第二，优于无记忆基线、LSTM 和非线性变体；记忆的必要性用配对 bootstrap 测出而非假设。
3. **控制有效**：用该预测器做 MPC，在 IFBench / IFEval / COLLIE 的 11 个子任务上，Qwen3-4B 平均 +14.5、Qwen3-8B 平均 +14.8 个点（相对 Base），比 TMPC 高约 6 点、比 RE-Control 高约 5 点，且不需要白盒访问、候选评估不增加 LLM 调用。

---

## 五、每个贡献背后已有的事实

### 5.1 预测（来源：PPT 09-17 主表，`numbers.yaml` n_main_*，commit 2d2718c / 6304abc / e8ac78d）

- 主表 11 列 × 6 行，Ours（线性延迟嵌入）11 列中 4 列 bold、3 列 underline。
- Claim A（记忆必要）：五个有量程的列上，多轮历史 vs 单轮的 Skill 差的 95% CI 不含 0；两个最小的列反向。
- 非线性版本 vs 线性：差异不超过 0.031、符号跨任务翻转（原 c3）——在正面口径下写成"线性特例已足够，编码器可按任务开关"。
- LSTM：11 列中没有一列显著好于线性。

### 5.2 控制（来源：`numbers_benchmark.yaml`，同事，暂视为可靠）

| 模型 | Koopman − Base 均值（范围） | TMPC − Base | RE-Control − Base | Koopman − RE-Control |
|---|---|---|---|---|
| Qwen3-4B | +14.5（+5.8 ~ +19.1） | +8.3 | +9.2 | +5.2（最小 +2.4） |
| Qwen3-8B | +14.8（+6.3 ~ +19.0） | +8.4 | +9.1 | +5.7（最小 +2.7） |

- 11 个子任务 × 2 个模型 = 22 格，Koopman 全部最优。
- Base = 冻结模型、不加任何控制；TMPC、RE-Control 为同事复现。
- 补充卖点：候选评估只在隐空间做，不调用 LLM（同事 §3.4 最后一段）；TMPC/RE-Control 需要隐层访问，我们不需要。

### 5.3 桥（预测 → 控制）

- 同事 §3.5 的两个界：$|y - \hat y| \le \Delta_h$，$J(c^\star) \le \min J + 2\delta_J$。
- 直觉：预测越准，选出来的指令越接近事后最优；这就是为什么先把预测做好。

---

## 六、表格与图的重排

**主文**

- **Fig. 1** 方法总览（同事 `figs/method.png`，补图注）。
- **Table 1（控制，headline）**：同事表重排为"按基准分块、按模型分组"，加一列 *Access*（white-box / black-box）和一列 *Avg.*（11 子任务均值），bold 保留。放在实验第一小节。
- **Table 2（预测）**：PPT 主表保留 7 个有结论的列（sentence_length_t10、vector_count_stage2_t10、vector_count_stage1_t10、character_length_t5、formality_t5、constraint、tsar_cefr）× 6 行；行名改成论文用语：Stateless / Last-turn / Memory w/o action / LSTM / Koopman (nonlinear lifting) / Koopman (linear, ours)。
- **Fig. 2（Claim A）**：记忆必要性的配对 bootstrap 森林图（PPT 第 5 页）。
- **Fig. 3（可选）**：某个基准子任务上 Base vs Koopman 的逐轮 $y_t$ 曲线，展示"逼近目标不过冲"——需要同事给轨迹；没有就先不画。

**附录**

- A1 估计器细节（现有 `sections/A1_estimator.tex` 改记号后可用）与 `equations.tex` 里的估计公式。
- A2 四个小样本列（sentiment_t5、average_word_length_t5、defense、gsm8k_sharded）：完整数字，声明不承担结论。
- A3 训练窗口匹配下 Claim A 的收缩（原 c2）。
- A4 等成本闭环对比（原 c5 / c7）：四个自建任务上 MPC 与最佳固定调度打平，tsar_cefr 3.37 步 / 322 token vs 1.03 步 / 100 token。**写法**：作为 "When does planning have headroom" 的边界讨论，指出这四类任务里指令效应几乎与时机无关、可控余量小，与基准任务的差别在于……（这里是要之后补理由的地方）。
- A5 一步预测误差（口径裁决 2，一步误差不做 headline）。

---

## 七、章节 → 文件映射与段落计划

| 章节 | 来源 | 状态 |
|---|---|---|
| Abstract | 新写，按第一节一句话故事展开 | 待写 |
| 1 Intro | 新写：P1 现象（2.1）；P2 三个 gap（2.2）；P3 为什么交互层（2.3）；P4 为什么 Koopman（3.1 压成一段）；P5 三贡献（四） | 待写 |
| 2 Related | 新写：多轮漂移测量；隐层控制方法（steering / CBF / PID / RE-Control / TMPC）；控制论视角的 prompting（Magic Word、GenCtrl）；Koopman 在 LLM 里的用法（ALQR） | 待写 |
| 3 Method | 同事 `3-method.tex` 全文保留；在 §3.2 加一句"线性特例"；把 Fig.1 图注补上 | 微改 |
| 4.1 Setup | 新写：两组任务（11 个多轮任务 + 3 个基准）、模型、基线、指标 | 待写 |
| 4.2 Control results | Table 1 + 文字 | 待写 |
| 4.3 Prediction results | Table 2 + Fig. 2 + 文字（含"线性够用"消融） | 待写 |
| 4.4 Analysis | 桥：预测误差 vs 控制收益；成本（零额外调用） | 待写 |
| 5 Limitations & Conclusion | 附录 A4 的边界讨论浓缩为一段；四小样本列；基准数字暂无 CI | 待写 |

记号统一：以同事的 $s_t, c_t, y_t, \xi_t, K, B, b, r, V, \mathcal{P}$ 为准；PPT 与旧稿里的 $z_t, u_t, v_t$ 全部换掉。

---

## 八、已知的不严谨与矛盾（之后修，现在只登记）

1. **非线性 vs 线性**：方法章的主角是 AE 编码器，预测表的最优行是线性特例。现在用"线性是特例、任务决定开关"过桥；基准任务上没有线性特例的数字。
2. **Base vs 等成本固定调度**：基准上只和 Base 比；自建任务上 MPC 和固定调度打平（附录 A4）。两组任务的结论方向不同，目前没有解释这个差别的实验。
3. **基准数字无 CI、无 seed 数、指标定义未写**。
4. **两组任务的指标不同**：基准是满足率（越高越好），自建任务是滚动误差（越低越好）；主文要用文字把两者的角色说清楚（一个测控制、一个测预测）。
5. **TMPC / RE-Control 是白盒方法**，与我们的比较在访问权限上不对等——写成优势而不是回避。
6. PPT 标题里的 "non-inferior control" 与本框架的正面控制口径不一致；组会材料不用改，论文以本框架为准。

---

## 九、Phase 0 重算后对第五节的更正（2026-09-18，Opus 登记）

`tools/build_tables.py` 从两个证据账本重算了主表和四组配对对比，结果与第五节有四处不符。
下面以重算结果为准；第五节的原文保留不改，供对照。所有数字都能用 `tables/_build_report.json` 复现。

| # | 第五节原话 | 重算结果 | 影响 |
|---|---|---|---|
| 7 | "Ours 11 列中 4 列 bold、3 列 underline"（七列口径下读作 4 最优、其余第二） | 七列口径：**4 列第一、1 列第二、2 列第三**。第三的两列是 character_length_t5 和 formality_t5 | contract `c8` 已按重算写。CP3c 的 Findings 不能说"其余第二" |
| 8 | "Claim A：五个有量程的列 CI 不含 0" | 七列里 **5 列 CI 不含 0**，与原话一致。两个例外是 vector_count_stage1（+0.246，CI 含 0）和 formality（−0.035，方向相反且含 0） | contract `c5`，措辞要点名这两列 |
| 9 | "非线性 vs 线性：差异不超过 0.031、符号跨任务翻转" | **不成立**。tsar_cefr 上线性比非线性高 **+0.599**（CI [+0.478,+0.741]）。七列里 4 列 CI 不含 0，全部偏向线性，没有一列偏向非线性且 CI 不含 0 | 这对我们是**有利**的更正：可以写"提升不来自 lifting"，而不是"两者打平" |
| 10 | "LSTM：11 列中没有一列显著好于线性" | **反了**。character_length_t5（−0.031，CI [−0.052,−0.001]）和 constraint（−0.027，CI [−0.037,−0.015]）两列 LSTM 更好且 CI 不含 0 | contract `c7` 降级为 observation，并加 `nc_6`。CP3c 不能声称对 LSTM 的普遍优势 |

### 11. 新登记：命令通道在预测实验里测不出来

四组配对对比里，"Ours − 记忆但不给动作"（row 3）**七列全部 CI 含 0**，点估计最大 0.025。原因分两层：

- 五个 core 列上，施加的命令就是跟踪误差 $r - y_t$，是状态的精确仿射函数（账本记 $R^2 = 1.0$，设计矩阵秩亏），
  所以这个设计**根本估不出独立的动作效应**，这五条 id 在账本里是 `caveated`。
- 剩下两列（constraint、tsar_cefr）能估，差分是 +0.0015 和 +0.0061，CI 仍含 0。

后果：方法章的核心设计之一（命令条件化的算子）在预测实验里没有正面证据。contract 已把
`choice_2` 改挂在 rq1（控制）下，并加了 `nc_1` / `nc_8` 两条不声称。**这是 Tier-1 冷审的 f1 阻断项，
需要用户裁决**：接受"控制实验即命令通道的检验"这个口径，还是补一个 no-command 消融。

### 12. 新登记：c7 两列失利缺机制解释

Rule 1 要求每个数字后面跟一句 driver。LSTM 在 character_length 和 constraint 两列赢，目前**没有机制解释**，
contract 里 `c7.driver_keyword` 暂填"no mechanism established"。CP3c 之前要么查清楚，要么这两列不带 driver 报出来。
