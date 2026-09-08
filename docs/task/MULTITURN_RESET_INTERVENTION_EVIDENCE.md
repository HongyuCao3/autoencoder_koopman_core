# reset 介入的外部证据：设计新激励臂之前的文献核查（2026-09-08）

**为什么有这份文件**：EK0 的闭合前审查判定要重设计激励臂（Q16）。在提交任何 GPU 作业之前先问
一句——**这种介入方式，别人验证过有效还是无效？** 体例沿用
[`CONTROL_THEORETIC_LLM_RELATED_WORK.md`](CONTROL_THEORETIC_LLM_RELATED_WORK.md)：只调研，不改代码与协议。

**结论先说**：**是佐证，不是反证。** reset 这类介入在**我们这个模型家族、这个规模、这个任务**上
被独立验证过有效（+6.8 分）。但外部证据同时给了三条硬约束，它们直接决定新臂怎么设计。

---

## 一、直接命中：有人在 Qwen3-4B / 分片数学上跑过 reset 介入

**Lin et al., *Same Evidence, Different Answers: Canonical-Context On-Policy Distillation for
Multi-Turn Language Models*, arXiv:2605.30251**（[html](https://arxiv.org/html/2605.30251v1)）

这是目前唯一一篇设定与我们几乎重合的工作：**Qwen3-4B / 8B / 14B + Llama3.1-8B**，分片（sharded）
多轮数学。

| | Qwen3-8B，math Raw-Sharded |
|---|---:|
| base（不介入） | 66.0 |
| **+RTA（Reset-Then-Answer，prompt 级）** | **72.8（+6.8）** |
| +DUC（Defer-Until-Complete） | 67.0（+1.0） |
| CCOPD（他们的训练法） | 82.5（+16.5） |
| CCOPD + RTA | 79.6（**−2.9**，反而更差） |

三件事直接可用：

1. **reset 在 Qwen 上是有效介入**，不是被证伪的死路。
2. 他们给我们的机制起了名字：**self-anchored drift**——"在信息不全时产生的早期回复引入了没有依据的
   假设，这些假设随后扭曲了最终答案"。**这与 R1 的证据是同一件事**（append 让模型抄自己上一轮的
   简短旧答案）。论文里有现成的术语和引用。
3. **他们把 prompt 级修复贬为"需要一个额外的外部控制回路，因而改变了推理过程本身"，主张应该由
   训练内化。** ——**那个"外部控制回路"就是本项目的 Koopman-MPC。** 这既是竞品定位，也是一个
   被点名的空位：他们论证了内化更好，但没有论证外部回路**做不好**；而且 CCOPD+RTA 掉 2.9 分说明
   两条路会互相干扰，**"什么时候该介入"这个问题在他们的框架里反而没被回答。**

---

## 二、效应量：0.07 ~ 0.18，这是设计新臂的硬约束

| 来源 | 模型 | 介入 | 相对无介入的增益 |
|---|---|---|---:|
| CCOPD RTA | Qwen3-8B | reset-then-answer（固定规则） | **+0.068** |
| Laban et al. 2505.06120 表 2 | GPT-4o-mini | RECAP | +0.161 |
| 同上 | GPT-4o | RECAP | +0.175 |
| 同上 | GPT-4o-mini | SNOWBALL | +0.114 |
| 同上 | GPT-4o | SNOWBALL | +0.062 |
| **本项目 R1** | **Qwen3-4B** | `always_reset − zero_control`（upstream profile） | **+0.181** |

**我们的 +0.181 落在文献区间内**，不是异常值。反过来说：

> **任何新臂的 MDE 必须 ≤ 0.07 才安全，≤ 0.15 是底线。**
> 现在这个 `random_excite p=0.5` 臂的 MDE 是 **+0.49**——比需要的钝 7 倍。这不是"再多跑几个 seed"
> 能解决的量级差距，必须换设计。

---

## 三、三条反证 / 限制，每条都改一处设计

### 3.1 ERGO 从未在 14B 以下测过，且没有拆开它的两个部件

核查 arXiv:2510.14077 全文：模型集是 **GPT-4o / GPT-4o-mini / Llama-3.1-405B / Phi-4(14B)**，
数据集 GSM8K / Spider / LiveCodeBench / BFCL / ToTTo。**没有 Qwen，最小 14B。**
**也没有把"熵触发的时机"与"模型重写的 consolidation prompt"分开的消融。**

→ **影响**：ERGO 的 +14.1/+9.3（自适应打赢最优固定规则）在 4B 上**没有任何外部支持**，本项目
不能把它当既有事实来推论。同时这条空缺是可占的贡献位——那个消融我们做得起。

### 3.2 Laban 的 `RECAP > SNOWBALL` 只有 2 个模型、没有误差棒

表 2 只报了 GPT-4o-mini 与 GPT-4o 两行，无 CI，无每格样本量。

→ **影响**：G-R1-3（我们没复现出这个排序）**本来就没多少东西可复现**。这独立支持了 2026-09-08
关闭 R4 的裁决（Q13），理由从"不填论文任何一格"又多了一条"被复现对象本身证据强度低"。

### 3.3 过度 / 时机不当的 reset 会因语义漂移抹掉收益

ERGO 自己指出：reset 的主要风险是 semantic drift——过度或时机不当的上下文重写会因抽象化而
丢失关键细节，**足以抵消甚至反转收益**。CCOPD 的 CCOPD+RTA **−2.9** 是同一现象的另一个例子。

→ **影响**：这条是**支持自适应的最强论据**（收益非单调 ⇒ 存在最优时机 ⇒ 值得用控制器找），
但它也否掉了"只跑 0/1 两端"的省事设计——**非单调性只在中间预算里看得见，中间档必须保留。**

### 3.4 小模型的多轮退化形态可能不同

多篇观察指出 4B 级模型在 3–5 轮后表现为**重复与照抄自己上一轮输出**，且比大模型更明显。

→ **影响**：4B 上的失效可能主要是"echo"而不是 ERGO 针对的"drift"。这两者对 reset 的响应未必相同，
**臂里要留能区分它们的记录项**（末轮回复与上一轮的逐字重合率，我们已有 `agent_message` 全文）。

---

## 四、一条意外的正面证据：Qwen3-4B 在 GSM8K 上的置信度校准是同类最好的

多篇校准研究报告 **Qwen3-4B 平均 ECE = 0.163（同类最佳），其中 GSM8K 上 ECE = 0.018**；
校准随任务难度单调恶化（GSM8K 0.110 → MATH500 0.359 → AIME 0.632），**GSM8K 是最好的一档。**

→ **影响**：本项目 2026-09-07 关掉熵读出族（RC-1/RC-3 不过）是在 **legacy profile** 上测的，而
R1 已证明那份数据被截断回复污染（`always_reset` 末轮 91% 是裸的 `Current answer: X`）——
**在退化输出上测熵，测的是退化，不是不确定性。** 外部证据说这个信号在 Qwen3-4B + GSM8K 上
应该是好的。**R5（熵触发臂）值得重新评估，但按 `.claude/global.md` 这属于"第二次修同一个仪器"，
必须有用户显式签字才能重开。**

---

## 五、对 Q16（新激励臂）的落地约束

| # | 约束 | 来源 |
|---|---|---|
| C1 | **MDE ≤ 0.07**（底线 0.15）。现设计 0.49，差 7 倍 | 第二节效应量表 |
| C2 | **同题配对 + 两端都要取到**，这是 8 倍效率的来源 | 本项目功效计算 + R1 的设计 |
| C3 | **中间预算不能砍**，非单调性只在那里可见，而它是自适应主张的立命之本 | 3.3 |
| C4 | **竞品基线要点名**：固定规则 RTA / RECAP，以及 ERGO 的熵触发。S1 = 同代价打赢这些，不是打赢 CCOPD（那是训练法，不同赛道） | 第一、三节 |
| C5 | **加一个记录项**：末轮与上一轮回复的逐字重合率，用来分开 "echo" 与 "drift" 两种失效 | 3.4 |
| C6 | 论文定位一句：CCOPD 主张把一致性内化进权重、并把外部控制回路当作缺点；**本文正面研究那个外部回路，并给出它何时值得** | 第一节 |

**本文件不提交任何 GPU 作业。** 新臂的具体规格与运行时估计另出，按
[`../.claude/experiments.md`](../.claude/experiments.md) → *提交 GPU 作业前* 六条走。

---

## 六、来源

- Lin et al., *Same Evidence, Different Answers: Canonical-Context On-Policy Distillation for Multi-Turn Language Models*, arXiv:[2605.30251](https://arxiv.org/abs/2605.30251)
- Laban et al., *LLMs Get Lost In Multi-Turn Conversation*, arXiv:[2505.06120](https://arxiv.org/abs/2505.06120)
- *ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models*, arXiv:[2510.14077](https://arxiv.org/abs/2510.14077) / [ACL Anthology](https://aclanthology.org/2025.uncertainlp-main.23/) / [OpenReview](https://openreview.net/forum?id=xPWycbxQn9)
- 校准数据来自多篇 Qwen3 校准实证研究（*From token probabilities to calibrated confidence*, arXiv:[2608.07827](https://arxiv.org/html/2608.07827)；*Different Facets of Verbalised Overconfidence*, arXiv:[2608.18106](https://arxiv.org/html/2608.18106)）

> **核查方式与局限**：全部通过网页检索与全文抓取完成（2026-09-08）。CCOPD 表 3 与 Laban 表 2 的
> 数字来自 HTML 全文抓取，**未逐一人工复核 PDF**；CCOPD 未公布 Qwen3-4B 的 RTA/DUC 分项、
> 也未报误差棒与样本量，因此 +6.8 这个数**只能当量级参考，不能当基线数字直接引用**。
