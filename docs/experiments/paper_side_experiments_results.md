# 论文副实验 A1 / M1 / M2：结果档案（2026-09-16）

> **状态**：结果档案。计划与事前签死的三行见
> [`../paper_side_experiments_plan_2026-09-16.md`](../paper_side_experiments_plan_2026-09-16.md)。
> 数字口径见 `../../.claude/global.md` → *报告口径*；术语见 [`../NAMING.md`](../NAMING.md)。
> **本文件不改** [`core_surrogate_rows_results.md`](core_surrogate_rows_results.md) /
> [`behavioral_surrogate_rows_results.md`](behavioral_surrogate_rows_results.md) /
> [`../article/CLOSED_LOOP_SYNTHESIS.md`](../article/CLOSED_LOOP_SYNTHESIS.md) 的任何结论段。

## 产物与谱系

| | |
|---|---|
| 脚本（core） | `scripts/paper_side_experiments.py`（新增，不改 `eval_surrogate_rows.py` / `core.py` / `train.py`） |
| 脚本（行为线） | `persona_drift_control/scripts/paper_side_experiments_behavioral.py`（新增，不改 `eval_surrogate_rows_behavioral.py`） |
| 打分 | `src/surrogate_eval/`（与 Table 1 同一套：`skill_h` / `trivial_nulls` / `bootstrap_ci` / `make_folds`） |
| 产物（core） | `results/{ablation_memory_depth, ablation_memory_depth_matched, mechanism_operator_profile, mechanism_item_effect}/` |
| 产物（行为线） | `persona_drift_control/outputs/{ablation_memory_depth, ablation_memory_depth_matched, mechanism_operator_profile_behavioral_fixed, mechanism_item_effect}_behavioral*/` |
| 单测 | `tests/test_paper_side_experiments.py` 10 条 + `persona_drift_control/tests/test_paper_side_experiments_behavioral.py` 5 条，全过；`tests/` 全量 48 条全过 |
| 基线 commit | `2d2718c` |
| **已作废** | `results/mechanism_spectrum/`（M1 第一版，只有谱没有 tap loading）；`persona_drift_control/outputs/mechanism_operator_profile_behavioral/`（行为线 M1 第一版，**tap 读了 `C`，而本代码体系的 `C` 恒为选择向量，该量恒等于 `[1,0,…]`，无信息**）。两者**原地保留不覆盖**，不得引用 |

**运行时**：A1 core 3m17s（事前估 <20 min，**高估**）、A1 matched core 同量级、M1 core 22 s、M2 core 1m23s、行为线三件各 5–20 s。**合计 < 10 min，事前估 35–55 min。**

**口径**：全部是 `Skill_H` + 按轨迹/题配对 bootstrap（2000 次），与 Table 1 逐行同构（裁决 6 的豁免覆盖此形）。
谱半径 / 半衰期 / tap loading / 脉冲响应是**拟合诊断量**，按 `global.md` **不得作为跨臂头条**。
`defense` 列全列 `judge_kind=self`（`global.md` 具名例外），引用处必带单向漏检局限句。

---

## 一、A1 记忆深度扫描 —— 判「**过**」，但深度比预期浅，且**暴露了 Table 1 第 2 行的一个训练窗口混杂**

### 1.1 曲线（primary 锚，与 Table 1 同 lag/H；`Skill_H` 逐 lag）

| 列 | lag/nu 0 → max | 增量（★ = CI 排除 0） | 判 |
|---|---|---|---|
| `sentence_length_t10` | +0.166 → +0.549 → +0.750 → +0.769 | +0.388★ / +0.128★ / +0.038★ | 单调、饱和 |
| `vector_count_stage2_t10` | +0.514 → +0.804 → +0.825 → +0.826 | +0.236★ / +0.018★ / +0.008★ | 单调、饱和 |
| `vector_count_stage1_t10` | +0.395 → +0.640 → +0.654 → +0.640 | +0.175★ / +0.010 / −0.007 | 单调、lag 1 后饱和 |
| `constraint`（nu 1→4） | +0.285 → +0.431 → +0.492 → +0.519 | +0.147★ / +0.061★ / +0.027★ | **单调、到 nu=4 仍在买** |
| T=5 四列 / `defense` / `gsm8k_sharded` | 只有两格或负向 | — | **本设计扫不动**（如实标注，不换口径） |

**deep 锚（H=3，lag 0–6）**确认 Table 1 的 lag=3 没有留下东西：lag 4/5/6 的增量在三列上分别是
+0.020★/+0.002/+0.003、−0.010/−0.005/−0.004、+0.000/−0.001/−0.005★。**深度到 3 为止。**
deep 锚与 primary 锚 **H 不同，两组格子不得并排比较**。

### 1.2 训练窗口混杂（**新发现，2026-09-16**）

浅状态比深状态更早可用，所以默认口径下 lag 0 行是在**深状态从未见过的早期转移**上拟合的。
把所有 lag 的训练转移对齐到同一集合（matched 变体）后：

| 列 | `ours − Markov` 默认口径 | 同窗口口径 | 变化 |
|---|---|---|---|
| `sentence_length_t10` | **+0.603★** | **+0.109★** | 缩到 1/5.5 |
| `vector_count_stage2_t10` | +0.312★ | +0.229★ | 缩 27% |
| `vector_count_stage1_t10` | +0.245★ | +0.133（跨 0） | **失去显著** |
| `character_length_t5` | +0.235★ | **−0.035★** | **翻号** |
| `sentiment_t5` | +0.221（跨 0） | +0.155★ | **变显著** |
| `average_word_length_t5` | **−0.409★** | +0.079（跨 0） | **翻号并失去显著** |
| **`constraint`** | **+0.234★** | **+0.231★** | **几乎不变** |

而且**同窗口下曲线在 lag 1 就平了**（`sentence_length_t10` +0.112/+0.110/+0.109；
`vector_count_stage2_t10` +0.228/+0.226/+0.229）。

**读法（措辞红线）**：
- **不能**写「Table 1 第 2 行是假的」。两种口径都成立，问的是不同问题：默认口径问「每个模型类用尽它状态允许的数据时谁更好」，同窗口口径问「只把深度换掉买到什么」。
- **消融要报的是同窗口口径**——它才把深度从训练窗口里分出来。
- **`core` 与 `constraint` 在这一点上不同**：`constraint` 两种口径差 0.003，**该线的记忆结论不受这个混杂影响**。
- 引用 `core` 侧任何 `ours − Markov` 数字时必须同址说明是哪种口径。

### 1.3 兑现的三行

**过了 →** 曲线单调并饱和 ⇒ Table 3 panel 1 + lag→`Skill_H` 曲线图，正文写「记忆是被测出来需要的，
**深度 ≈ 1 tap（core，同窗口口径）/ ≥4 turns（`constraint`）**」。**已兑现，主线动作 = 落图落表。**

---

## 二、M1 谱 + 脉冲响应 —— 判「**不过**」（按事前签死的第二行），机制图不画

### 2.1 数（行为线三列，动作为真随机化，秩检查全部通过）

| 线 | 谱半径 | 状态半衰期 | 非 Markov 份额 | 动作脉冲响应 | 第 1 步占比 |
|---|---|---|---|---|---|
| `constraint` | 0.943 | **11.9 轮** | 0.793 | [0.0146, 0.0031, 0.0041, 0.0047] | 0.554 |
| `defense` ⚠self | 0.379 | 0.71 轮 | 0.375 | [0.0617, 0.0268, 0.0184] | 0.577 |
| `gsm8k_sharded` | **1.104**（不稳定） | ∞ | 0.601 | [0.0323, **0.0653**, 0.0418, 0.0305] | 0.190 |

`core` 八列的动作通道是**每条轨迹恒定的参考**，脉冲在数据里从未出现 ⇒ 脉冲曲线是**从拟合外推的反事实**，
同址已标；其中 `even_odd_t5` 秩检查判**不可辨识**，按 P1-a **不出数**。

### 2.2 为什么判不过

事前签的第二行是：「可辨识的列 < 2，**或**测到脉冲响应长度 ≥2 却仍『重排买不到』⇒ 不画机制图」。
**第二个触发条件成立**：三条行为线的脉冲响应长度都 ≥2（按脚本的 ≥10%×峰值规则为 3/3/4），
`gsm8k_sharded` 的峰值甚至在**第 2 步**，而四条闭环列全部打平。
**「动作一步兑现所以重排不值钱」这个机制说法没有被数据支持。**

⚠️ **一处未签的自由度，交还用户**：判据里的「脉冲响应长度」阈值**事前没有签**，
脚本用的是「≥10%×峰值」。换一个同样合理的统计量（第 1 步占总响应的份额）读数是
0.554 / 0.577 / 0.190——`constraint` 与 `defense` 上一半以上的效应确实落在第 1 步。
**判词随统计量翻转，所以本条按「不过」记，措辞由用户裁决**，不由脚本的默认阈值决定。

### 2.3 后果

按第二行兑现：**§5.x 不画机制图**，只保留
[`../tsar_cefr_postmortem_and_next_tasks_2026-09-16.md`](../tsar_cefr_postmortem_and_next_tasks_2026-09-16.md)
§2.4 的数量级核算；`CLOSED_LOOP_SYNTHESIS.md` §六② 的措辞从机制解释**降为唯象并列**。

### 2.4 一条没被预期到的正面信息

半衰期与「记忆值不值钱」**不同号**：`formality_t5` 半衰期 11.5 但 `ours − Markov` 分不开，
`character_length_t5` 半衰期 0.82 却在默认口径下 +0.235★。
**慢 AR(1) 半衰期长但完全 Markov，延迟嵌入在它上面买不到东西**——半衰期是错的机制量，
对的是**旧 tap 的载荷**（`lag_loading.non_markov_share`：三条 T=10 列 0.873 / 0.915 / 0.982，
`constraint` 0.793）。这条改的是机制量的选择，不改任何已发表数字。

---

## 三、M2 题目截距替代 —— 判「**过**」（两列显著为正），但**题目身份吃掉了大部分增益**

| 列 | `ours − Markov`（同窗口） | 题目截距买到 | **`ours − (Markov+题目截距)`** | 截距吃掉的份额 |
|---|---|---|---|---|
| **`constraint`** | +0.2306★ | +0.1861★ | **+0.0445★** [+0.0153,+0.0859] | **81%** |
| `vector_count_stage2_t10` | +0.229★ | +0.154★ | **+0.075★** [+0.020,+0.145] | 67% |
| `sentence_length_t10` | +0.109★ | +0.068（跨 0） | +0.041（跨 0） | — |
| `vector_count_stage1_t10` | +0.133（跨 0） | +0.133（跨 0） | +0.000（跨 0） | 全部 |
| `sentiment_t5` | +0.155★ | +0.161★ | −0.006（跨 0） | 全部 |
| `average_word_length_t5` | +0.079（跨 0） | +0.042（跨 0） | +0.037（跨 0） | — |
| `formality_t5` | −0.025（跨 0） | −0.099（跨 0） | +0.074★ | 截距**反而有害**，不算支持 |
| `character_length_t5` | −0.035★ | +0.000 [0,0]★ | −0.035★ | ⚠️ 截距权重为 0，该 ★ 是浮点噪声上的**伪显著**，不得引用 |
| `defense` ⚠self | −0.033（跨 0） | −0.064（跨 0） | +0.031（跨 0） | 全格跨 0 |
| `gsm8k_sharded` | −0.290★ | **+0.609★** | −0.899★ | 记忆有害、题目身份极强；**64 个评测行**，不承重 |

### 3.1 兑现的三行与读法

**过了 →**「≥2 个信息量足够的列上仍显著为正」：`constraint` +0.0445★ 与
`vector_count_stage2_t10` +0.075★ 两列兑现 ⇒ **Table 3 panel 3 + 正文**。

**但正文必须同址带这句**：**延迟嵌入买到的东西里约 2/3 到 4/5 是持久的题目级异质性，
不是跨轮动力学**；剩下的那一截在两列上显著、在其余列上分不开。
写成「延迟嵌入携带跨轮动力学」而不带这个份额，是超出数据的。

---

## 四、对论文的直接后果（三条，供正文改写）

1. **Table 3（消融）**：panel 1 印同窗口口径的 lag 曲线；`core` 侧任何 `ours − Markov` 数字标明口径。
2. **§5.x（机制）**：**不画 M1 的机制图**（判不过）；改为把 M2 的份额分解写成机制段——
   「状态里占主导的是题目级持久异质性，动作相关结构只占一小截」，它同时解释
   §六①（记忆值钱）与 §六②（重排不值钱），且**两条都有实测**。
3. **Limitations**：①「joint 的多步项从未在可比强度下被扫过」（A3 不做）；
   ②「Table 1 第 2 行在 `core` 上含训练窗口成分，同窗口口径下缩到 1/5.5 至 1/1.3，
   一列翻号；`constraint` 不受影响」。
