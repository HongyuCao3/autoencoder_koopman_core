# 行为三线：主表六行的结果档案

> **状态（2026-09-12）**：结果档案。core 侧对应档案见
> [`core_surrogate_rows_results.md`](core_surrogate_rows_results.md)，设计见
> [`../article/MAIN_TABLE_DESIGN.md`](../article/MAIN_TABLE_DESIGN.md)（E3 行）。
> **第四条线 `tsar_cefr`（Table 1 第十一列）另有一份平行档案**：
> [`tsar_cefr_surrogate_rows_results.md`](tsar_cefr_surrogate_rows_results.md)（2026-09-15，
> MAIN_TABLE_DESIGN 的 E6 行）。本文件的标题与三行表**只覆盖这三条线**，不含它。
> 术语 [`../NAMING.md`](../NAMING.md)；口径 `../../.claude/global.md` → *报告口径*。
> **本文件不改任何既有实验文档的结论段。**

## 这不是复跑，是新测量

三条线签过的闸门**全部是一步、`nu=1`**——`constraint` 的 S2 配置字面就是 `{nu: 1, mu: 1}`，
**无延迟嵌入的 Markov 状态**。主表第 6 行（延迟嵌入 + 控制）在这三条线上**一次都没被拟合过**。
所以第 2 行才是对应各线已发表算子的那一行，第 6 行是新的。

## 产物与谱系

| | |
|---|---|
| 脚本 | `persona_drift_control/scripts/eval_surrogate_rows_behavioral.py`（新增） |
| 打分 | `src/surrogate_eval/`（与 core 侧同一套） |
| 模型类 | 本线自己的 `modeling.koopman` / `lstm_baseline` / `ae_baseline`（两套代码体系不互相 import） |
| 产物 | `persona_drift_control/outputs/surrogate_rows_behavioral/{line}.json`，含 `provenance` |

**深度与视界逐线取**（同 core 的规则：还能留 H≥3 的最深 y 块，目标 H=4 = MPC 规划视界）：
`constraint` nu=4/H=4、`gsm8k_sharded` nu=4/H=4、`defense` **nu=2/H=3**（只有 5 轮）。
**窗口由最深状态定义，每个模型在同一绝对轮次上用自己的状态播种**，六行落在逐行相同的评测集上。

## 结果（`Skill_H`，按题 bootstrap 95% CI；折按 `purpose="report"` 划分，与闸门折不同）

| 线 | nu/H | 题 | seed | 评测行 | 最好 null | Markov | 扣住 u | LSTM | AE | **Ours** |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| **`constraint`** | 4/4 | 40 | 3 | 12376 | `stateless` | +0.285 | +0.518 | +0.546 | +0.471 | **+0.519** [+0.38,+0.62] |
| `gsm8k_sharded` | 4/4 | 60 | **2** | **64** | `const` | +0.637 | −0.186 | −0.795 | −0.130 | **−0.129** [−0.50,−0.09] |
| `defense` | 2/3 | 30 | **2** | 180 | `const` | +0.068 | +0.023 | −0.054 | +0.021 | **+0.035** [−0.08,+0.15] |

### 配对 bootstrap（`ours − baseline`，★ = CI 排除 0）

| 线 | vs Markov | vs 扣住 u | vs LSTM | vs AE |
|---|---|---|---|---|
| `constraint` | **+0.234** [+0.18,+0.28] ★ | **+0.0015** [−0.00,+0.00] | −0.027 [−0.04,−0.01] ★ | +0.048 [+0.02,+0.07] ★ |
| `gsm8k_sharded` | −0.766 [−0.87,−0.76] ★ | +0.057 [−0.04,+0.16] | +0.665 [+0.56,+0.96] ★ | +0.001 [−0.15,+0.02] |
| `defense` | −0.033 [−0.10,+0.01] | +0.012 [−0.02,+0.04] | +0.089 [−0.03,+0.21] | +0.014 [−0.00,+0.03] |

## 绝对值（rollout MSE ↓，读出原生单位）

`MAIN_TABLE_DESIGN.md` §一 规定每个数据集 **2 列指标**：`Skill_H`（相对）与 **rollout MSE（绝对）**。
上表只登记了相对值，本节补上绝对值。取自
`persona_drift_control/outputs/surrogate_rows_behavioral/{line}.json` 的 `rows.*.rollout_mse`，
**同一批产物、同一个评测集，未重拟合**。（第四条线 `tsar_cefr` 的绝对列已在
[`tsar_cefr_surrogate_rows_results.md`](tsar_cefr_surrogate_rows_results.md) §结果 里。）

| 线 | nu/H | 评测行 | 最好 null | Markov | 扣住 u | LSTM | AE | **Ours** |
|---|---|---:|---|---|---|---|---|---|
| **`constraint`** | 4/4 | 12376 | 0.03844 (`stateless`) | 0.0275 | 0.01855 | 0.01746 | 0.02034 | **0.01849** |
| `gsm8k_sharded` | 4/4 | 64 | 0.2643 (`const`) | 0.09598 | 0.3134 | 0.4742 | 0.2987 | **0.2984** |
| `defense` | 2/3 | 180 | 0.07361 (`const`) | 0.06864 | 0.07193 | 0.07757 | 0.07207 | **0.07104** |

**绝对列让两件在相对列里看不出来的事变得可见**：

1. **`defense` 六行贴 0 的量级。** 最好 null 0.07361，六行全部落在 **0.0686–0.0776**——
   全距 0.009，不到 null 的 13%。相对列里 `+0.068` 与 `−0.054` 看着像两个方向，
   绝对列里它们是同一个数的两侧毛刺。这是"读出没量程"的直接呈现。
2. **`gsm8k_sharded` 的深状态是在 64 行上过拟合，不是任务难。** Markov 0.09598 对 Ours 0.2984
   ——深状态把误差**放大 3.11×**，LSTM 更放大到 4.94×。相对列的 `−0.129` / `−0.795` 是同一件事的压缩。

**三条读法与 core 侧同**（见 [`core_surrogate_rows_results.md`](core_surrogate_rows_results.md) §绝对值）：
跨列不可比（三线读出尺度不同，且 `defense` 是自判分、`constraint` 是独立 14B）；
绝对列无区间，**"两行有没有差别"一律仍看配对 bootstrap**；
AE / LSTM 行是按 seed 平均后的逐行平方误差。

## 三条读法

**1. `constraint` 上「记忆有用」成立，量级与 core 的样本足任务一致。** `ours − Markov`
**+0.234 ★**，12376 行 / 40 题 / 3 seed / 读出有量程（独立 14B）。加上 core 的
+0.603 / +0.312 / +0.235，**四个信息量足够的列全部显著支持延迟嵌入**。

**2. 「扣住控制量不花钱」跨线成立，而且 core 的解释在这里不适用。**
`constraint` 上 `ours − 扣住 u` = **+0.0015**，`gsm8k` +0.057、`defense` +0.012，全部跨 0。
（**2026-09-15 补记**：第四条线 `tsar_cefr` **+0.0061，CI 含 0**，同样跨 0——那条线的动作是
真随机化的**四值分类**、且被设计成本条的阳性对照，它没有兑现。第 3 行至此**十一列一致为零**。）
core 侧可以辩解「`r` 每条轨迹恒定、不是随机化动作」，**但 `constraint` 的 `u` 是伯努利随机化的真动作**。
随机化的动作被扣住，4 步预测一点不变差。
这与 S1a 的终点权威 **+0.1389** 不矛盾，但必须同址讲清楚：一步增益 `B` = **+0.0182**，
滚 4 步累计约 0.05，相对读出自身波动太小。
**提醒对「19 轮后的终点」有权威，对「4 步后的读出」没有预测力。**

**3. 非线性买不到东西，行为线与 core 一致。** `constraint` 上 `ours − LSTM` = −0.027 ★、
`ours − AE` = +0.048 ★——**两个方向都显著、都 ≤0.05**。措辞同 core：用效应量讲，不讲「不可区分」。

## 两列没有信息量（如实进表，不挑列）

- **`gsm8k_sharded`：只有 64 个评测行。** `nu=4 + H=4` 要求至少 9 轮，而该线轨迹多数更短。
  深状态在 64 行上明显过拟合（`ours − Markov` −0.766 ★，Markov 反而大幅更好）。
  报「本设计分辨不出来」，不作结论。
- **`defense`：六行全部贴 0，且最好的 null 是 `const`。** 连 `turn` 和动作都预测不了下一轮安全分。
  三条独立证据说明这是真的、不是实现问题：① 最好 null 是 `const`；
  ② 该线自己的 G-K2-1 是**一步**上 14/20 折，**恰好等于预注册阈值、零余量**；
  ③ `y_safety` 只有 5 个取值、ceiling 0.91。**读出没量程，算子就没东西可学**
  ——与 core 的 `even_odd_t5` 报 `DegenerateNullError` 是同一机制的连续版本。

## 一条把 Table 1 与 Table 2 连起来的结果

三条线的闭环都打不赢等代价固定日程（`defense` 的 periodic 追平、`gsm8k_sharded` 日程可分性=1、
`constraint` MPC − 最优固定 +0.0008）。**2026-09-15 补记：第四条线同向**——`tsar_cefr` 的
`koopman_mpc` 对 `fixed_ladder` −0.0073（含 0），等代价轴上花 3.2× token 换一个打平、被支配。**第 3 行的零给了这件事一个机制解释**：
算子在**控制器规划的那个视界上**根本没有动作依赖的预测结构，
那么用这个算子去规划的控制器，**在原理上就无法优于一个固定日程**。
Table 1 第 3 行与 Table 2 的打平不是两件事，是同一件事的两个观测面。

## 执行中修掉的三个问题（都在出数之前）

1. **Markov 行的窗口起点比深状态行早**，两者被打在不同行集上（断言抓到）。
   改为窗口由最深状态定义、各模型在同一绝对轮次上用自己的状态播种。
2. **判分失败写的是 `null` 不是 NaN**，`build_reduced_state_pairs` 只检测 NaN，`None` 会走到
   `float()` 才炸；且 NaN 丢行会破坏「第 j 对 ↔ 第 start+j 轮」的索引映射。
   改为载入时整条剔除含未解析读出的轨迹并计数（`constraint` 剔 2 条 / 240）。
3. **AE 被我喂瘦了。** 首轮没给 `val_dataset`，而该类的 `early_stopping_patience`
   在没有验证集时是**空操作**（跑满 `num_epochs=300`），同时 LSTM 却拿了验证集选隐层。
   `constraint` 上 AE 因此是 **−0.369**（比常数 null 还差）。补上与 LSTM 同一份 80/20 验证切分
   + `num_epochs=3000` 后是 **+0.471**。**首轮那个数不得引用。**

## 同址必带的局限

1. **`defense` 与 `gsm8k_sharded` 只有 2 个数据 seed。** ~~待用户裁定~~
   **已裁定（2026-09-13，用户裁决 6）：两列保留在正文 Table 1，不删不移。**
   `global.md` 的 ≥3 seed 条款约束的是「以 seed 为复制单元的 `mean ± std (n)`」，本表不印这种数——
   seed 在打分前已被平均掉（降噪手段），误差棒来自题/攻击级 bootstrap（30 / 60 个单位）。
   **豁免边界写死在 `MAIN_TABLE_DESIGN.md` §零裁决 6**，不外溢到任何以 seed 为复制单元的数字。
   **caption 义务**：同址写 `n_seed=2`、重采样单位、以及本列判读为「本设计分辨不出来」。
   不补第 3 个 seed——它改变不了这两列的判定（见本节第 2、3 条与正文两条"没有信息量"）。
2. **`defense` 列判分是自判**（`global.md` 具名例外）：自判 = 单向漏检、systematically 低估效应。
   例外不外溢到其它列。
3. 这三列的模型类是各线自己的实现，**打分路径统一但拟合路径不统一**；与 core 列并读时，
   跨列可比的是 `Skill_H` 的定义与区间构造，不是拟合器。
4. 为公平对齐观测前缀，给 `LSTMSurrogate` 加了 `warm_start`（空前缀逐位等于 `init_state()`）、
   给 `AEKoopmanSurrogate` 加了 `y_index`（默认 0 = 旧行为）。两者都不改既有方法，均带回归测试。
