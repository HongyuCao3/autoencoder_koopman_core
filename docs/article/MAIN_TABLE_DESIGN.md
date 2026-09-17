# 论文主表设计与填表清单

> **状态**：设计稿（2026-09-12，2026-09-13 更新 Table 2 与 ddof 签字），四条裁决已签，空壳先行（`../../.claude/paper.md` → *论文空壳先行*）。
> **本文件不是计划文档**，活跃计划仍是 [`../experiments/constraint_retention_plan.md`](../experiments/constraint_retention_plan.md)。
> 权威序上从属于 [`PAPER_EXECUTION_PLAN.md`](PAPER_EXECUTION_PLAN.md)；数字口径见
> `../../.claude/global.md` → *报告口径*；术语见 [`../NAMING.md`](../NAMING.md)。

## 零、2026-09-12 四条裁决（用户签字，均在看过证据盘点之后）

| # | 裁决 | 直接影响 |
|---|---|---|
| 1 | **闭环打平改框架留正文**，不挪附录 | 正文 RQ3 从"我们的控制器更好"降级为"闭环在什么条件下才买得到东西"。`paper.md` 诚实性红线不变，§1.5 不改写 |
| 2 | **定向解禁多步 `Skill_H`** 当正文头条数字 | 一步预测误差仍禁（只进附录）。报告用的折划分必须与闸门用的**不同且事前注册**——选择信号不得同时当报告信号 |
| 3 | **"ours" = 延迟嵌入受控 Koopman（线性）**，AE 降为消融行 | 叙事变成"先验成立且成立得更简单，简单性是被测出来的" |
| 4 | **先判定向提醒试点闸门（E0）再决定闭环 GPU** | G3 不在 E0 之前提交 |

**E0 已执行（2026-09-12，commit `79a08c9`，零 GPU）：主量 PASS。** 定向 − 整段 **+0.0748 ± 0.0409（n=3 seeds）**，
CI [+0.0367, +0.1202]，1.47×MDE，35 题 / 1499 条观测，22 条单测全过。→ 裁决 4 解为 **G3 值得花，但 S3 臂表与口径必须先重签**（§五 G3）。
**同址必读的次要量**：未点名约束的保持率 **−0.1059**（CI [−0.1506, −0.0639]），**量级大于收益**；净 `y` 定向 − 整段 −0.0225（CI 刚跨 0）。
次要量不得提拔为主量——但它决定了 S3 的动作空间必须 ≥3 个（见 §二）。

### 2026-09-13 两条裁决（用户签字，均在看过 E4 / E2 / E3 结果之后）

| # | 裁决 | 直接影响 |
|---|---|---|
| 5 | **Table 2 `defense` 列主量 = `late y`(t3–5)**，终点读出降为同格括号/附录 | 与该线 Phase E–J 事前注册的主量一致，且与 `constraint` 列 S3 的 late 窗口 t15–t20 同形 → Table 2 可横着读 |
| 6 | **Table 1 保留 10 列**，2-seed 的 `defense` / `gsm8k_sharded` 不删不移 | `global.md` 的 ≥3 seed 条款裁定为约束「以 seed 为复制单元的 `mean ± std`」，Table 1 不印这种数 |

**裁决 5 的理由不是"哪个数好看"——两个口径下结论方向一致**（Ours 对最优固定日程终点 −0.0688、
late −0.0250，都为负）。决定性的是三条：① late 是本线 Phase E–J 一路事前注册的主量，改终点等于
看过结果后换评分标准；② 终点是单轮（std 0.0568–0.1296），late 是三轮均值（std 0.0350–0.0553）；
③ **终点会把头条主张"提醒有用"从 +0.119 放大到 +0.244**，而 `constraint` 线同一件事测出来是
+0.1163——取 late 两条线对得上，取终点则只有 `defense` 一列翻倍。
**落地**：§二列头改 late；终点数同址以括号给出，不另起一列。

**裁决 6 的边界必须写死，否则外溢**：豁免**只**覆盖「误差棒来自题/攻击级 bootstrap、且 seed 在打分前
已被平均掉」这一种情形——seed 在 Table 1 里是降噪手段，不是复制单元，`defense` 30 攻击 /
`gsm8k_sharded` 60 题才是。**任何以 seed 为复制单元的 `mean ± std (n)` 一律不豁免**（Table 2 全表、
S1a、E0、S3 主量都属此类，n 仍须 ≥3）。
**不补 GPU**：第 3 个 seed 改变不了这两列的判定——`defense` 六行全贴 0 且最好的 null 是 `const`，
`gsm8k_sharded` 的瓶颈是 64 个评测行（nu=4+H=4 要求 ≥9 轮而轨迹多数更短），补 seed 只加三十几行。
**不删列的理由同样是硬的**：这两列恰好都是零结果，删零结果留好看的列就是选择性报告，
比 2-seed 本身严重。

---

**负面材料的边界（裁决 1 的推论）**：负*对照*与负面*诊断材料*进附录（§三）；**闭环打平本身留正文**，作为条件性结论的证据。

---

## 一、Table 1（正文 · RQ1）：算子质量，10 个数据集

**列 block = 数据集，block 内 2 列 = 指标；行 block 共用同一套 baseline，ours 垫底。**

| Panel | 列 block |
|---|---|
| **A** core 标量·样本足 | `sentence_length_t10` · `character_length_t5` · `formality_t5` |
| **B** core 小样本 + 多变量 | `average_word_length_t5` · `sentiment_t5` · `vector_count_stage1_t10` · `vector_count_stage2_t10` |
| **C** 多轮行为读出 | `defense`(judge=self ⚠) · `gsm8k_sharded`(确定性) · `constraint`(judge=indep 14B) |

**指标（每数据集 2 列）**
- **`Skill_H` ↑** = 1 − MSE_H(model) / MSE_H(最好平凡 null)。**`H` 与 `lag` 逐列标注**：
  每个数据集取「还能留 H≥3 的最深延迟嵌入」（core 的 T=10 用 lag=3 → H=6，T=5 用 lag=1 → H=3；
  行为线取 H=4 = MPC 规划视界）。`common_seed_turns=4` + `lag=3` 会让 T=5 只剩 **1 步**，
  而裁决 2 只解禁多步——这是判据，不是口味。
- **rollout MSE ↓**（读出原生单位，保留绝对尺度）。

**平凡 null（取最好者，角标注明谁赢）**——沿用 `defense` 线 G-K2-1 / `constraint` 线 G-S2-1
用的那三个，**不另写一份**：`const`（训练集均值）/ `turn_mean`（逐轮均值）/ `stateless`
（`y ~ 1 + 全部确定性外生量`）。**每个 null 都必须拿到 `turn`**，缺了就 raise。
外生量是环境事前固定的量（`turn`、core 的 `r`、`gsm8k_sharded` 的 `shard_frac`），
**不含控制量**——core 的控制是跟踪误差 `r − y_t`，它含状态，喂给"无状态 null"等于把它要扣住的东西漏回去。

**行 block（6 行，全 panel 共用）**

| 行 | 模型类 | 它否掉什么 |
|---|---|---|
| 1 | 最好平凡 null | 不看状态也能猜到的部分 |
| 2 | Markov 线性 + 控制（无记忆，`output_memory=1`） | 不需要记忆 |
| 3 | 延迟嵌入线性，**扣住控制量** | 执行器没进算子——**仅在动作被随机化过的列上有判别力**，见下 |
| 4 | LSTM 代理（隐层在 validation 上选） | 非线性循环模型更好 |
| 5 | AE-Koopman（非线性提升 k=16） | 非线性 lift 值不值 |
| **6** | **Ours：延迟嵌入受控 Koopman** | — |

**统计口径**：**每一行同一种区间**——点估计与 95% CI 都来自按 (轨迹/题) bootstrap（2000 次，
全部数据集同一套实现）；有训练 seed 的行（AE / LSTM）先把 3 个 seed 的逐行平方误差平均，
跨 seed 离散度**单独报**，不兼任误差棒。**「两行是否有差别」一律看配对 bootstrap**
（`ours − baseline`，三列联合重采样），不看两个逐格 CI 是否重叠。
core 的 n=3 是**训练** seed、行为线的 n=3 是**数据** seed，两者不可混读，脚注写死。

### E2 已测得，三条措辞据此改写（2026-09-12）

数与读法在 [`../experiments/core_surrogate_rows_results.md`](../experiments/core_surrogate_rows_results.md)。

1. **记忆有用是条件性的，不是普遍的。** core 7 个可判任务里 3 个显著支持（+0.603 / +0.312 / +0.235，
   都是 n≥42）、**1 个显著反对**（`average_word_length_t5` −0.409，n=14）、3 个分不开。
   正文写「在读出有量程、样本足的任务上成立」，**不写「惯性假设成立」**。
   与 `ABLATION_STUDY.md` 第五阶段「任务相关、无统一方向」一致。
2. **非线性买不到东西要用效应量讲。** `ours − AE`、`ours − LSTM` 在所有有信息的任务上都 **|Δ| < 0.03**
   且符号在任务间翻转；其中两格名义显著（含 `character_length_t5` 上 **LSTM 更好 −0.031**）。
   配对 bootstrap 精确到能把 0.005 判显著——**显著且可忽略**。
   写「差别小于 0.03 且方向不一致」，**不写「统计上不可区分」**（后者是错的）。
3. **第 3 行跨十列全部为零——E3 之后这条要改写成正面结论，不是 core 的局限。**
   > ⚠️ **2026-09-15：这条「要改写成正面结论」的前提已被证伪，措辞待裁决。**
   > 「正面结论」指的是拿 `tsar_cefr` 当阳性对照、把零说成条件性的。**该列第 3 行
   > `ours − 扣住 u` = +0.0061，CI95 [−0.0013,+0.0113]，含 0** ——第 3 行现为**十一列一致为零**，
   > 且这一列是三项条件（动作写入下一步输入 / 过冲有代价 / 动作真随机化）都满足的那一列。
   > 命名假设「第 3 行不为零 ⟺ 闭环为正」判据是**同号**，实测**同为零 → 未被否证，
   > 但非零侧仍未受检**。见
   > [`../experiments/tsar_cefr_surrogate_rows_results.md`](../experiments/tsar_cefr_surrogate_rows_results.md)
   > §命名假设的兑现。**本条正文怎么写、以及 `tsar_cefr` 线是否关闭，两者一并交还用户裁决。**
   `ours − 扣住控制量` 在 core 七个任务全部跨 0，在行为三线也全部跨 0
   （`constraint` +0.0015、`gsm8k_sharded` +0.057、`defense` +0.012），
   在 `tsar_cefr` 上同样跨 0（+0.0061）。
   **core 那套「`r` 恒定、不是随机化动作」的解释在 `constraint` 上不成立**——那里的 `u` 是伯努利随机化的真动作。
   与 S1a 的终点权威 +0.1389 并不矛盾：一步增益 `B`=+0.0182，滚 4 步累计约 0.05，小于读出自身波动。
   **正确表述：执行器对终点有权威，对规划视界内的读出没有预测力。**
   这正是 Table 2 三条线闭环全部打平的**机制解释**——算子在控制器规划的那个视界上没有动作依赖的结构，
   用它规划的控制器在原理上就赢不了固定日程。**Table 1 第 3 行与 Table 2 的打平是同一件事的两个观测面。**

**三列没有信息量，但仍进表**（否则等于按结果挑列）：`sentiment_t5`(10 条轨迹)、
`average_word_length_t5`(14)——ours 的 CI 分别是 [−0.11,+0.78]、[−1.52,+0.40]；
`even_odd_t5`(4，读出无量程)按用户裁定进附录 A1。前两列在表里如实标「本设计分辨不出来」。

## 二、Table 2（正文 · RQ3）：等代价下闭环买到了什么

> **四列的跨列读法见 [`CLOSED_LOOP_SYNTHESIS.md`](CLOSED_LOOP_SYNTHESIS.md)**（2026-09-15 新增）：
> 谁打平/压过 Ours、两类 baseline、四条独立证据为什么排除「baseline 太强」这个诊断、
> 效应量阶梯（动作本身 +0.12～0.18 对重排预算 0.00～0.01），以及由此派生的措辞红线。
> **本节只定义这张表怎么填；跨列结论不要在本节复述。**

**不作为"谁赢"的表读，作为三前提的分层证据读。** 列头额外标注该数据集上前提 (i)/(ii)/(iii) 的判定。

行（共用）：`zero_control` · `constant_remind`(满剂量) · 等代价随机 · **等代价最优固定日程** · 阈值反馈（无模型） · **Ours：Koopman-MPC（等代价）**
列（每数据集 3 个）：**主量读出 `mean ± std (n=seed)`**（`defense` = `late y`(t3–5)、`constraint` = late 窗口 t15–t20，**裁决 5**；终点数同址以括号给出） · vs 最优固定日程的配对差 95% CI · 代价（插入次数 / token）

`gsm8k_sharded` 的 MPC 格写"**构造上恒等于 `fixed_last`**"（日程可分性 = 1），不编数。

**`constraint` 列在 E0 之后改三处**（这是本线唯一能让闭环非退化的口子）：
1. **动作空间 ≥3**：`不提醒` / `整段提醒` / `定向提醒`。只有两个动作时 S2 已两次测出最优*日程*对所有人相同；
   定向的收益全部集中在 `n_violated=1`（+0.1489，对 `n=2` 的 +0.0018、`n=3` 的精确 +0.0000），
   **最优动作因此是状态的函数而非时刻的函数**——这正是固定日程模仿不了的依赖。
2. **代价轴改等 token**，不是等次数：定向块中位 21 token vs 整段 41（1.95×）。等次数下定向净 `y` 略劣，
   等 token 下不劣——**等 token 是它唯一站得住的卖点**。
3. **控制器不许恒选定向**：连带损失 −0.1059 > 收益 +0.0748，恒选定向在净 `y` 上输给整段。
   这不是坏消息——它意味着控制器**确实有事可决**，也是 Table 2 这一列的论点。

### Table 2 的伴随论证：自适应的上界（2026-09-13）

**Table 2 单独读会留一个审稿人形状的洞**：闭环打平既可以是"控制器不行"，也可以是"本来就没钱可赚"，
两种解释指向相反的下一步。[`../experiments/adaptivity_ceiling_results.md`](../experiments/adaptivity_ceiling_results.md)
把后者量出来了，零 GPU。

因为生成贪心定种子、提醒只在插入的那一轮进上下文，`fixed_t{k}` 在第 1..k−1 轮与 `zero_control`
**逐字节相同**（守卫实测 400/400）——所以任意策略都能在已落盘数据上精确回放。四个嵌套上界：

| 策略知道什么 | 对 `fixed_t5` 的增益 |
|---|---|
| Ours（运行时 `y` 历史） | −0.0688 [−0.1313, −0.0063] ★ |
| **因果 oracle：同样的信息 + 全部后见之明（所有读出可测策略的上确界）** | **+0.0063 [−0.0250, +0.0375]** |
| 按攻击 oracle（运行时不可观测） | +0.0250 [+0.0000, +0.0625] |
| 透视未来 oracle（松上界） | +0.1062 [+0.0500, +0.1625] ★ |

**机制**：决策时刻看不到东西——turn 1/2 的可观测前缀在 40 条轨迹上只有 **1 种**
（`y_1 ≡ 1.0`），到 turn 4–5 才分出 7/16 种，而那时动作集只剩 {t4, t5, 不花}。
**信息在该动手之后才到**，这与 Table 1 第 3 行是同一件事。
不是量程问题：`fixed_t5 − fixed_t2` = +0.2000 [+0.0750, +0.3375] ★。

**五个目标函数（终点 / 全轮均值 / t2–5 均值 / 最差单轮 / 累计失分 AUC）下因果 oracle 的 CI 全部跨 0**，
且最差单轮的最优固定日程换成 `fixed_t4`——所以不是"`fixed_t5` 恰好对上了终点"。

**正文可写**：闭环买不到东西不是因为控制器差，而是**可观测读出在该决策的时候几乎不携带信息**——
上界是量出来的，不是断言的。三条线合起来：`gsm8k_sharded` 可分性 1（构造上为 0）、
`constraint` 可分性 1（S3 实测 +0.0014）、`defense` 可分性 5 但因果上界 +0.0063（CI 跨 0）。

**这个方向已经关闭**（2026-09-13，闸门 R1，job 15833850）：三类读出的因果上界都量过了——`y_safety` 交叉验证 **−0.0271 ★**；七种文本/元数据信号**无一越过 0**；激活投影（layer 18 安全方向，零重叠）判据格 **−0.0354 ★**，全表最好 +0.0042（跨 0）。**关键不是读出太粗**：投影在每个决策点都能把 40 条轨迹完全区分开（turn 2 有 40 个不同取值，`y` 只有 1 个），上界仍然是 0。**分辨力不是瓶颈，可迁移的预测结构才是。** 正文据此写死：闭环买不到东西不是控制器的失败，是这个设定里没有可被读出携带、且能迁移到新攻击的动作相关结构。

> ✅ **~~待办 E5~~ 已补（2026-09-13，零 GPU）**：`late_t3_5` 已作为一个目标函数加进
> `analyze_adaptivity_ceiling.py`，伴随论证与主表现在同口径。主量下的四个上界：
> Ours **−0.0250 ★**、因果 oracle（in-sample）**+0.0063 [−0.0167, +0.0292]**（**仍跨 0**）、
> 因果交叉 −0.0271 ★、按攻击 +0.0375 ★、透视未来 +0.0896 ★。**结论与终点口径逐位一致。**
> 一处差别要写进 caption：按攻击 oracle 在主量下**显著**（终点口径下 CI 贴 0），
> 但那仍是运行时读不到的信息；读得到的那部分仍是 +0.0063、跨 0。
> 富信号筛查也已在主量下重跑，交叉验证最好的一格是提问长度的 **+0.0042（CI 跨 0）**，结论不变。

## 三、附录表（接收全部负对照与负面诊断材料）

| 表 | 内容 |
|---|---|
| A1 | `even_odd_t5` 负对照：读出无量程 → 算子正确地报"什么都没有" |
| A2 | 早停前后对照（`sentiment_t5` 78%→8%、`average_word_length_t5` 方向反转） |
| A3 | 读出天花板三连（defense 0.91 / gsm8k_sharded IQR=0 / constraint K1） |
| A4 | `gsm8k_sharded` 停在 S3 的全过程（可分性=1、熵读出 97% 衰减） |
| A5 | `constraint` 闭环退化诊断（两个模型类 × 两个目标函数都买不到东西） |
| A6 | 植入式对照单测（有状态依赖时可分性检查器确实报 >1 种日程） |
| A7 | 一步预测误差全表（裁决 2：正文不报） |
| A8 | 自适应上界的全表：四个 oracle × 五个目标函数 + 三条被实测关死的救援路线（交叉选 baseline / 等 token 前沿 / 逐攻击胜负） |
| A9 | **闸门 R1**：三类读出的因果上界（judge 分 / 七种文本信号 / 激活投影 × 2·3·4 箱 × 两条交叉轴），判 FAIL 关线；同址记「试点 n=16 把 in-sample 上界高估 11 倍」 |

## 四、单元格现状（Table 1）

| 行 | core（8 任务） | `defense` | `gsm8k_sharded` | `constraint` | `tsar_cefr` |
|---|---|---|---|---|---|
| 1–6（全部） | ✓ 8/8 | ✓ | ✓ | ✓ | ✓（2026-09-15） |

**十一列六行全满。** **Table 2 四列亦全满**：`defense` 六行见
[`../experiments/defense_table2_results.md`](../experiments/defense_table2_results.md)、
`constraint` 见 `../experiments/constraint_results.md` §S3、`gsm8k_sharded` 的 MPC 格按裁决
写"构造上恒等于 `fixed_last`"不编数、`tsar_cefr` 五臂见
[`../experiments/tsar_cefr_results.md`](../experiments/tsar_cefr_results.md) §Table 2（2026-09-14）。

**`tsar_cefr` 列（第十一列，2026-09-15）**：结果档案
[`../experiments/tsar_cefr_surrogate_rows_results.md`](../experiments/tsar_cefr_surrogate_rows_results.md)，
经 `persona_drift_control/scripts/eval_surrogate_rows_tsar_cefr.py`。**它是第三个调用方，不是第三套打分**
——`surrogate_eval` 同一份；单独写脚本的理由是这一列的动作是四值分类（3 自由度 one-hot），
行为侧脚本的数据路径只读一个标量动作列。**Table 2 的行在这一列不同**（本线五臂，非
`zero_control`/满剂量/等代价那套），按该线计划 §5.4；渲染正文表时这一列要单独排版，**待裁决**。

core 侧经 `scripts/eval_surrogate_rows.py`、行为侧经
`persona_drift_control/scripts/eval_surrogate_rows_behavioral.py`，**两侧共用同一个 `surrogate_eval` 打分路径**
（skill 定义、三个 null、bootstrap、折协议）；模型类各用本侧实现，两套代码体系不互相 import。
每侧六行都落在逐行相同的评测集上。core 的逐行预测存在 `results/surrogate_rows_paired/*_predictions.npz`。

**跨列可比的是 `Skill_H` 的定义与区间构造，不是拟合器**——这句话要进表注。

**两列指标现已全部登记（2026-09-17，零 GPU，只读产物）。** §一 规定每个数据集 2 列
（`Skill_H` 相对 + rollout MSE 绝对），但此前只有 `tsar_cefr` 列的档案印了绝对值，
core 八列与行为三列**只登记了相对值**。绝对值已从既有产物的 `rows.*.rollout_mse` 补进
[`../experiments/core_surrogate_rows_results.md`](../experiments/core_surrogate_rows_results.md) §绝对值
与 [`../experiments/behavioral_surrogate_rows_results.md`](../experiments/behavioral_surrogate_rows_results.md) §绝对值
——**同一批产物、未重拟合**，`Skill_H = 1 − MSE/MSE_null` 在列内代数等价，绝对列买的是尺度不是新证据。
**绝对列的三条边界**：① **跨列不可比**（读出尺度不同，禁求平均/排名/取最好，同 §六第 2 条）；
② **绝对列没有区间**（bootstrap 建在 `Skill_H` 上），"两行有没有差别"一律仍看配对 bootstrap；
③ `even_odd_t5` 的 `Skill_H` 无定义而绝对值存在（null 6.9e-32、线性行 ~1e-15、非线性行 ~1e-6），
**附录 A1 的实际内容是这三个数量级**。

**跨 seed 离散度已登记（2026-09-17，零 GPU，只读产物）。** §一 规定有训练 seed 的行「跨 seed 离散度**单独报**、
不兼任误差棒」，此前只在产物里（`rows.*.seed_spread_ddof{0,1}`），文档只给了指针。ddof=1（§六第 5 条已签）
的十四格数已印进 [`../experiments/core_surrogate_rows_results.md`](../experiments/core_surrogate_rows_results.md)
§跨 seed 离散度。**这个量只覆盖 core 七列 × AE / LSTM 两行**——`Ours` / `Markov` / `延嵌无 r` 是闭式 ridge，
无随机种子，离散度构造上为 0；`even_odd_t5` 的逐 seed `skill_h` 因 `DegenerateNullError` 全为 `None`，无定义；
**行为三线与 `tsar_cefr` 的任何行都没有这个量**，那边 3 个 seed 是拟合前就池化的**数据** seed（与裁决 6 一致）。
**登记买的不是新证据**：`mean` 那一半与行点估计逐位相同（残差 ≤2.2e-16，`Skill_H` 对 MSE 线性），
新信息只有 `± std`；且它**不得印成 Table 1 的区间**（区间一律按轨迹 bootstrap，混印过一次，
见结果档案「两个 bug」第 2 条）。

**缺的格子全是 CPU 重拟合，数据已落盘**（`datasets/`、`outputs/ergo_ekA_branch` 759 对、`outputs/sequor_s1_arm` 9600 行）——**Table 1 零 GPU 可填满。**

## 五、填表清单

| # | 做什么 | 成本（含假设） | 服务哪张表 |
|---|---|---|---|
| ~~E0~~ | ~~判定定向提醒试点闸门~~ **已完成 2026-09-12（`79a08c9`）：主量 PASS** | 零 GPU | Table 2 `constraint` 列成立 |
| ~~E1~~ | ~~统一代理评测 harness~~ **已完成 2026-09-12**：`src/surrogate_eval/`（用户裁定放 root，不依赖任一体系），17 条单测 + 行为侧 6 条等价性测试 | 零 GPU | Table 1 地基 |

**E1 落地说明**：`skill_h`（`horizon<2` 直接 raise，一步误差进不来）、`trivial_nulls`（`exogenous` 不含 `turn_next` 即 raise）、`best_null`（返回名字，谁赢本身是诊断）、`bootstrap_ci`（按组重采样，行重采样会低估区间）、`seed_aggregate`（`ddof` 无默认值，<3 seed 直接 raise）、`make_folds`（`purpose="gate"|"report"` 派生不同划分，两者重合即 raise）。**三个 null 没有写第四份拷贝**——`fit_koopman_defense_model._null_predictions` 是 G-K2-1/G-S2-1 用的那份，`surrogate_eval` 取同一套算术，行为侧 `test_surrogate_eval_null_equivalence.py` 逐值钉死（`atol=0`），漂移即红。
| ~~E2~~ | ~~core 8 补第 1/2/3/4 行~~ **已完成 2026-09-12**：`scripts/eval_surrogate_rows.py`，10 条单测；结果档案 [`../experiments/core_surrogate_rows_results.md`](../experiments/core_surrogate_rows_results.md) | 零 GPU | Table 1 core 侧**六行全满** |

**E2 改变了 Table 1 的三条措辞（待落到 §一）**：① 记忆有用是**条件性**的（3/7 显著支持、1/7 显著反对、3/7 分不开）；② 非线性无用要用**效应量**讲（|Δ| < 0.03 且符号翻转），不能讲「统计上不可区分」；③ **第 3 行「执行器进算子」core 一格都撑不住**（`ours − 无 r` 七个任务全部跨 0）——core 的 `r` 每条轨迹恒定、不是随机化动作，这条主张只能由行为线扛（当时三条，2026-09-15 起四条；**四条的第 3 行全部跨 0**）。**每个数据集的 lag 按「还能留 H≥3 的最深延迟嵌入」定**（T=10 用 lag=3、T=5 用 lag=1），表里逐列标注。
| ~~E3~~ | ~~行为 3 线补行~~ **已完成 2026-09-12**：`persona_drift_control/scripts/eval_surrogate_rows_behavioral.py`；结果档案 [`../experiments/behavioral_surrogate_rows_results.md`](../experiments/behavioral_surrogate_rows_results.md) | 零 GPU | Table 1 **十列全满** |

**E3 不是重新制表，是新测量**：三条线签过的闸门全部是**一步、`nu=1`**（`constraint` 的 S2 配置字面是 `{nu:1, mu:1}`），主表第 6 行从未在任何一条线上被拟合过；第 2 行才对应各线已发表的算子。
**E3 的三条结果**：① `constraint` 上 `ours − Markov` **+0.234 ★**，至此**四个信息量足够的列全部显著支持延迟嵌入**；② **`ours − 扣住 u` 跨线全部为零**，且 `constraint` 的 `u` 是伯努利随机化的真动作——core 那套「`r` 恒定」的解释在这里不成立；③ 非线性差别 ≤0.05、两个方向都显著，与 core 同形。
**两列无信息量**：`gsm8k_sharded` 只剩 64 个评测行（nu=4+H=4 要求 ≥9 轮）、`defense` 六行贴 0 且最好 null 是 `const`（读出无量程，与 `even_odd_t5` 同机制）。均如实进表标注，不挑列。
| ~~G1~~ | ~~`defense` 两个端点臂扩到 5 seed~~ **已完成 2026-09-13**：15829527 / 15829528，各 27 min，两臂各 200 行、判分失败 0、拒答 0 | 2 GPU 作业（已花） | Table 2 `defense` 列补洞 |
| ~~E4~~ | ~~`defense` Phase J 七臂按 seed 聚合~~ **已完成 2026-09-13**：`persona_drift_control/scripts/aggregate_table2_defense.py`，12 条单测；结果档案 [`../experiments/defense_table2_results.md`](../experiments/defense_table2_results.md) | 零 GPU | Table 2 `defense` 列**六行全满** |
| ~~G3~~ | ~~`constraint` S3 闭环臂~~ **已完成 2026-09-12**：砍成**三臂**执行（`8c75c03`），15815719 + 15815720 跑完并判读，判成**干净负结果**（`c094de0`）。主量 `koopman_mpc − best_fixed_schedule` = **+0.0014 ± 0.0145 (n=3)**，0.05×MDE；同批行里提醒本身买到 +0.1163。详见 [`../experiments/constraint_results.md`](../experiments/constraint_results.md) §S3 | 已花 ≈ 7.2 GPU-h | Table 2 `constraint` 列 |
| ~~G4~~ | ~~`tsar_cefr` 闭环五臂~~ **已完成 2026-09-14**：15944416（11m06）+ 锚趟 15943453（8m06）。签死的主对比 `koopman_mpc − dp_path` **−0.0730** [−0.1396,−0.0081] 过了，**但 `dp_degeneracy=1.0`**（199 题 DP 路径全等于相邻阶梯）→ 那个差主要是 prompt 家族不是控制；同家族内对 `fixed_ladder` −0.0073（含 0）、对手写 `greedy_reactive` **+0.0096**（含 0，符号不利），等代价 **3.37 步/322 token 对 1.03 步/100 token** → **被开环阶梯支配**。详见 [`../experiments/tsar_cefr_results.md`](../experiments/tsar_cefr_results.md) §Table 2 | 已花 ≈ 0.3 GPU-h | Table 2 **第四列** |
| ~~E6~~ | ~~`tsar_cefr` 补 Table 1 六行~~ **已完成 2026-09-15**：`persona_drift_control/scripts/eval_surrogate_rows_tsar_cefr.py`，22 条单测（连 `lstm_baseline` 的 `v_dim` 改动）；结果档案 [`../experiments/tsar_cefr_surrogate_rows_results.md`](../experiments/tsar_cefr_surrogate_rows_results.md) | 零 GPU | Table 1 **十一列全满** |

**E6 的三条结果**：① **预注册预测落空**——该线计划 §7.1 预注册「本列第 3 行显著不为零」，
实测 **+0.0061，CI 含 0**；第 3 行现为**十一列一致为零**，且这一列三项条件都满足。
**这是本线从「锦上添花」升级为承重的唯一理由，它没有兑现**（§一 第 3 条已加注）。
② `ours − Markov` **+0.0487 ★**，至此**五个信息量足够的列全部显著支持延迟嵌入**。
③ LSTM `Skill_H` **−0.101**、AE **−0.032**，两者都没越过最好的平凡 null（本列是 `stateless`）——**这是样本量不足
（597 轨迹 × 6 步），不得作为「非线性无用」的证据**；本列与其余十列「差别 <0.03 且方向不一致」
那句措辞**不同源**，正文不要合并。
**G4 与 E6 是同一个缺陷的两个读数**：第 3 行说「动作进不了算子」，锚趟说「算子把
`half_step_down`(−0.0696) 排在 `step_down`(−0.0665) 之前、MPC 一次 `step_down` 都不选」。


**E4 / G1 的三条结果**：① **12 天 + 一次重构，seed 0/1 的 160 行逐字节相同**——G1 选择整 5 seed 重跑
（而非续跑）换来的这次比对证明重构行为等价且环境没动，本线跨 phase 产物可以并排读；
② **满剂量买得到东西，等代价重分配买不到**：主量 late 从 `zero_control` 0.6687 抬到满剂量 0.7875（**+0.119**），
而六个等代价臂全挤在 0.6562–0.7312（终点口径 0.5500 → 0.7937 = +0.244，等代价臂 0.5375–0.7375）；
③ **Ours 对最优固定日程两个重采样单元下点估计都为负**（主量 late **−0.0250**、终点口径 −0.0688），按攻击重采样 CI 排除 0、按轨迹跨 0——**方向一致，显著性取决于单元**。
正文能不看单元就说死的只有一句：**Koopman-MPC 没有打赢最优固定日程，点估计为负。**

**两条均已签**：① ~~**Table 2 的主量是终点还是 late `y`**~~
**已签 2026-09-13（裁决 5）：主量 = `late y`(t3–5)**，终点降为同址括号并必须标「终点口径」。
② ~~**Table 2 配对差的重采样单元**~~ **已签 2026-09-16（用户裁决）：主口径 = 按攻击**。

**签的理由是它的来历，不是它的显著性**：`item_col=attack_id` 继承 Table 1 `defense` 列的
既有实现，**早于任何对比被算出**——不是在两个口径里挑出来的。
两个口径**点估计都为负**（主量 late −0.0250、终点口径 −0.0688），
**按攻击 CI 排除 0、按轨迹跨 0**。

**正文只写两个口径都撑得住的那句**：
> **Koopman-MPC 没有打赢最优固定日程，点估计为负；方向在两个重采样单元下一致，
> 显著性取决于单元。**

**显著性那半句降到 caption**，且 caption **必须同址带齐三样**：按轨迹的结果、
「只有 **8 个**攻击对聚类、bootstrap 偏少」、以及"同一攻击下的轨迹并不独立"。
❌ **只报按攻击、不提按轨迹**属于在两个口径里挑好看的那个，触 `.claude/global.md` 红线。

| ~~E5~~ | ~~伴随论证补算主量 t3–5 一行~~ **已完成 2026-09-13**：`late_t3_5` 进
`analyze_adaptivity_ceiling.py` 的 `OBJECTIVES`，产物
`persona_drift_control/outputs/koopman_case_study/adaptivity_ceiling_late_t3_5.json`，
结果落 §二 的 ✅ 注与 [`../experiments/adaptivity_ceiling_results.md`](../experiments/adaptivity_ceiling_results.md) §主量 | 零 GPU | Table 2 伴随论证与主表同口径 |

**本行 2026-09-15 曾以"待办"存在于本节、而 §二 同时印着"已补"——同一份文档两处互相矛盾，
与 09-15 清掉的那批"三代前的活计划"是同一种漂移。** 2026-09-16 按产物核实后合并：
回放 `--objective late_t3_5` 到新路径与已落盘产物逐字段比对，**102 个字段 0 处不同**，
§二 引的五个上界逐值对得上（Ours −0.0250、因果 oracle in-sample +0.0063 [−0.0167, +0.0292]、
留一交叉 −0.0271、按攻击 +0.0375、透视未来 +0.0896）。
⚠️ 同址记一个缺口：该产物**没有 harness 指纹**（无 `provenance` 字段，与
`.claude/global.md` → *产物与谱系* 不符）。本次**不重跑覆盖**；下次动这条链时补，
补的方式是写到新路径、不改已落盘文件。

## 六、同址必带的局限（写表时逐条落到 caption）

1. `defense` 列 `judge_kind=self`（`global.md` 具名例外）：自判 = 单向漏检、systematically 低估效应。**例外不外溢到任何其它列。**
2. 表内不设任何跨数据集聚合行（禁"平均排名"等序数聚合）。
3. `constraint` 的 `Skill_H` 曾用作 S3 准入闸门 → 正文格用**另一套事前注册的折划分**重算，闸门折的数进 A7。
4. core 的误差棒只含训练 seed，不含数据采样；行为线含数据 seed。
5. **`mean ± std` 的 `std` 一律用样本标准差 `ddof=1`——✅ 2026-09-13 用户签字。**
   `ABLATION_STUDY.md` 的 core 结果用的是**总体标准差**（ddof=0），`constraint` 线（S1a +0.1389 ± 0.0147、
   E0 +0.0748 ± 0.0409）用的是样本标准差（ddof=1）。两者恒差 √(n/(n−1))，**不改变任何排序**，
   也**不改变任何判定**——本仓库所有 ★ / PASS / UNDECIDABLE 都来自 bootstrap CI 与 MDE，
   跨 seed 的 `std` 只进显示字符串（判据见 `analyze_sequor_s3_gates.py` 的 `resolved`）。
   **签 ddof=1 的决定性理由是这张表混着 seed 数**：`defense` 列 n=5、`constraint` 列 n=3，
   而 ddof=0 把一列收缩 √((n−1)/n)——**n=3 收缩 18.4%、n=5 只收缩 10.6%**，
   seed 更少的那一列反而印出更紧的误差棒，正好把误差棒该传达的信息拧反。ddof=1 没有随 n 变的因子。
   （§六第 5 条此前写的「n=3 时恒差 1.2247 倍」在 G1 把 `defense` 补到 5 seed 之后已不成立。）
   **落地**：`surrogate_eval.seed_aggregate` 的 `ddof` 仍无默认值，没有人能默默选一个；
   正文表重算 core 的 ±（`results/*/run.json` 都在，零成本）；**历史文档不改**
   （`docs.md` → 历史陈述不改），脚注写明与 `ABLATION_STUDY.md` 的差异来源（√1.5 = 1.2247 倍）。
6. 定向提醒的 +0.0748 伴随 **−0.1059 的连带损失**，任何引用它的地方必须同址带这个数。
7. `constraint` 的算子**外推不出终点**（稳态位移 0.050 对实测 +0.1389，差 2.8×）——Table 1 的格子是辨识质量，不是终点预测力。
8. **`defense` 列主量 = `late y`(t3–5)（裁决 5）**：任何引用终点数的地方必须同址标「终点口径」，两处不得各取一个——终点把 `constant_remind − zero_control` 从 +0.119 放大到 +0.244。
9. **`defense` / `gsm8k_sharded` 在 Table 1 只有 2 个数据 seed（裁决 6）**：caption 必须同址写`n_seed=2`、重采样单位（30 攻击 / 60 题）、以及本列判读为**「本设计分辨不出来」**。豁免不外溢到任何以 seed 为复制单元的数字。
