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
   `ours − 扣住控制量` 在 core 七个任务全部跨 0，在行为三线也全部跨 0
   （`constraint` +0.0015、`gsm8k_sharded` +0.057、`defense` +0.012）。
   **core 那套「`r` 恒定、不是随机化动作」的解释在 `constraint` 上不成立**——那里的 `u` 是伯努利随机化的真动作。
   与 S1a 的终点权威 +0.1389 并不矛盾：一步增益 `B`=+0.0182，滚 4 步累计约 0.05，小于读出自身波动。
   **正确表述：执行器对终点有权威，对规划视界内的读出没有预测力。**
   这正是 Table 2 三条线闭环全部打平的**机制解释**——算子在控制器规划的那个视界上没有动作依赖的结构，
   用它规划的控制器在原理上就赢不了固定日程。**Table 1 第 3 行与 Table 2 的打平是同一件事的两个观测面。**

**三列没有信息量，但仍进表**（否则等于按结果挑列）：`sentiment_t5`(10 条轨迹)、
`average_word_length_t5`(14)——ours 的 CI 分别是 [−0.11,+0.78]、[−1.52,+0.40]；
`even_odd_t5`(4，读出无量程)按用户裁定进附录 A1。前两列在表里如实标「本设计分辨不出来」。

## 二、Table 2（正文 · RQ3）：等代价下闭环买到了什么

**不作为"谁赢"的表读，作为三前提的分层证据读。** 列头额外标注该数据集上前提 (i)/(ii)/(iii) 的判定。

行（共用）：`zero_control` · `constant_remind`(满剂量) · 等代价随机 · **等代价最优固定日程** · 阈值反馈（无模型） · **Ours：Koopman-MPC（等代价）**
列（每数据集 3 个）：终点读出 `mean ± std (n=seed)` · vs 最优固定日程的配对差 95% CI · 代价（插入次数 / token）

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

**唯一没关死的方向**：上界只约束按 `y` 读出可测的策略；读文本 / 激活的控制器不在它的约束内。

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

## 四、单元格现状（Table 1）

| 行 | core（8 任务） | `defense` | `gsm8k_sharded` | `constraint` |
|---|---|---|---|---|
| 1–6（全部） | ✓ 8/8 | ✓ | ✓ | ✓ |

**十列六行全满。** **Table 2 三列亦全满**（2026-09-13）：`defense` 六行见
[`../experiments/defense_table2_results.md`](../experiments/defense_table2_results.md)、
`constraint` 见 `../experiments/constraint_results.md` §S3、`gsm8k_sharded` 的 MPC 格按裁决
写"构造上恒等于 `fixed_last`"不编数。

core 侧经 `scripts/eval_surrogate_rows.py`、行为侧经
`persona_drift_control/scripts/eval_surrogate_rows_behavioral.py`，**两侧共用同一个 `surrogate_eval` 打分路径**
（skill 定义、三个 null、bootstrap、折协议）；模型类各用本侧实现，两套代码体系不互相 import。
每侧六行都落在逐行相同的评测集上。core 的逐行预测存在 `results/surrogate_rows_paired/*_predictions.npz`。

**跨列可比的是 `Skill_H` 的定义与区间构造，不是拟合器**——这句话要进表注。

**缺的格子全是 CPU 重拟合，数据已落盘**（`datasets/`、`outputs/ergo_ekA_branch` 759 对、`outputs/sequor_s1_arm` 9600 行）——**Table 1 零 GPU 可填满。**

## 五、填表清单

| # | 做什么 | 成本（含假设） | 服务哪张表 |
|---|---|---|---|
| ~~E0~~ | ~~判定定向提醒试点闸门~~ **已完成 2026-09-12（`79a08c9`）：主量 PASS** | 零 GPU | Table 2 `constraint` 列成立 |
| ~~E1~~ | ~~统一代理评测 harness~~ **已完成 2026-09-12**：`src/surrogate_eval/`（用户裁定放 root，不依赖任一体系），17 条单测 + 行为侧 6 条等价性测试 | 零 GPU | Table 1 地基 |

**E1 落地说明**：`skill_h`（`horizon<2` 直接 raise，一步误差进不来）、`trivial_nulls`（`exogenous` 不含 `turn_next` 即 raise）、`best_null`（返回名字，谁赢本身是诊断）、`bootstrap_ci`（按组重采样，行重采样会低估区间）、`seed_aggregate`（`ddof` 无默认值，<3 seed 直接 raise）、`make_folds`（`purpose="gate"|"report"` 派生不同划分，两者重合即 raise）。**三个 null 没有写第四份拷贝**——`fit_koopman_defense_model._null_predictions` 是 G-K2-1/G-S2-1 用的那份，`surrogate_eval` 取同一套算术，行为侧 `test_surrogate_eval_null_equivalence.py` 逐值钉死（`atol=0`），漂移即红。
| ~~E2~~ | ~~core 8 补第 1/2/3/4 行~~ **已完成 2026-09-12**：`scripts/eval_surrogate_rows.py`，10 条单测；结果档案 [`../experiments/core_surrogate_rows_results.md`](../experiments/core_surrogate_rows_results.md) | 零 GPU | Table 1 core 侧**六行全满** |

**E2 改变了 Table 1 的三条措辞（待落到 §一）**：① 记忆有用是**条件性**的（3/7 显著支持、1/7 显著反对、3/7 分不开）；② 非线性无用要用**效应量**讲（|Δ| < 0.03 且符号翻转），不能讲「统计上不可区分」；③ **第 3 行「执行器进算子」core 一格都撑不住**（`ours − 无 r` 七个任务全部跨 0）——core 的 `r` 每条轨迹恒定、不是随机化动作，这条主张只能由三条行为线扛。**每个数据集的 lag 按「还能留 H≥3 的最深延迟嵌入」定**（T=10 用 lag=3、T=5 用 lag=1），表里逐列标注。
| ~~E3~~ | ~~行为 3 线补行~~ **已完成 2026-09-12**：`persona_drift_control/scripts/eval_surrogate_rows_behavioral.py`；结果档案 [`../experiments/behavioral_surrogate_rows_results.md`](../experiments/behavioral_surrogate_rows_results.md) | 零 GPU | Table 1 **十列全满** |

**E3 不是重新制表，是新测量**：三条线签过的闸门全部是**一步、`nu=1`**（`constraint` 的 S2 配置字面是 `{nu:1, mu:1}`），主表第 6 行从未在任何一条线上被拟合过；第 2 行才对应各线已发表的算子。
**E3 的三条结果**：① `constraint` 上 `ours − Markov` **+0.234 ★**，至此**四个信息量足够的列全部显著支持延迟嵌入**；② **`ours − 扣住 u` 跨线全部为零**，且 `constraint` 的 `u` 是伯努利随机化的真动作——core 那套「`r` 恒定」的解释在这里不成立；③ 非线性差别 ≤0.05、两个方向都显著，与 core 同形。
**两列无信息量**：`gsm8k_sharded` 只剩 64 个评测行（nu=4+H=4 要求 ≥9 轮）、`defense` 六行贴 0 且最好 null 是 `const`（读出无量程，与 `even_odd_t5` 同机制）。均如实进表标注，不挑列。
| ~~G1~~ | ~~`defense` 两个端点臂扩到 5 seed~~ **已完成 2026-09-13**：15829527 / 15829528，各 27 min，两臂各 200 行、判分失败 0、拒答 0 | 2 GPU 作业（已花） | Table 2 `defense` 列补洞 |
| ~~E4~~ | ~~`defense` Phase J 七臂按 seed 聚合~~ **已完成 2026-09-13**：`persona_drift_control/scripts/aggregate_table2_defense.py`，12 条单测；结果档案 [`../experiments/defense_table2_results.md`](../experiments/defense_table2_results.md) | 零 GPU | Table 2 `defense` 列**六行全满** |
| ~~G3~~ | ~~`constraint` S3 闭环臂~~ **已完成 2026-09-12**：砍成**三臂**执行（`8c75c03`），15815719 + 15815720 跑完并判读，判成**干净负结果**（`c094de0`）。主量 `koopman_mpc − best_fixed_schedule` = **+0.0014 ± 0.0145 (n=3)**，0.05×MDE；同批行里提醒本身买到 +0.1163。详见 [`../experiments/constraint_results.md`](../experiments/constraint_results.md) §S3 | 已花 ≈ 7.2 GPU-h | Table 2 `constraint` 列 |

**E4 / G1 的三条结果**：① **12 天 + 一次重构，seed 0/1 的 160 行逐字节相同**——G1 选择整 5 seed 重跑
（而非续跑）换来的这次比对证明重构行为等价且环境没动，本线跨 phase 产物可以并排读；
② **满剂量买得到东西，等代价重分配买不到**：终点从 `zero_control` 0.5500 抬到满剂量 0.7937（+0.244），
而五个等代价臂全挤在 0.5375–0.7375；③ **Ours 对最优固定日程两个重采样单元下点估计都为负**
（终点 −0.0688、late −0.0250），按攻击重采样 CI 排除 0、按轨迹跨 0——**方向一致，显著性取决于单元**。
正文能不看单元就说死的只有一句：**Koopman-MPC 没有打赢最优固定日程，点估计为负。**

**两条待签（渲染正文表之前）**：① **Table 2 的主量是终点还是 late `y`**——§二写的是"终点读出"，
而 `defense` 线 Phase E–J 一路的预注册主量是 late `y`(t3–5)；两个都已算出且结论方向一致，
但不要两处各取一个。② **Table 2 配对差的重采样单元**——本次按攻击（继承 Table 1 `defense` 列
`item_col=attack_id` 的既有实现，早于任何对比被算出），Phase J 记录按轨迹；
**8 个攻击对聚类 bootstrap 偏少**，这是"Ours 显著为负"那句话唯一的软肋。

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
