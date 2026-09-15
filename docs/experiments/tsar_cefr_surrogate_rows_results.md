# `tsar_cefr`：主表六行的结果档案（Table 1 第十一列）

> **状态（2026-09-15）**：结果档案。core 侧档案见
> [`core_surrogate_rows_results.md`](core_surrogate_rows_results.md)、行为三线见
> [`behavioral_surrogate_rows_results.md`](behavioral_surrogate_rows_results.md)，
> 设计见 [`../article/MAIN_TABLE_DESIGN.md`](../article/MAIN_TABLE_DESIGN.md)。
> 本线的其余结果（G-T1 / G-S1 / G-S2 / D-2 / D-2.5 / Table 2 五臂）在
> [`tsar_cefr_results.md`](tsar_cefr_results.md)。
> 术语 [`../NAMING.md`](../NAMING.md)；口径 `../../.claude/global.md` → *报告口径*。
> **本文件不改任何既有实验文档的结论段。**

## 这一列买的是什么

计划 [§7.1 第 1 条](../operational_plan_tsar_cefr_line_2026-09-13.md) 预注册了一个预测：
**本列第 3 行（`ours − 扣住 u`）显著不为零，与三条行为线相反。** 这条线从「锦上添花」升级为
**承重**，靠的就是它——它是「第 3 行不为零 ⟺ 闭环为正」这条命名假设在**非零侧的唯一检验**
（[`tsar_cefr_kill_criterion.md`](tsar_cefr_kill_criterion.md) §Named assumption）。

**结果：预测落空。第 3 行 `+0.0061`，CI 含 0。** 第 3 行现在是**十一列一致为零**。

## 产物与谱系

| | |
|---|---|
| 脚本 | `persona_drift_control/scripts/eval_surrogate_rows_tsar_cefr.py`（新增，CPU） |
| 打分 | `src/surrogate_eval/`（与 core 侧、行为侧同一套） |
| 状态与动作构造 | 逐字 import 本线的 `scripts/fit_tsar_cefr_operator.py`，不重打 |
| 输入行 | `persona_drift_control/outputs/tsar_cefr_gpu1/trajectories.jsonl`（GPU-1 开环激励臂，只读） |
| 产物 | `persona_drift_control/outputs/surrogate_rows_tsar_cefr/tsar_cefr.json`，含 `provenance` |
| 谱系 | sha `6b16d40`、`git_dirty=true`（4 条脏路径 = 本脚本与它的单测 + `lstm_baseline` 的 `v_dim` 改动），交互节点 `node1642` |
| 单测 | **22 passed**（`test_eval_surrogate_rows_tsar_cefr.py` + `test_lstm_baseline.py`，4m46s） |

**为什么另写一个脚本，不在 `eval_surrogate_rows_behavioral.py` 里加一行 `LINES`**：那个脚本的
数据路径走 `modeling.dataset.build_reduced_state_pairs`，它只读**一个标量动作列**。本线的动作是
四值分类，按计划 §5.1 载成 **3 自由度 one-hot**（`copy` 为参考类），与 Phase 3 算子拟合时签的
编码同一份。把它压成标量正是这一列**不能**做的事——第 3 行量的就是「扣住动作损失多少」，
四个动作压成一个数会把第 3 行**朝零的方向**缩，也就是朝与其余十列一致、朝让论文主张好看的方向缩。

`MAIN_TABLE_DESIGN.md` §四的跨列契约照旧：**跨列可比的是 `Skill_H` 的定义与区间构造，不是拟合器**
——打分、三个 null、bootstrap、折协议全部走同一个 `surrogate_eval`。

## 设定

| 项 | 值 | 依据 |
|---|---|---|
| 状态 $\xi_t$ | $[\ell_t, s_t, \ell_{t-1}, s_{t-1}]$，lag=1 | **就是 MPC 实际跑的那个算子**，含它的两个判断题（$\ell_0$ = `source_level_expected`、$s_0$ = 1.0） |
| 打分读出 | $\ell$（被控变量，CEFR 期望级） | G-S2-1 是 $[\ell, s]$ 联合，Table 1 的 `Skill_H` 定义在单读出上 |
| 视界 H | 4 | MPC 规划视界（计划 §5.4），同其余行为列 |
| 外生量 | `turn_next`、`target_level_rank`、三个动作 dummy | `turn_next` 是 `trivial_nulls` 必需；`target_level_rank` 是本线的环境事先固定量（对应 core 的 `r`、`gsm8k_sharded` 的 `shard_frac`） |
| 题 / 轨迹 / seed | 100 源 / 597 轨迹 / 3 | GPU-1 臂 |
| 评测行 | 4776 | T=6、lag=1 下每条轨迹留 2 个 H=4 窗口 |
| 折 | 5，`purpose="report"` | 与闸门折不同 |
| 剔除 | `51-a2` | G-S1 具名剔题，全线同一处置 |
| 未解析读出 | 0 条轨迹被剔 | — |

**读出是固定 CEFR 分类器，与被控模型无关**，所以 *报告口径* 的「自判不得报告」不适用于本列。
区间是**按源文本 bootstrap（100 组）**，与其余十列同法；`mean ± std (n seeds)` 是 Table 2
跨臂比较的口径，不是 Table 1 的。

## 结果（`Skill_H`，按题 bootstrap 95% CI）

| 行 | 模型 | rollout MSE | `Skill_H` | CI95 |
|---|---|---|---|---|
| 1 | 最好 null（`stateless`） | 0.2078 | 0 | — |
| 2 | Markov 线性 + 控制 | 0.1001 | +0.518 | [+0.438, +0.614] |
| 3 | 延迟嵌入线性，**扣住 u** | 0.0912 | +0.561 | [+0.445, +0.691] |
| 4 | LSTM | 0.2286 | **−0.101** | [−0.301, +0.002] |
| 5 | AE-Koopman | 0.2144 | **−0.032** | [−0.073, +0.021] |
| 6 | **Ours（延迟嵌入 + 控制）** | **0.0899** | **+0.567** | [+0.455, +0.695] |

### 配对 bootstrap（`ours − baseline`，★ = CI 排除 0）

| 对比 | 值 | CI95 |
|---|---|---|
| vs Markov | **+0.0487** ★ | [+0.0069, +0.0971] |
| **vs 扣住 u** | **+0.0061** | **[−0.0013, +0.0113]** |
| vs LSTM | **+0.6676** ★ | [+0.4986, +0.9139] |
| vs AE | **+0.5992** ★ | [+0.4780, +0.7411] |

LSTM 隐层由验证集选，5 折全部选到 `hidden=8`（val MSE 0.92–1.00）。

## 三条读法

**1. 延迟嵌入有用，量级与四个信息量足够的列同阶。** `ours − Markov` **+0.0487 ★**。
至此**五个信息量足够的列全部显著支持延迟嵌入**（core 三列 +0.603 / +0.312 / +0.235、
`constraint` +0.234、本列 +0.0487）。本列的量级最小，机制在同址可查：$\xi$ 只有 4 维、
每条轨迹 6 步，lag 块买到的是滚动误差 0.1001 → 0.0899（**1.11×**）。

**2. 「扣住控制量不花钱」在最该失效的那一列上也成立。** `ours − 扣住 u` = **+0.0061**，CI 含 0。
这一列被设计成阳性对照，因为它三项条件都满足：动作直接写入下一步输入、过冲有代价、
**动作是真随机化的四值分类**（不是 core 那种每条轨迹恒定的 `r`，也不是伯努利二值）。
**执行器权威本身不缺**——D-2 的无模型配对差 `Δℓ(step_down) − Δℓ(copy)` = **−0.1356 ± 0.0099
(n=3 seeds)**，3.78× 本臂 MDE；算子的 `step_down` 增益 −0.0665，CI 排除 0（G-S2-2 过）。
**权威在一步上测得到，在 4 步滚动预测上买不到东西**——与 `constraint` 上「S1a 终点权威 +0.1389
对 `B`=+0.0182 滚 4 步约 0.05」是同一个形状。

**3. 非线性买不到东西，与 core、与行为三线同形。** LSTM `Skill_H` **−0.101**、AE **−0.032**，
两者都没越过最好的平凡 null（本列是 `stateless`）；`ours − LSTM` +0.668 ★、`ours − AE` +0.599 ★。
本列的措辞与其余十列不同：那里是「差别 <0.03 且方向不一致」，**这里两个非线性替身直接输给了
线性算子**，量级 0.6 级。原因写在同址：597 条轨迹 × 6 步的样本量喂不动这两类模型
（core 的同类现象出现在 n≤14 的列上）。**这不构成「非线性无用」的新证据，只是本列样本不足。**

## 它与 Table 2 的关系（同一件事的两个观测面）

Table 2 本列（[`tsar_cefr_results.md`](tsar_cefr_results.md) §Table 2）：`koopman_mpc` 0.7840
± 0.0074，对 `fixed_ladder` **−0.0073**（CI 含 0）、对手写 `greedy_reactive` **+0.0096**
（CI 含 0，符号不利），等代价轴上 **3.37 步 / 322 token 对 1.03 步 / 100 token** → 被开环阶梯支配。

**第 3 行的零给了它机制解释，方向与三条行为线一致**：算子在控制器规划的那个视界上没有
动作依赖的预测结构，用它规划的控制器在原理上赢不了固定日程。Table 2 的锚趟还给了更直接的
证据：MPC 因为 $B$ 把 `half_step_down`(−0.0696) 排在 `step_down`(−0.0665) 之前而**一次
`step_down` 都不选**——**第 3 行说的「动作进不了算子」与锚趟说的「动作次序排错了」是同一个缺陷
在预测侧与控制侧的两个读数。**

## 命名假设的兑现

| | 预注册 | 实测 | 判 |
|---|---|---|---|
| Table 1 第 3 行 | 显著不为零 | +0.0061，CI 含 0 → **零** | 预测**落空** |
| Table 2 主量 | 为正 | 同家族内两个对比都打平、等代价被支配 → **零** | 预测**落空** |
| 命名假设「第 3 行不为零 ⟺ 闭环为正」（判据：两者**同号**） | — | 两者**同为零** | **未被否证；但非零侧仍未受检** |

这三行必须一起读。假设本身没有被这一列推翻——它要求的是同号，实测同号。**被推翻的是
「这个任务族能提供非零侧」这个预测**，而那正是本线升级为承重的唯一理由。
后果按计划 §1.1 / §5.3 早已签死：Limitations 写「判定工具的阳性对照未成立」。
（**2026-09-15 重签**：这句的正文落点已具体化为新交付物里的「**反向未受检**——本文没有第 3 行
显著非零的实例，不主张『非零 ⇒ 闭环为正』」，见计划 §2.3。交付物由双向判定改为**单向的事前
否决判据**。）
**本档案不处置这条线，也不改写论文主张**——见 `tsar_cefr_results.md` §交还裁决。

## caption 义务（引用本列任何数字的地方同址带上）

1. 算在 **3582/3600 行、199/200 格**上（`51-a2` 具名剔除，G-S1）。
2. 读出是固定 CEFR 分类器，非被控模型自判；区间按源文本 bootstrap（100 组），非跨 seed `mean ± std`。
3. 第 4、5 行（LSTM / AE）为零以下，**不得作为「非线性无用」的证据**——本列样本量喂不动它们。
4. 第 3 行是**预注册预测的反面结果**，引用时须同址说明它原本被预测为非零。

## `lstm_baseline` 的改动与三条已发表列的关系

`LSTMSurrogate` 新增 `v_dim`（默认 1），因为本列的动作是 3 自由度 one-hot 而非二值 `u_remind`
/ `u_reset`。**`v_dim=1` 时 `input_size` 仍是 2、张量按同一顺序由同一批值构造**，所以
`behavioral_surrogate_rows_results.md` 的三列逐比特不变：
`tests/test_lstm_baseline.py::test_v_dim_one_matches_scalar` 钉住这一点。

**这条等价性 2026-09-15 独立复核过一次，不是只信单测**：用改动后的代码把三条行为线重新
导出到临时目录，与 09-12 发布的 `outputs/surrogate_rows_behavioral/*.json` 逐值比对
`rows` 与 `contrasts`。发布产物本身一个字节没动（`.claude/global.md` → *产物与谱系*：
`outputs/` 只增不改）。

- **复核结论：逐值相同。** 三条线的 `rows` 与 `contrasts` 共 **141 个数值字段，最大绝对差 0.0**；
  三个文件顶层**唯一不同的 key 是 `provenance`**（sha / 时间戳 / `git_dirty`）。
  即 `v_dim` 改动对三条已发表列的影响**精确为零**，不是「小到可忽略」。
  复核跑法：`scripts/eval_surrogate_rows_behavioral.py --out-dir <临时目录>`（2026-09-15，
  单核交互节点，1h47m；09-12 那趟多核约 45 min，**4× 差是核数不是代码**），
  比对脚本逐字段遍历两个 `rows`/`contrasts` 块，非数值字段与 `n_items` / `n_seeds` /
  `best_null_name` / `lag` / `horizon` / `n_folds` 一并比过。
