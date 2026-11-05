# 谱饱和探针：K 的特征值能否解释「两个读数就饱和」（#1，零 GPU）

> **状态**：✅ 已执行，判「**不过**」（2026-10-25）。三个饱和任务 **0/3** 在 $L{=}2,3$ 上满足 (a)/(b)，
> `constraint` 也不满足 ⇒ §二死亡条件成立，谱模式分不开饱和与不饱和。按 §三第二行：**不写进 proposal**，
> `docs/NEGATIVE_RESULTS.md` 已记，**不再设计第二个谱闸门**。向 Umesh 如实报告 null。结果见 §八、§九。
> 第三行「论文不填」：用户 2026-10-25 显式批准（主计划 §五.1）。三行为事前已写、先跑后补 LEDGER（零 GPU、只写 `results/`）。
>
> 原状态：2026-10-25 立项，主计划 [`post_meeting_2026-10-25_master_plan.md`](post_meeting_2026-10-25_master_plan.md)。
> 零 GPU、只读 `datasets/` 与行为线已落盘轨迹、只写 `results/spectral_saturation/`。按 `.claude/experiments.md`
> 「先跑后补三行」条款执行，当天补 LEDGER。**不是 M1 重跑**（区别见主计划 §四）。

## 一、问题与可证伪陈述

Umesh 的 conjecture：「两轮历史就饱和」应在 Koopman 矩阵的谱里看到。
可检验形式：延迟状态 $s_t=[y_t,\dots,y_{t-L}]$，$K\in\mathbb{R}^{(L+1)\times(L+1)}$。
从 $L{=}1$ 到 $L{=}2,3$ **新增的特征值**满足下列之一，则 conjecture 成立：
(a) 模长接近零（bootstrap 95% 区间上界 $< \tau$）；(b) 与已有特征值近似重复（距离 $<\tau$）。
$\tau$ **事前定为 0.1**（归一化读数，谱半径量级 1）。

**仅看谱不够**（M1 第一版因此作废）。必须连到得分：把 $K$ 按特征值模长截断，只保留前 $k$ 个模态
做多步预测，得分 $\mathrm{Skill}_H(k)$ 应在 $k{=}2$ 饱和。

**对照**：`constraint` 是唯一每加一次读数都继续涨分的任务（Table 5）。conjecture 若对，它的第 3、4 个
特征值应不满足 (a)(b)，且 $\mathrm{Skill}_H(k)$ 在 $k{=}2$ 不饱和。对照不成立 ⇒ 谱与饱和无关。

## 二、死亡条件

四个有深度数据的任务（`sentence_length_t10`、`vector_count_stage2_t10`、`vector_count_stage1_t10`、
`constraint`）里，饱和任务与 `constraint` 的谱模式**无法区分**（三个饱和任务中少于 2 个满足 (a)/(b)，
或 `constraint` 也满足）⇒ conjecture 不被支持，**向 Umesh 如实报告 null**，不换 $\tau$、不换统计量重判。

## 三、闸门三行

| 字段 | 内容 |
|---|---|
| 过了 → | 图进 proposal Task 1 prelim（深度–特征值 + 截断模态得分，一张双面板图）；主线动作：#2 proposal 修改 |
| 不过 → | 不写进 proposal；`docs/NEGATIVE_RESULTS.md` 记一条；不再设计第二个谱闸门 |
| 填哪张图/表 | **proposal 新图（Task 1 prelim）**；论文**不填**。是否算「填图」待用户裁决（主计划 §五.1） |

## 四、规格

### 4.1 数据与拟合（逐字复用，不改既有脚本）

- 载入：`scripts/eval_surrogate_rows.py::_load(task)`；状态配置：`scripts/paper_side_experiments.py::_state_config(lag, …)`，
  `lag = L`，`input_memory=0`，`control_mode="error"`（与 Table 5 / A1 matched 同口径）。
- 拟合：`paper_side_experiments._fit` → `AugmentedKoopmanModel`，取 `model.A` 作 $K$（`B`、`c` 不用）。
- **训练窗口**：A1 matched 口径（所有深度在同一批 transition 上拟合；`deep_lag` 决定最深深度）。
- 深度：$L\in\{0,1,2,3\}$（即状态里 1–4 个读数），`constraint` 到 4，T=10 三任务到 4。

### 4.2 输出量（每任务、每深度）

1. `eigvals`：$K$ 的特征值（复数，按模长降序）；轨迹级 paired bootstrap **2000 次**（与 A1 同生成器与种子），
   报告每个模长的中位数与 95% 区间。
2. `new_mode_test`：相对 $L{-}1$ 新增那个特征值的 (a)/(b) 判定与区间。
3. `truncated_skill`：对 $k=1..L+1$，$K_k = V_k\Lambda_k V_k^{+}$（保留前 $k$ 个模态，实部投影），
   用 `paper_side_experiments._score` 同一 held-out 面板算 $\mathrm{Skill}_H(k)$；配对区间。
4. `tap_loading`：$\|K[0, j]\|$ 对 $j$（最新读数对旧 tap 的载荷），与 M1 §2.4 的 `non_markov_share` 对照。

### 4.3 脚本与产物

- 新脚本 `scripts/spectral_saturation_probe.py`（**新增，不改既有文件**）；`--tasks` 默认四任务；
  `--out-dir` 默认 `results/spectral_saturation/`；存在即 `FileExistsError`。
- 产物 `results/spectral_saturation/<task>.json`；图 `results/spectral_saturation/fig_spectral_saturation.png`
  （白底，供 proposal；深色版供 deck）。
- 单测 `tests/test_spectral_saturation_probe.py`：① 合成 AR(2) 数据，$L{=}2$ 的第 3 特征值应 ≈0；
  ② 合成「题目水平 + 白噪声」数据，应出现一个 ≈1 的特征值且截断到 $k{=}1$ 得分不降。

### 4.4 解释时必须写的一句

若出现 ≈1 的慢特征值，它对应**题目基线水平**（M2 结论：增益 2/3–4/5 来自题目截距），不是轮间动态。
报告里写「饱和 = 状态记住了水平 + 一个快速衰减方向」，不写「状态捕捉了两步动态」。

## 五、执行位置

| 步骤 | 位置 | 原因 |
|---|---|---|
| 三个 `core` 任务（`sentence_length_t10`、`vector_count_stage*_t10`）全部 4.1–4.3 | **本地**（`datasets/` 在仓库内） | 只读 jsonl |
| `constraint` 对照 | **服务器**（`git pull` 后） | 轨迹在 `persona_drift_control/outputs/`，本地为空 |
| 单测、脚本开发 | 本地先做，用三个 `core` 任务跑通 | — |
| 结果 JSON 回传 | 服务器 → 本地：`results/` 被 gitignore，用 scp 或把 JSON 摘要贴进 §七 | — |

**先在本地出三个 `core` 任务的结果并填 §七前三行**；`constraint` 一行留空标「待服务器」。
判词必须等 `constraint` 对照回来再下，三行不得只凭饱和任务判「过」。

## 六、运行时估计

A1 matched 全量 3 分钟（结果档案 §运行时）。本探针多一层 $k$ 循环与 2000 次 bootstrap 的特征分解
（$4\times4$ 矩阵），预计 **< 15 分钟**，纯 CPU。

## 七、完成 = 什么

四任务 JSON + 一张图 + 本文件 §八结果表填好 + LEDGER 一行 + 三行判词（过 / 不过）写进本文件状态横幅。

## 八、结果表（执行后填，不动上文）

| 任务 | $L$ | $\|\lambda_1\|$ | $\|\lambda_2\|$ | $\|\lambda_3\|$ | $\|\lambda_4\|$ | 新增模式 (a)/(b) | $\mathrm{Skill}_H(k)$, $k=1..4$ |
|---|---|---|---|---|---|---|---|
| `sentence_length_t10` | 0 | 0.854 | — | — | — | — | +0.660 |
| `sentence_length_t10` | 1 | 0.961 | 0.577 | — | — | new 0.58, 上界 0.70, 距 1.43 ⇒ **否** | +0.743, +0.772 |
| `sentence_length_t10` | 2 | 0.973 | 0.433 | 0.433 | — | new 0.43, 上界 0.64, 距 0.35 ⇒ **否** | +0.563, +0.723, +0.770 |
| `sentence_length_t10` | 3 | 0.974 | 0.407 | 0.407 | 0.036 | new 0.04, 上界 0.39, 距 0.41 ⇒ **否** | -2.080, -0.144, +0.421, +0.769 |
| `vector_count_stage2_t10` | 0 | 0.879 | 0.811 | 0.769 | — | — | -2.394, -1.370, +0.597 |
| `vector_count_stage2_t10` | 1 | 0.987 | 0.973 | 0.965 | 0.876 (+2) | new 0.88/0.77/0.71, 上界 0.95, 距 1.65/1.54/1.48 ⇒ **否** | -2.451, -1.525, +0.786, +0.818 … +0.824 (k=6) |
| `vector_count_stage2_t10` | 2 | 0.991 | 0.979 | 0.972 | 0.849 (+5) | new 0.26/0.13/0.11, 上界 0.51, 距 0.45/0.57/0.59 ⇒ **否** | -2.104, -1.352, +0.749, +0.782 … +0.822 (k=9) |
| `vector_count_stage2_t10` | 3 | 0.994 | 0.988 | 0.975 | 0.904 (+8) | new 0.47/0.47/0.31, 上界 0.78, 距 0.47/0.47/0.31 ⇒ **否** | -2.963, -0.782, +0.734, +0.748 … +0.826 (k=12) |
| `vector_count_stage1_t10` | 0 | 0.840 | 0.790 | — | — | — | -2.630, +0.507 |
| `vector_count_stage1_t10` | 1 | 0.974 | 0.966 | 0.737 | 0.623 | new 0.74/0.62, 上界 0.81, 距 1.53/1.41 ⇒ **否** | -2.642, +0.653, +0.630, +0.634 |
| `vector_count_stage1_t10` | 2 | 0.975 | 0.975 | 0.614 | 0.602 (+2) | new 0.30/0.02, 上界 0.55, 距 0.33/0.64 ⇒ **否** | -11.600, +0.585, +0.582, +0.579 … +0.644 (k=6) |
| `vector_count_stage1_t10` | 3 | 0.983 | 0.980 | 0.722 | 0.722 (+4) | new 0.39/0.31, 上界 0.69, 距 0.39/0.31 ⇒ **否** | -9.801, +0.489, +0.501, +0.502 … +0.640 (k=8) |
| `constraint` | 0 | 0.642 | 0.000 | — | — | — | +0.287, +0.288 |
| `constraint` | 1 | 0.842 | 0.432 | 0.000 | — | new 0.43, 上界 0.47, 距 0.43 ⇒ **否** | +0.340, +0.438, +0.438 |
| `constraint` | 2 | 0.910 | 0.520 | 0.520 | 0.000 | new 0.52, 上界 0.56, 距 0.45 ⇒ **否** | +0.423, +0.479, +0.492, +0.492 |
| `constraint` | 3 | 0.943 | 0.592 | 0.592 | 0.557 (+1) | new 0.56, 上界 0.61, 距 0.51 ⇒ **否** | +0.476, +0.503, +0.516, +0.519 … +0.519 (k=5) |

产物 `results/spectral_saturation/{4 任务}.json` + `summary.json`；脚本 `scripts/spectral_saturation_probe.py`；单测 `tests/test_spectral_saturation_probe.py` 4 条全过。
模长为全数据拟合点估计（降序，`(+n)` = 还有 n 个更小的）；「上界」= 新增模式最大模长的轨迹级 bootstrap 95% 上界（2000 次，`default_rng(0)`）；
「距」= 新增模式到 $L{-}1$ 谱中最近特征值的距离。向量任务每加一层新增 $d$ 个模式（匈牙利匹配，用户 2026-10-25 裁决）。
`constraint` 的 $K$ 含 $v_t$ 槽，那个恒为 0 的模式就是它。$\mathrm{Skill}_H(k)$ 的 $k$ 取满时与 A1 matched 已落盘格子逐位一致（脚本内断言）。
环境：`persona_drift_pilot` 被 `/scratch` 清理损坏，本次用 `dkvar` env，换前复现 A1 matched 22 格最大差 1.9e-13。运行 8m52（事前估 <15 min）。

## 九、判词与读法（2026-10-25）

1. **判「不过」**：§二死亡条件两款都触发——饱和任务 0/3 满足（要求 ≥2），对照 `constraint` 也不满足，所以「谱模式」在两类任务上无法区分。不换 $\tau$、不换统计量。
2. **四个任务唯一的共同结构**是一个 $|\lambda|\approx0.96$–$0.99$ 的慢模式（向量任务每维一个）。按 §4.4：它对应**题目基线水平**（M2：增益 2/3–4/5 来自题目截距），不是轮间动态。
3. 新增模式的模长在 0.1–0.6，既不死也不重复；`sentence_length_t10` 在 $L{=}3$ 的第 4 个模式点估计 0.036 接近零，但 $L{=}2$ 那一步已经不满足，且上界 0.39。
4. ⚠️ **(a) 的功效问题**（事后发现，不改判）：合成 AR(2) 的真死模式在 100 条训练轨迹下 bootstrap 上界仍是 0.19 > $\tau$——零附近的模长在重采样下有向上偏。所以 (a) 在本项目的样本量上本来就难过；但 (b) 没有这个问题，而 (b) 也全部不满足（距离 0.31–0.64），结论不依赖 (a)。
5. 截断得分按规格用正交投影 $V_k\Lambda_kV_k^{+}$（不是谱投影 $V_kW_k^\top$），小 $k$ 下会出现很负的 Skill（例 `sentence_length_t10` $L{=}3,k{=}1$ −2.08），这是投影方式的产物，**不作解读**；它不进死亡条件。
