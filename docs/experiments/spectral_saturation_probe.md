# 谱饱和探针：K 的特征值能否解释「两个读数就饱和」（#1，零 GPU）

> **状态**：▶ 待执行（2026-10-25 立项，主计划 [`post_meeting_2026-10-25_master_plan.md`](post_meeting_2026-10-25_master_plan.md)）。
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
| | | | | | | | |
