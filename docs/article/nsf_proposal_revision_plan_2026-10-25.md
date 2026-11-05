# NSF MFAI proposal 修改计划（#2）

> **2026-11-05**：改动清单已按新版 PDF 重新对位，见 [`proposal_integration_plan_2026-11-05.md`](proposal_integration_plan_2026-11-05.md)（替代本文件「二」）。
> **状态**：▶ 待执行（2026-10-25，主计划 [`../experiments/post_meeting_2026-10-25_master_plan.md`](../experiments/post_meeting_2026-10-25_master_plan.md)）。
> 文件在仓库外：`E:\Kunpeng Liu\ICLR2027\`。本仓库 `paper/` **不碰**。
> Umesh 要求：preliminary 现在只是占位，**多放图、少放字**，让不懂 Koopman 也不懂 LLM 的 reviewer 能看懂。

## 一、前置（agent 第一步）

1. 向用户要 proposal 的 **.tex 源**（现只有 PDF）。副本命名 `Proposal_NSF_MFAI_2026_Cao_Liu.tex`，原稿不动。
2. 图：`E:\Kunpeng Liu\ICLR2027\proposal_deck\make_charts.py` 已能从论文表格数字生成六张深色图；
   加 `--theme light`（白底、墨色文字、同一套金/蓝/灰），配色过一遍 `dataviz` 验证脚本后再用。
   产物放 `E:\Kunpeng Liu\ICLR2027\proposal_figs\`，PDF 格式。

## 二、改动清单（顺序执行；每步独立可审）

### 第一步：替换 Task 1–3 的 Preliminary Results

| 位置 | 删 | 加 | 图 |
|---|---|---|---|
| Task 1 Prelim 开头 | — | 两句白话（读数 = 每轮回复的一个可测属性；状态 = 最近几次读数并排）+ 一句特例声明 + 对照公式 $s_{t+1}=Ks_t+Bc_t+b\subset$ 式 (1.25)，$\Phi=I$、$c^p$ 退化为标量、无 $u^{(\ell)}$ | — |
| Task 1 Prelim 正文 | 「MAE 3.778 → 2.007」段 | deck 第 1 页两条结论的英文版（趋势，不带数字）；引 Fig 3 作机器人类比的 LLM 版 | `c1_light`（分组柱）+ `c5_light`（深度折线）；#1 判过则再加谱图 |
| Task 2 §2.2 Prelim | — | Joint-2/3 两句 + 「学到的 lifting 无额外增益 ⇒ 表示复杂度应是识别问题的一部分（式 2.12）」 | `c2_light` + `c6_light` |
| Task 3 RQ3.3 | — | 三句：prompt 通道 MPC 可行 / 22 子任务全高于无控制 / 数字为点估计无区间；#3 的半页按其判定放 prelim 或 future work | `c3_light` |

### 第二步：新增

| 位置 | 内容 |
|---|---|
| Task 2 新小节 "Evaluation plan" | Skill$_H$ 定义（公式 + 两行解释）/ paired bootstrap（轨迹级，2000 次，95%）/ matched window（同一训练集比状态深度）；一张 `c4_light` |
| Overview / Intellectual Merit | 一句：preliminary work has validated interaction-level Koopman modeling on seven attribute-tracking tasks and three public instruction-following benchmarks |

### 第三步：收尾

1. 符号统一：全文只用 proposal 记号 $A, B_p, c^p, d_k$；引用论文时在括号里给对照 $(K, B, c_t, b)$，不混用。
2. 引用：KOMPAS 写 "under review, 2026"；补 Korda & Mezić 2018、Li et al. 2024、Dongre et al. 2025。
   **不进** Results from Prior NSF Support。
3. 白盒 vs 黑盒比较处加一句：该比较说明「先预测」有用，不说明哪种干预通道权限更大（后者是 RQ3.2/3.3 要研究的）。
4. 负面结果（Fig 2(b)(c)、App. F）**本轮不写**；另作分析后决定。

## 三、写作约束（来自 Umesh）

- 每个 prelim 段：**图在前、文字 ≤ 6 行**，每张图有一句图注说明"看什么"。
- 不出现未定义的术语：lifting、delay coordinates、receding horizon 第一次出现都要括号白话。
- 页数：三个 prelim 段合计增加 ≤ 1 页（图各占约 1/3 页）。超出则缩图不缩解释。

## 四、执行位置

全部本地（Windows 端，文件在 `E:\`）。需要 LaTeX 编译检查页数：本地若无 TeX，交付 .tex 由用户编译。

## 五、完成 = 什么

副本 .tex 改完三步 + `proposal_figs/` 六到七张白底 PDF + 一份 ≤ 20 行的改动摘要（给 Umesh）。
