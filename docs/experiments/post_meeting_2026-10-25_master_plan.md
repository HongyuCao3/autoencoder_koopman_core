# 组会后计划（2026-10-25）——主计划（▶ 活跃，唯一活跃计划）

> **状态**：▶ 活跃。开于 2026-10-25，依据 Kunpeng 会后消息（Umesh 认可三条证据后的六项任务）。
> 该消息即本计划的 outline（用户书写）；本文件与五份子规格只负责把它填成可执行规格。
> 规则不重抄：硬约束见 `.claude/global.md`，实验见 `.claude/experiments.md`，文档见 `.claude/docs.md`。
> 术语按 `docs/NAMING.md`。账本 `docs/LEDGER.md`。

## 一、六项任务与子规格

| # | 任务 | 子规格 | 类型 | 前置 | 估时 |
|---|---|---|---|---|---|
| 1 | 特征值分析（Umesh 最想看） | [`spectral_saturation_probe.md`](spectral_saturation_probe.md) | 零 GPU，只读 `datasets/`，写 `results/` | 无 | 1–2 天 |
| 4 | $B$ 可辨识性 + 持续激励采集 | [`command_channel_identifiability_plan.md`](command_channel_identifiability_plan.md) | 4a 零 GPU；4b GPU（**不自行提交**） | 无 | 4a 0.5 天；4b 2 天 |
| 3 | 证据③改写成 controllability | [`controllability_closed_form_evidence3.md`](controllability_closed_form_evidence3.md) | 零 GPU + 半页文字 | 4a | 1.5 天 |
| 2 | proposal 修改 | [`../article/nsf_proposal_revision_plan_2026-10-25.md`](../article/nsf_proposal_revision_plan_2026-10-25.md) | 写作 + 出图 | 源文件；1/3 的结论 | 2–3 天 |
| 5 | 复现仓库 | [`../repro_repo_plan_2026-10-25.md`](../repro_repo_plan_2026-10-25.md) | 基建 | Sriram 账号 | 1–2 天 |
| 6 | 后续论文 | 不立项；12/16 后再议 | — | ICLR 结果 | — |

## 二、执行顺序（agent 按此顺序，不并行开两条写 `outputs/` 的事）

```
D1   #4a  现有产物上判定哪些任务的 B 可信（只汇总 + 补 bootstrap 区间）
D1–2 #1   谱探针：各深度 K 的特征值 + 截断模态得分曲线 + Constraint 对照
D2   #2   准备：向用户要 proposal .tex；make_charts.py 加白底主题
D3–5 #2   proposal 三步改；#1 的图若判「过」一并放入 Task 1 prelim
D4   #3   闭式解 vs 枚举的离线一致性；半页初稿；按 #4a 结果定 prelim / future work
D5+  #4b  若 #4a 判「核心任务不可辨识且控制线日志不可得」→ 写 sbatch 干跑 → 停，等裁决
空档 #5   复现仓库
```

## 二·五、执行位置（2026-10-25 用户裁决：拿不到旧产物的步骤写进计划，`git pull` 到服务器再做）

本地（Windows 端 clone）只有 `datasets/` 的 `core` 轨迹；`persona_drift_control/outputs/` 与 `results/` 为空，产物在服务器。
各子规格的「执行位置」节已逐步标明。汇总：

| 本地可完成 | 服务器（`git pull` 后） | 待用户提供 |
|---|---|---|
| #1 三个 `core` 任务 + 脚本 + 单测；#2 全部；#4b 开关代码 + sbatch 干跑；#5 抽代码 + README | #1 `constraint` 对照；#4a 行为线三列 $B$ 区间；#3 `tsar_cefr` / `constraint` 一致性检查；#5 行为线轨迹拷出 | 控制线日志路径（#3、#4a、#5）；proposal .tex（#2）；Sriram 账号（#5） |

agent 在本地会话里做完左列后**停**，把中列的命令清单写进对应子规格的结果表上方，等下一会话在服务器执行。

## 三、agent 的硬边界（本计划内）

1. **不提交任何 GPU 作业**；#4b 只写到 sbatch + argparse 干跑 + 运行时估计，然后停。
2. **不碰 `paper/`**（`.claude/paper.md`）。proposal 文件在仓库外（`E:\Kunpeng Liu\ICLR2027\`），按 #2 子规格处理。
3. 零 GPU 探针只读 `datasets/`、`persona_drift_control/outputs/`、`results/`；新产物只写 `results/<新目录>/`，**存在即报错不覆盖**（沿用 `scripts/paper_side_experiments.py::main` 的 `FileExistsError` 做法）。
4. 每个零 GPU 探针当天在 `docs/LEDGER.md` §三 补一行（三行 + 「改变了哪个决定」），注明三行为补写。
5. 任何闸门判「不过」→ 停下来报告，不改统计量重判（历史：M1 的阈值自由度，见 `experiments/paper_side_experiments_results.md` §2.2）。
6. 计划外的判断题 → 问用户。本计划已知的待裁决项列在 §五。

## 四、与已有结论的衔接（防止重开已关闸门）

| 已有结论 | 出处 | 对本计划的约束 |
|---|---|---|
| M1 谱/脉冲响应判不过，机制图不画；`results/mechanism_spectrum/` 作废 | `experiments/paper_side_experiments_results.md` §二 | #1 **不是** M1 重跑：M1 问「动作脉冲是否一步兑现」，#1 问「深度 2 饱和能否由 K 的谱 + 截断模态得分解释」。#1 不出脉冲响应、不画机制图。 |
| 记忆增益 2/3–4/5 来自题目截距（M2） | 同上 §三 | #1 预期会看到一个 ≈1 的慢特征值；解释时必须写明它对应题目水平而非轮间动态 |
| `core` 八任务 $B$ 不可辨识（$R$ 是 $[Z\|1]$ 的精确仿射函数） | `experiments/core_multiobjective_planning_headroom.md` | #4a 对 `core` 不重算，直接引用；#4b 的采集必须打破这个恒等（命令不再是误差的函数） |
| 行为线 `constraint` / `defense` / `tsar_cefr` 秩检查通过 | `paper_side_experiments_results.md` §2.1、§5.1 | #4a 在这三列补 $B$ 的 bootstrap 区间即可 |
| `core.py::controllability_diagnostics` 已存在 | `src/koopman_ae/core.py` | #3 的可达集 / 能控性秩直接调用，不另写 |

## 五、待用户裁决（agent 不得自行决定）

1. #1 的闸门第三行填的是 **proposal** 的图（非论文图）——是否算「填图」而免于「不填需批准」？
2. 控制线（IFBench / IFEval / COLLIE）的日志与代码不在本仓库（论文 App. D 说明是独立管线）。**路径在哪？** 没有它，#3 的「闭式解 vs 枚举」只能在 `tsar_cefr` / `constraint` 的闭环日志上做。
3. #4b 若需要采集：在哪几个 `core` 任务上做、几个 seed。子规格给的是默认值。
4. proposal 的 .tex 源文件与副本命名。

## 六、完成判据

六项各自的子规格末尾有「完成 = 什么」。本主计划完成 = #1 / #4a / #3 / #2 有结果文档或交付物，#4b 的 sbatch 等待裁决，#5 仓库建好并发出邀请。届时本横幅改 ✅，`docs/README.md` 顶部状态行同步。

## 七、进展（截至 2026-11-05）

| # | 状态 | 一句结论 | 出处 |
|---|---|---|---|
| 1 谱探针 | ✅ 判「不过」（10-25） | 三个饱和任务 0/3 满足判据；proposal 不放谱图 | [`spectral_saturation_probe.md`](spectral_saturation_probe.md) |
| 4a $B$ 可信度 | ✅ | `core` 不可辨识（命令 $=r-y_t$）；行为线三列补了 bootstrap 区间 | [`command_channel_identifiability_results.md`](command_channel_identifiability_results.md) |
| 4b 持续激励采集 | ✅ 闸门**过**（11-05） | 作业 19443006，`sentence_length_t10` 3 seed 随机命令；$CB$ = **0.734 ± 0.009 (n=3)**，区间都不含 0，$c$ 对状态 $R^2\le0.003$。Gramian 由移位结构决定、信息量低；探索性可达集（无闸门）：100 词请求下稳态上界 **73.0 ± 0.5 词 (n=3)**，留出四档预测与实测在 ≤79 词时差 ≤2 词、最高档低估约 7 词 | [`core_excite_b_identifiability_results.md`](core_excite_b_identifiability_results.md)；`scripts/core_excite_{b_identifiability,reachability}.py` |
| 3 controllability 改写 | ✅ 判 future work（v1 10-25；v2 11-05） | `tsar_cefr` v1 一致率 0.876 < 0.95；v2（逐轮离散、不投影，闸门 = 保持首步命令）**0.922** < 0.95，仍判 future work（探索性「一步后停」1.000，事后选定，不参与判词）。4b 数据**不能**重开 #3：随机命令没有控制器日志，离线比对是同义反复 | [`controllability_closed_form_evidence3.md`](controllability_closed_form_evidence3.md) |
| 2 proposal | ▶ 规划 + 出图完成，**等合作者商议** | 整合规划 E1–E8 + F1（页/段/句级修改与英文原句）；Fig. 1–4 白底印刷尺寸 PDF 已出、figcheck 无问题。未编进 tex（无源文件），页数未实测 | [`../article/proposal_integration_plan_2026-11-05.md`](../article/proposal_integration_plan_2026-11-05.md)；`docs/article/proposal_figs/make_proposal_figs.py` → `proposal/figs/` |
| 5 复现仓库 | ⏸ 未开始 | 等 Sriram 账号 | [`../repro_repo_plan_2026-10-25.md`](../repro_repo_plan_2026-10-25.md) |
| 6 后续论文 | — | 12/16 后再议 | — |

**待裁决 / 待商议（agent 不自行决定）**

1. proposal 整合规划 §六 的 11 条商议项（页数预算、F1 替换 p.2 箭头式、E4/E5 删改合作者数字与段落、E2/E3/E6 位置、记号、#4b 结果是否进 E7、p.11 Joint 任务定义与论文 §4.1 不符、E8 带 #3 v2 的 92%）。
2. 是否把 $K,B$ 与可达集结果交给 Umesh（建议附读法 4：Gramian 无信息，有信息的是 $CB$、$K$ 第一行与可达范围）。
3. proposal .tex 源文件（编图、实测页数需要）。

**画图时发现、已记入商议清单的两处事实问题**：① p.11 的 Joint 任务属性（sentence length / character length / avg word length）与论文 §4.1 Joint-3（字数 / 平均词长 / 逗号数）不符；② 论文「四个任务里三个在第二个读数拿满增益」偏松——Joint-2 每个深度区间都含 0，proposal 稿改为「两个任务」。论文正文未动。

**环境**：`sequor_vllm` 于 11-05 重建，必须 `PYTHONNOUSERSITE=1`（`~/.local` 有 cu12 栈）；计算节点 glibc 2.28，pip 轮子需逐个查 GLIBC 版本。
