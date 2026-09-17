# 论文副实验：消融 A1 + 机制 M1/M2（2026-09-16 开，硬放弃线 09-20）

> **状态（2026-09-16，当日收尾）**：✅ **完成**——三件全部执行完：A1 判「过」、M1 判「不过」、
> M2 判「过」（两列）。结果与判词见
> [`experiments/paper_side_experiments_results.md`](experiments/paper_side_experiments_results.md)；
> LEDGER §九 已回填。**剩余动作是写作，不是实验。**
> 它接替
> [`tsar_cefr_postmortem_and_next_tasks_2026-09-16.md`](tsar_cefr_postmortem_and_next_tasks_2026-09-16.md)
> （✅ 完成，三个零 GPU 候选已出清，保留为复盘与判词出处）。
> outline 由用户写（2026-09-16 签），本文件是把 outline 填成的可执行规格。
> 数字口径见 `../.claude/global.md` → *报告口径*；术语见 [`NAMING.md`](NAMING.md)。

## 一、定位

本计划只服务 **Table 3（消融）与 §5.x（机制）两个空壳**，**不开新任务线**
（不走 `.claude/experiments.md` → *开一条新任务线* 五步，因为它不引入新数据源、新读出、新执行器），
**不动 Table 1 / Table 2 任何已落盘数字**。

它要护住的是论文唯一的正面主张——[`article/CLOSED_LOOP_SYNTHESIS.md`](article/CLOSED_LOOP_SYNTHESIS.md)
§六① **「延迟嵌入是被测出来需要的，非线性不值」**——并把该主张背后的机制量出来：
**状态有慢衰减的持久成分（⇒ 记忆值钱），动作的脉冲响应长度 ≈ 1（⇒ 重排预算不值钱）**。
这两件事是同一个算子的两个谱性质，讲对了同时解释 §六① 与 §六②。

**分类与配额**：三件都是**方法**（消融 / 辨识数据分析），零 GPU、无 sbatch，
不占 `.claude/experiments.md` 的「仪器」配额，无需核 `sacct`。
A1/M1/M2 **两侧都有正文可写，所以不是前置闸门，是主线测量**；三行按用户 2026-09-16 要求事前签，
第二行填的是「什么情况下这一格不进表 / 结论翻面」。

## 二、A1 记忆深度扫描（消融）

| 字段 | 内容 |
|---|---|
| **过了 →** | 曲线单调上升并在 lag\* 饱和 ⇒ 落 Table 3 panel 1 + Figure「lag → `Skill_H`」，Table 1 第 2 行那句改写成「记忆是需要的，深度 ≈ lag\*」。**不是再开闸门。** |
| **不过 →** | lag≥2 的配对差全部跨 0（lag=1 即饱和），或非单调且无可读结构 ⇒ **曲线图不进正文**，panel 1 只留一格，主张降级为「需要**一步**历史」，Limitations 写明深度在本设计下不可辨。 |
| **它填论文哪张图/表** | Table 3 panel 1 + 一张曲线图。非「不填」，不需额外批准。 |

**设计死口径（签死，改一条作废）**

1. `seed_turn = lag_max + 1` 对**所有** lag 固定，H 随之固定。现行
   `scripts/eval_surrogate_rows.py` 的 `horizon = n_turns - (lag+1)` 会让 H 随 lag 变——
   **不固定就是两个坐标同时动，`Skill_H` 跨 lag 不可比。**
   沿用行为线既有惯例：窗口由最深状态定义，每个模型在同一绝对轮次上用自己的状态播种。
2. 扫描空间：`core` 的 T=10 三列（lag 0–6，H=3）+ `constraint`（nu 1–4，H=4）。
   `defense`（5 轮）/ `gsm8k_sharded`（64 评测行）/ T=5 四列**没有扫描空间，如实标「本设计扫不动」**，
   不换口径去凑。lag=0 即 Table 1 第 2 行的 Markov。
3. 配对 bootstrap 与 Table 1 同一套实现（`src/surrogate_eval/`），2000 次，按轨迹/题重采样。

## 三、M1 谱 + 脉冲响应（机制主力）

| 字段 | 内容 |
|---|---|
| **过了 →** | 秩检查通过的列上测到「状态半衰期 ≫ 动作脉冲响应长度」⇒ 画 §5.x 机制图（两子图）+ 一段正文，把 §六①（记忆值钱）与 §六②（重排不值钱）挂到同一个算子的两个谱性质上。**不是再开闸门。** |
| **不过 →** | 秩检查后可辨识的列 < 2，**或**测到脉冲响应长度 ≥2 却仍「重排买不到」⇒ **不画机制图**，§5.x 只保留 [`tsar_cefr_postmortem_and_next_tasks_2026-09-16.md`](tsar_cefr_postmortem_and_next_tasks_2026-09-16.md) §2.4 已有的数量级核算，§六② 的措辞从机制解释降为唯象并列。 |
| **它填论文哪张图/表** | §5.x 机制图 + 一段正文。非「不填」。 |

**前置硬条件（P1-a，候选 A 的教训）**：每列先跑两行秩检查——设计矩阵 $[Z\,|\,R\,|\,1]$ 的秩、
$R$ 对 $[Z\,|\,1]$ 的逐维 $R^2$。`core` 八列的控制量是跟踪误差 $r-y_t$、是状态的精确仿射函数
⇒ $B$ 的脉冲响应**「未定义」，不是「短」**，图上标不可辨识、**不填数**。
判据与措辞出处 [`experiments/core_multiobjective_planning_headroom.md`](experiments/core_multiobjective_planning_headroom.md) §四。

**报告口径**：谱半径 / 半衰期 / 脉冲响应是**拟合诊断量**，按 `../.claude/global.md` → *报告口径*
**不得作为跨臂头条数字**，只与已落盘的 `Skill_H` 并排作机制解释。

## 四、M2 题目截距替代（机制，能改 abstract 的解读）

| 字段 | 内容 |
|---|---|
| **过了 →** | `ours − (Markov + 留出题目截距)` 在 ≥2 个信息量足够的列上仍显著为正 ⇒ Table 3 panel 3 + 正文「延迟嵌入携带的是跨轮动力学，不只是题目身份」。**不是再开闸门。** |
| **不过 →** | 配对差塌到跨 0 ⇒ **改写主张**：增益主要来自题目级持久异质性；`paper/sections/abstract.tex` 里 *delay-embedded memory is required in every informative comparison* 那句的**解读**要同址加限定。**反向答案照样进表**——这一侧不是关掉，是结论翻面。 |
| **它填论文哪张图/表** | Table 3 panel 3 + §5.x 正文两句。非「不填」。 |

**竞争假说**：延迟窗口的增益可能只是在估**题目级固定效应**，不是跨轮动力学。
已有旁证——`constraint` 的状态项大部分是持久题目难度（S2 已诊断），四条线题间异质性都占主导。

**泄漏红线**：题目截距只能用**评测窗口之外**的轮次估，fold 与 Table 1 的 `purpose="report"` 折一致。
按 `../.claude/code.md`，这条写成**运行前 raise 的守卫**，不是文档里的一句话。

## 五、口径与守卫

- 诊断量（谱半径、半衰期、脉冲响应、秩、$R^2$）**不得作跨臂头条**；可报的是 `Skill_H` 与配对 bootstrap。
- 配对 bootstrap 与 Table 1 同一套实现，不新写一份；`ours` 的拟合口径逐字沿用
  [`experiments/core_surrogate_rows_results.md`](experiments/core_surrogate_rows_results.md) 与
  [`experiments/behavioral_surrogate_rows_results.md`](experiments/behavioral_surrogate_rows_results.md)。
- **三条必带测试**（`../.claude/code.md`：新增行为必须带测试）：
  1. **A1 对拍**：`lag = lag_max` 那一格与 `results/surrogate_rows_paired/` 已落盘的 ours **逐值相同**（防口径漂移）。
  2. **M1 秩守卫**：秩亏 fixture 上标 undefined / raise，**不得输出数**（防「不可辨识」被误报成「零」）。
  3. **M2 泄漏守卫**：截距估计轮次 ∩ 评测轮次 = ∅，否则 raise。
- 新脚本独立成文件，**不改** `scripts/eval_surrogate_rows.py` / `src/koopman_ae/core.py` / `scripts/train.py`。

## 六、产物

**六条路径全部新建，2026-09-16 核过当前不存在，不覆盖任何已有产物**：

- `results/{ablation_memory_depth, mechanism_spectrum, mechanism_item_effect}/`（`core` 侧）
- `persona_drift_control/outputs/{同名}_behavioral/`（行为线侧）

每份产物按 `../.claude/global.md` → *产物与谱系* 记 `git rev-parse HEAD` 与关键开关
（`lag`/`nu`、`seed_turn`、`horizon`、fold 口径、bootstrap 种子）。

## 七、运行时估计（假设写明；本线最近三次估错两次，按数量级读）

| | 开发 | 跑 | 假设 |
|---|---|---|---|
| A1 | 3–4 h | <20 min | 21 格（3 列 × 7 lag）+ `constraint` 12 格（3 seed × 4 nu）；ridge 闭式、≤1.2e4 评测行、bootstrap 2000 次、串行 CPU |
| M1 | 3–4 h | <5 min | 复用 A1 的拟合，只做 eig(A) + H 步脉冲响应 + 秩检查 |
| M2 | 4–5 h | 10–30 min | 与 A1 同量级 bootstrap，多一层留出估计 |

**最可能估错的项**：M2 的留出估计若按 seed × fold 重复，bootstrap 层数翻倍 → 跑时可能到 1–1.5 h。

**硬放弃线 2026-09-20**：ICLR 2027 全文 09-25 AOE，`paper/sections/` 现在只有 `abstract.tex`，
**瓶颈是写作不是实验**。到期未出结果就丢掉，正文按现有措辞收口。

## 八、明确不做

- **A3（`lambda_multi` / joint 封口）**：`../ABLATION_STUDY.md` 自陈的前置条件仍未处理——多步项只占
  `1/(n_batches+1)` 的步数，跨任务扫出来的是「数据规模 × λ」的混合效应。用户 2026-09-16 裁决
  **不做**，Limitations 如实写一句「joint 的多步项从未在可比强度下被扫过」。
- **M3（Table 1 第 3 行的灵敏度标定）**：与候选 D 同构，但本次投稿不做，第 3 行继续按
  [`article/CLOSED_LOOP_SYNTHESIS.md`](article/CLOSED_LOOP_SYNTHESIS.md) §三 的灵敏度注引用。
