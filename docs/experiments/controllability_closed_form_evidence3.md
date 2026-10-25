# 证据③的 controllability 改写：从枚举筛选到用线性结构直接设计（#3）

> **状态**：▶ 待执行（2026-10-25 立项，主计划 [`post_meeting_2026-10-25_master_plan.md`](post_meeting_2026-10-25_master_plan.md)）。
> Umesh 的批评：KOMPAS 是「枚举候选 → 模型里滚 → 挑最好」，线性只让滚得快，没用线性结构**设计**干预。
> 交付：半页文字（proposal Thrust 3），加一个零 GPU 的离线一致性检查决定它放 prelim 还是 future work。

## 一、论点（半页的骨架，三段）

1. **约束满足是 $c$ 的线性不等式。** 模型仿射：$s_{t+h|t}=K^h s_t+\sum_{j<h}K^j(Bc+b)$，读出 $\hat y_{t+h}=C s_{t+h|t}$
   对标量 $c$ 是仿射的。「$h$ 步内 $y\in\mathcal Y=[y_{\min},y_{\max}]$」⇒ 对每个 $h$ 两条关于 $c$ 的线性不等式，
   可行集是区间 $[c_{\min}(t),c_{\max}(t)]$，闭式可解，**不需要枚举**。
2. **安全 = 用误差界收紧可行集。** 论文 App. E 的 $\Delta_h$ 是 $h$ 步预测误差上界；把 $\mathcal Y$ 向内缩 $\Delta_h$
   再求区间，就是 proposal 式 (3.24) 把 $u$ 换成 $c^p$、把 $g_i$ 取为盒约束的特例。
3. **单标量命令的可达集是线段，多属性任务秩不足。** $H$ 步可达集 $\subset s$-空间里过 $K^H s_t$ 的一条线段
   （方向 $\sum_j K^j B$）。Joint-2/3 要同时控两三个属性，`controllability_rank` 必然 $<$ 属性数
   ⇒ 这就是论文结论「扩展到多轴命令」的控制论表述，也直接引出 proposal RQ3.2 的内部通道（增加 $B$ 的列）。

## 二、离线一致性检查（决定 prelim 还是 future work）

### 2.1 检查什么

在已有闭环日志的每个决策点 $t$：用日志里的 $K,B,b$ 和候选集 $\mathcal C_t$，
算 (i) 枚举 argmin $\hat J_t(c)$（论文式 2）；(ii) 闭式：$\hat J_t$ 对标量 $c$ 是二次函数，极小点
$c^\star=\mathrm{clip}(-\beta/2\alpha,\,c_{\min},c_{\max})$，再投影到 $\mathcal C_t$ 最近点。
**一致率** = (i)(ii) 相同的决策点占比；不同时记录 $\hat J$ 差。

### 2.2 判定

| 字段 | 内容 |
|---|---|
| 过了 → | 一致率 ≥ 0.95 ⇒ 半页写成 **Thrust 3 prelim**：「现有控制器等价于闭式解，线性结构可直接设计」 |
| 不过 → | 半页写成 **future work**；不改控制器重跑 |
| 填哪张图/表 | proposal Thrust 3 一句 + 可达集示意（式 (3.14) 的数值例）；论文不填 |

### 2.3 数据与执行位置

| 日志 | 位置 | 可用性 |
|---|---|---|
| 控制线（IFBench / IFEval / COLLIE） | 不在本仓库（App. D 独立管线） | **待用户给路径**（主计划 §五.2）；拿到后在服务器跑 |
| `tsar_cefr` 闭环五臂 | 服务器 `persona_drift_control/outputs/tsar_cefr_gpu2*/` | `git pull` 后可做；$B$ 已辨识（#4a） |
| `constraint` S3 闭环 | 服务器 | 同上 |
| `core` 任务 | — | **不可用**：$B$ 不可辨识（headroom §二） |

默认先在 `tsar_cefr` 上做（动作是离散三值，闭式解 = 二次函数在三点上的最小值，仍可检验「等价」）。
脚本 `persona_drift_control/scripts/closed_form_vs_enumeration.py`（新增），产物
`persona_drift_control/outputs/closed_form_check_<line>/summary.json`，存在即报错。

### 2.4 可达集数值例（半页第 3 段用）

对每个 $B$ 可辨识的任务，调用 `src/koopman_ae/core.py::controllability_diagnostics(K, B, horizon=H)`，
报告 `controllability_rank`、`controllability_singular_values`、`gramian_condition`。
Joint-2/3 的 $B$ 不可辨识 ⇒ 用「单标量输入 ⇒ 秩 ≤ 1」的结构性论证代替数值，**不编数字**。

## 三、交付物

1. `docs/article/controllability_half_page.md`：半页英文（≤ 25 行），三段对应 §一，末尾一行写明 prelim / future work 的判定依据。
2. `summary.json` + 本文件 §四结果表。
3. 不改任何既有控制器代码、不重跑闭环。

## 四、结果表（执行后填）

| 线 | 决策点数 | 一致率 | 不一致时 $\Delta\hat J$ 中位数 | `controllability_rank` / $H$ | 判定 |
|---|---|---|---|---|---|
| `tsar_cefr` | | | | | |
| `constraint` | | | | | |
| 控制线 | 待路径 | | | | |

## 五、完成 = 什么

半页初稿 + 至少一条线的一致性检查 + §四填好 + LEDGER 一行（零 GPU）。
