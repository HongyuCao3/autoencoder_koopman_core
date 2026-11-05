# 证据③的 controllability 改写：从枚举筛选到用线性结构直接设计（#3）

> **状态**：✅ 已执行（2026-10-25，用户批准第三行「论文不填」后运行）。判「**future work**」：`tsar_cefr` 一致率 0.876 < 0.95。
> 回放先过：重写的枚举复现 **100%** 日志动作（2297 个决策点，含 284 个推断出的 `copy` 停止）。不一致**全部**是同一种——枚举选 `copy`（停），
> 闭式解给 `half_step_down`：单一命令保持整个时域的模型表达不了「现在停」。半页稿 [`../article/controllability_half_page.md`](../article/controllability_half_page.md)。
> `constraint` 不可做：S3 的 `koopman_mpc` 是 token 预算下的分格动作模型，不是线性 $K,B$ + 二次代价的 MPC，§二的闭式解在它上面无定义。
> 控制线日志不可得（`KOMPAS-code-release.zip` 不含 Table 2 harness）。
>
> 原状态：▶ 待执行（2026-10-25 立项，主计划 [`post_meeting_2026-10-25_master_plan.md`](post_meeting_2026-10-25_master_plan.md)）。
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
| `tsar_cefr` | 2297 | 0.876 | 0.151（284 个，全为 `copy`→`half_step_down`） | 4 / 4（状态维 4；奇异值 0.194 / 0.074 / 0.019 / 0.0054，Gramian 条件数 1286） | **future work** |
| `constraint` | — | — | — | — | 不适用（控制器非线性 MPC） |
| 控制线 | 不可得 | | | | 不可得 |

产物 `persona_drift_control/outputs/closed_form_check_tsar_cefr/summary.json`；脚本 `persona_drift_control/scripts/closed_form_vs_enumeration.py`；
单测 `persona_drift_control/tests/test_closed_form_vs_enumeration.py` 3 条全过。$\beta$=1.8333、$\lambda$=0.1156、$H$=4 取自该臂 `run_config.json`，算子取自 `outputs/tsar_cefr_phase3/operator_fit.json`（臂实际加载的那份）。
闭式解按 §一.1 的式子：一个命令保持整个时域，$\hat J(c)$ 是松弛 one-hot $c\in\mathbb R^3$ 的二次函数（`copy`=0；切换代价 $\lambda\mathbb 1[c\ne0]$ 在候选点上等于 $\lambda\sum c$），$c^\star=-Q^{-1}q/2$ 投影到最近候选。
注意该臂 MPC 落盘的动作只有 `half_step_down`（2013/2013）与推断的 `copy`，所以「一致」的那 87.6% 本身信息量有限。运行 33 s。

## 五、完成 = 什么

半页初稿 + 至少一条线的一致性检查 + §四填好 + LEDGER 一行（零 GPU）。

## 五、v2：逐轮重解 + 不放松离散动作（2026-11-05 立项并执行，判 **future work**）

> 不改 §四 的 v1 结果与判词。v2 脚本 `persona_drift_control/scripts/closed_form_vs_enumeration_v2.py`，
> 产物 `persona_drift_control/outputs/closed_form_check_tsar_cefr_v2/summary.json`。

### 5.1 动机（用户 2026-11-05）

v1 的 284 个不一致（枚举 `copy` → 闭式 `half_step_down`）被认为是实现缺口而非方法结论：
① 代价按「一条命令保持整个时域 $H$」算；② 四个离散动作放松成连续 one-hot 再投影到最近候选，投影可能把「停」投成「再降半级」。
v2 要求：每个决策点只解当前一步，不做连续放松与投影，直接在 {copy, step_down, half_step_down, paraphrase} 四点上取 $\hat J$ 最小；
$K,B,b$、$\beta$、$\lambda$、$H$、2297 个决策点、先回放 100% 复现与 v1 逐字一致。

### 5.2 闸门三行（照抄 §2.2，阈值对象换成 v2）

| 字段 | 内容 |
|---|---|
| 过了 → | v2 一致率 ≥ 0.95 ⇒ 半页写成 **Thrust 3 prelim** |
| 不过 → | 半页保持 **future work**；不改控制器重跑 |
| 填哪张图/表 | proposal Thrust 3 一句 + 可达集示意；论文不填（同 v1，用户 2026-11-05 指令照抄） |

### 5.3 写脚本前发现的口径歧义（零 GPU、只读、不写 `outputs/` 的 scratch 探针；**未事前签，无闸门**）

「后续步按 MPC 同样的保持假设」无法直接实现：部署的 `mpc_action`（`scripts/run_tsar_cefr_closed_loop_arms.py:338`）
**没有保持假设**——它枚举全部 $4^H$ 序列取最优续行、只执行第一步；v1 本来也在每个决策点各解一次，「保持」只发生在时域内部。
所以「后续 $H-1$ 步怎么填」有三种读法，判词随之翻转。探针在同一批 2297 点上（复用 v1 的 `decisions` / `sequence_costs` / `enumerate_first`，
首步在四个离散点上取精确 $\hat J$ 最小）：

| 读法 | 后续 $H-1$ 步 | 一致率 | 判词（若作闸门） | v1 那 284 个点的去向 | 新增不一致 |
|---|---|---|---|---|---|
| A | 继续保持首步命令 $c$（只修缺口②） | 0.922 | future work | 284/284 → 一致 | 180 个 `half_step_down`→`copy` |
| B | `copy`（一步后停，对应 `stop_on_arrival=on_copy`） | 1.000 | prelim | 284/284 → 一致 | 0 |
| C | 取最优续行 | 1.000（按定义） | 无信息 | — | — |

读法：
- A 说明 284 个不一致的直接成因是投影（缺口②）；但「保持一条命令 4 步」高估继续降级的代价，在另一侧造出 180 个反向不一致。
- B 的 100% 等价于「该臂 MPC 的首步决策 = 一步前瞻后停的贪心」，即 $H=4$ 的时域在这批决策上没起作用；它不等于「线性结构可直接设计干预」。
- C 就是枚举本身，不构成检验。
- v1 的局限不变：该臂 MPC 只用过 `half_step_down`（2013）与推断的 `copy`（284），一致只检验「降不降」二选一。
- **四个数字都已看过**，此后选哪种读法作闸门都要在本节标明是看过数字之后选的。

**已裁决（用户 2026-11-05，看过上表数字之后）**：**闸门用 A**（§5.2 三行的阈值对象 = 读法 A 的一致率）；B 只作探索性结果报告，标「事后」，不参与判词。
探针脚本未落盘（会话内一次性计算）；正式结果以 v2 脚本 + 单测 + `summary.json` 为准，填在 5.4。

### 5.4 v2 结果（2026-11-05）

| 读法 | 角色 | 决策点数 | 一致率 | 不一致时 $\Delta\hat J$ 中位数 | 不一致类型 | 判定 |
|---|---|---|---|---|---|---|
| A 保持首步命令 | **闸门** | 2297 | **0.922** | 0.046 | 180 个，全为 `half_step_down`→`copy` | **future work**（< 0.95） |
| B 首步后 `copy` | 探索性，事后选定 | 2297 | 1.000 | — | 0 | 不参与判词 |

回放先过：复现日志动作 100%（2297/2297），决策点数与 v1 `summary.json` 相同（脚本在不满足时 raise）。$\beta$、$\lambda$、$H$、算子与 v1 同源。
单测 `persona_drift_control/tests/test_closed_form_vs_enumeration_v2.py` 3 条 + v1 3 条全过；运行 44 s（估 < 1 min），环境 `dkvar`（`persona_drift_pilot` 已失效）。

读法：
- 去掉投影后，v1 的 284 个 `copy`→`half_step_down` 不一致全部消失，所以 v1 判词的直接成因是缺口②。但「一条命令保持 4 步」高估了继续降级的代价，在相反方向造出 180 个 `half_step_down`→`copy`，代价差中位数 0.046，比 v1 的 0.151 小。
- 按 §5.2：半页保持 **future work**，不改控制器重跑。v1 的局限仍在：该臂的选择只有「降半级 / 停」两种，一致只检验二选一。
- B 的 1.000 只说明这批决策上 MPC 的首步等于「走一步后停」的贪心。它是看过数字之后选的，不能写成「线性结构可直接设计干预」。
