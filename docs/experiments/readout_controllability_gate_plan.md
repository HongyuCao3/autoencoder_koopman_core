# 执行计划：两条线共用的"读出可控性前置闸门"（RC-gate）+ 各自的下一步

**状态**：待执行（2026-09-07 立项）。**适用智能体：Sonnet 5**——每个任务的规格、命令、预注册
判据都写死在本文档里，不需要自己做设计判断。
**触发来源**：2026-09-07 对两条在跑的工作线做的联合审计，输入是三份材料——
(a) 独立审计（本文档第零节把它的可复现数字全部固化）；
(b) [`adaptive_vs_fixed_claim_plan.md`](adaptive_vs_fixed_claim_plan.md) 第 11.4/11.5 节
（状态变量与 judge 打分未解耦；四条未裁决判断题）；
(c) [`ergo_koopman_mpc_opus_design_questions.md`](ergo_koopman_mpc_opus_design_questions.md)
（ERGO Phase C 的两个设计问题）。
**本文档同时是对 11.5 四条判断题与 ERGO 两个设计问题的裁决**，见第三节与第六节。
**阻塞**：T3 Step 2、T1b 的最终写法、ERGO Phase C、论文 §Experiments 主结论表。

> **开工前必读**（与 `adaptive_vs_fixed_claim_plan.md` 同一套纪律，逐条继承）
>
> 1. 本文档是**完整规格**。不要凭记忆复述、不要"优化"命令行、不要合并步骤。
> 2. 每个任务都有出口闸门。**闸门不过就停下来报告，不要自己想办法绕过去。**
> 3. 遇到本文档没写到的判断题，**停下来问用户**。作废判定与 evidence 台账是 Opus 的活。
> 4. **不要改** `paper/evidence/numbers.yaml`、`paper/evidence/superseded.md`，
>    **不要动 `paper/` 下任何文件**。
> 5. **不要覆盖任何已有产物**。每个任务的输出路径都是新的；目标已存在就停下来报告。
> 6. **不要改** `src/persona_drift/control.py` / `controller_cli.py`——`defense`/`stance` 两条线
>    共用它们。ERGO 的一切扩展都进 `src/persona_drift/ergo_koopman_mpc.py`（已存在）。
> 7. 环境：`export PATH=/scratch/hcao2/envs/persona_drift_pilot/bin:$PATH`，工作目录
>    `/home/hcao2/autoencoder_koopman_core/persona_drift_control`。登录节点没有 `conda`，
>    `scontrol` 坏了（用 `squeue -u hcao2` / `sacct -j <jobid>`）。GPU 一律 `sbatch`。

---

## 零、审计结论：两条线卡在同一个地方

本节每个数字都是 2026-09-07 在**已存产物**上离线复算的，未跑任何模型；每个任务的闸门都会
要求把它们逐位复现。

### 0.1 统一诊断

两条线现在都在做同一件事：**在没有先证明"读出里有可反馈的状态、且执行器能推动它"之前，
就去建/调闭环控制器。** 防御线花了 Phase A–J 才发现这一点（`koopman_defense_pilot.md`
第六节）；ERGO 线正准备用同一个顺序重犯一次。

所以两条线的下一步都是同一道**读出可控性前置闸门（RC-gate）**，三条判据：

| 判据 | 问的是什么 | 不过的含义 |
|---|---|---|
| RC-1 **有超过平凡基线的预测力** | 模型打得过"常数 / 只用轮次 / 只用外生输入无状态"这三个 null 吗 | 状态是多余的 |
| RC-2 **去掉共同趋势后仍有逐轨迹状态** | 去轮次（或 `shard_frac`）均值后 lag-1 自相关是否显著>0 | 没有可反馈的东西 |
| RC-3 **执行器能推动它，且不是提示词组成的机械效应** | 控制混淆项后输入增益是否显著同号 | 闭环等于开环 |

**RC-gate 三条全过，才准做控制器工作。任一不过，就地终止该分支并按本文档的终止写法收尾。**

### 0.2 线 A：T3 的 CONTINUE 判据在两个层面上都不成立

**(i) 因果可用性（这一条不需要跑数就能定）**：`analyze_refusal_direction_readout.py:89`
的 `agent_facing = f"{reminder}\n{stimulus}"`——`proj_pre_reply_t` 的测量上下文里**包含本轮
刚插入的提醒**。也就是说 `proj_pre_reply_t` 是 `u_t` 的函数。控制器在 t 轮决定 `u_t` 时
**看不到**它。**把 `proj_pre_reply` 当 MPC 的 t 时刻状态量是非因果的**，无论它的输入增益
多显著。控制器真正能条件化的只有 `proj_post_reply_{t-1}`，或者一个"本轮提醒插入之前"的
投影（现在还没算过，见 D2）。

**(ii) 规格稳健性**：预注册回归 `x_{t+1} ~ x_t + u_t + turn + arm_FE` 漏了 `u_{t+1}`，
而实测 `corr(u_t, u_{t+1}) = 0.5010`。同一批 736 个转移上补跑：

| 读出 | 规格 | `u_t` 系数 | `u_{t+1}` 系数 | R² |
|---|---|---:|---:|---:|
| `proj_pre_reply` | 预注册（无 `u_{t+1}`） | **+5.538** (p=6.04e-6) | — | 0.793 |
| `proj_pre_reply` | 加 `u_{t+1}` | **+3.912** (p=1.29e-3) | **−7.244** (p=4.51e-10) | 0.804 |
| `proj_post_reply` | 预注册 | −0.172 (p=0.782) | — | 0.798 |
| `proj_post_reply` | 加 `u_{t+1}` | −0.375 (p=0.556) | −0.903 (p=0.136) | 0.799 |

方向约定：`safety_direction_stats.json` 的 `harmless_mean_projection=+56.88` /
`harmful_mean_projection=−115.27`，即**正 = 无害极**。所以"当轮提醒"把读出推向**有害极**
（−7.244）、幅度还大于"上一轮提醒"推向无害极的幅度（+3.912）。这个符号翻转+幅度关系，更像
提醒文本本身在末 token 表征上的组成效应，不是执行器推动了一个持久状态。

**结论**：T3 Step 1 的 CONTINUE 不能直接支撑 Step 2，必须先过 D1。

### 0.3 线 A：11.4 的循环定义诊断成立，采纳

11.4 节追出的 `y_probe ≡ y_safety`（`trajectory_runner.py:141`）、`fit.y_col: y_safety`、
`rejudge.py:85-86` 原地覆盖——**状态变量就是评价信号本身**，这是本项目 house 规则 V8 的
blocking 级缺陷，也是 T1b"饿死"的直接机制。这个判断本文档全盘采纳，并据此在 D3 里改写
T1b 的结论限定语（对应 11.5 第 1 条）。

### 0.4 线 B：Phase B 的 GO 判据是空的

**(i) `richer_abs_sign` 这个对照是 vacuous 的**：`y_task_success ∈ {0,1}`（实测 666 行只有
0.0/1.0），而 richer 的额外特征是 `|y|` 与 `sign(y)`——对 0/1 变量**恒等于 `y` 本身**，与
`y` 精确共线。证据：两模型 `train_one_step_mse` 到 15 位有效数字相同
（0.08956779213150870 / 0.08956779213150866），richer 的 `A` 第 1/4/5 行逐位相同、`C` 是三个
相同分量的平均。它 rollout 更差是退化参数化的数值产物。**"ARX 赢过 richer 45%" 不构成
任何选型证据。**

**(ii) 无状态 null 比 ARX 更好**。同一 45/15 item split（`koopman_fit_report.json` 的
`held_out_item_ids`）：

| 预测器 | held-out MSE |
|---|---:|
| 常数（train 全局均值 0.1403） | 0.1096 |
| train 的逐轮均值（只用 `turn`） | 0.0832 |
| OLS `[1, shard_frac, u_reset]`，**完全不用 `y` 的滞后** | **0.0792** |
| ARX Koopman（报告值 `held_out_rollout_mse`） | 0.0893 |

ARX 连"零状态、只用外生输入"的回归都没打过（它自己的 `train_one_step_mse`=0.0896 也高于
这个 null 的 held-out）。机制上也看得出来：拟合出的第 3 维状态就是 `shard_frac`
（`A[2]=[-0.0295,-0.00118,1.00508]`，自系数 1.005、几乎不被其余分量驱动），第 2 维就是
`v=u`（`B[1]=1.0`），真正的状态只有 `y_lag` 一维（自持续 0.490），而 `y_{t+1}` 的解释力
主要来自 `0.819 × shard_frac` 这个**确定性轮次斜坡**——正是
`adaptive_vs_fixed_claim_plan.md` §0.3 花力气去均值要剔掉的那一项。

**(iii) 另外两条判据不是 binding constraint**：`controllability_rank=3` 对随机矩阵几乎必然
成立；`spectral_radius=0.953<1` 里那一维是 `shard_frac` 这个外生确定性斜坡（自系数 1.005>1），
把外生斜坡算进被控状态再谈稳定性是范畴错误。

**结论**：pilot 文档"没有出现防御线当年'读出没有可反馈状态'那种结构性卡点"这句，现有证据
不支持、方向上还相反。**Phase C 在过 E0 之前不准开工。**

### 0.5 线 B：reset 的代价不是编的（这条是好消息，直接回答设计问题 2）

设计问题 2 担心"预算是编出来的"。不需要编——**代价已经在数据里，而且是单调增长的**。
`inserted_tokens` 字段实测（Phase B 666 行，335 次 reset）：

| | 值 |
|---|---|
| `u_reset=0` 的行 | 全部 0 token |
| `u_reset=1` 的行 | 均值 **121.3** token，中位数 116，最大 229 |
| 随轮次 | turn1 **88.0** → turn3 114.3 → turn5 145.1 → turn8 187.0 → turn12 **229.0** |

原因是机制性的：reset 把"到目前为止揭示过的所有 shard"重新拼成一条消息，所以它的 prompt
代价**随轮次线性增长**。这给出一句可以直接写进论文、不需要任何假设的代价论证，也把
"永远 reset"从免费变成最贵的策略，从而让"什么时候花"重新成为一个真问题（早花便宜但信息少，
晚花贵但有效——实测两臂差距从 turn4 起才拉开）。

---

## 一、D1：线 A 输入增益的规格稳健性与因果可用性审查

**类型**：CPU，秒级，零 GPU。**依赖**：无。**必须先于 D2 完成。**
**这个任务要回答什么**：T3 Step 1 测到的 `u_remind → proj_pre_reply` 增益，是"执行器推动了
一个持久状态"，还是"提醒文本在被测上下文里"。

### 1.1 产物

新建 `scripts/analyze_input_gain_robustness.py`（CPU-only，纯 numpy/pandas，**不要 import
matplotlib**）。复用
`from analyze_readout_state import group_by_key, ols`（**不要改 `analyze_readout_state.py`
本体**，它的 G0 闸门必须保持不动），数据源与 `analyze_readout_state_t3.py` 完全相同：

```
outputs/koopman_case_study/refusal_direction_readout_expanded.json   (6 臂, 600 行)
outputs/koopman_case_study/refusal_direction_readout.json            (4 臂, 320 行)
```
轨迹键仍是 `arm + "|" + trajectory_id`。输出写
`outputs/koopman_case_study/input_gain_robustness_report.json` 并把表打印到 stdout。

### 1.2 四段计算

**S0 · 复现基线（闸门用）**：照 `analyze_readout_state_t3.py` 的行 C 规格
（736 转移、10 臂、加臂固定效应）重跑 `proj_pre_reply` 与 `proj_post_reply`。

**S1 · 加 `u_{t+1}`**：同上，回归量改成
`[1, x_t, u_t, turn, u_{t+1}, arm_onehot(去一列)]`，`u_{t+1}` 取 `b["u_remind"]`。
同时报 `corr(u_t, u_{t+1})`（全部转移上的 Pearson）。

**S2 · 外生子样本**（执行器动作与状态无关的臂，消除内生性）：
- S2a（主）：**只用 `phaseG_periodic`**（交替日程，64 个转移），无臂固定效应，
  分别跑无/有 `u_{t+1}` 两个规格；
- S2b（次）：`phaseG_periodic + phaseJ_fixed_t1 + phaseJ_fixed_t4`（192 个转移），加臂固定
  效应，同样两个规格。**S2b 只作参考不作判据**——`fixed_tk` 的 `u_t` 等价于 `1{turn==k}`，
  与 `turn` 项近共线，这一点要写进汇报。

**S3 · 持久性（提醒已离开紧邻位置）**：在同一 736 转移上构造二阶滞后样本
（同一轨迹内连续三轮 `t-1, t, t+1` 齐全的转移），回归
`x_{t+1} ~ 1 + x_t + u_{t-1} + u_t + u_{t+1} + turn + arm_onehot`，对
`proj_pre_reply` 与 `proj_post_reply` 各跑一遍，报 `u_{t-1}` 的系数/se/p。

**S4 · 因果可用性（同期检验，作为事实登记，不作判据）**：报
`proj_pre_reply_t` 对 `u_t` 的同期回归（`x_t ~ 1 + u_t + turn + arm_onehot`）的系数/p，
以及一句结论行 `"proj_pre_reply is a function of the same-turn action; not observable at
decision time"`。

### 1.3 闸门

**G-D1-0（复现）**：S0 必须给出 `proj_pre_reply` `u_t=+5.5380`（p=6.04e-6）、
`proj_post_reply` `u_t=−0.1719`（p=0.782），四位小数一致。不同就停——说明读文件或建样本
走偏了。

**G-D1-1（复现审计）**：S1 必须给出 `u_t=+3.912`（p≈1.29e-3）、`u_{t+1}=−7.244`
（p≈4.51e-10）、`corr=0.5010`，三位小数一致。不同就停下报告。

### 1.4 预注册判定 **RC-A**（跑之前写下，跑完不改）

**三条全过才做 D2**：

1. **S1**：`proj_pre_reply` 上 `u_t` 系数仍 `> 0` 且 `p < 0.05`；
2. **S2a**：`phaseG_periodic` 单臂上（有 `u_{t+1}` 的规格）`u_t` 系数 `> 0` 且 `p < 0.05`；
3. **S3**：`u_{t-1}` 在 `proj_pre_reply` **或** `proj_post_reply` 上系数 `> 0` 且 `p < 0.05`
   （至少一个通道说明效应能活过一轮）。

**任一不过 → T3 就地终止**，不做 D2，直接走 D3 的"T3 终止写法"。

**预注册预期**：第 1 条会过（已实测）；第 2 条不确定（64 个转移，功效有限）；**第 3 条大概率
不过**（`proj_post_reply` 在一阶上已经是 p=0.78 的零）。**如果 RC-A 不过，那是本任务的正面
产出，不是失败**——它把 T3 从"还有希望"变成"可判定"，并把"读出携带状态但不可控"这句话的
证据从一个可疑规格换成一组稳健规格。

---

## 二、D2：T3 Step 2（**仅在 RC-A 三条全过时执行**）

**类型**：GPU 1 个小作业（920 行 × 1 次前向，不生成 token）+ CPU 拟合。**依赖**：D1。
**这个任务要回答什么**：把 Koopman 的状态量从 judge 打分换成 judge-独立的确定性读出
（拆掉 11.4 的循环），然后问 RC-1：这个状态打得过平凡基线吗。

### 2.1 Step 1 · 算"动作之前"的投影（GPU）

因果可用性（第 0.2 节 (i)）要求状态在决定 `u_t` **之前**可观测，所以要新算一列
`proj_pre_action`：**不含本轮提醒**的上下文投影。

新建 `scripts/analyze_preaction_projection.py`：照抄
`scripts/analyze_refusal_direction_readout.py` 的模型加载/方向加载/重叠守卫/输出结构，
**唯一的差别**是 `rebuild_turn_messages` 里把 `agent_facing = f"{reminder}\n{stimulus}"`
改成 `agent_facing_noreminder = stimulus`（历史里前面几轮的提醒**保留**，只去掉本轮的），
每行输出 `proj_pre_action` 一列。臂集合与 `--layer 18`、`--direction-path`、
`--direction-stats-path` 与 T3 Step 1 的
`environment/run_refusal_direction_readout_expanded.sbatch` **逐字相同**，外加 4 个
Phase J 臂（与 `refusal_direction_readout.json` 同一批）。
输出：`outputs/koopman_case_study/preaction_projection.json`（**新文件名**）。
sbatch：`environment/run_preaction_projection.sbatch`，照抄上面那个，只改 job-name/日志/命令。

**闸门 G-D2-1**：`COMPLETED 0:0`；`rows` 数 == 920；`overlap_with_scored_attacks` 是空列表；
在 `u_remind==0` 的行上，`proj_pre_action` 必须与已缓存的 `proj_pre_reply` **逐行相等**
（容差 1e-3）——那些行本来就没有提醒，两种构造应当一致。**不相等就停下报告**，说明上下文
重建走偏了。

### 2.2 Step 2 · 用 `proj_pre_action` 做状态重拟合（CPU）

新建 `scripts/fit_koopman_projection_state.py`：

- 状态列 `x = proj_pre_action`，**标准化**：用 train split 的均值/标准差 z-score
  （投影量纲是 ±100 量级，ridge 对尺度敏感）；标准化参数写进报告；
- `u = u_remind`；`nu=1`、`mu=1`、**`contemporaneous_v=False`**（`u_t` 在 t 轮决策后才生效，
  其效应体现在 `t+1`，这与 `proj_pre_action` 的"动作之前"语义配套）；
- split：按 `attack_id` 切 held-out（**不是按轨迹**，否则同一攻击的两个 seed 会跨 split），
  留出比例 1/4，`--split-seed 0`；
- **`y_safety` 一行都不进状态**，只在报告里作为事后评价列出。

**闸门 G-D2-2（RC-1，这是本任务的主判据）**：报告必须同时给出四个数字——本模型的
`held_out_rollout_mse`，以及在同一 held-out 行上的三个 null：
(a) train 全局均值；(b) train 的逐轮均值（只用 `turn`）；(c) OLS `[1, turn, u_t]` 的
无状态回归。**本模型必须低于三者中的最小值**，否则**终止**：状态是多余的，写进 D3。

**闸门 G-D2-3（RC-3 复核）**：报告 `B`（输入通道）的系数与 95% bootstrap CI（1000 次，
按 `attack_id` 重采样）。CI 跨零就在汇报里明说，不要只报点估计。

### 2.3 出口

不管过不过，都**停在这里报告**，不要自行提交任何 MPC/闭环 GPU 作业——闭环规格是 Opus 看到
G-D2-2/G-D2-3 的数字之后再写。

---

## 三、D3：线 A 的文档修正与 11.5 四条判断题的裁决

**类型**：纯文档，零计算。**状态：✅ 已由 Opus 在立项同一次会话里执行完毕（2026-09-07），
Sonnet 不需要重做本节，只需在后续任务里遵守这里的裁决。**
下面保留全文，是为了让接手会话知道改了什么、以及第 3.2 节的裁决内容。

### 3.1 三处必须改的表述（都是已 commit 的内容）——**已改完**

1. **`docs/README.md:381`** 仍写着"投影上 p=0.0995/0.363，即'有状态的读出执行器够不到，
   执行器够得到的读出没有状态'"。这是 320 行时代的数字，已被 T3 Step 1 的 736 转移
   （p=6.04e-6）取代。改成：执行器能移动**回复前**投影（`u_remind` +5.538, p=6.04e-6），
   移不动**回复后**投影（p=0.782）；且该增益的规格稳健性由本文档 D1 裁定，**在 D1 出结果
   之前不要把任一方向写成结论**。
2. **`independent_judge_reactive_rerun_plan.md` 第八节**（尚未 commit）写着"独立 judge 口径
   下读出不携带任何逐轨迹状态，**任何闭环控制器都不可能赢开环**"。这句话的射程超过了证据：
   它只对 `y_safety` 这一个读出成立。按 11.4 的诊断改成"**在状态变量=judge 打分的当前架构
   下**，独立 judge 口径没有给出可学习的动作效应"。
3. **`adaptive_vs_fixed_claim_plan.md` 第 11.2 节第 2 条**把 `randsched_p75` 称为
   "真实等代价随机调度臂"。实测它花 **0.900** 提醒/轨迹（36/40；common16 上 0.9375），
   而 `koopman_b1` 是 0.750——**不是等代价，对手多花 20%**。抽签是 sha256 派生、独立的，
   36/40 只是运气（单侧 p≈0.014），不是 bug，**不要重跑、不要换种子**。改成
   "cost-matched to within +20%（实测 0.900 vs 0.750）"，并补一句方向说明：对手多花预算
   仍 n.s.，所以"自适应没打赢"这个方向是保守的；严格等代价的对照是 `randsched_p100`
   （1.000，与全部 `fixed_t*` 一致）。**11.1 节记录的原始数字一个字都不要改**——那里是
   如实记录，问题只在 11.2 的措辞。

### 3.2 对 11.5 四条判断题的裁决

1. **第 1 条（T1b 限定语 / Step2 是否提前）——采纳 11.4 的写法，并给出执行顺序**：
   T1b 的 NO-GO **加限定语**，标准表述是"在状态变量与 judge 打分未解耦的当前架构下，
   独立 judge 口径没有给出可学习的动作效应"，**不写**"自适应控制在独立 judge 下已确认无效"。
   T3 Step 2 **确实应该在 T1b 重新裁决之前**，但**前置是 D1 而不是直接开跑**：D1 的 RC-A
   不过，T3 就地终止，此时 T1b 的限定语**保留但改写成终局版**——"读出替换这条路已被 D1
   证否（增益不能在控制 `u_{t+1}` / 外生子样本 / 隔轮持久性中存活），因此 NO-GO 是当前
   读出族下的最终结论"。**这一句只有在 RC-A 不过时才准写。**
2. **第 2 条（"从不行动"是否写进本节 / 是否换数据集复核）——裁决：写进去，不换数据集。**
   "重拟合模型在 40/40 条轨迹上从不提醒"是一个诚实且信息量大的观察，按 11.4 的因果链
   （天花板 0.91 → `y_safety` 方差被压平 → 模型正确学出动作无效）写成机制性结果。
   换数据集复核**不做**——那问的是"这个结果多稳健"，而 11.4 已经说明它测的不是它自称测的
   那件事，稳健性检验会把 GPU 花在一个定义有问题的量上。
3. **第 3 条（T3 Step 2 的回归规格）——本文档 D2 第 2.2 节就是答案**，四个悬而未决的选项
   逐一定死：量纲 → train split z-score；pre 还是 post → **都不用**，用新算的
   `proj_pre_action`（理由见 0.2(i) 的因果可用性）；是否标准化 → 是；
   `y_safety` 当第二观测 → **否**，一行都不进状态，只做事后评价。
4. **第 4 条（数字如何落进 `paper/`）——维持原状：不动 `paper/`。** 等 D1（若通过则等 D2）
   出结果后由 Opus 走 evidence 台账流程。

### 3.3 落地位置与闸门（已执行，记录用）

改动落在这三个文件 + `adaptive_vs_fixed_claim_plan.md` 末尾新增的"十二、审计后续"一节。
**闸门（已核对）**：`git diff` 里没有 `paper/` 下任何路径；11.1 节的原始数字一个字未动。

---

## 四、E0：ERGO 读出可控性闸门（**Phase C 的准入条件**）

**类型**：CPU，秒级，零 GPU。**依赖**：无（可与 D1/D3 并行）。
**这个任务要回答什么**：ERGO 的读出满足 RC-1/RC-2/RC-3 吗。第 0.4 节已经用离线复算给出了
RC-1 的预期答案（**不过**），本任务把它固化成脚本并补齐另外两条。

### 4.1 产物

新建 `scripts/analyze_ergo_readout_state.py`（CPU-only）。数据源
`outputs/ergo_math_phaseB_random_excite/trajectories.jsonl`（666 行，60 items）与
`outputs/ergo_math_phaseB_random_excite/koopman_fit_report.json`（取 `held_out_item_ids`）。
输出 `outputs/ergo_math_phaseB_random_excite/readout_state_report.json` + stdout 表。
轨迹键用 `trajectory_id`；`shard_frac = turn / num_shards`。

### 4.2 四段计算

**第 1 段 · RC-1，null 对照**。在 `koopman_fit_report.json` 的同一 45/15 split 上，对
held-out 的全部行算四个 MSE：
(a) 常数 = train 全局 `y_task_success` 均值；(b) train 的逐 `turn` 均值（缺失的 turn 退回全局
均值）；(c) OLS `[1, shard_frac, u_reset]`，**不含任何 `y` 滞后**，在 train 上拟合；
(d) 直接抄报告里的 `arx.held_out_rollout_mse`。

**闸门 G-E0-1（复现）**：(a)=0.1096、(b)=0.0832、(c)=0.0792、(d)=0.08928，四位小数一致；
train/held-out 行数 = 506/160，train 均值 = 0.1403。对不上就停。

**第 2 段 · RC-2，去趋势后的逐 item 状态**。照
`analyze_readout_state.py` 第 1 段的定义，但把"轮次均值"换成 **`shard_frac` 十等分箱均值**
（ERGO 的轨迹长度不同，直接按 `turn` 池化会把长短轨迹混在一起）：报
`var_by_shardfrac`、`lag1_raw`、`lag1_demeaned`（去箱均值后的残差，轨迹内相邻配对，
池化后 Pearson，带 p）、`stable_traj_share`。

**第 3 段 · RC-3，输入增益**。
`y_{t+1} ~ 1 + y_t + u_t + u_{t+1} + shard_frac + item_onehot(去一列)`——
`u_{t+1}` 这一项是从线 A 的 D1 直接搬过来的教训（第 0.2 节 (ii)），**不要省**。
再跑一个不含 `u_{t+1}` 的对照，两个都报。

**第 4 段 · `richer_abs_sign` 共线性登记**。断言并打印：全部 666 行满足
`abs(y)==y` 且 `sign(y)==y`；两模型 `train_one_step_mse` 之差的绝对值 `< 1e-12`。
在报告里把该 baseline 标记为 `"vacuous_for_binary_y": true`。

### 4.3 预注册判定 **RC-B**（跑之前写下，跑完不改）

**三条全过才准进 Phase C（第六节）**：

1. **RC-1**：`arx.held_out_rollout_mse` **严格低于** (a)(b)(c) 三个 null 的最小值；
2. **RC-2**：`lag1_demeaned` 显著 `> 0`（p<0.05）；
3. **RC-3**：含 `u_{t+1}` 的规格里 `u_t` 系数显著 `> 0`（p<0.05）。

**预注册预期**：**RC-1 不过**（0.0893 > 0.0792，第 0.4 节已实测）。所以本任务的预期结局是
**Phase C 不开工，转 E1**。若 RC-1 意外通过，说明第 0.4 节的复算与脚本口径不一致——
**这是个真发现，停下来报告**，不要顺手往下走。

### 4.4 无论过不过都要做的文档修正

`ergo_multiturn_reliability_pilot.md` 的"拟合结果"小节现在写着三条 GO 理由，全部需要按
第 0.4 节改写：
- "ARX 赢过 richer baseline（rollout MSE 低 45%）"→ 标注 **vacuous**（二值 `y` 下 `|y|`/
  `sign(y)` 与 `y` 精确共线，train MSE 15 位相同），删除它作为选型证据的地位；
- "held-out rollout 误差不大（约 0.09）"→ 补上三个 null 的对照表，如实写"低于常数基线、
  **高于**只用轮次均值和只用外生输入的无状态回归"；
- "满秩可控 + 谱半径<1"→ 保留数字，但补一句：`controllability_rank` 对随机矩阵几乎必然满秩，
  且 `spectral_radius` 里那一维是外生的 `shard_frac`（自系数 1.005>1），两者都不构成
  "值得往下投"的证据；
- "没有出现防御线当年那种结构性卡点"这句 → **删掉**，换成 E0 的实测判定。
- **不要改**该文档"结果""样本扩充"两节里的任何原始数字（那些是执行器权威检查的结果，
  没有问题，本次审计不触碰）。

---

## 五、E1：读出替换（**仅在 RC-B 不过时执行**，这是 ERGO 线的真问题）

**类型**：Step 1 一个小 GPU 作业 + CPU 诊断。**依赖**：E0。

**为什么是读出而不是模型**：E0 若不过，问题是 `y_task_success` 这个读出本身——二值、稀疏
（Phase B 随机激励下成功率只有 **13.7%**，91/666）、且它的可预测部分几乎全被
`shard_frac` 这个确定性斜坡占掉。换更强的拟合器解决不了这个。

**默认候选：token 熵读出**。理由：这条线的名字（ERGO = entropy-guided resetting，
arXiv:2510.14077）本来就来自熵，但整条线至今没用过熵；它是连续量（解决二值/稀疏）、
一次前向的 logits 就能拿到（不需要新 judge）、且**与 `y_task_success` 这个评价指标天然解耦**
（直接修掉线 A 11.4 那个循环，不用等它踩一遍）。

### 5.1 Step 1（GPU 1 个小作业）

新建 `scripts/analyze_ergo_entropy_readout.py`：对
`outputs/ergo_math_phaseB_random_excite/trajectories.jsonl` 的 666 行，用与生成时相同的
`ChatModel`/decoding config 重建每轮的 agent-facing 上下文（复用
`ergo_math_trajectory.py` 的上下文构造，**不要自己重写拼接逻辑**），对该行已记录的
`agent_message` 做一次 teacher-forcing 前向，输出两列：
`entropy_mean`（回复 token 的平均预测熵）与 `entropy_answer_span`（"Current answer:"
之后那一段的平均熵；抽不到该 span 时写 null 并计数）。
产物 `outputs/ergo_math_phaseB_random_excite/entropy_readout.json`。
sbatch `environment/run_ergo_entropy_readout.sbatch`，资源照抄
`environment/run_ergo_phaseB_random_excite.sbatch`，`--time 01:00:00`。

**闸门 G-E1-1**：`COMPLETED 0:0`；行数 666；`entropy_answer_span` 的缺失率 < 10%
（超了就只用 `entropy_mean` 并在汇报里说明）。

### 5.2 Step 2（CPU）

把 `entropy_mean` / `entropy_answer_span` 各自当作 `x`，跑 **E0 第 2/3 段完全相同的诊断**
（去 `shard_frac` 箱均值后的 lag-1、稳定 item 偏移占比、含 `u_{t+1}` 的输入增益），
判据也**完全相同**（RC-2、RC-3）。

**预注册判定**：RC-2 与 RC-3 都过 → 报给用户，由 Opus 写"用熵做状态"的 Phase B'
重拟合规格（**不要自己设计**）。任一不过 → **ERGO 这条线在当前读出族下终止**，把结论写进
pilot 文档，与防御线并列成为"读出层面的结构性负结果"的第二个案例——这本身是论文里站得住的
一条内容，不是失败。

---

## 六、E2：ERGO Phase C 规格（**仅在 RC-B 三条全过时执行**）——对两个设计问题的裁决

**类型**：代码 + 3–5 个 GPU 臂。**依赖**：E0 通过。
本节同时是对 `ergo_koopman_mpc_opus_design_questions.md` 的答复。

### 6.1 设计问题 1 裁决：选 (a) 真值覆盖，且先跑量化检查

**裁决：`shard_frac` 在展望时必须用真值覆盖，不交给 `A`/`B`/`b` 外推。**
理由不是误差量级，而是定义：`shard_frac_{t+k} = (turn+k)/num_shards` 是**确定性、外生、
完全已知**的量，让线性模型去预测一个已知量只有误差没有信息收益；而且 `A[2,2]=1.005>1`
在 `horizon` 步递归下是发散方向，逐 item 的 `1/num_shards` 增量差异会被复利放大。

**实现方式（不要改父类签名）**：在 `ErgoKoopmanMPCController` 里覆写 `_simulate`，
用实例属性传递上下文而不是加参数——
```python
def next_u_remind(self, turn, history):
    self._lookahead_turn = int(turn)
    self._num_shards = int(history[-1]["num_shards"]) if history else None
    return super().next_u_remind(turn, history)

def _simulate(self, z, action, remaining_steps, remaining_budget):
    # 父类逐步递归；每步返回前把 aux 分量（状态最后一维）替换成真值
    ...
```
覆写里每推进一步就把 `z[-1]`（`aux_now` 的 `shard_frac` 分量，位置由
`modeling/dataset.py::build_reduced_state_pairs` 的 `y_hist + v_hist + aux_now` 顺序确定）
强制设为 `min(1.0, (self._lookahead_turn + k) / self._num_shards)`。
`self._num_shards is None` 时**退回父类行为**并计数一次（不要抛异常，兜底路径要有测试）。

**同时要跑的量化检查（记录用，不作判据）**：在 Phase B 的 60 个 item 上，比较 naive 递归
与真值覆盖在 3 步展望后的 `shard_frac` 偏差（按 `num_shards` 分组报均值/最大值），写进
Phase C 的文档小节。这条**算在本任务里**，Sonnet 直接做，不需要再问。

**测试**：`tests/test_ergo_koopman_mpc.py` 加至少 3 条——真值覆盖后 3 步的 aux 分量精确等于
`(turn+k)/num_shards`；`num_shards` 缺失时退回父类且不抛；覆盖不影响 `y`/`v` 两维的
父类算法（用一个 `A`/`B` 已知的小例子对拍）。

### 6.2 设计问题 2 裁决：加预算，代价论证用实测 token 成本

**裁决：加预算，`remind_budget = 1`（每条轨迹最多 reset 一次），与防御线 Phase J budget-1
对齐。** 这个预算**不是编的**，代价论证直接来自第 0.5 节的实测：

> A reset re-sends every shard revealed so far, so its prompt cost grows monotonically with
> turn: measured over 335 resets in the Phase B random-excitation data, an average reset
> costs 121.3 prompt tokens, rising from 88.0 at turn 1 to 229.0 at turn 12. Always-reset is
> therefore the most expensive policy, not a free upper bound, and the open question is
> **when** to spend a single reset — early resets are cheap but carry little information,
> late resets are expensive but land where the zero-control arm has already failed.

这句话可以逐字进论文。它同时解决了设计问题 2 担心的诚实性红线：预算不是为了制造一个能赢的
设定，而是这个动作**本来就有随轮次增长的实测代价**。

**汇报货币是 token，不是次数**：每个臂都必须报 `sum(inserted_tokens)/n_trajectories`，
主表里与成功率并列。**不要**只报"reset 次数"。

**附带的工程子问题（一并裁决：做）**：`episode_length` 从 `history[-1]["num_shards"]` 每条
轨迹动态取，不用 CLI 全局常数；`_planning_steps` 相应用动态值裁剪 horizon。这部分是机械
改动，在 `ErgoKoopmanMPCController` 内完成，**不要碰 `control.py`**。

### 6.3 臂表（预注册，五个臂，全部 60 items × 2 seeds）

| 臂 | 控制器 | 预算 | 作用 |
|---|---|---|---|
| `ergo_phaseC_zero_control` | `zero_control` | — | 下界 |
| `ergo_phaseC_always_reset` | `constant_remind` | 无 | 上界；**允许它更好，因为它更贵**，报 token 成本 |
| `ergo_phaseC_fixed_t{k}` | `fixed_schedule` | 1 | "事先选定的最好一轮"，`k` 取 1..4（`num_shards` 最小值是 4） |
| `ergo_phaseC_randsched_p100` | `random_schedule` | 1 | **等代价随机分配**——复用 T2 已实现并测过的 `RandomScheduleController`，`spend_prob=1.0`，`turns` 逐 item 取 `1..num_shards`（见 6.3.1） |
| `ergo_phaseC_mpc` | `ergo_koopman_mpc` | 1 | 被检验的对象 |

#### 6.3.1 两个 ERGO 专用臂怎么接进去（不碰共用代码）

`run_ergo_math_screening.py:80` 现在无条件调 `controller_cli.make_controller_factory`，而
`RandomScheduleController` 的 `turns` 在构造时就要定死——ERGO 每个 item 的 `num_shards` 不同，
不能传一个全局常数。**不要为此改 `controller_cli.py`。** 做法：在
`run_ergo_math_screening.py` 里对两个 ERGO 专用控制器名走一条**本地分支**，其余名字仍然
落到共用工厂（`entry_id` 就是 `item_id`，工厂签名 `(seed, entry_id) -> Controller` 已经
支持逐 item 定制）：

```python
if args.controller == "random_schedule":
    shards_by_item = {item.item_id: len(item.shards) for item in load_ergo_math_bank()}
    controller_factory = lambda seed, entry_id="": RandomScheduleController(
        turns=tuple(range(1, shards_by_item[entry_id] + 1)),
        spend_prob=1.0,
        seed=_excitation_seed(seed, entry_id),
    )
elif args.controller == "ergo_koopman_mpc":
    controller_factory = load_ergo_koopman_mpc_controller(...)   # 已存在于 ergo_koopman_mpc.py
else:
    controller_factory = make_controller_factory(...)            # 原样不动
```
`RandomScheduleController` 与 `_excitation_seed` 从 `persona_drift.control` /
`persona_drift.controller_cli` **import**，一行都不改。
**闸门**：改完 `git status --short src/persona_drift/control.py
src/persona_drift/controller_cli.py` 必须干净。

### 6.4 item 划分（预注册，跑之前定死）

题库 103 个 item。Phase B 用了 60 个（`--item-rng-seed 2`），其中 **45 个是辨识集**、
15 个是 Phase B 的 held-out。**Phase C 的评测集 = 与那 45 个辨识 item 不相交的 58 个**
（43 个 Phase B 完全没用过 + 15 个 Phase B held-out）。实测该池的 `num_shards` 分布：
`{4:12, 5:17, 6:13, 7:8, 8:7, 9:1}`。

**闸门 G-E2-1**：跑前打印评测集与辨识集的交集大小，**必须为 0**，非 0 就停。
用 `--item-ids` 显式传这 58 个 id（**不要随机抽样**），五个臂用**完全相同**的 id 列表。

### 6.5 预注册判定（跑之前写下，跑完不改）

主指标 `final_turn_success`，按 item 配对，bootstrap 10000 次（`default_rng(0)`）：

1. **闸门 1**：`mpc − zero_control` 的 95% CI 不含 0 且为正；
2. **闸门 2（主判据）**：`mpc − randsched_p100`（**等代价**）的 95% CI 不含 0 且为正；
3. **闸门 3（参照，不作准入）**：`mpc − max_k(fixed_t{k})`。

**闸门 2 不过 → 结论就是"自适应调度在 ERGO 上没有打过等代价随机分配"，照实写，不要调参
重跑、不要换 horizon、不要换 repeat_penalty。** 这是防御线 T2 那一条纪律的直接移植。
另报 token 成本表（第 6.2 节）与"达到同等成功率所需轮数"作为描述性维度。

**V8（指标泄漏）登记**：`y_probe ≡ y_task_success`（实测 666/666 行相同），MPC 直接优化被
汇报的指标。ERGO 这里的指标是确定性正则判定、不像 LLM judge 那样可被讨好，所以不是 blocking，
但**必须写进 limitation**，措辞："the controller conditions on the same deterministic
task-success signal that the headline metric reports; unlike an LLM judge this signal cannot
be gamed by the policy, but the comparison is not an independent evaluation."

---

## 七、依赖与分派

```
D1（CPU，秒级）─── RC-A ──过──> D2（GPU 1 作业 + CPU）──> 停下报告，等 Opus
                       └─不过─> T3 终止，并入 D3 的终止写法
D3（纯文档）─────────────────────> ✅ 已完成（Opus，2026-09-07），不需要重做
E0（CPU，秒级）─── RC-B ──过──> E2（Phase C，代码 + 5 个 GPU 臂）
                       └─不过─> E1（GPU 1 小作业 + CPU）──> 过则等 Opus 写 Phase B'
                                                        └─不过─> ERGO 线终止并收尾
```

**建议顺序**：D1、D3、E0 三个都是零 GPU、可并行，**先把这三个跑完再决定任何 GPU 支出**。
按第 0.2/0.4 节的预期，很可能两条线都会停在闸门上——那样这一轮的产出是两个可判定的负结果
加三处文档修正，**这是符合预期的结果，不要试图绕过闸门去拿一个正结果。**

**文件占用**（避免并行冲突）：D1 只写 `scripts/analyze_input_gain_robustness.py`；
E0 只写 `scripts/analyze_ergo_readout_state.py` 与 ERGO pilot 文档的"拟合结果"小节。
两者互不相碰。

---

## 八、失败模式清单

1. **改了 `analyze_readout_state.py` 本体**。它的 G0-1/2/3 闸门是 T0–T3 全部结果的基线，
   任何改动都会让已记录的数字不可复现。D1/D2 一律新建脚本 + import。
2. **改了 `control.py` / `controller_cli.py`**。`defense`/`stance` 共用。ERGO 的一切进
   `ergo_koopman_mpc.py`。提交前 `git status --short` 确认这两个文件干净。
3. **用 `proj_pre_reply` 当控制器状态量**。它是同轮动作的函数，非因果（第 0.2 节 (i)）。
4. **回归里漏 `u_{t+1}`**。两条线都要带（第 0.2 节 (ii)、第 4.2 节第 3 段）。
5. **拿"打过一个更复杂的 baseline"当 RC-1 的证据**。RC-1 的对手是**平凡 null**
   （常数 / 只用趋势 / 无状态外生回归），不是另一个自己拟合的模型（第 0.4 节 (i)）。
6. **Phase C 评测集与辨识集相交**。G-E2-1 就是为这个存在。
7. **闸门不过就调参重跑**。T2 已经立过这条规矩：偏离如实记录，不重新播种。
8. **写进已有目录**。每个任务的输出路径都是新的；目标已存在就停。
9. **顺手改 `paper/`**。见开工前必读第 4 条。

---

## 九、汇报格式

每个任务跑完给用户一段汇报，包含且仅包含：

- 任务编号与它的出口闸门逐条过/不过；
- job id 与各自 `State`/`Elapsed`/`ExitCode`（CPU 任务写"无作业"）；
- 该任务的**预注册判定**（RC-A / RC-B / G-D2-2 / 6.5 节三道闸门）逐条：预期 vs 实测，
  同号？同判定？；
- 所有系数**照抄**，带 se 与 p 或 95% CI，不要四舍五入、不要只报"显著/不显著"；
- 遇到的任何偏离本文档的地方，逐条列出并说明为什么；
- 需要用户/Opus 裁决的判断题，单独列一节。

---

## 十、与其他文档的关系

- [`adaptive_vs_fixed_claim_plan.md`](adaptive_vs_fixed_claim_plan.md)：本文档是对它第十一节
  （尤其 11.4/11.5）的续作。D2 是它 T3 Step 2 的规格；D3 第 3.2 节是对它 11.5 四条判断题的
  裁决。它第六节的 T7 写法与第八节失败模式清单被本文档继承。
- [`ergo_koopman_mpc_opus_design_questions.md`](ergo_koopman_mpc_opus_design_questions.md)：
  本文档第六节是对它两个设计问题的答复（问题 1 → 6.1 真值覆盖；问题 2 → 6.2 token 代价 +
  budget=1；附带工程子问题 → 6.2 末段"做"）。**但 E2 整节的准入条件是 E0 的 RC-B**，
  在 RC-B 通过之前不要执行第六节的任何一步。
- [`ergo_multiturn_reliability_pilot.md`](ergo_multiturn_reliability_pilot.md)：其"拟合结果"
  小节的三条 GO 理由按 4.4 节改写；"结果""样本扩充"两节的执行器权威结论**不受影响**。
- [`koopman_defense_pilot.md`](koopman_defense_pilot.md) 第一节（读出分辨率不足）、第五节
  （投影不能替代 judge）、第六节（一直没赢过 periodic）：RC-gate 三条判据是这三节教训的
  形式化。
- [`../evaluation/EVALUATION_METRICS.md`](../evaluation/EVALUATION_METRICS.md) §3.3
  （等代价 Pareto）：6.3 节等代价随机臂与 6.5 节闸门 2 的规范依据。

---

## 十一、指针：D1 执行结果（2026-09-07，线 A）

D1 已完成（Sonnet 5，CPU-only）。**RC-A 三条预注册判据未能全过**（第2条 `phaseG_periodic`
单臂检验字面规格因该臂日程严格交替导致 `u_t+u_{t+1}≡1` 而精确降秩、不可识别，退而求其次的
可识别参考值不显著；第3条隔轮持久性在 pre/post 两个通道均不显著），**T3 就地终止，未执行
D2**。G-D1-0/G-D1-1 两道复现闸门全部逐位通过。完整数字、S0–S4 全表、RC-A 逐条判定、
T1b 终局版限定语见 [`adaptive_vs_fixed_claim_plan.md`](adaptive_vs_fixed_claim_plan.md)
第十三节；产物 `outputs/koopman_case_study/input_gain_robustness_report.json`；脚本
`scripts/analyze_input_gain_robustness.py`（未改 `analyze_readout_state.py` 本体）。

**线 B（ERGO，E0/E1/E2）状态不受影响，仍待执行**——本节只是线 A 的收尾指针。
