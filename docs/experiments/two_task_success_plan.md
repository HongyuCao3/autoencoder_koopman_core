# 执行计划：两条任务线上争取 Koopman 正面结果（ERGO-append 主攻 + 防御线时间盒）

**日期**：2026-09-08 · **起草**：Hongyu Cao（与 Claude）· **适用**：Opus 5（裁决）/ Sonnet 5（执行）/ `general-purpose` subagent（盲标）
**状态**：**E0 已签字**（2026-09-07，Opus 5，见第十二节）。第十二节的规格补丁 B1–B4 与预注册项 C1–C2、Q7–Q8 **是本计划的一部分，优先于本文档其余各节中被它修改的文字**。
**依据**：截至 commit `ae167a5`（2026-09-07 17:23）的全部结果——`defense_line_redesign_plan.md` §9–12、`ergo_multiturn_reliability_pilot.md` G4、`adaptive_vs_fixed_claim_plan.md` §11–15、`signal_resolution_plan.md`、`measurement_validity_plan.md`、`article/PAPER_EXECUTION_PLAN.md` §1–2。
**时间假设**：ICLR 2027 摘要截止约 9/18、全文约 9/24（**用户核对中**；整张日程只压在这一个数上，若实际晚一周，第五节的 9/13 冻结同步后移）。

> **开工前必读**（继承既有纪律，一条不减）
>
> 1. 本文档是完整规格。凭记忆复述、"优化"命令行、合并步骤，都算偏离。
> 2. 闸门不过就停下来报告；**不调参、不换臂、不扩样本重跑**去找想要的答案。
> 3. 文档没写到的判断题，停下来问用户或 Opus；Sonnet 不替他们判。
> 4. **不改** `control.py` / `controller_cli.py` / `modeling/evaluate.py` / `modeling/dataset.py`；ERGO 侧一切改动放在 `ergo_*` 文件与 `run_ergo_math_screening.py` 本地分支里。**不动 `paper/`**（那是 Opus 的 evidence 台账流程）。
> 5. **不覆盖任何已有产物**，所有输出目录都是新的；`reset_mode="overwrite"` 的既有行为逐字节不变。
> 6. 盲标协议与 `measurement_validity_plan.md` P/G1 节逐字相同。
> 7. 环境：`export PATH=/scratch/hcao2/envs/persona_drift_pilot/bin:$PATH`，工作目录 `/home/hcao2/autoencoder_koopman_core/persona_drift_control`。
> 8. 每个任务完成后按第八节格式汇报，数字照抄带 CI；**Opus 独立重算后才算数**。
> 9. **每个会话除本任务一节外，必读第十二节**（E0 复核）。第十二节改写了 E1/E2/E3/E4/E5 的部分规格；只读任务节会漏掉 B1（分析器）与 B3（`forced_last_reset` 的预算路径），两者都会让作业提交后算不出闸门。

---

## 零、目标：先把"成功"写成可判定的分级

用户目标是"Koopman 在两条任务上都有正面表现"。直接以这句为目标会诱导调参找结果，所以先把它拆成可判定的等级，**跑之前写定，跑完不改**：

| 等级 | ERGO 线 | 防御线 |
|---|---|---|
| **S1 正面闭环** | 同代价下 Koopman-MPC 的最终轮成功率对**随机分配**配对差 CI>0，且对**同代价最优固定日程**不显著更差（同代价的定义见 Q8：模式 A 是 `fixed_t*`/`fixed_last`，模式 B 是 `fixed_(t,last)`） | 在新设定（第四节）下，Koopman-MPC 对同代价随机分配 CI>0 |
| **S2 不劣于先知** | 对随机分配 CI>0，对最优固定日程打平——写成"自适应不需要事先知道哪一轮最好即可达到最优固定日程" | 前提 1–2 在新设定下成立，闭环未及跑完（写为 ongoing） |
| **S3 方法论** | 三个前提全过但设定退化——写成"控制问题被执行器/终点构造性清空"的诊断 | 已在手：三个前提的检验程序、9.8% 与兑换率表、六个读出否决、judge 双向失效 |

**当前状态**：ERGO 处于 S3（G4），防御处于 S3（B1）。**本计划的目标是把 ERGO 推到 S1/S2，把防御在两个 GPU-天内判定能否离开 S3；两条线的 S3 都已经是可写的正文内容，不是失败。**

### 0.1 为什么 ERGO 还有戏：五层里有四层已经站住

`defense_line_redesign_plan.md` §12 的分层表：ERGO 线第 1 层（退化效应 −0.448）、第 2 层（reset 权威 +0.425）、第 3 层（`closeness` 四条判据全过）都成立，只有第 0 层（终点被末轮单个动作决定）不成立，而它的根因是**一行代码**——`ergo_math_trajectory.py:114` 把 reset 实现成**覆盖整个历史**。改成**追加**，历史重新进入被控对象，"状态"重新成为真问题。这是全项目里唯一一处"改一行就可能让四层变五层"的地方（§12.3 已指出），本计划的主攻就是它。

### 0.2 为什么防御线只给时间盒

三个前提在现有设定下全部失效，且截断已被排除为混淆（B1），基线率 9.8% 是攻击对这个模型的真实属性。要离开 S3 只能换设定（攻击池或目标模型），而换设定后前提 2、3 都要重测。两周内做完概率很低，所以只给**两个 GPU-天**做"新设定能不能让前提 1–2 成立"的判定，不做闭环。

---

## 一、分工原则与会话协议

### 1.1 谁做什么

| 角色 | 负责 | 不负责 |
|---|---|---|
| **Opus 5** | 本计划签字（第九节判断题）；每道闸门的**独立重算与裁决**；文档没写到的规格；`paper/` evidence 台账；论文 §1.3 第 5 句的改写 | 跑作业、写代码 |
| **Sonnet 5** | 代码 + 单测；sbatch；提交与监控作业；跑分析脚本；按第八节格式**只报数字**；把结果回写到各实验记录文档的新小节 | 解释结果、改判据、决定下一步 |
| `general-purpose` subagent | 盲标（D2），每批一个全新实例，prompt 与 P 节逐字相同 | 任何能看到 judge 分/臂名/轮次的事 |
| Fable 5.1 | 论文 Tier 3 正文（按 `PAPER_EXECUTION_PLAN.md` Step 5） | 实验 |

### 1.2 会话协议（每个任务一个会话）

1. Sonnet 会话开头只读：本文档对应任务一节 + 该任务点名的源文件；**不读**其他任务的结果节，避免"知道结果后改规格"。
2. 任务结束 Sonnet 写一份 ≤ 300 字的 handoff（做了什么 / 闸门逐条过否 / 偏离 / 产物路径），追加到对应实验记录文档，并 commit。
3. Opus 在**新会话**里只读 handoff + 产物 JSON，独立重算，出裁决，写进同一小节的"复核"段。
4. 裁决决定下一个任务能否开工；Sonnet 在裁决前不预跑下一步的 GPU 作业（CPU 准备可以）。

---

## 二、ERGO 线：E0–E6（主攻）

### E0 · Opus 签字（零成本，9/8）— **已完成**

第九节 8 个判断题全部裁决，第十二节记录复核过的事实、四处规格补丁（B1–B4）与两条预注册项（C1–C2）。**B1 与 B2 落地前，E2/E5 的 GPU 作业不得提交**——不是纪律问题，是那两步的闸门在现有代码里算不出来。

### E1 · `reset_mode="append"` + append 分析器（Sonnet，代码 + 单测，一天，零 GPU）

**改动范围**（只这四处；第 4 项来自 B1）：

1. `src/persona_drift/ergo_math_trajectory.py`
   - `ErgoMathTrajectoryConfig` 加字段 `reset_mode: str = "overwrite"`（取值 `"overwrite" | "append"`，其他值抛 `ValueError`）。
   - 第 112–117 行改为：
     ```python
     if u_reset:
         stimulus = _consolidated_stimulus(revealed_shards)
         if config.reset_mode == "overwrite":
             agent_history = [{"role": "user", "content": stimulus}]
         else:  # append
             agent_history.append({"role": "user", "content": stimulus})
     else:
         ...
     ```
   - `row` 加 `"reset_mode": config.reset_mode`。`inserted_tokens` 的计法不变（仍是合并消息的 token 数）。
2. `scripts/run_ergo_math_screening.py` 加 `--reset-mode {overwrite,append}`（默认 `overwrite`），透传进 config；`excitation_design` 名字后缀 `_append`（例如 `constant_remind_append`），否则不同模式的产物会同名。
3. `tests/test_ergo_math_trajectory.py`（或现有测试文件）加 4 条：`overwrite` 下 `agent_history` 在 reset 后长度为 1（既有行为）；`append` 下长度单调递增；两种模式下 `stimulus` 文本相同；非法值抛错。
4. **新增 `scripts/analyze_ergo_append_comparison.py`**（B1）。现有 `analyze_ergo_phaseC_comparison.py:36` 把 `ARM_DIRS` 硬编码在 `outputs/ergo_math_phaseC_*`，闸门集合固定为 gate1–gate6，**G-E2-1/2/3 与 P1/P2/P3 一个都不在里面**。新脚本要求：
   - 臂目录走 CLI（`--arm name=path`，可重复），不硬编码；
   - per-item 指标沿用 `_per_item_final_turn_success` 的定义（末轮 `y_task_success`，两 seed 求均值）与那条 `turn == num_shards` 断言，逐字复用，不重写；
   - 闸门由 `--gate a-b`（配对差 + bootstrap 10000 + `default_rng(0)`）与 `--identity a,b`（末轮 `judge_raw_output` 逐字相同的**轨迹**占比，分母是两臂 `trajectory_id` 的交集）声明；
   - 报告里带 `n_pairs`、两臂 `item_id` 集合是否相同、`trajectory_id` 交集大小；集合不同直接抛错（失败模式 3）。
   - 单测 ≥ 3 条：合成两臂数据上配对差与手算一致；`identity` 在人造"全同"与"全不同"上给出 1.00 / 0.00；`item_id` 集合不一致抛错。

**闸门 G-E1-1**：CPU 全套测试通过（基线 406 passed / 416 collected，同一批 5 个 NLTK 失败无关）；新分析器的单测全绿；`git diff --stat` 只涉及上述四个文件（加新测试文件）。
**闸门 G-E1-2（回归锁）**：新分析器以 overwrite 的 Phase C 目录跑一遍 `--identity always_reset,fixed_last`，必须复现 **116/116 = 1.000**（E0 已独立重算过这个数，见第十二节）。这条把 G-E2-2 的参照基线钉死在代码里，而不是钉在文档的一句话里。

### E2 · append 下的权威与恒等式检验（Sonnet 提交，3 个 GPU 臂，约 1.5 小时）

**臂**（58 个 held-out item × seeds 0 1，与 Phase C 逐字相同的 `--item-ids`）：

| 臂 | 控制器 | 输出目录 |
|---|---|---|
| `zero_control` | **复用** `outputs/ergo_math_phaseC_zero_control/`（u≡0，与 reset_mode 无关，逐字节相同，不重跑） | — |
| `always_reset_append` | `constant_remind` + `--reset-mode append` | `outputs/ergo_appendC_always_reset/` |
| `fixed_last_append` | `fixed_last` + `--reset-mode append` | `outputs/ergo_appendC_fixed_last/` |
| `fixed_t2_append` | `fixed_schedule` turns=(2,) + `--reset-mode append` | `outputs/ergo_appendC_fixed_t2/` |

sbatch 照抄 `environment/run_ergo_phaseC_*.sbatch`，只改 job-name / 日志 / `--reset-mode append` / `--output-dir`。**注意 Phase C 的 sbatch 不设 `--agent-max-new-tokens`（默认 512），照抄即保持一致，不要顺手改。**

**C2 · 上下文护栏**（append 让 prompt 单调膨胀：`always_reset_append` 每轮在保留的历史之上再追加一遍完整合并题面，12 × (题面 + 512 token 回复) ≈ 8–9k token）：逐轮记录 agent prompt 的 token 数，断言最大值 < 模型上下文上限，**汇报里报出这个最大值**。防御线 B1 就是一次截断混淆，这条零成本，先关掉这个口。

**预注册闸门**（按 item 配对，58 对，bootstrap 10000，`default_rng(0)`，用 **E1 新建的 `analyze_ergo_append_comparison.py`**——`analyze_ergo_phaseC_comparison.py` 算不出下面任何一道，见 B1）：

| 闸门 | 判据 | 过 → | 不过 → |
|---|---|---|---|
| **G-E2-1 权威保留** | `always_reset_append − zero_control` 的 95% CI 下界 > 0 | 继续 | **ERGO 线停在 S3**：append 把权威也抹掉了，写进 G4.5 |
| **G-E2-2 恒等式打破** | `fixed_last_append` 与 `always_reset_append` 末轮 `judge_raw_output` 逐字相同的轨迹占比 **< 0.90**（overwrite 下 E0 独立重算为 **116/116 = 1.000**，`agent_message` 亦 116/116） | 继续 | 停：追加没有让历史进入输出，报告并等 Opus 判断是否换 append 形式（第九节 Q1） |
| **G-E2-3 记录项** | `fixed_last_append − fixed_t2_append`、`always_reset_append − fixed_last_append` 的差与 CI；C2 的 prompt token 最大值 | 只记录，不判定 | — |

**预注册预期**：G-E2-1 大概率过（合并题面仍在 prompt 里）；G-E2-2 不确定——这是本计划最早的真正未知。

**G-E2-2 是整条 ERGO 线的开关**（Q2 裁决）：`final_turn_success` 作为主指标**只在 G-E2-2 通过的条件下**成立。若占比仍 ≥ 0.90，终点依旧是单个动作的函数，**E3–E5 全部不跑**，不是"换个次指标继续"。

### E3 · append 下的辨识数据、读出闸门与两道离线门槛（Sonnet 提交 1 个 GPU 臂 + CPU 一天）

**Step 1 · Phase B′ 采集**：`environment/run_ergo_phaseB_random_excite.sbatch` 照抄一份，加 `--reset-mode append`，`--item-rng-seed 2`（与 Phase B 同一批 60 item），`--random-excite-p 0.5`，seeds 0 1，输出 `outputs/ergo_appendB_random_excite/`。
**闸门 G-E3-0**：`COMPLETED 0:0`；行数与 Phase B 相同（666）；逐轮检查两种动作在每个 `shard_frac` 十等分箱里都出现过（任一箱只有单一动作 → 报告，不自行重采）。

**Step 2 · 读出闸门**（CPU，秒级）：`scripts/analyze_ergo_readout_state.py`（F0 修正版）与 `analyze_ergo_closeness_readout.py` 指向新目录，跑 RC-0..3。
**闸门 G-E3-1**：`closeness` 四条全过（判据定义沿用 `signal_resolution_plan.md` 0.5 节，不改阈值）。

**Step 3 · 拟合**（CPU）：`fit_koopman_ergo_closeness.py` 指向新目录 → `koopman_fit_report_closeness_append.json`。**额外拟合一个交互模型**：在同一 45/15 item split 上，回归

$$
c_{t+1} = a\,c_t + b_0\,u_{t+1} + b_2\,u_{t+1}\,c_t + g\,\text{shard\_frac}_{t+1} + \text{const}
$$

直觉：自适应能赢的前提是"reset 的效果取决于现在卡得多深"，$b_2$ 就是这一项。式中 $c_t$ 是第 $t$ 轮的 closeness，$u_{t+1}$ 是下一轮是否 reset（`contemporaneous_v=True` 的语义），$b_0$ 是 reset 的基础效应，$b_2$ 是效应随 closeness 的变化，$g$ 是信息累积斜坡。复用 `analyze_state_action_interaction.py` 的回归骨架（防御线 Phase H 用过），按 item 做 1000 次 bootstrap。**新脚本落在 `scripts/analyze_ergo_state_action_interaction.py`**（不改防御线那个：它读 `y_probe`，这里读 `closeness`），带一条在合成数据上恢复已知 $b_2$ 的单测。

**闸门 G-E3-2 状态依赖**：$b_2$ 的 95% CI 不含 0，且符号为负（越接近答案 reset 越没用）。不过 → 最优策略仍是常数规则，**ERGO 线停在 S3**，写"reset 效应与状态无关"。

**Step 4 · 预算模式判定**（CPU）——照 `measurement_validity_plan.md` 0.4 节的方法，在 Phase B′ 数据里按"reset 之后还剩几轮"分层看最终轮成功率：

**闸门 G-E3-3 退化判定**：

| 结果 | 预算模式（预注册） |
|---|---|
| "剩 0 轮"层比其余各层合计高 ≥ 0.15，且 Spearman(剩余轮数, 成功率) p<0.05 | **模式 B：k=2，末轮 reset 固定**。末轮那次不是决策（已知最优），自适应只决定**第一次**放哪。对手臂改为 `fixed_(t,last)`、`randsched_(t~U,last)` |
| 否，且**明确落在阈值之外**（差 < 0.10 **且** p > 0.10） | **模式 A：k=1 计数预算**，臂表同 Phase C 加 `fixed_last_append` |
| **阈值近失**（差 ∈ [0.10, 0.15) **或** p ∈ [0.05, 0.10]） | **停下报告，等 Opus 裁决，不许自动落回模式 A**（E0 裁决，理由见第十二节第五段：近失更像"末轮优势仍占主导"→ 模式 B 的证据，而不是"没有优势"→ 模式 A 的证据） |

不引入 token 预算（G4.3(3)：低于一次末轮重述的 token 预算是人为稀缺）。

**Step 5 · 离线回放准入**（CPU）：用 E4 的控制器在 Phase B′ 的 60 条轨迹上做离线回放（同 T1b Step 4 的方法）。
**闸门 G-E3-4**：`n_distinct_spend_turns ≥ 3`，且任一轮次的花费占比 ≤ 0.60。不过 → 控制器退化成固定臂，**不提交 E5**，报告并回到 E4 检查目标函数/视野（只允许改这两项，且改后重新过 G-E3-4）。

### E4 · 全视野、终值目标的 MPC + 模式 B 对手臂（Sonnet，代码 + 单测，一天，零 GPU；可与 E3 Step 1 并行）

只改 `src/persona_drift/ergo_koopman_mpc.py`、**新增 `src/persona_drift/ergo_controllers.py`**（B2）与 `run_ergo_math_screening.py` 本地分支：

1. **视野到 $T$**：`--koopman-horizon 12`。**E0 已核：这一项和第 4 项基本做完了**——`control.py:290` 的 `_planning_steps` 已按 `episode_length` 裁剪，`ergo_koopman_mpc.py:118` 已在每次 `next_u_remind` 里设 `episode_length = num_shards`，`:100` 已覆写 `_remaining_budget` 读 `u_col`（F3 那个 bug 已修）。E4 只需确认 + 单测，不要重写。动作二值、$T\le 12$，`_simulate` 的递归枚举 ≤ 2^12 = 4096 叶子/轮、约 8k/轨迹，毫秒级，不需要 DP。
2. **目标函数**：加 `objective: str = "terminal"`；覆写 `_simulate`，`"terminal"` 只返回最后一步的 `readout`（`remaining_steps > 0` 时只取子树 `max`，不加本层 `value`），`"sum"` 保留父类行为作消融。Phase C 退化成 `fixed_t2` 的直接原因是 horizon=2 + 逐轮和，两处都必须改。
   **B4 · 两条随之而来的预注册**：(a) terminal 目标下中间动作不再承担 `repeat_penalty`，所以 terminal 臂**固定 `repeat_penalty=0.0`**；(b) 父类 `control.py:319` 的 `if value > best_value` 在**精确平手时偏向 action=0**，这条平手规则**照原样保留并预注册**。两条一起，使"终值读出对早期位置不敏感"表现为 G-E3-4 不过（读作"没有状态依赖"），而不是留下"旋钮没调对"的解释空间。
3. **模式 B 支持**：`forced_last_reset: bool`。**B3 · 这里就是 F3 那个 bug 的形状，规格写死**：
   - `turn < num_shards`：可分配预算 = `k − spent − 1`（为末轮预留 1 次）；
   - `turn == num_shards`：**无条件返回 1，绕过 `remaining_budget <= 0` 的提前返回**。若只是把预算均匀减 1，父类 `control.py:315` 在末轮看到预算耗尽会直接 `return 0`，强制 reset 永远不会发生。
   - 规划器把 $u_T=1$ 当作已知常量（末轮不是决策变量）。
4. `shard_frac` 展望真值覆盖保持（F3 4.1 已做，`ergo_koopman_mpc.py:131`）。
5. **C1 · 最早决策轮**：`_current_state`（`ergo_koopman_mpc.py:78`）在历史不足 `max(nu−1, mu−shift)+1` 时返回 `None`；`contemporaneous_v=True, nu=mu=1` 下第一次真决策在 turn 2，turn 1 恒为 0。而第七节讲的"早 reset 防止锚定"有一部分住在 turn 1。**预注册 `pad_short_history=True`**，并把它写进臂名（`mpc_terminal_pad_append`）。**这个值现在定死，不许在看到 G-E3-4 之后再改。**
6. **B2 · 模式 B 的两个对手臂现在不存在**，必须一并实现（`CONTROLLER_CHOICES` 只有 7 个；`fixed_schedule` 只吃绝对轮次、`fixed_last` 只吃 per-item 末轮）：
   - `fixed_t_and_last`：在绝对轮次 $t$ 与该 item 自己的末轮各 reset 一次（`--fixed-schedule-turns t` + 末轮），$t = $ `num_shards` 时退化为单次并**报告**而非静默；
   - `randsched_t_and_last`：$t \sim U\{1,\dots,T-1\}$ 加末轮，RNG 播种方式照抄 `RandomScheduleController`（`control.py:120`）的 per-item 播种，同 seed 可复现。
   - 两者都实现在新的 `ergo_controllers.py` 里（**不改 `control.py`**），在 `run_ergo_math_screening.py` 的 `CONTROLLER_CHOICES` 与工厂里接上。
7. 单测 ≥ 10 条：terminal 与 sum 在手工小例子上给出不同决策；terminal 的平手取 0；horizon 裁剪到 `num_shards`；`forced_last` 下**每条轨迹恰好 k 次**且末轮 `u_reset=1`（失败模式 4）；`forced_last` 且 k=2 时中途恰好 1 次；`pad_short_history=True` 下 turn 2 已是真决策；`reset_mode` 不影响控制器；预算读 `u_reset` 列；`fixed_t_and_last` 在不同 `num_shards` 的 item 上各恰好 2 次；`randsched_t_and_last` 同 seed 可复现且末轮必 reset。

**闸门 G-E4-1**：测试全绿；`git diff --stat` 不含 `control.py` / `controller_cli.py` / `modeling/`。

### E5 · Phase C′（Sonnet 提交，6–8 个 GPU 臂，约 4 小时）

58 held-out item × seeds 0 1，全部 `--reset-mode append`。臂表按 G-E3-3 的模式：

| 臂 | 模式 A（k=1） | 模式 B（k=2，末轮固定） | 代价 |
|---|---|---|---|
| `zero_control` | 复用 | 复用 | 0 |
| `always_reset_append` | 复用 E2 | 复用 E2 | T 次（**不是同代价对手**，只作权威参照） |
| `fixed_last_append` | 复用 E2（**同代价**） | 复用 E2（**更便宜的参照**，1 次） | 1 |
| 固定日程（**P2 对手**） | `fixed_t1..t4_append` | `fixed_t_and_last_append`，t=1..4 | = k |
| 随机等代价（**P1 对手**） | `randsched_p100_append` | `randsched_t_and_last_append` | = k |
| **被测臂** | `mpc_terminal_pad_append`（k=1） | `mpc_terminal_forcedlast_pad_append`（k=2） | = k |
| 消融（可选，Opus 定） | `mpc_sum_append` | 同 | = k |
| ERGO 熵阈值 | **不做**（Q4 裁决） | 不做 | — |

**预注册主判据**（按 item 配对，bootstrap 10000，`default_rng(0)`；主指标 `final_turn_success`，次指标 turn≥3 的 `closeness` 均值只记录不判定）：

| 闸门 | 判据 | 含义 |
|---|---|---|
| **P1 主** | `mpc − randsched` 的 95% CI 下界 > 0 | 看状态选时机优于不看状态随便选 → **S2 起步** |
| **P2** | `mpc − 同代价最优固定日程` 的 CI 包含 0 或下界 > −0.05。**同代价的定义（Q8）**：模式 A 取 `fixed_t1..t4_append` 与 `fixed_last_append` 中最好的那个（都是 1 次）；模式 B **只取 `fixed_t_and_last_append` 中最好的那个**（都是 2 次） | 不劣于事先知道最好一轮的先知 → 与 P1 合起来是 **S1** |
| **P2b 参照（模式 B 专有，不判定 S1）** | `mpc − fixed_last_append` 的差与 CI | `fixed_last` 只花 1 次而 mpc 花 2 次，**用 2× 的 reset 赢过它不构成 S1 证据**；但 mpc 至少要追平它，否则"多花一次还没有更好"要在正文里明说 |
| **P3 诊断** | mpc 花费轮次的分布；按 mpc 自选轮次拆开的配对差（同 Phase J 11.5 的表） | 解释赢/输在哪 |

P1 不过 → **ERGO 线停在 S3**，Phase C′ 作为"前提全满足仍无自适应收益"的更强负结果写进论文；**不调 horizon、不换目标函数重跑**。

### E6 · 汇报与回写（Sonnet 半天 + Opus 半天）

按第八节格式回写 `ergo_multiturn_reliability_pilot.md` 新一节"E1–E5：append 执行器下的闭环"；Opus 独立重算后在 `paper/evidence/` 走台账，并按第五节改论文叙事。

---

## 三、ERGO 线的成本与日程

（E0 复核后修订：新增全部是 9/8–9/9 的 CPU 工作，吃掉的是原有缓冲，**GPU 总量与"9/12 前出结果"这个点不变**。）

| 任务 | 类型 | 预计 | 依赖 |
|---|---|---|---|
| E0 | Opus 裁决 | **9/7 已完成** | — |
| E1（含 B1 分析器） | 代码 | 9/8 全天 | E0 |
| E4（含 B2 两个控制器、B3、B4、C1） | 代码 | 9/8–9/9（与 E1/E2 并行） | E0 |
| E2（含 C2 护栏） | 3 GPU 臂 | 9/9 上午提交，1.5 h | E1 |
| E3 Step 1 | 1 GPU 臂 | 9/9 下午（G-E2-1/2 过后） | E2 |
| E3 Step 2–5 | CPU | 9/10 | E3 Step 1, E4 |
| E5 | 6–8 GPU 臂 | 9/11 提交，约 4 h | G-E3-4 |
| E6 | 回写 + 复核 | 9/12 | E5 |
| 缓冲 / 消融 | — | 9/13–9/14 | — |

GPU 总量约 8 小时，全部在 9/12 前出结果，留 6 天进论文。**E4 提前到与 E1 并行**：模式 B 的两个对手臂（B2）是 E5 的前置，而 G-E3-3 的裁决可能在 9/10 才出，届时再写控制器就压线。

---

## 四、防御线：D0–D3（时间盒，两个 GPU-天，9/9–9/12）

### D0 · Opus 决定是否花这两天（零成本）

要回答的只有一个问题：**换设定后，前提 1（权威可测）与前提 2（早期可预测）能否成立。** 若 Opus 判断两周内即使成立也来不及做闭环，可以直接跳到 D3 写法，把 GPU 全给 ERGO。

### D1 · 基线率筛查（Sonnet，2 个 GPU 臂 + 盲标约 12 次调用）

目的：找一个盲标成功率 ≥ 0.30 的（攻击池 × 目标模型）组合。

| 臂 | 攻击 | 模型 | 规模 |
|---|---|---|---|
| `screen_qwen4b` | SafeMTData 里**未用过的** 100 条（`default_rng(0)` 从 600 中剔除已用 40 条后抽） | Qwen3-4B | 100 × 1 seed × 5 轮，`--agent-max-new-tokens 1024` |
| `screen_alt` | 同 100 条 | 备选目标模型（Q3：Llama-3.1-8B-Instruct 或 Qwen2.5-7B-Instruct） | 同 |

零控制、无 judge 依赖。跑完各抽 late(3,4,5) 的 60 行盲标（P 协议，两遍，加 5 拒答对照 + 5 重复），得每个组合的 assisted 率与 Wilson CI。

**闸门 G-D1-1**：任一组合 assisted 率 CI 下界 ≥ 0.20 且点估计 ≥ 0.30 → 进 D2；否则 **防御线定格在 S3**，写第十节负结果（含"换模型也没用"这一条新证据）。

### D2 · 前提 2 在新设定下的检验（CPU + 约 6 次盲标调用）

在过闸门的组合上，按盲标结果把 100 条攻击排序，取 assisted 率最高的 40 条为新池。检验早期可预测性：用 turn 1–2 的**确定性特征**（回复长度、拒答标志、`closeness` 式的目标词覆盖、句子嵌入前 8 个主成分）做留一攻击的 ridge，预测 late 的盲标 assisted。
**闸门 G-D2-1**：留一 AUC 的 bootstrap CI 下界 > 0.60。过 → 记为 **S2（ongoing）**，闭环留到摘要之后；不过 → S3。

### D3 · 写法（Opus）

无论 D1/D2 结果，`defense_line_redesign_plan.md` 第十节的负结果写法都要加两句：换目标模型后基线率如何（D1）、早期可预测性如何（D2）。这两句把"你只是挑了个坏设定"的审稿风险再关掉一层。

---

## 五、论文并行线与合流点（Opus / Fable）

- 9/8–9/12：Opus 按 `PAPER_EXECUTION_PLAN.md` Step 3–4 继续，**以 S3 版本建 contract**（两条线都是方法论 + 诊断），这样即使 ERGO 全部不过，论文也能按期成稿。
- **§1.3 第 5 句必须改**：现写的"瓶颈是测量分辩率"已被 F4 否定（软 judge 恢复了分辩率仍不过 RC-1/RC-3）。改为：闭环要成立需要三个可检验前提（执行器权威、早期可预测、读出耦合且推得动），本文给出检验程序，并展示两条线各自在哪一条失效、以及一处"改一行"让前提全部成立后的闭环结果（E5 的结论无论正负都填这里）。
- **合流点 9/13**：E5 结果与 Opus 复核到位 → 若 S1/S2，RQ2/RQ3 的主表换成 Phase C′；若 S3，Phase C′ 作为"前提全满足仍无自适应收益"进 §Experiments 收尾。9/13 之后不再接受新实验数字进正文。
- 9/18 摘要；9/19–9/24 Tier 3 正文与审计。

---

## 六、失败模式清单（都在本项目真实发生过）

1. **E5 之前调 horizon / 目标函数去凑 G-E3-4**。只允许在 E4 规格里的两个选项间切换，且切换要记录；不允许第三种。
2. **看到 E2 结果再改 append 形式**。append 的形式在 E0 定死（Q1），E2 不过就报告，不试第二种。
3. **复用臂时 item 集合不一致**。E2/E5 复用 `zero_control` 前必须断言 `item_id` 集合与新臂完全相同。
4. **预算读错列**。F3 那个 `_remaining_budget` 读 `u_remind` 的 bug 已修，但新加的 `forced_last_reset` 路径要有单测锁定"每条轨迹恰好 k 次"。
5. **把 P1 过、P2 不过写成 S1**。那是 S2，措辞按第零节表。
6. **盲标泄漏臂身份**。D1 的 `screen_alt` 回复风格与 Qwen 不同，可能被裁决者猜到模型——不影响本任务（不做臂间比较，只测基线率），但**不要**在 D1 里做跨模型的配对比较。
7. **Sonnet 在裁决前提交下一步 GPU**。第一节 1.2 第 4 条。
8. **改 `paper/`**。任何数字进论文都走 Opus 的台账。
9. **用 `analyze_ergo_phaseC_comparison.py` 去算 append 的闸门**。它的 `ARM_DIRS` 硬编码在 overwrite 的 Phase C 目录，闸门集合固定为 gate1–gate6；指向新目录也只会算出 overwrite 的旧数或报错。必须用 E1 新建的分析器（B1）。
10. **G-E3-3 阈值近失时自行落回模式 A**。近失区间的处置见 E3 Step 4 的第三行——停下报告。
11. **`forced_last_reset` 把预算均匀减 1**。末轮那次会被父类的预算提前返回吃掉（B3）。单测必须断言"末轮 `u_reset=1`"，而不只是"总数等于 k"。
12. **在模式 B 里把"赢过 `fixed_last`"写成 S1**。那是 2 次 vs 1 次，不同代价（Q8）。S1 的对手是 `fixed_t_and_last`。
13. **看到 G-E3-4 不过之后再改 `pad_short_history` 或 `repeat_penalty`**。两者在 E0 已预注册（C1、B4），不在失败模式 1 允许切换的"两项"之内。

---

## 七、ERGO 线的两条设计说明（给 Opus 复核用）

**为什么 append 能恢复"状态"**：overwrite 下末轮 prompt 与历史无关，所以末轮输出只是"末轮有没有 reset"的函数（G4.2 的 116/116 恒等式）。append 下末轮 prompt = 历史（含此前的错误尝试）+ 合并题面，历史里"是否已经锚定在错误答案上"会影响输出——这正是 Laban 描述的"走错就不再恢复"机制。此时早 reset 的价值是**防止锚定**，晚 reset 的价值是**信息完整**，两者的权衡取决于这道题目前卡得多深，也就是取决于状态。G-E3-2 的 $b_2$ 就是在测这个权衡是否真的存在。

**为什么模式 B 比 token 预算干净**：G4 已证明末轮 reset 是无预算上界的等价物，把它从决策变量里拿掉，剩下的"第一次放哪"才是自适应唯一能发挥的地方；这样对手臂 `fixed_(t,last)` 与 `randsched_(t~U,last)` 都在同一预算、同一末轮动作下比较，差异只来自第一次的时机。token 预算要制造稀缺才有权衡，会被审稿人质疑设定人为。

---

## 八、汇报格式（Sonnet 每个任务结束时）

- 任务编号；每道闸门：判据 / 实测值（带 CI）/ 过或不过；
- 作业 id、`COMPLETED` 状态、行数、`item_id` 集合断言结果；
- 偏离本文档之处逐条列出（包括"文档没预见到的技术偏差"，如 S2a 那种精确降秩）；
- 产物路径（全部为新目录）；
- 留给 Opus 的判断题单独一节；**不写结论句**。

---

## 九、判断题与 E0 裁决

**全部已裁决**（Opus 5，2026-09-07）。"裁决"列是最终规格；Sonnet 不再就这 8 项提问。

| # | 问题 | 起草建议 | **E0 裁决** |
|---|---|---|---|
| Q1 | append 的具体形式：(a) 只追加合并题面；(b) 追加合并题面 + 一句"忽略你此前的尝试，重新求解" | (a) | **(a)**，同意，理由如起草：一次只动一个执行器变量，(b) 会让 E2 不过时无法归因 |
| Q2 | 主指标仍用 `final_turn_success`，还是换 turn≥3 的 `closeness` 均值 | 保留 `final_turn_success` | **保留**，但把依赖关系写显：它作为主指标**只在 G-E2-2 通过的条件下**成立。G-E2-2 不过则 E3–E5 全部不跑（已写进 E2 一节） |
| Q3 | D1 的备选目标模型 | Llama-3.1-8B-Instruct | **Llama-3.1-8B-Instruct**，同意 |
| Q4 | 是否实现在线 ERGO 熵阈值臂 | 一天内能加就加 | **不做。** 这是全计划唯一一个压在关键路径上、工程尾巴无界（要 `chat_model.generate` 在线吐 per-token 熵）却按计划自己的表就属"可选"的项目。定性对照 + 注明不可比。若 E5 过了 P1/P2 且 9/13 后有余量再议 |
| Q5 | D0：防御线的两个 GPU-天是否花 | 花 | **花**，同意。补一条：D1 的价值不对称——"换模型也没用"是能直接进正文的一句话，"换模型就有"会开一条 9/24 前收不了口的线。两种结果都只值第十节一句话，所以跑，但**只用队列空档，永不排在 ERGO 作业之前** |
| Q6 | 9/13 合流截止是否接受 | 接受 | **接受** |
| **Q7（E0 新增）** | `pad_short_history` 取什么值——决定 MPC 第一次真决策是 turn 1 还是 turn 2 | 计划未提 | **`True`**，写进臂名。见 C1；**现在定死，不许在 G-E3-4 之后改** |
| **Q8（E0 新增）** | 模式 B 下 P2 的同代价对手是谁——mpc 花 k=2，`fixed_last` 只花 1 | 计划写的是"含 `fixed_last`" | **P2 的对手是 `fixed_t_and_last` 中最好的那个（同为 2 次）**；`fixed_last` 降级为 P2b 的更便宜参照，mpc 至少要追平它，但**赢过它不构成 S1 证据** |

---

## 十、新会话启动语（直接粘贴）

**每条启动语都要求同时读第十二节**（E0 复核）——它改写了下面每个任务的规格。

> **E1**：读 `docs/experiments/two_task_success_plan.md` 第二节 E1 **与第十二节** 与 `src/persona_drift/ergo_math_trajectory.py`、`scripts/run_ergo_math_screening.py`、`scripts/analyze_ergo_phaseC_comparison.py`（只读，作为新分析器的骨架来源）。实现 `reset_mode`（4 条单测）**并新建 `scripts/analyze_ergo_append_comparison.py`**（B1，≥3 条单测），跑全套测试，报 G-E1-1 与 G-E1-2（overwrite 下 `--identity always_reset,fixed_last` 必须复现 116/116 = 1.000）。不提交任何 GPU 作业。

> **E2**：读第二节 E2 **与第十二节**。建 3 个 sbatch（照抄 `environment/run_ergo_phaseC_*.sbatch`，只改四处，**不要动 `--agent-max-new-tokens`**），提交，等 `COMPLETED`，跑 **`analyze_ergo_append_comparison.py`**（不是 `phaseC` 那个，见 B1），按第八节格式报 G-E2-1/2/3 与 C2 的 prompt token 最大值。`zero_control` 复用前断言 item 集合相同。

> **E3**：读第二节 E3 **与第十二节**。Step 1 提交 Phase B′；跑完后 Step 2–4 全部 CPU（Step 3 新建 `scripts/analyze_ergo_state_action_interaction.py`），报 G-E3-0..3；**G-E3-3 落在近失区间就停下报告，不自行选模式**；Step 5 用 E4 的控制器做离线回放，报 G-E3-4。不提交 E5。

> **E4**：读第二节 E4 **与第十二节** 与 `src/persona_drift/ergo_koopman_mpc.py`、`control.py`（只读）。先确认第 1/4 项已由 F3/F0 做完（别重写），再实现 `objective="terminal"`（含 B4 的 `repeat_penalty=0.0` 与平手取 0）、`forced_last_reset`（按 B3 的两条分支写死）、`pad_short_history=True`（C1），**并新建 `src/persona_drift/ergo_controllers.py` 实现 `fixed_t_and_last` 与 `randsched_t_and_last`**（B2）。≥10 条单测，报 G-E4-1。不改 `control.py`。

> **E5**：读第二节 E5 **与第十二节** 与 E3 的 G-E3-3 裁决。按对应模式建臂（P2 的对手按 Q8）、提交、分析，报 P1/P2/P2b/P3。不解释结果。

> **Opus 复核（新会话）**：只读 `<任务> handoff` 与产物 JSON，独立重算每个闸门数字，写"复核"段，出裁决与下一步是否开工。不读 Sonnet 的过程叙述。

---

## 十一、小结

ERGO 线五层里已有四层站住，唯一没站住的第 0 层根因是 reset 覆盖历史这一行代码，本计划以 append 变体为主攻，用四道预注册闸门（权威保留、恒等式打破、reset 效应依赖状态、控制器不退化成固定臂）决定是否投 Phase C′，全部 GPU 约 8 小时、9/12 前出结果。防御线三个前提在现有设定下全部失效且截断已被排除，只给两个 GPU-天判定换目标模型能否让前提 1–2 成立，闭环不再追。无论结果如何，两条线的 S3 版本（前提检验程序 + 分层诊断）已经是可成稿的正文，论文按 S3 建契约、9/13 合流、9/18 摘要。

**E0 复核（第十二节）之后的净变化**：事实基础全部核实（含 116/116 恒等式的独立重算），四处会让闸门算不出来或让强制动作静默失效的规格缺口补进 E1/E4，`pad_short_history` 与 terminal 的 `repeat_penalty`/平手规则改为跑前预注册，模式 B 的 P2 对手改为同代价的 `fixed_t_and_last`，G-E3-3 阈值近失时不许自行选模式。新增工作全在 9/8–9/9 的 CPU 侧，**GPU 总量与 9/12 出结果这两个点不变**。

---

## 十二、E0 复核与裁决（Opus 5，2026-09-07）

**结论**：计划的事实基础全部成立，代码定位精确到行，**签字放行**。四处规格缺口（B1–B4）会让 E2/E5 在提交后算不出闸门或让强制动作静默失效，全部是零 GPU 就能补的，已并入 E1/E4 的范围。两条预注册项（C1–C2）与两个新判断题（Q7–Q8）已写进对应各节。

### 12.1 独立重算过的事实

| 计划的断言 | 复核结果 |
|---|---|
| `ergo_math_trajectory.py:114` 是覆盖历史那一行 | ✅ 精确到行；112–117 的补丁形状与现码一致，`overwrite` 分支逐字节不变 |
| overwrite 下 `fixed_last` ≡ `always_reset`，116/116 | ✅ **独立重算**：末轮 `judge_raw_output` **116/116 = 1.000**，`agent_message` 亦 **116/116**，per-item 均值同为 0.775862，`gate6` diff = 0.0 CI [0, 0]。这是全计划最强的单个事实，是真的 |
| 分层表的 −0.448 / +0.425 | ✅ 前者 `defense_line_redesign_plan.md:545`（退化效应），后者 `:62`（60-item 首个权威实验）。Phase C 内的 `always_reset − zero_control` 是 0.3276 → 0.7759 = **+0.448**，与 +0.425 是两个不同实验，不是笔误 |
| 58 held-out item / Phase B 666 行 / 基线 406 passed | ✅ `conf/experiment/ergo_phaseC_item_ids.txt` 58 个；666 行；416 collected（406 + 5 NLTK + 5 skip） |
| 7 个分析脚本、Phase B/C sbatch、`--item-ids` | ✅ 全部存在（`--item-ids` 在 `run_ergo_math_screening.py:62`，Phase C sbatch 从 `conf/experiment/ergo_phaseC_item_ids.txt` 读入） |
| E4 第 1、4 项（视野裁剪、预算读 `u_reset`、`shard_frac` 真值覆盖） | ✅ **已经做完**：`control.py:290` 按 `episode_length` 裁剪；`ergo_koopman_mpc.py:118` 每轮设 `episode_length = num_shards`；`:100` 覆写 `_remaining_budget` 读 `u_col`（F3 的 bug 已修）；`:131` 真值覆盖在位。E4 只需确认 + 单测 |
| "穷举 ≤ 4096 条日程，毫秒级" | ✅ `_simulate` 的递归枚举在 T=12 时 ≤ 2^12 叶子/轮、约 8k/轨迹。不需要 DP |

### 12.2 四处规格补丁（已写进 E1 / E4）

**B1 · 分析器算不出任何一道新闸门。** `analyze_ergo_phaseC_comparison.py:36` 把 `ARM_DIRS` 硬编码在 `outputs/ergo_math_phaseC_*`，闸门集合固定为 gate1–gate6。G-E2-1（always−zero）、G-E2-2（恒等式占比）、P1/P2/P3、模式 B 的臂**一个都不在里面**，而 E1 原来的"只这三处"又把它排除了。→ 新增 `scripts/analyze_ergo_append_comparison.py`，并入 E1；加 G-E1-2 把 116/116 这个参照基线钉在代码里。

**B2 · 模式 B 的两个对手臂不存在。** `CONTROLLER_CHOICES` 只有 7 个；`fixed_schedule` 只吃绝对轮次，`fixed_last` 只吃 per-item 末轮。**没有** `fixed_(t,last)`，也没有 `randsched_(t~U,last)`。若 G-E3-3 路由到模式 B，E5 的臂建不起来。→ 新增 `ergo_controllers.py`，并入 E4，且 E4 提前到与 E1 并行（G-E3-3 的裁决可能 9/10 才出，届时再写就压线）。

**B3 · `forced_last_reset` 是 F3 那个 bug 的形状。** 原文只写"`_remaining_budget` 相应减 1"。若均匀减 1，到 `turn == num_shards` 时父类 `control.py:315` 看到 `remaining_budget <= 0` 直接 `return 0`，**末轮强制 reset 永远不会发生**。→ 两条分支写死（`turn < T`：`k − spent − 1`；`turn == T`：无条件 1，绕过预算检查），单测断言"末轮 `u_reset=1`"而不只是"总数 = k"。

**B4 · terminal 目标改变了 `repeat_penalty` 的语义，平手规则未定。** 终值目标下中间动作不再承担任何惩罚；而 `control.py:319` 的 `if value > best_value` 在精确平手时偏向 action=0。于是"终值读出对早期位置不敏感"会直接表现为"永不早花"——正是 G-E3-4 要抓的退化。→ 预注册 terminal 臂 `repeat_penalty=0.0` 与"平手取 0"，使 G-E3-4 不过时只能读作"没有状态依赖"，不留"旋钮没调对"的解释空间。

### 12.3 两条预注册项

**C1 · 最早决策轮（Q7）。** `ergo_koopman_mpc.py:78` 的 `_current_state` 在历史不足 `max(nu−1, mu−shift)+1` 时返回 `None`；`contemporaneous_v=True, nu=mu=1` 下第一次真决策在 turn 2，turn 1 恒为 0。而第七节讲的"早 reset 防止锚定"有一部分住在 turn 1。`pad_short_history` 这个开关正是为此存在，计划一次都没提。→ 取 `True`，写进臂名，现在定死。

**C2 · append 让 prompt 单调膨胀。** `always_reset_append` 每轮在保留的历史之上再追加一遍完整合并题面：12 × (题面 + 512 token 回复) ≈ 8–9k token。Qwen3-4B 装得下，但**防御线 B1 就是一次截断混淆**。→ E2 加逐轮 prompt token 计数 + 上限断言 + 汇报最大值。零成本，先关掉这个口。

### 12.4 对 ERGO 线的实质风险：模式 A 近乎预定只能到 S2

overwrite 下最优固定臂是 `fixed_last` = 0.776（vs `zero_control` 0.328），且以 152.8 vs 696.1 的 token 代价打平 `always_reset`。**若 append 保留了这个次序的任何一部分，末轮对几乎每个 item 都接近最优，自适应就没什么可分配的。** 这正是模式 B 存在的理由——但原计划的路由是"过 +0.15 且 p<0.05 → 模式 B，否则模式 A"，而模式 A 的对手集合里留着 `fixed_last_append`，等于让模式 A 近乎预定只能到 S2。

**裁决**：G-E3-3 照原样跑，但**阈值近失（差 ∈ [0.10, 0.15) 或 p ∈ [0.05, 0.10]）时停下报告，不许自动落回模式 A**。阈值近失更像"末轮优势仍占主导"（→ 模式 B）的证据，而不是"没有优势"（→ 模式 A）的证据。这条已写进 E3 Step 4 的判定表与失败模式 10。

### 12.5 签字

E1（含 B1）与 E4（含 B2/B3/B4/C1）可即刻开工，两者零 GPU、可并行。**E2 的 GPU 作业在 G-E1-1 与 G-E1-2 过之后提交**；其余闸门链与依赖不变。GPU 总量约 8 小时、9/12 前出结果这两个点不变；新增工作吃掉的是原有缓冲。

未由我裁决、仍属用户的一项：**ICLR 2027 的档期**。9/18 摘要 / 9/24 全文与 ICLR 历年的 9 月中下旬吻合，但我核不了官网，而整张日程只压在这一个数上。
