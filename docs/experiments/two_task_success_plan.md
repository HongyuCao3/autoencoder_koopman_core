# 执行计划：两条任务线上争取 Koopman 正面结果（ERGO-append 主攻 + 防御线时间盒）

**日期**：2026-09-08 · **起草**：Hongyu Cao（与 Claude）· **适用**：Opus 5（裁决）/ Sonnet 5（执行）/ `general-purpose` subagent（盲标）
**建议存放**：`docs/experiments/two_task_success_plan.md`，并在 `docs/README.md` 的"实验"一节登记为**活计划**。
**依据**：截至 commit `ae167a5`（2026-09-07 17:23）的全部结果——`defense_line_redesign_plan.md` §9–12、`ergo_multiturn_reliability_pilot.md` G4、`adaptive_vs_fixed_claim_plan.md` §11–15、`signal_resolution_plan.md`、`measurement_validity_plan.md`、`article/PAPER_EXECUTION_PLAN.md` §1–2。
**时间假设**：ICLR 2027 摘要截止约 9/18、全文约 9/24（**请核对**；下文日程按此排）。

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

---

## 零、目标：先把"成功"写成可判定的分级

用户目标是"Koopman 在两条任务上都有正面表现"。直接以这句为目标会诱导调参找结果，所以先把它拆成可判定的等级，**跑之前写定，跑完不改**：

| 等级 | ERGO 线 | 防御线 |
|---|---|---|
| **S1 正面闭环** | 同代价下 Koopman-MPC 的最终轮成功率对**随机分配**配对差 CI>0，且对**最优固定日程**（含 `fixed_last`）不显著更差，且（若有该臂）对 ERGO 熵阈值不劣 | 在新设定（第四节）下，Koopman-MPC 对同代价随机分配 CI>0 |
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

### E0 · Opus 签字（零成本，9/8）

确认第九节 6 个判断题，尤其是 append 的具体形式与预算模式的预注册规则。**没有 E0 的签字，E1 之后的任何 GPU 作业不得提交。**

### E1 · `reset_mode="append"`（Sonnet，代码 + 单测，半天，零 GPU）

**改动范围**（只这三处）：

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

**闸门 G-E1-1**：CPU 全套测试通过（基线 406 passed，同一批 5 个 NLTK 失败无关）；`git diff --stat` 只涉及上述三个文件。

### E2 · append 下的权威与恒等式检验（Sonnet 提交，3 个 GPU 臂，约 1.5 小时）

**臂**（58 个 held-out item × seeds 0 1，与 Phase C 逐字相同的 `--item-ids`）：

| 臂 | 控制器 | 输出目录 |
|---|---|---|
| `zero_control` | **复用** `outputs/ergo_math_phaseC_zero_control/`（u≡0，与 reset_mode 无关，逐字节相同，不重跑） | — |
| `always_reset_append` | `constant_remind` + `--reset-mode append` | `outputs/ergo_appendC_always_reset/` |
| `fixed_last_append` | `fixed_last` + `--reset-mode append` | `outputs/ergo_appendC_fixed_last/` |
| `fixed_t2_append` | `fixed_schedule` turns=(2,) + `--reset-mode append` | `outputs/ergo_appendC_fixed_t2/` |

sbatch 照抄 `environment/run_ergo_phaseC_*.sbatch`，只改 job-name / 日志 / `--reset-mode append` / `--output-dir`。

**预注册闸门**（按 item 配对，58 对，bootstrap 10000，`default_rng(0)`，用 `analyze_ergo_phaseC_comparison.py`）：

| 闸门 | 判据 | 过 → | 不过 → |
|---|---|---|---|
| **G-E2-1 权威保留** | `always_reset_append − zero_control` 的 95% CI 下界 > 0 | 继续 | **ERGO 线停在 S3**：append 把权威也抹掉了，写进 G4.5 |
| **G-E2-2 恒等式打破** | `fixed_last_append` 与 `always_reset_append` 末轮抽取答案逐字相同的轨迹占比 **< 0.90**（overwrite 下是 116/116 = 1.00） | 继续 | 停：追加没有让历史进入输出，报告并等 Opus 判断是否换 append 形式（第九节 Q1） |
| **G-E2-3 记录项** | `fixed_last_append − fixed_t2_append`、`always_reset_append − fixed_last_append` 的差与 CI | 只记录，不判定 | — |

**预注册预期**：G-E2-1 大概率过（合并题面仍在 prompt 里）；G-E2-2 不确定——这是本计划最早的真正未知。

### E3 · append 下的辨识数据、读出闸门与两道离线门槛（Sonnet 提交 1 个 GPU 臂 + CPU 一天）

**Step 1 · Phase B′ 采集**：`environment/run_ergo_phaseB_random_excite.sbatch` 照抄一份，加 `--reset-mode append`，`--item-rng-seed 2`（与 Phase B 同一批 60 item），`--random-excite-p 0.5`，seeds 0 1，输出 `outputs/ergo_appendB_random_excite/`。
**闸门 G-E3-0**：`COMPLETED 0:0`；行数与 Phase B 相同（666）；逐轮检查两种动作在每个 `shard_frac` 十等分箱里都出现过（任一箱只有单一动作 → 报告，不自行重采）。

**Step 2 · 读出闸门**（CPU，秒级）：`scripts/analyze_ergo_readout_state.py`（F0 修正版）与 `analyze_ergo_closeness_readout.py` 指向新目录，跑 RC-0..3。
**闸门 G-E3-1**：`closeness` 四条全过（判据定义沿用 `signal_resolution_plan.md` 0.5 节，不改阈值）。

**Step 3 · 拟合**（CPU）：`fit_koopman_ergo_closeness.py` 指向新目录 → `koopman_fit_report_closeness_append.json`。**额外拟合一个交互模型**：在同一 45/15 item split 上，回归

$$
c_{t+1} = a\,c_t + b_0\,u_{t+1} + b_2\,u_{t+1}\,c_t + g\,\text{shard\_frac}_{t+1} + \text{const}
$$

直觉：自适应能赢的前提是"reset 的效果取决于现在卡得多深"，$b_2$ 就是这一项。式中 $c_t$ 是第 $t$ 轮的 closeness，$u_{t+1}$ 是下一轮是否 reset（`contemporaneous_v=True` 的语义），$b_0$ 是 reset 的基础效应，$b_2$ 是效应随 closeness 的变化，$g$ 是信息累积斜坡。复用 `analyze_state_action_interaction.py` 的回归骨架（防御线 Phase H 用过），按 item 做 1000 次 bootstrap。

**闸门 G-E3-2 状态依赖**：$b_2$ 的 95% CI 不含 0，且符号为负（越接近答案 reset 越没用）。不过 → 最优策略仍是常数规则，**ERGO 线停在 S3**，写"reset 效应与状态无关"。

**Step 4 · 预算模式判定**（CPU）——照 `measurement_validity_plan.md` 0.4 节的方法，在 Phase B′ 数据里按"reset 之后还剩几轮"分层看最终轮成功率：

**闸门 G-E3-3 退化判定**：

| 结果 | 预算模式（预注册） |
|---|---|
| "剩 0 轮"层比其余各层合计高 ≥ 0.15，且 Spearman(剩余轮数, 成功率) p<0.05 | **模式 B：k=2，末轮 reset 固定**。末轮那次不是决策（已知最优），自适应只决定**第一次**放哪。对手臂改为 `fixed_(t,last)`、`randsched_(t~U,last)` |
| 否 | **模式 A：k=1 计数预算**，臂表同 Phase C 加 `fixed_last_append` |

不引入 token 预算（G4.3(3)：低于一次末轮重述的 token 预算是人为稀缺）。

**Step 5 · 离线回放准入**（CPU）：用 E4 的控制器在 Phase B′ 的 60 条轨迹上做离线回放（同 T1b Step 4 的方法）。
**闸门 G-E3-4**：`n_distinct_spend_turns ≥ 3`，且任一轮次的花费占比 ≤ 0.60。不过 → 控制器退化成固定臂，**不提交 E5**，报告并回到 E4 检查目标函数/视野（只允许改这两项，且改后重新过 G-E3-4）。

### E4 · 全视野、终值目标的 MPC（Sonnet，代码 + 单测，一天，零 GPU；可与 E3 Step 1 并行）

只改 `src/persona_drift/ergo_koopman_mpc.py` 与 `run_ergo_math_screening.py` 本地分支：

1. **视野到 $T$**：`--koopman-horizon 12`；确认 `_planning_steps` 用逐轨迹的 `episode_length=num_shards` 裁剪（`ergo_multiturn_reliability_pilot.md` 第 6 条已加）。动作二值、$T\le 12$，穷举 ≤ 4096 条日程，毫秒级；若实测慢，改为对 `remaining_budget` 做 DP。
2. **目标函数**：加 `objective: str = "terminal"`；覆写 `_simulate`，`"terminal"` 只累计最后一步的 `readout`，`"sum"` 保留父类行为作消融。Phase C 退化成 `fixed_t2` 的直接原因是 horizon=2 + 逐轮和，两处都必须改。
3. **模式 B 支持**：`forced_last_reset: bool`。为真时规划器把 $u_T=1$ 当作已知常量，预算 `k=2` 里只有一次可分配；`_remaining_budget` 相应减 1。
4. `shard_frac` 展望真值覆盖保持（F3 4.1 已做）。
5. 单测 ≥ 6 条：terminal 与 sum 在手工小例子上给出不同决策；horizon 裁剪到 `num_shards`；forced_last 下末轮必 reset 且中途最多 1 次；`reset_mode` 不影响控制器；预算读 `u_reset` 列（锁定 F3 那个 bug）。

**闸门 G-E4-1**：测试全绿；`git diff --stat` 不含 `control.py` / `controller_cli.py`。

### E5 · Phase C′（Sonnet 提交，6–8 个 GPU 臂，约 4 小时）

58 held-out item × seeds 0 1，全部 `--reset-mode append`。臂表按 G-E3-3 的模式：

| 臂 | 模式 A | 模式 B |
|---|---|---|
| `zero_control` | 复用 | 复用 |
| `always_reset_append` | 复用 E2 | 复用 E2 |
| `fixed_last_append` | 复用 E2 | 复用 E2（= 只有末轮那次） |
| 固定日程 | `fixed_t1..t4_append` | `fixed_(t,last)_append`，t=1..4 |
| 随机等代价 | `randsched_p100_append` | `randsched_(t~U[1,T−1],last)_append` |
| **被测臂** | `mpc_terminal_append`（k=1） | `mpc_terminal_forcedlast_append`（k=2，末轮固定） |
| 消融（可选，Opus 定） | `mpc_sum_append` | 同 |
| ERGO 熵阈值（可选，Opus 定，见 Q4） | `ergo_entropy_threshold_append` | 同 |

**预注册主判据**（按 item 配对，bootstrap 10000，`default_rng(0)`；主指标 `final_turn_success`，次指标 turn≥3 的 `closeness` 均值只记录不判定）：

| 闸门 | 判据 | 含义 |
|---|---|---|
| **P1 主** | `mpc − randsched` 的 95% CI 下界 > 0 | 看状态选时机优于不看状态随便选 → **S2 起步** |
| **P2** | `mpc − 最优固定日程`（含 `fixed_last`）的 CI 包含 0 或下界 > −0.05 | 不劣于事先知道最好一轮的先知 → 与 P1 合起来是 **S1**（若有熵阈值臂，还需 `mpc − ergo_threshold` CI 下界 ≥ −0.05） |
| **P3 诊断** | mpc 花费轮次的分布；按 mpc 自选轮次拆开的配对差（同 Phase J 11.5 的表） | 解释赢/输在哪 |

P1 不过 → **ERGO 线停在 S3**，Phase C′ 作为"前提全满足仍无自适应收益"的更强负结果写进论文；**不调 horizon、不换目标函数重跑**。

### E6 · 汇报与回写（Sonnet 半天 + Opus 半天）

按第八节格式回写 `ergo_multiturn_reliability_pilot.md` 新一节"E1–E5：append 执行器下的闭环"；Opus 独立重算后在 `paper/evidence/` 走台账，并按第五节改论文叙事。

---

## 三、ERGO 线的成本与日程

| 任务 | 类型 | 预计 | 依赖 |
|---|---|---|---|
| E0 | Opus 裁决 | 9/8 | — |
| E1 | 代码 | 9/8 | E0 |
| E2 | 3 GPU 臂 | 9/9 上午提交，1.5 h | E1 |
| E3 Step 1 | 1 GPU 臂 | 9/9 下午（G-E2-1/2 过后） | E2 |
| E4 | 代码 | 9/9–9/10（与 E3 并行） | E1 |
| E3 Step 2–5 | CPU | 9/10 | E3 Step 1, E4 |
| E5 | 6–8 GPU 臂 | 9/11 提交，约 4 h | G-E3-4 |
| E6 | 回写 + 复核 | 9/12 | E5 |
| 缓冲 / 消融 | — | 9/13–9/14 | — |

GPU 总量约 8 小时，全部在 9/12 前出结果，留 6 天进论文。

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

## 九、E0 需要 Opus / 用户裁决的判断题

| # | 问题 | 建议 |
|---|---|---|
| Q1 | append 的具体形式：(a) 只追加合并题面；(b) 追加合并题面 + 一句"忽略你此前的尝试，重新求解" | **(a)**。(b) 引入第二个执行器变量，E2 若不过无法归因 |
| Q2 | 主指标仍用 `final_turn_success`，还是换 turn≥3 的 `closeness` 均值 | **保留 `final_turn_success` 为主**（与 ERGO 可比），`closeness` 均值只记录。append 后终点不再被单个动作决定，G4.3(2) 的反对理由消失 |
| Q3 | D1 的备选目标模型 | **Llama-3.1-8B-Instruct**：ERGO 与 NBF 论文都用过，数字可对照；Qwen2.5-7B 为备选 |
| Q4 | 是否实现在线 ERGO 熵阈值臂 | 需要 `chat_model.generate` 在线返回 token 熵（`analyze_ergo_entropy_readout.py` 是事后前向）。若一天内能加且过单测就加；否则 E5 不含此臂，论文里对 ERGO 只作定性对照并注明不可比 |
| Q5 | D0：防御线的两个 GPU-天是否花 | 建议**花**——D1 的"换模型也没用"或"换模型就有"都是论文能用的一句话，且不与 ERGO 抢队列 |
| Q6 | 9/13 合流截止是否接受 | 按截止倒推的最晚时间；再晚 Tier 3 来不及重生成 |

---

## 十、新会话启动语（直接粘贴）

> **E1**：读 `docs/experiments/two_task_success_plan.md` 第二节 E1 与 `src/persona_drift/ergo_math_trajectory.py`、`scripts/run_ergo_math_screening.py`。实现 `reset_mode`，加 4 条单测，跑全套测试，报 G-E1-1。不提交任何 GPU 作业。

> **E2**：读第二节 E2。建 3 个 sbatch（照抄 `environment/run_ergo_phaseC_*.sbatch`，只改四处），提交，等 `COMPLETED`，跑 `analyze_ergo_phaseC_comparison.py`，按第八节格式报 G-E2-1/2/3。`zero_control` 复用前断言 item 集合相同。

> **E3**：读第二节 E3。Step 1 提交 Phase B′；跑完后 Step 2–4 全部 CPU，报 G-E3-0..3；Step 5 用 E4 的控制器做离线回放，报 G-E3-4。不提交 E5。

> **E4**：读第二节 E4 与 `src/persona_drift/ergo_koopman_mpc.py`、`control.py`（只读）。实现 `objective`/`forced_last_reset`/全视野，≥6 条单测，报 G-E4-1。不改 `control.py`。

> **E5**：读第二节 E5 与 E3 的 G-E3-3 裁决。按对应模式建臂、提交、分析，报 P1/P2/P3。不解释结果。

> **Opus 复核（新会话）**：只读 `<任务> handoff` 与产物 JSON，独立重算每个闸门数字，写"复核"段，出裁决与下一步是否开工。不读 Sonnet 的过程叙述。

---

## 十一、小结

ERGO 线五层里已有四层站住，唯一没站住的第 0 层根因是 reset 覆盖历史这一行代码，本计划以 append 变体为主攻，用四道预注册闸门（权威保留、恒等式打破、reset 效应依赖状态、控制器不退化成固定臂）决定是否投 Phase C′，全部 GPU 约 8 小时、9/12 前出结果。防御线三个前提在现有设定下全部失效且截断已被排除，只给两个 GPU-天判定换目标模型能否让前提 1–2 成立，闭环不再追。无论结果如何，两条线的 S3 版本（前提检验程序 + 分层诊断）已经是可成稿的正文，论文按 S3 建契约、9/13 合流、9/18 摘要。
