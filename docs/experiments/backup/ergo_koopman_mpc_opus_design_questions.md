# 设计问题：ERGO 线 Phase C（`KoopmanMPCController` 闭环）

> 🗄 **已归档（只读）**。这两个设计问题已经裁决：`shard_frac` 展望用真值覆盖（且已从
> "建议"升级为"必须"）、reset 加 `k=1` 预算并用实测 token 代价论证——见
> [`../signal_resolution_plan.md`](../signal_resolution_plan.md) 4.1 / 4.2。
> **不要再把本文档当成"等 Opus 裁决"的开放状态。**

**状态**：待 Opus 规划，Sonnet 5 不擅自裁决。**写于**：2026-09-07。
**目的**：ERGO/Laban 多轮可靠性侵蚀线（`docs/experiments/ergo_multiturn_reliability_pilot.md`）
Phase B（开环拟合）已经跑完并判断"值得往下投"（见该文档"拟合结果"小节）。Phase C 是照防御线
Phase B→C 的模板，把拟合出的 A/B 包进 `KoopmanMPCController` 跑闭环、和固定基线对比。这份文档
只记两个/三个需要 Opus 判断的设计问题——**不是执行计划**，Sonnet 5 已经做完了不需要判断的
部分（见下面"已完成"一节），剩下的部分因为涉及会实质改变结果解释的设计选择，主动停下来问。

**如果你是 Opus，接手方式**：读完这份文件应该就够，不需要重读整条 pilot 文档，除非要核对
Phase B 拟合的具体数字（那份文档"拟合结果"小节已经摘录了关键数字）。请针对下面两/三个问题
给出**具体规格**（不是"考虑一下这个方向"那种开放式建议）——Sonnet 5 接手时会照抄执行，不会
自己再做设计判断，格式可以参照 `independent_judge_reactive_rerun_plan.md` 那种"照抄，不要
增删"的写法。

---

## 已完成（不需要 Opus 看）

出于"避免干扰 `defense`/`stance` 两条线共用的 `control.py`/`controller_cli.py`"这个明确要求，
新加了一个**继承**出来的子类，零改动共用代码：

- `persona_drift_control/src/persona_drift/ergo_koopman_mpc.py`：
  `ErgoKoopmanMPCController(KoopmanMPCController)`，只覆写 `_current_state`——加了可配置
  `y_col`/`u_col`（ERGO 用 `y_task_success`/`u_reset`，不是人格漂移的 `y_probe`/`u_remind`）
  和 `aux_fns`（`control.KoopmanMPCController` 完全没有 `aux_cols` 支持，这里补上，状态里
  `aux_now` 的拼接顺序和 `modeling/dataset.py::build_reduced_state_pairs` 一致：
  `y_hist + v_hist + aux_now`）。`shard_frac(row)` 辅助函数直接从 `row["turn"]`/
  `row["num_shards"]` 算，不需要行上预先有 `shard_frac` 列。
  `load_ergo_koopman_mpc_controller(...)`：仿 `controller_cli.load_koopman_mpc_controller`，
  把 `contemporaneous_v=True`、`aux_cols=("shard_frac",)` 写死（这两个是 Phase B 已经定下的
  必要设计，不是消融开关，见 pilot 文档"Koopman 建模，Phase B"一节）。
- `persona_drift_control/tests/test_ergo_koopman_mpc.py`：7 个新 CPU 单测（状态拼接顺序、
  可配置列名、reminds-when-it-helps/repeat-penalty-dominates 这两个和防御线对称的行为、
  insufficient-history 兜底、loader 从 `koopman_fit_report.json` 建出一个能跑的控制器），全绿。
  全套测试 401 passed（同一批 5 个 NLTK 缺数据失败，无关，无新增失败）——
  `git status --short src/persona_drift/control.py src/persona_drift/controller_cli.py`
  确认这两个共用文件没有被碰。

**`_simulate`/`next_u_remind`/`_remaining_budget`/`_planning_steps` 全部原样继承，一行没改**——
这正是下面问题 1 的来源：继承下来的 `_simulate` 让 `shard_frac` 那一维状态的多步展望
完全交给拟合出的 `A`/`B`/`b` 那一行自由外推，没有做任何针对性处理。这是刻意搭的一个"能跑但
naive"的占位版本，不是遗漏——它诚实但可能不是该问的正确问题。

---

## 问题 1：MPC 展望时，`shard_frac` 该不该被当成"预测对象"

**背景（Phase B 拟合出的具体数字，见 pilot 文档）**：3 维状态
`z=[y_lag, u_reset_lag, shard_frac]`（`nu=1, mu=1, aux_cols=("shard_frac",)`,
`contemporaneous_v=True`）。`shard_frac` 那一行的拟合系数是
`A[2]=[-0.0295, -0.00118, 1.00508]`、`B[2]=-0.00236`、`b[2]=0.1757`——自身持续系数
`≈1.005`（略超 1，孤立看不稳定）、来自 `u_reset` 的驱动幅度接近 0（符合预期：reset 不该
控制这个外生变量），但这一行本质是**跨 60 个 item 的线性回归**，而每个 item 的
`num_shards`（分母）不同——真实的 `shard_frac_(t+1) - shard_frac_t = 1/num_shards`
是一个**每个 item 不同、但在单个 item 内部完全确定、不需要预测**的量，线性模型只能给出
"平均意义下"的近似增量，逐 item 会有系统性误差，且继承下来的 `_simulate` 是**递归多步**
展望（`horizon` 步），这个误差会不会在 2-3 步内复利放大到影响 MPC 的动作选择，没有测过。

**两个候选做法**（都没实现，等规格）：

- **(a) 展望时用真值覆盖**：`_simulate` 递归时额外传入当前 lookahead 的真实 turn 号，每步
  算完 `z_next` 后把 aux 分量强制替换成 `(turn+k)/num_shards`（`num_shards` 从
  `history[-1]["num_shards"]` 读，同一 item 内不变），只让 `A`/`B`/`b` 处理 `y`/`v_lag` 两个
  真正需要预测的维度。改动面：`ErgoKoopmanMPCController` 需要再覆写 `_simulate`（继承的签名
  是 `_simulate(self, z, action, remaining_steps, remaining_budget)`，不带 turn 号，需要加一个
  参数或者用一个 helper 包一层）。
- **(b) 保持 naive（现状）**：不覆写 `_simulate`，接受"平均意义下"的近似，理由可以是
  "3 步以内的 `A[2,2]=1.005` 误差量级本来就小，没必要为了一个外生变量单独开一条特殊路径"——
  但这需要先算一下 3 步复利误差的实际量级（比如 20 个不同 `num_shards` 的 item 上，naive
  展望 vs 真值展望的 `shard_frac` 预测偏差有多大）才能说这话有根据，不是拍脑袋。

**请求**：Opus 看完这两个选项后给出裁决——选哪个、如果选 (a) 具体的 `_simulate` 签名怎么改、
如果选 (b) 需不需要先跑一个离线量化误差量级的检查（如果需要，是否也算在这份规格里让
Sonnet 去做）。

---

## 问题 2：Phase C 要不要引入 reset 预算，如果引入，预算约束的理由是什么

**背景**：Phase A/B 数据里 `reset` 这个动作**目前看不出任何下行代价**——
`judge_parse_failure_rate`/`refusal_rate` 两臂都很低、没有异常（见 pilot 文档 20-item/60-item
两节的"诊断"段落）。60-item 权威检查本身就是"每轮都 reset"的固定臂（`--controller
constant_remind`ish 全 1 日程）赢了 zero_control 一大截，也就是说**目前唯一验证过的最优固定
策略就是"永远 reset"**——防御线 unbudgeted 情形下踩过同一个坑（`docs/experiments/
koopman_defense_pilot.md`："unbudgeted 的最优策略是 trivially always remind，没有分配问题
可比"），如果 ERGO 线不加预算约束，MPC 唯一能学到的最优策略大概率还是"永远 reset"，
和已有的固定基线**逐字相同**，跑不出任何新结论。

**但防御线加预算约束是有理由的**（reminder 会占用上下文预算/有"提醒疲劳"的隐含代价，
`docs/next_step_diagnosis.md` 第四节步骤 2）——**ERGO 线现在还没有类似的自然代价论证**。
如果凭空加一个预算（比如"最多 reset `num_shards//2` 次"），得回答：这个约束在叙事上站得住吗，
还是会被读成"为了制造一个能赢的设定而人为限制基线"？`docs/README.md` 记录过的诚实性红线
（"打不赢等代价 periodic 必须进正文"）同一逻辑适用在这里——如果预算是编出来的,那"MPC 在
预算下赢了固定日程"这个结论的说服力也是编出来的。

**附带一个纯工程子问题（不一定需要 Opus，但因为和上面这个问题耦合就一并列出）**：
`KoopmanMPCController.episode_length` 是构造时定死的一个标量,而 ERGO 每个 item 的
`num_shards`（也就是这条轨迹的自然总轮数）不同——如果要用 `episode_length` 做"轮次数意义下
的预算规划"（`_planning_steps` 用它裁剪 horizon），需要从 `history[-1]["num_shards"]` 每条
轨迹动态算,不能像防御线那样在 CLI 传一个全局常数。这部分本身是机械改动,不需要 Opus
设计——**除非** Opus 认为"预算"这个框架本身对 ERGO 不成立(问题 2 主问题的裁决结果决定要不要做
这个工程改动)。

**请求**：Opus 判断 ERGO 线是否需要一个有叙事依据的 reset 预算（如果需要，给出具体数值/比例
和一句可以直接写进论文的代价论证；如果不需要，说明"不加预算、直接对比 MPC vs 固定基线在
unbudgeted 设定下谁的**展望策略**更好"这个替代框架下 MPC 到底还能比什么——比如是不是该比较
"更早、更少次数地达到同等最终成功率"而不是"最终成功率本身"，这样"效率"而不是"上限"才是
比较维度,类似防御线 Phase J 也曾经想用"提醒次数更省"这个维度)。

---

## 下一次接续时

Sonnet 5 在 Opus 给出上面两/三个问题的规格之前，不会：
- 改 `_simulate`/加预算逻辑/接 `run_ergo_math_screening.py` 的 `koopman_mpc` CLI 分支；
- 提交任何 GPU 作业。

规格写好之后照抄执行即可，`ErgoKoopmanMPCController`/`load_ergo_koopman_mpc_controller`/
7 个单测已经是可以在其上直接加代码的起点，不需要重新设计状态拼接这部分。
