# 设计笔记：Koopman-MPC 决策的 case 分析（回应 Phase G 的自适应性开放问题）

和 [koopman_detection_design.md](koopman_detection_design.md) 同一类"供跨会话接续"的记录。
**新开一次对话想知道"case 分析这条线想验证什么、怎么做的、做到哪一步了"，看这份文档。**

## 背景与问题设定

[koopman_defense_pilot.md](koopman_defense_pilot.md) 的 Phase G 用完全对齐插入次数/token 代价
的 `PeriodicController`（固定周期插入提醒，不看任何反馈信号）打平了 `koopman_mpc`：攻击场景
主判据和良性代价上都不输给、甚至略优于 `koopman_mpc`。这修正了 Phase E/F 原先"建模复杂度换来
了收益"的说法，把结论收窄为：

1. `koopman_mpc` 相对 `threshold`（反应式反馈基线）仍是明确的胜利；
2. `koopman_mpc` 相对 `periodic`（同代价固定日程）的优势**未被现有数据证明，也未被证伪**；
3. `koopman_mpc` 唯一还没被排除的潜在优势是**自适应性**——`periodic` 的插入密度是写死的常数，
   `koopman_mpc` 原则上会根据预测的安全分调整插入时机，但这批 8 个 held-out 攻击强度相对同质，
   没有制造出能体现这种自适应优势的条件。

Case 分析的目的**不是**重复证明"Koopman 比基线好"（Phase G 已经说明这句话现在不成立），而是
回答一个更具体、更诚实的问题：**`koopman_mpc` 的决策过程里，能不能找到"确实在利用预测的安全分
动态调整插入时机"的具体证据，而不只是凑巧和固定周期节奏重合？** 如果能找到，说明 1 里那种潜在
优势是有机制基础的，只是当前这批攻击样本太同质、规模太小，没能在聚合指标上体现出来；如果找不到
（比如发现 `koopman_mpc` 在每条轨迹上的插入 turn 都恰好是 2、4，和 `periodic` 完全一样），那就是
一个更负面、但同样诚实的 case 结论。

## 目标现象（五类，均可用现有记录离线复现，不需要新实验/新 GPU job）

1. **插入模式是否随攻击变化，还是退化为固定周期**（直接对应上面的开放问题，最该优先看）——
   逐条比较 `koopman_defense_phaseE_koopman_mpc` 和 `koopman_defense_phaseG_periodic`
   （period=2，固定在 turn 2、4 插入）两个 arm 里，每个 `attack_id` 的 `u_remind` 序列是否
   一致。如果 8 个攻击里 `koopman_mpc` 的插入 turn 全部等于 `{2,4}`，说明它在这批数据上确实
   退化成了周期规则；如果有攻击的插入 turn 是 `{1,3}`、`{3,4}` 等不同模式，就是自适应性的直接
   证据，可以进一步看该攻击轨迹在哪个 turn 出现了不一样的 `y_probe` 走势导致这个差异决策。

2. **前瞻性介入 vs `threshold` 的被动反应**——同一 `(attack_id, seed)` 上，`koopman_mpc`
   `u_remind=1` 而 `threshold` `u_remind=0` 的最早分歧 turn，且此时 `y_probe` 尚未跌破 0.7
   （`threshold` 的触发阈值）。说明 MPC 提前于阈值触发点行动。

3. **选择性节省 vs `constant_remind` 的冗余提醒**——`koopman_mpc` 判断 `u_remind=0` 而
   `constant_remind` 恒为 1 的 turn，且后续 `y_probe`/`y_safety` 并未恶化，支持"该省的提醒确实
   可以省"这个判断。

4. **horizon 真正生效（贪心 vs 前瞻决策分歧）**——把 `KoopmanMPCController._simulate` 拆成
   `value_immediate(a) = readout(step(z,a))`（只看下一步，相当于 horizon=1 贪心）和实际用到的
   `value_full(a) = _simulate(z,a,horizon-1=1)`（真正的 2 步前瞻值）。找 `argmax value_immediate`
   与 `argmax value_full` 不一致的 turn——这是"前瞻预测真的改变了决策，而不是退化成即时贪心规则"
   最硬的证据。

5. **均值回归/校准后的鲁棒性（措辞已修正）**——**重要限定**：Phase E 用的状态是 `nu=1, mu=2`，
   即 `z_t = [当前 y_probe, u_{t-1}, u_{t-2}]`（再做 abs/sign 非线性提升到 5 维），**不包含
   多轮 y 历史**。所以这一条不能说成"Koopman 靠平均历史读数滤除噪声"，准确的机制是：拟合出的
   `A` 矩阵里 y 这一维的自回归系数 `< 1`（已拟合的 `arx` 模型里是 0.82），代表模型学到"低读数
   会自然回升"的先验，加上最近是否已提醒过（`u_{t-1}, u_{t-2}`）共同决定预测。找
   `threshold` 触发（`y_probe<0.7`）但 `koopman_mpc` 未触发的 turn，用该 turn 的 `A` 系数和
   `z_t` 具体数值说明模型为何判断"会自愈"。

## 数据来源与复现方法

全部离线、纯 numpy/pandas，不需要 GPU、不需要新实验：

- `outputs/koopman_defense_phaseE_{zero_control,constant_remind,threshold,koopman_mpc}/trajectories.jsonl`
  （Phase E 四臂，攻击场景）
- `outputs/koopman_defense_phaseG_periodic/trajectories.jsonl`（Phase G 周期基线，攻击场景）
- `outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json` 的 `richer_abs_sign`
  （`nu=1, mu=2`）——Phase E/G `koopman_mpc`/`periodic` 实际加载的同一份拟合模型
- 复用 `scripts/analyze_koopman_innovation.py::load_model`（同样的 `surrogate_from_arrays` +
  `ReducedStateConfig` 构造方式），避免重复实现状态重建逻辑
- `KoopmanMPCController._current_state`/`_simulate` 的重建逻辑是纯函数，给定同样的
  `history`（`trajectories.jsonl` 里已有的 `y_probe`/`u_remind` 逐轮记录）和同一份拟合矩阵，
  可以精确复现当时控制器实际做的决策计算——包括未落盘的中间值（`value_immediate`/`value_full`）

## 计划产出

单个脚本 `scripts/analyze_koopman_mpc_cases.py`，逐 `(attack_id, seed)` 重放 `koopman_mpc`
的决策过程，产出五张候选表（对应上面五类现象），每条候选记录附带该 turn 的
`attacker_query`/`agent_message`/`inserted_reminder_text` 摘录。

## 执行结果（2026-09-01）

对 Phase E `koopman_mpc` 臂全部 16 条轨迹（8 攻击 × 2 seed）逐轮重放决策计算。结果比设计时
预想的更清晰，也更负面：**这批数据里 koopman_mpc 完全没有表现出任何状态依赖的自适应决策**。

1. **插入模式**：`_current_state` 要求 `len(history) >= max(nu-1,mu)+1 = 3`（nu=1, mu=2），
   每条轨迹 turn 1-3 一律走"历史不足"的 fallback（强制 `u_remind=0`），turn 4-5 才是真正的
   MPC 决策——而这两轮上 **16/16 条轨迹、32/32 次真实决策，全部选择 `u_remind=1`，没有一次
   例外**。和 `periodic`（period=2，在 turn 2/4 插入）比较，16/16 条轨迹的插入 turn 模式都不
   相同（`phenomenon_1_n_identical_to_periodic=0`）——koopman 固定在 turn4/5，periodic 固定
   在 turn2/4，两者是**不同的固定规则**，不是巧合重合成同一个规则。
2. **前瞻介入 vs `threshold`**：13/16 条轨迹上，koopman 在 `y_probe` 仍 ≥0.7（`threshold`
   触发线）时就已插入提醒。但结合现象 1，这不是"预测到即将恶化所以提前动手"，而是"koopman
   无条件在 turn4/5 插入，凑巧那时 `y_probe` 经常还没跌破 0.7"——表面像前瞻，机制上是定时。
3. **选择性节省 vs `constant_remind`**：48 个 `u_remind=0` 的 turn **全部**发生在 turn 1-3
   （`phenomenon_3_n_zero_remind_turns_with_real_state=0`）——koopman 从未在有真实状态可用时
   选择"不提醒"，"省"完全是热身期的副产品。
4. **horizon 是否真正生效**：32 次真实决策里 `argmax(value_immediate)` 与 `argmax(value_full)`
   **完全一致，0 次分歧**。根源：`value_full(1) - value_full(0)` 在全部 32 次决策上恰好等于
   同一个常数 0.3672（浮点抖动量级），`value_immediate(1)-value_immediate(0)` 恒为 0.2659。
   这是模型的结构性质，不是巧合：动作 `v` 只通过 `B@v` 这一项线性、可加地进入状态转移方程，
   和状态 `z`（含 `abs_sign` 提升特征）没有任何交互项，所以"提醒 vs 不提醒"带来的预测收益差
   在数学上必然与当前状态无关——不管 `z_t` 是什么、horizon 多长，动作的边际预测收益都是同一个
   常数。只要这个常数为正（这批数据里确实为正）且 `repeat_penalty=0.0`，MPC 在有真实状态时就
   会永远选择提醒，与前瞻计算无关。
5. **均值回归鲁棒性**：0 个 case——现象 4 结果的必然推论（真实状态下 koopman 从不选 0，不存在
   "threshold 触发但 koopman 判断会自愈所以不触发"的情况）。

### 诚实结论

**Phase G 留下的"koopman_mpc 相对 `periodic` 的自适应优势尚未被证明也未被证伪"这个开放问题，
在这批数据/这个模型架构下有了更明确的答案：不是"数据不够看出来"，而是"这个控制器的决策机制在
结构上就不可能依赖状态"。** 只要 `repeat_penalty=0` 且用的是这种"动作线性可加、无状态-动作
交互项"的 Koopman/ARX 模型，MPC 的最优动作就是一个和 `z_t` 无关的常数（提醒边际收益的符号），
horizon、`abs_sign` 非线性提升都不改变这个结论——多步前瞻和更丰富的 lifting 在这个问题结构下
加的是零信息量。这也解释了 Phase G 为什么会被 `periodic` 打平：两者本质上都是不看状态的固定
规则，只是巧合选中了不同的固定 turn。

**对下一步方向的启示**：如果想让 `koopman_mpc` 表现出真实的自适应优势，现有模型结构不够——
需要引入非零 `repeat_penalty`（让"是否已经提醒过"通过 `v_hist` 真正影响决策的边际项，而不只
影响 readout 的截距），或者在 lifting 里显式加入状态-动作交互项（如 `abs(y)*v`）。否则无论怎么
调 `horizon` 或换更"丰富"的非线性 lifting，决策都会继续退化成同一个状态无关的开关——这是比
"再跑一次 A/B 实验"更精确的诊断，建议作为改进方向记录，而非继续在当前架构上做更多消融。

产物：`outputs/koopman_case_study/`（`koopman_mpc_decisions.csv` 全量决策重放 +
`phenomenon{1..5}_*.csv` 五张现象表 + `case_study_summary.json` 汇总数字）。

**执行状态：脚本已写、已跑完，结论已记录。**
