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

单个脚本（暂定 `scripts/analyze_koopman_mpc_cases.py`），逐 `(attack_id, seed)` 重放
`koopman_mpc` 的决策过程，产出五张候选表（对应上面五类现象），每条候选记录附带该 turn 的
`attacker_query`/`agent_message`/`inserted_reminder_text` 摘录，供人工挑 1-2 个最有说服力的
具体案例写进报告。**执行状态：设计已记录，脚本待写。**
