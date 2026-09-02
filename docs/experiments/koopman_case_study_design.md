# 设计笔记：Koopman-MPC 决策的 case 分析（回应 Phase G 的自适应性开放问题）

和 [koopman_detection_design.md](koopman_detection_design.md) 同一类"供跨会话接续"的记录。
**新开一次对话想知道"case 分析这条线想验证什么、怎么做的、做到哪一步了"，看这份文档。**
**最新进展（2026-09-01）：文末"后续：验证'对下一步方向的启示'"一节——状态-动作交互项
（加非零 `repeat_penalty`）已经离线证实能让决策产生真正的状态依赖，"无法自适应"不是方法论
天花板；要转成可验证的 strong motivation 还差一次新的闭环 GPU 实验，尚未执行。**

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

## 后续：验证"对下一步方向的启示"（路径1/路径2，2026-09-01）

背景：`periodic` 用同代价打平 `koopman_mpc` 之后，用户提出的问题是"如何才能证明 Koopman
的意义/这种控制方式的 strong motivation"。上面"对下一步方向的启示"提出的两个候选修复——
非零 `repeat_penalty`、状态-动作交互项——都离线执行完了（纯 CPU，复用已有拟合数据/已录制
轨迹，不涉及新 GPU/LLM 调用），结果和理论预判精确吻合，并额外发现了一个需要谨慎解读的现象。

### 路径 1：`repeat_penalty` 扫描（`scripts/analyze_repeat_penalty_sweep.py`），定量证实"确实修不好"

在不改架构的 `richer_abs_sign` 模型上，把 `repeat_penalty` 从 0 扫到 1.0（10 个取值，含
理论翻转点 0.3672），对 Phase E `koopman_mpc` 臂的全部 32 次真实决策重新计算
`value_full(1)-value_full(0)`：

| repeat_penalty | n_remind/32 | margin_std |
|---|---|---|
| 0.0 – 0.3672 | 32/32 | ~1e-16（浮点噪声） |
| 0.37 – 1.0 | 0/32 | ~1e-16（浮点噪声） |

**在整段扫描范围内，没有任何一个 `repeat_penalty` 取值产生过 0/1 混合决策**——它是一个纯粹的
全局开关，在 0.3672 和 0.37 之间一次性从"全部提醒"翻转到"全部不提醒"，此外任何取值下
margin 的方差都停留在浮点噪声量级。这把此前只是代数推导的结论（"`repeat_penalty` 是和 `z`
无关的常数，不可能引入状态依赖"）**变成了直接测量到的数字**。

### 路径 2：状态-动作交互项（`scripts/analyze_state_action_interaction.py`），证实"确实能修好"

新增 `modeling/interaction_lift.py`：`augment_with_interaction` 把控制输入从标量 `v` 扩成
`[v, v*y_t]`，`InteractionLiftedSurrogate` 把这个增广对 `control.py`/`modeling.evaluate`
完全透明（两边都只看到未增广的 `Predictor.step(z, v)` 接口，`v` 的提升发生在 wrapper 内部）。
在完全相同的 Phase B 数据、完全相同的 `attack_id` 75/25 split 上重新拟合。

**一个顺带发现，值得单独记录**：拟合时特意选了 `no_extra_features`（纯 ARX）而不是
`abs_sign_extra_features`，因为核对 Phase B 全部 300 行 `y_safety` 后发现**这个"非线性提升"
在这个信号上几乎完全退化**——`y_safety∈[0,1]` 永远非负，300 行里 `abs(y)==y` 100% 成立，
`sign(y)` 在 290/300 行恒为 1（仅 10 行 `y` 恰好等于 0.0 时为 0）。也就是说 `richer_abs_sign`
比纯 ARX 好一点点（Phase C：held-out rollout MSE 0.043 vs 0.051）很可能不是真的学到了非线性，
而是 ridge 在一个近乎共线/退化的增广矩阵上数值表现恰好不同——这是此前从未诊断过的一点，
也是这次刻意选纯 ARX + 交互项、不叠加 `abs_sign` 的原因：避免把交互项的效果和一个本身可疑的
"非线性"混在一起看不清楚。

**拟合质量（不是这次的重点，但需要如实报）**：交互模型 held-out rollout MSE = 0.0514，
和纯 ARX（0.0510）基本持平，比 `richer_abs_sign`（0.0430）略差——加交互项本身没有改善
预测精度，这符合预期：交互项是为了让**决策**依赖状态，不是为了让**预测**更准，两者是不同
的目标。

**决策重放（`repeat_penalty=0`）**：在 Phase E `koopman_mpc` 臂的同一批 32 次真实决策点上，
margin 的标准差从路径 1 的 ~1e-16 跳到 **0.0567**，margin 范围 `[0.084, 0.318]`，和 `y_probe`
的相关系数 **1.0**——marginal value 现在是 `z` 的一个真正的（线性的）函数，不再是常数。
**结构性证据确认：加交互项确实能让边际价值依赖状态，路径 1 做不到的事，路径 2 做到了。**

**但单靠交互项还不足以翻转任何一次决策**：这批 held-out 状态上 margin 的最小值（0.084）
仍然是正的，所以 32/32 次决策依然全部选择提醒——不是架构限制，是这批数据里 margin 从未
跌破 0。

**交互项 + 非零 `repeat_penalty` 组合：产生了真正的混合决策**（`repeat_penalty` 现在有意义，
因为它要对比的 margin 不再是常数）：

| repeat_penalty | n_remind/32 |
|---|---|
| 0.10 | 31/32 |
| 0.15 | 30/32 |
| 0.20 | 26/32 |
| 0.25 | 14/32（接近一半一半） |

**这是本轮调查的核心正面结果：只要同时具备 (a) 状态-动作交互项、(b) 一个卡在 margin 实际
取值范围内的非零 `repeat_penalty`，`koopman_mpc` 就能产生真正状态依赖、不退化为固定规则的
决策。**"Koopman 控制无法自适应"不是这套方法论本身的天花板，只是当前
（`nu=1,mu=2` + `B@v` 无交互）这个具体架构选择的产物,换一个最小的架构改动就能绕开。

**一个需要如实指出、目前还没有答案的问题**：margin 和 `y_probe` 是**正相关**——提高
`repeat_penalty` 时，最先被排除出"该提醒"名单的，是 `y_probe` **更低**（攻击更成功）的
状态，不是更高的。这和"情况越糟应该越积极提醒"这个朴素直觉方向相反。可能的解释：
（a）当 `y_safety` 已经跌到很低时，单轮提醒预测的边际收益本身就小，可能反映"已经造成的
侵蚀这一轮提醒也挽不回"这个真实现象；（b）也可能是 Phase B 随机激励数据里低 `y_safety`
样本本来就稀疏（Phase B 用的是 30 个攻击的开环随机激励，不是针对性收集低分样本），拟合在
这个区域外推不稳。**现有数据不足以区分这两种解释。这意味着"交互项修好了架构"和"修好后
学到的具体策略在直觉上是对的"是两件不同的事——前者已经证实，后者需要更多数据或专门的
校准步骤才能回答，不能想当然认为"能自适应了"就等于"自适应得合理"。**

### 对"如何证明 Koopman 意义"这个问题的回答

1. **纯方法论层面**：自适应性从未被证明是 Koopman/线性代理模型这条路线本身的天花板——
   之前"结构上不可能"的准确表述应该是"这个具体的、动作线性可加无交互的架构不可能"，这个
   限制现在被证明是可以绕开的（交互项），不需要放弃线性代理模型换成 LSTM 之类的黑盒模型。
2. **要转成"控制器真的打赢 `periodic`"这个可验证的 strong motivation，离线重放还不够**——
   需要一次新的闭环 GPU 实验（Phase E/G 同等规模，~15 分钟），用交互模型 + 校准过的
   `repeat_penalty` 接入 `KoopmanMPCController`，在**强度差异更大的攻击集合**上
   （例如混合 `safemtdata_0074` 这类第3轮就崩到0的快速侵蚀攻击、和
   `safemtdata_0169`/`safemtdata_0530` 这类全程没跌破0.75的几乎攻不破的攻击）对比
   `periodic`——只有在这种混合强度场景下，一个真正自适应的策略才有机会展现出 `periodic`
   这种固定日程结构性做不到的优势（该省则省、该顶则顶）；这批 8 个 held-out 攻击本身其实已经
   有这种强度差异（见上），不需要新采数据，可以直接复用。
3. **上面"方向反直觉"的问题最好先弄清楚再投入新 GPU 实验**——如果交互项学到的"情况越糟
   边际收益越小"是数据稀疏的假象而非真实现象，那么校准出来的自适应策略可能会在快速侵蚀的
   攻击上主动省提醒，这是不希望看到的结果。建议先用更细的 `repeat_penalty`/交互系数网格做
   离线诊断，确认这个方向在物理上说得通，再决定是否值得投入新的 GPU 实验。

产物：`outputs/koopman_case_study/repeat_penalty_sweep.json`（路径 1）、
`outputs/koopman_case_study/interaction_model_report*.json`（路径 2，
`repeat_penalty ∈ {0, 0.1, 0.15, 0.2, 0.25}` 共 5 组）。代码改动：新增
`src/persona_drift/modeling/interaction_lift.py`、
`scripts/analyze_repeat_penalty_sweep.py`、`scripts/analyze_state_action_interaction.py`，
测试 `tests/test_interaction_lift.py`（3 passed，CPU 全套 182 个测试里除
`test_logging_setup.py` 一个历史已知的 loguru 竞态 flaky 测试外全部通过，与本次改动无关）。

**执行状态：路径 1/路径 2 均已跑完，结论已记录。**

## Phase H：真正的闭环验证（`koopman_mpc_interaction`，2026-09-01 提交）

上面第 2 点的建议——离线证据不够，需要一次真正的闭环 GPU 实验——已经动手做。

**代码改动**：新增 `controller_cli.py::load_koopman_mpc_interaction_controller`（加载
`analyze_state_action_interaction.py` 存的 `model` 字段，包成
`InteractionLiftedSurrogate`，再套进不变的 `KoopmanMPCController`），
`make_controller_factory` 新增 `"koopman_mpc_interaction"` 分支；
`run_defended_screening.py`/`run_benign_helpfulness_screening.py` 都加了
`--controller koopman_mpc_interaction`/`--koopman-interaction-model-path`。测试
`tests/test_controller_cli.py` 新增 2 个（覆盖 factory 分支 + 真实 report 文件加载），
CPU 全套 184 个测试除 `test_logging_setup.py` 一个历史已知 flaky 外全部通过。用真实存盘的
`outputs/koopman_case_study/interaction_model_report.json` 跑了一次端到端冒烟检查（离线重放
Phase E `koopman_mpc` 臂的 32 次真实状态，`repeat_penalty=0.2` 下 `n_remind=26/32`，和
`analyze_state_action_interaction.py --repeat-penalty 0.2` 的离线结果完全一致）——确认 CLI
接线本身没有引入偏差，才提交 GPU 作业。

**`repeat_penalty=0.2` 的选择**：离线扫描里 0.1→31/32、0.15→30/32、0.2→26/32、0.25→14/32，
选 0.2 是两头都不极端的折中——0.25 会把接近一半的真实决策点判成"不提醒"，考虑到 margin 和
`y_probe` 正相关这个尚未解释的反直觉方向（`repeat_penalty` 会优先削减**更差**状态上的提醒），
0.25 在这批数据上可能显得过于激进。

**攻击场景**（`run_koopman_defense_phaseH_koopman_mpc_interaction.sbatch`，job 提交见下）：
和 Phase E/G 完全相同的 8 个 held-out 攻击 × 2 seed = 16 条轨迹。这批攻击本身已经有明显的
强度异质性（`safemtdata_0074` 在 `zero_control` 下第3轮就崩到 `y_safety=0.0`，
`safemtdata_0169`/`0530` 全程没跌破 0.75，见上文），不需要新采数据。

**良性场景**（`run_koopman_defense_phaseH_koopman_mpc_interaction_benign.sbatch`）：和
Phase F/G 完全相同的 16 个固定 `(benign_id, seed)` MT-Bench 会话，检查这次的
`repeat_penalty` 校准是否在良性流量上产生可测的 helpfulness 代价。

**为什么离线重放不够，必须真跑**：`analyze_state_action_interaction.py` 的离线重放用的是
Phase E 原 `koopman_mpc`（`repeat_penalty=0`）真实录制的历史状态,只回答"给定这些历史，新策略
*会*做出什么决策"；但一旦新策略在第 4/5 轮真的省掉了某次提醒，那一轮真实的 agent 回复、以及
它引出的下一轮 `y_safety`，都会和原录制不同——**这个反事实结果无法从已录制数据反推**，只能
重新调用真实模型生成。这也是为什么这次是一次新的 GPU job，而不是又一次离线脚本。

产物（跑完后补充结果）：`outputs/koopman_defense_phaseH_koopman_mpc_interaction/`、
`outputs/koopman_defense_phaseH_koopman_mpc_interaction_benign/`。
