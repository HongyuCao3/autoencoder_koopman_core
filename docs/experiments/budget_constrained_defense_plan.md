# Phase J：预算约束下的对抗防御控制（执行中）

**状态（2026-09-03）**：代码/配置/离线预检已完成，CPU 全套 283 个测试通过；7 个 GPU 臂的
sbatch 已就绪，**尚未提交**。

## 一、为什么要改评测设定

`koopman_defense_pilot.md` Phase A→I 的收尾结论是：v 对齐 bug 修好之后，`nu=1,mu=2` 的交互
模型学出了方向正确、更经济的策略（攻击场景 21/80 提醒 vs `periodic` 的 32/80，终轮安全分
0.7812 vs 0.7500），但仍然没在 new-Q1 上打赢 `periodic`，原因是结构性的：反应式策略必须先
观测到异常才能动，而 `periodic` 在 turn2 就盲插一次，恰好抢在自然下滑之前。

`docs/next_step_diagnosis.md` 第二节早就指出了这个设定层面的问题，比"谁抢跑"更根本：

> 当前 MPC 的目标是最大化预测安全分之和，`repeat_penalty=0`，而 Phase F 显示良性代价很小。
> 在"提醒有正效应、几乎没有代价"的设定下，**理论最优策略就是常提醒**……自适应策略只有在
> **代价 binding** 时才有存在的理由。

也就是说：**Phase A-I 的评测里根本不存在"分配问题"**，所以"Koopman 没展现自适应优势"是设定
的必然结果，而不是方法的证据。Phase J 就是把设定改成有分配问题的版本：每条轨迹最多 `k` 次
提醒，策略的任务从"要不要提醒"变成"把这 `k` 次放在哪一轮"。

## 二、为什么是 k=1，不是计划里写的 k=2

`next_step_diagnosis.md` 建议的是 k=2。核对已收集数据后改成 k=1，理由是 **k=2 在这批轨迹上根本不 binding**：

| 臂 | 每条轨迹提醒次数 |
|---|---|
| Phase G `periodic(period=2)` | 恒为 2（turn 2、4） |
| Phase I v 对齐 koopman_mpc | 最多 2，均值 1.31（分布：0×3, 1×5, 2×8） |
| Phase E `threshold` | 最多 2，均值 0.44 |

8 个 held-out 攻击全部是 5 轮（已核实），所以 k=2 的约束对现有任何一臂都不产生作用——跑出来
会和 Phase I 逐字节相同，纯属浪费 GPU。**k=1 是能真正制造分配决策的最大预算**：固定日程必须
提前押一轮，自适应策略可以等证据。顺带一提，`periodic(period=2)` 恰好就是 k=2 的固定分配，
所以 k=2 那一档的"预算匹配对比"其实已经由 Phase G/I 现成给出了（结论：同预算下 Koopman 花得
更少、终轮更高，但 new-Q1 没赢）。

## 三、臂设计（8 个 held-out 攻击 × seed 0/1，与 Phase E/G/H/I 完全可比）

| 臂 | 控制器 | 说明 |
|---|---|---|
| `phaseJ_budget1_fixed_t1` … `_t5` | `fixed_schedule` | 固定把唯一一次提醒放在第 1/2/3/4/5 轮，无反馈 |
| `phaseJ_budget1_threshold` | `threshold(y_min=0.7)` + 预算封装 | 无模型的反应式基线："花在第一个看起来变糟的轮次" |
| `phaseJ_budget1_koopman` | v 对齐交互 `koopman_mpc` + 预算感知规划 | 自适应臂 |

**为什么固定日程要跑满 1–5 轮的扫描**：要检验的命题是"自适应打赢**最好的**固定分配"，而最好
的那一轮只能测出来；这个扫描本身也是一个结果（单次提醒放在哪一轮最值）。代价是把对比的门槛
抬高了（在同一批数据上挑最优基线，偏向不利于自适应臂），这是有意为之。

`threshold` 臂的作用是把两件事拆开：**有反馈**（它已经有了）和**用前向模型判断"这个坏轮次值
不值得花掉最后一次机会、还是预测会有更坏的在后面"**（只有 Koopman 臂有）。

### Koopman 臂相对 Phase I 只改两个旋钮，都是预算设定本身要求的

- `koopman_horizon: 5` + `episode_length: 5`（Phase I 是 horizon=2）：horizon 会被裁剪到轨迹
  真正剩下的轮数（`KoopmanMPCController._planning_steps`），"现在花掉还是留给预测更糟的一轮"
  这个权衡在 2 步 horizon 里根本表达不出来。
- `koopman_pad_short_history: true`：允许它和 `periodic(2)` 一样从 turn2 开始动作。Phase I 的
  离线重放已经证明它在 turn2 会选择不提醒（0/16，turn1 的 `y_safety` 恒为 1.00），但不打开这个
  开关就会把"选择不早花"和"结构上不能早花"混在一起，而这正是本实验要量化的东西。

模型本身、`nu/mu`、`contemporaneous_v`、`repeat_penalty=0` 全部与 Phase I 一致，用的是同一个
`interaction_model_report_valigned.json`——**Phase J 不重新拟合任何东西**，只换求解的问题。

## 四、提交 GPU 前的离线预检（已完成，结论：可以提交）

`scripts/analyze_budget_allocation_replay.py`（CPU，无需 GPU）把预算策略回放在两批已有真实
轨迹上，检查它是否会退化成一个固定日程——如果它每条轨迹都花在同一轮，那闭环对比不可能有信息量，
就应该像 Phase I 的 `pad_short_history` 检查一样在离线阶段判死，而不是花 GPU 去学这件事：

| 回放数据 | 花在 turn3 | turn4 | turn5 | 一次都不花 |
|---|---|---|---|---|
| Phase I 自己的轨迹（16 条） | 0 | 9 | 4 | 3 |
| Phase G periodic 的轨迹（16 条） | 1 | 6 | 4 | 5 |

**结论：分配是轨迹依赖的（3 个不同轮次 + 一部分完全不花），不可约化为任何单一固定日程。**
这是闭环实验值得跑的**必要**条件（不是充分条件）。注意回放只是近似：`y_safety` 是真实的，但
喂进滞后槽位的提醒历史不是这个策略自己产生的，真实闭环里不同的提醒模式会改变模型后续的真实
回复——这一点和 Phase H→I 的教训一样，只有闭环能回答。

## 五、指标

`scripts/analyze_budget_arm_comparison.py`（CPU）产出跨臂对比表：

- **主指标**：每条轨迹的效应量——终轮 `y_safety`、后段窗口（turn 3–5）均值——自适应臂 vs 每个
  固定臂**按 (攻击, seed) 配对**，报告均值差 + bootstrap 95% CI。
- **续接指标**：new-Q1（整条轨迹 OLS 斜率的一次性 t 检验），保留只是为了和 Phase E–I 可比。
  它的已知问题在 Phase I 已经写明：被 turn1–3 的自然下滑主导，而任何反应式策略都还没来得及动作，
  且只给 pass/fail 不给量级。
- **成本**：实际花掉的提醒次数（预算是上界，不是必须花完——离线预检里就有 3–5/16 一次都不花）。

**已知局限（写在前面，不等结果出来再补）**：seed 仍然只有 2 个，CI 必然很宽；"最好的固定臂"
是在同一批数据上选的；`next_step_diagnosis.md` 第三步（seed 扩到 5+、主指标全面换成效应量）没有在本阶段
执行，Phase J 只改设定不改样本量，这样"变化只有一处"，结果才能归因。

## 六、Hydra 配置与"不覆盖、可区分"的机制

新增 `conf/experiment/` 配置组（Hydra 的 experiment 模式），**一个文件 = 一个臂 = 一个
`output_dir`**：

```bash
python scripts/run_screening_hydra.py experiment=phaseJ_budget1_koopman
```

四道互相独立的隔离：

1. **`conf/task/defense.yaml` 保持行为不变**：新增的 `remind_budget`/`episode_length`/
   `fixed_schedule_turns`/`koopman_pad_short_history` 默认全是 `null`/`false`，等于 Phase A–I
   的无预算行为。只有 `experiment=phaseJ_*` 才会打开它们，`task=defense` 单独跑不可能意外复现
   出一个预算臂。
2. **`conf/experiment/phaseJ_base.yaml` 集中持有攻击集/seed/预算**，7 个臂文件全部 `defaults`
   继承它——逐个臂复制 `attack_ids` 的话，一个 typo 就会让对比变成不可比，而且不会报错。
3. **`output_dir` 由臂文件钉死**（`outputs/koopman_defense_phaseJ_budget1_*`），和已有的
   `phaseA`…`phaseI` 目录不共享前缀；`logs/` 的文件名带控制器名（`fixed_schedule_t3`、
   `koopman_mpc_interaction_budget1`、`threshold_budget1`，见 `controller_cli._budgeted_name`），
   `trajectories.jsonl` 的 `excitation_design` 列也记的是这个名字，所以日志和数据行本身就能区分臂。
4. **`persona_drift.run_config_guard`**：每个 `output_dir` 第一次运行时落一份
   `hydra_run_config.json`，之后再往同一目录跑就逐键比对，不一致直接退出并列出差异键
   （`allow_config_change=true` 可显式放行）。这条是必需的，因为 screening 循环**天生可续跑**
   （`screening_common.prepare_resumable_trajectories_file` 会复用已完成的轨迹），把新臂指到旧
   目录不会报错，只会安静地把两个控制器的轨迹混进同一份报告——Hydra 自带的 `.hydra/` 快照挡不住
   这个，因为它每次都写一个新的时间戳子目录，从不和上一次比。

## 七、代码改动清单

- `src/persona_drift/control.py`：新增 `FixedScheduleController`、`BudgetLimitedController`；
  `KoopmanMPCController` 新增 `remind_budget`/`episode_length`（默认 `None` = 原行为），枚举时
  按剩余预算剪枝、按剩余轮数裁剪 horizon。预算从 `history` 数出来而不是存在实例里——控制器保持
  无状态，所以仍可跨轨迹共享实例，续跑时也能从磁盘上的行精确恢复预算。
- `src/persona_drift/controller_cli.py`：`fixed_schedule` 分支；无模型控制器按 `remind_budget`
  自动封装（koopman 分支**不**封装——预算要进它的规划，不是事后截断）；`random_excite` 显式拒绝
  预算（会破坏辨识数据的 i.i.d. 设计）；固定日程超预算时报错而不是静默截断。
- `src/persona_drift/run_config_guard.py`（新）：上面第 6 节第 4 条。
- `scripts/run_screening_hydra.py`：接入以上参数 + 守卫；koopman 臂现在按 `s.controller` 记录
  自己的臂名（交互变体过去会把自己记成普通 `koopman_mpc`）。
- `scripts/analyze_budget_allocation_replay.py`（新）：第 4 节的离线预检。
- `scripts/analyze_budget_arm_comparison.py`（新）：第 5 节的跨臂对比。
- `conf/experiment/*.yaml`（新，8 个）、`conf/task/defense.yaml`、`conf/screening.yaml`。
- `environment/run_koopman_defense_phaseJ_budget1_*.sbatch`（新，7 个）。
- 测试：`tests/test_control.py`（+7，含"同一状态同一模型，只有预算不同 → 从'现在提醒'变成'留着'"
  的行为测试和无预算回归测试）、`tests/test_controller_cli.py`（+7）、`tests/test_task_config.py`
  （+6，臂间 `output_dir` 互不相同、攻击集/预算/seed 完全一致、experiment 不泄漏回默认 compose）、
  `tests/test_run_config_guard.py`（新，5）。CPU 全套 283 passed。

## 八、怎么跑

```bash
cd persona_drift_control
for arm in koopman threshold fixed_t1 fixed_t2 fixed_t3 fixed_t4 fixed_t5; do
  sbatch environment/run_koopman_defense_phaseJ_budget1_${arm}.sbatch
done
# 全部跑完后（CPU）：
python scripts/analyze_budget_arm_comparison.py
```

每臂 16 条 5 轮轨迹，按 Phase E/G/H/I 的实测约 30 分钟/臂（sbatch 里 `--time 00:30:00`），
7 臂可并行。

**没有安排良性 helpfulness 臂**：k=1 的预算把良性代价的上界压到每条会话最多 1 次提醒，比
Phase F/G/H/I 任何一臂都低（Phase I 已经是 9/96 且无可测代价），这一档不存在需要新数据回答的
helpfulness 问题。

## 九、可证伪的预期

- 如果自适应臂在**最好**的固定臂上拿到正的配对效应量差且 CI 不含 0：这是这条线第一个"Koopman
  代理模型的自适应性带来了可测收益"的正面结果，且是在一个 Phase I 之后专门为此设计的设定里。
- 如果它只打平最好的固定臂、但明显赢过 `threshold` 臂：说明前向模型带来的增益在"什么时候花"
  这个维度上是真实的、但不够大，结论应写成"同预算下持平最优固定分配，优于同样有反馈的无模型基线"。
- 如果它连 `threshold` 都赢不了：那么在 5 轮 × 二值动作这个规模下，前向模型对分配决策没有增量
  价值，应当接受并把结论写进论文的 limitation——这个规模的决策空间本来就只有 4–5 种模式，
  这也是 `next_step_diagnosis.md` 第三节早就点明的问题。
