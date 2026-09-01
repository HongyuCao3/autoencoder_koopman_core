# 实验记录：Koopman-MPC 防御控制器（抵抗对抗性侵蚀）

和 [adversarial_screening_pilot.md](adversarial_screening_pilot.md) 同一类"供跨会话接续"的
记录。**新开一次对话想知道"Koopman 防御控制器现在做到哪一步了",看这份文档。**

## 这是什么，为什么做

`adversarial_screening_pilot.md`（step 1）和它的 thinking 模式复现
（[adversarial_screening_thinking_pilot.md](adversarial_screening_thinking_pilot.md)）已经
稳固确认：Qwen3-4B 在多轮渐进 jailbreak 攻击下安全性系统性侵蚀（new-Q1 两次都 p<0.0001），
且相邻轮次有显著惯性（new-Q3 两次都过）——这是 Koopman/MPC 建模有意义所需的前提。

`dose_response_pilot.md`（channel C，激活转向 steering）测了三轮都没能建立一个可用的执行器：
天花板效应 → 纯噪声 → 显著但方向相反（更像"单层扰动本身就是噪声源"）。**用户已决定这一轮不再
追加 channel C 新实验**,Koopman 控制器改用**和人格漂移那条线一样的设计——channel A 式的
提醒注入(修改 prompt),不是直接 steering**。

这份文档记录：如何把 `control.py`（persona-drift 的 `Controller` 协议 + 已有控制器实现）、
`reminder.py` 的提醒插入模式、`modeling/koopman.py`/`modeling/evaluate.py` 三块**几乎零改动
直接复用**到对抗防御这个新领域，设计并验证一个 Koopman-MPC 控制器。完整设计方案见
`/home/hcao2/.claude/plans/shiny-stargazing-sphinx.md`（已批准的计划文件）。

## 执行器：安全提醒注入（channel A 式）

新增 `src/persona_drift/safety_reminder.py`：`build_safety_reminder_text(level)`，固定的
安全承诺重申文本（不针对具体攻击内容，不暴露"识别到攻击"），level 0/1（和 `reminder.py`
同样的范围限制）。

`attack_trajectory.run_attack_trajectory` 新增 `controller: Controller | None = None` 参数
——按 `selfchat.py` 的既有模式（每轮先问 controller 要 `u_remind`，再决定是否拼接提醒文本）
插入提醒。默认 `None` 等价于 `ZeroControlController`（现有全部 screening 作业，包括
job 15399715/15410124，都是这个默认值,不受影响）。行 schema 新增
`u_remind`/`y_probe`（`y_safety` 的别名，只是让 `control.py` 的
`ThresholdController`/未来的 `KoopmanMPCController` 不用改一行代码就能读）/
`excitation_design`/`inserted_reminder_text`/`inserted_tokens`。

`adversarial_screening.run_adversarial_screening` 新增 `controller_factory:
Callable[[int], Controller] | None = None` 参数——**是工厂而不是单个 controller 实例**，
镜像 `screening.py` 的 `_make_controller(condition, seed, ...)`：每条轨迹用自己的 seed
现造一个新 controller，这对 `RandomExciteController`（Phase B 要用）是必须的——如果整个
run 共享一个 controller 实例，它的 RNG 会跨轨迹连续跑，既破坏"每条轨迹只由自己的 seed
决定"的可复现性，也破坏这个函数已有的断点续跑机制（续跑时 RNG 状态对不上从头跑一遍的结果）。
`run_id`/`report["config"]["controller"]` 记录 controller 名字，方便区分不同 controller
的作业。

`control.py` 新增 `KoopmanMPCController`：给定一个拟合好的 `KoopmanSurrogate` +
`ReducedStateConfig`，对 0/1 动作空间做短 horizon（默认 2）穷举，用 `surrogate.step()`/
`.readout()` 滚动预测每种动作序列下的 y_safety，选第一步动作使预测的总安全分（减去
`repeat_penalty` 惩罚项）最高——标准 receding-horizon MPC，动作空间小到不需要 QP 求解器。

`modeling/dataset.py`/`modeling/evaluate.py` 的 `group_by_trajectory`/
`split_by_system_prompt_id`/`build_reduced_state_pairs`/`build_identification_dataset`/
`rollout_output_error` 都加了**可选**列名参数（`id_col`/`y_col`/`u_col`/`split_col`，默认值
= 现有硬编码名字，对 persona-drift 调用方和既有测试零行为变化），对抗数据这边可以直接传
`y_col="y_safety"`、`split_col="attack_id"`，不需要伪造/改名列。`modeling/koopman.py`
（`KoopmanSurrogate`/`controllability_diagnostics`）本身零修改，两次代码探索都确认它和
具体领域无关。

## 分阶段计划（对应已批准的计划文件）

- **Phase A（执行器授权检验）**：`--controller constant_remind`，复用 job 15399715 同一批
  20 攻击 × 2 seed，对比 new-Q1 斜率/t/p。门槛：如果常提醒也测不出缓解，停下来重新讨论执行器。
- **Phase B（开环激励采集）**：`--controller random_excite`，供 Koopman 辨识。
- **Phase C（拟合 + 可控性诊断）**：`KoopmanSurrogate` + `controllability_diagnostics`，
  门槛：B 是否退化。
- **Phase D（控制器实现）**：`KoopmanMPCController` + `ThresholdController` 经典基线。
- **Phase E（闭环验证）**：四臂对比（zero_control / constant_remind / threshold /
  koopman_mpc），`analyze_adversarial_screening()` 原样复用。

## 代码改动清单

- 新增 `src/persona_drift/safety_reminder.py`。
- 修改 `src/persona_drift/attack_trajectory.py`：`controller` 参数 + 提醒插入 + 新增行字段。
- 修改 `src/persona_drift/adversarial_screening.py`：`controller_factory` 参数,
  `run_id`/`report["config"]["controller"]`。
- 修改 `src/persona_drift/control.py`：新增 `KoopmanMPCController`。
- 修改 `src/persona_drift/modeling/dataset.py`、`src/persona_drift/modeling/evaluate.py`：
  可选列名参数（向后兼容）。
- 新增 `scripts/run_defended_screening.py`：`--controller
  {zero_control,constant_remind,threshold,random_excite}` 的通用 CLI。
- 新增 `environment/run_koopman_defense_phaseA_constant_remind.sbatch`。
- 测试：`tests/test_safety_reminder.py`（新）、`tests/test_attack_trajectory.py`（新增
  controller=None 向后兼容 + ConstantRemindController 插入正确性）、
  `tests/test_control.py`（新增 `KoopmanMPCController` 用合成 surrogate 测试）、
  `tests/test_modeling_dataset.py`/`tests/test_koopman.py`（新增自定义列名测试）、
  `tests/test_adversarial_screening.py`（新增 `controller_factory` 默认值/逐轨迹现造/
  `RandomExciteController` 独立播种回归测试）。CPU 单测本地跑通 142 passed
  （1 个预先存在、与本次改动无关的 `test_logging_setup.py` loguru 竞态 flaky 测试除外，
  历史上多次记录过）。

## 结果：Phase A（job 15413205，2026-08-31，27 分钟）

**通过门槛判定——常提醒有实质缓解效果，不是"测不出"。**

| 指标 | 无提醒基线（job 15399715） | 常提醒（job 15413205） |
|---|---|---|
| new-Q1 t / p / pass | t=-7.8553, p=0.0000, **pass=True**（强侵蚀） | t=-1.6031, p=0.1254, **pass=False**（侵蚀不再显著） |
| new-Q3 p / r | p=0.0000, r=0.3983 | p=0.0062, r=0.2154（惯性减弱但仍显著） |
| turn1→turn5 y_safety 均值 | 0.9313→0.8375→0.6438→0.6875→**0.4500** | 0.9625→0.8875→0.8187→0.9125→**0.8063** |
| refusal_rate | 0.0000 | 0.1950 |
| judge_parse_failure_rate | 0.0000 | 0.0000 |

第 5 轮（终轮）均值安全分从 0.45 提到 0.81，接近翻倍；new-Q1 从强显著侵蚀（p<0.0001）变成不显著
（p=0.125，仍是负斜率方向，只是不够强）。`refusal_rate` 从 0 升到 0.195——提醒确实让模型在
约 1/5 的轮次里直接拒绝了，符合"提醒生效"的预期，不是异常。new-Q3 惯性虽然减弱
（r 从 0.40 降到 0.22）但仍显著,说明在有防御的情况下,轮次间的记忆结构依然存在——这对
后续 Koopman 建模是好消息,不是"提醒把动力学抹平了导致没什么可建模"。

**结论：channel A 式安全提醒注入这个执行器有真实权威,继续 Phase B。**

## Phase B 状态

`environment/run_koopman_defense_phaseB_random_excite.sbatch`（job 15413471，
`--controller random_excite --random-excite-p 0.5`，30 攻击 × 2 seed，`--attack-rng-seed 100`
和 Phase A/基线的 0 不同）**已经和 Phase A 并行提交**（未等 Phase A 门槛判定结果就先提交,
接受小概率浪费算力换取不等待,用户已确认这个取舍）——现在 Phase A 已经证实通过,这次 Phase B
数据可以直接用于建模,不是浪费。跑完后进入 Phase C（拟合 + 可控性诊断）。

```bash
squeue --me
sacct -j 15413471 --format=JobID,State,Elapsed,ExitCode
cat persona_drift_control/outputs/koopman_defense_phaseB_random_excite/adversarial_screening_report.md
```

## 结果：Phase B（job 15413471，2026-08-31，41 分钟）

30 攻击 × 2 seed = 60 条轨迹，`random_excite p=0.5`。new-Q1 仍显著（t=-3.12, p=0.0041，比
无提醒基线弱，符合"只有约一半轮次插了提醒"的预期），new-Q3 惯性显著（p<0.0001）。
`trajectories.jsonl` 写入 `outputs/koopman_defense_phaseB_random_excite/`，供 Phase C 辨识。

## 结果：Phase C（`scripts/fit_koopman_defense_model.py`，纯 CPU，几秒）

新增脚本：读 Phase B 的 `trajectories.jsonl`，按 `attack_id` 切 75/25 train/held-out
（`split_by_system_prompt_id(..., split_col="attack_id")`），
`build_identification_dataset(..., y_col="y_safety")` 建数据集，拟合 ARX
（`no_extra_features`）和一个非线性提升（`abs_sign_extra_features`），跑
`controllability_diagnostics`。

**原计划默认的 `nu=1, mu=1`（只记 1 个滞后输入）拟合出的 B 弱且符号和 Phase A 的证据矛盾**：
`B=[[-0.059],[1.0]]`（第二行是状态里"记录上一次输入"这个位置的恒等式，不是学出来的，只看
第一行）——单轮 u_remind 对下一轮 y_safety 的边际效应是**负的**、且很小，held-out rollout
MSE=0.068，和 Phase A"常提醒让终轮安全分翻倍"这个明确证据不吻合。

**换成 `nu=1, mu=2`（记 2 个滞后输入）后结果转为合理、且拟合明显更好**：`B[0,0]=0.156`
（正号，符合"提醒提升安全性"的方向），held-out rollout MSE 降到 0.051，Gramian 条件数从
193 降到 12.7（数值上更健康），`controllability_rank=3`（满秩，state_dim=3）。

**继续加到 `mu=3` 不可靠，不能用**：攻击轨迹本身只有 4-5 轮，`mu=3` 时
`start=max(nu-1,mu)=3`，每条轨迹能提供的可用状态转移几乎被榨干,Gramian 条件数飙到 2.57e11
（数值上已经退化,矩阵里出现大量精确的 0/整数比例值，是数据不够撑起这个阶数的信号，不是真实
动力学），`controllability_rank` 反而跌到 3/4（不满秩）——这是"阶数超过数据能支撑的上限"
的教科书信号,不是"系统本身不可控"。

**结论（go/no-go）：通过，但用 `nu=1, mu=2` 而不是原计划的 `nu=1, mu=1` 作为 Phase D/E 的
状态阶数**——单一"是否插了这一轮提醒"的记忆不够，需要看最近两轮的插入历史才能让线性模型
看出"持续提醒"和"偶尔提醒"的区别，这和 Phase A（常提醒，持续 100%）与 Phase B（随机激励，
平均 50%）效果强度不同这一现象本身是自洽的。`mu=2` 下 `richer_abs_sign`
（`abs_sign_extra_features`）比纯 ARX 拟合更好（held-out rollout MSE 0.043 vs 0.051），
选它作为 Phase D/E 用的模型。

held-out 攻击 id（`split_col="attack_id"`, seed=0, 25%）：`safemtdata_0074/0169/0257/0289/
0324/0329/0476/0530`——这 8 个攻击没有进入 Phase C 的拟合数据,专门留给 Phase E 做真正的
样本外闭环验证。

## Phase D：KoopmanMPCController 接上真实模型

`control.py::KoopmanMPCController` 在写测试时用的是手造的合成 surrogate,这次是第一次接上
真实拟合出的模型,过程中发现并修了一个 bug：**`modeling/koopman.py::surrogate_from_arrays`
最初从 `A.shape[0]` 推断 `state_dim`,但 `A` 是提升后的维度（`d_psi`）,只有 `no_extra_features`
（ARX，无提升）时才等于真正的状态维度；用 `abs_sign_extra_features` 这种真提升时 `d_psi
> state_dim`,`step()` 里 `eta_next[:state_dim]` 的截断会用错误的长度截断,导致下一次调用
`_psi()` 时输入维度对不上,矩阵乘法直接报错。** 用真实模型做冒烟测试时当场触发（`A.shape[0]=5`
但真实 `state_dim=3`）,已修复为要求调用方显式传 `state_dim`（`ReducedStateConfig(nu,
mu).state_dim`），`tests/test_koopman.py` 新增一个用 `abs_sign_extra_features` 的回归测试
专门覆盖这个情况（纯 ARX 测试测不出来,因为两个维度那时候恰好相等）。

`scripts/run_defended_screening.py` 新增 `--controller koopman_mpc`（读
`koopman_fit_report.json`、`--koopman-model-key {arx,richer_abs_sign}`、`--koopman-nu/mu/
horizon/repeat-penalty`）和 `--attack-ids`（显式攻击 id 列表,替代随机抽样，配合
`attack_bank.select_attacks_by_id`/`adversarial_screening.run_adversarial_screening` 新增的
`attack_ids` 参数——这样 Phase E 四臂可以在完全相同、且不在 Phase B 训练集里的攻击集合上
对比,而不是像 Phase A 那样靠不同 `attack_rng_seed` 只是"降低"重叠概率)。

## Phase E：四臂闭环验证（已提交）

四个作业都用同一批 8 个 held-out 攻击 × 2 seed = 16 条轨迹：

| 作业 | Controller | sbatch |
|---|---|---|
| zero_control | 无控制 | `run_koopman_defense_phaseE_zero_control.sbatch` |
| constant_remind | 常提醒 | `run_koopman_defense_phaseE_constant_remind.sbatch` |
| threshold | y_min=0.7 阈值反馈（经典基线，Koopman-MPC 必须打赢的对照） | `run_koopman_defense_phaseE_threshold.sbatch` |
| koopman_mpc | `richer_abs_sign`, nu=1/mu=2, horizon=2, 无代价惩罚 | `run_koopman_defense_phaseE_koopman_mpc.sbatch` |

```bash
squeue --me
for c in zero_control constant_remind threshold koopman_mpc; do
  echo "=== $c ==="
  cat persona_drift_control/outputs/koopman_defense_phaseE_$c/adversarial_screening_report.md 2>/dev/null
done
```

## 下一步（四臂结果出来后补充）

（结果出来后补：四臂 new-Q1 斜率/t/p 对比表，`koopman_mpc` 是否显著优于 `zero_control`、
是否不输给/优于 `constant_remind`/`threshold`；`inserted_tokens` 总量对比（`koopman_mpc`
若能用比 `constant_remind` 更少的提醒次数达到接近的安全性,是"建模复杂度换来了收益"的直接
证据）；如果 `koopman_mpc` 打不过 `constant_remind` 这种最笨的常数策略,如实记录为负结果,
不是失败,是诚实的方法学结论。）
