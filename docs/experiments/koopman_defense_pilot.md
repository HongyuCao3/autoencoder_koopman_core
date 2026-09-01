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

## 下一步（Phase B 跑完后）

用 `outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl`：
1. `modeling.dataset.build_identification_dataset(rows, ReducedStateConfig(nu=1, mu=1),
   y_col="y_safety")`（`u_col`/`id_col` 用默认的 `u_remind`/`trajectory_id`，已经在
   `attack_trajectory.py` 里对齐）。
2. 用 `split_by_system_prompt_id(..., split_col="attack_id")` 切出一部分留作 Phase E
   的 held-out 验证集（注意：Phase B 本身的 30 个攻击和 Phase A/基线的 20 个攻击之间用了不同
   `attack_rng_seed`，重叠概率降低但未严格保证不相交，见 sbatch 脚本注释——如果 Phase E
   要更严格的不相交保证，需要在这里补一个显式的 attack_id allowlist）。
3. `KoopmanSurrogate(extra_features_fn=no_extra_features).fit(dataset)`（先 ARX 基线），
   `controllability_diagnostics(model.A, model.B, horizon=...)`——检查 B 是否退化，
   这是 Phase C 的 go/no-go 判定。
