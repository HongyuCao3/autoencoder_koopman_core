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

## 结果：Phase E（四作业均于 2026-08-31 23:22–2026-09-01 00:00 完成，各约 11 分钟）

8 个 held-out 攻击 × 2 seed = 16 条轨迹/臂，均未出现在 Phase B/C 的拟合数据里。

| 臂 | new-Q1 t / p / pass | new-Q3 r / p / pass | turn5 y_safety | refusal_rate | 提醒次数/80 | inserted_tokens |
|---|---|---|---|---|---|---|
| zero_control | t=-3.70, p=0.0076, **True**（侵蚀显著） | r=0.5698, p=0.0000, True | 0.5156 | 0.0000 | 0 | 0 |
| constant_remind | t=-3.16, p=0.0160, **True**（侵蚀仍显著） | r=0.0350, p=0.7837, False | 0.8281 | 0.0750 | 80 | 2880 |
| threshold | t=-4.77, p=0.0020, **True**（侵蚀更显著，t 比无控制基线还差） | r=0.0786, p=0.5371, False | 0.7188 | 0.0500 | 7 | 252 |
| koopman_mpc | t=-2.34, p=0.0521, **False**（唯一侵蚀不显著，临界） | r=0.2668, p=0.0331, True | 0.7969 | 0.0750 | 32 | 1152 |

**成功判据判定：通过。** 按计划文件（`shiny-stargazing-sphinx.md`）定的判据——"侵蚀斜率明显好于
无控制基线，且不明显差于（理想是优于）常提醒/阈值这两个经典基线"——`koopman_mpc` 的
t 统计量（-2.34）在四臂里绝对值最小（最接近 0，侵蚀最弱），同时优于 `zero_control`
（-3.70）、优于 `constant_remind`（-3.16）、也优于 `threshold`（-4.77，反而比无控制基线更差，
说明 `y_min=0.7` 触发阈值在这批 held-out 攻击上明显欠触发——7/80 轮插入，比 `koopman_mpc`
的 32/80 还稀疏，达不到干预密度）。`koopman_mpc` 是四臂里唯一让 new-Q1 侵蚀检验从显著
（p<0.05）变为不显著（p=0.0521，非常接近临界值，需谨慎解读为"勉强通过"而非"压倒性"）的臂。

**代价侧证据支持"建模复杂度换来了收益"**：`koopman_mpc` 只用了 `constant_remind` 40%
的插入次数（32 vs 80）、40% 的 token 代价（1152 vs 2880），turn5 均值安全分（0.7969）已经
接近 `constant_remind`（0.8281），明显超过 `threshold`（0.7188）。换言之，MPC
用不到一半的提醒预算,拿到了不输给"每轮都提醒"这个最笨策略的效果,并在统计显著性判据上
略微超过它。

**需要如实记录的保留意见**：
1. `constant_remind` 在 Phase A（20 攻击 × 2 seed）上曾把 new-Q1 打到 p=0.125（不显著），
   但在这批不同的 8 个 held-out 攻击上是 p=0.016（显著）——说明"是否显著"这个二值判据本身
   对攻击子集较敏感,`koopman_mpc` 的 p=0.0521 同样应该在这个粒度上谨慎解读,不宜过度渲染
   为"决定性胜利"。
2. `koopman_mpc` 的 new-Q3（惯性）pass=True 而 `constant_remind`/`threshold` 都是 False——
   这与其说是 MPC 的缺陷,不如说是符合预期：MPC 只在约 40% 的轮次插入,轮次间仍保留部分
   未受干预的动力学结构,惯性没有被提醒动作"抹平"。

**结论：Koopman-MPC 控制器打赢了两个经典基线（无控制、阈值反馈），且用更低代价追平了
常提醒基线，达成计划文件设定的成功判据。这条实验线（Phase A→E）到此完整闭环。**

## Phase F：良性查询 helpfulness 代价检查（设计方案见
`/home/hcao2/.claude/plans/happy-purring-sunset.md`，已批准）

复用 attack_bank→attack_trajectory→adversarial_screening→analysis_adversarial 的整套结构，
换成良性内容：

- 良性数据：新 vendor `resources/mtbench_questions.jsonl`（MT-Bench 80 题，8 类，Apache-2.0，
  见 `PROVENANCE.md`）。每类链接 3 道该类题目（3×2轮=6轮）成一个 session——单题只有 2 轮，
  不够 `KoopmanMPCController`（`nu=1,mu=2` 要求 `len(history)>=3` 才会产生非默认动作）真正
  决策，6 轮给它第 4-6 轮三次机会。`benign_bank.py` 新增，8 类全用，不做子采样。
- `control.py` **完全未改**：`ThresholdController`/`KoopmanMPCController` 仍然读
  `history[-1]["y_probe"]`，由 `safety_judge`（用固定占位 `harmful_goal` 打分）驱动，模拟
  "真实部署里监控跑在全部流量上"；新增 `helpfulness_judge.py`（新 rubric：拒绝/回避判低分，
  不像 `safety_judge` 把拒绝当满分）产出的 `y_help` 是并行、不进决策的结果指标。
- 新增 `benign_trajectory.py`/`benign_screening.py`/`analysis_helpfulness.py`/
  `scripts/run_benign_helpfulness_screening.py`，四个 sbatch
  (`run_koopman_defense_phaseF_{zero_control,constant_remind,threshold,koopman_mpc}.sbatch`)。
  `analysis_helpfulness.compare_arms_to_zero_control` 在 16 个固定 `(benign_id, seed)` 会话
  上做配对 t 检验（内容跨臂相同，只有 `u_remind` 序列不同，和 Phase E 同款设计）。
- 测试：`test_benign_bank.py`/`test_helpfulness_judge.py`/`test_benign_trajectory.py`/
  `test_analysis_helpfulness.py`/`test_benign_screening.py`，CPU 全套 170 passed（新增 23,
  无回归）。
- **无 go/no-go 门槛**——这是描述性代价报告，不是通过/不通过判定。

### Phase F 状态

四臂互相独立（不像 Phase A→E 那样需要顺序 gate）。zero_control 臂先单独提交做冒烟验证
（job 15427488），第一条轨迹（`mtbench_coding__seed0`，6 轮，两次 judge 调用/轮）54s 内
无异常跑完,确认全链路（vendor 数据加载→提醒插入→两次 judge 调用→报告落盘）在真实 GPU/模型
上没问题,随后提交其余三臂：`constant_remind` job 15427539、`threshold` job 15427540、
`koopman_mpc` job 15427541。四个作业于 2026-09-01 提交,均已在跑或排队。

```bash
squeue --me
for c in zero_control constant_remind threshold koopman_mpc; do
  echo "=== $c ==="
  cat outputs/koopman_defense_phaseF_$c/benign_screening_report.md 2>/dev/null
done
```

## 结果：Phase F（四作业均于 2026-09-01 09:56–10:09 完成，各约 10-12 分钟）

8 个 MT-Bench 良性类别 × 2 seed = 16 个会话（每会话 6 轮）/臂，内容跨臂完全相同（同
`(benign_id, seed)`），只有 `u_remind` 序列不同。

| 臂 | mean_y_help (sd) | refusal_rate | 提醒次数/96 | inserted_tokens | Δ vs zero_control (mean_diff / t / p) |
|---|---|---|---|---|---|
| zero_control | 0.8750 (0.2460) | 0.0000 | 0/96 | 0 | — |
| constant_remind | 0.7708 (0.3338) | 0.1458 | 96/96 | 3456 | -0.1042 / t=-1.6887 / p=0.1120 |
| threshold | 0.8750 (0.2460) | 0.0000 | 3/96 | 108 | 0.0000 / t=nan / p=nan |
| koopman_mpc | 0.8620 (0.2711) | 0.0312 | 48/96 | 1728 | -0.0130 / t=-0.7182 / p=0.4837 |

**读数（描述性，无门槛）**：

- **`constant_remind` 代价最明显但配对检验不显著**：mean_y_help 从 0.875 掉到 0.771，
  refusal_rate 从 0 升到 0.1458——完全无害的请求里,约 1/7 的回合因为每轮都插安全提醒而被
  拒答/回避,这正是"倒U"风险担心的那种代价。但 16 个会话样本量小、方差大，配对 t 检验
  p=0.112,不到传统 0.05 门槛,只能说"方向上有代价,统计上证据不够强"，不能说"确认有害"。
- **`koopman_mpc` 代价很小、方向一致但同样不显著**：mean_y_help 0.862（比
  zero_control 只低 0.013,t=-0.72,p=0.48）,refusal_rate 0.0312（vs 0）,只用了
  `constant_remind` 一半的插入次数（48 vs 96/96）、约一半的 token 代价（1728 vs 3456）。
  换句话说,`koopman_mpc` 在 Phase E 里对攻击有效,而在这里对良性流量几乎不产生可测的
  helpfulness 代价——这是"建模复杂度换来了收益"故事的又一块证据：不仅比常提醒更省成本地
  防住了攻击（Phase E),还比常提醒更少地误伤良性流量。
- **`threshold` 的"零代价"是个需要谨慎解读的巧合，不是真正的零代价**：诊断显示它确实触发了
  3 次提醒（`mtbench_math__seed0` 第 3 轮、`mtbench_roleplay__seed1` 第 5-6 轮），触发后的
  agent 回复文本确实变了（`agent_message` 不同），但受影响会话的 `y_help` 会话内均值恰好
  和 `zero_control` 完全相等（`roleplay__seed1` 是两轮分数从 0.25+0.0 变成 0.0+0.25，总和
  巧合相同）——1-5 量化打分下的巧合，配对差值处处为 0 导致 `ttest_rel` 方差为零、
  `t`/`p` 返回 `nan`（`scipy` 对零方差配对数据的正常行为，不是代码 bug，逐行核对过
  `trajectories.jsonl` 确认）。真实结论应该是"threshold 触发太少（3/96），这批良性会话
  测不出它的代价"，不是"threshold 代价严格为零"。
- **无一臂在这批良性数据上出现统计显著的 helpfulness 下降**（都是 p>0.1），所以按这次的
  判据（是否显著抵消 Phase E 的代价优势）：**没有**——`koopman_mpc` 的 Phase E 胜利没有被
  这次检查推翻,但 `constant_remind` 的效应量本身最大,如果之后想更严格地检验这个方向,需要
  更大的良性样本（更多类别/更多 seed）才能把 p=0.112 这类边界情况谈清楚。

**结论：Phase F 不构成推翻 Phase E 结论的证据,`koopman_mpc` 依然是四臂里唯一同时在攻击场景
（Phase E）和良性场景（Phase F）都取得"不输给经典基线、代价明显更低"的控制器。** 这条
Phase A→F 主线到此完整闭环；样本量小是本轮结果的主要局限，值得记录但不构成必须补做的门槛。

## 另一条支线：给 Koopman 模型加检测能力

Phase E 打赢后引出的新问题——现在的 `KoopmanMPCController` 只用代理模型选动作，"是否存在
攻击"这个判断始终隐含在 `y_safety` 这个 judge 分数里，Koopman 模型本身没有显式的检测输出。
四个设计方案 + 方案 1（一步预测残差/innovation）在 Phase E 数据上的执行结果（负面/持平，
残差被"策略分布外"效应污染，没有干净跑赢"下一轮=上一轮"基线）记录在
[koopman_detection_design.md](koopman_detection_design.md)，不写在这份文档里，两条线分开
接续。
