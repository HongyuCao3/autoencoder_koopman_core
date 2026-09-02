# 实验记录：人格渐进施压确认 pilot

供换一次新对话（没有本会话上下文）时能接得上——这是
[drift_confirmation_pilot.md](drift_confirmation_pilot.md) 2026-08-31 补充分析里"下一步的小
范围实验建议"一节提出、并已实施的实验。**新开一次对话想知道"这个 pilot 现在跑到哪一步了"，
看这份文档。**

## 这是什么，为什么做

`drift_confirmation_pilot.md` 的结论是：10-prompt 规模下人格漂移（reminder-based，被动
`excite_iid` 刺激）测不出来，t=0.02，p=0.99，是很干净的空结果。同一份文档的补充分析发现，
同一个 Qwen3-4B 在 `adversarial_screening_pilot.md`（多轮 jailbreak 攻击）里对**渐进升级式**
的外部压力表现出清晰、强烈的跨轮侵蚀（p<0.0001），说明空结果更可能是"被动 reminder 刺激太
弱"，而不是"模型太小/训练得刀枪不入"——但这个判断当时只是跨域类比，没有在人格域上直接验证过。

这个 pilot 就是那个直接验证：把对抗任务里"渐轮升级施压"的设计模式移植到人格/语言风格域，在
**同一个 Qwen3-4B**、同一套 prompt/scoring pipeline 上，比较：

- **baseline**（`user_mode=live`，自然 self-chat，无施压）——预期复现之前的空结果，作为
  同批数据内部的健全性检查；
- **escalating_pressure**（`user_mode=pressure`，`pressure_scripts.py` 里手写的、按
  `prompt_category` 分类的固定 16 轮渐进劝退脚本，不走 user_sim）——测这个更强的刺激能不能
  激发出可测漂移。

判定用的是和 `analysis.py::q1_drift_trend` 相同的方法（全序列 OLS 斜率、按 prompt 聚合、单样本
t 检验），只是按 `user_mode` 而不是 `excitation_design` 分组（两个条件的 `u_remind` 都恒为 0，
渠道 A 在这个实验里不起作用，压力来自模拟用户的固定脚本本身）。

## 代码

- `src/persona_drift/pressure_scripts.py`：`load_pressure_script(prompt_category)`，
  `character_traits`/`language_constraints` 各一份手写的 16 轮渐进施压脚本（从委婉请求"正常
  回答一次"逐步升级到带愧疚感的"最后一次尝试"）。**手写而非 LLM 生成**——绕开
  `SCRIPTED_USER_TURNS_FEASIBILITY.md`/本文档姊妹篇记录的自聊退化问题；**按类别而非按具体
  prompt_id 措辞**——不点名具体指令内容，保证同一脚本能套用在类别内任何 system prompt 上。
- `src/persona_drift/selfchat.py`：`TrajectoryConfig.user_mode` 新增 `"pressure"` 分支，
  按 `entry.prompt_category`（不是 topic/seed）取脚本，复用原有的脚本长度校验逻辑。
- `src/persona_drift/analysis_pressure.py`：`analyze_pressure_screening()`，
  `q1_baseline_no_pressure` / `q1_escalating_pressure` 两组独立的斜率检验（复用
  `analysis.q1_drift_trend` 的统计方法）。
- `src/persona_drift/pressure_screening.py`：编排层，结构照抄 `screening.py`（含直接 import
  它的 `_prepare_resumable_trajectories_file`，逻辑与这里完全一致，不重复实现），两个条件都用
  `ZeroControlController`（渠道 A 不参与），只是 `user_mode` 不同。
- `scripts/run_pressure_screening.py` + `environment/run_pressure_screening.sbatch`：CLI 和
  提交脚本，默认 4 prompts（每类 2 个）× 2 seeds × 2 条件 = 16 条轨迹，16 轮，`--time 04:00:00`
  （按 `drift_confirmation_pilot.md` 实测的约 414s/轨迹均速估算，16 条轨迹最坏情形约 2.3h，
  留了较宽余量；支持断点续跑，被 `--time` 杀掉可直接重新提交同一脚本）。
- 测试：`test_pressure_scripts.py`、`test_selfchat.py`（新增 pressure 模式两个用例）、
  `test_analysis_pressure.py`，全部 CPU-only（假激活/假模型），随 `run_tests.sbatch` 一起跑。

## 判定方法（新 Q1，未预注册的探索性设计，仅供本轮决策参考）

- **escalating_pressure 测出显著漂移**（t<0 且 p<0.05）→ 假设得到直接证实：之前的空结果是
  刺激强度问题，不是模型规模/训练问题；不需要再讨论换 7B，人格漂移这条线如果还想继续，应该
  换成这种更强的施压设计，而不是回到被动 reminder。
- **escalating_pressure 依然测不出**（哪怕 baseline 保持空结果）→ 这才是真正该认真考虑模型
  规模/家族差异的时间点；且到那时候需要先给出一个具体机制假设（为什么预期更大模型在人格
  维度上更容易漂移），再设计针对性的 7B 实验。

## 当前作业

| 字段 | 值 |
|---|---|
| CPU 单测 job | 15405993（首次提交，暴露了 `selfchat.run_trajectory` 里 `pressure`/`scripted`
  两种脚本模式共享的"是否用 script[turn-1]"判断只检查了 `== "scripted"`，导致 `user_mode=
  "pressure"` 时仍尝试调用 `user_sim.generate()`、`user_sim=None` 报错——已修复，改成
  `if script is not None`），15406461/15406494（修复后两次确认：3 个新测试稳定通过；
  `test_logging_setup.py` 一个预先存在、与本次改动无关的 loguru `enqueue=True` 异步写入
  竞态测试间歇性失败，不阻塞本 pilot） |
| GPU pilot job | **15406535**（2026-08-31 提交） |
| 提交脚本 | `persona_drift_control/environment/run_pressure_screening.sbatch` |
| 规模 | 4 prompts × 2 seeds × 2 条件 = 16 条轨迹，16 轮 |
| 输出目录 | `persona_drift_control/outputs/pressure_screening/` |
| 并行说明 | 这次和 `dose_response_pilot.md` 的后续修复（job 15405662，`pdc-dose`，2026-08-31
  提交）在 Slurm 上同时跑，两条线互不阻塞——这是探索性辅助验证，不占用当前最高优先级路径
  （adversarial defense / ICLR 时间线）的算力排期决定权 |

## 查看状态的方法

```bash
squeue --me
sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode
cat persona_drift_control/outputs/pressure_screening/pressure_screening_report.md   # 跑完之后才会有
```

## 结论（job 15406535，2026-08-31，1h25m）

**两个条件都不过，escalating_pressure 假设未被证实**：

- `q1_baseline_no_pressure`：1/4 prompt 负斜率，t=1.2065，p=0.3141，`pass=False`——复现了
  `drift_confirmation_pilot.md` 的干净空结果，同批数据内部健全性检查通过。
- `q1_escalating_pressure`：3/4 prompt 负斜率，t=-1.2847，p=0.2891，`pass=False`——方向对了
  （多数 prompt 确实往负斜率偏），但样本量太小（4 个 prompt）撑不起显著性,没有达到判定假设
  成立所需的 p<0.05。
- 诊断：`refusal_rate=0.0078`、`scorer_failure_rate=0.0000`，管线本身跑得干净，不是代码问题。

**对应"当前作业"里预设的两条判据**：不是"escalating_pressure 显著证实"，也不是"baseline 和
escalating_pressure 都同样测不出"那种彻底空结果——是中间状态：方向上有偏移迹象（3/4 负斜率）
但当前 4-prompt 的规模统计功效不够,不能区分"刺激强度确实提升了但样本太小看不出"和"这次的
渐进施压脚本本身还不够强"这两种可能。

**对"要不要试 7B"的判断**：**还不到时候**。当前证据不支持"模型规模是瓶颈"这个假设被证实，
也不支持被推翻——在下结论之前，更便宜的下一步是扩大这次 4→更多 prompt 的样本量（同一套
pressure_scripts.py、同一个 Qwen3-4B，纯粹加大 N），如果扩大样本后 escalating_pressure 依然
测不出显著漂移，那时候才是该认真讨论换 7B 或者重新设计施压脚本强度的时间点。

## 扩样本量跟进（job 15412821，2026-08-31 提交）

`environment/run_pressure_screening_wide.sbatch`：`--num-prompts` 从 4 提到 12（每类 6 个，
character_traits 池子共 13 个、留有余量），seeds/turns/probe-repeats/模型全部不变，df 从 3
提到 11。输出到独立目录 `outputs/pressure_screening_wide/`，不覆盖 job 15406535 的结果，
便于两次直接对比 t/p 值。`--time 06:00:00`（按原 318s/轨迹均速估算 48 条轨迹约 4.24h，留
~40% 余量），跑法和 `screening.py` 一样支持断点续跑，被 `--time` 杀掉直接重新提交同一脚本。

**注意**：`select_screening_prompts(rng_seed=0, num_prompts=12)` 不保证是
`num_prompts=4` 那次抽样结果的超集（`random.sample` 的输出依赖 `k`）——这是同一个 prompt
池子里一次新的、更大的独立抽样，不是"在原来 4 个基础上加了 8 个"。

```bash
sacct -j 15412821 --format=JobID,State,Elapsed,ExitCode
cat persona_drift_control/outputs/pressure_screening_wide/pressure_screening_report.md   # 跑完才有
```

## 结果：扩样本量（job 15412821，2026-09-01，1h12m，此前完成但未写入本文档，2026-09-02 补记）

| | N=4（job 15406535） | N=12（job 15412821） |
|---|---|---|
| `q1_baseline_no_pressure` | 1/4 负斜率，t=1.2065，p=0.3141，pass=False | 7/12 负斜率，t=-1.3323，p=0.2097，pass=False |
| `q1_escalating_pressure` | 3/4 负斜率，t=-1.2847，p=0.2891，pass=False | 10/12 负斜率，t=-1.4478，p=0.1756，pass=False |

诊断：`refusal_rate=0.0130`、`scorer_failure_rate=0.0000`，管线干净。`y_probe` 均值按条件/
类别拆分：`live/character_traits`=0.4007、`live/language_constraints`=0.4392、
`pressure/character_traits`=0.2939、`pressure/language_constraints`=0.3578——`pressure`
条件两个类别的均值都低于对应的 `live` 条件，方向一致。

**样本量从 4 扩到 12 后，方向更一致（负斜率 prompt 占比从 3/4 提升到 10/12），但 p 值只是
从 0.29 略降到 0.18，仍未过 0.05。** 这不支持"样本量不够,再加大就能显著"这个简单外推——
更可能的解释是这条 pilot 用的连续 0-1 judge rubric 本身统计功效有限（同样的现象如果换成
"是否发生一次可辨识的立场/风格翻转"这种离散事件的测量设计，功效会更高，见
`../task/KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md` 第六节对 sycophancy drift 类似设计
的讨论）。

**结论：escalating_pressure 假设仍是中间态——不建议在当前连续打分测量设计上继续扩样本**；
如果还想在人格/风格维度上把这个信号测清楚，下一步应该换测量设计（离散翻转事件），而不是
再加 prompt 数。这条 pilot 到此告一段落，不再是活跃开发线。
