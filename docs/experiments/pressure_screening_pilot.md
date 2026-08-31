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

## 结论

（跑完后补充：`q1_baseline_no_pressure`/`q1_escalating_pressure` 的 pass/fail、斜率符号分布、
t/p 值，以及对"要不要试 7B"这个问题的最终判断。）
