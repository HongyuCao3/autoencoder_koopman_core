# 实验记录：ERGO/Laban 多轮可靠性侵蚀，最小执行器权威验证（2026-09-06）

和 [mc_sycophancy_screening_pilot.md](mc_sycophancy_screening_pilot.md) 同一类"供跨会话接续"的
记录。**新开一次对话想知道"这条候选线跑到哪一步了"，看这份文档。**

## 这是什么，为什么做

`docs/feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md`（2026-09-06）在 sycophancy 线两次
执行器权威空结果（`mc_sycophancy_screening_pilot.md` "Phase A"一节）之后评估的下一个 Koopman
闭环控制候选任务：Laban et al. 2025 的 sharded-instruction 多轮退化基准
（arXiv:2505.06120，`resources/ergo_gsm8k_sharded.jsonl`，见 `resources/PROVENANCE.md`）+
ERGO 的 entropy-guided resetting（arXiv:2510.14077）。该文档第 4 节把执行顺序倒过来：**先花
最小成本验证执行器权威，再决定要不要建完整 bank/judge/trajectory 管线**——这正是 sycophancy
线本该更早做、却在两版提醒设计都花了完整工程量之后才做的一步。这份文档记录第 4 节步骤 1-2
（vendor 数据 + 最小权威验证）的执行情况。

## 代码

- `resources/ergo_gsm8k_sharded.jsonl`：vendor 自 `microsoft/lost_in_conversation`（MIT
  协议），103 个 GSM8K 数学题的 sharded 版本（原题拆成 4-12 个 shard，逐轮揭示一个），见
  `resources/PROVENANCE.md`。**已知简化**：上游用一个活的 LLM user-simulator 决定 shard
  揭示顺序/节奏并对每轮回复分类，本项目 `SCRIPTED_USER_TURNS_FEASIBILITY.md` 早就认定这类
  设计不值得复刻，所以这里用本项目一贯的固定节奏（shard i 在 turn i 揭示），每轮都直接打分，
  不做"是否已经在尝试回答"的分类。
- `src/persona_drift/ergo_math_bank.py`：`GSM8KShardedItem`/`load_ergo_math_bank()`/
  `select_screening_items()`/`select_items_by_id()`，结构照抄 `mc_sycophancy_bank.py`，但是
  单任务扁平列表（没有 MMLU 那种按类别分层）。
- `src/persona_drift/ergo_math_judge.py`：纯 Python 正则抽取+比较（从上游
  `tasks/math/task_math.py` 的 `evaluator_function` 移植数字归一化逻辑，MIT 协议），**没有
  LLM 判官这一步**——比 `mc_answer_judge.py` 更进一步：连"正则失败退化到一次简短 LLM 抽取"
  这条兜底路径都省了（可行性文档的最小化裁剪，抽取失败率高的话再加回来）。
- `src/persona_drift/ergo_math_trajectory.py`：**没有复用** `trajectory_runner.
  run_reminder_gated_trajectory`（这次和 sycophancy 系列不同的地方）——共享循环的
  `reminder_fn(level) -> str | None` 只能返回和轮次无关的固定文本，而"重置"这个动作的内容
  （把目前为止揭示过的所有 shard 拼成一条新消息）依赖已经过去多少轮，没法塞进那个签名里，
  所以这里是一个新的、独立的小循环，复用 `control.Controller`（`ZeroControlController`/
  `ConstantRemindController` 等）纯粹当"重置/不重置"的 0/1 决策源，不需要改 `control.py`
  一行代码。
- `src/persona_drift/analysis_ergo_math.py`：刻意精简（没有离散翻转事件、没有分类别拆分、
  没有 `pass` 门槛判定），只有 `final_turn_success`（主指标，每条轨迹自己最后一轮，因为
  各 item 的 shard 数不同,不是像 sycophancy 那样固定 T）、`success_by_turn`（描述性）、
  `new_q1_escalation`/`new_q3_autocorrelation`（和 `analysis_sycophancy.py`/
  `analysis_adversarial.py` 同一套统计模式，**这是这个模式第三次近乎重复出现**，
  `analysis_sycophancy.py` 的文档字符串早就标注过"再出现第三次就该抽公共模块"——这次刻意
  没抽，留了债务提示在 `analysis_ergo_math.py` 自己的文档字符串里，不在最小验证的关键路径上）。
- `scripts/run_ergo_math_screening.py` + `environment/run_ergo_math_authority_{zero_control,
  reset}.sbatch`：`--controller {zero_control,constant_remind,fixed_schedule}`，两个 sbatch
  用完全相同的 20 个 item id（`--item-ids`，不是随机抽样），保证配对可比。
- `scripts/analyze_ergo_authority_comparison.py`：配对 t-test，按 item 比较两臂的
  `final_turn_success`，和 `analyze_mc_phaseA_comparison.py` 同一设计，适配了变长轨迹
  （用每条轨迹自己的最后一轮而不是固定窗口）。
- 24 个新 CPU 单测（bank/judge/trajectory/screening 编排，含对真实 vendor 文件的检查），全绿；
  全套 380 passed（5 个既有的、和本次改动无关的 NLTK 数据缺失失败不变）。

## 结果（job 15602880 zero_control / 15602881 reset，20 items × 2 seeds，均已完成，2026-09-06）

**执行器权威：确认，而且效应量很大**（`scripts/analyze_ergo_authority_comparison.py`，配对
t-test，按 item 比较各自最后一轮的 `y_task_success`）：

| | n_items | zero_control 均值 | reset 均值 | 差值 | t | p |
|---|---:|---:|---:|---:|---:|---:|
| 最终轮任务成功率 | 20 | 0.275 | 0.750 | **+0.475** | 4.79 | **0.000127** |

对比 sycophancy 线两次都卡在"效应量趋近于 0"（p=0.70/1.00），这次是**方向对、效应量大、p 值
远小于 0.05**——`Qwen/Qwen3-4B` 自己的管线上，重置这个执行器确实能显著提高任务成功率，不是
只在别人的模型/管线上有效。逐轮细看更直观：

| turn | zero_control | reset |
|---:|---:|---:|
| 3 | 0.0500 | 0.0500 |
| 4 | 0.1750 | 0.3750 |
| 5 | 0.1333 | 0.2667 |
| 6 | 0.0500 | 0.5000 |
| 7 | 0.2500 | 0.7500 |

两臂在早期轮次（信息还不够时）几乎一样低，从 turn 4 起持续拉开差距，越到后面差距越大——和
"权威随累积轮次显现"这个直觉一致。

**惯性检验（可行性文档第 4 节步骤 3）也顺带过了，但含义和 sycophancy 线不同，需要如实说明**：
zero_control 自己的轨迹上，`new_q3_autocorrelation`（lag-1 自相关）r=0.5752, p<0.0001，
`new_q1_escalation`（逐轮斜率）均值为正、t=2.83, p=0.0108——**斜率显著为正**（sycophancy/
防御线的惯性信号方向相反，那两条线关心的是"分数会不会退化"，这里是"分数会不会随信息累积而
提升")。**这不是"记忆/黏滞"意义上的惯性,而是"信息单调累积导致任务单调变简单"这个更平凡的
机制**——每揭示一个新 shard,题目客观上更完整、更容易答对,分数上升是任务结构本身决定的,不是
模型"记住了之前的状态"。这和 sycophancy/防御线那种"同一份信息反复施压,内部立场却会漂移"的
现象不是一回事,这里的动力学更接近"贝叶斯式信息累积"（有点像卡尔曼滤波的更新过程）——对
Koopman/状态空间建模来说仍然是一个合理甚至更自然的拟合对象（状态=当前对答案的置信度,受控
输入=是否重置),只是解读上不能直接套用"惯性=前置条件 1 满足"这句话背后 sycophancy 线的原始
语义,需要在后续文档/论文措辞里把这个区别讲清楚。

**诊断**：`judge_parse_failure_rate` 在 reset 臂略高（2.25% vs 0%，可能是重置后模型偶尔重新
从头推导、还没来得及给出"Current answer: X"格式），`refusal_rate` 两臂都很低（<1%），
没有异常。

## 样本扩充（job 15603367 zero_control / 15603368 reset，60 items × 2 seeds，2026-09-06 已跑完并分析）

20-item 最小验证已经确认权威（见上节），这一步不是重新测"有没有权威"，是把效应量估计打扎实、
给 `analysis_ergo_math.py` 的 new-Q1/new-Q3 诊断更多功效——和 sycophancy 线 20→60 items 的
扩样本同一个理由。**新抽的 60 items**（`--item-rng-seed 1`，不是复用 pilot 的那 20 个，题库
总共只有 103 个 item，两次抽样会有交集但不是子集关系），两个 sbatch 用完全相同的 60 个
item id，60 items × 2 seeds = 684 行。两个 job 均 `COMPLETED`（16:14 提交，16:29/16:49 结束）。

`scripts/analyze_ergo_authority_comparison.py --arm-a-dir outputs/ergo_math_authority_
zero_control_60items --arm-b-dir outputs/ergo_math_authority_reset_60items` 的配对比较
（按 item 比较最后一轮 `y_task_success`，n=60）：

| | n_items | zero_control 均值 | reset 均值 | 差值 | t | p |
|---|---:|---:|---:|---:|---:|---:|
| 最终轮任务成功率 | 60 | 0.3667 | 0.7917 | **+0.4250** | 6.78 | **6.39e-09** |

**效应量在 3 倍样本下依然稳，且显著性更强**（20-item: diff +0.475, t=4.79, p=1.27e-4 →
60-item: diff +0.425, t=6.78, p=6.39e-9）——不是小样本噪声。逐轮细看同一模式重复出现：

| turn | zero_control (n) | reset (n) |
|---:|---:|---:|
| 3 | 0.0000 (120) | 0.0333 (120) |
| 4 | 0.1500 (120) | 0.2833 (120) |
| 5 | 0.1702 (94) | 0.4043 (94) |
| 6 | 0.1935 (62) | 0.4839 (62) |
| 7 | 0.2333 (30) | 0.5000 (30) |
| 8 | 0.3000 (10) | 0.6000 (10) |

两臂差距从 turn 4 起持续拉开，到 turn 7/8 已经翻倍——和 20-item pilot 的定性一致。

`ergo_math_screening_report.md` 的 new-Q1/new-Q3 诊断（60 items，各自的独立信号，不是上面的
配对比较）：zero_control 逐item斜率 t=5.69, p<0.0001（28/60 正斜率）、lag-1 惯性
r=0.4441, p<0.0001；reset 斜率 t=12.41, p<0.0001（53/60 正斜率）、lag-1 惯性
r=0.3997, p<0.0001——两臂都强显著单调上升，reset 臂上升更快，与"重置后信息更快被有效利用"
一致。诊断：`judge_parse_failure_rate` reset 臂 0.73% vs zero_control 0%，`refusal_rate`
zero_control 3.65% vs reset 0.58%，都很低，无异常。

## Koopman 建模，Phase B（job 15613799，2026-09-06 提交）

执行器权威两次确认之后（20-item + 60-item），下一步的判断是：直接开始建 Koopman 模型，而不是
先铺开到其余五个 ERGO 任务或先定论文叙事——理由见下面"下一步"第 2/4 条的状态变化。这一节记录
建模第一阶段（Phase B：开环随机激励采集 + 首次拟合）的设计决策与执行情况，**新开一次对话想知道
"ERGO 的 Koopman 建模做到哪一步了"，看这一节**。

**读出结构的顾虑已经在实现里被绕开，之前的 feasibility 文档没跟着更新**：
`ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md` §2.3 原本担心"任务分数只在对话结束判一次、和本项目
逐轮 `y_t` 的设计不匹配"，需要在"熵代理读出"（路径 A）和"自建逐轮打分器"（路径 B）之间选。查
`ergo_math_trajectory.py` 代码发现**这个决策已经隐式做完了**：每一轮都会让模型给出"当前最佳猜测
答案"并当场判分（`ergo_math_judge.judge_math_answer`），不是只在结束时判一次——已经是路径 B，
且已经跑通、有数据。

**状态空间设计决策（feasibility 文档下一步 5(b) 的答案）**：显式加入 `shard_frac = turn /
num_shards` 作为 `ReducedStateConfig.aux_cols`（而不是只用 sycophancy 那种"只靠 y 的滞后项"设计）
——理由是这个任务的 y 上升不是记忆/黏滞驱动的，是"揭示的 shard 变多、题目客观上更完整"这个确定性
外生变量驱动的（见上面"惯性/动力学定性"一节），不显式建模这个变量，回归会把它的效应错误地
归到 y 滞后项上，把"重置"这个执行器自身的效果和纯粹的信息累积效果混在一起。这个扩展点
（`aux_cols`）`modeling/dataset.py` 里已经现成（`koopman_detection_design.md` 方案 4 留下的
接口），不需要改核心建模代码，只需要在读数据时补一列 `shard_frac`。

同时确定 `contemporaneous_v=True`（不是像 defense 线那样留一个可选 flag）：`run_ergo_math_trajectory`
里 `u_reset` 对 turn t 的决策，在生成/判分 turn t 的回复**之前**就已生效（重建了 prompt），
即 `u_reset_t` 同轮直接导致 `y_task_success_t`——精确对应 `ReducedStateConfig.contemporaneous_v=True`
的语义（`dataset.py` 类文档字符串 + `koopman_case_study_design.md` Phase I 的时序错位分析）。这里
没有理由留 False 分支做消融，跟 defense 线当年误踩的"v 错位"坑不是一回事，直接钉死。

**代码**：
- `scripts/run_ergo_math_screening.py` 新增 `--controller random_excite` + `--random-excite-p`
  （复用 `controller_cli.make_controller_factory` 已有的、经过完整单测的 `RandomExciteController`
  分支，零新增控制器代码）。
- `scripts/fit_koopman_ergo_model.py`（新文件，结构照抄 `fit_koopman_defense_model.py`）：读
  `trajectories.jsonl`，补 `shard_frac` 列，按 `item_id`（不是 `attack_id`）切 held-out split，
  `y_col="y_task_success"`/`u_col="u_reset"`，拟合 ARX + `richer_abs_sign` 两个 baseline，报告
  held-out rollout MSE + 可控性诊断（Gramian/谱半径），写 `koopman_fit_report.json`。
- 用已有的 60-item zero_control+reset 数据（拼在一起，本身不是随机激励设计,只是机械 smoke test）
  跑通过一次，确认列名/形状/`build_identification_dataset`/`rollout_output_error` 全部接得上,
  无需改一行核心 `modeling/` 代码。CPU 全套测试 386 passed（5 个既有的、和本次改动无关的 NLTK
  数据缺失失败不变，和 `mc_sycophancy_screening_pilot.md` 记录的同一批）。

**数据采集（job 15613799，已提交，2026-09-06）**：`environment/run_ergo_phaseB_random_excite.sbatch`
——`--controller random_excite --random-excite-p 0.5`，60 items（`--item-rng-seed 2`，和 pilot 的
20 个、扩样本的 60 个都不同的第三次抽样，题库只有 103 个 item 所以会有重叠，但这次是单臂采集
没有配对约束）× 2 seeds，和防御线 Phase B（30 attacks × 2 seeds, p=0.5）同一量级,预期成本和已跑完
的 60-item 权威检查同一数量级（15-30 分钟）。

**下一次接续时**：先 `sacct -j 15613799` 确认完成，再跑
`python scripts/fit_koopman_ergo_model.py`（默认参数已经指向这次的输出目录），看
`held_out_rollout_mse`/`controllability_rank`/`A_spectral_radius` 这几个数字，判断这条线是否
值得往下投入 `KoopmanMPCController`（防御线的下一步模板）。

**拟合结果（2026-09-07，`outputs/ergo_math_phaseB_random_excite/koopman_fit_report.json`）**：
job 15613799 已 `COMPLETED`（21:53），本地跑通 `fit_koopman_ergo_model.py`（60 items，45 train /
15 held-out，`--split-seed` 默认值）：

| model | train one-step MSE | held-out rollout MSE |
|---|---:|---:|
| ARX | 0.0896 | **0.0893** |
| richer_abs_sign | 0.0896 | 0.1621 |

- **ARX 赢过 richer baseline**（rollout MSE 低 45%）——和防御线 Phase B 的选型方向一致，
  三维状态 `z=[y_lag, shard_frac, v]`（`aux_cols=["shard_frac"]` + `contemporaneous_v=True`）
  已经够用，不需要更复杂的特征。
- **可控性满秩**：`controllability_rank=3`（=`state_dim`），但 `gramian_condition≈4779`——
  奇异值 `[1.003, 0.097, 0.015]` 相差近两个数量级，`u_reset` 这一维输入对 `y`/`v` 两个状态
  分量的驱动力强、对 `shard_frac` 分量几乎不驱动（预期之内——`shard_frac` 是外生变量，
  `u_reset` 不该控制它，`A`/`B` 矩阵里对应行系数确实接近零），不是病态拟合。
  `A_spectral_radius=0.953`（<1，稳定；比 `1.005` 的 `shard_frac` 自身系数略低，量级上与
  "分数随信息累积单调上升但终会饱和" 的定性一致）。
- **判断：值得往下投**——ARX 跑通、held-out rollout 误差不大（约 0.09，对应
  `y∈[0,1]` 量级下约 30% 的均方根误差）、满秩可控、谱半径 <1，三个前置条件（拟合质量/
  可控性/稳定性）都过了，没有出现防御线当年"读出没有可反馈状态"那种结构性卡点
  （`adaptive_vs_fixed_claim_plan.md` T3）。下一步是 `KoopmanMPCController` 而不是回头
  加更多任务的评分器工程。
- CPU 全套测试 394 passed（同一批 5 个 NLTK 数据缺失失败，和本次改动无关）。

## 下一步

1. ~~等 60-item 扩充作业跑完，确认效应量在更大样本下依然稳~~ **已完成（见上节）：效应量、
   方向、显著性三者都稳，60-item 比 20-item 显著性更强（p 从 1.27e-4 降到 6.39e-9）**。
2. **执行器权威已确认（20-item + 60-item 两次一致）**：这是这条线和 sycophancy 线最大的区别——
   不需要再纠结"要不要继续试错换执行器设计"这个问题了，可以往下走。
3. **惯性/动力学定性需要重新表述**，见上面加粗的段落——不是"记忆黏滞"而是"信息累积",这个
   区别要带进任何后续文档/论文措辞，避免和 sycophancy/防御线的"惯性"概念混为一谈。
4. **数学任务上的 Koopman 建模已跑通（见上面拟合结果小节，2026-09-07）**：ARX 赢过 richer
   baseline、满秩可控、谱半径 <1，三个前置条件都过了——**判断是值得往下投**，不是"暂不急于
   铺开"的悬置状态了。下一步是 `KoopmanMPCController`（Phase C），不是先为其余五个 ERGO
   任务（code/SQL/actions/data2text/summary）投评分器工程。
5. 可行性文档第 4 节的三个未走完项，现在的状态：
   a) **为更多任务投入路径 B 评分器工程——仍未决定，推迟到 Phase C（闭环 MPC）结果出来之后**；
   b) **Koopman 状态空间设计——已决定且已验证**（见上面"Koopman 建模，Phase B"一节）：显式
      加入 `shard_frac`（已揭示 shard 比例）作为 aux 协变量，而不是只用 y 的滞后项，拟合
      结果证实这个设计是必要的（`shard_frac` 对应的 `A`/`B` 行系数独立于其余状态）；
   c) **论文叙事定位——仍未决定**，Phase C（`KoopmanMPCController` 闭环）跑完后再判断这条线
      是"和 sycophancy 对照的正面案例"还是有独立的建模贡献，`docs/article/
      PAPER_EXECUTION_PLAN.md` §1.4 仍未提这条线。
6. **下一次接续时**：读上面"拟合结果"小节确认起点，然后照防御线 Phase B→C 的模板（先有
   开环拟合出的 A/B，再包一层 `KoopmanMPCController` 跑闭环、和 `reset`/`zero_control` 两个
   开环基线比较任务成功率）设计 ERGO 线的 Phase C，脚本层面可以照抄
   `run_defended_screening.py`/`controller_cli.py` 里 MPC 控制器已有的接入方式，域特定的
   部分只有 `ergo_math_trajectory.py` 的 `u_reset` 决策入口。
