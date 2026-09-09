# 实验记录：ERGO/Laban 多轮可靠性侵蚀，最小执行器权威验证（2026-09-06）

> ⚠️ **本文档里"ERGO 线在当前读出族下终止"这个结论已于 2026-09-07 被撤回**：E0 的 RC-3 把
> `u_{t+1}` 实现成了 `u_{t+2}`，RC-1 让模型外推一个 null 白送的确定性外生量——改正后 RC-B
> 三条全过。证据与修正规格见 [`signal_resolution_plan.md`](signal_resolution_plan.md)
> 第 0.1／0.2 节与任务 F0，本文档的相关各节由 F0 回写。
> **不受影响的部分**："结果""样本扩充"两节的执行器权威结论（reset 显著提高最终轮成功率）。

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

- ~~ARX 赢过 richer baseline（rollout MSE 低 45%）~~ **2026-09-07 撤回，vacuous**：
  `y_task_success ∈ {0,1}`（666 行全部是 0.0/1.0），而 `richer_abs_sign` 的额外特征是
  `|y|` 与 `sign(y)`——对 0/1 变量恒等于 `y` 本身，与 `y` 精确共线。两模型
  `train_one_step_mse` 到 15 位有效数字相同（`0.08956779213150870` vs
  `0.08956779213150866`），`richer` rollout 更差是退化参数化的数值产物，**不构成任何
  选型证据**（`backup/readout_controllability_gate_plan.md` §0.4(i)）。
- ~~held-out rollout 误差不大（约 0.09）~~ **2026-09-07 补充对照表，结论反过来了**：
  同一 45/15 item split 上三个平凡 null 的 held-out MSE——常数 0.1096、train 逐轮均值
  0.0832、无状态 OLS `[1, shard_frac, u_reset]`（完全不用 `y` 的滞后）0.0792——**ARX
  的 0.0893 连"零状态、只用外生输入"的回归都没打过**，只比"什么都不建模"的常数基线好一点。
  拟合出的状态本质上只有 `y_lag` 一维在做事（自持续系数 0.490），`y_{t+1}` 的解释力主要来自
  `0.819 × shard_frac` 这个确定性轮次斜坡，不是反馈状态。
- **可控性满秩 + 谱半径 <1（数字不变，但不构成证据）**：`controllability_rank=3`
  （=`state_dim`）对随机矩阵几乎必然成立；`A_spectral_radius=0.953` 里那一维是外生的
  `shard_frac`（自系数 `1.005>1`），把外生确定性斜坡算进"被控状态"再谈稳定性是范畴错误。
  这两条都**不是**"值得往下投"的证据。
- **2026-09-07 判断：Phase C 在 E0 之前不准开工**——正式的读出可控性闸门（RC-gate，
  [experiments/backup/readout_controllability_gate_plan.md](backup/readout_controllability_gate_plan.md)
  §四）跑完，**RC-1（打过平凡 null）和 RC-3（输入增益显著）都不过**，只有 RC-2（去趋势后
  仍有逐轨迹状态，`lag1_demeaned=0.2578, p=9.69e-10`）过。RC-gate 要求三条全过才能进
  Phase C，所以**这条线在当前读出（`y_task_success`）下不满足"值得往下投"的条件**——上面
  "没有出现防御线当年那种结构性卡点"的判断撤回，方向其实相反：和防御线踩的是同一个坑
  （建控制器之前没先证明读出有可反馈状态）。下一步不是 `KoopmanMPCController`，是 E1
  （换读出，见该文档§五）。
- CPU 全套测试 401 passed（同一批 5 个 NLTK 数据缺失失败，和本次改动无关）。
- **2026-09-07 F0 修正，上面两条判定撤回**：`analyze_ergo_readout_state.py` 有两处测量错误
  （[`signal_resolution_plan.md`](signal_resolution_plan.md) §0.1/0.2）——RC-3 把 `u_{t+1}`
  取成了 `u_{t+2}`（差一行）；RC-1 的 rollout 让模型自由外推 `shard_frac` 这个确定性外生量，
  而三个 null 都拿到真值，比较本身不公平。修正后（`scripts/analyze_ergo_readout_state.py`
  重写版，20-split + aux 真值覆盖）：**RC-0/RC-1/RC-2/RC-3 全部通过**——
  RC-1 aux=真值在 20/20 个 split 里赢过最优 null（mean skill +0.128，naive 版只有 7/20）；
  RC-3 `u_t=+0.0588`（se=0.0270, p=0.0296, n=546，含 `u_{t+1}` 规格）。**"Phase C 在 E0
  之前不准开工"、"下一步是 E1"这两条判断撤回，Phase C 已经开工，见下面新增的
  "F0–F3：读出分辨率修正与 Phase C 结果"一节。**

## 下一步

1. ~~等 60-item 扩充作业跑完，确认效应量在更大样本下依然稳~~ **已完成（见上节）：效应量、
   方向、显著性三者都稳，60-item 比 20-item 显著性更强（p 从 1.27e-4 降到 6.39e-9）**。
2. **执行器权威已确认（20-item + 60-item 两次一致）**：这是这条线和 sycophancy 线最大的区别——
   不需要再纠结"要不要继续试错换执行器设计"这个问题了，可以往下走。
3. **惯性/动力学定性需要重新表述**，见上面加粗的段落——不是"记忆黏滞"而是"信息累积",这个
   区别要带进任何后续文档/论文措辞，避免和 sycophancy/防御线的"惯性"概念混为一谈。
4. **2026-09-07 撤回**：数学任务上的 Koopman 建模"值得往下投"这个判断被 E0 闸门推翻
   （见上面拟合结果小节）——RC-1/RC-3 不过，Phase C 不开工。下一步是 E1（换读出，默认候选
   token 熵，[experiments/backup/readout_controllability_gate_plan.md](backup/readout_controllability_gate_plan.md)
   §五），不是 `KoopmanMPCController`，也不是先为其余五个 ERGO 任务
   （code/SQL/actions/data2text/summary）投评分器工程。
5. 可行性文档第 4 节的三个未走完项，现在的状态：
   a) **为更多任务投入路径 B 评分器工程——仍未决定，推迟到 E1 结果出来之后**；
   b) **Koopman 状态空间设计——已决定但已知不够**（见上面"Koopman 建模，Phase B"一节）：
      `shard_frac` 作为 aux 协变量的拼接方式没问题，但 E0 显示可预测部分几乎全被这个
      确定性斜坡占掉，`y_task_success` 本身作为读出不满足 RC-gate；
   c) **论文叙事定位——仍未决定**，等 E1（或它失败后的终止收尾）结果出来后再判断这条线
      是"和 sycophancy/防御线并列的第三个读出层面结构性负结果"还是有独立的建模贡献，
      `docs/article/PAPER_EXECUTION_PLAN.md` §1.4 仍未提这条线。
6. **2026-09-07 追加，Phase C 起步**：`KoopmanMPCController` 硬编码 `y_probe`/`u_remind`
   且不支持 `aux_cols`，不能直接照抄防御线接线——按"不干扰 `control.py`/`controller_cli.py`
   共用代码"的要求，新增了继承子类 `src/persona_drift/ergo_koopman_mpc.py`
   （`ErgoKoopmanMPCController`，覆写 `_current_state` 支持可配置列名+aux；7 个新 CPU
   单测全绿，全套 401 passed）。**`_simulate` 的多步展望（`shard_frac` 该不该被当预测对象）
   和要不要给 reset 加预算约束，这两个设计问题写进了
   [experiments/backup/ergo_koopman_mpc_opus_design_questions.md](backup/ergo_koopman_mpc_opus_design_questions.md)
   ~~等 Opus 裁决，Sonnet 5 在规格出来之前不会继续往下接 GPU 作业。~~
   **⚠️ 2026-09-07：两个设计问题都已裁决**（真值覆盖已从"建议"升级为"必须"、reset 加 `k=1`
   预算并用实测 token 代价论证），见 [`signal_resolution_plan.md`](signal_resolution_plan.md)
   4.0–4.2。**这条"不接 GPU 作业"的指令已作废**，下一步是该计划的 F0→F1→F2→F3。**
7. **2026-09-07 追加，E0 已跑完，RC-B 不过**：`scripts/analyze_ergo_readout_state.py`
   （CPU，秒级）——RC-1 不过（ARX 0.0893 输给两个平凡 null）、RC-2 过
   （`lag1_demeaned=0.2578, p=9.69e-10`）、RC-3 不过（`u_t` 系数 `p=0.659`，不显著）。
   `backup/readout_controllability_gate_plan.md` 第 0.4 节的预期（RC-1 不过 → 转 E1）被证实。
   产物 `outputs/ergo_math_phaseB_random_excite/readout_state_report.json`。第六节记的两个
   设计问题（真值覆盖 / reset 预算）已经在 `backup/readout_controllability_gate_plan.md` §6.1/6.2
   有裁决，但**都要等 E1 通过后才用得上**——E0 不过直接堵住了整个 E2（Phase C）分支。
   ~~下一步是 E1（`scripts/analyze_ergo_entropy_readout.py`，token 熵读出，1 个小 GPU 作业）。~~
   **⚠️ 2026-09-07：E0 的这两条判定已被撤回**（RC-3 的 `u_next` 差一位、RC-1 让模型外推一个
   null 白送的确定性外生量），下一步是 [`signal_resolution_plan.md`](signal_resolution_plan.md)
   的 F0，不是 E1。
8. **2026-09-07 追加，E1 已跑完，两个熵读出都不过 RC-B——ERGO 线终止**（见下面独立一节
   "结论：ERGO 线在当前读出族下终止"）。job `15644575`（`pdc-ergo-entropy-readout`）
   `COMPLETED 0:0`，Elapsed 00:01:29，666 行，`entropy_answer_span` 缺失率 0.60%（<10% 闸门）。
   `entropy_mean`：RC-2 过（`lag1_demeaned=0.2193, p=2.27e-07`），**RC-3 不过**
   （`u_t=-0.0111, se=0.0068, p=0.104`，双侧也不显著）。`entropy_answer_span`：**RC-2 不过**
   （`lag1_demeaned=0.0204, p=0.635`），RC-3 双侧显著但**方向为负**
   （`u_t=-0.0201, se=0.0069, p=0.00368`）——reset 之后紧接着答案段的熵**下降**，方向和
   "读出可控性闸门"给 `y_task_success` 定的">0"字面判据相反（那条判据是为成功率读出校准的，
   熵读出"变好"的方向本来就是降低，不是升高，这是留给用户/Opus 的一个判断题，见下面独立
   一节）。产物 `outputs/ergo_math_phaseB_random_excite/entropy_readout.json` +
   `entropy_readout_state_report.json`。

## ~~结论：ERGO 线在当前读出族下终止~~（2026-09-07，本节判定已撤回，见下一节）

> **2026-09-07 撤回**：本节依据的 RC-gate 判定本身有两处测量错误（`signal_resolution_plan.md`
> §0.1/0.2，F0 已修正），`y_task_success` 的 RC-B 实际是**过**的，不是这里写的"不过"。
> `entropy_mean`/`entropy_answer_span` 两个熵读出仍然不过（这部分结论不受影响），但"三个读出
> 全灭、ERGO 线终止"这个总判定是错的——正确结论、Phase C 实际执行结果见下面
> "F0–F3：读出分辨率修正与 Phase C 结果"一节。本节以下内容保留作历史记录，**不要引用其
> "ERGO 线终止"这个结论**。

`backup/readout_controllability_gate_plan.md` §五预注册的判定规则：E1 两个熵读出列的 RC-2/RC-3
"任一不过 → ERGO 这条线在当前读出族下终止"。实测**两个熵读出都不过**：

| 读出 | RC-2（去趋势后 lag-1） | RC-3（含 `u_{t+1}` 的输入增益，字面判据 `u_t>0` 且 `p<0.05`） | RC-B |
|---|---|---|---|
| `y_task_success`（E0） | 过（`lag1_demeaned=0.2578, p=9.69e-10`） | 不过（`u_t` p=0.659） | **不过** |
| `entropy_mean`（E1） | 过（`lag1_demeaned=0.2193, p=2.27e-07`） | 不过（`u_t=-0.0111, p=0.104`，双侧也不显著） | **不过** |
| `entropy_answer_span`（E1） | 不过（`lag1_demeaned=0.0204, p=0.635`） | 双侧显著但方向为负（`u_t=-0.0201, p=0.00368`） | **不过** |

三个候选读出没有一个同时满足"有超过平凡基线的预测力/去趋势后仍有逐轨迹状态/执行器能推动它"。
按防御线 T2/Phase J 立下的纪律：**闸门不过就照实记录，不调参重跑、不换读出族继续试**。

**这构成论文里站得住的一条内容**：和防御线（`koopman_defense_pilot.md` 第一/五/六节）、
`adaptive_vs_fixed_claim_plan.md`（T1b/T3）并列，成为"建闭环控制器之前必须先证明读出有
可反馈状态"这条方法论教训的**第三个独立案例**——三条完全不同的任务（越狱攻击安全侵蚀、
持续反驳下的立场维持、sharded 数学题多轮可靠性）用三套不同的读出（LLM judge 安全分/立场分、
确定性任务成功率+token 熵）全部撞上同一个结构性缺陷，这本身是一个可以写进论文讨论/方法论
部分的、有一定普遍性的负结果，不是这条线本身的失败。

**执行器权威结论不受影响，仍然站得住**：`reset` 这个动作对 `y_task_success`
的因果效应本身是真实、显著、稳定的（20-item pilot 与 60-item 扩样本两次确认，见上面
"结果""样本扩充"两节，本次审计没有触碰这部分数字）——问题始终是**这个动作的效应能不能被
一个逐轮反馈状态捕捉并用来做自适应决策**，不是"reset 有没有用"。

**留给用户/Opus 的判断题（未裁决）**：

1. `entropy_answer_span` 的 RC-3 双侧显著、方向为负（reset 后答案段熵下降）——这本身是一个
   有机制含义的发现（"reset 让模型对当前答案更确定"），但**只有 RC-3 一条过、RC-2 不过**
   （去趋势后没有可反馈的逐轨迹状态），按三条判据必须全过的规则仍然不能进 Phase C。要不要
   把这一条单独写成一句描述性观察（不是"读出可控"的证据，只是"reset 有即时效应"的又一个
   佐证），还是完全不提？
2. 要不要为其余五个 ERGO 任务（code/SQL/actions/data2text/summary）投入路径 B 评分器工程，
   在"三个读出全灭"之后，这个问题的答案大概率是"不值得"（同一读出族问题很可能在其他任务上
   重现），但没有正式裁决过。
3. 论文叙事定位：这条线现在的产出是"执行器权威确认 + 读出可控性结构性负结果"，要不要正式
   写入 `docs/article/PAPER_EXECUTION_PLAN.md` §1.4，以及以什么篇幅（一段讨论 vs. 独立小节）。

~~**这个任务不需要任何自动化后续动作**——终止收尾到此为止，除非用户/Opus 就上面三条给出裁决。~~

**⚠️ 2026-09-07 撤回上面这句**：终止判定本身已被推翻（见本文档顶部横幅）。当前的后续动作是
[`signal_resolution_plan.md`](signal_resolution_plan.md) 的 F0→F1→F2→F3，本节各段由 F0/F2 回写。

## F0–F3：读出分辨率修正与 Phase C 结果（2026-09-07）

**新开一次对话想知道"ERGO 这条线现在到底是终止了还是在跑 Phase C"，看这一节，不要看上面
"结论：ERGO 线在当前读出族下终止"那一节（已撤回）。**

### F0：修正 E0 的两处测量错误（CPU，无作业）

`scripts/analyze_ergo_readout_state.py` 重写：`build_input_gain_transitions` 的 `u_{t+1}`
改成直接读被预测目标那一行（`b[U_COL]`），不再多看一行；`run_segment1`（RC-1）改成 20 个
随机 item split（每个 45 train / 15 held-out），且 ARX 的多步 rollout 在每一步后把状态最后一维
（`shard_frac`）强制换成真值 `pair["z_next"][-1]`，不再让模型自由外推这个确定性外生量。

**RC-0/RC-1/RC-2/RC-3 全部通过**：

| 判据 | 结果 |
|---|---|
| RC-0（同一仪器） | 过（`y_task_success` 就是被汇报的指标本身，trivial） |
| RC-1（20-split，aux=真值 rollout vs 三个 null） | 过——20/20 个 split 里 ARX 赢过最优 null；均值 MSE：const=0.1257, turn_mean=0.1010, stateless_ols=0.0874, naive rollout=0.1127（只 7/20 赢）, **aux=真值 rollout=0.0759**（20/20 赢） |
| RC-2（去 shard_frac 十等分箱均值后的 lag-1） | 过：`lag1_demeaned=0.2578`（p=9.69e-10） |
| RC-3（含 `u_{t+1}` 的输入增益，item 固定效应） | 过：`u_t=+0.0588`（se=0.0270, p=0.0296, n=546）；`u_{t+1}=+0.0716`（se=0.0270, p=0.0083） |

`richer_abs_sign` 的 vacuous 登记不变（二值 `y` 下 `train_one_step_mse` 与 ARX 差 4.16e-17）。
CPU 全套测试 406 passed（同一批 5 个 NLTK 失败无关）。**偏离记录**：RC-3 的 p 值和
`signal_resolution_plan.md` §0.1 表格给的参考值差约 0.0004（`p=0.0300` vs 参考 `0.0296`；
`closeness` 那一行也有类似量级的差），beta/se 都逐位一致，怀疑是参考计算用了略有差异的
自由度/协方差估计口径，不影响任何判定方向。

### F1：把 `closeness` 固化成正式读出

`closeness = 1/(1+|a-g|/max(|g|,1))`（`a`=正则抽取答案，`g`=gold answer，和硬 0/1 判分
同一套抽取器）。`scripts/analyze_ergo_closeness_readout.py`（新脚本）：

- **G-F1-1（分布）**：666 行，`mean=0.6058, sd=0.2158, ==1.0 占比 0.1366, ==0.0 占比 0.0090`
  ——与预注册数字逐位一致。
- **候选读出对照表**（`y_task_success`/`closeness`/`coverage`/`reply_len`(词数)/`churn`/
  `entropy_mean`/`entropy_answer_span`，与目标 ρ / 去趋势 ρ / lag1_demeaned / 输入增益 `u_t`,p）：

  | 读出 | ρ(目标) | ρ(去趋势) | lag1_demeaned | `u_t` | p |
  |---|---:|---:|---:|---:|---:|
  | y_task_success | +1.0000 | +1.0000 | +0.2578 | +0.0588 | 3.00e-02 |
  | **closeness** | **+0.6315** | **+0.4519** | **+0.2757** | **+0.0540** | **5.29e-04** |
  | coverage | +0.4335 | +0.2519 | +0.4082 | +0.0434 | 2.46e-01 |
  | reply_len | +0.4810 | +0.2322 | +0.4525 | +0.8565 | 8.50e-01 |
  | churn | +0.1951 | +0.0231 | +0.1627 | −0.0572 | 1.41e-01 |
  | entropy_mean | +0.1144 | +0.0477 | +0.2193 | −0.0006 | 9.16e-01 |
  | entropy_answer_span | −0.1219 | +0.0759 | +0.0204 | −0.0121 | 3.15e-02 |

- **closeness 自己的 RC-0/1/2/3**：全过——RC-1 20-split mean skill **+0.1754**（20/20 赢），
  RC-2 `lag1_demeaned=0.2757`（p=5.61e-11），RC-3 `u_t=+0.0540`（p=5.29e-04）。
  **closeness 在四条判据上都不比二值 `y_task_success` 差**，采纳为 Phase C 的状态读出
  （汇报指标仍是二值 `final_turn_success`，两者是同一次抽取的两个函数，不违反 RC-0）。

### F2：用 closeness 重拟合 Phase B

`scripts/fit_koopman_ergo_closeness.py` → `koopman_fit_report_closeness.json`（新文件，未覆盖
原文件）。**G-F2-1**：`y_col="closeness"`、`aux_cols=["shard_frac"]`、`contemporaneous_v=true`、
`nu=1,mu=1`、held-out item 与原 45/15 split 一致，全部通过。`richer_abs_sign` 改标注
`vacuous_for_binary_y=false`（连续量下不再退化）。ARX rollout MSE：naive=0.0310, aux=真值=0.0275
（与 F1 20-split 均值 0.0274 一致）；三个 null：const=0.0441, turn_mean=0.0344,
stateless=0.0346——aux=真值仍然赢。

### F3：ERGO Phase C（闭环 MPC vs 四个基线）

**代码改动**（`control.py`/`controller_cli.py` 确认未改动）：
`ErgoKoopmanMPCController`（`src/persona_drift/ergo_koopman_mpc.py`）新增
`_simulate` 覆写（`shard_frac` 展望时每步强制换成真值 `(turn+k)/num_shards`）、
`next_u_remind` 覆写（逐轨迹动态设 `episode_length=num_shards`）；`run_ergo_math_screening.py`
新增 `random_schedule`/`ergo_koopman_mpc` 两个本地分支（不改
`controller_cli.make_controller_factory`）。12 个新/改单测全绿。

**⚠️ 执行中发现并修复一个真 bug**：继承来的 `_remaining_budget`（`control.py`）硬编码读
`u_remind` 列，ERGO 行里根本没有这个字段，导致预算从未真正生效——第一次跑
`ergo_koopman_mpc` 臂（job 15645632）在 664 行里 reset 了 548 次（budget=1 应该最多
116 次，每条轨迹 1 次）。已覆写 `_remaining_budget` 改读 `self.u_col`，新增单测
`test_remaining_budget_reads_u_reset_not_u_remind` 直接复现并锁定这个场景，旧产物存档为
`outputs/ergo_math_phaseC_mpc_INVALID_budget_not_enforced_job15645632`（不删除，仅存档），
重跑 job **15646133**（`COMPLETED 0:0`，6:52），确认 664 行里 reset 恰好 116 次、
每条轨迹恰好 1 次。

**评测集**（G-F3-1）：58 items（Phase B 45 个辨识 item 之外的全部，交集=0），
`num_shards` 分布 `{4:12, 5:17, 6:13, 7:8, 8:7, 9:1}`，与预注册一致。8 个臂全部
`COMPLETED 0:0`，664 行/臂（58 items × 2 seeds，行数随 `num_shards` 变化，8 个臂逐位相同）。

**结果**（`scripts/analyze_ergo_phaseC_comparison.py`，按 item 配对，10000 次 bootstrap，
`default_rng(0)`）：

| 臂 | final_turn_success 均值 | token/轨迹 |
|---|---:|---:|
| zero_control | 0.3276 | 0.0 |
| always_reset（无预算） | 0.7759 | 696.1 |
| fixed_t1（预算 1） | 0.2069 | 88.1 |
| fixed_t2（预算 1） | 0.2069 | 100.9 |
| fixed_t3（预算 1） | 0.3190 | 114.6 |
| fixed_t4（预算 1） | 0.6466 | 128.7 |
| randsched_p100（预算 1） | 0.4655 | 118.6 |
| **mpc（预算 1）** | **0.2069** | **100.9** |

三道预注册闸门**全部不过**（mpc 全面更差，不是跨零而是整段落在负区间）：

| 闸门 | mean_diff | 95% CI | 判定 |
|---|---:|---|---|
| 闸门 1（mpc − zero_control） | −0.1207 | [−0.2241, −0.0259] | 不过（更差） |
| 闸门 2（mpc − randsched_p100，主判据） | −0.2586 | [−0.3621, −0.1552] | 不过（更差） |
| 闸门 3（mpc − 最优 fixed_t，`fixed_t4`） | −0.4397 | [−0.5603, −0.3190] | 不过（更差，参照用） |

**诊断（不是判据，但解释了为什么）**：mpc 的 116 条轨迹**全部**在 turn 2 花掉唯一的一次
reset——`mean_final_turn_success`/`token/轨迹` 与 `fixed_t2` 逐位相同（0.2069/100.9），
不是巧合，是 116 条轨迹的决策序列完全相同，MPC 退化成了一个固定臂。机制上看：
`horizon=2` 下，turn 2（第一次有足够历史做决策）只能展望到 turn 3——**看不到 `fixed_t4`
（最好的固定臂）所在的 turn 4**，所以"越早花掉预算"在这个短视窗里永远显得更优，MPC
根本没有机会发现更晚花更好。这不是本次执行的偏差，是预注册规格（`horizon=2`）本身的
局限——按闸门 2 不过的纪律，**不调 horizon 重跑**，如实记录这个诊断。

**结论**：**自适应调度（`ergo_koopman_mpc`）在 ERGO 上没有打过等代价随机分配**（闸门 2 不过，
且差距很大），也没打过任何一个固定臂。执行器权威结论不受影响（`always_reset` 无预算下
0.7759，远高于 zero_control 的 0.3276）——问题仍然是"预算下怎么调度"，不是"reset 有没有用"。

**留给用户/Opus 的判断题**：
1. `horizon=2` 的短视是否值得作为独立诊断写进论文（"budget-1 + horizon-2 下 MPC 退化为
   fixed_t2，因为看不到 turn 4"），还是只在 limitation 里带一句；
2. 要不要认为"至少证明了 MPC 会做出一个自洽（虽然是退化的）决策"这件事本身值得记录，
   还是纯粹归为负结果；
3. 论文叙事定位（同上一节判断题 3，现在有了 Phase C 的真实数字，需要重新讨论）。

---

## G4：`fixed_t_last` 臂——设定退化坐实，并追出一个恒等式（2026-09-07）

**任务**：`measurement_validity_plan.md` 第七节 G4——补上 Phase C 臂表里缺失的那个最优固定臂
（reset 落在 `turn = num_shards`），判定 k=1 计数预算的设定是不是退化的。
**产物**：`outputs/ergo_math_phaseC_fixed_last/`（116 条轨迹，58 held-out item × 2 seeds）、
`outputs/ergo_case_study/phaseC_comparison_report_g4.json`。

### G4.1 结果

| 臂 | final_turn_success | token/轨迹 | 每 100 token 增益（相对 zero_control） |
|---|---:|---:|---:|
| `zero_control` | 0.3276 | 0 | — |
| **`fixed_last`** | **0.7759** | **152.8** | **+0.293** |
| `always_reset` | 0.7759 | 696.1 | +0.064 |
| `randsched_p100` | 0.4655 | 118.6 | +0.116 |
| `fixed_t4`（此前记为"最优固定臂"） | 0.6466 | 128.7 | +0.248 |
| `mpc`（≡ `fixed_t2`） | 0.2069 | 100.9 | −0.120 |

预注册闸门（按 item 配对，58 对，bootstrap 10000，`default_rng(0)`）：

| 闸门 | 差值 | 95% CI | 判定 |
|---|---:|---|---|
| `fixed_last − randsched_p100` | **+0.3103** | [+0.2069, +0.4224] | 排除 0 → **第一格：设定退化** |
| `fixed_last − fixed_t4` | **+0.1293** | [+0.0259, +0.2414] | 此前的"最优固定臂"不是最优 |
| `always_reset − fixed_last` | **0.0000** | **[0, 0]** | 完全相同 |

**G-G4-1 闸门通过**：116 条轨迹，落点全部恰好等于 `num_shards`（= 每条自己的最后一轮），
评测 item 与其余 7 个臂逐字相同。

**按预注册规则：k=1 计数预算下最优策略是一个只用 `num_shards` 就能写出、不需要任何反馈的
固定规则 → Phase C 的三道闸门作废，不要重跑 MPC。** 另外 Phase C 的 gate3 用 `fixed_t4`
当"最优固定日程"，比错了对手——对真正的最优臂，MPC 是 **−0.569** 而不是记录里的 −0.4397。

### G4.2 `fixed_last ≡ always_reset` 是恒等式，不是实验发现

两臂不只是均值相同——**116/116 条轨迹的最终抽取答案逐字相同**。根因在
`src/persona_drift/ergo_math_trajectory.py:112-117`：

```python
if u_reset:
    stimulus = _consolidated_stimulus(revealed_shards)
    agent_history = [{"role": "user", "content": stimulus}]   # ← 覆盖，不是追加
```

**reset 把整个对话历史抹掉**，只留一条含合并题面的消息。所以任何在**最后一轮** reset 的策略，
末轮 prompt 都是同一条消息，而 agent seed 由 `(seed, turn)` 决定——输出**必然**逐字相同。

配套验证全部对上：非末轮的 **548/548** 行 `fixed_last` 与 `zero_control` 逐字相同
（末轮之前不动作）；末轮与 `zero_control` 只有 38/116 相同。

### G4.3 三个推论

**(1) 缺失的"未分片基线"一直在手里，只是贴错了标签。**
`always_reset` 的 0.7759 就是"所有 shard 一次性给出、单轮作答"的准确率
（严格说是**合并后的 shard 列表**，不是 GSM8K 原始题面）。所以这个模型 + 这批 item 上，
Laban 的分片退化效应是 **0.3276 → 0.7759，−0.448**——一个干净的复现。

**(2) `final_turn_success` 是错误的终点指标。**
在会抹历史的执行器下，它**只是"末轮有没有 reset"这一个二元决策的函数**，前面所有轮的动作
对它毫无影响。**Phase C 的控制问题是被指标定义清空的，不是模型没有动力学。**

**(3) token 预算的补救价值大幅下降。**
一次末轮重述已经达到无限预算的天花板，成本 152.8 vs 696.1 token（**4.56×**）。
token 预算只有在**低于 ~153 token**、连一次完整重述都买不起时才产生真实权衡——那是人为制造
稀缺。`measurement_validity_plan.md` 7.3 节据此标记为**不再推荐**。

### G4.4 这条线现在的产出（三条，都不是负结果）

1. **分片退化效应的干净复现**：−0.448（58 held-out item × 2 seeds）；
2. **一个实用结论**：末轮一次重述 = 全程重述，成本降到 22%（每 100 token 的增益 0.293 vs 0.064）；
3. **一个方法论结论**：**会抹历史的执行器 + 末轮终点指标 = 控制问题被构造性地清空**。
   这与防御线的信息结构结论（`defense_line_redesign_plan.md` 0.1 节）配成一对——
   "LLM 控制 benchmark 可以怎样在开跑前就已经死了"的两种方式。

执行器权威结论（reset 显著提高最终轮成功率）**不受影响**，仍然成立。

### G4.5 若要让 ERGO 重新成为真正的控制问题（**都是新实验，当前不建议投**）

必须改**执行器或终点**，不是控制器：

- **最便宜的一处**：让 reset **追加**一条合并摘要而不是抹掉历史
  （`ergo_math_trajectory.py:114` 从赋值改成 append）。这样历史才有意义，"状态"才重新成为
  真问题；
- 换终点：禁止末轮 reset 之后的成功率，或逐轮成功率曲线下面积；
- token 预算严格低于一次末轮重述的成本。

---

## E1：`reset_mode` 与 append 分析器

**任务**：`docs/experiments/two_task_success_plan.md` 第二节 E1（四处改动）+ 第十二节 B1。零 GPU。

**改动**（只这四处，加新测试文件）：
1. `src/persona_drift/ergo_math_trajectory.py`：`ErgoMathTrajectoryConfig` 加 `reset_mode: str = "overwrite"`（`__post_init__` 对非法值抛 `ValueError`）；112–117 行按文档补丁形状改为 `overwrite`/`append` 分支；`row` 加 `"reset_mode"`；`inserted_tokens` 计法未改。
2. `scripts/run_ergo_math_screening.py`：加 `--reset-mode {overwrite,append}`（默认 `overwrite`），透传进 `ErgoMathTrajectoryConfig`；append 模式下包一层 `controller_factory`，把 `controller.name` 加 `_append` 后缀（带幂等判断，避免复用同一控制器实例的臂被重复加后缀）。
3. `tests/test_ergo_math_trajectory.py` 新增 4 条：`overwrite` reset 后 `agent_history` 长度为 1；`append` 下长度单调递增；两种模式下末轮消费的 stimulus 文本逐字相同；非法 `reset_mode` 抛 `ValueError`。
4. 新增 `scripts/analyze_ergo_append_comparison.py`（B1）：臂目录走 `--arm name=path`（可重复）；gate（`--gate a-b`，配对差 + bootstrap 10000 + `default_rng(0)`）直接调用从 `analyze_ergo_phaseC_comparison.py` 原样导入的 `_per_item_final_turn_success`（未重写）；identity（`--identity a,b`）另写 `_final_turn_field_by_trajectory`，按 `trajectory_id` 交集比较末轮 `judge_raw_output` 逐字相同占比；两种比较在 item_id 集合不一致时都直接抛 `ValueError`。新增 `tests/test_analyze_ergo_append_comparison.py`，7 条（≥3 条要求）。

**G-E1-1**：本任务范围内测试稳定全绿；"全套 CPU 测试通过"这一判据在并发写作环境下拿不到稳定单一数字，见下方偏离说明，判定留给 Opus。
- 我自己四个改动文件 + 两个新文件范围内的测试：**35 passed / 35**（`test_ergo_math_bank.py` + `test_ergo_math_judge.py` + `test_ergo_math_screening.py` + `test_ergo_math_trajectory.py`(10) + `test_analyze_ergo_append_comparison.py`(7)），稳定复现多次。
- **偏离（重要）**：`git diff --stat`/全套测试无法给出稳定单一数字——工作目录在本任务执行期间被至少三个并行会话（可辨认为 E3/E4/D1：`ergo_koopman_mpc.py`、`ergo_controllers.py`+`test_ergo_controllers.py`、`analyze_ergo_state_action_interaction.py`+其测试、`run_d1_screening.py`、`select_d1_attacks.py`、`conf/experiment/d1_attack_ids.txt`、以及 `docs/experiments/two_task_success_plan.md` 本身）同时读写。全套 `pytest -q` 在同一未改动状态下多次运行给出的 collected/passed 数字持续变化（观测到 423→433→444→449→458，passed 410→439→442 不等），且间歇性地把 `test_ergo_koopman_mpc.py`、`test_analyze_ergo_state_action_interaction.py` 也计入失败——这些文件不在我的改动范围内，`ErgoMathTrajectoryConfig` 也未在其中被构造性使用（已用 grep 核实全项目里该类只被关键字参数构造）。**取本任务结束时刻的一次快照**：`7 failed, 442 passed, 458 collected`，失败集合为 `test_surface_features.py` 5 条（NLTK，与基线一致）+ `test_ergo_koopman_mpc.py::test_remaining_budget_reads_u_reset_not_u_remind` 1 条 + `test_logging_setup.py::test_configure_run_logger_writes_config_to_a_file_under_logs_dir` 1 条（同一探针在孤立运行时稳定通过，怀疑与并行会话产生的额外 loguru sink/日志目录写入有关，不在本任务改动范围内）。我自己范围内的测试从未在任何一次快照中失败。
- 我自己的 `git diff --stat`（限定四个改动文件）：`3 files changed, 121 insertions(+), 2 deletions(-)`；新增两个文件 `scripts/analyze_ergo_append_comparison.py`、`tests/test_analyze_ergo_append_comparison.py`。**不含** `control.py`/`controller_cli.py`/`modeling/evaluate.py`/`modeling/dataset.py`/`paper/`。

**G-E1-2（回归锁）**：**通过**。
```
python scripts/analyze_ergo_append_comparison.py \
  --arm always_reset=outputs/ergo_math_phaseC_always_reset \
  --arm fixed_last=outputs/ergo_math_phaseC_fixed_last \
  --identity always_reset,fixed_last
```
输出：`n_identical=116, n_trajectories=116, rate=1.0`（`trajectory_id_intersection=116`，`item_id_sets_equal=true`）。额外交叉验证：同一对臂上 `--gate always_reset-fixed_last` 给出 `mean_diff=0.0, ci=[0.0,0.0], n_pairs=58`，与 `analyze_ergo_phaseC_comparison.py` 的既有 `gate6` 数字逐字一致。**未写任何文件到 `outputs/ergo_math_phaseC_*`，只读**。

**产物路径**：
- `src/persona_drift/ergo_math_trajectory.py`（改）
- `scripts/run_ergo_math_screening.py`（改）
- `tests/test_ergo_math_trajectory.py`（改，+4 条）
- `scripts/analyze_ergo_append_comparison.py`（新）
- `tests/test_analyze_ergo_append_comparison.py`（新，7 条）

**留给 Opus 的判断题**：
1. G-E1-1 的"全套 CPU 测试通过"这一判据在多会话并发写同一工作目录时无法给出单一可复核数字——是否需要改会话协议（例如每个任务用独立 worktree），还是接受"本任务范围内测试 + 一次带时间戳的全套快照 + 偏离说明"作为等价证据？
2. `test_logging_setup.py::test_configure_run_logger_writes_config_to_a_file_under_logs_dir` 在全套快照里间歇性失败、孤立运行必过——这是否值得单独立项排查（疑似 loguru sink 未清理导致的跨测试状态泄漏），还是等并发压力消失后视为噪声不予理会？

---

## E2–E3：append 执行器下的闭环——第 0 层修好了，第 2 层塌了（2026-09-08）

**一句话结论**：把 `ergo_math_trajectory.py:114` 的 reset 从"覆盖历史"改成"追加历史"，
`defense_line_redesign_plan.md` §12 分层表里唯一不成立的第 0 层（终点被末轮单个动作决定）
**确实被修好了**——状态持久性翻倍、G4 那条 116/116 恒等式掉到 30/116。**同一改动把第 2 层
（reset 权威）抹掉了**：reset 不再推动读出，也不再推动终点。两个性质由同一个自由度控制，
在这个设定里互斥。按预注册判据 **ERGO 线定格 S3**（G-E2-1、G-E3-1、G-E3-2 三道不过）。

### 执行摘要

四个 GPU 臂于 2026-09-07 夜间提交、全部 `COMPLETED 0:0`（job 15669504/05/06 三个 append Phase C′
臂 + 15669523 的 append Phase B′）。E2/E3 的全部闸门于 2026-09-08 在 CPU 上算出，Opus 用
独立重写、不复用项目函数的代码复算过 G-E2-1 与 G-E2-2，数字逐位一致。

按 §1.2 的两会话分离协议，本轮执行与裁决同在一个 Opus 会话完成（用户 2026-09-08 明确批准）。
分析脚本是 E1 已过单测的确定性 CLI，重算空间有限；记录在此以免被读成纪律松动。

### E2 闸门（58 held-out item × seeds 0 1，配对 bootstrap 10000，`default_rng(0)`）

| 闸门 | 判据 | 实测 | 判定 |
|---|---|---|---|
| **G-E2-1 权威保留** | `always_reset_append − zero_control` CI 下界 > 0 | **−0.0948**，CI [−0.2328, **+0.0431**] | **不过** |
| **G-E2-2 恒等式打破** | 末轮 `judge_raw_output` 逐字相同占比 < 0.90 | **30/116 = 0.259**（`agent_message` 16/116 = 0.138） | **通过** |
| G-E2-3 记录项 | 只记录 | `fixed_last_append − fixed_t2_append` = +0.259 [+0.129, +0.388]；`always_reset_append − fixed_last_append` = −0.345 [−0.466, −0.224] | — |
| **C2 上下文护栏** | prompt token 最大值 < 上下文上限 | **2960**（`always_reset_append`, turn 7），三臂 `n_turns_exceeding_limit=0` | **通过，截断混淆排除** |

`item_id` 集合三臂与复用的 `zero_control` 逐一相同（`item_id_sets_equal=true`，
`trajectory_id_intersection=116`，失败模式 3 的断言在位）。

臂均值（`final_turn_success`）：

| 臂 | overwrite | append |
|---|---:|---:|
| `zero_control` | 0.328 | 同一份数据（u≡0，与 `reset_mode` 无关） |
| `always_reset` | 0.776 | **0.233** |
| `fixed_last` | 0.776 | 0.578 |
| `fixed_t2` | 0.207 | 0.319 |

### E3 闸门（Phase B′，60 item × seeds 0 1，`--random-excite-p 0.5`）

**G-E3-0 采集检查：通过。** 666 行（与 overwrite Phase B 逐行同数）、`item_id` 集合与 Phase B
相同（60）、`reset_mode` 全为 `append`、动作分布 335/331（p̂=0.503）、`shard_frac` 十等分箱
逐箱两种动作都出现。

**G-E3-1 读出闸门：不过。** `closeness` 的 RC-3 失效：

| 判据 | overwrite Phase B | append Phase B′ |
|---|---|---|
| RC-1 预测技巧 | skill=0.175，20/20 split | skill=0.123，16/20 split → **不过** |
| RC-2 状态持久性（lag-1 去均值） | +0.276，p=5.6e-11 | **+0.588，p=4.6e-52** → 过，且强得多 |
| RC-3 动作耦合（`u_t` 系数） | +0.054，p=5.3e-4 | **+0.020，p=0.145** → **不过** |

**G-E3-2 状态依赖：不过。** 交互回归 `c_{t+1} = a·c_t + b0·u_{t+1} + b2·u_{t+1}·c_t + g·shard_frac + const`，
全 60 item、`n_pairs=546`、1000 次按 item bootstrap：

| 系数 | overwrite（§13.5 的负对照） | append Phase B′ |
|---|---|---|
| a（自回归） | +0.352 [+0.107, +0.594] | **+0.658 [+0.341, +0.855]** |
| b0（reset 基础效应） | +0.066 [−0.071, +0.195] | −0.017 [−0.119, +0.089] |
| **b2（效应 × 状态）** | −0.063 [−0.286, +0.178] | **+0.054 [−0.113, +0.222]** |
| g（信息累积斜坡） | +0.449 [+0.372, +0.521] | +0.289 [+0.209, +0.379] |

b2 跨 0，**且点估计符号为正**——与预注册要求的负号相反。判据要求"CI 不含 0 且符号为负"，
两条都不满足。

**G-E3-3 预算模式：路由到模式 A，但已无意义。** 按"最后一次 reset 之后还剩几轮"分层：

| | 剩 0 轮 | 其余 | 差 | Spearman(剩余轮数, 成功率) |
|---|---:|---:|---:|---|
| overwrite Phase B | 0.804 (n=56) | 0.475 (n=61) | **+0.328** | **−0.421，p<1e-5** |
| append Phase B′ | 0.429 (n=56) | 0.377 (n=61) | +0.052 | −0.054，**p=0.56** |

差 0.052 < 0.10 且 p=0.56 > 0.10，落在模式 A 区间、不在阈值近失区间，所以 §12.4 的"停下报告"
条款未触发。但 G-E3-2 已不过，E5 不跑，路由结果没有下游用途。

**E3 Step 5（离线回放准入）未执行。** 它的唯一用途是给 E5 放行，E5 已被 G-E3-2 关掉。

### 机制：让历史进入终点的那一行，和让 reset 触发重新求解的那一行，是同一行

append 下 reset 把合并题面追加到一个**已经含有模型自己旧答案**的历史后面。模型照抄旧答案，
不再重推：

| 臂 | 末轮回复中位字符数 | 末轮 ≤80 字符占比 |
|---|---:|---:|
| `always_reset` overwrite | 646 | 0.09 |
| `fixed_last_append` | 298 | 0.45 |
| `always_reset_append` | **18** | **0.91** |

`always_reset_append` 的典型末轮回复逐字是 `Current answer: 12`。overwrite 下 reset 把历史清空，
模型只看见合并题面，必须从头解一遍——**那次"从头解"才是 reset 的权威的来源，不是题面被重述
这件事本身**。append 保留了历史，模型于是复用历史里的答案，reset 退化成一句被忽略的重复。

这解释了全部四个失效：G-E2-1（`always_reset_append` 0.233 < `zero_control` 0.328，重复 reset
把"复读旧答案"的对话格式教给了模型）、G-E3-1 的 RC-3、G-E3-2 的 b0/b2、以及末轮优势从
+0.328 掉到 +0.052。

### 这条负结果比 G4 强

G4 的结论是"设定退化：终点被末轮单个动作构造性决定"。本轮做的是**对那个退化的干预实验**：
根因定位到一行代码，改掉它，退化确实消失（恒等式 1.000 → 0.259，a 从 +0.352 → +0.658，
lag-1 自相关 +0.276 → +0.588）——**而可控性的另一半随之塌掉**。执行器要同时满足两个性质，
"状态进入终点"与"动作推得动读出"，在这个设定里由同一个自由度控制，构成互斥。

这是带反事实、双向都测到的负结果，`article/PAPER_EXECUTION_PLAN.md` 的 §1.3 第 5 句要的正是
这种内容：三个前提可检验，本文给出检验程序，并展示一处"改一行让前提 0 成立"的干预如何让
前提 2 失效。

### 产物

- `scratchpad/e2_gates.json`（G-E2-1/2/3）、`c2_context.json`（C2）
- `scratchpad/closeness_readout_appendB.json`（G-E3-1）
- `scratchpad/interaction_appendB_all60.json`（G-E3-2）
- 轨迹：`outputs/ergo_appendC_{always_reset,fixed_last,fixed_t2}/`、`outputs/ergo_appendB_random_excite/`

### 偏离

1. **`analyze_ergo_closeness_readout.py` 无 CLI**，把 `ROWS_PATH`/`ENTROPY_PATH`/`OUT_PATH`
   硬编码在模块常量上（与 B1 同一类问题）。E3 Step 2 用一个 wrapper 改模块常量运行，
   **未改脚本文件**；entropy 两列跳过（append 目录无 `entropy_readout.json`，且熵读出族已被
   F 系列否决），候选表因此是 5 行而非 7 行。RC-0..3 与熵列无关。
2. **`analyze_ergo_append_context_length.py` 的 docstring 有一处错误声明**：它说重建对
   `overwrite` 与 `append` 两种模式都成立。对 overwrite 不成立——overwrite 下 reset 把
   `agent_history` 截成单条消息，而脚本按"逐轮累积 rows 1..t−1"重建，会系统性高估。用它跑
   overwrite 臂得到的 3282 token 是伪影，不可引用。C2 只判 append 臂，闸门不受影响。
   **待修**。
3. **执行与裁决同会话**（见上文执行摘要）。

### 留给下一步的判断题

1. 失效被归因到"模型照抄历史里的旧答案"。这是 Qwen3-4B 的性质，还是这个 harness 对上游
   协议的简化（固定节奏揭示、无 user-simulator、无"是否在尝试作答"分类、无 LLM judge）的
   性质？两者的修法完全不同。
2. 若是后者，向上游 Laban et al. 2025（arXiv:2505.06120）/ ERGO（arXiv:2510.14077）的
   prompt 结构靠拢能否恢复 reset 权威——且这种靠拢会不会把"状态"重新洗掉？

---

## R1：把格式指令搬进 system prompt——权威回来了，而且状态没走（2026-09-08）

**一句话结论**：E2/E3 判定的"append 抹掉了 reset 权威"**是我们自己 prompt 结构的产物**。把答案
格式指令从每轮重复改为 system prompt 只给一次，同一个 append 执行器在同一批 58 个 item 上
从 `−0.095`（CI 跨 0）变成 **`+0.181`（CI [+0.086, +0.284]，不跨 0）**。**权威与状态这次同时成立。**

### 闸门

三臂 job 15690582 / 15690584 / 15690586，全部 `COMPLETED 0:0`，各 664 行，
`prompt_profile=upstream`、`reset_mode=append`、`item_id` 集合与 Phase C 相同。

| 闸门 | 判据 | 实测 | 判定 |
|---|---|---|---|
| **G-R1-0 实现锁** | legacy 逐字节不变；upstream 侧结构正确 | legacy：**5980/5980** 条 `user_message` 逐字节复现（9 个已有产物目录，零 GPU）。upstream：三臂 664/664 行 `prompt_profile=upstream`，**含格式指令的 user_message = 0**，臂名 `*_up_append` | **通过** |
| **G-R1-1 机制闸门** | `always_reset_up` 末轮回复中位 ≥ 200 字符且 ≤80 字符占比 ≤ 0.30 | **563 字符 / 0.00**（legacy 同臂为 18 / 0.91） | **通过** |
| **G-R1-2 权威闸门** | `always_reset_up − zero_control_up` CI 下界 > 0 | **+0.1810**，CI **[+0.0862, +0.2845]** | **通过** |
| **G-R1-3 上游排序** | `fixed_last > always_reset > zero`，两两 CI 下界 > 0 | 实得 **`always_reset` > `fixed_last` > `zero`**：`fixed_last − always_reset` = **−0.0776** CI [−0.1638, +0.0000] | **不通过**（排序与上游相反） |
| **G-R1-4 记录项** | — | C2 prompt token 最大 **2395**，三臂 `n_turns_exceeding_limit=0`；`attempt_rate` 三臂**皆 1.0000** | — |

Opus 用独立重写、不复用项目函数的代码复算了 G-R1-2/G-R1-3 的三个差值，逐位一致。

### 臂均值（`final_turn_success`）

| 臂 | legacy profile | **upstream profile** |
|---|---:|---:|
| `zero_control`（append） | 0.3276 | **0.5776** |
| `always_reset`（append） | 0.2328 | **0.7586** |
| `fixed_last`（append） | 0.5776 | **0.6810** |
| `always_reset`（overwrite，参照） | 0.7759 | — |

`always_reset_up` 的 **0.7586 已经逼近 overwrite 的 0.7759 天花板**，而它保留了全部历史。

### 状态没有被换掉

| 量 | 值 | 读作 |
|---|---|---|
| 末轮 `agent_message` 逐字相同（`always_reset_up` vs `fixed_last_up`） | **2/116 = 0.017** | 历史真的进了终点 |
| 末轮 `judge_raw_output` 逐字相同 | 79/116 = 0.681 | 抽取后的数字会撞，但仍 < 0.90 |
| overwrite legacy 的同一对 | **116/116 = 1.000** | G4 的恒等式 |

**权威（+0.181，CI 不跨 0）与状态（`agent_message` 2/116）在同一个臂上同时成立。**

### 对本文档「E2–E3」一节的更正

那一节写的"执行器要同时满足的两个性质由同一个自由度控制、**构成互斥**"——**这个推广被证伪了**。
E2/E3 的测量本身没错（在 legacy prompt 结构下逐条复现），错的是把它读成 `reset_mode` 这个自由度
的性质。互斥是**每轮重复的答案格式指令**造成的：它在 append 下变成一个教模型简洁作答的 k-shot
示范，而 reset 的权威恰恰来自"逼模型重新推导一遍"。指令只给一次，权威就回来了。

**S3 那一节的结论要改写，`article/PAPER_EXECUTION_PLAN.md` 的 §1.3 第 5 句随之要改第二次。**
不改 E2/E3 的任何数字。

### 产物

`outputs/ergo_upC_{zero_control,always_reset,fixed_last}/`；闸门 JSON 在
`scratchpad/r1_gates.json`、`r1_dual.json`。

---

# R 相位的记录与裁决（2026-09-08 从计划文档原样迁入）

> 以下三节原本写在 [`ergo_fidelity_restoration_plan.md`](ergo_fidelity_restoration_plan.md)
> 的第十三～十五节。按 `.claude/docs.md`「计划文档 ≤ 150 行，超出的部分是结果，写到结果文档」，
> **逐字迁入本结果档案，一个字未改**。计划文档只保留仍要执行的规格。
>
> 同批迁入的还有原**第零节**（归因诊断与上游对照，本节末尾）——它是诊断结果，不是待执行的
> 规格。**外部文档引用的「`ergo_fidelity_restoration_plan.md` 第 0.1 节」现在指本文件末尾
> 那一节**（`docs/experiments/defense_line_redesign_plan.md:639`、`backup/README.md:36`）。

## 十三、R0 签字与 R1 提交记录（2026-09-08）

### 13.1 签字

用户于 2026-09-08 裁决 Q9–Q12（见第十节），**R0 签字完成**。R1 的三个 GPU 臂随即提交：

| 臂 | job | 输出目录 | 对应上游条件 |
|---|---|---|---|
| `zero_control_up` | **15690582** | `outputs/ergo_upC_zero_control/` | SHARDED |
| `always_reset_append_up` | **15690584** | `outputs/ergo_upC_always_reset/` | SNOWBALL |
| `fixed_last_append_up` | **15690586** | `outputs/ergo_upC_fixed_last/` | RECAP |

三臂全部 `--reset-mode append --prompt-profile upstream`，58 held-out item × seeds 0 1，
`--time 06:00:00`，`Qwen/Qwen3-4B`。

### 13.2 Q9 推迟 R5 的两个下游后果（记录，避免以后被读成遗漏）

1. **R1–R4 的任何闸门结果都不自动触发 R5。** 即使 G-R1-2 与 G-R1-3 全过、harness 判为忠实，
   熵触发臂也要等 ERGO 的 Koopman 工作跑完之后重新签字才开工。第七节保留完整规格，状态是待办。
2. **本轮能达到的上限因此是 S2，不是 S1。** 上游 ERGO 相对 RECAP 的 +14.1/+9.3 来自熵触发的
   自适应时机；不做 R5，本轮的被测臂就只有 Koopman-MPC 自己，S1 要靠它打赢同代价最优固定日程。
   这不是坏消息——**S1 本来就该由本项目自己的控制器去拿，用上游的熵触发去拿反而说明不了
   Koopman 的价值**——但要写清楚，免得以后把"没做 R5"记成"忘了做"。

### 13.3 Q10 与 Q11 合起来定义了 R2 的产物形状

- **主指标不换**（Q11）：`final_turn_success` 仍是主指标，attempt-only 分数**并列新增**，
  不是替换。论文里两个都报。
- **旧数据重算**（Q10）：既有的 overwrite Phase B/C 与 append Phase B′/C′ 全部按新口径重算，
  **旧数字不删**。全 CPU，用已有数据。
- 两条合起来 → 第四节 4.1 的 `analyze_ergo_dual_metric.py` 是 R2 的**唯一**汇报入口，
  任何单口径汇报算偏离（失败模式 23）。

### 13.4 Q12：合流点不变

9/13 合流点保持。R1（9/8 已提交）→ R2（9/9–9/10）→ R3（9/10–9/12）仍在窗口内；
R5 推迟本来就把日程压力去掉了。

---

## 十四、G-R2-1 不通过：规则层单向过判（2026-09-08）

**判定：不通过。** 按 4.2 节，下一步是**启用 LLM 层并重测**；若 LLM 层仍不过，则**不换口径**，
R2 终止并报告。在 LLM 层通过之前，`final_attempt_success` 与 `attempt_rate` **不得写进任何
对外结论**。

### 14.1 数字

60 行盲标样本（按规则层判定分层各 30，臂名/轮次/judge 分全部剥离，两遍独立全新 subagent 实例，
P 协议）：

| 比较 | agreement |
|---|---|
| 标注者间 A vs B | **0.9833**（59/60） |
| 规则层 vs A | 0.8333（50/60） |
| 规则层 vs B | 0.8167（49/60） |
| 规则层 vs 两遍共识（59 行） | 0.8305（49/59） |

判据要求 ≥ 0.85，**两遍都不到**。标注者间 0.983 说明分歧是真的，不是标注噪声。

### 14.2 失效是单向的，这一点决定了怎么读现有数字

混淆矩阵（规则层 × 两遍共识，59 行）：

| | 共识 = attempt | 共识 = 非 attempt |
|---|---:|---:|
| **规则 = attempt** | 19 | **10 ← 全部误差在这一格** |
| **规则 = 非 attempt** | **0** | 30 |

**规则层从不漏判**：它说不是 attempt 的，人判 30/30 全部同意。误差 100% 是过判。

过判的 10 行是同一种东西——**长篇的"拒绝求解"**：模型写一整段解释为什么信息不足、算不出来，
然后给一个兜底数字。例如 `row_33`「The percentage ... varies by school, district, and region.
Without specific data ...」、`row_36`「However, the total charge ...」、`row_57`「it is not
possible to calcula...」。这正是 Laban 七分类里的 hedging / discussion 类，不是 answer attempt。
**80 字符的规则分不出"80 个字符的推导"和"80 个字符的解释为什么推不出来"。**

### 14.3 对既有 Q10 数字的两条限定（必须一起引用）

1. **`attempt_rate` 是上界，不是估计。** 非 attempt 一侧 30/30 可靠，所以真实 attempt 率
   **不高于**已报的数。
2. **修正后臂间差距会缩小，不会重排。** 过判需要**长回复**才可能发生，而长回复正是
   overwrite 与 `fixed_last_append` 多、`always_reset_append`（末轮中位 18 字符）几乎没有的东西。
   在规则-attempt 那一层上过判率是 10/30 = 33%；若各臂同率，`always_reset_append` 的 0.086
   几乎不动而 overwrite 的 0.905 会明显下来。**过判率是否跨臂均匀是未知的，且没有理由假设它均匀**
   ——这正是必须先过 G-R2-1 的原因。

**主结论不受影响**：Q11 裁决主指标不换，E2/E3 与 Q10 的定性结论（append 下每轮 reset 不比
从不 reset 好）在 legacy 口径上独立成立，不依赖这个分类器。

### 14.4 重测时必须一并修的抽样缺陷

本次样本按**规则层**分层各 30，于是共识 attempt 类只有 **19 行 < 判据要求的 20**。
即使 agreement 达标，这一条也不满足。重测时改为：先用当前分类器抽一个更大的候选池
（例如各 45），标完后按**共识**类别核对两类各 ≥ 20；或直接过采样规则-attempt 层。
**这不是放宽判据，是修一个抽样设计缺陷**——原判据"两类各 ≥ 20"指的就是共识类别。

---

## 十五、R1 裁决（Opus，2026-09-08）

结果与数字见 [`ergo_multiturn_reliability_pilot.md`](ergo_multiturn_reliability_pilot.md) 的
「R1」一节。本节只出裁决。

### 15.1 G-R1-0/1/2 通过；第零节的归因成立

`always_reset_append − zero_control` 从 `−0.0948`（CI 跨 0）到 **`+0.1810`（CI [+0.0862, +0.2845]）**，
唯一改动是那条答案格式指令的位置。**E2/E3 的 G-E2-1 失效被解释并逆转。**

同时 `agent_message` 末轮逐字相同只有 2/116，所以**状态仍在终点里**。E2/E3 那句"两个性质互斥"
是对 legacy prompt 结构的描述，不是对 `reset_mode` 的描述——已在 pilot 文档里更正。

### 15.2 G-R1-3 不通过：裁决为**进 R4**，且**不许把它论证成通过**

上游三次测量都是 RECAP > SNOWBALL；我们得到 SNOWBALL > RECAP（`fixed_last − always_reset`
= −0.0776，CI [−0.1638, +0.0000]）。**判为不通过，记录在案。**

我现在认为这条判据当初写得不好：`always_reset` 花 T 次 reset、`fixed_last` 花 1 次，**二者不同代价**，
而本计划自己在 E5 臂表里就写明 `always_reset`"不是同代价对手，只作权威参照"。把一个代价不对称的
比较当作保真度判据，是把两件事混在了一起。**但这个事后认识不能把"不通过"变成"通过"**
（失败模式 26 的同构：判据写完就该认，改判据要在跑之前）。

**裁决：进 R4**（模型对照，约 1 GPU-小时）。上游模型集是 Phi-4 / Llama-3.1-8B / GPT-4o 系，
**没有 Qwen**；一个 4B 模型比 GPT-4o 更依赖反复的上下文合并，是这个排序差异最省事的解释，且可测。
**不进 R2 的 LLM 层**（理由见 15.3）。

### 15.3 R2 的剩余工作降级

三个 upstream 臂的 `attempt_rate` **皆为 1.0000**，`final_attempt_success` 与
`final_turn_success` 逐位相同。**在修好的 prompt 结构下，attempt / 非 attempt 的区分整个消失了**
——R2 这个口径本来就是为了处理 legacy harness 产生的非作答回复而存在的。

因此：
- **R1 及其之后的任何结论都不依赖 R2 的分类器**，G-R2-1 不过不阻塞 R 相位；
- R2 的 LLM 层仍然欠着，但它现在只影响 **Q10 对 legacy 旧数据的重算**这一件事，优先级降到
  R3/R4 之后；
- 14.3 的两条限定继续有效，且**只适用于 legacy 数据的 upstream 口径列**。

### 15.4 对 S1 前景的诚实修正：0.4 节的论证没了一半

第 0.4 节把"RECAP > SNOWBALL 的非单调性"当作"控制问题真实存在"的证据。**在我们的 harness 里
这个非单调性不存在**（G-R1-3），所以那条论证作废。

剩下的、也是唯一还站着的问题是：**在固定预算 k 下，最优的 reset 时机是否随 item / 状态变化。**
那是 b₂ 测的东西，而 b₂ 上一次是在 legacy prompt 结构的 Phase B′ 上测的，
**那份数据现在已知被同一个混淆污染**。

**因此下一步是 R1.5**：在 upstream profile 下重跑随机激励的 Phase B′（1 个 GPU 臂，约 1 小时），
重算 G-E3-1（读出闸门 RC-0..3）与 G-E3-2（b₂ 与它的 overwrite 负对照）。**这不是把旧闸门翻案**
——旧判定在 legacy 数据上仍然成立；这是在一个已修正的 harness 上**重新提出同一个问题**，
判据逐字沿用、不改阈值。R1.5 的结果决定 ERGO 线还有没有 S1/S2 可谈。

---

## 零、这份计划为什么存在：E2/E3 的失效被归因到我们自己的 prompt 结构

### 0.1 决定性证据：我们测到的"reset 权威"其实是"模型有没有写出推导"

E2/E3 的结论是 append 下 reset 失去权威（G-E2-1、G-E3-1、G-E3-2 三道不过）。2026-09-08 的
事后诊断把这件事解释掉了：

| 臂 | 末轮回复中位字符 | 末轮 ≤80 字符占比 | `final_turn_success` |
|---|---:|---:|---:|
| `zero_control`（从不 reset） | 19 | 0.66 | 0.328 |
| `always_reset_append` | 18 | 0.91 | 0.233 |
| `fixed_t2_append` | 19 | 0.69 | 0.319 |
| `fixed_last_append` | 298 | 0.45 | 0.578 |
| `fixed_last`（overwrite） | 646 | 0.09 | 0.776 |
| `always_reset`（overwrite） | 646 | 0.09 | 0.776 |

跨 6 个臂 **Pearson(末轮回复中位长度, 成功率) = +0.984**。臂内同样成立：长回复（>80 字符）
成功率 0.67–1.00，短回复 0.15–0.36。**reset 的全部权威 = 它让模型重新写了一遍推导**，
而不是题面被重述这件事本身。

让"不写推导"成为可选项的是我们自己加的一行（`ergo_math_trajectory.py:42-46`）：

```python
_ANSWER_FORMAT_INSTRUCTION = (
    'Give your current best-guess final numeric answer to the math problem, even if you are '
    'not fully confident yet or do not have all the details, on its own line in this exact '
    'format: "Current answer: X" (a single number).'
)
```

它被拼在**每一轮** user message 末尾（`:90` 与 `:97`），而这个 harness **完全没有 system prompt**
（`agent_history` 从空列表开始）。overwrite 下无害——历史被清空，模型只看见一次指令、必须从头解；
append 下模型看见 k 轮「指令 + 自己回的 `Current answer: 12`」，这是一个 **k-shot 的简洁作答示范**。

### 0.2 与上游的三条偏离（ground truth，非论文措辞）

| | 上游 | 我们 |
|---|---|---|
| 格式指令位置 | **system prompt，只给一次**（`prompts/math/math_system_prompt.txt`，`generate_system_prompt()`）：*"The answer should be a single number..."* | **每轮 user message 末尾**，重复 4–12 次 |
| system prompt | 有，minimal，turn 1 之前给一次 | **完全没有** |
| 打分对象 | **strategy classifier** 把回复分成 7 类（澄清/拒答/对冲/追问/讨论/缺失/**answer attempt**），**只给 answer attempt 打分**，再由 answer extractor 抽 span | 每轮正则抽数字打分，**无 attempt 分类** |
| ERGO 的 reset 内容 | **让模型把累积输入重写成一个优化的单轮 prompt** | 机械拼接的 bullet list |
| ERGO 的 reset 语义 | **stateless**："passed into a new instance of the model, simulating a stateless chat environment with no memory of prior turns" | 我们的 `overwrite` **就是这个** |
| 模型 | Phi-4 / Llama-3.1-8B-Instruct / GPT-4o / GPT-4o-mini / GPT-4.1 | Qwen3-4B（**上游模型集里没有 Qwen**） |

第三行最要命：**上游会把 `Current answer: 12` 这类回复判为「非作答」而不打分，我们给它打 0 分
并计入指标。** 所以 append 下的 `y_task_success` 有相当一部分在测「模型愿不愿意写推导」，
而上游明确把这一维过滤掉了。

来源：
- Laban et al. 2025, *LLMs Get Lost In Multi-Turn Conversation*, arXiv:2505.06120（[abs](https://arxiv.org/abs/2505.06120) / [html](https://arxiv.org/html/2505.06120v1)）
- *ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models*, arXiv:2510.14077（[abs](https://arxiv.org/abs/2510.14077) / [html](https://arxiv.org/html/2510.14077v1) / [ACL Anthology](https://aclanthology.org/2025.uncertainlp-main.23/)）
- 上游代码与 prompt：[microsoft/lost_in_conversation](https://github.com/microsoft/lost_in_conversation)（本仓库 `resources/ergo_gsm8k_sharded.jsonl` 的来源）

### 0.3 我们的结果与上游同向，只是幅度更大

| | SHARDED（无 reset） | SNOWBALL（每轮 recap） | RECAP（末轮一次 recap） |
|---|---:|---:|---:|
| Laban GPT-4o-mini | 50.4 | 61.8 | **66.5** |
| Laban GPT-4o | 59.1 | 65.3 | **76.6** |
| ERGO 论文复现 4o-mini | 44.3 | 54.0 | **57.7** |
| **我们（append，Qwen3-4B）** | 0.328 | 0.233 | **0.578** |

三次独立测量都给出 **RECAP > SNOWBALL**。我们的排序一致，唯一差异是我们的 SNOWBALL 掉到
SHARDED 之下。Laban 的 root cause #4 是 *"overly rely on previous (incorrect) answer attempts
leading to lengthier 'bloated' answers"*——上游那里表现为**回复臃肿**，我们这里表现为**回复简短**，
因为我们把输出格式钉死了。**同一现象的两个表面形态。**

### 0.4 对 Koopman 的意义：非单调性就是控制问题，而它只在 append 下存在

overwrite 下 `always_reset ≡ fixed_last` 是 116/116 的恒等式——reset 次数完全不影响结果，
没有任何可权衡的东西（G4 的结论）。append 下「多 reset 反而更差」，上游两个模型 + 我们的数据
三次确认。**一个自适应控制器需要的正是这种非单调性。**

而上游还有一个数字直接就是本项目的 S1：

| | GPT-4o-mini | GPT-4o |
|---|---:|---:|
| RECAP（最优固定规则） | 57.7 | 66.3 |
| **ERGO（熵触发的自适应时机）** | **71.8** | **75.6** |

**+14.1 / +9.3 分——"自适应时机打赢最好的固定规则"。** 上游声称它存在。

同时上游留了一个**它自己没做的消融**：ERGO 的增益同时来自 (a) 熵触发的自适应时机与
(b) 模型重写的 consolidation prompt，论文没有报「固定日程 + 同样的重写 prompt」这一臂，
**两者贡献是混淆的**。拆开它是本计划 R3 的副产品，也是一个能独立成立的贡献。

### 0.5 三个不能忘的诚实前提

1. **R1/R2 可能恢复权威，但最优解仍停在「末轮 recap 一次」这个固定规则上。** 上游的 RECAP
   就是这个固定规则，Laban 从未声称自适应能赢它。真正能翻盘的是熵触发臂（R5），而它正是
   `backup/two_task_success_plan.md` 第九节 Q4 裁决掉的那一项。**Q4 值得重开，但那是用户的决定（Q9）。**
2. **`defense_line_redesign_plan.md` §12.3 曾预测「第 2 层需重测但大概率保留」。它被证伪了。**
   本计划的预测同样可能被证伪；G-R1-1 就是为了让证伪来得便宜且明确。
3. **改打分口径会让所有历史数字不可直接比较**（Q11）。R2 因此强制双口径报告。

---

---

# EK0 / EK1：upstream harness 上的状态依赖与 Koopman 辨识（2026-09-08）

数据 `outputs/ergo_upB_random_excite/`（job 15696226，upstream profile，append，
60 item × seeds 0 1，`random_excite p=0.5`，666 行）。全部 CPU，无新 GPU 作业。

## G-R15-0 采集：✅ 过

666 行 ✅；60 item ✅；`item_id` 集合与 overwrite Phase B **逐一相同** ✅；激励动作 335 / 331 两种都有 ✅。

## G-R15-1 读出（`closeness` 的 RC-0..3）：❌ 不过

`analyze_ergo_closeness_readout.py`，产物 `outputs/ergo_upB_random_excite/closeness_readout_state_report.json`。

| 闸门 | 判据 | 实测 | 判定 |
|---|---|---|---|
| RC-0 | 与 `y_task_success` 同一仪器 | 构造性成立 | ✅ |
| RC-1 | ARX(aux=truth) 在 ≥18/20 split 上打过最好 null | **14/20**；mean skill **+0.1077**（mse 0.0273 vs best null 0.0295） | ❌ |
| RC-2 | 去趋势 lag-1 自相关 > 0，p<0.05 | **+0.5355**，p=7.6e-42 | ✅ |
| RC-3 | 输入增益 `u_t` 系数 > 0，p<0.05 | **+0.0131，p=0.339**，n=546 | ❌ |

## G-R15-2 状态依赖：❌ 不过

`analyze_ergo_state_action_interaction.py --held-out-frac 0`（预注册的全 60-item 口径）。
**首次运行误用了默认 `--held-out-frac 0.25`（45 item / n_pairs=416），已按预注册重跑**；
两次结论一致，45-item 版留作纸面痕迹（`state_action_interaction_report.json`）。

判据：b₂ 的 95% CI 不含 0 **且符号为负**。实测 **b₂ = +0.0786，CI [−0.1229, +0.2407]**——跨 0，
且符号为正。**不过。**

## G-R15-3 记录项：a / b0 / g 三方对照（全部 60 item，n_pairs=546）

| 数据 | a | b0 | b2 | g |
|---|---|---|---|---|
| overwrite Phase B（负对照，引用 `backup/two_task_success_plan.md` 13.5，不重跑） | +0.3522 | +0.0661 | **−0.0632** [−0.2861, +0.1779] | +0.4493 |
| legacy append B′（本次新算） | +0.6577 | −0.0165 | **+0.0538** [−0.1129, +0.2220] | +0.2895 |
| **upstream append B（本次新算）** | **+0.5963** | **−0.0457** | **+0.0786** [−0.1229, +0.2407] | **+0.4755** |

**b₂ 在三份数据上全部跨 0。** 15.4 节曾把 legacy Phase B′ 的 b₂ 判为"被混淆污染、需要在修正后的
harness 上重问"——重问之后答案没变。自回归系数 `a` 在 append 下（0.60/0.66）显著高于 overwrite
（0.35），信息累积斜坡 `g` 三处都强显著。

## EK1 Koopman 辨识（单 45/15 split，记录用）

`fit_koopman_ergo_closeness.py --rows-path outputs/ergo_upB_random_excite/trajectories.jsonl`，
产物 `koopman_fit_report_closeness.json`：

- ARX held-out rollout MSE（aux=truth）**0.023459** vs 最好 null（stateless OLS）**0.024214** → skill **+3.1%**；
  naive rollout 0.033188 输给全部 null。
- `controllability_rank=3`（满秩），`gramian_condition=3.87e5`，`A_spectral_radius=1.0299`。
- **`B` 在 closeness 维上是 −3.5e−4**——与 RC-3 的 `u_t` p=0.34、交互回归的 b0/b2 跨 0 三处同向。
- **单 split 不作判定**：G-EK1-1 判在 20-split 上，即上面 RC-1 的 14/20。

## 读法：失效定位在**输入通道**，不在状态

`closeness` 在 upstream harness 上的**状态**比以往任何一份 ERGO 数据都强（RC-2 = 0.5355，
`a` = 0.60），Koopman 拟合平均也打得过 null（skill +10.8%）。**塌的是 u 这一路**：
reset 对下一轮 closeness 的边际效应，在四个互相独立的估计上都不可与 0 区分。

**这与 R1 存在一处必须解释的张力**：R1 在同一 profile 下测到 `always_reset − zero_control`
= **+0.1810**，CI [+0.0862, +0.2845]。**一个推不动读出的执行器，不该推得动终点。** 两种读法：

1. **口径不匹配**（更可能）：R1 测的是**持续策略的终点效应**，RC-3/b0 测的是**一步边际效应**。
   `random_excite p=0.5` 的一步设计对前者没有功效——需要先算这个设计能探测的最小效应量。
2. **指标失效**：`closeness` 是在 legacy harness（末轮中位 18 字符）上标定的；upstream 下末轮
   中位 563 字符，抽取与打分行为可能已经不同（本次实测 `mean=0.6057`、`frac==1.0=0.17`）。

按第九节的**闭合前审查**，这两条查清之前不闭合 ERGO 线。

## 闭合前审查（第九节 A/B/C 三项，2026-09-08，全部零 GPU）

问题：EK0 的 u 通道全线为零，是**模型变强导致任务饱和**，还是**动力学有某种特殊之处让 Koopman
拟合不了**？答案是**两者都不是**——是激励臂的功效配不上要测的效应量。

### B 指标有效性：没有饱和，读出也没坏

| | `closeness==1.0` 占比 | 从已解出状态出发的转移 | `P(y'=1｜y=0)` | 逐行 y 均值 |
|---|---:|---:|---:|---:|
| upstream append B | 17.0% | 3.7% | 0.177 | 0.170 |
| legacy append B′ | 9.2% | 2.2% | 0.092 | 0.092 |
| overwrite Phase B | 13.7% | 2.7% | 0.145 | 0.137 |

终点成功率 0.775 是**尾部陡升**，不是全程贴顶：绝大多数转移仍在"未解出"区域，动作空间是够的。
把样本限制在未解出子集（n=526）重估 u，系数与全样本一致（−0.0032 vs −0.0035）——**不是饱和把效应吃掉了。**

### 动力学：状态是真的，一阶结构也没漏东西

- 在确定性斜坡 `shard_frac` 之上加状态 `c_t`，R² **0.3964 → 0.5652（+0.169）**；再加 u 只 +0.0001。
- 加分布滞后（`u_t`、`u_{t-1}`）与**累积 u**，四项**没有一项** CI 排除 0 → `nu=1/mu=1` 不是太窄，
  效应不是藏在更长的滞后里。
- RC-2 = **0.5355** 是三份 ERGO 数据里最强的状态持续性，`a` = 0.60。

**所以不是"Koopman 拟合不了这套动力学"**——状态项是这套数据里最结实的部分。

### C 口径匹配：激励臂的最小可检测效应是 R1 效应的 2.7 倍

终点剂量–反应 `y_final ~ frac_resets`（n=120 轨迹，平均 5.5 轮）：

| | sd(frac_resets) | 残差 sd | se | **MDE@80%** | 能否分辨 R1 的 +0.1810 |
|---|---:|---:|---:|---:|---|
| upstream append B | 0.218 | 0.419 | 0.176 | **+0.49** | ❌ 差 2.7 倍 |
| overwrite Phase B | 0.218 | 0.463 | 0.194 | +0.54 | ❌ |

要在这个设计下分辨 +0.1810 需要 **885 条轨迹**（2 seed 下约 442 题）= 现有数据的 **7.4 倍**。
**upB 里每一个"u 是死的"的数字，在 R1 的效应量上什么都排除不了。**

而且 pooled 估计本身读错了方向。换成 **item 固定效应**（每题两个 seed 的 reset 模式不同，
47/60 题有题内变异）：

| | item-FE 终点效应（y） | 95% CI |
|---|---:|---|
| upstream append B | **+0.3463** | [−0.0708, +0.7032] |
| overwrite Phase B | **+0.8459** | [+0.2562, +1.3600] |

upB 的点估计从 ≈0 **翻成 +0.35**，**R1 的 +0.1810 稳稳落在 CI 内**。题间难度方差才是压住 pooled
估计的东西，不是 reset 没有效应。

### A 过程正确性

查出并已修一处：G-R15-2 首跑用了默认 `--held-out-frac 0.25`，预注册是全 60 item `--held-out-frac 0`；
按预注册重跑，结论未变。其余参数与负对照引用无误。

### 审查结论：**不闭合**，且要回写一条旧判定

1. **ERGO 线不因 EK0/EK1 闭合。** 三道闸门是被一个 MDE ≈ 0.5 的仪器判的，而要测的效应是 0.18。
2. **append 的执行器确实弱于 overwrite**（upB 的 CI 上界 +0.70 vs overwrite 点估计 +0.85），
   但"更弱"不等于"零"，**这个设计分不出来**。
3. **⚠️ 回写**：同一个功效地板适用于 overwrite Phase B。G-E3-2 当初"负对照通过（b₂ 跨 0）"
   是在同一个分辨不出效应的仪器上宣布的——它证明的是**测不到伪影**，不是**没有伪影**。
   凡引用该负对照的地方都要带这条限定。
4. **下一步是激励设计，不是换读出、也不是换模型。** R1 的配对设计（58 题配对，CI 半宽 0.10）
   比这个 i.i.d. 激励臂**效率高约 8 倍**，因为它把题间难度配对消掉了、且两端都取到
   （Δfrac = 1 vs 现在的 sd 0.354）。代价是配对到两端就没有中间档的 u 变异供辨识。
   **需要用户裁决的取舍**：分层激励（一部分题跑 0/1 配对拿功效，一部分题跑中间预算拿辨识）
   还是别的设计。属新 GPU 作业，**未提交**。

---

# EK 臂设计的功效依据（2026-09-08，零 GPU，全部从既有数据反推）

设计取舍见 [`ergo_fidelity_restoration_plan.md`](ergo_fidelity_restoration_plan.md) 第四、五节；
外部效应量区间见 [`../task/MULTITURN_RESET_INTERVENTION_EVIDENCE.md`](../task/MULTITURN_RESET_INTERVENTION_EVIDENCE.md)。
本节只放数字。

## 一、题库是 103 题，且辨识集与评测集目前有 15 题重叠

`resources/ergo_gsm8k_sharded.jsonl` 共 **103 题**，恰好被切成三块：
**Phase B 独占 45 + 两边共有 15 + held-out 独占 43 = 103**。

→ 若拿 Phase B 的 60 题辨识、held-out 58 题评测，**有 15 题既进辨识又进评测**。
**设计决定**：辨识只用 **B 独占的 45 题**，评测用 **held-out 全部 58 题**（与 R1 / Phase C 逐字相同，
保持可比）。两集合此时严格不交。

## 二、配对对比的方差分解：seed 还买得到功效，item 不是硬上限

对 `fixed_last − always_reset`（policy-vs-policy，S1 的形状）做单因素分解
`Var(配对均值差) = σ_b²/I + σ_w²/(I·S)`：

| 指标 | σ_within² | σ_between² | I=103 且 S→∞ 的地板 |
|---|---:|---:|---:|
| `y_final` | 0.14655 | 0.03025 | MDE **0.048** |
| `closeness` | 0.02428 | 0.00461 | MDE **0.019** |

`σ_within² ≈ 5 倍 σ_between²` → **加 seed 仍然有效**，题目数不是当前的硬约束。

| I | S | se(`y_final`) | MDE@80% | MDE(`closeness`) | 轨迹/臂 | GPU-h/臂 |
|---:|---:|---:|---:|---:|---:|---:|
| 43 | 6 | 0.0357 | 0.100 | 0.040 | 258 | 1.50 |
| **58** | **6** | **0.0307** | **0.086** | **0.034** | **348** | **2.03** |
| 58 | 10 | 0.0278 | 0.078 | 0.031 | 580 | 3.38 |
| 103 | 6 | 0.0230 | 0.065 | 0.026 | 618 | 3.60 |

运行时基数：R1 三臂实测 **19.1 / 21.2 / 22.8 秒每轨迹**（116 轨迹，37/41/44 分钟），
取 **21 s/轨迹**。假设：Qwen3-4B、`max_new_tokens=512`、upstream profile、同一队列同型号卡。

**选定 58 题 × 6 seed**：`y_final` MDE **0.086**，`closeness` MDE **0.034**。
对照外部效应量：ERGO 声称的"自适应打赢最优固定规则"是 +0.141（4o-mini）/ +0.093（4o）。
**预注册限定：本设计能可靠分辨 ≥0.09 的 S1 效应；若真实效应小于 0.09，结果必须写成
"本设计对 <0.09 的效应功效不足"，不得写成"自适应没有优势"。**（EK0 的教训，失败模式 29。）

## 三、为什么辨识改用反事实分叉，而不是继续做观测式激励

同一 (item, turn) 跨 seed 的 `closeness` 离散度，与观测式一步回归的残差对比：

| | 跨 seed 同 (item,turn) 的 sd | 观测式一步残差 sd | 方差比 |
|---|---:|---:|---:|
| `ergo_upC_always_reset` | 0.0219 | 0.1573 | ~51× |
| `ergo_upB_random_excite` | 0.0409 | 0.1534 | ~14× |

跨 seed 的那一列**仍然包含前缀分叉带来的噪声**，而反事实分叉对**共享完全相同的前缀**，
所以它是分叉设计噪声下界的**上界**。取保守端 14×，分叉每轮生成成本约 2×，
→ **每 GPU-秒的信息量约 7 倍于观测式激励臂**。

补充：解码是随机的（`temperature=0.7`），两个 seed 的 `agent_message` 有 **94.9%** 不逐字相同，
所以 seed 确实是独立样本，不是重复。（turn 1 的 `closeness` 跨 seed sd 为 0 是**读出在 turn 1
分辨不出差别**，不是解码确定性——不要据此认为 seed 无效。）

---

# G-EKA-0 实现锁：通过（2026-09-08，零 GPU）

规格见 [`ergo_fidelity_restoration_plan.md`](ergo_fidelity_restoration_plan.md) 第四节。

## 改了什么

| 文件 | 改动 |
|---|---|
| `src/persona_drift/ergo_math_trajectory.py` | 把 `run_ergo_math_trajectory` 的循环体抽成 `_run_one_turn`（**不改变任何行为，不改动历史列表，返回续接后的历史**），新增 `run_ergo_math_branch_trajectory` 返回 `(base_rows, counterfactual_rows)` |
| `scripts/run_ergo_branch_arm.py` | **新脚本**。不改 `run_ergo_math_screening.py`——那个 driver 的续跑靠 `expected_rows_by_trajectory_id`，而反事实行是单轮、有自己的 `trajectory_id`，走它会同时污染续跑记账和所有按 `trajectory_id` 分组的分析器 |
| `tests/test_ergo_math_trajectory.py` | 新增 8 条单测 |
| `environment/run_ergo_ekA_branch.sbatch` | 新 sbatch，**未提交** |

## 判据与实测

| 判据 | 实测 |
|---|---|
| 重构后 `run_ergo_math_trajectory` **逐字节不变** | ✅ **8/8 配置**（`overwrite`/`append` × `legacy`/`upstream` × `zero_control`/`constant_remind`）行内容与 generate/judge **调用序列**都逐字节相同。用的是内容敏感桩（回复 = 消息全文 + seed 的哈希），常量回复桩分不出"共享前缀"和"碰巧相同" |
| 分叉不扰动基线分支 | ✅ 分叉跑出的 `base_rows` 与同配置普通跑**完全相同**（4 种 reset_mode × profile 组合，单测 `test_branch_base_rows_are_identical_to_a_plain_run`） |
| 前缀真的共享 | ✅ 每轮 base 与反事实看到的消息列表**除最后一条 user 外逐条相同**，且最后一条不同 |
| 动作确实翻转 | ✅ `u_cf = 1 − u_base` 全轮成立；base 用 `constant_remind` 时反事实是 0 |
| 共同随机数 | ✅ 反事实**复用 base 的 `agent_seed`**（`seed·10⁶ + turn·100 + 1`），judge 用 +3 区分 |
| 成本 | ✅ **每轮恰好 2 次 generate**（judge 是正则，不调模型）→ 成本正好 2× |
| 全量测试 | **496 passed / 5 failed**；5 个失败全在 `test_surface_features.py`，是 NLTK 语料未下载的环境问题，与本改动无关 |

## 题集与运行时（sbatch 里写死，可审计）

`--item-ids` 是 **Phase B 用而 held-out 评测集不用的那 45 题**（103 题的库切成 45 / 15 共有 / 43）。
写死而不是运行时推导，是为了题集不随 bank 或选择种子漂移。EK-B 在 58 held-out 题上评测，
**辨识与评测严格不交**。

45 题 × 3 seed = **135 条轨迹**，每轮生成两次 → 约 42 s/轨迹（R1 实测单次 19–23 s）→ **约 95 分钟**。
`--time 04:00:00`（约 2.5× 余量）。加了 `--requeue`（本队列会抢占，K1 就被抢占过）与 pip 的 `flock`。

**干跑验证**：用 sbatch 里逐字的命令行跑通了完整路径（把 135 条轨迹预标记为已完成，
因此不加载模型），item 选择、续跑判定、两文件写入、退出码全部正常。

---

# EK-A 落地：G-EKA-1/2/3/4（2026-09-08，job 15708895）

规格见 [`ergo_fidelity_restoration_plan.md`](ergo_fidelity_restoration_plan.md) 第四节。
数据 `outputs/ergo_ekA_branch/`（COMPLETED 1h11m，**759 个同前缀反事实对**，45 题 × seeds 0 1 2，
`--reset-mode append --prompt-profile upstream`，base 分支 `zero_control`）。
分析全部零 GPU：`analyze_ergo_branch_pairs.py` → `branch_pair_gates_report.json`；
`fit_koopman_ergo_branch.py` → `koopman_fit_report_branch.json`。item 自举 4000 次，`seed=0`。

**判分口径**：ERGO 的 judge 是**正则抽取**（`ergo_math_judge`），不是模型自判，
`.claude/global.md` 的具名例外与本节无关。**3 seed** ✅。

## G-EKA-1 执行器权威（同前缀反事实对）：✅ 过

| 量 | 判据 | 实测 |
|---|---|---|
| `mean(Δc_k)` | 95% CI（item 自举）不含 0 | **+0.0265，CI [+0.0128, +0.0416]** ✅ |
| `mean(Δy_k)`（记录） | — | +0.0369，CI [+0.0138, +0.0616] |
| 本设计的 MDE@80% | 失败模式 29 要求同址给出 | **+0.0206**（se≈0.0074） |

**效应是稀疏的，不是弥散的**：759 对里 **87.5% 的 `Δc_k` 恰好为 0**，9.2% 为正、3.3% 为负。
均值由不到十分之一的轮次撑起来。**实测效应 +0.0265 只比本设计的 MDE 0.0206 高 29%**——
过是过了，没有余量。

## G-EKA-2 状态依赖（`Δc_k ~ c_{k−1}`）：❌ 不过，但**判为不可判定，不是判为无状态依赖**

判据：斜率 95% CI 不含 0（预期为负）。实测 **斜率 −0.0059，CI [−0.0737, +0.1020]**，
n=624（丢弃 135 个 turn-1 对，无前态）。

**必须与判定同址引用的两条（失败模式 29）**：

1. **MDE**：本设计的斜率 MDE@80% = **±0.1255**（se≈0.0448）。若 reset 的效应在 `c_{k−1}=1` 处
   恰好归零，斜率应为 **−0.0281**（= −截距）——**MDE 是它的 4.5 倍**。这个设计对预期量级的
   状态依赖**本来就没有功效**。
2. **杠杆**：`c_{k−1}` 几乎没有变异。624 行里 **469 行（75%）取值恰为 0.500**，IQR = **0**，
   sd = 0.165，80.6% 落在 [0.4, 0.6)。回归没有 x 轴可用。

分箱均值（记录，非判据）：

| `c_{k−1}` | n | mean `Δc_k` |
|---|---|---|
| [0.0, 0.2) | 26 | −0.0110 |
| [0.2, 0.4) | 6 | +0.1304 |
| [0.4, 0.6) | **503** | +0.0266 |
| [0.6, 0.8) | 44 | +0.0373 |
| [0.8, 1.0) | 45 | +0.0021 |

**这是读出量程问题的第三次现身**（`defense` 的 judge ceiling 0.91、`stance` 的两次空结果之后）。
`Δc_k` 的 87.5% 零值与 `c_{k−1}` 的 75% 单点堆积是同一件事：`closeness` 在 upstream harness 上
**是一个几乎不动的四值量**，不是连续读出。

## G-EKA-3 拟合：✅ 过

1248 个转移（624 个同前缀对 × 2 个动作），45 题，20 折 item-disjoint。
判据是 skill 的 item 自举 CI 不含 0（**不是折数投票**，理由见 `fit_koopman_ergo_branch.py` 模块 docstring）。

| null / 模型 | 折均一步 MSE |
|---|---|
| `constant` | 0.055553 |
| `turn_mean` | 0.043047 |
| `stateless_ols`（含 `[1, shard_frac, u]`） | **0.032569**（最好 null） |
| `arx` | **0.024445** |

**mean skill = +0.007241，CI [+0.002285, +0.015312]** → ✅ 过。折数记录项 **16/20**（对标 RC-1 的 14/20、K2 的 14/20）。

算子（`c_next = const + a·c_prev + B·u + g·shard_frac`），CI 为 item 自举：

| 系数 | 值 | CI |
|---|---|---|
| const | −0.0514 | [−0.1208, +0.0282] |
| **a**（`c_prev`） | **+0.6386** | [+0.4828, +0.7949] 不含 0 |
| **B**（`u`） | **+0.0250** | [+0.0095, +0.0418] 不含 0 |
| **g**（`shard_frac`） | **+0.4964** | [+0.4118, +0.5929] 不含 0 |

**`B` 的 CI 第一次不含 0。** EK0 的四个估计（RC-3 `u` p=0.34、b0/b2 跨 0、closeness 维 `B`=−3.5e−4）
全部不可与 0 区分——换掉的是设计，不是数据处理：反事实分叉把同前缀差**测**了出来。

非退化性：谱半径 |a| = 0.6386，可控性 **rank = 1**，5 步 Gramian = 1.047e−03。
**rank 1 是标量状态的构造性上界**，不是缺陷；但 Gramian 量级说明输入通道很弱。

**最好 null 已经含 `shard_frac`**，所以 skill 不是被确定性斜坡买单的——`c_prev` 在斜坡之外仍有增量预测力。

## G-EKA-4 记录项

| 量 | 实测 |
|---|---|
| 相邻两轮回复**完全逐字重复**率 | **0.2644** |
| 相邻两轮回复 token Jaccard 均值 | **0.6051** |
| `Δc_k`（前态已解出，`c_prev ≥ 0.99`） | **0.0**（n=19） |
| `Δc_k`（前态未解出） | +0.0272 |
| `Δc_k` 按轮次 | t1 +0.0333 · t2 −0.0151 · t3 +0.0119 · t4 +0.0183 · t5 +0.0370 · t6 +0.0520 · t7 +0.0662 · t8 +0.2800 · t9 0.0 · t10 0.0 · t11 +0.2308 · t12 +0.4275 |

**echo 是主要失效模态**：四分之一的轮次回复与上一轮**一字不差**，Jaccard 0.61。
`Δc_k` 随轮次单调走高（尾部轮次样本少，不作判定）——reset 在**长对话后段**才有力气。

## 4.1 三个搭车分析的状态

| # | 分析 | 状态 |
|---|---|---|
| 1 | 熵触发规则 vs 反事实真值的 precision/recall | ⏸ **未跑**——`analyze_ergo_entropy_readout.py` 需 teacher-forced 前向，是 GPU 作业（约 1540 行，15–30 min），**待签字** |
| 2 | echo vs drift | ✅ 已出，见 G-EKA-4 |
| 3 | `b₂` 散点图 | ⏸ 数据已在 `branch_pair_gates_report.json` 的 `g_eka_2.scatter`；作图未做（该环境无 matplotlib）。**若进论文需一个仓库内可复现的脚本** |

## G-EKA-2 的失效定位：`closeness` 的众数是**占位答案**，不是状态（2026-09-08，零 GPU）

`closeness = 1/(1 + |a−g| / max(|g|,1))`（`ergo_math_trajectory.py:84`）。

EK-A 全部 1518 行（base + 反事实）上的实测分布：

| `closeness` | 行数 | 占比 | 它是什么 |
|---|---|---|---|
| **0.500** | **949** | **62.5%** | **其中 934 行（98.4%）抽出的答案是 `0`** |
| 1.000 | 272 | 17.9% | 答对 |
| 0.000 | 42 | 2.8% | 抽不出数 / 非数值 |
| 其余 62 个值 | 255 | 16.8% | 真正错但有数的答案 |

**机制**：upstream system prompt 要求模型**每轮都给出 `Current answer: X`，即使还不确定**；
信息不全时 4B 的默认输出就是 `Current answer: 0`。本题库 gold **全部 |g| ≥ 1**，
于是 `|0−g|/max(|g|,1) = 1` → `closeness` **恒等于 0.5**。

两个后果：

1. **没有 x 轴。** `c_{k−1}` 的 IQR = 0，75% 的回归行取值恰为 0.500。G-EKA-2 的斜率不是"测出来接近 0"，
   是**没有变异可回归**。
2. **轴序是错的。** 一个"错但差得不远"的答案（例如偏离 0.5×gold）得 0.667，**高于**"还没有答案"的 0.5。
   `closeness` 在"进展"这个语义维度上**不单调**，所以即使有变异，斜率也不对应前提 (iii) 要问的东西。

### 同一批数据上把状态换成分类变量（诊断，非闸门）

预注册的 G-EKA-2 判在连续斜率上，**下表不改变该判定**，只做失效定位。item 自举 4000 次。

| `c_{k−1}` 类别 | n | mean `Δc_k` | CI |
|---|---|---|---|
| 0.5（占位答案 `0`） | 469 | +0.0253 | [+0.0072, +0.0451] |
| 1.0（已答对） | 19 | **+0.0000** | [+0.0000, +0.0000]（零方差） |
| 0.0（抽取失败） | 25 | +0.0151 | [−0.2100, +0.0877] |
| 其它（真错数） | 111 | +0.0306 | [−0.0063, +0.0733] |

**读法**：唯一可判定的状态依赖是**平凡的**——已答对是两个动作下的吸收态，reset 对它恰好无效
（n=19，零方差，对比 CI [+0.0072, +0.0451] 不含 0）。**在"尚未答对"的三个格子之间，看不出差别。**
MPC 需要的那种非平凡状态依赖（在未解出的状态里，时机会带来差别）**在这份读出上仍然测不到**。

**分叉设计本身没有问题**：同前缀、共同随机数、每轮恰好 2 次 generate，正是它让 `B` 的 CI 第一次
不含 0（G-EKA-3）。**坏的是被回归的那个状态变量。** 按 `.claude/global.md` → *null 结果的默认后果是关闭*
的"同一个仪器只修一次"，**不得再修 `closeness` 本身**（judge/readout 链已修四轮）。

## 提升版 Koopman + 双线性辨识：EK-B 的闭环在这份算子上是退化的（2026-09-08，零 GPU）

**为什么做这一步**（2026-09-08 用户指令：跳过 G-EKA-2 的修补，直接把 Koopman 做对）：
标量模型 `c_next = const + a·c_prev + B·u + g·sf` **在结构上无法表达状态依赖的输入增益**。
而 `B` 若是常数，终端目标下预算 k 的最优解**对每条轨迹都是"最后 k 轮"**——
MPC 退化成固定日程。**闭环能赢开环，当且仅当增益依赖状态**，算子形式就是双线性项 `N`：

`ψ_{t+1} = A ψ_t + B u_t + N ψ_t u_t + c`

反事实分叉数据正是辨识 `N` 的正确数据：每个提升态都各贡献一条 `u=0` 与一条 `u=1` 的转移。

脚本 `scripts/fit_koopman_ergo_branch_lifted.py`（新增，不动任何复现路径），
产物 `outputs/ergo_ekA_branch/koopman_fit_report_lifted.json`。
978 个转移 / 45 题（**turn 1–2 丢弃**：提升态需要前一轮算 echo 坐标）。三个 null 逐字复用，
**每折选一个 null**（不是逐行取 min——那是任何 null 模型都当不成的 oracle），
计分仍在 `closeness` 一步误差上，与 G-EKA-3 / RC-1 / K2 可比。

### 提升坐标（8 维，全部是决策时可得的纯文本统计量，不新增 judge）

| 坐标 | mean | sd | distinct | IQR |
|---|---|---|---|---|
| `closeness` | +0.5403 | 0.1708 | 44 | **0.0000** |
| `placeholder`（答案抽出来是 `0`） | +0.6892 | 0.4628 | 2 | 1.0000 |
| `no_number` | +0.0368 | 0.1883 | 2 | 0.0000 |
| `echo_jaccard`（与上一轮回复） | +0.6511 | 0.3068 | **267** | 0.6458 |
| `echo_exact` | +0.3088 | 0.4620 | 2 | 1.0000 |
| `log_len` | +0.3827 | 0.0405 | 91 | 0.0432 |
| `refusal` | +0.2106 | 0.4078 | 2 | 0.0000 |
| `shard_frac` | +0.5862 | 0.1849 | 21 | 0.3214 |

**提升确实买到了变异**（`echo_jaccard` 267 个不同取值、`placeholder` IQR = 1），
这正是 `closeness` 单独没有的东西。

### held-out 一步 MSE（20 折 item-disjoint）与四个检验

| 模型 | 折均 MSE |
|---|---|
| null `constant` | 0.061924 |
| null `turn_mean` | 0.054754 |
| null `stateless_ols` | 0.037853（19/20 折的胜出 null） |
| `arx_scalar`（EK-A 的算子） | **0.030790** |
| `lift_linear` | 0.032756 |
| `lift_bilinear` | 0.032784 |

| 检验 | 问的是什么 | skill | CI（item 自举） | |
|---|---|---|---|---|
| **T0** `arx_scalar` vs 最好 null | 一步动力学存在吗 | +0.006124 | [+0.000660, +0.013053] | ✅ 不含 0 |
| **T1** `lift_linear` vs 最好 null | 提升空间里还有动力学吗 | +0.004159 | [−0.000123, +0.008739] | ❌ 跨 0 |
| **T2** `lift_bilinear` vs `lift_linear` | **状态依赖增益（前提 (iii)）** | **−0.000028** | **[−0.000381, +0.000276]** | ❌ 跨 0 |
| **T3** `lift_linear` vs `arx_scalar` | 提升买到东西了吗 | −0.001966 | [−0.005187, +0.000581] | ❌ 跨 0 |

**T0 复现了 G-EKA-3**（换了样本仍然过）：**一步动力学是真的**。
**T2 是本节的主结果**：在一个有真实变异的 8 维提升空间里，用平衡的反事实设计直接辨识双线性项，
**状态依赖增益仍然与 0 不可区分**。这不再是"回归没有 x 轴"（G-EKA-2 的失效），
提升坐标有 x 轴——**是耦合本身不在那里。**

### 策略可分性：算子想要的日程对每条轨迹都一样

用拟合出的**全提升算子**（预测整个 `ψ`，`shard_frac` 逐步真值覆盖，与 `ergo_koopman_mpc._simulate` 同款）
穷举每条轨迹在预算 k 下的最优日程（T ≤ 12、k ≤ 3，穷举即 MPC 的 argmax，不是近似）：

| 算子 | k=1 | k=2 | k=3 |
|---|---|---|---|
| `lift_linear` | **1 种日程 / 135 条轨迹**，`[最后 1 轮]`，100% | 1 种，`[最后 2 轮]`，100% | 1 种，`[最后 3 轮]`，100% |
| `lift_bilinear` | 1 种，100% | 1 种，100% | 2 种，modal 97.1% |

**读法**：在这份辨识结果上，**闭环 ≡ 开环**。MPC 臂与固定日程臂会给出同一个动作序列，
EK-B 的 6.1 GPU-小时买不到一个可区分的比较。

**这个判读只有在检查器本身能发现状态依赖时才成立**，因此配了负对照单测
（`tests/test_ergo_branch_lifted.py`，8 条，全过）：植入一个只在 `placeholder` 态上起作用、
且衰减快于被控状态的增益，检查器必须报出 **2 种日程**；不植入则报 1 种。
**校准过程中发现一个陷阱并记录在测试里**：若被调制的状态与被控状态**衰减速率相同**，
两个衰减恰好抵消，任何植入强度都仍然选"最后一轮"——那是植入退化，不是检查器失效。

### 限定（必须与结论同址引用）

1. 本节说的是**"辨识出的算子蕴含固定日程"**，不是"任何控制器都赢不了"。
   拟合看不见的状态依赖，检查器同样看不见。
2. `turn 1–2` 被丢弃，样本从 1248 降到 978；T0 在这个更小的样本上仍然过。
3. `closeness` 的 IQR 在本子集上仍是 0——上一节的读出退化诊断在这里原样成立。

## 熵状态轴：`entropy_mean` 是轮次斜坡的代理，前提 (iii) 仍然不成立（2026-09-08，job 15716425）

**签字来源**：2026-09-08 用户「先执行2」——即 `.claude/global.md` *同一个仪器只修一次* 所要求的
重开熵读出族的显式签字（Q17）。作业 `pdc-ergoEKA-entropy` **COMPLETED 2m41s**
（估计 5–10 分钟，偏保守）。产物 `outputs/ergo_ekA_branch/entropy_readout.json`，759 行 base 轨迹。

### 提交前改了脚本：计划 §4.1 的「不需要改代码」是错的

`analyze_ergo_entropy_readout.py` 的重放有两处会**静默**算错（不报错，只是数不对）：
① upstream profile 下丢掉 system message；② 对 `reset_mode="append"` 的 reset 用 overwrite 语义。
第三处是 rows-path 该指哪个文件：**必须是 base 轨迹**——状态要在 turn k−1 读，那是 base 行；
反事实行是单轮记录、自带 `trajectory_id`、且记在 append 下。

改动：新增 `--prompt-profile`（**默认 `legacy` = 旧行为逐字节不变**，已发布的 phaseB 产物仍可由
裸命令行复现）+ 两道拒绝式守卫；`tests/test_ergo_entropy_replay.py` 7 条单测，含 legacy 回归检验。
干跑中守卫如期挡下了 `counterfactual_pairs.jsonl`。

### 读出量程：熵有量程，`closeness` 没有

| 读出 | mean | sd | distinct | IQR |
|---|---|---|---|---|
| `entropy_mean` | 0.0583 | 0.0556 | **731 / 759** | 0.0777 |
| `entropy_answer_span` | 0.0100 | 0.0335 | 188 | **0.0005** |
| （对照）`closeness` | 0.5403 | 0.1708 | 44 | **0.0000** |

`entropy_answer_span` 75.7% 的行 < 1e−3——**占位答案把答案跨度的熵钉死了**，与上一节预测一致。

### 前提 (iii) 在熵轴上：marginal 过，控住轮次斜坡后死掉

`Δc_k` 是同一批反事实对测出来的一步因果增益，只换回归变量。item 自举 4000 次，
**每个斜率同址给 MDE**（失败模式 29）。

| 回归 | slope | CI | MDE@80% | |
|---|---|---|---|---|
| `Δc ~ entropy_mean` | −0.2453 | [−0.5435, −0.0044] | ±0.3851 | ✅ 勉强不含 0 |
| `Δc ~ entropy_mean + shard_frac` | **−0.0070** | [−0.3313, +0.2457] | ±0.4122 | ❌ **衰减 97%** |
| `Δc ~ entropy_mean + shard_frac + c_prev` | +0.0010 | [−0.3304, +0.2651] | ±0.4254 | ❌ 衰减 100% |
| `Δc ~ entropy_answer_span` | −0.5551 | [−0.8132, −0.2829] | ±0.3788 | ✅ |
| `Δc ~ entropy_answer_span + shard_frac` | −0.3527 | [−0.5952, −0.0645] | ±0.3791 | ✅ 衰减 36% |
| **`Δc ~ shard_frac`（确定性斜坡本身）** | **+0.1161** | **[+0.0503, +0.1796]** | **±0.0924** | ✅ **效应大于 MDE** |

`corr(entropy_mean, shard_frac) = −0.5162`。分位表把机制摆明了：

| entropy_mean 分位 | n | mean `Δc` | mean `shard_frac` |
|---|---|---|---|
| Q1（最低熵） | 156 | +0.0522 | 0.773 |
| Q2 | 156 | +0.0225 | 0.763 |
| Q3 | 156 | +0.0114 | 0.703 |
| Q4（最高熵） | 156 | +0.0141 | 0.464 |

**熵的排序就是轮次的排序。** 唯一稳健地预测"reset 什么时候有用"的量是 `shard_frac`——
确定性、外生、开场前就已知，**而且它已经在每个 null 和算子里**（`g` = +0.4964）。
这正是 `koopman-judge-dynamics-not-prediction` 第 (d) 条要求的检验：
**null 必须拿到模型拿到的每一个确定性外生量**。熵轴过不了这一关。

### 一条值得单独记下的观察（方向与 ERGO 的触发规则相反）

`entropy_answer_span` 控住斜坡后仍留下信号，且**不是少数离群点**（去掉回归量最高的 1%，
斜率 −0.5551 → −0.5425，只掉 7 行）。按是否犹豫分组：

| 组 | n | mean `Δc` | CI | mean `shard_frac` |
|---|---|---|---|---|
| 答案跨度熵 ~0（已经笃定一个数） | 458 | **+0.0341** | [+0.0140, +0.0563] | 0.749 |
| 答案跨度熵 >1e−3（还在犹豫） | 147 | **−0.0071** | [−0.0235, +0.0091] | 0.461 |

**reset 在模型已经笃定时有用，在它犹豫时没用——与 ERGO「熵尖峰触发 reset」正好相反。**
这是上游从未做过的那个消融（触发时机 vs 重写 prompt）第一次有了逐轮反事实真值。

**限定，必须同址引用**：斜率 −0.3527 仍**小于本设计的 MDE ±0.3791**，
低功效下的显著估计有系统性夸大风险；两组的 `shard_frac` 也仍差 0.75 vs 0.46。
**这是讨论段级别的观察，不是可以写进结果表的主张。**

### 算子层面：加了熵之后仍然没有双线性耦合

| 检验 | 无熵 | 加熵 |
|---|---|---|
| T1 `lift_linear` vs 最好 null | +0.004159 跨 0 | +0.003960 跨 0 |
| **T2 `lift_bilinear` vs `lift_linear`** | −0.000028 跨 0 | **−0.000121 跨 0** |
| T3 `lift_linear` vs `arx_scalar` | −0.001966 跨 0 | −0.002165 跨 0 |

策略可分性：`lift_linear` 在 k=1/2/3 仍是**1 种日程 / 100%**；`lift_bilinear` 在 k=3 从 2 种变 4 种
（modal 75.2%），但那是熵坐标带进来的噪声，不是信号——它在 T2 上买不到任何预测力。

**结论**：熵轴不是救兵。按 LEDGER 登记的预注册后果——**不过 → ERGO 停在 S3，预算转 `constraint` 线**。
