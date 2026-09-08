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
