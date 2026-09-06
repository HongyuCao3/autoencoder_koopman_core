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

## 结果（job 15602880 zero_control / 15602881 reset，20 items × 2 seeds，尚未跑完）

作业已提交，尚未归档结果——下一次接续这条线时先检查这两个作业是否跑完
（`sacct -j 15602880,15602881`），跑完后用 `scripts/analyze_ergo_authority_comparison.py`
做配对比较，把结果和结论补写进这一节。

## 下一步

1. **等两个作业跑完，做配对比较，判定执行器权威**：这是这份文档存在的唯一理由，其余步骤
   都要等这一步的结果。
2. 权威确认 → 按可行性文档第 4 节步骤 3-4：用同一批零控制轨迹做 `new_q3_autocorrelation`
   风格的惯性检验（已经在 `analysis_ergo_math.py` 里实现,只需要真实数据),确认后再决定要不要
   为路径 B 的其余任务/更精细读出投入工程。
3. 权威不确认 → 回到 `KOOPMAN_MECHANISM_AND_TRANSFER_ANALYSIS.md` 第五节，重议 context
   equilibria（当前排序里机制最相似但尚未评估执行器权威的候选）或其他任务，而不是像 sycophancy
   线那样在同一个执行器上换第二版设计——这次的执行器（重置）已经比 sycophancy 的提醒文案更贴近
   "已被外部证实有效"，如果连这个都测不出来，更应该怀疑是本项目自己模型/简化重置动作的问题，
   而不是继续在这个执行器的变体上试错。
