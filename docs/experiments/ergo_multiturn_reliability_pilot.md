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

## 样本扩充（job 15603367 zero_control / 15603368 reset，60 items × 2 seeds，已提交，2026-09-06）

20-item 最小验证已经确认权威（见上节），这一步不是重新测"有没有权威"，是把效应量估计打扎实、
给 `analysis_ergo_math.py` 的 new-Q1/new-Q3 诊断更多功效——和 sycophancy 线 20→60 items 的
扩样本同一个理由。**新抽的 60 items**（`--item-rng-seed 1`，不是复用 pilot 的那 20 个，题库
总共只有 103 个 item，两次抽样会有交集但不是子集关系），两个 sbatch 用完全相同的 60 个
item id，60 items × 2 seeds = 684 行，作业已提交，尚未跑完。下一次接续时先
`sacct -j 15603367,15603368` 确认完成，再用 `scripts/analyze_ergo_authority_comparison.py`
（默认参数指向 pilot 的 20-item 目录，这次需要 `--arm-a-dir outputs/ergo_math_authority_
zero_control_60items --arm-b-dir outputs/ergo_math_authority_reset_60items`）算配对比较，
补写进这一节。

## 下一步

1. **等 60-item 扩充作业跑完，确认效应量在更大样本下依然稳**：这是当前最直接的下一步。
2. **执行器权威已确认（20-item 结果）**：这是这条线和 sycophancy 线最大的区别——不需要
   再纠结"要不要继续试错换执行器设计"这个问题了，可以往下走。
3. **惯性/动力学定性需要重新表述**，见上面加粗的段落——不是"记忆黏滞"而是"信息累积",这个
   区别要带进任何后续文档/论文措辞，避免和 sycophancy/防御线的"惯性"概念混为一谈。
4. **暂不急于铺开到其余五个 ERGO 任务**（code/SQL/actions/data2text/summary）——那些的评分器
   工程量更大（`PROVENANCE.md` 的"已知简化"一节），且数学任务本身已经足够回答"权威在不在"
   这个问题；扩样本确认稳定后再决定是否值得为其余任务投入工程。
5. **下一次真正投入建模前，还有可行性文档第 4 节没走完的部分**：a) 决定要不要为更多任务投入
   路径 B 评分器工程；b) 想清楚"信息累积"这种动力学形态下，Koopman 状态空间怎么设计最自然
   （e.g. 状态是否应该显式包含"已揭示多少个 shard"这个确定性协变量，而不是像 sycophancy 那样
   只用 y 的滞后项）；c) 论文叙事层面要不要把这条线定位成"和 sycophancy 形成对照的正面案例"
   （`docs/article/PAPER_EXECUTION_PLAN.md` §1.4 需要相应更新，目前完全没提这条线）。
