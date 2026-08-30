# 实验记录：人格漂移功效确认 pilot（10 prompt 规模）

记录这次跑在 Palmetto 2 上的作业本身的状态，供换一次新对话（没有本会话上下文）时能接得上——
和 [signal_screening_pilot.md](signal_screening_pilot.md) 是同一类文档，但这是**另一个、更晚的
实验**：signal_screening_pilot.md 记录的是协议第 7 节要求的 5-prompt gate screening（已完成，
`overall_pass=False`）；这份文档记录的是那次 screening 三问全挂之后，为了判断"到底是真的没有
漂移，还是样本量太小测不出来"而做的功效放大 pilot。

## 这是什么实验，为什么要做

`signal_screening_pilot.md` 的 5-prompt screening 三问全挂（Q1/Q2/Q3 都不过）。排查发现：

1. 5 个 prompt 里 2 个是打分函数饱和的（`character_traits_013`、`language_constraints_004`，
   已经在 `prompt_bank.KNOWN_SATURATED_PROMPT_IDS` 里排除）。
2. 排除饱和 prompt 后，剩下 3 个"健康" prompt 的独立样本量太小——用全序列 OLS 斜率重新分析
   （而不是协议原来的 turn1-vs-turn16 端点比较），6 条轨迹里 5 条斜率同向为负，以 **prompt**
   为独立单位（而非轨迹，因为同一 prompt 的两个 seed 不是独立重复）做单样本 t 检验，
   t≈-2.49, df=5, p≈0.056（双尾）/ 0.028（单尾）——卡在统计边界上，证据方向上像是有一个很小
   的真实漂移，但样本量（只有 3 个独立 prompt）太小，不足以下结论。

结论：**当前证据不足以判断"16 轮内 Qwen3-4B 是否存在可测的人格漂移"**，需要把独立 prompt
数量从 3 提升到能给出确定结论的量级（本次目标 ~10 个 prompt，作为分阶段验证的第一步，而不是
一步到位的 20 个——见下方"规模决策"）。这个结论本身也决定了整个 Koopman 控制方向值不值得往
下做：如果漂移/控制效应在更大样本下依然测不出来，说明协议前提站不住，需要换方向而不是继续
往 320 条正式采集里投入算力。

## 排查/准备阶段做的工程改动（都在 GPU 作业开始前完成并跑过 CPU 测试）

1. **离线打分函数安全性预筛**（零 GPU 成本）：`prompt_bank.classify_scorer_screening_safety`
   用一批多样化的"典型回复"字符串离线跑一遍每个候选 prompt 的 `score_fn`，标出结构性二值
   （`binary_across_battery`）或超出 [0,1] 范围（`out_of_unit_range`）的候选项。
   `select_screening_prompts` 默认（`avoid_unsafe_scorers=True`）优先从"ok"标记的池子里采样。
   `scripts/audit_prompt_scorers.py` 是可以随时重新跑的诊断脚本。
   - **已知局限**：这个启发式对"检查特定小词表命中比例"类打分函数（比如
     `character_traits_006` 的 teenager_score、`character_traits_009` 的 cat/meow-purr）会
     误判成二值型——因为诊断字符串库里没有覆盖这些特定词汇，导致这些本来连续的打分函数在
     整个电池上都算出常数 0.0。已知这两个被误伤（006 已经从真实 GPU 数据确认是连续的），
     但因为"ok"池子本身已经够 10 个 prompt 用，这次没有再手动纠正，接受这个保守的效率损失。
   - `analyze_screening` 也新增了 `diagnostics.saturated_prompt_ids`：按 `system_prompt_id`
     分组，本次运行内 `y_probe` 方差恒为 0 的 prompt 自动列出来，作为运行后的事后防御网。
2. **断点续跑**（`screening._prepare_resumable_trajectories_file` +
   `run_screening` 里 `trajectories.jsonl` 改成追加写）：作业被 `--time` 杀掉后，重新提交
   同一个 sbatch 脚本会跳过已经完整写入的 trajectory_id，只补跑剩下的，不用重新猜一个
   "绝对够用"的 `--time`（这是 `signal_screening_pilot.md` 里第二次提交因为 `--time` 估
   不准、主动取消两次的教训）。
3. **Q1 的补充统计**（`analysis.py` 新增 `q1_drift_trend`，不改 `q1_drift_exists` 的判定
   逻辑）：每条 `zero_control` 轨迹用全部 16 个点做 OLS 斜率（而不是只比较 turn1/turn16 两个
   端点），按 **prompt** 聚合（同一 prompt 的不同 seed 取平均斜率）后做符号统计和单样本 t
   检验。这是这次能不能"确认"漂移的关键读数，不是原来的 Q1 pass/fail。
4. **scripted-user 方案：尝试过，已放弃**（见下方"历史"）。

## 规模决策：为什么是 10 个 prompt 而不是一步到位 20 个

原计划是直接冲到 20 个 prompt（40→80 条轨迹），但考虑到断点续跑、离线预筛、
scripted-user 都是这次话新加的机制，一次性投入大规模 GPU 时间去验证一套没实跑过的新流程
风险较高，所以先用 10 个 prompt（40 条轨迹）跑一遍，确认新机制稳定之后再决定要不要扩到 20。

## 当前作业

| 字段 | 值 |
|---|---|
| Job ID | **15393201**（2026-08-30 提交，无排队立刻开始跑） |
| 提交脚本 | `persona_drift_control/environment/run_drift_confirmation_pilot.sbatch` |
| 规模 | 10 prompts（`select_screening_prompts` 默认设置，从"打分函数安全"的池子分层采样）
  × 2 seeds × 2 条件（`zero_control`/`excite_iid`）= **40 条轨迹** |
| user_mode | `live`（scripted 已放弃，见下方历史） |
| 申请时长 | `--time 08:00:00`（按第一次 20 条轨迹跑 10944s≈547s/条的均速估算，40 条约
  21,880s≈6.1h，留了约 30% 余量；如果不够，断点续跑机制支持重新提交同一脚本接着跑，不用
  重新猜时间） |
| 输出目录 | `persona_drift_control/outputs/drift_confirmation_pilot/` |

## 查看状态的方法

```bash
squeue -j 15393201
tail -f persona_drift_control/environment/slurm_logs/drift-confirmation-15393201.out
ls persona_drift_control/logs/                       # 找到这次运行对应的 loguru 文件
wc -l persona_drift_control/outputs/drift_confirmation_pilot/trajectories.jsonl   # 40 x 16 = 640 行才算完整跑完
cat persona_drift_control/outputs/drift_confirmation_pilot/screening_report.md   # 跑完之后才会有；重点看 q1_drift_trend 和 q1_drift_exists 两节
```

如果作业被 `--time` 杀掉：直接 `sbatch environment/run_drift_confirmation_pilot.sbatch` 重新
提交同一个脚本（不要改 `--output-dir`），`_prepare_resumable_trajectories_file` 会自动跳过
已经完整写入的轨迹，只补跑剩下的。

## 历史：scripted-user 方案，尝试后放弃

`docs/SCRIPTED_USER_TURNS_FEASIBILITY.md`（分析文档，2026-08-30）提出用离线预生成的用户轮次
脚本替代每轮实时调用的 `user_sim`，边际收益约 17%（每轮生成次数 6→5）外加省掉加载第二个模型
的开销。按该文档第 5 节的验证顺序实施：

1. 实现 `TrajectoryConfig.user_mode: Literal["live","scripted"]`、`user_scripts.py`
   （脚本加载）、`scripts/generate_user_scripts.py`（离线生成：每个话题先跑一次真实的
   agent+user 自聊拿到参考 agent 回复序列，再用这个固定参考序列驱动生成每个 seed 的用户
   脚本）。全部改动都有 CPU 单测覆盖（`test_selfchat.py` 用 fake ChatModel 桩，不需要 GPU）。
2. **第一次生成**（job 15392309，12 话题×4 seed×16 轮，约 74 分钟）：人工抽查 3 个话题
   （`public-transportation`、`weekend-hiking`、`electric-vehicles`）的脚本，**全部**在
   第 10-13 轮左右退化成逐字重复的段落（比如同一段关于芝加哥 CTA 交通的话重复 8 轮）。
   根因：长（16 轮）、无真实反馈闭环（用户脚本生成时用的是固定参考 agent 序列，不是真实
   反应）的自聊，是 LLM 自聊的已知病理场景，且 `ChatModel.generate` 当时没有配置任何抗
   重复参数。
3. **修复尝试**：给 `GenerationConfig` 新增 `repetition_penalty`/`no_repeat_ngram_size`
   字段（默认是 transformers 的 no-op 值 1.0/0，不影响任何已验证过的 live 流程行为），
   只在 `generate_user_scripts.py` 里设成 1.15/4。**第二次生成**（job 15392875，同样规模，
   约 73 分钟）：逐字重复问题确实解决了，但暴露了另一种退化——被强制不能重复用词后，
   模型在长自聊里转向了语气逐渐失控的独白（`electric-vehicles` 从第 10 轮开始变成"加油稿"
   式的夸张鼓励；`public-transportation` 更严重，第 12 轮后完全脱离"公交"话题，变成
   "my dearest soul...sacred love..."这种准情话/灵性独白，违反了
   `USER_SYSTEM_PROMPT_TEMPLATE` 里"保持随意、扣题"的设定）。
4. **判断：放弃**。两轮生成都在质量检验（`SCRIPTED_USER_TURNS_FEASIBILITY.md` 第 5 节
   步骤 2"确认不是退化成高度重复的文本"）上失败，且第二次的失败模式（语气失控/脱离话题）
   比第一次（逐字重复）更难被自动化检测抓到，对可信度的潜在杀伤力更大。根因是"固定参考
   agent 脚手架 + 无真实反馈闭环"这个方法本身的结构性问题，不是靠调 decoding 参数能修好的
   ——按该文档自己写的规则（"任一步不达标，都应该停在这里，继续用活的 user_sim"）执行，
   回退到 `live` 模式做本次主实验。
   `resources/user_scripts/` 里的脚本文件和 `user_mode="scripted"` 代码路径都保留在代码库
   里（有 CPU 单测覆盖），供以后想换一种脚手架方法重新尝试时参考，但不会被本次或默认流程
   使用（`TrajectoryConfig.user_mode` 默认值是 `"live"`）。

## 结论（2026-08-30，作业已完成）

Job 15393201 正常跑完，40/40 条轨迹，总耗时 16585s（≈4.6h），未触发 `--time 08:00:00`，
断点续跑机制这次没用上（一次过）。`saturated_prompt_ids` 为空——离线预筛的安全池这次没有
再让任何饱和 prompt 混进来，之前 40% 算力浪费的问题解决了。

**overall_pass = False，但这次的关键读数是 `q1_drift_trend`，而不是端点法的 `q1_drift_exists`：**

- **Q1 补充统计（10 个独立 prompt，全序列 OLS 斜率）：5/10 正、5/10 负，一分不差的对半开；
  t=0.0166，p=0.9871。** 这是这次 pilot 最重要的发现——上一轮只有 3 个独立 prompt 时，
  5/6 条轨迹同向、t≈-2.49、p≈0.056，看起来像是有一个小的真实漂移信号只是功效不够；
  这次把独立 prompt 数从 3 提到 10（超过 3 倍），符号分布从"5:1 明显偏向一边"直接变成
  "5:5 完全打平"，t 统计量从 -2.49 掉到几乎为零。**这是小样本噪声被更大样本"打回原形"
  的典型例子**——之前的边界显著性大概率就是运气，不是真实效应的早期信号。
- **Q1 端点法**：mean_drop=0.0154 vs 阈值 0.0501，比值 0.0154/0.0501≈0.31，比第一次
  pilot 的比值（0.019/0.032≈0.59，健康 prompt 单独算是 0.0317/0.0533≈0.59）明显更低——
  样本越大，端点法测出的"漂移"占测量噪声阈值的比例反而越小，方向上和 `q1_drift_trend`
  的结论一致：更大样本没有让漂移信号变得更清楚，而是让它更接近于零。
- **Q2**：diff=0.0063（方向转正，比第一次 pilot 的 -0.0058/-0.0097 好看一点），但只有阈值
  0.0647 的约 10%，n_pairs=300（这次数据量很大，不是功效问题）——提醒信号在下一轮探针
  分数上仍然测不出统计意义上的效应。
- **Q3**：斜率=-0.0026，p=0.9541，n_pairs=280——滞后 2 轮同样测不出任何关系。
- 诊断：refusal_rate 0.63%，scorer_failure_rate 0%（生成流程健康），
  y_probe 均值 character_traits=0.4456 (sd 0.35)，language_constraints=0.5628 (sd 0.38)，
  和第一次 pilot 的量级基本一致。

## 这次证据强度怎么判断

和第一次 pilot 时"证据不足以下结论,需要更大样本"不同，**这次的证据强度已经足够支撑一个
相对有把握的判断**：

- 10 个独立 prompt（不是轨迹，是真正不同的 system prompt/probe/scorer）的漂移方向对半开，
  t 检验 p=0.99，是"这批数据里找不到任何一致方向的漂移"能给出的最干净的空结果之一——不是
  "勉强不显著"，是完全没有倾向性。
  这仍然只能说是"在这套设计下（16 轮、Qwen3-4B、这批 persona/pattern 类 system prompt、
  K=4 探针重复）测不到"，不是"任何情况下都不存在"的全称结论——更长对话、更强的诱导手段、
  不同模型规模，都还是没有排除的可能性，但就"当前协议设想的实验设计"而言，已经有比较
  扎实的证据说明它测不到预期的信号。
- Q2/Q3 在 n_pairs=280-300（比第一次 pilot 的 n=140-150 又翻了一倍）的规模下依然测不出
  提醒信号的因果效应，且效应量本身（diff 只有阈值的 10%）看不出"差一点就显著"的迹象，
  不像是继续加大样本就会转为显著的情况。

## 对研究方向的含义（留给下一步决定）

`DATA_COLLECTION_PROTOCOL.md` 设想的"用提醒信号纠正可测的人格漂移，再拿 Koopman/ARX 建模"
这条路径，目前两个前提（有漂移可测、提醒信号有可测的因果效应）在 10-prompt 规模下都没有
得到支持。是否要：(a) 认为协议前提不成立，转向别的诱导/测量设计（更强的输入、更长的对话、
换一批更容易漂移的 system prompt 类别）；(b) 认为 Qwen3-4B 在这个规模下确实相当稳定，
把"稳定性"本身作为一个（不同方向的）研究发现；(c) 换一个更大/更容易漂移的模型再测一遍——
这是需要往下讨论的决定，不是这份 pilot 本身能回答的。
