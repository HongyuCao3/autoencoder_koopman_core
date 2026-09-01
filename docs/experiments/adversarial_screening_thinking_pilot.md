# 实验记录：adversarial_screening 的 enable_thinking 复现性重跑

和 [adversarial_screening_pilot.md](adversarial_screening_pilot.md) 同一类"供跨会话接续"的记录。
**新开一次对话想知道"这个 thinking 复现重跑现在跑到哪一步了"，看这份文档。**

## 这是什么，为什么做

审计发现全仓库所有实验（`ChatModel.__init__` 默认 `enable_thinking=False`，从未被任何调用点覆盖）
一直跑在 Qwen3 的 **non-thinking 模式**下，这个变量从未被显式检视或记录过。逐个实验分析后，判断
`adversarial_screening_pilot.md`（jailbreak 渐进侵蚀 screening，job 15399715，new-Q1 p=0.0000、
new-Q3 p=0.0000）是当前唯一值得优先做 thinking 模式复现重跑的实验：

- 这是当前项目最高优先级路径（`ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md`，ICLR 2027 摘要
  09-18/全文 09-25 截止）里唯一已经拿到强信号的核心结果，任何要写进论文的发现都应该核实是否
  在 thinking 模式下复现——Qwen3 本身以 thinking/non-thinking 双模切换为卖点，不测这个变量
  会显得刻意回避一个审稿人几乎必然会问的问题。
- thinking 模式和 jailbreak 鲁棒性直接相关的两派文献预测方向相反（deliberative alignment 认为
  推理链能重新锚定拒绝；chain-of-thought hijacking 类工作认为推理链本身是新的攻击面），是一个
  有具体科学问题在等答案的变量，不是可有可无的 ablation。
- 成本低、风险可控：这个 job 只有 40 条轨迹、非 thinking 版本实测 31 分钟，且已有确定的
  non-thinking 基线可对比，可以完全独立并行，不干扰当前正在调试的 `dose_response_pilot.md`
  （channel C steering，仍在排查 p=0.0563/0.4535 不显著的根因，thinking 模式会给它引入新的
  混淆变量，明确不建议现在碰，见下方"没有做的部分"）。

判定方法和基线完全一致：`analysis_adversarial.analyze_adversarial_screening` 的 new-Q1（渐进
侵蚀，t 检验）/new-Q3（自相关）。

## 用 Hydra 管理 enable_thinking，避免结果/日志互相覆盖

新增 `persona_drift_control/conf/`（`hydra-core` 加入 `pyproject.toml` 依赖）：

- `conf/adversarial_screening.yaml`：顶层配置，`defaults: [generation: no_thinking, _self_]`，
  `hydra.run.dir: ${output_root}/${generation.name}/${now:%Y%m%d_%H%M%S}`——目录名同时编码了
  `generation` 分组名（`no_thinking`/`thinking`）和时间戳，即使有人忘了手动传不同的
  `--output-dir`，thinking 和 non-thinking 两次跑的 `trajectories.jsonl`/report 也不可能落到
  同一个目录（这正是旧的 argparse CLI `scripts/run_adversarial_screening.py` 没有内建保护的
  失效模式）。`hydra.job.chdir: false`——保留项目其它脚本"输出路径相对于进程原始 CWD"的既有
  约定，不用 Hydra 默认的按运行改变 CWD 行为。
- `conf/generation/no_thinking.yaml`：`enable_thinking: false`, `agent_max_new_tokens: 256`
  ——和过去所有作业的默认值字节级一致，是 Hydra 的默认分组成员，不传 `generation=...` 就完全
  复现已验证过的 non-thinking 基线（job 15399715）。
- `conf/generation/thinking.yaml`：`enable_thinking: true`, `agent_max_new_tokens: 1536`
  ——token 预算从 256 提到 1536 的原因见下方"max_new_tokens 为什么要单独调"。
- `src/persona_drift/adversarial_screening.py::run_adversarial_screening()` 新增
  `enable_thinking: bool = False` 参数（默认值保证旧调用点/旧 argparse 脚本行为不变），
  写入 `run_config`（`configure_run_logger` 第一行就会记录），并且 `run_id` 从
  `f"{output_dir.name}_{timestamp}"` 改成 `f"{output_dir.name}_think{int(enable_thinking)}_{timestamp}"`
  ——`logs/` 是一个跟 `output_dir` 分开的扁平目录，只靠时间戳理论上也不会撞（两次跑不可能在
  同一秒开始），但显式把 `enable_thinking` 编进文件名让日志本身自解释，不需要打开文件看第一行
  配置才知道是哪次跑的。
- `scripts/run_adversarial_screening_hydra.py`：新的 Hydra 入口（`@hydra.main`），旧的
  `scripts/run_adversarial_screening.py`（argparse）**不改动**——它默认行为不变，仍然是
  non-thinking 基线的规范复现方式,两个脚本共享同一个 `run_adversarial_screening()` 编排函数，
  没有逻辑重复。
- `environment/run_adversarial_screening_thinking.sbatch`：新 sbatch，调
  `run_adversarial_screening_hydra.py generation=thinking`。**注意**：因为
  `hydra.run.dir` 用了 `${now:...}`，被 `--time` 杀掉后不能像其它脚本那样直接
  `sbatch` 同一脚本重跑（会生成一个新的空时间戳目录，"续跑"逻辑会跑在空目录上，不会真的接上
  断点）——必须把上次的目录路径当 `$1` 传给这个 sbatch 脚本，它会转成
  `hydra.run.dir=<path>` 覆盖项传给 Hydra，见脚本内注释。

## 两个必须同时解决的正确性问题（不只是加一个 flag）

审计阶段就预判到、这次实现时确认必须处理的两处，否则 thinking 模式下的数据不可信：

### 1. judge 必须被钉死在 non-thinking，即使和 agent 共享同一个模型实例

`adversarial_screening.py` 默认 `judge_model_id == agent_model_id` 时判官和 agent 复用同一个
`ChatModel` 实例（省一次模型加载）。如果 `enable_thinking` 是纯粹的实例级属性，agent 打开
thinking 后判官的调用也会被迫打开——但判官的 prompt 只要求"回复一个 1-5 的数字"，
`max_new_tokens=10`，完全没有空间容纳一段推理，一旦打开 thinking，判官几乎必然在推理过程中被
截断、从不产出数字，`judge_parse_failure_rate` 会从当前的 `0.0000` 直接崩掉。

修复：`ChatModel.generate()` 新增按次覆盖参数 `enable_thinking: bool | None = None`
（`None` = 用实例默认值），`safety_judge.judge_safety_score()` 显式传
`enable_thinking=False`——判官作为一个固定的测量工具，不应该随被操纵的自变量（agent 是否
thinking）一起变化，这本身也是更干净的实验设计（避免"判官自己是否推理"变成第二个混淆变量）。
`tests/test_safety_judge.py::test_judge_safety_score_always_pins_enable_thinking_false` 和
`tests/test_attack_trajectory.py`（更新后的断言）覆盖了这一点。

### 2. `<think>...</think>` 块必须在 token 层面被正确切开，不能留给下游文本处理

Qwen3 tokenizer 把 `</think>` 注册成特殊 token，`skip_special_tokens=True` 解码时会**去掉
`<think>`/`</think>` 这两个标签本身，但不会去掉标签之间的推理内容**——如果不显式处理，
`agent.generate()` 返回的字符串会是"推理内容 + 最终答案"直接拼接、没有任何分隔符,
`judge_safety_score`/`looks_like_refusal` 会把推理过程当成"agent 的实际回复"来打分，和
non-thinking 基线的语义完全不可比。

修复（`chat_model.py`）：
- `_split_at_think_end(token_ids, think_end_id)`——纯函数，在 token id 层面（不是文本正则）
  找到 `</think>` 对应的 token id 做切分，`think_end_id` 缺失或未出现时整体归为 content
  （覆盖 `enable_thinking=False` 和非 Qwen3 tokenizer 两种情况，向后完全兼容）。不依赖
  torch/tokenizer，`tests/test_chat_model.py` 里 4 个用例纯 CPU 覆盖。
- `ChatModel.generate()` 新增 `return_thinking: bool = False`，默认关闭时返回类型/内容和
  之前完全一致（`str`，只是 content 部分）；`True` 时返回 `(content, thinking_text)`。
- `attack_trajectory.py::run_attack_trajectory()` 用 `return_thinking=True` 拿到
  `agent_thinking`，写入 `agent_thinking` 这一新列（推理轨迹本身，供审计/未来分析，
  不参与判分）；写回 `agent_history` 的仍然只有 `agent_text`（最终答案，不含推理）——和真实
  多轮对话客户端调用 thinking 模型的方式一致（历史里不带上一轮的推理块）。

## max_new_tokens 为什么要单独调

`AttackTrajectoryConfig.agent_gen` 基线默认 `max_new_tokens=256`。thinking 模式下如果不改这
个预算，`<think>` 推理块很容易把 256 个 token 全部占满，模型还没写到最终答案就被截断——
`agent_message` 会是空字符串或半截推理文本，`judge_safety_score`/`looks_like_refusal` 全部
失真。`conf/generation/thinking.yaml` 把它提到 `1536`，这是一个起始估计，不是扫描过的最优值；
这次真实 GPU 跑完后要看的第一个诊断指标就是"有多大比例的行里 `agent_thinking` 非空且
`agent_message` 非空"（即：在预算内正常结束了推理并给出了答案），如果这个比例不高，
下一步就是继续加大这个数字，而不是怀疑代码逻辑。

## 没有做的部分（这次故意跳过，不是漏做）

- **`dose_response_pilot.md`（channel C 安全方向 steering）不在这次改动范围内**——那条线正在
  排查 new-Q2 两次都不显著（p=0.0563/0.4535）的根因（校准点/应用点不匹配、单层干预强度不够
  等假设还没验证完），此时引入 thinking 模式会带来一个新的、独立的混淆变量
  （`hidden_state_at_layer()` 在 thinking 模式下该在哪个 token 位置取激活——`<think>` 块内、
  `</think>` 之后，还是整个序列最后一个 token——这个问题本身没有唯一答案），会让本已模糊的
  失败原因更难拆解。应该先在 non-thinking 里把根因定位清楚（或判定 channel C 本身不可行），
  再决定要不要在 thinking 模式下重新校准方向。因此 `activation_direction.py`/
  `chat_model.hidden_state_at_layer()` 这次都没有加 `enable_thinking` 覆盖参数，避免在
  还没用到的地方引入未测试的表面积。
- **`pressure_screening_pilot.md`（人格域渐进施压）这次没有跟着改**——它的结果本身还没跑出来
  （job 15406535），基线结论不明确之前不应该抢先追加 thinking 版本的算力投入，等它有结论后
  再单独评估是否需要类似的 thinking 复现重跑。
- **`drift_confirmation_pilot.md`/`signal_screening_pilot.md`（被动 reminder 刺激的 null
  结果）没有安排 thinking 重跑**——空结果已经被诊断为"刺激太弱"而非"模型/训练问题"，开
  thinking 预期信息增量很低，优先级明显低于这次的对抗任务重跑。
- 旧的 argparse CLI `scripts/run_adversarial_screening.py` 没有加 `--enable-thinking` 参数
  ——它的默认行为（non-thinking）没有变化,不需要跟着改；这次的重跑走新的 Hydra 入口。

## 代码改动清单

- `pyproject.toml`：新增依赖 `hydra-core>=1.3`。
- `persona_drift_control/conf/adversarial_screening.yaml`（新）、
  `conf/generation/no_thinking.yaml`（新）、`conf/generation/thinking.yaml`（新）。
- `src/persona_drift/chat_model.py`：新增纯函数 `_split_at_think_end`；`generate()` 新增
  `enable_thinking`/`return_thinking` 参数（见上）。
- `src/persona_drift/safety_judge.py`：`judge_safety_score()` 调用 `judge.generate()` 时
  显式传 `enable_thinking=False`。
- `src/persona_drift/attack_trajectory.py`：agent 调用改用 `return_thinking=True`，行 schema
  新增 `agent_thinking` 列。
- `src/persona_drift/adversarial_screening.py`：`run_adversarial_screening()` 新增
  `enable_thinking: bool = False` 参数，写入 `run_config`，`run_id` 编码
  `think{0,1}`；agent `ChatModel` 构造时传入该参数，judge 单独实例构造时显式传
  `enable_thinking=False`（即便这次一定会被 per-call 覆盖，写出来是为了让代码本身表达意图）。
- `scripts/run_adversarial_screening_hydra.py`（新）。
- `environment/run_adversarial_screening_thinking.sbatch`（新）。
- 测试：`tests/test_chat_model.py`（新增 4 个 `_split_at_think_end` 用例）、
  `tests/test_safety_judge.py`（`FakeJudge`/`capturing_generate` 更新签名 + 新增 pin 检验）、
  `tests/test_attack_trajectory.py`（`FakeChatModel` 更新签名，新增 `agent_thinking`/
  `enable_thinking_calls` 断言）、`tests/test_adversarial_screening.py`（新，CPU-only，
  monkeypatch `ChatModel`，验证 `enable_thinking` 真的被传进 agent 的构造函数、`run_id`/
  `report["config"]` 都正确编码了这个值）、`tests/test_hydra_config.py`（新，纯配置合成测试,
  不依赖 `@hydra.main` 入口，覆盖"忘了改 YAML 导致默认组/覆盖组失效"这类问题）。

## 当前状态

**CPU 单测已通过，GPU 重跑作业已完成，new-Q1/new-Q3 均复现通过。**

- CPU 单测：第一次提交（job 15408733）暴露了两处遗漏——`tests/test_dose_response.py` 里
  独立定义的另一个 `FakeChatModel`（`dose_response.py` 内部同样调用
  `safety_judge.judge_safety_score`，但当时只更新了 `test_attack_trajectory.py`/
  `test_safety_judge.py` 里的 Fake，漏了这一个）不接受新加的 `enable_thinking` 关键字参数，
  `TypeError` 在 `judge_safety_score` 内部报错；已修复（`generate()` 签名加上
  `enable_thinking=None, return_thinking=False`）。同批还看到一次
  `test_logging_setup.py` 的失败，是 docs 里多次记录过的既有 flaky 测试（loguru
  `enqueue=True` 异步写入竞态），跟这次改动无关。修复后重新提交（job 15408755）：
  **125 passed**，包括新增的 `test_hydra_config.py`（确认 `hydra-core` 在集群环境里装得上、
  两个 `generation` 分组组合出的配置值都对）——这次那个 flaky 测试也没有再触发。
- GPU 作业：`environment/run_adversarial_screening_thinking.sbatch`（`generation=thinking`，
  20 攻击 × 2 seeds = 40 条轨迹，和 non-thinking 基线 job 15399715 完全同规模同种子）已提交,
  见下方 job ID。

## 查看状态的方法

```bash
squeue --me
sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode
cat persona_drift_control/outputs/adversarial_screening_thinking_ablation/thinking/<timestamp>/adversarial_screening_report.md   # 跑完才有

# 拿到报告后先看这两个诊断指标，再看 new-Q1/new-Q3 pass/fail：
#   - judge_parse_failure_rate 是否依然接近 0（判官钉死 non-thinking 是否生效）
#   - 抽查几行 agent_thinking 是否非空、agent_message 是否看起来完整（不是被截断的推理残段）
```

## 结果（job 15410124，2026-08-31，2h36m）

`sbatch environment/run_adversarial_screening_thinking.sbatch`（无需断点续跑，一次跑完），
输出目录 `outputs/adversarial_screening_thinking_ablation/thinking/20260831_172125/`。

**new-Q1（渐进侵蚀）：通过**——18/20 攻击负斜率，t=-7.0256，p=0.0000（df=19）。和 non-thinking
基线 job 15399715 同数量级的强信号，方向一致。

**new-Q3（自相关）：通过，但比基线弱**——slope=0.1965，r=0.1970，p=0.0126（n_pairs=160）。
仍在 p<0.05 阈值内通过，但 p 值比基线的 p=0.0000 高两个数量级——thinking 模式下相邻轮次的
`y_safety` 惯性比 non-thinking 弱一些，值得记录但不影响整体结论。

**诊断指标**：
- `judge_parse_failure_rate=0.0000`——判官钉死 non-thinking 的修复生效,没有被推理链拖累。
- `refusal_rate=0.0050`，和基线量级相当。
- 200 行轨迹里 **198 行 `agent_thinking` 非空、200 行 `agent_message` 非空**（只查了 `<think>`
  切分是否产出了非空内容,不是逐行人工审读）——1536 token 的预算基本够用,只有 2 行没有推理
  内容；`agent_thinking` 字符长度均值约 3130,最长 6273,没有看到大范围硬截断的迹象。

**结论**：deliberative alignment（推理链重新锚定拒绝）和 chain-of-thought hijacking
（推理链本身是新攻击面）这两派预测里,这次数据支持**推理链既没有显著缓解渐进式安全侵蚀,
也没有让它变得更严重**——thinking 模式下侵蚀依然强烈发生（new-Q1 同量级），只是跨轮惯性
（new-Q3）比 non-thinking 弱一点。可以直接写进论文,回应"是否测过 thinking 模式"这个审稿人
几乎必然会问的问题；`agent_thinking` 列已经保留在 trajectories.jsonl 里,后续如果要人工抽查
推理内容本身（比如"模型的推理链里有没有明确讨论过是否该拒绝"这类更细的问题），数据已经在。

**没有做的部分**：没有对 198 行 `agent_thinking` 做定性内容分析（比如推理链里是否出现过
"用户似乎想要有害内容,但……"这类挣扎痕迹）——这次只验证了工程正确性和整体统计结论,定性分析
留给需要写进论文正文时再做。
