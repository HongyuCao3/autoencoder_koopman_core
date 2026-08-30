# 轨迹生成机制：agent 与模拟用户具体如何对话

本文件记录 `DATA_COLLECTION_PROTOCOL.md` 里数学层面描述的采集协议，在代码里具体是怎么实现
的——对话怎么发生、长度/主题/随机性各自怎么控制。协议本身不记录这些实现细节，此前只散落在
`persona_drift_control/src/persona_drift/*.py` 各处的 docstring 和注释里，没有一份文档把它们
放在一起，本文件补上这个空缺。代码是唯一的事实来源，本文件仅作导航和解释，实现变了这里也要
跟着改。

## 总览：一条轨迹的构成

一条轨迹（trajectory）= 一个 system prompt entry × 一个 seed × 一个 controller 条件，固定跑
`num_turns`（screening 阶段 16 轮，冒烟测试 2 轮）轮，一轮 = 用户一句 + agent 一句。核心循环在
`persona_drift_control/src/persona_drift/selfchat.py::run_trajectory`，每轮做六件事：

1. **控制器决策**：`controller.next_u_remind(turn, rows)` 决定这轮的 `u_remind`（见
   [controllers.md](controllers.md)）。
2. **模拟用户发言**：模拟用户模型根据自己的对话历史生成一句话。
3. **提醒注入**（如果 `u_remind=1`）：在这句用户消息前面拼一段提醒文本
   （`reminder.py::build_reminder_text`），一起作为这轮喂给 agent 的 user turn 内容。
4. **agent 回复**：agent 模型根据自己的对话历史（含刚插入的提醒）生成回复，写入主线对话。
5. **探针分叉**：把 agent 当前对话历史（含刚生成的这轮回复）加上一个探针问题单独拿出去问，
   **不进入主线对话**，避免探针污染后续轮次的漂移过程（ContextEcho 做法）。
6. **打分记录**：在分叉上重复问 `probe_repeats`（screening 阶段 4）次，每次用该 entry 自带的
   确定性 Python 打分函数评分，取均值 `y_probe` 和标准差 `y_probe_sd`（后者用于估计测量噪声），
   连同这一轮的所有字段打包成一行，累加进这条轨迹的行列表。

agent 和模拟用户各自是独立的 `ChatModel`（`chat_model.py`）实例，各自维护一条完整的、独立的
对话历史（`agent_history` / `user_history`），互相看到的是对方生成的文本，而不是共享同一个消息
列表。

## 长度控制

- **对话轮数**：`TrajectoryConfig.num_turns`。screening 阶段固定 16 轮（协议第 1 节规定），
  冒烟测试临时改成 2 轮以便快速验证流水线。
- **单次生成的 token 上限**（`GenerationConfig.max_new_tokens`，只是生成上限，模型可以提前
  用 EOS 结束，没有强制拉满或事后裁剪到固定长度）：

  | 生成对象 | 上限 |
  |---|---:|
  | agent 主线回复 | 256 tokens |
  | 模拟用户回复 | 96 tokens（比 agent 短，因为普通用户说话本来就更简短） |
  | 探针回答 | 256 tokens |

- 没有对文本长度设下限约束。

## 内容主题控制

- **话题**：从写死的 12 个候选池（`selfchat.py::TOPICS`：公共交通、远程工作、周末徒步等中性
  生活化话题）里选，选择粒度是**按 system prompt entry 选一次**，不是按轮次或按 seed 选——
  `screening.py` 里 `topic = topic_rng.choice(TOPICS)` 处于 `for entry in prompts` 这层循环，
  所以**同一个 system prompt 的所有 seed、所有 controller 条件的轨迹都共享同一个话题**，只有
  换 prompt 才换话题。
- **模拟用户的 system prompt**（`selfchat.py::USER_SYSTEM_PROMPT_TEMPLATE`）明确要求："围绕
  给定话题自然聊天，不要提及、询问或引用对方的 instructions、persona 或 system prompt"——
  防止模拟用户主动剧透或诱导漂移，让漂移只来自 agent 自身的注意力衰减机制，而不是用户话术。
- **agent 的 system prompt**（决定它要维持的"人格"）来自 `prompt_bank.py` 从 HF
  `Naomibas/llm-system-prompts-benchmark`（离线缓存在 `resources/`）筛出的两类：
  `persona_system_prompts`（映射为 `character_traits`）与 `pattern_system_prompts`（映射为
  `language_constraints`）。具体映射依据和与协议原文的偏离见 `persona_drift_control/README.md`
  "探针题库到协议分类的映射"一节。
- **探针问题**是每个 entry 自带的；题库里标成 `"random"` 的占位符会在**加载题库时**（不是每次
  运行时）用固定种子解析成一个具体问题，保证同一个 entry 每次加载都问同一个探针问题。

## 随机性控制

协议要求"给定 seed 完全可复现"，同时对话本身又需要随机采样解码来模拟真实、有噪声的多轮
对话——这两个要求通过给每一路随机性显式分配互不干扰的种子来同时满足：

| 随机性来源 | 播种方式 | 作用范围 |
|---|---|---|
| agent/用户/探针文本生成 | `ChatModel.generate()` 内 `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)`，seed 由轨迹 seed 派生：用户发言 `seed*1_000_000+turn*100`，agent 回复 `...+1`，第 k 次探针重复 `...+2+k` | 同一条轨迹内，用户发言、agent 回复、每次探针重复各自独立、确定、不互相干扰 |
| 控制器决策（如 `RandomExciteController`） | 独立的 `random.Random(seed)`（Python 标准库，非 torch），同样以轨迹 seed 播种 | 与上面生成用的 torch RNG 完全分开管理，见 [controllers.md](controllers.md) |
| 话题分配 / screening 阶段的 prompt 抽样 | `prompt_rng_seed`（默认 0）播种的独立随机源，与轨迹 seed 无关 | 给每个 system prompt 分配一次话题；分层抽样 screening 用的 prompt 子集 |
| 探针题库 `"random"` 占位符解析 | `random.Random(f"{raw_category}:{index}")`，只在 `load_prompt_bank()` 内执行一次 | 保证不同运行之间同一个 entry 解析出的探针问题一致 |

解码本身用随机采样（`temperature=0.7, top_p=0.95, do_sample=True`），不是贪心/beam search；
但因为上述每一路随机源都被显式 seed 锁定，整条流水线是"随机但完全可复现"的——同样的 seed 重跑
一遍会得到逐字节相同的对话和分数。

## 代码索引

| 内容 | 位置 |
|---|---|
| 每轮循环、长度/主题的 system prompt 拼接 | `src/persona_drift/selfchat.py::run_trajectory` |
| 模型加载与带种子的生成 | `src/persona_drift/chat_model.py::ChatModel` |
| 提醒文本构造（channel A） | `src/persona_drift/reminder.py` |
| 题库加载、prompt 抽样、打分 | `src/persona_drift/prompt_bank.py` |
| 轨迹编排、写盘 schema | `src/persona_drift/screening.py` |
| 控制器（u_remind 取值策略） | `src/persona_drift/control.py`，见 [controllers.md](controllers.md) |
