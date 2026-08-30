# 人格漂移闭环控制：数据来源与候选清单（草案 v0.1，2026-08-28）

本文件回答"人格漂移任务的数据从哪来、各自扮演什么角色、现阶段用不用"。与 `DATA_COLLECTION_PROTOCOL.md`（怎么采）、`EVALUATION_METRICS.md`（怎么评）配套。按流程角色分四组：被控对象的 prompt 库、已有轨迹数据、待采集的激励轨迹、评价用外部数据。表格中"状态"一列：**在用** / **候选** / **不用**。

## 1. System prompt 库：漂移的"设定"从哪来

### 1.1 Naomibas/llm-system-prompts-benchmark（HF，100 条）— 在用

Li et al. 2024 persona_drift 论文配套的 benchmark，已 vendored 至 `persona_drift_control/resources/hundred_system_prompts.py`（来源与两处本地补丁见 `resources/PROVENANCE.md`：词表改读本地缓存；修复 `random_probes` 中把两条探针合并的漏逗号）。每条记录 = (system prompt, 探针问题, 确定性 Python 打分函数)；探针为 `random` 时从 20 条 `random_probes` 中抽。上游五个类别（条数用 `ast` 解析核对，合计 100）：

| 上游类别 | 条数 | 内容 | 打分性质 | 协议标签 | 状态 |
|---|---:|---|---|---|---|
| `pattern_system_prompts` | 29 | 格式/字母/句子模式约束（"以字母 A 开头""只答一句话"） | 纯字符串规则，噪声最低 | `language_constraints` | 在用 |
| `persona_system_prompts` | 14 | 性格/语气（"总是充满喜悦""只用生僻词"） | 词表 + 启发式，接近二值 | `character_traits` | 在用（须过 `EVALUATION_METRICS.md` 1.1 的 SNR 门槛） |
| `memorization_system_prompts` | 27 | 记住一个事实（"你有三个孩子""每次提到东京"），探针直接问事实 | 近似 0/1，信号强噪声小 | — | 候选（扩展）：可辨识性好，但 y 过于离散，Koopman 拟合不利 |
| `multiple_choice_system_prompts` | 27 | 含政治立场等，探针为选择题 | 选择题匹配 | — | 不用（敏感、且与"人格"关系弱） |
| `language_system_prompts` | 3 | 法语作答等 | 语言检测 | — | 不用（太少） |

命名说明：协议中的 "language constraints" **不是**上游只有 3 条的 French 类，而是 `pattern_system_prompts`；映射写在 `prompt_bank.py` 的 `CATEGORY_LABELS`。

规模核对：两个在用类别合计 29 + 14 = 43 条，满足协议第 6 节"40 条 → 28/6/6"的假设，余 3 条可作 screening 专用或备用。

### 1.2 IFEval（Stolfo et al. 2025 所用）— 候选

可验证指令集合（格式 163 / 关键词 203 / 合成长度），性质与 `pattern_system_prompts` 相同但规模大一个量级。用途：若 pattern 类 29 条不够切分或需要更多 `language_constraints`，IFEval 是最自然的扩充源；Stolfo 的 difference-in-means steering 向量即在其上计算，可顺带把 `u_gain`/`u_steer` 通道的执行器搬过来。代价：需要为每条指令写或移植探针 + 打分函数，且 IFEval 是单轮 QA 格式，要改造成 system prompt 形式。

## 2. 已有轨迹数据：同事的 8 份 Koopman 数据集 — 在用（仅接口与 readout）

`datasets/` 下的 `sentence_length_t10`、`character_length_t5`、`average_word_length_t5`、`sentiment_t5`、`formality_t5`、`even_odd_t5`、`vector_count_stage1/2_t10`（详见 `DATASETS.md`、`DATASET_MANIFEST.csv`）。它们**不是**人格漂移数据，而是输出属性受控的轨迹：输入由误差模板决定、目标恒定、闭环采集。

对本任务的价值：
- 接口模板：`trajectory_id / topic_split / turn / normalized_output / effective_norm` 的列约定就是新协议表结构的来源，`core.py` 已在其上验证。
- 连续 readout 复用：`formality`（calibrated scorer）与 `sentiment`（Cardiff RoBERTa）直接成为 `y_formality`、`y_sentiment`。

局限（`KV_INJECTION_MONITORING.md` 第 8 节）：输入方向未被持续激励，无法辨识 B；不能替代第 3 节的采集。

## 3. 待采集的激励轨迹：主数据 — 在用（尚不存在）

`DATA_COLLECTION_PROTOCOL.md` 第 6 节定义：40 prompt × 2 通道 × 4 seed 的 iid/hold 激励轨迹（320 条 × 16 轮），加两组对照（u≡0 自由漂移 160 条；u_remind≡1 固定重申 160 条）；KV 通道按 `KV_INJECTION_MONITORING.md` 第 5 节另加 m 的激励。与同事数据的本质区别：输入每轮独立随机抽取（开环激励），这是 B 可辨识的必要条件。

采集前的 5 prompt × 2 seed × 16 轮信号探针（screening）是其缩小版，`scripts/run_signal_screening.py` 已能产出相同表结构；`analysis.py` 的 Q1–Q3 即门槛检验。

### 3.1 likenneth/persona_drift 官方数据 — 候选（参照用）

Li et al. 发布的 LLaMA2-70B-chat 自聊对话与逐轮探针结果。u≡0 的自由漂移数据，**不能**用于辨识 B；用途是在采集前先验证 `EVALUATION_METRICS.md` 1.1 的 SNR_drift 与 1.2 的衰减率 λ 在另一模型上的量级，作为 Qwen3-4B 结果的参照。获取：GitHub 仓库（当前沙盒与设备 VM 均不能访问外网，需手动下载）。另见 `OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md` 2.1/2.3 节：ContextEcho 长会话数据是同类"参照用、不能辨识 B"的数据，覆盖的是长程 agentic 场景，与本条互补。

### 3.2 模拟用户的话题池 — 待定

`selfchat.py` 的用户模板为 "chatting naturally with an assistant about {topic}"，但话题来源尚未在协议中写明。候选：同事数据的 `topic` 列（已有 train/validation/test 锁定切分）；Li 代码中的对话起始句；`OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md` 2.4 节新增的 LMSYS-Chat-1M/WildChat 真实对话话题分布。需要统一的一点：`topic_split` 列名暗示按话题切分，而协议第 6 节实际按 `system_prompt_id` 切分——两者若同时存在，须明确以后者为准、前者只作记录列，或改名。

## 4. 评价用外部数据

| 数据 | 用途 | 来源 | 状态 |
|---|---|---|---|
| MMLU | 能力代价：第 4 轮插入 MMLU 题，只报干预前后准确率差；带 system prompt 与历史时的绝对值无意义，每次比较须有同条件无干预基线 | Li §6.3 | 在用（`EVALUATION_METRICS.md` 3.2 已定） |
| LLM-judge 打分样本 | 对话质量代价；小参数量本地模型（≤8B），先做一致率检验 | Stolfo Fig. 4 的做法，模型换低成本 | 在用（judge 模型待定） |
| Alhafni 2024 的 ~50 个连续表层特征 | 免费连续 readout（词数、POS 频率、可读性、依存关系），放在 `y_probe` 旁边缓解探针分近似二值的问题；其 251 作者/106k 文档数据本身不用；任务/数据集适配性核实见 [`ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md`](ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md) | Alhafni et al. 2024 及其特征抽取代码 | 候选（`EVALUATION_METRICS.md` 1.4） |

## 5. 未决事项

- 话题池来源与 `topic_split` 的语义（3.2）。
- `memorization_system_prompts` 是否作为第三类进入采集（离散 y 与可辨识性的权衡）。
- 是否引入 IFEval 扩充 `language_constraints`；若引入，探针/打分函数的移植成本。
- persona_drift 官方数据的下载与放置位置（不入库，放 scratch）。
