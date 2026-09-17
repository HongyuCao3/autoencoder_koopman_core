# 第二 backbone 组（Table 1 稳健性）：计划

> **状态**：进行中（2026-09-17）。闸门 B0 判过；§八 第 2 步 `tsar_cefr` gemma-4 激励臂**已于 14:47 提交**
> （job **16018827**，用户裁决「提交」），终态见 [`../LEDGER.md`](../LEDGER.md) §6。其余作业仍未提交。
> 口径 `../../.claude/global.md` → *报告口径*；提交规则 `../../.claude/experiments.md`；
> 术语 [`../NAMING.md`](../NAMING.md)；主表设计 [`../article/MAIN_TABLE_DESIGN.md`](../article/MAIN_TABLE_DESIGN.md)。
> **outline 由 agent 起草**（`../../.claude/docs.md` 要求 outline 由用户写）——这一条是偏离，请一并裁。

## 一、它买什么

Table 1 现在十一列**全部出自同一个 backbone（Qwen3-4B）**。所以「延迟嵌入有用、非线性不额外买东西」
这两条结论，与「Qwen3-4B 这一个模型上延迟嵌入有用」在证据上**无法区分**。本组加第二个 4B 级
backbone，只为回答一个问题：**这两条结论换一个模型家族还成不成立。**

**不重开任何一条线的科学问题。** `defense` ⛔已关闭、`gsm8k_sharded` ⏸挂起、`constraint` ✅收尾、
`core` ✅收尾——本组**不动这些判定**，只在 Table 1 的算子质量测量上加一个 backbone 轴。

## 二、backbone：`google/gemma-4-E4B-it`（2026-09-17 用户签字）

实测可用性（登录节点上跑过，非记忆）：不 gated / `transformers` 5.16.1 与 `vllm` 0.28.0 均支持
`Gemma4ForConditionalGeneration` / 权重 16.02 GB 已落 `hf_cache` / chat template 支持 `system` 角色
与多轮 / `enable_thinking` kwarg 被接受并忽略（`chat_model.py` 的 TypeError 回退用不上）。

**表注必须同址带的两条**：
1. **「同 4B 级」是有效参数口径，不是总参/显存口径。** E4B 总参 7.996 B、BF16 16 GB，是 Qwen3-4B
   （7.6 GB）的 2.1×。Gemma 4 没有稠密 4B（稠密档从 12B 起跳）。
2. **tokenizer 不同（262144 对 ~151k）→ 等 token 代价轴跨 backbone 不可比。** 只影响 Table 2，
   本组只做 Table 1，不受影响。

## 三、范围：四列，不是十一列

| 列 | 怎么换 | Qwen 侧实测 |
|---|---|---|
| `tsar_cefr` | `--agent-model` 一个 flag | 7m11（15850523） |
| `gsm8k_sharded` | 同上 | 43m（15696226） |
| `constraint` | 同上 + 同一个 14B 独立判分重跑 | 4h22m + 判分约 1h |
| core 7 列 | **需重建采集 harness**，见 §四 | — |

**`defense` 列不做**（2026-09-17 用户裁决）。该列判分是自判，换 backbone 等于同时换 agent 与 judge；
且该列在 Table 1 本就判为「本设计分辨不出来」（六行贴 0、最好 null 是 `const`）。

## 四、core 七列：协议可重建，且已验过

仓库里**没有 core 的生成器**（`datasets/` 是同事给的一整份，单个 commit `ca2f107`）。
但协议可以从已落盘行里**recover 并逐行验证**——`scripts/verify_core_protocol_reconstruction.py`
（24 条单测），零 GPU、只读、不写 `outputs/`：

**轮结构三条约定，八个任务 12,840 个 turn-pair 上 100% 成立**：历史 append-only /
assistant 轮 = `raw_generation` / user 轮 = 上一行的 `feedback_text`。
→ **turn-1 prompt 逐字回放即可，不需要重建**；要重建的只有打分器与反馈规则。

**打分器逐行保真度**（对已落盘 Qwen 行）：

| 任务 | 字段 | 保真 |
|---|---|---|
| `sentence_length_t10` | `measured_raw_count` | **1.0000** |
| `character_length_t5` | `measured_raw_count` | **1.0000** |
| `average_word_length_t5` | `measured_raw_value` | **1.0000** |
| `even_odd_t5` | `integer_value` | **1.0000** |
| `vector_count_stage1_t10` | `word_count` / `average_word_length` | 1.0000 / 0.9927 |
| `vector_count_stage2_t10` | `word_count` / `awl` / `comma_count` | 0.9807 / 0.9693 / **1.0000** |
| `formality_t5` | 外部模型 `hf_lendiglearn_mdeberta_v3_formality_isotonic` | 未定（isotonic 标定不在仓库） |
| `sentiment_t5` | 外部模型 `hf_cardiff_roberta_latest` | 未定（标准模型，可下载） |

分词规则是**拟合出来的、不是从示例猜的**：按空白 / `/` / U+2019 切，每 token 计
`1 + count('-')//2`（`well-being` 记 1、`up-to-date` 记 2、`24/7` 记 2）。任一条去掉都会
在 2520 行上留下可见残差。`vector_stage2` 残差 1.93%（±1/±2 混合，30 条含 `3 PM`）**没有再逆推**
——继续猜就是「从示例推断架构」。

### core 的两条硬限制（写进表注，不许省）

1. **第 3 行在 core 上结构性为零，换 backbone 也还是零。** `NAMING.md` 已记死：core 的控制量
   是跟踪误差 $r-y_t$，是状态的**精确仿射函数**（八任务 $R^2=1.0$）。所以 core 的第二 backbone
   只能支持第 1/2/4/5/6 行的结论，**不能**用来谈执行器。
2. **保真度 <1.0 的任务，已落盘 Qwen 列不是合法对照。** 那两列的打分器与同事的差 1.9–3.1%，
   backbone 效应会与打分器效应混在一起。→ **凡进本组的 core 任务，必须同时跑一个同 harness 的
   Qwen3-4B 对照臂**，比较只在本组内部做，不跨到已发表的 Table 1 列。
   （先例：G1 选整 5 seed 重跑而非续跑，买的就是"同一个 harness 指纹"。）

## 五、闸门准入三行（`../../.claude/experiments.md`）

**B0：core 协议重建保真闸门（零 GPU，已执行，2026-09-17）**

| 字段 | 内容 |
|---|---|
| 过了 → | 主线动作：提交 `tsar_cefr` + `gsm8k_sharded` 的 gemma-4 激励臂（辨识数据） |
| 不过 → | core 七列**退出**本组，四列缩成三列（行为线），不再找第三种重建法 |
| 它填论文哪张图/表 | Table 1 的第二 backbone 组（新增表，与现 Table 1 并列） |

**已判：轮结构八任务全过；打分器 5 个字段 1.0000、3 个字段 0.969–0.993、2 个任务待外部模型。**
→ 按第二行，`vector_count_stage2_t10` 与两个外部打分器任务**暂不进本组**，其余五个 core 任务进，
且每个都带同 harness 的 Qwen 对照臂。

## 六、死亡条件

1. gemma-4 在某条线上**读出无量程**（最好 null 是 `const`、或 `Skill_H` 六行全贴 0）→ 该列如实
   标「本设计分辨不出来」，**不调参、不换 judge**（`defense` 的死法）。
2. 同 harness 的 Qwen 对照臂与已发表 Qwen 列**结论方向相反** → 说明重建 harness 不等价，
   **core 七列整体退出本组**，只留行为线。
3. 三条行为线里有两条的 gemma 臂跑不出可判读的行（截顶 >5%/题，按 `constraint` 线预注册判据）
   → 本组关闭，不抬 cap 第二次。

## 七、运行时估计（依据逐条列出）

| 作业 | 估计 | 依据 |
|---|---|---|
| `tsar_cefr` gemma 臂 | **0.4–1.0 h** | Qwen 实测 7m11，3600 次生成；gemma 权重 2.1× 大、vocab 1.7×，取 2–4× 余量 |
| `gsm8k_sharded` gemma 臂 | **1.0–2.0 h** | Qwen 实测 43m，666 行 |
| `constraint` gemma 臂 + 判分 | **6–11 h** | Qwen 实测 4h22m（9600 次生成）+ 判分 1h51m（两份读出，本组只需独立 14B 那份 ≈ 55m） |
| core 5 任务 × 2 backbone | **1–3 h** | 约 7,740 次生成/backbone（除去 stage2 与两个外部打分任务），单句短生成，vLLM 批处理 |

**合计约 9–17 GPU-h。** 未实测项：gemma-4 E4B 在本节点的 vLLM 吞吐、以及它的
`max_new_tokens` 需求——**cap 必须先标定**（见 §八）。

## 八、提交顺序（一条都不许并发做 editable install）

1. **cap 标定 smoke**（基建，不入配额，约 10 min）：gemma-4 在三条行为线各跑 ~24 行，量回复
   token 中位/最长与截顶率。**Qwen 的 cap 在 gemma 上不成立**是有先例的——15739196 就是
   14B 的 cap 512 拿到 4B 上炸出 8.1% 解析失败、15756689 触顶 18.6% 被迫重跑。
2. `tsar_cefr` 臂（最便宜、唯一活线）→ 判读 → 3. `gsm8k_sharded` → 4. core 对照臂对 → 5. `constraint`。

**每一步跑完回填 `../LEDGER.md` §3，失败作业要么重提、要么写明为什么不重提。**
