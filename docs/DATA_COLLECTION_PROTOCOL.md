# 人格/指令漂移闭环控制：激励数据采集协议（草案 v0.1，2026-08-27）

本文件描述的采集与同事原始数据的方法不同，代码实现见本仓库 `persona_drift_control/`（`autoencoder_koopman_core` 是我 fork 之后的仓库，不是需要与同事分开维护的共享仓库，因此不再单独建仓库）。目标是采集一份**带自由输入 u_t 的轨迹数据**，使得 Koopman/线性 surrogate 中的输入矩阵 B 可辨识，并为 MPC 闭环验证提供训练集。与同事数据的根本区别：同事数据里的输入是由误差模板决定的（闭环、常目标），本协议里的输入每轮独立随机抽取（开环激励）。

## 1. 被控对象（plant）

- Agent 模型：`Qwen/Qwen3-4B`（与同事数据同模型，便于对照），HF backend，`temperature=0.7, top_p=0.95`，每条轨迹固定 seed。
- 模拟用户：同为 Qwen3-4B（或 Qwen3-8B），system prompt 为"普通用户，围绕给定话题自然聊天，不提及对方设定"。沿用 Harvard persona_drift 的 self-chat 协议（`likenneth/persona_drift` 的 `run.py`）。
- 对话长度 T = 16 轮（一轮 = 用户一句 + agent 一句）。Harvard 数据显示 8 轮内已明显漂移，16 轮留出"漂出去再拉回来"的空间。
- Agent 的 system prompt 来自 HF `Naomibas/llm-system-prompts-benchmark`（100 条）。首期只取两类：**character traits**（性格/语气）与 **language constraints**（语言/风格约束），因为它们的探针分数更接近连续量；其余三类留作扩展。

## 2. 输出测量（readout）y_t

- 每轮 agent 回复后，**分叉**当前对话副本（ContextEcho 做法），在副本上提出该 system prompt 配套的探针问题，用配套的确定性 Python 打分函数得到分数；主线对话不包含探针，避免污染。
- 探针在同一轮重复 K = 4 次（不同 seed）取均值，记 `y_probe`，同时记录 4 次的标准差 `y_probe_sd`，用于估计测量噪声。
- 附加连续 readout（与同事管线兼容）：对主线回复本身打 `formality`（同事的 calibrated scorer）与 `sentiment`（Cardiff RoBERTa），记 `y_formality`, `y_sentiment`。首期 y_t 取 `[y_probe]`，向量版取 `[y_probe, y_formality]`。

## 3. 输入通道 u_t（每轮独立随机抽取）

首期做两个通道，第三个留作扩展。每条轨迹只激活一个通道（便于单独辨识各通道的 B 列），通道标签写入 `input_channel` 字段。

| 通道 | 符号 | 电平 | 物理实现 | 代价度量 |
|---|---|---|---|---|
| A 提醒强度 | `u_remind` | {0, 0.25, 0.5, 0.75, 1} | 在本轮用户消息前插入 system prompt 的摘要：0 = 不插入；0.25/0.5/0.75 = 分别保留原文约 25%/50%/75% 的关键句（预先人工/LLM 生成三档摘要并固定）；1 = 完整重申 | 插入 token 数 |
| B 注意力增益 | `u_gain` | {0, 0.5, 1.0, 1.5, 2.0} | Harvard split-softmax 对 system prompt token 的注意力放大系数（0 = 关闭）；或 CFG 的引导强度 | 推理开销 + 通用能力损失 |
| C 激活转向（扩展） | `u_steer` | {−1, −0.5, 0, 0.5, 1} × α₀ | 第 ℓ 层残差流加 α·v，v 为该人格的对比方向 | ‖α‖ |

电平序列设计：每条轨迹的 u_1..u_T 从上表电平中独立均匀抽取（多电平随机序列）。为保证低频激励也被覆盖，另取 20% 的轨迹使用"持续段"设计：随机长度 2–4 轮内保持同一电平再切换。两种设计用 `excitation_design ∈ {iid, hold}` 标记。

## 4. 目标/设定 r

调节任务里 r 是允许的最低遵循分 `y_min`（首期固定 0.7），不进入采集阶段；采集阶段 r 不出现在 prompt 中。表中仍保留 `target_norm` 字段（=NaN）以兼容同事的列约定。

## 5. 表结构（JSONL，一行一轮）

必备列（与同事 `core.py` 的接口一致）：`trajectory_id, topic_split, turn, y_probe, u_remind|u_gain|u_steer`（未激活通道填 0）。
其余列：`run_id, system_prompt_id, prompt_category, input_channel, excitation_design, seed, model, decoding_config, user_message, agent_message, probe_question, probe_answers(list of 4), y_probe_sd, y_formality, y_sentiment, inserted_reminder_text, inserted_tokens, refusal_flag, parse_failure`。

## 6. 切分与规模

- 切分按 `system_prompt_id`：40 条 prompt → train 28 / validation 6 / test 6，同一 prompt 的所有轨迹在同一 split。
- 首期规模：40 prompt × 2 通道 × 4 seed = 320 条轨迹 × 16 轮 = 5,120 轮；每轮 1 次主线生成 + 4 次探针生成 ≈ 25,600 次生成。Qwen3-4B 单卡（A100/4090）约 6–10 小时。
- 对照组（必须）：同一 40 prompt × 4 seed，`u ≡ 0` 的自由漂移轨迹 160 条，作为"无控制"基线曲线；以及 `u_remind ≡ 1` 的每轮重申轨迹 160 条，作为"固定重申"基线。

## 7. 采集前的 1 小时信号探针（必须先做）

按 `01_pre_experiment_signal_screening` 的要求，先用 5 条 prompt × 2 seed × 16 轮回答三个问题，任一失败则修改协议再采：
1. 漂移是否存在：`u ≡ 0` 时 y_probe 从 turn 1 到 turn 16 的下降是否 > 2 × 平均 `y_probe_sd`。
2. 输入是否有效：`u_remind = 1` 与 `u_remind = 0` 的下一轮 y_probe 差是否 > 2 × `y_probe_sd`。
3. 是否有惯性：u_t 对 y_{t+2} 的效应是否显著非零（否则动力学是一步即时的，Koopman 无用武之地，应改用更长记忆或更弱的干预电平）。

## 8. 文件位置

代码写入本仓库 `persona_drift_control/`；数据（原始生成文本）保留，checkpoint 与缓存放 scratch，不入库。
