# 候选 baseline 清单（用于和 Koopman 控制方法对比）

本文件记录 2026-08-29 调研出的、值得和 `persona_drift_control` 的 Koopman-MPC 方法做对比的
baseline，来源与本地 PDF 缓存（`references/`，已 gitignore，不入库，仅供离线阅读）的对应
关系。分三层：①激励/对照设计层（已在 `DATA_COLLECTION_PROTOCOL.md` 里）②控制器设计层
③代理建模/可控性分析方法层。是否实现见后续的"公平对比实现分析"部分（本文件先只记录调研结果）。

## ① 对照组（协议里已有，不需要额外实现）

| 名称 | 说明 |
|---|---|
| `zero_control` | `u ≡ 0` 自由漂移，无控制基线 |
| `constant_remind` | `u_remind ≡ 1` 每轮完整重申，"暴力"常控基线 |

## ② 控制器设计层 baseline

| 名称 | 论文 | 本地 PDF | 对应协议通道 / 角色 |
|---|---|---|---|
| Split-softmax | Measuring and Controlling Instruction (In)Stability in Language Model Dialogs, COLM 2024. [arXiv:2402.10962](https://arxiv.org/abs/2402.10962) · 代码 [likenneth/persona_drift](https://github.com/likenneth/persona_drift) | `references/split_softmax_persona_drift_colm2024.pdf` | 对应通道 B `u_gain`（对 system prompt token 的注意力放大），训练/参数无关，是与本项目最同源的强 baseline |
| 激活转向 / persona vector steering（CAA 类） | Identifying and Manipulating Personality Traits in LLMs Through Activation Engineering. [arXiv:2412.10427](https://arxiv.org/abs/2412.10427) | `references/activation_engineering_personality_traits_2412.10427.pdf` | 对应扩展通道 C `u_steer`（残差流加对比方向 α·v） |
| LF-Steering | LF-Steering: Latent Feature Activation Steering for Enhancing Semantic Consistency in LLMs. [arXiv:2501.11036](https://arxiv.org/abs/2501.11036) | `references/lf_steering_2501.11036.pdf` | 同上，`u_steer` 通道的另一种转向方向构造法（稀疏特征而非均值差） |
| UniSteer | UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering. [arXiv:2605.30076](https://arxiv.org/abs/2605.30076) | `references/unisteer_flow_matching_2605.30076.pdf` | 同上，`u_steer` 通道的更通用（flow-matching）转向法，规模较大，供参考 |
| 经典反馈控制器（PID / 阈值触发重提醒） | 无独立论文；形式见 `Control_of_Foundational_Model_revised.pdf` 第 9 节公式 (27)-(28)：`e_t = r - y_t`，直接比例/阈值反馈，不经过 Koopman 代理 | — | MPC 必须打赢的"简单对手"：如果 PID/阈值触发就能达到接近效果，Koopman-MPC 的复杂度不值得 |
| 周期性/事件触发重提醒 | 思路参考 [Nautilus Compass: Black-box Persona Drift Detection for Production LLM Agents](https://arxiv.org/abs/2605.09863)（"behavioral invariant checklist + re-anchor"） | `references/nautilus_compass_persona_drift_detection_2605.09863.pdf` | 接近生产环境实际部署策略的对照（固定周期或误差超阈值才触发重提醒，而非每轮独立随机激励） |

## ③ 代理建模 / 可控性分析方法层 baseline

| 名称 | 论文 | 本地 PDF | 角色 |
|---|---|---|---|
| 线性 ARX / LSTM 一步-多步预测模型 | 无独立论文，属于 `Control_of_Foundational_Model_revised.pdf` 第 8 节要求的 ablation（"output-only, finite-memory, augmented-state models" 对比） | — | 证明 Koopman 代理相对任意非线性/记忆模型的增益不是平凡的 |
| GenCtrl | GenCtrl — A Formal Controllability Toolkit for Generative Models, ICLR 2026（Apple）. [arXiv:2601.05637](https://arxiv.org/abs/2601.05637) · 代码 [apple/ml-genctrl](https://github.com/apple/ml-genctrl) | `references/genctrl_controllability_toolkit_2601.05637.pdf` | 无分布假设的黑盒 PAC 可控集估计，可作为独立于 Koopman 代理的"ground-truth 可控集"参照，对应 `Control_of_Foundational_Model_revised.pdf` 第 8 节的 "controllable-set agreement" 验证步骤 |

## 检测/评测类参考（非控制器，用于失败分析与评测设计）

| 名称 | 论文 | 本地 PDF |
|---|---|---|
| Attractor States / Assistant Axis | Attractor States Emerge in Multi-Turn LLM Conversations. [arXiv:2606.30571](https://arxiv.org/abs/2606.30571) | `references/attractor_states_multiturn_2606.30571.pdf` |
| SPASM | SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation. [arXiv:2604.09212](https://arxiv.org/abs/2604.09212) | `references/spasm_stable_persona_simulation_2604.09212.pdf` |
