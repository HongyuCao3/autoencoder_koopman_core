# 人格漂移闭环控制：评价指标设计（草案 v0.1，2026-08-28）

> **⚠️ 状态（2026-09-03 补记）：这份草案是为人格漂移这条线写的，没有随后来的任务转向更新，
> 不是现行判据清单。** 项目在 2026-08-31 转向对抗防御、2026-09-02 再转向 sycophancy drift；
> 实际在用的判据（new-Q1 渐进侵蚀 / new-Q3 惯性 / 离散翻转事件率与趋势）定义在
> `../task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md`、
> `../task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md` 与各 `../experiments/*_pilot.md` 里，本文件
> **完全没有提到它们**。相应地，下面那句"指标定义若与代码实现（`persona_drift/analysis.py`）
> 冲突，以本文件为准并更新代码"**已不适用**——现在的分析代码是
> `analysis_adversarial.py`/`analysis_sycophancy.py`/`analysis_helpfulness.py`，以它们和上面
> 那些任务文档为准。
>
> 本文件仍然有用的部分：第 1 层"测量本身的有效性"（readout SNR、判分器噪声下界）这套方法论、
> 三层指标（测量/建模/控制）的分层框架、以及四篇论文各自评价设计的出处与短板分析——这些在
> 新任务上依然适用，也确实被沿用了（例如"连续 rubric 打分噪声接近 bin 宽度时控制是假象"这条
> 直接导致 sycophancy 线改用三分类判据）。

本文件是 `DATA_COLLECTION_PROTOCOL.md`（采集协议）、`KV_INJECTION_MONITORING.md`（通道 D）与 `LLM_LATENT_STATE_FEASIBILITY.md`（状态增广可行性）之外的第四份草案，回答"实验做完之后用什么数来判断成败"。它是一份**后续参考与迭代**的文件：每个指标标明来源（哪篇论文的哪种做法）、现阶段的决定、以及尚未确定的地方。指标定义若与代码实现（`persona_drift/analysis.py`）冲突，以本文件为准并更新代码。

依据的文献为 Zotero「Koopman」分类中的四篇：Li et al. 2024（persona drift / split-softmax）、Stolfo et al. 2025（activation steering）、Alhafni et al. 2024（fine-grained linguistic control）、Wang et al. 2026（TMPC）。

## 0. 指标为哪条 claim 服务

四篇论文的评价设计都严格跟着各自的 claim 走：Li 的 claim 是"漂移存在 + split-softmax 在同等能力代价下更稳"，评价核心是逐轮稳定性曲线和等 MMLU 代价比较；Stolfo 的 claim 是"有/无指令时 steering 都提高遵循率"，评价是 IFEval 准确率 + LLM-judge 质量；TMPC 的 claim 是"每次 LLM 调用的收益更高"，横轴是调用预算。

本项目有两条候选 claim，对应的主指标不同：

| claim | 内容 | 主指标层 |
|---|---|---|
| C1 建模 | 人格漂移在提升坐标下近似线性动力系统，输入矩阵 B 可辨识 | 第 2 层（预测误差、可辨识性） |
| C2 控制 | 用辨识模型做闭环 MPC，在同等代价下比任何固定强度的开环方案更稳 | 第 3 层（adherence–cost Pareto） |

**现阶段决定：** headline 为 C2，C1 作支撑。理由：Li 的 Fig. 6（SPR 后期强、split-softmax 前期强）是"按轮调度执行器"的直接动机，而四篇中没有一篇做闭环（都是固定强度、不反馈 y_t）。指标分三层设计，第 1 层是地基，第 2 层支撑 C1，第 3 层支撑 C2。

## 1. 第一层：测量本身的有效性（readout y_t）

四篇在这一层的短板：Li 的探针分数接近二值、未报 CI、mitigation 实验只用 5 个 persona；Alhafni 指出打分器噪声接近 bin 宽度时"控制"是假象。因此以下量必须在任何建模/控制结果之前先报告。

### 1.1 漂移信噪比 SNR_drift（来源：Li Fig. 3 + 本项目 screening Q1）

直觉：u ≡ 0 的轨迹从第 1 轮到第 T 轮的下降，只有明显超过探针自身的重复噪声时才算"漂移存在"。

$$\mathrm{SNR}_{\text{drift}} = \frac{\bar y_1 - \bar y_T}{\overline{\sigma}_{\text{probe}}}$$

$\bar y_1, \bar y_T$：首轮、末轮 `y_probe` 在 u≡0 轨迹上的均值；$\overline{\sigma}_{\text{probe}}$：K=4 次重复探针标准差 `y_probe_sd` 的平均。

用途：按 `prompt_category` 分别报告。某类别 SNR < 2 时，承认探针对该类（尤其 character traits）测不准，该类不进入 C2 的 headline，而不是硬做。对应 `analysis.py` 的 `q1_drift_exists`（现有实现只报差值和阈值，需新增比值本身）。

### 1.2 经验衰减率 λ（来源：Koopman 主特征值的物理先验）

直觉：若漂移是线性动力系统，u ≡ 0 时 y 应按几何速率趋近一个平台；把曲线拟合成一阶衰减，得到一个与模型无关、但能直接与 A 的主特征值对照的数。

$$y_t \approx y_\infty + (y_1 - y_\infty)\,\lambda^{\,t-1}$$

$y_\infty$：漂移平台；$y_1$：首轮分数；$\lambda \in (0,1)$：每轮保留率；$t$：轮次。

用途：第 2 层中辨识出的 A 的最大特征值若与 λ 相差很远，说明提升坐标或延迟嵌入出了问题。新增函数 `fit_decay(rows) -> {y_inf, lambda, r2}`，按 prompt 与 category 汇总。

### 1.3 执行器静态增益曲线与单调性（来源：Stolfo Fig. 5a；反例 Alhafni Fig. 3）

直觉：Stolfo 的权重 c 从 0 到 40、字数从 324 单调降到 23，本质是一条 dose–response 曲线；Alhafni 的 bin 位移响应是双峰的。每个通道在建模前都要先知道自己是哪一种。

对每个通道 $u \in \{u_{\text{remind}}, u_{\text{gain}}, m\}$，报告档位 → 下一轮增量的曲线：

$$g(\ell) = \mathbb{E}\big[\,y_{t+1} - y_t \;\big|\; u_t = \ell\,\big]$$

$\ell$：该通道的离散电平；期望在 `excitation_design = iid` 的所有 (t, 轨迹) 对上取。附 Spearman 相关及其 p 值作单调性检验。

用途：若 `u_remind` 五档不单调，MPC 的可行集改为离散枚举而非连续优化——这是指标层面就能提前发现的设计问题。新增 `static_gain_curve(rows, channel)`。

### 1.4 多 readout 一致性（来源：Alhafni 的连续表层特征）

`y_formality`、`y_sentiment`（以及可扩展的词数、POS 频率、可读性）适合放在 `y_probe` 旁边，因为探针分近似二值时 Koopman 拟合很差。报告 `y_probe` 与各连续 readout 的相关系数矩阵。低相关不是坏事（测的是不同维度），但要防止后续把"formality 稳定"误报为"人格稳定"。

对 character traits 类，另加小样本（如 100 轮）LLM-judge 或人工核对与确定性打分器的一致率（见 3.2 的低成本 judge）。

## 2. 第二层：辨识模型的质量（支撑 C1）

评价对象是 `autoencoder_koopman_core` 拟合出的 (A, B, c)。测试集按 `system_prompt_id` 切分（协议第 6 节）。

### 2.1 相对预测误差 Skill_H（来源：系统辨识常规；针对 Koopman 论文常见只报一步误差的漏洞）

直觉：绝对 MSE 没有参照意义，应问"比最笨的预测好多少"。基线取持续预测 $\hat y_{t+1} = y_t$。

$$\mathrm{Skill}_H = 1 - \frac{\sum_t (y_{t+H} - \hat y_{t+H\mid t})^2}{\sum_t (y_{t+H} - y_t)^2}$$

$\hat y_{t+H\mid t}$：从第 t 轮出发、把真实 u 序列喂入模型、滚动 H 步的预测；$H$ 取 1 与 MPC 规划视界（暂定 4）。多步误差比一步更重要，因为 MPC 使用多步预测。第二条基线：u≡0 的类别平均曲线（"不看输入也能猜到的部分"）。

### 2.2 B 的可辨识性与一致性（来源：本项目 screening Q2 的正式版）

对轨迹做 bootstrap 重采样（≥ 200 次），报告 B 各分量的置信区间与符号一致率。并且 B 预测的"u=1 与 u=0 的下一轮差"必须能复现 Q2 直接测出的经验差值 `q2_input_effective.diff`——这是"模型学到了输入效应而非拟合噪声"的直接检验。

### 2.3 惯性 / 脉冲响应（来源：screening Q3 的正式版）

报告经验上 $u_t$ 对 $y_{t+1}, y_{t+2}, y_{t+3}$ 的效应（分层回归系数与 CI），与模型脉冲响应 $C A^{k-1} B$ 逐 k 对比。若经验上只有 $y_{t+1}$ 受影响，Koopman 相对静态回归没有优势，指标必须诚实显示这一点。

### 2.4 衰减率表

同一张表并列：1.2 的经验 λ；A 的最大特征值 $\rho(A)$；KV 通道下 `a_bank` 与 `a_sys` 的保持率（`KV_INJECTION_MONITORING.md` 第 4 节层次二的 K 对角元）。三者应量级一致。

## 3. 第三层：控制效果（headline，支撑 C2）

吸收 Li 的两个关键设计：**等代价比较**与**逐轮拆解**，并针对本项目的"阈值调节"任务修正主指标。

### 3.1 adherence 指标

直觉：本项目的 r 是允许的最低遵循分 $y_{\min}$（暂定 0.7），是阈值调节任务；Li 报的轮均稳定性对调节任务有误导——前 8 轮很高、后 8 轮塌掉的轨迹均值可能不错但完全失败。

$$\bar S = \frac{1}{T}\sum_{t=1}^{T} y_t,\qquad S_{\min} = \min_t y_t,\qquad V = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\,y_t < y_{\min}\,]$$

$\bar S$：轮均稳定性（与 Li 直接可比）；$S_{\min}$：最差轮；$V$：越界轮占比；$T$：对话轮数（16）；$\mathbf{1}[\cdot]$：指示函数。

**现阶段决定：** $V$ 与 $\bar S$ **并列**作为考察指标，暂不指定唯一主指标；等首批数据看清两者分歧程度后再定 headline 用哪一个。$S_{\min}$ 作辅助。可扩展：越界深度积分 $\sum_t \max(0, y_{\min} - y_t)$。

### 3.2 代价指标

三类代价，全部单独报告，不做加权合成：

| 代价 | 定义 | 来源 | 现阶段决定 |
|---|---|---|---|
| 干预代价 | $\sum_t c(u_t)$；remind 用 `inserted_tokens`，gain/steer 用强度绝对值，KV 用 m | 协议第 3 节代价度量 | 采用 |
| 能力代价 | turn-4 处 MMLU 准确率的下降（只报干预前后差值，绝对值无意义） | Li §6.3 | **采用 MMLU** 作为能力代价的主指标 |
| 对话质量代价 | LLM-judge 对主线回复的质量打分下降 | Stolfo Fig. 4（GPT-4o rubric） | 采用**低成本 judge**：小参数量本地模型（候选 Qwen3-4B/8B，与 agent 同族，或其他 ≤ 8B 开源模型），固定 rubric、固定 seed、每样本打分 K=3 取中位数；先在 ~100 条样本上测与人工判断的一致率，一致率不达标则只报 MMLU |

注意：MMLU 是**校准干预强度的工具**，不是能力本身（Li 原话："MMLU drop should be thought of as a budget"）。对话质量 judge 的作用是堵"MMLU ≠ 对话能力"这个漏洞，其自身噪声要用一致率检验约束。

### 3.3 核心比较：等代价下的 Pareto 曲线（来源：Li Fig. 5）

直觉：任何方法都能靠加大干预买稳定性，只有代价相同时比较才公平。

做法：对每个开环基线（常数 `u_remind`、常数 `u_gain`、常数 m，以及协议第 6 节的 u≡0 与 u_remind≡1 两条对照）扫强度得到一条 adherence–cost 曲线；MPC 通过调 $y_{\min}$ 或代价权重得到自己的曲线。C2 的 claim 即：MPC 曲线在每个代价预算上处于或高于所有单旋钮曲线的上包络。压成一个数：

$$\Delta(c^\star) = V_{\text{MPC}}(c^\star) - \min_{b \in \text{baselines}} V_b(c^\star)$$

$c^\star$：固定的代价预算（横轴取三类代价中的任意一类，分别报）；$V_b$：基线 b 在该预算下的越界率（$\bar S$ 版本同理，符号相反）。$\Delta < 0$ 且 CI 不跨 0 才算支持 C2。

### 3.4 逐轮拆解（来源：Li Fig. 6）

在等代价下画每轮 $y_t$ 曲线（MPC vs 各基线）。这是 C2 的机制证据：若 MPC 前期像 split-softmax、后期像 SPR，说明它确实在调度执行器，而不只是平均意义上更好。同时画每轮 $u_t$ 的调度序列。

### 3.5 调用预算视角（来源：TMPC）

闭环的真实成本包含测量：探针每轮 4 次额外生成。单独报告"每轮附加生成次数"，并做 K=1（单次探针）消融，检验闭环在测量变便宜、变噪之后是否仍成立。

### 3.6 统计要求

四篇的公平性标记里全部出现"单 seed / 无 CI"。本项目：≥ 4 seed；所有第 3 层指标报基于轨迹的 bootstrap CI；test split 的 6 条 prompt **逐条**列出结果而不只是平均——6 条太少，被均值掩盖的失败案例必须可见。

## 4. 指标 → 数据列 → 代码 对照

| 指标 | 依赖列（JSONL） | 函数（`analysis.py`，新增用 * 标） |
|---|---|---|
| SNR_drift | `y_probe, y_probe_sd, excitation_design, prompt_category` | `q1_drift_exists`（补比值） |
| λ | `y_probe, turn, trajectory_id` | *`fit_decay` |
| 静态增益 / 单调性 | `u_remind/u_gain/m, y_probe, turn` | *`static_gain_curve` |
| readout 一致性 | `y_probe, y_formality, y_sentiment` | *`readout_correlation` |
| Skill_H | 模型预测 + `y_probe` | *`skill_score`（在 core 侧或 analysis 侧） |
| B 一致性 | 训练输出 + `q2_input_effective.diff` | *`bootstrap_B` |
| 脉冲响应 | `u_*, y_probe` | `q3_inertia`（扩展到 lag 1–3） |
| $\bar S, S_{\min}, V$ | `y_probe, turn`，常量 `y_min` | *`adherence_metrics` |
| 代价 | `inserted_tokens, u_gain, m`；新增列 `mmlu_acc_turn4`, `judge_score` | *`cost_metrics` |
| Pareto / Δ | 上两项按 run 汇总 | *`pareto_compare` |

新增列：`mmlu_acc_turn4`（每条轨迹一个值，其余轮 NaN）、`judge_score`、`judge_model`、`judge_seed`。

## 5. 未决事项（迭代时更新）

- headline 用 $V$ 还是 $\bar S$：待首批数据，看两者排序是否一致。
- LLM-judge 的具体模型与 rubric：先做一致率检验再定；一致率阈值暂定 Cohen's κ ≥ 0.6。
- 代价预算 $c^\star$ 的取值网格：待基线曲线出来后按分位数取 3 个点。
- character traits 类是否进入 headline：取决于 1.1 的 SNR。
- Skill_H 的 H 是否与最终 MPC 视界一致：MPC 视界定了之后同步。
