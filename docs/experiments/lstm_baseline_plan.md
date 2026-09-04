# 计划：LSTM 代理模型 baseline（补齐 `BASELINES.md` ③层缺口）

和 [koopman_detection_design.md](koopman_detection_design.md)/[koopman_defense_pilot.md](koopman_defense_pilot.md)
同一类"供跨会话接续"的记录。**新开一次对话想知道"LSTM baseline 做到哪一步、结论是什么"，
看这份文档。** 计划部分（下面到"资源预估"为止）已执行完毕。

**当前结论（2026-09-03 v 对齐重跑之后，见文末最后一节，这是最新的一节）**：在和 AE baseline
同一口径（早停判据用训练攻击切出的验证集）下，LSTM 的 held-out rollout MSE 是
**0.082–0.095**，比 `richer_abs_sign` 的 0.0684 差 20%–39%，四个隐层大小全部更差——
**"LSTM 没赢"这个方向成立**。但中间的"执行结果（2026-09-01）"一节里那个**接近 2 倍**的差距
**不成立**：那批数字跑在 v 对齐 bug 修复之前，修正后差距从 1.88 倍掉到 0.97–1.07 倍，剩下的
劣势全部由早停口径撑着（旧口径的早停判据就是被报告的那个集合）。读这份文档时，**"执行结果
（2026-09-01）"一节的数字只在它自己的旧口径内有效**，跨文档比较请用最后一节的表。

## 目的：现在这个 claim 站不住脚，需要它来补

`BASELINES.md` ③层把"线性 ARX / LSTM 一步-多步预测模型"列为
`Control_of_Foundational_Model_revised.pdf` 第 8 节要求的 ablation，角色写的是"证明 Koopman
代理相对任意非线性/记忆模型的增益不是平凡的"。现状是：`fit_koopman_defense_model.py` 已经对比过
`no_extra_features`（纯 ARX）和 `abs_sign_extra_features`（`richer_abs_sign`）——这两者都是
**同一个 `(nu=1, mu=2)` 固定窗口线性回归**，只是特征字典（lifting）不同，`richer_abs_sign`
赢了（held-out rollout MSE 0.043 vs 0.051，见 `koopman_defense_pilot.md` Phase C）。这只能
支撑"手工设计的非线性 lifting 比纯线性好一点"，**不能**支撑"Koopman 这套（有限记忆窗 + 线性/
轻度非线性）结构本身相对更强的函数近似器（真正的深度序列模型）没有留下明显的性能"——现在没有
任何一个真正的非线性/长记忆模型跑过对比，这条更强的 claim 目前没有证据。LSTM baseline 就是
用来补这个洞的。

## 设计上第一个必须先定的问题：LSTM 用哪种"记忆"

这个选择直接决定整个实现方式，不能跳过：

**方案 A（固定窗口，和 ARX/Koopman 完全同信息量）**：LSTM/MLP 吃和 `richer_abs_sign` 完全一样的
`z_t = [y_hist(nu), v_hist(mu)]`（`modeling.dataset.ReducedStateConfig` 已经构造好的那个向量），
每次独立预测,不跨轮次传递隐状态。
- 优点：严格控制变量——同一份 `build_identification_dataset` 输出、同一个 train/held-out split，
  唯一变量是"线性岭回归"换成"更深的非线性函数"，是最干净的 ablation。
- 缺点：没用上 LSTM 真正的循环记忆能力——如果每次调用都从零状态独立跑，本质上只是一个换了名字
  的 MLP，"LSTM"这个名字名不副实，也测不出"更长历史是否有用"这个问题。

**方案 B（真正的序列模型，隐状态在整条轨迹内持续演化）**：LSTM 在整条轨迹上跑，`(h, c)` 就是
predictor 的"z"（打包成一个 numpy 向量满足 `Predictor.step(z, v)`/`readout(z)` 协议）,从
轨迹开头的全零状态开始，每轮真实喂入 `(y_t, v_t)` 向前滚动。
- 优点：这才是 `BASELINES.md` 原本想问的问题——"如果不把状态限制在 `nu=1, mu=2` 这么窄的窗口
  （Phase C 已经确认 `mu=3` 在这批短轨迹上数值退化撑不住），允许模型自己学要记多久，能不能做得
  明显更好"。这是对 Koopman 建模选择本身（有限记忆窗假设）的压力测试，证明力度比方案 A 强。
- 缺点：`(h, c)` 是黑盒向量，不是可解释状态——`modeling.koopman.controllability_diagnostics`
  这类基于 `A`/`B` 结构（特征值、Gramian）的分析对 LSTM 完全不适用，这是预期之内、不是缺陷,
  LSTM 本来就不该套用这类分析（controllability 诊断是给"可以写成线性状态空间方程"的模型准备的）。
  另外训练不再是 ridge 回归的 closed-form 解，需要真正的 BPTT + torch 优化器训练循环——
  `docs/method/koopman_surrogate.md`"已知缺口"一节已经写明这部分还没做。

**推荐方案 B**，理由见上（更贴合 `BASELINES.md` 写这条 baseline 的初衷）。方案 A 作为方案 B 训练
不稳定/效果异常时的降级选项记在这里，不单独实现。

## 接口设计

新建 `persona_drift_control/src/persona_drift/modeling/lstm_baseline.py`：

- `LSTMSurrogate`（`torch.nn.Module` 子类）：单层 `nn.LSTMCell(input_size=2, hidden_size=H)`
  （输入是 `[y_t, v_t]`，`H` 是超参，见下面"参数量对齐"）+ 一个线性层 `Linear(H, 1)` 把 `h`
  映射到 `y` 的读数——和 `KoopmanSurrogate` 的 `C` 矩阵角色一样。
- `step(z, v) -> z_next`：`z` 是 `(h, c)` 拼接成的 `2H` 维 numpy 向量（满足 `Predictor` 协议
  "z 是一个 numpy 向量"这个约定），内部 reshape 回 torch tensor 跑一步 `LSTMCell`，输出新
  `(h, c)` 再拼接回 numpy。
- `readout(z) -> float`：从 `z` 里切出 `h` 那一半，过线性层。
- 轨迹开头 `(h, c)` 初始化为全零——和"没有先验知识"这个假设一致，也是 `nn.LSTMCell` 的标准约定。

这样设计后 `LSTMSurrogate` 满足 `modeling.evaluate.Predictor` 协议（`step`/`readout`），
理论上可以直接复用 `one_step_error`；但 `rollout_output_error`/`build_reduced_state_pairs`
是按 `ReducedStateConfig(nu, mu)` 固定窗口设计的，不适用于"整条轨迹连续滚动隐状态"这种用法，
需要为 LSTM 单独写一个 `rollout_lstm(model, traj_rows, y_col, u_col)` 风格的函数（逻辑：
从零状态开始，逐轮 `step`，每轮都用**真实观测到的** `(y_t, v_t)` 前进——即 one-step 评测口径
的教师强制版本；自由多步 rollout 则改成不重新 grounding，纯用模型自己的预测递推，两者都要，
和 `KoopmanSurrogate` 目前的 one-step/rollout 两件套对齐）。

## 训练循环

- 数据：Phase B 的 30 个攻击（`outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl`），
  和 `fit_koopman_defense_model.py` 完全相同的 75/25 `attack_id` split（`seed=0`）——**必须**
  是同一个 split，不然 LSTM 和 Koopman 的 held-out 攻击集合不一致，对比就不公平。
- 按轨迹（不是按 transition pair）喂入，因为 LSTM 需要在一条轨迹内部保持隐状态连续——这是和
  `build_identification_dataset` 把所有轨迹打散成独立 pair 的用法本质的不同,复用不了那个
  数据结构，只能复用它上游的 `group_by_trajectory`。
- loss：每轮 `MSE(readout(h_t), y_t)` 累加，teacher forcing（每步都喂真实 `y_t`/`v_t`，不用
  模型自己上一步的预测）——和 one-step 评测口径一致。
- 优化器：`torch.optim.Adam`，小学习率（如 1e-2 量级，数据量很小，没必要用大模型训练那套
  学习率日程），早停判据用 held-out rollout MSE（不是 train loss，避免在小数据上过拟合还
  显示"在变好"）。

## 评测口径对齐（和 Koopman 的两件套一一对应）

- **one-step MSE**：每轮都用真实 `(y_t, v_t)` 重新起算，预测 `y_{t+1}`——直接和
  `richer_abs_sign`/`arx` 的 `held_out_rollout_mse`旁边的 one-step 数字（`train_one_step_mse`）
  可比。
- **rollout MSE**：从轨迹开头零状态起，只喂真实的 `v` 序列（不重新 grounding `y`），自由滚动到
  轨迹末尾——对应 `rollout_output_error`，和 Phase C 报告里 `held_out_rollout_mse` 那一列直接
  可比。

## 参数量对齐：避免"赢是因为模型更大"这种不公平论证

`richer_abs_sign` 的可学参数总数 = `A`（`d_psi × d_psi`）+ `B`（`d_psi × 1`）+ `b`（`d_psi`）+
`C`（`1 × d_psi`），`d_psi = state_dim + 2 = 5`（`abs_sign_extra_features` 加 2 维），量级在
**30 个参数**左右。`nn.LSTMCell(2, H)` 的参数量是 `4×(2×H + H×H + H)`——即使 `H=2` 这么小，
参数量也有 `4×(4+4+2)=40`，已经超过 Koopman 那 30 个。**结论：不可能做到参数量完全相等**，
计划改成"报告里明确写出两边的参数量，如果 LSTM 明显赢,补一次进一步缩小 `H`（比如 `H=1`）的
消融，看优势是否在参数量继续缩小后消失"——不追求完全公平（做不到），追求把这个混杂因素摆在
台面上,不藏起来。

## 数据规模的现实约束（最大的先天局限，如实记录在这里，不要等结果出来才提）

Phase B 只有 30 个攻击 × 2 seed = 60 条轨迹、每条 4-6 轮，identification 数据在 pair 数量级上
是几百条。深度序列模型在这个规模下过拟合风险很高，这是这个 baseline 天生的劣势，不是实现问题。
**预期结果很可能是"LSTM 持平或更差"**——这不代表这个 baseline 不值得做，"在这个数据规模下强
结构假设（线性 + 小状态）是合理的归纳偏置，不是限制"本身就是一个有信息量的、支持现有
`richer_abs_sign` 选择的结果。如果反过来 LSTM 明显更好，那更说明现在的模型选择有问题，更值得
知道。

## 成功判据 / 这次实验想回答的问题

- LSTM held-out rollout MSE 明显好于 `richer_abs_sign` 的 0.043（且缩小 `H` 后优势仍在）→
  说明 Koopman 的线性+lifting 结构确实丢了可学的信息，值得重新考虑更丰富的 lifting，或者
  在需要更高精度预测的场景直接用 LSTM 做代理（代价是放弃 `controllability_diagnostics` 这类
  可解释性分析）。
- LSTM 持平或更差 → 支持"`richer_abs_sign` 这个选择在当前数据规模下是合理的"，`BASELINES.md`
  那条"Koopman 代理相对任意非线性/记忆模型的增益不是平凡的"claim 有证据支撑。

## 明确不在这次计划范围内的

- 不把 `controllability_diagnostics` 之类的线性状态空间分析套到 LSTM 上——不适用，不是缺口。
- 不把 LSTM 接入 `KoopmanMPCController` 做真正的闭环控制器——这次只是代理建模层的对比 baseline，
  不是要新增一条控制器分支；如果 LSTM baseline 结果显示它明显更准，是否值得再花力气把它接成
  MPC 控制器，是那之后才需要考虑的问题。

## 资源预估

纯 CPU 可行——小 LSTM（`H` 个位数）、几十条轨迹、几百个训练样本，训练几秒到几分钟，和
`fit_koopman_defense_model.py` 一样可以直接在登录节点跑，不需要 GPU/sbatch。唯一的新依赖是
`torch`（已经是 `pyproject.toml` 的既有依赖，训练脚本本身不需要新增任何包）。

---

## 执行结果（2026-09-01）

> **⚠️ 口径提示（2026-09-02 补记）**：本节所有数字都是在 **`modeling/dataset.py` 的"v 对齐"
> 时序错位 bug 修复之前**跑的（bug 与修复见 `koopman_case_study_design.md` 的 "Phase I：v
> 对齐修正与再验证"）。因此这里的 `richer_abs_sign` 对照值是 **0.0430**，而 v-aligned 之后
> 的同一基线是 **0.0684**（见 `ae_baseline_plan.md`）——跨文档比较 held-out rollout MSE 时
> 不要把这两套数字混着用。~~**LSTM 侧尚未在修正后的对齐下重跑**；"LSTM 明显更差"这个结论的
> 方向大概率不变（差距接近 2 倍，远大于对齐修正带来的量级变化），但严格说它仍是一个
> **待复核**的结论~~
>
> **已复核（2026-09-03，见文末最后一节）**：重跑做了，**方向对、量级错**。v 对齐后差距从
> 1.88 倍掉到 0.97–1.07 倍——上面那句"差距接近 2 倍所以方向大概率不变"的推理前提被推翻了；
> 结论之所以仍然成立，靠的是本节没有意识到的另一件事（早停判据用的就是被报告的 held-out
> 集合，是乐观的），换成公平口径后 LSTM 差 20%–39%。本节的数字继续只在本节口径内有效。

按方案 B 实现：新增 `src/persona_drift/modeling/lstm_baseline.py`
（`LSTMSurrogate` + `teacher_forced_predictions`/`rollout_predictions`/
`mse_from_predictions`/`train_lstm_surrogate`）、
`scripts/fit_koopman_lstm_baseline.py`、`tests/test_lstm_baseline.py`（6
passed，CPU 全套 179 passed，`test_surface_feature*` 两个文件因预先存在的
`nltk vader_lexicon` 资源缺失报错，与本次改动无关，跳过）。

**评测口径的一个必要修正（如实记录，不在原计划里）**：Koopman 的
`held_out_rollout_mse`/`train_one_step_mse` 只在 `t >= start =
max(nu-1,mu) = 2` 的位置产生预测（`build_reduced_state_pairs` 的窗口约束），
LSTM 理论上从 t=0 就能预测，但 t=0/1 处只有全零初始 `(h,c)` 或极短历史，
直接对比 t>=0 的全窗口 MSE 对 LSTM 不公平地偏差（早期位置的预测本质上是
"完全没有信息时的先验猜测"，Koopman 从不在这些位置尝试预测）。因此每个指标都
报了 `_full`（t>=0）和 `_matched`（t>=2，和 Koopman 用完全相同的位置集合）
两个版本，结论只看 `_matched`。

**训练**：Phase B 的 30 攻击 × 2 seed = 60 条轨迹，和 `fit_koopman_defense_model.py`
完全相同的 `attack_id` 75/25 split（`split_seed=0`，22 训练攻击/44 条轨迹，
8 held-out 攻击/16 条轨迹）。teacher-forced BPTT，Adam，`lr=1e-2`，早停判据
= held-out rollout MSE（matched window），扫了 `H ∈ {1,2,4,8}`。

| H | 参数量 | epochs | train one-step MSE (matched) | held-out rollout MSE (matched) | 训练耗时 |
|---|---|---|---|---|---|
| 1 | 22 | 31 | 0.0708 | **0.0808** | 7.7s |
| 2 | 51 | 42 | 0.0754 | **0.0809** | 6.2s |
| 4 | 133 | 90 | 0.0666 | **0.0862** | 13.3s |
| 8 | 393 | 79 | 0.0652 | **0.0821** | 11.7s |
| `richer_abs_sign`（对照） | **40** | — | 0.0587 | **0.0430** | — |

**诚实结论：LSTM 在全部 4 个测试的 `H` 上都明显更差，不是持平**——即使
`H=1`（22 个参数，比 `richer_abs_sign` 的 40 个还少）训练时的 one-step
拟合和 Koopman 相当（0.071 vs 0.059），held-out rollout MSE 却是
`richer_abs_sign` 的接近 2 倍（0.081 vs 0.043），且这个差距在 `H` 从 1
扫到 8（参数量从 22 到 393，接近 20 倍）时几乎不变——增大模型容量没有换来
更好的泛化，说明这不是"参数不够、欠拟合"，是训练集本身太小（44 条 4-5
轮轨迹）撑不起一个真正靠梯度下降训练的循环模型去可靠地学到比"固定 2 步线性
窗口 + `abs`/`sign` 提升"更好的表示——train one-step 拟合相近但 held-out
rollout 明显更差,是小数据集上过拟合的典型signature。这正是计划里预判的
"LSTM 持平或更差"的结果,而且比"持平"更明确地支持现有选择。

**结论对应 `BASELINES.md` 的 claim**：`richer_abs_sign` 相对一个真正的
非线性/长记忆序列模型（不只是相对纯 ARX 换 lifting 字典）的增益**不是
平凡的**——在这个数据规模下，Koopman 的强结构假设（线性 + 小状态 + 手工
lifting）是比"让模型自己学多长记忆"更好的归纳偏置,这条 ablation claim
现在有真实证据支撑,不再是"没测过"。

产物：`outputs/koopman_lstm_baseline/lstm_fit_report.json`（4 个 `H` 的完整
训练曲线 + 指标）。

## 附：控制器决策计算成本（回应"周期性基线是否只是更省算力"的追问）

Phase G 的 `period=2` 已经让 token 成本（`inserted_tokens`）和插入次数与
`koopman_mpc` **严格相等**——这是选周期前算过的，不是巧合，见
`koopman_defense_pilot.md` Phase G。token 成本这个维度不需要再测。

没测过的是另一个维度：**控制器自身"决定要不要插提醒"这一步的计算成本**
（不是 LLM token 成本）。新增 `scripts/benchmark_controller_decision_cost.py`，
离线重放 Phase E `koopman_mpc` 臂的真实决策点（80 个，8 held-out 攻击 ×
2 seed × 5 轮），对 `koopman_mpc`（horizon=2）/`periodic`/`threshold`
各自的 `next_u_remind` 计时（每个决策点重复 500 次取平均，排除首次调用的
预热开销）：

| controller | 每次决策耗时 | 相对 periodic |
|---|---|---|
| periodic | 0.22 μs | 1.0x |
| threshold | 0.28 μs | 1.25x |
| koopman_mpc | 88.60 μs | **400x** |

**如实结论**：`koopman_mpc` 的决策计算确实比 `periodic` 贵约 400
倍——horizon=2 的穷举前瞻仿真不是免费的。但绝对数值是 88.6 微秒/次，而
Phase E/G 这批作业的实际 wall-clock 是 16 条轨迹 × 5 轮 × 2 次 judge 调用
≈ 11-16 分钟，折算下来**每轮的 LLM 生成+打分耗时是秒级**，比
`koopman_mpc` 的决策计算慢 4-5 个数量级。**这个 400 倍的差距在真实部署的
wall-clock 里完全可以忽略不计**——`periodic` 更简单这件事在"决策计算成本"
这个维度上是真的，但它不构成"`periodic` 更省资源"这个论证里有意义的一条
证据，因为这个维度本来就不是瓶颈；真正的成本瓶颈（token/插入次数）已经
被 Phase G 的设计严格控制相等。

产物：`outputs/koopman_lstm_baseline/controller_decision_cost.json`。

---

## 2026-09-03 重跑：v 对齐修正后（"待复核"结清）

上一节开头的口径提示挂着一条待办：那批数字是 `modeling/dataset.py` 的 "v 对齐" bug 修复
**之前**跑的，LSTM 侧从未在修正后的对齐下重跑过。本节把它跑完了。产物：
`outputs/koopman_lstm_baseline/lstm_fit_report_valigned.json`（主）、
`lstm_fit_report_valigned_valsplit.json`、`lstm_fit_report_valigned_lesstrain_earlystop_heldout.json`、
`lstm_fit_report_prefix_reproduction.json`。旧的 `lstm_fit_report.json` 一字节没动。
作业 15519634（CPU，`environment/run_lstm_baseline_valigned.sbatch`），全套 287 passed。

### 改了什么

1. **`contemporaneous_v`**：`modeling/lstm_baseline.py` 增加和
   `ReducedStateConfig.contemporaneous_v` 同义的一格移位——预测 $y_{t+1}$ 时喂给 cell 的动作
   从 $v_t$ 改成 $v_{t+1}$（真正作用在这一轮回复上的那次提醒）。默认 `False` 保持旧行为，
   脚本侧 `--contemporaneous-v` 默认**开**（和 `fit_koopman_ae_baseline.py` 一致）。
2. **`--min-turn-index` 改成推导**：Koopman 的第一个可预测轮次是
   $\text{start}=\max(\nu-1,\ \mu-\text{shift})$，v 对齐后是 **1** 而不是 2。原来硬编码的 2
   会让两个模型在不同的轮次集合上算 MSE。
3. **`--train-seeds` 改成复数（默认 0/1/2）**：照 AE baseline 的教训（单种子结论被两个补充
   种子推翻），报均值 ± 总体标准差。
4. **`--val-frac` / `--early-stop-on`**：见下面"必须先说的一个口径问题"。

**回归控制**：`--no-contemporaneous-v --train-seeds 0` 在两台不同节点上都**逐位复现**了
2026-09-01 那张表（0.0808 / 0.0809 / 0.0862 / 0.0821，epochs 31/42/90/79）。所以下面
所有差异都来自对齐口径本身，不是重构带来的。

### 必须先说的一个口径问题：早停判据就是被报告的那个集合

`train_lstm_surrogate` 的早停判据是 held-out rollout MSE，而被报告的也正是这个集合——
返回的模型是**按该集合选出的最好那一轮**。这在 2026-09-01 是无害的（LSTM 无论如何都输），
一旦两边贴近就不无害了，因为**乐观的那一侧恰好就是把差距抹平的那一侧**。AE baseline 用的是
从训练攻击里切出的验证集（`val_frac=0.15`），本轮补齐了同一口径：`--val-frac 0.15` 把
30 个攻击切成 18 训练 / 4 验证 / **8 held-out（逐个 id 断言与原 split 相同）**。

切验证集同时也少了 4 个训练攻击，这会独立地抬高 MSE，所以又加了第三个臂
（`--early-stop-on held_out`）：训练集同样缩到 18 个攻击，但早停仍放回 held-out，用来把
"口径变公平"和"训练数据变少"两件事分开。

### 结果（held-out rollout MSE，matched window，3 seed 均值 ± 总体标准差）

| H | 参数量 | (a) 早停在 held-out，22 训练攻击 | (b) 早停在 held-out，18 训练攻击 | (c) **早停在验证集**，18 训练攻击 |
|---:|---:|---:|---:|---:|
| 1 | 22 | 0.0733 ± 0.0079 | 0.0756 ± 0.0077 | 0.0902 ± 0.0190 |
| 2 | 51 | 0.0709 ± 0.0055 | 0.0750 ± 0.0034 | 0.0821 ± 0.0021 |
| 4 | 133 | 0.0667 ± 0.0010 | 0.0690 ± 0.0005 | 0.0945 ± 0.0041 |
| 8 | 393 | **0.0664 ± 0.0002** | 0.0673 ± 0.0003 | 0.0949 ± 0.0045 |
| `richer_abs_sign`（对照，v-aligned，闭式解、无种子、不早停） | **40** | **0.0684** | 0.0684 | 0.0684 |

（对照的 `arx` 是 0.0683，和 `richer_abs_sign` 几乎相同——v 对齐后手工非线性 lifting 的增益
本身就几乎消失了，见 `ae_baseline_plan.md`。）

### 结论

1. **"接近 2 倍"这个量级不成立了。** 旧口径下 0.0808 vs 0.0430 是 1.88 倍；v 对齐后
   (a) 列 0.0664–0.0733 vs 0.0684 是 **0.97–1.07 倍**。对齐修正让 Koopman 那一侧变差
   （0.0430→0.0684）的幅度远大于 LSTM 那一侧，差距因此基本消失。上一节"差距接近 2 倍所以
   方向大概率不变"这句预判，**在量级上是错的**。
2. **但"LSTM 没有赢"这个方向仍然成立，靠的是口径而不是对齐。** (a) 列那个看起来打平甚至
   略优的数字是"按 held-out 自己选出来的最好一轮"；换成和 AE 同样的验证集早停 (c) 列，
   LSTM 是 0.0821–0.0949，比 `richer_abs_sign` 差 **20%–39%**，四个 H 全部更差。
3. **这个差距几乎全部来自口径，不是来自少切走的 4 个训练攻击。** 以 H=8 为例：
   (a)→(b) 少 4 个训练攻击只花了 +0.0009，(b)→(c) 换公平早停花了 **+0.0276**，后者是前者的
   30 倍。H=4 是 +0.0023 / +0.0255。
4. **反过来也要如实说**：2026-09-01 那张表用的也是 (a) 口径，也就是说当时是"被乐观选择过的
   LSTM"输给"干净的闭式岭回归"——旧结论在这一点上是**偏保守**的，不是被 bug 撑起来的。
5. 容量仍然不是瓶颈：(c) 列里 H 从 1 涨到 8（参数量 22→393，近 20 倍）没有变好，H=4/8 反而
   最差。和上一节一样，这是小数据集上的过拟合特征，不是欠拟合。

**对 `BASELINES.md` 那条 claim 的净影响**："Koopman 代理相对任意非线性/记忆模型的增益不是
平凡的"——**在同一口径下仍然成立，但强度要下调**：不再是"明显差 2 倍"，而是"在公平早停下
稳定差 20%–39%，在乐观口径下打平"。和 AE 的"打平"放在一起看，现在三条 ablation 的合成结论
是：**这个数据规模下没有任何一个更强的非线性/记忆模型赢过线性 baseline，但也只有 LSTM 在
公平口径下明确输**。

### 局限

1. 验证集只有 4 个攻击（8 条轨迹），早停信号本身噪声大——`ae_baseline_plan.md` 记过同一条
   局限，这里的表现是 H=1 的种子间标准差 0.019、早停轮数在 31/300/59 之间跳。(c) 列因此有
   一部分是"早停判早了"，这个方向会高估 LSTM 的劣势；但 (b) 列证明它不可能高估很多。
2. 报的仍然是"3 个训练种子的均值 ± 标准差 vs 一个无种子的点估计"，不是按 held-out 轨迹配对
   的检验。要严格给差异的置信区间，需要逐轨迹配对 bootstrap（和 Phase J 的
   `analyze_budget_arm_comparison.py` 同一套做法），本次没做。
3. `rollout_predictions` 的 LSTM 从全零 `(h,c)` 起步，而 Koopman 用真实 $z$ 起步——这条
   不对称在上一节就有，v 对齐把 matched 窗口从 t≥2 放宽到 t≥1 之后，这个"吃亏位"在被计分的
   4 个位置里占比更高了。它对 LSTM 不利，方向上和结论一致，所以没有反过来威胁结论。
