# 计划：LSTM 代理模型 baseline（补齐 `BASELINES.md` ③层缺口）

和 [koopman_detection_design.md](koopman_detection_design.md)/[koopman_defense_pilot.md](koopman_defense_pilot.md)
同一类"供跨会话接续"的记录。**新开一次对话想知道"LSTM baseline 计划到哪一步、要不要现在做"，
看这份文档。** 本文档只是计划,**尚未开始实现**。

## 目的：现在这个 claim 站不住脚，需要它来补

`docs/BASELINES.md` ③层把"线性 ARX / LSTM 一步-多步预测模型"列为
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
