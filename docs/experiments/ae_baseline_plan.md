# 计划：encoder-decoder（AE）Koopman baseline，对照 `core` 的 `DeepAugmentedKoopmanAutoencoder`

和 [lstm_baseline_plan.md](lstm_baseline_plan.md) 同一类"供跨会话接续"的记录。**新开一次对话
想知道"AE baseline 做到哪一步、结论是什么"，看这份文档。** 计划部分（下面到"资源预估"为止）
已执行完毕，**结果见文末"执行结果"一节**。

## 背景：为什么现在要补这个

`koopman_defense_pilot.md`/`koopman_case_study_design.md` 记录的对抗防御调查线已经在
2026-09-02 收尾（v-alignment 修正后，`koopman_mpc` 仍未打赢 `periodic`，架构层面的自适应性
问题已查清）。这条线用的代理模型（`modeling/koopman.py::KoopmanSurrogate`）从头到尾都是
**纯仿射 + 手工特征字典**（`richer_abs_sign` = `[y_hist, v_hist, abs(y), sign(y)]` 上的岭回归），
从未用过本仓库另一半——根目录 `src/koopman_ae/core.py` 里的**非线性 encoder/decoder**
（`DeepAugmentedKoopmanAutoencoder`）。`ABLATION_STUDY.md`（核心 Koopman-AE 任务自己的消融
研究）刚刚在标量/向量任务上系统对比过"AE 非线性提升 vs 纯线性 baseline"，结论是**任务相关、
没有跨任务通用答案**——8 个任务里 AE 赢 2-4 个、线性赢 2-3 个、1 个退化无信息量。这份文档把
同一个问题（"encoder-decoder 的非线性 lift 在这个具体任务上是否值得"）搬到对抗防御任务上，
为探索的完整性补一个缺口：**这里目前唯一测过的"更强模型能不能超过手工线性 lifting"的对比是
LSTM（`lstm_baseline_plan.md`，负结果，LSTM 明显更差），architecture 意义上更贴近 `core` 本身
设计思路的 AE 对比从未做过。**

需要说明：这条调查线本身已经收尾，这次补测不是要重新打开"该不该用 koopman_mpc"这个问题
（`periodic` 打平/略赢的结论不会因为换一个代理模型架构而改变——`periodic` 根本不用任何代理
模型），而是回答一个更窄的问题：**在 `richer_abs_sign`/ARX/LSTM 这几个已经测过的代理建模
选择之外，`core` 自己的 encoder-decoder 架构作为代理模型的预测质量（one-step/rollout MSE）
处在什么位置**——纯粹是代理建模层的消融，不涉及重新评估控制器/闭环效果。

## 设计上的关键决定：不新写评测代码，直接复用 `evaluate.py`

LSTM baseline 当时被迫**没有**复用 `modeling/evaluate.py::one_step_error`/`rollout_output_error`：
LSTM 的状态是 `(h, c)` 隐状态，形状和 `ReducedStateConfig` 的原始 `z_t`（`[y_hist, v_hist]`）
完全不同，`rollout_output_error` 用真实 `z_t` 播种轨迹起点这一步对 LSTM 不成立，只能另写
`rollout_predictions`/`teacher_forced_predictions`（见 `lstm_baseline.py`）。

`core.py` 的 `DeepAugmentedKoopmanAutoencoder` 不同：它的 `encode`/`decode` 是**同一个原始
状态空间的往返映射**（`z -> xi -> z_hat`，`z_hat` 和 `z` 同维度），线性动力学只发生在
`encode()` 之后的隐空间，`step()`/`predict_next_z()` 最终仍然吃、吐**原始 `z`**。这正好
满足 `modeling.evaluate.Predictor` 协议（`step(z, v) -> z_next`、`readout(z) -> y`，两者都是
在原始 `z` 空间上定义的）——`docs/method/koopman_surrogate.md`"已知缺口"一节写的"`Predictor`
协议已经为它留好了扩展点"指的就是这种情况。所以这次**新增的 `AEKoopmanSurrogate` 直接实现
`step`/`readout`，内部做 `encode -> 隐空间线性递推 -> decode`，外部行为和 `KoopmanSurrogate`
完全一样**，可以直接喂给 `fit_koopman_defense_model.py` 已经在用的
`one_step_error`/`rollout_output_error`，不需要像 LSTM 那样另写一套评测函数——这是这次和
LSTM baseline 最大的设计差异，也是更贴近 `core` 原始设计意图的地方。

`readout(z)` 的具体实现：`core.py::DeepAugmentedKoopmanAutoencoder.predict_y` 不学一个单独的
读出矩阵，而是直接从（预测出的）`z` 里切前 `output_dim` 维——因为 `AugmentedStateConfig` 的
状态设计本来就把 `y` 放在 `z` 的最前面。`ReducedStateConfig` 在 `nu=1` 时同样成立：`z[0]` 就是
`y_t` 本身（`koopman.py::abs_sign_extra_features` 已经用了这个假设）。这次沿用 `core` 的
写法：`readout(z) = float(z[0])`，不额外学一个读出层——这是"多大程度上照搬 `core` 的方法"这个
问题上刻意做出的选择，读出方式本身也是 `core` 架构的一部分，不是可以自由替换的实现细节。
**前提：只在 `nu=1` 下有效**，和 `richer_abs_sign`/LSTM baseline 一致，这次同样固定 `nu=1`。

## 训练方式：照搬 `core` 的 `reconstruction_then_ridge`，不用 `joint`

`ABLATION_STUDY.md` 第三阶段（本仓库自己的消融，`sentence_length_t10` 任务）刚测出
`reconstruction_then_ridge` 比 `joint` 稳健得多（rollout MSE 低 46%，种子间方差低 4 倍）。
这次直接采用这个已有证据支持的训练方式，不重新在对抗防御这个更小的数据集上验证 `joint` 是否
更差——两个任务数据规模差两个数量级，`joint` 在更小数据上过拟合风险只会更高，没有理由在这里
反而选一个已经被自己的消融数据否定的训练方式。具体流程完全照抄
`core.py::DeepAugmentedKoopmanAutoencoder.fit`（`training_mode="reconstruction_then_ridge"`
分支）：

1. **阶段一（重建）**：只训练 encoder/decoder，loss = `MSE(decoder(encoder(Z_t)), Z_t)`
   （`lambda_rec=1.0`，`core` 默认），`torch.optim.AdamW`，固定 `num_epochs` 轮（不做早停——
   `core.py` 本身在这个训练模式下就是跑满 `num_epochs`，这次不额外发明一个 `core` 没有的早停
   机制，保持"照搬 `core` 的方法"这个约束）。
2. **阶段二（岭回归拟合动力学）**：冻结 encoder，对 `xi_t=encode(Z_t)`、`xi_next=encode(Z_next)`
   做闭式岭回归 `xi_next ≈ K xi_t + B v_t + c`（`core.py::_fit_latent_ridge` 原样搬过来，
   `dynamics_alpha` 对应 `core` 的同名超参，默认 `1e-4`）。

## 接口设计

新增 `persona_drift_control/src/persona_drift/modeling/ae_baseline.py`：

- `AEKoopmanConfig`（dataclass）：`latent_dim`、`hidden_dim`、`num_layers`、`activation`、
  `learning_rate`、`weight_decay`、`num_epochs`、`dynamics_alpha`、`random_state`——字段名和
  语义直接对应 `core.py::DeepAugmentedKoopmanConfig` 的子集（去掉 `joint` 模式独有的
  `lambda_pred`/`lambda_latent`/`lambda_multi`/`multi_step_horizon`/`batch_size`/`device`，
  这次不用 `joint`、数据集小到不需要按 batch 训练、CPU 足够）。
- `AEKoopmanSurrogate`：`encoder`/`decoder` 两个 MLP（复用 `core.py::_make_mlp` 同款结构，
  独立实现一份而不是跨包 import——和 `koopman.py::controllability_diagnostics` 当年的理由
  一样，见 `docs/method/koopman_surrogate.md`"为什么不直接复用 core.py"一节：两个子项目故意
  保持独立可安装，数值工具重复十几行比引入包间依赖划算）：
  - `fit(dataset)`：`dataset` 是 `modeling.dataset.build_identification_dataset` 的输出
    （和 `KoopmanSurrogate.fit` 吃同一个数据结构），内部跑上面两阶段训练。
  - `step(z, v) -> z_next`、`readout(z) -> y`：满足 `Predictor` 协议，直接可喂
    `evaluate.one_step_error`/`rollout_output_error`。
  - `encode`/`decode`：暴露出来供诊断/调试使用，行为对齐 `core.py` 的同名方法。

## 参数量对齐（如实记录，不追求相等）

`richer_abs_sign` 约 40 个参数（`lstm_baseline_plan.md` 已算过）。`core.py` 默认
`hidden_dim=64, num_layers=2, latent_dim=16` 对一个只有 3 维（`nu=1,mu=2`）原始状态的任务
明显过大，必然在几十条轨迹上过拟合——这次不用默认值，改用能覆盖 `state_dim=3` 的最小配置：
`hidden_dim=4, num_layers=1`，扫 `latent_dim ∈ {1, 2, 4}`。三档参数量（encoder+decoder+K+B+c）
分别约为 47 / 63 / 95——仍然比 `richer_abs_sign` 的 40 多，但量级接近（LSTM 当年最小档也要
22-40+，同样做不到完全相等），报告里明确写出来，不藏起来。

## 数据规模的现实约束

和 `lstm_baseline_plan.md` 完全一样的先天局限：Phase B 只有 30 个攻击 × 2 seed = 60 条轨迹、
`nu=1,mu=2` 起算后每条产出的 transition pair 更少，identification 数据在几十到一百多条的
量级。**预期结果大概率和 LSTM 类似（持平或更差）**——线性 + 手工 lifting 在这个规模下是合理
的强归纳偏置，这不是这次实验的失败,是有信息量的负结果（进一步确认，而不是重复验证同一件
事——AE 和 LSTM 是两种不同的"更强非线性模型"，即使结论方向相同,也排除了"只是 LSTM 这一种
架构选择不合适"这个混杂解释）。

## 评测口径

和 `richer_abs_sign`/ARX 完全一样，直接复用 `fit_koopman_defense_model.py` 已经在用的
`one_step_error`（train split）/`rollout_output_error`（held-out split），同一个 `nu=1,mu=2`、
同一个 `attack_id` 75/25 split（`seed=0`）、同一份 `y_col="y_safety"`。**用
`contemporaneous_v=True`（v-alignment 修正后的当前正确配置，见
`koopman_case_study_design.md`"Phase I"）而不是最早的 `koopman_fit_report.json`**——2026-09-02
的修正已经确认原始配对方式时序错位，继续拿旧配置当参照会把一个已知 bug 的产物当作基准,不
科学。对照数字来自 `koopman_fit_report_valigned.json`：`richer_abs_sign`
`train_one_step_mse=0.05398`、`held_out_rollout_mse=0.06844`（`arx` 几乎相同，
`0.05399`/`0.06835`——v-alignment 修正后手工非线性 lifting 的增益本身也几乎消失了，这本身
就是这次 AE 对比要面对的一个新背景：**当前"对照组"已经不是一个明显跑赢 ARX 的强 baseline**）。

## 成功判据 / 这次实验想回答的问题

- AE held-out rollout MSE 明显好于 `richer_abs_sign`/`arx` 的 0.068（且缩小 `latent_dim` 后
  优势仍在）→ 说明这个任务上非线性 encoder/decoder 确实能学到线性 lifting 学不到的结构，
  值得作为未来重新评估这条线时的候选代理模型。
- 持平或更差 → 和 LSTM 的结论一致方向，进一步坐实"这个数据规模下强线性结构假设合理"，也
  回应了 `ABLATION_STUDY.md` 里"AE vs 线性是任务相关的"这个结论在对抗防御任务上具体落在
  哪一边。

## 明确不在这次计划范围内的

- 不重新评估 `koopman_mpc` 控制器闭环效果，不接入 `KoopmanMPCController`——这条线已经收尾，
  这次只是代理建模层的补充对比，和 `lstm_baseline_plan.md` 当年划的范围一致。
- 不测 `joint` 训练模式——已在上面"训练方式"一节说明理由。
- 不做 `latent_dim`/`hidden_dim` 的完整网格搜索——只扫 `latent_dim`,`hidden_dim` 固定小值,
  和 LSTM baseline 只扫 `hidden_size` 一个维度的克制程度一致。

## 资源预估

纯 CPU 可行：3 维原始状态、`hidden_dim=4`、几十到一百多条训练样本，训练几秒到十几秒一档,
和 `fit_koopman_lstm_baseline.py` 同量级,不需要 GPU/sbatch。唯一依赖 `torch`,已是既有依赖。

---

## 执行结果（2026-09-02）

新增 `src/persona_drift/modeling/ae_baseline.py`（`AEKoopmanConfig` + `AEKoopmanSurrogate`）、
`scripts/fit_koopman_ae_baseline.py`、`tests/test_ae_baseline.py`（5 passed）。设计按计划实现：
`step`/`readout` 直接吃/吐原始 `z`，`fit_koopman_ae_baseline.py` 不写任何新评测代码，直接调用
`modeling.evaluate.one_step_error`/`rollout_output_error`——验证了这次和 LSTM baseline 最大的
设计差异是可行的。

**训练**：Phase B 的 22 训练攻击/44 条轨迹、8 held-out 攻击/16 条轨迹（和 `richer_abs_sign`
`valigned` 版本完全相同的 split），`nu=1, mu=2, contemporaneous_v=True`（`state_dim=3`），
`hidden_dim=4, num_layers=1`，`num_epochs=300`，扫 `latent_dim ∈ {1, 2, 4}` × `train_seed ∈
{0, 1, 2}`（原计划只扫 `latent_dim` 一个维度；latent_dim=2 首个种子的 held-out rollout MSE
看起来略优于 `richer_abs_sign` 后，补种子扫描确认不是噪声——和 `lstm_baseline_plan.md`"如果
LSTM 明显赢，补一次…消融"同一条审慎原则,这次触发条件是"看起来赢"而不是"看起来输"）。

### 结果（3 seed 均值 ± 总体标准差）

| latent_dim | 参数量 | train one-step MSE | held-out rollout MSE |
|---:|---:|---:|---:|
| 1 | 47 | 0.5380 ± 0.0406 | 0.0675 ± 0.0072 |
| 2 | 61 | 0.4136 ± 0.1333 | 0.0675 ± 0.0070 |
| 4 | 95 | 0.4412 ± 0.0973 | 0.0675 ± 0.0054 |
| `richer_abs_sign`（对照，v-aligned，无随机性） | **40** | **0.0540** | **0.0684** |
| `arx`（对照，v-aligned，无随机性） | 40 | 0.0540 | 0.0683 |

### 结论

1. **held-out rollout MSE 上，AE 和线性 baseline 基本打平——不是"略赢"也不是"略输"**：三个
   `latent_dim` 的均值几乎完全相同（0.0675，种子间标准差 0.005–0.007），`richer_abs_sign`/`arx`
   的 0.068 落在这个分布的正中间，差距远小于一个种子标准差。第一个种子（`train_seed=0`）
   `latent_dim=2` 那次 0.0613 一度看起来"赢了"，补两个种子后打回原形——这正是这次没有跳过种子
   扫描的原因。**这是三个代理建模 baseline（ARX/`richer_abs_sign` vs LSTM vs AE）里第一个
   "打平"的结果，不同于 LSTM 的"明确更差"**——用两种不同的非线性模型类别（自由循环记忆 vs
   非线性 lifting + 线性隐空间动力学）分别验证过，排除了"只是 LSTM 这一种架构选择不适合"这个
   混杂解释：在这个数据规模下，"非线性能不能帮上忙"这个问题的答案更接近"帮不上也不添乱"，
   而不是单一模型的偶然结果。
2. **`latent_dim` 在 1–4 这个区间同样不敏感**（三档 held-out 均值几乎相等），和
   `ABLATION_STUDY.md` 第四阶段（core 任务上 `latent_dim` 8–32 同样不敏感）的结论方向一致——
   两个完全不同的任务上，都观察到"隐空间维度在小范围内不是关键超参数"。
3. **train one-step MSE 上 AE 明显更差（0.41–0.54 vs 0.054，约 8–10 倍），但这个对比本身不公平，
   不能反向读成"AE 拟合能力更弱"**：`richer_abs_sign`/`arx` 的 `A/B/b` 是**直接**对原始 `z` 空间
   一步预测误差做闭式岭回归的解——这个 MSE 数字就是它们的训练目标本身，理论下界。AE 用
   `reconstruction_then_ridge`（照抄 `core.py`）：`K/B/c` 的岭回归解minimizes 的是**隐空间**里
   `xi_next` 的误差，配合一个有损的 `decode(encode(z))` 往返，两者叠加后的 raw-z-space one-step
   MSE 从未被直接优化过——`train_one_step_mse` 对 AE 而言是一个"训练目标的下游副产物"，对
   `richer_abs_sign` 而言就是训练目标本身。**这解释了为什么 one-step 上差距巨大、但 rollout
   上打平**：rollout 表现由"隐空间动力学是否学到了有用的长期结构"决定，不直接受限于原始状态
   空间的重建精度。这是这次实验一个值得记录的架构性观察，不是负结果也不是正结果，是"两类模型
   在同一个数字上不可比"的方法论提醒——后续如果还要新增第三类代理模型，应该先确认它的训练目标
   和 `one_step_error` 测的是不是同一件事,不能默认可比。
4. **和 LSTM baseline 合并看**：现在有三个独立的"更强模型能否超过线性 lifting"的对照
   （ARX vs `richer_abs_sign` 内部已经几乎打平、LSTM 明确更差、AE 打平），没有一个在这批 Phase B
   数据规模下明确赢过线性 baseline。`docs/BASELINES.md` 第③层"Koopman 代理相对任意非线性/记忆
   模型的增益不是平凡的"这条 claim，现在有三条独立证据支撑（不再只有 LSTM 一条）。
5. **和 `ABLATION_STUDY.md`（core 任务）对照**：那边"AE vs 线性"结论是任务相关、没有统一方向
   （8 个任务里有明确 AE 赢、也有明确线性赢的）。对抗防御这个任务落在"打平"这个中间地带——不是
   矛盾，是给"任务相关"这个结论又加了一个具体样本点：**数据规模极小（这里 44 条训练轨迹 vs core
   任务动辄几百上千条样本）时，AE 相对线性的潜在优势和劣势可能都被压缩到噪声量级里，"打平"本身
   可能就是小数据规模的特征，而不是这个任务的特有属性**——这是一个假设,不是本次实验能直接验证
   的结论,留给以后有更大规模开环激励数据时重新检验。

产物：`outputs/koopman_ae_baseline/ae_fit_report.json`（`train_seed=0`）、
`ae_fit_report_seed1.json`、`ae_fit_report_seed2.json`。

## 明确的局限（如实记录）

- **重建 loss 在 300 epoch 后还没有明显收敛到接近零**（最终 reconstruction loss 在 0.09–0.17
  之间，`z` 各分量量级在 0–1），说明 encoder/decoder 本身没有学到一个高精度的恒等映射式往返——
  这也是"train one-step MSE 差距巨大"的一部分原因（不完全是"目标不同"，encode/decode 的重建
  误差本身也不小）。这次没有加长训练轮数或调学习率去把它压得更低，因为最终关心的 rollout 指标
  已经打平,进一步压低重建 loss 是否会改变 rollout 结论没有验证过,留作后续如果要更精细地
  区分"打平是真的打平"还是"AE 还没训好、有更大潜力"时的下一步。
- 只在 `hidden_dim=4, num_layers=1` 这一档 MLP 容量下测过，没有像 `latent_dim` 一样做容量扫描——
  和 LSTM baseline 当年"不追求完全公平，把混杂因素摆在台面上"是同一个取舍。
