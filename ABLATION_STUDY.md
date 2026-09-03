# 核心 Koopman-AE 任务的消融研究

## 背景与动机

`persona_drift_control/` 那条线（人格漂移 → 对抗防御）已经做过多轮消融：LSTM 代理模型 baseline
（`docs/experiments/lstm_baseline_plan.md`）、周期性提醒 baseline（Phase G）、检测方案 1/3/4 的状态
特征消融、state-action interaction 消融。相比之下，本仓库最早、也是默认起点的**核心
Autoencoder–Koopman 任务**（`sentence_length_t10` 句子长度控制，以及另外 7 个标量/向量任务）
从未有过任何一份消融记录：`results/` 此前只有 `.gitkeep`，`slurm/sweep_example.txt` 只是扫描
命令的模板从未被执行归档，`CODE_DESIGN.md` 里"`reconstruction_then_ridge` 通常更稳健"这类判断
也没有配套数据。

本文件记录这条空白的消融计划，以及已经跑完的部分。**只使用现有 CLI/库代码做不同配置组合的对比
跑，不修改 `src/koopman_ae/core.py`、`scripts/train.py` 或任何已有实验的产物**；新脚本一律作为
独立文件添加，不改动既有文件。

## 待验证的模块与优先级

| 优先级 | 模块 | 待验证问题 | 状态 |
|---|---|---|---|
| 1 | 状态定义（Markov / Memory-L / Augmented-L） | 加历史记忆、加控制误差历史是否真的提升预测——这是框架"需要惯性建模"这一核心主张的直接检验 | **已完成，见下** |
| 2 | AE 非线性提升 vs 直接线性 Koopman baseline | `DeepAugmentedKoopmanAutoencoder` 的非线性 encoder/decoder 相对 `AugmentedKoopmanModel`（仓库里已有但从未接入 CLI 的纯仿射 baseline）是否有增益——对应"为什么要用 Koopman"的核心论点 | **已完成，见下** |
| 3 | 训练方式（joint vs reconstruction_then_ridge） | 把 `CODE_DESIGN.md` 里的定性判断换成数据 | **已完成，见下** |
| 4a | `latent_dim` 扫描（8/16/32） | 是否存在明显更优的容量 | **已完成，见下** |
| 4b | loss 权重（`lambda_latent`、`lambda_multi`）、joint + 多步 rollout loss 组合 | 调参类问题，优先级最低，尚未做 | 待做 |

其他 7 个标量/向量任务（`character_length_t5` 等）优先级低于 `sentence_length_t10`——它是文档里
唯一被称为"主要长时域标量实验"的任务，其余几个要么是 smoke 规模、要么受 scorer/readout 质量限制
（`CODE_DESIGN.md`"Known Limitations"一节已自陈）。

## 实验环境与复现性说明

- 复用 `persona_drift_control` 现有的 `/scratch/hcao2/envs/persona_drift_pilot` conda 环境
  （numpy/pandas/torch/hydra 版本均已满足 `pyproject.toml` 的依赖下限）。
- 用 `pip install --no-deps -e .` 把本包装进该环境，**不解析/不升级任何已装依赖**，避免影响
  该环境里正在跑的对抗防御实验。
- checkpoint 写到单独隔离的 `/scratch/hcao2/checkpoints/autoencoder_koopman_core_ablation/`，
  与任何既有 checkpoint 目录都不重名，不会覆盖任何已有产物。
- 所有跑数均为 CPU、`reconstruction_then_ridge`，单次 30–40 秒，不需要占用 GPU 资源。

## 第一阶段：状态定义消融（已完成）

### 设计

固定其余超参数为 README 的"推荐起点"配置（`sentence_length_t10`、
`training_mode=reconstruction_then_ridge`、`latent_dim=16`、`epochs=200`），只切换
`state=markov / memory(lag=3) / augmented(lag=3, control_mode=error)`，每种配置跑
`seed=0,1,2` 三个种子。命令：

```bash
python scripts/train.py dataset=sentence_length_t10 state=<markov|memory|augmented> \
  state.lag=3 model.training_mode=reconstruction_then_ridge model.latent_dim=16 \
  trainer.epochs=200 trainer.device=cpu trainer.seed=<0|1|2> \
  trainer.checkpoint_root=/scratch/hcao2/checkpoints/autoencoder_koopman_core_ablation
```

结果落在 `results/sentence_length_t10-<family>-lag<L>-reconstruction_then_ridge-k16-seed<N>/run.json`。

### 结果（test split，y 空间指标，3 seed 均值 ± 总体标准差）

| 状态定义 | one_step_mse | rollout_mse |
|---|---:|---:|
| markov（无记忆） | 0.002357 ± 0.000018 | 0.002309 ± 0.000031 |
| memory-3（历史观测） | 0.000810 ± 0.000014 | **0.000941 ± 0.000021** |
| augmented-3（历史观测 + 控制误差历史） | 0.000839 ± 0.000021 | 0.001183 ± 0.000254 |

### 结论

1. **惯性/记忆假设成立**：memory / augmented 相对 markov 的 rollout MSE 低约 2.5 倍
   （0.00094–0.00118 vs 0.00231），3 个种子方向完全一致，不是噪声——句子长度这个 readout 确实
   有跨轮记忆，不是纯 Markov 过程，Koopman 框架"提升到延迟嵌入状态"这一核心设计对这个任务是
   有效的。
2. **加控制历史（augmented）在当前配置下没有帮助，反而更不稳定**：augmented 的 one-step MSE
   与 memory 持平，但 rollout MSE 明显更差（0.00118 vs 0.00094，高 26%）且种子间方差大一个数量
   级（sd 0.000254 vs 0.000021）。说明在这个任务/超参数下，多步 rollout 时把控制误差历史也塞进
   状态反而引入了额外噪声源，没有换来预测收益——`state=memory` 应该是这个任务更稳的默认选择，
   而不是 README 示例 2 暗示的 `augmented`。
3. 这个结论目前只在 `latent_dim=16`、`reconstruction_then_ridge` 下验证；是否在 `joint` 训练
   或不同 `latent_dim` 下同样成立，留给第二/三阶段。

## 第二阶段：AE 非线性提升 vs 纯线性 Koopman baseline（已完成）

### 设计

新增独立脚本 `scripts/ablation_linear_baseline.py`（不改动 `core.py`/`scripts/train.py`），调用
库里已有但从未接入 CLI 的 `AugmentedKoopmanModel`（纯仿射 `z_(t+1) = A z_t + B r + c`，无 AE
非线性 encoder/decoder），在与第一阶段完全相同的 `state=memory, lag=3` 数据切分和评测函数
（`one_step_predictions` / `rollout_augmented_from_trajectories`，与 `scripts/train.py::_evaluate`
用的是同一套库函数）上拟合、评测，与第一阶段 memory 的结果直接对比。`AugmentedKoopmanModel`
的拟合是闭式岭回归，无随机性，因此不需要多 seed。

```bash
python scripts/ablation_linear_baseline.py
```

### 结果（test split，y 空间指标）

| 模型 | one_step_mse | rollout_mse |
|---|---:|---:|
| 纯线性 Koopman（`AugmentedKoopmanModel`，无 AE） | 0.000758 | 0.000880 |
| AE + 岭回归（第一阶段 memory-3，k=16，3 seed 均值） | 0.000810 ± 0.000014 | 0.000941 ± 0.000021 |

### 结论

**AE 的非线性提升目前没有带来增益，纯线性 baseline 反而略优**（rollout_mse 低约 6%，
one_step_mse 低约 6%，差距在 3 个 seed 标准差以上，不是噪声）。也就是说在
`sentence_length_t10` + `memory-3` + `latent_dim=16` 这个配置下，把 4 维延迟嵌入状态
再升维到 16 维隐空间做线性动力学，并不比直接在原始 4 维状态上做仿射拟合更好——"为什么要用
Koopman 的非线性 lift"这一核心论点，在当前默认超参数下**没有被这次实验支持**。这不代表
AE 在其他配置（更小/更大 `latent_dim`、`joint` 训练、多变量任务）下同样无效，但作为默认起点
配置的第一份对照数据，这是一个需要正视的负结果。

## 第三阶段：训练方式消融（joint vs reconstruction_then_ridge，已完成）

### 设计

固定第一阶段胜出的状态定义 `state=memory, lag=3`，`latent_dim=16`、其余 loss 权重维持默认
（`lambda_multi=0`，即 joint 模式下不额外开多步 rollout loss——避免同时改两个变量），只切换
`model.training_mode`，3 个种子：

```bash
python scripts/train.py dataset=sentence_length_t10 state=memory state.lag=3 \
  model.training_mode=joint model.latent_dim=16 trainer.epochs=200 trainer.device=cpu \
  trainer.seed=<0|1|2> \
  trainer.checkpoint_root=/scratch/hcao2/checkpoints/autoencoder_koopman_core_ablation
```

### 结果（test split，y 空间指标，3 seed 均值 ± 总体标准差）

| 训练方式 | one_step_mse | rollout_mse |
|---|---:|---:|
| `reconstruction_then_ridge`（第一阶段 memory-3） | 0.000810 ± 0.000014 | 0.000941 ± 0.000021 |
| `joint` | 0.000817 ± 0.000015 | 0.001371 ± 0.000084 |

### 结论

单步预测两者相当，但**多步 rollout 上 `joint` 明显更差且更不稳定**：rollout_mse 高约 46%，
种子间标准差高约 4 倍。这把 `CODE_DESIGN.md` 里"`reconstruction_then_ridge` 通常更稳健"这句
未经数据支持的判断，换成了具体证据——至少在默认权重（`lambda_multi=0`，joint 没有专门优化
多步 rollout 的信号）下成立。**注意**：这不是对 `joint` 模式本身的最终判决——`joint` 的一个
卖点正是可以打开 `lambda_multi`/`multi_step_horizon` 直接优化多步 rollout loss（README 的
"Joint Augmented-L3"示例就是这样配的），这次没测；把训练方式和 loss 权重两个变量分开消融
正是第四阶段（b）要做的事。

## 第四阶段（a）：`latent_dim` 扫描（已完成）

固定 `state=memory, lag=3`、`reconstruction_then_ridge`，扫 `latent_dim ∈ {8, 16, 32}`，3 个
种子：

| `latent_dim` | one_step_mse | rollout_mse |
|---:|---:|---:|
| 8 | 0.000793 ± 0.000025 | 0.000984 ± 0.000055 |
| 16 | 0.000810 ± 0.000014 | 0.000941 ± 0.000021 |
| 32 | 0.000788 ± 0.000027 | 0.000916 ± 0.000034 |

三者差异都在彼此的种子标准差量级内，没有明显的单调趋势或最优点——**在 8–32 这个区间，
`latent_dim` 不是敏感超参数**，默认值 16 没有问题，但也谈不上被证明是最优选择。结合第二阶段
的结果（纯线性、也就是"`latent_dim` 等价于原始 4 维状态、没有非线性 lift"反而更好），这进一步
削弱了"更大的隐空间维度能带来增益"这个假设。

## 第五阶段：其余 7 个任务上重复"AE vs 纯线性"对照（已完成）

### 设计

第二阶段的"线性 baseline 反而略优"只在 `sentence_length_t10` 一个任务上验证过。本阶段把同一套
对照方法（`state=memory, lag=3`，`latent_dim=16`，`reconstruction_then_ridge`，AE 侧 3 个种子
`seed=0,1,2`，线性侧 `AugmentedKoopmanModel` 闭式岭回归、无随机性）铺到其余 7 个已注册任务
（`average_word_length_t5`、`character_length_t5`、`even_odd_t5`、`formality_t5`、
`sentiment_t5`、`vector_count_stage1_t10`、`vector_count_stage2_t10`），检验"线性优于 AE"是
`sentence_length_t10` 特有还是普遍现象。

新增独立脚本 `scripts/ablation_linear_baseline_all_tasks.py`（`ablation_linear_baseline.py` 的
参数化版本，从 `configs/dataset/*.yaml` 读路径/列名，不改动原脚本），AE 侧复用
`scripts/train.py` 与第一/二阶段完全相同的调用方式，只切换 `dataset=<task>`：

```bash
python scripts/train.py dataset=<task> state=memory state.lag=3 \
  model.training_mode=reconstruction_then_ridge model.latent_dim=16 \
  trainer.epochs=200 trainer.device=cpu trainer.seed=<0|1|2> \
  trainer.checkpoint_root=/scratch/hcao2/checkpoints/autoencoder_koopman_core_ablation
python scripts/ablation_linear_baseline_all_tasks.py
```

### 结果（test split，y 空间 rollout_mse，AE 为 3 seed 均值 ± 总体标准差；`test n` 是该任务的
**rollout 评测点数**，即 test 轨迹数 × 每条可 rollout 的轮数，**不是** test split 的行数、
也不是轨迹数。例如 `sentiment_t5` 的 test split 是 10 条轨迹/50 行，T=5 且
`common_seed_turns=4` 只剩 1 轮可 rollout，所以 `test n`=10；`sentence_length_t10` 是
42 条轨迹 × 6 轮=252。标注它是因为几个 T5 任务的评测点只有个位数到几十个，`CODE_DESIGN.md`
"Known Limitations" 已自陈这类任务受数据规模限制，此处结果置信度相应更低。下面结论 4 讨论
"数据规模"时用的就是这一列，读作评测点数而不是轨迹数）

| 任务 | test n | AE rollout_mse | 纯线性 rollout_mse | 胜者 | 差距（相对 AE） |
|---|---:|---:|---:|---|---:|
| average_word_length_t5 | 14 | 0.002138 ± 0.000081 | 0.003092 | **AE**（~12 倍 sd，明确） | 线性差 44.6% |
| character_length_t5 | 60 | 0.002682 ± 0.000376 | 0.003135 | AE（~1.2 倍 sd，弱） | 线性差 16.9% |
| even_odd_t5 | 4 | ~0.000000 ± 0.000000 | ~0.000000 | 平局（任务退化，两者近乎零误差，无信息量） | — |
| formality_t5 | 42 | 0.006543 ± 0.000128 | 0.006863 | AE（~2.5 倍 sd，中等） | 线性差 4.9% |
| sentiment_t5 | 10 | 0.009925 ± 0.000561 | 0.005572 | **线性**（~7.8 倍 sd，明确） | AE 差 78.1% |
| vector_count_stage1_t10 | 288 | 0.006207 ± 0.000300 | 0.005743 | 线性（~1.5 倍 sd，弱） | AE 差 8.1% |
| vector_count_stage2_t10 | 540 | 0.006783 ± 0.000066 | 0.006685 | 线性（~1.5 倍 sd，但 sd 很小、方向一致） | AE 差 1.5% |
| （参照）sentence_length_t10（第二阶段） | 大 | 0.000941 ± 0.000021 | 0.000880 | 线性（~2.9 倍 sd） | AE 差 6.5% |

### 结论

**不是"线性普遍优于 AE"，而是结果按任务分裂，没有一致方向**——这纠正了只看
`sentence_length_t10` 一个任务时容易得出的过度概括：

1. **一个任务上 AE 明确更好**：`average_word_length_t5`（差距达 12 倍种子标准差，是本次 8 个
   任务里最大的效应量，也是除下面 `sentiment_t5` 之外唯一一个方向性明确的案例）。
   `character_length_t5`、`formality_t5` 上 AE 也占优但效应量较弱（1.2–2.5 倍 sd）。
   （**第八阶段修正**：`average_word_length_t5` 这个效应量在打开早停后反转成打平，见第八阶段——
   本条现在只剩"弱/中等的 AE 占优"，一个明确案例都不剩。）
2. **`sentiment_t5` 上线性明显更好**（AE 误差高 78%，是本次除 `average_word_length_t5` 外效应
   量最大的一项），叠加 `sentence_length_t10`、两个 `vector_count` 任务上线性的弱/中等优势，
   "线性 baseline 有竞争力"这个第二阶段的结论在多任务上依然基本成立，但优势幅度普遍从
   sentence_length_t10 的 6.5% 稀释到 1.5–8%，且不再是唯一模式。
3. `even_odd_t5` 完全退化（两个模型 test rollout_mse 都约等于 0，test n 仅 4 行），对"AE vs 线性"
   这个问题没有提供任何信息，不计入上面的方向统计。
4. **数据规模是明显的混杂变量**：test n 越小（`even_odd_t5`=4、`sentiment_t5`=10、
   `average_word_length_t5`=14）方向和效应量看起来越极端，n 较大的两个 vector 任务
   （288、540）反而效应量最小且最一致——这与"小样本 T5 任务结果噪声大"的既有认知
   （`CODE_DESIGN.md`）吻合，`sentiment_t5`、`average_word_length_t5` 这两个最大效应量的结果
   需要更多数据或更多种子才能确认不是采样噪声，不能直接当作任务本身的稳定属性。
5. 综合本阶段与第二阶段：**"encoder-decoder 的非线性 lift 是否值得"这个问题目前没有跨任务
   通用答案，是任务相关的**（task-dependent），且现有证据的置信度受限于几个任务的小样本量。

## 第六阶段：检验"rollout horizon 长度决定 AE-vs-线性胜负"这个假设（已完成，假设未被支持）

### 动机

第五阶段的原始读法很诱人："T=5 的任务（`common_seed_turns=4` 时只剩 1 步可 rollout）里 AE 多数
赢，T=10 的三个任务（6 步真·多步 rollout）里线性全赢"，看起来像是 rollout 步数越长、AE 隐空间
线性动力学 `K` 的复合误差就越吃亏。但 T=5 vs T=10 这个划分和"task 是哪个任务"是完全绑定的混杂
变量——光看第五阶段的数据，无法把"horizon 长"和"这就是 sentiment_t5/vector_count 这些任务本身
的特性"区分开。本阶段设计了两个探针实验，**固定任务和 state 定义不变，只改 rollout horizon**，
直接检验这个假设，而不是停留在相关性猜测上。

### 探针 A：`average_word_length_t5` + `state=memory,lag=1`（新增脚本 `ablation_horizon_probe.py`）

第五阶段用的是 `lag=3`，但 `lag=3` 要求至少 4 个观测轮次做种子状态，T=5 的数据只剩 1 轮可 rollout
（`common_seed_turns` 没法小于 4），结构上无法在 `lag=3` 下把 horizon 拉长。退而求其次，把 lag
降到 1（种子轮次最小可到 2），这样能在同一个任务、同一个 T=5 数据集上比较 `common_seed_turns=4`
（horizon=1）vs `common_seed_turns=2`（horizon=3）：

| horizon | seed_turns | AE rollout_mse（3 seed） | 线性 rollout_mse | 胜者 | 差距 |
|---:|---:|---:|---:|---|---:|
| 1 | 4 | 0.001867 ± 0.000010 | 0.001902 | AE | 线性差 1.9% |
| 3 | 2 | 0.003566 ± 0.000018 | 0.003703 | AE | 线性差 3.8% |

horizon 从 1 拉到 3，AE 的优势不但没有收窄反而略微扩大——**方向和第五阶段"horizon 越长线性
越占优"的直觉相反**。但要注意这里换成了 `lag=1`（state 维度从 4 降到 2），AE 的优势幅度本身也从
第五阶段 `lag=3` 时的 44.6% 骤降到不到 4%——说明 `lag` 本身对这个任务的 AE-vs-线性差距影响
可能比 horizon 更大，这个探针没能把 lag 和 horizon 完全解耦。

### 探针 B：`sentence_length_t10` + `state=memory,lag=3`（新增脚本 `ablation_horizon_sweep_t10.py`）

T=10 的数据允许 `common_seed_turns` 在 4–9 之间取值，而不用换 lag，这样能在**和第二阶段完全
一致的 `lag=3` state 定义**下扫 horizon，是比探针 A 更干净的对照（任务、state 定义都不变，
只变 horizon）：

| horizon | seed_turns | AE rollout_mse（3 seed） | 线性 rollout_mse | test n | 胜者 | 差距（线性 vs AE） |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 9 | 0.000418 ± 0.000009 | 0.000429 | 42 | AE | −2.8% |
| 2 | 8 | 0.000689 ± 0.000015 | 0.000642 | 84 | 线性 | +6.8% |
| 4 | 6 | 0.000629 ± 0.000015 | 0.000637 | 168 | AE | −1.2% |
| 6 | 4 | 0.000941 ± 0.000021 | 0.000880 | 252 | 线性（第二阶段原始结果） | +6.5% |

四个 horizon 上胜负交替（AE、线性、AE、线性），**不是随 horizon 单调变化的趋势**——如果
"horizon 越长线性越占优"成立，应该看到差距随 horizon 增大单调地从负变正，但实际是 1→2 从
−2.8%跳到+6.8%，2→4 又从+6.8%回落到−1.2%，4→6 再跳回+6.5%。差距量级本身也不大（1–7%），
数量级和第五阶段其余弱信号（`character_length_t5`、`vector_count_stage1/2_t10` 等）相当，
不像第五阶段最强的两个信号（`average_word_length_t5` 的 44.6%、`sentiment_t5` 的 78.1%）。

（注：`rollout_mse` 在这里是"种子轮次之后所有轮次"的平均误差，不是"horizon 处那一个点"的
误差，所以 horizon=6 的指标里其实混入了 horizon=1..6 每一步的误差，会稀释纯粹的"远端复合误差"
信号——这是这个指标定义本身的局限，也是交替模式的一个可能来源，但不构成"horizon 有单调
影响、只是被指标平均掉了"的证据，因为如果真有强烈的单调复合误差，稀释后至少应该看到微弱但
一致的方向，而不是正负交替。）

### 结论

**"rollout horizon 长度决定 AE 是否更好"这个假设没有被两个探针支持，予以否定**。第五阶段
"T=5 任务多数 AE 赢、T=10 任务全部线性赢"这个表面模式，更可能是任务本身的特性（读出函数的
性质、数据规模等）恰好和 T=5/T=10 这个任务分组相关，而不是 rollout 步数本身的因果效应——这
纠正了看到第五阶段结果后最直觉的一个解释。真正驱动 AE-vs-线性差距的因素目前仍不确定，候选
（未经因果验证，只是描述性观察，见下）包括：

1. **state 维度/`lag` 选择**：探针 A 换成 `lag=1` 后 AE 的优势幅度骤降，暗示"延迟嵌入状态里塞
   进多少历史"本身可能比 horizon 更影响 AE 能不能找到线性模型抓不到的非线性组合。
2. **读出函数的性质**：`average_word_length_t5`（AE 赢最多，44.6%）的读出是
   `总字符数/词数`，是对底层生成文本的一个真正非线性（比值型）函数；`character_length_t5`、
   `sentence_length_t10` 的读出是接近直接计数的量，AE 优势小或线性反赢。`formality_t5`、
   `sentiment_t5` 的读出来自外部 NLP 分类器打分，`formality_t5`（252 条轨迹）AE 中等幅度赢，
   `sentiment_t5`（60 条轨迹，本次样本量最小的几个任务之一）AE 输得最多（78.1%）——不排除是
   AE 在小样本、噪声更大的分类器打分上过拟合（`hidden_dim=64` 两层 MLP encoder+decoder 参数量
   级约 1.1 万，远超线性模型的几十个参数）。
3. 这两条目前都只是**描述性关联，不是像本阶段这样做过因果探针的结论**——如果要验证，需要专门
   的后续实验（例如：固定 lag、只对 `sentiment_t5` 做数据量下采样 vs 不下采样对照；或者构造
   一个"读出是直接计数 vs 读出是比值"的最小任务对照）。

## 第七阶段：检验"数据规模"和"lag"是否解释 `sentiment_t5` 这个最大反例（已完成，两个假设都未被支持）

### 动机

第六阶段否定了"rollout horizon 决定胜负"。第六阶段末尾提出了两个尚未验证、仅基于描述性关联的
候选解释：(a) 数据规模——`sentiment_t5`（60 条轨迹）AE 输得最多，`formality_t5`（252 条轨迹，
同样是外部 NLP 分类器读出）AE 反而赢，猜测 AE 在小样本上过拟合；(b) `lag`——第六阶段探针 A
发现 `average_word_length_t5` 换成 `lag=1` 后 AE 的优势幅度骤降，猜测 `lag` 本身可能是驱动因素。
本阶段直接对 `sentiment_t5`（第五阶段里效应量最大、最异常的一项：线性反超 AE）做两个针对性探针。

### 探针 A（数据规模）：把 `formality_t5` 下采样到 `sentiment_t5` 的轨迹数（新增脚本
`ablation_downsample_probe.py`）

固定 `lag=3`、`common_seed_turns=4` 不变，只把 `formality_t5` 按 `sentiment_t5` 的
train/validation/test 轨迹数（40/10/10，共 60 条，用固定随机种子 0 抽样）下采样，直接检验
"更少数据是否让 AE 优势消失/反转"：

| 数据规模 | AE rollout_mse（3 seed） | 线性 rollout_mse | 胜者 | 差距（相对 AE 均值） |
|---|---:|---:|---|---:|
| `formality_t5` 原始（n=252） | 0.006543 ± 0.000128 | 0.006863 | AE | 线性差 4.9% |
| `formality_t5` 下采样到 n=60 | 0.009700 ± 0.000548 | 0.012068 | AE | 线性差 24.4% |

**方向和假设相反**：数据从 252 条降到 60 条后，AE 的优势不但没消失，反而从 4.9% 扩大到
24.4%。这排除了"小样本让 AE 过拟合、从而输给线性"这个通用解释——至少在 `formality_t5` 这个
读出上，样本量减少并没有伤害 AE，反而伤害了线性模型更多。`sentiment_t5` 本身的小样本量
不能用来解释它为什么线性会赢。

### 探针 B（lag）：`sentiment_t5` 上扫 `lag ∈ {1, 2}`（新增脚本 `ablation_lag_sweep.py`）

固定 `common_seed_turns=4`（T=5 数据下 `lag∈{1,2,3}` 都不改变 horizon，一直是 1 步，不引入
第六阶段那个混杂变量）：

| `lag` | AE rollout_mse（3 seed） | 线性 rollout_mse | 胜者 | 差距（相对 AE 均值） |
|---:|---:|---:|---|---:|
| 1 | 0.011515 ± 0.000322 | 0.010468 | 线性 | 线性优 9.1% |
| 2 | 0.007965 ± 0.000694 | 0.007688 | 线性 | 线性优 3.5% |
| 3（第五阶段原始结果） | 0.009925 ± 0.000561 | 0.005572 | 线性 | 线性优 43.9% |

**线性在全部三个 `lag` 上都赢**，方向完全没有随 `lag` 反转，只是幅度非单调波动（9.1%→3.5%→
43.9%，不是随 `lag` 增大而单调增大或减小）。这排除了"`lag` 选择本身决定 `sentiment_t5` 胜负
方向"这个假设——`average_word_length_t5` 上 `lag` 确实明显改变了效应量（第六阶段），但
`sentiment_t5` 上 `lag` 不改变方向，两个任务对 `lag` 的敏感度本身就不一样，说明 `lag` 效应
（如果存在）也是任务相关的，不是一个能跨任务解释胜负方向的通用因子。

### 结论

**"数据规模"和"lag"这两个候选解释都被证伪，`sentiment_t5` 上线性显著优于 AE 这件事目前没有
找到可以跨任务泛化的原因**。到本阶段为止，horizon（第六阶段）、数据规模、lag 三个"结构性/
配置性"假设全部被直接实验否定。剩下站得住脚的解释都指向**任务本身的读出信号性质**是任务相关
（task-specific）的，而不是某个可调超参数或数据规模能统一解释的——但"读出信号性质"目前仍然
只是描述性观察（第六阶段列出的"比值型 vs 计数型"区分），本仓库现有实验还没有一个直接测量
"信号非线性程度"或"信号噪声结构"、并证明它能预测 AE-vs-线性胜负的方法；这需要专门设计一个
可跨任务比较的"任务非线性度"指标才能验证，超出了当前"改配置重新跑"这类消融能覆盖的范围。

## 第八阶段：早停复现，排除"训练预算不足"作为混杂变量（已完成）

### 动机

第七阶段排除了 horizon/数据规模/`lag` 之后，`sentiment_t5`（线性赢 78.1%，第五阶段效应量
最大的"线性赢"案例）和 `average_word_length_t5`（AE 赢 44.6%，效应量最大的"AE 赢"案例，也是
第六阶段"比值型 readout 需要非线性"这个假说的核心证据）仍然没有解释。这两个任务恰好都是
`ABLATION_STUDY.md` 全篇效应量最大的两个异常点，但此前所有阶段都固定用 README 起点配置的
`trainer.epochs=200`（一个人工选定的数字，从未验证是否足够训练收敛）——"AE 的 encoder/decoder
是否单纯没有训练到收敛"这个最基础的可能性，一直没有被直接检验过。

### 设计

给 `core.py::DeepAugmentedKoopmanAutoencoder.fit`（`reconstruction_then_ridge` 阶段一）新增基于
验证集重建 loss 的早停（`DeepAugmentedKoopmanConfig.early_stopping_patience`/
`early_stopping_min_delta`，默认 `None`/不生效，向后兼容），`scripts/train.py` 相应接入
`topic_split="validation"` 切分作为早停验证集。`patience=30`、`min_delta=1e-6`、
`trainer.epochs=3000`（新的硬上限，只是安全网，不是目标训练轮数）。其余配置（`state=memory,
lag=3`、`latent_dim=16`、`reconstruction_then_ridge`、`seed∈{0,1,2}`）与第二/五阶段完全一致，
只改训练时长的决定方式，保持和已有结果的可比性。8 个任务全部重跑：

```bash
python scripts/train.py dataset=<task> state=memory state.lag=3 \
  model.training_mode=reconstruction_then_ridge model.latent_dim=16 \
  trainer.epochs=3000 trainer.device=cpu trainer.seed=<0|1|2> \
  trainer.early_stopping_patience=30 trainer.early_stopping_min_delta=1e-6 \
  trainer.checkpoint_root=/scratch/hcao2/checkpoints/autoencoder_koopman_core_ablation
```

### 结果（test split，y 空间 rollout_mse，早停侧 3 seed 均值±总体标准差；"早停 epoch" 是 3 个
seed 实际训练到的轮数范围，对照组固定为 200）

| 任务 | 旧（固定200ep） | 新（早停） | 线性 baseline | 早停 epoch 范围 | 方向变化 |
|---|---:|---:|---:|---:|---|
| average_word_length_t5 | 0.002138±0.000081（AE 赢 44.6%） | 0.0031053±0.0000065（AE 反输 0.4%） | 0.003092 | 615–853 | **反转** |
| character_length_t5 | 0.002682±0.000376（AE 赢 16.9%） | 0.002627±0.000351（AE 赢 19.3%） | 0.003135 | 381–619 | 不变 |
| even_odd_t5 | ≈0（退化） | ≈0（退化） | ≈0 | 118–151 | 不变，仍无信息量 |
| formality_t5 | 0.006543±0.000128（AE 赢 4.9%） | 0.006518±0.000112（AE 赢 5.3%） | 0.006863 | 267–289 | 不变 |
| sentence_length_t10 | 0.000941±0.000021（线性赢 6.5%） | 0.0009444±0.0000062（线性赢 6.8%） | 0.000880 | 198–263 | 不变 |
| sentiment_t5 | 0.009925±0.000561（线性赢 78.1%） | 0.006061±0.000077（线性赢 8.1%） | 0.005572 | 705–859 | 不变，**差距锐减 78%→8%** |
| vector_count_stage1_t10 | 0.006207±0.000300（线性赢 8.1%） | 0.0059637±0.0000324（线性赢 3.7%） | 0.005743 | 189–204 | 不变，差距略缩小 |
| vector_count_stage2_t10 | 0.006783±0.000066（线性赢 1.5%） | 0.0068648±0.0000065（线性赢 2.6%） | 0.006685 | 158–184 | 不变 |

重建 loss（阶段一训练目标，seed=0）佐证了"训练不足"确实是部分任务的真实瓶颈：`sentiment_t5`
从 2.41×10⁻³ 降到 2.38×10⁻⁵（约 100 倍），`average_word_length_t5` 从 5.01×10⁻⁴ 降到
7.43×10⁻⁶（约 67 倍）；相比之下 `formality_t5`、`sentence_length_t10`（原本重建 loss 就已经
是 1–2×10⁻⁵ 量级）早停后只小幅改善或几乎不变，`vector_count_stage1/2_t10` 甚至略高于旧值
（3×10⁻⁵ vs 2×10⁻⁵，噪声量级，因为验证集早停判定的停止点和固定 200 epoch 不严格是同一个点）
——这与它们的 rollout_mse 结果方向"6/8 个任务不变"完全对应：只有 `sentiment_t5` 和
`average_word_length_t5` 原本就明显没训到收敛，其余任务在固定 200 epoch 时就已经接近收敛，
早停对它们只是锦上添花或不起作用。

### 结论

1. **"任务相关、无统一方向"这个第五阶段的总体结论没有被推翻**——8 个任务里 7 个方向不变
   （其中 `even_odd_t5` 本就退化、无信息量，所以在有信息的 7 个任务里是 6 个不变）。但**全篇
   两个效应量最大的"旗舰案例"都被显著削弱或直接推翻**：`sentiment_t5` 的线性优势从 78.1% 收窄
   到 8.1%（欠拟合假设成立，此前第七阶段排除的"数据规模"/"lag"都不是真正原因，真正原因是
   训练轮数本身）；`average_word_length_t5` 的"AE 赢 44.6%"直接反转成打平（AE 反而略输
   0.4%）——这是全篇除 `sentiment_t5` 外效应量最大的一项，此前从未被怀疑是训练不足的产物。
2. **第六阶段"比值型 readout（如 average_word_length_t5 的字符数/词数）需要非线性提升"这个
   假说，核心证据现在站不住了**：它唯一的支撑就是 `average_word_length_t5` 44.6% 的效应量，
   而这个效应量本身被证明主要是训练预算不足、不是任务的真实非线性需求。"读出信号性质决定
   AE-vs-线性胜负"这条第七阶段末尾留下的、唯一没被证伪的候选解释，其证据基础比看起来的更弱。
3. **训练预算（是否收敛）此前从未被当作一个需要控制的混杂变量**——第一到第七阶段的每一次
   对照都固定用同一个人工选定的 `epochs=200`，隐含假设"200 轮足够所有任务收敛"，这个假设本身
   从未被验证过，而这次证明它对至少 2/8 个任务不成立。这提示：*任何*后续新增任务或新的
   AE-vs-线性对照，都应该默认打开早停（或至少检查训练/验证 loss 曲线是否已经平台化），不能
   再默认沿用 `epochs=200` 这个未经验证的固定值。
4. **`persona_drift_control` 的对抗防御 AE baseline（`docs/experiments/ae_baseline_plan.md`）
   做了同样的早停改造，但没有看到同样的改善**（结果记录见该文档）：那个任务训练数据只有
   22 条训练攻击（44 条轨迹），切验证集后进一步压到 18 条，早停判定信号本身噪声很大（同一
   `latent_dim` 不同 seed 实际训练轮数从 200 多到 5000 都有），这提示"欠拟合"假设的适用性
   依赖数据规模本身要足够支撑一个独立的验证切分，不是所有 AE 实验都能靠早停"免费"改善。

产物：`results/<task>-memory-lag3-reconstruction_then_ridge-k16-earlystop-seed<N>/`（8 任务
× 3 seed = 24 个新增结果目录），`src/koopman_ae/core.py`/`scripts/train.py`/
`configs/trainer/default.yaml` 的早停相关改动。

## 后续阶段（未执行，计划）

- **第四阶段（b）**：`lambda_latent`、`lambda_multi` 消融；以及把 `joint` 训练方式和
  `multi_step_horizon>0` 的多步 rollout loss 一起打开，重新对比 `joint` vs
  `reconstruction_then_ridge`（第三阶段的结论目前只在 `lambda_multi=0` 下成立）。
- 第五/八阶段结论目前只在 `latent_dim=16`、`reconstruction_then_ridge`、`state=memory,lag=3`
  下验证；`average_word_length_t5`、`sentiment_t5` 这两个效应量最大的任务值得优先补更多种子
  （当前只有 AE 侧 3 个种子，线性侧本就无随机性）以确认早停后的新效应量不是噪声。
- 第一阶段（状态定义：markov/memory/augmented）目前也只在 `sentence_length_t10` 上验证过，
  尚未像本阶段这样铺开到其余 7 个任务；且第一阶段本身也是用固定 `epochs=200` 跑的，第八阶段
  的发现同样适用——"markov 明显更差"这个结论也应该在早停下复核一次。
- **第九阶段（候选，未执行，原第八阶段候选，因编号冲突后移）**：第七阶段排除了
  horizon/数据规模/`lag` 之后，唯一还站得住的方向是"读出信号本身的非线性/噪声结构"——需要先
  设计一个可跨任务计算的量化指标（例如：控制输入 `r` 到输出 `y` 之间残差非线性度的某种度量，
  或信号的高阶矩/间断性统计量），再检验它是否能预测第五/八阶段任务里 AE 是否赢。**第八阶段的
  发现削弱了这个方向的优先级**：它最初的动机主要来自 `average_word_length_t5` 的比值型
  readout 假说，而那个假说的核心证据已经被第八阶段推翻，这个方向现在需要先在早停结果上
  重新确认还有没有值得解释的效应量,再决定是否投入设计新指标。
- **第十阶段（候选，未执行，原第七阶段候选，因编号冲突后移）**：验证第六阶段末尾提出的
  `lag`/state 维度对 AE-vs-线性差距的影响，在多个任务上扫 `lag∈{1,2,3}`（探针 A 只做了
  `average_word_length_t5` 一个任务的 `lag=1 vs 3`，且是在早停之前做的，`average_word_length_t5`
  本身的基线结论已经变了，这个探针的解读也需要在早停下重新做）。（数据规模那一支已经在正式
  第七阶段做过，不再重复列出。）
