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

## 后续阶段（未执行，计划）

- **第四阶段（b）**：`lambda_latent`、`lambda_multi` 消融；以及把 `joint` 训练方式和
  `multi_step_horizon>0` 的多步 rollout loss 一起打开，重新对比 `joint` vs
  `reconstruction_then_ridge`（第三阶段的结论目前只在 `lambda_multi=0` 下成立）。
- 其余 7 个标量/向量任务（`character_length_t5` 等）上重复第一、二阶段的消融，确认
  "memory 优于 augmented"、"线性优于 AE"是否是 `sentence_length_t10` 特有还是普遍现象。
