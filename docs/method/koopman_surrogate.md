# Koopman 代理与 ARX baseline：实现与公平对比设计

`src/persona_drift/modeling/` 是之前缺失的那一块——把采集到的 `trajectories.jsonl` 拟合成
`Control_of_Foundational_Model_revised.pdf` 第 4/5/6/7 节描述的受控 Koopman 代理，并把 ARX
baseline（`docs/BASELINES.md` 第③层）实现为同一套代码的特例，而不是另写一份。目前只用**合成的
已知线性系统数据**验证过（见 `tests/test_koopman.py`），还没有接到真实采集数据上——那要等
screening 通过、正式 320 条轨迹采完之后。

## 为什么不直接复用 `src/koopman_ae/core.py`

仓库根目录已经有一套通用 Koopman 实现（`AugmentedStateConfig` / `build_augmented_state_dataset`
/ `AugmentedKoopmanModel`），同事的标量/向量数据集就是用它训的。但它的控制输入是**写死的**
`u_t = r - y_t`（`control_mode: error` / `error_abs_sign`，闭环、跟踪误差）。`DATA_COLLECTION_PROTOCOL.md`
开篇就说明了这正是本协议和同事数据的根本区别：人格漂移用的是每轮独立随机抽取的开环激励
`u_remind`，采集阶段 `r` 根本不出现在 prompt 里，也就没有误差可算。所以 `build_augmented_state_dataset`
这条路径原则上不适用，`modeling/dataset.py` 是它的开环输入版本，为 `y_probe`/`u_remind` schema
重新写的。

真正可以、也确实复用的是和 schema 无关的纯数值工具：`modeling/koopman.py::controllability_diagnostics`
和 `koopman_ae.core.controllability_diagnostics` 字段名完全一致（有意保持一致），但**没有跨包
import**——它只有十几行纯 numpy、不依赖任何 pandas 列约定，直接复制一份比引入两个独立打包子
项目之间的依赖更划算。真正有"表面积"的部分（数据加载、切分、拟合、评测）没有这样重复，而是
按需各写各的，因为两边的控制语义本来就不兼容。

## 模块

- **`modeling/dataset.py`**：所有方法共享的数据加载/切分/状态构造。任何新 baseline（包括以后
  的 LSTM）都必须走这里，不能自己重新写一套 `z_t`/`v_t`/`y_t` 的构造逻辑——哪怕差异很小，也会
  让"方法更好"和"预处理不一样"混在一起，无法公平对比。
  - `load_trajectories` / `group_by_trajectory`：读盘、按轨迹分组排序。
  - `split_by_system_prompt_id`：按 `system_prompt_id`（不是按轮次）切 train/val/test，原因
    见 `DATA_COLLECTION_PROTOCOL.md` 第 6 节和 PDF 第 5/8 节——同一条对话内相邻轮次不独立，
    按轮次切分会泄漏。
  - `ReducedStateConfig(nu, mu)` + `build_reduced_state_pairs` / `build_identification_dataset`：
    构造 PDF 公式 (8) 的 `z_t = [y_t,...,y_(t-nu+1), v_(t-1),...,v_(t-mu)]`（本实现里顺序是从旧
    到新，PDF 是从新到旧，只是记法不同，不影响线性拟合的正确性）。NaN 打分（探针打分函数失败）
    的轮次会被整体跳过，不会污染拟合。
- **`modeling/koopman.py`**：`KoopmanSurrogate` —— `eta_t=[z_t, extra_features(z_t)]`，
  `eta_(t+1) ≈ A eta_t + B v_t + b`（PDF eq 15），`y_t ≈ C eta_t`（eq 16），岭回归最小二乘拟合
  （eq 18/19）。**ARX 就是 `extra_features_fn=no_extra_features`**（即 `eta_t=z_t`，不做任何
  lifting）——不是单独的模型类或单独的代码路径，和更丰富的 lifting（比如
  `abs_sign_extra_features`）走同一个 `fit`/`step`/`readout`，唯一的自由变量是 lifting 字典本身。
  也提供 `controllability_diagnostics(A, B, horizon)`（PDF eq 23-26 的可控性矩阵/Gramian/秩/
  谱半径）。
  - **参考矩阵 E 没有实现**：协议第 4 节采集阶段 `r` 不出现在 prompt 里，这批数据里 `r` 根本不
    变化，硬拟合一个 E 只会得到一个没有意义的矩阵。
- **`modeling/evaluate.py`**：`one_step_error` / `rollout_output_error`，对应 PDF 第 8 节验证
  协议的前两项（one-step prediction、multi-step rollout）。这两个函数只依赖一个最小的
  `Predictor` 协议（`step(z, v) -> z_next`、`readout(z) -> y`），不专门为 `KoopmanSurrogate`
  写——任何未来的 baseline（LSTM 等）只要实现这两个方法就能复用同一套评测代码。

## 已知缺口 / 还没做的事

- **只在合成数据上验证过**：`tests/test_koopman.py` 用已知 `a,g,c` 的线性系统生成无噪声数据，
  验证 `fit()` 能把参数 recover 回来、`one_step_error`/`rollout_output_error` 在这种理想情况下
  接近零。还没有在任何真实 `trajectories.jsonl` 上跑过——等 screening 过关、正式数据采出来后
  要做的第一件事就是这个。
- **LSTM baseline 还没实现**：`docs/BASELINES.md` 第③层要求的另一个 ablation。`Predictor` 协议
  已经为它留好了扩展点（实现 `step`/`readout` 即可复用 `evaluate.py`），但训练循环本身
  （nonlinear，需要 torch 优化器）还没写，属于范围内明确推迟的部分，不是遗漏。
- **可达集/可控集与黑盒数据的对照**（PDF 第 8 节验证协议后两项）需要真实数据和黑盒采样对比，
  现在拿不到，`GenCtrl` 的对接（见 `docs/BASELINES.md`）也还没做。
