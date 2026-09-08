# 执行计划：Koopman 快车道（K1–K4）——一个激励臂直接进建模

**状态**：⏳ 已就绪，**等待提交 K1**（2026-09-08）。**尚未取代**
[`defense_line_redesign_plan.md`](defense_line_redesign_plan.md)。
K1/K1.2 的 sbatch 已写好、K2 的 null 补丁已落地并通过回归检验（0.2 节），**GPU 作业未提交**。
**适用智能体**：Sonnet 5。
**触发来源**：对防御线证据基础的复核发现，用来否掉"有没有状态"的三个数字没有一个是
Koopman/DMDc 需要的量（详见本文档第五节）。结论是**停止再做代理性前置闸门，直接跑真正的
辨识与闭环评测**——因为 Koopman 拟合本身就是比任何代理闸门更强的判据。

> **开工前必读**（继承既有纪律）
>
> 1. 本文档是完整规格。不要凭记忆复述、不要"优化"命令行、不要合并步骤。
> 2. 闸门不过就停下来报告，不要自己想办法绕过去。
> 3. 本文档没写到的判断题，停下来问用户。
> 4. **不要动 `paper/`**；**不要改** `control.py` / `controller_cli.py` / `modeling/evaluate.py`。
> 5. **不要覆盖任何已有产物**，输出路径都是新的。
> 6. **不要自行提交 GPU 作业**。
> 7. 环境：`export PATH=/scratch/hcao2/envs/persona_drift_pilot/bin:$PATH`，工作目录
>    `/home/hcao2/autoencoder_koopman_core/persona_drift_control`。

---

## 零、这份计划相对旧计划砍掉了什么

| 旧计划里的动作 | 处置 | 理由 |
|---|---|---|
| D2「turn 1–2 静态分类」闸门 | ❌ 删 | 测的是一次性可分性；MPC 逐轮重决策，不需要它（§14.3 用户更正） |
| 「重新定义前提 2」的独立闸门 | ❌ 删 | 由 K2 的 held-out 一步误差取代——拟合本身就是前提 2 的判据 |
| RC-0..RC-3 电池 | ❌ 删 | 被 K2 的「预测 vs 三个平凡 null」+ `B` 的 CI 取代，且 RC-3 的判据挂在错的 `u` 上（第五节） |
| Phase A 执行器权威臂（`constant_remind`） | ❌ 删 | K1 的激励臂里约 250 个 reminded turn 已经给出 `B`，功效远高于 D1 的 41 行/臂 |
| D1 盲标续做（§13.2 阻塞） | ⏸ 降级 | 不再是前置条件，挪到 K4 的报告口径决策 |
| 路线 A / A2 | ❌ 不复活 | A1 已出局 |
| 读出量程前置检验 | ✅ **已过，零成本** | 见 0.1，本次会话已算完，不需要新实验 |

**净结果：前置实验 = 0，新工程 = 1 个 null 基线补丁（K2.1，已完成）。**

### 0.2 用户裁决与已完成的准备（2026-09-08）

| 项 | 裁决 / 状态 |
|---|---|
| `--random-excite-p` | **0.5**（用户裁决）。`B` 的功效最大；代价是约一半轮次带提醒，偏离低干预叙事——这条臂只用于辨识，不用于报告防御率 |
| K1 的 seed 数 | **只跑 seed 0**（用户裁决）。500 行足够 K2 辨识；若 K2.4 算出 K3 需要更多 seed，届时再回头补 K1 |
| `environment/run_defense_excite_qwen4b.sbatch` | ✅ 已写好，命令行已用真实 argparse 干跑验证（controller=random_excite, 100 个 attack_id, seeds=[0], max_new_tokens=1024） |
| `environment/run_defense_rejudge_d1_qwen4b.sbatch` | ✅ 已写好 |
| K2.1 null 补丁 | ✅ 已落地（`--n-folds`，默认 0）。**回归检验**：按已提交报告的原始参数（`--mu 2`）重跑，`arx` / `richer_abs_sign` / `controllability_arx` / 切分全部逐字节相同 |
| GPU 作业 | ❌ **未提交**（§1.2 第 6 条） |

### 0.1 唯一的前置条件：读出量程（已满足）

`outputs/d1_screen_qwen4b/`（100 攻击 × 5 轮 × seed 0，`--agent-max-new-tokens 1024`）是
防御线上**第一个五轮都有量程**的数据：

| turn | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| `y_safety` 均值 | 0.897 | 0.700 | 0.552 | 0.432 | 0.435 |
| 跨轨迹 sd | 0.251 | 0.384 | 0.390 | 0.394 | 0.414 |
| 贴天花板占比 | 0.81 | 0.54 | 0.35 | 0.25 | 0.30 |

去轮次趋势后 lag-1 = **+0.5285**（p=3.7e-30，400/400 个有效对）。对照：旧的独立 judge
口径在 `phaseE_zero_control` 上 turn 1/2/4 的 sd **精确为 0**，有效对 **0 个**——那里的
"lag-1 = 0.000, p=1"是算术上被强制的，不是系统性质。

**这条量程检验从此是硬前置**：任何用来下"有/无状态"结论的读出，必须每一轮 sd > 0 且
≥ 3 个取值。不过就不许用它下结论。

---

## 一、K1：激励臂（**唯一的新 GPU 采集**）

**类型**：1 个 GPU 臂，~4.5h（对标 `d1_screen_qwen4b` 的 4:33:03）。**依赖**：无。

**为什么这不是前置实验**：它同时是 (a) Koopman 的训练集、(b) `B` 的估计、(c) 把 `A` 与
"每 episode 一个静态偏移 α"分开的**唯一**手段。在 `u ≡ 0` 的数据上，`A` 与 α 不可分辨、
`B` 完全不可识别——防御线此前所有"有没有状态"的测试都跑在零控制臂上，这是根因。

### 1.1 规格

新建 `environment/run_defense_excite_qwen4b.sbatch`，照抄
`environment/run_d1_screen_qwen4b.sbatch` 的头部，正文改为：

```bash
python scripts/run_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/defense_excite_qwen4b \
  --controller random_excite \
  --random-excite-p 0.5 \
  --attack-ids $(cat conf/experiment/d1_attack_ids.txt) \
  --seeds 0 \
  --agent-max-new-tokens 1024
```

`--attack-ids` 是 `nargs="+"`（`run_defended_screening.py:84`），`d1_attack_ids.txt` 是
空白分隔的 100 个 id，所以 `$(cat ...)` 展开即可——**不要**新写一个 CLI。
`RandomExciteController` 见 `control.py:61`（每轮 i.i.d. Bernoulli(0.5)）。

**闸门 G-K1**：`COMPLETED 0:0`；500 行；`attack_id` 集合与 `outputs/d1_screen_qwen4b/`
**完全相同**；`u_remind` 的均值落在 [0.40, 0.60]；reminded 行数 ≥ 200；
`y_safety` 每一轮的跨轨迹 sd > 0（0.1 节的量程检验，在新臂上重跑一遍）。

### 1.2 与 K1 并行的零成本动作（不占 GPU 排队之外的时间）

`scripts/rejudge_safety_runs.py` 对**已有**的 `outputs/d1_screen_qwen4b/` 做离线独立
rejudge（Qwen3-4B-Instruct-2507），**不产生任何新生成**。目的只有一个：**看独立 judge
在这个新设定上有没有量程**（旧池上它 75/80 行贴 1.0）。

- 有量程 → K4 可以用它作第二口径，§14.1 的自评局限被大幅缓解；
- 没量程 → K4 只报自评口径，并按 §14.1 原样带上局限。

**这一步不设闸门、不阻塞任何后续动作**，结果只影响 K4 的报告方式。

---

## 二、K2：拟合 + 真正的判据（CPU，分钟级）

**依赖**：K1。**这一步取代旧计划里的全部代理闸门。**

跑法（`--contemporaneous-v` **不是可选项**，见 2.3）：

```bash
python scripts/fit_koopman_defense_model.py \
  --rows-path outputs/defense_excite_qwen4b/trajectories.jsonl \
  --nu 1 --mu 2 --contemporaneous-v \
  --n-folds 20 --split-seed 0 \
  --out-path outputs/defense_excite_qwen4b/koopman_fit_report.json
```

### 2.1 唯一的代码改动：给拟合报告补三个平凡 null（✅ 已完成）

`scripts/fit_koopman_defense_model.py` 已经报 `held_out_rollout_mse`、
`controllability_rank`、`gramian_condition`、`A_spectral_radius`，**但没有平凡基线**——
而 RC-1 的教训正是"没有 null 就读不出 MSE 的意义"。补三个（直接复用
`scripts/analyze_soft_judge_readout.py:133` 的 `stateless_null_predict` 写法）：

| null | 定义 |
|---|---|
| `const` | 训练集全局均值 |
| `turn_mean` | 训练集的逐轮均值（**这是必须打过的那个**——它就是主导 Koopman 模态） |
| `stateless` | `y ~ 1 + turn + u`，不含 `y_t` |

并把评价从"一次 held-out"改成**按 `attack_id` 切的 20 折**，每折报 `arx` / `richer` /
三个 null 的一步 MSE。新增 `--n-folds`（默认 0 = 原行为，报告一字不变），
产出 `fold_evaluation` 块。

**实现里两个必须保留的细节**（都写进了脚本注释）：

1. **比的是标量 `y_{t+1}`，不是 `modeling.evaluate.one_step_error` 的整个 `z_{t+1}`。**
   `z_next` 的 `v` 块只是这条轨迹自己已实现的输入平移一格，代理模型免费复现它——用整
   `z` 的 MSE 去比只预测 `y` 的 null 是在给代理模型送分。`y_{t+1}` 在 `z_next` 的
   `nu-1` 位（与 `rollout_output_error` 读的是同一个槽），因此不改 `modeling/`。
2. **三个 null 都拿到 `turn`。** d1 池上 `y_safety` 逐轮从 0.90 掉到 0.43，逐轮均值是
   最强的平凡预测器，而它就是主导 Koopman 模态——打过它才是问题本身。不给 null 一个
   模型有的确定性外生量，是 ERGO RC-1 缺陷的镜像（那次是**白送**给 null 一个模型得外推
   的量），会制造一个假的 PASS。

**旧数据上的合理性对照**（在 `phaseB_random_excite` 上 5 折干跑）：`arx` 只在 **3/5** 折
打过最好 null，与 F4 的 RC-1（6/20）同向。**闸门不是白送的。**

### 2.2 预注册闸门（跑之前写下，跑完不改）

| 闸门 | 判据 |
|---|---|
| **G-K2-1（状态）** | Koopman 的**一步** MSE 在 ≥ 14/20 折上打过三个 null 里最好的那个 |
| **G-K2-2（可控）** | 同轮动作系数 `B`（即 `u_{t+1}`，见 2.3）的 95% CI 不含 0，且符号 = 提醒使 `y_safety` **上升**（更安全） |
| **G-K2-3（非退化）** | `A` 的谱半径 ∈ (0.1, 1.05)；可控性矩阵满秩；Gramian 条件数 < 1e12 |
| **G-K2-4（多步）** | `held_out_rollout_mse` 打过 `turn_mean` 在同一批行上的 MSE——单步过、多步不过意味着 MPC 的 horizon 只能取 1 |

**三条主闸门（1/2/3）全过 → 进 K3。任一不过 → 停下来报告，不要调 `nu`/`mu`/`ridge`
重跑找一个想要的答案**（要扫超参只能在**看结果之前**声明扫描范围与选择规则）。

### 2.3 `u` 的索引必须写进代码注释

`trajectory_runner.py:96-116`：`u_remind[t]` 是拼进**第 t 轮 prompt** 的提醒，产出
`agent_message[t]`，判成 `y[t]`。所以作用在 `t → t+1` 转移上的控制是 **`u_remind[t+1]`**。
`analyze_soft_judge_readout.py:219` 的 RC-3 判据挂在滞后的 `u_remind[t]` 上（那里
`u_t` p=0.96/0.72，而同轮的 `u_{t+1}` 是 +0.037/+0.033, p=0.067/0.082）——**这是
ERGO 线 E0 已经撤回过一次的同一类错误，不许再犯第三次。**

代码库自己已经有这个修正的名字：`ReducedStateConfig.contemporaneous_v`
（`modeling/dataset.py:48-73`）。它的 docstring 明写了默认的 `False` 配的是
persona-drift 的 probe-then-decide 时序，**不是 `attack_trajectory.py` 的时序**，那样拟合
出的 `B` "不是提醒的直接效应，只是提醒文本还留在历史里对 `y_(t+1)` 的残留效应"。
**所以 K2 必须传 `--contemporaneous-v`，K3 的 hydra 配置必须带
`koopman_contemporaneous_v: true`，两边不一致会让控制器在错的动作槽上决策。**

已知的解释性风险，必须在报告里写明而不是回避：同轮的 `u` 同时进入了状态转移与观测函数
（judge 读到的回复所处的上下文里有提醒文本）。K3 的闭环比较不受此影响（那里比的是终点
结果，不是系数），但 K2 的 `B` 是"控制 + 观测"的合成增益，**措辞不许写成纯因果状态增益**。

### 2.4 K2 顺带产出 K3 的样本量（这才是省钱的关键）

用 K2 估到的 `B` 与 late `y_safety` 的配对差 sd，按 80% 功效 / α=0.05 算 K3 每臂需要多少
条轨迹，**再决定 K3 跑几个 seed**。旧计划全线的失败模式是先跑臂、后发现没功效；这里顺序
反过来。**若算出来的需求 > 每臂 150 条轨迹，停下来报告**——那说明这个设定仍然太弱，应该
转负结果而不是硬跑。

---

## 三、K3：闭环评测（GPU，臂数与 seed 数由 K2.4 决定）

**依赖**：K2 三条主闸门全过。全部是 hydra 配置，**零新代码**。

### 3.1 臂

新建 `conf/experiment/fast_base.yaml`（照 `phaseJ_base.yaml` 的结构），共享：
`attack_ids` = K2 报告里的 `held_out_attack_ids`（25 个）、`remind_budget: 1`、
`episode_length: 5`、`agent_max_new_tokens: 1024`、`seeds` = K2.4 算出来的那个数。
`koopman_mpc` 臂另外必须带 **`koopman_contemporaneous_v: true`**，与 K2 的拟合口径一致（2.3）。

| 臂 | 角色 |
|---|---|
| `koopman_mpc` | 待证明的自适应策略（模型路径 = K1 的拟合产物） |
| `threshold` | 等预算**反应式**对手（经典基线） |
| `fixed_schedule`（turn = koopman 的众数消费轮） | 等预算**非自适应**对手——**这是主对照** |
| `zero_control` | 不防御下界。**seed 0 已经在 `outputs/d1_screen_qwen4b/` 里，只需补 seed ≥ 1** |

跑法：`python scripts/run_screening_hydra.py experiment=fast_<arm>`（`run_screening_hydra.py`
已支持 `remind_budget` / `attack_ids` / `koopman_*`）。

### 3.2 预注册判定

主量：late(3,4,5) 的 `y_safety` 轨迹均值，按 `(attack_id, seed)` 配对，配对 bootstrap
10000 次（`default_rng(0)`）。

| 结果 | 裁决 |
|---|---|
| `koopman_mpc − fixed_schedule` 的 95% CI 不含 0 且为正 | **自适应闭环有正面证据**——这是这条线一直想要的那个结果，报给用户由 Opus 写论文口径 |
| CI 含 0，但 `koopman_mpc − zero_control` 显著为正 | 提醒有用、**自适应不比固定日程强**。这是干净的负结果，也是可发表的（"闭环的额外价值为零"） |
| `koopman_mpc − zero_control` 也含 0 | 在这个设定上执行器仍然摸不到结果 → 回第十节式的负结果，但这次是在**有量程、有激励、有辨识**的条件下得出的，比现在的版本强得多 |

**不预测方向。**

---

## 四、K4：报告口径

- K1.2 的独立 rejudge 有量程 → 主表双口径并排（自评 + 独立）；无量程 → 只报自评，
  并原样带上 §14.1 的局限句；
- §13.2 的盲标阻塞在这里再决策：**只有当论文需要"实质帮助率"这个绝对量时才需要续做**；
  K3 比的是臂间差，用不到它；
- 所有数字照抄，带 CI；偏离本文档的地方逐条列出。

---

## 五、为什么这样重排——旧证据基础的三个错位（复核记录）

1. **长程聚合 ≠ 一步转移。** §0.1 的 `r = −0.066/+0.041` 测的是 mean(turn 1–2) →
   mean(turn 3–5)，约等于 lag-3 的聚合预报；A1 测的是跨 episode 的静态回归；D2 测的是
   一次性分类。三者都不是 receding-horizon 控制需要的量。
2. **唯一形状对的判据跑在了不可能产出非零值的数据上。** §0.2 的 lag-1 = 0.000 有 0 个
   有效对（见 0.1）。同一份 `readout_state_report.json` 里，自评 @ 同一批行是 **+0.424**
   (p=4.7e-4)、screening 是 +0.347、Phase B 硬标签 +0.170、软 judge +0.229、激活投影
   +0.861（自持系数 0.60）。
3. **在零控制数据上，"静态异质"与"AR(1) 动力学"观测等价。** T=5、200 次重复的模拟：
   纯静态偏移给 de-turned lag-1 = +0.504，纯 AR(1) a=0.58 给 +0.552——实测的 +0.528 落在
   两者之间，**两个假设都没被拒绝**。分开它们的唯一手段是激励，即 K1。

第五节三条的诊断脚本本次会话只在 scratchpad 里跑过、**未提交**（照 §14.2 的先例：诊断计算不落盘为正式脚本）。
要复算，按 0.1 与本节的数字重写即可；若决定长期保留，落到
`scripts/diag_dynamics_vs_prediction.py` 并在本节记下 commit。

---

## 六、失败模式清单

1. **K2 不过就调 `nu`/`mu`/`ridge`/换特征重跑。** 那是在找想要的答案。要扫参只能在看结果
   之前声明范围与选择规则。
2. **把 K2 的 `B` 写成纯因果状态增益。** 它是控制 + 观测的合成增益（2.3）。
3. **在 K3 里用非等预算的对手。** `remind_budget` 必须由 `fast_base.yaml` 统一给，
   不许逐臂抄。
4. **跳过 K2.4 直接开 K3 的臂。** 先算功效再花 GPU，这是旧计划全线失败的模式。
5. **回头用零控制臂论证"有/无状态"。** 那里 `A` 与 α 不可分辨。
6. **在任何地方把 `u_remind[t]` 当成 `t→t+1` 的控制。**（2.3）
7. 自行提交 GPU 作业；改 `control.py` / `controller_cli.py` / `modeling/evaluate.py`；改 `paper/`。

---

## 七、成本合计

| 阶段 | GPU | CPU | 新代码 |
|---|---|---|---|
| K1 激励臂 | ~4.5h × 1 | — | 1 个 sbatch |
| K1.2 离线 rejudge | ~0.5h（与 K1 并行） | — | 0 |
| K2 拟合 + 闸门 | 0 | 分钟级 | ✅ null 补丁已落地 |
| K3 闭环 | 3–4 臂 × (seed 数由 K2.4 定) | — | 4 个 hydra yaml |

**到"能不能做真 Koopman"这个问题有答案为止：1 个 GPU 臂 + 分钟级 CPU。**

提交命令（**等用户下令**）：

```bash
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control
sbatch environment/run_defense_excite_qwen4b.sbatch        # K1，~4.5h
sbatch environment/run_defense_rejudge_d1_qwen4b.sbatch    # K1.2，并行，~0.5h
```
