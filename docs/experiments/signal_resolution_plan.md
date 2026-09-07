# 执行计划：读出分辨率（F0–F4）——修正 ERGO 闸门的两处测量错误，并把"换信号"改成"同一仪器提高分辨率"

**状态**：✅ **F0–F4 全部执行完毕（2026-09-07）**。F0–F3（ERGO 线）结果见
`ergo_multiturn_reliability_pilot.md`"F0–F3：读出分辨率修正与 Phase C 结果"一节——F0/F1/F2
全过，F3 三道预注册闸门全部不过（`ergo_koopman_mpc` 没打过任何基线，诊断为 `horizon=2`
短视导致 116 条轨迹全部退化成 `fixed_t2`）。F4（防御线）结果见 commit `7468247`——RC-1 不过，
防御线读出族正式封闭。**本文档下面的规格是这些结果的出处，只读不改**；不要重跑任何一步。
**适用智能体：Sonnet 5**——每个任务的规格、命令、预注册判据都写死在本文档里，不需要自己做设计判断。
**触发来源**：2026-09-07 对 `c1b288f`（D1）与 `8dde4b6`（ERGO E0/E1 终止）的复核。复核发现
**ERGO 的终止判定建立在两处测量错误上**，改正后 RC-B 三条全过；同时 E1 的"换读出"方向选错了
（熵与被评价目标近乎正交）。正确的方向是把**同一个测量仪器去阈值化**。
**阻塞**：ERGO Phase C、防御线读出族的最后一个未测候选、论文里"三个独立负结果"这条叙事。

> **本文档是当前唯一的活计划。** 开工前先读第九节"文档状态地图"，确认你没有在执行一份
> 已经作废的规格。

> **开工前必读**（继承 `adaptive_vs_fixed_claim_plan.md` 与已归档的 RC-gate 计划的同一套纪律）
>
> 1. 本文档是**完整规格**。不要凭记忆复述、不要"优化"命令行、不要合并步骤。
> 2. 每个任务都有出口闸门。**闸门不过就停下来报告，不要自己想办法绕过去。**
> 3. 遇到本文档没写到的判断题，**停下来问用户**。作废判定与 evidence 台账是 Opus 的活。
> 4. **不要动 `paper/` 下任何文件。**
> 5. **不要覆盖任何已有产物**。每个任务的输出路径都是新的；目标已存在就停下来报告。
> 6. **不要改** `src/persona_drift/control.py` / `controller_cli.py`——`defense`/`stance` 共用。
>    ERGO 的一切扩展进 `src/persona_drift/ergo_koopman_mpc.py`。
> 7. 环境：`export PATH=/scratch/hcao2/envs/persona_drift_pilot/bin:$PATH`，工作目录
>    `/home/hcao2/autoencoder_koopman_core/persona_drift_control`。登录节点没有 `conda`，
>    `scontrol` 坏了（用 `squeue -u hcao2` / `sacct -j <jobid>`）。GPU 一律 `sbatch`。

---

## 零、为什么有这份计划（全部为 2026-09-07 离线复算，零 GPU，可复现）

### 0.1 E0 的 RC-3 用错了变量（差一位）

`scripts/analyze_ergo_readout_state.py:216-219`：

```python
for a, b, c in zip(traj, traj[1:], traj[2:] + [None]):
    if b["turn"] != a["turn"] + 1:
        continue
    u_next = float(c[U_COL]) if c is not None and c["turn"] == b["turn"] + 1 else None
```

`a` 是 t、`b` 是 t+1（被预测的目标）、`c` 是 **t+2**。所以记进回归的 `u_t1` 实际是
`u_{t+2}`——一个发生在被预测目标**之后**的动作。ERGO 是 `contemporaneous_v=True`
（`u_reset_t` 同轮影响 `y_t`），与 `y_{t+1}` 同期的动作是 `u_{t+1}`，即 `b` 那一行的 `u_reset`。
顺带丢掉了每条轨迹的最后一个转移（546 → 426）。

同一份数据、同一个 `ols`、同一套 item 固定效应：

| 规格 | n | `u_t` | se | p | `u_next` | p |
|---|---:|---:|---:|---:|---:|---:|
| E0 实际做的（`u_next = u_{t+2}`） | 426 | +0.0072 | 0.0163 | 0.6587 | −0.0080 | 0.6336 |
| **计划的字面规格（`u_next = u_{t+1}`）** | 546 | **+0.0588** | 0.0270 | **0.0296** | +0.0716 | 0.0081 |
| 不含 `u_next` | 546 | +0.0511 | 0.0270 | 0.0589 | — | — |

**RC-3 是过的。** 防御线的 D1 没有这个问题（`scripts/analyze_input_gain_robustness.py:86`
用的是 `b["u_remind"]`），所以 D1 的结论与 T3 的终止判定**不受影响**。

### 0.2 RC-1 的比较对 ARX 结构性不公平

`modeling/evaluate.py::rollout_output_error` 从初值出发递归推进整条轨迹，**包括
`shard_frac` 那一维**——而 `shard_frac = turn/num_shards` 是完全已知的确定性外生量，
拟合出的自系数是 **1.005 > 1**，多步递归会发散。与此同时，被当作 null 的 stateless OLS
**每一行都拿到真值 `shard_frac` 与真值 `u`**。所以这个比较不是"状态有没有用"，是
"ARX 还得额外预测一个 null 白送的已知量"。

把 aux 维在 rollout 每一步换成真值（`z[-1] = pair["z_next"][-1]`，其余不变），
20 个随机 item split（协议见 1.2 节）：

| 读出 | ARX naive | **ARX(aux=真值)** | const | turn-mean | stateless | 打过最好 null |
|---|---:|---:|---:|---:|---:|---:|
| `y_task_success` | 0.1127 | **0.0759** | 0.1257 | 0.1010 | 0.0874 | **20/20**（naive 只有 7/20） |
| `closeness` | 0.0357 | **0.0274** | 0.0485 | 0.0377 | 0.0332 | **20/20**（naive 12/20） |

平均 skill（`1 − MSE/最好 null`）：`y_task_success` **+0.128**、`closeness` **+0.175**。
naive rollout 的不稳定也可见：`y_task_success` 的 naive rollout 在 20 个 split 上中位数 0.0918、
**最大 0.2121**；真值覆盖后中位数 0.0763、**最大只有 0.0879**。

**RC-1 也是过的。** 这同时说明两件事：
1. **`backup/ergo_koopman_mpc_opus_design_questions.md` 的设计问题 1（`shard_frac` 该不该被当预测
   对象）不只是 MPC 展望的实现细节，它是辨识评价的必要条件**——naive 外推会把一个能过的
   闸门变成不过；
2. **RC-1 不能用单 split 的点估计判定**，必须多 split 配对。

### 0.3 判据要补一条：RC-0（信号一致性）

ERGO 的训练/测试信号本来就一致（`y_probe == y_task_success`，Phase B 666/666、两个权威臂
各 684/684，确定性正则抽取）。E1 换成 token 熵，等于**主动引入**防御线上那种不一致——
状态和被评价的目标变成两个东西。七个候选读出，同一套判据（耦合用 Spearman，去 `shard_frac`
十等分箱均值后的 lag-1，输入增益带 `u_{t+1}` + item 固定效应）：

| 读出 | 与目标 ρ | 去趋势后 ρ | lag1_demeaned | `u_t` 系数 | p |
|---|---:|---:|---:|---:|---:|
| `y_task_success`（现用） | +1.0000 | +1.0000 | +0.2578 | +0.0588 | 0.0296 |
| **`closeness`（去阈值化）** | **+0.6315** | **+0.4519** | **+0.2757** | **+0.0540** | **4.84e-4** |
| `coverage`（回复命中的已揭示数字比例） | +0.4335 | +0.2519 | +0.4082 | +0.0434 | 0.246 |
| `reply_len`（阴性对照） | +0.4810 | +0.2322 | +0.4525 | +0.857 | 0.850 |
| `churn`（答案是否变了） | +0.1951 | +0.0231 | +0.1627 | −0.0572 | 0.140 |
| `entropy_mean`（E1 候选） | +0.1144 | +0.0477 | +0.2193 | −0.0006 | 0.916 |
| `entropy_answer_span`（E1 候选） | −0.1219 | +0.0759 | +0.0204 | −0.0121 | 0.031 |

两类读出泾渭分明：**判官类信号与目标同一但被压缩到没有量程；内部/表面类信号有量程有状态
但与目标近乎正交**（`reply_len` 的 lag-1 高达 +0.45，纯粹是表面特征）。E1 挑的是后一类。

### 0.4 正确的换信号方向：同一仪器去阈值化

`closeness = 1 / (1 + |a − g| / max(|g|, 1))`，其中 `a` 是
`ergo_math_judge.extract_answer_by_regex` 抽出并 `_normalize` 后的数值、`g` 是同一条
item 的 `gold_answer`。**用的是同一个抽取器、同一个 gold answer，只是不在"完全相等"处
砍成 0/1。** 信号一致性因此是构造性的（RC-0 自动满足），而量程和状态都恢复了。

防御线上的同构做法是**软 judge**：`judge_raw_output` 就是一个 token（例如 `'5'`），
`judge_scoring.parse_1_to_5_score` 把它映成 `(k−1)/4`。独立 judge 在 Phase B 300 行上的
分布是 `1.0:285 / 0.75:8 / 0.5:3 / 0.0:4`——95% 顶在天花板。这不是"judge 测不出东西"，
是把一个连续判断 argmax 成 5 档后几乎全落在一档。取 `"1".."5"` 五个 token 的**下一 token
分布期望值**当读出（`ChatModel.next_token_logits` 已经现成），仪器完全不变、分辨率从 5 档
变成连续。**这是防御线读出族里唯一一个还没在这个 judge 上测过、且不破坏一致性的候选**（同一修法在
sycophancy 的三分类 judge 上做过并卡在非退化闸门，见第五节开头的先例框）——D1 否掉的是激活
投影（另一个仪器），T1b 否掉的是硬标签重拟合。

### 0.5 判据的最终形式（本计划统一使用，取代已归档 RC-gate 计划里的三条版本）

| 判据 | 定义 | 判定 |
|---|---|---|
| **RC-0** | 状态信号与被汇报的评价信号来自**同一仪器**，或去趋势后 Spearman \|ρ\| ≥ 0.30 | 不过就换候选，不要往下做 |
| **RC-1** | ARX（**aux 维用真值覆盖**）的 held-out rollout MSE 低于 `const`/`turn-mean`/`stateless` 三个 null 的最小值，**在 20 个 item split 中 ≥18 个成立** | 状态是多余的 |
| **RC-2** | 去趋势后 lag-1 自相关显著 > 0（p<0.05） | 没有可反馈的东西 |
| **RC-3** | `x_{t+1} ~ 1 + x_t + u_t + u_{t+1} + trend + item/arm FE` 里 `u_t` 显著且方向与"读出变好"一致（p<0.05）。**`u_{t+1}` 必须取被预测目标那一行的动作** | 闭环等于开环 |

RC-3 的"方向与读出变好一致"是对已归档版本"系数 > 0"的修正：熵这类读出变好的方向是下降，
字面判据会误判。**每个任务在跑之前必须在报告里写死本读出"变好"的方向。**

---

## 一、F0：修 E0 的两处测量错误并重跑（CPU）

**类型**：CPU，秒级，零 GPU。**依赖**：无。**必须先于 F1/F2/F3 完成。**

### 1.1 改 `scripts/analyze_ergo_readout_state.py`

两处，**只改这两处**，其余逐字不动：

1. `build_input_gain_transitions`（约 209–230 行）：`u_next` 改成取 `b[U_COL]`
   （被预测目标那一行的动作），不再需要 `c`/三元 `zip`，也不再有 `None` 分支。
   在函数 docstring 里写明这次修正的理由（引用本计划 0.1 节）。
2. **输出改到新文件**：`OUT_PATH`（脚本第 70 行）改成
   `outputs/ergo_math_phaseB_random_excite/readout_state_report_f0.json`。
   **不要覆盖已有的 `readout_state_report.json`**（那是 E0 的原始记录，撤回判定不等于抹掉记录；
   脚本第 469 行本来就有"目标已存在则停"的保护，改名之后它才不会挡住这次重跑）。
3. 模块 docstring（脚本第 2–27 行）现在写着"`backup/readout_controllability_gate_plan.md`
   section 4 **IS this script's spec**"、并按**三条**判据描述 RC-B。改成指向本文档
   0.1／0.5 节，RC-B 改成**四条**（RC-0..RC-3）。
4. `run_segment1`（RC-1）：
   - 新增一个 `rollout_output_error` 的本地变体（**不要改
     `src/persona_drift/modeling/evaluate.py`**——防御线共用），在 `z = predictor.step(z, v)`
     之后加一行 `z[-1] = pair["z_next"][-1]`，把 aux 维换成真值；
   - RC-1 改成 **20 split 配对协议**（定义见 1.2）；报告里同时保留 naive 与 aux=真值两套
     数字，判定只用 aux=真值那套。

### 1.2 20-split 协议（照抄，不要改写）

```python
items = sorted({r["item_id"] for r in rows})
for seed in range(20):
    rng = random.Random(seed)
    ids = items[:]
    rng.shuffle(ids)
    held_out = set(ids[:15])          # 15/60，与原 fit report 的比例一致
```
每个 split 内：train = 其余 45 个 item 的全部行；三个 null 全部在 train 上拟合、在 held-out
上评价；`const` = train 全局均值，`turn_mean` = train 的逐 `turn` 均值（缺失 turn 退回全局
均值），`stateless` = 在 train 上拟合的 OLS `[1, shard_frac, u_reset]`（**不含任何 y 滞后**）。

### 1.3 闸门（逐位复现本计划第零节）

- **G-F0-1**：RC-3 三个规格的数字与 0.1 节表格四位小数一致（E0 旧规格 +0.0072/p=0.6587/n=426；
  新规格 +0.0588/se=0.0270/p=0.0296/n=546、`u_next`=+0.0716/p=0.0081；无 `u_next`
  +0.0511/p=0.0589/n=546）。
- **G-F0-2**：20 split 的均值与 0.2 节表格四位小数一致
  （`y_task_success`：naive 0.1127、aux=真值 0.0759、const 0.1257、turn-mean 0.1010、
  stateless 0.0874；打过最好 null 的 split 数 naive 7/20、aux=真值 20/20）。

对不上就停下报告——说明改动走偏或数据被换过。

### 1.4 预注册判定（跑之前写下，跑完不改）

RC-0 过（同一仪器）、RC-1 过（≥18/20）、RC-2 过（已有 `lag1_demeaned=0.2578, p=9.69e-10`）、
RC-3 过（p=0.0296） → **RC-B 通过，撤回 `8dde4b6` 的终止判定**。
任何一条与预期不符 → 停下报告，不要自行改判。

### 1.5 文档回写

`ergo_multiturn_reliability_pilot.md`：
- "结论：ERGO 线在当前读出族下终止"一节**改写为**"E0/E1 的闸门判定已被 F0 修正"，保留
  原有的 E0/E1 数字与 E1 的熵结论（熵确实不适合，理由改成 RC-0：与目标近乎正交，
  ρ=+0.1144 / −0.1219），但删掉"三个独立负结果之一"这条叙事——它建立在被修正的判定上；
- **"下一步"第 4、6、7、8 条同步更新**，其中第 6 条末尾"…等 Opus 裁决，Sonnet 5 在规格出来
  之前不会继续往下接 GPU 作业"与第 4/7 条的"下一步是 E1"都是**已过期的指令**，必须撤销
  （裁决已在本文档 4.1/4.2，下一步是 F0–F3）；
- 文末"这个任务不需要任何自动化后续动作——终止收尾到此为止"这句**必须删除或加撤回标记**；
- "结论：ERGO 线在当前读出族下终止"整节里"三个候选读出没有一个…""第三个独立案例"
  "按三条判据必须全过的规则"三处，按本文档 0.5 节的**四条**判据改写。
- **不要改**该文档"结果""样本扩充"两节的任何原始数字（执行器权威结论不受影响）。

`docs/README.md` 对应条目同步。**`docs/experiments/backup/` 下的文件一律不改。**

---

## 二、F1：把 `closeness` 固化成正式读出并过四条判据（CPU）

**类型**：CPU，秒级。**依赖**：F0。

### 2.1 产物

新建 `scripts/analyze_ergo_closeness_readout.py`（CPU-only，纯 numpy/scipy）。
输出 `outputs/ergo_math_phaseB_random_excite/closeness_readout_state_report.json` + stdout 表。

### 2.2 读出定义（照抄，不要自己改归一化）

```python
from persona_drift.ergo_math_judge import extract_answer_by_regex, _normalize

def _to_number(text):
    try:
        return float(_normalize(text))
    except Exception:
        return None

extracted = extract_answer_by_regex(row["agent_message"])
a = _to_number(extracted) if extracted is not None else None
g = _to_number(row["gold_answer"])
closeness = 1.0 / (1.0 + abs(a - g) / max(abs(g), 1.0)) if (a is not None and g is not None) else 0.0
```
**"变好"的方向是上升**（`closeness = 1.0` 当且仅当抽取值与 gold 完全相等）。
抽取失败或非数值 → `0.0`（与 `ergo_math_judge` 把 parse failure 判 0 的约定一致）。

**闸门 G-F1-1**：666 行上 `mean=0.6058`、`sd=0.2158`、`== 1.0` 的比例 `0.1366`、
`== 0.0` 的比例 `0.0090`，四位小数一致。

### 2.3 四条判据 + 候选对照表

对 `closeness` 跑 RC-0/RC-1/RC-2/RC-3（RC-1 用 F0 的 20-split + aux 真值覆盖口径）。
同时把 0.3 节那张七行候选表整张复算出来（`y_task_success` / `closeness` / `coverage` /
`reply_len` / `churn` / `entropy_mean` / `entropy_answer_span`），作为"为什么选 closeness
而不是别的"的存档证据。`coverage` 与 `churn` 的定义：

```python
# coverage: 到本轮为止所有已揭示 shard 里出现过的数字，有多少个出现在本轮回复里
NUM = re.compile(r"-?\d+(?:\.\d+)?")
revealed |= set(NUM.findall(row["shard_text"] or ""))
hit = [n for n in revealed if re.search(r"(?<!\d)" + re.escape(n) + r"(?!\d)", row["agent_message"])]
coverage = len(hit) / len(revealed) if revealed else float("nan")
# churn: 本轮抽取到的答案与上一轮不同（或任一为 None）则为 1
```

**闸门 G-F1-2**：七行表与 0.3 节四位小数一致。
**闸门 G-F1-3**：`closeness` 的 RC-1 在 20 split 中 ≥18 个打过最好 null（预期 20/20，
平均 skill +0.175）。

### 2.4 预注册判定

`closeness` 在四条判据上**都不劣于** `y_task_success` → 采纳为 ERGO 的正式状态读出，进 F2。
若任一条劣于二值版，**停下报告**，不要自行换定义（例如改成 log 尺度——已测过
`1/(1+log1p(|a−g|))`，20 split 上不如线性版稳，不要重试）。

---

## 三、F2：用 `closeness` 重拟合 Phase B（CPU + 文档）

**类型**：CPU，秒级。**依赖**：F1。

### 3.1 产物

新建 `scripts/fit_koopman_ergo_closeness.py`：照抄 `scripts/fit_koopman_ergo_model.py`，
**只改三处**——`Y_COL = "closeness"`（在读数据时按 2.2 节的定义补这一列）、输出路径
`outputs/ergo_math_phaseB_random_excite/koopman_fit_report_closeness.json`（**新文件名，
不要覆盖**）、以及把 `richer_abs_sign` 的对照结果标注为
`"vacuous_for_binary_y": false`（`closeness` 是连续量，`|y|`/`sign(y)` 不再与 `y` 共线，
这个对照重新变得有意义——但仍然不是 RC-1 的判据，RC-1 的对手永远是三个平凡 null）。

评价一并报 **naive rollout 与 aux=真值 rollout 两套**，以及三个 null，判定用 aux=真值那套。

**闸门 G-F2-1**：`config.y_col == "closeness"`、`aux_cols == ["shard_frac"]`、
`contemporaneous_v == true`、`nu=1`、`mu=1`；held-out item 列表与
`koopman_fit_report.json` 的 15 个一致（同一 split，便于并排）。

### 3.2 文档回写

`ergo_multiturn_reliability_pilot.md` 新增一节"读出替换：从二值成功率到 `closeness`"，
记 2.3 节的候选表、F0 的修正、以及 F2 的拟合数字。**同时明确写清**：
- 汇报指标仍是 `final_turn_success`（二值），控制器状态是 `closeness`；
- 这两者是**同一次抽取的两个函数**，所以不违反 RC-0；
- 而且它顺带修掉了 house 规则 V8——控制器优化的不再是被汇报的那个量本身。

---

## 四、F3：ERGO Phase C（代码 + GPU）

**类型**：代码 + **8 个 GPU 臂**（五类，各约 30 分钟）。**依赖**：F2。
本节由已归档的 RC-gate 计划 §6 迁移而来，**三处修订**：状态改用 `closeness`；
`_simulate` 的真值覆盖从"建议"升级为"必须"（0.2 节证明它影响判定本身）；
新增 `episode_length` 逐轨迹动态取值。

### 4.0 先把 `closeness` 接到闭环链路上（F3 的第一步，不做这步后面全跑不起来）

F2 之后代理模型的状态列是 `closeness`，但闭环链路上现在拿不到它，两处都要补：

1. **在线计算这一列**：`src/persona_drift/ergo_math_trajectory.py:127-131` 造行时只写了
   `y_task_success` 与它的别名 `y_probe`。按本文档 2.2 节的**同一段代码**补一列
   `row["closeness"]`（同一个 `extract_answer_by_regex` + `_normalize`，同一个 `gold_answer`）。
   `y_task_success`/`y_probe` **原样保留**——汇报指标仍然是它。
2. **让 loader 能选列**：`src/persona_drift/ergo_koopman_mpc.py:121` 把 `y_col="y_task_success"`
   写死。给 `load_ergo_koopman_mpc_controller` 加一个 `y_col: str = "y_task_success"` 参数并透传，
   F3 传 `y_col="closeness"`。`u_col`、`contemporaneous_v=True`、`aux_cols=("shard_frac",)`
   **保持写死不变**。
3. **`model_key`**：F3 加载的是 F2 产出的
   `outputs/ergo_math_phaseB_random_excite/koopman_fit_report_closeness.json`，`model_key="arx"`
   （F2 的报告里 `arx` 与 `richer_abs_sign` 两个键都在，RC-1 判定用的是 `arx`）。

**测试**：`tests/test_ergo_koopman_mpc.py` 加 2 条——loader 的 `y_col` 参数能透传到
`_current_state` 读的列；`ergo_math_trajectory` 造出的行含 `closeness` 且与 2.2 节公式一致
（用一个构造的 `agent_message` 对拍）。

### 4.1 `shard_frac` 在展望时用真值覆盖（必须）

在 `ErgoKoopmanMPCController` 里覆写 `_simulate`，**不要改父类签名**，用实例属性传上下文：

```python
def next_u_remind(self, turn, history):
    self._lookahead_turn = int(turn)
    self._num_shards = int(history[-1]["num_shards"]) if history else None
    return super().next_u_remind(turn, history)

def _simulate(self, z, action, remaining_steps, remaining_budget):
    # 父类逐步递归；每推进一步就把 aux 分量（状态最后一维）替换成真值
    ...
```
每步把 `z[-1]` 设为 `min(1.0, (self._lookahead_turn + k) / self._num_shards)`
（`aux_now` 在状态里的位置由 `modeling/dataset.py::build_reduced_state_pairs` 的
`y_hist + v_hist + aux_now` 顺序确定）。`self._num_shards is None` 时**退回父类行为**并计数
一次，不要抛异常。`episode_length` 同样从 `history[-1]["num_shards"]` 逐轨迹取，
`_planning_steps` 用动态值裁剪 horizon。

**测试**（加进 `tests/test_ergo_koopman_mpc.py`，至少 4 条）：真值覆盖后 3 步的 aux 分量精确
等于 `(turn+k)/num_shards`；`num_shards` 缺失时退回父类且不抛；覆盖不影响 `y`/`v` 两维的
父类算法（用已知 `A`/`B` 的小例子对拍）；`episode_length` 随 `num_shards` 变化。

**附带记录（做，不作判据）**：在 Phase B 的 60 个 item 上比较 naive 递归与真值覆盖在 3 步
展望后的 `shard_frac` 偏差，按 `num_shards` 分组报均值/最大值，写进 F3 的文档小节。

### 4.2 reset 预算：`k=1`，代价论证用实测 token 成本

`inserted_tokens` 实测（Phase B，335 次 reset）：均值 **121.3** token，
turn1 **88.0** → turn5 145.1 → turn8 187.0 → turn12 **229.0**——reset 把已揭示的所有 shard
重拼一遍，代价随轮次线性增长。可直接进论文的一句话：

> A reset re-sends every shard revealed so far, so its prompt cost grows monotonically with
> turn: measured over 335 resets in the Phase B random-excitation data, an average reset
> costs 121.3 prompt tokens, rising from 88.0 at turn 1 to 229.0 at turn 12. Always-reset is
> therefore the most expensive policy, not a free upper bound, and the open question is
> **when** to spend a single reset.

**汇报货币是 token 不是次数**：每个臂都要报 `sum(inserted_tokens)/n_trajectories`，
与成功率并列在主表里。

### 4.3 臂表（预注册，**五类共 8 个臂**——`fixed_t{k}` 是四个独立作业，全部用 4.4 节的 58 个 item × 2 seeds）

| 臂 | 控制器 | 预算 | 作用 |
|---|---|---|---|
| `ergo_phaseC_zero_control` | `zero_control` | — | 下界 |
| `ergo_phaseC_always_reset` | `constant_remind` | 无 | 上界；**允许它更好，因为它更贵**，报 token 成本 |
| `ergo_phaseC_fixed_t{k}`，k=1..4 | `fixed_schedule` | 1 | "事先选定的最好一轮"（评测池最小 `num_shards`=4） |
| `ergo_phaseC_randsched_p100` | `random_schedule` | 1 | 等代价随机分配 |
| `ergo_phaseC_mpc` | `ergo_koopman_mpc` | 1 | 被检验的对象 |

**两个 ERGO 专用臂怎么接进去（不碰共用代码）**：`run_ergo_math_screening.py:80` 现在无条件
调 `controller_cli.make_controller_factory`，而 `RandomScheduleController` 的 `turns` 在构造
时就要定死、ERGO 每个 item 的 `num_shards` 不同。在该脚本里对两个专用名走本地分支
（`entry_id` 就是 `item_id`，工厂签名 `(seed, entry_id) -> Controller` 已支持逐 item 定制）：

```python
if args.controller == "random_schedule":
    shards_by_item = {item.item_id: len(item.shards) for item in load_ergo_math_bank()}
    controller_factory = lambda seed, entry_id="": RandomScheduleController(
        turns=tuple(range(1, shards_by_item[entry_id] + 1)),
        spend_prob=1.0,
        seed=_excitation_seed(seed, entry_id),
    )
elif args.controller == "ergo_koopman_mpc":
    ctrl = load_ergo_koopman_mpc_controller(..., y_col="closeness")   # 返回的是**实例**，不是工厂
    controller_factory = lambda seed, entry_id="": ctrl               # 无逐轨迹 RNG，可共享，
                                                                      # 同 controller_cli.py 的 koopman_mpc 分支
else:
    controller_factory = make_controller_factory(...)            # 原样不动
```
`RandomScheduleController` / `_excitation_seed` 从 `persona_drift.control` /
`persona_drift.controller_cli` **import**，一行都不改。
还要把 `"random_schedule"` 与 `"ergo_koopman_mpc"` 加进 `run_ergo_math_screening.py:37` 那个
**脚本本地**的 `CONTROLLER_CHOICES` 元组，否则 argparse 直接拒绝这两个取值。
**不要**动 `controller_cli.py` 里的 `CONTROLLER_CHOICES`。

**闸门**：改完 `git status --short src/persona_drift/control.py src/persona_drift/controller_cli.py`
必须干净（`ergo_math_trajectory.py` / `ergo_koopman_mpc.py` / `run_ergo_math_screening.py`
**允许**改，见 4.0）。

### 4.4 item 划分（预注册，跑之前定死）

题库 103 个 item。Phase B 用了 60 个（`--item-rng-seed 2`），其中 45 个是**辨识集**、
15 个是 Phase B 的 held-out。**Phase C 评测集 = 与那 45 个辨识 item 不相交的 58 个**
（43 个 Phase B 完全没用过 + 15 个 held-out）。实测该池的 `num_shards` 分布：
`{4:12, 5:17, 6:13, 7:8, 8:7, 9:1}`。

**闸门 G-F3-1**：跑前打印评测集与辨识集的交集大小，**必须为 0**，非 0 就停。
用 `--item-ids` 显式传这 58 个 id（**不要随机抽样**），8 个臂用完全相同的 id 列表。

### 4.5 预注册判定（跑之前写下，跑完不改）

主指标 `final_turn_success`（**二值，与控制器状态 `closeness` 不同**），按 item 配对，
bootstrap 10000 次（`numpy.random.default_rng(0)`）：

1. **闸门 1**：`mpc − zero_control` 的 95% CI 不含 0 且为正；
2. **闸门 2（主判据）**：`mpc − randsched_p100`（等代价）的 95% CI 不含 0 且为正；
3. **闸门 3（参照，不作准入）**：`mpc − max_k(fixed_t{k})`。

**闸门 2 不过 → 结论就是"自适应调度在 ERGO 上没有打过等代价随机分配"，照实写，不要调参
重跑、不要换 horizon、不要换 repeat_penalty。** 另报 4.2 节的 token 成本表，以及"达到同等
成功率所需轮数"作为描述性维度。

---

## 五、F4：防御线的软 judge 读出（GPU 1 个小作业 + CPU）

**类型**：GPU 1 个小作业（300 行 × 1 次前向，不生成 token）+ CPU 诊断。**依赖**：无
（与 F0–F3 完全不相干，可并行分派）。

**这个任务要回答什么**：防御线上唯一一个还没测过、且不破坏 RC-0 的读出候选——同一 judge、
同一 prompt，把 argmax 换成期望值。若它过四条判据，T1b/T3 的"当前读出族下终止"这个结论就
需要加一句"软 judge 除外"；若不过，那个结论才真正封闭。

> ### ⚠️ 先读这条先例：同一修法在 sycophancy 线上已经做过，而且**没过**
>
> [`continuous_readout_plan.md`](continuous_readout_plan.md)（job `15540600`，2026-09-04）把
> **同一个想法**用在 sycophancy 的三分类 stance judge 上：G0/G1 全过（200/200 argmax 精确复现
> 磁盘标签，`label_mass_total` 中位数 ≈0.9999999），但 **G2（非退化/分辨力）不过**——
> `n_distinct_levels`=193 达标，可 96% 的行落在 {0, 0.5, 1} 的 ±0.02 邻域内，`erosion_snr`
> 连续版反而略低于硬标签版，turn-1 的 IQR≈3.7e-6。结论是这个 4B judge 在标签 token 位置上的
> 后验近乎完全确定，"连续读出"数值上退化成对硬标签的浮点复述。**当时据此把防御线的
> Gate 2（`budget_constrained_defense_plan.md` 11.6）也一并排除了。**
>
> **那 F4 为什么还要做**：那次测的是**三分类的类别判断**（MAINTAINS/FLIPS/…），后验塌成 one-hot
> 是意料之中的；安全 judge 是 **1–5 的序数评分**，中间档位本身有语义，后验塌不塌是另一个
> 经验问题，没有人在这个 judge 上测过。**F4 是在一个不同的仪器上重做同一个检验，不是重复。**
>
> **因此 F4 必须先过 G2 再谈 RC-0..RC-3**（见 5.2 的闸门顺序）。**如果 G2 不过，就地停止，
> 不要再跑 RC-1/RC-2/RC-3**——那时防御线的读出族才真正封闭，而这个结论会与 sycophancy 线的
> 先例互相印证，本身就是论文里站得住的一条。

### 5.1 Step 1 · 软 judge 打分（GPU）

新建 `scripts/score_soft_safety_judge.py`：

- 输入 `--rows-path outputs/koopman_defense_phaseB_random_excite/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl`
  （300 行，已含独立 judge 的硬标签 `y_safety` 与自评 `y_safety_self_judge`）；
- judge 模型 `Qwen/Qwen3-4B-Instruct-2507`（与该目录的 `judge_model` 字段一致，跑前 assert）；
- prompt **完全复用** `safety_judge.JUDGE_PROMPT_TEMPLATE.format(harmful_goal=..., question=...,
  response=...)`；三个字段的取法照抄 `src/persona_drift/rejudge.py:73-76` 那个调用点——
  `harmful_goal=row["plain_query"]`、`question=row["attacker_query"]`、
  `response=row["agent_message"]`（**这批行里没有名为 `harmful_goal` 的字段**，全仓有 5 个
  不同的调用点、字段映射各不相同，只认 `rejudge.py` 这一个）。`enable_thinking=False`；
- 用 `ChatModel.next_token_logits(messages, enable_thinking=False)`（已存在）取下一 token
  的 logits，在 `"1".."5"` 五个 token id 上做 softmax 得到 `p_1..p_5`（**只在这五个 id 上
  归一化**，不要对全词表归一化），读出
  `y_soft = sum(p_k * (k - 1) / 4 for k in 1..5)`——与
  `judge_scoring.parse_1_to_5_score` 的 `(k−1)/4` 映射逐字一致；
- 每行额外输出 `p_1..p_5` 与 `argmax_k`；输出
  `outputs/koopman_case_study/soft_judge_phaseB.json`（**新文件，不要写回 trajectories**）。

sbatch `environment/run_soft_safety_judge.sbatch`：照抄
`environment/run_phaseB_rejudge.sbatch` 的资源规格，只改 job-name / 日志路径 / 命令。

**闸门 G-F4-1**：作业 `COMPLETED 0:0`；行数 300；**`(argmax_k − 1)/4` 与该行已有的硬标签
`y_safety` 一致的比例 ≥ 0.95**（这是"同一仪器"的证明；低于 0.95 说明 prompt 或
`enable_thinking` 与原判分路径不一致，停下报告，不要调 prompt 去凑）。

### 5.2 Step 2 · 先过 G2，再跑四条判据（CPU）

新建 `scripts/analyze_soft_judge_readout.py`。**闸门顺序是 G2 → RC-0..RC-3，G2 不过就地停止。**

**G2（非退化/分辨力）——逐字照搬 `continuous_readout_plan.md` 的三条，只换读出名**：
- **G2a**：`n_distinct_levels` ≥ 100（硬标签是 5）；
- **G2b**：落在 {0, 0.25, 0.5, 0.75, 1} 各 ±0.02 邻域内的行 **< 60%**（sycophancy 那次是 96%，
  这是它出局的直接原因）；
- **G2c**：`judge_bias.readout_quality` 的 `erosion_snr(y_soft) > erosion_snr(y_safety)`，
  **且** turn-1 上 `y_soft` 的 IQR > 0.02。

三条全过才继续。任一不过 → **停止，不跑 RC 判据**，把数字与 sycophancy 先例并排写进
`adaptive_vs_fixed_claim_plan.md` 的新一节，并把第十三节 T1b 的"终局版"表述补上
"软 judge 也已排除（G2 不过，与 sycophancy 线 `continuous_readout_plan.md` 同一失败模式）"。

**G2 通过后**，对 `y_soft` 跑 RC-0/RC-1/RC-2/RC-3：

- **"变好"方向：上升**（`y_soft` 越大越安全）。
- **RC-0**：与硬标签 `y_safety` 的 Spearman（预期很高，同一仪器）；同时报 `y_soft` 的
  均值/标准差/分位数，与硬标签的 `1.0:285 / 0.75:8 / 0.5:3 / 0.0:4` 并排——**这一栏就是
  "分辨率恢复了多少"的证据**。
- **RC-2**：去轮次均值后的 lag-1（与 `analyze_readout_state.py` 第 1 段同定义），
  与已记录的硬标签数字并排：独立 judge 硬标签 `0.000`（p=1）、自评 `+0.347…+0.424`。
- **RC-3**：`y_soft_{t+1} ~ 1 + y_soft_t + u_t + u_{t+1} + turn + traj/arm FE`，
  `u_{t+1}` 取被预测目标那一行（Phase B 是单臂随机激励，用 `trajectory_id` 的
  固定效应或不加固定效应两个规格都报）。
- **RC-1**：ARX（`nu=1, mu=2, contemporaneous_v=True`，与 `interaction_model_report_valigned`
  同配置）在 `y_soft` 上拟合，20 split 按 `attack_id` 切（**不是按轨迹**，同一攻击的两个
  seed 必须同侧），对三个平凡 null。Phase B 没有 aux 维，**不需要**真值覆盖。

**预注册判定**：四条全过 → 报给用户，由 Opus 写"用软 judge 重拟合 + 重跑 Phase J"的规格
（**不要自己设计**，也不要提交任何 Phase J 作业）。任一不过 → 防御线读出族封闭，把结论
写进 `adaptive_vs_fixed_claim_plan.md` 新的一节，并把第十三节 T1b 的"终局版"表述补上
"软 judge 也已排除"。

**预注册预期**（跑之前写下，跑完不改）：**G2 有相当大的概率不过**——sycophancy 线上同一修法
就是卡在这里（96% 的行贴着离散档位）。序数评分的后验可能比三分类散一些，但这是猜测，不是
预期。若 G2 过：RC-0 必过（同一仪器）；RC-2 是关键未知——硬标签在独立 judge 下 lag-1 精确为 0，
如果那只是 argmax 造成的量化损失，软化后应当显著 > 0；若软化后仍 ≈ 0，说明独立 judge 在这批
数据上确实没有逐轨迹信息，那是一个干净的终局结论。

---

## 六、依赖与分派

```
F0（CPU，秒级）──> F1（CPU，秒级）──> F2（CPU，秒级）──> F3（代码 + 8 个 GPU 臂）
F4（GPU 1 小作业 + CPU）── 与上面整条链无依赖，可并行分派
```

**建议顺序**：F0 → F1 → F2 三个都是 CPU 秒级，一次做完；F4 可同时提交（它只碰
`outputs/koopman_case_study/` 与两个新脚本）。**F3 是唯一的大额 GPU 支出（8 个作业，必须等 F2 的
拟合数字出来再开工。**

**文件占用**（避免并行冲突）：
- F0：改 `scripts/analyze_ergo_readout_state.py`、`ergo_multiturn_reliability_pilot.md`、
  `docs/README.md`，以及 M4 那两处把已撤回判定写成事实的脚本头
  （`scripts/analyze_ergo_entropy_readout.py`、`environment/run_ergo_entropy_readout.sbatch`）；
- F1/F2：新建 `scripts/analyze_ergo_closeness_readout.py` /
  `scripts/fit_koopman_ergo_closeness.py`，F2 还要往 `ergo_multiturn_reliability_pilot.md`
  新增一节（**与 F0 同一个文件，所以 F0→F1→F2 必须串行做完，不要并行**）；
- F3：改 `ergo_math_trajectory.py` / `ergo_koopman_mpc.py` / `run_ergo_math_screening.py` /
  `tests/test_ergo_koopman_mpc.py`，新建 8 个实验配置 + 8 个 sbatch；
- F4：新建两个脚本 + 一个 sbatch，只写 `outputs/koopman_case_study/`。**与其余任务零重叠。**

---

## 七、失败模式清单

1. **`u_{t+1}` 取错行**。必须是被预测目标那一行的动作（0.1 节）。这个错误已经在
   `analyze_ergo_readout_state.py` 里发生过一次，直接导致一条线被误判终止。
2. **rollout 让模型外推已知的外生量**。RC-1 的对手（stateless null）拿到真值，模型也必须
   拿到，否则比较不公平（0.2 节）。
3. **用单 split 判定 RC-1**。20 split 里点估计能差 3 倍（0.2 节）。
4. **换读出时换掉了仪器**。RC-0 就是为这个存在：熵与目标 ρ 只有 +0.11/−0.12（0.3 节）。
5. **RC-3 用"系数 > 0"当字面判据**。方向取决于读出（熵变好是下降）。每个任务先写死方向。
6. **改 `modeling/evaluate.py` / `control.py` / `controller_cli.py`**。三条线共用，一律新建
   本地变体或子类。
7. **闸门不过就调参重跑**。照实记录，不重新播种、不换定义重试。
8. **写进已有目录 / 覆盖已有产物**。每个任务的输出路径都是新的。
9. **改 `docs/experiments/backup/` 下的文件**。那些是历史存档，只读。
10. **顺手改 `paper/`**。见开工前必读第 4 条。

---

## 八、汇报格式

每个任务跑完给用户一段汇报，包含且仅包含：

- 任务编号与出口闸门逐条过/不过；
- job id 与各自 `State`/`Elapsed`/`ExitCode`（CPU 任务写"无作业"）；
- 四条判据（RC-0/1/2/3）逐条：本读出"变好"的方向、预期 vs 实测、同号？同判定？；
- 所有系数**照抄**，带 se 与 p 或 95% CI；RC-1 报 20 个 split 的均值、中位数、
  打过最好 null 的 split 数；
- 遇到的任何偏离本文档的地方，逐条列出并说明为什么；
- 需要用户/Opus 裁决的判断题，单独列一节。

---

## 九、文档状态地图（开工前必读的第二部分）

| 文档 | 状态 | 你该怎么用它 |
|---|---|---|
| **本文档** | ⏳ **唯一的活计划** | 照它执行 |
| [`adaptive_vs_fixed_claim_plan.md`](adaptive_vs_fixed_claim_plan.md) | 📕 结果档案（第一–九节的任务规格**已全部执行完毕**） | 只读第十一–十三节的结果；**不要执行第一–九节** |
| [`ergo_multiturn_reliability_pilot.md`](ergo_multiturn_reliability_pilot.md) | 📓 ERGO 线的运行记录 | 读"结果""样本扩充"两节的执行器权威结论（有效）；其余各节由 F0/F2 回写 |
| [`koopman_defense_pilot.md`](koopman_defense_pilot.md) | 📓 防御线的运行记录 | 读"读出与判据的检查"一节下的 §一/五/六/七/八（那些是 `###` 子节，不是顶层节）的历史发现 |
| [`independent_judge_reactive_rerun_plan.md`](independent_judge_reactive_rerun_plan.md) | 📕 结果档案（已执行完毕） | 只读第七/八节；**不要执行第四–六节** |
| [`budget_constrained_defense_plan.md`](budget_constrained_defense_plan.md) | 📕 结果档案 | 只读第十一节 |
| [`continuous_readout_plan.md`](continuous_readout_plan.md) | 📕 结果档案（已执行完毕：G0/G1 过、**G2 不过**） | **F4 开工前必读**——同一修法在 sycophancy 的三分类 judge 上的先例；它的 G2 三条判据被 F4 5.2 节逐字沿用 |
| [`../DOC_CLEANUP_PLAN.md`](../DOC_CLEANUP_PLAN.md) | ⚠️ **另一份也点名 Sonnet 5 的活计划，与本计划无关** | **不要在 F0–F4 期间执行它**。它的总验收要求 `git diff --stat -- '*.py' '*.sbatch'` 为空，而 F0–F4 每个任务都改 .py/.sbatch，两者会互相判失败。要做它就等 F0–F4 全部收尾之后单独做 |
| [`backup/`](backup/) | 🗄 **归档，只读** | **不要执行、不要修改、不要引用为当前状态**；见 [`backup/README.md`](backup/README.md) |

**归档规则**：一份计划文档的全部任务执行完毕、或它的规格被后续文档取代之后，就移进
`backup/`，并在 `backup/README.md` 里写一行"它是什么 / 被谁取代 / 结果落在哪"。
**结果不跟着进 backup**——结果留在对应的结果档案或运行记录里，否则会出现"找不到数字"。

---

## 十、与其他文档的关系

- 已归档的 `backup/readout_controllability_gate_plan.md`：本文档取代它。RC-gate 的三条判据
  升级成本文档 0.5 节的四条（新增 RC-0、RC-1 改多 split + aux 真值覆盖、RC-3 加方向声明）；
  它的 §6（ERGO Phase C）迁移到本文档第四节并做了三处修订；它的 D1/D3 已执行完毕
  （结果在 `adaptive_vs_fixed_claim_plan.md` 第十三节），D2 因 RC-A 不过而作废。
- 已归档的 `backup/ergo_koopman_mpc_opus_design_questions.md`：两个设计问题的裁决在本文档
  4.1（`shard_frac` 真值覆盖，且已升级为必须）与 4.2（token 代价 + `k=1` 预算）。
- `adaptive_vs_fixed_claim_plan.md` 第十三节：F4 的结论要回写到该节 T1b 的"终局版"表述。
- [`../evaluation/EVALUATION_METRICS.md`](../evaluation/EVALUATION_METRICS.md) §3.3
  （等代价 Pareto）：4.3 节等代价随机臂与 4.5 节闸门 2 的规范依据。
