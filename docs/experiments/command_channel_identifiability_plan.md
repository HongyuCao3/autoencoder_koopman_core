# 命令通道 $B$ 的可辨识性与持续激励采集（#4）

> **状态**：4a ✅ 完成（结果 [`command_channel_identifiability_results.md`](command_channel_identifiability_results.md)）；
> 4b ✅ 闸门**过**（2026-11-05，作业 19443006；三 seed $CB$ = 0.734 ± 0.009，区间均不含 0）→ 结果 [`core_excite_b_identifiability_results.md`](core_excite_b_identifiability_results.md)。
>
> 原状态：▶ 待执行（2026-10-25 立项，主计划 [`post_meeting_2026-10-25_master_plan.md`](post_meeting_2026-10-25_master_plan.md)）。
> 目的：在 Umesh 用 $K,B$ 算 Gramian / 可达集之前，确定哪些任务的 $B$ 是数据辨识出来的。
> 4a 零 GPU；4b 是 GPU 采集，**agent 只写到 sbatch 干跑，不提交**（`.claude/experiments.md`）。

## 一、已知事实（不重算，直接引用）

| 任务组 | $B$ 可辨识？ | 依据 |
|---|---|---|
| `core` 八任务（Table 1 前五列 + Joint + 三个 T=5） | **否**。$R=r-y_t$ 是 $[Z\|1]$ 的精确仿射函数，设计矩阵秩亏 | [`core_multiobjective_planning_headroom.md`](core_multiobjective_planning_headroom.md) §二 |
| `constraint` / `defense`（⚠自判）/ `gsm8k_sharded` | **是**（动作真随机化，秩检查过） | [`paper_side_experiments_results.md`](paper_side_experiments_results.md) §2.1 |
| `tsar_cefr` | **是**（秩 8/8，$v$ 对 $[Z\|1]$ 逐维 $R^2<10^{-3}$） | 同上 §5.1 |
| 控制线（IFBench / IFEval / COLLIE，Table 2） | **未知**。控制器选的 $c_t\ne r$ 常见，理论上有变异 | 代码与日志**不在本仓库**（论文 App. D） |

论文 App. F「4/7 列命令通道不携带信息」与第一行同源：不是效应小，是**问题无定义**。

## 二、4a：现有产物上补齐（零 GPU）

### 2.1 做什么

1. 对 `constraint` / `defense` / `tsar_cefr`：在已落盘算子上做轨迹级 bootstrap 2000 次，报告 $B$ 每个元素的
   95% 区间与符号稳定率；复用 `paper_side_experiments._identifiability` 的输出作同址附注。
2. 对 `core` 八任务：**不重算**，表里写「不可辨识，引用 headroom §二」。
3. 控制线：向用户要日志路径（主计划 §五.2）。拿到后用同一脚本跑 `_identifiability` + bootstrap。
4. 产出 `docs/experiments/command_channel_identifiability_results.md`：一张表，列 = 任务 / 可辨识 / $B$ 区间 / 可用于 Gramian？

### 2.2 执行位置

| 步骤 | 位置 |
|---|---|
| 2.1-1（行为线三列） | **服务器**：`persona_drift_control/outputs/` 本地为空；`git pull` 后跑 `persona_drift_control/scripts/` 下新增的 `b_identifiability_bootstrap.py` |
| 2.1-2 | 本地（只写表） |
| 2.1-3 | 取决于日志位置 |

### 2.3 判定（给 Umesh 的回答）

「可用于 Gramian」= 秩检查过 **且** $B$ 主元素区间不含 0 **且** 符号稳定率 ≥ 0.95。
按此，预期答案：`tsar_cefr`、`constraint`（`defense` 带自判标注）可用；`core` 八列不可用；控制线待查。

## 三、4b：持续激励采集（仅当 4a 判「`core` 不可用且控制线不可得」时启动）

### 3.1 设计

- 任务：`sentence_length_t10`（默认；用户可加 `sentiment_t5`）。模型、解码、题目与原采集一致
  （`scripts/run_core_collection_arm.py`）。
- **唯一改动**：每轮命令 $c_t$ 不再取 $r-y_t$，改为从命令范围均匀随机抽（持续激励），记录到轨迹字段 `command`。
  若 `run_core_collection_arm.py` 无此开关，按 `.claude/code.md` 新增 `--command-policy {error,random}`，默认 `error` 保持旧行为。
- 规模：42 题 × 6 轮 × **3 seed**（`global.md` 最低 seed 数）。
- 产物：`persona_drift_control/outputs/core_excite_sentence_length_<seed>/`，记 `git rev-parse HEAD`、`command_policy`、judge 版本。

### 3.2 闸门三行（事前签）

| 字段 | 内容 |
|---|---|
| 过了 → | $B$ 区间不含 0 且三 seed 符号一致 ⇒ 把 $K,B$ 交 Umesh 算 Gramian / 可达集；#3 的 prelim 用这组 |
| 不过 → | **关闭**「`core` 任务命令通道」这条线；记为「命令通道权限弱」，对应 proposal 式 (3.33)；不二次修采集 |
| 填哪张图/表 | proposal Thrust 3 prelim 一句 + 可达集小图；论文不填 |

### 3.3 运行时估计

756 次生成 / seed-任务，`max_new_tokens=256`，Qwen3-4B，batch 16，单卡 A100：原采集 42 题 × 10 轮的实测时长
按 `docs/LEDGER.md` §一同类作业折算，**预计 20–40 分钟/seed**，三 seed 串行 < 2 小时。缺的信息：原采集单作业时长（agent 在 LEDGER 里查后更新此行）。

**更新（2026-10-25，agent 查后）**：本仓库没有原采集的作业记录（`core` 语料是同事交付的成品）；最近的同类作业是
16035914（同一 runner、同一任务、Qwen3-4B、A100），`sentence_length_t10` 含一次引擎启动约 8 min。
落盘协议是 **252 条轨迹 × 10 轮**（168/42/42 划分），不是上文的「42 题 × 6 轮」，按 runner 的单位三 seed 共 7,560 次生成，
估 **0.25–0.75 h**（依据写在 `slurm/run_core_excite_sentence_length.sbatch` 头注）。**用户 2026-10-25 裁决：按实际协议 252×10×3 seed。**
**两个提交前阻塞**：① `/scratch/hcao2/envs/sequor_vllm` 的 Python 标准库被 scratch 清理删掉，env 须重建并在其中重做干跑；
② runner 原先不关 Qwen3 的 thinking——16035914 在本任务上 90.6% 行触顶 256 token、字数数进了 `<think>`；已加 `--disable-thinking`（默认关、旧调用不变），sbatch 打开它。

### 3.4 执行位置与停机点

- 开关实现 + 单测 + `argparse` 干跑：**本地**。
- sbatch 写到 `slurm/run_core_excite_sentence_length.sbatch`；LEDGER §三追加一行（「改变哪个决定」= Umesh 能否算 Gramian）。
- **到此停止，等用户裁决。** 提交、`sacct` 回填、结果拟合：服务器，用户或下一会话执行。

## 四、死亡条件

4b 跑完三 seed 后 $B$ 仍不可辨识或区间含 0 ⇒ 3.2 第二行兑现，不再在 `core` 任务上追命令效应。

## 五、完成 = 什么

4a：结果表文档 + LEDGER 一行。4b：开关代码 + 单测过 + sbatch 干跑通过 + LEDGER 预登记 + 本文件状态改「⏸ 等裁决」。
