# 命令通道 $B$ 的可辨识性与持续激励采集（#4）

> **状态**：4a ✅ 完成（结果 [`command_channel_identifiability_results.md`](command_channel_identifiability_results.md)）；
> 4b ✅ 闸门**过**（2026-11-05，作业 19443006；三 seed $CB$ = 0.734 ± 0.009，区间均不含 0）→ 结果 [`core_excite_b_identifiability_results.md`](core_excite_b_identifiability_results.md)。
> 4b v2（分叉设计，§3.5）⏸ **等裁决**：开关 + 单测 + 干跑 + LEDGER 预登记已完成，**未提交**。
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

### 3.5 4b v2：同前缀分叉（2026-11-05 用户提出；参照 `tsar_cefr` GPU-2a 反事实树）

**为什么**：3.1 的随机命令拆掉了「命令 = 目标 − 读数」，但 $CB$ 仍是**跨 prompt** 比出来的。v2 在**同一 prompt、同一段历史**下只改命令。

- **主轨迹**：`--command-policy branch` 按落盘协议推进（每轮重问目标 $r$，即 `error`），`trajectories.jsonl` 与 `error` 臂逐字段相同，只多 `command_*` 四列（runner 级回归单测已证）。
- **分叉**：每条轨迹一个分叉点 $k\sim U\{2,4,6\}$。固定回复 $1..k{-}1$ 与之前的反馈，把第 $k{-}1$ 轮之后的反馈换成 4b 模板（如实报告上一问 $r$，再问 $c_j$），$c_j\sim U\{5..100\}$，$j=1..3$，与目标、读数无关（`default_rng([seed, i])`）。各分支生成回复 $k$ 后即停，不回灌主轨迹。脱靶路径下 $c_j=r$ 的分支文本与落盘反馈逐字相同（干跑 212–214/252；其余是「差 1 词以内」的回复，4b 模板措辞不同，但三个兄弟分支用同一模板）⇒ 兄弟分支之间**只差所问的数**。
- **记录**（`branch_points.jsonl`，每分叉点一行）：`trajectory_id`、`k`、`prefix_state`（$y_{k-1},y_{k-2}$ 的原始值与归一化值；$k=2$ 时 $y_0$ 为 null）、三分支的命令 / 反馈 / 完整 prompt / 回复 $k$ 读数 / 是否触顶、主轨迹自身的回复 $k$、`git_sha`、`command_policy=branch`、`disable_thinking=true`。runner 拒绝不带 `--disable-thinking` 的 branch 调用。
- **规模**：3 seed × 252 条 ×（10 轮主轨迹 + 1 分叉 × 3 分支）= 3 × (2520 + 756) = 9,828 次生成。产物 `persona_drift_control/outputs/core_branch_sentence_length_<seed>/`（全新，runner 遇到已有文件就报错）。
- **运行时**：估 **8–20 min**（`--time 1:00:00`）。依据：v1 作业 19443006 跑 3 × 2520 次用了 6m11，其中引擎启动约 1.5 min × 3；v2 每 seed 生成多 1.3 倍；题目数 252、轮数 10、`max_new_tokens=256`（落盘解码）、thinking 关、Qwen3-4B、单卡 A100。
- ⚠ 主轨迹用贪心 + `error` 策略，**三 seed 的主轨迹相同**（只有 $k$ 与分支命令随 seed 变；vLLM 批次不确定性除外）。生成量的 2/3 是重复的，约 4 min，按指令保留。

**拟合（采集前固定，`scripts/core_branch_b_identifiability.py`）**：

1. **配对差分**：每个分叉点内对 $(c_j, y_k^{(j)})$ 去均值，$\widehat{CB}=\sum\Delta c\,\Delta y/\sum\Delta c^2$。$K z_{k-1}$ 与 prompt 截距在点内是常数，被消掉。单位与 v1 相同（归一化请求字数 → 归一化回复字数），可直接和 0.734 比。轨迹级 bootstrap 2000 次（`default_rng(0)`，每条轨迹一个点）。
2. **秩检查**：在全部转移上（主轨迹 + $k\ge3$ 的分支）跑 `scripts/paper_side_experiments.py::_identifiability`，取 $z_t=[y_t,y_{t-1},r]$，$V$ = 下一轮所问的归一化字数；要求秩满且 $R^2(V\,|\,[Z|1])<0.99$。$z$ 与 headroom 的状态 $[y_t,y_{t-1},r-y_t,r-y_{t-1}]$ 列空间相同，但去掉了内部恒等式 $r-y_{t-1}=(r-y_t)+y_t-y_{t-1}$——留着它，$[Z|V|1]$ 无论命令如何都秩亏，秩检查就测不到命令。同时把**只用主轨迹**的那一列写成 `main_only`（预期秩亏，即在新语料上复现 headroom §二）。
3. 单测：合成线性系统（prompt 截距与目标相关 0.4）上，配对差分复原 $CB=0.6$ 到 ±0.02、区间不含 0；主轨迹单独秩亏、$R^2=1$；加上分支后秩满、$R^2=0.61$；$CB=0$ 时区间含 0。

**闸门三行（事前签；1、3 行照 3.2 抄，只改「过了」的判据）**：

| 字段 | 内容 |
|---|---|
| 过了 → | 秩满且 $R^2<0.99$（三 seed）**且**配对差分 $\widehat{CB}$ 区间不含 0 **且**三 seed 符号一致 ⇒ 把 $K,B$ 交 Umesh 算 Gramian / 可达集；#3 的 prelim 用这组 |
| 不过 → | **关闭**「`core` 任务命令通道」这条线；记为「命令通道权限弱」，对应 proposal 式 (3.33)；不二次修采集 |
| 填哪张图/表 | proposal Thrust 3 prelim 一句 + 可达集小图；论文不填 |

**照抄后与现状不符、须用户裁决的三处**（agent 没有改）：① v1 已经「过」，交 Umesh 这一步在 v1 之后已经解锁、在等裁决。v2「过了」时的实际增量是：交出去的 $CB$ 用 v1 还是 v2 的。② #3 已判 future work，v1 结果文档也说明随机命令数据做不了 #3（v2 同理，没有控制器日志）。③ 若 v2「不过」而 v1「过」，第二行（关闭）就和已经落盘的 v1 结论冲突。
**执行位置与停机点**：开关、单测、干跑、本节、sbatch `slurm/run_core_branch_sentence_length.sbatch`、LEDGER 预登记——本地，已完成。**到此停止，不提交。**

## 四、死亡条件

4b 跑完三 seed 后 $B$ 仍不可辨识或区间含 0 ⇒ 3.2 第二行兑现，不再在 `core` 任务上追命令效应。

## 五、完成 = 什么

4a：结果表文档 + LEDGER 一行。4b：开关代码 + 单测过 + sbatch 干跑通过 + LEDGER 预登记 + 本文件状态改「⏸ 等裁决」。
