# 交接记录：adaptive_vs_fixed_claim_plan.md 执行中断点

**写于**：2026-09-06 22:23 EDT，因用户预计离开 6+ 小时、当前会话可能失联而创建。
**目的**：让任何一个全新的 Claude Code 会话（不带本次对话上下文）能看这一份文件，
在几分钟内判断"哪些做完了、哪些还在跑、接下来具体敲什么命令"，不需要从头重跑。

**如果你是接手的新会话，先读这份文件，再读
[`adaptive_vs_fixed_claim_plan.md`](adaptive_vs_fixed_claim_plan.md) 本体获取每个任务的完整规格。**
本文件只记状态和"接下来做什么"，不重复规格。

---

## 0. 运行环境提醒（新会话大概率会踩的坑）

- 工作目录：`/home/hcao2/autoencoder_koopman_core/persona_drift_control`
- 登录节点**没有** `conda`/`activate` 命令。激活环境用：
  ```bash
  export PATH=/scratch/hcao2/envs/persona_drift_pilot/bin:$PATH
  ```
  这样 `python`（3.10.20，能 `import persona_drift, torch`）和其余工具都可用。
  `sbatch`/`squeue`/`sacct` 已经在系统 PATH（`/usr/bin`）。
- `scontrol` 在这台机器上坏了（`libhistory.so.7` 缺失），**不要用它**查作业详情，
  用 `squeue -u hcao2` 和 `sacct -j <jobid>` 即可。
- **这个 Claude Code 会话本身跑在一个 SLURM 交互作业里**（Open OnDemand VSCode，
  作业号 `15610886`，`interact` 分区），通过 code-server 提供。code-server 配置了
  `--idle-timeout-seconds=1800`——**断开连接超过 30 分钟，code-server 可能自行关闭**，
  连带杀掉当时还活着的所有子 agent。**已经提交给 SLURM 的 GPU 作业不受影响，会自己跑完**，
  只是"作业跑完后自动做闸门检查/写报告/commit"这一段可能会被打断，需要人工接手。

---

## 1. 各任务状态一览（写这份文件时刻）

| 任务 | 状态 | 备注 |
|---|---|---|
| T0 | ✅ 完成，已 commit | commit `d315b4e` |
| T1a | 🟡 GPU 作业跑着，闸门检查未做 | job `15617755`，见第 2 节 |
| T1b | ✅ 完成（NO-GO，正确停在 Step 4，未提交 Step 5） | commit `d137ed4`，见第 3 节 |
| T2 | 🟡 代码已改完（未 commit），2 个 GPU 作业跑着 | job `15617773`/`15617774`，见第 4 节 |
| T3 Step 1 | ✅ 完成，判定 CONTINUE | commit `410067d`，见第 5 节；Step 2 未做，等 Opus 出规格 |
| T7（收尾） | 未开始 | 依赖 T1a/T1b/T2/T3 全部有结果 |

`git status --short` 在写这份文件时的样子（**这是预期中的中间状态，不要 `git checkout`/`reset`/`clean` 掉**）：
```
 M persona_drift_control/conf/task/defense.yaml
 M persona_drift_control/scripts/run_defended_screening.py
 M persona_drift_control/scripts/run_screening_hydra.py
 M persona_drift_control/src/persona_drift/control.py
 M persona_drift_control/src/persona_drift/controller_cli.py
 M persona_drift_control/tests/test_control.py
?? persona_drift_control/conf/experiment/phaseJ_budget1_randsched_p100.yaml
?? persona_drift_control/conf/experiment/phaseJ_budget1_randsched_p75.yaml
?? persona_drift_control/conf/experiment/phaseJ_indepjudge_threshold_ymin1.yaml
?? persona_drift_control/environment/run_phaseJ_budget1_randsched_p100.sbatch
?? persona_drift_control/environment/run_phaseJ_budget1_randsched_p75.sbatch
?? persona_drift_control/environment/run_phaseJ_indepjudge_threshold_ymin1.sbatch
```
（`docs/README.md`、`docs/experiments/independent_judge_reactive_rerun_plan.md`、
`docs/experiments/koopman_defense_pilot.md` 的修改是本次会话开始前就存在的、与本计划无关的
其他工作，不要碰。）

---

## 2. T1a：threshold y_min=1 独立 judge 重跑

**SLURM job**：`15617755`（`pdc-phaseJ-threshold-indepjudg`），写这份文件时 `RUNNING`，已跑 24+ 分钟。
**产物目录**：`persona_drift_control/outputs/koopman_defense_phaseJ_budget1_threshold_indepjudge_ymin1/`
（写这份文件时只有 `hydra_run/` 和 `hydra_run_config.json`，`trajectories.jsonl` 还没写出来）。
**已创建但未 commit 的文件**：
`conf/experiment/phaseJ_indepjudge_threshold_ymin1.yaml`、
`environment/run_phaseJ_indepjudge_threshold_ymin1.sbatch`。

**接手步骤**：
1. `sacct -j 15617755 --format=JobID,State,ExitCode,Elapsed` 确认 `COMPLETED 0:0`。
2. 按计划文档 T1a 第 2.4 节的出口闸门逐条核对（200 行、`judge_model` 全部是
   `Qwen/Qwen3-4B-Instruct-2507`、`judge_parse_failure` 全 false、报提醒/轨迹）。
3. 写一个新的独立脚本（例如 `scripts/analyze_readout_state_t1a.py`，**不要改**
   `scripts/analyze_readout_state.py` 本体），import 它里面的
   `per_trajectory_arm_metrics`/`paired_bootstrap`/`mixed_baseline_bootstrap` 等函数，
   把这个新臂加进两道闸门重算一遍（`threshold_ymin1` vs `zero_control` 的独立 judge 结果）。
4. 对照 T1a 2.4 节末尾的预注册预期（提醒/轨迹落在 0.075–0.30；两道闸门仍不过）。
5. `git add` 只加这次新建的文件，commit，然后按计划文档第九节格式汇报。

---

## 3. T1b：koopman 独立 judge 重拟合（已完成，NO-GO）

**结论**：Step 1–3 全过，Step 3 的 B 系数符号未翻转，但 Step 4 离线回放显示重拟合后的模型
在全部 40 条轨迹上**从不提醒**（`n_never_spends=40`，`n_distinct_spend_turns=0`）——按准入条件
判定 NO-GO，**未提交** Step 5 的 GPU 重跑，`conf/experiment/phaseJ_indepjudge_koopman_refit.yaml`
和对应 sbatch **都没有被创建**，这是有意为之，不是遗漏。

**已产出并 commit 的文件**（commit `d137ed4`）：`environment/run_phaseB_rejudge.sbatch`。
**未 commit 但已在磁盘上的数据产物**（`outputs/` 已 gitignore，不用管）：
- `outputs/koopman_defense_phaseB_random_excite/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl`（300行）
- `outputs/koopman_case_study/rejudge_manifest_phaseB.json`
- `outputs/koopman_fits/defense/fit_koopman/20260906_215723/koopman_fit_report.json`
- `outputs/koopman_case_study/interaction_model_report_valigned_indepjudge.json`
- `outputs/koopman_case_study/budget_allocation_replay_indepjudge.json`

**关键数字**：独立 judge 重拟合模型 `held_out_rollout_mse=0.0333`（自评版 0.0703，参照 arx/richer
自评 0.0510/0.0430，独立 0.0315/0.0315）；决策回放 `n_remind=0/48`（自评版 20/48）。

**留给用户/Opus 的判断题**（尚未裁决，写这份文件时仍开放）：
1. 这个"模型本身在独立 judge 分数上学出提醒无用"是否要正式写进 T7 收尾；
2. 要不要换一批独立 judge 回放数据再验证一次这个"从不行动"对数据集是否敏感。

**这个任务不需要任何后续动作**，除非用户/Opus 就上面两条给出裁决。

---

## 4. T2：等代价随机调度基线

**代码改动状态**：`src/persona_drift/control.py`（新增 `RandomScheduleController`）、
`src/persona_drift/controller_cli.py`（新增 `random_schedule` 工厂分支）、
`conf/task/defense.yaml`（新增两个 null 配置键）、`scripts/run_screening_hydra.py`、
`scripts/run_defended_screening.py`、`tests/test_control.py` 均已修改，**但尚未 commit**——
这些改动本身应该已经跑过 `pytest tests/ -q` 验证过（fork 的任务里包含这一步），接手时可以先
重新跑一遍确认没有回归，再决定要不要相信它已经测试过。

**SLURM jobs**（写这份文件时都 `RUNNING`）：
- `15617773`（`pdc-phaseJ-randsched-p100`），已跑 23+ 分钟，输出目录
  `outputs/koopman_defense_phaseJ_budget1_randsched_p100/`
- `15617774`（`pdc-phaseJ-randsched-p75`），已跑 17+ 分钟，输出目录
  `outputs/koopman_defense_phaseJ_budget1_randsched_p75/`

**已创建但未 commit 的文件**：
`conf/experiment/phaseJ_budget1_randsched_p100.yaml`、`conf/experiment/phaseJ_budget1_randsched_p75.yaml`、
`environment/run_phaseJ_budget1_randsched_p100.sbatch`、`environment/run_phaseJ_budget1_randsched_p75.sbatch`。

**接手步骤**：
1. `python -m pytest tests/ -q` 确认只有已知的 5 个 NLTK 失败（`test_surface_features.py`），
   没有新增失败——尤其确认 `tests/test_control.py` 里新增的 5 条 `RandomScheduleController`
   测试全过（spend_prob=0/1 边界、同 seed 可复现、0.75 与 1.0 同 seed 落点相同的配对不变量、
   `remind_budget=0` 报错）。
2. `sacct -j 15617773 --format=...` 和 `-j 15617774` 确认都 `COMPLETED 0:0`。
3. 按计划文档 T2 第 4.7 节的出口闸门逐条核对：200 行/臂、`judge_parse_failure` 全 false、
   `judge_model` 是自评（`Qwen/Qwen3-4B`，不是独立 checkpoint）、`p100` 提醒/轨迹恰好 1.000、
   `p75` 落在 0.60–0.90、报 `p100` 五轮的落点分布。
4. 跑 `scripts/analyze_budget_arm_comparison.py`（4.7 节给的确切命令）。
5. 写一个新脚本（例如 `scripts/analyze_readout_state_t2.py`，同样**不要改**
   `scripts/analyze_readout_state.py` 本体）把 `randsched_p100`/`randsched_p75` 加进两道闸门
   （只用自评口径，这两个臂没有独立 judge 版本）。**这是本任务的主结果**——重点看
   `koopman_b1` vs `randsched_p75`（真正等代价的对手）的 `late_y` 差是否落在文档预注册的
   [-0.03, +0.08] 且 CI 跨零；若显著偏离，不要自己下结论，报给用户/Opus。
6. `git add` 只加 T2 涉及的具体文件（上面列的改动文件 + 新建文件），不要 `git add -A`，commit。
7. 按计划文档第九节格式汇报。

---

## 5. T3 Step 1：已完成，判定 CONTINUE

**结论**：G3-1/G3-2 全过，行 C（920行，加臂固定效应）的 `proj_pre_reply` 上 `u_remind` 系数
`+5.538`，`p=6.0e-6` → **继续 Step 2**。

**已 commit**（commit `410067d`）：`environment/run_refusal_direction_readout_expanded.sbatch`、
`scripts/analyze_readout_state_t3.py`。
数据产物（gitignored，只在磁盘上）：`outputs/koopman_case_study/refusal_direction_readout_expanded.json`、
`outputs/koopman_case_study/readout_state_report_t3.json`。

**这个任务不需要任何自动化后续动作**——Step 2 的回归规格（投影量纲/是否标准化/pre 还是
post/要不要把 y_safety 当第二观测）明确是 Opus 看到这些数字之后再写，接手会话**不要**
自己设计 Step 2，除非用户明确要求。

---

## 6. 全部四个任务都有结果之后

按计划文档第六节（T7）综合收尾：把 T1a/T1b/T2/T3 的最终数字汇总，判断能不能支持
"自适应调度抗侵蚀更强"这个 claim（目前看：T1b NO-GO、T3 continue 但 Step2 未做，
claim 大概率仍然站不住，具体措辞见文档第 6.1-6.3 节），在
`adaptive_vs_fixed_claim_plan.md` 追加"执行结果"一节，并在
`koopman_defense_pilot.md` 末尾加一节指针。**不要改 `paper/` 下任何文件**——
论文措辞是 Opus 的活。
