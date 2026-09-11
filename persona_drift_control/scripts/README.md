# scripts/ 清单

这份清单盘点 `persona_drift_control/scripts/` 下全部 95 个脚本，按术语基准（[`docs/NAMING.md`](../../docs/NAMING.md)）
标"线"、按引用计数（docs/environment/tests/scripts 四处 grep + 集群 `outputs/` 产物扫描）判"状态"。
**冻结-复现用**指该脚本是某条已收尾/暂停线（`defense`/`stance`/`benign`/`detect`/`persona_drift`/ERGO）
已提交结果的复现路径——按 `.claude/code.md` 不得删改，只是不再产生新结果。
此前文档里写作「ERGO」的那条线，代号是 **`gsm8k_sharded`**（2026-09-10 用户裁定按数据集命名：
它用 Laban et al. 的 sharded GSM8K，`constraint` 用 SEQUOR；ERGO 只是它借用的执行器方法名）。
见 `docs/NAMING.md` 消歧说明第 4 条。**脚本文件名里的 `ergo_*` 不改**（历史标识符）。（2026-09-10 更新：清单初版写作时 `score_sequor_trajectories.py` 正被作业 15772957 调用，
该作业已于 16:44 COMPLETED，脚本未被改动。）

**分布（2026-09-10，S1 落地后 +2）**：活跃 13 · 冻结-复现用 81 · 候删 1。计划一 C6 点名的 4 个"零引用"里
**只有 1 个真的可删**——另 3 个的引用/产物落在计划的扫描口径之外（`conf/` 下的 Hydra 配置组、
不叫 `run_*.json` 的产物目录、只在脚本 docstring 里留下的谱系）。**判"候删"前把这三处一起扫**，
否则删掉的是已提交结果的复现路径（`.claude/code.md` 硬约束）。

| 脚本 | 线 | 状态 | 说明 |
|---|---|---|---|
| `analyze_attack_risk_predictability.py` | defense | 冻结-复现用 | A1：检验多轮攻击的危险度能否仅从攻击文本本身（回复生成前）预测，决定 A2 跨轨迹预算分配是否成立 |
| `analyze_budget_allocation_replay.py` | defense | 冻结-复现用 | Phase J 预算约束臂闭环跑之前的离线 go/no-go 检查：k 次提醒预算下 v-aligned 交互 Koopman MPC 是否真的对不同轨迹在不同轮次下手 |
| `analyze_budget_arm_comparison.py` | defense | 冻结-复现用 | Phase J 预算约束设定的跨臂比较：读取各臂 trajectories.jsonl，报告实际花费的提醒次数、终轮/晚窗安全分与 new-Q1 侵蚀检验 |
| `analyze_compounding_hypothesis.py` | defense | 冻结-复现用 | v 对齐修正后，检验 mu=1 单步 ARX 模型能否解释 Phase A 持续提醒 5 轮的终轮安全分增益 |
| `analyze_continuous_readout.py` | stance | 冻结-复现用 | sycophancy judge 连续（label-token 分布）读出的 G0-G3 闸门报告 |
| `analyze_d1_self_judge_gate.py` | defense | 冻结-复现用 | D1 闸门 G-D1-1 改用自判分数评估（盲标 P 协议因 8/14 子代理不可用而无法完成时的偏离记录） |
| `analyze_ergo_append_comparison.py` | gsm8k_sharded | 冻结-复现用 | ERGO append-mode 臂比较：把 phaseC 比较脚本硬编码的闸门集合泛化到任意 append-mode 输出目录 |
| `analyze_ergo_append_context_length.py` | gsm8k_sharded | 冻结-复现用 | 重建 append-mode 下每轮 agent 实际看到的 prompt token 数，检验 E2 的 context 增长护栏 C2 |
| `analyze_ergo_authority_comparison.py` | gsm8k_sharded | 冻结-复现用 | ERGO/Laban 最小执行器权威检查的配对比较：reset 相对 zero_control 是否移动终轮任务成功率 |
| `analyze_ergo_branch_pairs.py` | gsm8k_sharded | 冻结-复现用 | EK-A 反事实分叉臂上的 G-EKA-1/2/4 闸门：同前缀下 u=1/u=0 两次生成的 closeness 差值直接估计 |
| `analyze_ergo_closeness_readout.py` | gsm8k_sharded | 冻结-复现用 | F1：把 closeness（去阈值化的正确性打分）形式化为 ERGO 候选状态读出，跑 RC-0..RC-3 四条闸门 |
| `analyze_ergo_dual_metric.py` | gsm8k_sharded | 冻结-复现用 | R2 双指标报告：每臂只报三个数，是 R2 唯一允许的报告入口 |
| `analyze_ergo_entropy_readout.py` | gsm8k_sharded | 冻结-复现用 | 计算 token 熵读出（entropy_mean/entropy_answer_span）；E0 撤回后该读出仍不可用作状态读出，但原因改了 |
| `analyze_ergo_entropy_state.py` | gsm8k_sharded | 冻结-复现用 | E1 Step2：对熵读出的两列跑与 analyze_ergo_readout_state.py 相同的 RC-2/RC-3 诊断。**产物零命中但不删**：ergo_multiturn_reliability_pilot.md 报告的 `entropy_mean`/`entropy_answer_span` 两列 RC-2/RC-3 判定（`u_t=-0.0201, p=0.00368` 等）正是本脚本 docstring 描述的计算，文档只点名了产出熵列的 analyze_ergo_entropy_readout.py——这是谱系缺口，不是脚本没用过 |
| `analyze_ergo_phaseC_comparison.py` | gsm8k_sharded | 冻结-复现用 | F3：ERGO Phase C 五臂（zero_control/always_reset/fixed_t{1..4}/randsched_p100/mpc）配对 bootstrap 比较 |
| `analyze_ergo_readout_state.py` | gsm8k_sharded | 冻结-复现用 | E0/F0：y_task_success 的可控性闸门 RC-0..RC-3，判定 Koopman MPC 控制前状态是否可用 |
| `analyze_ergo_state_action_interaction.py` | gsm8k_sharded | 冻结-复现用 | E3 Step3：在 ERGO 的 closeness 读出上拟合状态-动作交互回归 |
| `analyze_input_gain_robustness.py` | defense | 冻结-复现用 | D1：检验 u_remind→proj_pre_reply 输入增益是持续状态效应还是提醒文本留在上下文里的同轮伪影 |
| `analyze_koopman_innovation.py` | detect | 冻结-复现用 | 检测设计选项1：用已拟合的 Phase C/D Koopman 代理模型计算 Phase E 四条闭环臂的一步预测残差 |
| `analyze_koopman_mpc_cases.py` | defense | 冻结-复现用 | KoopmanMPCController 决策案例研究：在 Phase E/G 已记录轨迹上重放控制器决策逻辑，找出五类候选现象 |
| `analyze_mc_phaseA_comparison.py` | stance | 冻结-复现用 | MMLU Phase A 重跑及提醒设计变体的执行器权威比较（与 SYCON-Bench Phase A 设计相同） |
| `analyze_mu2_interaction_replay.py` | defense | 冻结-复现用 | mu 扫描后续：nu=1,mu=2（Phase C-H 用的记忆长度）配 v-alignment 修正是否解释 Phase A 观测累积增益 |
| `analyze_phaseB_seed_confound.py` | defense | 冻结-复现用 | 量化 Phase B RandomExciteController 播种 bug 的严重程度（seed 未按 (attack,seed) 对隔离） |
| `analyze_q1_turn_window_sensitivity.py` | defense | 冻结-复现用 | 检验 new-Q1 的开场轮天花板效应是否扭曲侵蚀判定（攻击序列刻意以无害问题开场） |
| `analyze_readout_state.py` | defense | 冻结-复现用 | 把 adaptive_vs_fixed_claim_plan.md 第0节的诊断固化为可重跑脚本，三段式离线诊断（CPU-only） |
| `analyze_readout_state_t1a.py` | defense | 冻结-复现用 | T1a 后续：加入 threshold_ymin1（独立 judge 最敏感阈值重跑）重算两条 claim 闸门 |
| `analyze_readout_state_t2.py` | defense | 冻结-复现用 | T2 后续：加入 randsched_p100/p75（等代价随机分配基线）重算两条 claim 闸门 |
| `analyze_readout_state_t3.py` | defense | 冻结-复现用 | T3 Step1：控制臂固定效应后，检验执行器（u_remind）是否移动激活投影读出 |
| `analyze_refusal_direction_readout.py` | defense | 冻结-复现用 | 读出(1)激活投影 vs 读出(2) judge 分数的相关性检验（安全方向投影的有效性核对） |
| `analyze_repeat_penalty_sweep.py` | defense | 冻结-复现用 | "如何证明 Koopman 动机"路径1：检验提高 repeat_penalty 能否让提醒决策状态依赖，还是只会产生一致翻转 |
| `analyze_sequor_fidelity.py` | constraint | 活跃 | 上游保真度臂的判定（预注册）：任一 harness 变体能否把留存曲线拉到基准水平，CPU-only |
| `analyze_sequor_s0_0_gates.py` | constraint | 活跃 | constraint 线 S0-0 的三条闸门（K1 读出量程、K2 状态超越单轮、K3…），CPU-only |
| `analyze_sequor_s1_pilot.py` | constraint | 活跃 | 用 S1 试点臂（job 15768661）实测方差重算 S1a 的 MDE 与所需 N，不出裁决 |
| `analyze_sequor_s1_gates.py` | constraint | 活跃 | S1 的准入闸门 G-S1（按臂算量程）与跑前签死的 S1a 估计量（t15..t20，按 (题,seed) 配对），CPU-only |
| `analyze_soft_judge_readout.py` | defense | 冻结-复现用 | F4 Step2：把四条 RC 闸门套用到 y_soft（期望值软安全 judge），判定防御线读出族是否升级 |
| `analyze_state_action_interaction.py` | defense | 冻结-复现用 | "如何证明 Koopman 动机"路径2：拟合带显式状态-动作交互控制输入的 Koopman 代理，对比 richer_abs_sign |
| `analyze_surface_feature_input_effect.py` | persona_drift | 冻结-复现用 | Q2/Q3 式 u_remind 效应检验套用到自由表层特征而非 y_probe，纯离线分析 |
| `analyze_v_alignment_fix.py` | defense | 冻结-复现用 | 验证"v 槽位错位"诊断的两条可证伪预测：v 应该是 z_next 的 y 分量直接成因而非与 z_t 同轮 |
| `audit_ground_truth_quality.py` | stance | 冻结-复现用 | 人工审计 SYCON-Bench False Presuppositions 20 题的 correction 真值，报告剔除不成立题目后头条数字的变化 |
| `audit_judge_disagreements.py` | defense | 冻结-复现用 | 打印自判/独立 judge 最大分歧背后的原始回复文本，供人工判断哪个 judge 对 |
| `audit_prompt_scorers.py` | persona_drift | 冻结-复现用 | 零 GPU 审计：用 classify_scorer_screening_safety 对排除已饱和题后的题库里每个 score_fn 分类 |
| `backfill_surface_features.py` | persona_drift | 冻结-复现用 | 在已完成的 trajectories.jsonl 上回填自由表层特征，并对每行跑 zero_control 漂移检验 |
| `benchmark_controller_decision_cost.py` | defense | 冻结-复现用 | 控制器自身决策延迟基准（与 token/插入次数成本分开）：koopman_mpc 的 MPC 前向模拟 vs periodic/threshold |
| `build_authority_check_sample.py` | defense | 冻结-复现用 | D1 §1.2：构建执行器权威裁决检查的盲样本（constant_remind 相对 zero_control 是否减少"实质协助"） |
| `build_d1_blind_sample.py` | defense | 冻结-复现用 | D1 基线率筛查：构建每臂计算 G-D1-1 assisted rate 所用的盲样本 |
| `build_judge_validation_sample.py` | defense | 冻结-复现用 | 构建 judge 有效性研究（P 协议）的盲标样本：`--pilot` 建 70 题分歧层试点 |
| `build_len1024_authority_check_sample.py` | defense | 冻结-复现用 | B1 §4.2：盲标 len1024 zero_control 诊断臂，检验回复上限从 256 提到 1024 后基线协助率是否上升。**产物零命中但不删**：它的产物是 `outputs/authority_check_len1024/`（blind_items/key/labels_pass1-2/b1_analysis_report/gate_G_B1_2，实测存在），不是 `run_*.json`，计划的扫描口径看不见它 |
| `calibrate_safety_direction.py` | defense | 冻结-复现用 | channel-C 安全方向校准 CLI：需 GPU，写出 safety_direction.npy 供 dose-response 筛查用 |
| `calibrate_sequor_judge.py` | constraint | 活跃 | S0 判分校准（HF-transformers）：候选 judge 在 2000 行 SEQUOR 金标集上能否分开约束遵守/违反 |
| `calibrate_sequor_judge_vllm.py` | constraint | 活跃 | 与 calibrate_sequor_judge.py 相同的 S0 判分校准，换 vLLM 后端把 2000 行耗时从 4.12h 降下来 |
| `compare_judge_runs.py` | stance | 冻结-复现用 | 自评 judge vs 独立 judge 的配对比较，量化自评偏差（sycophancy 三分类标签） |
| `compare_safety_judge_runs.py` | defense | 冻结-复现用 | 自判 vs 独立 judge 比较（防御线版，CPU-only），对照 compare_judge_runs.py 在 sycophancy 线的做法 |
| `evaluate_koopman_detector.py` | detect | 冻结-复现用 | 检测设计选项3 step2：双 regime 残差比较检测器，按一步预测谁拟合更好分类 held-out 轨迹 |
| `fast_track_status.py` | 通用 | 活跃 | Koopman fast track 的现状仪表盘脚本，每次接手工作前先跑，汇总各线状态 |
| `fit_koopman_ae_baseline.py` | defense | 冻结-复现用 | AE（编码器-解码器）Koopman 基线：在与 richer_abs_sign 相同的 Phase B 数据/切分/状态配置上拟合 |
| `fit_koopman_ae_hydra.py` | defense | 冻结-复现用 | AE Koopman 基线的任务级 Hydra 入口（新增入口，不替换 fit_koopman_ae_baseline.py）。**不删**：`conf/fit_koopman_ae.yaml` 整组配置就是为它写的，`conf/task/defense.yaml` 里也有一段注明"只被它读"的旋钮——计划的四处扫描不含 `conf/` |
| `fit_koopman_benign_model.py` | detect | 冻结-复现用 | 检测设计选项3 step1：拟合"良性 regime" Koopman 代理，与已有"攻击 regime"模型配对 |
| `fit_koopman_defense_model.py` | defense | 冻结-复现用 | Phase C：在 Phase B 开环随机激励轨迹上拟合 Koopman 代理，评估一步/rollout 误差与可控性诊断 |
| `fit_koopman_ergo_branch_lifted.py` | gsm8k_sharded | 冻结-复现用 | EK-A 反事实分叉臂上的升维（含双线性）Koopman-with-control，正规化版 fit_koopman_ergo_branch.py |
| `fit_koopman_sequor_model.py` | constraint | 活跃 | S2：在 S1b 反相激励臂上拟合 y_{t+1}=Ay_t+Bu_t+c，跑 G-S2-1..4 与状态来源诊断，CPU-only |
| `fit_koopman_ergo_branch.py` | gsm8k_sharded | 冻结-复现用 | EK-A 门 G-EKA-3：在反事实分叉臂上拟合受控算子，用三个平凡零假设按题目不相交折检验 |
| `fit_koopman_ergo_closeness.py` | gsm8k_sharded | 冻结-复现用 | F2：用 closeness 读出替代 y_task_success 重跑 Phase B 拟合（复用 fit_koopman_ergo_model.py 结构） |
| `fit_koopman_ergo_model.py` | gsm8k_sharded | 冻结-复现用 | ERGO/Laban 线 Phase B/C：在开环随机激励轨迹（u_reset/y_task_success）上拟合 Koopman 代理 |
| `fit_koopman_hydra.py` | defense | 冻结-复现用 | ARX/richer_abs_sign 岭回归拟合的任务级 Hydra 入口（新增入口，不替换 fit_koopman_defense_model.py） |
| `fit_koopman_lstm_baseline.py` | defense | 冻结-复现用 | LSTM 代理基线：在与 richer_abs_sign 相同的 Phase B 数据/切分上拟合，报告一步与 rollout MSE |
| `generate_user_scripts.py` | persona_drift | 冻结-复现用 | 一次性离线生成脚本化用户轮次库：每个话题先跑一次真实自聊拿回复脚手架，再生成多份用户轮次脚本 |
| `interim_report.py` | 通用 | 候删 | 对进行中（未跑完）的 trajectories.jsonl 跑 analyze_screening，不用等任务写出 screening_report.md。四处引用 + 集群 `outputs/` 全递归扫描均零命中，且不产出任何被报告的数字 |
| `probe_sequor_response_length.py` | constraint | 活跃 | constraint 线响应长度探测：S1 需要多大的回复上限，哪些题会撞到它 |
| `reanalyze_dose_response_subset.py` | defense | 冻结-复现用 | 对已有 dose_response_rows.jsonl 重跑分析，可限制部分 alpha 值，诊断极端 alpha 是否驱动了报告效应 |
| `rebuild_sequor_arm_report.py` | constraint | 活跃 | 从磁盘已有的行重建 S0-0 某臂的 arm_report.json（生成阶段完好，只有汇总阶段崩了时用） |
| `rejudge_safety_runs.py` | defense | 冻结-复现用 | 用独立 judge 模型重新打分已收集的攻击轨迹（GPU），修正防御线此前自判==agent 的判分 |
| `run_adversarial_screening_hydra.py` | defense | 冻结-复现用 | 对抗防御筛查试点的 Hydra 驱动 CLI，把 enable_thinking 做成可换配置组 |
| `run_adversarial_screening.py` | defense | 冻结-复现用 | 对抗防御筛查试点 CLI（需 GPU），写出 trajectories.jsonl 与筛查报告 |
| `run_benign_helpfulness_screening.py` | benign | 冻结-复现用 | Phase F 良性代价检查 CLI：把 Phase E 某控制器臂部署到固定 MT-Bench 良性会话集而非攻击题库 |
| `run_d1_screening.py` | defense | 冻结-复现用 | D1 基线率筛查的轻量 CLI：接受外部确定好的 attack-id 列表，不做自己的分层采样 |
| `run_defended_screening.py` | defense | 冻结-复现用 | 带防御控制器的对抗筛查 CLI：channel-A 安全提醒执行器由 control.py 的 Controller 实现驱动 |
| `run_dose_response_screening.py` | defense | 冻结-复现用 | 单轮安全方向剂量-反应筛查 step CLI（需已校准方向、需 GPU） |
| `run_ergo_branch_arm.py` | gsm8k_sharded | 冻结-复现用 | EK-A 反事实分叉识别臂：每题×seed 跑一条基础轨迹，每轮额外从同前缀生成相反动作的一轮 |
| `run_ergo_math_screening.py` | gsm8k_sharded | 冻结-复现用 | ERGO/Laban 分片 GSM8K 最小执行器权威检查 CLI：简化版"reset"执行器移动 y_task_success 的效应 |
| `run_eroded_dose_response_screening.py` | defense | 冻结-复现用 | 侵蚀上下文剂量-反应筛查变体 CLI（需已校准方向与 step-1 筛查轨迹） |
| `run_mc_sycophancy_defended_screening.py` | stance | 冻结-复现用 | MMLU-sycophancy 线执行器权威检查（Phase A 对应版）：channel-A 提醒能否移动 y_consistency |
| `run_mc_sycophancy_screening.py` | stance | 冻结-复现用 | 基于 MMLU 的 sycophancy 筛查 CLI，替代 run_sycophancy_screening.py 的 SYCON-Bench 题库 |
| `run_pressure_screening.py` | persona_drift | 冻结-复现用 | 渐进人格施压确认试点 CLI（需 GPU），persona_drift 放弃前的最后一批筛查之一 |
| `run_screening_hydra.py` | defense | 冻结-复现用 | 带防御控制器的对抗筛查流水线的任务级 Hydra 入口（与 run_defended_screening.py 底层代码相同） |
| `run_sequor_fidelity_arm.py` | constraint | 活跃 | constraint 线上游保真度臂：本 harness 相对上游基准的留存曲线差距有多大 |
| `run_sequor_s0_0_branch_arm.py` | constraint | 活跃 | constraint 线 S0-0 反事实分叉臂：每轮从同前缀生成两次（u=0/u=1），测单次提醒的一步增益 |
| `run_sequor_s1_pilot_arm.py` | constraint | 活跃 | constraint 线 S1 定量试点：实测 S1a 的 MDE 到底是多少（计划里假设的 sd≈0.30 待验证） |
| `run_signal_screening.py` | persona_drift | 冻结-复现用 | 实验前信号筛查试点 CLI（persona_drift 线最初的筛查，需 GPU） |
| `run_sycophancy_defended_screening.py` | stance | 冻结-复现用 | sycophancy 漂移线执行器权威检查 CLI：consistency_reminder.py 的 channel-A 提醒能否移动 y_consistency |
| `run_sycophancy_screening.py` | stance | 冻结-复现用 | sycophancy 漂移筛查试点 CLI（需 GPU），镜像 run_adversarial_screening.py |
| `score_sequor_trajectories.py` | constraint | 活跃（当前被作业 15772957 调用，正在跑） | 对 constraint 线某臂轨迹套用分级读出 y_t=(满足的约束数)/k，k=3，复用上游 SEQUOR 判分 prompt |
| `score_soft_safety_judge.py` | defense | 冻结-复现用 | F4 Step1：用软（期望值）安全 judge 给 Phase B 已判分的行重新打分，分辨率从5档变连续值 |
| `score_sycophancy_continuous.py` | stance | 冻结-复现用 | 把连续（label-token 分布）judge 读出回填到两个已收集的 sycophancy 筛查跑，不重新生成 agent 文本 |
| `select_d1_attacks.py` | defense | 冻结-复现用 | 为 D1 基线率筛查选出 100 个攻击题池（按 Opus 的排除/分层裁决） |
| `split_d1_blind_batches.py` | defense | 冻结-复现用 | 把 D1 盲样本切成 pass1/pass2 两批×7 组共 14 份，各用独立打乱种子（P 协议） |
