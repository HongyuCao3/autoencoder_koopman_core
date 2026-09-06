# 执行计划：反应式臂的独立 judge 重跑（高优先级）

**状态**：待执行。**适用智能体：Sonnet 5**（规格已写死在本文档里，不需要自己做设计判断）。
**触发来源**：论文 Step 2b 的作废判定（`../../paper/evidence/superseded.md` 事件 E3
与第五节第 2 条）。**阻塞**：论文 §Experiments 里任何"以独立 judge 为准"的臂间结论。

> **开工前必读**：本文档是**完整规格**。不要凭记忆复述、不要"优化"命令行、不要合并步骤。
> 每一步都有出口闸门，闸门不过就停下来报告，不要自己想办法绕过去。
> 遇到本文档没写到的判断题（例如"某个数字该不该改进论文"），**停下来问用户**，
> 不要自行裁决——作废判定是 Opus 的活。

---

## 一、为什么必须真跑 GPU（这一段决定了整个设计，先读懂）

`koopman_defense_pilot.md` 第七节做过一次**离线重打分**：用独立 judge
（`Qwen/Qwen3-4B-Instruct-2507`）把 10 个臂 1605 行重新打了一遍分，agent 文本直接复用、
一个 token 都没重新生成。那次结论是：自评偏差单向、随轮次增长（+0.072/轮，p=0.0000），
**独立 judge 下 10 个臂的 new-Q1 全部不显著**。

但离线重打分**只能重新测量、不能重新决策**：

- **固定臂**（`zero_control` / `constant_remind` / `periodic` / `fixed_t1`–`fixed_t5`）的
  提醒日程与 judge 分**无关**，所以"换 judge 重新打分"在数学上等价于"换 judge 重跑"该臂。
  **这些臂不需要重跑，已有的离线重打分产物就是最终结果。**
- **反应式臂**（`threshold` / `koopman_mpc` / `koopman_mpc_interaction` / `threshold_budget1`
  / `koopman_budget1`）把 `y_probe` 当反馈来**决定下一步插不插提醒**。它们的**决策**是在
  自评分数上做出的，离线改不了——重打分只会给"一条由自评 judge 引导出来的轨迹"换一套分数，
  那不是"独立 judge 下这个控制器会怎么表现"。

所以本次要跑的，**只有反应式臂**。这不是"为了更保险再跑一遍"，是这五个臂的现有数字在
独立 judge 口径下**不存在**。

---

## 二、范围（照抄，不要增删臂）

### 2.1 必跑（5 个臂）

| # | 臂 | 原产物目录 | 新产物目录 | 轨迹数 | 驱动方式 |
|---|---|---|---|---:|---|
| A1 | Phase J 自适应臂 | `outputs/koopman_defense_phaseJ_budget1_koopman` | `outputs/koopman_defense_phaseJ_budget1_koopman_indepjudge` | 40 | Hydra |
| A2 | Phase J threshold | `outputs/koopman_defense_phaseJ_budget1_threshold` | `outputs/koopman_defense_phaseJ_budget1_threshold_indepjudge` | 40 | Hydra |
| A3 | Phase E threshold | `outputs/koopman_defense_phaseE_threshold` | `outputs/koopman_defense_phaseE_threshold_indepjudge` | 16 | argparse |
| A4 | Phase E koopman_mpc | `outputs/koopman_defense_phaseE_koopman_mpc` | `outputs/koopman_defense_phaseE_koopman_mpc_indepjudge` | 16 | argparse |
| A5 | Phase I（v-aligned 交互模型） | `outputs/koopman_defense_phaseI_koopman_mpc_valigned` | `outputs/koopman_defense_phaseI_koopman_mpc_valigned_indepjudge` | 16 | argparse |

**优先级**：A1/A2 最高（Phase J 是论文的主结论），A3/A4/A5 次之。**如果 GPU 排队紧张，
先交 A1/A2，跑完再交 A3–A5**——两批之间没有依赖。

### 2.2 明确不跑

- 全部固定臂（理由见第一节）。
- **Phase H**（`koopman_mpc_interaction` 旧对齐）：已被 Phase I 取代，Step 2b 判为
  `superseded`，重跑一个已作废的臂没有意义。
- **良性场景（Phase F/G/I benign）**：`y_help` 用的是另一个 judge prompt，本次的偏差量化
  只覆盖 `y_safety`。良性侧要不要一起做是**另一个决定**，不在本次范围内，不要顺手加。
- **不要重跑任何固定臂"以求一致"**。固定臂的独立 judge 分已经存在且是精确配对的
  （复用同一段 agent 文本），重跑反而会引入 agent 采样噪声，让对比变差。

### 2.3 成本估算

按 `budget_constrained_defense_plan.md` 11.4 节记的 ~42 秒/轨迹算：
A1+A2 = 80 条 ≈ 56 分钟 + 模型加载；A3+A4+A5 = 48 条 ≈ 34 分钟 + 加载。
每个作业单独申请 1 张 A100、`--time 01:00:00` 足够（沿用现有 sbatch 的规格）。
**总计约 1.5 GPU-小时。**

---

## 三、开工前的前置检查（全部通过才提交作业）

按顺序执行，任何一条不过就停下报告：

```bash
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

# P1 原产物存在且行数符合预期（40*5=200 / 16*5=80）
for d in koopman_defense_phaseJ_budget1_koopman koopman_defense_phaseJ_budget1_threshold \
         koopman_defense_phaseE_threshold koopman_defense_phaseE_koopman_mpc \
         koopman_defense_phaseI_koopman_mpc_valigned; do
  echo -n "$d: "; wc -l < outputs/$d/trajectories.jsonl
done
# 期望：前两个 200，后三个 80

# P2 新产物目录必须【尚不存在】——这是本次最重要的一道闸门
ls -d outputs/*_indepjudge 2>/dev/null && echo "!! 已存在，停下来问用户 !!" || echo "P2 OK: 无同名目录"

# P3 环境可用
source activate /scratch/hcao2/envs/persona_drift_pilot
python -c "import persona_drift, torch; print('env OK', torch.__version__)"

# P4 CPU 测试全绿（改任何代码之前先记下基线）
python -m pytest tests/ -q 2>&1 | tail -3
```

**P4 的已知既有失败**（与本次无关，不要去修）：1 个 loguru flaky 测试、5 个
`test_surface_features.py` 的 NLTK 数据缺失失败。除此之外必须全绿。

---

## 四、执行步骤

### Step 1 · 建两个 Phase J 的 Hydra 实验文件

Phase J 走 `run_screening_hydra.py`，纪律是**一个文件 = 一个臂 = 一个 `output_dir`**
（`conf/screening.yaml` 顶部写明了原因：`run_config_guard` 靠这个防止两个臂混进同一个目录）。
所以**不要用命令行覆盖 `output_dir`**，老老实实新建两个文件。

新建 `conf/experiment/phaseJ_indepjudge_koopman.yaml`：

```yaml
# @package _global_
# Phase J 自适应臂的独立 judge 重跑（docs/experiments/independent_judge_reactive_rerun_plan.md）。
# 与 phaseJ_budget1_koopman.yaml 逐字相同，只有两处差异，且两处都是本次实验的自变量：
#   * judge_model: 从 null（自评，judge==agent）改成独立 checkpoint；
#   * output_dir: 新目录，绝不覆盖自评那批已记录的产物。
# 为什么反应式臂必须真跑而不能离线重打分：控制器拿 y_probe 当反馈决定下一步动作，
# 离线换 judge 只能换分数、换不掉决策。
defaults:
  - phaseJ_base

output_dir: outputs/koopman_defense_phaseJ_budget1_koopman_indepjudge
task:
  screening:
    judge_model: Qwen/Qwen3-4B-Instruct-2507
    controller: koopman_mpc_interaction
    koopman_interaction_model_path: outputs/koopman_case_study/interaction_model_report_valigned.json
    koopman_nu: 1
    koopman_mu: 2
    koopman_horizon: 5
    koopman_repeat_penalty: 0.0
    koopman_contemporaneous_v: true
    koopman_pad_short_history: true
```

新建 `conf/experiment/phaseJ_indepjudge_threshold.yaml`：

```yaml
# @package _global_
# Phase J threshold 臂的独立 judge 重跑。与 phaseJ_budget1_threshold.yaml 逐字相同，
# 只有 judge_model 与 output_dir 两处差异。
defaults:
  - phaseJ_base

output_dir: outputs/koopman_defense_phaseJ_budget1_threshold_indepjudge
task:
  screening:
    judge_model: Qwen/Qwen3-4B-Instruct-2507
    controller: threshold
    threshold_y_min: 0.7
```

**出口闸门（不启动 GPU 就能查）**：

```bash
python scripts/run_screening_hydra.py experiment=phaseJ_indepjudge_koopman --cfg job \
  | grep -E 'output_dir|judge_model|controller|horizon|pad_short_history|seeds|remind_budget'
```
必须看到 `judge_model: Qwen/Qwen3-4B-Instruct-2507`、`output_dir: ..._indepjudge`、
`seeds: [0,1,2,3,4]`、`remind_budget: 1`、`koopman_horizon: 5`、
`koopman_pad_short_history: true`。两个文件都查。**任何一项对不上就停下**。

### Step 2 · 建 5 个 sbatch

照抄现有 sbatch 的头部规格（`--cpus-per-task 8 --mem 32G --time 01:00:00 --gpus a100:1`、
`module load anaconda3`、`source activate /scratch/hcao2/envs/persona_drift_pilot`、
`export HF_HOME=/scratch/hcao2/hf_cache`、`export NLTK_DATA=/scratch/hcao2/nltk_data`、
`cd .../persona_drift_control`、`python -m pip install -e . -q`），只改 job-name、
日志路径与最后的命令。文件名：

- `environment/run_phaseJ_indepjudge_koopman.sbatch`
  → `python scripts/run_screening_hydra.py experiment=phaseJ_indepjudge_koopman`
- `environment/run_phaseJ_indepjudge_threshold.sbatch`
  → `python scripts/run_screening_hydra.py experiment=phaseJ_indepjudge_threshold`

**注意：不要传 `allow_config_change=true`。** 新目录里没有旧 config 可冲突，守卫本来就会放行；
传了反而会把这道保险关掉。

- `environment/run_phaseE_indepjudge_threshold.sbatch`：

```bash
python scripts/run_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/koopman_defense_phaseE_threshold_indepjudge \
  --seeds 0 1 \
  --controller threshold --threshold-y-min 0.7 \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
```

- `environment/run_phaseE_indepjudge_koopman_mpc.sbatch`：

```bash
python scripts/run_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/koopman_defense_phaseE_koopman_mpc_indepjudge \
  --seeds 0 1 \
  --controller koopman_mpc \
  --koopman-model-key richer_abs_sign \
  --koopman-nu 1 --koopman-mu 2 --koopman-horizon 2 --koopman-repeat-penalty 0.0 \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
```

> ⚠️ **A4 有一个坑，必须照上面写、不要"顺手修正"**：Phase E 的 `koopman_mpc` 原臂用的是
> **旧对齐**的 `koopman_fit_report.json`（`--koopman-contemporaneous-v` 没有传）。本次是
> "同一个控制器换 judge"的对照，**必须保持旧对齐**，否则同时改了两个变量，结果无法归因。
> 已知这个臂的模型带 v 对齐 bug（`paper/evidence/superseded.md` 事件 E1），这正是它在论文里
> 只能作为历史对照的原因，不是本次要修的东西。

- `environment/run_phaseI_indepjudge_koopman_mpc_valigned.sbatch`：

```bash
python scripts/run_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/koopman_defense_phaseI_koopman_mpc_valigned_indepjudge \
  --seeds 0 1 \
  --controller koopman_mpc_interaction \
  --koopman-nu 1 --koopman-mu 2 --koopman-horizon 2 --koopman-repeat-penalty 0.0 \
  --koopman-contemporaneous-v \
  --koopman-interaction-model-path outputs/koopman_case_study/interaction_model_report_valigned.json \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
```

**出口闸门**：`bash -n <每个 sbatch>` 语法通过；逐字比对新 sbatch 与被复制的原 sbatch，
差异**只能**出现在 job-name、日志路径、`--judge-model`、`--output-dir`/`experiment=` 这几处。
把这个 diff 贴进汇报里。

### Step 3 · 提交并等待

```bash
sbatch environment/run_phaseJ_indepjudge_koopman.sbatch
sbatch environment/run_phaseJ_indepjudge_threshold.sbatch
# 跑完再交这三个（或同时交，若队列宽松）
sbatch environment/run_phaseE_indepjudge_threshold.sbatch
sbatch environment/run_phaseE_indepjudge_koopman_mpc.sbatch
sbatch environment/run_phaseI_indepjudge_koopman_mpc_valigned.sbatch

squeue --me
sacct -j <jobid> --format=JobID,JobName%30,State,Elapsed,ExitCode
```

**记下每个 job id**，最后要写进文档。

**出口闸门**：全部 `COMPLETED` + `ExitCode 0:0`；行数对得上（A1/A2 各 200 行，
A3/A4/A5 各 80 行）；`judge_parse_failure_rate` 为 0：

```bash
for d in outputs/*_indepjudge; do
  echo -n "$d rows="; wc -l < $d/trajectories.jsonl
  python - <<PY
import json,collections
rows=[json.loads(l) for l in open("$d/trajectories.jsonl")]
print("  judge_model:", collections.Counter(r.get("judge_model") for r in rows))
print("  parse_failures:", sum(bool(r.get("judge_parse_failure")) for r in rows))
PY
done
```
`judge_model` 必须**全部**是 `Qwen/Qwen3-4B-Instruct-2507`——只要有一行是
`Qwen/Qwen3-4B`，说明 `--judge-model` 没传进去，**整个作业作废重来**。

### Step 4 · 把新臂纳入统一口径的对比

现有的 `analyze_budget_arm_comparison.py` 用一个 `--scores-name` 读所有臂，所以要让
"固定臂用离线重打分文件、反应式臂用新跑出来的文件"落在**同一个相对路径**上。做法是对新臂
也跑一次 `rejudge_safety_runs.py`——它用同一个 judge、同一套 judge seed
（`seed*1e6 + turn*100 + 2`）重打分，对一份已经是独立 judge 的文件来说**应当逐位复现原分数**，
所以这一步既统一了路径，**又是一次自洽性检查**：

```bash
# CPU 少量 GPU，几分钟；只写 <arm>/rejudge_qwen3_4b_instruct_2507/，不动原文件
python scripts/rejudge_safety_runs.py --device cuda \
  --arm-dir outputs/koopman_defense_phaseJ_budget1_koopman_indepjudge \
  --arm-dir outputs/koopman_defense_phaseJ_budget1_threshold_indepjudge \
  --arm-dir outputs/koopman_defense_phaseE_threshold_indepjudge \
  --arm-dir outputs/koopman_defense_phaseE_koopman_mpc_indepjudge \
  --arm-dir outputs/koopman_defense_phaseI_koopman_mpc_valigned_indepjudge \
  --manifest-path outputs/koopman_case_study/rejudge_manifest_indepjudge.json
```

**出口闸门（自洽性检查，很重要）**：逐行比对每个新臂的
`trajectories.jsonl` 与 `rejudge_qwen3_4b_instruct_2507/trajectories.jsonl` 的 `y_safety`。
**期望 100% 相同**。如果不同，说明在线 judge 调用与离线 rejudge 的 seed/prompt 口径不一致
——**这是一个真 bug，停下来报告，不要继续往下算**（它同时意味着第七节那批离线重打分数字
本身需要复核）。

然后跑对比（Phase J，5 seed，独立 judge 口径）：

```bash
python scripts/analyze_budget_arm_comparison.py \
  --scores-name rejudge_qwen3_4b_instruct_2507/trajectories.jsonl \
  --arm koopman_budget1=outputs/koopman_defense_phaseJ_budget1_koopman_indepjudge \
  --arm threshold_budget1=outputs/koopman_defense_phaseJ_budget1_threshold_indepjudge \
  --arm fixed_t1_budget1=outputs/koopman_defense_phaseJ_budget1_fixed_t1 \
  --arm fixed_t2_budget1=outputs/koopman_defense_phaseJ_budget1_fixed_t2 \
  --arm fixed_t3_budget1=outputs/koopman_defense_phaseJ_budget1_fixed_t3 \
  --arm fixed_t4_budget1=outputs/koopman_defense_phaseJ_budget1_fixed_t4 \
  --arm fixed_t5_budget1=outputs/koopman_defense_phaseJ_budget1_fixed_t5 \
  --out-path outputs/koopman_case_study/budget_arm_comparison_indepjudge_rerun.json
```

（固定臂指向**原目录**是对的——它们的 `rejudge_.../trajectories.jsonl` 早就存在。
`--out-path` 显式给了新文件名，绝不覆盖已有产物。）

**出口闸门**：JSON 产出成功；`n` 对得上（自适应臂 40 条配对）；把
`koopman_budget1` vs 最优固定臂、vs `threshold_budget1` 的均值差与 95% CI 抄出来。

### Step 5 · 归档

1. 在**本文档末尾**追加一节"执行结果"，写清楚：job id、每臂行数、Step 4 自洽性检查的结论、
   Phase J 在独立 judge 口径下的臂间对比表（均值差 + CI）、以及**与自评口径的定性对照**
   （结论是否同号、是否同判定）。数字**照抄**，不要四舍五入、不要重新措辞。
2. 在 `koopman_defense_pilot.md` 第七节末尾追加一句指针，指向本文档（不要改第七节原文——
   那一节记录的是离线重打分，本次是另一件事）。
3. 在 `budget_constrained_defense_plan.md` 第十一节末尾追加一句指针。
4. 在 `docs/README.md` 的 experiments 列表里给本文档加一条索引。
5. `git commit`。

**不要做的**：不要改 `paper/evidence/numbers.yaml` 的任何 `status`，不要往里加条目，
不要改 `paper/evidence/superseded.md`。新数字进 evidence 台账是论文 Step 2a/2b 的活，
由 Opus 在另一个会话按那边的 schema 处理。你只负责把结果写进实验文档并汇报。

---

## 五、失败模式清单（这几条都真实发生过或差点发生）

1. **写进已有目录**。screening 循环天生可续跑，指错目录不会报错，只会安静地把两个控制器的
   轨迹混进同一份报告。P2 那道检查就是为这个存在。
2. **`--judge-model` 没生效**。`run_defended_screening.py` 里 `judge_model = args.judge_model
   or args.agent_model`——参数名拼错会静默退回自评。Step 3 的 `judge_model` 计数就是查这个。
3. **A4 顺手加了 `--koopman-contemporaneous-v`**。见第四节 Step 2 的警告框：那会同时改两个
   变量。
4. **用命令行覆盖 Hydra 的 `output_dir` 而不是新建实验文件**。能跑通，但破坏了
   "一个文件 = 一个臂 = 一个目录"的可审计性，下次有人复现时无从知道这个目录是怎么来的。
5. **拿新臂和自评口径的固定臂比**。整个重跑的意义就是让所有臂落在同一个 judge 口径下；
   Step 4 的 `--scores-name` 必须对所有臂生效，混口径比较等于白跑。
6. **把 Phase F/G 的良性结果一起改了**。不在范围内，见 2.2。

---

## 六、汇报格式

跑完后给用户一段汇报，包含且仅包含：

- 5 个 job id 与各自 `State`/`Elapsed`/`ExitCode`；
- 每臂行数与 `judge_model` 计数；
- Step 4 自洽性检查结论（100% 一致 / 不一致的行数与例子）；
- 独立 judge 口径下 Phase J 的臂间对比表；
- 与自评口径的定性对照一句话（同号？同判定？）；
- 遇到的任何偏离本文档的地方，逐条列出并说明为什么。
