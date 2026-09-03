# Persona Drift Control

`autoencoder_koopman_core`（这个仓库）是我自己 fork 之后的仓库，不是需要
和同事分开维护的共享仓库，所以这个子项目就直接放在这里，不再单独建仓库。
（早前一版曾以为要与同事仓库分开维护，短暂建过一个独立仓库
`persona_drift_control`，后来发现前提不成立，已经用 `git merge
--allow-unrelated-histories` 把那个仓库的完整历史并回这里，`git log --all
--oneline` 仍能看到原来的独立 commit。）

协议、方法实现细节、baseline 调研等文档全部在 `../docs/`，
以 [`../docs/README.md`](../docs/README.md)
为索引入口，本目录不重复存放，以那边为准。

## 现在做到哪一步了

**最初只有协议第 7 节要求的"采集前 1 小时信号探针"（`signal_screening`），后来在同一套
`Controller`/`modeling` 骨架上长出了四条实验线。** 各线的状态、job id、结论一律以
`../docs/README.md` 的索引为准，本文件不重复：

| 实验线 | 入口脚本 | 状态 |
|---|---|---|
| ① 人格漂移 screening（最初的 gate） | `run_signal_screening.py` | 三问全挂；10-prompt 放大后仍是空结果；渐进施压版（`run_pressure_screening.py`）中间态 |
| ② 对抗防御 Koopman-MPC（主线） | `run_adversarial_screening.py` / `run_defended_screening.py` | Phase A→I 完整闭环并已收尾：`koopman_mpc` 打赢 `zero_control`/`threshold`，但未打赢同代价的 `periodic` |
| ③ Koopman 显式检测支线 | `evaluate_koopman_detector.py` | 方案 1/3/4 跑完，修完"v 对齐"bug 后方案 1/3 由负结果转为正向 |
| ④ sycophancy drift screening | `run_sycophancy_screening.py` | 两次 GPU 跑完（自评 judge + 独立 judge 配对重跑），欠功效的空结果 |

代理建模层另有 ARX / `richer_abs_sign` / LSTM / AE 四个 baseline 的对照（`fit_koopman_*.py`）。

## 目录结构

```text
persona_drift_control/
├── resources/                 # 第三方题库的离线缓存（探针题库、MT-Bench、SYCON-Bench；见 PROVENANCE.md）
├── conf/                      # Hydra 配置层（screening / fit_koopman / task / experiment / generation）
├── src/persona_drift/
│   ├── chat_model.py           # transformers 封装：加载模型、按 seed 生成
│   ├── control.py              # Controller 协议 + 全部控制器（zero/constant/threshold/periodic/koopman_mpc…）
│   ├── trajectory_runner.py    # 三条领域线共享的"逐轮提醒-生成-判分"循环骨架
│   ├── selfchat.py             # ① 人格漂移：模拟用户+agent 自聊，逐轮探针分叉打分
│   ├── attack_trajectory.py    # ② 对抗防御：攻击序列回放（trajectory_runner 的薄封装）
│   ├── benign_trajectory.py    # ② 良性 helpfulness 代价对照（同上）
│   ├── sycophancy_trajectory.py# ④ sycophancy：SYCON-Bench 反驳回放（同上）
│   ├── *_bank.py / *_reminder.py / *_judge.py   # 各线的题库、提醒文本、LLM-judge 判分
│   ├── analysis*.py            # 各线的判据（new-Q1/new-Q3、离散翻转、helpfulness、剂量-响应…）
│   ├── *_screening.py          # 各线的编排层：断点续跑 + 写 trajectories.jsonl + report
│   └── modeling/               # koopman.py / dataset.py / evaluate.py + lstm_baseline / ae_baseline / interaction_lift
├── scripts/                   # 36 个 CLI 入口（run_* 跑实验，fit_* 拟合模型，analyze_* 离线分析）
├── environment/               # setup_env.sh + 每个实验一份的 *.sbatch，日志在 slurm_logs/
└── tests/                     # 45 个离线单测文件，不需要模型/GPU
```

## 探针题库到协议分类的映射（需要确认）

协议文档写的"character traits"和"language constraints"两类，在实际用到的
`Naomibas/llm-system-prompts-benchmark`（100 条）数据集里，并不是现成字段——
该数据集自己的分类是 `pattern` / `multiple_choice` / `persona` /
`memorization` / `language`（且 `language` 只有 3 条，指"讲法语"，不是协议
意义上的"语言/风格约束"）。按协议自己给出的筛选标准（"探针分数更接近连续
量"），`prompt_bank.py` 把 `persona_system_prompts`（14 条）当作
character_traits，把 `pattern_system_prompts`（29 条，字母/大小写/词数/句
式等格式约束）当作 language_constraints。这个映射是我按标准做的判断，不是
协议原文写死的，值得你确认一遍是否符合本意。

## 与协议字面规模的一处偏离

协议第 7 节要求"5 条 prompt × 2 seed × 16 轮"（10 条轨迹）。但问题 1 需要
`u≡0` 的纯漂移轨迹，问题 2/3 需要 `u_remind` 有变化的轨迹——两者不能用同一
批轨迹回答。因此 `screening.py` 对每个 (prompt, seed) 都跑了 `zero_control`
和 `excite_iid` 两种条件，一共 20 条轨迹，粗略是协议估计的"1 小时"的两倍
左右。这个决定和取舍见 `screening.py` 顶部注释。

## 运行

环境已经用 `environment/setup_env.sh` 建好了（`/scratch/hcao2/envs/persona_drift_pilot`）。
Slurm 命令行调试/提交作业的具体做法见 `RUNNING_ON_PALMETTO.md`（这台机器是
Clemson Palmetto 2，写代码用的沙箱没有 python/conda/nvidia-smi/sbatch，
所以那份文档也没有实际跑过，第一次用时留意资源参数是否合理）。以下命令都
在本目录（`autoencoder_koopman_core/persona_drift_control/`）下执行：

```bash
cd persona_drift_control                # 若当前在仓库根目录，先进这个子目录
bash environment/setup_env.sh          # 一次性：建 conda env，装依赖，下 nltk 数据
# 之后每次运行前:
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
source activate /scratch/hcao2/envs/persona_drift_pilot

python scripts/run_signal_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --user-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/signal_screening
```

上面是实验线 ① 的入口。**其余三条线各有自己的 CLI 和 sbatch**，参数、job id 和判定口径写在
对应的实验文档里（`../docs/experiments/`），不在这里重复：

| 线 | CLI | sbatch |
|---|---|---|
| ② 对抗防御 screening / 带防御重跑 | `run_adversarial_screening.py`、`run_defended_screening.py`、`run_benign_helpfulness_screening.py` | `environment/run_adversarial_screening.sbatch`、`run_koopman_defense_phase*.sbatch` |
| ③ 检测支线 | `evaluate_koopman_detector.py` | 离线 CPU，无 sbatch |
| ④ sycophancy screening | `run_sycophancy_screening.py` | `environment/run_sycophancy_screening.sbatch`、`run_sycophancy_screening_independent_judge.sbatch` |
| 代理模型拟合/对照 | `fit_koopman_defense_model.py`、`fit_koopman_lstm_baseline.py`、`fit_koopman_ae_baseline.py` | 多为 CPU 直跑 |

集群上一律用对应的 `environment/*.sbatch` 提交，而不是手敲上面的裸命令——sbatch 里固定了
`HF_HOME`/`NLTK_DATA`/环境激活和资源参数。部分脚本另有 Hydra 版本（`*_hydra.py` + `conf/`），
用于需要按 task/experiment 切配置、且不能让两次跑互相覆盖输出目录的场合。

模型权重和 HF 缓存目录默认在 `/scratch/hcao2/hf_cache`（`chat_model.py` 里
兜底设置了 `HF_HOME`，忘记 export 也不会落到 home 目录）。结果写到
`outputs/signal_screening/`：`trajectories.jsonl`（逐轮原始记录，字段与
协议第 5/8 节及同事 `core.py` 的列约定兼容）、`screening_report.json` 和
`screening_report.md`（三个问题的判定结果）。

## 测试

```bash
pytest -q
```

45 个测试文件，覆盖各线的题库加载、提醒文本构造、判分解析、轨迹循环、判据统计、控制器决策
与 `modeling/` 的拟合/评测，全部是纯逻辑，不需要 GPU 或联网（第三方题库已离线缓存在
`resources/`，见 `resources/PROVENANCE.md`）。已知的既有环境问题：1 个 loguru flaky 测试、
`test_surface_features.py` 因缺 `nltk vader_lexicon` 数据报 5 个错——都与具体改动无关，
排查时先排除这两项。
