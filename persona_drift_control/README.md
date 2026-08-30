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

当前只实现了协议第 7 节要求的**采集前 1 小时信号探针**——这是正式采集
（40 prompt × 2 通道 × 4 seed）之前的强制关卡：三个问题任一不过，都要先改协议
再采数据，而不是先跑几千次生成再发现设计有问题。

## 目录结构

```text
persona_drift_control/
├── resources/                    # 第三方探针题库的离线缓存（见 PROVENANCE.md）
├── src/persona_drift/
│   ├── prompt_bank.py             # 加载探针题库，选 character_traits / language_constraints 两类
│   ├── reminder.py                 # 通道 A（u_remind）文本构造，探针阶段只用 0/1 两档
│   ├── chat_model.py                # transformers 封装：加载模型、按 seed 生成
│   ├── selfchat.py                   # 单条轨迹：模拟用户+agent 自聊，逐轮探针分叉打分
│   ├── analysis.py                    # 协议第7节三个问题的判定
│   └── screening.py                    # 串起以上模块，写 trajectories.jsonl + report
├── scripts/run_signal_screening.py       # CLI 入口
├── environment/setup_env.sh               # conda 环境搭建（在真实交互 shell 里跑，不是这里）
└── tests/                                  # 离线单元测试，不需要模型/GPU
```

## 探针题库到协议分类的映射（需要确认）

协议文档写的"character traits"和"language constraints"两类，在实际用到的
`Naomibas/llm-system-prompts-benchmark`（100 条）数据集里，并不是现成字段——
该数据集自己的分类是 `pattern` / `multiple_choice` / `persona` /
`memorization` / `language`（且 `language` 只有 3 条，指"讲法语"，不是协议
意义上的"语言/风格约束"）。按协议自己给出的筛选标准（"探针分数更接近连续
量"），`prompt_bank.py` 把 `persona_system_prompts`（14 条）当作
character_traits，把 `pattern_system_prompts`（28 条，字母/大小写/词数/句
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

模型权重和 HF 缓存目录默认在 `/scratch/hcao2/hf_cache`（`chat_model.py` 里
兜底设置了 `HF_HOME`，忘记 export 也不会落到 home 目录）。结果写到
`outputs/signal_screening/`：`trajectories.jsonl`（逐轮原始记录，字段与
协议第 5/8 节及同事 `core.py` 的列约定兼容）、`screening_report.json` 和
`screening_report.md`（三个问题的判定结果）。

## 测试

```bash
pytest -q
```

只测 `prompt_bank` / `reminder` / `analysis` 的纯逻辑，不需要 GPU 或联网
（探针题库已离线缓存在 `resources/`，见 `resources/PROVENANCE.md`）。
