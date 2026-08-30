# 实验记录：采集前信号探针 pilot（真实规模）

记录这次跑在 Palmetto 2 上的作业本身的状态，供换一次新对话（没有本会话上下文）时能接得上——
git 里只有代码和协议，不会有"现在有个作业在跑"这种进行时状态，所以单独记一份。作业跑完/被杀之后
把结论（`overall_pass` 等）也写回这里，不要只留在 `outputs/` 里（那里 gitignore 了，换台机器
或清理 scratch 后就没了）。

## 这是什么实验

`DATA_COLLECTION_PROTOCOL.md` 第 7 节要求的、在正式采集 320 条轨迹之前必须先跑的 gate：
5 个 system prompt × 2 个 seed × 16 轮，`zero_control` 和 `excite_iid` 两种条件各跑一遍
（每个 (prompt, seed) 都跑两种条件，不是只跑一种），共 20 条轨迹，用来回答三个问题：
漂移是否存在／输入是否有效／是否有惯性。任一不过就要先改协议，不能直接开始正式采集。

## 作业信息

| 字段 | 值 |
|---|---|
| Job ID | 15382858 |
| 提交脚本 | `persona_drift_control/environment/run_screening.sbatch` |
| 实际命令 | `python scripts/run_signal_screening.py --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B --device cuda --output-dir outputs/signal_screening`（未传 `--num-prompts/--seeds/--num-turns/--probe-repeats`，用的是协议默认规模：5 prompts, seeds 0 1, 16 turns, 4 probe repeats） |
| 资源 | `work1` 分区，`--gpus a100:1`，`--cpus-per-task 8 --mem 32G` |
| 申请时长 | `--time 03:00:00` |
| 提交时间 | 2026-08-29T20:48:14（提交后立刻开始跑，无排队） |
| 输出目录 | `persona_drift_control/outputs/signal_screening/`（trajectories.jsonl + screening_report.json/md，整个 `outputs/` 被 `persona_drift_control/.gitignore` 排除，不入库） |
| 日志 | `persona_drift_control/environment/slurm_logs/screening-15382858.out` |

## 查看状态的方法

```bash
squeue -j 15382858                                  # 是否还在跑、剩余时间
wc -l persona_drift_control/outputs/signal_screening/trajectories.jsonl   # 已完成的行数
cat persona_drift_control/outputs/signal_screening/screening_report.md   # 跑完之后才会有
```

`screening.py` 是每条轨迹整体跑完才一次性写盘的（不是每轮增量写），所以行数在最后一条轨迹完成
前都可能是 0，属于正常现象，不代表卡住了。

## 进度与耗时预估（写这份文档时的快照，之后应该更新或删掉这一节）

- 2026-08-29 21:21（作业运行 33 分钟时）：模型已加载完成，但 `trajectories.jsonl` 仍是
  0 行——20 条轨迹里连第 1 条都还没跑完。
- `RUNNING_ON_PALMETTO.md` 里 `--time 03:00:00` 的预估依据是"协议自己估计 10 条轨迹约 1 小时，
  20 条轨迹翻倍到约 2 小时，`03:00:00` 留了余量"，即预期约 6 分钟/条。**实际观察到第 1 条轨迹
  超过 33 分钟仍未完成，明显慢于预期**，原因待查（可能是每轮上下文变长导致 prefill 变慢、
  也可能是探针 4 次重复 + 主线生成完全串行没有 batch）。如果这个速度持续，20 条轨迹可能需要
  远超 3 小时，作业会被 Slurm 在到时后直接杀掉、拿不到 `screening_report.md`。
- **下次打开这份文档时先做的事**：`squeue -j 15382858` 看作业还在不在。
  - 如果不在了且 `outputs/signal_screening/screening_report.md` 存在：作业正常跑完了，去看
    `overall_pass`，把结论更新到本文件的"结论"一节，再决定要不要开始正式的 320 条采集。
  - 如果不在了但没有 `screening_report.md`：大概率是被 `--time 03:00:00` 到点杀掉了。看
    `trajectories.jsonl` 实际写了几条、日志时间戳，估出真实的单条轨迹耗时，改大
    `run_screening.sbatch` 的 `--time` 后重新 `sbatch` 提交。
  - 如果还在跑：继续等，或者用上面"查看状态的方法"里的命令看进度。

## 更新：加了进度日志，但这个作业本身不受影响

2026-08-29 稍晚给 `screening.py` 加了逐条轨迹的开始/结束时间日志（含 `handle.flush()`，让
`trajectories.jsonl` 增量可读）。**这个改动对 job 15382858 没有任何效果**——Python 在作业开始
时就已经把旧版模块读进内存了，编辑磁盘上的源码不会让正在跑的进程重新加载。也就是说这个作业
从头到尾都不会在日志里看到进度信息，是预期行为，不是改坏了。以后新提交的 screening 作业才会
有这些日志。

2026-08-29 21:36（作业运行 48 分钟时）：仍然是 0 行、模型加载完成后再没有任何输出——因为没有
进度日志，无法判断第 1 条轨迹具体卡在哪一轮。比最初预估的"6 分钟/条"慢了至少 8 倍。

## 结论（作业完成后补充）

（尚未完成，等 `screening_report.md` 产出后把 `overall_pass` 和三问的具体数值写在这里。如果
这个作业最终被 `--time 03:00:00` 杀掉且一条轨迹都没跑完，说明这个规模/配置在当前推理速度下
不可行，需要重新评估是加大 `--time`、减小 `--max-new-tokens`，还是排查是不是某个环节真的卡死
了，而不是简单地重新提交同一个作业。）
