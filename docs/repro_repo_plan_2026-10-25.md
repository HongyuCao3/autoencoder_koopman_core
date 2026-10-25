# 复现仓库计划（#5，给 Sriram）

> **状态**：▶ 待执行（2026-10-25，主计划 [`experiments/post_meeting_2026-10-25_master_plan.md`](experiments/post_meeting_2026-10-25_master_plan.md)）。
> 原则：**抽一个干净仓库，不开放本仓库**。本仓库有 99 个 run 目录与多处 `.claude/global.md` 判为不可报告的数。

## 一、仓库

- 名称 `kompas-repro`（私有）；owner 为用户账号；邀请 Sriram（账号由 Kunpeng 提供）。
- 许可证沿用本仓库 `LICENSE`。
- **不含**：`persona_drift_control/outputs/`、`results/`、`paper/`、`docs/`（本仓库文档含内部裁决）、`.claude/`。

## 二、内容（四段管线 + 数据）

| 目录 | 来自 | 说明 |
|---|---|---|
| `kompas/` | `src/koopman_ae/core.py` 中 `AugmentedStateConfig` / `build_augmented_state_dataset` / `AugmentedKoopmanModel` / `controllability_diagnostics` | 只保留线性路径；AE 路径（`torch`）作为可选 extra |
| `scripts/collect.py` | `scripts/run_core_collection_arm.py` | 采集轨迹；默认 `command_policy=error`（与论文一致）；#4b 的 `random` 若已实现则一并带上 |
| `scripts/fit.py` | `scripts/paper_side_experiments.py::_fit/_state_config` + `eval_surrogate_rows.py::_load` | 一条命令从 `trajectories.jsonl` 拟到 $K,B,b$ 并输出 Skill$_H$ |
| `scripts/evaluate.py` | `eval_surrogate_rows.py` | 复现 Table 1 的七列（`core` 四列本地数据即可；行为线三列需对应轨迹，见 §四） |
| `scripts/control.py` | 控制线管线（**不在本仓库**，待用户给路径） | 复现 Table 2；拿不到则 README 写明「控制线代码另行提供」 |
| `data/` | `datasets/scalar/*`、`datasets/vector/*` 的 `trajectories.jsonl` | 体积检查：> 100 MB 的用 Git LFS 或 release 附件 |
| `tests/` | `tests/` 中只依赖上述模块的用例 | `pytest` 一条命令全绿 |

## 三、README 必写

1. 环境：Python 版本、`requirements.txt` 锁版本、GPU 可选（拟合与评估纯 CPU）。
2. 三条命令：`collect` → `fit` → `evaluate`，各自输入输出路径、预计时长。
3. **报告口径**：Table 1 数字为 Skill$_H$ + 轨迹级 paired bootstrap；**`defense` 列为自判分数，单向漏检，不可作为报告数字**（`.claude/global.md` 具名例外的义务句）。
4. Table 2 数字为点估计、无区间。
5. 与论文记号的对照表（$K,B,b$ / 状态深度 / 时域 $H$ 的取值表，抄论文 Table 4）。

## 四、执行位置

| 步骤 | 位置 |
|---|---|
| 抽代码、写 README、单测 | 本地 |
| 行为线轨迹（`constraint` / `defense` / `tsar_cefr`）与控制线日志 | 服务器 `git pull` 后拷出；体积过大则不进仓库，README 给获取方式 |
| 推送与邀请 | 本地，用户账号 |

## 五、完成 = 什么

从零 clone 后 `pytest` 绿 + `fit`/`evaluate` 在 `core` 四列复现 Table 1 对应数（容差 1e-3）+ README 五项齐 + 邀请发出。
