# 命令通道 $B$ 可辨识性：结果（#4a）

> **状态**：✅ 4a 完成（2026-10-25）。规格 [`command_channel_identifiability_plan.md`](command_channel_identifiability_plan.md) §二。
> 口径见 `../../.claude/global.md`。$B$ 区间是**拟合诊断量**，不是跨臂头条数字；它只回答「能不能把 $K,B$ 交出去算 Gramian」。

## 产物与谱系

| | |
|---|---|
| 脚本 | `persona_drift_control/scripts/b_identifiability_bootstrap.py`（新增；`_identifiability` 复用 M1 的那份，状态配置与 Table 1 / M1 相同） |
| 单测 | `persona_drift_control/tests/test_b_identifiability_bootstrap.py` 4 条，全过 |
| 产物 | `results/b_identifiability/{constraint,defense,tsar_cefr}.json`（存在即报错，不覆盖） |
| 设置 | 全数据拟合给点估计；轨迹级 bootstrap 2000 次（`default_rng(0)`）给区间；item 级区间作同址稳健性列 |
| 环境 | ⚠️ `persona_drift_pilot` env 的 Python 标准库被 `/scratch` 清理删掉，本次用 `/scratch/hcao2/envs/dkvar`（numpy 1.26.4 / pandas 2.3.3）+ `PYTHONPATH=src`。换 env 之前先复现了已落盘的 A1 matched 两列，22 格 `Skill_H` 最大差 **1.9e-13** |
| 运行时 | 三条线合计约 2 分钟，纯 CPU（规格估 0.5 天是人时，不是机时） |

**主元素**的定义：$C B$，即每个命令列对读数 $y$ 的一步效应。`constraint` / `defense` 的 $C$ 是选择向量，
$CB$ 就是 $B$ 落在最新 $y$ 上的那一行；`tsar_cefr` 的 $CB$ 是三个非参照动作（参照 = `copy`）对 CEFR 等级的效应。
$B$ 其余元素（进旧 tap、进 $s$）也写进 JSON，但不进判词。

## 结果表

判词按规格 §2.3：秩检查过 **且** 每个主元素 95% 区间不含 0 **且** 符号稳定率 ≥ 0.95。签的是轨迹级单位。

| 任务 | 秩 | $v$ 对 $[Z\|1]$ 的 $R^2$ | 主元素 $CB$（点估计 [轨迹级 95%]，符号稳定率） | 可用于 Gramian（轨迹级） | item 级 |
|---|---|---|---|---|---|
| `core` 八任务 | — | — | 不可辨识：$R=r-y_t$ 是 $[Z\|1]$ 的精确仿射函数，引用 [`core_multiobjective_planning_headroom.md`](core_multiobjective_planning_headroom.md) §二，不重算 | **否** | — |
| `tsar_cefr` | 8/8 | 0.0008 / 0.0009 / 0.0006 | `step_down` −0.0665 [−0.0841, −0.0502] 1.000；`half_step_down` −0.0696 [−0.0863, −0.0522] 1.000；`paraphrase` −0.0261 [−0.0465, −0.0045] 0.989 | **是** | 是 |
| `constraint` | 7/7 | 0.0014 | `u_remind` +0.0146 [+0.0049, +0.0247] 0.999 | **是** | 是 |
| `defense` ⚠self | 5/5 | 0.0004 | `u_remind` +0.0617 [−0.0183, +0.1469] 0.930 | **否** | 是（[+0.0031, +0.1237] 0.982） |
| 控制线（IFBench / IFEval / COLLIE） | — | — | **不可得**：用户指向的 `KOMPAS-code-release.zip` 是代码发布包，README 明写 Table 2 的 benchmark harness 不在其中，包内无任何日志 | 待查 | — |

`defense` 列：自判分数（`judge_kind=self`，`global.md` 具名例外），自判 = 单向漏检，systematically 低估效应。

## 读法

1. **交给 Umesh 的回答**：`tsar_cefr` 与 `constraint` 的 $K,B$ 可以拿去算 Gramian / 可达集；`core` 八列不行（问题无定义，不是效应小）；
   `defense` 在签好的轨迹级单位下不过（区间含 0，符号稳定率 0.930 < 0.95）。
2. **`defense` 的两种单位判词不一致**：item 级（30 个攻击）判「是」，轨迹级（60 条）判「否」。按规格签的轨迹级记「否」，
   不换单位改判。即使按 item 级，它也带自判局限，不是首选的 Gramian 输入。
3. 与规格 §2.3 的预期（`tsar_cefr`、`constraint` 可用；`defense` 带自判标注可用）相比，**`defense` 比预期差一档**。
4. `tsar_cefr` 的三个动作都把 CEFR 等级往下推（简化方向），`paraphrase` 最弱且区间贴近 0（上界 −0.0045）。
   规格写的「动作离散三值」不准确：该线是四值（三个 one-hot + 参照 `copy`）。

## 对 4b 的后果

4b 的启动条件是「`core` 不可用 **且** 控制线不可得」。两条都成立（控制线日志不在 zip 里），所以按规格 §三进入 4b：
开关代码 + 单测 + sbatch 干跑，然后停在 ⏸ 等裁决。
