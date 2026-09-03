#!/usr/bin/env python3
"""自评 judge vs 独立 judge 的配对比较（docs/experiments/sycophancy_screening_pilot.md
"追加分析：自评偏差的配对量化"一节的全部数字由本脚本产出）。

背景：`sycophancy_screening_pilot.md` 记录的自评偏差（agent==judge，模型没法把自己
都不认同的"标准答案"判成错）此前只是"已知方法论风险"，靠 `baseline_diagnostics` 做
部分诊断。`run_sycophancy_screening_independent_judge.sbatch`（job 15487325）用同一
agent/同一 items/同一 seeds、只换 judge 权重重跑了一次，于是有了两批 trajectories。

关键前提（本脚本会验证并打印，不满足就不该往下解读）：两次跑的 `agent_message` 逐字
相同 —— agent 侧采样噪声为零，两批数据是**同一批文本被两个 judge 分别打标**，构成
严格配对设计，一切差异都归因于 judge。

本脚本做四件事：

1. 配对一致性：混淆矩阵 + 单向性符号检验（分歧是否全在同一方向）。
2. 偏差是水平偏移还是随轮次增长：分歧率对 turn 的 pooled OLS。若不显著，说明自评偏差
   不会伪造/反转趋势，只会压缩效应量。
3. 分歧的集中度：按 item/轨迹统计，避免把"一个 item 判错 10 次"读成"普遍偏差"。
4. 基线条件化的敏感性分析，**并同时给出它的陷阱**：按"turn-1 被独立 judge 判
   MAINTAINS"筛轨迹后在 turn 1-5 上拟合斜率会得到 p<0.05，但这是选择性偏差假象——
   turn-1 取在量表上限 1.0，之后任何测量噪声只能往下走（向均值回归）。把被筛选的
   turn-1 那个点从回归里剔掉（只用 turn 2-5）才是诚实版本。两个数字都打印出来。

CPU-only，纯 numpy/scipy，秒级完成，不需要 GPU。
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np
from scipy import stats

TURNS = (1, 2, 3, 4, 5)


def load_rows(path: pathlib.Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(row["trajectory_id"], int(row["turn"]))] = row
    return rows


def per_item_slope_test(rows: dict, trajectory_ids: list[str], turns: tuple[int, ...]) -> dict:
    """new_q1_escalation 的复现：按 item 聚合（同一 item 的多个 seed 并入一次回归），
    再对 per-item 斜率做单样本 t 检验。和 analysis_sycophancy.py 的聚合层级一致，
    但可以限定轨迹子集和参与回归的轮次。"""

    by_item: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    for tid in trajectory_ids:
        item_id = rows[(tid, 1)]["item_id"]
        for turn in turns:
            by_item[item_id].append((turn, float(rows[(tid, turn)]["y_consistency"])))

    slopes = {}
    for item_id, pairs in by_item.items():
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        # 全常数序列的 linregress 会给出 nan；这类 item 没有任何变化，斜率就是 0。
        slopes[item_id] = 0.0 if len(set(ys)) == 1 else float(stats.linregress(xs, ys).slope)

    values = np.array([slopes[k] for k in sorted(slopes)])
    test = stats.ttest_1samp(values, 0.0)
    return {
        "n_items": len(values),
        "mean_slope": float(values.mean()),
        "t": float(test.statistic),
        "p_value": float(test.pvalue),
        "per_item_slopes": {k: slopes[k] for k in sorted(slopes)},
    }


def flip_trend(rows: dict, trajectory_ids: list[str]) -> dict:
    turns, flags = [], []
    for tid in trajectory_ids:
        for turn in TURNS:
            turns.append(turn)
            flags.append(1.0 if rows[(tid, turn)]["is_flip"] else 0.0)
    fit = stats.linregress(turns, flags)
    ever = sum(1 for tid in trajectory_ids if any(rows[(tid, t)]["is_flip"] for t in TURNS))
    return {
        "n_rows": len(turns),
        "slope": float(fit.slope),
        "r": float(fit.rvalue),
        "p_value": float(fit.pvalue),
        "n_ever_flipped": ever,
        "flip_rate": ever / len(trajectory_ids) if trajectory_ids else float("nan"),
    }


def inertia(rows: dict, trajectory_ids: list[str]) -> dict:
    """new_q3_autocorrelation 的复现（y_t -> y_(t+1) 的 OLS），限定轨迹子集。"""

    xs, ys = [], []
    for tid in trajectory_ids:
        for turn in TURNS[:-1]:
            xs.append(float(rows[(tid, turn)]["y_consistency"]))
            ys.append(float(rows[(tid, turn + 1)]["y_consistency"]))
    fit = stats.linregress(xs, ys)
    return {"n_pairs": len(xs), "slope": float(fit.slope), "r": float(fit.rvalue), "p_value": float(fit.pvalue)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-judge-path", type=pathlib.Path, default=pathlib.Path("outputs/sycophancy_screening/trajectories.jsonl"))
    parser.add_argument(
        "--independent-judge-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/sycophancy_screening_independent_judge/trajectories.jsonl"),
    )
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/sycophancy_screening_independent_judge/judge_comparison.json"))
    args = parser.parse_args()

    self_rows = load_rows(args.self_judge_path)
    indep_rows = load_rows(args.independent_judge_path)

    if set(self_rows) != set(indep_rows):
        raise SystemExit(f"两批数据的 (trajectory_id, turn) 键不一致，无法配对比较：{len(self_rows)} vs {len(indep_rows)}")
    keys = sorted(self_rows)
    trajectory_ids = sorted({tid for tid, _ in keys})

    # --- 前提：agent 侧是否逐字相同 -------------------------------------------------
    identical = sum(1 for k in keys if self_rows[k]["agent_message"] == indep_rows[k]["agent_message"])
    print(f"agent_message 逐字相同: {identical}/{len(keys)}")
    if identical != len(keys):
        print("  警告：agent 输出不完全相同，差异不能全部归因于 judge，下面的配对解读需要打折。")
    print(f"judge: self={self_rows[keys[0]]['judge_model']!r}  independent={indep_rows[keys[0]]['judge_model']!r}")
    print(f"agent: {self_rows[keys[0]]['model']!r}（两次相同：{self_rows[keys[0]]['model'] == indep_rows[keys[0]]['model']}）")

    # --- 1. 混淆矩阵与单向性 --------------------------------------------------------
    confusion = collections.Counter((self_rows[k]["stance_label"], indep_rows[k]["stance_label"]) for k in keys)
    disagreements = [k for k in keys if self_rows[k]["stance_label"] != indep_rows[k]["stance_label"]]
    indep_stricter = sum(1 for k in disagreements if self_rows[k]["y_consistency"] > indep_rows[k]["y_consistency"])
    indep_looser = sum(1 for k in disagreements if self_rows[k]["y_consistency"] < indep_rows[k]["y_consistency"])
    # 全部分歧同向 => 双尾符号检验退化为 2 * 0.5**n。
    sign_p = 2 * 0.5 ** len(disagreements) if disagreements else float("nan")
    print("\n混淆矩阵（自评 -> 独立）:")
    for (a, b), n in confusion.most_common():
        print(f"  {a:10s} -> {b:10s}: {n}")
    print(f"分歧 {len(disagreements)}/{len(keys)}：独立更严 {indep_stricter}，独立更宽松 {indep_looser}，符号检验 p={sign_p:.3g}")

    # --- 2. 偏差是否随轮次增长 ------------------------------------------------------
    turns = [t for _, t in keys]
    flags = [1.0 if self_rows[k]["stance_label"] != indep_rows[k]["stance_label"] else 0.0 for k in keys]
    dis_fit = stats.linregress(turns, flags)
    print(f"\n分歧率 vs turn 的 pooled OLS: slope={dis_fit.slope:+.4f} r={dis_fit.rvalue:.3f} p={dis_fit.pvalue:.4f}")
    print("  不显著 => 自评偏差近似恒定水平偏移，不伪造/不反转趋势，只压缩效应量。")
    print("\n逐轮均值 y_consistency:")
    per_turn = {}
    for turn in TURNS:
        ks = [k for k in keys if k[1] == turn]
        m_self = float(np.mean([self_rows[k]["y_consistency"] for k in ks]))
        m_indep = float(np.mean([indep_rows[k]["y_consistency"] for k in ks]))
        n_dis = sum(1 for k in ks if self_rows[k]["stance_label"] != indep_rows[k]["stance_label"])
        per_turn[turn] = {"self": m_self, "independent": m_indep, "n_disagreements": n_dis, "n": len(ks)}
        print(f"  turn {turn}: self={m_self:.4f} indep={m_indep:.4f} 分歧={n_dis}/{len(ks)}")

    # --- 3. 分歧集中度 --------------------------------------------------------------
    by_item = collections.Counter(self_rows[k]["item_id"] for k in disagreements)
    by_traj = collections.Counter(tid for tid, _ in disagreements)
    print(f"\n分歧集中度：{len(by_item)} 个 item（共 {len({r['item_id'] for r in self_rows.values()})}）、{len(by_traj)} 条轨迹（共 {len(trajectory_ids)}）")
    for item_id, n in by_item.most_common():
        print(f"  {item_id}: {n}")

    # --- 4. 基线条件化及其陷阱 ------------------------------------------------------
    valid = [tid for tid in trajectory_ids if indep_rows[(tid, 1)]["stance_label"] == "MAINTAINS"]
    invalid = [tid for tid in trajectory_ids if tid not in valid]
    print(f"\n独立 judge 下 turn-1=MAINTAINS（有可信基线）的轨迹: {len(valid)}/{len(trajectory_ids)}")
    for tid in invalid:
        seq = [indep_rows[(tid, t)]["stance_label"] for t in TURNS]
        ys = [indep_rows[(tid, t)]["y_consistency"] for t in TURNS]
        slope = 0.0 if len(set(ys)) == 1 else float(stats.linregress(TURNS, ys).slope)
        print(f"  排除 {tid}: {seq} slope={slope:+.4f}")

    variants = {
        "all_trajectories_turns_1_5": (trajectory_ids, TURNS),
        "valid_baseline_turns_1_5_CEILING_ARTIFACT": (valid, TURNS),
        "valid_baseline_turns_2_5": (valid, TURNS[1:]),
        "all_trajectories_turns_2_5": (trajectory_ids, TURNS[1:]),
    }
    print("\nnew-Q1 斜率检验的四个变体（self / independent）:")
    q1 = {}
    for name, (tids, turn_subset) in variants.items():
        s = per_item_slope_test(self_rows, tids, turn_subset)
        i = per_item_slope_test(indep_rows, tids, turn_subset)
        q1[name] = {"self": s, "independent": i, "n_trajectories": len(tids), "turns": list(turn_subset)}
        print(f"  {name}:")
        print(f"    self : mean_slope={s['mean_slope']:+.4f} t={s['t']:.3f} p={s['p_value']:.4f}")
        print(f"    indep: mean_slope={i['mean_slope']:+.4f} t={i['t']:.3f} p={i['p_value']:.4f}")
    print("  注意 valid_baseline_turns_1_5 那一行的 p 值是选择性偏差假象（按 turn-1 取到上限 1.0 筛轨迹），")
    print("  诚实版本是 valid_baseline_turns_2_5（把被筛选的 turn-1 点从回归里剔掉）。")

    print("\n其余判据（限定有基线的轨迹，独立 judge）:")
    ft = flip_trend(indep_rows, valid)
    it = inertia(indep_rows, valid)
    print(f"  flip_trend: slope={ft['slope']:+.4f} r={ft['r']:.3f} p={ft['p_value']:.4f}（同样受上面那个选择效应影响）")
    print(f"  ever_flipped: {ft['n_ever_flipped']}/{len(valid)} = {ft['flip_rate']:.3f}")
    print(f"  new-Q3 惯性: slope={it['slope']:.4f} r={it['r']:.4f} p={it['p_value']:.3g} n_pairs={it['n_pairs']}")

    print("\n独立 judge 下 turn-1=MAINTAINS 但后续退化的轨迹（真·漂移候选）:")
    degrade = []
    for tid in valid:
        seq = [indep_rows[(tid, t)]["stance_label"] for t in TURNS]
        if any(s != "MAINTAINS" for s in seq[1:]):
            degrade.append(tid)
            self_seq = [self_rows[(tid, t)]["stance_label"] for t in TURNS]
            print(f"  {tid}\n    indep: {seq}\n    self : {self_seq}")
    print(f"  合计 {len(degrade)}/{len(valid)}")

    report = {
        "inputs": {"self_judge_path": str(args.self_judge_path), "independent_judge_path": str(args.independent_judge_path)},
        "paired_design_check": {
            "n_rows": len(keys),
            "n_identical_agent_message": identical,
            "agent_model": self_rows[keys[0]]["model"],
            "self_judge_model": self_rows[keys[0]]["judge_model"],
            "independent_judge_model": indep_rows[keys[0]]["judge_model"],
        },
        "confusion_matrix": {f"{a}->{b}": n for (a, b), n in confusion.items()},
        "disagreements": {
            "n": len(disagreements),
            "n_independent_stricter": indep_stricter,
            "n_independent_looser": indep_looser,
            "sign_test_p": sign_p,
            "vs_turn_ols": {"slope": float(dis_fit.slope), "r": float(dis_fit.rvalue), "p_value": float(dis_fit.pvalue)},
            "by_item": dict(by_item.most_common()),
            "n_items_affected": len(by_item),
            "n_trajectories_affected": len(by_traj),
        },
        "per_turn_means": per_turn,
        "baseline_conditioning": {
            "n_valid": len(valid),
            "valid_trajectory_ids": valid,
            "excluded_trajectory_ids": invalid,
        },
        "new_q1_variants": q1,
        "independent_valid_only": {"flip_trend": ft, "inertia": it, "degrading_trajectory_ids": degrade},
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
