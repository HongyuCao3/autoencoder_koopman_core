#!/usr/bin/env python3
"""docs/next step.md 第一步之后浮现的新问题（2026-09-02）: 修正 v 对齐后，
nu=1,mu=1 ARX 拟合出的同轮直接效应 B 只有 ~0.02（且 95% CI 包含 0，见
scripts/analyze_phaseB_seed_confound.py），远小于 Phase A "持续提醒 5 轮，
终轮安全分从 0.45 拉到 0.81"（差距 ~0.36）这个量级。第一轮检验（mu=1）已经
证实：从每臂 turn1 均值播种、按固定策略滚动，nu=1,mu=1 模型在终轮只解释了
观测 gap 的 -1.6%（几乎为零、符号还错）——单步模型的自回归项衰减太快
（a~0.30），完全解释不了这个跨轮累积/复合效应。

这个脚本把 mu 从 1 扫到 4（5 轮轨迹里 mu=5 会导致 Phase B 拟合数据一条 pair
都凑不出，见下面 --mu-values 的上界）：本工程的目标是把 Koopman/EDMD 框架
用到 LLM 防御上，所以不脱离"lifted space 里线性动力学"这个框架 -- 加长 mu
只是扩大原始状态 z 本身（更长的动作历史），A/B 仍然是线性拟合，这仍然是
纯粹的 Koopman/ARX 路线，不是换成非线性模型。如果某个 mu 能让模型的滚动
预测追上 Phase A 观测到的 gap，说明之前的记忆窗口开小了，问题在状态维度
不在方法论；如果 mu 一路加到 4 都追不上，说明需要一个专门设计的、能表达
"跨轮累积侵蚀阻力"的 lifting（例如把"最近 k 轮提醒次数"这样的工程特征作为
Koopman 观测量的一部分，而不是继续加 mu 的原始滞后阶数）。

复用同一对配对数据: Phase A（`--controller constant_remind`, 20 个攻击 x 2
seed, 100% 每轮提醒）和它的 zero_control 基线（job 15399715,
`outputs/adversarial_screening/`, 同一批 20 个攻击 x 2 seed, 从不提醒）。

CPU-only, 纯 numpy -- 不需要 GPU。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    load_trajectories,
)
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phase-b-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl")
    )
    parser.add_argument(
        "--zero-control-path", type=pathlib.Path, default=pathlib.Path("outputs/adversarial_screening/trajectories.jsonl")
    )
    parser.add_argument(
        "--constant-remind-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseA_constant_remind/trajectories.jsonl"),
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--mu-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="max is 4 for a 5-turn trajectory: build_reduced_state_pairs needs start=mu-1 < len(traj_rows)-1=4, i.e. mu<5.",
    )
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study/compounding_hypothesis_report.json"))
    return parser.parse_args()


def _mean_y_by_turn(rows: list[dict]) -> dict[int, float]:
    by_turn: dict[int, list[float]] = {}
    for row in rows:
        by_turn.setdefault(row["turn"], []).append(row["y_safety"])
    return {t: float(np.mean(ys)) for t, ys in by_turn.items()}


def _rollout(model: KoopmanSurrogate, y1: float, u_policy: int, n_turns: int, mu: int) -> list[float]:
    """z_1 = [y_1, u_hist] under contemporaneous_v=True, nu=1: u_hist is the
    mu most recent actions up to and including turn 1's own (=u_policy) --
    turns before 1 don't exist, treated as u=0 (no reminder could have been
    inserted before the trajectory started). model.step's fitted A/B then
    carries the shift-register update for the whole mu-length lag window
    automatically (the same mechanism it was fit to reproduce), so only the
    seed needs this explicit zero-padding -- every later step is just
    `model.step(z, [u_policy])`."""

    u_hist_at_turn1 = [0.0] * (mu - 1) + [float(u_policy)]
    z = np.array([y1] + u_hist_at_turn1)
    preds = [float(model.readout(z))]
    for _ in range(n_turns - 1):
        z = model.step(z, np.array([float(u_policy)]))
        preds.append(float(model.readout(z)))
    return preds


def main() -> None:
    args = parse_args()

    phase_b_rows = load_trajectories(args.phase_b_path)
    zero_rows = load_trajectories(args.zero_control_path)
    remind_rows = load_trajectories(args.constant_remind_path)
    zero_attacks = {r["attack_id"] for r in zero_rows}
    remind_attacks = {r["attack_id"] for r in remind_rows}
    assert zero_attacks == remind_attacks, "zero_control and constant_remind must be the same attack panel"

    zero_by_turn = _mean_y_by_turn(zero_rows)
    remind_by_turn = _mean_y_by_turn(remind_rows)
    turns = sorted(zero_by_turn)
    n_turns = len(turns)
    observed_gap = {t: remind_by_turn[t] - zero_by_turn[t] for t in turns}

    print("observed: mean y_safety per turn (zero_control / constant_remind / gap)")
    for t in turns:
        print(f"  turn={t}: zero={zero_by_turn[t]:.4f} remind={remind_by_turn[t]:.4f} gap={observed_gap[t]:+.4f}")

    last_turn = turns[-1]
    sweep_report = {}
    for mu in args.mu_values:
        config = ReducedStateConfig(nu=1, mu=mu, contemporaneous_v=True)
        dataset = build_identification_dataset(phase_b_rows, config, y_col="y_safety")
        n_pairs = dataset["Z"].shape[0]
        print(f"\n=== mu={mu} (n_fit_pairs={n_pairs}, state_dim={config.state_dim}) ===")
        if n_pairs == 0:
            print("  no fittable pairs at this mu -- skipping (trajectory too short)")
            continue
        model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=args.ridge).fit(dataset)
        print(f"  A={model.A.tolist()}")
        print(f"  B={model.B.tolist()}")

        pred_zero = _rollout(model, zero_by_turn[turns[0]], u_policy=0, n_turns=n_turns, mu=mu)
        pred_remind = _rollout(model, remind_by_turn[turns[0]], u_policy=1, n_turns=n_turns, mu=mu)
        predicted_gap = {turns[i]: pred_remind[i] - pred_zero[i] for i in range(n_turns)}

        for i, t in enumerate(turns):
            print(f"  turn={t}: zero_pred={pred_zero[i]:.4f} remind_pred={pred_remind[i]:.4f} gap_pred={predicted_gap[t]:+.4f}  (observed={observed_gap[t]:+.4f})")

        explained_fraction = predicted_gap[last_turn] / observed_gap[last_turn] if observed_gap[last_turn] else float("nan")
        print(f"  => terminal turn ({last_turn}): model explains {explained_fraction:.1%} of the observed gap "
              f"({predicted_gap[last_turn]:+.4f} predicted vs {observed_gap[last_turn]:+.4f} observed)")

        sweep_report[mu] = {
            "n_fit_pairs": n_pairs,
            "A": model.A.tolist(),
            "B": model.B.tolist(),
            "b": model.b.tolist(),
            "predicted_gap_by_turn": predicted_gap,
            "explained_fraction_at_terminal_turn": explained_fraction,
        }

    print("\n=== summary across mu ===")
    print(f"{'mu':>4} {'n_pairs':>8} {'explained_fraction_at_turn' + str(last_turn):>28}")
    for mu, r in sweep_report.items():
        print(f"{mu:>4} {r['n_fit_pairs']:>8} {r['explained_fraction_at_terminal_turn']:>28.1%}")

    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "observed": {"zero_control_by_turn": zero_by_turn, "constant_remind_by_turn": remind_by_turn, "gap_by_turn": observed_gap},
        "sweep": sweep_report,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
