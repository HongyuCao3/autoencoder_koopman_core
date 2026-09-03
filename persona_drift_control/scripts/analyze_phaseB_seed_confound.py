#!/usr/bin/env python3
"""Quantifies how severe the Phase B RandomExciteController seeding bug is
(found while executing docs/next_step_diagnosis.md's step 1, 2026-09-02).

`RandomExciteController.__post_init__` does `random.Random(self.seed)`, and
`screening_common.py::run_trajectories_loop` calls
`controller_factory(seed)` fresh for every `(attack, seed)` pair using only
the outer `--seeds 0 1` value -- NOT anything attack-specific. Since
`random.Random(N)` is a pure function of `N`, every attack sharing the same
`seed` gets a byte-identical 5-turn `u_remind` draw. Phase B's 300 rows (30
attacks x 2 seeds x 5 turns) turn out to contain exactly two distinct
`u_remind` sequences -- `seed0` is always `[0,0,1,1,0]`, `seed1` is always
`[1,0,0,1,1]`, confirmed directly from `trajectories.jsonl` (see this
session's transcript). This is NOT the "i.i.d. Bernoulli(p) each turn" the
class's docstring promises: there are only 2 independent excitation
realizations, not 300 (or even 60).

The `seed` value also drives `attack_seed`/`judge_seed` in
`attack_trajectory.py` (`seed * 1_000_000 + turn * 100 + 1/2`), i.e. every
turn's agent/judge sampling draw, for every attack. So `seed0` vs `seed1`
differ in BOTH `u_remind` (at turns 1/3/5 -- turns 2/4 are IDENTICAL, `u=0`
and `u=1` respectively, across both seeds) AND in whatever the generation
RNG happens to produce that has nothing to do with reminding. Any naive
`u_remind`-grouped contrast is confounded with this pure "which decoding
draw did you get" seed effect.

This script separates the two using exactly the turns where they don't
overlap: turns 2 and 4 have IDENTICAL u_remind across both seeds (u=0 and
u=1 respectively), so `mean(y|seed1) - mean(y|seed0)` at those two turns is
a clean, u_remind-free estimate of the pure seed-decoding confound. Turns
1/3/5 have u_remind flipped between the seeds (`du = u1-u0` = +1,-1,+1), so
`mean(y|seed1) - mean(y|seed0)` there is `seed_effect + u_eff * du`;
subtracting the turn-2/4 seed_effect estimate and dividing by `du` isolates
a deconfounded estimate of `u_eff`, the thing `B` is supposed to measure.

CPU-only, pure numpy -- no GPU needed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories  # noqa: E402

# Confirmed directly from outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl:
# every seed0 trajectory's u_remind sequence, every seed1 trajectory's.
_U_BY_SEED = {0: [0, 0, 1, 1, 0], 1: [1, 0, 0, 1, 1]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rows-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl")
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study/phaseB_seed_confound_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_trajectories(args.rows_path)
    traj = group_by_trajectory(rows)

    # sanity check: confirm every trajectory's u_remind sequence still matches
    # the exact two known patterns before trusting anything downstream.
    attacks_seed0, attacks_seed1 = {}, {}
    for tid, traj_rows in traj.items():
        u_seq = [int(r["u_remind"]) for r in traj_rows]
        y_seq = [float(r["y_safety"]) for r in traj_rows]
        attack_id = traj_rows[0]["attack_id"]
        seed = 0 if tid.endswith("__seed0") else 1
        expected = _U_BY_SEED[seed]
        if u_seq != expected:
            raise AssertionError(f"{tid}: u_remind={u_seq} != expected {expected} -- data has changed, re-derive _U_BY_SEED")
        (attacks_seed0 if seed == 0 else attacks_seed1)[attack_id] = y_seq

    common_attacks = sorted(set(attacks_seed0) & set(attacks_seed1))
    print(f"n_attacks_with_both_seeds={len(common_attacks)} (of {len(attacks_seed0)}/{len(attacks_seed1)})")

    # d[turn] = y(seed1) - y(seed0), paired per attack, turn = 1..5 (1-indexed)
    Y0 = np.array([attacks_seed0[a] for a in common_attacks])  # (n_attacks, 5)
    Y1 = np.array([attacks_seed1[a] for a in common_attacks])
    D = Y1 - Y0  # (n_attacks, 5): paired seed1-seed0 diff at each turn

    du = np.array([_U_BY_SEED[1][t] - _U_BY_SEED[0][t] for t in range(5)])  # [+1, 0, -1, 0, +1]
    print(f"du (u_seed1 - u_seed0) by turn = {du.tolist()}")

    seed_only_turns = np.where(du == 0)[0]  # turns 2, 4 (0-indexed 1, 3)
    u_turns = np.where(du != 0)[0]  # turns 1, 3, 5

    d_mean = D.mean(axis=0)  # (5,)
    print("\nmean paired diff d(turn) = mean(y_seed1 - y_seed0) per turn (1-indexed):")
    for t in range(5):
        print(f"  turn={t + 1}: du={du[t]:+d}  d={d_mean[t]:+.4f}")

    seed_effect = float(d_mean[seed_only_turns].mean())
    print(f"\nseed_effect estimate (mean of d at turns where u is identical across seeds, i.e. turns 2/4) = {seed_effect:+.4f}")

    # deconfounded u_eff at each u-varying turn: (d(turn) - seed_effect) / du(turn)
    u_eff_per_turn = (d_mean[u_turns] - seed_effect) / du[u_turns]
    u_eff = float(u_eff_per_turn.mean())
    print("deconfounded u_eff per turn (turns 1/3/5): " + ", ".join(f"{v:+.4f}" for v in u_eff_per_turn))
    print(f"deconfounded u_eff estimate (mean over turns 1/3/5) = {u_eff:+.4f}")

    naive_confounded = float(d_mean[u_turns].mean() * np.sign(du[u_turns]).mean())  # rough, for comparison only
    print(f"\n(for scale) naive |seed_effect| = {abs(seed_effect):.4f} vs |deconfounded u_eff| = {abs(u_eff):.4f}")
    print(f"ratio |seed_effect| / |u_eff| = {abs(seed_effect) / abs(u_eff) if u_eff else float('inf'):.2f}")

    # cluster bootstrap over attacks (n=30 independent clusters -- the real
    # sample size here, not 300 rows) for a rough CI on both quantities.
    rng = np.random.default_rng(args.bootstrap_seed)
    n_attacks = len(common_attacks)
    boot_seed_effect = np.empty(args.n_bootstrap)
    boot_u_eff = np.empty(args.n_bootstrap)
    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n_attacks, size=n_attacks)
        Db = D[idx]
        db_mean = Db.mean(axis=0)
        se_b = db_mean[seed_only_turns].mean()
        ue_b = ((db_mean[u_turns] - se_b) / du[u_turns]).mean()
        boot_seed_effect[b] = se_b
        boot_u_eff[b] = ue_b

    seed_effect_ci = (float(np.percentile(boot_seed_effect, 2.5)), float(np.percentile(boot_seed_effect, 97.5)))
    u_eff_ci = (float(np.percentile(boot_u_eff, 2.5)), float(np.percentile(boot_u_eff, 97.5)))
    print(f"\ncluster bootstrap (n={args.n_bootstrap}, resampling the {n_attacks} attacks):")
    print(f"  seed_effect 95% CI = [{seed_effect_ci[0]:+.4f}, {seed_effect_ci[1]:+.4f}]")
    print(f"  u_eff (deconfounded) 95% CI = [{u_eff_ci[0]:+.4f}, {u_eff_ci[1]:+.4f}]")
    print(f"  u_eff CI excludes 0: {u_eff_ci[0] > 0 or u_eff_ci[1] < 0}")

    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "n_attacks": n_attacks,
        "du_by_turn": du.tolist(),
        "d_mean_by_turn": d_mean.tolist(),
        "seed_effect": seed_effect,
        "seed_effect_ci95": seed_effect_ci,
        "u_eff_per_turn": u_eff_per_turn.tolist(),
        "u_eff": u_eff,
        "u_eff_ci95": u_eff_ci,
        "naive_confounded_reference": naive_confounded,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
