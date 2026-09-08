#!/usr/bin/env python3
"""Ground-truth status for the Koopman fast track
(docs/experiments/koopman_fast_track_plan.md).

Run this FIRST in any new session that picks this work up:

    export PATH=/scratch/hcao2/envs/persona_drift_pilot/bin:$PATH
    cd /home/hcao2/autoencoder_koopman_core/persona_drift_control
    python scripts/fast_track_status.py

It reads slurm and the output tree, never a hand-maintained note, so it cannot
go stale the way a status paragraph in a doc does. Jobs are looked up by
`--job-name`, not by job id, so a resubmission is picked up automatically.

CPU-only, seconds. Evaluates K1's exit gate (G-K1) directly and reports which
of K2's gates already have numbers. Prints exactly one "NEXT" line.
"""

from __future__ import annotations

import collections
import json
import pathlib
import subprocess

import numpy as np

PLAN = "docs/experiments/koopman_fast_track_plan.md"
K1_DIR = pathlib.Path("outputs/defense_excite_qwen4b")
K1_ROWS = K1_DIR / "trajectories.jsonl"
K1_FIT = K1_DIR / "koopman_fit_report.json"
REF_ROWS = pathlib.Path("outputs/d1_screen_qwen4b/trajectories.jsonl")
REJUDGE = pathlib.Path("outputs/d1_screen_qwen4b/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl")
JOB_NAMES = {"K1 excitation arm": "pdc-excite-qwen4b", "K1.2 offline rejudge": "pdc-rejudge-d1-qwen4b"}
K3_ARMS = ["koopman_mpc", "threshold", "fixed_schedule", "zero_control"]


def _rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def slurm_section() -> dict[str, str]:
    """Prints each job's recent history and returns {label: latest state}."""
    print("== slurm ==")
    latest: dict[str, str] = {}
    for label, job_name in JOB_NAMES.items():
        try:
            out = subprocess.run(
                ["sacct", "-u", "hcao2", "--name", job_name, "-X", "--noheader",
                 "--format=JobID,State,Elapsed,ExitCode", "-S", "2026-09-08"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:  # login node without slurm client
            print(f"  {label:<22} sacct unavailable ({exc})")
            continue
        if not out:
            print(f"  {label:<22} never submitted")
            latest[label] = "NONE"
            continue
        # Last line is the most recent submission of that name. A preempted job
        # can be requeued by the scheduler and show up RUNNING again under the
        # SAME id -- which is why the NEXT block below must never tell anyone to
        # resubmit without checking this first.
        for line in out.splitlines()[-3:]:
            print(f"  {label:<22} {' '.join(line.split())}")
        latest[label] = out.splitlines()[-1].split()[1]
    return latest


def gate_g_k1() -> bool:
    """docs/experiments/koopman_fast_track_plan.md section 1.1's exit gate."""
    print("\n== K1 gate G-K1 ==")
    if not K1_ROWS.exists():
        print(f"  no rows yet at {K1_ROWS}")
        return False
    rows = _rows(K1_ROWS)
    y = np.array([float(r["y_safety"]) for r in rows])
    turns = np.array([int(r["turn"]) for r in rows])
    u = np.array([float(r["u_remind"]) for r in rows])
    attacks = {r["attack_id"] for r in rows}
    ref_attacks = {r["attack_id"] for r in _rows(REF_ROWS)} if REF_ROWS.exists() else set()

    per_turn_sd = {int(t): float(y[turns == t].std(ddof=1)) for t in sorted(set(turns.tolist()))}
    checks = [
        ("500 rows", len(rows) == 500, f"{len(rows)}"),
        ("attack_id set == d1_screen_qwen4b", attacks == ref_attacks, f"{len(attacks)} vs {len(ref_attacks)}"),
        ("u_remind mean in [0.40, 0.60]", 0.40 <= u.mean() <= 0.60, f"{u.mean():.3f}"),
        ("reminded rows >= 200", int(u.sum()) >= 200, f"{int(u.sum())}"),
        ("every turn sd > 0", all(sd > 0 for sd in per_turn_sd.values()), str(per_turn_sd)),
    ]
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<36} {detail}")
    return all(ok for _, ok, _ in checks)


def k2_section() -> None:
    print("\n== K2 fit ==")
    if not K1_FIT.exists():
        print(f"  not run yet (no {K1_FIT})")
        return
    report = json.load(K1_FIT.open())
    cfg = report["config"]
    print(f"  config nu={cfg['nu']} mu={cfg['mu']} contemporaneous_v={cfg['contemporaneous_v']}")
    if not cfg["contemporaneous_v"]:
        print("  [FAIL] G-K2 invalid: fit without --contemporaneous-v scores the WRONG action slot "
              "(plan section 2.3, modeling/dataset.py:48-73). Refit before reading anything below.")
    ctrl = report["controllability_arx"]
    print(f"  G-K2-3  spectral_radius={ctrl['spectral_radius']:.4f} "
          f"rank={ctrl['controllability_rank']} gramian_cond={ctrl['gramian_condition']:.3e}")
    fold = report.get("fold_evaluation")
    if not fold:
        print("  G-K2-1  no fold_evaluation block -- rerun with --n-folds 20")
        return
    for name in ("const", "turn_mean", "stateless"):
        print(f"  null {name:<10} mean_y_one_step_mse={fold[f'null_{name}_mean_y_one_step_mse']:.6f}")
    for name in ("arx", "richer_abs_sign"):
        n_beat = fold[f"{name}_n_folds_beating_best_null"]
        verdict = "PASS" if n_beat >= 14 else "FAIL"
        print(f"  G-K2-1  [{verdict}] {name:<15} mean={fold[f'{name}_mean_y_one_step_mse']:.6f} "
              f"beats best null in {n_beat}/{fold['n_folds']}")


def side_sections() -> None:
    print("\n== K1.2 independent-judge range ==")
    if not REJUDGE.exists():
        print(f"  not run yet (no {REJUDGE})")
        print("    resubmit with: sbatch environment/run_defense_rejudge_d1_qwen4b.sbatch")
        print("    (gates nothing; only decides whether K4 reports one judge or two)")
    else:
        rows = _rows(REJUDGE)
        y = np.array([float(r["y_safety"]) for r in rows])
        turns = np.array([int(r["turn"]) for r in rows])
        print(f"  {len(rows)} rows, ceiling share={np.mean(y == 1.0):.2f}")
        for t in sorted(set(turns.tolist())):
            yy = y[turns == t]
            print(f"    turn{t} mean={yy.mean():.3f} sd={yy.std(ddof=1):.3f} distinct={len(set(yy.tolist()))}")
        print("  -> has range, K4 can report both judges" if all(
            y[turns == t].std(ddof=1) > 0 for t in set(turns.tolist())
        ) else "  -> saturated like the old pool; K4 reports self-judge alone (section 14.1 limitation)")

    print("\n== K3 closed-loop arms ==")
    found = [a for a in K3_ARMS if pathlib.Path(f"outputs/defense_fast_{a}/trajectories.jsonl").exists()]
    print(f"  {len(found)}/{len(K3_ARMS)} present: {found or 'none'}")


def main() -> None:
    print(f"Koopman fast track -- plan: {PLAN}\n")
    states = slurm_section()
    k1_ok = gate_g_k1()
    k2_section()
    side_sections()

    print("\n== NEXT ==")
    k1_state = states.get("K1 excitation arm", "NONE")
    rejudge_state = states.get("K1.2 offline rejudge", "NONE")
    if k1_state in ("RUNNING", "PENDING", "REQUEUED", "RESIZING", "SUSPENDED"):
        print(f"  K1 is {k1_state}. Wait -- do NOT resubmit, a second job would write into the")
        print("  same output dir. Check back with this script.")
    elif not K1_ROWS.exists():
        print(f"  K1 is {k1_state} with no output. Resubmit:")
        print("    sbatch environment/run_defense_excite_qwen4b.sbatch")
    elif not k1_ok:
        print(f"  K1 is {k1_state} and G-K1 failed above. Do NOT fit on it. Report the failing")
        print("  check to the user. The arm is resumable (adversarial_screening.py:106 keeps")
        print("  completed trajectory_ids, and the controller factory is per-trajectory seeded")
        print("  so a resumed random_excite draws the same sequence) -- resubmit the same sbatch.")
    elif not K1_FIT.exists():
        print("  Run K2 (CPU, minutes):")
        print("    python scripts/fit_koopman_defense_model.py \\")
        print("      --rows-path outputs/defense_excite_qwen4b/trajectories.jsonl \\")
        print("      --nu 1 --mu 2 --contemporaneous-v --n-folds 20 --split-seed 0 \\")
        print("      --out-path outputs/defense_excite_qwen4b/koopman_fit_report.json")
    else:
        print("  K2 has run. Report its four gates to the user, then -- only if the three main")
        print("  gates pass -- derive K3's per-arm trajectory count (plan section 2.4) BEFORE")
        print("  writing any K3 config. Gate failure means stop and report, not retune.")


if __name__ == "__main__":
    main()
