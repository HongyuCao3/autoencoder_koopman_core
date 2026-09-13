#!/usr/bin/env python3
"""Gate R1: does the activation projection raise the adaptivity ceiling?

The three rows were signed before the job ran
(docs/experiments/adaptivity_ceiling_results.md, appendix 2):

  pass -> re-run Table 2's `defense` closed loop with `proj` as the readout
          (main-line action, ~3.5 GPU-h), NOT another gate.
  fail -> close the line. Table 2's negative stands as written.
  table -> appendix A8 gains a row; section 2's "the one open direction"
           sentence is rewritten either way.

CRITERION, fixed in advance: the causal ceiling conditioned on `proj_pre_reply`
binned into THREE bins, cross-fitted. Pass iff the 95% CI lower bound clears
zero on EITHER axis -- leave-one-attack-out (generalise to a new attack) or
leave-one-seed-out (generalise to a new conversation against a known attack).

Two and four bins are reported and are NOT the criterion. Three was named
before the 40-trajectory data existed precisely because three was the best of
the three on the 16-trajectory pilot: letting the pilot's winner pick the
criterion would make a selection signal into the reported one.

`proj_pre_reply` is the criterion because it is the position the direction was
calibrated at and the position that exists before the reply does. The
`proj_post_reply` column is a labelled diagnostic, not a second shot at the
gate.

CPU-only; consumes the GPU job's artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

from surrogate_eval import bootstrap_ci

HERE = pathlib.Path(__file__).resolve().parent
TURNS = (1, 2, 3, 4, 5)
CRITERION_BINS = 3
CRITERION_FIELD = "proj_pre_reply"
OBJECTIVE = "late_t3_5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projection-path", type=pathlib.Path,
                        default=pathlib.Path("outputs/koopman_case_study/zero_control_projection_readout.json"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out-path", type=pathlib.Path,
                        default=pathlib.Path("outputs/koopman_case_study/gate_r1_projection_verdict.json"))
    return parser.parse_args()


def _ceiling():
    spec = importlib.util.spec_from_file_location("_ceiling", HERE / "analyze_adaptivity_ceiling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_projections(path: pathlib.Path, field: str, keys, arm: str) -> dict:
    """One projection per (attack, seed, turn) off the unfired arm. Raises on a
    hole rather than quietly conditioning on a shorter history: a missing turn
    would silently shrink the policy's information set and depress the ceiling
    it is supposed to measure."""

    payload = json.loads(path.read_text())
    if payload.get("overlap_with_scored_attacks"):
        raise SystemExit(
            f"direction overlaps the scored attacks {payload['overlap_with_scored_attacks']} -- "
            "its harmless pole is those attacks' own turn-1 queries, so the ceiling would be leaked")
    table = {(row["attack_id"], row["seed"], row["turn"]): float(row[field])
             for row in payload["rows"] if row["arm"] == arm}
    missing = [(key, turn) for key in keys for turn in TURNS if (key[0], key[1], turn) not in table]
    if missing:
        raise SystemExit(f"{len(missing)} (trajectory, turn) cells missing from {path}, e.g. {missing[:3]}")
    return table


def main() -> None:
    args = parse_args()
    ceiling = _ceiling()
    score, sign = ceiling.OBJECTIVES[OBJECTIVE]

    zero = ceiling.load("outputs/koopman_defense_phaseE_zero_control_5seed")
    fixed = {t: ceiling.load(f"outputs/koopman_defense_phaseJ_budget1_fixed_t{t}") for t in TURNS}
    ceiling.guard_prefix_identity(zero, fixed)
    keys = sorted(set(zero) & set.intersection(*(set(a) for a in fixed.values())))
    attacks = sorted({key[0] for key in keys})
    seeds = sorted({key[1] for key in keys})
    groups = [key[0] for key in keys]

    def series(rows, key):
        return [float(rows[key][turn]["y_safety"]) for turn in sorted(rows[key])]

    terminal = {t: {key: score(series(fixed[t], key)) for key in keys} for t in TURNS}
    never = {key: score(series(zero, key)) for key in keys}
    levels = {t: float(np.mean([terminal[t][key] for key in keys])) for t in TURNS}
    best_fixed = max(TURNS, key=lambda t: levels[t])
    baseline = np.array([terminal[best_fixed][key] for key in keys])

    def evaluate(values):
        boot = bootstrap_ci(np.asarray(values) - baseline, groups,
                            seed=args.bootstrap_seed, n_resamples=args.bootstrap)
        return {"gain": boot["point"], "ci": [boot["ci_low"], boot["ci_high"]],
                "excludes_zero": boot["excludes_zero"], "clears_zero_upward": boot["ci_low"] > 0}

    def ceiling_for(prefix):
        in_sample = ceiling.causal_oracle(keys, keys, prefix, terminal, never, sign)
        out = {"in_sample": evaluate([in_sample[key] for key in keys])}
        for axis, index, buckets in (("leave_one_attack_out", 0, attacks), ("leave_one_seed_out", 1, seeds)):
            replayed = {}
            for bucket in buckets:
                replayed.update(ceiling.causal_oracle(
                    [k for k in keys if k[index] != bucket], [k for k in keys if k[index] == bucket],
                    prefix, terminal, never, sign))
            out[axis] = evaluate([replayed[key] for key in keys])
        return out

    results = {"y_safety_reference": ceiling_for(
        {(key, turn): tuple(float(zero[key][s]["y_safety"]) for s in range(1, turn))
         for key in keys for turn in TURNS})}

    for field in ("proj_pre_reply", "proj_post_reply"):
        table = load_projections(args.projection_path, field, keys, "zero_control_5seed")
        for n_bins in (2, 3, 4):
            raw = [table[(key[0], key[1], turn)] for key in keys for turn in TURNS]
            edges = np.quantile(np.asarray(raw), np.linspace(0, 1, n_bins + 1)[1:-1])
            prefix = {(key, turn): tuple(int(np.searchsorted(edges, table[(key[0], key[1], s)]))
                                         for s in range(1, turn))
                      for key in keys for turn in TURNS}
            results[f"{field}_{n_bins}bins"] = ceiling_for(prefix)
            results[f"{field}_{n_bins}bins"]["distinct_prefixes_by_turn"] = {
                turn: len({prefix[(key, turn)] for key in keys}) for turn in TURNS}

    criterion = results[f"{CRITERION_FIELD}_{CRITERION_BINS}bins"]
    passed = (criterion["leave_one_attack_out"]["clears_zero_upward"]
              or criterion["leave_one_seed_out"]["clears_zero_upward"])
    verdict = "PASS" if passed else "FAIL"

    print(f"objective={OBJECTIVE}  baseline=fixed_t{best_fixed} ({levels[best_fixed]:.4f})  "
          f"{len(attacks)} attacks x {len(seeds)} seeds\n")
    print(f"{'conditioning signal':<26}{'in-sample':>24}{'xfit: new attack':>24}{'xfit: new conversation':>26}")
    for name, value in results.items():
        cell = lambda c: f"{c['gain']:+.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]{'*' if c['excludes_zero'] else ' '}"
        mark = "  <= CRITERION" if name == f"{CRITERION_FIELD}_{CRITERION_BINS}bins" else ""
        print(f"{name:<26}{cell(value['in_sample']):>24}{cell(value['leave_one_attack_out']):>24}"
              f"{cell(value['leave_one_seed_out']):>26}{mark}")

    print(f"\nGATE R1: {verdict}")
    print("  pass -> re-run Table 2's defense closed loop on the projection readout (~3.5 GPU-h)"
          if passed else
          "  fail -> close the line; Table 2's negative stands, section 2's open-direction sentence is rewritten")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps({
        "gate": "R1", "verdict": verdict, "objective": OBJECTIVE,
        "criterion": {"field": CRITERION_FIELD, "n_bins": CRITERION_BINS,
                      "rule": "95% CI lower bound > 0 on either cross-fit axis",
                      "signed_before_the_job_ran": True},
        "baseline": {"best_fixed_turn": best_fixed, "level": levels[best_fixed]},
        "judge_kind": "self",
        "results": results,
    }, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
