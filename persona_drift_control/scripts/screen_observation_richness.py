#!/usr/bin/env python3
"""Does ANY richer observable signal raise the adaptivity ceiling above what
the `y` readout gives?

`analyze_adaptivity_ceiling.py` bounds every policy measurable w.r.t. the
`y_safety` history at +0.0063 (CI crosses zero) and names the mechanism: at
the two decision points that matter, all 40 trajectories look identical to a
`y`-conditioned controller. That bound explicitly does NOT cover a controller
reading something else, which left exactly one direction open. This screens
it, on data already on disk.

WHAT IS SCREENABLE HERE AND WHAT IS NOT. Per-turn activation projections are
NOT: `outputs/safety_direction/` holds the direction vector, not per-row
projections, and producing them means re-running the model. So this screens
the text- and metadata-derived signals the trajectories already carry. A null
here therefore closes the cheap half of the open direction, not all of it.

WHY THE IN-SAMPLE NUMBER IS NOT THE GATE. `len(agent_message)` separates 39 of
40 trajectories by turn 2 -- it is a fingerprint. Fitted and scored on the same
40 trajectories, any such feature drives the oracle to the clairvoyant bound by
memorising, which measures cardinality, not signal. The gate is cross-fitted.

TWO CROSS-FIT AXES, BECAUSE THEY ARE TWO DIFFERENT DEPLOYMENT CLAIMS:
  * leave-one-ATTACK-out  -- generalise to an attack never seen before.
  * leave-one-SEED-out    -- generalise to a new conversation against an
    attack family already seen. This is the weaker claim, and the only one
    under which conditioning on attack identity can legitimately pay; the
    per-attack oracle's +0.0250 is its ceiling.

The binning quantiles are computed over all trajectories rather than per fold,
which leaks a little in the screen's favour. That is deliberate: a null under a
generous construction is a stronger null.

CPU-only.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import pathlib

import numpy as np

from surrogate_eval import bootstrap_ci

HERE = pathlib.Path(__file__).resolve().parent
TURNS = (1, 2, 3, 4, 5)
ZERO_CONTROL = "outputs/koopman_defense_phaseE_zero_control_5seed"
FIXED = {t: f"outputs/koopman_defense_phaseJ_budget1_fixed_t{t}" for t in TURNS}

# name -> (row -> raw value, n_bins, looks_ahead). `looks_ahead` marks a signal
# available for the turn being decided, not just the turns before it: the
# attacker script is fixed and identical across arms (checked below), so the
# query the agent is about to answer is already on screen when the controller
# has to choose.
FEATURES = {
    "y_safety": (lambda row: float(row["y_safety"]), 0, False),
    "y_probe": (lambda row: float(row["y_probe"]), 0, False),
    "message_len": (lambda row: len(row["agent_message"]), 3, False),
    "query_len": (lambda row: len(row["attacker_query"]), 3, False),
    "query_len_lookahead": (lambda row: len(row["attacker_query"]), 3, True),
    "y_safety_plus_message_len": (None, 3, False),  # handled as a pair below
    "y_safety_plus_query_lookahead": (None, 3, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    # Ruling 5 makes the late window the reported primary, so the screen is
    # scored on it by default rather than on the terminal turn.
    parser.add_argument("--objective", default="late_t3_5")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--out-path", type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_case_study/observation_richness_screen.json"))
    return parser.parse_args()


def _ceiling_module():
    spec = importlib.util.spec_from_file_location("_ceiling", HERE / "analyze_adaptivity_ceiling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binner(values: list[float], n_bins: int):
    """Coarse quantile bins. Without them a continuous signal is a per-row id
    and the induction memorises instead of generalising."""
    if n_bins <= 0:
        return lambda value: value
    edges = np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1])
    return lambda value: int(np.searchsorted(edges, value))


def build_prefixes(zero, keys, name):
    """prefix[(key, turn)] = everything the controller can see when it must
    decide at `turn`, under this feature set."""
    pair = name.startswith("y_safety_plus_")
    if pair:
        other = name.replace("y_safety_plus_", "").replace("query_lookahead", "query_len")
        parts = [("y_safety", *FEATURES["y_safety"][1:]), (other, *FEATURES[other][1:])]
        extractors = [FEATURES["y_safety"][0], FEATURES[other][0]]
        looks_ahead = FEATURES[name][2]
    else:
        parts = [(name, FEATURES[name][1], FEATURES[name][2])]
        extractors = [FEATURES[name][0]]
        looks_ahead = FEATURES[name][2]

    binners = []
    for (part_name, n_bins, _), extract in zip(parts, extractors):
        raw = [extract(zero[key][turn]) for key in keys for turn in TURNS]
        binners.append(_binner(raw, n_bins))

    prefix = {}
    for key in keys:
        for turn in TURNS:
            seen = list(range(1, turn + 1)) if looks_ahead else list(range(1, turn))
            prefix[(key, turn)] = tuple(
                binner(extract(zero[key][s]))
                for s in seen for binner, extract in zip(binners, extractors)
            )
    return prefix


def main() -> None:
    args = parse_args()
    ceiling = _ceiling_module()

    zero = ceiling.load(ZERO_CONTROL)
    fixed = {t: ceiling.load(d) for t, d in FIXED.items()}
    ceiling.guard_prefix_identity(zero, fixed)

    keys = sorted(set(zero) & set.intersection(*(set(a) for a in fixed.values())))
    attacks = sorted({key[0] for key in keys})
    seeds = sorted({key[1] for key in keys})
    groups = [key[0] for key in keys]

    # A look-ahead feature is only replayable if the attacker script does not
    # react to the reminder. Checked, not assumed.
    divergent = sum(
        fixed[t][key][turn]["attacker_query"] != zero[key][turn]["attacker_query"]
        for t in TURNS for key in keys for turn in TURNS
    )
    if divergent:
        raise SystemExit(
            f"{divergent} attacker queries differ across arms -- the attacker reacts to the reminder, "
            "so a look-ahead feature cannot be replayed off these arms")

    score, sign = ceiling.OBJECTIVES[args.objective]
    if sign != 1:
        raise SystemExit(f"--objective {args.objective} is a loss; this screen's induction assumes a gain")

    def series(rows, key):
        return [float(rows[key][turn]["y_safety"]) for turn in sorted(rows[key])]

    terminal = {t: {key: score(series(fixed[t], key)) for key in keys} for t in TURNS}
    never = {key: score(series(zero, key)) for key in keys}
    levels = {t: float(np.mean([terminal[t][key] for key in keys])) for t in TURNS}
    best_fixed = max(TURNS, key=lambda t: levels[t])
    baseline = np.array([terminal[best_fixed][key] for key in keys])

    def evaluate(values):
        boot = bootstrap_ci(np.asarray(values) - baseline, groups, seed=args.bootstrap_seed, n_resamples=args.bootstrap)
        return {"gain": boot["point"], "ci": [boot["ci_low"], boot["ci_high"]], "excludes_zero": boot["excludes_zero"]}

    results = {}
    for name in FEATURES:
        prefix = build_prefixes(zero, keys, name)
        in_sample = ceiling.causal_oracle(keys, keys, prefix, terminal, never, +1)
        folds = {}
        for axis, buckets in (("attack", attacks), ("seed", seeds)):
            index = 0 if axis == "attack" else 1
            replayed = {}
            for bucket in buckets:
                fit = [key for key in keys if key[index] != bucket]
                held = [key for key in keys if key[index] == bucket]
                replayed.update(ceiling.causal_oracle(fit, held, prefix, terminal, never, +1))
            folds[axis] = evaluate([replayed[key] for key in keys])
        results[name] = {
            "distinct_prefixes_by_turn": {t: len({prefix[(key, t)] for key in keys}) for t in TURNS},
            "in_sample_upper_bound": evaluate([in_sample[key] for key in keys]),
            "crossfit_leave_one_attack_out": folds["attack"],
            "crossfit_leave_one_seed_out": folds["seed"],
        }

    print(f"baseline = fixed_t{best_fixed} ({levels[best_fixed]:.4f}); {len(attacks)} attacks x {len(seeds)} seeds\n")
    print(f"{'signal':<32}{'prefixes @t2':>13}{'in-sample':>12}{'xfit:attack':>14}{'xfit:seed':>13}")
    for name, value in results.items():
        star = lambda cell: "*" if cell["excludes_zero"] else " "
        print(f"{name:<32}{value['distinct_prefixes_by_turn'][2]:>13}"
              f"{value['in_sample_upper_bound']['gain']:>+11.4f}{star(value['in_sample_upper_bound'])}"
              f"{value['crossfit_leave_one_attack_out']['gain']:>+13.4f}{star(value['crossfit_leave_one_attack_out'])}"
              f"{value['crossfit_leave_one_seed_out']['gain']:>+12.4f}{star(value['crossfit_leave_one_seed_out'])}")

    report = {
        "objective": args.objective,
        "baseline": {"best_fixed_turn": best_fixed, "level": levels[best_fixed]},
        "reference_points": {
            "ours_koopman_mpc": -0.0688,
            "per_attack_oracle": 0.0250,
            "clairvoyant_oracle": 0.1062,
        },
        "gate": "a signal is worth GPU only if a CROSS-FITTED gain clears zero; in-sample measures cardinality",
        "not_screened": "per-turn activation projections (would require re-running the model)",
        "results": results,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
