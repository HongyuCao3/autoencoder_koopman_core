#!/usr/bin/env python3
"""How much was adaptivity ever worth here? -- an upper bound on every
controller, not a verdict on ours.

Phase J and the main table both answer "did Koopman-MPC beat the best fixed
schedule" (no, docs/experiments/defense_table2_results.md). That question has
a reviewer-shaped hole in it: a loss is consistent with "the controller is
bad" AND with "there was nothing to win", and those call for opposite next
moves. This script measures the second one, so the first stops being a
speculation either side can assert.

WHAT MAKES THE BOUNDS EXACTLY COMPUTABLE. Decoding is sampled (temperature
0.7) but DETERMINISTIC GIVEN THE SEED -- `agent_seed = seed*1e6 + turn*100 + 1`
-- and a reminder enters the context only at the turn it is inserted, so every
`fixed_t{k}` arm is byte-identical to `zero_control` for turns 1..k-1. That is
checked as a guard below, not assumed: it means the observable history at any
decision point IS the `zero_control` prefix, and firing at turn t yields
exactly `fixed_t{t}`'s outcome. Every policy over these five actions can
therefore be replayed off data already on disk, with no counterfactual
modelling and no new generation.

FOUR BOUNDS, nested by what the policy is allowed to know:

  global fixed schedule    knows nothing        <- the baseline
  causal oracle            the observed y-prefix at decision time
  per-attack oracle        which attack this is (NOT observable at runtime)
  clairvoyant oracle       the outcome of every action on this trajectory

The causal oracle is the one that binds: it is the supremum over ALL policies
measurable with respect to the observed readout history -- Koopman, LSTM, RL,
a hand-written rule. Fitted and scored on the same 40 trajectories, so it is
an overfit upper bound on purpose; the leave-one-attack-out number beside it
says how much of that survives contact with a trajectory it has not seen.

SCOPE. It bounds policies that condition on the y readout. A controller
reading something richer (the message text, activations) is NOT bounded by
it, which is the one direction this analysis leaves open rather than closes.

CPU-only.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from surrogate_eval import bootstrap_ci

SCORES_NAME = "trajectories.jsonl"
TURNS = (1, 2, 3, 4, 5)
ZERO_CONTROL = "outputs/koopman_defense_phaseE_zero_control_5seed"
FIXED = {t: f"outputs/koopman_defense_phaseJ_budget1_fixed_t{t}" for t in TURNS}
OURS = "outputs/koopman_defense_phaseJ_budget1_koopman"
# Readout aggregations. A policy that wins only under the aggregation it was
# scored on has not won; `auc_below` is a loss, so it carries the flipped sign.
OBJECTIVES = {
    "terminal": (lambda v: v[-1], +1),
    "mean_all": (lambda v: float(np.mean(v)), +1),
    "mean_t2_5": (lambda v: float(np.mean(v[1:])), +1),
    # The `defense` line's signed primary (ruling 5, 2026-09-13): the same
    # late window Phase E-J pre-registered and the same shape as the
    # `constraint` column's t15-t20.
    "late_t3_5": (lambda v: float(np.mean(v[2:])), +1),
    "min_any": (lambda v: float(np.min(v)), +1),
    "auc_below_one": (lambda v: float(np.sum([1.0 - x for x in v])), -1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--objective", default="terminal", choices=sorted(OBJECTIVES))
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_case_study/adaptivity_ceiling.json"),
    )
    return parser.parse_args()


def load(directory: str) -> dict[tuple, dict[int, dict]]:
    rows = [json.loads(line) for line in (pathlib.Path(directory) / SCORES_NAME).read_text().splitlines() if line.strip()]
    out: dict[tuple, dict[int, dict]] = collections.defaultdict(dict)
    for row in rows:
        out[(row["attack_id"], row["seed"])][row["turn"]] = row
    return out


def guard_prefix_identity(zero: dict, fixed: dict[int, dict]) -> dict:
    """The whole construction rests on `fixed_t{k}` matching `zero_control`
    before turn k. If a rerun ever breaks it (non-greedy decoding, a changed
    prompt assembly), every bound below silently becomes a different quantity,
    so this raises rather than warns."""

    checked = mismatched = 0
    for turn, arm in fixed.items():
        shared = sorted(set(arm) & set(zero))
        if not shared:
            raise SystemExit(f"fixed_t{turn} shares no trajectory key with zero_control")
        fired = sorted({t for key in shared for t in arm[key] if int(arm[key][t]["u_remind"]) == 1})
        if fired != [turn]:
            raise SystemExit(f"fixed_t{turn} fires on turns {fired}, not [{turn}] -- arm is mislabelled")
        for key in shared:
            for before in range(1, turn):
                checked += 1
                if arm[key][before]["agent_message"] != zero[key][before]["agent_message"]:
                    mismatched += 1
    if mismatched:
        raise SystemExit(
            f"{mismatched}/{checked} pre-reminder turns differ from zero_control: the observed history "
            "at a decision point is NOT the zero-control prefix, so none of these bounds hold"
        )
    return {"pre_reminder_turns_checked": checked, "all_identical": True}


def causal_oracle(fit_keys, eval_keys, prefix, terminal, never, sign: int) -> dict:
    """Backward induction over the observable prefix. Decisions are made per
    prefix GROUP (that is what "measurable with respect to what it can see"
    means); the value is settled per trajectory."""

    decide: dict[tuple, bool] = {}
    # Decision to fall back on when a replay presents a history the fit never
    # saw. Truncating the prefix does not help when it is the FIRST
    # observation that is unseen, so the fallback is the pooled decision over
    # every fit trajectory still unfired at that turn.
    marginal: dict[int, bool] = {}
    value = {key: never[key] for key in fit_keys}  # turn 6: never fired
    for turn in range(5, 0, -1):
        pooled_fire = float(np.mean([terminal[turn][key] for key in fit_keys]))
        pooled_wait = float(np.mean([value[key] for key in fit_keys]))
        marginal[turn] = sign * pooled_fire >= sign * pooled_wait
        groups: dict[tuple, list] = collections.defaultdict(list)
        for key in fit_keys:
            groups[prefix[(key, turn)]].append(key)
        updated = {}
        for pre, members in groups.items():
            fire = float(np.mean([terminal[turn][key] for key in members]))
            wait = float(np.mean([value[key] for key in members]))
            # `sign` carries the objective's direction: `auc_below_one` is a
            # loss, and a maximising induction on it would build the WORST
            # policy while still being called an oracle.
            prefer_fire = sign * fire >= sign * wait
            decide[(turn, pre)] = prefer_fire
            for key in members:
                updated[key] = terminal[turn][key] if prefer_fire else value[key]
        value = updated

    replayed = {}
    for key in eval_keys:
        fired = None
        for turn in TURNS:
            pre = prefix[(key, turn)]
            choice = decide.get((turn, pre))
            if choice is None:
                # Unseen history: the longest seen ancestor if there is one,
                # else the pooled decision for this turn. A policy at
                # deployment has to answer either way.
                for length in range(len(pre) - 1, -1, -1):
                    if (turn, pre[:length]) in decide:
                        choice = decide[(turn, pre[:length])]
                        break
                else:
                    choice = marginal[turn]
            if choice:
                fired = turn
                break
        replayed[key] = terminal[fired][key] if fired else never[key]
    return replayed


def main() -> None:
    args = parse_args()
    score, sign = OBJECTIVES[args.objective]

    zero_rows, fixed_rows, ours_rows = load(ZERO_CONTROL), {t: load(d) for t, d in FIXED.items()}, load(OURS)
    prefix_check = guard_prefix_identity(zero_rows, fixed_rows)

    keys = sorted(set(zero_rows) & set(ours_rows) & set.intersection(*(set(a) for a in fixed_rows.values())))
    attacks = sorted({key[0] for key in keys})
    groups = [key[0] for key in keys]

    def series(rows, key):
        return [float(rows[key][turn]["y_safety"]) for turn in sorted(rows[key])]

    terminal = {t: {key: score(series(fixed_rows[t], key)) for key in keys} for t in TURNS}
    never = {key: score(series(zero_rows, key)) for key in keys}
    ours = {key: score(series(ours_rows, key)) for key in keys}
    # The observable history before firing is the zero-control prefix, which
    # the guard above just established.
    y_prefix = {(key, turn): tuple(float(zero_rows[key][s]["y_safety"]) for s in range(1, turn))
                for key in keys for turn in TURNS}

    levels = {t: float(np.mean([terminal[t][key] for key in keys])) for t in TURNS}
    best_fixed = max(TURNS, key=lambda t: sign * levels[t])
    baseline = np.array([terminal[best_fixed][key] for key in keys])

    per_attack_best = {}
    for attack in attacks:
        members = [key for key in keys if key[0] == attack]
        per_attack = {t: float(np.mean([terminal[t][key] for key in members])) for t in TURNS}
        per_attack_best[attack] = {
            "best_turn": max(TURNS, key=lambda t: sign * per_attack[t]),
            "levels": per_attack,
            "gain_over_global_best": sign * (max(sign * v for v in per_attack.values()) - sign * per_attack[best_fixed]),
        }
    # Counting distinct argmaxes overstates separability when turns tie, so
    # the gain over the global choice is reported next to the count.
    n_distinct = len({a["best_turn"] for a in per_attack_best.values()})
    n_strict = sum(a["gain_over_global_best"] > 1e-12 for a in per_attack_best.values())

    policies = {
        "ours_koopman_mpc": np.array([ours[key] for key in keys]),
        "oracle_per_attack": np.array([terminal[per_attack_best[key[0]]["best_turn"]][key] for key in keys]),
        "oracle_clairvoyant": np.array([sign * max(sign * terminal[t][key] for t in TURNS) for key in keys]),
        "oracle_causal_in_sample": np.array(
            [causal_oracle(keys, keys, y_prefix, terminal, never, sign)[key] for key in keys]),
    }
    crossfit = {}
    for attack in attacks:
        crossfit.update(causal_oracle([k for k in keys if k[0] != attack], [k for k in keys if k[0] == attack],
                                      y_prefix, terminal, never, sign))
    policies["oracle_causal_leave_one_attack_out"] = np.array([crossfit[key] for key in keys])

    bounds = {}
    for name, values in policies.items():
        diffs = values - baseline
        boot = bootstrap_ci(diffs, groups, seed=args.bootstrap_seed, n_resamples=args.bootstrap)
        bounds[name] = {
            "level": float(np.mean(values)),
            "gain_over_best_fixed": boot["point"],
            "ci": [boot["ci_low"], boot["ci_high"]],
            "excludes_zero": boot["excludes_zero"],
        }

    # What a controller can actually see at each decision point. This is the
    # mechanism behind whatever the causal bound turns out to be.
    visibility = {
        turn: {"distinct_observable_prefixes": len({y_prefix[(key, turn)] for key in keys}),
               "n_trajectories": len(keys)}
        for turn in TURNS
    }

    report = {
        "objective": args.objective,
        "prefix_identity_guard": prefix_check,
        "baseline": {"best_fixed_turn": best_fixed, "level": levels[best_fixed], "all_levels": levels},
        "separability": {
            "n_distinct_best_turns": n_distinct,
            "n_attacks_strictly_better_than_global": n_strict,
            "per_attack": per_attack_best,
        },
        "bounds_vs_best_fixed": bounds,
        "observable_history_at_decision_time": visibility,
        "n_attacks": len(attacks),
        "n_trajectories": len(keys),
        "judge_kind": "self",
        "limitation_required_at_every_quotation": (
            "self-judged readout (.claude/global.md named exception, `defense` only); and these bounds "
            "constrain policies measurable w.r.t. the y readout only -- a controller reading text or "
            "activations is not bounded by them"
        ),
    }

    print(f"objective={args.objective}  {len(attacks)} attacks x {len(keys) // len(attacks)} seeds  "
          f"baseline = fixed_t{best_fixed} ({levels[best_fixed]:.4f})\n")
    print(f"{'policy':<38}{'level':>9}{'gain':>10}  95% CI")
    for name, value in bounds.items():
        star = " *" if value["excludes_zero"] else ""
        print(f"{name:<38}{value['level']:>9.4f}{value['gain_over_best_fixed']:>+10.4f}  "
              f"[{value['ci'][0]:+.4f}, {value['ci'][1]:+.4f}]{star}")

    print(f"\nseparability: {n_distinct} distinct best turns across {len(attacks)} attacks, "
          f"but only {n_strict} beat the global choice by anything")
    print("\nwhat the controller can see when it has to decide:")
    for turn, info in visibility.items():
        print(f"  turn {turn}: {info['distinct_observable_prefixes']:>2} distinct observable prefixes "
              f"/ {info['n_trajectories']} trajectories")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
