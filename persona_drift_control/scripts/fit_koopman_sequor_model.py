#!/usr/bin/env python3
"""S2: fit `y_(t+1) = A y_t + B u_t + c` on the `constraint` line's S1b
excitation arms (docs/experiments/constraint_retention_plan.md section 5).
CPU only, pure numpy -- run directly, no sbatch.

    G-S2-1 state        one-step scalar-y MSE beats the best of three trivial
                        nulls (`const` / `turn_mean` / `stateless`) on >= 14
                        of 20 item-disjoint folds. Every null is handed
                        `turn`, the deterministic exogenous quantity the
                        surrogate also has -- denying it would manufacture a
                        pass (the ERGO RC-1 defect, mirrored).
    G-S2-2 controllable `B`'s 95% CI excludes 0 with the sign that says a
                        reminder raises `y`.
    G-S2-3 degeneracy   spectral radius in (0.1, 1.05), full controllability
                        rank, Gramian condition < 1e12. RECORDED, not a gate.
    G-S2-4 sizing       trajectories per arm S3 needs to resolve half the
                        full-dose contrast (user ruling 2026-09-10), on the
                        PAIRED per-(item, seed) difference variance that plan
                        section 6's primary actually has. > 150 per arm ->
                        stop and report.

The input is S1b (`bernoulli` + `antithetic`) and only S1b: the S1a arms are a
full-dose endpoint contrast with `u` constant within an arm, which identifies
no `B` at all. `--contemporaneous-v` is mandatory, not default -- the reminder
for turn t is injected BEFORE that turn's reply is generated and scored, so
`u_t` acts on `y_t`, and fitting the lagged slot would estimate the carryover
of last turn's reminder text while calling it the actuator's effect. Both the
`defense` line and ERGO retracted a verdict over exactly that off-by-one.

OPTION (a), SIGNED 2026-09-11 (screening section 10 item 11). The S3 pre-check
found the scalar operator's budget-k optimum to be the same k turns for every
trajectory, while 60-67% of real late turns sit at y = 1 -- a ceiling the
scalar fit smooths away, and saturation is exactly the mechanism that makes
optimal timing state dependent. So the same two checks are re-run on the state
the judge already returns: the per-constraint binary `followed` vector,
collapsed to the count `m` in {0,1,2,3} with `y = m/3` unchanged. The kernel
`P(m_(t+1) | m_t, u_t)` is counted, not regressed, so the boundary is
represented rather than approximated. PRE-REGISTERED verdict, written before
the numbers were seen: read `mpc_minus_best_fixed` against the SAME MDE the
scalar check used (G-S2-4's design resolution, 0.0690, signed 2026-09-10). At
or above it the degeneracy was the model class; below it a model class that
carries the ceiling explicitly still wants one plan for everyone, and the
degeneracy belongs to the task. No third reading and no moving the bar. Zero
GPU: the 9600 rows of `outputs/sequor_s1_arm/` are the input, unchanged.

OPTION (b) ON PAPER, SIGNED 2026-09-11 (screening section 10 item 12). Option
(a) showed the degeneracy is the objective's shape, not the model class: under
a FIXED budget over a FIXED window the reminders must all be spent and only
their placement is free, so the state scales what a reminder is worth without
moving the argmax over turns. The threshold objective removes exactly that --
"keep m >= theta, pay for every reminder" makes WHETHER to spend the question,
and "do I still need one" is a function of the state by construction of the
task. Before rewriting plan section 6's arm table, the same identified kernel
is asked on paper: at EQUAL EXPECTED COST, how much more of the late window
does the feedback policy hold above the threshold than the best fixed
schedule? Matched cost is what keeps the comparison from being won by
definition. PRE-REGISTERED: the largest equal-cost gap is read against THIS
objective's own MDE at the current design, computed from the same paired
variance components on the new primary. At or above it, the rewrite is worth
the GPU; below it, the degeneracy survives the objective change too. Zero GPU
again: same 9600 rows, same kernel, no new arm.

WHAT A WEAK RESULT HERE DOES NOT MEAN (screening section 10 item 9, signed
2026-09-10): 65% of the judged constraints on this item set lie outside the
judge's calibration set, and that observation noise biases the state
coefficient toward zero. A weak or null G-S2-1 on these rows therefore has two
possible causes this design cannot separate, and it does NOT close the line.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import itertools
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    group_by_trajectory,
)
from persona_drift.modeling.koopman import (  # noqa: E402
    KoopmanSurrogate,
    abs_sign_extra_features,
    no_extra_features,
)

S1B_ARMS = ("bernoulli", "antithetic")
N_BOOTSTRAP = 10000
N_BOOTSTRAP_INTERACTION = 4000  # refits a 4-column design per draw; 4k is stable to 1e-3 here
POWER_Z = 2.80  # two-sided alpha=0.05 at 80% power, as in every gate on this line
S3_TRAJECTORIES_PER_ARM_CEILING = 150  # plan section 5, G-S2-4


def fit_bilinear(transitions: list[dict], seed: int) -> dict:
    """G-S2-5: is the actuator's effect state dependent?

    `y_(t+1) = c + a y_t + b u_t + d (u_t * y_t)`. `d` is the whole question
    for S3, and the reason is arithmetic rather than cautious: on a purely
    linear operator a reminder at turn t contributes `b * a^(T-t)` to the
    objective NO MATTER what the state is, so the budget-k optimum is the same
    k turns for every trajectory and `koopman_mpc` cannot differ from the best
    fixed schedule. ERGO died exactly here -- its bilinear term was -0.000028,
    CI [-0.000381, +0.000276], and enumeration then found one distinct
    schedule over 135 trajectories.

    Reported in two specifications because they answer different questions.
    POOLED `d` mixes "reminders help on hard items" with "reminders help when
    THIS dialogue has slipped"; ITEM-DEMEANED keeps only the second, which is
    the one a within-dialogue feedback loop can act on.

    Verdict vocabulary follows the K2 clause ordering signed 2026-09-09
    (screening section 10 item 7): a degenerate axis first, then a CI that
    excludes zero is a PASS whose power is demonstrated after the fact, and a
    CI crossing zero is UNDECIDABLE rather than FAIL whenever the MDE is
    larger than the effect the gate is meant to see.
    """

    y0 = np.array([t["y_now"] for t in transitions])
    y1 = np.array([t["y_next"] for t in transitions])
    u = np.array([t["u"] for t in transitions])
    items = np.array([t["item_id"] for t in transitions])
    unique = sorted(set(items.tolist()))
    mean_gain = float(np.mean(y1[u == 1]) - np.mean(y1[u == 0])) if (u == 1).any() and (u == 0).any() else 0.0

    def spec(demean: bool) -> dict:
        a0, a1, au = y0.copy(), y1.copy(), u.copy()
        if demean:
            for item in unique:
                mask = items == item
                a0[mask] -= a0[mask].mean()
                a1[mask] -= a1[mask].mean()
                au[mask] -= au[mask].mean()
        design = np.column_stack([np.ones_like(a0), a0, au, au * a0])
        if np.linalg.matrix_rank(design) < design.shape[1] or len(set(a0.tolist())) < 3:
            return {"degenerate_axis": True, "verdict": "UNDECIDABLE",
                    "reason": "y_t has no usable spread: there is no axis for an interaction"}
        beta = np.linalg.lstsq(design, a1, rcond=None)[0]
        per_item = {item: (design[items == item], a1[items == item]) for item in unique}
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(N_BOOTSTRAP_INTERACTION):
            chosen = rng.choice(len(unique), size=len(unique), replace=True)
            x = np.vstack([per_item[unique[i]][0] for i in chosen])
            y = np.concatenate([per_item[unique[i]][1] for i in chosen])
            draws.append(float(np.linalg.lstsq(x, y, rcond=None)[0][3]))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        sd = float(np.std(draws, ddof=1))
        mde = POWER_Z * sd
        d = float(beta[3])
        excludes_zero = bool(lo * hi > 0)
        if excludes_zero:
            verdict = "PASS"
            reason = "the CI excludes 0: the actuator's effect depends on the state"
        elif mde > abs(mean_gain):
            verdict = "UNDECIDABLE"
            reason = ("the CI crosses 0 but the interaction MDE exceeds the mean one-step gain "
                      "itself -- this arm could not resolve a state dependence even as large as "
                      "the whole average effect (ERGO G-EKA-2 precedent)")
        else:
            verdict = "NULL"
            reason = "the CI crosses 0 on a design powered to see an effect the size of the mean gain"
        return {
            "degenerate_axis": False, "interaction_d": d, "ci": [float(lo), float(hi)],
            "bootstrap_sd": sd, "mde_at_80pct": mde, "effect_over_mde": abs(d) / mde if mde else None,
            "y_prev_coefficient": float(beta[1]), "u_coefficient": float(beta[2]),
            "marginal_effect_of_a_reminder": {f"y={v:.2f}": float(beta[2] + beta[3] * v)
                                              for v in (0.0, 1 / 3, 2 / 3, 1.0)},
            "ci_excludes_zero": excludes_zero,
            "underpowered": bool(abs(d) < mde),
            "verdict": verdict, "reason": reason,
        }

    pooled, demeaned = spec(False), spec(True)
    return {
        "criterion": "the u x y_t coefficient's 95% CI excludes 0 (bootstrap over items)",
        "mean_one_step_gain_for_scale": mean_gain,
        "pooled": pooled, "item_demeaned": demeaned,
        "verdict": pooled["verdict"],
        "why_it_gates_s3": "on a linear operator the budget-k optimum is the same k turns for "
                           "every trajectory, so koopman_mpc is identical to the best fixed "
                           "schedule by construction. Only this term can make the closed loop "
                           "non-degenerate.",
        "read_the_two_specs_together": (
            "pooled mixes per-item difficulty into the state; item-demeaned keeps only the "
            "within-dialogue part, which is what a feedback loop acts on. A pooled PASS with an "
            "UNDECIDABLE demeaned spec means the evidence for closed-loop headroom is not yet "
            "separated from 'reminders help on hard items'."),
    }


def _step(y: float, u: int, op: dict) -> float:
    return float(np.clip(op["c"] + op["a"] * y + op["b"] * u + op["d"] * u * y, 0.0, 1.0))


def _objective(y0: float, schedule: tuple[int, ...], start_turn: int, late_from: int,
               op: dict) -> float:
    """Late-window mean `y` under a schedule -- S3's primary quantity, not the
    endpoint. Deterministic rollout: this is the planner's own model.

    TURN LABELLING (fixed 2026-09-11, screening section 10 item 12): the start
    state is turn 1, so the state produced by step `offset` is turn
    `start_turn + offset` -- with `start_turn=2` and `horizon = n_turns -
    start_turn + 1` the rollout covers turns 2..20, one decision per reminder
    slot. The first version wrote `+ 1` here, which labelled that state turn 3
    and let the rollout run to a turn 21 the arm does not have; the late window
    then averaged seven states instead of six. Found because option (b) divides
    by the window length and reported a SHARE of 1.09."""

    y, kept = y0, []
    for offset, u in enumerate(schedule):
        y = _step(y, u, op)
        if start_turn + offset >= late_from:
            kept.append(y)
    return float(np.mean(kept)) if kept else y


def schedule_separability(operators: dict, starts: dict, budgets: tuple[int, ...],
                          start_turn: int, n_turns: int, late_from: int) -> dict:
    """THE question S3 hinges on, asked before any GPU is spent: does the
    identified operator want a different schedule for different items, or the
    same turns for everyone?

    Enumerated exactly over all C(horizon, k) placements rather than through a
    receding-horizon controller, which is an approximation of this same argmax.
    Both operators are run: the linear one is a NEGATIVE CONTROL that must
    return exactly one distinct schedule -- if it ever returns more, the
    checker is broken, not the operator interesting.
    """

    horizon = n_turns - start_turn + 1
    out = {}
    for name, op in operators.items():
        per_budget = {}
        for k in budgets:
            if k > horizon:
                continue
            chosen = {}
            for item, y0 in starts.items():
                best, best_value = None, -np.inf
                for turns in itertools.combinations(range(horizon), k):
                    schedule = tuple(1 if i in turns else 0 for i in range(horizon))
                    value = _objective(y0, schedule, start_turn, late_from, op)
                    if value > best_value:
                        best, best_value = turns, value
                # offsets from the END, so "remind on the last turn" is one schedule
                chosen[item] = tuple(sorted(horizon - 1 - t for t in best))
            counts: dict[tuple, int] = {}
            for value in chosen.values():
                counts[value] = counts.get(value, 0) + 1
            modal, modal_n = max(counts.items(), key=lambda kv: kv[1])
            per_budget[k] = {
                "n_items": len(chosen), "n_distinct_schedules": len(counts),
                "modal_schedule_from_end": list(modal), "modal_share": modal_n / len(chosen),
                "schedule_counts": {str(list(kk)): v for kk, v in sorted(counts.items(), key=lambda kv: -kv[1])},
            }
        out[name] = per_budget
    linear_distinct = {k: v["n_distinct_schedules"] for k, v in out.get("linear", {}).items()}
    bilinear_distinct = {k: v["n_distinct_schedules"] for k, v in out.get("bilinear", {}).items()}
    return {
        "is_a_gate": False,
        "start_turn": start_turn, "horizon": horizon, "objective": f"mean y over t{late_from}..t{n_turns}",
        "by_operator": out,
        "linear_distinct_schedules": linear_distinct,
        "bilinear_distinct_schedules": bilinear_distinct,
        "negative_control_holds": all(v == 1 for v in linear_distinct.values()),
        "closed_loop_has_something_to_do": any(v > 1 for v in bilinear_distinct.values()),
        "reading": "linear must give exactly 1 distinct schedule (it is the negative control). "
                   "If the bilinear operator also gives 1, closed loop IS open loop on this "
                   "operator and the MPC arm cannot differ from the best fixed schedule -- the "
                   "ERGO EK-B finding, reached there for 6.1 GPU-hours less than running it.",
    }


def simulate_s3_gap(op: dict, starts: dict, residuals: np.ndarray, budget: int, start_turn: int,
                    n_turns: int, late_from: int, n_sims: int, seed: int, grid_n: int = 1001) -> dict:
    """How big an arm gap S3 would see, simulated on the identified operator,
    against the MDE this design actually has.

    Three policies, equal budget: the DP-optimal feedback policy (what
    `koopman_mpc` approximates), the single best FIXED schedule shared by all
    items, and a random placement. Planning is deterministic -- the planner
    only has the model -- while the simulated trajectory carries residual
    noise resampled from the fit, so re-planning has something to react to. On
    a noiseless simulation feedback is worthless by construction and the
    comparison would be empty.

    This is a DIAGNOSTIC on a model, not evidence about the agent: it inherits
    every way the operator is wrong, and the operator is already known to
    under-predict the sustained endpoint gap by 2.8x. Its use is one-sided --
    a simulated gap far below the design's MDE says S3 as sized cannot resolve
    what its own model predicts, which is a reason to change the design before
    spending GPU, not a prediction of the result.
    """

    horizon = n_turns - start_turn + 1
    # The DP plans on a discretised state. Too coarse a grid makes the planner
    # lose to the exactly-enumerated fixed schedule on its own model, which
    # would show up as a spurious negative gap -- the zero-noise test pins this.
    grid = np.linspace(0.0, 1.0, grid_n)
    index = lambda y: int(round(float(np.clip(y, 0.0, 1.0)) * (grid_n - 1)))

    # backward DP over (offset, remaining budget, y grid) -> whether to act
    value = np.zeros((horizon + 1, budget + 1, len(grid)))
    act = np.zeros((horizon, budget + 1, len(grid)), dtype=bool)
    for offset in range(horizon - 1, -1, -1):
        turn_next = start_turn + offset
        counts = 1.0 if turn_next >= late_from else 0.0
        for k in range(budget + 1):
            for gi, y in enumerate(grid):
                y0_next = _step(y, 0, op)
                best = counts * y0_next + value[offset + 1, k, index(y0_next)]
                take = False
                if k > 0:
                    y1_next = _step(y, 1, op)
                    alt = counts * y1_next + value[offset + 1, k - 1, index(y1_next)]
                    if alt > best:
                        best, take = alt, True
                value[offset, k, gi] = best
                act[offset, k, gi] = take

    def run(policy, y0: float, noise: np.ndarray) -> float:
        """One trajectory under one policy. `policy` is a fixed schedule or
        None for the DP feedback policy; `noise` is the SAME draw for every
        policy on this (item, sim) so the gap is a paired difference rather
        than the difference of two noise draws."""

        y, k, kept = y0, budget, []
        for offset in range(horizon):
            if policy is None:
                u = int(act[offset, k, index(y)])
            else:
                u = policy[offset]
            k -= u
            y = float(np.clip(_step(y, u, op) + noise[offset], 0.0, 1.0))
            if start_turn + offset >= late_from:
                kept.append(y)
        return float(np.mean(kept))

    # the best fixed schedule the model can name, shared by all items (S3's arm 3)
    best_fixed, best_value = None, -np.inf
    mean_start = float(np.mean(list(starts.values())))
    for turns in itertools.combinations(range(horizon), budget):
        schedule = tuple(1 if i in turns else 0 for i in range(horizon))
        v = _objective(mean_start, schedule, start_turn, late_from, op)
        if v > best_value:
            best_fixed, best_value = schedule, v

    rng = np.random.default_rng(seed)
    gaps_vs_fixed, gaps_vs_random = [], []
    for item, y0 in starts.items():
        for _ in range(n_sims):
            noise = rng.choice(residuals, size=horizon, replace=True)
            turns = tuple(sorted(rng.choice(horizon, size=budget, replace=False).tolist()))
            random_schedule = tuple(1 if i in turns else 0 for i in range(horizon))
            mpc = run(None, y0, noise)
            gaps_vs_fixed.append(mpc - run(best_fixed, y0, noise))
            gaps_vs_random.append(mpc - run(random_schedule, y0, noise))

    return {
        "is_a_gate": False, "budget": budget, "n_sims_per_item": n_sims, "planner_grid": grid_n,
        "best_fixed_schedule_from_end": [horizon - 1 - i for i, u in enumerate(best_fixed) if u],
        "mpc_minus_best_fixed": float(np.mean(gaps_vs_fixed)),
        "mpc_minus_best_fixed_sd": float(np.std(gaps_vs_fixed, ddof=1)),
        "mpc_minus_random": float(np.mean(gaps_vs_random)),
        "paired_on": "common random numbers: every policy sees the same noise draw on each "
                     "(item, sim), so the gap is not the difference of two noise draws",
        "residual_sd_used": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "caveat": "a model simulation, not a prediction: it inherits the operator's errors, and "
                  "the operator already under-predicts the sustained endpoint gap by 2.8x. Use it "
                  "one-sidedly -- a gap far below the design's MDE means S3 as sized cannot see "
                  "what its own model expects.",
    }


BINARY_SPARSE_CELL = 30  # transitions below which a (state, action) cell is flagged, not trusted


def binary_transitions(readout: dict, excluded: list[str]) -> list[dict]:
    """(followed_t, u_t, followed_(t+1)) per adjacent pair on the S1b arms.

    The judge already returns `followed: [b, b, b]` per turn and `y_graded` is
    their mean -- this reads the vector the scalar fit averaged away. The `u`
    slot follows `one_step_transitions` exactly: the action credited to a
    transition is the reminder injected at the NEXT turn, because it lands
    before that turn's reply is generated. A pair touching an unparsed
    constraint is dropped whole rather than imputed, the same rule the scalar
    path uses, so the two models see the same dialogue turns.
    """

    rows = []
    for row in readout["rows"]:
        if row["branch"] not in S1B_ARMS or row["item_id"] in excluded:
            continue
        rows.append(row)
    by_traj: dict[str, list[dict]] = {}
    for row in rows:
        by_traj.setdefault(row["trajectory_id"], []).append(row)

    usable = lambda r: r.get("followed") is not None and all(b is not None for b in r["followed"])
    out = []
    for traj_rows in by_traj.values():
        ordered = sorted(traj_rows, key=lambda r: r["turn"])
        for now, nxt in zip(ordered, ordered[1:]):
            if not usable(now) or not usable(nxt):
                continue
            out.append({
                "b_now": tuple(bool(b) for b in now["followed"]),
                "b_next": tuple(bool(b) for b in nxt["followed"]),
                "u": int(nxt["u_remind"]), "turn": int(nxt["turn"]),
                "item_id": now["item_id"], "trajectory_id": now["trajectory_id"],
            })
    return out


def fit_constraint_kernel(transitions: list[dict], seed: int) -> dict:
    """`P(b_(t+1)=1 | b_t, u_t)` per constraint, pooled over the three slots.

    Four cells, counted rather than regressed: the state is binary, so the
    conditional mean IS the whole model and a link function would only add an
    assumption. Pooling over slots is the primary spec because the three
    constraints of an item are interchangeable by construction (the judge
    grades them in the order the item lists them, not by kind); the per-slot
    fit is reported beside it so a slot that behaves differently cannot hide
    inside the pool.

    The reported contrast is the binary analogue of G-S2-5's `d`:
    `gain(0) - gain(1)`, where `gain(b) = P(1|b,u=1) - P(1|b,u=0)`. It is
    bootstrapped over ITEMS for the same reason `B` is -- a dialogue is the
    independent unit, and three seeds of one item replay the same constraints.
    """

    items = sorted({t["item_id"] for t in transitions})
    # counts[item][b][u] = (n, successes)
    counts = np.zeros((len(items), 2, 2, 2))  # item, b_now, u, (n, successes)
    index = {item: i for i, item in enumerate(items)}
    per_slot = np.zeros((3, 2, 2, 2))
    for t in transitions:
        i = index[t["item_id"]]
        for slot, (now, nxt) in enumerate(zip(t["b_now"], t["b_next"])):
            counts[i, int(now), t["u"], 0] += 1
            counts[i, int(now), t["u"], 1] += int(nxt)
            per_slot[slot, int(now), t["u"], 0] += 1
            per_slot[slot, int(now), t["u"], 1] += int(nxt)

    def rates(block: np.ndarray) -> np.ndarray:
        n, s = block[..., 0], block[..., 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    pooled_counts = counts.sum(axis=0)
    p = rates(pooled_counts)
    gain = p[:, 1] - p[:, 0]
    contrast = float(gain[0] - gain[1])

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOTSTRAP_INTERACTION):
        chosen = rng.choice(len(items), size=len(items), replace=True)
        block = counts[chosen].sum(axis=0)
        q = rates(block)
        g = q[:, 1] - q[:, 0]
        draws.append(float(g[0] - g[1]))
    draws = np.array(draws)
    finite = draws[np.isfinite(draws)]
    # an empty cell is a real possibility on a thin readout: say so instead of
    # bootstrapping a NaN into a number nobody can read
    if len(finite) < 2 or not np.isfinite(contrast):
        lo = hi = float("nan")
        sd = float("nan")
        mde = float("nan")
    else:
        lo, hi = np.percentile(finite, [2.5, 97.5])
        sd = float(np.std(finite, ddof=1))
        mde = POWER_Z * sd

    slot_rates = rates(per_slot)
    cells = {}
    for b in (0, 1):
        for u in (0, 1):
            n = int(pooled_counts[b, u, 0])
            cells[f"b={b},u={u}"] = {"n": n, "p_next_1": float(p[b, u]),
                                     "sparse": n < BINARY_SPARSE_CELL}
    return {
        "is_a_gate": False,
        "spec": "pooled over the three constraint slots; counts, not a regression",
        "cells": cells,
        "gain_when_violated": float(gain[0]), "gain_when_kept": float(gain[1]),
        "state_dependence_of_the_gain": contrast,
        "ci": [float(lo), float(hi)], "bootstrap_sd": sd, "mde_at_80pct": mde,
        "effect_over_mde": abs(contrast) / mde if mde and np.isfinite(mde) else None,
        "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0),
        "underdetermined": [k for k, v in
                            {f"b={b},u={u}": pooled_counts[b, u, 0] for b in (0, 1) for u in (0, 1)}.items()
                            if v == 0],
        "per_slot": {f"slot_{s}": {f"b={b},u={u}": {"n": int(per_slot[s, b, u, 0]),
                                                    "p_next_1": float(slot_rates[s, b, u])}
                                   for b in (0, 1) for u in (0, 1)} for s in range(3)},
        "reading": "the scalar bilinear `d` and this contrast ask the same question in two state "
                   "spaces. Here the ceiling is explicit: a constraint already kept can only be "
                   "held or lost, so `gain(1)` is bounded by `1 - P(1|1,0)` no matter how strong "
                   "the actuator is -- the censoring the scalar fit smoothed away.",
    }


def count_kernel(transitions: list[dict]) -> dict:
    """`P(m_(t+1) | m_t, u_t)` on the count state `m = #followed`, empirical.

    This is the kernel the planner runs on, and it is estimated WITHOUT the
    conditional-independence assumption the per-constraint kernel would need:
    the three constraints of one turn share a reply, so a reminder that fixes
    one may well fix all three. `independence_deviation` below reports what
    that assumption would have cost, rather than making it silently.

    The objective is `y = m / 3`, the same readout every gate on this line is
    reported on -- not a new quantity.
    """

    n = np.zeros((2, 4, 4))
    for t in transitions:
        m, m_next = sum(t["b_now"]), sum(t["b_next"])
        n[t["u"], m, m_next] += 1
    totals = n.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = np.where(totals[..., None] > 0, n / np.maximum(totals[..., None], 1), np.nan)
    return {"kernel": kernel, "counts": n, "totals": totals}


def product_count_kernel(constraint_cells: dict) -> np.ndarray:
    """The count kernel conditional independence WOULD imply, for comparison."""

    p = np.zeros((2, 2))  # b_now, u -> P(next=1)
    for b in (0, 1):
        for u in (0, 1):
            p[b, u] = constraint_cells[f"b={b},u={u}"]["p_next_1"]
    kernel = np.zeros((2, 4, 4))
    for u in (0, 1):
        for m in range(4):
            for kept in range(m + 1):  # of the m already-kept constraints
                for gained in range(3 - m + 1):  # of the (3-m) violated ones
                    prob = (math.comb(m, kept) * p[1, u] ** kept * (1 - p[1, u]) ** (m - kept)
                            * math.comb(3 - m, gained) * p[0, u] ** gained
                            * (1 - p[0, u]) ** (3 - m - gained))
                    kernel[u, m, kept + gained] += prob
    return kernel


def kernel_diagnostics(counts: dict, product: np.ndarray) -> dict:
    """What the planner's kernel looks like, and where it is thin.

    `action_value_interaction` is the count-space reading of the same term
    G-S2-5 measured: regress `E[m_(t+1)] / 3` on `m/3`, `u` and their product
    over the occupied cells, weighted by cell counts. A scalar-linear operator
    forces that product term to 0 by construction; if the fitted kernel gives
    it a large negative value, the state dependence the linear fit could not
    represent is in the data rather than in the model class.
    """

    kernel, n, totals = counts["kernel"], counts["counts"], counts["totals"]
    grid = np.arange(4) / 3.0
    rows, weights, targets = [], [], []
    cells = {}
    for u in (0, 1):
        for m in range(4):
            total = float(totals[u, m])
            expected = float(np.nansum(kernel[u, m] * grid)) if total else float("nan")
            cells[f"m={m},u={u}"] = {
                "n": int(total), "sparse": total < BINARY_SPARSE_CELL,
                "expected_next_y": expected,
                "distribution": [float(v) for v in kernel[u, m]] if total else None,
            }
            if total:
                rows.append([1.0, grid[m], float(u), grid[m] * u])
                weights.append(total)
                targets.append(expected)
    design = np.array(rows) * np.sqrt(np.array(weights))[:, None]
    beta = np.linalg.lstsq(design, np.array(targets) * np.sqrt(np.array(weights)), rcond=None)[0]

    occupied = totals > 0
    deviation = np.abs(kernel - product)[occupied[..., None].repeat(4, axis=-1)]
    deviation = deviation[np.isfinite(deviation)]
    return {
        "is_a_gate": False,
        "cells": cells,
        "action_value_interaction": float(beta[3]),
        "y_coefficient": float(beta[1]), "u_coefficient": float(beta[2]),
        "independence_deviation_mean_abs": float(np.mean(deviation)) if len(deviation) else None,
        "independence_deviation_max_abs": float(np.max(deviation)) if len(deviation) else None,
        "sparse_cells": [k for k, v in cells.items() if v["sparse"]],
        "reading": "the interaction term is the count-space analogue of G-S2-5's `d`. The "
                   "independence deviation is what a per-constraint product kernel would have "
                   "got wrong; the planner does not use the product kernel, this only prices the "
                   "assumption. Sparse cells are reported because the m=0 state is rare on this "
                   "readout -- a policy that depends on it is extrapolating.",
    }


def _propagate(dist: np.ndarray, kernel: np.ndarray, u: int) -> np.ndarray:
    return dist @ kernel[u]


def _binary_objective(start: np.ndarray, schedule: tuple[int, ...], start_turn: int,
                      late_from: int, kernel: np.ndarray) -> float:
    """Expected late-window mean `y` of a FIXED schedule, propagated exactly.

    An open-loop schedule does not observe the state, so the expectation is
    computed by pushing the distribution through the kernel rather than by
    simulating -- no Monte-Carlo error enters the enumeration.
    """

    dist, kept, grid = start.copy(), [], np.arange(4) / 3.0
    for offset, u in enumerate(schedule):
        dist = _propagate(dist, kernel, u)
        if start_turn + offset >= late_from:
            kept.append(float(dist @ grid))
    return float(np.mean(kept)) if kept else float(dist @ grid)


def binary_schedule_separability(kernels: dict, starts: dict, budgets: tuple[int, ...],
                                 start_turn: int, n_turns: int, late_from: int) -> dict:
    """The same enumeration `schedule_separability` runs, on the count kernel.

    Deliberately the same question and the same output shape, so the two model
    classes are compared rather than two checks. It carries the same warning
    too: enumerating FIXED schedules answers "does the best open-loop plan
    depend on where you start", which is necessary for a closed loop to earn
    its arm but not sufficient -- `simulate_binary_s3_gap` asks the sufficient
    version, whether reacting to the realised state beats the best plan made
    in advance.
    """

    horizon = n_turns - start_turn + 1
    out = {}
    for name, kernel in kernels.items():
        per_budget = {}
        for k in budgets:
            if k > horizon:
                continue
            chosen = {}
            for item, start in starts.items():
                best, best_value = None, -np.inf
                for turns in itertools.combinations(range(horizon), k):
                    schedule = tuple(1 if i in turns else 0 for i in range(horizon))
                    value = _binary_objective(start, schedule, start_turn, late_from, kernel)
                    if value > best_value:
                        best, best_value = turns, value
                chosen[item] = tuple(sorted(horizon - 1 - t for t in best))
            counts: dict[tuple, int] = {}
            for value in chosen.values():
                counts[value] = counts.get(value, 0) + 1
            modal, modal_n = max(counts.items(), key=lambda kv: kv[1])
            per_budget[k] = {
                "n_items": len(chosen), "n_distinct_schedules": len(counts),
                "modal_schedule_from_end": list(modal), "modal_share": modal_n / len(chosen),
                "schedule_counts": {str(list(kk)): v for kk, v in sorted(counts.items(), key=lambda kv: -kv[1])},
            }
        out[name] = per_budget
    distinct = {name: {k: v["n_distinct_schedules"] for k, v in per.items()}
                for name, per in out.items()}
    return {
        "is_a_gate": False, "start_turn": start_turn, "horizon": horizon,
        "objective": f"expected mean y over t{late_from}..t{n_turns}",
        "by_operator": out, "distinct_schedules": distinct,
        "open_loop_optimum_is_state_dependent": any(
            v > 1 for per in distinct.values() for v in per.values()),
    }


def simulate_binary_s3_gap(kernel: np.ndarray, starts: dict, budget: int, start_turn: int,
                           n_turns: int, late_from: int, n_sims: int, seed: int) -> dict:
    """`koopman_mpc` against the best fixed schedule, on the count kernel.

    THE number option (a) was bought to produce. The planner is exact here --
    the state space is four points, so the DP is the optimal feedback policy
    rather than an approximation of it, and no discretisation grid can bias
    the comparison (the failure the scalar simulation had to pin with a
    zero-noise test).

    Common random numbers: one uniform per (item, sim, offset) drives the
    transition under EVERY policy, so the gap is a paired difference. The
    stochasticity is the model's own -- a binary constraint either survives
    the turn or does not -- rather than bootstrapped residuals bolted onto a
    deterministic operator.
    """

    horizon = n_turns - start_turn + 1
    grid = np.arange(4) / 3.0

    value = np.zeros((horizon + 1, budget + 1, 4))
    act = np.zeros((horizon, budget + 1, 4), dtype=bool)
    for offset in range(horizon - 1, -1, -1):
        counts_toward = 1.0 if start_turn + offset >= late_from else 0.0
        for k in range(budget + 1):
            for m in range(4):
                rewards = counts_toward * grid + value[offset + 1, k]
                best = float(kernel[0, m] @ rewards)
                take = False
                if k > 0:
                    alt = float(kernel[1, m] @ (counts_toward * grid + value[offset + 1, k - 1]))
                    if alt > best:
                        best, take = alt, True
                value[offset, k, m] = best
                act[offset, k, m] = take

    cdf = np.cumsum(kernel, axis=2)

    def run(policy, m0: int, draws: np.ndarray) -> float:
        m, k, kept = m0, budget, []
        for offset in range(horizon):
            u = int(act[offset, k, m]) if policy is None else policy[offset]
            k -= u
            m = int(np.searchsorted(cdf[u, m], draws[offset]))
            if start_turn + offset >= late_from:
                kept.append(grid[m])
        return float(np.mean(kept))

    mean_start = np.mean(np.vstack(list(starts.values())), axis=0)
    best_fixed, best_value = None, -np.inf
    for turns in itertools.combinations(range(horizon), budget):
        schedule = tuple(1 if i in turns else 0 for i in range(horizon))
        v = _binary_objective(mean_start, schedule, start_turn, late_from, kernel)
        if v > best_value:
            best_fixed, best_value = schedule, v

    rng = np.random.default_rng(seed)
    gaps_vs_fixed, gaps_vs_random, reminders_used = [], [], []
    for item, start in starts.items():
        for _ in range(n_sims):
            m0 = int(rng.choice(4, p=start))
            draws = rng.random(horizon)
            turns = tuple(sorted(rng.choice(horizon, size=budget, replace=False).tolist()))
            random_schedule = tuple(1 if i in turns else 0 for i in range(horizon))
            mpc = run(None, m0, draws)
            gaps_vs_fixed.append(mpc - run(best_fixed, m0, draws))
            gaps_vs_random.append(mpc - run(random_schedule, m0, draws))
    return {
        "is_a_gate": False, "budget": budget, "n_sims_per_item": n_sims,
        "planner": "exact DP over (turn, remaining budget, m) -- four states, no grid",
        "best_fixed_schedule_from_end": [horizon - 1 - i for i, u in enumerate(best_fixed) if u],
        "mpc_minus_best_fixed": float(np.mean(gaps_vs_fixed)),
        "mpc_minus_best_fixed_sd": float(np.std(gaps_vs_fixed, ddof=1)),
        "mpc_minus_random": float(np.mean(gaps_vs_random)),
        "paired_on": "one uniform draw per (item, sim, turn), shared by every policy",
        "caveat": "a simulation on a kernel estimated from 40 items; it inherits every way that "
                  "kernel is wrong, and the m=0 row is thin. Use it one-sidedly, as with the "
                  "scalar version: a gap far below the design's MDE means S3 as specified cannot "
                  "resolve what its own planner expects.",
    }


def binary_state_model(readout: dict, excluded: list[str], all_rows: list[dict], budgets: tuple[int, ...],
                       start_turn: int, n_turns: int, late_from: int, simulate_budget: int,
                       n_sims: int, mde: float, seed: int) -> dict:
    """Option (a), signed 2026-09-11: refit the state the judge actually
    returns and re-run the separability check that stopped S3.

    PRE-REGISTERED before the numbers were seen (screening section 10 item 11):
    the verdict is read off `mpc_minus_best_fixed` against the SAME MDE the
    scalar check used -- 0.0690, G-S2-4's design resolution signed 2026-09-10.
    Above it, the degeneracy was the scalar model class and the `koopman_mpc`
    arm has something to do; below it, a model class that represents the
    ceiling explicitly still wants one plan for everyone, which makes the
    degeneracy a property of the task and not of the fit. No third reading,
    and no re-tuning of the bar after seeing the gap.
    """

    transitions = binary_transitions(readout, excluded)
    if not transitions:
        return {"is_a_gate": False, "verdict": None,
                "reason": "no usable per-constraint verdicts in this readout"}
    constraint = fit_constraint_kernel(transitions, seed)
    counts = count_kernel(transitions)
    diagnostics = kernel_diagnostics(counts, product_count_kernel(constraint["cells"]))

    first_turn: dict[str, list[int]] = {}
    for row in all_rows:
        if row["turn"] != 1 or row.get("followed") is None or any(b is None for b in row["followed"]):
            continue
        first_turn.setdefault(row["item_id"], []).append(sum(1 for b in row["followed"] if b))
    starts = {}
    for item, values in sorted(first_turn.items()):
        dist = np.zeros(4)
        for m in values:
            dist[m] += 1
        starts[item] = dist / dist.sum()

    if not starts:
        return {"is_a_gate": False, "verdict": None,
                "reason": "no turn-1 row carries a parsed constraint vector: there is no start state"}

    kernel = counts["kernel"]
    if np.isnan(kernel[counts["totals"] > 0]).any():
        raise SystemExit("count kernel has a NaN in an occupied cell")
    planner_kernel = np.nan_to_num(kernel, nan=0.0)
    for u in (0, 1):
        for m in range(4):
            if counts["totals"][u, m] == 0:  # unvisited: hold the state, never invent a transition
                planner_kernel[u, m] = 0.0
                planner_kernel[u, m, m] = 1.0

    separability = binary_schedule_separability(
        {"binary_count": planner_kernel}, starts, budgets, start_turn, n_turns, late_from)
    pure = {f"m={m}": np.eye(4)[m] for m in range(4)}
    sweep = binary_schedule_separability(
        {"binary_count": planner_kernel}, pure, budgets, start_turn, n_turns, late_from)
    separability["state_sweep"] = {
        "states": list(pure),
        "distinct_schedules": sweep["distinct_schedules"]["binary_count"],
        "by_budget": sweep["by_operator"]["binary_count"],
        "reading": "the count state has exactly four points, so this sweep is the WHOLE state "
                   "space, not a sample of it",
    }
    simulation = (simulate_binary_s3_gap(planner_kernel, starts, simulate_budget, start_turn,
                                         n_turns, late_from, n_sims, seed) if n_sims else None)

    gap = simulation["mpc_minus_best_fixed"] if simulation else None
    verdict = None
    if gap is not None:
        verdict = "CLOSED_LOOP_HAS_HEADROOM" if gap >= mde else "DEGENERACY_SURVIVES_THE_MODEL_CLASS"
    return {
        "is_a_gate": False,
        "signed": "2026-09-11, option (a) (screening section 10 item 11)",
        "state": "per-constraint binary `followed`, collapsed to the count m in {0,1,2,3}; y = m/3",
        "n_transitions": len(transitions),
        "n_items": len({t["item_id"] for t in transitions}),
        "constraint_kernel": constraint,
        "count_kernel_diagnostics": diagnostics,
        "schedule_separability": separability,
        "s3_gap_simulation": simulation,
        # the planner's own inputs travel with the report: a schedule nobody can
        # recompute from the artifact is a claim, not a result
        "planner_kernel": planner_kernel.tolist(),
        "starting_distributions": {item: [float(v) for v in dist] for item, dist in starts.items()},
        "mde_used": mde,
        "verdict": verdict,
        "decision_rule": (
            "pre-registered: MPC minus best fixed schedule at or above the design MDE "
            f"({mde:.4f}) means the scalar model class was the problem and the koopman_mpc arm "
            "has something to do; below it, a model class that represents the ceiling explicitly "
            "still wants one plan for everyone and the degeneracy belongs to the task"),
    }


THRESHOLD_MAX_REMINDERS = 6  # open-loop enumeration ceiling; every frontier point above it is absent
THRESHOLD_LAMBDAS = tuple(np.round(np.linspace(0.0, 0.30, 61), 5))


def _late_reward(theta: int, start_turn: int, n_turns: int, late_from: int) -> tuple[np.ndarray, int]:
    """Per-turn reward vector over m, and the number of late turns it averages."""

    keep = np.array([1.0 if m >= theta else 0.0 for m in range(4)])
    n_late = sum(1 for turn in range(start_turn, n_turns + 1) if turn >= late_from)
    return keep / max(n_late, 1), n_late


def _binary_objective_threshold(kernel: np.ndarray, starts: dict, theta: int,
                                schedule: tuple[int, ...], start_turn: int, n_turns: int,
                                late_from: int) -> float:
    """Service level of one explicit schedule, averaged over the items' own
    starting distributions."""

    reward, _ = _late_reward(theta, start_turn, n_turns, late_from)
    total = 0.0
    for start in starts.values():
        dist, value = start.copy(), 0.0
        for offset, u in enumerate(schedule):
            dist = _propagate(dist, kernel, u)
            if start_turn + offset >= late_from:
                value += float(dist @ reward)
        total += value
    return total / len(starts)


def fixed_schedule_frontier(kernel: np.ndarray, starts: dict, theta: int, start_turn: int,
                            n_turns: int, late_from: int, max_k: int) -> dict:
    """Open loop: the best FIXED schedule at each number of reminders.

    Every schedule's value is linear in the starting distribution, so the
    backward recursion is run once for all schedules at once and each item is
    then a dot product. That is what makes an exhaustive enumeration up to
    `max_k` reminders affordable -- and exhaustive matters, because a fixed
    schedule chosen badly would hand the closed loop a win it did not earn.

    Two open-loop competitors are reported, not one. `shared` is the single
    schedule the whole arm would run, which is what S3 actually compares
    against. `per_item_oracle` lets every item pick its own best schedule in
    advance -- it cannot be run without knowing each item's starting state, so
    it is not an arm, but it is the strict competitor: a closed-loop gap that
    survives it is not just the closed loop rediscovering that items differ.
    """

    horizon = n_turns - start_turn + 1
    reward, _ = _late_reward(theta, start_turn, n_turns, late_from)
    schedules = []
    for k in range(min(max_k, horizon) + 1):
        for turns in itertools.combinations(range(horizon), k):
            schedules.append((k, tuple(1 if i in turns else 0 for i in range(horizon))))
    actions = np.array([s for _, s in schedules], dtype=bool)  # (S, horizon)

    w = np.zeros((len(schedules), 4))
    for offset in range(horizon - 1, -1, -1):
        counts = reward if start_turn + offset >= late_from else np.zeros(4)
        future = counts + w
        under_0 = future @ kernel[0].T
        under_1 = future @ kernel[1].T
        w = np.where(actions[:, offset][:, None], under_1, under_0)

    ks = np.array([k for k, _ in schedules])
    mean_start = np.mean(np.vstack(list(starts.values())), axis=0)
    shared_values = w @ mean_start
    per_item = np.vstack([w @ start for start in starts.values()])  # (items, S)

    shared, oracle = {}, {}
    for k in range(min(max_k, horizon) + 1):
        mask = ks == k
        best = int(np.argmax(np.where(mask, shared_values, -np.inf)))
        shared[k] = {
            "service_level": float(shared_values[best]),
            "schedule_from_end": [horizon - 1 - i for i, u in enumerate(schedules[best][1]) if u],
        }
        oracle[k] = {"service_level": float(np.mean(np.max(per_item[:, mask], axis=1)))}
    return {"shared": shared, "per_item_oracle": oracle, "max_k": max_k,
            "n_schedules_enumerated": len(schedules)}


def _upper_envelope(points: list[tuple[float, float]]):
    """Concave upper envelope of (cost, value) points, as a lookup function.

    A randomised mixture of two fixed schedules is itself an open-loop policy
    and achieves the chord between them, so the honest open-loop competitor at
    a fractional cost is the envelope, not the lower of the two integer points.
    Rounding the closed loop's cost down to the next integer schedule would
    manufacture part of the gap.
    """

    kept: list[tuple[float, float]] = []
    best = -np.inf
    for cost, value in sorted(points):
        if value <= best:  # a cheaper schedule is already at least this good
            continue
        kept.append((cost, value))
        best = value
    hull: list[tuple[float, float]] = []
    for point in kept:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # concave: slopes must fall. If the middle point sits on or below the
            # chord, a mixture of its neighbours already dominates it.
            if (y2 - y1) * (point[0] - x1) <= (point[1] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(point)

    def at(cost: float) -> float:
        if cost <= hull[0][0]:
            return hull[0][1]
        if cost >= hull[-1][0]:
            return hull[-1][1]
        for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
            if x1 <= cost <= x2:
                return y1 + (y2 - y1) * (cost - x1) / (x2 - x1)
        return hull[-1][1]

    return at, hull


def threshold_feedback_frontier(kernel: np.ndarray, starts: dict, theta: int, start_turn: int,
                                n_turns: int, late_from: int, lambdas) -> list[dict]:
    """Closed loop under objective (b): keep `m >= theta`, pay `lambda` per reminder.

    No budget to spend down -- that is the whole point of changing the
    objective. Under a fixed budget the reminders must all be used and only
    their placement is free, which is why timing decoupled from state; here the
    controller decides WHETHER to spend at all, and "do I still need one" is a
    function of the state by construction of the task rather than of the
    planner.

    Each lambda gives one deterministic policy; its expected cost and service
    level are both computed by propagating the state distribution exactly, so
    no simulation noise enters the frontier.
    """

    horizon = n_turns - start_turn + 1
    reward, _ = _late_reward(theta, start_turn, n_turns, late_from)
    mean_start = np.mean(np.vstack(list(starts.values())), axis=0)

    out = []
    for lam in lambdas:
        value = np.zeros((horizon + 1, 4))
        act = np.zeros((horizon, 4), dtype=bool)
        for offset in range(horizon - 1, -1, -1):
            counts = reward if start_turn + offset >= late_from else np.zeros(4)
            future = counts + value[offset + 1]
            for m in range(4):
                hold = float(kernel[0, m] @ future)
                remind = float(kernel[1, m] @ future) - lam
                take = remind > hold
                value[offset, m] = remind if take else hold
                act[offset, m] = take
        dist, service, cost = mean_start.copy(), 0.0, 0.0
        for offset in range(horizon):
            u = act[offset].astype(float)
            cost += float(dist @ u)
            moved = np.zeros(4)
            for m in range(4):
                moved += dist[m] * kernel[int(act[offset, m]), m]
            dist = moved
            if start_turn + offset >= late_from:
                service += float(dist @ reward)
        out.append({"lambda": float(lam), "expected_reminders": cost, "service_level": service,
                    "acts_on_state": bool(len({tuple(act[o]) for o in range(horizon)}) > 1
                                          or any(len(set(act[o])) > 1 for o in range(horizon)))})
    return out


def threshold_objective(kernel: np.ndarray, starts: dict, theta: int, start_turn: int, n_turns: int,
                        late_from: int, max_k: int, lambdas, mde: float,
                        measured_full_dose: float | None = None) -> dict:
    """Option (b) on paper, before any GPU: at EQUAL EXPECTED COST, how much
    more of the late window does a state-feedback policy hold above the
    threshold than the best fixed schedule?

    Matched cost is the whole design of this comparison. "Keep m >= theta"
    rewards acting when the state has slipped, so a closed loop that simply
    spent MORE reminders would win by definition; pricing every reminder and
    reading the gap at the same expected spend removes that. The open-loop
    competitor is the exhaustive best fixed schedule, mixed along its own
    upper envelope, not a schedule chosen for it.
    """

    open_loop = fixed_schedule_frontier(kernel, starts, theta, start_turn, n_turns, late_from, max_k)
    closed = threshold_feedback_frontier(kernel, starts, theta, start_turn, n_turns, late_from, lambdas)
    shared_at, shared_hull = _upper_envelope(
        [(float(k), v["service_level"]) for k, v in open_loop["shared"].items()])
    oracle_at, _ = _upper_envelope(
        [(float(k), v["service_level"]) for k, v in open_loop["per_item_oracle"].items()])

    points = []
    for point in closed:
        if point["expected_reminders"] > max_k:  # off the enumerated frontier: no honest competitor
            continue
        points.append({**point,
                       "open_loop_shared_at_same_cost": shared_at(point["expected_reminders"]),
                       "open_loop_oracle_at_same_cost": oracle_at(point["expected_reminders"]),
                       "gap_vs_shared": point["service_level"] - shared_at(point["expected_reminders"]),
                       "gap_vs_oracle": point["service_level"] - oracle_at(point["expected_reminders"])})
    # Model check, the one the 2026-09-10 note made mandatory for this operator:
    # what the model says the FULL dose buys, against what the arms measured.
    horizon = n_turns - start_turn + 1
    always = _binary_objective_threshold(kernel, starts, theta, (1,) * horizon, start_turn,
                                         n_turns, late_from)
    never = _binary_objective_threshold(kernel, starts, theta, (0,) * horizon, start_turn,
                                        n_turns, late_from)
    model_full_dose = always - never

    best = max(points, key=lambda p: p["gap_vs_shared"]) if points else None
    verdict = None
    if best is not None:
        verdict = ("WORTH_THE_GPU" if best["gap_vs_shared"] >= mde
                   else "DEGENERACY_SURVIVES_THE_OBJECTIVE")
    return {
        "is_a_gate": False,
        "signed": "2026-09-11, option (b) on paper (screening section 10 item 12)",
        "primary": f"share of turns t{late_from}..t{n_turns} with m >= {theta}",
        "threshold": theta, "max_reminders_enumerated": max_k,
        "n_schedules_enumerated": open_loop["n_schedules_enumerated"],
        "open_loop_frontier": open_loop,
        "open_loop_shared_hull": [list(p) for p in shared_hull],
        "frontier": points,
        "best_matched_cost_point": best,
        "mde_used": mde,
        "model_full_dose_contrast": model_full_dose,
        "measured_full_dose_contrast": measured_full_dose,
        "model_under_predicts_full_dose_by": (measured_full_dose / model_full_dose
                                              if measured_full_dose and model_full_dose else None),
        "sensitivity": "the kernel under-predicts what the full dose buys, as the scalar operator "
                       "did (2.8x on the endpoint). Scaling the equal-cost gap by that same factor "
                       "is not an estimate -- it is the crudest way to ask whether the verdict "
                       "could survive the model being wrong by that much, and it is reported so "
                       "the answer is visible rather than assumed.",

        "verdict": verdict,
        "decision_rule": (
            "pre-registered: the largest equal-cost gap against the SHARED best fixed schedule, "
            f"read against this objective's own MDE at the current design ({mde:.4f}). At or "
            "above it, changing the objective buys a closed loop worth running; below it, the "
            "degeneracy survives the objective change too and no arm table rewrite is warranted"),
        "why_matched_cost": "the threshold objective rewards acting when the state has slipped, so "
                            "an unpriced closed loop would win by spending more. Every reminder is "
                            "priced and the gap is read at equal expected spend.",
    }


def threshold_sizing(pilot, all_rows: list[dict], theta: int, late_from: int, n_turns: int,
                     seeds: int, fraction: float, seed: int) -> dict:
    """What objective (b) would cost in design terms: the paired variance and
    MDE of the NEW primary, computed the same way G-S2-4 computed the old one.

    Reported, not signed. The 2026-09-10 note is explicit that the arm table,
    the primary, the MDE and G-S2-4's 50% clause move together; this produces
    the numbers that rewrite would need, and leaves the ruling to the user.
    """

    rows = []
    for row in all_rows:
        followed = row.get("followed")
        usable = followed is not None and all(b is not None for b in followed)
        rows.append({**row,
                     "y_graded": float(sum(1 for b in followed if b) >= theta) if usable else None})
    late = range(late_from, n_turns + 1)
    contrast = pilot.per_item_contrast(rows, "constant_remind", "zero_control", late)
    if len(contrast) < 2:
        return {"is_a_gate": False, "primary": f"share of turns t{late_from}..t{n_turns} with m >= {theta}",
                "status": "NOT COMPUTABLE: fewer than two items carry both full-dose arms with a "
                          "parsed constraint vector"}
    components = pilot.variance_components(contrast)
    full_dose = pilot.bootstrap_contrast(contrast, seed)
    var = components["sd_between_items"] ** 2 + components["sd_within_item_across_seeds"] ** 2 / seeds
    n_items = len(contrast)
    mde = POWER_Z * math.sqrt(var / n_items) if n_items else float("nan")
    target = fraction * full_dose["point"]
    needed = math.ceil(POWER_Z ** 2 * var / target ** 2) if target else float("inf")
    return {
        "is_a_gate": False,
        "primary": f"share of turns t{late_from}..t{n_turns} with m >= {theta}",
        "full_dose_contrast": full_dose["point"], "full_dose_ci": full_dose.get("ci"),
        "sd_between_items": components["sd_between_items"],
        "sd_within_item_across_seeds": components["sd_within_item_across_seeds"],
        "n_items": n_items, "seeds": seeds,
        "mde_at_current_design": mde,
        "n_items_for_half_the_full_dose": needed,
        "trajectories_per_arm_for_half": needed * seeds,
        "within_ceiling": needed * seeds <= S3_TRAJECTORIES_PER_ARM_CEILING,
        "status": "REPORTED, NOT SIGNED: adopting it means re-signing the arm table, the primary, "
                  "the MDE and G-S2-4's 50% clause together (results archive, 2026-09-10 note 3)",
    }


def print_threshold_section(th: dict, sizing: dict) -> None:
    print(f"\n阈值型目标（选项 b 的纸面版，主量 = {th['primary']}）")
    print(f"  新主量的设计代价（只报不签）: 满剂量差 {sizing['full_dose_contrast']:+.4f}，"
          f"配对 sd 题间 {sizing['sd_between_items']:.4f} / 题内 {sizing['sd_within_item_across_seeds']:.4f} → "
          f"本设计 MDE {sizing['mde_at_current_design']:.4f}；"
          f"分辨满剂量一半需 {sizing['n_items_for_half_the_full_dose']} 题 × {sizing['seeds']} seed "
          f"= {sizing['trajectories_per_arm_for_half']} 条/臂"
          f"（{'在' if sizing['within_ceiling'] else '超出'} 150 停止线）")
    print(f"  枚举 {th['n_schedules_enumerated']} 个固定日程（≤{th['max_reminders_enumerated']} 次提醒），"
          f"开环上包络 {th['open_loop_shared_hull']}")
    for p in th["frontier"][::10]:
        print(f"    λ={p['lambda']:.3f}  闭环代价 {p['expected_reminders']:.2f} 次、服从率 "
              f"{p['service_level']:.4f}；等代价开环（共享/按题预知）"
              f"{p['open_loop_shared_at_same_cost']:.4f}/{p['open_loop_oracle_at_same_cost']:.4f}；"
              f"差 {p['gap_vs_shared']:+.4f}/{p['gap_vs_oracle']:+.4f}")
    print(f"  模型 vs 实测的满剂量: 模型 {th['model_full_dose_contrast']:+.4f}，"
          f"实测 {th['measured_full_dose_contrast']:+.4f}"
          + (f"，低估 {th['model_under_predicts_full_dose_by']:.1f}×"
             if th["model_under_predicts_full_dose_by"] else ""))
    best = th["best_matched_cost_point"]
    if best:
        print(f"  最大等代价差: {best['gap_vs_shared']:+.4f}（λ={best['lambda']:.3f}，"
              f"代价 {best['expected_reminders']:.2f} 次），对按题预知的开环 {best['gap_vs_oracle']:+.4f}；"
              f"本目标自己的 MDE {th['mde_used']:.4f}")
    print(f"  裁定（预注册）: {th['verdict']}")


def print_binary_section(bs: dict) -> None:
    if bs.get("constraint_kernel") is None:
        print(f"\n二值约束状态模型（选项 a）: 跳过 — {bs['reason']}")
        return
    ck, kd = bs["constraint_kernel"], bs["count_kernel_diagnostics"]
    print(f"\n二值约束状态模型（选项 a，{bs['n_transitions']} 个转移，{bs['n_items']} 题）")
    print("  P(next=1 | b, u): " + "  ".join(
        f"{k} {v['p_next_1']:.4f} (n={v['n']}{', 稀疏' if v['sparse'] else ''})"
        for k, v in ck["cells"].items()))
    print(f"  一次提醒的增益: 违反时 {ck['gain_when_violated']:+.4f}，已守住时 "
          f"{ck['gain_when_kept']:+.4f}；状态依赖 {ck['state_dependence_of_the_gain']:+.4f} "
          f"CI [{ck['ci'][0]:+.4f}, {ck['ci'][1]:+.4f}] MDE {ck['mde_at_80pct']:.4f}")
    print(f"  计数核: 交互项 {kd['action_value_interaction']:+.4f}（标量线性算子恒为 0）；"
          f"独立性偏差 均值 {kd['independence_deviation_mean_abs']} / 最大 "
          f"{kd['independence_deviation_max_abs']}；稀疏格 {kd['sparse_cells'] or '无'}")
    bsep = bs["schedule_separability"]
    print(f"  日程可分性: 按题 {bsep['distinct_schedules']['binary_count']}，"
          f"全状态空间（m=0..3）{bsep['state_sweep']['distinct_schedules']}")
    bsim = bs["s3_gap_simulation"]
    if bsim:
        print(f"  S3 臂间差模拟（预算 {bsim['budget']}，精确 DP）: MPC − 最优固定日程 "
              f"{bsim['mpc_minus_best_fixed']:+.4f}（sd {bsim['mpc_minus_best_fixed_sd']:.4f}），"
              f"MPC − 随机 {bsim['mpc_minus_random']:+.4f}；本设计 MDE {bs['mde_used']:.4f}")
    print(f"  裁定（预注册）: {bs['verdict']}")


def _sibling_module(name: str):
    """Import the sibling script that already defines a criterion rather than
    restating it. The three nulls come from the `defense` line's fit script,
    where G-K2-1 used them: two copies of a null definition drift apart, and
    the whole point of G-S2-1 is that it is the same comparison K2 ran."""

    path = pathlib.Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--independent-readout", type=pathlib.Path, required=True,
                   help="score_sequor_trajectories.py report with judge_kind=independent.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--nu", type=int, default=1)
    p.add_argument("--mu", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--n-folds", type=int, default=20, help="Item-disjoint folds for G-S2-1.")
    p.add_argument("--folds-to-pass", type=int, default=14, help="Plan section 5: 14 of 20.")
    p.add_argument("--controllability-horizon", type=int, default=5)
    p.add_argument("--late-from", type=int, default=15, help="Late window used by G-S2-4's sd.")
    p.add_argument("--separability-budgets", type=int, nargs="*", default=[1, 2, 3],
                   help="Reminder budgets at which the model-optimal schedule is enumerated.")
    p.add_argument("--simulate-budget", type=int, default=3,
                   help="Budget used by the S3 arm-gap simulation.")
    p.add_argument("--simulate-n", type=int, default=200,
                   help="Simulated trajectories per item (0 skips the simulation).")
    p.add_argument("--s3-target-fraction", type=float, default=0.5,
                   help="G-S2-4's target: the fraction of the full-dose (zero_control vs "
                        "constant_remind) contrast S3 must resolve. 0.5 signed 2026-09-10.")
    p.add_argument("--threshold", type=int, default=2,
                   help="Option (b)'s objective: keep at least this many of the three constraints.")
    p.add_argument("--threshold-max-reminders", type=int, default=THRESHOLD_MAX_REMINDERS,
                   help="Open-loop schedules are enumerated exhaustively up to this many reminders.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-item", action="append", default=[], metavar="ITEM_ID")
    p.add_argument("--contemporaneous-v", action="store_true", required=True,
                   help="Mandatory on this line. `u_t` is injected before turn t's reply is "
                        "generated, so it acts within the turn; the lagged slot would estimate "
                        "carryover instead of the actuator.")
    return p.parse_args()


def rows_for_identification(readout: dict, excluded: list[str]) -> list[dict]:
    """S1b rows in the shape `modeling.dataset` expects.

    An unusable readout (the judge's verdict did not parse) becomes NaN rather
    than being dropped here: `build_reduced_state_pairs` drops every pair that
    would touch it, which is the difference between losing one transition and
    silently splicing turn t-1 onto turn t+1 as if they were adjacent.
    """

    rows = []
    for row in readout["rows"]:
        if row["branch"] not in S1B_ARMS or row["item_id"] in excluded:
            continue
        rows.append({
            "trajectory_id": row["trajectory_id"], "item_id": row["item_id"],
            "turn": row["turn"], "seed": row["seed"],
            "y_graded": float("nan") if row["y_graded"] is None else float(row["y_graded"]),
            "u_remind": float(row["u_remind"]),
        })
    return rows


def design_matrix(rows: list[dict], config: ReducedStateConfig) -> dict:
    """The same transitions `build_identification_dataset` produces, tagged
    with the item each one came from, as an explicit regression design.

    The design is `[z_t, v_t, 1]` with target `y_(t+1)` -- exactly the column
    of `KoopmanSurrogate.fit`'s ridge solve that carries `y` when the lift is
    the identity. It exists so `B` can be bootstrapped over items in closed
    form (per-item Gram matrices summed per resample) instead of refitting the
    surrogate ten thousand times. `_assert_matches_surrogate` is what keeps
    the two from drifting: if they ever disagree, the CI would belong to a
    model nobody fit.

    The transitions come from the builder itself, one item at a time, rather
    than from a hand-rolled copy of its loop -- including its NaN filter,
    which drops every pair touching an unparsed verdict. A private copy of
    that loop would drift from the builder the first time either changed, and
    the drift would be silent.
    """

    blocks, targets, items = [], [], []
    by_item: dict[str, list[dict]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)
    for item in sorted(by_item):
        data = build_identification_dataset(by_item[item], config, y_col="y_graded", u_col="u_remind")
        if data["Z"].shape[0] == 0:
            continue
        n = data["Z"].shape[0]
        blocks.append(np.hstack([data["Z"], data["V"], np.ones((n, 1))]))
        targets.append(data["Z_next"][:, config.nu - 1])
        items.extend([item] * n)
    return {
        "X": np.vstack(blocks), "y": np.concatenate(targets), "items": np.array(items),
        "u_index": config.nu + config.mu,  # z = [y lags, u lags]; v sits right after
    }


def _ridge_solve(gram: np.ndarray, moment: np.ndarray, ridge: float) -> np.ndarray:
    return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), moment)


def _assert_matches_surrogate(design: dict, dataset: dict, ridge: float) -> float:
    """The explicit solve must reproduce the surrogate's own `B` exactly."""

    beta = _ridge_solve(design["X"].T @ design["X"], design["X"].T @ design["y"], ridge)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=ridge).fit(dataset)
    explicit = float(beta[design["u_index"]])
    fitted = float(model.B[0, 0])
    if not math.isclose(explicit, fitted, rel_tol=1e-8, abs_tol=1e-10):
        raise SystemExit(
            f"explicit design gives B={explicit:.8f} but KoopmanSurrogate gives {fitted:.8f}: "
            f"the bootstrap would describe a model nobody fit")
    return fitted


def bootstrap_b(design: dict, ridge: float, seed: int) -> dict:
    """Resample ITEMS with replacement. A dialogue is the independent unit --
    turns inside one are dependent, and three seeds of one item replay the
    same constraints against the same user turns."""

    items = sorted(set(design["items"].tolist()))
    per_item = {}
    for item in items:
        mask = design["items"] == item
        x, y = design["X"][mask], design["y"][mask]
        per_item[item] = (x.T @ x, x.T @ y)

    rng = np.random.default_rng(seed)
    idx = design["u_index"]
    draws = []
    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(len(items), size=len(items), replace=True)
        gram = sum(per_item[items[i]][0] for i in chosen)
        moment = sum(per_item[items[i]][1] for i in chosen)
        draws.append(float(_ridge_solve(gram, moment, ridge)[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    point = float(_ridge_solve(design["X"].T @ design["X"], design["X"].T @ design["y"], ridge)[idx])
    return {"B": point, "ci": [float(lo), float(hi)], "bootstrap_sd": float(np.std(draws, ddof=1)),
            "n_items": len(items), "n_transitions": int(design["X"].shape[0])}


def fold_evaluation(defense, rows: list[dict], config: ReducedStateConfig, ridge: float,
                    n_folds: int, seed: int) -> dict:
    """Item-disjoint folds: the surrogate's scalar-y one-step MSE against the
    three nulls, on the identical transition set.

    Splitting by item (not by turn, not by trajectory) is the same anti-leak
    rule the `defense` line used for `attack_id`: turns within one dialogue
    are dependent, and the three seeds of one item share its constraints.
    """

    items = sorted({row["item_id"] for row in rows})
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(items))
    groups = [shuffled[i::n_folds] for i in range(n_folds)]

    folds = []
    for held_out in groups:
        if not held_out:
            continue
        held = set(held_out)
        train_rows = [r for r in rows if r["item_id"] not in held]
        test_rows = [r for r in rows if r["item_id"] in held]
        train_dataset = build_identification_dataset(train_rows, config, y_col="y_graded", u_col="u_remind")
        test_dataset = build_identification_dataset(test_rows, config, y_col="y_graded", u_col="u_remind")
        if train_dataset["Z"].shape[0] == 0 or test_dataset["Z"].shape[0] == 0:
            continue

        keys = ("y_now", "v", "y_next", "turn_next")
        train_aligned = dict(zip(keys, defense._aligned_transitions(train_rows, config, y_col="y_graded")))
        test_aligned = dict(zip(keys, defense._aligned_transitions(test_rows, config, y_col="y_graded")))
        assert len(test_aligned["y_next"]) == test_dataset["Z"].shape[0], "aux-turn alignment drifted"

        fold = {"held_out_items": sorted(held),
                "n_train_transitions": int(train_dataset["Z"].shape[0]),
                "n_test_transitions": int(len(test_aligned["y_next"]))}
        for name, extra in (("arx", no_extra_features), ("richer_abs_sign", abs_sign_extra_features)):
            model = KoopmanSurrogate(extra_features_fn=extra, ridge=ridge).fit(train_dataset)
            pred = np.array([model.step(z, v)[config.nu - 1]
                             for z, v in zip(test_dataset["Z"], test_dataset["V"])])
            fold[f"{name}_y_one_step_mse"] = float(np.mean((pred - test_aligned["y_next"]) ** 2))
        for name, pred in defense._null_predictions(train_aligned, test_aligned).items():
            fold[f"null_{name}_y_one_step_mse"] = float(np.mean((pred - test_aligned["y_next"]) ** 2))
        fold["best_null_y_one_step_mse"] = min(
            fold[f"null_{name}_y_one_step_mse"] for name in ("const", "turn_mean", "stateless"))
        for name in ("arx", "richer_abs_sign"):
            fold[f"{name}_beats_best_null"] = bool(
                fold[f"{name}_y_one_step_mse"] < fold["best_null_y_one_step_mse"])
        folds.append(fold)

    summary = {"n_folds": len(folds), "folds": folds}
    for name in ("arx", "richer_abs_sign"):
        summary[f"{name}_n_folds_beating_best_null"] = sum(1 for f in folds if f[f"{name}_beats_best_null"])
        summary[f"{name}_mean_y_one_step_mse"] = float(np.mean([f[f"{name}_y_one_step_mse"] for f in folds]))
    for name in ("const", "turn_mean", "stateless"):
        summary[f"null_{name}_mean_y_one_step_mse"] = float(
            np.mean([f[f"null_{name}_y_one_step_mse"] for f in folds]))
    return summary


def gate_s2_1(folds: dict, folds_to_pass: int, n_folds: int) -> dict:
    won = folds["arx_n_folds_beating_best_null"]
    return {
        "criterion": f"arx one-step scalar-y MSE beats the best trivial null on >= {folds_to_pass} "
                     f"of {n_folds} item-disjoint folds; every null gets `turn`",
        "arx_folds_beating_best_null": won, "n_folds": folds["n_folds"],
        "richer_abs_sign_folds_beating_best_null": folds["richer_abs_sign_n_folds_beating_best_null"],
        "arx_mean_mse": folds["arx_mean_y_one_step_mse"],
        "best_null_mean_mse": min(folds[f"null_{n}_mean_y_one_step_mse"]
                                  for n in ("const", "turn_mean", "stateless")),
        "verdict": "PASS" if won >= folds_to_pass else "FAIL",
        "a_weak_result_does_not_close_the_line": (
            "65% of judged constraints sit outside the judge's calibration set; that observation "
            "noise biases the state coefficient toward zero. A weak result here has two causes "
            "this design cannot separate (screening section 10 item 9) -- report and hand back."),
    }


def gate_s2_2(b: dict) -> dict:
    excludes_zero = bool(b["ci"][0] * b["ci"][1] > 0)
    mde = POWER_Z * b["bootstrap_sd"]
    return {
        "criterion": "B's 95% CI excludes 0, sign positive (a reminder raises y)",
        **b, "mde_at_80pct": mde, "effect_over_mde": abs(b["B"]) / mde if mde else None,
        "ci_excludes_zero": excludes_zero, "sign_is_positive": b["B"] > 0,
        "verdict": "PASS" if (excludes_zero and b["B"] > 0) else "FAIL",
        "caveat": None if abs(b["B"]) >= mde else (
            "significant at ~1.96 sigma but below the 80%-power MDE: the point estimate is "
            "likely inflated, as with S0-0's K3 under the old harness"),
    }


def gate_s2_3(diagnostics: dict, state_dim: int) -> dict:
    radius = diagnostics["spectral_radius"]
    return {
        "criterion": "recorded, not a gate: spectral radius in (0.1, 1.05), full controllability "
                     "rank, Gramian condition < 1e12",
        "spectral_radius": radius, "controllability_rank": diagnostics["controllability_rank"],
        "state_dim": state_dim, "gramian_condition": diagnostics["gramian_condition"],
        "radius_in_band": bool(0.1 < radius < 1.05),
        "full_rank": diagnostics["controllability_rank"] == state_dim,
        "gramian_well_conditioned": diagnostics["gramian_condition"] < 1e12,
    }


def s3_sizing_variance(pilot, rows: list[dict], late_from: int, n_turns: int) -> dict:
    """The variance S3's primary quantity actually has: the sd of the PAIRED
    per-(item, seed) difference between two schedules, split into a
    between-item and a within-item-across-seeds part.

    This is the caliber correction signed 2026-09-10. The first version of
    G-S2-4 used the sd of the late-window mean ACROSS trajectories (0.1632 on
    this arm), but plan section 6's primary is a bootstrap over differences
    paired by `(item_id, seed)`, and pairing removes the between-item level
    that dominates this readout. Sizing an arm comparison on an unpaired sd
    prices a design nobody is going to run.

    The split matters for the same reason it did when S1 was sized: adding
    SEEDS shrinks only the within-item term, so a pooled sd would let someone
    buy seeds expecting power that only items can deliver.
    """

    contrast = pilot.per_item_contrast(rows, "constant_remind", "zero_control",
                                       range(late_from, n_turns + 1))
    components = pilot.variance_components(contrast)
    return {
        "paired_on": "(item_id, seed), plan section 6's primary",
        "sd_between_items": components["sd_between_items"],
        "sd_within_item_across_seeds": components["sd_within_item_across_seeds"],
        "sd_of_item_means": components["sd_of_item_means"],
        "measured_on": "the S1a arms (constant_remind vs zero_control), the only pair of arms "
                       "in this data that differ in schedule",
    }


def gate_s2_4(full_dose_contrast: float, sizing: dict, seeds: int, fraction: float,
              unpaired_late_sd: float, b: float) -> dict:
    """Trajectories per arm S3 needs at 80% power, against the signed target.

    THE TARGET (user ruling 2026-09-10, option A): S3 must resolve
    `fraction` x the full-dose contrast -- half of what never-remind vs
    always-remind buys. The plan said to back the number out of `B`, but S3's
    three arms spend the SAME budget and differ only in where the reminders
    land, so `B` (the value of one more reminder) does not name that contrast;
    and S2 showed the operator under-predicts the sustained endpoint gap by
    2.8x, so a `B`-derived target would be priced off a quantity the operator
    itself cannot extrapolate.

    What the ruling costs is recorded with it, not left implicit: at half the
    full-dose contrast the design is blind to a closed-loop advantage smaller
    than that, so a CI covering zero in S3 is UNDECIDABLE, never "the closed
    loop is worth nothing" -- the clause ERGO's pre-closure review had to
    invent after the fact.

    Other fractions are computed beside it so the cost of the ruling stays
    visible, and the old unpaired sd is kept as a labelled secondary so the
    superseded number remains traceable.
    """

    var = lambda s: sizing["sd_between_items"] ** 2 + sizing["sd_within_item_across_seeds"] ** 2 / s
    def items_needed(effect: float, s: int) -> int:
        return math.ceil(POWER_Z ** 2 * var(s) / effect ** 2)

    target = fraction * full_dose_contrast
    n_items = items_needed(target, seeds)
    per_arm = n_items * seeds
    table = {}
    for frac in (1.0, 0.5, 1 / 3, 0.25):
        effect = frac * full_dose_contrast
        n = items_needed(effect, seeds)
        table[f"{frac:.2f}x_full_dose"] = {
            "effect": effect, "n_items": n, "trajectories_per_arm": n * seeds,
            "within_ceiling": n * seeds <= S3_TRAJECTORIES_PER_ARM_CEILING,
        }
    n_b = items_needed(abs(b), seeds)
    table["B_one_reminder"] = {
        "effect": abs(b), "n_items": n_b, "trajectories_per_arm": n_b * seeds,
        "within_ceiling": n_b * seeds <= S3_TRAJECTORIES_PER_ARM_CEILING,
        "note": "the plan's literal reading; it prices S3 off a quantity the operator cannot "
                "extrapolate, and it does not name an equal-budget contrast at all",
    }
    return {
        "criterion": f"trajectories per arm at 80% power to resolve {fraction:.2f} x the full-dose "
                     f"contrast; > {S3_TRAJECTORIES_PER_ARM_CEILING} per arm -> stop and report",
        "signed": "2026-09-10, option A (screening section 10 item 10)",
        "full_dose_contrast": full_dose_contrast, "target_fraction": fraction,
        "target_effect": target, "seeds": seeds,
        "paired_sd_components": sizing,
        "n_items_required": n_items, "trajectories_per_arm": per_arm,
        "mde_at_current_design": POWER_Z * math.sqrt(var(seeds) / n_items),
        "sizing_table": table,
        "unpaired_late_window_sd_superseded": unpaired_late_sd,
        "why_superseded": "sd of the late-window mean across trajectories, i.e. an UNPAIRED "
                          "caliber. Plan section 6 pairs by (item, seed), and pairing removes the "
                          "between-item level; the paired components above are the right "
                          "denominator (user ruling 2026-09-10).",
        "s3_reporting_rules_signed_with_this": [
            "a CI covering zero in S3 is UNDECIDABLE, not a negative result about the closed loop",
            f"every S3 number carries: this design resolves arm gaps of "
            f"{POWER_Z * math.sqrt(var(seeds) / n_items):.4f} or larger, about "
            f"{fraction:.0%} of the full-dose gain",
        ],
        "verdict": "PASS" if per_arm <= S3_TRAJECTORIES_PER_ARM_CEILING else "STOP_AND_REPORT",
    }


def one_step_transitions(rows: list[dict]) -> list[dict]:
    """(y_t, u_t, y_(t+1)) per adjacent pair, dropping any pair that touches an
    unparsed verdict. One definition shared by every diagnostic below, so a
    change to what counts as a transition cannot reach one of them and not the
    others."""

    out = []
    for traj_rows in group_by_trajectory(rows).values():
        ordered = sorted(traj_rows, key=lambda r: r["turn"])
        for now, nxt in zip(ordered, ordered[1:]):
            if now["y_graded"] != now["y_graded"] or nxt["y_graded"] != nxt["y_graded"]:
                continue
            out.append({
                "y_now": now["y_graded"], "y_next": nxt["y_graded"], "u": nxt["u_remind"],
                "turn": float(nxt["turn"]), "item_id": now["item_id"],
                "trajectory_id": now["trajectory_id"],
            })
    return out


def state_provenance(rows: list[dict]) -> dict:
    """DIAGNOSTIC, not a gate: where the state term's predictive power comes
    from -- turn-to-turn dynamics, or a level that never changes.

    `y_t` can predict `y_(t+1)` for two very different reasons: the dialogue
    moves and the operator tracks the movement, or this item is simply harder
    than the others and both turns inherit the same level. The fold nulls
    cannot separate them (a held-out item's own mean is not available at test
    time, so it is not a legitimate null), but the same regression run on
    demeaned data can say how much survives.

    Three specifications of `y_(t+1) ~ y_t + u + turn`: pooled, with item
    means removed, and with each trajectory's own mean removed. The last is
    the harshest and is downward biased at fixed T (Nickell, order
    -(1+rho)/(T-1)), so it is a lower bound, not the estimate. The `u`
    coefficient is printed in all three because an actuator effect that moves
    with the specification would be a level artifact rather than an action.
    """

    def fit(group_key: str | None) -> dict:
        y0 = np.array([r["y_now"] for r in transitions])
        y1 = np.array([r["y_next"] for r in transitions])
        u = np.array([r["u"] for r in transitions])
        turn = np.array([r["turn"] for r in transitions])
        if group_key is not None:
            keys = np.array([r[group_key] for r in transitions])
            for key in set(keys.tolist()):
                mask = keys == key
                y0[mask] -= y0[mask].mean()
                y1[mask] -= y1[mask].mean()
                u[mask] -= u[mask].mean()
        design = np.column_stack([np.ones_like(y0), y0, u, turn - turn.mean()])
        beta = np.linalg.lstsq(design, y1, rcond=None)[0]
        return {"y_prev_coefficient": float(beta[1]), "u_coefficient": float(beta[2])}

    transitions = one_step_transitions(rows)
    return {
        "is_a_gate": False,
        "n_transitions": len(transitions),
        "pooled": fit(None),
        "item_demeaned": fit("item_id"),
        "trajectory_demeaned": fit("trajectory_id"),
        "reading": "the gap between pooled and item-demeaned is the share of the state term "
                   "carried by persistent per-item difficulty rather than turn-to-turn dynamics; "
                   "the trajectory-demeaned row is a downward-biased lower bound at fixed T",
    }


def late_window_paired_sd(readout: dict, excluded: list[str], late_from: int) -> float:
    """sd across (item, seed) cells of the late-window mean `y`, pooled over
    the S1b arms -- the unit S3's paired comparison will be made on."""

    cells: dict[tuple[str, str, int], list[float]] = {}
    for row in readout["rows"]:
        if row["branch"] not in S1B_ARMS or row["item_id"] in excluded:
            continue
        if row["y_graded"] is None or row["turn"] < late_from:
            continue
        cells.setdefault((row["branch"], row["item_id"], row["seed"]), []).append(row["y_graded"])
    means = [statistics.fmean(v) for v in cells.values()]
    return statistics.stdev(means)


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    gates = _sibling_module("analyze_sequor_s0_0_gates.py")
    defense = _sibling_module("fit_koopman_defense_model.py")

    readout = gates.load_readout(args.independent_readout, "independent")
    arm_dir = pathlib.Path(readout["arm_dir"])
    arm_report = json.loads((arm_dir / "arm_report.json").read_text())
    cap_record = gates.refuse_if_the_cap_bound(arm_report, args.exclude_item)

    config = ReducedStateConfig(nu=args.nu, mu=args.mu, contemporaneous_v=args.contemporaneous_v)
    rows = rows_for_identification(readout, args.exclude_item)
    if not rows:
        raise SystemExit("no S1b rows: this readout has no bernoulli/antithetic arm")
    dataset = build_identification_dataset(rows, config, y_col="y_graded", u_col="u_remind")
    design = design_matrix(rows, config)
    if design["X"].shape[0] != dataset["Z"].shape[0]:
        raise SystemExit(f"explicit design has {design['X'].shape[0]} transitions but the "
                         f"identification dataset has {dataset['Z'].shape[0]}")

    _assert_matches_surrogate(design, dataset, args.ridge)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=args.ridge).fit(dataset)
    diagnostics = model.controllability(args.controllability_horizon)

    folds = fold_evaluation(defense, rows, config, args.ridge, args.n_folds, args.seed)
    b = bootstrap_b(design, args.ridge, args.seed)
    unpaired_sd = late_window_paired_sd(readout, args.exclude_item, args.late_from)

    pilot = _sibling_module("analyze_sequor_s1_pilot.py")
    all_rows = [r for r in readout["rows"] if r["item_id"] not in args.exclude_item]
    n_turns = arm_report["n_turns"]
    seeds = len(arm_report["seeds"])
    late = range(args.late_from, n_turns + 1)
    full_dose = pilot.bootstrap_contrast(
        pilot.per_item_contrast(all_rows, "constant_remind", "zero_control", late), args.seed)["point"]
    sizing = s3_sizing_variance(pilot, all_rows, args.late_from, n_turns)

    transitions = one_step_transitions(rows)
    bilinear = fit_bilinear(transitions, args.seed)
    spec = bilinear["pooled"]
    linear_op = {"a": float(model.A[0, 0]), "b": float(model.B[0, 0]), "c": float(model.b[0]), "d": 0.0}
    bilinear_op = {"a": spec["y_prev_coefficient"], "b": spec["u_coefficient"],
                   "c": float(model.b[0]), "d": spec["interaction_d"]}
    # each item's state at the last action-free turn: turn 1 states the constraints
    first_turn: dict[str, list[float]] = {}
    for row in all_rows:
        if row["turn"] == 1 and row["y_graded"] is not None:
            first_turn.setdefault(row["item_id"], []).append(row["y_graded"])
    starts = {item: statistics.fmean(v) for item, v in sorted(first_turn.items())}
    separability = schedule_separability(
        {"linear": linear_op, "bilinear": bilinear_op}, starts,
        tuple(args.separability_budgets), start_turn=2, n_turns=n_turns, late_from=args.late_from)
    # The 40 items agreeing could just mean their starting states are too alike.
    # Sweeping the WHOLE state range answers the stronger question: is there ANY
    # reachable state at which this operator would want a different schedule?
    sweep = schedule_separability(
        {"bilinear": bilinear_op}, {f"y={v:.2f}": float(v) for v in np.linspace(0.0, 1.0, 101)},
        tuple(args.separability_budgets), start_turn=2, n_turns=n_turns, late_from=args.late_from)
    separability["state_sweep"] = {
        "n_states": 101, "range": [0.0, 1.0],
        "distinct_schedules": {k: v["n_distinct_schedules"]
                               for k, v in sweep["by_operator"]["bilinear"].items()},
        "reading": "one schedule across the entire state range means the degeneracy is structural, "
                   "not an accident of where these 40 items happen to start",
    }
    residuals = np.array([t["y_next"] - _step(t["y_now"], int(t["u"]), bilinear_op)
                          for t in transitions])
    simulation = (simulate_s3_gap(bilinear_op, starts, residuals, args.simulate_budget, 2, n_turns,
                                  args.late_from, args.simulate_n, args.seed)
                  if args.simulate_n else None)

    # Option (a): the same two checks on the state the judge actually returns.
    # It is sized against G-S2-4's resolution, not a bar of its own.
    sizing_gate = gate_s2_4(full_dose, sizing, seeds, args.s3_target_fraction, unpaired_sd, b["B"])
    binary = binary_state_model(
        readout, args.exclude_item, all_rows, tuple(args.separability_budgets), 2, n_turns,
        args.late_from, args.simulate_budget, args.simulate_n,
        sizing_gate["mde_at_current_design"], args.seed)

    # Option (b) on paper: the same kernel under a threshold objective, priced
    # per reminder so the closed loop cannot win by spending more.
    threshold_size = threshold_sizing(pilot, all_rows, args.threshold, args.late_from, n_turns,
                                      seeds, args.s3_target_fraction, args.seed)
    threshold = (threshold_objective(
        np.array(binary["planner_kernel"]),
        {item: np.array(dist) for item, dist in binary["starting_distributions"].items()},
        args.threshold, 2, n_turns, args.late_from, args.threshold_max_reminders,
        THRESHOLD_LAMBDAS, threshold_size["mde_at_current_design"],
        threshold_size.get("full_dose_contrast"))
        if binary.get("planner_kernel") is not None else None)

    report = {
        "arm_dir": str(arm_dir), "provenance": readout.get("provenance"),
        "judge_kind": readout["judge_kind"], "judge_model": readout["judge_model"],
        "arms_used": list(S1B_ARMS), "excluded_items": sorted(args.exclude_item),
        "response_cap": cap_record,
        "config": {"nu": args.nu, "mu": args.mu, "ridge": args.ridge,
                   "contemporaneous_v": args.contemporaneous_v,
                   "n_folds": args.n_folds, "folds_to_pass": args.folds_to_pass,
                   "late_from": args.late_from, "seed": args.seed,
                   "n_bootstrap": N_BOOTSTRAP},
        "n_trajectories": len({r["trajectory_id"] for r in rows}),
        "n_transitions": int(dataset["Z"].shape[0]),
        "operator": {"A": model.A.tolist(), "B": model.B.tolist(), "b": model.b.tolist(),
                     "C": model.C.tolist()},
        "fold_evaluation": folds,
        "state_provenance": state_provenance(rows),
        "g_s2_5_bilinear": bilinear,
        "schedule_separability": separability,
        "s3_gap_simulation": simulation,
        "g_s2_1": gate_s2_1(folds, args.folds_to_pass, args.n_folds),
        "g_s2_2": gate_s2_2(b),
        "g_s2_3": gate_s2_3(diagnostics, model.state_dim),
        "g_s2_4": sizing_gate,
        "binary_state_model": binary,
        "threshold_objective": threshold,
        "threshold_objective_sizing": threshold_size,
        "gold_coverage": arm_report["gold_coverage"],
        "caveat": (
            f"{arm_report['gold_coverage']['share_outside_calibration_set']:.0%} of the judged "
            "constraints lie outside the judge's calibration set. That observation noise biases "
            "the state coefficient toward zero -- the direction that would fake this line's "
            "death -- so a weak G-S2-1 is reported and handed back, never used to close "
            "(screening section 10 item 9)."),
    }

    print(f"{report['n_trajectories']} trajectories, {report['n_transitions']} transitions "
          f"({', '.join(S1B_ARMS)}), contemporaneous_v={args.contemporaneous_v}")
    print(f"A={model.A.tolist()}  B={model.B.tolist()}  b={model.b.tolist()}\n")
    for name in ("const", "turn_mean", "stateless"):
        print(f"  null {name:<10} mean one-step MSE {folds[f'null_{name}_mean_y_one_step_mse']:.6f}")
    for name in ("arx", "richer_abs_sign"):
        print(f"  {name:<15} mean {folds[f'{name}_mean_y_one_step_mse']:.6f}  beats best null in "
              f"{folds[f'{name}_n_folds_beating_best_null']}/{folds['n_folds']} folds")
    g1, g2, g3, g4 = (report["g_s2_1"], report["g_s2_2"], report["g_s2_3"], report["g_s2_4"])
    print(f"\nG-S2-1 {g1['verdict']}: arx wins {g1['arx_folds_beating_best_null']}/{g1['n_folds']} "
          f"(needs {args.folds_to_pass})")
    print(f"G-S2-2 {g2['verdict']}: B {g2['B']:+.5f} CI [{g2['ci'][0]:+.5f}, {g2['ci'][1]:+.5f}], "
          f"MDE {g2['mde_at_80pct']:.5f}, effect/MDE {g2['effect_over_mde']:.2f}")
    print(f"G-S2-3 record: spectral radius {g3['spectral_radius']:.4f} (in band {g3['radius_in_band']}), "
          f"rank {g3['controllability_rank']}/{g3['state_dim']}, "
          f"gramian cond {g3['gramian_condition']:.3e}")
    print(f"G-S2-4 {g4['verdict']}: full-dose contrast {g4['full_dose_contrast']:+.4f}, target "
          f"{g4['target_fraction']:.0%} = {g4['target_effect']:.4f} -> {g4['n_items_required']} items "
          f"x {g4['seeds']} seeds = {g4['trajectories_per_arm']} trajectories/arm "
          f"(ceiling {S3_TRAJECTORIES_PER_ARM_CEILING})")
    for name, row in g4["sizing_table"].items():
        print(f"    {name:<18} effect {row['effect']:.4f} -> {row['n_items']} items, "
              f"{row['trajectories_per_arm']} traj/arm  "
              f"{'ok' if row['within_ceiling'] else 'over the ceiling'}")
    print(f"    paired sd: between-items {sizing['sd_between_items']:.4f}, within-item across "
          f"seeds {sizing['sd_within_item_across_seeds']:.4f}  "
          f"(superseded unpaired sd {unpaired_sd:.4f})")
    bl = report["g_s2_5_bilinear"]
    print(f"\nG-S2-5 双线性 {bl['verdict']}: pooled d {bl['pooled']['interaction_d']:+.4f} "
          f"CI [{bl['pooled']['ci'][0]:+.4f}, {bl['pooled']['ci'][1]:+.4f}] "
          f"MDE {bl['pooled']['mde_at_80pct']:.4f} ({bl['pooled']['effect_over_mde']:.2f}x); "
          f"item-demeaned {bl['item_demeaned']['verdict']} d {bl['item_demeaned']['interaction_d']:+.4f} "
          f"CI [{bl['item_demeaned']['ci'][0]:+.4f}, {bl['item_demeaned']['ci'][1]:+.4f}]")
    print("  一次提醒的边际效应 " + "  ".join(
        f"{k} {v:+.4f}" for k, v in bl["pooled"]["marginal_effect_of_a_reminder"].items()))
    sep = report["schedule_separability"]
    print(f"\n日程可分性（{sep['objective']}，决策自 t{sep['start_turn']}）: "
          f"linear {sep['linear_distinct_schedules']} (负对照须全为 1: {sep['negative_control_holds']})  "
          f"bilinear {sep['bilinear_distinct_schedules']}  "
          f"闭环有事可做: {sep['closed_loop_has_something_to_do']}")
    for k, rec in sep["by_operator"]["bilinear"].items():
        print(f"    k={k}: {rec['n_distinct_schedules']} 种日程，众数 {rec['modal_schedule_from_end']} "
              f"占 {rec['modal_share']:.0%}")
    print(f"    全状态区间扫描（101 个起始 y）: {sep['state_sweep']['distinct_schedules']}")
    sim = report["s3_gap_simulation"]
    if sim:
        print(f"\nS3 臂间差模拟（预算 {sim['budget']}，每题 {sim['n_sims_per_item']} 次，共同随机数）: "
              f"MPC − 最优固定日程 {sim['mpc_minus_best_fixed']:+.4f}, "
              f"MPC − 随机 {sim['mpc_minus_random']:+.4f}；"
              f"本设计的 MDE {g4['mde_at_current_design']:.4f}")
    print_binary_section(report["binary_state_model"])
    if threshold is not None:
        print_threshold_section(threshold, threshold_size)

    prov = report["state_provenance"]
    print(f"\nstate provenance (diagnostic, not a gate): y_prev coefficient pooled "
          f"{prov['pooled']['y_prev_coefficient']:+.4f} -> item-demeaned "
          f"{prov['item_demeaned']['y_prev_coefficient']:+.4f} -> trajectory-demeaned "
          f"{prov['trajectory_demeaned']['y_prev_coefficient']:+.4f} (lower bound); "
          f"u coefficient {prov['pooled']['u_coefficient']:+.4f} / "
          f"{prov['item_demeaned']['u_coefficient']:+.4f} / "
          f"{prov['trajectory_demeaned']['u_coefficient']:+.4f}")
    print(f"\n{report['caveat']}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
