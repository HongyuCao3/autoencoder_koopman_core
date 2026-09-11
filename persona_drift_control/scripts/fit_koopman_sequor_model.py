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
    endpoint. Deterministic rollout: this is the planner's own model."""

    y, kept = y0, []
    for offset, u in enumerate(schedule):
        y = _step(y, u, op)
        if start_turn + offset + 1 >= late_from:
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
        turn_next = start_turn + offset + 1
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
            if start_turn + offset + 1 >= late_from:
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
        "g_s2_4": gate_s2_4(full_dose, sizing, seeds, args.s3_target_fraction,
                            unpaired_sd, b["B"]),
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
