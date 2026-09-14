#!/usr/bin/env python3
"""D-2.5, the causal upper bound, computed EXACTLY on GPU-2a's depth-4 tree.

Zero GPU: no model is loaded, no text is generated, no readout is re-run. The
67,660 rows of outputs/tsar_cefr_gpu2a_branch/tree.jsonl are read and the
signed criteria are applied.

WHAT D-2.5 ASKS. Does watching the runtime readout buy anything over the best
schedule fixable before the run starts? If the best bin-measurable closed-loop
policy cannot beat the best open-loop schedule, feedback control has no
headroom on this task and the line closes (kill_criterion.md section 2, D-2.5:
CI containing zero fires).

WHY THE TREE HAD TO BE BOUGHT. GPU-1's excitation arm samples 3 of 4^6 paths
per cell; an IPW estimate of one 6-step deterministic policy has an effective
sample size of 0.146 trajectories. The upper bound was not small there, it was
not estimable. The tree enumerates the action axis instead, so every policy in
the class is read off exactly, with no importance weights and no operator in
between.

THE POLICY CLASSES, AND THE ONE THAT WAS REVISED.

  open loop   pi: (source level, target level, step) -> action.  UNCHANGED.
  closed loop pi: (bin of d_t, step) -> action, shared across cells, one fit
              per target level.  REVISED 2026-09-14 (kill_criterion section
              2.6): the signed version was STEP-INVARIANT while the open-loop
              class was step-dependent, so the comparison asked "sighted with
              one hand tied vs blind with two" instead of "does runtime
              feedback pay". The revision is a post-hoc loosening and section
              2.6 writes out all four disclosures. Its direction was known in
              advance: the asymmetry depresses the closed-loop side, i.e. it
              makes the line MORE likely to close.

  The rationale the signed text gave for step-invariance ("otherwise you must
  enumerate 4^12") was arithmetically wrong, which is why `solve_closed_loop`
  below is an EXACT solver and not a search. See THE SOLVER.

THE BINS ARE NOT FITTED. Bin edges on d_t = ell_t - ell* are verbatim the three
rules of plan section 5.4's `greedy_reactive` (far >= 1.0 -> step_down,
near [0.5, 1.0) -> half_step_down, arrived < 0.5 -> copy), so the supremum has
a direct reading: how much better than that hand-written rule can ANY
bin-measurable policy be. Four bins (splitting arrived into arrived/overshoot)
is the secondary reading, never the verdict.

THE SOLVER IS EXACT, AND THAT WORD IS LOAD-BEARING. A policy shared across
cells cannot be optimised by per-node backward induction -- that would let each
node pick its own action and collapse the supremum onto the omniscient path,
which is the failure section 2.5 was written to prevent. The exact method is
prefix enumeration: the decisions of steps 1..t-1 determine where every cell
sits, so states are enumerated as DECISION PREFIXES and deduplicated by the
resulting position vector; only bins actually occupied are branched on. At the
last step the choice separates per bin given the prefix, so it is taken by
argmin rather than enumerated. `tests/test_analyze_tsar_cefr_d25.py` pins the
word by brute-forcing all 4^(bins*steps) policies on a small instance and
asserting equality.

LEAVE ONE SOURCE OUT, BECAUSE IN-SAMPLE BOUNDS LIE BY AN ORDER OF MAGNITUDE.
The same in-sample bound on the `defense` line read +0.0469 * on 16
trajectories and +0.0042 (crossing zero) on 40 -- an 11x overstatement
(kill_criterion.md section 3 (1)). Both classes are fitted on 99 sources and
scored on the held-out one, for all 100. The in-sample number is reported
beside it and DOES NOT JUDGE.

NO LEAKAGE THROUGH AN UNVISITED BIN. If the held-out source reaches a (bin,
step) no training cell occupied, that cell of the policy table carries no
in-fold support and reading it would import the full-sample fit. The fallback
is the bin's namesake action (section 2.6 (3)); every fallback is counted in
the report, because a bound resting on many fallbacks means something else.

WHAT THIS SCRIPT DOES NOT DECIDE. The residual asymmetry on the source-level
axis (open loop conditions on source level, closed loop sees it only through
the bin) is NOT repaired here. A superset class pi: (source level, bin, step)
is computed and reported in the same file as a DIAGNOSTIC that is explicitly
not the verdict (section 2.6 (2)) -- it answers "what is that axis worth", and
section 2.6 flags in writing that it must not become a back door for reopening
a closed line.

THE NUMBER IS SINGLE-REALIZATION. The tree is greedy, temperature 0; the action
axis is enumerated, not sampled, so there is no seed axis to aggregate over.
Uncertainty here is over SOURCES (bootstrap by source_id, 2000 draws, as
signed), not over seeds. Per .claude/global.md the D-2.5 number therefore may
not be quoted as a cross-arm headline result in `mean +- std (n seeds)` form;
it is a gate verdict. The report carries that obligation as a field.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import sys

import numpy as np

CEFR_CODE = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
ACTIONS = ("step_down", "half_step_down", "paraphrase", "copy")
N_ACTIONS = len(ACTIONS)
DEPTH = 4
OFFSETS = (0, 1, 5, 21, 85)
N_NODES = 341

CAP_ITEM_LIMIT = 0.05
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260914
MDE_Z = 2.80
PREFIX_CAP = 3_000_000


def bin_edges_3(d: float) -> int:
    if d >= 1.0:
        return 0
    if d >= 0.5:
        return 1
    return 2


def bin_edges_4(d: float) -> int:
    if d >= 1.0:
        return 0
    if d >= 0.5:
        return 1
    if d >= 0.0:
        return 2
    return 3


BIN_SCHEMES = {
    3: (bin_edges_3, ("far", "near", "arrived"), (0, 1, 3)),
    4: (bin_edges_4, ("far", "near", "arrived", "overshoot"), (0, 1, 3, 3)),
}


def child_index(idx: np.ndarray, depth: int, action: np.ndarray | int) -> np.ndarray:
    local = idx - OFFSETS[depth]
    return OFFSETS[depth + 1] + local * N_ACTIONS + action


class Cell:
    __slots__ = ("text_id", "source_id", "target", "lstar", "ell", "termsq", "n_cap")

    def __init__(self, text_id: str, source_id: str, target: str) -> None:
        self.text_id = text_id
        self.source_id = source_id
        self.target = target
        self.lstar = CEFR_CODE[target]
        self.ell = np.full(N_NODES, np.nan, dtype=np.float64)
        self.termsq = np.full(N_NODES, np.nan, dtype=np.float64)
        self.n_cap = 0


def load_tree(path: pathlib.Path) -> dict[str, Cell]:
    cells: dict[str, Cell] = {}
    action_idx = {a: i for i, a in enumerate(ACTIONS)}
    for line in path.open():
        row = json.loads(line)
        tid = row["text_id"]
        cell = cells.get(tid)
        if cell is None:
            cell = cells[tid] = Cell(tid, row["source_id"], row["target_cefr"].upper())
            cell.ell[0] = row["source_level_expected"]
        depth = row["depth"]
        local = 0
        for a in row["path"]:
            local = local * N_ACTIONS + action_idx[a]
        idx = OFFSETS[depth] + local
        cell.ell[idx] = row["level_expected"]
        if depth == DEPTH:
            cell.termsq[idx] = (CEFR_CODE[row["level_official"]] - cell.lstar) ** 2.0
        cell.n_cap += int(row["hit_token_cap"])
    for cell in cells.values():
        if np.isnan(cell.ell[: OFFSETS[DEPTH]]).any():
            raise ValueError(f"{cell.text_id}: missing a node at depth 0..3")
        if np.isnan(cell.termsq[OFFSETS[DEPTH] :]).any():
            raise ValueError(f"{cell.text_id}: missing a terminal node")
    return cells


def cap_pressure(cells: dict[str, Cell]) -> dict:
    per_item = {c.text_id: c.n_cap / (N_NODES - 1) for c in cells.values()}
    over = sorted(k for k, v in per_item.items() if v > CAP_ITEM_LIMIT)
    pooled = sum(c.n_cap for c in cells.values()) / (len(cells) * (N_NODES - 1))
    worst = max(per_item, key=per_item.get)
    return {
        "limit_per_item": CAP_ITEM_LIMIT,
        "rows_per_item": N_NODES - 1,
        "design_resolution": 1.0 / (N_NODES - 1),
        "pooled_share": pooled,
        "worst_item": worst,
        "worst_share": per_item[worst],
        "items_over_limit": over,
        "nonzero_items": {k: v for k, v in sorted(per_item.items()) if v > 0},
        "note": (
            "Unlike GPU-1's 51-a2 (18 rows per item, so one truncation = 0.0556 and the "
            "threshold was zero tolerance), an item here holds 340 rows and the "
            "resolution is 0.0029, so a flagged rate is a real rate."
        ),
    }


class Group:
    """Cells that share one fitted policy, with their tree flattened into arrays."""

    def __init__(self, cells: list[Cell], nbins: int) -> None:
        binner, self.bin_names, self.fallback = BIN_SCHEMES[nbins]
        self.nbins = nbins
        self.cells = cells
        self.n = len(cells)
        self.binv = np.zeros((self.n, N_NODES), dtype=np.int8)
        self.termsq = np.zeros((self.n, N_NODES), dtype=np.float64)
        for i, cell in enumerate(cells):
            for idx in range(OFFSETS[DEPTH]):
                self.binv[i, idx] = binner(cell.ell[idx] - cell.lstar)
            self.termsq[i] = np.nan_to_num(cell.termsq)
        self.index = {cell.text_id: i for i, cell in enumerate(cells)}

    def walk(self, table: np.ndarray, i: int) -> tuple[int, list[int], list[int]]:
        """Follow one cell under a full decision table; return terminal idx, bins, actions."""
        idx, bins, acts = 0, [], []
        for t in range(DEPTH):
            b = int(self.binv[i, idx])
            a = int(table[t, b])
            bins.append(b)
            acts.append(a)
            idx = int(child_index(np.int64(idx), t, a))
        return idx, bins, acts


def enumerate_prefixes(g: Group) -> tuple[np.ndarray, np.ndarray]:
    """All step-1..3 decision prefixes, deduplicated by the position vector they induce.

    Deduplication is done on the FULL cell set of the group, so two prefixes kept
    apart here can differ on any subset too, and two prefixes merged here agree on
    every subset -- which is what makes one enumeration serve all leave-one-out folds.
    """
    pos = np.zeros((1, g.n), dtype=np.int32)
    dec = np.zeros((1, DEPTH, g.nbins), dtype=np.int8)
    for t in range(DEPTH - 1):
        bins_now = g.binv[np.arange(g.n)[None, :], pos]
        occ = np.zeros((pos.shape[0], g.nbins), dtype=bool)
        for b in range(g.nbins):
            occ[:, b] = (bins_now == b).any(axis=1)
        new_pos, new_dec = [], []
        for pattern in {tuple(r) for r in occ}:
            rows = np.flatnonzero((occ == np.array(pattern)).all(axis=1))
            free = [b for b, o in enumerate(pattern) if o]
            for combo in np.ndindex(*([N_ACTIONS] * len(free))):
                table = np.array(g.fallback, dtype=np.int8)
                for b, a in zip(free, combo):
                    table[b] = a
                acts = table[bins_now[rows]]
                new_pos.append(child_index(pos[rows], t, acts).astype(np.int32))
                d = dec[rows].copy()
                d[:, t, :] = table
                new_dec.append(d)
        pos = np.concatenate(new_pos)
        dec = np.concatenate(new_dec)
        _, keep = np.unique(pos, axis=0, return_index=True)
        keep.sort()
        pos, dec = pos[keep], dec[keep]
        if pos.shape[0] > PREFIX_CAP:
            raise MemoryError(f"prefix count {pos.shape[0]} exceeds cap {PREFIX_CAP}")
    return pos, dec


def step4_sums(g: Group, pos3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S[prefix, bin, action] summed over ALL cells, and each cell's step-4 bin."""
    n_p = pos3.shape[0]
    bins3 = g.binv[np.arange(g.n)[None, :], pos3]
    S = np.zeros((n_p, g.nbins, N_ACTIONS), dtype=np.float64)
    for a in range(N_ACTIONS):
        vals = g.termsq[np.arange(g.n)[None, :], child_index(pos3, DEPTH - 1, a)]
        for b in range(g.nbins):
            S[:, b, a] = np.where(bins3 == b, vals, 0.0).sum(axis=1)
    return S, bins3


def solve_closed_loop(
    g: Group, pos3: np.ndarray, dec: np.ndarray, S_all: np.ndarray,
    bins3: np.ndarray, members: np.ndarray
) -> tuple[np.ndarray, float, dict]:
    """Exact minimiser of mean terminal squared error over `members` (indices into g)."""
    S = S_all.copy()
    drop = np.setdiff1d(np.arange(g.n), members)
    for i in drop:
        col = pos3[:, i]
        b = bins3[:, i]
        rows = np.arange(pos3.shape[0])
        for a in range(N_ACTIONS):
            S[rows, b, a] -= g.termsq[i, child_index(col, DEPTH - 1, a)]
    occupied4 = np.zeros((pos3.shape[0], g.nbins), dtype=bool)
    for b in range(g.nbins):
        occupied4[:, b] = (bins3[:, members] == b).any(axis=1)
    best_a4 = S.argmin(axis=2)
    totals = np.take_along_axis(S, best_a4[:, :, None], axis=2)[:, :, 0].sum(axis=1)
    p = int(totals.argmin())
    table = dec[p].copy()
    table[DEPTH - 1] = np.where(occupied4[p], best_a4[p], np.array(g.fallback))
    support = {
        "prefix_index": p,
        "n_prefixes": int(pos3.shape[0]),
        "bins_unoccupied_at_step4": [
            g.bin_names[b] for b in range(g.nbins) if not occupied4[p, b]
        ],
    }
    return table, float(totals[p] / len(members)), support


def fold_occupancy(g: Group, table: np.ndarray, members: np.ndarray) -> np.ndarray:
    occ = np.zeros((DEPTH, g.nbins), dtype=bool)
    for i in members:
        _, bins, _ = g.walk(table, int(i))
        for t, b in enumerate(bins):
            occ[t, b] = True
    return occ


def score_held_out(g: Group, table: np.ndarray, occ: np.ndarray, i: int) -> tuple[float, int]:
    idx, fallbacks = 0, 0
    for t in range(DEPTH):
        b = int(g.binv[i, idx])
        if occ[t, b]:
            a = int(table[t, b])
        else:
            a = int(g.fallback[b])
            fallbacks += 1
        idx = int(child_index(np.int64(idx), t, a))
    return float(g.termsq[i, idx]), fallbacks


def open_loop_values(cells: list[Cell]) -> np.ndarray:
    """[n_cells, 4^DEPTH] terminal squared error of every fixed action sequence."""
    out = np.zeros((len(cells), N_ACTIONS ** DEPTH), dtype=np.float64)
    for i, cell in enumerate(cells):
        for s, seq in enumerate(np.ndindex(*([N_ACTIONS] * DEPTH))):
            idx = 0
            for t, a in enumerate(seq):
                idx = int(child_index(np.int64(idx), t, a))
            out[i, s] = cell.termsq[idx]
    return out


class ClosedLoopFitter:
    """One group's tree, pre-enumerated once and reused by every fold."""

    def __init__(self, cells: list[Cell], nbins: int) -> None:
        self.g = Group(cells, nbins)
        self.pos3, self.dec = enumerate_prefixes(self.g)
        self.S_all, self.bins3 = step4_sums(self.g, self.pos3)

    def fit(self, members: np.ndarray):
        return solve_closed_loop(
            self.g, self.pos3, self.dec, self.S_all, self.bins3, members
        )


def closed_loop_scores(cells: list[Cell], nbins: int, loo: bool) -> dict:
    """Terminal squared error per cell under the exact optimum fitted with/without it."""
    fitter = ClosedLoopFitter(cells, nbins)
    g = fitter.g
    src = np.array([c.source_id for c in cells])
    out: dict[str, float] = {}
    fallbacks = 0
    prefixes = int(fitter.pos3.shape[0])
    if not loo:
        members = np.arange(g.n)
        table, _, _ = fitter.fit(members)
        occ = fold_occupancy(g, table, members)
        for i in range(g.n):
            sq, fb = score_held_out(g, table, occ, i)
            out[cells[i].text_id] = sq
            fallbacks += fb
        return {"sq": out, "fallbacks": fallbacks, "n_prefixes": prefixes}
    for i in range(g.n):
        members = np.flatnonzero(src != src[i])
        table, _, _ = fitter.fit(members)
        occ = fold_occupancy(g, table, members)
        sq, fb = score_held_out(g, table, occ, i)
        out[cells[i].text_id] = sq
        fallbacks += fb
    return {"sq": out, "fallbacks": fallbacks, "n_prefixes": prefixes}


def open_loop_scores(cells: list[Cell], loo: bool) -> dict:
    """Best fixed (source level, target level, step) schedule, fitted with/without each cell."""
    vals = open_loop_values(cells)
    src = np.array([c.source_id for c in cells])
    lvl = np.array([round(c.ell[0]) for c in cells])
    out: dict[str, float] = {}
    backoff = 0
    for i in range(len(cells)):
        members = np.flatnonzero(src != src[i]) if loo else np.arange(len(cells))
        same = members[lvl[members] == lvl[i]]
        if same.size == 0:
            same = members
            backoff += 1
        out[cells[i].text_id] = float(vals[i, int(vals[same].sum(axis=0).argmin())])
    return {"sq": out, "backoff_to_pooled": backoff}


def step_invariant_values(cells: list[Cell], nbins: int) -> np.ndarray:
    """[n_cells, 4^nbins] terminal squared error of every STEP-INVARIANT bin policy.

    This is the class section 2.5 originally signed for the closed loop, and the form
    the 4-bin secondary reading keeps: the step-dependent 4-bin class would need
    256^3 = 16.7M prefixes, which is not enumerable at this scale, so a secondary that
    claimed to be step-dependent would be a search result, not a supremum.
    """
    g = Group(cells, nbins)
    out = np.zeros((g.n, N_ACTIONS ** nbins), dtype=np.float64)
    for s, combo in enumerate(np.ndindex(*([N_ACTIONS] * nbins))):
        table = np.tile(np.array(combo, dtype=np.int8), (DEPTH, 1))
        for i in range(g.n):
            idx, _, _ = g.walk(table, i)
            out[i, s] = g.termsq[i, idx]
    return out


def step_invariant_scores(cells: list[Cell], nbins: int, loo: bool) -> dict:
    vals = step_invariant_values(cells, nbins)
    src = np.array([c.source_id for c in cells])
    out: dict[str, float] = {}
    for i in range(len(cells)):
        members = np.flatnonzero(src != src[i]) if loo else np.arange(len(cells))
        out[cells[i].text_id] = float(vals[i, int(vals[members].sum(axis=0).argmin())])
    return {"sq": out, "n_policies": int(vals.shape[1]), "step_invariant": True}


def superset_scores(cells: list[Cell], nbins: int, loo: bool, pooled: dict | None = None) -> dict:
    """DIAGNOSTIC ONLY (kill_criterion 2.6 (2)): closed loop also conditioned on source level."""
    lvl = {c.text_id: round(c.ell[0]) for c in cells}
    out: dict[str, float] = {}
    fallbacks = backoff = 0
    if pooled is None:
        pooled = closed_loop_scores(cells, nbins, loo)
    for level in sorted(set(lvl.values())):
        sub = [c for c in cells if lvl[c.text_id] == level]
        if len({c.source_id for c in sub}) < 2:
            for c in sub:
                out[c.text_id] = pooled["sq"][c.text_id]
                backoff += 1
            continue
        got = closed_loop_scores(sub, nbins, loo)
        out.update(got["sq"])
        fallbacks += got["fallbacks"]
    return {"sq": out, "fallbacks": fallbacks, "backoff_to_pooled": backoff}


def per_source_rmse(cells: list[Cell], sq: dict[str, float]) -> dict[str, float]:
    acc: dict[str, list[float]] = collections.defaultdict(list)
    for cell in cells:
        acc[cell.source_id].append(sq[cell.text_id])
    return {s: math.sqrt(sum(v) / len(v)) for s, v in acc.items()}


def paired_delta(cells: list[Cell], closed: dict, open_: dict) -> tuple[list[str], list[float]]:
    """Per source: RMSE(open) - RMSE(closed).  Positive = closed loop is better."""
    rc = per_source_rmse(cells, closed)
    ro = per_source_rmse(cells, open_)
    keys = sorted(rc)
    return keys, [ro[s] - rc[s] for s in keys]


def bootstrap_ci(values: list[float], draws: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def summarise(cells: list[Cell], closed: dict, open_: dict, judged: bool) -> dict:
    keys, deltas = paired_delta(cells, closed["sq"], open_["sq"])
    mean = sum(deltas) / len(deltas)
    lo, hi = bootstrap_ci(deltas, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    sd = statistics_stdev(deltas)
    mde = MDE_Z * sd / math.sqrt(len(deltas))
    pooled = lambda d: math.sqrt(sum(d[c.text_id] for c in cells) / len(cells))
    return {
        "judged": judged,
        "n_sources": len(keys),
        "n_cells": len(cells),
        "rmse_closed_pooled": pooled(closed["sq"]),
        "rmse_open_pooled": pooled(open_["sq"]),
        "delta_mean": mean,
        "delta_ci95": [lo, hi],
        "delta_sd_across_sources": sd,
        "mde_reported_not_judged": mde,
        "effect_over_mde": abs(mean) / mde if mde > 0 else None,
        "ci_contains_zero": lo <= 0.0 <= hi,
        "sign_convention": "RMSE(open) - RMSE(closed); positive = closed loop better",
        "closed_loop_support": {k: v for k, v in closed.items() if k != "sq"},
        "open_loop_support": {k: v for k, v in open_.items() if k != "sq"},
    }


def statistics_stdev(values: list[float]) -> float:
    n = len(values)
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def greedy_reactive_rmse(cells: list[Cell], nbins: int) -> float:
    g = Group(cells, nbins)
    table = np.tile(np.array(g.fallback, dtype=np.int8), (DEPTH, 1))
    total = 0.0
    for i in range(g.n):
        idx, _, _ = g.walk(table, i)
        total += g.termsq[i, idx]
    return math.sqrt(total / g.n)


def fixed_ladder_rmse(cells: list[Cell]) -> dict:
    """Plan 5.4's `fixed_ladder`, read off the tree: step_down until nominally arrived.

    The point of having it is that it is OPEN LOOP AND UNFITTED, so putting it beside
    `greedy_reactive` (closed loop, unfitted) compares the two information sets with the
    estimator taken out of the comparison -- which is what the leave-one-out margin
    cannot do on its own, since the two classes do not overfit equally.
    """
    total = 0.0
    steps = []
    for cell in cells:
        k = max(0, min(DEPTH, round(cell.ell[0]) - cell.lstar))
        idx = 0
        for t in range(DEPTH):
            a = ACTIONS.index("step_down") if t < k else ACTIONS.index("copy")
            idx = int(child_index(np.int64(idx), t, a))
        total += cell.termsq[idx]
        steps.append(k)
    return {
        "rmse": math.sqrt(total / len(cells)),
        "mean_descent_steps": sum(steps) / len(steps),
        "note": "open loop, unfitted; `one_shot` has no counterpart in this action set "
                "(no jump-to-level action), the same mismatch dp_path has",
    }


def run_variant(cells: list[Cell], nbins: int) -> dict:
    closed_loo = closed_loop_scores(cells, nbins, loo=True)
    open_loo = open_loop_scores(cells, loo=True)
    closed_ins = closed_loop_scores(cells, nbins, loo=False)
    open_ins = open_loop_scores(cells, loo=False)
    sup_loo = superset_scores(cells, nbins, loo=True, pooled=closed_loo)
    signed_loo = step_invariant_scores(cells, nbins, loo=True)
    out = {
        "leave_one_source_out": summarise(cells, closed_loo, open_loo, judged=True),
        "in_sample_reported_not_judged": summarise(cells, closed_ins, open_ins, judged=False),
        "superset_diagnostic_not_judged": summarise(cells, sup_loo, open_loo, judged=False),
        "as_originally_signed_not_judged": summarise(cells, signed_loo, open_loo, judged=False),
        "greedy_reactive_rmse": greedy_reactive_rmse(cells, nbins),
        "bins": BIN_SCHEMES[nbins][1],
    }
    out["as_originally_signed_not_judged"]["what_it_answers"] = (
        "what section 2.5's step-invariant closed-loop class would have returned against "
        "the same step-dependent open-loop class; the gap to the verdict row IS the "
        "asymmetry section 2.6 repaired"
    )
    out["superset_diagnostic_not_judged"]["what_it_answers"] = (
        "value of the source-level axis, which the verdict's closed-loop class does not "
        "see directly; kill_criterion 2.6 (2) forbids using it to change the verdict"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", default="outputs/tsar_cefr_gpu2a_branch/tree.jsonl")
    ap.add_argument("--out-dir", default="outputs/tsar_cefr_d25")
    ap.add_argument("--tables-only", action="store_true",
                    help="write only the IN-SAMPLE fitted policy tables (descriptive, never judged)")
    args = ap.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; outputs are append-only, pick a new path")
    cells = load_tree(pathlib.Path(args.tree))
    cap = cap_pressure(cells)
    kept = [c for c in cells.values() if c.text_id not in cap["items_over_limit"]]
    allc = list(cells.values())

    if args.tables_only:
        tables = {"note": "IN-SAMPLE fits, descriptive only; the verdict is leave-one-source-out",
                  "bins": list(BIN_SCHEMES[3][1]), "actions": list(ACTIONS), "targets": {}}
        for tgt in sorted({c.target for c in kept}):
            sub = [c for c in kept if c.target == tgt]
            fitter = ClosedLoopFitter(sub, 3)
            table, mse, support = fitter.fit(np.arange(len(sub)))
            tables["targets"][tgt] = {
                "n_cells": len(sub),
                "in_sample_mse": mse,
                "policy_by_step": {
                    f"step{t+1}": {BIN_SCHEMES[3][1][b]: ACTIONS[int(table[t, b])]
                                   for b in range(3)} for t in range(DEPTH)},
                "support": support,
            }
        tables["unfitted_references"] = {
            "greedy_reactive_closed_loop": greedy_reactive_rmse(kept, 3),
            "fixed_ladder_open_loop": fixed_ladder_rmse(kept),
        }
        out_dir.mkdir(parents=True)
        (out_dir / "policy_tables.json").write_text(
            json.dumps(tables, indent=2, ensure_ascii=False))
        print(json.dumps(tables["unfitted_references"], indent=2))
        return 0

    report = {
        "gate": "D-2.5 causal upper bound",
        "criterion": "kill_criterion.md sec 2.5 as revised by sec 2.6 (both classes step-dependent)",
        "tree": str(args.tree),
        "cap_gate": cap,
        "variants": {},
    }
    report["variants"]["excluded_3bin"] = run_variant(kept, 3)
    report["variants"]["all_cells_3bin"] = run_variant(allc, 3)
    sec = step_invariant_scores(kept, 4, loo=True)
    report["variants"]["excluded_4bin_secondary_step_invariant"] = summarise(
        kept, sec, open_loop_scores(kept, loo=True), judged=False
    )
    report["variants"]["excluded_4bin_secondary_step_invariant"]["what_it_answers"] = (
        "the 4-bin secondary reading, kept in its originally signed STEP-INVARIANT form: "
        "the step-dependent 4-bin class needs 256^3 = 16.7M prefixes and is not enumerable "
        "here, and a secondary that could only be searched would not be a supremum"
    )

    judged = report["variants"]["excluded_3bin"]["leave_one_source_out"]
    alt = report["variants"]["all_cells_3bin"]["leave_one_source_out"]
    report["verdict"] = {
        "fires": judged["ci_contains_zero"],
        "consequence": "close the tsar_cefr line" if judged["ci_contains_zero"] else "D-2.5 not triggered",
        "judged_on": "excluded_3bin / leave_one_source_out",
        "agrees_with_unexcluded_variant": judged["ci_contains_zero"] == alt["ci_contains_zero"],
    }
    report["choices_made_by_this_script"] = [
        "source level = round(source_level_expected); the plan names the axis, not the coding",
        "per-source RMSE paired across the two classes, then averaged over sources; the "
        "bootstrap unit is source_id as signed, matching D-2",
        "sign convention RMSE(open) - RMSE(closed), positive = closed loop better, matching "
        "the defense precedent's +0.0469 *; the verdict (CI contains 0) is sign-invariant",
        "open loop with an empty (source level, target) cell in a fold backs off to the "
        "target-level pooled fit, mirroring the closed loop's unvisited-bin fallback; counted",
        "ties in the exact argmin are broken by first index; equivalent prefixes give the "
        "same objective by construction",
    ]
    report["reporting_obligations"] = [
        "SINGLE REALIZATION: greedy decoding, action axis enumerated not sampled, so there is "
        "no seed axis. Uncertainty is over sources. Per .claude/global.md this number may not "
        "be quoted as a cross-arm headline result in mean +- std (n seeds) form.",
        "Readout is a fixed classifier, not ground truth; carry the G-T1 caption (2 of 40 "
        "pairs reversed: 03-b1 a classifier misread, 12-b1 a data property, both in the "
        "denominator).",
        "Every number here is computed on 3582/3600 GPU-1 rows' admission (51-a2 dropped) and "
        "on the tree's own cap gate; the excluded item is named in cap_gate.",
    ]
    out_dir.mkdir(parents=True)
    (out_dir / "d25_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["verdict"], indent=2))
    print(json.dumps(judged, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
