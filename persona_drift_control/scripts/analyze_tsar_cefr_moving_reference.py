#!/usr/bin/env python3
"""Candidate B's P1 probe: is there planning headroom when the reference MOVES?

Zero GPU. Reads the 67,660 rows of GPU-2a's depth-4 tree
(`outputs/tsar_cefr_gpu2a_branch/tree.jsonl`) and re-scores them against a
TIME-VARYING reference. No model is loaded, no text is generated, no readout is
re-run, and nothing under `outputs/` is written or overwritten.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT.

  G_plan = J(best myopic policy) - J(best anticipating policy)

is the quantity `tsar_cefr_postmortem_and_next_tasks_2026-09-16.md` section 2.6
says should have gated GPU-2 and did not. Under the CONSTANT reference the
tree was bought for, the answer is already known to be small (the closed-loop
supremum sits 0.0130 above the hand-written `greedy_reactive`). Candidate B
asks whether a reference that MOVES mid-trajectory opens that gap, because a
moving reference is the only mechanism on this task that can make anticipation
pay: the action set is one-directional (`step_down`, `half_step_down`,
`paraphrase`, `copy` -- there is no step UP) and one step buys about -0.13, so
a target that drops later must be started on early or it cannot be reached.

  THE PROMPT CARRIES THE TARGET, SO THIS IS THE PLANT-SIDE QUESTION ONLY.
  `tsar_cefr_actions.user_message` opens every step with "The reader you are
  writing for is at CEFR level {target_cefr}", so the tree holds two SEPARATE
  trees per source -- one all-A2, one all-B1 -- and a trajectory that was told
  B1 for two steps and A2 thereafter WAS NEVER GENERATED. Splicing the B1 tree
  into the A2 tree at depth 2 would continue from text the A2 prompt never
  produced. This script therefore moves the reference in the COST ONLY: the
  prompt keeps saying the cell's own target, and the schedule r_t says which
  level we score against at step t.

  That makes this a NECESSARY-CONDITION probe, not candidate B itself. If the
  plant's own dynamics admit no planning gain when the scored reference moves,
  announcing the switch in the prompt cannot create one out of nothing, and
  candidate B dies here for zero GPU. If the gain IS there, candidate B is not
  established either -- it earns a GPU arm that regenerates the tree with
  switching prompts. The probe is one-directional on purpose; it is the same
  shape of instrument as Table 1's row 3.

THE OBJECTIVE IS A RUNNING COST, AND THAT IS A DEPARTURE. D-2.5 scores terminal
squared error on `level_official` (the argmax label). Reference TRACKING needs
a cost at every step, so this script uses

    J = mean_t (ell_t - r_t)^2  over t = 1..4,  on `level_expected`,

which is a DIFFERENT metric from the published one. Its numbers are therefore
not comparable to Table 2's RMSE or to the 0.0130 headroom, and the Table 2
MDE of 0.095 does not transfer. An MDE on THIS metric is derived in-place from
the paired per-source deltas (same z = 2.80 as the D-2.5 analyzer).

THE TWO POLICY CLASSES ARE THE D-2.5 CLASSES, SPLIT BY WHETHER THEY SEE t.

  myopic       pi: bin(d_t) -> action, STEP-INVARIANT. Sees the current error
               against the CURRENT r_t and nothing else; it cannot know a
               switch is coming. 4^3 = 64 tables, brute-forced exactly.
  anticipating pi: (bin(d_t), t) -> action, STEP-DEPENDENT. The step index is
               where knowledge of the schedule enters. Solved exactly by the
               prefix enumeration of `analyze_tsar_cefr_d25.enumerate_prefixes`
               (dedup by position vector is sound for a running cost too,
               because a node index encodes its whole action path, so equal
               position vectors imply equal intermediate positions).

Bin edges are verbatim plan section 5.4's `greedy_reactive` (far >= 1.0,
near [0.5, 1.0), arrived < 0.5) and are NOT fitted, so "myopic" has a direct
reading: the best any static per-error rule can do. Unoccupied bins fall back
to the bin's namesake action, as in D-2.5.

THE PLANTED CONTROL IS THE CONSTANT SCHEDULE. The same two classes are solved
against r_t == ell* at every step. If the switching schedule's G_plan is not
clearly above the constant schedule's, the moving reference bought nothing and
the mechanism claim in postmortem section 2.2 fails on its own task.

LEAVE ONE SOURCE OUT, because in-sample bounds on this line have overstated by
11x before (kill_criterion.md section 3 (1)). Both classes are fitted on 99
sources and scored on the held-out one, for all 100; the in-sample number is
reported beside it and does not judge. Uncertainty is over SOURCES (bootstrap
by `source_id`, 2000 draws), not over seeds -- per .claude/global.md this is a
GATE VERDICT and may not be quoted as a cross-arm headline in mean +- std (n).
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from persona_drift.run_provenance import provenance  # noqa: E402
from analyze_tsar_cefr_d25 import (  # noqa: E402
    BIN_SCHEMES,
    DEPTH,
    N_ACTIONS,
    N_NODES,
    OFFSETS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    MDE_Z,
    bootstrap_ci,
    child_index,
    load_tree,
)

SCHEDULES = {
    "constant": (0.0, 0.0, 0.0, 0.0),
    "switch_at_3": (0.0, 0.0, -1.0, -1.0),
}


def parent_index(idx: int, depth: int) -> int:
    return OFFSETS[depth - 1] + (idx - OFFSETS[depth]) // N_ACTIONS


def path_positions(idx: np.ndarray, depth: int = DEPTH) -> list[np.ndarray]:
    """Positions at depths 1..`depth` for a node given at `depth` (idx encodes its path).

    `depth` is explicit because the anticipating solver hands this function
    DEPTH-1 positions (the prefix end), not terminal ones; defaulting it to DEPTH
    and passing a depth-3 array silently walks off the wrong offsets.
    """
    out = [idx]
    cur = idx
    for d in range(depth, 1, -1):
        cur = OFFSETS[d - 1] + (cur - OFFSETS[d]) // N_ACTIONS
        out.append(cur)
    return out[::-1]


class MovingRefGroup:
    """Cells sharing one fitted policy, flattened, with a per-step reference."""

    def __init__(self, cells, nbins: int, offsets: tuple[float, ...]) -> None:
        binner, self.bin_names, self.fallback = BIN_SCHEMES[nbins]
        self.nbins = nbins
        self.cells = cells
        self.n = len(cells)
        self.offsets = offsets
        # r[i, t] for t = 1..DEPTH; step t is scored against r[:, t-1]
        self.r = np.array([[c.lstar + o for o in offsets] for c in cells], dtype=np.float64)
        self.ell = np.array([c.ell for c in cells], dtype=np.float64)
        # cost[i, idx] is only meaningful once we know which step idx sits at
        self.depth_of = np.zeros(N_NODES, dtype=np.int8)
        for d in range(1, DEPTH + 1):
            self.depth_of[OFFSETS[d] : OFFSETS[d] + (N_ACTIONS ** d)] = d
        self.cost = np.zeros((self.n, N_NODES), dtype=np.float64)
        self.binv = np.zeros((self.n, N_NODES), dtype=np.int8)
        for idx in range(N_NODES):
            d = int(self.depth_of[idx])
            # the bin AT a node is the error the policy sees when deciding the
            # NEXT step, so it uses the next step's reference
            r_here = self.r[:, min(max(d, 1), DEPTH) - 1] if d >= 1 else self.r[:, 0]
            r_next = self.r[:, min(d, DEPTH - 1)]
            self.cost[:, idx] = (self.ell[:, idx] - r_here) ** 2 if d >= 1 else 0.0
            for i in range(self.n):
                self.binv[i, idx] = binner(float(self.ell[i, idx] - r_next[i]))

    def walk_cost(self, table: np.ndarray, i: int, step_invariant: bool) -> float:
        idx, total = 0, 0.0
        for t in range(DEPTH):
            b = int(self.binv[i, idx])
            a = int(table[b] if step_invariant else table[t, b])
            idx = int(child_index(np.int64(idx), t, a))
            total += float(self.cost[i, idx])
        return total / DEPTH

    def walk_cost_occ(self, table: np.ndarray, occ: np.ndarray, i: int,
                      step_invariant: bool) -> tuple[float, int]:
        idx, total, fb = 0, 0.0, 0
        for t in range(DEPTH):
            b = int(self.binv[i, idx])
            seen = occ[b] if step_invariant else occ[t, b]
            if seen:
                a = int(table[b] if step_invariant else table[t, b])
            else:
                a = int(self.fallback[b])
                fb += 1
            idx = int(child_index(np.int64(idx), t, a))
            total += float(self.cost[i, idx])
        return total / DEPTH, fb


def solve_myopic(g: MovingRefGroup, members: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact: brute-force all nbins^... step-invariant tables (4^3 = 64 for 3 bins)."""
    best, best_j = None, np.inf
    for combo in itertools.product(range(N_ACTIONS), repeat=g.nbins):
        table = np.array(combo, dtype=np.int8)
        j = float(np.mean([g.walk_cost(table, int(i), True) for i in members]))
        if j < best_j:
            best, best_j = table, j
    return best, best_j


def enumerate_prefixes_running(g: MovingRefGroup) -> tuple[np.ndarray, np.ndarray]:
    """Steps 1..DEPTH-1 decision prefixes, deduplicated by induced position vector.

    Same contract as the D-2.5 analyzer's, restated here because this group holds
    a different cost array; the dedup key is unchanged and is sound for a running
    cost because a node index encodes its entire action path.
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
    return pos, dec


def solve_anticipating(g: MovingRefGroup, pos3: np.ndarray, dec: np.ndarray,
                       members: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact minimiser of the mean running cost over `members`, step-dependent table."""
    sub = pos3[:, members]
    # running cost of steps 1..DEPTH-1, recovered from the prefix-end position
    pre = np.zeros(pos3.shape[0], dtype=np.float64)
    for arr in path_positions(sub, DEPTH - 1):
        pre += g.cost[members[None, :], arr].sum(axis=1)
    bins3 = g.binv[members[None, :], sub]
    S = np.zeros((pos3.shape[0], g.nbins, N_ACTIONS), dtype=np.float64)
    for a in range(N_ACTIONS):
        vals = g.cost[members[None, :], child_index(sub, DEPTH - 1, a)]
        for b in range(g.nbins):
            S[:, b, a] = np.where(bins3 == b, vals, 0.0).sum(axis=1)
    occupied4 = np.zeros((pos3.shape[0], g.nbins), dtype=bool)
    for b in range(g.nbins):
        occupied4[:, b] = (bins3 == b).any(axis=1)
    best_a4 = S.argmin(axis=2)
    totals = pre + np.take_along_axis(S, best_a4[:, :, None], axis=2)[:, :, 0].sum(axis=1)
    p = int(totals.argmin())
    table = dec[p].copy()
    table[DEPTH - 1] = np.where(occupied4[p], best_a4[p], np.array(g.fallback))
    return table, float(totals[p] / (len(members) * DEPTH))


def occ_myopic(g: MovingRefGroup, table: np.ndarray, members: np.ndarray) -> np.ndarray:
    occ = np.zeros(g.nbins, dtype=bool)
    for i in members:
        idx = 0
        for t in range(DEPTH):
            b = int(g.binv[i, idx])
            occ[b] = True
            idx = int(child_index(np.int64(idx), t, int(table[b])))
    return occ


def occ_anticipating(g: MovingRefGroup, table: np.ndarray, members: np.ndarray) -> np.ndarray:
    occ = np.zeros((DEPTH, g.nbins), dtype=bool)
    for i in members:
        idx = 0
        for t in range(DEPTH):
            b = int(g.binv[i, idx])
            occ[t, b] = True
            idx = int(child_index(np.int64(idx), t, int(table[t, b])))
    return occ


def brute_force_anticipating(g: MovingRefGroup, members: np.ndarray,
                             occupied_bins: list[int] | None = None) -> float:
    """Reference implementation: every step-dependent table, no solver involved.

    Unoccupied bins are pinned to their fallback, exactly as the solver leaves
    them, so the enumeration is N_ACTIONS^(DEPTH * |occupied|) rather than
    N_ACTIONS^(DEPTH * nbins) -- 4^8 instead of 4^12 on a two-bin fixture.
    Only tractable for tiny groups; it exists to pin the word 'exact'.
    """
    if occupied_bins is None:
        occupied_bins = sorted({int(b) for b in g.binv[members].ravel()})
    best = np.inf
    for flat in itertools.product(range(N_ACTIONS), repeat=DEPTH * len(occupied_bins)):
        table = np.tile(np.array(g.fallback, dtype=np.int8), (DEPTH, 1))
        k = 0
        for t in range(DEPTH):
            for b in occupied_bins:
                table[t, b] = flat[k]
                k += 1
        j = float(np.mean([g.walk_cost(table, int(i), False) for i in members]))
        best = min(best, j)
    return best


def greedy_reactive_cost(g: MovingRefGroup, members: np.ndarray) -> float:
    table = np.array(g.fallback, dtype=np.int8)
    return float(np.mean([g.walk_cost(table, int(i), True) for i in members]))


def run_schedule(cells_by_target, name: str, nbins: int, verify: bool) -> dict:
    per_source_my: dict[str, list[float]] = collections.defaultdict(list)
    per_source_an: dict[str, list[float]] = collections.defaultdict(list)
    in_sample = {}
    fallbacks = {"myopic": 0, "anticipating": 0}
    tables = {}
    for target, cells in sorted(cells_by_target.items()):
        g = MovingRefGroup(cells, nbins, SCHEDULES[name])
        sources = sorted({c.source_id for c in cells})
        all_idx = np.arange(g.n)
        pos3, dec = enumerate_prefixes_running(g)
        t_my, j_my = solve_myopic(g, all_idx)
        t_an, j_an = solve_anticipating(g, pos3, dec, all_idx)
        in_sample[target] = {
            "myopic": j_my, "anticipating": j_an, "G_plan": j_my - j_an,
            "greedy_reactive": greedy_reactive_cost(g, all_idx),
            "n_prefixes": int(pos3.shape[0]),
        }
        tables[target] = {"myopic": t_my.tolist(), "anticipating": t_an.tolist()}
        if verify:
            small = np.array([i for i, c in enumerate(cells)
                              if c.source_id in sources[:3]], dtype=np.int64)
            exact = solve_anticipating(g, pos3, dec, small)[1]
            brute = brute_force_anticipating(g, small)
            assert abs(exact - brute) < 1e-12, (
                f"{target}/{name}: prefix solver {exact} != brute force {brute}")
        for s in sources:
            held = np.array([i for i, c in enumerate(cells) if c.source_id == s], dtype=np.int64)
            keep = np.setdiff1d(all_idx, held)
            tm, _ = solve_myopic(g, keep)
            ta, _ = solve_anticipating(g, pos3, dec, keep)
            om, oa = occ_myopic(g, tm, keep), occ_anticipating(g, ta, keep)
            for i in held:
                cm, fm = g.walk_cost_occ(tm, om, int(i), True)
                ca, fa = g.walk_cost_occ(ta, oa, int(i), False)
                per_source_my[s].append(cm)
                per_source_an[s].append(ca)
                fallbacks["myopic"] += fm
                fallbacks["anticipating"] += fa
    srcs = sorted(set(per_source_my) & set(per_source_an))
    deltas = [float(np.mean(per_source_my[s]) - np.mean(per_source_an[s])) for s in srcs]
    g_plan = float(np.mean(deltas))
    lo, hi = bootstrap_ci(deltas, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    se = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
    return {
        "schedule": name,
        "offsets": SCHEDULES[name],
        "n_sources": len(srcs),
        "loo": {
            "J_myopic": float(np.mean([np.mean(per_source_my[s]) for s in srcs])),
            "J_anticipating": float(np.mean([np.mean(per_source_an[s]) for s in srcs])),
            "G_plan": g_plan,
            "ci95": [lo, hi],
            "se": se,
            "mde": MDE_Z * se,
            "passes_2x_mde": bool(g_plan >= 2.0 * MDE_Z * se and lo > 0.0),
        },
        "in_sample": in_sample,
        "tables": tables,
        "fallback_reads": fallbacks,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", type=pathlib.Path,
                    default=pathlib.Path("outputs/tsar_cefr_gpu2a_branch/tree.jsonl"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--bins", type=int, default=3, choices=sorted(BIN_SCHEMES))
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the brute-force cross-check of the exact solver")
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    cells = load_tree(args.tree)
    for c in cells.values():
        if np.isnan(c.ell).any():
            raise ValueError(f"{c.text_id}: level_expected missing at some node")
    by_target = collections.defaultdict(list)
    for c in cells.values():
        by_target[c.target].append(c)

    report = {
        "provenance": provenance(),
        "tree": str(args.tree),
        "metric": "J = mean_t (level_expected_t - r_t)^2 over t=1..4 (RUNNING cost)",
        "not_comparable_to": ("Table 2 RMSE and the 0.0130 headroom, which score terminal "
                              "squared error on level_official; the Table 2 MDE 0.095 does "
                              "not transfer to this metric"),
        "prompt_limitation": ("the tree's action prompts name the cell's own target level, so "
                              "the reference moves in the COST ONLY; an announced switch was "
                              "never generated. Necessary-condition probe, not candidate B"),
        "reporting_caliber": ("gate verdict; uncertainty is over sources (bootstrap by "
                              "source_id), not seeds. Per .claude/global.md may not be quoted "
                              "as a cross-arm headline in mean +- std (n seeds) form"),
        "bins": args.bins,
        "cells": len(cells),
        "schedules": {},
    }
    for name in SCHEDULES:
        report["schedules"][name] = run_schedule(by_target, name, args.bins,
                                                 verify=not args.no_verify)
    c = report["schedules"]["constant"]["loo"]
    s = report["schedules"]["switch_at_3"]["loo"]
    report["verdict"] = {
        "G_plan_constant": c["G_plan"],
        "G_plan_switching": s["G_plan"],
        "lift_from_moving_reference": s["G_plan"] - c["G_plan"],
        "switching_passes_2x_mde": s["passes_2x_mde"],
        "reading": ("candidate B earns a GPU arm only if the switching schedule clears "
                    "2x MDE with a CI excluding zero AND clearly exceeds the constant "
                    "schedule's G_plan (the planted control)"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    for name, r in report["schedules"].items():
        L = r["loo"]
        print(f"\n[{name}] LOO  J_myopic={L['J_myopic']:.4f}  J_antic={L['J_anticipating']:.4f}"
              f"  G_plan={L['G_plan']:+.4f}  CI95=[{L['ci95'][0]:+.4f}, {L['ci95'][1]:+.4f}]"
              f"  MDE={L['mde']:.4f}  2xMDE={'PASS' if L['passes_2x_mde'] else 'FAIL'}")
        for t, v in r["in_sample"].items():
            print(f"   in-sample {t}: myopic={v['myopic']:.4f} antic={v['anticipating']:.4f}"
                  f" G={v['G_plan']:+.4f} greedy_reactive={v['greedy_reactive']:.4f}"
                  f" prefixes={v['n_prefixes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
