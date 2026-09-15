#!/usr/bin/env python3
"""GPU-2 for the `tsar_cefr` line: the five closed-loop arms of plan section 5.4.

100 test sources x 2 target levels x 3 seeds x 5 arms x <= 6 steps.

WHY THE BRANCH TREE DOES NOT REPLACE THIS JOB. GPU-2a enumerated every length-4
action sequence, so four of the five arms can be read off it exactly -- at
HORIZON 4 and in ONE realization. Table 2 is signed at <= 6 steps and at
`mean +- std (n = 3 seeds)`. The tree is depth 4 and greedy, so it satisfies
neither axis. It stays what it was bought for: the exact D-2.5 bound, and a
ground truth the fitted operator's rollouts can be checked against.

WHAT THE SEED AXIS MEANS HERE, AND WHY IT IS A RULING. GPU-1's three seeds drew
ACTION SEQUENCES; these arms are deterministic policies, so the action axis
carries no randomness at all and three seeds under greedy decoding would be
three byte-identical copies. The only remaining source of variation is
DECODING. `.claude/global.md` requires >= 3 seeds for any reportable number, so
either decoding is sampled here -- which breaks byte-comparability with GPU-1
and the tree, both greedy -- or Table 2 has no error bars. This script REFUSES
to run canonical under greedy decoding with more than one seed rather than
quietly emit identical rows.

THE UNSIGNED PARAMETERS ARE GUARDS, NOT DEFAULTS. Plan 5.4 leaves four values
to be fixed on the TRIAL 20 sources: the MeaningBERT admission floor (recorded
as NOT YET SIGNED in kill_criterion 2.5 (3)), the MPC's beta and lambda, and
dp_path's reward matrix. Selection on trial, reporting on test, is the split
.claude/global.md asks for; canonical mode therefore refuses to start until
each is supplied or calibrated, because a report-discipline check that lives in
a document is invisible exactly when it fails.

DP_PATH IS NOT IMPLEMENTABLE WITHOUT A HARNESS RULING, AND THE REASON IS SHARP.
Its policy is a path of CEFR LEVELS with possibly multi-level jumps. This action
set is relative ({step_down, half_step_down, paraphrase, copy}) and names no
level. Restricting the DP to adjacent transitions does not rescue it: with only
adjacent moves the strictly-decreasing path from source to target is UNIQUE, so
`dp_path` collapses into `fixed_ladder` and the signed main contrast
`koopman_mpc - dp_path` loses its reference. The alternative is a prompt that
names an arbitrary level -- which `tsar_cefr_actions.saturation_message` almost
is, except it is restricted to the two target levels -- and that is a harness
change. Neither is taken here on this script's own authority.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "persona_drift_control" / "src"))

from persona_drift import tsar_cefr_actions as actions          # noqa: E402
from persona_drift import tsar_cefr_bank as bank                # noqa: E402
from persona_drift import tsar_cefr_readout as readout          # noqa: E402
from persona_drift.run_provenance import provenance             # noqa: E402

SIGNED_TEMPERATURE = 6.5             # G-T1's readout temperature; never re-fitted here
CAP_LENGTH_MULTIPLIER = 1.5          # GPU-1's rule, unchanged
MIN_MAX_NEW_TOKENS = 128
EXCLUDED_ITEMS = ("51-a2",)          # GPU-1's admission ruling, carried forward
MAX_STEPS = 6                        # plan 5.4: all arms cost at most six steps
MPC_HORIZON = 4                      # plan 5.4
CEFR_CODE = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
OPEN_LOOP_ARMS = ("one_shot", "fixed_ladder", "dp_path")
CLOSED_LOOP_ARMS = ("greedy_reactive", "koopman_mpc")
ARMS = OPEN_LOOP_ARMS + CLOSED_LOOP_ARMS


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-model", required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--operator", type=pathlib.Path,
                   default=pathlib.Path("outputs/tsar_cefr_phase3/operator_fit.json"))
    p.add_argument("--split", default="test", choices=sorted(bank.SPLITS))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--decoding", choices=["greedy", "sample"], default=None,
                   help="RULING REQUIRED in canonical mode: these arms are deterministic "
                        "policies, so greedy decoding makes every seed byte-identical.")
    p.add_argument("--decoding-temperature", type=float, default=None)
    p.add_argument("--stop-on-arrival", choices=["never", "on_copy"], default=None,
                   help="RULING REQUIRED in canonical mode: whether a policy that selects "
                        "`copy` ends the trajectory (cheaper, and what plan 6.1's cost "
                        "estimate assumes) or generates a near-identity step (what the tree "
                        "did, so comparable with it).")
    p.add_argument("--beta", type=float, default=None, help="MPC meaning weight; plan 5.4 fixes it on trial.")
    p.add_argument("--lambda-cost", type=float, default=None, help="MPC per-action cost; plan 5.4 calibrates it on the token axis.")
    p.add_argument("--dp-path-mapping", choices=["adjacent_only", "level_prompt"], default=None)
    p.add_argument("--dp-path-reward", choices=["published", "trial_measured"], default=None)
    p.add_argument("--reward-probe-split", default="trial", choices=sorted(bank.SPLITS))
    p.add_argument("--reward-probe-depth", type=int, default=2)
    p.add_argument("--top-p", type=float, default=0.8, help="Qwen3 non-thinking card default")
    p.add_argument("--top-k", type=int, default=20, help="Qwen3 non-thinking card default")
    p.add_argument("--n-sources", type=int, default=None, help="debug only")
    p.add_argument("--readout-device", type=int, default=0)
    p.add_argument("--readout-batch-size", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mode", choices=["debug", "canonical"], required=True)
    return p.parse_args(argv)


def check_rulings(args: argparse.Namespace) -> None:
    """Every unsigned value is a pre-run raise, never a silent default (.claude/code.md)."""
    if args.mode != "canonical":
        return
    if args.decoding is None:
        raise SystemExit(
            "--decoding is required in canonical mode: with deterministic arm policies the "
            "seed axis lives entirely in decoding, and .claude/global.md requires >= 3 seeds")
    if args.decoding == "greedy" and len(args.seeds) > 1:
        raise SystemExit(
            f"greedy decoding with {len(args.seeds)} seeds would write {len(args.seeds)} "
            "byte-identical copies and report a standard deviation of exactly 0")
    if args.decoding == "sample" and args.decoding_temperature is None:
        raise SystemExit("--decoding-temperature is required when --decoding sample")
    if args.stop_on_arrival is None:
        raise SystemExit("--stop-on-arrival is required in canonical mode (plan 5.4 compares "
                         "arms at equal cost but does not say whether `copy` ends a run)")
    if args.beta is None or args.lambda_cost is None:
        raise SystemExit("--beta and --lambda-cost are required in canonical mode; plan 5.4 "
                         "fixes both on the trial 20 sources")
    if "dp_path" in args.arms:
        if args.dp_path_mapping is None:
            raise SystemExit(
                "--dp-path-mapping is required when dp_path is in the arms: its policy is a "
                "path of CEFR LEVELS and this action set names none")
        if args.dp_path_mapping == "adjacent_only":
            raise SystemExit(
                "--dp-path-mapping adjacent_only makes the strictly-decreasing path unique, "
                "so dp_path becomes fixed_ladder and the signed contrast koopman_mpc - dp_path "
                "loses its reference; choose level_prompt (ruled 2026-09-14) or drop the arm")
        if args.dp_path_reward is None:
            raise SystemExit(
                "--dp-path-reward is required with dp_path: R is a MODEL-DEPENDENT empirical "
                "matrix, so copying the published one imports another model's dynamics")
        if args.dp_path_reward == "published":
            raise SystemExit(
                "the published R matrix is not in this repo -- only its filling scheme "
                "(hit +1, off-by-one +0.5, else -1, min-max to [0,1]). Use trial_measured, "
                "which applies that same scheme to this model on the trial split")


def token_cap(n_source_tokens: int) -> int:
    return max(MIN_MAX_NEW_TOKENS, int(-(-n_source_tokens * CAP_LENGTH_MULTIPLIER // 1)))


def select_items(items, n_sources, mode):
    if n_sources is not None:
        if mode == "canonical":
            raise SystemExit("--n-sources truncates the bank; it is a debug switch")
        keep = set(bank.source_ids(items)[:n_sources])
        items = [item for item in items if item.source_id in keep]
    kept = [item for item in items if item.text_id not in set(EXCLUDED_ITEMS)]
    dropped = sorted({item.text_id for item in items} & set(EXCLUDED_ITEMS))
    if mode == "canonical" and set(dropped) != set(EXCLUDED_ITEMS):
        raise SystemExit(f"expected to exclude {EXCLUDED_ITEMS}, split contains {dropped}")
    return kept, dropped


def greedy_reactive_action(ell: float, target: str) -> str:
    """Plan 5.4's three rules, verbatim. The D-2.5 bins are these same edges."""
    d = ell - CEFR_CODE[target.upper()]
    if d >= 1.0:
        return "step_down"
    if d >= 0.5:
        return "half_step_down"
    return "copy"


def fixed_ladder_actions(source_ell: float, target: str, max_steps: int) -> list[str]:
    k = max(0, min(max_steps, round(source_ell) - CEFR_CODE[target.upper()]))
    return ["step_down"] * k + ["copy"] * (max_steps - k)


LEVELS_LOW_TO_HIGH = actions.CEFR_LEVELS
REWARD_HIT, REWARD_ADJACENT, REWARD_MISS = 1.0, 0.5, -1.0
REWARD_PROBE_CAP = 800


def reward_of(asked: str, landed: str) -> float:
    """The published scheme, applied to this model's own transitions."""
    gap = abs(CEFR_CODE[landed] - CEFR_CODE[asked])
    return REWARD_HIT if gap == 0 else (REWARD_ADJACENT if gap == 1 else REWARD_MISS)


def normalise_rewards(raw: dict[tuple[str, str], list[float]]) -> dict:
    """Mean per ordered pair, then min-max to [0,1] as the paper does.

    A pair nobody attempted stays ABSENT rather than defaulting to anything: the DP
    may not plan a transition this model was never observed making.
    """
    means = {pair: sum(v) / len(v) for pair, v in raw.items()}
    lo, hi = min(means.values()), max(means.values())
    span = hi - lo
    return {
        "R": {f"{j}->{i}": (0.5 if span == 0 else (m - lo) / span)
              for (j, i), m in means.items()},
        "n_attempts": {f"{j}->{i}": len(raw[(j, i)]) for (j, i) in means},
        "raw_mean": {f"{j}->{i}": m for (j, i), m in means.items()},
        "scheme": "hit +1, off-by-one +0.5, else -1; min-max to [0,1]",
        "unobserved_pairs_are_unavailable_to_the_dp": True,
    }


def adjacent_path(source_level: str, target_level: str) -> list[str]:
    return [lvl for lvl in reversed(LEVELS_LOW_TO_HIGH)
            if CEFR_CODE[target_level] <= CEFR_CODE[lvl] < CEFR_CODE[source_level]]


def dp_degeneracy(plans: dict) -> dict:
    """How often the DP's plan is just the ladder -- a property of the OBJECTIVE, not a bug.

    After min-max normalisation every R is non-negative, so a sum over hops rewards
    LONGER paths: three mediocre transitions (0.5 each) outscore one excellent jump
    (1.0). The published objective therefore leans toward the adjacent path unless some
    transition is actively bad. If every plan here is the ladder then `dp_path` IS
    `fixed_ladder` empirically -- not by the action-set collapse this arm was rebuilt to
    avoid, but by the method's own arithmetic on this model's measured R. That is a
    reportable finding and a caption obligation, so it is counted rather than repaired.
    """
    same = sum(1 for p in plans.values() if p["path"] == p["adjacent_path"])
    return {"n_items": len(plans), "n_plans_equal_to_fixed_ladder": same,
            "share": same / len(plans) if plans else None,
            "note": "if this share is 1.0, dp_path and fixed_ladder are the same arm on "
                    "this model and the signed contrast koopman_mpc - dp_path has no "
                    "independent reference; the analyzer's zero-variance guard will fire"}


def dp_plan(R: dict, source_level: str, target_level: str, max_steps: int) -> dict:
    """The strictly-decreasing level path maximising sum R (plan 5.4 / arXiv 2602.07499).

    Enumerated rather than recursed: between two CEFR levels there are at most 2^4
    intermediate subsets, so the DP recursion buys nothing here and enumeration cannot
    silently mis-handle the unavailable-transition case.
    """
    src, tgt = CEFR_CODE[source_level], CEFR_CODE[target_level]
    if src <= tgt:
        return {"path": [], "adjacent_path": [], "total_reward": 0.0, "unobserved_used": 0,
                "note": "source already at or below target"}
    middles = [lvl for lvl in LEVELS_LOW_TO_HIGH if tgt < CEFR_CODE[lvl] < src]
    best, best_score, best_unobs = None, None, None
    for mask in range(1 << len(middles)):
        chosen = [m for k, m in enumerate(middles) if mask >> k & 1]
        chosen.sort(key=lambda lvl: -CEFR_CODE[lvl])
        path = chosen + [target_level]
        if len(path) > max_steps:
            continue
        hops = list(zip([source_level] + path[:-1], path))
        score = sum(R.get(f"{j}->{i}", 0.0) for j, i in hops)
        unobs = sum(1 for j, i in hops if f"{j}->{i}" not in R)
        key = (unobs, -score)          # prefer fully observed, then higher reward
        if best_score is None or key < best_score:
            best, best_score, best_unobs = path, key, unobs
    return {"path": best, "adjacent_path": adjacent_path(source_level, target_level),
            "total_reward": -best_score[1], "unobserved_used": best_unobs}


def probe_reward_matrix(llm, sampling_cls, probes, tokenizer, split: str, depth: int) -> tuple[dict, int]:
    """Fill R on the TRIAL split with this model, by the published scheme.

    SELECTION ON TRIAL, REPORTING ON TEST -- the split .claude/global.md asks for. The
    probe is GREEDY whatever the arms use: it is a measurement of this model's level
    transitions, so it should not carry decoding noise into the DP's plan.

    One text per SOURCE, not per item: a level transition does not depend on which of
    the two task targets the item was written for.
    """
    items = bank.load_tsar_bank(bank.fetch(split))
    seen, texts = set(), []
    for item in items:
        if item.source_id not in seen:
            seen.add(item.source_id)
            texts.append(item.original)
    levels = probes.levels(texts, temperature=SIGNED_TEMPERATURE)
    frontier = [{"text": t, "level": lv.level_official,
                 "cap": token_cap(len(tokenizer.encode(t)))}
                for t, lv in zip(texts, levels)]
    raw: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    attempts = 0
    for _ in range(depth):
        jobs = [{**node, "asked": lvl} for node in frontier for lvl in LEVELS_LOW_TO_HIGH
                if CEFR_CODE[lvl] < CEFR_CODE[node["level"]]]
        if not jobs:
            break
        attempts += len(jobs)
        if attempts > REWARD_PROBE_CAP:
            raise SystemExit(f"reward probe would issue {attempts} generations, over the "
                             f"cap of {REWARD_PROBE_CAP}; lower --reward-probe-depth")
        conversations = [
            [{"role": "system", "content": actions.SYSTEM_PROMPT},
             {"role": "user", "content": actions.level_message(job["text"], job["asked"])}]
            for job in jobs
        ]
        params = [sampling_cls(temperature=0.0, max_tokens=job["cap"], seed=0) for job in jobs]
        outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False})
        produced = [o.outputs[0].text.strip() for o in outputs]
        landed = probes.levels(produced, temperature=SIGNED_TEMPERATURE)
        frontier = []
        for job, text, lvl in zip(jobs, produced, landed):
            raw[(job["level"], job["asked"])].append(reward_of(job["asked"], lvl.level_official))
            frontier.append({"text": text, "level": lvl.level_official, "cap": job["cap"]})
        print(f"  reward probe: {len(jobs)} attempts, {len(raw)} ordered pairs so far", flush=True)
    return normalise_rewards(raw), attempts


class Operator:
    """xi_{t+1} = K xi_t + B u_t + c, with xi = [ell_t, s_t, ell_{t-1}, s_{t-1}]."""

    def __init__(self, path: pathlib.Path) -> None:
        fit = json.loads(path.read_text())["final_operator"]
        self.K = fit["K"]
        self.B = fit["B"]
        self.c = fit["c"]
        self.u_layout = list(fit["u_layout"])
        self.reference_action = "copy"

    def onehot(self, action: str) -> list[float]:
        return [1.0 if action == name else 0.0 for name in self.u_layout]

    def step(self, xi: list[float], action: str) -> list[float]:
        u = self.onehot(action)
        out = []
        for i in range(len(self.K)):
            v = self.c[i] + sum(self.K[i][j] * xi[j] for j in range(len(xi)))
            v += sum(self.B[i][j] * u[j] for j in range(len(u)))
            out.append(v)
        return out


def mpc_action(op: Operator, xi: list[float], target: str, beta: float, lam: float,
               horizon: int = MPC_HORIZON) -> str:
    """Plan 5.4's objective, enumerated over all 4^H sequences -- no approximate planner."""
    star = float(CEFR_CODE[target.upper()])
    best, best_cost = None, None
    def walk(state, seq, cost):
        nonlocal best, best_cost
        if len(seq) == horizon:
            if best_cost is None or cost < best_cost:
                best, best_cost = seq[0], cost
            return
        for action in actions.ACTION_NAMES:
            nxt = op.step(state, action)
            step_cost = (nxt[0] - star) ** 2 + beta * (1.0 - nxt[1])
            step_cost += lam * (0.0 if action == op.reference_action else 1.0)
            walk(nxt, seq + [action], cost + step_cost)
    walk(list(xi), [], 0.0)
    return best


def next_action(arm: str, state: dict, args: argparse.Namespace, op: Operator | None) -> str:
    if arm == "one_shot":
        return "one_shot_prompt" if state["step"] == 0 else "copy"
    if arm == "fixed_ladder":
        return state["schedule"][state["step"]]
    if arm == "dp_path":
        plan = state["dp_plan"]["path"]
        return f"level:{plan[state['step']]}" if state["step"] < len(plan) else "copy"
    if arm == "greedy_reactive":
        return greedy_reactive_action(state["ell"], state["target_cefr"])
    if arm == "koopman_mpc":
        return mpc_action(op, state["xi"], state["target_cefr"], args.beta, args.lambda_cost)
    raise SystemExit(f"arm {arm!r} has no policy in this script")


def render(action: str, text: str, target: str) -> str:
    if action == "one_shot_prompt":
        return actions.saturation_message(text, target)
    if action.startswith("level:"):
        return actions.level_message(text, action.split(":", 1)[1])
    return actions.user_message(text, action, target)


def census(n_items: int, arms, seeds, max_steps: int) -> dict:
    return {"n_items": n_items, "n_arms": len(arms), "n_seeds": len(seeds),
            "max_steps": max_steps,
            "upper_bound_generations": n_items * len(arms) * len(seeds) * max_steps,
            "note": "upper bound; arms that stop on arrival generate fewer"}


def main(argv=None) -> None:
    args = parse_args(argv)
    check_rulings(args)
    if args.out_dir.exists():
        raise SystemExit(f"refusing to write into existing {args.out_dir}")

    items = bank.load_tsar_bank(bank.fetch(args.split))
    kept, dropped = select_items(items, args.n_sources, args.mode)
    plan = census(len(kept), args.arms, args.seeds, args.max_steps)
    print(json.dumps({"kept_items": len(kept), "excluded": dropped, **plan}, indent=2), flush=True)

    op = Operator(args.operator) if "koopman_mpc" in args.arms else None
    if args.dry_run:
        item = kept[0]
        cap = token_cap(len(item.original.split()))
        print(f"first item {item.text_id!r}, proxy cap {cap}")
        # A stand-in R so the DP path is exercised without a GPU. The real one is
        # measured on the trial split at run time; only its VALUES differ, and every
        # other failure on this path -- the plan, the level action, the template -- is
        # reachable here. (GPU-2a died at 2m21s on a field name a narrower dry run
        # could not reach: a check that passes exactly when it is not needed.)
        stand_in_R = {f"{j}->{i}": 0.5 for j in LEVELS_LOW_TO_HIGH for i in LEVELS_LOW_TO_HIGH
                      if CEFR_CODE[i] < CEFR_CODE[j]}
        for arm in args.arms:
            state = {"step": 0, "ell": 3.5, "xi": [3.5, 1.0, 3.5, 1.0],
                     "target_cefr": item.target_cefr,
                     "schedule": fixed_ladder_actions(3.5, item.target_cefr, args.max_steps),
                     "dp_plan": dp_plan(stand_in_R, "B2", item.target_cefr, args.max_steps)}
            act = next_action(arm, state, args, op)
            extra = f"  plan {state['dp_plan']['path']}" if arm == "dp_path" else ""
            print(f"  {arm}: first action {act!r}{extra}")
        if "dp_path" in args.arms:
            print(f"  reward probe: <= {REWARD_PROBE_CAP} generations on the "
                  f"{args.reward_probe_split} split, depth {args.reward_probe_depth}, greedy")
        print("--- one rendered user message (greedy_reactive, step 1) ---")
        print(render(greedy_reactive_action(3.5, item.target_cefr), item.original,
                     item.target_cefr)[:300])
        print("dry run: no model loaded, nothing generated")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.agent_model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enforce_eager=False)
    tokenizer = llm.get_tokenizer()
    probes = readout.CefrProbes(device=args.readout_device, batch_size=args.readout_batch_size)

    calibration = None
    if "dp_path" in args.arms:
        rewards, attempts = probe_reward_matrix(
            llm, SamplingParams, probes, tokenizer, args.reward_probe_split,
            args.reward_probe_depth)
        calibration = {"reward_matrix": rewards, "probe_generations": attempts,
                       "probe_split": args.reward_probe_split,
                       "probe_decoding": "greedy (a measurement, not an arm)"}

    sources = probes.levels([item.original for item in kept], temperature=SIGNED_TEMPERATURE)
    base = {item.text_id: lvl for item, lvl in zip(kept, sources)}
    source_text = {item.text_id: item.original for item in kept}

    live = {}
    for arm in args.arms:
        for seed in args.seeds:
            for item in kept:
                ell0 = base[item.text_id].level_expected
                live[(arm, item.text_id, seed)] = {
                    "arm": arm, "seed": seed, "step": 0, "text_id": item.text_id,
                    "source_id": item.source_id, "target_cefr": item.target_cefr,
                    "text": item.original, "ell": ell0, "s": 1.0,
                    "xi": [ell0, 1.0, ell0, 1.0],
                    "cap": token_cap(len(tokenizer.encode(item.original))),
                    "schedule": fixed_ladder_actions(ell0, item.target_cefr, args.max_steps),
                    "dp_plan": (dp_plan(calibration["reward_matrix"]["R"],
                                        base[item.text_id].level_official,
                                        item.target_cefr, args.max_steps)
                                if arm == "dp_path" else None),
                    "done": False,
                }

    # A STEP-0 ROW FOR EVERY TRAJECTORY, WRITTEN BEFORE ANY GENERATION.
    # Without it, `--stop-on-arrival on_copy` silently deletes trajectories: a policy
    # that picks `copy` at step 1 generates nothing, emits no row, and vanishes from
    # the analyzer's denominator -- so two arms would be scored on different samples
    # and nothing would say so. The source paragraph IS the terminal state of such a
    # trajectory, and this row records it with zero generated tokens.
    rows = [{
        "arm": st["arm"], "seed": st["seed"], "step": 0,
        "text_id": st["text_id"], "source_id": st["source_id"],
        "target_cefr": st["target_cefr"], "action": "start",
        "text_in": None, "text_out": st["text"], "n_tokens_out": 0,
        "hit_token_cap": False,
        "level_expected": base[st["text_id"]].level_expected,
        "level_official": base[st["text_id"]].level_official,
        "meaning_to_source": 1.0,
        "fkgl": readout.fkgl(st["text"]),
        "source_level_expected": base[st["text_id"]].level_expected,
        "source_level_official": base[st["text_id"]].level_official,
    } for st in live.values()]

    for step in range(args.max_steps):
        batch = [st for st in live.values() if not st["done"]]
        if not batch:
            break
        chosen = [next_action(st["arm"], st, args, op) for st in batch]
        if args.stop_on_arrival == "on_copy":
            keep = [i for i, a in enumerate(chosen) if a != "copy"]
            for i, st in enumerate(batch):
                if i not in set(keep):
                    st["done"] = True
            batch = [batch[i] for i in keep]
            chosen = [chosen[i] for i in keep]
        if not batch:
            break
        conversations = [
            [{"role": "system", "content": actions.SYSTEM_PROMPT},
             {"role": "user", "content": render(a, st["text"], st["target_cefr"])}]
            for st, a in zip(batch, chosen)
        ]
        if args.decoding == "sample":
            params = [SamplingParams(temperature=args.decoding_temperature, top_p=args.top_p,
                                     top_k=args.top_k, max_tokens=st["cap"], seed=st["seed"])
                      for st in batch]
        else:
            params = [SamplingParams(temperature=0.0, max_tokens=st["cap"], seed=st["seed"])
                      for st in batch]
        started = time.time()
        outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False})
        texts = [o.outputs[0].text.strip() for o in outputs]
        levels = probes.levels(texts, temperature=SIGNED_TEMPERATURE)
        meanings = probes.meaning([source_text[st["text_id"]] for st in batch], texts)
        for st, action, out, text, level, meaning in zip(batch, chosen, outputs, texts, levels, meanings):
            st["step"] += 1
            rows.append({
                "arm": st["arm"], "seed": st["seed"], "step": st["step"],
                "text_id": st["text_id"], "source_id": st["source_id"],
                "target_cefr": st["target_cefr"], "action": action,
                "text_in": st["text"], "text_out": text,
                "n_tokens_out": len(out.outputs[0].token_ids),
                "hit_token_cap": out.outputs[0].finish_reason == "length",
                "level_expected": level.level_expected,
                "level_official": level.level_official,
                "meaning_to_source": meaning,
                "fkgl": readout.fkgl(text),
                "source_level_expected": base[st["text_id"]].level_expected,
                "source_level_official": base[st["text_id"]].level_official,
            })
            st["xi"] = [level.level_expected, meaning, st["ell"], st["s"]]
            st["text"], st["ell"], st["s"] = text, level.level_expected, meaning
        print(f"  step {step + 1}: {len(batch)} live in {(time.time() - started) / 60:.1f} min",
              flush=True)

    dp_plans = ({tid: live[("dp_path", tid, args.seeds[0])]["dp_plan"]
                 for tid in sorted({i.text_id for i in kept})}
                if "dp_path" in args.arms else None)

    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "tsar_cefr GPU-2 five closed-loop arms",
        "plan": plan, "excluded_items": dropped, "n_rows": len(rows),
        "rulings": {"decoding": args.decoding, "decoding_temperature": args.decoding_temperature,
                    "top_p": args.top_p, "top_k": args.top_k,
                    "stop_on_arrival": args.stop_on_arrival, "beta": args.beta,
                    "lambda_cost": args.lambda_cost, "dp_path_mapping": args.dp_path_mapping,
                    "dp_path_reward": args.dp_path_reward},
        "calibration": calibration,
        "dp_plans": dp_plans, "dp_degeneracy": dp_degeneracy(dp_plans) if dp_plans else None,
        "provenance": provenance({
            "agent_model": args.agent_model, "split": args.split, "arms": list(args.arms),
            "seeds": list(args.seeds), "max_steps": args.max_steps,
            "readout_temperature": SIGNED_TEMPERATURE, "mpc_horizon": MPC_HORIZON,
            "operator": str(args.operator), "excluded_items": list(EXCLUDED_ITEMS)}),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
