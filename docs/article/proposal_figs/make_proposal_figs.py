#!/usr/bin/env python3
"""Figures 1-4 for the NSF MFAI proposal (plan: docs/article/proposal_integration_plan_2026-11-05.md §二).

Nothing plotted is typed by hand: Figs 2-3 read paper/evidence/numbers.yaml by id (read-only;
paper/ is not modified), Fig 4 reads results/ and the #4b trajectories. Every id or file used is
written to proposal/figs/_proposal_fig_report.json.

Sizes are print sizes (NSF letter, 6.5 in text width), so font sizes are printed points.
Palette = dataviz reference slots 1-4 (validated light mode; slots 3-4 below 3:1 -> direct labels).

    /scratch/hcao2/envs/dkvar/bin/python docs/article/proposal_figs/make_proposal_figs.py [fig1 fig2 fig3 fig4]
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "proposal" / "figs"
LEDGER = REPO / "paper" / "evidence" / "numbers.yaml"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
OLD, INK, MUTED, GRID, FAINT = "#9a9a9a", "#1d1d1d", "#5f5f5f", "#dedede", "#c9c9c9"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
TASKS = ["sentence_length_t10", "vector_count_stage2_t10", "vector_count_stage1_t10",
         "sentiment_t5", "defense", "constraint", "tsar_cefr"]
# Joint-2 = word count + average word length; Joint-3 adds comma count (paper/sections/04a_setup.tex).
PLAIN = {"sentence_length_t10": "Sentence\nlength", "vector_count_stage2_t10": "Joint\n(3 attr.)",
         "vector_count_stage1_t10": "Joint\n(2 attr.)", "sentiment_t5": "Sentiment",
         "defense": "Jailbreak\ndefense", "constraint": "Constraint\nfollowing", "tsar_cefr": "Simpli-\nfication"}
# Table 1 (paper/tables/tab_prediction.tex) ids: Last-turn row, KOMPAS (linear) row, paired contrast Ours - Markov.
LAST = ["n_main_006", "n_main_038", "n_main_022", "n_main_102", "n_main_139", "n_main_123", "n_main_171"]
MEM = ["n_main_004", "n_main_036", "n_main_020", "n_main_100", "n_main_147", "n_main_131", "n_main_179"]
GAIN = ["n_main_013", "n_main_045", "n_main_029", "n_main_109", "n_main_148", "n_main_132", "n_main_180"]
# Ours - AE-Koopman (linear minus learned lifting), as in paper Fig. mechanism (c).
LIFT = ["n_main_015", "n_main_047", "n_main_031", "n_main_111", "n_main_151", "n_main_135", "n_main_183"]
REPORT: dict = {}


def ledger() -> dict[str, dict]:
    rows = yaml.safe_load(open(LEDGER, encoding="utf-8"))
    return {r["id"]: r for r in rows if r.get("status") != "superseded"}


def task_check(row: dict, task: str) -> dict:
    hay = row["what"] + " " + str(row.get("source_run", ""))
    if task not in hay:
        raise SystemExit(f"{row['id']} is not about {task}: {row['what'][:80]}")
    return row


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.5, colors=MUTED, labelcolor=INK)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1 framework (schematic)
def box(ax, x, y, w, h, title, body, edge=INK, face="white", dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", lw=0.9,
                                ec=edge, fc=face, ls=(0, (3, 2)) if dashed else "-", zorder=2))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 7.4, body, ha="center", va="top", fontsize=7, color=MUTED, zorder=3, linespacing=1.25)


def arrow(ax, p, q, color=INK, dashed=False, rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 ls=(0, (3, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=1))


def fig1():
    fig = plt.figure(figsize=(6.5, 2.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 40); ax.axis("off")
    y, h = 15, 18
    box(ax, 1, y, 27, h, "Interaction history", "prompts and replies\nso far in the\nconversation")
    box(ax, 35, y, 27, h, r"Behavioral state $s_k$", "a few measured\nattributes of recent\nreplies (e.g., lengths)")
    box(ax, 69, y, 27, h, "Controlled linear\nmodel (2.5)", "", edge=BLUE)
    ax.text(82.5, y + 5.6, r"$z_{k+1}=Az_k+B_p c^p_k+\sum_\ell B_\ell u^{(\ell)}_k$", ha="center", va="center",
            fontsize=7, color=INK, zorder=3)
    box(ax, 103, y, 26, h, "Future behavior", "predicted attributes\nof the next replies")
    for (a, b), lab in zip(((28, 35), (62, 69), (96, 103)), ("compress", "encode", "predict")):
        arrow(ax, (a + 0.3, y + h / 2), (b - 0.3, y + h / 2))
        ax.text((a + b) / 2, y + h / 2 + 1.8, lab, ha="center", fontsize=6.3, color=MUTED)
    # closed loop: the next reply joins the history
    arrow(ax, (116, y + h + 0.3), (14.5, y + h + 0.3), color=MUTED, rad=0.12, lw=0.8)
    ax.text(65.5, 35.6, "next turn: the new reply joins the history", ha="center", va="center", fontsize=6.3, color=MUTED)
    # two input channels into the model box
    arrow(ax, (68, 6.5), (78, y - 0.3), color=ORANGE, lw=1.3)
    arrow(ax, (97, 6.5), (87, y - 0.3), color=OLD, dashed=True, lw=1.3)
    ax.text(67, 7.6, r"Prompt $c^p_k$ (instruction text)", ha="right", va="center", fontsize=7.2, color=INK)
    ax.text(67, 3.8, "realized in preliminary work (Sec. 3.2)", ha="right", va="center", fontsize=6.3, color=MUTED)
    ax.text(98, 7.6, r"Activation change $u^{(\ell)}_k$", ha="left", va="center", fontsize=7.2, color=INK)
    ax.text(98, 3.8, "internal layers; proposed (Thrusts A.3, C.2)", ha="left", va="center", fontsize=6.3, color=MUTED)
    REPORT["fig1"] = {"kind": "schematic", "data": None}
    save(fig, "fig1_framework")


# ---------------------------------------------------------------- Fig. 2 what memory buys
def fig2():
    L = ledger()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.05), gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.28})
    x = np.arange(len(TASKS)); w = 0.36
    used = []
    for i, t in enumerate(TASKS):
        last, mem, gain = (task_check(L[ids[i]], t) for ids in (LAST, MEM, GAIN))
        excl = gain["ci"][0] > 0 or gain["ci"][1] < 0
        ax.bar(x[i] - w / 2 - 0.01, last["value"], w, color=OLD, zorder=2)
        ax.bar(x[i] + w / 2 + 0.01, mem["value"], w, color=BLUE if excl else "white", ec=BLUE, lw=0.9, zorder=2)
        used.append({"task": t, "last_turn": last["id"], "memory": mem["id"], "contrast": gain["id"], "excludes_zero": excl})
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x, [PLAIN[t] for t in TASKS], fontsize=6.3)
    ax.set_ylabel(r"Prediction skill (Skill$_H$)")
    ax.set_ylim(-0.12, 1.0); style(ax)
    ax.legend(handles=[Patch(color=OLD, label="current reading only"),
                       Patch(color=BLUE, label="last few readings, gain's 95% interval excludes 0"),
                       Patch(fc="white", ec=BLUE, lw=0.9, label="last few readings, interval contains 0")],
              fontsize=6.3, frameon=False, loc="upper right", handlelength=1.2)
    ax.set_title("(a) Memory vs. current reading", fontsize=8, loc="left")
    # (b) depth sweep, matched training window
    sweep = {"sentence_length_t10": ("lag {k} minus lag 0", "matched caliber, primary anchor", [1, 2, 3]),
             "vector_count_stage2_t10": ("lag {k} minus lag 0", "matched caliber, primary anchor", [1, 2, 3]),
             "vector_count_stage1_t10": ("lag {k} minus lag 0", "matched caliber, primary anchor", [1, 2, 3]),
             "constraint": ("nu {k} minus nu 1", "matched caliber", [2, 3, 4])}
    run = "ablation_memory_depth_matched"
    depth_ids = {}
    for (t, (pat, anchor, ks)), c in zip(sweep.items(), (BLUE, ORANGE, AQUA, YELLOW)):
        path_re = re.compile(re.escape(run) + r"(_behavioral|_tsar_cefr)?/" + re.escape(t) + r"\.json$")
        xs, ys, ids, excl = [1], [0.0], [], [False]
        for k in ks:
            hits = [r for r in L.values() if path_re.search(str(r.get("source_run", "")))
                    and re.search(re.escape(anchor) + r".*" + re.escape(pat.format(k=k)) + r" =", r["what"])]
            if len(hits) != 1:
                raise SystemExit(f"depth {t} k={k}: {len(hits)} rows")
            r = hits[0]; ids.append(r["id"])
            xs.append(k + 1 if pat.startswith("lag") else k); ys.append(r["value"])
            excl.append(r["ci"][0] > 0 or r["ci"][1] < 0)
        depth_ids[t] = {"ids": ids, "excludes_zero": excl[1:]}
        bx.plot(xs, ys, color=c, lw=1.4, zorder=2)
        for xx, yy, e in zip(xs, ys, excl):
            bx.plot(xx, yy, "o", ms=3.8, color=c if e or xx == 1 else "white", mec=c, mew=0.9, zorder=3)
        lab_xy = {"vector_count_stage2_t10": (2.0, ys[1] + 0.016, "center")}.get(t, (xs[-1] + 0.12, ys[-1] - (0.012 if t == "constraint" else 0), "left"))
        bx.text(lab_xy[0], lab_xy[1], PLAIN[t].replace("\n", " "), fontsize=6.2, va="center", ha=lab_xy[2], color=INK)
    bx.axhline(0, color=MUTED, lw=0.6)
    bx.set_xticks([1, 2, 3, 4]); bx.set_xlim(0.85, 6.3); bx.set_ylim(-0.01, 0.27)
    bx.set_xlabel("Past readings in the state"); bx.set_ylabel("Gain over one reading")
    bx.text(0.98, 0.003, "hollow = interval contains 0", fontsize=6.0, color=MUTED, ha="right", va="bottom", transform=bx.get_yaxis_transform())
    style(bx); bx.set_title("(b) How many readings", fontsize=8, loc="left")
    REPORT["fig2"] = {"a": used, "b_depth_ids": depth_ids, "ledger": str(LEDGER.relative_to(REPO))}
    save(fig, "fig2_memory")


# ---------------------------------------------------------------- Fig. 3 learned lifting
def fig3():
    L = ledger()
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    used = []
    dims = {"vector_count_stage2_t10": "D", "vector_count_stage1_t10": "s"}
    for i, t in enumerate(TASKS):
        r = task_check(L[LIFT[i]], t)
        lo, hi = r["ci"]; excl = lo > 0 or hi < 0
        y = len(TASKS) - 1 - i
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.2, zorder=2)
        ax.plot(r["value"], y, marker=dims.get(t, "o"), ms=5, color=BLUE if excl else "white", mec=BLUE, mew=1.0, zorder=3)
        used.append({"task": t, "id": r["id"], "value": r["value"], "ci": r["ci"], "excludes_zero": excl})
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(TASKS)), [PLAIN[t].replace("\n", " ") for t in reversed(TASKS)], fontsize=6.6)
    ax.set_xlabel("Skill difference: linear minus learned lifting")
    ax.text(1.0, 1.02, "right of 0: linear model better; filled = interval excludes 0", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=MUTED)
    style(ax, grid_axis="x")
    REPORT["fig3"] = {"ids": used}
    save(fig, "fig3_lifting")


# ---------------------------------------------------------------- Fig. 4 prompt authority, measured
def fig4():
    spec = importlib.util.spec_from_file_location("reach", REPO / "scripts" / "core_excite_reachability.py")
    reach = importlib.util.module_from_spec(spec); spec.loader.exec_module(reach)
    ident = reach.ident
    summ = json.load(open(REPO / "results" / "core_excite_reachability" / "summary.json"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.2), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.42})
    # (a) held-out test prompts: requested vs delivered words; model line from train fit (per seed, then mean)
    req_all, got_all = [], []
    for seed in ident.SEEDS:
        rows = [json.loads(l) for l in open(REPO / ident.IN_TEMPLATE.format(seed=seed))]
        split = {r["trajectory_id"]: r["topic_split"] for r in rows}
        units = ident.transitions(rows, ident.LAG)
        test = [u for t, u in units.items() if split[t] == "test"]
        req_all.append(reach.words(np.concatenate([u[1] for u in test])))
        got_all.append(reach.words(np.vstack([u[2] for u in test])[:, 0]))
    ax.scatter(np.concatenate(req_all), np.concatenate(got_all), s=2.5, color=FAINT, lw=0, zorder=1, rasterized=True)
    ax.plot([5, 100], [5, 100], color=MUTED, lw=0.7, ls=":", zorder=1)
    centers = [(lo + hi - 1) / 2 for lo, hi in reach.BINS]
    keys = [f"{lo}-{hi - 1}" for lo, hi in reach.BINS]
    pm = [summ["across_seeds"][f"heldout_predicted_{k}"]["mean"] for k in keys]
    om = [summ["across_seeds"][f"heldout_observed_{k}"]["mean"] for k in keys]
    os_ = [summ["across_seeds"][f"heldout_observed_{k}"]["std"] for k in keys]
    ax.plot(centers, pm, color=ORANGE, lw=1.6, marker="s", ms=3.5, zorder=3)
    ax.errorbar(centers, om, yerr=os_, fmt="o", ms=4, color=BLUE, ecolor=BLUE, elinewidth=1, capsize=0, zorder=4)
    ceil = summ["across_seeds"]["ceiling_words_c1"]["mean"]
    ax.axhline(ceil, color=OLD, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.text(2, ceil + 1.5, f"model's reachable average: {ceil:.0f} words", fontsize=6.2, color=MUTED, ha="left", va="bottom")
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="", ms=4, label="held-out prompts (mean, sd over 3 seeds)"),
                       Line2D([], [], color=ORANGE, marker="s", ms=3.5, lw=1.6, label="model fitted on training prompts"),
                       Line2D([], [], color=FAINT, marker="o", ls="", ms=2.5, label="individual held-out replies"),
                       Line2D([], [], color=MUTED, lw=0.7, ls=":", label="delivered = requested")],
              fontsize=6.0, frameon=False, loc="upper left", handlelength=1.6)
    ax.set_xlim(0, 105); ax.set_ylim(0, 135)
    ax.set_xlabel("Requested words"); ax.set_ylabel("Delivered words")
    style(ax); ax.set_title("(a) Sentence length: what the prompt can reach", fontsize=8, loc="left")
    # (b) identified one-step effects, each task on its own scale
    ex = json.load(open(REPO / "results" / "core_excite_b_identifiability" / "summary.json"))
    rows_b = [("Sentence length, seed %d" % s, ex["per_seed"][str(s)]["main_element_CB"]["point"],
               *ex["per_seed"][str(s)]["main_element_CB"]["ci95"], "words per requested word") for s in ident.SEEDS]
    cefr = json.load(open(REPO / "results" / "b_identifiability" / "tsar_cefr.json"))["bootstrap"]["trajectory"]["main_elements_CB"]
    names = ["step down", "half step down", "paraphrase"]
    rows_c = [(f"Simplification: {n}", e["point"], e["ci_low"], e["ci_high"], "CEFR level") for n, e in zip(names, cefr)]
    con = json.load(open(REPO / "results" / "b_identifiability" / "constraint.json"))["bootstrap"]["trajectory"]["main_elements_CB"][0]
    rows_d = [("Constraint: reminder", con["point"], con["ci_low"], con["ci_high"], "satisfaction")]
    bx.axis("off")
    groups = [(rows_b, "sentence length (normalized words)"), (rows_c, "CEFR level"), (rows_d, "constraint satisfaction")]
    sub_h = [0.33, 0.33, 0.14]; top = 0.92
    for (rows, unit), hh in zip(groups, sub_h):
        sub = bx.inset_axes([0.42, top - hh, 0.56, hh - 0.06])
        for j, (lab, p, lo, hi, _) in enumerate(rows):
            yy = len(rows) - 1 - j
            sub.plot([lo, hi], [yy, yy], color=BLUE, lw=1.2); sub.plot(p, yy, "o", ms=3.8, color=BLUE)
            bx.text(0.40, top - 0.03 - (j + 0.5) * (hh - 0.06) / len(rows), lab, transform=bx.transAxes,
                    ha="right", va="center", fontsize=6.2, color=INK)
        sub.axvline(0, color=INK, lw=0.7)
        lo_all = min(min(r[2] for r in rows), 0); hi_all = max(max(r[3] for r in rows), 0)
        pad = 0.15 * (hi_all - lo_all); sub.set_xlim(lo_all - pad, hi_all + pad)
        sub.set_yticks([]); sub.set_ylim(-0.6, len(rows) - 0.4)
        for s in ("top", "right", "left"):
            sub.spines[s].set_visible(False)
        sub.tick_params(axis="x", labelsize=5.8, length=2, width=0.5, colors=MUTED, labelcolor=INK)
        sub.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        sub.set_xlabel(unit, fontsize=5.8, color=MUTED, labelpad=1)
        top -= hh + 0.1
    bx.set_title("(b) One-step instruction effect, 95% interval", fontsize=8, loc="left")
    REPORT["fig4"] = {"a": ["results/core_excite_reachability/summary.json", ident.IN_TEMPLATE],
                      "b": ["results/core_excite_b_identifiability/summary.json",
                            "results/b_identifiability/tsar_cefr.json", "results/b_identifiability/constraint.json"]}
    save(fig, "fig4_prompt_authority")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4"]
    for name in which:
        globals()[name]()
    rp = OUT / "_proposal_fig_report.json"
    old = json.load(open(rp)) if rp.exists() else {}
    old.update(REPORT)
    rp.write_text(json.dumps(old, indent=2, default=str))
    print("wrote", ", ".join(which), "->", OUT)
