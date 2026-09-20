#!/usr/bin/env python3
"""Generate the modeling-line ablation figures from the evidence ledger.

Same rule as build_tables.py: nothing in figs/fig_*.pdf is typed by hand. Every plotted
point is read out of evidence/numbers.yaml and the ids that fed each panel are written to
figs/_fig_report.json so a mark can always be traced back to one evidence id.

Figures (2026-09-19, side experiments P1 and P2 of claude/side_experiments_plan_2026-09-19.md):
  fig_memory_depth.pdf   P1  paired Skill_H gain of a deeper memory state over a single reading,
                             matched-window caliber (paper_side_experiments_results.md section 1.2)
  fig_item_effect.pdf    P2  what the memory gain decomposes into: an item intercept versus the
                             remainder beyond it (section 3)

Usage:
    /scratch/hcao2/envs/research-template/bin/python tools/build_figs.py

matplotlib is not in the persona_drift_pilot env; research-template has it.

To look at a typeset page (no pdftoppm, pdftocairo, gs or ImageMagick on this host),
rasterise main.pdf with the typst that ships inside quarto:

    T=~/.local/share/quarto-1.10.18/bin/tools/x86_64/typst
    printf '#set page(width: 8.5in, height: 11in, margin: 0pt)\n#image("main.pdf", page: 9, width: 100%%)\n' > p.typ
    $T compile --format png --ppi 120 p.typ page_9.png
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
LEDGER = os.path.join(PAPER, "evidence", "numbers.yaml")
FIGS = os.path.join(PAPER, "figs")

# Paper-facing column names, same vocabulary as build_tables.py PRED_COLUMNS / tab_prediction.
NAMES = {
    "sentence_length_t10": "Sentence len.",
    "vector_count_stage2_t10": "Joint-3",
    "vector_count_stage1_t10": "Joint-2",
    "sentiment_t5": "Sentiment",
    "defense": "Defense",
    "constraint": "Constraint",
    "tsar_cefr": "CEFR",
}
TABLE2 = ["sentence_length_t10", "vector_count_stage2_t10", "vector_count_stage1_t10",
          "sentiment_t5", "defense", "constraint", "tsar_cefr"]

# Categorical palette, fixed slot order (dataviz skill reference palette, validated 2026-09-19).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "pdf.fonttype": 42,
})


def load():
    with open(LEDGER, encoding="utf-8") as f:
        rows = yaml.safe_load(f)
    return [r for r in rows if r.get("status") != "superseded"]


def task_of(row):
    return os.path.basename(str(row.get("source_run", ""))).replace(".json", "")


def pick(rows, run_dir, task, pattern):
    """One ledger row whose source_run sits in run_dir/task.json and whose `what` matches."""
    # core runs live in results/<run_dir>/<task>.json; behavioral and CEFR runs in
    # persona_drift_control/outputs/<run_dir>_behavioral/ or <run_dir>_tsar_cefr/.
    path_re = re.compile(re.escape(run_dir) + r"(_behavioral|_tsar_cefr)?/" + re.escape(task) + r"\.json$")
    hits = [r for r in rows
            if path_re.search(str(r.get("source_run", "")))
            and re.search(pattern, r["what"])]
    if len(hits) != 1:
        raise SystemExit(f"expected one row for {run_dir}/{task} ~ /{pattern}/, got {len(hits)}: "
                         f"{[h['id'] for h in hits]}")
    return hits[0]


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.5)


# ---------------------------------------------------------------------------
# P1. Memory depth: paired gain over a single reading, matched caliber.
# ---------------------------------------------------------------------------
def fig_memory_depth(rows, report):
    run = "ablation_memory_depth_matched"
    # (task, x = readings in the state, pattern for "that depth minus single reading")
    sweep = {
        "sentence_length_t10":     ("lag {k} minus lag 0", "matched caliber, primary anchor", [1, 2, 3]),
        "vector_count_stage2_t10": ("lag {k} minus lag 0", "matched caliber, primary anchor", [1, 2, 3]),
        "vector_count_stage1_t10": ("lag {k} minus lag 0", "matched caliber, primary anchor", [1, 2, 3]),
        "constraint":              ("nu {k} minus nu 1",   "matched caliber",                 [2, 3, 4]),
    }
    two_cell = {  # designs with two depths only: the single extra reading
        "sentiment_t5": ("lag 1 minus lag 0", "matched caliber, primary anchor"),
        "defense":      ("nu 2 minus nu 1",   "matched caliber"),
        "tsar_cefr":    ("lag 1 minus lag 0", "matched caliber"),
    }
    used = {}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.25), gridspec_kw={"width_ratios": [3.0, 1.35]})
    slot = 0
    for task, (pat, anchor, ks) in sweep.items():
        xs, ys, lo, hi, ids = [1], [0.0], [0.0], [0.0], []
        for k in ks:
            r = pick(rows, run, task, re.escape(anchor) + r".*" + re.escape(pat.format(k=k)) + r" =")
            depth = k + 1 if pat.startswith("lag") else k
            xs.append(depth); ys.append(r["value"]); lo.append(r["ci"][0]); hi.append(r["ci"][1])
            ids.append(r["id"])
        used[task] = ids
        c, m = PALETTE[slot], MARKERS[slot]
        # Same encoding as panel (b) and fig_item_effect: filled marker = the paired gain over a
        # single reading has a 95% interval excluding zero; hollow = interval contains zero.
        ax.plot(xs, ys, color=c, linewidth=1.6, zorder=2, label=NAMES[task])
        for x, y, l, h in zip(xs, ys, lo, hi):
            excl = (x > 1) and ((l > 0) or (h < 0))
            ax.plot(x, y, marker=m, markersize=5, color=c if excl else "white",
                    markeredgecolor=c, markeredgewidth=1.0, zorder=3)
        dy = {"constraint": 5, "vector_count_stage2_t10": -5}.get(task, 0)
        ax.annotate(NAMES[task], (xs[-1], ys[-1]), xytext=(5, dy), textcoords="offset points",
                    va="center", ha="left", fontsize=7, color=INK)
        slot += 1
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 5.0)
    ax.set_xlabel("Past readings in the state (filled = interval excludes 0)")
    ax.set_ylabel(r"$\mathrm{Skill}_H$ gain over one reading")
    ax.set_title("(a) Depth sweep, matched training window", loc="left", fontsize=8)
    style_axes(ax)

    # (b) two-depth designs.
    labels, vals, los, his = [], [], [], []
    for task, (pat, anchor) in two_cell.items():
        r = pick(rows, run, task, re.escape(anchor) + r".*" + re.escape(pat) + r" =")
        used[task] = [r["id"]]
        labels.append(NAMES[task]); vals.append(r["value"]); los.append(r["ci"][0]); his.append(r["ci"][1])
    ypos = list(range(len(labels)))[::-1]
    ax2.axvline(0, color=MUTED, linewidth=0.6, zorder=1)
    for y, v, l, h in zip(ypos, vals, los, his):
        excl = (l > 0) or (h < 0)
        ax2.plot([l, h], [y, y], color=INK, linewidth=0.8, zorder=2)
        ax2.plot(v, y, marker="o", markersize=5, color=INK if excl else "white",
                 markeredgecolor=INK, markeredgewidth=0.8, zorder=3)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Gain from one extra reading")
    ax2.set_title("(b) Two-depth designs", loc="left", fontsize=8)
    ax2.set_ylim(-0.6, len(labels) - 0.4)
    style_axes(ax2)
    ax2.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax2.grid(False, axis="y")

    fig.tight_layout(w_pad=1.5)
    out = os.path.join(FIGS, "fig_memory_depth.pdf")
    fig.savefig(out)
    plt.close(fig)
    report["fig_memory_depth"] = {"file": os.path.relpath(out, PAPER), "caliber": "matched window",
                                  "ids": used}


# ---------------------------------------------------------------------------
# P2. Item intercept: what the memory gain decomposes into.
# ---------------------------------------------------------------------------
def fig_item_effect(rows, report):
    run = "mechanism_item_effect"
    series = [  # (label, regex on `what`, palette slot, marker)
        ("Memory gain: ours $-$ Last-turn", r"ours minus markov =", 0, "o"),
        ("Bought by an item intercept", r"markov plus item minus markov =", 1, "s"),
        ("Remainder: ours $-$ (Last-turn + intercept)", r"ours minus markov plus item =", 2, "^"),
    ]
    data = {}
    for task in TABLE2:
        data[task] = {}
        for label, pat, _, _ in series:
            r = pick(rows, run, task, pat)
            data[task][label] = r
    order = sorted(TABLE2, key=lambda t: -data[t][series[0][0]]["value"])
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    off = [0.24, 0.0, -0.24]
    ypos = {t: i for i, t in enumerate(order[::-1])}
    ax.axvline(0, color=MUTED, linewidth=0.6, zorder=1)
    for t in order:
        ax.axhline(ypos[t] - 0.5, color=GRID, linewidth=0.4, zorder=0)
    used = {}
    for (label, pat, slot, m), o in zip(series, off):
        c = PALETTE[slot]
        first = True
        for t in order:
            r = data[t][label]
            used.setdefault(t, []).append(r["id"])
            y = ypos[t] + o
            lo, hi = r["ci"]
            v = r["value"]
            excl = (lo > 0) or (hi < 0)
            ax.plot([lo, hi], [y, y], color=c, linewidth=1.0, zorder=2)
            ax.plot(v, y, marker=m, markersize=4.8, color=c if excl else "white",
                    markeredgecolor=c, markeredgewidth=0.9, zorder=3,
                    label=label if first else None)
            first = False
    ax.set_yticks([ypos[t] for t in order])
    ax.set_yticklabels([NAMES[t] for t in order])
    ax.set_xlabel(r"Paired $\mathrm{Skill}_H$ difference (95% interval; filled = interval excludes 0)")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.legend(loc="lower right", frameon=False, handlelength=1.2)
    style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.grid(False, axis="y")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_item_effect.pdf")
    fig.savefig(out)
    plt.close(fig)
    report["fig_item_effect"] = {"file": os.path.relpath(out, PAPER), "caliber": "matched window",
                                 "row_order": [NAMES[t] for t in order], "ids": used}



# ---------------------------------------------------------------------------
# Mechanism figures (2026-09-20). Three scatters, one per mechanism of
# contract.yaml's mechanism_lattice that the main text asserts without showing:
#   fig_two_futures.pdf       m1  one reading is consistent with two opposite next steps
#   fig_command_channel.pdf   m5  the applied command does not vary with the state
#   fig_lifting_contrast.pdf  m4  what a learned lifting costs, against trajectory count
# A and B read the bundled trajectory files directly, not the evidence ledger, because
# they display the collected data rather than an estimate. Provenance for those two is
# the file sha256 plus the transform, both written to _fig_report.json. C reads the
# ledger like every other figure. Neither A nor B is a Skill_H number and neither may be
# quoted as one (.claude/global.md -> report caliber).
# ---------------------------------------------------------------------------
import collections
import hashlib

DATASETS = os.path.join(os.path.dirname(PAPER), "datasets")

# Table 1 columns whose raw trajectories ship in datasets/. The other three columns
# (Defense, Constraint, CEFR) are collected elsewhere and have no file here.
RAW = {
    "sentence_length_t10": ("scalar/sentence_length_t10", "normalized_output"),
    "vector_count_stage2_t10": ("vector/vector_count_stage2_t10", "word_count_norm"),
    "vector_count_stage1_t10": ("vector/vector_count_stage1_t10", "word_count_norm"),
    "sentiment_t5": ("scalar/sentiment_t5", "normalized_output"),
}


def trajectories(task):
    """Group one bundled dataset into turn-ordered trajectories; return (rows, sha256)."""
    path = os.path.join(DATASETS, RAW[task][0], "trajectories.jsonl")
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    grouped = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            grouped[row["trajectory_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["turn"])
    return grouped, sha, os.path.relpath(path, os.path.dirname(PAPER))


def binned_mean(xs, ys, edges):
    """Mean of ys inside each x bin; bins holding under 5 points are dropped."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [y for x, y in zip(xs, ys) if lo <= x < hi]
        if len(vals) >= 5:
            out.append(((lo + hi) / 2, sum(vals) / len(vals)))
    return [p[0] for p in out], [p[1] for p in out]


# ---------------------------------------------------------------------------
# m1. One reading, two futures.
# ---------------------------------------------------------------------------
def fig_two_futures(report):
    tasks = ["sentence_length_t10", "vector_count_stage2_t10",
             "vector_count_stage1_t10", "sentiment_t5"]
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.8), sharey=False)
    ids = {}
    for ax, task in zip(axes, tasks):
        grouped, sha, rel = trajectories(task)
        col = RAW[task][1]
        up_x, up_y, dn_x, dn_y, ties = [], [], [], [], 0
        for rows in grouped.values():
            series = [r.get(col) for r in rows]
            for i in range(1, len(series) - 1):
                a, b, c = series[i - 1], series[i], series[i + 1]
                if None in (a, b, c):
                    continue
                prev, nxt = b - a, c - b
                if prev == 0:
                    ties += 1
                elif prev > 0:
                    up_x.append(b); up_y.append(nxt)
                else:
                    dn_x.append(b); dn_y.append(nxt)
        ax.axhline(0, color=GRID, linewidth=0.6, zorder=1)
        for xs, ys, colour, marker in ((up_x, up_y, PALETTE[0], "o"),
                                       (dn_x, dn_y, PALETTE[1], "^")):
            ax.scatter(xs, ys, s=2.5, c=colour, marker=marker, alpha=0.22,
                       linewidths=0, zorder=2)
        lo = min(up_x + dn_x); hi = max(up_x + dn_x)
        edges = [lo + (hi - lo) * i / 8 for i in range(9)]
        for xs, ys, colour, marker in ((up_x, up_y, PALETTE[0], "o"),
                                       (dn_x, dn_y, PALETTE[1], "^")):
            bx, by = binned_mean(xs, ys, edges)
            ax.plot(bx, by, color=colour, linewidth=1.3, marker=marker,
                    markersize=3.2, zorder=3,
                    label="previous step up" if colour == PALETTE[0] else "previous step down")
        style_axes(ax)
        ax.set_title(NAMES[task], pad=3)
        ax.set_xlabel("reading $y_t$")
        gap = (sum(dn_y) / len(dn_y)) - (sum(up_y) / len(up_y))
        ids[task] = {"file": rel, "sha256": sha, "column": col,
                     "n_up": len(up_x), "n_down": len(dn_x), "n_flat_dropped": ties,
                     "mean_next_change_up": sum(up_y) / len(up_y),
                     "mean_next_change_down": sum(dn_y) / len(dn_y),
                     "separation_down_minus_up": gap}
    axes[0].set_ylabel("next change $y_{t+1}-y_t$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.09), handlelength=1.6, columnspacing=1.6)
    fig.tight_layout(pad=0.4)
    out = os.path.join(FIGS, "fig_two_futures.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    report["fig_two_futures"] = {
        "file": os.path.relpath(out, PAPER), "mechanism": "m1",
        "caliber": "descriptive, raw readings -- NOT Skill_H, never quote as a result number",
        "transform": "per trajectory: x=y_t, y=y_{t+1}-y_t, grouped by sign(y_t-y_{t-1}); "
                     "exact ties on the previous step dropped; line overlay = mean of y in 8 "
                     "equal-width x bins, bins under 5 points dropped",
        "sources": ids}
    return out


# ---------------------------------------------------------------------------
# m5. The command channel does not move.
# ---------------------------------------------------------------------------
def fig_command_channel(report):
    tasks = ["sentence_length_t10", "sentiment_t5"]  # the Table 1 columns logging a per-turn command
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1))
    ids = {}
    for ax, task in zip(axes, tasks):
        grouped, sha, rel = trajectories(task)
        xs, ys, targets, same = [], [], [], 0
        for rows in grouped.values():
            for a, b in zip(rows, rows[1:]):
                u, r_, y = b.get("requested_norm"), a.get("effective_norm"), a.get("normalized_output")
                if None in (u, r_, y):
                    continue
                xs.append(r_ - y); ys.append(u); targets.append(r_)
                same += abs(u - r_) < 1e-9
        sc = ax.scatter(xs, ys, s=3.5, c=targets, cmap="Blues", vmin=-0.45,
                        alpha=0.55, linewidths=0, zorder=2)
        ax.axvline(0, color=GRID, linewidth=0.6, zorder=1)
        style_axes(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=0.5)
        ax.set_title(NAMES[task], pad=3)
        ax.set_xlabel("tracking error $r-y_t$")
        ax.annotate(f"{same}/{len(xs)} transitions\nhave command $=r$",
                    xy=(0.04, 0.04), xycoords="axes fraction", fontsize=6.5,
                    color=MUTED, ha="left", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                              edgecolor="none", alpha=0.85))
        ids[task] = {"file": rel, "sha256": sha, "n_transitions": len(xs),
                     "n_command_equals_target": same}
    axes[0].set_ylabel("applied command $u_{t+1}$")
    cb = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("target $r$", size=7); cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.4)
    out = os.path.join(FIGS, "fig_command_channel.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    report["fig_command_channel"] = {
        "file": os.path.relpath(out, PAPER), "mechanism": "m5",
        "caliber": "descriptive, raw collection log -- NOT Skill_H, never quote as a result number",
        "transform": "per consecutive turn pair: x=effective_norm-normalized_output at t, "
                     "y=requested_norm at t+1, colour=effective_norm",
        "note": "Joint-2 / Joint-3 log no per-turn scalar command; their command channel was "
                "checked by the rank test in docs/experiments/core_multiobjective_planning_headroom.md 2.1",
        "sources": ids}
    return out


# ---------------------------------------------------------------------------
# m4. What the learned lifting costs, against how much structure the readout has.
# ---------------------------------------------------------------------------
# Ids are the paired Ours - AE-Koopman contrast for the seven columns
# build_tables.py::PRED_COLUMNS prints. x is the readout dimension, the half of m4 the
# data supports; the trajectory count rides along as an annotation because the other
# half of m4 -- that the cost is estimation variance at small n -- is not what the
# numbers show (CEFR carries the largest cost on the largest task).
# (task, ledger id, readout dimension, x offset inside its category, label offset in pt)
LIFTING = [("sentence_length_t10", "n_main_015", 1, -0.36, (-27, -17)),
           ("vector_count_stage2_t10", "n_main_047", 3, 0.0, (-20, 9)),
           ("vector_count_stage1_t10", "n_main_031", 2, 0.0, (-18, 9)),
           ("sentiment_t5", "n_main_111", 1, 0.0, (-33, -3)),
           ("defense", "n_main_151", 1, -0.18, (3, -18)),
           ("constraint", "n_main_135", 1, 0.18, (4, -3)),
           ("tsar_cefr", "n_main_183", 1, 0.36, (-13, 9))]


def fig_lifting_contrast(rows, report):
    by = {r["id"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(4.6, 2.0))
    ax.axhline(0, color=MUTED, linewidth=0.7, zorder=1)
    ids = {}
    for task, rid, dim, dx, offset in LIFTING:
        r = by[rid]
        n, v, ci = r["n"], r["value"], r["ci"]
        excl = ci[0] > 0 or ci[1] < 0
        caveated = r.get("status") == "caveated"
        x = dim + dx
        ax.plot([x, x], ci, color=MUTED, linewidth=0.7, zorder=2)
        ax.scatter([x], [v], s=26, marker="o", zorder=3,
                   facecolor=(PALETTE[0] if excl else "white"),
                   edgecolor=(MUTED if caveated else PALETTE[0]),
                   linewidths=(1.1 if caveated else 0.9))
        dagger = "$^{\\dagger}$" if caveated else ""
        ax.annotate(NAMES[task] + dagger + "\n$n{=}$" + str(n),
                    xy=(x, v), xytext=offset, textcoords="offset points",
                    fontsize=6.2, color=INK, linespacing=1.0)
        ids[task] = {"id": rid, "n_trajectories": n, "value": v, "ci": ci,
                     "readout_dim": dim, "interval_excludes_zero": bool(excl),
                     "status": r.get("status")}
    style_axes(ax)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1-d", "2-d", "3-d"])
    ax.set_xlim(0.40, 3.45)
    ax.set_xlabel("dimension of the attribute readout")
    ax.set_ylabel("$\\mathrm{Skill}_H$: linear $-$ learned lifting")
    fig.tight_layout(pad=0.4)
    out = os.path.join(FIGS, "fig_lifting_contrast.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    report["fig_lifting_contrast"] = {
        "file": os.path.relpath(out, PAPER), "mechanism": "m4",
        "caliber": "paired grouped bootstrap, evidence ledger",
        "marker_rule": "filled = 95% interval excludes zero; grey edge + dagger = caveated id; "
                       "x position inside a category is a fixed spacing offset, not data",
        "not_shown": "the cost does not fall with the trajectory count: CEFR (n=100) carries "
                     "the largest cost and Joint-3 (n=90) the smallest. m4's estimation-variance "
                     "clause is unsupported and was removed from the mechanism lattice.",
        "ids": ids}
    return out


def main():
    rows = load()
    os.makedirs(FIGS, exist_ok=True)
    report = {}
    fig_memory_depth(rows, report)
    fig_item_effect(rows, report)
    fig_two_futures(report)
    fig_command_channel(report)
    fig_lifting_contrast(rows, report)
    with open(os.path.join(FIGS, "_fig_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
