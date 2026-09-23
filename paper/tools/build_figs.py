#!/usr/bin/env python3
"""Generate the modeling-line ablation figures from the evidence ledger.

Same rule as build_tables.py: nothing in figs/fig_*.pdf is typed by hand. Every plotted
point is read out of evidence/numbers.yaml and the ids that fed each panel are written to
figs/_fig_report.json so a mark can always be traced back to one evidence id.

Figures (merged 2026-09-20; former fig_memory_depth.pdf and fig_item_effect.pdf are gone):
  fig_memory.pdf       P1+P2  (a,b) paired Skill_H gain of a deeper memory state over a single
                              reading under the matched window; (c) item intercept vs remainder
  fig_two_futures.pdf  m1     next change vs current reading, split by previous step direction
  fig_mechanism.pdf    m5,m4  (a,b) applied command vs tracking error; (c) skill lost to a lifting

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
def fig_memory(rows, report):
    """P1 + P2 in one row: depth sweep, two-depth designs, and the intercept decomposition."""
    run1, run2 = "ablation_memory_depth_matched", "mechanism_item_effect"
    sweep = {
        "sentence_length_t10":     ("lag {k} minus lag 0", "matched caliber, primary anchor", [1, 2, 3]),
        "vector_count_stage2_t10": ("lag {k} minus lag 0", "matched caliber, primary anchor", [1, 2, 3]),
        "vector_count_stage1_t10": ("lag {k} minus lag 0", "matched caliber, primary anchor", [1, 2, 3]),
        "constraint":              ("nu {k} minus nu 1",   "matched caliber",                 [2, 3, 4]),
    }
    two_cell = {
        "sentiment_t5": ("lag 1 minus lag 0", "matched caliber, primary anchor"),
        "defense":      ("nu 2 minus nu 1",   "matched caliber"),
        "tsar_cefr":    ("lag 1 minus lag 0", "matched caliber"),
    }
    series = [  # (short legend label, regex on `what`, palette slot, marker)
        ("Memory gain", r"ours minus markov =", 0, "o"),
        ("Item intercept", r"markov plus item minus markov =", 1, "s"),
        ("Remainder", r"ours minus markov plus item =", 2, "^"),
    ]
    used_depth, used_item = {}, {}
    fig, (ax, ax2, ax3) = plt.subplots(
        1, 3, figsize=(5.4, 1.7), gridspec_kw={"width_ratios": [1.55, 0.95, 2.05]})

    # (a) depth sweep. Filled marker = the paired gain over one reading has a 95% interval
    # excluding zero; hollow = it contains zero. The same rule holds in (b) and (c).
    slot = 0
    for task, (pat, anchor, ks) in sweep.items():
        xs, ys, lo, hi, ids = [1], [0.0], [0.0], [0.0], []
        for k in ks:
            r = pick(rows, run1, task, re.escape(anchor) + r".*" + re.escape(pat.format(k=k)) + r" =")
            depth = k + 1 if pat.startswith("lag") else k
            xs.append(depth); ys.append(r["value"]); lo.append(r["ci"][0]); hi.append(r["ci"][1])
            ids.append(r["id"])
        used_depth[task] = ids
        c, m = PALETTE[slot], MARKERS[slot]
        ax.plot(xs, ys, color=c, linewidth=1.4, zorder=2)
        for x, y, l, h in zip(xs, ys, lo, hi):
            excl = (x > 1) and ((l > 0) or (h < 0))
            ax.plot(x, y, marker=m, markersize=4.2, color=c if excl else "white",
                    markeredgecolor=c, markeredgewidth=0.9, zorder=3)
        dy = {"constraint": 6, "vector_count_stage2_t10": -6}.get(task, 0)
        ax.annotate(NAMES[task], (xs[-1], ys[-1]), xytext=(3, dy), textcoords="offset points",
                    va="center", ha="left", fontsize=6, color=INK)
        slot += 1
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=1)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.85, 5.6)
    ax.set_xlabel("past readings in the state")
    ax.set_ylabel(r"$\mathrm{Skill}_H$ gain over one reading")
    ax.set_title("(a) Depth sweep", loc="left", fontsize=7.5)
    style_axes(ax)

    # (b) designs with two depths only.
    labels, vals, los, his = [], [], [], []
    for task, (pat, anchor) in two_cell.items():
        r = pick(rows, run1, task, re.escape(anchor) + r".*" + re.escape(pat) + r" =")
        used_depth[task] = [r["id"]]
        labels.append(NAMES[task]); vals.append(r["value"]); los.append(r["ci"][0]); his.append(r["ci"][1])
    ypos = list(range(len(labels)))[::-1]
    ax2.axvline(0, color=MUTED, linewidth=0.6, zorder=1)
    for y, v, l, h in zip(ypos, vals, los, his):
        excl = (l > 0) or (h < 0)
        ax2.plot([l, h], [y, y], color=INK, linewidth=0.8, zorder=2)
        ax2.plot(v, y, marker="o", markersize=4.2, color=INK if excl else "white",
                 markeredgecolor=INK, markeredgewidth=0.8, zorder=3)
    ax2.set_yticks(ypos); ax2.set_yticklabels(labels)
    ax2.set_xlabel("one extra reading")
    ax2.set_title("(b) Two depths", loc="left", fontsize=7.5)
    ax2.set_ylim(-0.6, len(labels) - 0.4)
    style_axes(ax2)
    ax2.grid(True, axis="x", color=GRID, linewidth=0.5); ax2.grid(False, axis="y")

    # (c) what the gain decomposes into.
    data = {t: {lab: pick(rows, run2, t, pat) for lab, pat, _, _ in series} for t in TABLE2}
    order = sorted(TABLE2, key=lambda t: -data[t][series[0][0]]["value"])
    off = [0.24, 0.0, -0.24]
    ypos3 = {t: i for i, t in enumerate(order[::-1])}
    ax3.axvline(0, color=MUTED, linewidth=0.6, zorder=1)
    for t in order:
        ax3.axhline(ypos3[t] - 0.5, color=GRID, linewidth=0.4, zorder=0)
    for (label, pat, slot, m), o in zip(series, off):
        c, first = PALETTE[slot], True
        for t in order:
            r = data[t][label]
            used_item.setdefault(t, []).append(r["id"])
            lo, hi = r["ci"]; v = r["value"]
            excl = (lo > 0) or (hi < 0)
            ax3.plot([lo, hi], [ypos3[t] + o, ypos3[t] + o], color=c, linewidth=0.9, zorder=2)
            ax3.plot(v, ypos3[t] + o, marker=m, markersize=4.0, color=c if excl else "white",
                     markeredgecolor=c, markeredgewidth=0.8, zorder=3,
                     label=label if first else None)
            first = False
    ax3.set_yticks([ypos3[t] for t in order]); ax3.set_yticklabels([NAMES[t] for t in order])
    ax3.set_xlabel(r"paired $\mathrm{Skill}_H$ difference")
    ax3.set_title("(c) What the gain decomposes into", loc="left", fontsize=7.5)
    ax3.set_ylim(-0.6, len(order) - 0.4)
    ax3.legend(loc="lower right", frameon=True, framealpha=0.88, edgecolor="none",
               handlelength=1.0, fontsize=6, labelspacing=0.25, borderpad=0.2)
    style_axes(ax3)
    ax3.grid(True, axis="x", color=GRID, linewidth=0.5); ax3.grid(False, axis="y")

    fig.tight_layout(pad=0.4, w_pad=1.1)
    out = os.path.join(FIGS, "fig_memory.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    report["fig_memory"] = {
        "file": os.path.relpath(out, PAPER), "caliber": "matched window",
        "panels": {"a": "depth sweep", "b": "two-depth designs", "c": "intercept decomposition"},
        "marker_rule": "filled = 95% paired interval excludes zero, in all three panels",
        "ids_depth": used_depth, "ids_item": used_item}
    return out


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
    fig, axes = plt.subplots(1, 4, figsize=(5.4, 1.45), sharey=False)
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
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(45); lbl.set_ha("right"); lbl.set_rotation_mode("anchor")
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
               bbox_to_anchor=(0.5, 1.11), handlelength=1.6, columnspacing=1.6)
    fig.tight_layout(pad=0.4, w_pad=0.7)
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
# (task, ledger id, readout dimension). The dimension is the half of m4 the data supports:
# a one-dimensional readout leaves a lifting little structure to expose.
LIFTING = [("sentence_length_t10", "n_main_015", 1), ("vector_count_stage2_t10", "n_main_047", 3),
           ("vector_count_stage1_t10", "n_main_031", 2), ("sentiment_t5", "n_main_111", 1),
           ("defense", "n_main_151", 1), ("constraint", "n_main_135", 1),
           ("tsar_cefr", "n_main_183", 1)]
DIM_MARKER = {1: "o", 2: "s", 3: "D"}


def fig_mechanism(rows, report):
    """m5 and m4 in one row: the command channel on two tasks, then the cost of a lifting."""
    by = {r["id"]: r for r in rows}
    # 4-column grid with a blank spacer (col 2) between (b) and (c): panel (c)'s rotated
    # yticklabels need a wide gutter to its left, while (a)-(b) sit close together.
    fig = plt.figure(figsize=(6.1, 1.05))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.55, 1.55, 0.85, 1.3], wspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 3])

    # (a), (b): the applied command against the tracking error, one panel per task.
    chan = {}
    for ax, task, tag in ((ax1, "sentence_length_t10", "(a)"), (ax2, "sentiment_t5", "(b)")):
        grouped, sha, rel = trajectories(task)
        xs, ys, targets, same = [], [], [], 0
        for trows in grouped.values():
            for a, b in zip(trows, trows[1:]):
                u, r_, y = b.get("requested_norm"), a.get("effective_norm"), a.get("normalized_output")
                if None in (u, r_, y):
                    continue
                xs.append(r_ - y); ys.append(u); targets.append(r_)
                same += abs(u - r_) < 1e-9
        ax.scatter(xs, ys, s=3.0, c=targets, cmap="Blues", vmin=-0.45,
                   alpha=0.55, linewidths=0, zorder=2)
        ax.axvline(0, color=GRID, linewidth=0.6, zorder=1)
        style_axes(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=0.5)
        ax.set_title(tag + " " + NAMES[task], loc="left", fontsize=7.5)
        ax.set_xlabel("tracking error $r-y_t$")
        ax.annotate(f"{same}/{len(xs)}\nat $u=r$", xy=(0.04, 0.04), xycoords="axes fraction",
                    fontsize=6.0, color=MUTED, ha="left", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.85))
        chan[task] = {"file": rel, "sha256": sha, "n_transitions": len(xs),
                      "n_command_equals_target": same}
    ax1.set_ylabel("command $u_{t+1}$")

    # (c): what a learned lifting costs, rows grouped by readout dimension then by value.
    ordered = sorted(LIFTING, key=lambda e: (e[2], -by[e[1]]["value"]))
    ids, ticks, labels = {}, [], []
    for row, (task, rid, dim) in enumerate(ordered):
        r = by[rid]
        y = len(ordered) - 1 - row
        n, v, ci = r["n"], r["value"], r["ci"]
        excl = ci[0] > 0 or ci[1] < 0
        caveated = r.get("status") == "caveated"
        ax3.plot(ci, [y, y], color=MUTED, linewidth=0.7, zorder=2)
        ax3.scatter([v], [y], s=20, marker=DIM_MARKER[dim], zorder=3,
                    facecolor=(PALETTE[0] if excl else "white"),
                    edgecolor=(MUTED if caveated else PALETTE[0]),
                    linewidths=(1.0 if caveated else 0.9))
        ticks.append(y)
        labels.append(NAMES[task] + ("$^{\\dagger}$" if caveated else "")
                      + "  $n{=}$" + str(n))
        ids[task] = {"id": rid, "n_trajectories": n, "value": v, "ci": ci,
                     "readout_dim": dim, "interval_excludes_zero": bool(excl),
                     "status": r.get("status")}
    ax3.axvline(0, color=MUTED, linewidth=0.7, zorder=1)
    ax3.set_yticks(ticks)
    ax3.set_yticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax3.set_ylim(-0.7, len(ordered) - 0.3)
    for side in ("top", "right"):
        ax3.spines[side].set_visible(False)
    ax3.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax3.set_axisbelow(True); ax3.tick_params(length=2.5, width=0.5)
    ax3.set_title("(c) Skill lost to a learned lifting", loc="left", fontsize=7.5)
    ax3.set_xlabel(r"$\Delta\,\mathrm{Skill}_H$")

    fig.tight_layout(pad=0.4, w_pad=0.3)
    out = os.path.join(FIGS, "fig_mechanism.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    report["fig_mechanism"] = {
        "file": os.path.relpath(out, PAPER),
        "panels": {"a": "command channel, sentence length (m5)",
                   "b": "command channel, sentiment (m5)",
                   "c": "cost of a learned lifting by readout dimension (m4)"},
        "caliber": "(a),(b) descriptive raw collection log, NOT Skill_H; "
                   "(c) paired grouped bootstrap from the evidence ledger",
        "marker_rule": "(c) filled = 95% interval excludes zero; grey edge + dagger = caveated; "
                       "shape = readout dimension (circle 1-d, square 2-d, diamond 3-d)",
        "not_shown": "the lifting cost does not fall with the trajectory count: CEFR (n=100) "
                     "carries the largest and Joint-3 (n=90) the smallest.",
        "note": "Joint-2 / Joint-3 log no per-turn scalar command; their channel was checked by "
                "the rank test in docs/experiments/core_multiobjective_planning_headroom.md 2.1",
        "sources": chan, "ids": ids}
    return out


def main():
    rows = load()
    os.makedirs(FIGS, exist_ok=True)
    report = {}
    fig_memory(rows, report)
    fig_two_futures(report)
    fig_mechanism(rows, report)
    with open(os.path.join(FIGS, "_fig_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
