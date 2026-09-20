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
    python3 tools/build_figs.py
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
    "vector_count_stage2_t10": "Items (s2)",
    "vector_count_stage1_t10": "Items (s1)",
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


def main():
    rows = load()
    os.makedirs(FIGS, exist_ok=True)
    report = {}
    fig_memory_depth(rows, report)
    fig_item_effect(rows, report)
    with open(os.path.join(FIGS, "_fig_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
