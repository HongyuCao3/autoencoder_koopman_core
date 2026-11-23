#!/usr/bin/env python3
"""Figures 1-4 for the NSF MFAI proposal (plan: docs/article/proposal_integration_plan_2026-11-05.md §二).

Nothing plotted is typed by hand: Figs 2-3 read paper/evidence/numbers.yaml by id (read-only;
paper/ is not modified), Fig 4 reads results/ and the #4b trajectories. Every id or file used is
written to proposal/figs/_proposal_fig_report.json.

Sizes are print sizes (NSF letter, 6.5 in text width), so font sizes are printed points.
Palette = dataviz reference slots 1-4 (validated light mode; slots 3-4 below 3:1 -> direct labels).

    /scratch/hcao2/envs/dkvar/bin/python docs/article/proposal_figs/make_proposal_figs.py [--layout {wide,single}] [fig1 fig2 fig3 fig4]

--layout wide (default) reproduces the 2026-11-05 figures in proposal/figs/. --layout single
(2026-11-23 materials request: vector PDF, single-column width, no text below 8 pt) writes to
proposal/figs/single/: Figs 2-4 are 3.3 in wide (Figs 2 and 4 stacked 2x1), Fig 1 stays full width,
and the run ends with a pdfplumber font/width check that exits 1 on failure. Spec:
docs/article/proposal_figs/TASK_single_column_redraw_2026-11-23.md. Same data, same ids in both layouts.
"""
from __future__ import annotations

import argparse
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
LAYOUT = "wide"  # set from --layout in __main__
SPLIT = False    # set from --split in __main__: also write each panel of fig2 / fig4 as its own file
SPLIT_NAMES = {"fig2_memory": ("fig2a_memory_gain", "fig2b_depth"),
               "fig4_prompt_authority": ("fig4a_reachable", "fig4b_effect")}


def PT(letter: str, text: str) -> str:
    """Panel title: '(a) text' in the combined figure, bare text when panels are also written separately."""
    return text if SPLIT else f"({letter}) {text}"


def FS(x: float) -> float:
    """Font size: unchanged in the wide layout, floored at 8 pt in the single-column layout."""
    return max(x, 8.0) if LAYOUT == "single" else x


def SINGLE() -> bool:
    return LAYOUT == "single"


def TS() -> float:
    return 9 if SINGLE() else 8


SINGLE_RC = {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8}


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


def one_line(t: str) -> str:
    """Task name on one line ("Simpli-\nfication" -> "Simplification")."""
    return PLAIN[t].replace("-\n", "").replace("\n", " ")


def save(fig, name, groups=None):
    """groups: [(axes list), (axes list)] -> with --split, each group is also cropped to its own PDF/PNG
    (vector content untouched; full 3.3 in width kept, only the vertical extent is trimmed)."""
    OUT.mkdir(parents=True, exist_ok=True)
    if SINGLE():  # exact canvas: width is what was asked for, not what a tight bbox grows to
        if SPLIT and groups:  # panels only; the combined file (with "(a)/(b)" titles) comes from a run without --split
            split_panels(fig, name, groups)
            plt.close(fig)
            return
        fig.savefig(OUT / f"{name}.pdf")
        fig.savefig(OUT / f"{name}.png", dpi=200, facecolor="white")
        plt.close(fig)
        return
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1 framework (schematic)
def split_panels(fig, name, groups):
    from pypdf import PdfReader, PdfWriter
    from PIL import Image
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    H = fig.get_figheight(); W = fig.get_figwidth(); dpi = fig.dpi
    pad = 0.04  # inches
    spans = []
    for axes in groups:
        bbs = [a.get_tightbbox(rend) for a in axes]
        y0 = min(b.y0 for b in bbs) / dpi - pad; y1 = max(b.y1 for b in bbs) / dpi + pad
        spans.append((max(0.0, y0), min(H, y1)))
    # keep the groups from overlapping: cut midway between them
    for i in range(len(spans) - 1):
        (a0, a1), (b0, b1) = spans[i], spans[i + 1]
        if a0 < b1 and b0 < a1:  # overlap in figure-inch coordinates (y grows upward)
            mid = (max(a0, b0) + min(a1, b1)) / 2
            if a0 > b0:
                spans[i] = (mid, a1); spans[i + 1] = (b0, mid)
            else:
                spans[i] = (a0, mid); spans[i + 1] = (mid, b1)
    import tempfile
    tdir = tempfile.TemporaryDirectory()
    tmp_pdf = pathlib.Path(tdir.name) / f"{name}.pdf"; tmp_png = pathlib.Path(tdir.name) / f"{name}.png"
    fig.savefig(tmp_pdf); fig.savefig(tmp_png, dpi=200, facecolor="white")
    rd = PdfReader(tmp_pdf)
    png = Image.open(tmp_png)
    for (y0, y1), out in zip(spans, SPLIT_NAMES[name]):
        wr = PdfWriter(); pg = wr.add_page(rd.pages[0])
        pg.mediabox.lower_left = (0, y0 * 72); pg.mediabox.upper_right = (W * 72, y1 * 72)
        pg.cropbox.lower_left = (0, y0 * 72); pg.cropbox.upper_right = (W * 72, y1 * 72)
        wr.write(OUT / f"{out}.pdf")
        s = png.height / H  # px per inch
        png.crop((0, int(round((H - y1) * s)), png.width, int(round((H - y0) * s)))).save(OUT / f"{out}.png")
        REPORT.setdefault("split", {})[out] = {"from": name, "height_in": round(y1 - y0, 2)}
    png.close()
    tdir.cleanup()


def box(ax, x, y, w, h, title, body, edge=INK, face="white", dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", lw=0.9,
                                ec=edge, fc=face, ls=(0, (3, 2)) if dashed else "-", zorder=2))
    if SINGLE():  # title 9, body 8; body starts below a one- or two-line title
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9, fontweight="bold", color=INK, zorder=3,
                linespacing=1.15)
        ax.text(x + w / 2, y + h - (12.4 if "\n" in title else 7.6), body, ha="center", va="top", fontsize=8,
                color=MUTED, zorder=3, linespacing=1.2)
        return
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 7.4, body, ha="center", va="top", fontsize=FS(7), color=MUTED, zorder=3, linespacing=1.25)


def arrow(ax, p, q, color=INK, dashed=False, rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 ls=(0, (3, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=1))


def fig1_single():
    """Full width (replaces the p.2 arrow row, F1 plan B); text >= 8 pt. Mathtext scripts would print
    below 8 pt, so symbols are written as plain text: s(k), c(k), u_l(k) -> s(k), c(k), u(k)."""
    fig = plt.figure(figsize=(6.5, 2.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 48); ax.axis("off")
    y, h, w = 13.5, 23, 25.5
    xs = [0.5, 35.0, 69.0, 103.5]
    box(ax, xs[0], y, w, h, "Interaction\nhistory", "prompts and replies\nso far")
    box(ax, xs[1], y, w, h, "Behavioral\nstate s(k)", "measured attributes\nof recent replies\n(e.g., lengths)")
    box(ax, xs[2], y, w, h, "Controlled linear\nmodel (2.5)", "", edge=BLUE)
    ax.text(xs[2] + w / 2, y + 7.2, "z(k+1) = A z(k)\n+ B c(k) + Σ Bℓ uℓ(k)", ha="center", va="center",
            fontsize=8, color=INK, zorder=3, linespacing=1.25)
    box(ax, xs[3], y, w, h, "Future\nbehavior", "predicted attributes\nof the next replies")
    for i, lab in enumerate(("compress", "encode", "predict")):
        a, b = xs[i] + w, xs[i + 1]
        arrow(ax, (a + 0.3, y + h / 2), (b - 0.3, y + h / 2))
        ax.text((a + b) / 2, y + h / 2 + 1.2, lab, ha="center", va="bottom", fontsize=8, color=MUTED)
    arrow(ax, (xs[3] + w / 2, y + h + 0.3), (xs[0] + w / 2, y + h + 0.3), color=MUTED, rad=0.1, lw=0.8)
    ax.text(65.5, 46.0, "next turn: the new reply joins the history", ha="center", va="center", fontsize=8, color=MUTED)
    arrow(ax, (68, 7.4), (77, y - 0.3), color=ORANGE, lw=1.3)
    arrow(ax, (96, 7.4), (88, y - 0.3), color=OLD, dashed=True, lw=1.3)
    ax.text(67, 8.4, "Prompt c(k) (instruction text)", ha="right", va="center", fontsize=8, color=INK)
    ax.text(67, 3.6, "preliminary work, Sec. 3.2", ha="right", va="center", fontsize=8, color=MUTED)
    ax.text(98, 8.4, "Layer activation change u(k)", ha="left", va="center", fontsize=8, color=INK)
    ax.text(98, 3.6, "proposed, Thrusts A.3, C.2", ha="left", va="center", fontsize=8, color=MUTED)
    REPORT["fig1"] = {"kind": "schematic", "data": None}
    save(fig, "fig1_framework")


def fig1():
    if SINGLE():
        return fig1_single()
    fig = plt.figure(figsize=(6.5, 2.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 40); ax.axis("off")
    y, h = 15, 18
    box(ax, 1, y, 27, h, "Interaction history", "prompts and replies\nso far in the\nconversation")
    box(ax, 35, y, 27, h, r"Behavioral state $s_k$", "a few measured\nattributes of recent\nreplies (e.g., lengths)")
    box(ax, 69, y, 27, h, "Controlled linear\nmodel (2.5)", "", edge=BLUE)
    ax.text(82.5, y + 5.6, r"$z_{k+1}=Az_k+B_p c^p_k+\sum_\ell B_\ell u^{(\ell)}_k$", ha="center", va="center",
            fontsize=FS(7), color=INK, zorder=3)
    box(ax, 103, y, 26, h, "Future behavior", "predicted attributes\nof the next replies")
    for (a, b), lab in zip(((28, 35), (62, 69), (96, 103)), ("compress", "encode", "predict")):
        arrow(ax, (a + 0.3, y + h / 2), (b - 0.3, y + h / 2))
        ax.text((a + b) / 2, y + h / 2 + 1.8, lab, ha="center", fontsize=FS(6.3), color=MUTED)
    # closed loop: the next reply joins the history
    arrow(ax, (116, y + h + 0.3), (14.5, y + h + 0.3), color=MUTED, rad=0.12, lw=0.8)
    ax.text(65.5, 35.6, "next turn: the new reply joins the history", ha="center", va="center", fontsize=FS(6.3), color=MUTED)
    # two input channels into the model box
    arrow(ax, (68, 6.5), (78, y - 0.3), color=ORANGE, lw=1.3)
    arrow(ax, (97, 6.5), (87, y - 0.3), color=OLD, dashed=True, lw=1.3)
    ax.text(67, 7.6, r"Prompt $c^p_k$ (instruction text)", ha="right", va="center", fontsize=FS(7.2), color=INK)
    ax.text(67, 3.8, "realized in preliminary work (Sec. 3.2)", ha="right", va="center", fontsize=FS(6.3), color=MUTED)
    ax.text(98, 7.6, r"Activation change $u^{(\ell)}_k$", ha="left", va="center", fontsize=FS(7.2), color=INK)
    ax.text(98, 3.8, "internal layers; proposed (Thrusts A.3, C.2)", ha="left", va="center", fontsize=FS(6.3), color=MUTED)
    REPORT["fig1"] = {"kind": "schematic", "data": None}
    save(fig, "fig1_framework")


# ---------------------------------------------------------------- Fig. 2 what memory buys
def fig2():
    L = ledger()
    if SINGLE():  # 2x1 stacked, single column
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.3, 4.2), layout="constrained", gridspec_kw={"height_ratios": [1.15, 1.0]})
    else:
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
    if SINGLE():  # one-line names at 30 deg; mathtext subscript would print below 8 pt -> plain text
        ax.set_xticks(x, [one_line(t) for t in TASKS], fontsize=8,
                      rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Prediction skill (Skill-H)")
        ax.set_ylim(-0.12, 1.0); style(ax)
        ax.legend(handles=[Patch(color=OLD, label="current reading only"),
                           Patch(color=BLUE, label="last few readings, gain's 95% interval excludes 0"),
                           Patch(fc="white", ec=BLUE, lw=0.9, label="last few readings, interval contains 0")],
                  fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=1.2,
                  borderaxespad=0.2, labelspacing=0.25)
        ax.set_title(PT("a", "Memory vs. current reading"), fontsize=TS(), loc="left", pad=42)
    else:
        ax.set_xticks(x, [PLAIN[t] for t in TASKS], fontsize=FS(6.3))
        ax.set_ylabel(r"Prediction skill (Skill$_H$)")
        ax.set_ylim(-0.12, 1.0); style(ax)
        ax.legend(handles=[Patch(color=OLD, label="current reading only"),
                           Patch(color=BLUE, label="last few readings, gain's 95% interval excludes 0"),
                           Patch(fc="white", ec=BLUE, lw=0.9, label="last few readings, interval contains 0")],
                  fontsize=FS(6.3), frameon=False, loc="upper right", handlelength=1.2)
    if not SINGLE():
        ax.set_title("(a) Memory vs. current reading", fontsize=TS(), loc="left")
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
        if SINGLE():  # task names go to the legend instead of line-end labels
            continue
        lab_xy = {"vector_count_stage2_t10": (2.0, ys[1] + 0.016, "center")}.get(t, (xs[-1] + 0.12, ys[-1] - (0.012 if t == "constraint" else 0), "left"))
        bx.text(lab_xy[0], lab_xy[1], PLAIN[t].replace("\n", " "), fontsize=FS(6.2), va="center", ha=lab_xy[2], color=INK)
    bx.axhline(0, color=MUTED, lw=0.6)
    if SINGLE():
        bx.legend(handles=[Line2D([], [], color=c, lw=1.4, marker="o", ms=3.8, label=one_line(t))
                           for t, c in zip(sweep, (BLUE, ORANGE, AQUA, YELLOW))],
                  fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.42), ncol=2, handlelength=1.6,
                  borderaxespad=0.2, labelspacing=0.3, columnspacing=1.0)
        bx.set_xticks([1, 2, 3, 4]); bx.set_xlim(0.85, 4.15); bx.set_ylim(-0.01, 0.27)
    else:
        bx.set_xticks([1, 2, 3, 4]); bx.set_xlim(0.85, 6.3); bx.set_ylim(-0.01, 0.27)
    bx.set_xlabel("Past readings in the state"); bx.set_ylabel("Gain over one reading")
    bx.text(0.98, 0.003, "hollow = interval contains 0", fontsize=FS(6.0), color=MUTED, ha="right", va="bottom", transform=bx.get_yaxis_transform())
    style(bx); bx.set_title(PT("b", "How many readings") if SINGLE() else "(b) How many readings", fontsize=TS(), loc="left")
    REPORT["fig2"] = {"a": used, "b_depth_ids": depth_ids, "ledger": str(LEDGER.relative_to(REPO))}
    save(fig, "fig2_memory", groups=[[ax], [bx]])


# ---------------------------------------------------------------- Fig. 3 learned lifting
def fig3():
    L = ledger()
    fig, ax = plt.subplots(figsize=(3.3, 2.3), layout="constrained") if SINGLE() else plt.subplots(figsize=(3.2, 2.0))
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
    names = [one_line(t) if SINGLE() else PLAIN[t].replace("\n", " ") for t in reversed(TASKS)]  # wide kept as 11-05
    ax.set_yticks(range(len(TASKS)), names, fontsize=FS(6.6))
    ax.set_xlabel("Skill difference: linear minus learned lifting")
    note = "right of 0: linear model better; filled = interval excludes 0"
    ax.text(1.0, 1.02, note.replace("; ", ";\n") if SINGLE() else note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS(6.2), color=MUTED)
    style(ax, grid_axis="x")
    REPORT["fig3"] = {"ids": used}
    save(fig, "fig3_lifting")


# ---------------------------------------------------------------- Fig. 4 prompt authority, measured
def fig4():
    spec = importlib.util.spec_from_file_location("reach", REPO / "scripts" / "core_excite_reachability.py")
    reach = importlib.util.module_from_spec(spec); spec.loader.exec_module(reach)
    ident = reach.ident
    summ = json.load(open(REPO / "results" / "core_excite_reachability" / "summary.json"))
    if SINGLE():  # 2x1 stacked, single column; (b) = three stacked axes, one per task scale
        fig = plt.figure(figsize=(3.3, 4.4), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2])
        ax = fig.add_subplot(gs[0])
        sg = gs[1].subgridspec(3, 1, height_ratios=[3, 3, 1.4])
        bsubs = [fig.add_subplot(sg[i]) for i in range(3)]
    else:
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
    ax.text(2, ceil + 1.5, f"model's reachable average: {ceil:.0f} words", fontsize=FS(6.2), color=MUTED, ha="left", va="bottom")
    if SINGLE():  # two columns at 8 pt need shorter labels; same four entries
        ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="", ms=4, label="held-out, mean \u00b1 sd (3 seeds)"),
                           Line2D([], [], color=ORANGE, marker="s", ms=3.5, lw=1.6, label="model fit on train prompts"),
                           Line2D([], [], color=FAINT, marker="o", ls="", ms=2.5, label="single held-out replies"),
                           Line2D([], [], color=MUTED, lw=0.7, ls=":", label="delivered = requested")],
                  fontsize=8, frameon=False, loc="upper left", ncol=2, handlelength=1.3, columnspacing=0.7,
                  handletextpad=0.4, borderaxespad=0.2)
    else:
        ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="", ms=4, label="held-out prompts (mean, sd over 3 seeds)"),
                           Line2D([], [], color=ORANGE, marker="s", ms=3.5, lw=1.6, label="model fitted on training prompts"),
                           Line2D([], [], color=FAINT, marker="o", ls="", ms=2.5, label="individual held-out replies"),
                           Line2D([], [], color=MUTED, lw=0.7, ls=":", label="delivered = requested")],
                  fontsize=FS(6.0), frameon=False, loc="upper left", handlelength=1.6)
    ax.set_xlim(0, 105); ax.set_ylim(0, 150 if SINGLE() else 135)  # single: room for the 2-column legend
    ax.set_xlabel("Requested words"); ax.set_ylabel("Delivered words")
    style(ax); ax.set_title(PT("a", "Sentence length: what the prompt can reach") if SINGLE() else "(a) Sentence length: what the prompt can reach", fontsize=TS(), loc="left")
    # (b) identified one-step effects, each task on its own scale
    ex = json.load(open(REPO / "results" / "core_excite_b_identifiability" / "summary.json"))
    rows_b = [("Sentence length, seed %d" % s, ex["per_seed"][str(s)]["main_element_CB"]["point"],
               *ex["per_seed"][str(s)]["main_element_CB"]["ci95"], "words per requested word") for s in ident.SEEDS]
    cefr = json.load(open(REPO / "results" / "b_identifiability" / "tsar_cefr.json"))["bootstrap"]["trajectory"]["main_elements_CB"]
    names = ["step down", "half step down", "paraphrase"]
    rows_c = [(f"Simplification: {n}", e["point"], e["ci_low"], e["ci_high"], "CEFR level") for n, e in zip(names, cefr)]
    con = json.load(open(REPO / "results" / "b_identifiability" / "constraint.json"))["bootstrap"]["trajectory"]["main_elements_CB"][0]
    rows_d = [("Constraint: reminder", con["point"], con["ci_low"], con["ci_high"], "satisfaction")]
    if SINGLE():
        groups = [(rows_b, "Sentence length (normalized words)"), (rows_c, "Simplification (CEFR level)"),
                  (rows_d, "Constraint (satisfaction)")]
        for sub, (rows, unit) in zip(bsubs, groups):
            for j, (lab, p, lo, hi, _) in enumerate(rows):
                yy = len(rows) - 1 - j
                sub.plot([lo, hi], [yy, yy], color=BLUE, lw=1.2); sub.plot(p, yy, "o", ms=3.8, color=BLUE)
            sub.axvline(0, color=INK, lw=0.7)
            lo_all = min(min(r[2] for r in rows), 0); hi_all = max(max(r[3] for r in rows), 0)
            pad = 0.15 * (hi_all - lo_all); sub.set_xlim(lo_all - pad, hi_all + pad)
            sub.set_ylim(-0.6, len(rows) - 0.4)
            sub.set_yticks(range(len(rows)), [r[0].split(", ")[-1].split(": ")[-1] for r in reversed(rows)], fontsize=8)
            for s in ("top", "right", "left"):
                sub.spines[s].set_visible(False)
            sub.tick_params(axis="x", labelsize=8, length=2, width=0.5, colors=MUTED, labelcolor=INK)
            sub.tick_params(axis="y", length=0)
            sub.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            sub.set_xlabel(unit, fontsize=8, color=MUTED, labelpad=1)
        bsubs[0].set_title(PT("b", "One-step instruction effect, 95% interval"), fontsize=TS(), loc="left")
        REPORT["fig4"] = {"a": ["results/core_excite_reachability/summary.json", ident.IN_TEMPLATE],
                          "b": ["results/core_excite_b_identifiability/summary.json",
                                "results/b_identifiability/tsar_cefr.json", "results/b_identifiability/constraint.json"]}
        return save(fig, "fig4_prompt_authority", groups=[[ax], bsubs])
    bx.axis("off")
    groups = [(rows_b, "sentence length (normalized words)"), (rows_c, "CEFR level"), (rows_d, "constraint satisfaction")]
    sub_h, top, gap, lab_x = [0.33, 0.33, 0.14], 0.92, 0.1, 0.40
    for (rows, unit), hh in zip(groups, sub_h):
        sub = bx.inset_axes([lab_x + 0.02, top - hh, 0.98 - lab_x - 0.02, hh - 0.06])
        for j, (lab, p, lo, hi, _) in enumerate(rows):
            yy = len(rows) - 1 - j
            sub.plot([lo, hi], [yy, yy], color=BLUE, lw=1.2); sub.plot(p, yy, "o", ms=3.8, color=BLUE)
            bx.text(lab_x, top - 0.03 - (j + 0.5) * (hh - 0.06) / len(rows), lab, transform=bx.transAxes,
                    ha="right", va="center", fontsize=FS(6.2), color=INK)
        sub.axvline(0, color=INK, lw=0.7)
        lo_all = min(min(r[2] for r in rows), 0); hi_all = max(max(r[3] for r in rows), 0)
        pad = 0.15 * (hi_all - lo_all); sub.set_xlim(lo_all - pad, hi_all + pad)
        sub.set_yticks([]); sub.set_ylim(-0.6, len(rows) - 0.4)
        for s in ("top", "right", "left"):
            sub.spines[s].set_visible(False)
        sub.tick_params(axis="x", labelsize=FS(5.8), length=2, width=0.5, colors=MUTED, labelcolor=INK)
        sub.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        sub.set_xlabel(unit, fontsize=FS(5.8), color=MUTED, labelpad=1)
        top -= hh + gap
    bx.set_title("(b) One-step instruction effect, 95% interval", fontsize=TS(), loc="left")
    REPORT["fig4"] = {"a": ["results/core_excite_reachability/summary.json", ident.IN_TEMPLATE],
                      "b": ["results/core_excite_b_identifiability/summary.json",
                            "results/b_identifiability/tsar_cefr.json", "results/b_identifiability/constraint.json"]}
    save(fig, "fig4_prompt_authority")


def glyph_pt(c: dict) -> float:
    """Printed font size of one pdfplumber char. pdfplumber's c["size"] is the bbox extent, which equals
    the font size only for unrotated text: a 90-deg y-label 'i' at 8 pt reads 2.2 (its advance), a '.' in a
    30-deg tick label reads 7.93 (= adv*sin + size*cos). The glyph box is adv x size rotated by the text
    matrix angle, so recover size from the bbox side that the rotation leaves dominated by it."""
    import math
    a, b = c["matrix"][0], c["matrix"][1]
    cs, sn = abs(math.cos(math.atan2(b, a))), abs(math.sin(math.atan2(b, a)))
    h, w = c["bottom"] - c["top"], c["x1"] - c["x0"]
    return (h - c["adv"] * sn) / cs if cs >= sn else (w - c["adv"] * cs) / sn


def check_single() -> None:
    """Acceptance (task spec 1.3): every figure <= 3.4 in wide except Fig 1; no glyph below 8 pt."""
    import pdfplumber
    bad = []
    for pdf in sorted(OUT.glob("fig*.pdf")):
        with pdfplumber.open(pdf) as doc:
            p = doc.pages[0]
            w_in = p.width / 72
            sizes = [round(glyph_pt(c), 1) for c in p.chars if c["text"].strip()]
        small = [s for s in sizes if s < 8.0]
        print(pdf.name, f"{w_in:.2f} in", "min font", min(sizes), "n<8:", len(small))
        if not (w_in <= 3.4 or pdf.name.startswith("fig1")):
            bad.append(f"{pdf.name}: {w_in:.2f} in wide")
        if small:
            bad.append(f"{pdf.name}: {len(small)} glyphs below 8pt")
    if bad:
        print("ACCEPTANCE FAILED:", "; ".join(bad))
        raise SystemExit(1)
    print("acceptance passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["wide", "single"], default="wide")
    ap.add_argument("--split", action="store_true", help="single layout only: also write each panel of fig2/fig4 as its own file")
    ap.add_argument("figs", nargs="*", default=["fig1", "fig2", "fig3", "fig4"])
    args = ap.parse_args()
    LAYOUT = args.layout
    SPLIT = args.split
    if SPLIT and LAYOUT != "single":
        raise SystemExit("--split requires --layout single")
    if SINGLE():
        OUT = OUT / "single"
        plt.rcParams.update(SINGLE_RC)
    which = args.figs
    for name in which:
        globals()[name]()
    rp = OUT / "_proposal_fig_report.json"
    old = json.load(open(rp)) if rp.exists() else {}
    split_old = old.get("split", {})
    old.update(REPORT)
    if "split" in REPORT:  # merge per panel: a run on fig4 alone must not drop the fig2 panel records
        old["split"] = {**split_old, **REPORT["split"]}
    if SINGLE():
        old["layout"] = {"layout": "single", "width_in": 3.3, "min_font_pt": 8.0,
                         "spec": "docs/article/proposal_figs/TASK_single_column_redraw_2026-11-23.md"}
    rp.write_text(json.dumps(old, indent=2, default=str))
    print("wrote", ", ".join(which), "->", OUT)
    if SINGLE():
        check_single()
