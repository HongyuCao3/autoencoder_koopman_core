#!/usr/bin/env python3
"""Generate the paper's two main tables from the evidence ledgers.

Nothing in tables/*.tex is typed by hand. Every printed number is read out of
evidence/numbers.yaml (self-built tasks, ids n_main_*) or evidence/numbers_benchmark.yaml
(colleague benchmark results, ids n_bench_*), so a table cell can always be traced back
to one evidence id. See WRITING_PLAN_2026-09-18.md §0.3 and §5.

Usage:
    python3 tools/build_tables.py                 # writes tables/*.tex and tables/_build_report.json
    python3 tools/build_tables.py --method-name KOMPAS   # after decision D10 names the method
"""
import argparse
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Table 2 (prediction). Row and column vocabulary.
# Keys are the ledger's wording; values are the paper's wording. The story
# framework requires paper-facing names, not code names. CP3a may revise the
# right-hand side; the left-hand side is fixed by the ledger.
# ---------------------------------------------------------------------------
ROW_ORDER = [
    ("best trivial null", "Stateless"),
    ("Markov state + control", "Last-turn"),
    ("delay-embedded state, control withheld", "Memory w/o command"),
    ("LSTM sequence baseline", "LSTM"),
    ("AE-Koopman lifted operator", "KOMPAS (learned lifting)"),
    ("delay-embedded linear operator with control", "KOMPAS (linear)"),
]

# Seven main-text columns. 2026-09-19: user swapped character_length_t5 and formality_t5
# out for sentiment_t5 and defense (both also stay in appendix A2). 2026-09-20: the two
# vector_count tasks are joint multi-attribute tracking tasks (stage1 = word count +
# average word length, stage2 = + comma count; see configs/dataset/vector_count_stage*.yaml),
# so the paper-facing names are "Joint-2" / "Joint-3", not "Items (s1)/(s2)", which
# collided with the "per-item intercept" of the P2 side experiment.
# Defense carries a known self-judge caveat (evidence/numbers.yaml n_main_136-151: judge_kind=self,
# n_seed=2) that D4/D18 had kept out of the main text; the user chose to include it without an
# in-text caveat sentence (decision recorded in chat, not yet logged as a DECISIONS.md entry).
PRED_COLUMNS = [
    ("sentence_length_t10", "Sentence len."),
    ("vector_count_stage2_t10", "Joint-3"),
    ("vector_count_stage1_t10", "Joint-2"),
    ("sentiment_t5", "Sentiment"),
    ("defense", "Defense"),
    ("constraint", "Constraint"),
    ("tsar_cefr", "CEFR"),
]
APPENDIX_COLUMNS = [
    ("sentiment_t5", "Sentiment"),
    ("average_word_length_t5", "Word len."),
    ("defense", "Defense"),
    ("gsm8k_sharded", "GSM8K-sharded"),
]

# ---------------------------------------------------------------------------
# Table 1 (control).
# ---------------------------------------------------------------------------
BENCH_BLOCKS = [
    ("IFBench", ["WC", "UWC", "NS", "NP"]),
    ("IFEval", ["KF", "WC", "NS", "NP"]),
    ("COLLIE", ["Word", "Sent.", "Para."]),
]
MODELS = ["Qwen3-4B", "Qwen3-8B"]
# Paper order puts the proposed method last; the ledger's own order is the colleague's.
CONTROL_ROWS = [
    ("Base", "--", "no control applied"),
    ("TMPC", "white-box", "colleague reproduction"),
    ("RE-Control", "white-box", "colleague reproduction"),
    ("Koopman", "black-box", "ours"),
]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Prediction table
# ---------------------------------------------------------------------------
def parse_main(entries):
    """Return {(column, row_key): {'skill': v, 'ci': [lo, hi], 'mse': v, 'ids': {...}}}."""
    grid = {}
    col_re = re.compile(r" on ([a-z0-9_]+), (?:lag|nu)=")
    LABEL_RE = re.compile(r"column: (.+?) on [a-z0-9_]+, (?:lag|nu)=")
    for e in entries:
        eid = str(e.get("id", ""))
        if not eid.startswith("n_main_"):
            continue
        what = " ".join(e["what"].split())
        if "paired contrast" in what:
            continue
        m = col_re.search(what)
        if not m:
            continue
        column = m.group(1)
        # Match the row label only, not the whole sentence: every entry ends with
        # "best trivial null = stateless", so a whole-string search would file every
        # row under the stateless null.
        lm = LABEL_RE.search(what)
        if not lm:
            continue
        label = lm.group(1)
        row_key = None
        for ledger_name, _ in ROW_ORDER:
            if ledger_name in label:
                row_key = ledger_name
                break
        if row_key is None:
            continue
        cell = grid.setdefault((column, row_key), {"skill": None, "ci": None, "mse": None, "ids": {}})
        if what.startswith("Table 1 relative column"):
            cell["skill"] = e["value"]
            cell["ci"] = e.get("ci")
            cell["ids"]["skill"] = eid
        elif what.startswith("Table 1 absolute column"):
            cell["mse"] = e["value"]
            cell["ids"]["mse"] = eid
    return grid


CONTRAST_KINDS = {
    "row 2 paired contrast": "ours_minus_last_turn",      # what the delay embedding buys
    "row 3 paired contrast": "ours_minus_memory_no_action",  # what the actuator buys
    "row 4 paired contrast": "ours_minus_lstm",           # linear vs a nonlinear sequence model
    "row 5 paired contrast": "ours_minus_nonlinear",      # linear vs a learned lifting
}


def parse_contrasts(entries):
    """Paired Skill_H differences with grouped-bootstrap CIs. These, not the point
    estimates in the table, are what a memory-necessity or linear-vs-nonlinear claim
    may lean on."""
    out = {}
    col_re = re.compile(r" on ([a-z0-9_]+), (?:lag|nu)=")
    for e in entries:
        eid = str(e.get("id", ""))
        if not eid.startswith("n_main_"):
            continue
        what = " ".join(e["what"].split())
        kind = None
        for needle, name in CONTRAST_KINDS.items():
            if needle in what:
                kind = name
                break
        if kind is None:
            continue
        m = col_re.search(what)
        if not m:
            continue
        ci = e.get("ci")
        out.setdefault(kind, {})[m.group(1)] = {
            "value": e["value"],
            "ci": ci,
            "excludes_zero": bool(ci) and (ci[0] > 0 or ci[1] < 0),
            "id": eid,
        }
    return out


def fmt_skill(value, best, second):
    if value is None:
        return "--"
    s = f"{value:.3f}"
    if best is not None and abs(value - best) < 1e-12:
        return r"\textbf{" + s + "}"
    if second is not None and abs(value - second) < 1e-12:
        return r"\underline{" + s + "}"
    return s


def build_prediction(grid, columns, label, caption, note_ids=True, header_note=None):
    ncol = len(columns)
    lines = []
    lines.append("% GENERATED by tools/build_tables.py. Do not edit by hand; edit the evidence ledger instead.")
    if header_note:
        lines.append("% " + header_note)
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\footnotesize")
    # Only stretch a wide table to the text width. A four-column appendix table blown up
    # to \textwidth reads as a poster, not a table. 2026-09-22: the eight-column Table 1 at
    # \footnotesize is narrower than \textwidth, so stretching it only made it taller; the
    # threshold now leaves it at natural width (page budget, class B).
    wide = ncol >= 10
    if wide:
        lines.append(r"\resizebox{\textwidth}{!}{")
    else:
        # 2026-09-22: at natural width the eight-column Table 1 overflowed \textwidth by
        # ~19pt (Overfull \hbox). Tightening the column padding from 6pt to 4pt saves
        # 2pt x 16 gaps = 32pt and keeps the font size. Scoped to this table environment.
        lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{l" + "c" * ncol + "}")
    lines.append(r"\toprule")
    lines.append("Predictor & " + " & ".join(disp for _, disp in columns) + r" \\")
    lines.append(r"\midrule")

    report = {}
    per_col_rank = {}
    for col_key, _ in columns:
        values = []
        for ledger_name, _disp in ROW_ORDER:
            cell = grid.get((col_key, ledger_name))
            if cell and cell["skill"] is not None:
                values.append(cell["skill"])
        ordered = sorted(set(values), reverse=True)
        per_col_rank[col_key] = (
            ordered[0] if ordered else None,
            ordered[1] if len(ordered) > 1 else None,
        )

    for ledger_name, disp in ROW_ORDER:
        cells, ids = [], []
        for col_key, _ in columns:
            cell = grid.get((col_key, ledger_name))
            best, second = per_col_rank[col_key]
            value = cell["skill"] if cell else None
            cells.append(fmt_skill(value, best, second))
            ids.append(cell["ids"].get("skill") if cell else None)
        row = disp + " & " + " & ".join(cells) + r" \\"
        if note_ids:
            row += "  % " + " ".join(i or "MISSING" for i in ids)
        lines.append(row)
        report[disp] = {
            col_key: (grid.get((col_key, ledger_name)) or {}).get("skill")
            for col_key, _ in columns
        }

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if wide:
        lines.append("}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n", report, per_col_rank


# ---------------------------------------------------------------------------
# Control table
# ---------------------------------------------------------------------------
def parse_bench(entries):
    grid = {}
    for e in entries:
        eid = str(e.get("id", ""))
        if not eid.startswith("n_bench_"):
            continue
        key = (e["model"], e["method"], e["benchmark"], e["subtask"])
        grid[key] = {"value": e["value"], "id": eid}
    return grid


def build_control(grid, method_name):
    header_top = [r"\multirow{2}{*}{Model}", r"\multirow{2}{*}{Method}", r"\multirow{2}{*}{Access}"]
    for bench, subs in BENCH_BLOCKS:
        header_top.append(r"\multicolumn{" + str(len(subs)) + "}{c}{" + bench + "}")
    header_top.append(r"\multirow{2}{*}{Avg.}")

    subheader = ["", "", ""]
    for _, subs in BENCH_BLOCKS:
        subheader.extend(subs)
    subheader.append("")

    ncols = 3 + sum(len(s) for _, s in BENCH_BLOCKS) + 1
    lines = []
    lines.append("% GENERATED by tools/build_tables.py. Do not edit by hand; edit the evidence ledger instead.")
    lines.append("% MAIN-TEXT TABLE 2 of 2 (control line). D18: the main text carries exactly two tables.")
    lines.append("% Every cell is a point estimate as printed by the colleague. No seeds, no intervals.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Constraint satisfaction on IFBench, IFEval and COLLIE. "
        r"Access states what each controller needs from the model. Values are point estimates; "
        r"intervals are not yet available.}"
    )
    lines.append(r"\label{tab:control}")
    lines.append(r"\footnotesize")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{lll" + "c" * (ncols - 3) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_top) + r" \\")
    start = 4
    cmids = []
    for _, subs in BENCH_BLOCKS:
        cmids.append(r"\cmidrule(lr){" + f"{start}-{start + len(subs) - 1}" + "}")
        start += len(subs)
    lines.append(" ".join(cmids))
    lines.append(" & ".join(subheader) + r" \\")
    lines.append(r"\midrule")

    report = {}
    for mi, model in enumerate(MODELS):
        # Best value per column within this model block.
        best_per_col = {}
        for bench, subs in BENCH_BLOCKS:
            for sub in subs:
                vals = [
                    grid[(model, meth, bench, sub)]["value"]
                    for meth, _, _ in CONTROL_ROWS
                    if (model, meth, bench, sub) in grid
                ]
                best_per_col[(bench, sub)] = max(vals) if vals else None

        for ri, (meth, access, _why) in enumerate(CONTROL_ROWS):
            cells, ids, vals = [], [], []
            for bench, subs in BENCH_BLOCKS:
                for sub in subs:
                    rec = grid.get((model, meth, bench, sub))
                    if rec is None:
                        cells.append("--")
                        ids.append("MISSING")
                        continue
                    v = rec["value"]
                    vals.append(v)
                    s = f"{v:.1f}"
                    if best_per_col[(bench, sub)] is not None and abs(v - best_per_col[(bench, sub)]) < 1e-9:
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
                    ids.append(rec["id"])
            avg = sum(vals) / len(vals) if vals else None
            avg_s = f"{avg:.1f}" if avg is not None else "--"
            if meth == "Koopman":
                avg_s = r"\textbf{" + avg_s + "}"
            label = method_name if meth == "Koopman" else meth
            # Model label rotated 90 degrees (manual edit of 2026-09-19, now reproduced here).
            first = (r"\multirow{" + str(len(CONTROL_ROWS)) + r"}{*}{\rotatebox{90}{" + model + "}}"
                     if ri == 0 else "")
            lines.append(
                f"{first} & {label} & {access} & " + " & ".join(cells) + f" & {avg_s}" + r" \\"
                + "  % " + " ".join(ids)
            )
            report[f"{model}/{meth}"] = {"cells": vals, "avg": avg}
        if mi < len(MODELS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n", report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-name", default="KOMPAS",
                    help="display name of the proposed method in Table 1 (decision D10)")
    args = ap.parse_args()

    main_entries = load(os.path.join(PAPER, "evidence", "numbers.yaml"))
    bench_entries = load(os.path.join(PAPER, "evidence", "numbers_benchmark.yaml"))

    grid_main = parse_main(main_entries)
    grid_bench = parse_bench(bench_entries)

    outdir = os.path.join(PAPER, "tables")
    os.makedirs(outdir, exist_ok=True)

    ctrl_tex, ctrl_report = build_control(grid_bench, args.method_name)
    with open(os.path.join(outdir, "tab_control.tex"), "w", encoding="utf-8") as fh:
        fh.write(ctrl_tex)

    pred_tex, pred_report, ranks = build_prediction(
        grid_main, PRED_COLUMNS, "tab:prediction",
        r"Multi-step prediction skill on seven multi-turn tasks (forecasts from step $t$ only, no reading fed back within the horizon). "  # 2026-09-22: was "Free-running", a term the text never defines
        r"Higher is better. Best per column in bold, second underlined.",
        header_note="MAIN-TEXT TABLE 1 of 2 (modeling line). D18: the main text carries exactly two tables.",
    )
    with open(os.path.join(outdir, "tab_prediction.tex"), "w", encoding="utf-8") as fh:
        fh.write(pred_tex)

    app_tex, app_report, _ = build_prediction(
        grid_main, APPENDIX_COLUMNS, "tab:prediction_small_app",
        r"The four small-sample columns, reported for completeness. "
        r"No conclusion in the main text is drawn from them.",
        header_note="APPENDIX A2 table. Not one of the two main-text tables.",
    )
    with open(os.path.join(outdir, "tab_prediction_small.tex"), "w", encoding="utf-8") as fh:
        fh.write(app_tex)

    # How many of the seven columns does the linear special case win outright?
    ours = "KOMPAS (linear)"
    wins = sum(
        1 for col_key, _ in PRED_COLUMNS
        if pred_report[ours][col_key] is not None
        and ranks[col_key][0] is not None
        and abs(pred_report[ours][col_key] - ranks[col_key][0]) < 1e-12
    )
    seconds = sum(
        1 for col_key, _ in PRED_COLUMNS
        if pred_report[ours][col_key] is not None
        and ranks[col_key][1] is not None
        and abs(pred_report[ours][col_key] - ranks[col_key][1]) < 1e-12
    )

    contrasts = parse_contrasts(main_entries)
    report = {
        "contrasts": contrasts,
        "prediction_main": pred_report,
        "prediction_appendix": app_report,
        "control": ctrl_report,
        "ours_best_of_7": wins,
        "ours_second_of_7": seconds,
        "control_deltas": {},
    }
    for model in MODELS:
        base = report["control"][f"{model}/Base"]["avg"]
        for meth in ("TMPC", "RE-Control", "Koopman"):
            report["control_deltas"][f"{model}/{meth} - Base"] = round(
                report["control"][f"{model}/{meth}"]["avg"] - base, 2)
        report["control_deltas"][f"{model}/Koopman - RE-Control"] = round(
            report["control"][f"{model}/Koopman"]["avg"]
            - report["control"][f"{model}/RE-Control"]["avg"], 2)

    with open(os.path.join(outdir, "_build_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("wrote tables/tab_prediction.tex (Table 1, main), tables/tab_control.tex (Table 2, main),")
    print("      tables/tab_prediction_small.tex (appendix A2, NOT a main-text table)")
    print(f"prediction: ours best in {wins}/7 columns, second in {seconds}/7")
    print("control deltas (avg over 11 subtasks):")
    for k, v in report["control_deltas"].items():
        print(f"  {k}: {v:+.2f}")
    print("paired contrasts (Skill_H difference, CI excluding zero marked *):")
    for kind, cols in contrasts.items():
        shown = [c for c, _ in PRED_COLUMNS if c in cols]
        marks = " ".join(
            f"{c}={cols[c]['value']:+.3f}{'*' if cols[c]['excludes_zero'] else ''}"
            for c in shown)
        n_star = sum(1 for c in shown if cols[c]["excludes_zero"])
        print(f"  {kind}: {n_star}/{len(shown)} exclude 0 | {marks}")

    missing = [k for k, v in grid_main.items() if v["skill"] is None]
    if missing:
        print(f"WARNING: {len(missing)} cells with no relative value: {missing[:5]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
