#!/usr/bin/env python3
"""Equation gate (user requirement, 2026-09-19; WRITING_PLAN §4.1 rule 3 made mechanical).

Two things must hold for every displayed equation in the paper:

  BEFORE  an intuition sentence, in the same paragraph, immediately preceding the equation.
          A reader should know what the equation is about before meeting the symbols.
  AFTER   a gloss for every symbol the equation puts on its right-hand side, within the
          prose that follows it.

Ground truth for "which symbols" is contract.yaml: equation_lattice[].rhs_symbols points into
symbol_lattice, so this gate checks the prose against the contract rather than against a guess.

Usage:
    python3 tools/equation_gate.py paper/sections/02_method.tex [more.tex ...]
Exit code 1 if any equation fails.
"""
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
LOOKAHEAD_LINES = 18          # how far after \end{equation} a gloss may sit
ENV = r"equation\*?|align\*?|gather\*?"


def load_contract():
    with open(os.path.join(PAPER, "contract.yaml"), encoding="utf-8") as fh:
        c = yaml.safe_load(fh)
    symbols = {s["id"]: s["symbol"] for s in c.get("symbol_lattice", [])}
    eqs = {}
    for e in c.get("equation_lattice", []):
        names = []
        for sid in e.get("rhs_symbols", []) or []:
            raw = symbols.get(sid)
            if not raw:
                continue
            # one lattice row may carry a pair, e.g. "\varepsilon_{a}, \varepsilon_{b}"
            for piece in raw.split(","):
                piece = piece.strip()
                if piece:
                    names.append((sid, piece))
        eqs[e["label"]] = names
    return eqs


def strip_comment(line):
    return re.sub(r"(?<!\\)%.*$", "", line)


def blocks(lines):
    """Yield (label, begin_idx, end_idx) for each displayed equation carrying a \\label."""
    begin = re.compile(r"\\begin\{(" + ENV + r")\}")
    end = re.compile(r"\\end\{(" + ENV + r")\}")
    i, n = 0, len(lines)
    while i < n:
        if begin.search(strip_comment(lines[i])):
            j = i + 1
            while j < n and not end.search(strip_comment(lines[j])):
                j += 1
            body = "\n".join(lines[i:j + 1])
            m = re.search(r"\\label\{([^}]*)\}", body)
            yield (m.group(1) if m else None), i, min(j, n - 1)
            i = j + 1
        else:
            i += 1


def mask_algorithmic(lines):
    """Blank out the body of algorithmic blocks (D34).

    Pseudocode names symbols the prose has already introduced, in an order the gate would
    otherwise read as a first occurrence. Rule 21's result-number grep in mech_audit.sh still
    covers the block; only this gate's symbol bookkeeping ignores it.
    """
    out, inside = [], False
    for raw in lines:
        stripped = strip_comment(raw)
        if re.search(r"\\begin\{algorithmic\}", stripped):
            inside = True
            out.append(raw)
            continue
        if re.search(r"\\end\{algorithmic\}", stripped):
            inside = False
            out.append(raw)
            continue
        out.append("%" + raw if inside else raw)
    return out


def intuition_before(lines, begin_idx):
    """True when a prose sentence sits in the same paragraph just above the equation.

    D34: a Proposition's opening line is not a paragraph break for this purpose. The intuition
    sentence for a bound stated inside a proposition sits in the paragraph above the environment.
    """
    k = begin_idx - 1
    while k >= 0:
        raw = strip_comment(lines[k]).strip()
        if re.match(r"\\(begin|end)\{(proposition|lemma|theorem)\}", raw):
            k -= 1
            continue
        if not raw:
            return False                      # paragraph break: the equation opens a paragraph
        if re.match(r"\\(section|subsection|subsubsection|paragraph|label|end|begin|item)", raw):
            return False
        # a line with actual words, not only math or a command
        words = re.sub(r"\$[^$]*\$", " ", raw)
        words = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", words)
        if len(re.findall(r"[A-Za-z]{3,}", words)) >= 3:
            return True
        k -= 1
    return False


def first_occurrence(lines, latex):
    """Line index where this symbol string first appears anywhere in the file, or None."""
    for i, raw in enumerate(lines):
        if latex in strip_comment(raw):
            return i
    return None


def doc_order(main_tex):
    """The \input order in main.tex, as paths relative to the paper directory.

    A symbol introduced in an earlier section is already introduced by the time a later
    section uses it. Without this the gate reads each file as if it were the whole paper
    and demands a re-gloss the prose should not contain (Rule 10).
    """
    try:
        with open(main_tex, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return []
    out = []
    for m in re.finditer(r"\\input\{([^}]*)\}", text):
        rel = m.group(1)
        if not rel.endswith(".tex"):
            rel += ".tex"
        out.append(os.path.join(PAPER, rel))
    return out


def prior_text(target, order):
    """Everything that appears before `target` in document order, concatenated."""
    chunks = []
    for path in order:
        if os.path.abspath(path) == os.path.abspath(target):
            break
        try:
            with open(path, encoding="utf-8") as fh:
                chunks.append(fh.read())
        except OSError:
            continue
    return "\n".join(chunks)


def gloss_after(lines, end_idx, wanted):
    """Which of `wanted` [(sid, latex)] do not appear in the prose after the equation."""
    window = []
    k = end_idx + 1
    taken = 0
    while k < len(lines) and taken < LOOKAHEAD_LINES:
        raw = strip_comment(lines[k])
        window.append(raw)
        if raw.strip():
            taken += 1
        k += 1
    text = "\n".join(window)
    missing = []
    for sid, latex in wanted:
        if latex not in text:
            missing.append((sid, latex))
    return missing


def check(path, eqs, earlier=""):
    with open(path, encoding="utf-8") as fh:
        lines = mask_algorithmic(fh.read().split("\n"))
    problems = []
    seen = 0
    for label, b, e in blocks(lines):
        seen += 1
        name = label or f"(unlabelled, line {b + 1})"
        if not intuition_before(lines, b):
            problems.append(
                f"  {path}:{b + 1}  [{name}] no intuition sentence immediately before the equation. "
                "Say in one sentence what the equation is about, in the same paragraph."
            )
        if label and label in eqs:
            # A symbol that this equation INTRODUCES must be explained right after it. A symbol
            # already introduced earlier in the file does not need re-glossing; demanding that
            # would push the prose into repeating itself, which Rule 9 and Rule 10 both fight.
            fresh = []
            for sid, latex in eqs[label]:
                if latex in earlier:
                    continue          # already introduced in an earlier section
                first = first_occurrence(lines, latex)
                if first is None or b <= first <= e:
                    fresh.append((sid, latex))
            missing = gloss_after(lines, e, fresh)
            if missing:
                shown = ", ".join(f"{latex} ({sid})" for sid, latex in missing)
                problems.append(
                    f"  {path}:{e + 1}  [{name}] introduces these symbols with no gloss within "
                    f"{LOOKAHEAD_LINES} lines: {shown}"
                )
        elif label:
            problems.append(f"  {path}:{b + 1}  [{name}] label is not in contract.yaml equation_lattice")
    return seen, problems


def main(argv):
    if len(argv) < 2:
        print("usage: equation_gate.py <file.tex> [more.tex ...]", file=sys.stderr)
        return 2
    eqs = load_contract()
    order = doc_order(os.path.join(PAPER, "main.tex"))
    total, all_problems = 0, []
    for path in argv[1:]:
        seen, problems = check(path, eqs, earlier=prior_text(path, order))
        total += seen
        all_problems.extend(problems)
    if not all_problems:
        print(f"equation_gate: PASS ({total} displayed equation(s) checked)")
        return 0
    print(f"equation_gate: FAIL ({len(all_problems)} problem(s) over {total} equation(s))")
    print("\n".join(all_problems))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
