#!/usr/bin/env python3
"""fig_symbol_audit.py -- gate G2 of FIG_PLAN_2026-09-19.

Every symbol that appears inside math mode in a figure fragment must also
appear, with the same spelling, in paper/sections/02_method.tex.  The figure
writes symbols only through the macros in figs/method_fig/symbols.tex, so the
audit first expands those macros, then compares symbol ATOMS (a base symbol
plus its sub/superscripts) against the atoms found in the method section.

usage:  python paper/tools/fig_symbol_audit.py <fragment.tex> [...]
exit 0  = every atom matched; exit 1 = at least one unmatched atom.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
METHOD = PAPER / "sections" / "02_method.tex"
SYMBOLS = PAPER / "figs" / "method_fig" / "symbols.tex"

# TeX spelling/typography, not symbols of the model.
IGNORE = {
    r"\left", r"\right", r"\big", r"\Big", r"\bigl", r"\bigr", r"\qquad",
    r"\quad", r"\,", r"\;", r"\!", r"\:", r"\top", r"\cdot", r"\cdots",
    r"\ldots", r"\dots", r"\vdots", r"\mid", r"\sim", r"\le", r"\leq",
    r"\ge", r"\geq", r"\in", r"\to", r"\subseteq", r"\star", r"\sum",
    r"\prod", r"\min", r"\max", r"\arg", r"\operatorname", r"\text",
    r"\mathrm", r"\mathbb", r"\mathcal", r"\widehat", r"\hat", r"\bm",
    r"\lvert", r"\rvert", r"\lVert", r"\rVert", r"\vert", r"\Vert",
    r"\times", r"\circ", r"\colon", r"\begin", r"\end", r"\\",
    r"\bmatrix", r"\aligned", r"\triangleq", r"\approx", r"\neq",
    r"\partial", r"\nabla", r"\mathbb{R}", r"\pm", r"\cup", r"\cap",
    r"\mathbf", r"\displaystyle", r"\notin", r"\forall", r"\exists",
}

MATH_PATTERNS = [
    re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.S),
    re.compile(r"\\\((.+?)\\\)", re.S),
]
ENV_PATTERN = re.compile(
    r"\\begin\{(equation|align|aligned|equation\*|align\*|gather)\}(.*?)"
    r"\\end\{\1\}", re.S)

ATOM = re.compile(
    r"(?:\\[a-zA-Z]+|[A-Za-z])"          # base symbol
    r"(?:[_^](?:\{[^{}]*\}|\\[a-zA-Z]+|[A-Za-z0-9\*\\]))*"   # sub/superscripts
)


def load_macros(path):
    """macro name -> replacement body, from \newcommand{\name}{body}."""
    out = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"\\newcommand\{(\\[A-Za-z]+)\}\{", text):
        name = m.group(1)
        i = m.end() - 1
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[name] = text[i + 1:j]
    return out


def expand(s, macros, rounds=6):
    for _ in range(rounds):
        new = re.sub(r"\\[A-Za-z]+",
                     lambda m: macros.get(m.group(0), m.group(0)), s)
        if new == s:
            break
        s = new
    return s


def math_spans(text, include_envs=False):
    spans = []
    for pat in MATH_PATTERNS:
        spans += pat.findall(text)
    if include_envs:
        spans += [m.group(2) for m in ENV_PATTERN.finditer(text)]
    return spans


def normalise(s):
    # every \cmd{X} becomes a single backslash token, so that stripping the
    # whitespace afterwards can never glue two symbols into one atom
    # (\mid\mathcal{H}_t must not read as one symbol "\midcalH_t").
    s = re.sub(r"\\(?:widehat|hat)\{([^{}]*)\}", r"\\hat\1", s)
    s = re.sub(r"\\(?:bm|mathbf)\{([^{}]*)\}", r"\\bf\1", s)
    s = re.sub(r"\\mathcal\{([^{}]*)\}", r"\\cal\1", s)
    s = re.sub(r"\\mathbb\{([^{}]*)\}", r"\\bb\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\\rm\1", s)
    s = re.sub(r"\\operatorname\*?\{[^{}]*\}", " ", s)
    s = re.sub(r"\\begin\{[^{}]*\}|\\end\{[^{}]*\}", " ", s)
    s = re.sub(r"\\[,;!: ]", " ", s)
    s = re.sub(r"\s+", "", s)
    return s


def atoms(spans):
    found = set()
    for sp in spans:
        for a in ATOM.findall(normalise(sp)):
            base = re.match(r"\\[a-zA-Z]+|[A-Za-z]", a).group(0)
            if base in IGNORE or a in IGNORE:
                continue
            if re.fullmatch(r"[0-9]+", a):
                continue
            found.add(a)
    return found


def main(argv):
    if not METHOD.exists():
        print(f"FATAL: {METHOD} not found")
        return 2
    macros = load_macros(SYMBOLS)
    ref = atoms(math_spans(METHOD.read_text(encoding="utf-8",
                                            errors="replace"),
                           include_envs=True))
    bad_total = 0
    for arg in argv:
        p = Path(arg)
        if not p.is_absolute():
            p = Path.cwd() / arg
        if not p.exists():
            print(f"{arg}: NOT FOUND")
            bad_total += 1
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        spans = [expand(s, macros) for s in math_spans(raw)]
        got = atoms(spans)
        missing = sorted(a for a in got if a not in ref)
        # a bare base letter counts as matched if the base occurs in §3
        missing = [a for a in missing
                   if not any(r.startswith(a) or a.startswith(r) for r in ref)]
        if missing:
            bad_total += len(missing)
            print(f"{p.name}: {len(missing)} UNMATCHED -> {', '.join(missing)}")
        else:
            print(f"{p.name}: OK ({len(got)} atoms, 0 unmatched)")
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
