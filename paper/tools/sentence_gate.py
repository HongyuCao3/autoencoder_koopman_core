#!/usr/bin/env python3
"""Sentence gate for the ICLR 2027 draft (WRITING_PLAN_2026-09-18.md §4.2, user decision D8).

Two hard limits per sentence:
  * at most 30 words (LaTeX commands, math and \ref{} do not count),
  * at most 2 logical turns (neurips-write SKILL.md Rule 9 token table).

Exit code 1 when anything fails, so mech_audit.sh can stop before compilation.
"""
import re
import sys

MAX_WORDS = 30
MAX_TURNS = 2

# Rule 9 token table, plus the semicolon. "and", \emph{(i)} list markers and
# that/which/where relative clauses are explicitly NOT turns.
TURN_WORDS = [
    "however", "while", "yet", "whereas", "rather", "because", "so",
    "once", "when", "if", "although", "but",
]
TURN_RE = re.compile(r"\b(" + "|".join(TURN_WORDS) + r")\b", re.IGNORECASE)

STRIP_ENVS = [
    "equation", "equation*", "align", "align*", "aligned", "gather", "gather*",
    "table", "table*", "tabular", "figure", "figure*", "itemize", "enumerate",
    "abstract", "displaymath",
]


def strip_latex(text):
    """Return (clean_text, line_map) where line_map[i] is the source line of clean line i."""
    lines = text.split("\n")
    out, src = [], []
    skip_depth = 0
    env_begin = re.compile(r"\\begin\{(" + "|".join(re.escape(e) for e in STRIP_ENVS) + r")\}")
    env_end = re.compile(r"\\end\{(" + "|".join(re.escape(e) for e in STRIP_ENVS) + r")\}")
    for i, raw in enumerate(lines, start=1):
        # Drop comments, but keep escaped percent signs.
        line = re.sub(r"(?<!\\)%.*$", "", raw)
        if env_begin.search(line):
            skip_depth += 1
            # A displayed block ends the sentence a reader is parsing. Without this
            # marker the clause before the block glues onto the clause after it and
            # the gate reports a violation no reader would see.
            out.append(" . ")
            src.append(i)
            continue
        if env_end.search(line):
            skip_depth = max(0, skip_depth - 1)
            out.append(" . ")
            src.append(i)
            continue
        if skip_depth:
            continue
        # Display and inline math carry no words.
        line = re.sub(r"\$\$.*?\$\$", " MATH ", line, flags=re.S)
        line = re.sub(r"\$[^$]*\$", " MATH ", line)
        line = re.sub(r"\\\[.*?\\\]", " MATH ", line, flags=re.S)
        # Cross references and citations are not words.
        line = re.sub(r"\\(ref|eqref|autoref|cref|Cref|cite[a-z]*|label)\{[^}]*\}", " REF ", line)
        # Section titles are labels, not prose sentences: drop them whole, and leave a
        # boundary behind so the title's words never join the first body sentence.
        line = re.sub(r"\\(section|subsection|subsubsection)\*?\{[^}]*\}", " . ", line)
        # \paragraph{Why X.} is a prose lead-in: keep its text, drop the command.
        line = re.sub(r"\\paragraph\*?\{", " ", line)
        # Any remaining command: drop the command name, keep braces content.
        line = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", line)
        line = line.replace("{", " ").replace("}", " ").replace("&", " ").replace("\\\\", " ")
        out.append(line)
        src.append(i)
    return out, src


def split_sentences(clean_lines, src_lines):
    """Yield (sentence, source_line). Split on . ! ? followed by whitespace + capital."""
    blob, index = [], []
    for line, ln in zip(clean_lines, src_lines):
        for ch in line + " ":
            blob.append(ch)
            index.append(ln)
    text = "".join(blob)
    sentences = []
    start = 0
    pat = re.compile(r"[.!?](?=\s+[A-Z(])|[.!?]\s*$")
    for m in pat.finditer(text):
        end = m.end()
        chunk = text[start:end].strip()
        if chunk:
            sentences.append((chunk, index[min(start, len(index) - 1)]))
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append((tail, index[min(start, len(index) - 1)]))
    return sentences


def word_count(sentence):
    # MATH / REF placeholders do not count as words.
    tokens = [t for t in sentence.split() if t not in ("MATH", "REF")]
    tokens = [t for t in tokens if re.search(r"[A-Za-z0-9]", t)]
    return len(tokens)


def turn_count(sentence):
    hits = [m.group(1).lower() for m in TURN_RE.finditer(sentence)]
    hits += [";"] * sentence.count(";")
    return hits


def check(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    clean, src = strip_latex(text)
    violations = []
    for sentence, line in split_sentences(clean, src):
        n = word_count(sentence)
        turns = turn_count(sentence)
        if n > MAX_WORDS or len(turns) > MAX_TURNS:
            violations.append({
                "file": path,
                "line": line,
                "words": n,
                "turns": turns,
                "head": " ".join(sentence.split())[:60],
            })
    return violations


def main(argv):
    if len(argv) < 2:
        print("usage: sentence_gate.py <file.tex> [more.tex ...]", file=sys.stderr)
        return 2
    all_v = []
    for path in argv[1:]:
        all_v.extend(check(path))
    if not all_v:
        print(f"sentence_gate: PASS ({len(argv) - 1} file(s), limits: <={MAX_WORDS} words, <={MAX_TURNS} turns)")
        return 0
    print(f"sentence_gate: FAIL ({len(all_v)} violation(s))")
    for v in all_v:
        why = []
        if v["words"] > MAX_WORDS:
            why.append(f"{v['words']} words")
        if len(v["turns"]) > MAX_TURNS:
            why.append(f"{len(v['turns'])} turns {v['turns']}")
        print(f"  {v['file']}:{v['line']}  [{'; '.join(why)}]  {v['head']}...")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
