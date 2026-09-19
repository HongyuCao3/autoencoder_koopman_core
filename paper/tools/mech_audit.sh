#!/usr/bin/env bash
# Mechanical audit gate (WRITING_PLAN_2026-09-18.md §4.3).
# Runs every check that a script can decide, in order. Nothing here judges meaning;
# a clean run is the precondition for compiling and for sending the file to cold review.
# Usage: tools/mech_audit.sh paper/sections/03b_control.tex [more.tex ...]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FAIL=0
FILES=("$@")

if [ ${#FILES[@]} -eq 0 ]; then
  echo "usage: mech_audit.sh <file.tex> [more.tex ...]" >&2
  exit 2
fi

banner() { printf '\n--- %s ---\n' "$1"; }

banner "1. sentence gate (<=30 words, <=2 turns; D8)"
python3 "$HERE/sentence_gate.py" "${FILES[@]}" || FAIL=1

banner "2. equation gate (intuition before, symbol gloss after)"
python3 "$HERE/equation_gate.py" "${FILES[@]}" || FAIL=1

banner "3. em-dash (Rule 19)"
if grep -nE -- '---' "${FILES[@]}"; then
  echo "FAIL: replace every --- with a sentence break, colon, or parenthetical."
  FAIL=1
else
  echo "PASS"
fi

banner "4. adverbs (Rule 20)"
ADV='\b(significantly|substantially|considerably|dramatically|drastically|vastly|remarkably|notably|markedly|materially|highly|extremely|particularly|especially|decisively|strongly|weakly|robustly|sharply|severely|clearly|obviously|essentially|fundamentally|intrinsically|inherently|truly|indeed|surely|certainly|actually|effectively|practically|simply|merely|just|relatively|comparatively|somewhat|fairly|slightly|moderately|largely|mostly|primarily|nearly|virtually|roughly|approximately|freshly|already|uniformly|overwhelmingly|empirically|eventually|explicitly|directly|monotonically|exactly|formally)\b'
# A line may keep one adverb only by declaring why, on that same line:
#   % GATE-EXEMPT: adverb <word> -- <reason>
# The exemption is visible in every diff and greppable, so "PASS" keeps meaning PASS.
# Rule 20 exists to kill vague intensifiers; a mathematical quantifier is a different animal.
if grep -nEi "$ADV" "${FILES[@]}" | grep -v 'GATE-EXEMPT: adverb'; then
  echo "FAIL: replace each adverb with a number, a delta, or a structural fact."
  echo "      Whitelist (precision-bearing, house-voice V3): deterministically; exactly for cardinality;"
  echo "      monotonically when the sequence is not spelled out."
  echo "      To keep one deliberately, append on the same line: % GATE-EXEMPT: adverb <word> -- <reason>"
  FAIL=1
else
  echo "PASS"
  n=$(grep -c 'GATE-EXEMPT: adverb' "${FILES[@]}" 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
  [ "$n" -gt 0 ] && echo "      ($n declared exemption(s); read them, they are not free)"
fi

banner "5. \\cite (user decision D7)"
if grep -nE '\\cite[a-z]*\{' "${FILES[@]}"; then
  echo "FAIL: no \\cite in this phase. Mark the place with a standalone line: % CITE: <key>"
  FAIL=1
else
  echo "PASS"
fi

# Method-only rules.
METHOD_FILES=()
for f in "${FILES[@]}"; do
  case "$f" in *02_method.tex) METHOD_FILES+=("$f");; esac
done

if [ ${#METHOD_FILES[@]} -gt 0 ]; then
  banner "6. results inside Method (Rule 21)"
  if grep -nE 'Table~\\ref|sec:exp|outperform|baseline[s]? (beat|improve)|\+[0-9.]+ ?(pp|%)' "${METHOD_FILES[@]}"; then
    echo "FAIL: Method explains what the design does and why, never how well it scored."
    FAIL=1
  else
    echo "PASS"
  fi

  banner "7. math inside Why-X paragraphs (Rule 22)"
  python3 - "${METHOD_FILES[@]}" <<'PY' || FAIL=1
import re, sys
bad = 0
for path in sys.argv[1:]:
    lines = open(path, encoding='utf-8').read().split('\n')
    inside = False
    for i, line in enumerate(lines, 1):
        if re.search(r'\\(textbf|paragraph)\{\s*Why', line):
            inside = True
        if inside and not line.strip():
            inside = False
            continue
        if inside and ('$' in line or re.search(r'\\eqref|\\ref\{eq:', line)):
            print(f"  {path}:{i}  math or equation ref inside a Why-X paragraph: {line.strip()[:70]}")
            bad += 1
print("FAIL: Why-X paragraphs are formula-free, number-free, symbol-free." if bad else "PASS")
sys.exit(1 if bad else 0)
PY
fi

banner "result"
if [ "$FAIL" -eq 0 ]; then
  echo "mech_audit: PASS — file may be compiled and sent to cold review."
else
  echo "mech_audit: FAIL — fix every hit above before compiling."
fi
exit "$FAIL"
