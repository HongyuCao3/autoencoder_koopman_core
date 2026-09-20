#!/usr/bin/env bash
# usage: ./build.sh modA_memory   |  ./build.sh method_figure
set -e; cd "$(dirname "$0")"; f=$1
v=$(ls preview/step_${f}_v*.png 2>/dev/null | wc -l); v=$((v+1))
if [ "$f" = method_figure ]; then
  pdflatex -interaction=nonstopmode -halt-on-error method_figure.tex >/dev/null
  # sections/03_method.tex includes figs/method.pdf; publish the build product there.
  cp method_figure.pdf ../method.pdf
else
  pdflatex -interaction=nonstopmode -halt-on-error -jobname=$f "\def\MOD{$f}\input{preview_module}" >/dev/null
fi
pdftoppm -r 200 -png -singlefile $f.pdf preview/step_${f}_v$v
echo preview/step_${f}_v$v.png
