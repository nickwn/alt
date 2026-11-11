#!/bin/sh
# Regenerates every figure in blog/img/conformal_sphere (and prints the numbers quoted in the post).
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/repos/shapegen/.venv/bin/python}
for f in overview coords iterates mobius convergence spectrum cmcf_time metrics; do
  echo "== $f"; "$PY" $f.py || exit 1
done
