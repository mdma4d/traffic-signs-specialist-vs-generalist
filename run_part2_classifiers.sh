#!/usr/bin/env bash
# Part 2: fine-tunes and measures every model in bench/finetune.py, each in its own process,
# so one failing model does not stop the others. Needs an NVIDIA GPU (up to 10 GB for ViT-B).
#   bash run_part2_classifiers.sh                    all 22 models, plus 3 repeats of ResNet-18
#   bash run_part2_classifiers.sh big_cnn vit_base   only the listed models
#   SKIP_DONE=1 bash run_part2_classifiers.sh        keep results that already exist
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ -f data/testset/labels.csv ] || { echo "data/testset is missing: run  $PY scripts/make_testset.py"; exit 1; }

run() {  # run <key> [seed]
  local key="$1" seed="${2:-0}" out="results/classifiers/$1.json"
  [ "$seed" = 0 ] || out="results/classifiers/${key}_seed$seed.json"
  if [ -n "${SKIP_DONE:-}" ] && [ -f "$out" ]; then echo "=== $key seed $seed: done, skipping"; return; fi
  echo "=== $key seed $seed"
  SEED="$seed" "$PY" -u bench/finetune.py "$key" 2>&1 | grep --line-buffered -v -E 'Warning|warn'
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!!! FAILED: $key"
}

if [ "$#" -gt 0 ]; then
  for key in "$@"; do run "$key"; done
else
  for key in $("$PY" -c "import sys; sys.path.insert(0, 'bench'); import finetune; print(' '.join(finetune.MODELS))"); do
    run "$key"
  done
  for seed in 1 2 3; do run resnet18_tv "$seed"; done  # run-to-run variance
fi
"$PY" bench/report_classifiers.py > /dev/null && echo "=== table: results/classifiers.md"
echo "=== ALL DONE"
