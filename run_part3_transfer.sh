#!/usr/bin/env bash
# Part 3: signs from another country (BelgiumTSC). Every model is trained on GTSRB again and
# evaluated on both sets right away; results go to results/transfer/. Needs an NVIDIA GPU.
#   bash run_part3_transfer.sh             all models, zero-shot CLIP/SigLIP 2, 3 repeats of 3 fast models
#   bash run_part3_transfer.sh big_cnn     only the listed models
#   SKIP_DONE=1 bash run_part3_transfer.sh keep results that already exist
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ -f data/testset/labels.csv ] || { echo "data/testset is missing: run  $PY scripts/make_testset.py"; exit 1; }
[ -f data/belgium/labels.csv ] || { echo "data/belgium is missing: run  $PY scripts/make_belgium.py"; exit 1; }
export RESULTS_SUBDIR=transfer
mkdir -p results/transfer

run() {  # run <output name> <script and args...>
  local out="results/transfer/$1.json"; shift
  if [ -n "${SKIP_DONE:-}" ] && [ -f "$out" ]; then echo "=== $out: done, skipping"; return; fi
  echo "=== $*"
  "$PY" -u "$@" 2>&1 | grep --line-buffered -v -E 'Warning|warn'
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!!! FAILED: $*"
}

if [ "$#" -gt 0 ]; then
  for key in "$@"; do run "$key" bench/finetune.py "$key"; done
else
  run zs_clip bench/zeroshot.py --name transfer/zs_clip --testset belgium --model openai/clip-vit-large-patch14
  run zs_siglip2 bench/zeroshot.py --name transfer/zs_siglip2 --testset belgium --model google/siglip2-so400m-patch14-384
  for key in $("$PY" -c "import sys; sys.path.insert(0, 'bench'); import finetune; print(' '.join(finetune.MODELS))"); do
    run "$key" bench/finetune.py "$key"
  done
  for seed in 1 2 3; do  # run-to-run variance of the three fast models
    for key in resnet18_tv small_cnn big_cnn; do
      SEED=$seed run "${key}_seed$seed" bench/finetune.py "$key"
    done
  done
fi
"$PY" bench/report_transfer.py > /dev/null && echo "=== table: results/transfer.md"
echo "=== ALL DONE"
