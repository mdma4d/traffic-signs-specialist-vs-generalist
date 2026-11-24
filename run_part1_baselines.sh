#!/usr/bin/env bash
# Part 1: zero-shot models (CLIP, SigLIP 2, a vision LLM) vs a fine-tuned ResNet-18.
# Needs an NVIDIA GPU (about 9 GB of GPU memory for the vision LLM) and data/testset.
# One failing approach does not stop the others.
#   bash run_part1_baselines.sh              all approaches
#   bash run_part1_baselines.sh siglip2      one of: clip siglip2 classifier vlm
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ -f data/testset/labels.csv ] || { echo "data/testset is missing: run  $PY scripts/make_testset.py"; exit 1; }

run() {
  echo "=== $*"
  "$PY" -u "$@" 2>&1 | grep --line-buffered -v -E 'Warning|warn'
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!!! FAILED: $*"
}

ONLY="${1:-all}"
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

want clip       && run bench/zeroshot.py --name clip --model openai/clip-vit-large-patch14
want siglip2    && run bench/zeroshot.py --name siglip2 --model google/siglip2-so400m-patch14-384
want classifier && run bench/classifier.py
want vlm        && run bench/vlm.py --name vlm_number --answer number --model "${VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
want vlm        && run bench/vlm.py --name vlm_name --answer name --model "${VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
"$PY" bench/report.py > /dev/null && echo "=== table: results/report.md"
echo "=== ALL DONE"
