#!/usr/bin/env bash
# Part 4: classifier with a reject option. Trains every variant, saves the weights to models/,
# then checks multi-view and two-model agreement. Needs an NVIDIA GPU and all prepared sets:
# data/belgium, data/belgium_train, data/belgium_unknown, data/background.
#   bash run_part4_reject.sh           everything (about 40 minutes on an RTX 4090 Laptop)
#   bash run_part4_reject.sh final     only the final model: models/big_cnn_oe_all.pt (about 6 minutes)
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
for d in belgium belgium_train background/test; do
  [ -f "data/$d/labels.csv" ] || { echo "data/$d is missing: see \"Prepare the data\" in README.md"; exit 1; }
done
[ -f data/belgium_unknown/test/labels.csv ] || { echo "data/belgium_unknown is missing: run  $PY scripts/make_belgium_unknown.py"; exit 1; }

run() {
  echo "=== $*"
  "$PY" -u "$@" 2>&1 | grep --line-buffered -v -E 'Warning|warn|epoch'
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!!! FAILED: $*"
}

if [ "${1:-all}" = final ]; then
  run bench/reject.py oe_all big_cnn
else
  for key in resnet18_tv small_cnn big_cnn; do
    for mode in base oe oe_mix oe_bg; do run bench/reject.py "$mode" "$key"; done
  done
  for key in resnet18_tv big_cnn; do run bench/reject.py oe_all "$key"; done
  run bench/agree.py oe_bg
  run bench/agree.py oe_all
  run bench/tta.py models/big_cnn_oe_bg.pt
fi
"$PY" bench/report_reject.py > /dev/null && echo "=== table: results/reject.md"
echo "=== ALL DONE"
