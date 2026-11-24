#!/usr/bin/env bash
# Creates .venv next to this script and installs the dependencies.
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/pip ]; then
  rm -rf .venv
  if ! python3 -m venv .venv 2>/dev/null; then
    # Debian/Ubuntu without the python3-venv package: create the environment without pip
    # and install pip with the official PyPA bootstrap script (no sudo needed).
    rm -rf .venv
    python3 -m venv --without-pip .venv
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python - -q
  fi
fi
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda', torch.cuda.is_available())"
