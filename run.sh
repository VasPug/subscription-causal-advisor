#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"  # ~/.cache is not writable here
if command -v uv >/dev/null; then
  [ -x .venv/bin/python ] || uv venv -q .venv
  uv pip install -q -p .venv/bin/python -r requirements.txt
else
  # fallback: this box's python3 lacks ensurepip, so bootstrap pip into the venv
  if [ ! -x .venv/bin/python ]; then
    python3 -m venv --without-pip .venv
    curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python - -q
  fi
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python test_sanity.py
.venv/bin/python run.py "$@"
