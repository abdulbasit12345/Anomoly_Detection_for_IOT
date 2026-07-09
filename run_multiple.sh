#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_multiple.sh  —  Batch/Multi-file launcher (no manual venv activation needed)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

# Activate the virtual environment automatically
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
  echo "[run_multiple.sh] Virtual environment activated (.venv/bin/python)"
else
  echo "[run_multiple.sh] WARNING: .venv not found — using system Python"
fi

# Run the batch script
python run_multiple.py "$@"
