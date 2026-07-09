#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  —  One-command launcher (no manual venv activation needed)
#
# Usage:
#   ./run.sh                                          # runs main.py (default)
#   ./run.sh --csv 02-15-2018.csv --tag 02_15_2018   # runs run_on_file.py
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

# Activate the virtual environment automatically
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
  echo "[run.sh] Virtual environment activated (.venv/bin/python)"
else
  echo "[run.sh] WARNING: .venv not found — using system Python"
fi

# If --csv flag is passed, route to run_on_file.py, otherwise main.py
if echo "$@" | grep -q "\-\-csv"; then
  echo "[run.sh] Running: python run_on_file.py $@"
  python run_on_file.py "$@"
else
  echo "[run.sh] Running: python main.py $@"
  python main.py "$@"
fi
