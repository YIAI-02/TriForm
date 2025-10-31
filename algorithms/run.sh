#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"

if [ -d "$ROOT/output" ]; then
  echo "Removing: $ROOT/output"
  rm -rf -- "$ROOT/output"
fi

while IFS= read -r -d '' d; do
  echo "Removing: $d"
  rm -rf -- "$d"
done < <(find "$ROOT" -type d -name "format tunning" -print0)

if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "Python not found (tried: python, python3)" >&2
  exit 1
fi

exec "$PY" main.py
