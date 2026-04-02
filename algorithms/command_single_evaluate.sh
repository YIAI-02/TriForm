#!/usr/bin/env bash
set -e

python main.py evaluate \
  --config ./examples/evaluate_test_config.json \
  --debug \
  "$@"

echo "Evaluate mode done."
