#!/usr/bin/env bash
set -e

python main.py weight-suggest \
  --config ./examples/weight_suggest_test_config.json \
  --debug
  
echo "weight suggest mode done."