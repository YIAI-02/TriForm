#!/usr/bin/env bash
set -e

python main.py weight-suggest \
  --config ./examples/evaluate_test_config.json \
  --debug \
  --npu_backend lut
  
echo "weight suggest mode done."