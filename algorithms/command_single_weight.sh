#!/usr/bin/env bash
set -e

python main.py weight-suggest --config ./examples/evaluate_test_config.json --debug --npu_fast_mode --pim_fast_mode

echo "weight suggest mode done."