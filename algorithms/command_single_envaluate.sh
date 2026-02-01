#!/usr/bin/env bash
set -e

# Evaluate mode: run multiple algos + baselines, derive outputs from result_dir
python main.py evaluate --config ./examples/evaluate_test_config.json --debug --pim_fast_mode --npu_backend llmcompass

echo "Evaluate mode done."

# Weight-suggest mode: multi-pass SA to propose weight formats
# python main.py weight-suggest --config ./examples/weight_suggest_config.json --debug

# echo "Weight-suggest mode done. See ./output/weight_suggestion_30/"
