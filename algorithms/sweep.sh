#!/usr/bin/env bash
set -e

mkdir -p ./output/len_sweep

# Four cases
python main.py --prefill_len 128  --decode_len 128  \
  --all_passes_json ./output/len_sweep/all_passes_128x128.json \
  --best_summary_json ./output/len_sweep/best_summary_128x128.json \
  --schedulers heft,sa,ga,rl \
  --run_baselines_after

python main.py --prefill_len 1024 --decode_len 128  \
  --all_passes_json ./output/len_sweep/all_passes_1024x128.json \
  --best_summary_json ./output/len_sweep/best_summary_1024x128.json \
  --schedulers heft,sa,ga,rl \
  --run_baselines_after

python main.py --prefill_len 128  --decode_len 1024 \
  --all_passes_json ./output/len_sweep/all_passes_128x1024.json \
  --best_summary_json ./output/len_sweep/best_summary_128x1024.json \
  --schedulers heft,sa,ga,rl \
  --run_baselines_after

python main.py --prefill_len 1024 --decode_len 1024 \
  --all_passes_json ./output/len_sweep/all_passes_1024x1024.json \
  --best_summary_json ./output/len_sweep/best_summary_1024x1024.json \
  --schedulers heft,sa,ga,rl \
  --run_baselines_after

echo "Done. See ./output/len_sweep/"
