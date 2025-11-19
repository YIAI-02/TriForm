#!/usr/bin/env bash
set -e

mkdir -p ./output/len_sweep

ALGOS=("heft" "sa" "rl" "ga" "astar")
ALGOS_CSV=$(IFS=, ; echo "${ALGOS[*]}")
# ALGOS=("heft")

python main.py --once \
      --result_dir ./output/len_sweep \
      --algo "$ALGOS_CSV" \
      --prefill_len 128  \
      --decode_len 128  \
      --decode_sample_stride 16 \
      --baselines ianus,neupims,attacc,facil,pd,weights_on_pim,attn_on_pim \
      --debug

python main.py --once \
      --result_dir ./output/len_sweep \
      --algo "$ALGOS_CSV" \
      --prefill_len 1024 \
      --decode_len 128  \
      --decode_sample_stride 16 \
      --baselines ianus,neupims,attacc,facil,pd,weights_on_pim,attn_on_pim \
      --debug

python main.py --once \
      --result_dir ./output/len_sweep \
      --algo "$ALGOS_CSV" \
      --prefill_len 128  \
      --decode_len 1024  \
      --decode_sample_stride 16 \
      --baselines ianus,neupims,attacc,facil,pd,weights_on_pim,attn_on_pim \
      --debug

python main.py --once \
      --result_dir ./output/len_sweep \
      --algo "$ALGOS_CSV" \
      --prefill_len 1024 \
      --decode_len 1024  \
      --decode_sample_stride 16 \
      --baselines ianus,neupims,attacc,facil,pd,weights_on_pim,attn_on_pim \
      --debug


echo "Done. See ./output/len_sweep/"

python main.py \
      --result_dir ./output/weight_suggestion_30 \
      --algo "heft" \
      --all_passes_json ./output/weight_suggestion_30/all_passes_results.json \
      --best_summary_json ./output/weight_suggestion_30/best_summary.json \
      --prefill_len 1024 \
      --decode_len 1024  \
      --decode_sample_stride 16 \
      --debug