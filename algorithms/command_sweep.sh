set -e 

python3 sweep_hefthint.py   --mode grid   --gamma 0 0.2 0.4 0.6 --lambda_ 0 4 8 --eta 300 400 500 --objective total   --outdir ./output/sweep_hefthint_manual_scale_down_qwen1.8b  --resume

echo "Sweep done."