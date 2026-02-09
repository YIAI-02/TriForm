#!/bin/bash
#SBATCH -J verify_run_gpu
#SBATCH -p GPU40G
#SBATCH --qos=low
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o job.%j.out

set -eo pipefail

mkdir -p logs

echo "[slurm] jobid=${SLURM_JOB_ID} nodelist=${SLURM_NODELIST}"
echo "[slurm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

source ~/.bashrc
conda activate triform310

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

python schedule_deploy_verify.py run-gpu \
    --tasks ./out/hefthint_seg.gpu_tasks.json \
    --out ./out/hefthint_seg.gpu_results.json \
    --warmup 3 --iters 10