#!/bin/bash
#SBATCH --job-name=tiny-moe-bundle
#SBATCH --partition=C064M0256G
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=results/tiny_moe_bundle.%j.out
#SBATCH --error=results/tiny_moe_bundle.%j.err
set -euo pipefail
cd /lustre/home/2501111916/workspace/Het-Infer/dops-camc-work
export RAMULATOR2_BIN=/lustre/home/2501111916/workspace/Het-Infer/current/vendor/dops/submodules/CENT/aim_simulator/build/ramulator2
export LD_LIBRARY_PATH=/lustre/software/gcc/12.2.0/lib64:/lustre/home/2501111916/workspace/Het-Infer/current/vendor/dops/submodules/CENT/aim_simulator
export PIM_TRACE_SCALE_REPEATS=0 PIM_TRACE_STRICT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
export PIM_LATENCY_CACHE_FILE="${PIM_LATENCY_CACHE_FILE:-$PWD/output/tiny_moe_experiment/expert_aim_exact.pkl}"
/lustre/home/2501111916/anaconda3/bin/python src/hetinfer_experiment_export.py "$@"
