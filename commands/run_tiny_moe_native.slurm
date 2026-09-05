#!/bin/bash
#SBATCH -J tiny_moe_native
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 04:00:00
#SBATCH -o results/tiny_moe_native.%j.out
#SBATCH -e results/tiny_moe_native.%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
cd /lustre/home/2501111916/workspace/Het-Infer/dops-camc-work
export RAMULATOR2_BIN=/lustre/home/2501111916/workspace/Het-Infer/current/vendor/dops/submodules/CENT/aim_simulator/build/ramulator2
export LD_LIBRARY_PATH=/lustre/software/gcc/12.2.0/lib64:/lustre/home/2501111916/workspace/Het-Infer/current/vendor/dops/submodules/CENT/aim_simulator
export PIM_TRACE_SCALE_REPEATS=0
export PIM_TRACE_STRICT=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/lustre/home/2501111916/anaconda3/bin/python commands/export_hetinfer_tiny_suite.py "$@"
