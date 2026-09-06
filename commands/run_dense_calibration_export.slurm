#!/bin/bash
#SBATCH -J dense-cal-export
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o results/dense_cal_export.%j.out
#SBATCH -e results/dense_cal_export.%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
cd /lustre/home/2501111916/workspace/Het-Infer/dops-camc-work
export RAMULATOR2_BIN=/lustre/home/2501111916/workspace/Het-Infer/current/vendor/dops/submodules/CENT/aim_simulator/build/ramulator2
export LD_LIBRARY_PATH=/lustre/software/gcc/12.2.0/lib64:/lustre/home/2501111916/workspace/Het-Infer/current/vendor/dops/submodules/CENT/aim_simulator
export PIM_TRACE_SCALE_REPEATS=0 PIM_TRACE_STRICT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
/lustre/home/2501111916/anaconda3/bin/python commands/export_hetinfer_dense_calibration_suite.py "$@"
