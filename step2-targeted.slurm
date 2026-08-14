#!/bin/bash
#SBATCH --partition=C064M0256G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --job-name=het-step2-targeted
#SBATCH --chdir=/lustre/home/2501111916/Het-Infer-workspace/DOPS
#SBATCH --output=/lustre/home/2501111916/Het-Infer-workspace/DOPS/step2-targeted-%j.out

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/het-step2-pycache-${SLURM_JOB_ID}"
PY=/lustre/home/2501111916/anaconda3/bin/python3.12
HET=/lustre/home/2501111916/Het-Infer-workspace/Het-Infer

host_name="$(hostname -f 2>/dev/null || hostname)"
case "${host_name}" in
  *login*|*data*)
    echo "Refusing to validate on non-compute host ${host_name}" >&2
    exit 2
    ;;
esac

"$PY" -m py_compile "$HET/scripts/hpc/train_dqn_digital_twin.py"
bash -n "$HET/scripts/hpc/train_dqn_digital_twin.slurm"
if grep -n -E -- '--dops-prior|DQN_PRIOR|load_real_dops_warm_start_samples' \
  "$HET/scripts/hpc/train_dqn_digital_twin.py" \
  "$HET/scripts/hpc/train_dqn_digital_twin.slurm"; then
  echo "retired score-bearing prior training entry remains" >&2
  exit 1
fi

"$PY" -m unittest discover -s tests -p 'test_hetinfer_prior.py' -v
"$PY" -m unittest discover -s tests -p 'test_hetinfer_prior_export.py' -v
"$PY" -m unittest discover -s tests -p 'test_hetinfer_scheduler_export.py' -v
"$PY" -m unittest discover -s tests -p 'test_hetinfer_prior_cli.py' -v
git diff --check
cd "$HET"
PYTHONPATH="$HET/src" "$PY" -m unittest tests.test_prior_contract -v
PYTHONPATH="$HET/src" "$PY" -m unittest \
  tests.test_cli.CLITest.test_inspect_prior_validates_the_static_dops_artifact \
  tests.test_dqn_vertical_slice.DQNVerticalSliceTest.test_retired_score_prior_is_not_a_public_training_input \
  -v
git diff --check
printf '%s\n' STEP2_TARGETED_OK
