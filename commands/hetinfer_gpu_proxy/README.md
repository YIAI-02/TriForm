# GPU0 + PIM0 prior-profile grid for Het-Infer

This directory prepares DOPS placement priors for the Het-Infer online device
selector. It does **not** execute the model on a physical GPU or PIM.

## Evidence classes

| Mode | GPU0 cost | PIM0 cost | Intended use |
| --- | --- | --- | --- |
| `fast` | LLMCompass `A100_80GB_fp16` proxy | Analytical FLOP/bandwidth estimate | Broad 18-point prior grid |
| `ramulator-scaled` | Same GPU proxy | CENT/AiM unit trace + Ramulator2, repeated work scaled arithmetically | Three trace-backed anchors |
| `ramulator-strict` | Same GPU proxy | Fully expanded CENT/AiM trace + Ramulator2 | One small integration validation |

The current DOPS source does not directly load the ATLAS C++ simulator. The
trace modes above are DOPS' own `CENT/AiM -> Ramulator2` path. An ATLAS result
must remain a separate Het-Infer co-simulation telemetry source until an
explicit DOPS-to-ATLAS cost-backend adapter exists.

`GPU0` is intentionally encoded as hardware `type: "npu"`, because that is
the accelerator device class understood by the current DOPS scheduler. Its
exported device name remains `GPU0`, so Het-Infer can consume it without
renaming the placement action.

## Profile axes

The broad grid is the Cartesian product:

- batch: `1, 4, 8`
- prefill length: `128, 512, 2048`
- decode horizon: `64, 256`

It produces 18 separate `dops.hetinfer_prior.v1` artifacts. Keeping runs
separate preserves each run's exact graph/config provenance. Het-Infer may
load the resulting manifest and select/interpolate among the profile
workloads; DOPS timeline timestamps are not a runtime dispatch contract.

## Submit from the login node

Only submit from the login node; all Python/simulator work runs in Slurm. Set
the checkout, Python, output, and Ramulator paths explicitly:

```bash
export DOPS_ROOT=/lustre/home/2501111916/HeteroLLM-workspace/DOPS-HetInfer
export DOPS_PYTHON=/absolute/path/to/python
export DOPS_PRIOR_OUTPUT_ROOT=/lustre/home/2501111916/HeteroLLM-workspace/results/dops_gpu_pim_prior_grid

sbatch --export=ALL commands/hetinfer_gpu_proxy/prior_grid_fast.slurm
```

For the trace-backed anchors, `RAMULATOR2_BIN` must be an already compiled
executable. Compilation must also happen in a compute allocation, never on the
login node.

```bash
export RAMULATOR2_BIN=/absolute/path/to/ramulator2
sbatch --export=ALL commands/hetinfer_gpu_proxy/prior_grid_ramulator_scaled.slurm
sbatch --export=ALL commands/hetinfer_gpu_proxy/prior_grid_ramulator_strict.slurm
```

After an array completes, build a validated manifest in a compute allocation:

```bash
export PYTHONPATH="${DOPS_ROOT}/src"
"${DOPS_PYTHON}" commands/hetinfer_gpu_proxy/build_prior_grid_manifest.py \
  --grid-root "${DOPS_PRIOR_OUTPUT_ROOT}/fast" \
  --require-count 18
```

The corresponding expected counts are 3 for `ramulator-scaled` and 1 for
`ramulator-strict`.

Each `prior.json` is accompanied by a uniquely named canonical
`<artifact_id>.source.json`. Treat those two files as one bundle. The manifest
records both relative paths under `bundle_files`; a training-stage copy must
preserve both files in the same directory and must not rename the source
sidecar. Het-Infer intentionally rejects a `DQN_PRIOR` whose source companion
is absent or whose byte digest differs.

## What this does and does not prove

Successful jobs prove that DOPS can build the Qwen 1.8B DAG, score legal
`GPU0`/`PIM0` candidates under the named cost models, and export validated
placement priors. They do not prove physical A100/A800 latency, real PIM
latency, GPU/PIM overlap, token correctness, or end-to-end inference speedup.
