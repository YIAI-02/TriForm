# Measured GPU calibration for DOPS

This workflow replaces the broad-grid GPU **proxy** with parameters measured
on the CUDA device actually allocated by Slurm. It leaves the PIM model
unchanged. The proxy files under `configs/hetinfer_gpu_proxy/` remain the
fallback when no measured artifact is available.

## Outputs and provenance

One GPU job writes five JSON files:

1. `gpu_microbench_raw.json`: machine, Slurm allocation, torch/CUDA/cuDNN,
   detected GPU name/capability/memory, benchmark shapes, warmups, repeats,
   every raw CUDA-event latency sample, and any skipped/OOM cases.
2. `gpu_calibration_fit.json`: per-shape summaries, the GEMM latency fit,
   fitted utilization curve, launch intercept, D2D HBM bandwidth, pinned
   H2D/D2H bandwidth, recommendations, and the raw-file SHA-256.
3. `gpu0_runtime_model.json`: the data-only utilization and GEMM-overhead
   functions loaded by DOPS at runtime.
4. `hardware_gpu0_pim0_calibrated.json`: `GPU0` effective peak, HBM bandwidth,
   capacity, and measured host link; PIM values remain inherited/unmeasured.
5. `evaluate_qwen1p8b_gpu_calibrated.json`: selects `npu_backend=fast` and
   points to the measured hardware/runtime model.

The export uses GiB/s because the current DOPS implementation multiplies its
legacy `*_GBs` fields by `1024^3`. The raw file also reports conventional
decimal GB/s to keep the measurement interpretation explicit.

## Submit

There is intentionally no partition in the Slurm file. Supply the real GPU
partition at submission time:

```bash
export DOPS_ROOT=/lustre/home/2501111916/HeteroLLM-workspace/DOPS-HetInfer
export DOPS_PYTHON=/absolute/path/to/python-with-torch
export DOPS_GPU_CALIBRATION_OUTPUT=/lustre/home/2501111916/HeteroLLM-workspace/results/gpu_calibration/run_001

sbatch \
  --partition=<actual_gpu_partition> \
  --export=ALL \
  measurements/gpu_calibration/gpu_calibration.slurm
```

To ensure Slurm allocated the intended family, add a regular expression. The
tool checks the name returned by `torch.cuda.get_device_properties`; it never
infers the GPU from the partition name.

```bash
export DOPS_EXPECT_GPU_REGEX='A800|A100'
```

Do not set this to `A800` merely because an A800 is expected. If the detected
device does not match, the job fails before producing a fit.

The script contains `#SBATCH --gres=gpu:1` but no `#SBATCH --partition`.
Compilation or execution must occur in a compute allocation, not on a login
node.

## Use the calibrated config for the DOPS prior grid

The broad grid runner keeps LLMCompass as its default fallback. Override both
the config and backend after calibration:

```bash
export DOPS_PRIOR_FAST_CONFIG="${DOPS_GPU_CALIBRATION_OUTPUT}/evaluate_qwen1p8b_gpu_calibrated.json"
export DOPS_PRIOR_FAST_NPU_BACKEND=fast
sbatch --partition=<cpu_partition> --export=ALL \
  commands/hetinfer_gpu_proxy/prior_grid_fast.slurm
```

This second job is still an offline simulator job and does not need a GPU.

## Model boundary

The calibration covers `torch.mm`, device-to-device `Tensor.copy_`, and pinned
host-to-device/device-to-host `Tensor.copy_`. It does not measure softmax,
normalization, activation, attention kernels, collectives, KV-cache handling,
CUDA Graphs, actual model execution, or the PIM. Therefore it improves the
GPU roofline prior but is not an end-to-end inference calibration.
