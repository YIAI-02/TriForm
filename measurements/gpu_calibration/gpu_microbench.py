"""Run CUDA GEMM and copy microbenchmarks and emit lossless raw samples.

PyTorch is imported only inside ``main`` so this module can be syntax-tested
on machines without CUDA. A successful run always verifies CUDA availability
and records the device name instead of inferring it from the Slurm partition.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import re
import socket
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calibration_common import RAW_SCHEMA, atomic_write_json, load_json_object


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expect-device-regex",
        help="Fail before benchmarking unless the detected torch device name matches",
    )
    return parser


def _positive_int(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{field} must be positive, got {number}")
    return number


def _auto_inner_iterations(flops: int, target_flops: int, maximum: int) -> int:
    if flops <= 0:
        return 1
    return max(1, min(maximum, math.ceil(target_flops / flops)))


def _cuda_samples_ms(
    torch: Any,
    operation: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    inner_iterations: int,
) -> list[float]:
    for _ in range(warmup):
        for _ in range(inner_iterations):
            operation()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_iterations):
            operation()
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end)) / inner_iterations
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
            raise RuntimeError(
                f"CUDA event returned invalid elapsed time {elapsed_ms!r}"
            )
        samples.append(elapsed_ms)
    return samples


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _nvidia_smi_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "query": command[1],
        "rows": [line.strip() for line in result.stdout.splitlines() if line.strip()],
        "stderr": result.stderr.strip(),
    }


def _slurm_snapshot() -> dict[str, str | None]:
    fields = {
        "job_id": "SLURM_JOB_ID",
        "job_name": "SLURM_JOB_NAME",
        "partition": "SLURM_JOB_PARTITION",
        "node_list": "SLURM_JOB_NODELIST",
        "job_gpus": "SLURM_JOB_GPUS",
        "step_gpus": "SLURM_STEP_GPUS",
        "cpus_per_task": "SLURM_CPUS_PER_TASK",
    }
    return {name: os.environ.get(variable) for name, variable in fields.items()}


def _dtype_from_name(torch: Any, name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise ValueError(
            f"unsupported dtype {name!r}; expected float16, bfloat16, or float32"
        )
    return mapping[key]


def _benchmark_matmuls(
    torch: Any, cfg: dict[str, Any], dtype: Any, device: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warmup = _positive_int(cfg.get("warmup", 10), "warmup")
    repeats = _positive_int(cfg.get("repeats", 30), "repeats")
    target_flops = _positive_int(
        cfg.get("matmul_target_sample_flops", 5_000_000_000),
        "matmul_target_sample_flops",
    )
    max_inner = _positive_int(
        cfg.get("matmul_max_inner_iterations", 100),
        "matmul_max_inner_iterations",
    )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    shapes = cfg.get("matmul_shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("matmul_shapes must be a non-empty array")

    for ordinal, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise TypeError(f"matmul_shapes[{ordinal}] must be an object")
        label = str(shape.get("label") or f"matmul_{ordinal}")
        m = _positive_int(shape.get("m"), f"{label}.m")
        n = _positive_int(shape.get("n"), f"{label}.n")
        k = _positive_int(shape.get("k"), f"{label}.k")
        flops = 2 * m * n * k
        inner = _positive_int(
            shape.get(
                "inner_iterations",
                _auto_inner_iterations(flops, target_flops, max_inner),
            ),
            f"{label}.inner_iterations",
        )
        lhs = rhs = output = None
        try:
            lhs = torch.empty((m, k), device=device, dtype=dtype)
            rhs = torch.empty((k, n), device=device, dtype=dtype)
            output = torch.empty((m, n), device=device, dtype=dtype)

            def operation(lhs: Any = lhs, rhs: Any = rhs, output: Any = output) -> Any:
                return torch.mm(lhs, rhs, out=output)

            samples_ms = _cuda_samples_ms(
                torch,
                operation,
                warmup=warmup,
                repeats=repeats,
                inner_iterations=inner,
            )
            tflops_samples = [
                flops / (sample_ms * 1e-3) / 1e12 for sample_ms in samples_ms
            ]
            results.append(
                {
                    "label": label,
                    "m": m,
                    "n": n,
                    "k": k,
                    "flops_per_op": flops,
                    "warmup": warmup,
                    "repeats": repeats,
                    "inner_iterations": inner,
                    "samples_ms_per_op": samples_ms,
                    "achieved_tflops_samples": tflops_samples,
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-shape CUDA/OOM failures in the raw artifact
            errors.append(
                {
                    "stage": "matmul",
                    "label": label,
                    "m": m,
                    "n": n,
                    "k": k,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            lhs = rhs = output = None
            torch.cuda.empty_cache()
    return results, errors


def _benchmark_copies(
    torch: Any, cfg: dict[str, Any], device: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warmup = _positive_int(cfg.get("warmup", 10), "warmup")
    repeats = _positive_int(cfg.get("repeats", 30), "repeats")
    inner = _positive_int(cfg.get("copy_inner_iterations", 10), "copy_inner_iterations")
    sizes = cfg.get("copy_sizes_bytes")
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("copy_sizes_bytes must be a non-empty array")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for raw_size in sizes:
        size = _positive_int(raw_size, "copy size")
        for direction in ("d2d", "h2d", "d2h"):
            source = destination = operation = None
            try:
                if direction == "d2d":
                    source = torch.empty(size, dtype=torch.uint8, device=device)
                    destination = torch.empty(size, dtype=torch.uint8, device=device)
                    traffic_multiplier = 2
                    operation = lambda source=source, destination=destination: (
                        destination.copy_(source, non_blocking=True)
                    )
                elif direction == "h2d":
                    source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                    destination = torch.empty(size, dtype=torch.uint8, device=device)
                    traffic_multiplier = 1
                    operation = lambda source=source, destination=destination: (
                        destination.copy_(source, non_blocking=True)
                    )
                else:
                    source = torch.empty(size, dtype=torch.uint8, device=device)
                    destination = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                    traffic_multiplier = 1
                    operation = lambda source=source, destination=destination: (
                        destination.copy_(source, non_blocking=True)
                    )

                samples_ms = _cuda_samples_ms(
                    torch,
                    operation,
                    warmup=warmup,
                    repeats=repeats,
                    inner_iterations=inner,
                )
                logical_gbs = [
                    size / (sample_ms * 1e-3) / 1e9 for sample_ms in samples_ms
                ]
                aggregate_gbs = [
                    size * traffic_multiplier / (sample_ms * 1e-3) / 1e9
                    for sample_ms in samples_ms
                ]
                results.append(
                    {
                        "direction": direction,
                        "size_bytes": size,
                        "warmup": warmup,
                        "repeats": repeats,
                        "inner_iterations": inner,
                        "samples_ms_per_copy": samples_ms,
                        "logical_bandwidth_GB_s_samples": logical_gbs,
                        "aggregate_traffic_multiplier": traffic_multiplier,
                        "aggregate_bandwidth_GB_s_samples": aggregate_gbs,
                        "measurement_semantics": (
                            "aggregate read+write HBM traffic"
                            if direction == "d2d"
                            else "one-way pinned-host torch copy payload"
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-copy CUDA/pinned-memory failures
                errors.append(
                    {
                        "stage": "copy",
                        "direction": direction,
                        "size_bytes": size,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            finally:
                source = destination = operation = None
                torch.cuda.empty_cache()
    return results, errors


def main() -> int:
    args = _parser().parse_args()
    cfg = load_json_object(args.config)
    if cfg.get("schema") != "dops.gpu_microbench.config.v1":
        raise ValueError(
            "benchmark config schema must be dops.gpu_microbench.config.v1"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is false; run this tool only inside a Slurm GPU allocation"
        )
    device = str(cfg.get("device", "cuda:0"))
    torch.cuda.set_device(device)
    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    device_name = str(properties.name)
    if args.expect_device_regex and not re.search(
        args.expect_device_regex, device_name
    ):
        raise RuntimeError(
            f"detected CUDA device {device_name!r} does not match --expect-device-regex "
            f"{args.expect_device_regex!r}"
        )

    dtype_name = str(cfg.get("dtype", "float16"))
    dtype = _dtype_from_name(torch, dtype_name)
    allow_tf32 = bool(cfg.get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    seed = int(cfg.get("seed", 0) or 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    repo_root = Path(__file__).resolve().parents[2]
    with torch.inference_mode():
        matmul, matmul_errors = _benchmark_matmuls(torch, cfg, dtype, device)
        copies, copy_errors = _benchmark_copies(torch, cfg, device)
    errors = matmul_errors + copy_errors
    if not matmul:
        raise RuntimeError(f"all matmul benchmarks failed: {matmul_errors}")
    if not any(item["direction"] == "d2d" for item in copies):
        raise RuntimeError(f"all d2d copy benchmarks failed: {copy_errors}")

    capability = torch.cuda.get_device_capability(device_index)
    raw = {
        "schema": RAW_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not errors else "partial",
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "executable": sys.executable,
            "slurm": _slurm_snapshot(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": _nvidia_smi_snapshot(),
        },
        "software": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cudnn_version": torch.backends.cudnn.version(),
            "git_revision": _git_revision(repo_root),
        },
        "device": {
            "requested": device,
            "torch_index": device_index,
            "name": device_name,
            "compute_capability": [int(capability[0]), int(capability[1])],
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
        },
        "benchmark": {
            "config_path": str(args.config.expanduser().resolve()),
            "dtype": dtype_name,
            "element_size_bytes": int(torch.empty((), dtype=dtype).element_size()),
            "seed": seed,
            "allow_tf32": allow_tf32,
            "warmup": int(cfg.get("warmup", 10)),
            "repeats": int(cfg.get("repeats", 30)),
        },
        "matmul": matmul,
        "copy": copies,
        "errors": errors,
    }
    output = atomic_write_json(args.output, raw)
    print(
        f"[GPU-MICROBENCH] wrote {output} device={device_name!r} "
        f"matmul_shapes={len(matmul)} copy_cases={len(copies)} errors={len(errors)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
