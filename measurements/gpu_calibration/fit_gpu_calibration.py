"""Fit a conservative DOPS roofline/runtime model from GPU microbench JSON."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calibration_common import (
    FIT_SCHEMA,
    RAW_SCHEMA,
    RUNTIME_MODEL_SCHEMA,
    atomic_write_json,
    finite_positive,
    load_json_object,
    ordinary_least_squares,
    quantile,
    sha256_file,
    summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-name-prefix", default="GPU0")
    parser.add_argument(
        "--large-matmul-flops",
        type=float,
        default=1.0e9,
        help="Only shapes at or above this FLOP count estimate peak throughput",
    )
    return parser


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def _fit_matmul(raw: dict[str, Any], *, large_flops: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in raw.get("matmul", []):
        if not isinstance(item, dict):
            continue
        times_ms = finite_positive(item.get("samples_ms_per_op", []))
        flops = int(item.get("flops_per_op", 0) or 0)
        if not times_ms or flops <= 0:
            continue
        time_summary = summary(times_ms)
        median_s = float(time_summary["median"]) * 1e-3
        throughput_samples = [flops / (time_ms * 1e-3) / 1e12 for time_ms in times_ms]
        rows.append(
            {
                "label": str(item.get("label", "")),
                "m": int(item.get("m", 0) or 0),
                "n": int(item.get("n", 0) or 0),
                "k": int(item.get("k", 0) or 0),
                "flops": flops,
                "median_s": median_s,
                "latency_ms": time_summary,
                "achieved_tflops": summary(throughput_samples),
            }
        )
    if len(rows) < 2:
        raise ValueError(
            "at least two successful matmul shapes are required for fitting"
        )

    ols = ordinary_least_squares((float(row["flops"]), row["median_s"]) for row in rows)
    smallest_median_s = min(float(row["median_s"]) for row in rows)
    launch_overhead_s = _clamp(float(ols["intercept"]), 0.0, smallest_median_s * 0.95)

    for row in rows:
        adjusted_s = max(float(row["median_s"]) - launch_overhead_s, 1e-9)
        row["overhead_adjusted_tflops"] = float(row["flops"]) / adjusted_s / 1e12

    large_rows = [row for row in rows if float(row["flops"]) >= float(large_flops)]
    if not large_rows:
        large_rows = sorted(rows, key=lambda row: int(row["flops"]))[
            -max(1, len(rows) // 3) :
        ]
    peak_tflops = quantile(
        [row["overhead_adjusted_tflops"] for row in large_rows],
        0.90,
    )
    if peak_tflops <= 0.0:
        raise ValueError("fitted peak TFLOPS is not positive")

    for row in rows:
        row["utilization_of_fitted_peak"] = _clamp(
            float(row["overhead_adjusted_tflops"]) / peak_tflops,
            1e-4,
            1.0,
        )
    rows_by_flops = sorted(rows, key=lambda row: int(row["flops"]))
    split = max(1, len(rows_by_flops) // 3)
    low_rows = rows_by_flops[:split]
    high_rows = rows_by_flops[-split:]
    min_util = _clamp(
        quantile([row["utilization_of_fitted_peak"] for row in low_rows], 0.25),
        0.01,
        1.0,
    )
    max_util = _clamp(
        quantile([row["utilization_of_fitted_peak"] for row in high_rows], 0.75),
        min_util,
        1.0,
    )
    flops_values = [float(row["flops"]) for row in rows]
    flops_low = quantile(flops_values, 0.10)
    flops_high = quantile(flops_values, 0.90)
    if flops_high <= flops_low:
        flops_high = max(flops_low + 1.0, max(flops_values))

    runtime_curve = {
        "enabled": True,
        "curve": "log_linear",
        "min_util": min_util,
        "max_util": max_util,
        "flops_low": flops_low,
        "flops_high": flops_high,
        "power": 1.0,
        "fit_scope": "torch.mm only",
    }
    kernel_overhead_us = launch_overhead_s * 1e6
    kernel_model = {
        "enabled": kernel_overhead_us > 0.0,
        "apply_backends": ["fast"],
        "phase_scale": {"prefill": 1.0, "decode": 1.0},
        "scale_by_time_scale": False,
        "default_us": 0.0,
        "by_category_us": {
            "norm": 0.0,
            "softmax": 0.0,
            "activation": 0.0,
            "elem": 0.0,
            "gemm": kernel_overhead_us,
        },
        "by_op_us": {},
        "fit_scope": "single shared torch.mm intercept; non-GEMM overhead is unmeasured",
    }
    return {
        "shape_count": len(rows),
        "large_shape_threshold_flops": float(large_flops),
        "large_shape_count": len(large_rows),
        "latency_fit_s_equals_intercept_plus_slope_times_flops": ols,
        "launch_overhead_s_clamped": launch_overhead_s,
        "peak_tflops_overhead_adjusted_p90": peak_tflops,
        "large_shape_raw_tflops": summary(
            [float(row["achieved_tflops"]["median"]) for row in large_rows]
        ),
        "runtime_compute_utilization": runtime_curve,
        "runtime_kernel_launch_overhead": kernel_model,
        "per_shape": rows,
    }


def _fit_copy_direction(
    items: list[dict[str, Any]], direction: str
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    traffic_multiplier = 2 if direction == "d2d" else 1
    for item in items:
        if not isinstance(item, dict) or item.get("direction") != direction:
            continue
        size = int(item.get("size_bytes", 0) or 0)
        times_ms = finite_positive(item.get("samples_ms_per_copy", []))
        if size <= 0 or not times_ms:
            continue
        time_stats = summary(times_ms)
        median_s = float(time_stats["median"]) * 1e-3
        bytes_modeled = size * traffic_multiplier
        bytes_per_second = bytes_modeled / median_s
        rows.append(
            {
                "size_bytes": size,
                "modeled_traffic_bytes": bytes_modeled,
                "traffic_multiplier": traffic_multiplier,
                "latency_ms": time_stats,
                "bandwidth_GB_s_decimal": bytes_per_second / 1e9,
                "bandwidth_GiB_s_binary": bytes_per_second / (1024**3),
            }
        )
    if not rows:
        return None
    bandwidths_gib = [row["bandwidth_GiB_s_binary"] for row in rows]
    fit = None
    if len(rows) >= 2:
        fit = ordinary_least_squares(
            (
                float(row["modeled_traffic_bytes"]),
                float(row["latency_ms"]["median"]) * 1e-3,
            )
            for row in rows
        )
        slope = float(fit["slope"])
        fit["bandwidth_GiB_s_binary_from_slope"] = (
            1.0 / slope / (1024**3) if slope > 0.0 else None
        )
        fit["overhead_s_clamped"] = _clamp(
            float(fit["intercept"]),
            0.0,
            min(float(row["latency_ms"]["median"]) * 1e-3 for row in rows) * 0.95,
        )
    conservative = quantile(bandwidths_gib, 0.25)
    if fit and fit.get("bandwidth_GiB_s_binary_from_slope"):
        conservative = min(
            conservative,
            float(fit["bandwidth_GiB_s_binary_from_slope"]),
        )
    return {
        "direction": direction,
        "measurement_semantics": (
            "aggregate read+write HBM traffic"
            if direction == "d2d"
            else "one-way pinned-host torch copy payload"
        ),
        "bandwidth_GiB_s_binary": summary(bandwidths_gib),
        "latency_fit_s_equals_intercept_plus_slope_times_bytes": fit,
        "recommended_conservative_GiB_s": conservative,
        "per_size": rows,
    }


def _fit_copies(raw: dict[str, Any]) -> dict[str, Any]:
    items = [item for item in raw.get("copy", []) if isinstance(item, dict)]
    fitted = {
        direction: _fit_copy_direction(items, direction)
        for direction in ("d2d", "h2d", "d2h")
    }
    if fitted["d2d"] is None:
        raise ValueError("at least one successful d2d copy measurement is required")
    host_directions = [
        float(fitted[direction]["recommended_conservative_GiB_s"])
        for direction in ("h2d", "d2h")
        if fitted[direction] is not None
    ]
    host_link = min(host_directions) if host_directions else None
    host_overheads = [
        float(
            fitted[direction]["latency_fit_s_equals_intercept_plus_slope_times_bytes"][
                "overhead_s_clamped"
            ]
        )
        for direction in ("h2d", "d2h")
        if fitted[direction] is not None
        and fitted[direction]["latency_fit_s_equals_intercept_plus_slope_times_bytes"]
        is not None
    ]
    return {
        "directions": fitted,
        "recommended_hbm_mem_bw_GiB_s": float(
            fitted["d2d"]["recommended_conservative_GiB_s"]
        ),
        "recommended_host_gpu_link_GiB_s": host_link,
        "recommended_host_gpu_overhead_s": max(host_overheads)
        if host_overheads
        else None,
        "unit_note": (
            "DOPS multiplies mem_bw_GBs and link bw_GBs by 1024^3; export therefore uses GiB/s "
            "even though the legacy field name says GB/s"
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    raw = load_json_object(args.input)
    if raw.get("schema") != RAW_SCHEMA:
        raise ValueError(f"input schema must be {RAW_SCHEMA}")
    if raw.get("status") not in {"complete", "partial"}:
        raise ValueError("raw benchmark status must be complete or partial")
    device = raw.get("device")
    if not isinstance(device, dict) or not str(device.get("name", "")).strip():
        raise ValueError("raw benchmark is missing the detected CUDA device name")

    matmul = _fit_matmul(raw, large_flops=args.large_matmul_flops)
    copies = _fit_copies(raw)
    device_prefix = str(args.device_name_prefix).strip()
    if not device_prefix:
        raise ValueError("--device-name-prefix must not be empty")
    runtime_model = {
        "schema": RUNTIME_MODEL_SCHEMA,
        "device_name_prefix": device_prefix,
        "source_scope": "torch.mm plus torch Tensor.copy_ microbenchmarks",
        "compute_utilization": matmul["runtime_compute_utilization"],
        "kernel_launch_overhead": matmul["runtime_kernel_launch_overhead"],
    }
    fit = {
        "schema": FIT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "raw_path": str(args.input.expanduser().resolve()),
            "raw_sha256": sha256_file(args.input),
            "raw_status": raw.get("status"),
            "machine": raw.get("machine"),
            "software": raw.get("software"),
            "device": device,
            "benchmark": raw.get("benchmark"),
            "raw_errors": raw.get("errors", []),
        },
        "matmul_fit": matmul,
        "copy_fit": copies,
        "recommendations": {
            "dops_hardware": {
                "tflops": matmul["peak_tflops_overhead_adjusted_p90"],
                "mem_bw_GBs": copies["recommended_hbm_mem_bw_GiB_s"],
                "mem_capacity_GB": int(device.get("total_memory_bytes", 0) or 0)
                / float(1024**3),
            },
            "host_gpu_link": {
                "bw_GBs": copies["recommended_host_gpu_link_GiB_s"],
                "overhead_s": copies["recommended_host_gpu_overhead_s"],
            },
            "runtime_model": runtime_model,
        },
        "limitations": [
            "Only torch.mm GEMM and torch Tensor.copy_ paths are calibrated.",
            "Softmax, normalization, activation, collectives, KV-cache kernels, and end-to-end inference are not measured.",
            "The utilization curve depends only on FLOP count and cannot represent every matrix aspect ratio.",
            "The fitted values describe the recorded GPU/software/node state, not an assumed GPU family.",
        ],
    }
    output = atomic_write_json(args.output, fit)
    print(
        f"[GPU-FIT] wrote {output} device={device['name']!r} "
        f"tflops={fit['recommendations']['dops_hardware']['tflops']:.6g} "
        f"hbm_GiB_s={fit['recommendations']['dops_hardware']['mem_bw_GBs']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
