"""Convert a measured GPU fit into DOPS hardware, run config, and runtime model."""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calibration_common import (
    FIT_SCHEMA,
    RUNTIME_MODEL_SCHEMA,
    atomic_write_json,
    load_json_object,
    sha256_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--base-hardware", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--hardware-out", required=True, type=Path)
    parser.add_argument("--config-out", required=True, type=Path)
    parser.add_argument("--runtime-model-out", required=True, type=Path)
    parser.add_argument("--gpu-device-name", default="GPU0")
    parser.add_argument(
        "--expect-device-regex",
        help="Refuse export unless the device recorded in the fit matches",
    )
    return parser


def _positive_finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive, got {value!r}")
    return number


def _relative_path(target: Path, owner: Path) -> str:
    return os.path.relpath(
        target.expanduser().resolve(), owner.expanduser().resolve().parent
    )


def _rebase_existing_config_paths(
    config: dict[str, Any],
    *,
    base_config_path: Path,
    output_config_path: Path,
) -> None:
    path_keys = (
        "shape_file",
        "pim_config_path",
        "ramulator_config_path",
        "result_dir",
        "simulation_log_file",
        "dump_graph_dir",
        "hetinfer_prior_out",
    )
    base_dir = base_config_path.expanduser().resolve().parent
    output_dir = output_config_path.expanduser().resolve().parent
    for key in path_keys:
        raw = config.get(key)
        if raw in (None, ""):
            continue
        source = Path(str(raw)).expanduser()
        if not source.is_absolute():
            source = (base_dir / source).resolve()
        config[key] = os.path.relpath(source, output_dir)


def _unwrap_hardware(value: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(value.get("hardware"), dict):
        return "hardware", copy.deepcopy(value["hardware"])
    if isinstance(value.get("cluster"), dict):
        return "cluster", copy.deepcopy(value["cluster"])
    if isinstance(value.get("devices"), list):
        return "hardware", copy.deepcopy(value)
    raise ValueError("base hardware must contain hardware.devices or cluster.devices")


def _find_link_bandwidth(
    links: list[dict[str, Any]],
    left: str,
    right: str,
) -> float | None:
    endpoints = {left, right}
    for link in links:
        if not isinstance(link, dict):
            continue
        if {str(link.get("a", "")), str(link.get("b", ""))} != endpoints:
            continue
        try:
            bandwidth = float(link.get("bw_GBs"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(bandwidth) and bandwidth > 0.0:
            return bandwidth
    return None


def _build_hardware(
    base: dict[str, Any],
    fit: dict[str, Any],
    *,
    gpu_device_name: str,
    fit_path: Path,
) -> dict[str, Any]:
    wrapper, hardware = _unwrap_hardware(base)
    devices = hardware.get("devices")
    if not isinstance(devices, list):
        raise TypeError("base hardware devices must be an array")
    detected = fit["source"]["device"]
    recommendation = fit["recommendations"]["dops_hardware"]
    tflops = _positive_finite(recommendation.get("tflops"), "recommended tflops")
    mem_bw = _positive_finite(
        recommendation.get("mem_bw_GBs"), "recommended mem_bw_GBs"
    )
    capacity = _positive_finite(
        recommendation.get("mem_capacity_GB"),
        "recommended mem_capacity_GB",
    )
    matches = [device for device in devices if device.get("name") == gpu_device_name]
    if len(matches) != 1:
        raise ValueError(
            f"base hardware must contain exactly one device named {gpu_device_name!r}, "
            f"found {len(matches)}"
        )
    gpu = matches[0]
    if str(gpu.get("type", "")).lower() != "npu":
        raise ValueError(
            f"DOPS accelerator {gpu_device_name!r} must remain type='npu', got {gpu.get('type')!r}"
        )
    gpu["tflops"] = tflops
    gpu["mem_bw_GBs"] = mem_bw
    gpu["mem_capacity_GB"] = capacity
    gpu["arch"] = "measured-torch-cuda:" + str(detected["name"])
    for key in (
        "llmcompass_kind",
        "llmcompass_device",
        "llmcompass_device_name",
        "llmcompass_arch",
    ):
        gpu.pop(key, None)
    gpu["calibration"] = {
        "kind": "torch_cuda_microbench_roofline_fit",
        "detected_device_name": detected["name"],
        "fit_sha256": sha256_file(fit_path),
        "fit_created_at_utc": fit.get("created_at_utc"),
        "tflops_semantics": "overhead-adjusted p90 of large measured torch.mm shapes",
        "mem_bw_semantics": "conservative aggregate read+write D2D torch copy GiB/s",
    }

    cpu_names = [
        str(device.get("name"))
        for device in devices
        if str(device.get("type", "")).lower() == "cpu"
    ]
    pim_names = [
        str(device.get("name"))
        for device in devices
        if str(device.get("type", "")).lower() == "pim"
    ]
    if not cpu_names:
        raise ValueError(
            "base hardware needs a CPU host for the measured host-GPU link"
        )
    host = cpu_names[0]
    old_links = [link for link in hardware.get("links", []) if isinstance(link, dict)]
    fallback_link_bw = _positive_finite(
        hardware.get("fc_bw_GBs", 32.0),
        "base PIM-link fallback bandwidth",
    )
    measured_link = fit["recommendations"].get("host_gpu_link", {})
    host_gpu_bw = _positive_finite(
        measured_link.get("bw_GBs"),
        "measured host-GPU link bandwidth",
    )
    host_gpu_overhead = max(0.0, float(measured_link.get("overhead_s") or 0.0))
    links: list[dict[str, Any]] = [
        {
            "a": host,
            "b": gpu_device_name,
            "bw_GBs": host_gpu_bw,
            "latency_s": 0.0,
            "overhead_s": host_gpu_overhead,
            "measurement_scope": "conservative minimum of pinned H2D and D2H torch copy fits",
        }
    ]
    for pim_name in pim_names:
        pim_bw = _find_link_bandwidth(old_links, host, pim_name) or fallback_link_bw
        links.append(
            {
                "a": host,
                "b": pim_name,
                "bw_GBs": pim_bw,
                "latency_s": 0.0,
                "overhead_s": 0.0,
                "measurement_scope": "inherited unmeasured PIM link from base hardware",
            }
        )
    hardware["topology"] = "star"
    hardware.pop("fc_bw_GBs", None)
    hardware["links"] = links
    hardware["modeling_scope"] = "offline_cost_model_with_measured_gpu_roofline"
    hardware["gpu_cost_model"] = (
        f"measured on {detected['name']} with torch CUDA microbench; "
        "only GEMM, HBM copy, and pinned host copy are calibrated"
    )
    hardware["calibration_provenance"] = {
        "fit_sha256": sha256_file(fit_path),
        "raw_sha256": fit["source"].get("raw_sha256"),
        "machine": fit["source"].get("machine"),
        "software": fit["source"].get("software"),
        "device": detected,
    }
    return {wrapper: hardware}


def _build_runtime_model(fit: dict[str, Any], *, fit_path: Path) -> dict[str, Any]:
    runtime = copy.deepcopy(fit["recommendations"].get("runtime_model"))
    if not isinstance(runtime, dict) or runtime.get("schema") != RUNTIME_MODEL_SCHEMA:
        raise ValueError(
            f"fit is missing recommendations.runtime_model schema={RUNTIME_MODEL_SCHEMA}"
        )
    runtime["source_fit_sha256"] = sha256_file(fit_path)
    runtime["source_raw_sha256"] = fit["source"].get("raw_sha256")
    runtime["detected_device"] = fit["source"].get("device")
    runtime["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return runtime


def _build_config(
    base: dict[str, Any],
    fit: dict[str, Any],
    *,
    fit_path: Path,
    hardware_out: Path,
    runtime_model_out: Path,
    config_out: Path,
    base_config_path: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    _rebase_existing_config_paths(
        config,
        base_config_path=base_config_path,
        output_config_path=config_out,
    )
    detected = fit["source"]["device"]
    config["npu_backend"] = "fast"
    config["hardware_json"] = _relative_path(hardware_out, config_out)
    config["gpu_runtime_model_json"] = _relative_path(runtime_model_out, config_out)
    config["experiment_label"] = "hetinfer_gpu0_pim0_static_prior_gpu_calibrated_v1"
    pim_kind = (
        "analytical fast-mode"
        if bool(config.get("pim_fast_mode", False))
        else "CENT/AiM-to-Ramulator2 trace-mode"
    )
    config["evidence_class"] = (
        f"DOPS offline simulation; GPU0 roofline calibrated by torch CUDA on {detected['name']}; "
        f"PIM0 remains {pim_kind}"
    )
    config["gpu_calibration"] = {
        "fit_sha256": sha256_file(fit_path),
        "raw_sha256": fit["source"].get("raw_sha256"),
        "detected_device": detected,
        "scope": "torch.mm, D2D copy, pinned H2D/D2H copy",
        "limitations": fit.get("limitations", []),
    }
    return config


def main() -> int:
    args = _parser().parse_args()
    fit = load_json_object(args.fit)
    if fit.get("schema") != FIT_SCHEMA:
        raise ValueError(f"fit schema must be {FIT_SCHEMA}")
    device = fit.get("source", {}).get("device")
    if not isinstance(device, dict) or not str(device.get("name", "")).strip():
        raise ValueError("fit has no recorded CUDA device")
    if args.expect_device_regex and not re.search(
        args.expect_device_regex,
        str(device["name"]),
    ):
        raise ValueError(
            f"fit device {device['name']!r} does not match --expect-device-regex "
            f"{args.expect_device_regex!r}"
        )

    base_hardware = load_json_object(args.base_hardware)
    base_config = load_json_object(args.base_config)
    hardware = _build_hardware(
        base_hardware,
        fit,
        gpu_device_name=args.gpu_device_name,
        fit_path=args.fit,
    )
    runtime_model = _build_runtime_model(fit, fit_path=args.fit)
    config = _build_config(
        base_config,
        fit,
        fit_path=args.fit,
        hardware_out=args.hardware_out,
        runtime_model_out=args.runtime_model_out,
        config_out=args.config_out,
        base_config_path=args.base_config,
    )
    runtime_path = atomic_write_json(args.runtime_model_out, runtime_model)
    hardware_path = atomic_write_json(args.hardware_out, hardware)
    config_path = atomic_write_json(args.config_out, config)
    print(
        "[GPU-EXPORT] wrote "
        f"hardware={hardware_path} config={config_path} runtime_model={runtime_path} "
        f"device={device['name']!r}; npu_backend=fast"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
