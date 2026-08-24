#!/usr/bin/env python3
"""Export the fixed Qwen-1.8B Het-Infer full-request sidecar suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


BATCHES = (1, 2, 4, 8)
PREFIX_LENGTHS = (128, 512, 2048, 8192)
DECODE_TOKENS = 128


def _config(
    *,
    dops_root: Path,
    het_infer_root: Path,
    batch: int,
    prefix_length: int,
    workload: str,
) -> dict[str, object]:
    sidecar_root = (
        het_infer_root
        / "artifacts"
        / "full_qwen_1p8b"
        / "sidecars"
        / workload
    )
    result_root = dops_root / "output" / "full_qwen_1p8b" / workload
    return {
        "experiment_label": f"hetinfer_full_qwen_1p8b_{workload}",
        "evidence_class": "DOPS Ascend LUT plus CENT/AiM trace and Ramulator2",
        "model_family": "qwen",
        "model_variant": "1.8b",
        "model_revision": "shape-only:qwen_1.8b_shape.json;layer_num=28",
        "hetinfer_graph_id": "qwen-1.8b-28layer-1npu2pim",
        "hetinfer_workload_id": workload,
        "shape_file": str(dops_root / "configs" / "qwen_1.8b_shape.json"),
        "dtype": "fp16",
        "batch": batch,
        "max_batch_size": batch,
        "prefill_len": prefix_length,
        "decode_len": DECODE_TOKENS,
        "max_seq_len": prefix_length + DECODE_TOKENS,
        "decode_sample_stride": 1,
        "decode_plan_refresh_stride": 0,
        "pim_config_path": str(dops_root / "src" / "aim_simulator" / "PIM_AiM.json"),
        "ramulator_config_path": str(dops_root / "src" / "aim_simulator" / "example.yaml"),
        "pim_ramulator_timeout_s": 300,
        "pim_trace_strict": True,
        "pim_trace_keep_traces": False,
        "result_dir": str(result_root),
        "simulation_log_file": str(result_root / "pim_simulation.log"),
        "hardware_json": str(dops_root / "src" / "examples" / "hardware_1npu_2aim.json"),
        "hetinfer_prior_out": str(sidecar_root / "prior.json"),
        "hetinfer_network_out": str(sidecar_root / "network.json"),
        "algo": ["Bifocal"],
        "baselines": [],
        "tp_qkv": 1,
        "tp_ffn": 1,
        "tp_moe": 1,
        "pp": 1,
        "ep": 1,
        "npu_backend": "lut",
        "npu_lut_strict": True,
        "pim_fast_mode": False,
        "scheduler_seed": 0,
        "dump_graph": False,
        "pim_weight_load_overlap_ratio": 0.0,
        "weight_load_compute_overlap_ratio": 0.0,
    }


def _is_complete(network_path: Path, batch: int, prefix_length: int) -> bool:
    if not network_path.is_file():
        return False
    manifest = json.loads(network_path.read_text(encoding="utf-8"))
    networks = manifest.get("networks", [])
    if len(networks) != 1 + DECODE_TOKENS:
        return False
    prefill = networks[0]["workload"]
    return (
        prefill["batch"] == batch
        and prefill["sequence_length"] == prefix_length
        and prefill["scheduled_tokens"] == batch * prefix_length
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--het-infer-root", required=True, type=Path)
    parser.add_argument("--batch", action="append", type=int, choices=BATCHES)
    parser.add_argument(
        "--prefix-length", action="append", type=int, choices=PREFIX_LENGTHS
    )
    args = parser.parse_args()

    dops_root = Path(__file__).resolve().parents[1]
    het_infer_root = args.het_infer_root.expanduser().resolve()
    output_root = dops_root / "output" / "full_qwen_1p8b"
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    simulator_root = (
        het_infer_root
        / "vendor"
        / "dops"
        / "submodules"
        / "CENT"
        / "aim_simulator"
    )
    env = os.environ.copy()
    env["RAMULATOR2_BIN"] = str(simulator_root / "build" / "ramulator2")
    env["DYLD_LIBRARY_PATH"] = str(simulator_root)
    env["PIM_TRACE_SCALE_REPEATS"] = "1"

    selected_batches = tuple(args.batch or BATCHES)
    selected_prefix_lengths = tuple(args.prefix_length or PREFIX_LENGTHS)
    for batch in selected_batches:
        run_env = env.copy()
        run_env["PIM_LATENCY_CACHE_FILE"] = str(
            output_root / f"pim_latency_cache_b{batch}.pkl"
        )
        for prefix_length in selected_prefix_lengths:
            workload = f"b{batch}-p{prefix_length}-h128"
            sidecar_root = (
                het_infer_root
                / "artifacts"
                / "full_qwen_1p8b"
                / "sidecars"
                / workload
            )
            prior_path = sidecar_root / "prior.json"
            network_path = sidecar_root / "network.json"
            if prior_path.is_file() and _is_complete(
                network_path, batch, prefix_length
            ):
                print(f"[skip] {workload}", flush=True)
                continue

            sidecar_root.mkdir(parents=True, exist_ok=True)

            config = _config(
                dops_root=dops_root,
                het_infer_root=het_infer_root,
                batch=batch,
                prefix_length=prefix_length,
                workload=workload,
            )
            config_path = config_root / f"{workload}.json"
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            log_path = output_root / workload / "export.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[start] {workload} log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(dops_root / "src" / "main.py"),
                        "evaluate",
                        "--config",
                        str(config_path),
                    ],
                    cwd=dops_root,
                    env=run_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not prior_path.is_file() or not _is_complete(
                network_path, batch, prefix_length
            ):
                raise RuntimeError(f"incomplete sidecar export for {workload}")
            print(f"[done] {workload}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
