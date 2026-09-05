#!/usr/bin/env python3
"""Export the fixed Qwen-1.8B Het-Infer full-request sidecar suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DOPS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DOPS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hetinfer_tensor_bindings_export import (  # noqa: E402
    export_tensor_bindings_manifest_from_artifacts,
)


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
        "hetinfer_tensor_bindings_out": str(
            sidecar_root / "tensor_bindings.json"
        ),
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


def _network_is_complete(
    network_path: Path, batch: int, prefix_length: int
) -> bool:
    if not network_path.is_file():
        return False
    manifest = json.loads(network_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "dops.hetinfer_network.v1"
        or manifest.get("schema_version") != 1
    ):
        return False
    networks = manifest.get("networks", [])
    if len(networks) != 1 + DECODE_TOKENS:
        return False
    graph_id = networks[0]["graph_id"]
    workload_id = networks[0]["workload_id"]
    required_workload = {
        "batch",
        "sequence_length",
        "past_kv_len",
        "query_len",
        "scheduled_tokens",
        "mean_context",
    }
    for network_index, network in enumerate(networks):
        if (
            network["graph_id"] != graph_id
            or network["workload_id"] != workload_id
            or set(network["workload"]) != required_workload
        ):
            return False
        workload = network["workload"]
        if workload["batch"] != batch:
            return False
        if network_index == 0:
            if (
                network["phase"] != "prefill"
                or workload["sequence_length"] != prefix_length
                or workload["past_kv_len"] != 0
                or workload["query_len"] != prefix_length
                or workload["scheduled_tokens"] != batch * prefix_length
            ):
                return False
        elif (
            network["phase"] != "decode"
            or workload["past_kv_len"] != workload["sequence_length"]
            or workload["query_len"] != 1
            or workload["scheduled_tokens"] != batch
        ):
            return False
        for operator_index, operator in enumerate(network["operators"]):
            if (
                operator.get("operator_index") != operator_index
                or "layer_index" not in operator
                or not operator.get("canonical_op_slot")
            ):
                return False
    return True


def _is_complete(
    network_path: Path,
    tensor_bindings_path: Path,
    batch: int,
    prefix_length: int,
) -> bool:
    if not _network_is_complete(network_path, batch, prefix_length):
        return False
    if not tensor_bindings_path.is_file():
        return False
    manifest = json.loads(network_path.read_text(encoding="utf-8"))
    networks = manifest["networks"]
    tensor_bindings = json.loads(
        tensor_bindings_path.read_text(encoding="utf-8")
    )
    if tensor_bindings.get("schema") != "dops.hetinfer_tensor_bindings.v1":
        return False
    if tensor_bindings.get("schema_version") != 1:
        return False
    if tensor_bindings.get("graph_id") != networks[0]["graph_id"]:
        return False
    if tensor_bindings.get("workload_id") != networks[0]["workload_id"]:
        return False
    bindings = tensor_bindings.get("bindings", [])
    if {item["network_index"] for item in bindings} != set(range(len(networks))):
        return False
    return True


def _backfill_tensor_bindings(
    *, prior_path: Path, network_path: Path, output: Path
) -> None:
    prior_artifact = json.loads(prior_path.read_text(encoding="utf-8"))
    network_manifest = json.loads(network_path.read_text(encoding="utf-8"))
    export_tensor_bindings_manifest_from_artifacts(
        prior_artifact=prior_artifact,
        network_manifest=network_manifest,
        output=output,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--het-infer-root", required=True, type=Path)
    parser.add_argument("--batch", action="append", type=int, choices=BATCHES)
    parser.add_argument(
        "--prefix-length", action="append", type=int, choices=PREFIX_LENGTHS
    )
    args = parser.parse_args()

    dops_root = DOPS_ROOT
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
    env["LD_LIBRARY_PATH"] = str(simulator_root)
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
            tensor_bindings_path = sidecar_root / "tensor_bindings.json"
            if prior_path.is_file() and _is_complete(
                network_path, tensor_bindings_path, batch, prefix_length
            ):
                print(f"[skip] {workload}", flush=True)
                continue
            if (
                prior_path.is_file()
                and not tensor_bindings_path.exists()
                and _network_is_complete(network_path, batch, prefix_length)
            ):
                _backfill_tensor_bindings(
                    prior_path=prior_path,
                    network_path=network_path,
                    output=tensor_bindings_path,
                )
                if not _is_complete(
                    network_path, tensor_bindings_path, batch, prefix_length
                ):
                    raise RuntimeError(
                        f"incomplete tensor binding backfill for {workload}"
                    )
                print(f"[backfill] {workload}", flush=True)
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
                network_path, tensor_bindings_path, batch, prefix_length
            ):
                raise RuntimeError(f"incomplete sidecar export for {workload}")
            print(f"[done] {workload}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
