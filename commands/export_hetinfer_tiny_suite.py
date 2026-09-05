#!/usr/bin/env python3
"""Generate native DOPS inputs for the agreed seven-workload experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "commands"))
from export_hetinfer_full_suite import _config

WORKLOADS = ((1, 16), (2, 16), (4, 16), (8, 16),
             (1, 64), (1, 256), (1, 1024))
HORIZON = 32


def experiment_config(model: str, batch: int, prefill: int) -> dict:
    workload = f"b{batch}-p{prefill}-h{HORIZON}"
    out = ROOT / "output" / "tiny_moe_experiment" / model / workload
    cfg = _config(dops_root=ROOT, het_infer_root=ROOT.parent / "v1-cama-work",
                  batch=batch, prefix_length=prefill, workload=workload)
    cfg.update({
        "experiment_label": f"tiny_moe_experiment_{model}_{workload}",
        "decode_len": HORIZON,
        "max_seq_len": prefill + HORIZON,
        "result_dir": str(out / "native_run"),
        "simulation_log_file": str(out / "pim_simulation.log"),
        "hetinfer_prior_out": str(out / "native" / "prior.json"),
        "hetinfer_network_out": str(out / "native" / "network.json"),
        "hetinfer_tensor_bindings_out": str(out / "native" / "tensor_bindings.json"),
        "pim_trace_strict": True,
        "pim_ramulator_timeout_s": 1800,
    })
    if model == "mixtral":
        cfg.update({
            "model_family": "mixtral", "model_variant": "8x7b",
            "model_revision": "shape-only:tiny_mixtral_experiment.json",
            "shape_file": str(ROOT / "configs" / "tiny_mixtral_experiment.json"),
            "hetinfer_graph_id": "tiny-mixtral-4layer-1npu2pim",
            "moe_control_timing": "analytic_npu",
            "evidence_class": "NPU LUT / pointwise AIM; Router and Combine use explicit analytic NPU timing",
        })
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mixtral", "qwen"), required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--prefill", type=int)
    args = parser.parse_args()
    if (args.batch is None) != (args.prefill is None):
        parser.error("--batch and --prefill must be supplied together")
    selected = WORKLOADS if args.batch is None else ((args.batch, args.prefill),)
    if any(pair not in WORKLOADS for pair in selected):
        parser.error("workload is outside the agreed seven-point grid")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run DOPS/AIM inside a Slurm allocation")
    env = os.environ.copy()
    env["PIM_TRACE_SCALE_REPEATS"] = "0"
    env["PIM_TRACE_STRICT"] = "1"
    env["PIM_LATENCY_CACHE_FILE"] = str(
        ROOT / "output" / "tiny_moe_experiment" / f"{args.model}_aim_exact.pkl")
    for batch, prefill in selected:
        cfg = experiment_config(args.model, batch, prefill)
        out = Path(cfg["hetinfer_prior_out"]).parents[1]
        out.mkdir(parents=True, exist_ok=True)
        (out / "native").mkdir(exist_ok=True)
        config_path = out / "config.json"
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"START {args.model} b{batch}-p{prefill}-h32", flush=True)
        with (out / f"native_export.{os.environ['SLURM_JOB_ID']}.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "src" / "main.py"),
                            "evaluate", "--config", str(config_path)],
                           cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        network = json.loads(Path(cfg["hetinfer_network_out"]).read_text())
        if len(network["networks"]) != HORIZON + 1:
            raise RuntimeError("Native export must contain prefill plus 32 decode snapshots")
        print(f"NATIVE_OK {out}", flush=True)


if __name__ == "__main__":
    main()
