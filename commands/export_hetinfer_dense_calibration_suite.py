#!/usr/bin/env python3
"""Export the six approved Dense calibration workloads on a compute node."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "commands"))
from export_hetinfer_full_suite import _config

CASES = {
    "W1": ("1.8b", 1, 16, 32),
    "W2": ("1.8b", 4, 16, 32),
    "W3": ("1.8b", 1, 256, 32),
    "W4": ("1.8b", 1, 16, 128),
    "W5": ("7b", 1, 16, 32),
    "W6": ("7b", 4, 256, 32),
    "W7": ("7b", 4, 256, 24),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use a Slurm compute allocation")
    variant, batch, prefill, horizon = CASES[args.case]
    output = ROOT / "output" / "dense_calibration_suite" / args.case
    output.mkdir(parents=True, exist_ok=True)
    existing = ROOT / "output" / "tiny_moe_experiment" / "qwen" / (
        f"b{batch}-p{prefill}-h{horizon}") / "bundle"
    if variant == "1.8b" and horizon == 32 and existing.is_dir():
        meta = json.loads((existing / "experiment.json").read_text())
        cfg = json.loads((existing.parent / "config.json").read_text())
        if (meta["layer_count"], meta["batch"], meta["prefill"], meta["decode_rounds"]) != (28, batch, prefill, horizon):
            raise ValueError("Existing bundle has a different workload")
        if cfg["model_variant"] != variant or meta["pim_trace_scale_repeats"] != 0:
            raise ValueError("Existing bundle has incompatible model/timing provenance")
        (output / "bundle_source.json").write_text(json.dumps({
            "case": args.case, "variant": variant, "bundle": str(existing),
            "reused": True, "config": str(existing.parent / "config.json")}, indent=2) + "\n")
        print(f"DENSE_BUNDLE_OK {args.case} {existing}", flush=True)
        return
    cfg = _config(dops_root=ROOT, het_infer_root=ROOT.parent / "v1-cama-work",
                  batch=batch, prefix_length=prefill,
                  workload=f"{args.case}-qwen{variant}-b{batch}-p{prefill}-h{horizon}")
    cfg.update({
        "model_variant": variant,
        "model_revision": f"shape-only:qwen_{variant}_shape.json",
        "shape_file": str(ROOT / "configs" / f"qwen_{variant}_shape.json"),
        "hetinfer_graph_id": f"qwen-{variant}-28layer-1npu2pim",
        "experiment_label": f"dense_calibration_{args.case}",
        "decode_len": horizon, "max_seq_len": prefill + horizon,
        "result_dir": str(output / "native_run"),
        "simulation_log_file": str(output / "pim_simulation.log"),
        "hetinfer_prior_out": str(output / "native" / "prior.json"),
        "hetinfer_network_out": str(output / "native" / "network.json"),
        "hetinfer_tensor_bindings_out": str(output / "native" / "tensor_bindings.json"),
        "pim_ramulator_timeout_s": 1800, "pim_trace_strict": True,
    })
    config = output / "config.json"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    (output / "native").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PIM_TRACE_SCALE_REPEATS="0", PIM_TRACE_STRICT="1",
               PIM_LATENCY_CACHE_FILE=str(output / "aim_exact.pkl"))
    known = ROOT / "output" / "tiny_moe_experiment" / "qwen_aim_exact.pkl"
    if args.case == "W7":
        known = ROOT / "output" / "dense_calibration_suite" / "W6" / "aim_exact.pkl"
    if (variant == "1.8b" or args.case == "W7") and known.is_file() and not (output / "aim_exact.pkl").exists():
        shutil.copy2(known, output / "aim_exact.pkl")
    native = output / "native" / "network.json"
    if not native.exists():
        with (output / f"native_export.{os.environ['SLURM_JOB_ID']}.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "src" / "main.py"),
                "evaluate", "--config", str(config)], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True)
    networks = json.loads(native.read_text())["networks"]
    if len(networks) != horizon + 1:
        raise ValueError("Native export is incomplete")
    if not (output / "bundle" / "experiment.json").exists():
        subprocess.run([sys.executable, str(ROOT / "src" / "hetinfer_experiment_export.py"),
                        "--config", str(config)], cwd=ROOT, env=env, check=True)
    (output / "bundle_source.json").write_text(json.dumps({
        "case": args.case, "variant": variant, "bundle": str(output / "bundle"),
        "reused": False, "config": str(config)}, indent=2) + "\n")
    print(f"DENSE_BUNDLE_OK {args.case} {output / 'bundle'}", flush=True)

if __name__ == "__main__":
    main()
