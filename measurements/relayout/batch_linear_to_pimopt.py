#!/usr/bin/env python3
"""
export PROJECT_ROOT=/path/to/TriForm
export RELAYOUT_DIR=${PROJECT_ROOT}/measurements/relayout
export RAMU_EXE=${PROJECT_ROOT}/src/ramulator2
export RAMU_REPO=${PROJECT_ROOT}/src

python3 ${RELAYOUT_DIR}/batch_linear_to_pimopt.py \
  --runner ${RELAYOUT_DIR}/run_linear_to_pimopt.py \
  --repo ${RAMU_REPO} \
  --exe ${RAMU_EXE} \
  --config ${RELAYOUT_DIR}/linear_to_pimopt_gddr6.yaml \
  --workdir ${RELAYOUT_DIR}/batch_runs \
  --shape 128x128 128x512 512x512 512x1024 4096x1024 4096x4096 4096x14336 14336x4096
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_shape(s: str) -> tuple[int, int]:
    try:
        r, c = s.lower().split('x')
        return int(r), int(c)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid shape '{s}', expected ROWSxCOLS") from exc


def run_one(
    python_exe: str,
    runner: Path,
    repo: Path,
    exe: Path,
    config: Path,
    workdir: Path,
    rows: int,
    cols: int,
    dtype_bytes: int,
    burst_bytes: int,
    channels: int,
    banks: int,
    gpr_count: int,
    tck_ps: float,
    extra_args: list[str] | None = None,
) -> dict:
    tag = f"{rows}x{cols}_fp{dtype_bytes * 8}"
    cmd = [
        python_exe,
        str(runner),
        "--repo", str(repo),
        "--exe", str(exe),
        "--config", str(config),
        "--rows", str(rows),
        "--cols", str(cols),
        "--dtype-bytes", str(dtype_bytes),
        "--burst-bytes", str(burst_bytes),
        "--channels", str(channels),
        "--banks", str(banks),
        "--gpr-count", str(gpr_count),
        "--tck-ps", str(tck_ps),
        "--workdir", str(workdir),
        "--tag", tag,
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"run failed for {rows}x{cols}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    summary_path = workdir / tag / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


DEFAULT_SHAPES = [
    (4096, 4096),
    (4096, 14336),
    (14336, 4096),
    (8192, 8192),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple Linear->PIM-OPT simulations and aggregate bandwidth")
    parser.add_argument("--runner", type=Path, required=True, help="path to run_linear_to_pimopt.py")
    parser.add_argument("--repo", type=Path, required=True, help="repo/work directory for ramulator")
    parser.add_argument("--exe", type=Path, required=True, help="path to ramulator executable")
    parser.add_argument("--config", type=Path, required=True, help="path to YAML config")
    parser.add_argument("--workdir", type=Path, required=True, help="directory to store per-shape outputs")
    parser.add_argument("--shape", type=parse_shape, nargs="*", default=None, help="list like 4096x4096 4096x14336")
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--burst-bytes", type=int, default=32)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--gpr-count", type=int, default=16)
    parser.add_argument("--tck-ps", type=float, default=500.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    args, extra = parser.parse_known_args()

    shapes = args.shape if args.shape else DEFAULT_SHAPES
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    per_run = []
    for rows, cols in shapes:
        summary = run_one(
            python_exe=args.python,
            runner=args.runner.resolve(),
            repo=args.repo.resolve(),
            exe=args.exe.resolve(),
            config=args.config.resolve(),
            workdir=workdir,
            rows=rows,
            cols=cols,
            dtype_bytes=args.dtype_bytes,
            burst_bytes=args.burst_bytes,
            channels=args.channels,
            banks=args.banks,
            gpr_count=args.gpr_count,
            tck_ps=args.tck_ps,
            extra_args=extra,
        )
        per_run.append(summary)
        print(
            f"done {rows}x{cols}: "
            f"time={summary['total_time_ms']:.6f} ms, "
            f"payload_bw={summary['payload_equivalent_bandwidth_GBps']:.3f} GB/s, "
            f"dram_bw={summary['dram_traffic_equivalent_bandwidth_GBps']:.3f} GB/s"
        )

    total_payload = sum(x["payload_bytes"] for x in per_run)
    total_dram = sum(x["dram_traffic_bytes"] for x in per_run)
    total_time_s = sum(x["total_time_s"] for x in per_run)

    agg = {
        "num_runs": len(per_run),
        "tCK_ps": args.tck_ps,
        "shapes": [x["shape"] for x in per_run],
        "per_run": per_run,
        "aggregate": {
            "total_payload_bytes": total_payload,
            "total_dram_traffic_bytes": total_dram,
            "total_time_s": total_time_s,
            "total_time_ms": total_time_s * 1e3,
            "aggregate_payload_equivalent_bandwidth_GBps": total_payload / total_time_s / 1e9,
            "aggregate_dram_traffic_equivalent_bandwidth_GBps": total_dram / total_time_s / 1e9,
            "arithmetic_mean_payload_bandwidth_GBps": statistics.mean(x["payload_equivalent_bandwidth_GBps"] for x in per_run),
            "arithmetic_mean_dram_bandwidth_GBps": statistics.mean(x["dram_traffic_equivalent_bandwidth_GBps"] for x in per_run),
            "median_payload_bandwidth_GBps": statistics.median(x["payload_equivalent_bandwidth_GBps"] for x in per_run),
            "median_dram_bandwidth_GBps": statistics.median(x["dram_traffic_equivalent_bandwidth_GBps"] for x in per_run),
        },
    }

    out_json = workdir / "batch_summary.json"
    out_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print("\n=== aggregate ===")
    print(json.dumps(agg["aggregate"], indent=2))
    print(f"\nwritten: {out_json}")


if __name__ == "__main__":
    main()
