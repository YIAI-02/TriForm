#!/usr/bin/env python3
import argparse, os, subprocess, sys
from pathlib import Path

def run_one(bin_path, cfg_path, trace_path, outdir):
    outdir = Path(outdir)
    outlog_dir = outdir / "outlog"
    stats_dir = outdir / "stats"
    outlog_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    base = Path(trace_path).stem
    stdout_file = outlog_dir / f"{base}.stdout.txt"
    stats_file = stats_dir / f"{base}.stats.txt"
    cmd = [bin_path, cfg_path, "--mode=dram", "--stats", str(stats_file), trace_path]
    with open(stdout_file, "w") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=False)
    return stdout_file, stats_file, proc.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ramulator-bin", required=True)
    ap.add_argument("--config", help="Config YAML path if your ramulator supports -t TRC")
    ap.add_argument("--trace-lists", required=True, help="Path to trace_list.txt produced by 01_generate_traces.py")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    traces = [line.strip() for line in Path(args.trace_lists).read_text().splitlines() if line.strip()]
    for tr in traces:
        stdout_file, stats_file, rc = run_one(args.ramulator_bin, args.config, tr, args.outdir)
        print(f"[{Path(tr).name}] return_code={rc} -> {stats_file}")

if __name__ == "__main__":
    main()
