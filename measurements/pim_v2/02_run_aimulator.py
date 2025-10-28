#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, subprocess, re, csv, json, sys
from pathlib import Path
from typing import Dict, Any, List, Optional

def parse_cycles_from_output(output_text: str, pattern: Optional[str] = None) -> Optional[int]:
    """find and parse cycles from Ramulator output text"""
    if pattern:
        patterns = [pattern]
    else:
        patterns = [
            r"memory_system_cycles:\s*([0-9]+)"
        ]
    
    for pat in patterns:
        m = re.search(pat, output_text)
        if m:
            try:
                return int(m.group(1))
            except:
                continue
    return None

def run_ramulator(config: Path, trace: Path, 
                  extra_args: str, cmd_template: str) -> tuple[int, str, str]:
    """run Ramulator and return (returncode, output_text, cmd)"""
    cmd = f"./ramulator2 -f {config} -t {trace} {extra_args}"
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
    
    return result.returncode, result.stdout, cmd

def load_metadata(trace_path: Path) -> Dict[str, Any]:
    """load metadata from JSON file or parse from filename"""
    json_path = trace_path.with_suffix(".json")
    
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to load {json_path}: {e}", file=sys.stderr)
    
    # try parsing from filename as fallback
    return parse_metadata_from_filename(trace_path)

def parse_metadata_from_filename(trace_path: Path) -> Dict[str, Any]:
    """parse metadata from trace filename"""
    meta = {
        "op": None, "dim": None, "n_heads": None, "n_kv_heads": None,
        "seqlen": None, "vector_dim": None, "matrix_col": None,
        "with_af": 0
    }
    
    name = trace_path.name
    
    # op
    m = re.match(r"^([A-Za-z0-9_]+)_", name)
    if m:
        meta["op"] = m.group(1)
    
    # dim_h{n_heads}_hk{n_kv_heads}
    m = re.search(r"dim(\d+)_h(\d+)_hk(\d+)", name)
    if m:
        meta["dim"] = int(m.group(1))
        meta["n_heads"] = int(m.group(2))
        meta["n_kv_heads"] = int(m.group(3))
        meta["vector_dim"] = meta["dim"]  # 默认vector_dim=dim
    
    # seqlen
    m = re.search(r"seq(\d+)", name)
    if m:
        meta["seqlen"] = int(m.group(1))
    
    # vector_dim
    m = re.search(r"vec(\d+)", name)
    if m:
        meta["vector_dim"] = int(m.group(1))
    
    # matrix_col
    m = re.search(r"col(\d+)", name)
    if m:
        meta["matrix_col"] = int(m.group(1))
    
    # with_af
    if "withaf" in name.lower():
        meta["with_af"] = 1
    
    return meta

def main():
    ap = argparse.ArgumentParser(
        description="Run Ramulator on traces and collect results"
    )
    ap.add_argument("--traces-dir", type=Path, required=True,
                    help="Directory containing trace files")
    ap.add_argument("--glob", type=str, default="**/*.trace",
                    help="Glob pattern for trace files (default: **/*.trace)")
    ap.add_argument("--config", type=Path, required=True,
                    help="Ramulator config file")
    ap.add_argument("--out-csv", type=Path, required=True,
                    help="Output CSV file with results")
    ap.add_argument("--extra-args", type=str, default="",
                    help="Extra arguments for Ramulator")
    ap.add_argument("--metric-regex", type=str, default=None,
                    help="Custom regex to extract cycles (default: memory_system_cycles)")
    ap.add_argument("--cmd-template", type=str, 
                    default="{bin} -f {config} -t {trace} {extra}",
                    help="Command template for running Ramulator")
    ap.add_argument("--save-logs", action="store_true",
                    help="Save Ramulator output to .log files")
    
    args = ap.parse_args()
    traces = sorted(args.traces_dir.rglob(args.glob))
    
    if not traces:
        print(f"Error: No traces found under {args.traces_dir} with glob {args.glob}", 
              file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(traces)} trace files")


    fieldnames = [
        "trace_file",
        "op",
        "dim",
        "n_heads", 
        "n_kv_heads",
        "seqlen",
        "vector_dim",
        "matrix_col",
        "ffn_dim",
        "with_af",
        "cycles",
        "returncode",
        "ramulator_cmd",

        "DRAM_column",
        "DRAM_row",
        "burst_length",
        "num_banks",
        "num_channels",
        "threads",
        "reuse_size",
        "channels_per_block",
        "max_seq_len",
    ]

    results = []
    

    for idx, trace in enumerate(traces, 1):
        print(f"[{idx}/{len(traces)}] Processing {trace.name}...", end=" ")
        
        meta = load_metadata(trace)
        
        returncode, output, cmd = run_ramulator(
            args.config,
            trace,
            args.extra_args,
            args.cmd_template
        )

        if args.save_logs:
            log_path = trace.with_suffix(trace.suffix + ".log")
            log_path.write_text(output, encoding="utf-8")
  
        cycles = parse_cycles_from_output(output, args.metric_regex)

        row = {
            "trace_file": str(trace.relative_to(args.traces_dir)),
            "op": meta.get("op", ""),
            "dim": meta.get("dim", ""),
            "n_heads": meta.get("n_heads", ""),
            "n_kv_heads": meta.get("n_kv_heads", ""),
            "seqlen": meta.get("seqlen", ""),
            "vector_dim": meta.get("vector_dim", ""),
            "matrix_col": meta.get("matrix_col", ""),
            "ffn_dim": meta.get("ffn_dim", ""),
            "with_af": meta.get("with_af", 0),
            "cycles": cycles if cycles is not None else "",
            "returncode": returncode,
            "ramulator_cmd": cmd,

            "DRAM_column": meta.get("DRAM_column", ""),
            "DRAM_row": meta.get("DRAM_row", ""),
            "burst_length": meta.get("burst_length", ""),
            "num_banks": meta.get("num_banks", ""),
            "num_channels": meta.get("num_channels", ""),
            "threads": meta.get("threads", ""),
            "reuse_size": meta.get("reuse_size", ""),
            "channels_per_block": meta.get("channels_per_block", ""),
            "max_seq_len": meta.get("max_seq_len", ""),
        }
        
        results.append(row)

        status = "OK" if returncode == 0 else f"FAILED(rc={returncode})"
        cycles_str = f"{cycles}" if cycles is not None else "N/A"
        print(f"{status}, cycles={cycles_str}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n[SUCCESS] Results written to {args.out_csv}")
    print(f"Total traces processed: {len(results)}")

    successful = sum(1 for r in results if r["returncode"] == 0)
    with_cycles = sum(1 for r in results if r["cycles"] != "")
    print(f"Successful runs: {successful}/{len(results)}")
    print(f"Traces with cycles: {with_cycles}/{len(results)}")

if __name__ == "__main__":
    main()