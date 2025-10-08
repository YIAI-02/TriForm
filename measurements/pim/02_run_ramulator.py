#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, subprocess, re, csv, sys, json
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# ------------------------- 规范化特征与匹配模式 -------------------------
# 规范化后的原语名（列名中会出现 <NAME>_calls 与 <NAME>_opsize）
FEATURE_SPECS = [
    # name, has_opsize, regex patterns（兼容多种书写）
    ("MAC_ABK",     True,  [r"^AiM\s+MAC_ABK\s+(\d+)"]),
    ("MAC_BK_BK",   True,  [r"^AiM\s+MAC_BK_BK\s+(\d+)"]),
    ("MAC_BK_GB",   True,  [r"^AiM\s+MAC_BK_GB\s+(\d+)"]),
    ("WR_GB",       True,  [r"^AiM\s+WR_GB\s+(\d+)"]),
    ("COPY_BK_GB",  True,  [r"^AiM\s+COPY_BK_GB\s+(\d+)", r"^AiM\s+COPY_BKGB\s+(\d+)"]),
    ("COPY_GB_BK",  True,  [r"^AiM\s+COPY_GB_BK\s+(\d+)", r"^AiM\s+COPY_GBBK\s+(\d+)"]),
    ("EWMUL",       True,  [r"^AiM\s+EWMUL\s+(\d+)"]),
    ("EWADD",       True,  [r"^AiM\s+EWADD\s+(\d+)"]),
    ("AF",          False, [r"^AiM\s+AF\b"]),
    ("RD_MAC",      False, [r"^AiM\s+RD_MAC\b"]),
    ("RD_AF",       False, [r"^AiM\s+RD_AF\b"]),
    ("WR_BIAS",     False, [r"^AiM\s+WR_BIAS\b"]),
    ("RD_SBK",      False, [r"^AiM\s+RD_SBK\b"]),
    ("WR_SBK",      False, [r"^AiM\s+WR_SBK\b"]),
]
FEATURE_NAMES = [n for n, _, _ in FEATURE_SPECS]
FEATURE_HAS_SIZE = {n: has for n, has, _ in FEATURE_SPECS}
FEATURE_PATTERNS = {n: [re.compile(p) for p in pats] for n, _, pats in FEATURE_SPECS}

DEFAULT_PATTERNS = [
    r"memory_system_cycles:\s*([0-9]+)"
]

def parse_metric(text: str, pattern: Optional[str]) -> Optional[int]:
    pats = [pattern] if pattern else DEFAULT_PATTERNS
    for pat in pats:
        m = re.search(pat, text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None

def parse_features(trace_path: Path) -> Dict[str, Tuple[int, int]]:
    counts = {name: [0, 0] for name in FEATURE_NAMES}  # name -> [calls, opsize]
    with trace_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for name in FEATURE_NAMES:
                for pat in FEATURE_PATTERNS[name]:
                    m = pat.search(line)
                    if not m:
                        continue
                    counts[name][0] += 1  # calls
                    if FEATURE_HAS_SIZE[name] and m.lastindex:
                        try:
                            counts[name][1] += int(m.group(1)) # opsize
                        except Exception:
                            pass
                    break
    return {k: (v[0], v[1]) for k, v in counts.items()} #calls  opsize

def parse_meta(trace_path: Path) -> Dict[str, str | int | None]:

    meta = {
        "op": None, "seqlen": None, "vector_dim": None, "matrix_col": None, "with_af": None,
        "dim": None, "n_heads": None, "n_kv_heads": None,
        "DRAM_column": None, "DRAM_row": None, "burst_length": None,
        "num_banks": None, "num_channels": None, "threads": None, "reuse_size": None,
        "channels_per_block": None, "max_seq_len": None,
    }
    j = trace_path.with_suffix(".json")
    if j.exists():
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
            for k in meta.keys():
                if k in d:
                    meta[k] = d[k]
        except Exception:
            pass

    name = trace_path.name
    # op
    m = re.match(r"^(score|output|weight)_", name)
    if m and meta["op"] is None:
        meta["op"] = m.group(1)
    # with_af
    if meta["with_af"] is None:
        meta["with_af"] = 1 if "_withaf" in name else 0
    # seqlen
    m = re.search(r"_seq(\d+)_", name)
    if m and meta["seqlen"] is None:
        meta["seqlen"] = int(m.group(1))
    # vector/matrix
    m = re.search(r"_vec(\d+)_", name)
    if m and meta["vector_dim"] is None:
        meta["vector_dim"] = int(m.group(1))
    m = re.search(r"_col(\d+)_", name)
    if m and meta["matrix_col"] is None:
        meta["matrix_col"] = int(m.group(1))
    # dim / heads
    m = re.search(r"_dim(\d+)_h(\d+)", name)
    if m:
        if meta["dim"] is None:
            meta["dim"] = int(m.group(1))
        if meta["n_heads"] is None:
            meta["n_heads"] = int(m.group(2))
    return meta

def run_one(bin_path: Path, config: Path, trace: Path, extra_args: str, cmd_template: str) -> tuple[int, str, str]:
    cmd = cmd_template.format(bin=str(bin_path), config=str(config), trace=str(trace), extra=extra_args)
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout, cmd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", type=Path, required=True)
    ap.add_argument("--glob", type=str, default="*.aim", help="Glob for trace files")
    ap.add_argument("--ramulator-bin", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--extra-args", type=str, default="", help="Extra args appended to the command")
    ap.add_argument("--metric-regex", type=str, default=None, help="Regex with one capture group yielding cycles")
    ap.add_argument("--cmd-template", type=str, default="{bin} -f {config} -t {trace} {extra}")
    args = ap.parse_args()

    traces = sorted(args.traces_dir.rglob(args.glob))
    if not traces:
        print(f"No traces found under {args.traces_dir} with glob {args.glob}", file=sys.stderr)
        sys.exit(1)

    base_cols = [
        "trace", "returncode", "cycles", "ramulator_cmd",
        "op", "with_af", "seqlen", "vector_dim", "matrix_col",
        "dim", "n_heads", "n_kv_heads",
        "DRAM_column", "DRAM_row", "burst_length", "num_banks", "num_channels",
        "threads", "reuse_size", "channels_per_block", "max_seq_len",
    ]
    feat_cols = []
    for name in FEATURE_NAMES:
        feat_cols.append(f"{name}_calls")
        feat_cols.append(f"{name}_opsize")
    fieldnames = base_cols + feat_cols

    rows = []
    for t in traces:
        # 1) trace features & meta
        feats = parse_features(t)  # name -> (calls, opsize)
        meta = parse_meta(t)

        # 2) ramulator
        rc, out, cmd = run_one(args.ramulator_bin, args.config, t, args.extra_args, args.cmd_template)
        log_path = t.with_suffix(t.suffix + ".log")
        log_path.write_text(out, encoding="utf-8")
        cycles = parse_metric(out, args.metric_regex)

        # 3) 写一行
        row = {
            "trace": str(t),
            "returncode": rc,
            "cycles": cycles if cycles is not None else "",
            "ramulator_cmd": cmd,
            **meta, #字典解包语法
        }
        for name in FEATURE_NAMES: #把trace的统计信息也写进去
            c, s = feats.get(name, (0, 0))
            row[f"{name}_calls"] = c
            row[f"{name}_opsize"] = s if FEATURE_HAS_SIZE[name] else 0
        rows.append(row)
        print(f"[done] {t.name}: rc={rc}, cycles={cycles}")

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
