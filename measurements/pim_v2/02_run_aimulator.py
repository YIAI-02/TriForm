#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_run_aimulator.py
读取 pim_v2/traces/<op> 下的 AiM trace（每算子一个目录），
将每个 trace 聚合为“微操作统计 + 估算 cycles”，写到一个 CSV（每行一个 trace）。
用法：
  python 02_run_aimulator.py --traces-dir pim_v2/traces --out-csv pim_v2/aim_results.csv

备注：这里没有真正“跑” aim_sim，而是按 aim_sim.py 里的 timing_constant
对每条 `AiM XXX ...` 进行累计（等价于 only_trace 的时间模型）。
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 与 aim_sim.py 保持一致的 timing 常数（如需更新请同步）
TIMING_CONSTANT = {
    "COPY_GB_BK": 45.5,
    "COPY_BK_GB": 42.5,
    "WR_GB": 32,
    "MAC_ABK": 49,
    "MAC_BK_BK": 49,
    "MAC_BK_GB": 49,
    "EWMUL": 47,
    "EWADD": 0,
    "AF": 60,
    "RD_MAC": 37.5,
    "RD_AF": 37.5,
    "WR_BIAS": 37.5,
    "RD_SBK": 30.5,
    "WR_SBK": 45.5,
    # 其它未列到的 AiM 指令（如 WR_ABK、SYNC、EOC 等）不计入 cycles
}

# 支持的 AiM 行解析（正则）
LINE_PATTERNS = [
    # 三元：<op> <op_size> <mask> <row_idx>
    (re.compile(r'^AiM\s+(MAC_ABK|MAC_BK_BK|MAC_BK_GB|EWMUL|WR_GB|EWADD|COPY_BK_GB|COPY_GB_BK|COPY_BKGB|COPY_GBBK)\s+(\d+)'), 2),
    # 一元：只带掩码，无 op_size（AF / RD_AF / RD_MAC / WR_BIAS / WR_SBK / RD_SBK）
    (re.compile(r'^AiM\s+(AF|RD_AF|RD_MAC|WR_BIAS|WR_SBK|RD_SBK)\b'), 0),
]

# 从 trace 文件名提取 (op, dim, n_heads, n_kv_heads, seqlen, V, N, withaf)
NAME_RE = re.compile(
    r'(?P<op>[a-zA-Z0-9_]+)'
    r'_dim(?P<dim>\d+)_h(?P<h>\d+)_hk(?P<hk>\d+)'
    r'(?:_seq(?P<seq>\d+))?'
    r'(?:_vec(?P<V>\d+))?'
    r'(?:_col(?P<N>\d+))?'
    r'(?:_withaf)?\.trace$'
)

MICRO_OPS = sorted(TIMING_CONSTANT.keys())

def parse_trace_file(path: Path) -> Tuple[Dict[str, int], Dict[str, int], float]:
    """
    解析单个 AiM trace：返回 (calls, opsize_sums, cycles_total)
    - calls[name]：某微操作出现次数
    - opsize_sums[name]：该微操作 op_size 的总和（没有 op_size 的默认为 0）
    - cycles_total：按 timing 常数 + op_size 累积的总 cycles
    """
    calls = {k: 0 for k in MICRO_OPS}
    opsizes = {k: 0 for k in MICRO_OPS}
    cycles = 0.0

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        matched = False
        for creg, idx in LINE_PATTERNS:
            m = creg.match(line)
            if not m:
                continue
            op = m.group(1)
            # 兼容 COPY_BKGB / COPY_GBBK 的别名
            if op == "COPY_BKGB":
                op = "COPY_BK_GB"
            if op == "COPY_GBBK":
                op = "COPY_GB_BK"
            # 统计
            calls.setdefault(op, 0)
            opsizes.setdefault(op, 0)
            calls[op] += 1
            op_size = int(m.group(2)) if idx == 2 else 0
            opsizes[op] += op_size
            # 累加 cycles
            if op in TIMING_CONSTANT:
                cycles += TIMING_CONSTANT[op] + op_size
            matched = True
            break
        # 其它行（WR_ABK / SYNC / EOC 等）忽略
    return calls, opsizes, cycles

def parse_meta_from_name(name: str) -> Dict[str, Optional[int]]:
    m = NAME_RE.search(name)
    if not m:
        return {"op": None, "dim": None, "n_heads": None, "n_kv_heads": None,
                "seqlen": None, "V": None, "N": None, "withaf": False}
    gd = m.groupdict()
    return {
        "op": gd["op"],
        "dim": int(gd["dim"]),
        "n_heads": int(gd["h"]),
        "n_kv_heads": int(gd["hk"]),
        "seqlen": int(gd["seq"]) if gd["seq"] else None,
        "V": int(gd["V"]) if gd["V"] else None,
        "N": int(gd["N"]) if gd["N"] else None,
        "withaf": ("withaf" in name),
    }

def scan_traces(traces_dir: Path) -> Tuple[List[Path], Dict[str, List[Path]]]:
    all_traces = []
    by_op = {}
    for sub in sorted(traces_dir.iterdir()):
        if sub.is_dir():
            for t in sorted(sub.glob("*.trace")):
                all_traces.append(t)
                by_op.setdefault(sub.name, []).append(t)
        elif sub.is_file() and sub.suffix == ".trace":
            all_traces.append(sub)
            by_op.setdefault("unknown", []).append(sub)
    return all_traces, by_op

def main():
    ap = argparse.ArgumentParser(description="Aggregate AiM traces (per-op) into a CSV of features + cycles")
    ap.add_argument("--traces-dir", type=Path, required=True, help="root directory that contains per-op trace subfolders")
    ap.add_argument("--out-csv", type=Path, required=True, help="output CSV path")
    args = ap.parse_args()

    all_traces, by_op = scan_traces(args.traces_dir)
    if not all_traces:
        raise SystemExit(f"No .trace files under {args.traces_dir}")

    # CSV header
    fieldnames = [
        "op", "trace_file", "dim", "n_heads", "n_kv_heads", "seqlen", "V", "N", "withaf", "cycles_total"
    ]
    for k in MICRO_OPS:
        fieldnames += [f"{k}_calls", f"{k}_opsize"]

    rows = []
    for t in all_traces:
        calls, opsizes, cycles = parse_trace_file(t)
        meta = parse_meta_from_name(t.name)
        row = {
            "op": meta["op"] or t.parent.name,
            "trace_file": str(t),
            "dim": meta["dim"], "n_heads": meta["n_heads"], "n_kv_heads": meta["n_kv_heads"],
            "seqlen": meta["seqlen"], "V": meta["V"], "N": meta["N"], "withaf": int(meta["withaf"]),
            "cycles_total": float(cycles),
        }
        for k in MICRO_OPS:
            row[f"{k}_calls"] = int(calls.get(k, 0))
            row[f"{k}_opsize"] = int(opsizes.get(k, 0))
        rows.append(row)

    # 写 CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[ok] aggregated {len(rows)} traces from {args.traces_dir} -> {args.out_csv}")

if __name__ == "__main__":
    main()
