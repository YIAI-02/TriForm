#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a LUT (CSV) for MMAD / matmul total time on NPU simulator from profiler outputs.

It searches recursively under a given root directory for files ending with "*_code_exe.csv".
For each CSV it extracts running_time(us) for `mmad_custom_cube_only.h` lines 70/71/72/73,
sums them as the matmul total time, and writes a merged LUT CSV.

Typical directory layout (example):
  profile/
    1x128x128/
      OPPROF_.../
        simulator/core0.cubecore0/core0.cubecore0_code_exe.csv

Usage:
  python build_mmad_lut.py --root ./profile --out mmad_lut.csv
"""

from __future__ import annotations
import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


LINE_DEFAULT = (70, 71, 72, 73)


def parse_shape_from_path(p: Path) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    """
    Try to parse (M, N, K) from any path segment like '1x128x128' or similar.
    Returns (M, N, K, shape_str).
    """
    text = str(p)
    # Find patterns like 1x128x128 (3 dims) or more dims; we prefer exactly 3 numbers if possible.
    candidates = re.findall(r'(\d+(?:x\d+){2,})', text)
    # Prefer the last occurrence (closest to file path tail) and prefer exactly 3 dims
    best = None
    best_len = None
    for cand in candidates[::-1]:
        parts = cand.split('x')
        if len(parts) == 3:
            best = cand
            best_len = 3
            break
        if best is None:
            best = cand
            best_len = len(parts)
    if best is None:
        return None, None, None, None
    parts = best.split('x')
    if len(parts) >= 3:
        try:
            m, n, k = int(parts[0]), int(parts[1]), int(parts[2])
            return m, n, k, best
        except Exception:
            return None, None, None, best
    return None, None, None, best


@dataclass
class ExtractResult:
    csv_path: Path
    m: Optional[int]
    n: Optional[int]
    k: Optional[int]
    shape: Optional[str]
    total_us: Optional[float]
    per_line_us: Dict[int, float]
    missing_lines: List[int]


def extract_total_time(csv_path: Path, lines: Sequence[int] = LINE_DEFAULT) -> ExtractResult:
    df = pd.read_csv(csv_path)
    # Basic column normalization
    if "running_time(us)" not in df.columns:
        # sometimes it could be named differently; try best-effort
        time_col = None
        for c in df.columns:
            if "running_time" in c and "us" in c:
                time_col = c
                break
        if time_col is None:
            raise ValueError(f"Cannot find running_time(us) column in {csv_path}")
    else:
        time_col = "running_time(us)"

    # Filter rows for target file
    code_col = "code" if "code" in df.columns else df.columns[0]
    mask = df[code_col].astype(str).str.contains("mmad_custom_cube_only.h:", regex=False)
    sub = df.loc[mask, [code_col, time_col]].copy()

    # Extract line numbers
    sub["line"] = sub[code_col].astype(str).str.extract(r"mmad_custom_cube_only\.h:(\d+)")
    sub = sub.dropna(subset=["line"])
    sub["line"] = sub["line"].astype(int)

    # Keep only requested lines
    line_set = set(lines)
    sub = sub[sub["line"].isin(line_set)]

    per_line: Dict[int, float] = {}
    for ln, grp in sub.groupby("line"):
        # In case multiple rows exist for same line, sum them
        per_line[int(ln)] = float(pd.to_numeric(grp[time_col], errors="coerce").fillna(0).sum())

    missing = [ln for ln in lines if ln not in per_line]
    total = sum(per_line.values()) if per_line else None

    m, n, k, shape = parse_shape_from_path(csv_path)

    return ExtractResult(
        csv_path=csv_path,
        m=m,
        n=n,
        k=k,
        shape=shape,
        total_us=total,
        per_line_us=per_line,
        missing_lines=missing,
    )


def pick_latest_per_shape(results: List[ExtractResult]) -> List[ExtractResult]:
    """
    If multiple runs exist for the same shape, keep the latest modified CSV.
    """
    by_shape: Dict[str, ExtractResult] = {}
    for r in results:
        key = r.shape or str(r.csv_path)
        if key not in by_shape:
            by_shape[key] = r
        else:
            old = by_shape[key]
            if r.csv_path.stat().st_mtime > old.csv_path.stat().st_mtime:
                by_shape[key] = r
    # Sort by (M,N,K) if available, else by shape string
    def sort_key(r: ExtractResult):
        if r.m is not None and r.n is not None and r.k is not None:
            return (0, r.m, r.n, r.k, str(r.csv_path))
        return (1, r.shape or "", str(r.csv_path))
    return sorted(by_shape.values(), key=sort_key)


def write_lut_csv(results: List[ExtractResult], out_csv: Path, lines: Sequence[int] = LINE_DEFAULT) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["shape", "M", "N", "K", "total_time_us"] + [f"line{ln}_us" for ln in lines] + ["missing_lines", "csv_path"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {
                "shape": r.shape,
                "M": r.m,
                "N": r.n,
                "K": r.k,
                "total_time_us": r.total_us,
                "missing_lines": ",".join(map(str, r.missing_lines)) if r.missing_lines else "",
                "csv_path": str(r.csv_path),
            }
            for ln in lines:
                row[f"line{ln}_us"] = r.per_line_us.get(ln, 0.0)
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory that contains per-shape profiling folders.")
    ap.add_argument("--out", type=str, required=True, help="Output LUT csv path.")
    ap.add_argument("--lines", type=int, nargs="+", default=list(LINE_DEFAULT), help="Line numbers to sum, default: 70 71 72 73")
    ap.add_argument("--keep", choices=["latest", "all"], default="latest", help="If multiple runs per shape: keep latest or keep all.")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    lines = tuple(args.lines)

    csv_files = [p for p in root.rglob("*_code_exe.csv") if p.is_file()]
    if not csv_files:
        raise SystemExit(f"No '*_code_exe.csv' found under: {root}")

    results: List[ExtractResult] = []
    for p in csv_files:
        try:
            r = extract_total_time(p, lines=lines)
            # Keep only those that actually contain target lines
            if r.total_us is not None:
                results.append(r)
        except Exception as e:
            # Skip files that are not in expected format
            continue

    if not results:
        raise SystemExit("No valid MMAD code_exe.csv found (missing mmad_custom_cube_only.h:70-73).")

    if args.keep == "latest":
        results = pick_latest_per_shape(results)

    write_lut_csv(results, out, lines=lines)
    print(f"Saved LUT: {out} (rows={len(results)})")


if __name__ == "__main__":
    main()
