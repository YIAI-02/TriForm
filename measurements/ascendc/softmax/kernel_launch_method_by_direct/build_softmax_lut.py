#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_softmax_lut.py

Build a LUT (CSV) for Softmax kernel time on NPU simulator from profiler outputs.

This script is intentionally written in the same *style/format* as `build_mmad_lut.py`:

* Recursively searches under a given root directory for files ending with "*_code_exe.csv".
* From each matched CSV it extracts `running_time(us)` for `softmax_kernel.h:<line>`.
* By default it only extracts **line 98** (no accumulation across multiple lines).
* Writes a merged LUT CSV with columns aligned to `build_mmad_lut.py`.

Typical directory layout (example):

  profile/
    256x1024/
      OPPROF_.../
        simulator/core0.veccore0/core0.veccore0_code_exe.csv

Usage:
  python build_softmax_lut.py --root ./profile --out softmax_lut.csv

Notes:
* "Only one core" requirement: by default we only keep paths containing
  "core0.veccore" (so both `core0.veccore0` and `core0.veccore1` match) to avoid
  duplicating the same shape across multiple cores. Change via `--core-substr`
  if needed.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


LINE_DEFAULT = (98, 103)
TOTAL_PICK_LINES = (98, 103)
DEFAULT_CORE_SUBSTR = "core0.veccore"


def parse_shape_from_path(p: Path) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    """Try to parse shape dims from any path segment like '256x1024' or '1x128x128'.

    Returns (M, N, K, shape_str).
      * For 2D shape like '256x1024' -> (256, 1024, None, '256x1024')
      * For 3D shape like '1x128x128' -> (1, 128, 128, '1x128x128')
    """
    text = str(p)
    candidates = re.findall(r"(\d+(?:x\d+){1,})", text)
    if not candidates:
        return None, None, None, None

    best: Optional[str] = None
    best_rank = 999
    for cand in candidates[::-1]:
        parts = cand.split("x")
        # Rank: exactly 3 dims best, then exactly 2 dims, then others.
        if len(parts) == 3:
            best = cand
            best_rank = 0
            break
        if len(parts) == 2 and best_rank > 1:
            best = cand
            best_rank = 1
        elif best is None:
            best = cand

    if best is None:
        return None, None, None, None

    parts = best.split("x")
    try:
        m = int(parts[0]) if len(parts) >= 1 else None
        n = int(parts[1]) if len(parts) >= 2 else None
        k = int(parts[2]) if len(parts) >= 3 else None
        return m, n, k, best
    except Exception:
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


def _resolve_time_col(df: pd.DataFrame, csv_path: Path) -> str:
    if "running_time(us)" in df.columns:
        return "running_time(us)"
    for c in df.columns:
        cc = str(c)
        if "running_time" in cc and "us" in cc:
            return cc
    raise ValueError(f"Cannot find running_time(us) column in {csv_path}")


def _resolve_code_col(df: pd.DataFrame) -> str:
    if "code" in df.columns:
        return "code"
    return str(df.columns[0])


def extract_total_time(csv_path: Path, lines: Sequence[int] = LINE_DEFAULT) -> ExtractResult:
    df = pd.read_csv(csv_path)

    time_col = _resolve_time_col(df, csv_path)
    code_col = _resolve_code_col(df)

    mask = df[code_col].astype(str).str.contains("softmax_kernel.h", regex=False)
    sub = df.loc[mask, [code_col, time_col]].copy()

    sub["line"] = sub[code_col].astype(str).str.extract(r"softmax_kernel\.h\s*:\s*(\d+)")
    sub = sub.dropna(subset=["line"])
    sub["line"] = sub["line"].astype(int)
    per_line_all: Dict[int, float] = {}
    for ln, grp in sub.groupby("line"):
        per_line_all[int(ln)] = float(pd.to_numeric(grp[time_col], errors="coerce").fillna(0).sum())

    line_set = set(int(x) for x in lines)
    per_line: Dict[int, float] = {ln: v for ln, v in per_line_all.items() if ln in line_set}
    missing = [int(ln) for ln in lines if int(ln) not in per_line_all]

    candidates: List[float] = []
    for ln in TOTAL_PICK_LINES:
        v = per_line_all.get(int(ln))
        if v is not None:
            candidates.append(float(v))
    total = max(candidates) if candidates else None

    m, n, k, shape = parse_shape_from_path(csv_path)
    return ExtractResult(csv_path, m, n, k, shape, total, per_line, missing)


def pick_latest_per_shape(results: List[ExtractResult]) -> List[ExtractResult]:
    """If multiple runs exist for the same shape, keep the latest modified CSV."""

    def _core_rank(p: Path) -> Tuple[int, int, int]:
        s = str(p)
        m = re.search(r"core(\d+)\.veccore(\d+)", s)
        if m:
            return (int(m.group(1)), 0, int(m.group(2)))
        m = re.search(r"core(\d+)\.aicore(\d+)", s)
        if m:
            return (int(m.group(1)), 1, int(m.group(2)))
        return (999, 999, 999)

    by_shape: Dict[str, ExtractResult] = {}
    for r in results:
        key = r.shape or str(r.csv_path)
        if key not in by_shape:
            by_shape[key] = r
        else:
            old = by_shape[key]
            r_m = r.csv_path.stat().st_mtime
            o_m = old.csv_path.stat().st_mtime
            if r_m > o_m:
                by_shape[key] = r
            elif r_m == o_m and _core_rank(r.csv_path) < _core_rank(old.csv_path):
                by_shape[key] = r

    def sort_key(r: ExtractResult):
        if r.m is not None and r.n is not None:
            k = r.k if r.k is not None else -1
            return (0, r.m, r.n, k, str(r.csv_path))
        return (1, r.shape or "", str(r.csv_path))

    return sorted(by_shape.values(), key=sort_key)


def write_lut_csv(results: List[ExtractResult], out_csv: Path, lines: Sequence[int] = LINE_DEFAULT) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shape",
        "M",
        "N",
        "K",
        "total_time_us",
        *[f"line{ln}_us" for ln in lines],
        "missing_lines",
        "csv_path",
    ]

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
                row[f"line{ln}_us"] = r.per_line_us.get(int(ln), 0.0)
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory that contains per-shape profiling folders.")
    ap.add_argument("--out", type=str, required=True, help="Output LUT csv path.")
    ap.add_argument(
        "--lines",
        type=int,
        nargs="+",
        default=list(LINE_DEFAULT),
        help="Line numbers to extract (default: 98).",
    )
    ap.add_argument(
        "--core-substr",
        type=str,
        default=DEFAULT_CORE_SUBSTR,
        help=(
            "Only keep *_code_exe.csv whose path contains this substring to avoid multi-core duplicates. "
            "Default: core0.veccore (matches core0.veccore0/core0.veccore1 ...)"
        ),
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print reasons when a *_code_exe.csv is skipped (format/parse issues).",
    )
    ap.add_argument(
        "--keep",
        choices=["latest", "all"],
        default="latest",
        help="If multiple runs per shape: keep latest or keep all.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    lines = tuple(int(x) for x in args.lines)

    csv_files_all = [p for p in root.rglob("*_code_exe.csv") if p.is_file()]
    if not csv_files_all:
        raise SystemExit(f"No '*_code_exe.csv' found under: {root}")

    csv_files = csv_files_all
    if args.core_substr:
        csv_files = [p for p in csv_files_all if args.core_substr in str(p)]
        if not csv_files:
            csv_files = csv_files_all
            print(
                f"[WARN] No '*_code_exe.csv' path contains '{args.core_substr}'. "
                "Falling back to all cores; you may want to pass a correct --core-substr."
            )

    results: List[ExtractResult] = []
    for p in csv_files:
        try:
            r = extract_total_time(p, lines=lines)
            # Keep only those that actually contain target lines
            if r.total_us is not None:
                results.append(r)
            elif args.verbose:
                print(f"[SKIP] {p} -> no softmax_kernel.h rows")
        except Exception as e:
            if args.verbose:
                print(f"[SKIP] {p} -> {e}")
            continue

    if not results:
        raise SystemExit(
            "No valid softmax code_exe.csv found (missing softmax_kernel.h). "
            "Check --root/--core-substr and whether *_code_exe.csv contains softmax_kernel.h entries."
        )

    if args.keep == "latest":
        results = pick_latest_per_shape(results)

    write_lut_csv(results, out, lines=lines)
    print(f"Saved LUT: {out} (rows={len(results)})")


if __name__ == "__main__":
    main()