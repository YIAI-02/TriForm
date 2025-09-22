#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_parse_and_fit.py  (per-model, read-only)
------------------------------------------
What this script does
  - Scan ONLY stats files whose names end with "*_read.stats.txt".
  - Parse `read_latency_sum` from each file.
  - Relate it to ACTUAL read traffic of the paired "*_read.trace":
        bytes_read = (#lines in *_read.trace) * cacheline_B
    where `cacheline_B` and `trace_read` are taken from 01's address_map.csv.

Multi-model support (this version)
  - Accept multiple address_map.csv files (produced by 01). Each address_map row
    carries the `model` field; results are grouped strictly by `model`.
  - Outputs (per model only; NO global/combined outputs):
      <run-outdir>/fit_results/<model>/summary_read.csv
      <run-outdir>/fit_results/<model>/read_fit.txt

Usage example
  python3 03_parse_and_fit.py \
      --run-outdir ./out_runs \
      --stats-dir  ./out_runs/stats \
      --address-map ./out_nd/mistral_shape/address_map.csv ./out_nd/mpt_shape/address_map.csv

Notes
  - If --stats-dir is omitted, the script auto-discovers one of:
        <run-outdir>/stats, <run-outdir>/outlog/stats, <run-outdir>/logs
  - If a trace file referenced by address_map is not present locally,
        fallback to rows*cols*dtype_B to estimate request count (ceil division).
"""

import argparse, csv, json, os, re, sys, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# -------------------------------
# Address map ingestion (from 01)
# -------------------------------

def _read_address_map_one(path: Path) -> Dict[str, Dict]:
    """
    Parse one address_map.csv and build a dict:
        stem -> {
            'model': str,
            'trace_read': Path,
            'cacheline_B': int,
            'rows': int, 'cols': int, 'dtype_B': int, 'total_bytes': int,
            'amap_dir': Path
        }
    """
    out: Dict[str, Dict] = {}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return out
    hdr = [x.strip() for x in lines[0].split(",")]
    idx = {name:i for i,name in enumerate(hdr)}
    rows_k = 'rows' if 'rows' in idx else None
    cols_k = 'cols' if 'cols' in idx else None

    for ln in lines[1:]:
        parts = [x.strip() for x in ln.split(",")]
        def _get(k: str) -> Optional[str]:
            j = idx.get(k)
            return parts[j] if j is not None and j < len(parts) else None

        tr_read = _get('trace_read')
        if not tr_read:
            continue
        stem = Path(tr_read).stem
        rec = {
            'model':       _get('model') or 'model',
            'trace_read':  Path(tr_read),
            'cacheline_B': int(_get('cacheline_B')) if _get('cacheline_B') else None,
            'rows':        int(_get(rows_k)) if rows_k and _get(rows_k) else None,
            'cols':        int(_get(cols_k)) if cols_k and _get(cols_k) else None,
            'dtype_B':     int(_get('dtype_B')) if _get('dtype_B') else None,
            'total_bytes': int(_get('total_bytes')) if _get('total_bytes') else None,
            'amap_dir':    path.parent
        }
        out[stem] = rec
    return out


def _read_address_maps(paths: List[Path]) -> Dict[str, Dict]:
    """Merge multiple address_map.csv files into a single stem->meta map."""
    mp: Dict[str, Dict] = {}
    for p in paths:
        if not p.exists():
            print(f"[WARN] address_map not found: {p}")
            continue
        sub = _read_address_map_one(p)
        mp.update(sub)  # later files override earlier on collision (should be rare)
    return mp

# -------------------------------
# Stats helpers
# -------------------------------

def _find_stats_dir(run_outdir: Path) -> Optional[Path]:
    """Auto-discover stats directory under run_outdir."""
    cand = [run_outdir / "stats", run_outdir / "outlog" / "stats", run_outdir / "logs"]
    for d in cand:
        if d.exists():
            return d
    return None

def _discover_read_stats(stats_dir: Path) -> List[Path]:
    """Locate *_read.stats.txt files only."""
    return sorted(stats_dir.glob("*_read.stats.txt"))

def _parse_read_latency_sum(text: str) -> Optional[float]:
    # Sum all "ramulator.read_latency_sum_<idx> <value>" occurrences
    pat_v1 = re.compile(
        r"ramulator\.read_latency_sum(?:_\d+)?\s+([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE
    )
    vals = [float(m.group(1)) for m in pat_v1.finditer(text)]
    if vals:
        return float(sum(vals))

    # Fallback (kept for compatibility)
    pat_generic = re.compile(
        r"\bread_latency_sum\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE
    )
    m = pat_generic.search(text)
    if m:
        return float(m.group(1))
    return None


# -------------------------------
# Utils
# -------------------------------

def _count_lines(path: Path) -> int:
    """Count lines in a text file quickly (used for counting read requests)."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)

def _ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b

def _least_squares(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Simple least squares fit: return (a, b) s.t. y ≈ a*x + b."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x*x for x in xs); sxy = sum(x*y for x,y in zip(xs,ys))
    denom = n*sxx - sx*sx
    if denom == 0:
        return 0.0, sy/n if n>0 else 0.0
    a = (n*sxy - sx*sy) / denom
    b = (sy - a*sx) / n
    return a, b

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-outdir", required=True, help="02 output root; per-model results will be written under this directory.")
    ap.add_argument("--stats-dir", default=None, help="Directory containing *_read.stats.txt. If omitted, auto-discover under run-outdir.")
    ap.add_argument("--address-map", nargs="+", required=True, help="One or more 01-generated address_map.csv paths (e.g., out_nd/*/address_map.csv).")
    ap.add_argument("--cacheline", type=int, default=64, help="Fallback cacheline bytes if address_map lacks it (default 64).")
    args = ap.parse_args()

    run_dir = Path(args.run_outdir)
    fit_root = run_dir / "fit_results"
    fit_root.mkdir(parents=True, exist_ok=True)

    stats_dir = Path(args.stats_dir) if args.stats_dir else _find_stats_dir(run_dir)
    if not stats_dir or not stats_dir.exists():
        raise SystemExit(f"Stats directory not found. Pass --stats-dir or ensure one of {{stats, outlog/stats, logs}} exists under {run_dir}")

    # Load address maps (multi-model)
    amap_paths = [Path(p) for p in args.address_map]
    stem_info = _read_address_maps(amap_paths)  # stem -> info{model,trace_read,cacheline_B,...}
    print(f"[INFO] Loaded {len(stem_info)} entries from {len(amap_paths)} address_map.csv files.")

    # Discover read stats
    stats_files = _discover_read_stats(stats_dir)
    if not stats_files:
        raise SystemExit(f"No *_read.stats.txt found in {stats_dir}")

    # Accumulators per model
    per_model_rows: Dict[str, List[List[str]]] = defaultdict(list)
    per_model_x: Dict[str, List[float]] = defaultdict(list)
    per_model_y: Dict[str, List[float]] = defaultdict(list)

    for sf in stats_files:
        # keep suffix "_read" in the stem to match trace_read's stem
        stem = sf.name.replace(".stats.txt", "")
        print(f"[INFO] Processing stats: {sf.name} (stem: {stem})")
        txt = sf.read_text(encoding="utf-8", errors="ignore")
        rls = _parse_read_latency_sum(txt)
        if rls is None:
            print(f"[WARN] read_latency_sum not found in {sf.name}; skip")
            continue

        meta = stem_info.get(stem)
        if not meta:
            print(f"[WARN] address_map has no entry for {stem}; skip")
            continue

        model = str(meta.get("model", "model"))
        clB = int(meta.get("cacheline_B") or args.cacheline)

        # Count requests from the paired *_read.trace; fallback to rows*cols if missing
        tr_path = meta['trace_read']
        if not tr_path.is_absolute():
            tr_path = meta['amap_dir'] / tr_path
        if tr_path.exists():
            reqs = _count_lines(tr_path)
            bytes_read = reqs * clB
        else:
            rows = meta.get('rows'); cols = meta.get('cols'); dtype_B = meta.get('dtype_B')
            if rows and cols and dtype_B:
                nd_bytes = int(rows) * int(cols) * int(dtype_B)
                reqs = _ceil_div(nd_bytes, clB)
                bytes_read = reqs * clB
            else:
                print(f"[WARN] cannot infer bytes for {stem}; skip")
                continue

        per_model_rows[model].append([stem, str(reqs), str(clB), str(bytes_read), f"{rls}"])
        per_model_x[model].append(float(bytes_read))
        per_model_y[model].append(float(rls))

    # Write per-model outputs only
    for model, rows in per_model_rows.items():
        mdir = fit_root / model
        mdir.mkdir(parents=True, exist_ok=True)
        # Summary CSV
        with (mdir / "summary_read.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["trace_stem", "requests", "cacheline_B", "bytes_read", "read_latency_sum"])
            w.writerows(rows)
        # Per-model linear fit
        a, b = _least_squares(per_model_x[model], per_model_y[model])
        (mdir / "read_fit.txt").write_text(
            f"Linear fit ({model}): read_latency_sum ≈ {a:.6e} * bytes_read + {b:.6e}\n",
            encoding="utf-8"
        )

    print(f"[OK] Per-model results are under: {fit_root}/<model>/")

if __name__ == "__main__":
    main()
