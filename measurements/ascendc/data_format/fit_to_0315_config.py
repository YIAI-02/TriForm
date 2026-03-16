#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

BLOCK = 16
ELEM_BYTES = 2.0  # fp16 / half


def ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def rows_cols_from_mode(mode: str, m: int, n: int, k: int) -> Tuple[int, int]:
    mode = str(mode or "").strip().lower()
    if mode in ("nd2nz_a", "nz2zz_a", "nd2zz_a"):
        return int(m), int(k)
    return int(k), int(n)


def nd_bytes(rows: int, cols: int) -> float:
    return float(rows) * float(cols) * ELEM_BYTES


def nz_bytes(rows: int, cols: int) -> float:
    return float(ceil_div(rows, BLOCK) * ceil_div(cols, BLOCK) * BLOCK * BLOCK) * ELEM_BYTES


def moved_bytes_for_mode(mode: str, m: int, n: int, k: int) -> Optional[float]:
    rows, cols = rows_cols_from_mode(mode, m, n, k)
    nd = nd_bytes(rows, cols)
    nz = nz_bytes(rows, cols)
    mode = str(mode or "").strip().lower()
    if mode in ("nd2nz_a", "nd2nz_b"):
        return nd + nz
    if mode == "nz2zz_a":
        return nz + nz
    if mode == "nz2zn_b":
        return nz + nz
    if mode == "nd2zz_a":
        return (nd + nz) + (nz + nz)
    if mode == "nd2zn_b":
        return (nd + nz) + (nz + nz)
    return None


def fit_line(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    if len(xs) != len(ys) or not xs:
        raise ValueError("empty fit")
    n = float(len(xs))
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
    return slope, intercept, r2


def bw_from_slope_us_per_byte(slope: float) -> float:
    if slope <= 0:
        return float("inf")
    return 1.0 / (1000.0 * slope)


def point_bw_gbs(moved_bytes: float, time_us: float) -> float:
    if moved_bytes <= 0 or time_us <= 0:
        return float("inf")
    return moved_bytes / (time_us * 1000.0)


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_time_us(row: Dict[str, str]) -> Optional[float]:
    for key in ("avg_time_us", "per_conversion_us", "normalized_time_us"):
        v = row.get(key, "")
        if v not in (None, "", "None"):
            try:
                return float(v)
            except Exception:
                pass
    return None


def fit_edge(pairs: List[Tuple[float, float]]) -> Dict[str, object]:
    xs = [p[0] for p in pairs if p[0] > 0 and p[1] is not None and p[1] > 0]
    ys = [p[1] for p in pairs if p[0] > 0 and p[1] is not None and p[1] > 0]
    if not xs:
        return {}
    if len(xs) == 1:
        bw = point_bw_gbs(xs[0], ys[0])
        return {
            "bw_gbs": bw,
            "overhead_us": 0.0,
            "r2": None,
            "samples": 1,
            "method": "single_point",
        }

    slope, intercept, r2 = fit_line(xs, ys)
    if slope > 0:
        bw = bw_from_slope_us_per_byte(slope)
        oh = max(0.0, intercept)
        return {
            "bw_gbs": bw,
            "overhead_us": oh,
            "r2": r2,
            "samples": len(xs),
            "method": "ols",
        }

    # No positive slope: fall back to median pointwise bandwidth through origin.
    pws = [point_bw_gbs(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    bw = median(pws) if pws else float("inf")
    return {
        "bw_gbs": bw,
        "overhead_us": 0.0,
        "r2": None,
        "samples": len(xs),
        "method": "median_through_origin",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    groups: Dict[str, List[Tuple[float, float, Dict[str, str]]]] = defaultdict(list)
    for row in rows:
        try:
            mode = str(row.get("mode", "") or "").strip().lower()
            m = int(row.get("m", 0) or 0)
            n = int(row.get("n", 0) or 0)
            k = int(row.get("k", 0) or 0)
            y = _parse_time_us(row)
            if y is None:
                continue
            x = moved_bytes_for_mode(mode, m, n, k)
            if x is None or x <= 0:
                continue
            groups[mode].append((x, y, row))
        except Exception:
            continue

    direct_map = {
        "nd2nz_a": "ND->NZ",
        "nd2nz_b": "ND->NZ",
        "nz2zz_a": "NZ->ZZ",
        "nz2zn_b": "NZ->ZN",
    }
    combined: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    raw_samples: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for mode, edge in direct_map.items():
        for x, y, row in groups.get(mode, []):
            combined[edge].append((x, y))
            raw_samples[edge].append({
                "mode": mode,
                "m": int(row.get("m", 0) or 0),
                "n": int(row.get("n", 0) or 0),
                "k": int(row.get("k", 0) or 0),
                "moved_bytes": x,
                "time_us": y,
                "point_bw_gbs": point_bw_gbs(x, y),
                "source_csv": row.get("source_csv", ""),
            })

    fits: Dict[str, Dict[str, object]] = {}
    paths_bw: Dict[str, float] = {}
    paths_oh: Dict[str, float] = {}
    for edge, pairs in combined.items():
        fit = fit_edge(pairs)
        if not fit:
            continue
        fits[edge] = fit
        bw = fit.get("bw_gbs")
        oh = fit.get("overhead_us")
        if isinstance(bw, (int, float)):
            paths_bw[edge] = float(bw)
        if isinstance(oh, (int, float)):
            paths_oh[edge] = float(oh)

    if "ND->NZ" in paths_bw:
        paths_bw.setdefault("NZ->ND", paths_bw["ND->NZ"])
        paths_oh.setdefault("NZ->ND", paths_oh.get("ND->NZ", 0.0))
    if "NZ->ZZ" in paths_bw:
        paths_bw.setdefault("ZZ->ND", paths_bw["NZ->ZZ"])
        paths_oh.setdefault("ZZ->ND", paths_oh.get("NZ->ZZ", 0.0))
    if "NZ->ZN" in paths_bw:
        paths_bw.setdefault("ZN->ND", paths_bw["NZ->ZN"])
        paths_oh.setdefault("ZN->ND", paths_oh.get("NZ->ZN", 0.0))

    cross_checks = []
    for chain_mode, chain_edges in {
        "nd2zz_a": ("ND->NZ", "NZ->ZZ"),
        "nd2zn_b": ("ND->NZ", "NZ->ZN"),
    }.items():
        pred_ready = all(e in fits for e in chain_edges)
        for x, y, row in groups.get(chain_mode, []):
            pred = None
            if pred_ready:
                m = int(row.get("m", 0) or 0)
                n = int(row.get("n", 0) or 0)
                k = int(row.get("k", 0) or 0)
                rows_, cols_ = rows_cols_from_mode(chain_mode, m, n, k)
                nd = nd_bytes(rows_, cols_)
                nz = nz_bytes(rows_, cols_)
                e0 = fits[chain_edges[0]]
                e1 = fits[chain_edges[1]]
                pred = (
                    float(e0.get("overhead_us", 0.0)) + (nd + nz) / (float(e0["bw_gbs"]) * 1000.0)
                    + float(e1.get("overhead_us", 0.0)) + (nz + nz) / (float(e1["bw_gbs"]) * 1000.0)
                )
            cross_checks.append({
                "mode": chain_mode,
                "m": row.get("m"),
                "n": row.get("n"),
                "k": row.get("k"),
                "measured_us": y,
                "predicted_from_direct_edges_us": pred,
                "abs_err_us": (None if pred is None else abs(pred - y)),
                "source_csv": row.get("source_csv", ""),
            })

    out = {
        "fits": fits,
        "raw_samples": raw_samples,
        "cross_checks": cross_checks,
        "format_conv_bw_gbs": {
            "npu": {
                "default": 409.6,
                "allow_dense_fallback": False,
                "paths": paths_bw,
            }
        },
        "format_conv_overhead_us": {
            "npu": {
                "default": 0.0,
                "allow_dense_fallback": False,
                "paths": paths_oh,
            }
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    main()
