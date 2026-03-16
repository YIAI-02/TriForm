#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TIME_HEADER_CANDS = [
    "task duration(us)",
    "total time(us)",
    "device time(us)",
    "kernel execution time(us)",
    "aic_total_time(us)",
    "execution_time(us)",
    "duration(us)",
    "avg time(us)",
    "running_time(us)",
    "running time(us)",
]
NAME_HEADER_CANDS = [
    "kernel name",
    "op name",
    "task name",
    "name",
]
TARGET_NAME_TOKENS = ("format_conv_bench",)

# Bench-body line ranges. These are the mode-specific function bodies in format_conv_bench_kernel.h.
MODE_LINE_RANGES: Dict[str, List[Tuple[str, int, int]]] = {
    "nd2nz_a": [("format_conv_bench_kernel.h", 93, 120)],
    "nd2nz_b": [("format_conv_bench_kernel.h", 122, 149)],
    "nz2zz_a": [("format_conv_bench_kernel.h", 151, 173)],
    "nz2zn_b": [("format_conv_bench_kernel.h", 175, 197)],
    "nd2zz_a": [("format_conv_bench_kernel.h", 199, 234)],
    "nd2zn_b": [("format_conv_bench_kernel.h", 236, 271)],
}

# Optional common wrapper/dispatch ranges if the user wants full-kernel time instead of bench-body only.
COMMON_FULL_KERNEL_RANGES: List[Tuple[str, int, int]] = [
    ("auto_gen_format_conv_bench.cpp", 39, 40),
    ("format_conv_bench.cpp", 11, 15),
    ("format_conv_bench_kernel.h", 45, 69),
]

CODE_PATH_RE = re.compile(r"(?P<file>[^/\\:]+):(\d+)$")


def _norm(s: str) -> str:
    return " ".join(str(s or "").strip().lower().replace("_", " ").split())


def _find_header(headers: Iterable[str], cands: Iterable[str]) -> Optional[str]:
    norm_map = {_norm(h): h for h in headers}
    for c in cands:
        if c in norm_map:
            return norm_map[c]
    for h in headers:
        nh = _norm(h)
        for c in cands:
            if c in nh:
                return h
    return None


def _parse_path_line(raw: str) -> Tuple[str, Optional[int]]:
    m = CODE_PATH_RE.search(str(raw or ""))
    if not m:
        return str(raw or ""), None
    return m.group("file"), int(m.group(2))


def _ranges_for_mode(mode: str, scope: str) -> List[Tuple[str, int, int]]:
    ranges = list(MODE_LINE_RANGES.get(mode, []))
    if scope == "full_kernel":
        ranges.extend(COMMON_FULL_KERNEL_RANGES)
    return ranges


def _match_any_range(file_name: str, line_no: int, ranges: Sequence[Tuple[str, int, int]]) -> bool:
    for file_pat, lo, hi in ranges:
        if file_name == file_pat and lo <= line_no <= hi:
            return True
    return False


def _parse_code_exe_csv(path: Path, mode: str, repeat: int, inner_loops: int, scope: str) -> Tuple[Optional[float], Dict[str, object]]:
    ranges = _ranges_for_mode(mode, scope)
    if not ranges:
        return None, {"parser": "code_exe", "reason": f"no line range for mode={mode}"}

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None, {"parser": "code_exe", "reason": "empty fieldnames"}
            code_h = _find_header(reader.fieldnames, ["code"])
            time_h = _find_header(reader.fieldnames, ["running_time(us)", "running time(us)", "time(us)"])
            call_h = _find_header(reader.fieldnames, ["call_count", "call count"])
            cyc_h = _find_header(reader.fieldnames, ["cycles", "cycle"])
            if code_h is None or time_h is None:
                return None, {
                    "parser": "code_exe",
                    "reason": "missing code/time header",
                    "fieldnames": list(reader.fieldnames),
                }

            selected_total_us = 0.0
            selected_rows = 0
            selected_call_count = 0.0
            selected_cycles = 0.0
            matched_lines: List[Dict[str, object]] = []
            for row in reader:
                file_name, line_no = _parse_path_line(row.get(code_h, ""))
                if line_no is None:
                    continue
                if not _match_any_range(file_name, line_no, ranges):
                    continue
                raw_t = str(row.get(time_h, "") or "").replace(",", "")
                try:
                    t_us = float(raw_t)
                except Exception:
                    continue
                selected_total_us += t_us
                selected_rows += 1
                try:
                    if call_h is not None:
                        selected_call_count += float(str(row.get(call_h, "") or "0").replace(",", ""))
                except Exception:
                    pass
                try:
                    if cyc_h is not None:
                        selected_cycles += float(str(row.get(cyc_h, "") or "0").replace(",", ""))
                except Exception:
                    pass
                matched_lines.append({
                    "file": file_name,
                    "line": line_no,
                    "running_time_us": t_us,
                })

            norm = max(1, int(repeat)) * max(1, int(inner_loops))
            if selected_rows == 0:
                return None, {
                    "parser": "code_exe",
                    "reason": "no matched rows in selected ranges",
                    "ranges": ranges,
                    "time_header": time_h,
                    "code_header": code_h,
                }
            avg_us = selected_total_us / float(norm)
            return avg_us, {
                "parser": "code_exe",
                "scope": scope,
                "repeat": int(repeat),
                "inner_loops": int(inner_loops),
                "normalization": norm,
                "time_header": time_h,
                "code_header": code_h,
                "selected_total_us": selected_total_us,
                "selected_rows": selected_rows,
                "selected_call_count": selected_call_count,
                "selected_cycles": selected_cycles,
                "ranges": ranges,
                "matched_lines": matched_lines,
            }
    except Exception as e:
        return None, {"parser": "code_exe", "reason": f"exception: {e}"}


# Legacy/fallback parser for summary-style msprof CSVs.
def _parse_summary_csv(path: Path) -> Tuple[Optional[float], Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None, {}
            time_h = _find_header(reader.fieldnames, TIME_HEADER_CANDS)
            name_h = _find_header(reader.fieldnames, NAME_HEADER_CANDS)
            if time_h is None:
                return None, {}
            vals: List[float] = []
            for row in reader:
                if not isinstance(row, dict):
                    continue
                if name_h is not None:
                    name_val = _norm(row.get(name_h, ""))
                    if TARGET_NAME_TOKENS and not any(tok in name_val for tok in TARGET_NAME_TOKENS):
                        continue
                raw = str(row.get(time_h, "") or "").replace(",", "")
                try:
                    vals.append(float(raw))
                except Exception:
                    continue
            if not vals:
                return None, {"parser": "summary", "time_header": time_h, "name_header": name_h or ""}
            return mean(vals), {"parser": "summary", "time_header": time_h, "name_header": name_h or ""}
    except Exception as e:
        return None, {"parser": "summary", "reason": f"exception: {e}"}


def _latest_opprof_dir(case_dir: Path) -> Optional[Path]:
    opprofs = sorted([p for p in case_dir.rglob("OPPROF_*") if p.is_dir()])
    return opprofs[-1] if opprofs else None


def parse_tree(root: Path, repeat: int, inner_loops: int, scope: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for mode_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        mode = mode_dir.name.strip().lower()
        for case_dir in sorted([p for p in mode_dir.iterdir() if p.is_dir()]):
            shape = case_dir.name
            try:
                m, n, k = [int(x) for x in shape.split("x")[:3]]
            except Exception:
                m = n = k = 0

            best_us: Optional[float] = None
            best_meta: Dict[str, object] = {}
            best_source: List[str] = []

            search_root = _latest_opprof_dir(case_dir) or case_dir
            code_exe_csvs = sorted(search_root.rglob("*_code_exe.csv"))
            if code_exe_csvs:
                total_us = 0.0
                merged_meta: Dict[str, object] = {
                    "parser": "code_exe",
                    "scope": scope,
                    "repeat": int(repeat),
                    "inner_loops": int(inner_loops),
                    "normalization": max(1, int(repeat)) * max(1, int(inner_loops)),
                    "files": [],
                }
                any_ok = False
                for p in code_exe_csvs:
                    avg_us, meta = _parse_code_exe_csv(p, mode=mode, repeat=repeat, inner_loops=inner_loops, scope=scope)
                    if avg_us is None:
                        continue
                    any_ok = True
                    total_us += avg_us
                    merged_meta["files"].append({"path": str(p), **meta})
                    best_source.append(str(p))
                if any_ok:
                    best_us = total_us
                    best_meta = merged_meta

            if best_us is None:
                # Fallback: scan any summary-style CSV under the chosen profile tree.
                for csv_path in search_root.rglob("*.csv"):
                    avg_us, meta = _parse_summary_csv(csv_path)
                    if avg_us is None:
                        continue
                    if best_us is None or avg_us < best_us:
                        best_us = avg_us
                        best_meta = meta
                        best_source = [str(csv_path)]

            rows.append({
                "mode": mode,
                "m": m,
                "n": n,
                "k": k,
                "avg_time_us": best_us,
                "repeat": int(repeat),
                "inner_loops": int(inner_loops),
                "source_csv": ";".join(best_source),
                "meta": json.dumps(best_meta, ensure_ascii=False),
            })
    return rows


def write_csv(rows: List[Dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "m", "n", "k", "avg_time_us", "repeat", "inner_loops", "source_csv", "meta"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--inner-loops", type=int, default=1)
    ap.add_argument("--scope", choices=["bench_body", "full_kernel"], default="bench_body")
    args = ap.parse_args()

    rows = parse_tree(Path(args.root), repeat=args.repeat, inner_loops=args.inner_loops, scope=args.scope)
    write_csv(rows, Path(args.out))
    print(f"[INFO] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
