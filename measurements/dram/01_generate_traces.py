#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_generate_traces.py (ND layout)
---------------------------------
Purpose
  - Generate DRAM-side READ/WRITE traces using **ND layout**,
    where "ND" means the matrix is laid out **row-major** in DRAM.
    (Note: "ND" here is a layout tag only; it is NOT "N×D" as a size symbol.)
  - Matrix sizes are derived from model shape JSONs (e.g., llama/opt/palm *_shape.json).
  - For each matrix shape (rows, cols), we emit two traces:
        layoutND_<model>_<name>_<rows>x<cols>_read.trace
        layoutND_<model>_<name>_<rows>x<cols>_write.trace

Trace format
  - One request per line:
        r 0x<address>   # read cacheline start
        w 0x<address>   # write cacheline start
  - Addresses enumerate the **start** of each cacheline that covers
    the row-major array, stepping by `cacheline_bytes`.
  - No timestamps are emitted.

Side artifacts
  - address_map.csv  : name,model,rows,cols,dtype_B,cacheline_B,base_addr,total_bytes,trace_read,trace_write
  - trace_list.txt   : list of generated traces, one path per line (for batch running by 02 script)
  - README_traces.txt: parameters & notes, including the definition of ND (row-major layout)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set


# -------------------------------
# Address iterator (ND layout = row-major)
# -------------------------------

def iter_cacheline_addrs_rowmajor(base: int, rows: int, cols: int, dtype_bytes: int, line_bytes: int) -> Iterable[int]:
    """
    Yield cacheline *starting* addresses that cover a row-major (rows x cols) dense matrix.
    We step by whole cachelines, not by elements.
    """
    assert rows > 0 and cols > 0, "rows/cols must be positive"
    assert dtype_bytes > 0 and line_bytes > 0 and line_bytes % dtype_bytes == 0, "invalid dtype/cacheline size"
    elems_per_line = line_bytes // dtype_bytes
    total_elems = rows * cols
    for base_elem in range(0, total_elems, elems_per_line):
        yield base + base_elem * dtype_bytes


# -------------------------------
# Model shape helpers
# -------------------------------

def _canonical_model_name(shape: Dict) -> str:
    """Normalize model type/name for filenames (lowercase, trimmed)."""
    return str(shape.get("type", "model")).strip().lower()


def _derive_common_weight_shapes(shape_json: Dict) -> List[Tuple[str, int, int]]:
    """
    Derive common Transformer weight matrix sizes from a model shape.json.
    Returns [(logical_name, rows, cols), ...].
    Note: This function determines **sizes only**; ND means row-major layout when generating addresses.
    """
    model = _canonical_model_name(shape_json)
    h = int(shape_json["hidden_dim"])
    inter = int(shape_json["intermediate_dim"])
    qh = int(shape_json["q_head_num"])
    kvh = int(shape_json.get("kv_head_num", qh))
    head_dim = h // qh

    mats: List[Tuple[str, int, int]] = []
    # Attention projections (conservative common forms)
    mats.append((f"{model}_Wq", h, h))
    mats.append((f"{model}_Wo", h, h))
    kv_cols = kvh * head_dim  # shape for K/V when GQA is present
    mats.append((f"{model}_Wk", h, kv_cols))
    mats.append((f"{model}_Wv", h, kv_cols))
    # MLP
    mats.append((f"{model}_mlp_up", h, inter))
    mats.append((f"{model}_mlp_down", inter, h))
    return mats


def _auto_find_shape_files() -> List[Path]:
    """Find *_shape.json in current directory if --shapes is not provided."""
    found = sorted(Path(".").glob("*_shape.json"))
    return [Path(p) for p in found]


def _collect_unique_shapes(shape_files: List[Path], dedup_by_size: bool = True) -> List[Tuple[str, int, int]]:
    """
    Load all shape files and collect weight shapes.
    If dedup_by_size=True, only keep a single unique matrix size (name is different).
    """
    all_shapes: List[Tuple[str, int, int]] = []
    for sf in shape_files:
        j = json.loads(Path(sf).read_text(encoding="utf-8"))
        all_shapes.extend(_derive_common_weight_shapes(j))

    if not dedup_by_size:
        return all_shapes

    seen: Set[Tuple[int, int]] = set()
    keep: List[Tuple[str, int, int]] = []
    for name, r, c in all_shapes:
        key = (r, c)
        if key not in seen:
            keep.append((name, r, c))
            seen.add(key)
    return keep


# -------------------------------
# Trace writer
# -------------------------------

def write_trace(path: Path, addrs: Iterable[int], is_write: bool) -> None:
    op = "W" if is_write else "R"
    with path.open("w", encoding="utf-8") as f:
        for a in addrs:
            f.write(f"0x{a:x} {op}\n")


# -------------------------------
# CLI
# -------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate DRAM READ/WRITE traces using ND (row-major) layout.")
    p.add_argument("--shapes", nargs="+", help="Model shape JSON files. If omitted, auto-discovers *_shape.json in CWD.")
    p.add_argument("--outdir", type=str, required=True, help="Output directory.")
    p.add_argument("--dtype-bytes", type=int, default=2, help="Element size in bytes (default: 2=FP16).")
    p.add_argument("--cacheline-bytes", type=int, default=64, help="Cacheline size in bytes (default: 64).")
    p.add_argument("--base-addr", type=lambda x: int(x, 0), default=0x10000000, help="Starting base address (hex OK).")
    p.add_argument("--gap-mb", type=int, default=2, help="Gap between matrices' base addresses (in MB, default: 2).")
    p.add_argument("--no-dedup", action="store_true", help="Do not deduplicate by (rows,cols); keep all.")
    return p


def main():
    args = build_argparser().parse_args()


    # Resolve shape files (explicit or auto-discovered)
    if args.shapes is None or len(args.shapes) == 0:
        shape_files = _auto_find_shape_files()
        if not shape_files:
            raise SystemExit("No *_shape.json found; pass via --shapes or place files in CWD.")
    else:
        shape_files = [Path(p) for p in args.shapes]

    # Collect common weight shapes
    mats = _collect_unique_shapes(shape_files, dedup_by_size=(not args.no_dedup))

    # Emit traces (layout = ND=row-major)
    mapping_rows: List[str] = []
    manifest: List[str] = []
    base = int(args.base_addr)
    gap_bytes = int(args.gap_mb) * 1024 * 1024

    outdir = Path(args.outdir)
    traces_dir = outdir / shape_files[0].stem / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    for (name, rows, cols) in mats:
        # align base to cacheline
        cl = int(args.cacheline_bytes)
        base = (base + (cl - 1)) // cl * cl

        total_bytes = rows * cols * int(args.dtype_bytes)
        addrs = list(iter_cacheline_addrs_rowmajor(base, rows, cols, int(args.dtype_bytes), cl))

        read_path  = traces_dir / f"layoutND_{name}_{rows}x{cols}_read.trace"
        write_path = traces_dir / f"layoutND_{name}_{rows}x{cols}_write.trace"

        write_trace(read_path,  addrs, is_write=False)
        write_trace(write_path, addrs, is_write=True)

        mapping_rows.append(",".join([
            name, name.split("_", 1)[0] if "_" in name else "model",
            str(rows), str(cols),
            str(args.dtype_bytes), str(cl),
            f"0x{base:x}", str(total_bytes),
            str(read_path), str(write_path),
        ]))

        # move base beyond this matrix region plus a gap; re-align
        base += total_bytes + gap_bytes
        base = (base + (cl - 1)) // cl * cl

        manifest.append(str(read_path))
        manifest.append(str(write_path))

    # Side artifacts
    (outdir / shape_files[0].stem / "address_map.csv").write_text(
        "name,model,rows,cols,dtype_B,cacheline_B,base_addr,total_bytes,trace_read,trace_write\n" +
        "\n".join(mapping_rows),
        encoding="utf-8",
    )
    (outdir / shape_files[0].stem / "trace_list.txt").write_text("\n".join(manifest), encoding="utf-8")
    (outdir / shape_files[0].stem / "README_traces.txt").write_text(
        (
            "Traces generated with ND layout (row-major) for DRAM-side simulation.\n"
            f"dtype={args.dtype_bytes} B, cacheline={args.cacheline_bytes} B\n"
            f"base_addr={hex(int(args.base_addr))}, gap={args.gap_mb} MB\n"
            f"shapes={[str(p) for p in shape_files]}\n"
            f"dedup={'by (rows,cols)' if not args.no_dedup else 'disabled'}\n"
        ),
        encoding="utf-8",
    )

    print(f"[OK] layout=ND (row-major). Traces under: {outdir}")


if __name__ == "__main__":
    main()
