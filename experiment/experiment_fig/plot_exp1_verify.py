#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot experiment results from *merge_all.csv files.
find . -type f -print > files.txt
python3 plot_exp1_verify.py \
  --file-list ../../verify/hw_hardware_1npu_2aim/st64/llama_7b_bf16_b16_s64/files.txt \
  --dims "2048x64, 2048x128, 2048x256, 2048x512" \
  --output ../../figs/verify/hw_hardware_1npu_2aim/st64/llama_7b_bf16_b16_s64/exp1_2048_x.pdf 
  
  --search-dir /path/to/verify/evaluate_single_test/hardware_1gpu_4aim \
    
    
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D


# --- Config ---
# Strategy order: keep preferred methods first, but ALWAYS put hefthint at the very end.
PREFERRED_ORDER: List[str] = ["pd", "ianus", "facil", "hefthint"]

# Display name overrides (only affects x-axis tick labels).
DISPLAY_NAME_MAP: Dict[str, str] = {
    # user-facing label for hefthint
    "hefthint": "this work",
}

# Manually exclude strategies from plotting (edit this list if you want).
# You can also pass the same via CLI: --exclude ianus,facil
MANUAL_EXCLUDE: List[str] = []

# For plotting label "hefthint", candidates on disk can be either "heft" or "hefthint"
HEFT_VARIANTS: List[str] = ["heft", "hefthint"]

# Parse: <strategy>_<Lin>x<Lout>.merge_all.csv
FNAME_RE = re.compile(r"^(?P<algo>.+?)_(?P<lin>\d+)x(?P<lout>\d+)\.merge_all\.csv$")


@dataclass
class Metrics:
    algo_label: str               # label used in plot (strategy name)
    source_variant: str           # actual variant on disk (may differ for hefthint chosen from heft/hefthint)
    csv_path: Path
    n_rows: int

    trace_prefill: float
    trace_decode: float
    trace_total: float

    meas_prefill: float
    meas_decode: float
    meas_total: float

    delta_total_mean: float       # mean(delta_total_s) = mean(meas_total - trace_total)
    mean_abs_rel_err_pct: float   # mean(|delta_total|/meas_total)*100


def _read_text_lines(p: Path) -> List[str]:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for raw in txt.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def resolve_file_list_paths(file_list: Path, root: Optional[Path] = None) -> List[Path]:
    """
    Read a file-list and resolve lines to existing Paths.

    Robust resolution order for each line:
      1) absolute path: itself
      2) <cwd>/<line>
      3) <root>/<line>                (if --root provided)
      4) <file_list_dir>/<line>
      5) for parent in parents(file_list_dir):
            <parent>/<line_stripped>   (line_stripped removes leading "./")
    """
    file_list = file_list.resolve()
    base_dir = file_list.parent
    cwd = Path.cwd().resolve()
    candidates = _read_text_lines(file_list)

    resolved: List[Path] = []
    missing: List[str] = []

    for line in candidates:
        line_stripped = line[2:] if line.startswith("./") else line
        p = Path(line)
        tries: List[Path] = []
        if p.is_absolute():
            tries = [p]
        else:
            tries.append(cwd / p)
            if root is not None:
                tries.append(root / p)
            tries.append(base_dir / p)

            for parent in [base_dir] + list(base_dir.parents):
                tries.append(parent / line_stripped)

        hit: Optional[Path] = None
        for t in tries:
            if t.exists():
                hit = t.resolve()
                break

        if hit is None:
            missing.append(line)
            continue

        resolved.append(hit)

    if missing:
        print(f"[WARN] {len(missing)} paths from file list do NOT exist in this environment.", file=sys.stderr)
        for ex in missing[:5]:
            print(f"       missing: {ex}", file=sys.stderr)

    return resolved


def glob_merge_all(search_dir: Path) -> List[Path]:
    return sorted(search_dir.rglob("*.merge_all.csv"))


def index_merge_all(csv_paths: List[Path]) -> Dict[Tuple[int, int], Dict[str, List[Path]]]:
    """
    Build an index:
      (Lin, Lout) -> { algo_on_disk -> [csv_paths...] }
    """
    idx: Dict[Tuple[int, int], Dict[str, List[Path]]] = {}
    for p in csv_paths:
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        algo = m.group("algo").lower()
        lin = int(m.group("lin"))
        lout = int(m.group("lout"))
        idx.setdefault((lin, lout), {}).setdefault(algo, []).append(p)
    return idx


def load_metrics(csv_path: Path, algo_label: str, source_variant: str) -> Metrics:
    df = pd.read_csv(csv_path)

    required = [
        "prefill_time_s", "decode_time_s", "total_time_s",
        "trace_prefill_s", "trace_decode_s", "trace_total_s",
        "delta_total_s",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["total_time_s", "trace_total_s"])
    if df.empty:
        raise ValueError(f"{csv_path} has no valid rows after dropping NaNs in total columns")

    means = df[required].mean(numeric_only=True)

    rel_err = np.abs(df["delta_total_s"]) / df["total_time_s"]
    mean_abs_rel_err_pct = float(np.nanmean(rel_err) * 100.0)

    return Metrics(
        algo_label=algo_label,
        source_variant=source_variant,
        csv_path=csv_path,
        n_rows=int(len(df)),

        trace_prefill=float(means["trace_prefill_s"]),
        trace_decode=float(means["trace_decode_s"]),
        trace_total=float(means["trace_total_s"]),

        meas_prefill=float(means["prefill_time_s"]),
        meas_decode=float(means["decode_time_s"]),
        meas_total=float(means["total_time_s"]),

        delta_total_mean=float(means["delta_total_s"]),
        mean_abs_rel_err_pct=mean_abs_rel_err_pct,
    )


def pick_best_by_trace_total(paths: List[Path], algo_label: str, source_variant: str) -> Metrics:
    """
    If multiple csv candidates exist, pick the one with the smallest mean(trace_total_s).
    """
    best: Optional[Metrics] = None
    for p in paths:
        try:
            m = load_metrics(p, algo_label=algo_label, source_variant=source_variant)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}", file=sys.stderr)
            continue
        if best is None or m.trace_total < best.trace_total:
            best = m
    if best is None:
        raise RuntimeError(f"No valid csv found for {algo_label} (variant={source_variant}).")
    return best


def infer_dims(idx: Dict[Tuple[int, int], Dict[str, List[Path]]]) -> List[Tuple[int, int]]:
    return sorted(idx.keys(), key=lambda t: (t[0] * t[1], t[0], t[1]))


def parse_dims_arg(s: str) -> List[Tuple[int, int]]:
    dims: List[Tuple[int, int]] = []
    for part in s.split(","):
        part = part.strip().lower().replace("×", "x")
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Bad dim '{part}', expected like 1024x1024")
        a, b = part.split("x", 1)
        dims.append((int(a), int(b)))
    return dims


def build_dim_results(idx: Dict[Tuple[int, int], Dict[str, List[Path]]],
                      dims: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Dict[str, Metrics]]:
    """
    Return:
      (Lin,Lout) -> { strategy_label -> Metrics }

    Notes:
      - All discovered strategies are included.
      - "heft" and "hefthint" are merged into a single plotted label "hefthint",
        selecting the faster variant by mean(trace_total_s).
    """
    out: Dict[Tuple[int, int], Dict[str, Metrics]] = {}

    for dim in dims:
        lin, lout = dim
        dim_entry = idx.get(dim, {})
        res: Dict[str, Metrics] = {}

        # 1) Normal strategies (everything except heft/heftHint)
        for algo_on_disk, paths in dim_entry.items():
            if algo_on_disk in HEFT_VARIANTS:
                continue
            try:
                res[algo_on_disk] = pick_best_by_trace_total(paths, algo_label=algo_on_disk, source_variant=algo_on_disk)
            except Exception as e:
                print(f"[WARN] {lin}x{lout}: cannot load '{algo_on_disk}': {e}", file=sys.stderr)

        # 2) Special: heft vs hefthint -> plotted label "hefthint"
        variant_metrics: List[Metrics] = []
        for variant in HEFT_VARIANTS:
            if variant in dim_entry:
                try:
                    variant_metrics.append(
                        pick_best_by_trace_total(dim_entry[variant], algo_label="hefthint", source_variant=variant)
                    )
                except Exception as e:
                    print(f"[WARN] {lin}x{lout}: cannot load '{variant}': {e}", file=sys.stderr)

        if variant_metrics:
            best = min(variant_metrics, key=lambda m: m.trace_total)
            res["hefthint"] = best
            print(f"[INFO] {lin}x{lout}: choose '{best.source_variant}' for plotted label 'hefthint' "
                  f"(mean trace_total_s={best.trace_total:.6g}) from {best.csv_path}")
        else:
            # not an error; just no such strategy for this dim
            pass

        out[dim] = res

    return out


def pretty_strategy_name(strategy_key: str) -> str:
    """Human-facing label for x-axis ticks."""
    return DISPLAY_NAME_MAP.get(strategy_key, strategy_key)


def format_speedup(sp: float) -> str:
    """Format speedup multiplier text shown next to speedup points."""
    if not np.isfinite(sp):
        return ""
    if sp >= 100:
        return f"{sp:.0f}×"
    if sp >= 10:
        return f"{sp:.1f}×"
    return f"{sp:.2f}×"


def _normalize_strategy_token(s: str) -> str:
    """Normalize strategy tokens for exclude-list matching (case-insensitive, alias-aware)."""
    raw = (s or "").strip().lower()
    compact = re.sub(r"[\s_\-]+", "", raw)

    # aliases for hefthint
    if compact in {"thiswork", "hefthint", "heft"}:
        return "hefthint"

    return raw


def parse_exclude_arg(exclude_arg: Optional[str]) -> List[str]:
    if not exclude_arg:
        return []
    parts = [p.strip() for p in exclude_arg.split(",")]
    return [p for p in parts if p]


def strategy_order(res: Dict[str, Metrics]) -> List[str]:
    present = list(res.keys())
    # Keep preferred order (except hefthint), then any other discovered strategies,
    # and ALWAYS put hefthint at the very end if present.
    preferred_no_heft = [s for s in PREFERRED_ORDER if s != "hefthint"]
    order: List[str] = [s for s in preferred_no_heft if s in res]
    rest = sorted([s for s in present if s not in set(PREFERRED_ORDER) and s != "hefthint"])
    order += rest
    if "hefthint" in res:
        order.append("hefthint")
    return order


def plot_results(dim_results: Dict[Tuple[int, int], Dict[str, Metrics]],
                 dims: List[Tuple[int, int]],
                 output: Path,
                 title: Optional[str] = None,
                 show_error_text: bool = True,
                 dpi: int = 200,
                 border_lw: float = 0.5) -> None:

    # Colors (tweak as needed)
    deep_green = "#2ca02c"
    light_green = "#98df8a"
    deep_blue = "#1f77b4"
    light_blue = "#aec7e8"

    # Speedup styling
    trace_sp_color = "#9e9e9e"  # trace speedup line (simulation) in grey
    meas_sp_color  = "k"        # measured speedup (black squares, connected)

    # Determine global y-lims (time) and speedup y-lims
    max_time = 0.0
    max_speedup = 1.0
    max_categories = 1

    for dim in dims:
        res = dim_results.get(dim, {})
        if not res:
            continue

        order = strategy_order(res)
        max_categories = max(max_categories, len(order))

        for m in res.values():
            max_time = max(max_time, m.meas_total, m.trace_total)

        pd_m = res.get("pd")
        if pd_m:
            # Trace-based speedup (simulation)
            if pd_m.trace_total > 0:
                base_trace = pd_m.trace_total
                for m in res.values():
                    if m.trace_total and m.trace_total > 0:
                        sp = base_trace / m.trace_total
                        if np.isfinite(sp):
                            max_speedup = max(max_speedup, float(sp))

            # Measurement-based speedup (real run time)
            if pd_m.meas_total > 0:
                base_meas = pd_m.meas_total
                for m in res.values():
                    if m.meas_total and m.meas_total > 0:
                        sp = base_meas / m.meas_total
                        if np.isfinite(sp):
                            max_speedup = max(max_speedup, float(sp))

    if max_time <= 0:
        max_time = 1.0

    # Headroom for arrows/text
    y_max = max_time * 1.25
    # Extra headroom for per-point speedup labels
    sp_max = max(1.2, max_speedup * 1.30)

    # Dynamic figure width (keeps labels readable when there are many strategies)
    per_subplot_w = max(4.5, 0.45 * max_categories)
    fig_w = max(20.0, 4.0 * per_subplot_w)
    fig_w = min(fig_w, 80.0)  # avoid absurdly wide figures
    fig_h = 3.0

    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(fig_w, fig_h))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    bar_w = 0.34

    for i, ax in enumerate(axes):
        # Spine width to match bar borders visually
        for sp in ax.spines.values():
            sp.set_linewidth(border_lw)
        ax.tick_params(width=border_lw * 0.6)

        if i >= len(dims):
            ax.axis("off")
            continue

        lin, lout = dims[i]
        res = dim_results.get((lin, lout), {})

        if not res:
            ax.set_title(f"{lin}x{lout}\n(no data)")
            ax.set_xticks([])
            ax.set_ylim(0, y_max)
            continue

        order = strategy_order(res)
        n = len(order)
        x = np.arange(n, dtype=float)

        # Build arrays in order
        trace_prefill = np.array([res[a].trace_prefill for a in order], dtype=float)
        trace_decode  = np.array([res[a].trace_decode  for a in order], dtype=float)
        meas_prefill  = np.array([res[a].meas_prefill  for a in order], dtype=float)
        meas_decode   = np.array([res[a].meas_decode   for a in order], dtype=float)
        trace_total   = np.array([res[a].trace_total   for a in order], dtype=float)
        meas_total    = np.array([res[a].meas_total    for a in order], dtype=float)
        err_pct       = np.array([res[a].mean_abs_rel_err_pct for a in order], dtype=float)

        pos_trace = x - bar_w / 2
        pos_meas  = x + bar_w / 2

        # Stacked bars (no edges on segments; we draw a single outline per stacked bar)
        ax.bar(pos_trace, trace_prefill, width=bar_w, color=deep_green, edgecolor="none", linewidth=0, zorder=2)
        ax.bar(pos_trace, trace_decode,  width=bar_w, bottom=trace_prefill, color=light_green, edgecolor="none", linewidth=0, zorder=2)

        ax.bar(pos_meas,  meas_prefill, width=bar_w, color=deep_blue, edgecolor="none", linewidth=0, zorder=2)
        ax.bar(pos_meas,  meas_decode,  width=bar_w, bottom=meas_prefill, color=light_blue, edgecolor="none", linewidth=0, zorder=2)

        # Thick outlines around each stacked bar
        trace_stack_total = trace_prefill + trace_decode
        meas_stack_total  = meas_prefill + meas_decode
        for px, h in zip(pos_trace, trace_stack_total):
            if not np.isfinite(h):
                continue
            ax.add_patch(Rectangle((px - bar_w / 2, 0), bar_w, h,
                                   fill=False, edgecolor="k", linewidth=border_lw, zorder=4))
        for px, h in zip(pos_meas, meas_stack_total):
            if not np.isfinite(h):
                continue
            ax.add_patch(Rectangle((px - bar_w / 2, 0), bar_w, h,
                                   fill=False, edgecolor="k", linewidth=border_lw, zorder=4))

        ax.set_title(f"{lin}x{lout}")
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_strategy_name(a) for a in order], rotation=30, ha="right", fontsize=12)
        ax.set_ylim(0, y_max)

        if i == 0:
            ax.set_ylabel("Time (s)")

        # Mean error arrows (between trace_total and meas_total)
        # Place the arrow above the *shorter* of the two bars (instead of in-between).
        cap_half = bar_w * 0.55  # cap spans roughly one bar width
        for j in range(n):
            yt = trace_total[j]
            ym = meas_total[j]
            if not (np.isfinite(yt) and np.isfinite(ym)):
                continue

            # Arrow x-position: align with the shorter bar
            xc = pos_trace[j] if yt <= ym else pos_meas[j]
            ax.plot([xc - cap_half, xc + cap_half], [yt, yt], color="k", lw=border_lw, zorder=5)
            ax.plot([xc - cap_half, xc + cap_half], [ym, ym], color="k", lw=border_lw, zorder=5)
            ax.annotate(
                "",
                xy=(xc, yt),
                xytext=(xc, ym),
                arrowprops=dict(arrowstyle="<->", color="k", lw=border_lw * 0.8),
                zorder=6,
            )

            if show_error_text and np.isfinite(err_pct[j]):
                y_text = max(yt, ym) + y_max * 0.015
                ax.text(xc, y_text, f"{err_pct[j]:.1f}%", ha="center", va="bottom", fontsize=12, zorder=7)

        # Speedup line (secondary axis)
        ax2 = ax.twinx()
        for sp in ax2.spines.values():
            sp.set_linewidth(border_lw)
        ax2.tick_params(width=border_lw * 0.6)

        pd_m = res.get("pd")

        # Trace-based speedup (simulation)
        if pd_m and pd_m.trace_total > 0:
            base_trace = pd_m.trace_total
            speedup_trace = np.array(
                [base_trace / res[a].trace_total if res[a].trace_total > 0 else np.nan for a in order],
                dtype=float,
            )
        else:
            speedup_trace = np.array([np.nan] * n, dtype=float)

        # Measurement-based speedup (real run time)
        if pd_m and pd_m.meas_total > 0:
            base_meas = pd_m.meas_total
            speedup_meas = np.array(
                [base_meas / res[a].meas_total if res[a].meas_total > 0 else np.nan for a in order],
                dtype=float,
            )
        else:
            speedup_meas = np.array([np.nan] * n, dtype=float)

        # Trace speedup line in grey (was black)
        ax2.plot(
            x,
            speedup_trace,
            color=trace_sp_color,
            lw=border_lw * 0.8,
            marker="o",
            markersize=4,
            zorder=10,
        )

        # Measured speedup: black squares, connected
        ax2.plot(
            x,
            speedup_meas,
            linestyle="-",
            lw=border_lw * 0.8,
            color=meas_sp_color,
            marker="s",
            markersize=3.5,
            zorder=10,
        )

        ax2.set_ylim(0, sp_max)

        for j, sp in enumerate(speedup_trace):
            if not np.isfinite(sp):
                continue
            ax2.annotate(
                format_speedup(float(sp)),
                xy=(x[j], float(sp)),
                textcoords="offset points",
                xytext=(0, -9),
                ha="center",
                va="top",
                fontsize=10,
                color=trace_sp_color,
                zorder=11,
            )

        # Label measured speedup at each point (black)
        for j, sp in enumerate(speedup_meas):
            if not np.isfinite(sp):
                continue
            ax2.annotate(
                format_speedup(float(sp)),
                xy=(x[j], float(sp)),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                va="bottom",
                fontsize=10,
                color=meas_sp_color,
                zorder=11,
            )

        # Only show speedup ticks on the right-most visible subplot
        if i == min(len(dims), 4) - 1:
            ax2.set_ylabel("Speedup (vs pd)")
        else:
            ax2.set_yticks([])
            ax2.set_ylabel("")

        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

    if title:
        fig.suptitle(title, y=1.02)

    # Global legend
    handles = [
        Patch(facecolor=deep_green, label="trace_prefill_s"),
        Patch(facecolor=light_green, label="trace_decode_s"),
        Patch(facecolor=deep_blue, label="prefill_time_s"),
        Patch(facecolor=light_blue, label="decode_time_s"),
        Line2D([0], [0], color=trace_sp_color, lw=border_lw * 0.8, marker="o", markersize=4,
               label="speedup (trace_total, vs pd)"),
        Line2D([0], [0], color=meas_sp_color, lw=border_lw * 0.8, linestyle="-", marker="s", markersize=4,
               label="speedup (meas_total, vs pd)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.12),fontsize=14)

    # # Note like your sample
    # fig.text(0.99, 1.02, "mean error relative to measurement%", ha="right", va="bottom", fontsize=12)

    # Layout: leave more space for rotated x labels
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"[OK] saved figure -> {output.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-list", type=str, default=None,
                    help="Path to a text file listing result files (recommended).")
    ap.add_argument("--root", type=str, default=None,
                    help="Optional root directory to resolve relative paths in file-list.")
    ap.add_argument("--search-dir", type=str, default=None,
                    help="If --file-list is not provided, recursively search this directory for *.merge_all.csv")
    ap.add_argument("--dims", type=str, default=None,
                    help="Comma-separated dims to plot, like: 1024x1024,2048x1024,4096x4096,.... "
                         "If not given, infer from discovered csv files.")
    ap.add_argument("--output", type=str, default="plot.png", help="Output image path.")
    ap.add_argument("--title", type=str, default=None, help="Figure title.")
    ap.add_argument("--no-error-text", action="store_true", help="Do not draw % text for mean error.")
    ap.add_argument("--exclude", type=str, default=None,
                    help="Comma-separated strategy names to exclude from plotting (case-insensitive). "
                         "Example: --exclude ianus,facil . Aliases: 'this work'/'heft' -> hefthint.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--border-lw", type=float, default=1.0, help="Bar outline / spine linewidth.")
    ap.add_argument("--show", action="store_true", help="Show interactively (in addition to saving).")

    args = ap.parse_args()

    csv_paths: List[Path] = []
    if args.file_list:
        fl = Path(args.file_list)
        if not fl.exists():
            ap.error(f"--file-list not found: {fl}")
        root = Path(args.root).resolve() if args.root else None
        paths = resolve_file_list_paths(fl, root=root)
        csv_paths = [p for p in paths if p.name.endswith(".merge_all.csv")]
    else:
        if not args.search_dir:
            ap.error("Provide either --file-list or --search-dir")
        sd = Path(args.search_dir)
        if not sd.exists():
            ap.error(f"--search-dir not found: {sd}")
        csv_paths = glob_merge_all(sd)

    if not csv_paths:
        ap.error("No *.merge_all.csv found. Check inputs / paths.")

    idx = index_merge_all(csv_paths)

    # Decide dims
    if args.dims:
        dims = parse_dims_arg(args.dims)
    else:
        dims = infer_dims(idx)

    # Keep at most 4 dims (to match required 4 subplots)
    if len(dims) >= 4:
        dims = dims[:4]
    else:
        print(f"[WARN] only found {len(dims)} dims; remaining subplots will be blank.", file=sys.stderr)

    dim_results = build_dim_results(idx, dims)

    # Apply manual/CLI excludes
    exclude_tokens: List[str] = []
    exclude_tokens.extend(MANUAL_EXCLUDE)
    exclude_tokens.extend(parse_exclude_arg(args.exclude))
    exclude_set = {_normalize_strategy_token(t) for t in exclude_tokens if t and _normalize_strategy_token(t)}

    if exclude_set:
        for dim in list(dim_results.keys()):
            res = dim_results.get(dim, {})
            for k in list(res.keys()):
                if _normalize_strategy_token(k) in exclude_set:
                    res.pop(k, None)

    plot_results(
        dim_results=dim_results,
        dims=dims,
        output=Path(args.output),
        title=args.title,
        show_error_text=(not args.no_error_text),
        dpi=args.dpi,
        border_lw=float(args.border_lw),
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

