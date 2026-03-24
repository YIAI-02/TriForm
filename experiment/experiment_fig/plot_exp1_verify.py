
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot latency comparison from *merge_all.csv files.

find . -type f -name "*.merge_all.csv" | sort > files.txt
python3 plot_exp1_verify.py \
  --file-list ../../verify/sst8_rst8/llama_7b_fp16_b16_s8/files.txt \
  --algo-order "pd,attn_on_pim, ianus,facil,attacc,hefthint" \
  --exclude "weights_on_pim"\
  --dims "128x128,128x512,128x1024,1024x128,1024x512,1024x1024"\
  --name-map "pd=PD,ianus=PD+FFN,facil=PD+Linear,attacc=PD+Attention,attn_on_pim=AF,hefthint=Bifocal" \
  --output ../../figs/verify/sst8_rst8/llama_7b_fp16_b16_s8.pdf 

python3 plot_exp1_verify.py \
  --file-list ../../verify/sst8_rst8/qwen_1.8b_fp16_b8_s8/files.txt \
  --algo-order "pd,attn_on_pim, ianus,facil,attacc,hefthint" \
  --exclude "weights_on_pim"\
  --dims "128x128,128x512,128x1024,1024x128,1024x512,1024x1024"\
  --name-map "pd=PD,ianus=PD+FFN,facil=PD+Linear,attacc=PD+Attention,attn_on_pim=AF,hefthint=Bifocal" \
  --output ../../figs/verify/sst8_rst8/qwen_1p8b_fp16_b8_s8.pdf 

  
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

ARIAL_FONT_FAMILY = "Arial"
MIN_FONT_PT = 7.0


def apply_global_plot_style() -> None:
    plt.rcParams.update({
        "font.family": [ARIAL_FONT_FAMILY],
        "font.sans-serif": [ARIAL_FONT_FAMILY],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def enforce_figure_fonts(
    fig: plt.Figure,
    *,
    min_font_pt: float = MIN_FONT_PT,
    font_family: str = ARIAL_FONT_FAMILY,
) -> None:
    for text in fig.findobj(Text):
        try:
            current_size = float(text.get_fontsize())
        except (TypeError, ValueError):
            current_size = min_font_pt
        text.set_fontfamily(font_family)
        text.set_fontsize(max(min_font_pt, current_size))


apply_global_plot_style()


# ============================================================================
# USER EDIT ZONE
# ============================================================================
# 1) Default algorithm order on the x-axis.
#    You can change the order directly here, or override it from CLI by --algo-order.
DEFAULT_ALGO_ORDER: List[str] = [
    "pd",
    "ianus",
    "facil",
    "attacc",
    "attn_on_pim",
    "weights_on_pim",
    "hefthint",
]

# 2) Mapping from the name read from files -> name shown on the plot.
#    You can change it here, or override it from CLI by --name-map.
DEFAULT_DISPLAY_NAME_MAP: Dict[str, str] = {
    "pd": "PD",
    "ianus": "IANUS",
    "facil": "Facil",
    "attacc": "AttAcc",
    "attn_on_pim": "Attn-on-PIM",
    "weights_on_pim": "Weights-on-PIM",
    "hefthint": "Bifocal (this work)",
}

# 3) Highlight these x tick labels (red + bold).
HIGHLIGHT_KEYS = {"hefthint"}

# 4) Manual exclude list (can also pass --exclude on CLI).
MANUAL_EXCLUDE: List[str] = []

# 5) For plotted label "hefthint", candidates on disk can be either "heft" or "hefthint".
HEFT_VARIANTS: List[str] = ["heft", "hefthint"]

# 6) Spacing knobs.
#    BAR_WIDTH controls how narrow each bar is.
BAR_WIDTH = 0.01

#    PAIR_SEP is the center-to-center distance between trace and verification bars of ONE algorithm.
#    If PAIR_SEP == BAR_WIDTH, the two bars just touch each other.
PAIR_SEP = BAR_WIDTH

#    GROUP_STEP is the center-to-center distance between neighboring algorithms.
#    If GROUP_STEP == BAR_WIDTH + PAIR_SEP, neighboring algorithms also just touch each other.
#    This is the parameter to change if you want more / less blank between algorithms.
GROUP_STEP = BAR_WIDTH + PAIR_SEP + 0.5 * BAR_WIDTH

# 7) No blank between subplots.
SUBPLOT_WSPACE = 0.03

# 8) Figure size. Width kept close to old 4-panel figure; height is about 1.5x taller.
DEFAULT_FIG_WIDTH = 20.0
DEFAULT_FIG_HEIGHT = 3.0
# ============================================================================


# Parse: <strategy>_<Lin>x<Lout>.merge_all.csv
FNAME_RE = re.compile(r"^(?P<algo>.+?)_(?P<lin>\d+)x(?P<lout>\d+)\.merge_all\.csv$")


@dataclass
class Metrics:
    algo_label: str
    source_variant: str
    csv_path: Path
    n_rows: int

    trace_prefill: float
    trace_decode: float
    trace_total: float

    meas_prefill: float
    meas_decode: float
    meas_total: float


def _read_text_lines(p: Path) -> List[str]:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    lines: List[str] = []
    for raw in txt.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def resolve_file_list_paths(file_list: Path, root: Optional[Path] = None) -> List[Path]:
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
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required)
    if df.empty:
        raise ValueError(f"{csv_path} has no valid rows after dropping NaNs")

    means = df[required].mean(numeric_only=True)
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
    )


def pick_best_by_trace_total(paths: List[Path], algo_label: str, source_variant: str) -> Metrics:
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
    return sorted(idx.keys(), key=lambda t: (t[0], t[1]))


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


def _normalize_strategy_token(s: str) -> str:
    raw = (s or "").strip().lower()
    compact = re.sub(r"[\s_\-]+", "", raw)
    if compact in {"thiswork", "bifocal(thiswork)", "bifocal", "hefthint", "heft"}:
        return "hefthint"
    return raw


def parse_exclude_arg(exclude_arg: Optional[str]) -> List[str]:
    if not exclude_arg:
        return []
    parts = [p.strip() for p in exclude_arg.split(",")]
    return [p for p in parts if p]


def parse_algo_order_arg(algo_order_arg: Optional[str]) -> List[str]:
    if not algo_order_arg:
        return list(DEFAULT_ALGO_ORDER)
    parts = [p.strip() for p in algo_order_arg.split(",")]
    return [_normalize_strategy_token(p) for p in parts if p.strip()]


def parse_name_map_arg(name_map_arg: Optional[str]) -> Dict[str, str]:
    out = dict(DEFAULT_DISPLAY_NAME_MAP)
    if not name_map_arg:
        return out
    for item in name_map_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --name-map item '{item}', expected key=value")
        k, v = item.split("=", 1)
        out[_normalize_strategy_token(k)] = v.strip()
    return out


def build_dim_results(
    idx: Dict[Tuple[int, int], Dict[str, List[Path]]],
    dims: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict[str, Metrics]]:
    out: Dict[Tuple[int, int], Dict[str, Metrics]] = {}

    for dim in dims:
        lin, lout = dim
        dim_entry = idx.get(dim, {})
        res: Dict[str, Metrics] = {}

        # Normal strategies (except heft/hefthint)
        for algo_on_disk, paths in dim_entry.items():
            if algo_on_disk in HEFT_VARIANTS:
                continue
            try:
                res[algo_on_disk] = pick_best_by_trace_total(paths, algo_label=algo_on_disk, source_variant=algo_on_disk)
            except Exception as e:
                print(f"[WARN] {lin}x{lout}: cannot load '{algo_on_disk}': {e}", file=sys.stderr)

        # Merge heft / hefthint into plotted key "hefthint"
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
            print(
                f"[INFO] {lin}x{lout}: choose '{best.source_variant}' for plotted label 'hefthint' "
                f"(mean trace_total_s={best.trace_total:.6g}) from {best.csv_path}"
            )

        out[dim] = res

    return out


def strategy_order(res: Dict[str, Metrics], preferred_order: List[str]) -> List[str]:
    present = list(res.keys())
    preferred = [_normalize_strategy_token(s) for s in preferred_order]
    order: List[str] = [s for s in preferred if s in res]
    rest = sorted([s for s in present if s not in set(order)])
    order.extend(rest)
    return order


def pretty_strategy_name(strategy_key: str, display_name_map: Dict[str, str]) -> str:
    return display_name_map.get(strategy_key, strategy_key)


def format_speedup(sp: float) -> str:
    if not np.isfinite(sp):
        return ""
    if sp >= 100:
        return f"{sp:.0f}×"
    if sp >= 10:
        return f"{sp:.1f}×"
    return f"{sp:.2f}×"


def add_dim_separators(fig: plt.Figure, axes: np.ndarray, color: str = "#dddddd", lw: float = 0.8) -> None:
    if len(axes) <= 1:
        return
    for i in range(len(axes) - 1):
        p0 = axes[i].get_position()
        p1 = axes[i + 1].get_position()
        x = (p0.x1 + p1.x0) * 0.5
        y0 = min(p0.y0, p1.y0)
        y1 = max(p0.y1, p1.y1)
        fig.add_artist(
            Line2D(
                [x, x], [y0, y1],
                transform=fig.transFigure,
                linestyle="--",
                linewidth=lw,
                color=color,
                zorder=20,
            )
        )


def plot_results(
    dim_results: Dict[Tuple[int, int], Dict[str, Metrics]],
    dims: List[Tuple[int, int]],
    output: Path,
    title: Optional[str] = None,
    dpi: int = 200,
    border_lw: float = 0.7,
    max_panels: int = 8,
    algo_order: Optional[List[str]] = None,
    display_name_map: Optional[Dict[str, str]] = None,
    fig_width: float = DEFAULT_FIG_WIDTH,
    fig_height: float = DEFAULT_FIG_HEIGHT,
) -> None:
    deep_green = "#39a937"
    light_green = "#aee4ad"
    deep_blue = "#3760a9"
    light_blue = "#add9e4"

    trace_sp_color = "#8c8c8c"
    meas_sp_color = "k"

    algo_order = list(DEFAULT_ALGO_ORDER if algo_order is None else algo_order)
    display_name_map = dict(DEFAULT_DISPLAY_NAME_MAP if display_name_map is None else display_name_map)

    dims_to_plot = dims[:max_panels]

    max_time = 0.0
    max_speedup = 1.0
    width_ratios: List[float] = []

    for dim in dims_to_plot:
        res = dim_results.get(dim, {})
        if not res:
            width_ratios.append(1.0)
            continue
        order = strategy_order(res, algo_order)
        width_ratios.append(float(max(1, len(order))))
        for m in res.values():
            max_time = max(max_time, m.trace_total, m.meas_total)

        pd_m = res.get("pd")
        if pd_m is not None and pd_m.trace_total > 0 and pd_m.meas_total > 0:
            for m in res.values():
                if m.trace_total > 0:
                    max_speedup = max(max_speedup, pd_m.trace_total / m.trace_total)
                if m.meas_total > 0:
                    max_speedup = max(max_speedup, pd_m.meas_total / m.meas_total)

    if max_time <= 0:
        max_time = 1.0

    y_max = max_time * 1.02
    sp_max = max(1.2, max_speedup * 1.02)

    fig, axes = plt.subplots(
        1,
        len(dims_to_plot),
        sharey=True,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": SUBPLOT_WSPACE, "width_ratios": width_ratios},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    cluster_half_width = (PAIR_SEP + BAR_WIDTH) / 2.0

    for i, ax in enumerate(axes):
        for sp in ax.spines.values():
            sp.set_linewidth(border_lw)
            sp.set_color("#bfbfbf")
        ax.tick_params(width=max(border_lw * 0.8, 0.5), length=3)
        ax.tick_params(axis="x", pad=1)

        lin, lout = dims_to_plot[i]
        res = dim_results.get((lin, lout), {})

        if not res:
            ax.set_title(f"{lin}×{lout}\n(no data)", fontsize=12)
            ax.set_xticks([])
            ax.set_ylim(0, y_max)
            continue

        order = strategy_order(res, algo_order)
        n = len(order)
        x = np.arange(n, dtype=float) * GROUP_STEP
        pos_trace = x - PAIR_SEP / 2.0
        pos_meas = x + PAIR_SEP / 2.0

        trace_prefill = np.array([res[a].trace_prefill for a in order], dtype=float)
        trace_decode = np.array([res[a].trace_decode for a in order], dtype=float)
        meas_prefill = np.array([res[a].meas_prefill for a in order], dtype=float)
        meas_decode = np.array([res[a].meas_decode for a in order], dtype=float)

        ax.bar(pos_trace, trace_prefill, width=BAR_WIDTH, color=deep_green, edgecolor="none", linewidth=0, zorder=2)
        ax.bar(pos_trace, trace_decode, width=BAR_WIDTH, bottom=trace_prefill, color=light_green, edgecolor="none", linewidth=0, zorder=2)
        ax.bar(pos_meas, meas_prefill, width=BAR_WIDTH, color=deep_blue, edgecolor="none", linewidth=0, zorder=2)
        ax.bar(pos_meas, meas_decode, width=BAR_WIDTH, bottom=meas_prefill, color=light_blue, edgecolor="none", linewidth=0, zorder=2)

        trace_stack_total = trace_prefill + trace_decode
        meas_stack_total = meas_prefill + meas_decode
        for px, h in zip(pos_trace, trace_stack_total):
            ax.add_patch(Rectangle((px - BAR_WIDTH / 2, 0), BAR_WIDTH, h, fill=False, edgecolor="k", linewidth=border_lw, zorder=4))
        for px, h in zip(pos_meas, meas_stack_total):
            ax.add_patch(Rectangle((px - BAR_WIDTH / 2, 0), BAR_WIDTH, h, fill=False, edgecolor="k", linewidth=border_lw, zorder=4))

        # ax.set_title(f"{lin}×{lout}", fontsize=12, pad=8)
        ax.text(
            0.03, 0.97, f"{lin}×{lout}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [pretty_strategy_name(a, display_name_map) for a in order],
            rotation=45,
            ha="center",
            va="top",
            fontsize=11,
        )
        ax.set_ylim(0, y_max)
        ax.set_xlim(x[0] - cluster_half_width, x[-1] + cluster_half_width)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.30)

        if i == 0:
            ax.set_ylabel("Latency (s)", fontsize=12)

        for tick_label, strategy_key in zip(ax.get_xticklabels(), order):
            if strategy_key in HIGHLIGHT_KEYS:
                tick_label.set_color("blue")
                # tick_label.set_fontweight("bold")

        ax2 = ax.twinx()
        for sp in ax2.spines.values():
            sp.set_linewidth(border_lw)
            sp.set_color("#bfbfbf")
        ax2.tick_params(width=max(border_lw * 0.8, 0.5), length=3)

        pd_m = res.get("pd")
        if pd_m and pd_m.trace_total > 0:
            speedup_trace = np.array([pd_m.trace_total / res[a].trace_total if res[a].trace_total > 0 else np.nan for a in order], dtype=float)
        else:
            speedup_trace = np.full(n, np.nan, dtype=float)

        if pd_m and pd_m.meas_total > 0:
            speedup_meas = np.array([pd_m.meas_total / res[a].meas_total if res[a].meas_total > 0 else np.nan for a in order], dtype=float)
        else:
            speedup_meas = np.full(n, np.nan, dtype=float)

        ax2.plot(x, speedup_trace, color=trace_sp_color, lw=0.8, marker="o", markersize=3.2, zorder=10)
        ax2.plot(x, speedup_meas, color=meas_sp_color, lw=0.8, linestyle="-", marker="s", markersize=2.9, zorder=10)
        ax2.set_ylim(0, sp_max)

        for j, sp in enumerate(speedup_trace):
            if np.isfinite(sp):
                ax2.annotate(format_speedup(float(sp)), xy=(x[j], float(sp)), textcoords="offset points", xytext=(0, -8), ha="center", va="top", fontsize=9, color=trace_sp_color, zorder=11)
        for j, sp in enumerate(speedup_meas):
            if np.isfinite(sp):
                ax2.annotate(format_speedup(float(sp)), xy=(x[j], float(sp)), textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=9, color=meas_sp_color, zorder=11)

        if i == len(dims_to_plot) - 1:
            ax2.set_ylabel("Speedup (vs pd)", fontsize=12)
        else:
            ax2.set_yticks([])
            ax2.set_ylabel("")

    if title:
        fig.suptitle(title, y=0.985, fontsize=13)

    handles = [
        Patch(facecolor=deep_green, label="Simulated prefill"),
        Patch(facecolor=light_green, label="Simulated decode"),
        Patch(facecolor=deep_blue, label="Verification prefill"),
        Patch(facecolor=light_blue, label="Verification decode"),
        Line2D([0], [0], color=trace_sp_color, lw=0.8, marker="o", markersize=4, label="Speedup (trace total, vs pd)"),
        Line2D([0], [0], color=meas_sp_color, lw=0.8, linestyle="-", marker="s", markersize=4, label="Speedup (verification total, vs pd)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.94), fontsize=11, handlelength=1.6, columnspacing=1.0)

    enforce_figure_fonts(fig)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.31, top=0.80, wspace=SUBPLOT_WSPACE)
    add_dim_separators(fig, axes)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"[OK] saved figure -> {output.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-list", type=str, default=None, help="Path to a text file listing result files (recommended).")
    ap.add_argument("--root", type=str, default=None, help="Optional root directory to resolve relative paths in file-list.")
    ap.add_argument("--search-dir", type=str, default=None, help="If --file-list is not provided, recursively search this directory for *.merge_all.csv")
    ap.add_argument("--dims", type=str, default=None, help="Comma-separated dims to plot, like: 128x128,128x256,...")
    ap.add_argument("--output", type=str, default="plot_latency.png", help="Output image path.")
    ap.add_argument("--title", type=str, default=None, help="Figure title.")
    ap.add_argument("--exclude", type=str, default=None, help="Comma-separated strategy names to exclude from plotting.")
    ap.add_argument("--algo-order", type=str, default=None, help="Comma-separated x-axis order, e.g. 'pd,ianus,facil,hefthint'")
    ap.add_argument("--name-map", type=str, default=None, help="Comma-separated display-name map, e.g. 'pd=PD,hefthint=Bifocal (this work)'")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--border-lw", type=float, default=0.7, help="Bar outline / spine linewidth.")
    ap.add_argument("--max-panels", type=int, default=8, help="Maximum number of dims to show in one row.")
    ap.add_argument("--fig-width", type=float, default=DEFAULT_FIG_WIDTH)
    ap.add_argument("--fig-height", type=float, default=DEFAULT_FIG_HEIGHT)
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
    dims = parse_dims_arg(args.dims) if args.dims else infer_dims(idx)

    if len(dims) > args.max_panels:
        print(f"[WARN] {len(dims)} dims provided; only the first {args.max_panels} will be shown in the latency plot.", file=sys.stderr)
        dims = dims[: args.max_panels]

    dim_results = build_dim_results(idx, dims)

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

    algo_order = parse_algo_order_arg(args.algo_order)
    display_name_map = parse_name_map_arg(args.name_map)

    plot_results(
        dim_results=dim_results,
        dims=dims,
        output=Path(args.output),
        title=args.title,
        dpi=int(args.dpi),
        border_lw=float(args.border_lw),
        max_panels=int(args.max_panels),
        algo_order=algo_order,
        display_name_map=display_name_map,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()