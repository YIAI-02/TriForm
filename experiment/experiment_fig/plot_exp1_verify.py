#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot merged trace-versus-reference latency comparisons from ``*.merge_all.csv`` files.

Examples
--------
Create a file list first::

    find "$(pwd)" -name "*.merge_all.csv" | sort > files.txt

Single model::

    python3 plot_exp1_verify.py \
      --file-list ../../verify/llama_7b_fp16_b16_s2/files.txt \
      --algo-order "PD,AF,PD+FFN,PD+Linear,PD+Attn,HEFT,Bifocal" \
      --dims "128x128,128x512,128x1024,1024x128,1024x512,1024x1024" \
      --name-map "PD=PD,AF=AF,PD+FFN=PD+FFN,PD+Linear=PD+Linear,PD+Attn=PD+Attn,HEFT=HEFT,Bifocal=Bifocal" \
      --output ../../figs/exp1/verify/llama_7b_fp16_b16_s2.pdf

Multiple models::

    python3 plot_exp1_verify.py \
      --file-list ../../verify/llama_7b_fp16_b16_s2/files.txt \
      --file-list ../../verify/qwen_1.8b_fp16_b8_s2/files.txt \
      --model-label "Qwen-1.8B" \
      --model-label "Llama-7B" \
      --algo-order "PD,AF,PD+FFN,PD+Linear,PD+Attn,HEFT,Bifocal" \
      --dims "128x128,128x512,128x1024,1024x128,1024x512,1024x1024" \
      --name-map "PD=PD,AF=AF,PD+FFN=PD+FFN,PD+Linear=PD+Linear,PD+Attn=PD+Attn,HEFT=HEFT,Bifocal=Bifocal" \
      --output ../../figs/exp1/verify/exp1_verify.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
        # "font.family": [ARIAL_FONT_FAMILY],
        # "font.sans-serif": [ARIAL_FONT_FAMILY],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.default": "regular",
        "mathtext.fontset": "dejavusans",
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
DEFAULT_ALGO_ORDER: List[str] = [
    "PD",
    "AF",
    "PD+FFN",
    "PD+Linear",
    "PD+Attn",
    "HEFT",
    "Bifocal",
]

# 2) Mapping from the name read from files -> name shown on the plot.
DEFAULT_DISPLAY_NAME_MAP: Dict[str, str] = {
    "PD": "PD",
    "PD+FFN": "PD+FFN",
    "PD+Linear": "PD+Linear",
    "PD+Attn": "PD+Attn",
    "AF": "AF",
    "HEFT": "HEFT",
    "Bifocal": "Bifocal",
}

# 3) Highlight these x tick labels (red + bold).
HIGHLIGHT_KEYS = {"Bifocal"}

# 4) Manual exclude list (can also pass --exclude on CLI).
MANUAL_EXCLUDE: List[str] = []

# 5) Leave this list empty to plot schedulers independently.
SCHEDULER_VARIANTS: List[str] = []

# 6) Spacing knobs.
BAR_WIDTH = 0.004
PAIR_SEP = BAR_WIDTH
GROUP_STEP = 0.011
INTER_GROUP_GAP = GROUP_STEP - (BAR_WIDTH + PAIR_SEP)
PANEL_LEFT_MARGIN = 0.002
PANEL_RIGHT_MARGIN = 0.002

# 7) Subplot spacing.
SUBPLOT_WSPACE = 0.05
SUBPLOT_HSPACE = 0.22

# 8) Font sizing.
FONT_SCALE = 2.30
TICK_FONT_PT = 11.0 * FONT_SCALE
AXIS_LABEL_FONT_PT = 12.0 * FONT_SCALE
DIM_LABEL_FONT_PT = 10.5 * FONT_SCALE
LEGEND_FONT_PT = 9.5 * FONT_SCALE
SPEEDUP_TEXT_FONT_PT = 7.6 * FONT_SCALE
TITLE_FONT_PT = 13.0 * FONT_SCALE
ROW_LABEL_FONT_PT = 11.5 * FONT_SCALE
NO_DATA_FONT_PT = 11.5 * FONT_SCALE

# 9) Automatic figure sizing.
DEFAULT_PANEL_WIDTH = 5.2
DEFAULT_ROW_HEIGHT = 4.5
DEFAULT_MIN_FIG_WIDTH = 32.0
LEGEND_EXTRA_HEIGHT = 1.6
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
        algo = _normalize_strategy_token(m.group("algo"))
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



def infer_dims_from_all(indices: Sequence[Dict[Tuple[int, int], Dict[str, List[Path]]]]) -> List[Tuple[int, int]]:
    all_dims = set()
    for idx in indices:
        all_dims.update(idx.keys())
    return sorted(all_dims, key=lambda t: (t[0], t[1]))



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
    if raw.endswith("_linear"):
        raw = raw[: -len("_linear")]

    compact = re.sub(r"[\s_\-]+", "", raw)
    if compact in {"thiswork", "bifocal(thiswork)", "bifocal", "Bifocal", "HEFT"}:
        return "Bifocal"
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

        for algo_on_disk, paths in dim_entry.items():
            try:
                res[algo_on_disk] = pick_best_by_trace_total(paths, algo_label=algo_on_disk, source_variant=algo_on_disk)
            except Exception as exc:
                print(f"[WARN] {lin}x{lout}: cannot load '{algo_on_disk}': {exc}", file=sys.stderr)

        out[dim] = res

    return out


def strategy_order_from_keys(strategy_keys: Sequence[str], preferred_order: List[str]) -> List[str]:
    present = list(dict.fromkeys(strategy_keys))
    preferred = [_normalize_strategy_token(s) for s in preferred_order]
    order: List[str] = [s for s in preferred if s in present]
    rest = sorted([s for s in present if s not in set(order)])
    order.extend(rest)
    return order



def strategy_order(res: Dict[str, Metrics], preferred_order: List[str]) -> List[str]:
    return strategy_order_from_keys(list(res.keys()), preferred_order)



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



def format_dim_label(lin: int, lout: int) -> str:
    return rf"$L_{{\mathrm{{in}}}} = {lin},\ L_{{\mathrm{{out}}}} = {lout}$"



def set_panel_dim_label(ax: plt.Axes, lin: int, lout: int) -> None:
    ax.text(
        0.5,
        1.008,
        format_dim_label(lin, lout),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=DIM_LABEL_FONT_PT,
        clip_on=False,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.95,
            boxstyle="square,pad=0.08",
        ),
        zorder=15,
    )



def add_dim_separators(fig: plt.Figure, axes: np.ndarray, color: str = "#dddddd", lw: float = 0.8) -> None:
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    nrows, ncols = axes.shape
    if ncols <= 1:
        return

    for c in range(ncols - 1):
        p0 = axes[0, c].get_position()
        p1 = axes[0, c + 1].get_position()
        x = (p0.x1 + p1.x0) * 0.5
        y0 = min(axes[r, c].get_position().y0 for r in range(nrows))
        y1 = max(axes[r, c].get_position().y1 for r in range(nrows))
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



def _compute_row_limits(
    dim_results: Dict[Tuple[int, int], Dict[str, Metrics]],
    dims_to_plot: Sequence[Tuple[int, int]],
) -> Tuple[float, float]:
    max_time = 0.0
    max_speedup = 1.0

    for dim in dims_to_plot:
        res = dim_results.get(dim, {})
        for m in res.values():
            max_time = max(max_time, m.trace_total, m.meas_total)

        pd_m = res.get("PD")
        if pd_m is not None:
            if pd_m.trace_total > 0:
                for m in res.values():
                    if m.trace_total > 0:
                        max_speedup = max(max_speedup, pd_m.trace_total / m.trace_total)
            if pd_m.meas_total > 0:
                for m in res.values():
                    if m.meas_total > 0:
                        max_speedup = max(max_speedup, pd_m.meas_total / m.meas_total)

    if max_time <= 0:
        max_time = 1.0

    # Keep only a small headroom above the tallest bar so the speedup text stays
    # close to the centered L_in / L_out label instead of leaving a large empty band.
    y_max = max_time + max(0.22 * max_time, 4.0)
    sp_max = max(1.2, max_speedup * 1.05)
    return y_max, sp_max



def _broadcast_list(
    values: Optional[List[str]],
    n_targets: int,
    *,
    fill_value: Optional[str] = None,
    arg_name: str,
) -> List[Optional[str]]:
    if n_targets <= 0:
        return []
    if not values:
        return [fill_value] * n_targets
    if len(values) == 1 and n_targets > 1:
        return [values[0]] * n_targets
    if len(values) != n_targets:
        raise ValueError(
            f"{arg_name} must be provided either once or exactly {n_targets} times; got {len(values)}."
        )
    return list(values)



def collect_csv_paths_from_args(
    *,
    file_list_arg: Optional[str],
    search_dir_arg: Optional[str],
    root_arg: Optional[str],
) -> List[Path]:
    csv_paths: List[Path] = []

    if file_list_arg:
        fl = Path(file_list_arg)
        if not fl.exists():
            raise FileNotFoundError(f"--file-list not found: {fl}")
        root = Path(root_arg).resolve() if root_arg else None
        paths = resolve_file_list_paths(fl, root=root)
        csv_paths = [p for p in paths if p.name.endswith(".merge_all.csv")]
    else:
        if not search_dir_arg:
            raise ValueError("Provide either --file-list or --search-dir")
        sd = Path(search_dir_arg)
        if not sd.exists():
            raise FileNotFoundError(f"--search-dir not found: {sd}")
        csv_paths = glob_merge_all(sd)

    return csv_paths



def plot_results(
    model_dim_results_list: List[Dict[Tuple[int, int], Dict[str, Metrics]]],
    dims: List[Tuple[int, int]],
    output: Path,
    title: Optional[str] = None,
    dpi: int = 200,
    border_lw: float = 0.7,
    max_panels: int = 8,
    algo_order: Optional[List[str]] = None,
    display_name_map: Optional[Dict[str, str]] = None,
    model_labels: Optional[List[str]] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
) -> None:
    deep_green = "#39a937"
    light_green = "#aee4ad"
    deep_blue = "#3760a9"
    light_blue = "#add9e4"

    trace_sp_color = "#8c8c8c"
    meas_sp_color = "k"
    trace_text_color = deep_green
    meas_text_color = deep_blue

    algo_order = list(DEFAULT_ALGO_ORDER if algo_order is None else algo_order)
    display_name_map = dict(DEFAULT_DISPLAY_NAME_MAP if display_name_map is None else display_name_map)

    dims_to_plot = dims[:max_panels]
    nrows = len(model_dim_results_list)
    ncols = len(dims_to_plot)

    if nrows <= 0:
        raise ValueError("No model data to plot.")
    if ncols <= 0:
        raise ValueError("No dimensions to plot.")

    model_labels = list(model_labels or [])
    if len(model_labels) < nrows:
        model_labels.extend([""] * (nrows - len(model_labels)))

    column_orders: List[List[str]] = []
    width_ratios: List[float] = []
    for dim in dims_to_plot:
        present_keys: List[str] = []
        seen = set()
        for dim_results in model_dim_results_list:
            for key in dim_results.get(dim, {}).keys():
                if key not in seen:
                    seen.add(key)
                    present_keys.append(key)
        col_order = strategy_order_from_keys(present_keys, algo_order)
        column_orders.append(col_order)
        width_ratios.append(float(max(1, len(col_order))))

    row_y_maxs: List[float] = []
    row_sp_maxs: List[float] = []
    for dim_results in model_dim_results_list:
        y_max, sp_max = _compute_row_limits(dim_results, dims_to_plot)
        row_y_maxs.append(y_max)
        row_sp_maxs.append(sp_max)

    auto_fig_width = max(DEFAULT_MIN_FIG_WIDTH, DEFAULT_PANEL_WIDTH * ncols)
    auto_fig_height = DEFAULT_ROW_HEIGHT * nrows + LEGEND_EXTRA_HEIGHT + (0.8 if title else 0.0)
    fig_width = auto_fig_width if fig_width is None else fig_width
    fig_height = auto_fig_height if fig_height is None else fig_height

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex="col",
        gridspec_kw={
            "wspace": SUBPLOT_WSPACE,
            "hspace": SUBPLOT_HSPACE,
            "width_ratios": width_ratios,
        },
    )

    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(max(1.8, border_lw * 2.2))

    cluster_half_width = (PAIR_SEP + BAR_WIDTH) / 2.0
    spine_lw = max(1.6, border_lw * 2.0)
    bar_outline_lw = max(1.4, border_lw * 1.9)
    tick_lw = max(1.3, spine_lw * 0.85)
    # Speedup annotations should sit directly above each bar center.
    label_dx = 0.0

    for row_idx in range(nrows):
        dim_results = model_dim_results_list[row_idx]
        row_label = (model_labels[row_idx] or "").strip()
        row_y_max = row_y_maxs[row_idx]
        row_sp_max = row_sp_maxs[row_idx]
        del row_sp_max

        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            lin, lout = dims_to_plot[col_idx]
            res = dim_results.get((lin, lout), {})
            global_order = column_orders[col_idx]

            if global_order:
                x_all = np.arange(len(global_order), dtype=float) * GROUP_STEP
                x_left = x_all[0] - cluster_half_width - PANEL_LEFT_MARGIN
                x_right = x_all[-1] + cluster_half_width + PANEL_RIGHT_MARGIN
            else:
                x_all = np.array([0.0], dtype=float)
                x_left, x_right = -0.5, 0.5

            for sp in ax.spines.values():
                sp.set_linewidth(spine_lw)
                sp.set_color("black")
            ax.tick_params(width=tick_lw, length=4.5, labelsize=TICK_FONT_PT, colors="black")
            ax.tick_params(axis="x", pad=1.5)
            ax.tick_params(axis="y", pad=1.5)
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
            ax.set_xlim(x_left, x_right)
            ax.set_ylim(0, row_y_max)
            set_panel_dim_label(ax, lin, lout)

            if global_order:
                ax.set_xticks(x_all)
                if row_idx == nrows - 1:
                    ax.set_xticklabels(
                        [pretty_strategy_name(a, display_name_map) for a in global_order],
                        rotation=45,
                        ha="right",
                        va="top",
                        rotation_mode="anchor",
                        fontsize=TICK_FONT_PT,
                    )
                    for tick_label, strategy_key in zip(ax.get_xticklabels(), global_order):
                        if strategy_key in HIGHLIGHT_KEYS:
                            tick_label.set_color("blue")
                else:
                    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            else:
                ax.set_xticks([])

            if col_idx == 0:
                ax.set_ylabel("Latency (s)", fontsize=AXIS_LABEL_FONT_PT)
                if row_label:
                    ax.text(
                        0.00,
                        1.10,
                        row_label,
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=ROW_LABEL_FONT_PT,
                        fontweight="bold",
                        clip_on=False,
                    )
            else:
                ax.tick_params(axis="y", labelleft=False)

            if not res:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=NO_DATA_FONT_PT,
                )
                continue

            present_indices = [i for i, a in enumerate(global_order) if a in res]
            order = [global_order[i] for i in present_indices]
            x = x_all[present_indices]
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
                ax.add_patch(
                    Rectangle(
                        (px - BAR_WIDTH / 2, 0),
                        BAR_WIDTH,
                        h,
                        fill=False,
                        edgecolor="black",
                        linewidth=bar_outline_lw,
                        zorder=4,
                    )
                )
            for px, h in zip(pos_meas, meas_stack_total):
                ax.add_patch(
                    Rectangle(
                        (px - BAR_WIDTH / 2, 0),
                        BAR_WIDTH,
                        h,
                        fill=False,
                        edgecolor="black",
                        linewidth=bar_outline_lw,
                        zorder=4,
                    )
                )

            pd_m = res.get("PD")
            if pd_m and pd_m.trace_total > 0:
                speedup_trace = np.array([
                    pd_m.trace_total / res[a].trace_total if res[a].trace_total > 0 else np.nan
                    for a in order
                ], dtype=float)
            else:
                speedup_trace = np.full(len(order), np.nan, dtype=float)

            if pd_m and pd_m.meas_total > 0:
                speedup_meas = np.array([
                    pd_m.meas_total / res[a].meas_total if res[a].meas_total > 0 else np.nan
                    for a in order
                ], dtype=float)
            else:
                speedup_meas = np.full(len(order), np.nan, dtype=float)

            label_pad = row_y_max * 0.0035
            for j, sp in enumerate(speedup_trace):
                if np.isfinite(sp):
                    ax.text(
                        pos_trace[j],
                        float(trace_stack_total[j] + label_pad),
                        format_speedup(float(sp)),
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=SPEEDUP_TEXT_FONT_PT,
                        color=trace_text_color,
                        zorder=11,
                        clip_on=True,
                    )
            for j, sp in enumerate(speedup_meas):
                if np.isfinite(sp):
                    ax.text(
                        pos_meas[j],
                        float(meas_stack_total[j] + label_pad),
                        format_speedup(float(sp)),
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=SPEEDUP_TEXT_FONT_PT,
                        color=meas_text_color,
                        zorder=11,
                        clip_on=True,
                    )

    if title:
        fig.suptitle(title, y=0.995, fontsize=TITLE_FONT_PT)

    handles = [
        Patch(facecolor=deep_green, label="Simulated prefill"),
        Patch(facecolor=light_green, label="Simulated decode"),
        Patch(facecolor=deep_blue, label="Verification prefill"),
        Patch(facecolor=light_blue, label="Verification decode"),
        Line2D([0], [0], color=trace_sp_color, linestyle="None", marker="o", markersize=8, label="Trace total speedup"),
        Line2D([0], [0], color=meas_sp_color, linestyle="None", marker="s", markersize=7, label="Verification total speedup"),
    ]

    legend_y = 0.955 if title else 0.985
    legend_ncol = 6 if fig_width >= 32 else 3
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, legend_y),
        fontsize=LEGEND_FONT_PT,
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=0.85,
        borderaxespad=0.0,
    )

    enforce_figure_fonts(fig, min_font_pt=max(MIN_FONT_PT, 10.0 * FONT_SCALE))

    top_margin = 0.82 if title else 0.85
    bottom_margin = 0.20 if nrows == 1 else 0.16
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=bottom_margin,
        top=top_margin,
        wspace=SUBPLOT_WSPACE,
        hspace=SUBPLOT_HSPACE,
    )
    add_dim_separators(fig, axes)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    print(f"[OK] saved figure -> {output.resolve()}")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file-list",
        type=str,
        action="append",
        default=None,
        help="Path to a text file listing result files. Pass multiple times for multiple model rows.",
    )
    ap.add_argument(
        "--root",
        type=str,
        action="append",
        default=None,
        help="Optional root directory to resolve relative paths in --file-list. May be passed once or one per --file-list.",
    )
    ap.add_argument(
        "--search-dir",
        type=str,
        action="append",
        default=None,
        help="If --file-list is not provided, recursively search this directory for *.merge_all.csv. Pass multiple times for multiple model rows.",
    )
    ap.add_argument(
        "--model-label",
        type=str,
        action="append",
        default=None,
        help="Optional label shown above the first panel of each row. Pass once or one per model.",
    )
    ap.add_argument("--dims", type=str, default=None, help="Comma-separated dims to plot, like: 128x128,128x256,...")
    ap.add_argument("--output", type=str, default="plot_latency.png", help="Output image path.")
    ap.add_argument("--title", type=str, default=None, help="Figure title.")
    ap.add_argument("--exclude", type=str, default=None, help="Comma-separated strategy names to exclude from plotting.")
    ap.add_argument("--algo-order", type=str, default=None, help="Comma-separated x-axis order, e.g. 'PD,AF,PD+FFN,PD+Linear,PD+Attn,HEFT,Bifocal'")
    ap.add_argument("--name-map", type=str, default=None, help="Comma-separated display-name map, e.g. 'PD=PD,Bifocal=Bifocal'")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--border-lw", type=float, default=0.7, help="Bar outline / spine linewidth.")
    ap.add_argument("--max-panels", type=int, default=8, help="Maximum number of dims to show in one row.")
    ap.add_argument("--fig-width", type=float, default=None, help="Total figure width. Default is automatic.")
    ap.add_argument("--fig-height", type=float, default=None, help="Total figure height. Default is automatic.")
    ap.add_argument("--show", action="store_true", help="Show interactively (in addition to saving).")

    args = ap.parse_args()

    if args.file_list and args.search_dir:
        ap.error("Use either one or more --file-list values, or one or more --search-dir values, not both.")
    if not args.file_list and not args.search_dir:
        ap.error("Provide at least one --file-list or --search-dir")

    if args.search_dir and args.root:
        ap.error("--root is only valid together with --file-list")

    model_sources: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
    if args.file_list:
        roots = _broadcast_list(args.root, len(args.file_list), fill_value=None, arg_name="--root")
        for file_list_arg, root_arg in zip(args.file_list, roots):
            model_sources.append((file_list_arg, None, root_arg))
    else:
        for search_dir_arg in args.search_dir or []:
            model_sources.append((None, search_dir_arg, None))

    model_labels = _broadcast_list(args.model_label, len(model_sources), fill_value="", arg_name="--model-label")

    all_indices: List[Dict[Tuple[int, int], Dict[str, List[Path]]]] = []
    for file_list_arg, search_dir_arg, root_arg in model_sources:
        try:
            csv_paths = collect_csv_paths_from_args(
                file_list_arg=file_list_arg,
                search_dir_arg=search_dir_arg,
                root_arg=root_arg,
            )
        except Exception as e:
            ap.error(str(e))

        if not csv_paths:
            source_desc = file_list_arg if file_list_arg else search_dir_arg
            ap.error(f"No *.merge_all.csv found for input: {source_desc}")

        all_indices.append(index_merge_all(csv_paths))

    dims = parse_dims_arg(args.dims) if args.dims else infer_dims_from_all(all_indices)

    if len(dims) > args.max_panels:
        print(
            f"[WARN] {len(dims)} dims provided; only the first {args.max_panels} will be shown in the latency plot.",
            file=sys.stderr,
        )
        dims = dims[: args.max_panels]

    exclude_tokens: List[str] = []
    exclude_tokens.extend(MANUAL_EXCLUDE)
    exclude_tokens.extend(parse_exclude_arg(args.exclude))
    exclude_set = {_normalize_strategy_token(t) for t in exclude_tokens if t and _normalize_strategy_token(t)}

    algo_order = parse_algo_order_arg(args.algo_order)
    display_name_map = parse_name_map_arg(args.name_map)

    model_dim_results_list: List[Dict[Tuple[int, int], Dict[str, Metrics]]] = []
    for idx in all_indices:
        dim_results = build_dim_results(idx, dims)
        if exclude_set:
            for dim in list(dim_results.keys()):
                res = dim_results.get(dim, {})
                for k in list(res.keys()):
                    if _normalize_strategy_token(k) in exclude_set:
                        res.pop(k, None)
        model_dim_results_list.append(dim_results)

    plot_results(
        model_dim_results_list=model_dim_results_list,
        dims=dims,
        output=Path(args.output),
        title=args.title,
        dpi=int(args.dpi),
        border_lw=float(args.border_lw),
        max_panels=int(args.max_panels),
        algo_order=algo_order,
        display_name_map=display_name_map,
        model_labels=[str(x or "") for x in model_labels],
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
