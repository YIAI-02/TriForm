#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python plot_exp3_layout_compare.py \
  --csv ../../algorithms/output/ws_0p015_compare_adjust_strategy/results.csv \
  --outdir ../../figs/exp3/compare \
  --fig-format pdf \
  --share-y
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


# Keep the same font fallback behavior as the current script.
_available_font_names = {f.name for f in font_manager.fontManager.ttflist}
if "Arial" in _available_font_names:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
else:
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# Five fixed bars, in the requested order.
# Last column uses the HEFT-hint dual best result.
BAR_SPECS = [
    ("pd_linear_initial", "PD\nlinear", "#bdade4"),
    ("pd_dual_copy_initial", "PD\ndual", "#e4adb5"),
    ("nd_initial", "ND\ninit", "#aee4ad"),
    ("nd_best", "ND\nbest", "#add9e4"),
    ("hefthint_dual_copy_best", "HEFT\ndual\nbest", "#e4ddad"),
]

REQUIRED_COLUMNS = [
    "model",
    "batch",
    "prefill_len",
    "decode_len",
    *(col for col, _, _ in BAR_SPECS),
]


# Visually thinner bars, but bars still touch each other.
BAR_WIDTH = 0.46
BAR_STEP = BAR_WIDTH  # no gap between neighboring bars
SIDE_PADDING = 0.92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-(model, batch) multi-panel figures from results.csv. "
            "Each panel is one (prefill_len, decode_len) pair, and each panel "
            "contains five touching bars: PD linear, PD dual, ND init, ND best, "
            "and HEFT dual best."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to results.csv")
    parser.add_argument("--outdir", default="plots_pd_nd", help="Output directory")
    parser.add_argument(
        "--fig-format",
        nargs="+",
        default=["png"],
        help="Output format(s), e.g. png pdf",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    parser.add_argument(
        "--ncols",
        default="auto",
        help="Number of columns per figure. Use auto or a positive integer.",
    )
    parser.add_argument(
        "--share-y",
        action="store_true",
        help="Share the y-axis range across all panels in the same figure.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def format_length_value(value: object) -> str:
    try:
        value_f = float(value)
        if np.isfinite(value_f) and value_f.is_integer():
            return str(int(value_f))
        if np.isfinite(value_f):
            return f"{value_f:g}"
    except Exception:
        pass
    return str(value)


def format_seconds(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) < 1.0:
        return f"{value:.3f}s"
    return f"{value:.2f}s"


def format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.1f}%"


def auto_ncols(n_panels: int) -> int:
    if n_panels <= 3:
        return n_panels
    if n_panels <= 6:
        return 3
    if n_panels <= 8:
        return 4
    return 5


def resolve_ncols(arg_value: str, n_panels: int) -> int:
    if str(arg_value).strip().lower() == "auto":
        return auto_ncols(n_panels)
    ncols = int(arg_value)
    if ncols <= 0:
        raise ValueError("--ncols must be auto or a positive integer")
    return ncols


def compute_ylim(max_value: float) -> Tuple[float, float]:
    if not np.isfinite(max_value) or max_value <= 0:
        return (0.0, 1.0)
    return (0.0, max_value * 1.24)


def compute_nd_best_reduction_pct(nd_init: float, nd_best: float) -> float:
    if not np.isfinite(nd_init) or nd_init <= 0 or not np.isfinite(nd_best):
        return float("nan")
    return (nd_init - nd_best) / nd_init * 100.0


def add_value_labels(ax: plt.Axes, bars, values: Sequence[float], fontsize: float = 8.1) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.018
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + offset,
            format_seconds(float(value)),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="0.15",
        )


def make_panel(ax: plt.Axes, row: pd.Series, y_lim: Tuple[float, float] | None) -> None:
    labels = [label for _, label, _ in BAR_SPECS]
    values = np.array([float(row[col]) for col, _, _ in BAR_SPECS], dtype=float)
    colors = [color for _, _, color in BAR_SPECS]

    x = np.arange(len(labels), dtype=float) * BAR_STEP
    bars = ax.bar(
        x,
        values,
        width=BAR_WIDTH,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
        align="center",
    )

    if y_lim is None:
        ax.set_ylim(*compute_ylim(float(np.nanmax(values))))
    else:
        ax.set_ylim(*y_lim)

    add_value_labels(ax, bars, values)

    nd_reduction_pct = compute_nd_best_reduction_pct(
        nd_init=float(row["nd_initial"]),
        nd_best=float(row["nd_best"]),
    )
    ax.text(
        0.98,
        0.97,
        f"ND best reduction: {format_percent(nd_reduction_pct)}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8.4,
        color="0.12",
        bbox={
            "facecolor": "white",
            "edgecolor": "0.85",
            "linewidth": 0.6,
            "alpha": 0.95,
            "pad": 0.28,
        },
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.tick_params(axis="x", pad=2.0, length=0)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    left_edge = x[0] - BAR_WIDTH / 2.0
    right_edge = x[-1] + BAR_WIDTH / 2.0
    ax.set_xlim(left_edge - SIDE_PADDING, right_edge + SIDE_PADDING)

    ax.set_title(
        f"Prefill={format_length_value(row['prefill_len'])}, Decode={format_length_value(row['decode_len'])}",
        fontsize=10.5,
        pad=6,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_figure_title(group_key: Tuple, group_cols: Sequence[str]) -> str:
    kv = dict(zip(group_cols, group_key))
    parts: List[str] = []
    if "model" in kv:
        parts.append(str(kv["model"]))
    if "dtype" in kv:
        parts.append(f"dtype={kv['dtype']}")
    if "batch" in kv:
        parts.append(f"batch={kv['batch']}")
    return " | ".join(parts)


def make_figure(
    group_df: pd.DataFrame,
    group_key: Tuple,
    group_cols: Sequence[str],
    outdir: Path,
    fig_formats: Sequence[str],
    dpi: int,
    ncols_arg: str,
    share_y: bool,
) -> List[Path]:
    n_panels = len(group_df)
    ncols = resolve_ncols(ncols_arg, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig_w = max(4.6 * ncols, 9.2)
    fig_h = max(3.8 * nrows + 0.9, 5.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = np.array(axes).reshape(-1)

    shared_y_lim = None
    if share_y:
        max_value = float(group_df[[col for col, _, _ in BAR_SPECS]].to_numpy(dtype=float).max())
        shared_y_lim = compute_ylim(max_value)

    for ax, (_, row) in zip(axes, group_df.iterrows()):
        make_panel(ax, row, shared_y_lim)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(build_figure_title(group_key, group_cols), fontsize=14.2, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.14, wspace=0.26, hspace=0.36)

    stem = sanitize_filename(build_figure_title(group_key, group_cols))
    out_paths: List[Path] = []
    for fmt in fig_formats:
        fmt = str(fmt).lower().lstrip(".")
        out_path = outdir / f"{stem}_pd_nd_heft5.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(csv_path)
    require_columns(df, REQUIRED_COLUMNS)

    numeric_cols = ["batch", "prefill_len", "decode_len", *(col for col, _, _ in BAR_SPECS)]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(by=["model", "dtype", "batch", "prefill_len", "decode_len"]).reset_index(drop=True)

    group_cols = [col for col in ["model", "dtype", "batch"] if col in df.columns]
    saved_paths: List[Path] = []
    for group_key, group_df in df.groupby(group_cols, sort=False):
        group_df = group_df.reset_index(drop=True)
        saved_paths.extend(
            make_figure(
                group_df=group_df,
                group_key=group_key if isinstance(group_key, tuple) else (group_key,),
                group_cols=group_cols,
                outdir=outdir,
                fig_formats=args.fig_format,
                dpi=args.dpi,
                ncols_arg=str(args.ncols),
                share_y=bool(args.share_y),
            )
        )

    print(f"Saved {len(saved_paths)} figure(s) to: {outdir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
