#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot summary tables into compact multi-panel comparison figures.

Supported modes
---------------
1) all_compare:
   ND init, NZ init, PIM-OPT init, ND opt
2) nd_only:
   ND init, ND opt
3) both:
   generate both of the above

Expected summary columns
------------------------
Required columns:
- model_family, model_variant, dtype, batch, prefill_len, decode_len
- nd_initial_prefill_s, nd_initial_decode_s, nd_initial_total_s
- nz_initial_prefill_s, nz_initial_decode_s
- pim_opt_initial_prefill_s, pim_opt_initial_decode_s
- nd_final_total_s (or one alias listed below)

Optional columns (used automatically if present):
- nd_final_prefill_s / nd_best_prefill_s / nd_optimized_prefill_s / nd_opt_prefill_s
- nd_final_decode_s / nd_best_decode_s / nd_optimized_decode_s / nd_opt_decode_s

Config file
-----------
Use --cfg with JSON or TOML.
This script accepts BOTH the new internal config names and the older aliases:

Old aliases that are still supported:
- layout_display_names  -> bar_label_map
- panel_label_map       -> layout_name_map
- panel_label_position  -> layout_name_mode
- subplot_spacing       -> figure.*
- margins               -> figure.*
- xtick_rotation        -> ticks.x_tick_rotation
- xtick_label_mode      -> ticks.x_tick_label_mode
- xtick_wrap_width      -> ticks.x_tick_wrap_width
- xtick_stagger_points  -> ticks.x_tick_stagger_points
- xtick_line_spacing    -> ticks.x_tick_linespacing
- legend_ncol           -> figure.legend_ncol

Useful x tick label modes:
- plain
- wrap
- stagger
- wrap_stagger

Examples
--------
python plot_summary_compare.py \
    --summary /path/to/summary.csv \
    --mode both \
    --cfg /path/to/plot_cfg.json \
    --outdir ./plots
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None


# Dark = prefill, light = decode for every group.
COLOR_MAP = {
    "nd_init": ("#3760a9", "#add9e4"),
    "nz_init": ("#39a937", "#aee4ad"),
    "pim_opt_init": ("#5837a8", "#bdade4"),
    "nd_opt": ("#a83747", "#e4adb5"),
}

DEFAULT_LABEL_MAP = {
    "nd_init": "ND init",
    "nz_init": "NZ init",
    "pim_opt_init": "PIM-OPT init",
    "nd_opt": "ND opt",
}

MODE_TO_BAR_KEYS = {
    "all_compare": ["nd_init", "nz_init", "pim_opt_init", "nd_opt"],
    "nd_only": ["nd_init", "nd_opt"],
}

OPT_PREFILL_ALIASES = [
    "nd_final_prefill_s",
    "nd_best_prefill_s",
    "nd_optimized_prefill_s",
    "nd_opt_prefill_s",
]
OPT_DECODE_ALIASES = [
    "nd_final_decode_s",
    "nd_best_decode_s",
    "nd_optimized_decode_s",
    "nd_opt_decode_s",
]
OPT_TOTAL_ALIASES = [
    "nd_final_total_s",
    "nd_best_total_s",
    "nd_optimized_total_s",
    "nd_opt_total_s",
]

BAR_KEY_ALIAS_MAP = {
    "nd_init": "nd_init",
    "ndinit": "nd_init",
    "nd_initial": "nd_init",
    "ndinitial": "nd_init",
    "nz_init": "nz_init",
    "nzinit": "nz_init",
    "nz_initial": "nz_init",
    "nzinitial": "nz_init",
    "pim_opt_init": "pim_opt_init",
    "pimoptinit": "pim_opt_init",
    "pim_optinitial": "pim_opt_init",
    "pim_opt_initial": "pim_opt_init",
    "piminit": "pim_opt_init",
    "nd_opt": "nd_opt",
    "ndopt": "nd_opt",
    "nd_final": "nd_opt",
    "ndfinal": "nd_opt",
    "nd_optimized": "nd_opt",
    "ndoptimized": "nd_opt",
}

DEFAULT_CFG = {
    "bar_label_map": {},
    "layout_name_map": {},
    # title / xlabel / both / none
    "layout_name_mode": "title",
    "panel": {
        "layout_name_fontsize": 10.2,
        "layout_name_labelpad": 5.8,
        "show_title_when_no_alias": True,
    },
    "ticks": {
        "x_tick_fontsize": 8.8,
        "x_tick_rotation": 0,
        "x_tick_label_mode": "wrap_stagger",
        "x_tick_wrap_width": 11,
        "x_tick_stagger_points": 7.0,
        "x_tick_linespacing": 0.92,
        "x_tick_pad": 1.4,
        "y_tick_fontsize": 8.5,
        "bar_annotation_fontsize": 7.8,
        "speedup_fontsize": 8.5,
    },
    "figure": {
        "panel_width": 3.08,
        "panel_height": 3.35,
        "header_height": 1.00,
        "min_fig_width": 12.0,
        "min_fig_height": 4.3,
        "outer_wspace": 0.05,
        "outer_hspace": 0.14,
        "inner_hspace": 0.01,
        "left": 0.042,
        "right": 0.998,
        "top": 0.842,
        "bottom": 0.095,
        "bottom_with_note": 0.118,
        "legend_y": 0.942,
        "suptitle_y": 0.988,
        "bar_xstep": 0.92,
        "bar_width": 0.78,
        "save_pad_inches": 0.03,
        "legend_ncol": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot summary comparison figures.")
    parser.add_argument("--summary", required=True, help="Path to summary CSV/TSV/XLSX file.")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["all_compare", "nd_only", "both"],
        help="Plot mode.",
    )
    parser.add_argument(
        "--outdir",
        default="plots",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--cfg",
        default=None,
        help="Optional JSON/TOML config file for spacing and naming.",
    )
    parser.add_argument(
        "--ncols",
        default="auto",
        help="Subplots per row: auto, 4, 5, or 6.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--fig-format",
        nargs="+",
        default=["png"],
        help="Output figure format(s), e.g. png pdf.",
    )
    parser.add_argument(
        "--share-y",
        action="store_true",
        help="Share bar-axis y-limits within one figure.",
    )
    parser.add_argument(
        "--nd-final-split-strategy",
        default="keep_prefill",
        choices=["keep_prefill", "proportional"],
        help="How to split ND optimized total into prefill/decode when split columns are missing.",
    )
    parser.add_argument(
        "--sort-by",
        nargs="+",
        default=["prefill_len", "decode_len"],
        help="Subplot sort keys inside each figure.",
    )
    return parser.parse_args()


def deep_update(base: Dict, updates: Dict) -> Dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported summary file type: {path.suffix}")


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def auto_ncols(n_panels: int) -> int:
    if n_panels <= 4:
        return 4
    if n_panels <= 5:
        return 5
    return 6


def resolve_ncols(arg_value: str, n_panels: int) -> int:
    if str(arg_value).lower() == "auto":
        return auto_ncols(n_panels)
    ncols = int(arg_value)
    if ncols < 4 or ncols > 6:
        raise ValueError("--ncols must be auto, 4, 5, or 6")
    return ncols


def normalize_bar_key(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())
    return BAR_KEY_ALIAS_MAP.get(cleaned, str(text).strip())


def canonical_layout_key(prefill_len: int, decode_len: int) -> str:
    return f"{int(prefill_len)}x{int(decode_len)}"


def normalize_layout_key(key: str) -> str:
    nums = re.findall(r"\d+", str(key))
    if len(nums) >= 2:
        return canonical_layout_key(int(nums[0]), int(nums[1]))
    return str(key).strip().lower()


def normalize_user_cfg_aliases(user_cfg: Dict) -> Dict:
    """Accept both the new internal names and the previously shared alias names."""
    cfg = copy.deepcopy(user_cfg)

    if "layout_display_names" in cfg and "bar_label_map" not in cfg:
        cfg["bar_label_map"] = copy.deepcopy(cfg["layout_display_names"])
    if "panel_label_map" in cfg and "layout_name_map" not in cfg:
        cfg["layout_name_map"] = copy.deepcopy(cfg["panel_label_map"])
    if "panel_label_position" in cfg and "layout_name_mode" not in cfg:
        cfg["layout_name_mode"] = cfg["panel_label_position"]

    if "subplot_spacing" in cfg:
        subplot_spacing = cfg.get("subplot_spacing") or {}
        figure_cfg = cfg.setdefault("figure", {})
        alias_map = {
            "panel_width": "panel_width",
            "panel_height": "panel_height",
            "wspace": "outer_wspace",
            "hspace": "outer_hspace",
            "inner_hspace": "inner_hspace",
        }
        for src_key, dst_key in alias_map.items():
            if src_key in subplot_spacing and dst_key not in figure_cfg:
                figure_cfg[dst_key] = subplot_spacing[src_key]

    if "margins" in cfg:
        margins = cfg.get("margins") or {}
        figure_cfg = cfg.setdefault("figure", {})
        for key in ["left", "right", "top", "bottom", "bottom_with_note"]:
            if key in margins and key not in figure_cfg:
                figure_cfg[key] = margins[key]

    tick_aliases = {
        "xtick_rotation": "x_tick_rotation",
        "xtick_label_mode": "x_tick_label_mode",
        "xtick_wrap_width": "x_tick_wrap_width",
        "xtick_stagger_points": "x_tick_stagger_points",
        "xtick_line_spacing": "x_tick_linespacing",
        "xtick_fontsize": "x_tick_fontsize",
        "ytick_fontsize": "y_tick_fontsize",
    }
    if any(key in cfg for key in tick_aliases):
        ticks_cfg = cfg.setdefault("ticks", {})
        for src_key, dst_key in tick_aliases.items():
            if src_key in cfg and dst_key not in ticks_cfg:
                ticks_cfg[dst_key] = cfg[src_key]

    if "legend_ncol" in cfg:
        figure_cfg = cfg.setdefault("figure", {})
        if "legend_ncol" not in figure_cfg:
            figure_cfg["legend_ncol"] = cfg["legend_ncol"]

    return cfg


def load_cfg(path: Optional[Path]) -> Dict:
    cfg = copy.deepcopy(DEFAULT_CFG)
    if path is None:
        return cfg

    suffix = path.suffix.lower()
    if suffix == ".json":
        user_cfg = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML config requires Python 3.11+ tomllib support.")
        user_cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("--cfg only supports .json or .toml")

    if not isinstance(user_cfg, dict):
        raise ValueError("Config root must be an object / table")

    user_cfg = normalize_user_cfg_aliases(user_cfg)
    return deep_update(cfg, user_cfg)


def resolve_bar_label_map(cfg: Dict) -> Dict[str, str]:
    label_map = dict(DEFAULT_LABEL_MAP)
    user_map = cfg.get("bar_label_map", {}) or {}
    for raw_key, label in user_map.items():
        canonical_key = normalize_bar_key(raw_key)
        if canonical_key in label_map:
            label_map[canonical_key] = str(label)
    return label_map


def resolve_layout_name_map(cfg: Dict) -> Dict[str, str]:
    result: Dict[str, str] = {}
    raw_map = cfg.get("layout_name_map", {}) or {}
    for raw_key, value in raw_map.items():
        result[normalize_layout_key(raw_key)] = str(value)
    return result


def get_layout_display_name(
    prefill_len: int,
    decode_len: int,
    layout_name_map: Dict[str, str],
    show_title_when_no_alias: bool,
) -> Optional[str]:
    key = canonical_layout_key(prefill_len, decode_len)
    if key in layout_name_map:
        return layout_name_map[key]
    if show_title_when_no_alias:
        return f"{int(prefill_len)}×{int(decode_len)}"
    return None


def get_nd_opt_split(
    row: pd.Series,
    opt_prefill_col: Optional[str],
    opt_decode_col: Optional[str],
    opt_total_col: str,
    strategy: str,
) -> Tuple[float, float, str]:
    if opt_prefill_col and opt_decode_col:
        prefill = float(row[opt_prefill_col])
        decode = float(row[opt_decode_col])
        return prefill, decode, "explicit"

    nd_init_prefill = float(row["nd_initial_prefill_s"])
    nd_init_decode = float(row["nd_initial_decode_s"])
    nd_init_total = float(row["nd_initial_total_s"])
    nd_opt_total = float(row[opt_total_col])

    if strategy == "keep_prefill":
        prefill = nd_init_prefill
        decode = max(0.0, nd_opt_total - prefill)
        return prefill, decode, "keep_prefill"

    if strategy == "proportional":
        if nd_init_total <= 0:
            return 0.0, 0.0, "proportional"
        scale = nd_opt_total / nd_init_total
        prefill = max(0.0, nd_init_prefill * scale)
        decode = max(0.0, nd_init_decode * scale)
        return prefill, decode, "proportional"

    raise ValueError(f"Unknown strategy: {strategy}")


def build_bar_payload(
    row: pd.Series,
    mode: str,
    opt_prefill_col: Optional[str],
    opt_decode_col: Optional[str],
    opt_total_col: str,
    nd_final_split_strategy: str,
    bar_label_map: Dict[str, str],
) -> List[Dict[str, float]]:
    payload: List[Dict[str, float]] = []
    for key in MODE_TO_BAR_KEYS[mode]:
        if key == "nd_init":
            prefill = float(row["nd_initial_prefill_s"])
            decode = float(row["nd_initial_decode_s"])
            total = float(row["nd_initial_total_s"])
        elif key == "nz_init":
            prefill = float(row["nz_initial_prefill_s"])
            decode = float(row["nz_initial_decode_s"])
            total = prefill + decode
        elif key == "pim_opt_init":
            prefill = float(row["pim_opt_initial_prefill_s"])
            decode = float(row["pim_opt_initial_decode_s"])
            total = prefill + decode
        elif key == "nd_opt":
            prefill, decode, _ = get_nd_opt_split(
                row=row,
                opt_prefill_col=opt_prefill_col,
                opt_decode_col=opt_decode_col,
                opt_total_col=opt_total_col,
                strategy=nd_final_split_strategy,
            )
            total = prefill + decode
        else:
            raise KeyError(f"Unknown bar key: {key}")

        prefill_color, decode_color = COLOR_MAP[key]
        payload.append(
            {
                "key": key,
                "label": bar_label_map.get(key, DEFAULT_LABEL_MAP[key]),
                "prefill": prefill,
                "decode": decode,
                "total": total,
                "prefill_color": prefill_color,
                "decode_color": decode_color,
            }
        )
    return payload


def get_speedup(row: pd.Series, opt_total_col: str) -> float:
    nd_init_total = float(row["nd_initial_total_s"])
    nd_opt_total = float(row[opt_total_col])
    if nd_opt_total <= 0:
        return float("nan")
    return nd_init_total / nd_opt_total


def format_seconds(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) < 1.0:
        return f"{value:.3f}s"
    return f"{value:.2f}s"


def format_delta_pct(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:+.1f}%"


def compute_bar_ylim(max_total: float) -> Tuple[float, float]:
    if max_total <= 0:
        return (0.0, 1.0)
    return (0.0, max_total * 1.28)


def compute_top_ylim(totals: np.ndarray) -> Tuple[float, float]:
    if len(totals) == 0:
        return (0.0, 1.0)
    tmin = float(np.min(totals))
    tmax = float(np.max(totals))
    span = tmax - tmin
    ref = max(abs(tmax), 1e-9)
    pad = max(span * 0.75, ref * 0.04, 0.003)
    low = max(0.0, tmin - pad)
    high = tmax + pad * 1.15
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return (0.0, 1.0)
    return (low, high)


def legend_handles(mode: str, bar_label_map: Dict[str, str]) -> List:
    handles: List = []
    for key in MODE_TO_BAR_KEYS[mode]:
        prefill_color, decode_color = COLOR_MAP[key]
        label = bar_label_map.get(key, DEFAULT_LABEL_MAP[key])
        handles.append(Patch(facecolor=prefill_color, edgecolor="black", label=f"{label} prefill"))
        handles.append(Patch(facecolor=decode_color, edgecolor="black", label=f"{label} decode"))
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linewidth=1.6,
            markersize=4,
            label="ND init → ND opt",
        )
    )
    return handles


def plot_zoomed_totals_strip(
    ax: plt.Axes,
    x: np.ndarray,
    totals: np.ndarray,
    speedup: float,
    xpad: float,
    panel_title: Optional[str],
    title_fontsize: float,
    speedup_fontsize: float,
) -> None:
    ax.set_xlim(x[0] - xpad, x[-1] + xpad)
    ax.set_ylim(*compute_top_ylim(totals))

    nd_init_total = float(totals[0])
    nd_opt_total = float(totals[-1])

    ax.plot(x, totals, color="0.78", linewidth=1.0, zorder=1)
    ax.scatter(x, totals, s=20, color="0.45", zorder=2)

    ax.axhline(nd_init_total, color="0.65", linestyle="--", linewidth=0.8, zorder=0)
    ax.plot(
        [x[0], x[-1]],
        [nd_init_total, nd_opt_total],
        color="black",
        marker="o",
        linewidth=1.7,
        markersize=4.6,
        zorder=3,
    )

    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * 0.08
    label_y = max(nd_init_total, nd_opt_total) + offset
    if np.isfinite(speedup):
        ax.text(
            (x[0] + x[-1]) / 2.0,
            label_y,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=speedup_fontsize,
            color="black",
        )
    else:
        ax.text(
            0.5,
            0.60,
            "speedup N/A",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=speedup_fontsize,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if panel_title:
        ax.set_title(panel_title, fontsize=title_fontsize, pad=1.0)


def annotate_bar_totals(
    ax: plt.Axes,
    x: np.ndarray,
    totals: np.ndarray,
    fontsize: float,
) -> None:
    if len(totals) == 0:
        return
    baseline = float(totals[0])
    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * 0.022

    for idx, (xi, total) in enumerate(zip(x, totals)):
        if baseline > 0 and np.isfinite(total):
            delta_pct = (total / baseline - 1.0) * 100.0
        else:
            delta_pct = float("nan")

        if idx == 0:
            text = f"{format_seconds(total)}\nbase"
        else:
            text = f"{format_seconds(total)}\n{format_delta_pct(delta_pct)}"

        ax.text(
            xi,
            total + offset,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="0.15",
        )


def draw_stack_boundary(ax: plt.Axes, x: np.ndarray, width: float, prefill_vals: np.ndarray) -> None:
    for xi, prefill in zip(x, prefill_vals):
        ax.hlines(
            prefill,
            xi - width / 2.0,
            xi + width / 2.0,
            color="white",
            linewidth=1.1,
            zorder=3,
        )


def wrap_tick_label(text: str, width: int) -> str:
    text = str(text)
    if width <= 0:
        return text
    parts = text.split("\n")
    wrapped_parts: List[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            wrapped_parts.append("")
            continue
        wrapped = textwrap.wrap(
            stripped,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_parts.append("\n".join(wrapped) if wrapped else stripped)
    return "\n".join(wrapped_parts)


def format_tick_labels(labels: Sequence[str], tick_cfg: Dict) -> List[str]:
    style = str(tick_cfg.get("x_tick_label_mode", "wrap_stagger")).strip().lower()
    wrap_width = int(tick_cfg.get("x_tick_wrap_width", 11))
    if style in {"wrap", "wrap_stagger", "auto"}:
        return [wrap_tick_label(label, wrap_width) for label in labels]
    return [str(label) for label in labels]


def apply_tick_label_transforms(ax: plt.Axes, fig: plt.Figure, tick_cfg: Dict) -> None:
    style = str(tick_cfg.get("x_tick_label_mode", "wrap_stagger")).strip().lower()
    if style not in {"stagger", "wrap_stagger", "auto"}:
        return
    stagger_points = float(tick_cfg.get("x_tick_stagger_points", 7.0))
    if stagger_points == 0:
        return
    labels = ax.get_xticklabels()
    for idx, label in enumerate(labels):
        if idx % 2 == 1:
            offset = mtransforms.ScaledTranslation(0, -stagger_points / 72.0, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)


def set_bar_xticklabels(ax: plt.Axes, fig: plt.Figure, x: np.ndarray, labels: Sequence[str], tick_cfg: Dict) -> None:
    formatted_labels = format_tick_labels(labels, tick_cfg)
    ax.set_xticks(x)
    ax.set_xticklabels(formatted_labels, fontsize=float(tick_cfg["x_tick_fontsize"]))
    ax.tick_params(
        axis="x",
        pad=float(tick_cfg.get("x_tick_pad", 1.4)),
        rotation=float(tick_cfg.get("x_tick_rotation", 0)),
        length=0,
    )
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_va("top")
        label.set_linespacing(float(tick_cfg.get("x_tick_linespacing", 0.92)))
    apply_tick_label_transforms(ax=ax, fig=fig, tick_cfg=tick_cfg)


def plot_single_panel(
    fig: plt.Figure,
    outer_spec,
    row: pd.Series,
    mode: str,
    opt_prefill_col: Optional[str],
    opt_decode_col: Optional[str],
    opt_total_col: str,
    nd_final_split_strategy: str,
    plot_cfg: Dict,
    bar_label_map: Dict[str, str],
    layout_name_map: Dict[str, str],
    share_y_lim: Optional[Tuple[float, float]] = None,
    show_ylabel: bool = False,
) -> Tuple[plt.Axes, plt.Axes]:
    fig_cfg = plot_cfg["figure"]
    panel_cfg = plot_cfg["panel"]
    tick_cfg = plot_cfg["ticks"]
    layout_mode = str(plot_cfg.get("layout_name_mode", "title")).strip().lower()

    inner = outer_spec.subgridspec(2, 1, height_ratios=[1.10, 4.05], hspace=float(fig_cfg["inner_hspace"]))
    ax_top = fig.add_subplot(inner[0])
    ax_bar = fig.add_subplot(inner[1])

    payload = build_bar_payload(
        row=row,
        mode=mode,
        opt_prefill_col=opt_prefill_col,
        opt_decode_col=opt_decode_col,
        opt_total_col=opt_total_col,
        nd_final_split_strategy=nd_final_split_strategy,
        bar_label_map=bar_label_map,
    )

    labels = [item["label"] for item in payload]
    xstep = float(fig_cfg["bar_xstep"])
    width = float(fig_cfg["bar_width"])
    x = np.arange(len(payload), dtype=float) * xstep
    xpad = max(0.12, width * 0.56)

    prefill_vals = np.array([item["prefill"] for item in payload], dtype=float)
    decode_vals = np.array([item["decode"] for item in payload], dtype=float)
    totals = prefill_vals + decode_vals

    prefill_colors = [item["prefill_color"] for item in payload]
    decode_colors = [item["decode_color"] for item in payload]

    ax_bar.bar(
        x,
        prefill_vals,
        width=width,
        color=prefill_colors,
        edgecolor="black",
        linewidth=0.7,
        zorder=2,
    )
    ax_bar.bar(
        x,
        decode_vals,
        width=width,
        bottom=prefill_vals,
        color=decode_colors,
        edgecolor="black",
        linewidth=0.7,
        zorder=2,
    )
    draw_stack_boundary(ax_bar, x, width, prefill_vals)

    ax_bar.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55, zorder=0)
    ax_bar.set_axisbelow(True)
    set_bar_xticklabels(ax=ax_bar, fig=fig, x=x, labels=labels, tick_cfg=tick_cfg)
    ax_bar.tick_params(axis="y", labelsize=float(tick_cfg["y_tick_fontsize"]))
    ax_bar.set_xlim(x[0] - xpad, x[-1] + xpad)
    if show_ylabel:
        ax_bar.set_ylabel("Time (s)", fontsize=10)

    if share_y_lim is not None:
        ax_bar.set_ylim(*share_y_lim)
    else:
        ax_bar.set_ylim(*compute_bar_ylim(float(np.max(totals)) if len(totals) else 1.0))

    annotate_bar_totals(ax_bar, x, totals, fontsize=float(tick_cfg["bar_annotation_fontsize"]))

    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    layout_name = get_layout_display_name(
        prefill_len=row["prefill_len"],
        decode_len=row["decode_len"],
        layout_name_map=layout_name_map,
        show_title_when_no_alias=bool(panel_cfg.get("show_title_when_no_alias", True)),
    )

    tick_label_mode = str(tick_cfg.get("x_tick_label_mode", "wrap_stagger")).strip().lower()
    extra_labelpad = 0.0
    if tick_label_mode in {"wrap", "wrap_stagger", "auto"}:
        extra_labelpad += 3.0
    if tick_label_mode in {"stagger", "wrap_stagger", "auto"}:
        extra_labelpad += 2.0

    if layout_name and layout_mode in {"xlabel", "both"}:
        ax_bar.set_xlabel(
            layout_name,
            fontsize=float(panel_cfg["layout_name_fontsize"]),
            labelpad=float(panel_cfg.get("layout_name_labelpad", 5.8)) + extra_labelpad,
        )

    speedup = get_speedup(row, opt_total_col)
    plot_zoomed_totals_strip(
        ax=ax_top,
        x=x,
        totals=totals,
        speedup=speedup,
        xpad=xpad,
        panel_title=layout_name if layout_mode in {"title", "both"} else None,
        title_fontsize=float(panel_cfg["layout_name_fontsize"]),
        speedup_fontsize=float(tick_cfg["speedup_fontsize"]),
    )

    return ax_top, ax_bar


def compute_shared_y_lim(
    group_df: pd.DataFrame,
    mode: str,
    opt_prefill_col: Optional[str],
    opt_decode_col: Optional[str],
    opt_total_col: str,
    nd_final_split_strategy: str,
    bar_label_map: Dict[str, str],
) -> Tuple[float, float]:
    maxima: List[float] = []
    for _, row in group_df.iterrows():
        payload = build_bar_payload(
            row=row,
            mode=mode,
            opt_prefill_col=opt_prefill_col,
            opt_decode_col=opt_decode_col,
            opt_total_col=opt_total_col,
            nd_final_split_strategy=nd_final_split_strategy,
            bar_label_map=bar_label_map,
        )
        maxima.append(max(item["total"] for item in payload))
    ymax = max(maxima) if maxima else 1.0
    return compute_bar_ylim(ymax)


def make_figure(
    group_df: pd.DataFrame,
    group_key: Tuple,
    mode: str,
    outdir: Path,
    dpi: int,
    fig_formats: Sequence[str],
    ncols_arg: str,
    share_y: bool,
    nd_final_split_strategy: str,
    opt_prefill_col: Optional[str],
    opt_decode_col: Optional[str],
    opt_total_col: str,
    plot_cfg: Dict,
    bar_label_map: Dict[str, str],
    layout_name_map: Dict[str, str],
) -> List[Path]:
    fig_cfg = plot_cfg["figure"]

    n_panels = len(group_df)
    ncols = resolve_ncols(ncols_arg, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig_w = max(float(fig_cfg["min_fig_width"]), ncols * float(fig_cfg["panel_width"]))
    fig_h = max(
        float(fig_cfg["min_fig_height"]),
        nrows * float(fig_cfg["panel_height"]) + float(fig_cfg["header_height"]),
    )
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    outer = fig.add_gridspec(
        nrows,
        ncols,
        wspace=float(fig_cfg["outer_wspace"]),
        hspace=float(fig_cfg["outer_hspace"]),
    )

    if share_y:
        shared_y_lim = compute_shared_y_lim(
            group_df=group_df,
            mode=mode,
            opt_prefill_col=opt_prefill_col,
            opt_decode_col=opt_decode_col,
            opt_total_col=opt_total_col,
            nd_final_split_strategy=nd_final_split_strategy,
            bar_label_map=bar_label_map,
        )
    else:
        shared_y_lim = None

    for idx, (_, row) in enumerate(group_df.iterrows()):
        r = idx // ncols
        c = idx % ncols
        plot_single_panel(
            fig=fig,
            outer_spec=outer[r, c],
            row=row,
            mode=mode,
            opt_prefill_col=opt_prefill_col,
            opt_decode_col=opt_decode_col,
            opt_total_col=opt_total_col,
            nd_final_split_strategy=nd_final_split_strategy,
            plot_cfg=plot_cfg,
            bar_label_map=bar_label_map,
            layout_name_map=layout_name_map,
            share_y_lim=shared_y_lim,
            show_ylabel=(c == 0),
        )

    model_family, model_variant, dtype, batch = group_key
    title = f"{model_family} {model_variant} ({dtype}) | batch={batch} | {mode}"
    fig.suptitle(title, fontsize=14.5, y=float(fig_cfg["suptitle_y"]))

    handles = legend_handles(mode, bar_label_map=bar_label_map)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, float(fig_cfg["legend_y"])),
        ncol=int(fig_cfg.get("legend_ncol", 5)),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        columnspacing=1.0,
        handletextpad=0.45,
    )

    strategy_note = None
    if mode in {"all_compare", "nd_only"} and not (opt_prefill_col and opt_decode_col):
        strategy_text = {
            "keep_prefill": "ND opt split: keep ND prefill, adjust decode",
            "proportional": "ND opt split: proportional to ND init",
        }[nd_final_split_strategy]
        strategy_note = strategy_text

    if strategy_note:
        fig.text(0.5, 0.024, strategy_note, ha="center", va="center", fontsize=8.6, color="0.35")
        bottom_margin = float(fig_cfg["bottom_with_note"])
    else:
        bottom_margin = float(fig_cfg["bottom"])

    fig.subplots_adjust(
        left=float(fig_cfg["left"]),
        right=float(fig_cfg["right"]),
        bottom=bottom_margin,
        top=float(fig_cfg["top"]),
    )

    stem = sanitize_filename(f"{model_family}_{model_variant}_{dtype}_batch{batch}_{mode}")
    out_paths: List[Path] = []
    for fmt in fig_formats:
        fmt = fmt.lower().lstrip(".")
        out_path = outdir / f"{stem}.{fmt}"
        fig.savefig(
            out_path,
            bbox_inches="tight",
            pad_inches=float(fig_cfg["save_pad_inches"]),
            dpi=dpi,
        )
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    cfg_path = Path(args.cfg).expanduser().resolve() if args.cfg else None
    outdir.mkdir(parents=True, exist_ok=True)

    plot_cfg = load_cfg(cfg_path)
    bar_label_map = resolve_bar_label_map(plot_cfg)
    layout_name_map = resolve_layout_name_map(plot_cfg)

    df = load_table(summary_path)

    require_columns(
        df,
        [
            "model_family",
            "model_variant",
            "dtype",
            "batch",
            "prefill_len",
            "decode_len",
            "nd_initial_prefill_s",
            "nd_initial_decode_s",
            "nd_initial_total_s",
            "nz_initial_prefill_s",
            "nz_initial_decode_s",
            "pim_opt_initial_prefill_s",
            "pim_opt_initial_decode_s",
        ],
    )

    opt_total_col = first_existing_column(df, OPT_TOTAL_ALIASES)
    if not opt_total_col:
        raise ValueError(
            f"Could not find any ND optimized total column. Tried: {OPT_TOTAL_ALIASES}"
        )
    opt_prefill_col = first_existing_column(df, OPT_PREFILL_ALIASES)
    opt_decode_col = first_existing_column(df, OPT_DECODE_ALIASES)

    for col in ["batch", "prefill_len", "decode_len"]:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    sort_cols = [col for col in args.sort_by if col in df.columns]
    if sort_cols:
        df = df.sort_values(by=["model_family", "model_variant", "dtype", "batch", *sort_cols])

    modes = [args.mode] if args.mode != "both" else ["all_compare", "nd_only"]

    saved_paths: List[Path] = []
    group_cols = ["model_family", "model_variant", "dtype", "batch"]
    for group_key, group_df in df.groupby(group_cols, sort=False):
        group_df = group_df.reset_index(drop=True)
        for mode in modes:
            saved_paths.extend(
                make_figure(
                    group_df=group_df,
                    group_key=group_key,
                    mode=mode,
                    outdir=outdir,
                    dpi=args.dpi,
                    fig_formats=args.fig_format,
                    ncols_arg=str(args.ncols),
                    share_y=bool(args.share_y),
                    nd_final_split_strategy=args.nd_final_split_strategy,
                    opt_prefill_col=opt_prefill_col,
                    opt_decode_col=opt_decode_col,
                    opt_total_col=opt_total_col,
                    plot_cfg=plot_cfg,
                    bar_label_map=bar_label_map,
                    layout_name_map=layout_name_map,
                )
            )

    print(f"Saved {len(saved_paths)} figure(s) to: {outdir}")
    if cfg_path is not None:
        print(f"Loaded cfg: {cfg_path}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
