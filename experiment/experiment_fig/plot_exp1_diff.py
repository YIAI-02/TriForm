#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot histogram(s) of trace-vs-verification signed gap from *merge_all.csv files.

Gap definition:
    gap_pct = (trace_total_s - total_time_s) / total_time_s * 100

Changes in this version:
1. x-axis range is fixed to [-10, 10] (%) and samples outside this range are dropped.
2. Support multiple file-list inputs and render multiple panels horizontally.
3. Panels share a single y-axis; horizontal spacing is controlled by --wspace.
4. When --file-list points to a directory, <dir>/files.txt is used automatically.

Examples:
python3 plot_exp1_diff.py \
  --file-list ../../verify/sst64_rst64/llama_7b_fp16_b16_s64/files.txt ../../verify/sst64_rst64/qwen_1.8b_fp16_b4_s64/files.txt  \
  --output ../../figs/exp1/qwen_1_8b_fp16_b4_s64_128_llama_7b_fp16_b16_s64_128_histogram.pdf \
  --panel-titles "Llama 7b, Qwen 1.8b" \
  --wspace 0.05
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================================
# USER EDIT ZONE
# ============================================================================
MANUAL_EXCLUDE: List[str] = []
HEFT_VARIANTS: List[str] = ["heft", "hefthint"]
POSITIVE_COLOR = "#39a937"
NEGATIVE_COLOR = "#3760a9"
DEFAULT_FIG_WIDTH = 5.0   # interpreted as per-panel width
DEFAULT_FIG_HEIGHT = 1.6
DEFAULT_WSPACE = 0.12
XMIN = -10.0
XMAX = 10.0
XTICK_STEP = 2.0
MIN_SIDE_SPAN = 2.0
# ============================================================================


FNAME_RE = re.compile(r"^(?P<algo>.+?)_(?P<lin>\d+)x(?P<lout>\d+)\.merge_all\.csv$")


@dataclass
class Metrics:
    algo_label: str
    source_variant: str
    csv_path: Path
    n_rows: int
    trace_total: float
    meas_total: float


@dataclass
class PanelData:
    label: str
    gap_df: pd.DataFrame
    dropped_out_of_range: int
    total_before_range_filter: int


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def _read_text_lines(p: Path) -> List[str]:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    lines: List[str] = []
    for raw in txt.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def resolve_file_list_arg(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if p.is_file():
        return p

    if p.is_dir():
        default_file = p / "files.txt"
        if default_file.is_file():
            return default_file.resolve()

        txt_files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".txt"])
        if len(txt_files) == 1:
            return txt_files[0].resolve()

        if not txt_files:
            raise FileNotFoundError(
                f"Directory '{p}' does not contain files.txt (or any unique *.txt file) to use as file-list."
            )
        raise FileNotFoundError(
            f"Directory '{p}' contains multiple *.txt files; please pass the exact file-list path instead."
        )

    raise FileNotFoundError(f"--file-list path not found: {path_str}")


def derive_panel_label_from_file_list(file_list: Path) -> str:
    if file_list.name.lower() == "files.txt":
        return file_list.parent.name
    return file_list.stem


def derive_panel_label_from_search_dir(search_dir: Path) -> str:
    return search_dir.resolve().name


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


# ---------------------------------------------------------------------------
# Parsing / indexing helpers
# ---------------------------------------------------------------------------
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


def parse_panel_titles_arg(panel_titles: Optional[str]) -> Optional[List[str]]:
    if panel_titles is None:
        return None
    titles = [x.strip() for x in panel_titles.split(",")]
    return [x for x in titles]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_metrics(csv_path: Path, algo_label: str, source_variant: str) -> Metrics:
    df = pd.read_csv(csv_path)
    required = ["total_time_s", "trace_total_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required)
    if df.empty:
        raise ValueError(f"{csv_path} has no valid rows after dropping NaNs")

    return Metrics(
        algo_label=algo_label,
        source_variant=source_variant,
        csv_path=csv_path,
        n_rows=int(len(df)),
        trace_total=float(df["trace_total_s"].mean()),
        meas_total=float(df["total_time_s"].mean()),
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
            if algo_on_disk in HEFT_VARIANTS:
                continue
            try:
                res[algo_on_disk] = pick_best_by_trace_total(paths, algo_label=algo_on_disk, source_variant=algo_on_disk)
            except Exception as e:
                print(f"[WARN] {lin}x{lout}: cannot load '{algo_on_disk}': {e}", file=sys.stderr)

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


def collect_gap_records(dim_results: Dict[Tuple[int, int], Dict[str, Metrics]], dims: List[Tuple[int, int]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for lin, lout in dims:
        res = dim_results.get((lin, lout), {})
        if not res:
            continue

        dim_label = f"{lin}x{lout}"
        for algo_key, m in res.items():
            df = pd.read_csv(m.csv_path)
            required = ["total_time_s", "trace_total_s"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[WARN] skip {m.csv_path}: missing columns {missing}", file=sys.stderr)
                continue

            for c in required:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=required)
            if df.empty:
                continue

            denom = df["total_time_s"].replace(0, np.nan)
            gap_pct = (df["trace_total_s"] - df["total_time_s"]) / denom * 100.0
            valid = np.isfinite(gap_pct.to_numpy(dtype=float))
            if not np.any(valid):
                continue

            arr = gap_pct.to_numpy(dtype=float)[valid]
            sign = np.where(arr >= 0.0, "Positive gap", "Negative gap")

            frames.append(
                pd.DataFrame(
                    {
                        "dim": dim_label,
                        "algo": algo_key,
                        "gap_pct": arr,
                        "sign": sign,
                        "csv_path": str(m.csv_path),
                    }
                )
            )

    if not frames:
        return pd.DataFrame(columns=["dim", "algo", "gap_pct", "sign", "csv_path"])
    return pd.concat(frames, ignore_index=True)


def apply_exclude_filters(
    dim_results: Dict[Tuple[int, int], Dict[str, Metrics]],
    exclude_set: Sequence[str],
) -> None:
    if not exclude_set:
        return
    exclude_norm = {_normalize_strategy_token(t) for t in exclude_set if t and _normalize_strategy_token(t)}
    for dim in list(dim_results.keys()):
        res = dim_results.get(dim, {})
        for k in list(res.keys()):
            if _normalize_strategy_token(k) in exclude_norm:
                res.pop(k, None)


# ---------------------------------------------------------------------------
# Panel prep / plotting helpers
# ---------------------------------------------------------------------------
def round_up_to_step(x: float, step: float) -> float:
    return float(np.ceil(x / step) * step)


def round_down_to_step(x: float, step: float) -> float:
    return float(np.floor(x / step) * step)


def compute_panel_xlim(gap_values: np.ndarray) -> Tuple[float, float]:
    """
    Compute a display x-range that:
    - never exceeds [-10, 10]
    - shrinks automatically if the tails have no data
    - still keeps a small span around zero for readability
    """
    if gap_values.size == 0:
        return (-MIN_SIDE_SPAN, MIN_SIDE_SPAN)

    neg = gap_values[gap_values < 0.0]
    pos = gap_values[gap_values > 0.0]

    if neg.size:
        left = round_down_to_step(float(np.min(neg)), XTICK_STEP)
    else:
        left = -MIN_SIDE_SPAN

    if pos.size:
        right = round_up_to_step(float(np.max(pos)), XTICK_STEP)
    else:
        right = MIN_SIDE_SPAN

    left = min(left, -MIN_SIDE_SPAN)
    right = max(right, MIN_SIDE_SPAN)
    left = max(XMIN, left)
    right = min(XMAX, right)

    if left >= 0.0:
        left = -MIN_SIDE_SPAN
    if right <= 0.0:
        right = MIN_SIDE_SPAN

    return (left, right)


def build_panel_data_from_csv_paths(
    csv_paths: List[Path],
    label: str,
    dims_arg: Optional[str],
    exclude_tokens: Sequence[str],
) -> PanelData:
    if not csv_paths:
        return PanelData(
            label=label,
            gap_df=pd.DataFrame(columns=["dim", "algo", "gap_pct", "sign", "csv_path"]),
            dropped_out_of_range=0,
            total_before_range_filter=0,
        )

    idx = index_merge_all(csv_paths)
    dims = parse_dims_arg(dims_arg) if dims_arg else infer_dims(idx)
    dim_results = build_dim_results(idx, dims)
    apply_exclude_filters(dim_results, exclude_tokens)
    gap_df = collect_gap_records(dim_results, dims)

    total_before = int(len(gap_df))
    if gap_df.empty:
        return PanelData(
            label=label,
            gap_df=gap_df,
            dropped_out_of_range=0,
            total_before_range_filter=0,
        )

    in_range_mask = gap_df["gap_pct"].between(XMIN, XMAX, inclusive="both")
    dropped = int((~in_range_mask).sum())
    gap_df_in_range = gap_df.loc[in_range_mask].copy()

    if dropped > 0:
        print(
            f"[INFO] panel '{label}': dropped {dropped} samples outside [{XMIN:.0f}, {XMAX:.0f}]% "
            f"(kept {len(gap_df_in_range)}/{total_before})."
        )

    return PanelData(
        label=label,
        gap_df=gap_df_in_range,
        dropped_out_of_range=dropped,
        total_before_range_filter=total_before,
    )


def build_panels_from_args(args: argparse.Namespace) -> List[PanelData]:
    if args.file_list and args.search_dir:
        raise SystemExit("Please provide either --file-list or --search-dir, not both.")

    panel_specs: List[Tuple[str, str, List[Path]]] = []
    root = Path(args.root).resolve() if args.root else None

    if args.file_list:
        for raw in args.file_list:
            file_list_path = resolve_file_list_arg(raw)
            csv_paths = [p for p in resolve_file_list_paths(file_list_path, root=root) if p.name.endswith(".merge_all.csv")]
            panel_specs.append(("file-list", derive_panel_label_from_file_list(file_list_path), csv_paths))
    elif args.search_dir:
        for raw in args.search_dir:
            sd = Path(raw).expanduser().resolve()
            if not sd.exists():
                raise FileNotFoundError(f"--search-dir not found: {sd}")
            csv_paths = glob_merge_all(sd)
            panel_specs.append(("search-dir", derive_panel_label_from_search_dir(sd), csv_paths))
    else:
        raise SystemExit("Provide either --file-list or --search-dir")

    if not panel_specs:
        raise SystemExit("No panel inputs were resolved.")

    custom_titles = parse_panel_titles_arg(args.panel_titles)
    if custom_titles is not None and len(custom_titles) != len(panel_specs):
        raise SystemExit(
            f"--panel-titles count ({len(custom_titles)}) must match number of panels ({len(panel_specs)})."
        )

    exclude_tokens: List[str] = []
    exclude_tokens.extend(MANUAL_EXCLUDE)
    exclude_tokens.extend(parse_exclude_arg(args.exclude))

    panels: List[PanelData] = []
    for idx_panel, (_, default_label, csv_paths) in enumerate(panel_specs):
        label = custom_titles[idx_panel] if custom_titles is not None else default_label
        panel = build_panel_data_from_csv_paths(
            csv_paths=csv_paths,
            label=label,
            dims_arg=args.dims,
            exclude_tokens=exclude_tokens,
        )
        panels.append(panel)

    if not any(p.total_before_range_filter > 0 for p in panels):
        raise RuntimeError("No valid gap data found to plot.")

    return panels


def plot_gap_histograms(
    panels: Sequence[PanelData],
    output: Path,
    bins: int = 50,
    title: Optional[str] = None,
    dpi: int = 200,
    fig_width: float = DEFAULT_FIG_WIDTH,
    fig_height: float = DEFAULT_FIG_HEIGHT,
    wspace: float = DEFAULT_WSPACE,
) -> None:
    if not panels:
        raise RuntimeError("No panels to plot.")

    sns.set_theme(style="whitegrid")

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width * n_panels, fig_height),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    legend_handles = [
        Patch(facecolor=NEGATIVE_COLOR, edgecolor="white", linewidth=0.5, alpha=0.60, label="Negative gap"),
        Patch(facecolor=POSITIVE_COLOR, edgecolor="white", linewidth=0.5, alpha=0.60, label="Positive gap"),
    ]

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        arr = panel.gap_df["gap_pct"].to_numpy(dtype=float) if not panel.gap_df.empty else np.array([], dtype=float)
        x_left, x_right = compute_panel_xlim(arr)
        bin_edges = np.linspace(x_left, x_right, bins + 1)

        neg = arr[arr < 0.0]
        pos = arr[arr >= 0.0]

        if neg.size:
            ax.hist(
                neg,
                bins=bin_edges,
                color=NEGATIVE_COLOR,
                alpha=0.60,
                edgecolor="white",
                linewidth=0.5,
            )
        if pos.size:
            ax.hist(
                pos,
                bins=bin_edges,
                color=POSITIVE_COLOR,
                alpha=0.60,
                edgecolor="white",
                linewidth=0.5,
            )

        # ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_xlim(x_left, x_right)
        ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
        ax.set_xlabel("Signed gap (%)", fontsize=11)
        ax.set_title(panel.label, fontsize=12)

        if i == 0:
            ax.set_ylabel("Count", fontsize=11)
        else:
            ax.set_ylabel("")

        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.grid(axis="x", visible=False)

        if panel.total_before_range_filter > 0 and panel.gap_df.empty:
            ax.text(
                0.5,
                0.5,
                "No data\nin displayed range",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

    top_margin = 0.78 if title else 0.86
    fig.subplots_adjust(wspace=wspace, top=top_margin)

    if title:
        fig.suptitle(title, fontsize=13, y=0.97)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"[OK] saved figure -> {output.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file-list",
        type=str,
        nargs="+",
        default=None,
        help="One or more file-list paths, or directories that contain files.txt. Each input becomes one panel.",
    )
    ap.add_argument("--root", type=str, default=None, help="Optional root directory to resolve relative paths in file-list.")
    ap.add_argument(
        "--search-dir",
        type=str,
        nargs="+",
        default=None,
        help="If --file-list is not provided, recursively search one or more directories for *.merge_all.csv. Each directory becomes one panel.",
    )
    ap.add_argument("--dims", type=str, default=None, help="Comma-separated dims to include. If omitted, include all dims found per panel.")
    ap.add_argument("--output", type=str, default="plot_gap_hist.png", help="Output image path.")
    ap.add_argument("--title", type=str, default=None, help="Figure title.")
    ap.add_argument("--panel-titles", type=str, default=None, help="Comma-separated titles, one per panel.")
    ap.add_argument("--exclude", type=str, default=None, help="Comma-separated strategy names to exclude from plotting.")
    ap.add_argument("--bins", type=int, default=50, help="Histogram bins per panel.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--fig-width", type=float, default=DEFAULT_FIG_WIDTH, help="Per-panel width in inches.")
    ap.add_argument("--fig-height", type=float, default=DEFAULT_FIG_HEIGHT, help="Total figure height in inches.")
    ap.add_argument("--wspace", type=float, default=DEFAULT_WSPACE, help="Horizontal spacing between panels.")
    ap.add_argument("--show", action="store_true", help="Show interactively (in addition to saving).")

    args = ap.parse_args()
    panels = build_panels_from_args(args)
    plot_gap_histograms(
        panels=panels,
        output=Path(args.output),
        bins=int(args.bins),
        title=args.title,
        dpi=int(args.dpi),
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        wspace=float(args.wspace),
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

