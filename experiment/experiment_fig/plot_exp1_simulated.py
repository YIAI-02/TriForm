#!/usr/bin/env python3
"""Plot simulated latency and speedup from ``baseline_compare_*.json`` files.

Run this script from ``experiment/experiment_fig`` in the current repository
layout. The examples below assume simulation outputs are stored under the
repository-level ``output/`` directory produced by ``python src/main.py
evaluate`` or the scripts in ``commands/``.

Examples
--------
Scan one output subtree and write figures into a shared directory::

    python plot_exp1_simulated.py \
      --output-root ../../output/evaluate_single_test \
      --out-dir ../../figs/exp1/simulated

Plot one specific run directory that already contains
``baseline_compare_*.json`` files::

    python plot_exp1_simulated.py \
      --model-dir ../../output/evaluate_single_test/hardware_1npu_2aim/llama_7b_fp16_b1_s2 \
      --out-dir ../../figs/exp1/simulated

Preview discovered targets without generating figures::

    python plot_exp1_simulated.py \
      --output-root ../../output/evaluate_single_test \
      --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

ARIAL_FONT_FAMILY = "Arial"
MIN_FONT_PT = 7.0
PANELS_PER_PAGE = 4

PLOT_COLORS = {
    "prefill": "#326568",
    "decode": "#A2D091",
    "speedup": "#000000",
}

PREFERRED_ORDER = ["PD", "AF", "PD+FFN", "PD+Linear", "PD+Attn", "HEFT", "Bifocal"]


def apply_global_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": [ARIAL_FONT_FAMILY],
            "font.sans-serif": [ARIAL_FONT_FAMILY],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


apply_global_plot_style()


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


def canonical_policy_name(policy: str) -> str:
    name = (policy or "").strip()
    if name.startswith("algo:"):
        name = name.split(":", 1)[1].strip()
    return name


def ordered_policies(policies: Sequence[str]) -> List[str]:
    unique: List[str] = []
    seen: set[str] = set()
    for policy in policies:
        if policy not in seen:
            unique.append(policy)
            seen.add(policy)

    by_name: Dict[str, str] = {}
    for policy in unique:
        by_name.setdefault(canonical_policy_name(policy), policy)

    ordered: List[str] = []
    for name in PREFERRED_ORDER:
        policy = by_name.get(name)
        if policy is not None:
            ordered.append(policy)

    remaining = [policy for policy in unique if policy not in ordered]
    remaining.sort(key=lambda item: canonical_policy_name(item))
    return ordered + remaining


def parse_case_from_config_or_name(cfg: Dict[str, object], path: Path) -> Tuple[int, int]:
    prefill = cfg.get("prefill_len")
    decode = cfg.get("decode_len")
    if prefill is None or decode is None:
        match = re.search(r"(\d+)x(\d+)", path.stem)
        if match is None:
            return -1, -1
        prefill, decode = int(match.group(1)), int(match.group(2))
    return int(prefill), int(decode)


def load_baseline_compare(
    path: Path,
) -> Tuple[Tuple[int, int], Dict[str, Tuple[float, float, float]], Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload.get("config", {}) or {}
    case = parse_case_from_config_or_name(cfg, path)

    time_map: Dict[str, Tuple[float, float, float]] = {}
    for row in payload.get("results", []) or []:
        policy = row.get("policy")
        if not policy:
            continue
        prefill = float(row.get("prefill_time_s", 0.0))
        decode = float(row.get("decode_time_s", 0.0))
        total = float(row.get("total_time_s", 0.0))
        if total <= 0:
            total = prefill + decode
        time_map[str(policy)] = (prefill, decode, total)
    return case, time_map, cfg


def safe_filename(value: str, max_len: int = 240) -> str:
    value = value.replace(os.sep, "__")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_.=+\-]+", "_", value).strip("_")
    if len(value) > max_len:
        keep = max_len - 10
        value = value[: keep // 2] + "__TRUNC__" + value[-keep // 2 :]
    return value


def add_page_suffix(outfile: Path, page_idx: int, total_pages: int) -> Path:
    if total_pages <= 1:
        return outfile
    return outfile.with_name(f"{outfile.stem}__page{page_idx + 1}{outfile.suffix}")


def chunk_sequence(items: Sequence[Tuple[Tuple[int, int], Path, Dict[str, Tuple[float, float, float]]]], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def plot_one_case(
    ax: plt.Axes,
    time_map: Dict[str, Tuple[float, float, float]],
    *,
    title: str,
) -> Tuple[object | None, object | None, object | None, plt.Axes | None]:
    policies = ordered_policies(list(time_map.keys()))
    if not policies:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return None, None, None, None

    labels = [canonical_policy_name(policy) for policy in policies]
    prefill = np.array([time_map[policy][0] for policy in policies], dtype=float)
    decode = np.array([time_map[policy][1] for policy in policies], dtype=float)
    total = np.array([time_map[policy][2] for policy in policies], dtype=float)

    x = np.arange(len(policies))
    bars_prefill = ax.bar(
        x,
        prefill,
        label="Prefill",
        color=PLOT_COLORS["prefill"],
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )
    bars_decode = ax.bar(
        x,
        decode,
        bottom=prefill,
        label="Decode",
        color=PLOT_COLORS["decode"],
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )

    ax.set_title(title, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.85, zorder=0)

    pd_total = None
    for policy, total_s in zip(policies, total):
        if canonical_policy_name(policy) == "PD":
            pd_total = float(total_s)
            break

    line = None
    ax_right = None
    if pd_total is not None and pd_total > 0:
        speedup = pd_total / total
        ax_right = ax.twinx()
        ax_right.patch.set_visible(False)
        ax_right.set_zorder(ax.get_zorder() + 1)
        (line,) = ax_right.plot(
            x,
            speedup,
            color=PLOT_COLORS["speedup"],
            marker="o",
            linewidth=2.0,
            zorder=5,
        )

        finite = speedup[np.isfinite(speedup)]
        if finite.size:
            speedup_max = float(np.max(finite))
            span = float(np.max(finite) - np.min(finite))
            offset = 0.02 * span if span > 0 else 0.05 * max(speedup_max, 1.0)
            ax_right.set_ylim(0.0, speedup_max * 1.15 if speedup_max > 0 else 1.0)
            for xpos, value in zip(x, speedup):
                if not np.isfinite(value):
                    continue
                ax_right.text(
                    xpos,
                    value + offset,
                    f"{value:.2f}×",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=PLOT_COLORS["speedup"],
                )

    return bars_prefill, bars_decode, line, ax_right


def plot_compare_grid(
    loaded: Sequence[Tuple[Tuple[int, int], Path, Dict[str, Tuple[float, float, float]]]],
    *,
    outfile: Path,
    sharey: bool = False,
) -> None:
    loaded = sorted(loaded, key=lambda item: (item[0][0], item[0][1], str(item[1])))
    if not loaded:
        raise ValueError("No baseline_compare files to plot")

    cols = min(PANELS_PER_PAGE, len(loaded))
    rows = 1
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(max(6.0, 5.2 * cols), 4.8),
        sharey=sharey,
        squeeze=False,
        gridspec_kw={"wspace": 0.2, "hspace": 0.25},
    )
    axes_flat = axes.ravel()

    legend_handles = None
    legend_labels = None

    for index, (case, _path, time_map) in enumerate(loaded):
        ax = axes_flat[index]
        prefill, decode = case
        bars_prefill, bars_decode, line, ax_right = plot_one_case(
            ax,
            time_map,
            title=f"prefill={prefill}, decode={decode}",
        )

        if index == 0:
            ax.set_ylabel("Latency (s)")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

        if ax_right is not None:
            is_last = index == len(loaded) - 1
            if is_last:
                ax_right.set_ylabel("Speedup vs PD")
            else:
                ax_right.set_ylabel("")
                ax_right.tick_params(axis="y", which="both", right=False, labelright=False)
                ax_right.spines["right"].set_visible(False)

        if legend_handles is None and bars_prefill is not None and bars_decode is not None:
            legend_handles = [bars_prefill, bars_decode]
            legend_labels = ["Prefill", "Decode"]
            if line is not None:
                legend_handles.append(line)
                legend_labels.append("Speedup vs PD")

    for index in range(len(loaded), rows * cols):
        axes_flat[index].axis("off")

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 0.935),
            borderaxespad=0.1,
            handletextpad=0.6,
            columnspacing=1.0,
        )

    enforce_figure_fonts(fig)
    fig.subplots_adjust(right=0.99, left=0.06, bottom=0.18, top=0.84, wspace=0.2)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def find_model_dirs(output_root: Path, pattern: str) -> Iterator[Path]:
    seen: set[Path] = set()
    for path in sorted(output_root.rglob(pattern)):
        model_dir = path.parent
        if model_dir not in seen:
            seen.add(model_dir)
            yield model_dir


def collect_compare_files(model_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    files = sorted(model_dir.rglob(pattern) if recursive else model_dir.glob(pattern))
    if files or not recursive:
        return files
    return sorted(model_dir.rglob(pattern))


def build_outfile_name(model_dir: Path, cases: Sequence[Tuple[int, int]], *, ext: str = "pdf") -> str:
    case_tag = "__".join(f"{prefill}x{decode}" for prefill, decode in cases)
    rel_parts = model_dir.parts[-3:] if len(model_dir.parts) >= 3 else model_dir.parts
    base = "__".join(rel_parts) + "__" + case_tag
    return f"{safe_filename(base)}.{ext}"


def process_one_model_dir(
    *,
    model_dir: Path,
    pattern: str,
    recursive: bool,
    sharey: bool,
    out_dir: Optional[Path],
    dry_run: bool,
) -> List[Path]:
    files = collect_compare_files(model_dir, pattern=pattern, recursive=recursive)
    if not files:
        return []

    loaded = []
    for path in files:
        case, time_map, _ = load_baseline_compare(path)
        loaded.append((case, path, time_map))

    loaded.sort(key=lambda item: (item[0][0], item[0][1], str(item[1])))
    cases_sorted = [item[0] for item in loaded]

    out_name = build_outfile_name(model_dir, cases_sorted, ext="pdf")
    out_base = ((out_dir or model_dir) / out_name).resolve()

    pages = list(chunk_sequence(loaded, PANELS_PER_PAGE))
    out_paths: List[Path] = []

    for page_idx, page_items in enumerate(pages):
        out_path = add_page_suffix(out_base, page_idx, len(pages))
        out_paths.append(out_path)

        if dry_run:
            print(f"[DRY-RUN] {model_dir} -> {out_path.name}")
            continue

        plot_compare_grid(page_items, outfile=out_path, sharey=sharey)
        print(f"[OK] {model_dir} -> {out_path.name}")

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latency and speedup from evaluate outputs.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory to scan recursively for baseline_compare_*.json files.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Plot one directory that already contains baseline_compare_*.json files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="baseline_compare_*.json",
        help="Filename pattern used to find compare JSON files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively inside the selected model directory.",
    )
    parser.add_argument(
        "--sharey",
        action="store_true",
        help="Share the latency y-axis across panels.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory for generated figures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered targets without generating figures.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    if args.model_dir:
        process_one_model_dir(
            model_dir=Path(args.model_dir).resolve(),
            pattern=args.pattern,
            recursive=args.recursive,
            sharey=args.sharey,
            out_dir=out_dir,
            dry_run=args.dry_run,
        )
        return

    output_root = Path(args.output_root).resolve() if args.output_root else Path.cwd().resolve()
    found_any = False
    for model_dir in find_model_dirs(output_root, args.pattern):
        out_paths = process_one_model_dir(
            model_dir=model_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            sharey=args.sharey,
            out_dir=out_dir,
            dry_run=args.dry_run,
        )
        if out_paths:
            found_any = True

    if not found_any:
        print(f"[WARN] No '{args.pattern}' files found under: {output_root}")


if __name__ == "__main__":
    main()
