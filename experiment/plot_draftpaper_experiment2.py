#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot 6 heatmaps (2x3 subplots) for speedup = PD latency / Hefthint latency.
python3 plot_draftpaper_experiment2.py --out ../figs/draftpaper_exp2/speedup_heatmaps.pdf --batches 1,4,8,16,32
修改里面的dir
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


PALETTE = ["#0024FF", "#0349FF", "#006CFE", "#0092FE",
           "#FF3F04", "#FE5D00", "#FE8000", "#FFBF02"]

FILENAME_RE = re.compile(r"baseline_compare_(\d+)x(\d+)\.json$")


@dataclass
class PlotSpec:
    title: str
    root_dir: Path
    model_glob: str = "qwen_7b_*_b*_s64"
    pd_policy: str = "algo:pd"
    work_policy: str = "algo:hefthint"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _find_policy_latency(data: dict, policy: str, latency_field: str) -> float:
    for item in data.get("results", []):
        if item.get("policy") == policy:
            return _safe_float(item.get(latency_field))
    return float("nan")


def _parse_one_json(
    json_path: Path,
    pd_policy: str,
    work_policy: str,
    latency_field: str,
) -> Optional[dict]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    cfg = data.get("config", {}) or {}
    prefill = cfg.get("prefill_len")
    decode = cfg.get("decode_len")
    batch = cfg.get("batch")

    # fallback from filename
    if prefill is None or decode is None:
        m = FILENAME_RE.search(json_path.name)
        if m:
            prefill = int(m.group(1))
            decode = int(m.group(2))

    if prefill is None or decode is None:
        return None

    # IMPORTANT: pd & hefthint come from THE SAME json file => per-config PD baseline
    pd_lat = _find_policy_latency(data, pd_policy, latency_field)
    work_lat = _find_policy_latency(data, work_policy, latency_field)

    if not np.isfinite(pd_lat) or not np.isfinite(work_lat) or work_lat == 0.0:
        return None

    return {
        "prefill_len": int(prefill),
        "decode_len": int(decode),
        "batch": int(batch) if batch is not None else np.nan,
        "pd_latency": float(pd_lat),
        "work_latency": float(work_lat),
        "speedup": float(pd_lat) / float(work_lat),
        "file": str(json_path),
    }


def collect_records(spec: PlotSpec, latency_field: str) -> pd.DataFrame:
    records: List[dict] = []
    model_dirs = sorted([p for p in spec.root_dir.glob(spec.model_glob) if p.is_dir()])
    for model_dir in model_dirs:
        for jf in sorted(model_dir.glob("baseline_compare_*x*.json")):
            rec = _parse_one_json(jf, spec.pd_policy, spec.work_policy, latency_field)
            if rec is not None:
                rec["model_dir"] = str(model_dir)
                records.append(rec)
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            f"No valid records under {spec.root_dir} (check root_dir/model_glob/latency_field/policy names)."
        )
    return df


def build_matrix(df: pd.DataFrame, batch_agg: str, batch_filter: Optional[int]) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pd_latency", "work_latency", "speedup"])

    if batch_filter is not None:
        df = df[df["batch"] == int(batch_filter)]

    if df.empty:
        raise RuntimeError(
            f"No records after filtering batch={batch_filter}. "
            "Check whether JSON config has 'batch' and the directory contains that batch."
        )

    if batch_agg == "ratio_of_means":
        g = df.groupby(["decode_len", "prefill_len"]).agg(
            pd_mean=("pd_latency", "mean"),
            work_mean=("work_latency", "mean"),
        )
        mat = (g["pd_mean"] / g["work_mean"]).unstack("prefill_len")
    elif batch_agg == "mean_ratio":
        g = df.groupby(["decode_len", "prefill_len"]).agg(
            speedup=("speedup", "mean"),
        )
        mat = g["speedup"].unstack("prefill_len")
    else:
        raise ValueError(f"Unknown batch_agg: {batch_agg}")

    return mat.sort_index().sort_index(axis=1)


def default_specs(base_dir: Path) -> List[PlotSpec]:
    # Matches your requested 6 subplots.
    return [
        PlotSpec(title="NPU 1GB AiM", root_dir=base_dir / "../algorithms/output/experiment_npu/hw_npu_aim/st64"),
        PlotSpec(title="NPU 2GB AiM", root_dir=base_dir / "../algorithms/output/experiment_npu/hw_npu_2aim/st64"),
        PlotSpec(title="NPU 4GB AiM", root_dir=base_dir / "../algorithms/output/experiment_npu/hw_npu_4aim/st64"),
        PlotSpec(title="2×NPU 2GB AiM", root_dir=base_dir / "../algorithms/output/experiment_2npu/hw_2npu_2aim/st64"),
        PlotSpec(title="2×NPU 4GB AiM", root_dir=base_dir / "../algorithms/output/experiment_2npu/hw_2npu_4aim/st64"),
        PlotSpec(title="2×NPU 8GB AiM", root_dir=base_dir / "../algorithms/output/experiment_2npu/hw_2npu_8aim/st64"),
    ]


def _robust_vmax(values: np.ndarray, factor: float = 1.6, q: float = 90.0) -> float:
    vmax = float(np.nanmax(values))
    vq = float(np.nanpercentile(values, q))
    if np.isfinite(vq) and vq > 0 and vmax > factor * vq:
        return vq
    return vmax


def _pick_tick_step(vmin: float, vmax: float) -> float:
    rng = vmax - vmin
    if rng <= 2.0:
        return 0.2
    if rng <= 5.0:
        return 0.5
    if rng <= 10.0:
        return 1.0
    return 2.0


def _compute_norm(
    mats: List[pd.DataFrame],
    robust: bool,
    robust_factor: float,
    robust_quantile: float,
) -> Tuple[Union[Normalize, TwoSlopeNorm], float, float, float, float, float]:
    """Return (norm, vmin_plot, vmax_plot, step, gmin, gmax_raw)."""
    all_vals = np.concatenate([m.to_numpy().ravel() for m in mats])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise RuntimeError("No finite speedup values to plot.")

    gmin = float(np.min(all_vals))
    gmax_raw = float(np.max(all_vals))
    gmax = _robust_vmax(all_vals, factor=robust_factor, q=robust_quantile) if robust else gmax_raw

    step = _pick_tick_step(gmin, gmax)
    vmin_plot = math.floor(gmin / step) * step
    vmax_plot = math.ceil(gmax / step) * step

    if vmin_plot < 1.0 < vmax_plot:
        norm = TwoSlopeNorm(vmin=vmin_plot, vcenter=1.0, vmax=vmax_plot)
    else:
        norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)

    return norm, float(vmin_plot), float(vmax_plot), float(step), gmin, gmax_raw


def _make_one_figure(
    specs: List[PlotSpec],
    mats: List[pd.DataFrame],
    norm: Union[Normalize, TwoSlopeNorm],
    vmin_plot: float,
    vmax_plot: float,
    step: float,
    fig_w: float,
    fig_h: float,
    wspace: float,
    hspace: float,
    page_label: Optional[str] = None,
) -> plt.Figure:

    plt.rcParams.update({
        "font.size": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    })

    cmap = LinearSegmentedColormap.from_list("aim_palette", PALETTE, N=256)
    cmap.set_bad(color="#E0E0E0")

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1, 0.055],
        wspace=wspace,
        hspace=hspace,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    cax = fig.add_subplot(gs[:, 3])

    # page label
    fig.text(0.01, 0.985, page_label, ha="left", va="top", fontsize=7)

    mappable = None

    for i, (spec, mat) in enumerate(zip(specs, mats)):
        r = 0 if i < 3 else 1
        c = i % 3
        ax = axes[r][c]

        arr = mat.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(arr)

        im = ax.imshow(
            masked,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        mappable = im

        x_vals = list(mat.columns.astype(int))
        y_vals = list(mat.index.astype(int))

        ax.set_title(spec.title, pad=2)

        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))

        # bottom row x tick labels only
        if r == 1:
            ax.set_xticklabels([str(x) for x in x_vals])
            ax.tick_params(axis="x", bottom=True, labelbottom=True, width=1.0, length=3)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

        # left column y tick labels only
        if c == 0:
            ax.set_yticklabels([str(y) for y in y_vals])
            ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, width=1.0, length=3)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)

        # cell annotations
        for yy in range(arr.shape[0]):
            for xx in range(arr.shape[1]):
                v = arr[yy, xx]
                if not np.isfinite(v):
                    continue

                rr, gg, bb, _ = im.cmap(im.norm(v))
                luma = 0.299 * rr + 0.587 * gg + 0.114 * bb
                txt_color = "black" if luma > 0.6 else "white"

                ax.text(
                    xx, yy, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=4, fontweight="bold",
                    color=txt_color,
                )

        # grid lines
        ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.8, alpha=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xlabel("")
        ax.set_ylabel("")

    if mappable is not None:
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.tick_params(labelsize=7, width=1.0, length=3)
        ticks = np.arange(vmin_plot, vmax_plot + 1e-9, step)
        cbar.set_ticks(ticks)
        if step < 1.0:
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        else:
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        cbar.ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))
        cbar.ax.tick_params(which="minor", length=2, width=0.8)
        cbar.set_label("")

    # Global axis labels
    fig.supxlabel("decode length", fontsize=7)
    fig.supylabel("prefill length", fontsize=7)

    return fig


def parse_batches(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=".")
    ap.add_argument("--out", type=str, default="speedup_heatmaps_bybatch.pdf")

    ap.add_argument("--latency_field", type=str, default="total_time_s",
                    choices=["total_time_s", "prefill_time_s", "decode_time_s"])
    ap.add_argument("--batch_agg", type=str, default="ratio_of_means",
                    choices=["ratio_of_means", "mean_ratio"])

    ap.add_argument("--batches", type=str, default="1,4,8,16,32")
    ap.add_argument("--no_avg", action="store_true")

    ap.add_argument("--shared_scale", action="store_true")

    ap.add_argument("--no_robust", action="store_true")
    ap.add_argument("--robust_factor", type=float, default=1.6)
    ap.add_argument("--robust_quantile", type=float, default=90.0)

    ap.add_argument("--fig_w", type=float, default=5.0)
    ap.add_argument("--fig_h", type=float, default=3.0)
    ap.add_argument("--wspace", type=float, default=0.1)
    ap.add_argument("--hspace", type=float, default=0.14)

    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_path = Path(args.out).resolve()

    specs = default_specs(base_dir)

    batches = parse_batches(args.batches)
    keys: List[Tuple[str, Optional[int]]] = [(f"batch={b}", b) for b in batches]
    if not args.no_avg:
        keys.append(("AVG (all batches)", None))

    dfs = [collect_records(spec, args.latency_field) for spec in specs]

    mats_by_key: Dict[str, List[pd.DataFrame]] = {}
    for label, b in keys:
        mats: List[pd.DataFrame] = []
        for df in dfs:
            mats.append(build_matrix(df, args.batch_agg, batch_filter=b))
        mats_by_key[label] = mats

    if args.shared_scale:
        all_mats = []
        for mats in mats_by_key.values():
            all_mats.extend(mats)
        norm, vmin_plot, vmax_plot, step, gmin, gmax_raw = _compute_norm(
            all_mats,
            robust=(not args.no_robust),
            robust_factor=float(args.robust_factor),
            robust_quantile=float(args.robust_quantile),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    stem = out_path.with_suffix("")
    suffix = out_path.suffix

    for label, b in keys:
        mats = mats_by_key[label]
        if not args.shared_scale:
            norm, vmin_plot, vmax_plot, step, gmin, gmax_raw = _compute_norm(
                mats,
                robust=(not args.no_robust),
                robust_factor=float(args.robust_factor),
                robust_quantile=float(args.robust_quantile),
            )

        fig = _make_one_figure(
            specs=specs,
            mats=mats,
            norm=norm,
            vmin_plot=vmin_plot,
            vmax_plot=vmax_plot,
            step=step,
            fig_w=float(args.fig_w),
            fig_h=float(args.fig_h),
            wspace=float(args.wspace),
            hspace=float(args.hspace),
            page_label=None,   # 不在图中写 batch
        )

        tag = "avg" if b is None else f"batch{int(b)}"
        out_i = Path(str(stem) + f"_{tag}" + suffix)
        fig.savefig(out_i, dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        print(f"[OK] Saved: {out_i}")


if __name__ == "__main__":
    main()

