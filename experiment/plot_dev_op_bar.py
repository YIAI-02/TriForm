# -*- coding: utf-8 -*-
"""
python plot_dev_op_bar.py \
  --csv ../algorithms/output/evaluate_single_test/hardware_config_gpu_7aim.json/llama_7b_int8_b1_s64/algo_hefthint/hefthint_4096x4096_ops_trace.csv \
  --out_dir ../figs/dev_op_bar/

  --label_threshold 0.03
  --group_step 0.52 --bar_width 0.22 --inner_gap 0.04
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# =========================
# Color config (given)
# =========================
PREFILL_GPU_COLOR = "#A2D091"
DECODE_GPU_COLOR  = "#D8E69C"
PREFILL_PIM_COLOR = "#ADC1E4"
DECODE_PIM_COLOR  = "#CED8E8"


# =========================
# Text normalization helpers
# =========================
def norm_text(s: str) -> str:
    """
    Normalize operator name for filtering:
    - lower
    - '_' -> ' '
    - collapse multiple spaces
    NOTE: CamelCase like 'AllReduce' becomes 'allreduce' (fine).
    """
    s = str(s).strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def merge_and_rename_op(op: str) -> str:
    """
    Merge/rename operator names:
    - QK -> Score
    - SV -> context
    - O  -> projection
    - Q/K/V -> QKV
    - ffn1/2/3 or ffn_w1/2/3 or FFN_W1/2/3 -> FFN
    Otherwise keep original text.
    """
    raw = str(op).strip()
    low = raw.lower()

    # rename
    if low == "qk":
        return "Score"
    if low == "sv":
        return "Context"
    if low == "o":
        return "Projection"

    # merge Q/K/V
    if low in {"q", "k", "v"}:
        return "QKV"

    # merge FFN variants
    if low in {"ffn1", "ffn2", "ffn3", "ffn_w1", "ffn_w2", "ffn_w3"}:
        return "FFN"

    return raw


def infer_device_group(df: pd.DataFrame,
                       device_type_col: str = "device_type",
                       device_col: str = "device") -> pd.Series:
    """
    Return a Series with values 'GPU' or 'PIM'.

    Priority:
    - device_type column if exists (contains 'pim' -> PIM, else GPU)
    - else device column (contains 'PIM' -> PIM, else GPU)
    """
    if device_type_col in df.columns:
        dt = df[device_type_col].astype(str).str.lower()
        return np.where(dt.str.contains("pim", na=False), "PIM", "GPU")

    if device_col in df.columns:
        dv = df[device_col].astype(str).str.upper()
        return np.where(dv.str.contains("PIM", na=False), "PIM", "GPU")

    raise ValueError(
        f"Cannot infer device group: neither '{device_type_col}' nor '{device_col}' exists in CSV."
    )


# =========================
# Aggregation
# =========================
def build_metric_tables(
    df: pd.DataFrame,
    phase_col: str,
    op_group_col: str,
    dev_group_col: str,
    phases: list[str],
    op_order: list[str],
    metric: str,
    duration_col: str = "duration",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
    - weights table (GPU/PIM values) per (op, phase)
    - ratios table (row-normalized to sum=1) per (op, phase)

    metric:
    - 'count': weight = 1 per row
    - 'time' : weight = duration (sum)
    """
    d = df[df[phase_col].isin(phases)].copy()

    if metric == "count":
        d["__w__"] = 1.0
    elif metric == "time":
        if duration_col not in d.columns:
            raise ValueError(f"metric='time' requires duration column '{duration_col}' in CSV.")
        d["__w__"] = pd.to_numeric(d[duration_col], errors="coerce").fillna(0.0)
    else:
        raise ValueError("metric must be 'count' or 'time'")

    weights = (
        d.groupby([op_group_col, phase_col, dev_group_col])["__w__"]
         .sum()
         .unstack(fill_value=0.0)
    )

    # Ensure both columns exist
    for col in ["GPU", "PIM"]:
        if col not in weights.columns:
            weights[col] = 0.0
    weights = weights[["GPU", "PIM"]]

    # Ensure all (op, phase) combos exist in stable order
    full_index = pd.MultiIndex.from_product(
        [op_order, phases],
        names=[op_group_col, phase_col]
    )
    weights = weights.reindex(full_index, fill_value=0.0)

    ratios = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)
    return weights, ratios


# =========================
# Plotting
# =========================
def annotate_segments(
    ax: plt.Axes,
    x_positions: np.ndarray,
    gpu_vals: np.ndarray,
    pim_vals: np.ndarray,
    threshold: float = 0.0,
    fontsize: int = 9,
) -> None:
    """
    Annotate GPU% in the middle of GPU segment, and PIM% in middle of PIM segment.
    threshold: skip labels for segments <= threshold (e.g., 0.01 to skip <=1%)
    """
    for x, g, p in zip(x_positions, gpu_vals, pim_vals):
        if g > threshold:
            ax.text(x, g / 2, f"{g * 100:.0f}%", ha="center", va="center", fontsize=fontsize)
        if p > threshold:
            ax.text(x, g + p / 2, f"{p * 100:.0f}%", ha="center", va="center", fontsize=fontsize)


def plot_grouped_stacked_ratio(
    ratios: pd.DataFrame,
    op_order: list[str],
    out_png: Path,
    metric_title: str,
    group_step: float = 0.55,   # < 1 => reduce inter-group gaps
    bar_width: float = 0.22,    # narrow bars
    inner_gap: float = 0.04,    # gap between prefill and decode within group
    label_threshold: float = 0.0,
) -> None:
    """
    One figure:
    - X: operators
    - Two bars per operator: prefill, decode
    - Each bar stacked: GPU (bottom) + PIM (top)
    - Ratio bars always sum to 1
    """
    phases = ["prefill", "decode"]

    # Safety: avoid overlapping groups
    required_width = 2 * bar_width + inner_gap
    if group_step <= required_width:
        raise ValueError(
            f"group_step ({group_step}) too small: must be > 2*bar_width+inner_gap ({required_width})."
        )

    # ratios index: MultiIndex [op_group, phase]
    # Build per-phase matrices indexed by op
    if not isinstance(ratios.index, pd.MultiIndex):
        raise ValueError("ratios must have a MultiIndex (op, phase).")

    phase_level_name = ratios.index.names[-1]  # expects phase is one of the index levels

    def phase_slice(phase: str) -> pd.DataFrame:
        if phase not in ratios.index.get_level_values(phase_level_name):
            return pd.DataFrame(0.0, index=op_order, columns=["GPU", "PIM"])
        return ratios.xs(phase, level=phase_level_name).reindex(op_order, fill_value=0.0)

    rp = phase_slice("prefill")
    rd = phase_slice("decode")

    # Compress group spacing
    x = np.arange(len(op_order)) * group_step
    offset = (bar_width / 2) + (inner_gap / 2)
    pre_x = x - offset
    dec_x = x + offset

    # Figure width: scale by number of ops and spacing
    fig_w = max(10, 6 + 1.2 * len(op_order) * group_step)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    edge = "#2B2B2B"
    lw = 0.7

    # Prefill bars (stacked)
    ax.bar(pre_x, rp["GPU"].to_numpy(), width=bar_width,
           color=PREFILL_GPU_COLOR, edgecolor=edge, linewidth=lw, label="Prefill GPU", zorder=3)
    ax.bar(pre_x, rp["PIM"].to_numpy(), width=bar_width, bottom=rp["GPU"].to_numpy(),
           color=PREFILL_PIM_COLOR, edgecolor=edge, linewidth=lw, label="Prefill PIM", zorder=3)

    # Decode bars (stacked)
    ax.bar(dec_x, rd["GPU"].to_numpy(), width=bar_width,
           color=DECODE_GPU_COLOR, edgecolor=edge, linewidth=lw, label="Decode GPU", zorder=3)
    ax.bar(dec_x, rd["PIM"].to_numpy(), width=bar_width, bottom=rd["GPU"].to_numpy(),
           color=DECODE_PIM_COLOR, edgecolor=edge, linewidth=lw, label="Decode PIM", zorder=3)

    # Clear separator line inside each group (between prefill & decode bars)
    ax.vlines(x, 0, 1, colors=edge, linewidth=0.9, alpha=0.7, zorder=4)

    # Percentage labels in the middle of each stacked segment
    annotate_segments(ax, pre_x, rp["GPU"].to_numpy(), rp["PIM"].to_numpy(),
                      threshold=label_threshold, fontsize=9)
    annotate_segments(ax, dec_x, rd["GPU"].to_numpy(), rd["PIM"].to_numpy(),
                      threshold=label_threshold, fontsize=9)

    # Axes / formatting
    ax.set_xticks(x)
    ax.set_xticklabels(op_order, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Operator type")
    ax.set_ylabel("Proportion (stacked)")
    ax.set_title(f"GPU vs PIM ratio by operator (Prefill vs Decode) — {metric_title}")

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Reduce outer whitespace
    if len(x) > 0:
        pad = group_step * 0.9
        ax.set_xlim(x[0] - pad, x[-1] + pad)

    ax.legend(ncols=2)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot grouped stacked ratio bars (prefill/decode) for GPU vs PIM per operator, by count and by time."
    )
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--out_dir", required=True, help="Output directory")

    # Column name overrides
    p.add_argument("--phase_col", default="phase", help="Phase column name (default: phase)")
    p.add_argument("--op_col", default="op", help="Operator column name (default: op)")
    p.add_argument("--device_type_col", default="device_type", help="Device type column name (default: device_type)")
    p.add_argument("--device_col", default="device", help="Device column name (default: device)")
    p.add_argument("--duration_col", default="duration", help="Duration column name for time metric (default: duration)")

    # Plot layout knobs
    p.add_argument("--group_step", type=float, default=0.55, help="Group spacing step (<1 reduces gaps). Default 0.55")
    p.add_argument("--bar_width", type=float, default=0.22, help="Bar width (narrower columns). Default 0.22")
    p.add_argument("--inner_gap", type=float, default=0.04, help="Gap between prefill and decode within a group. Default 0.04")
    p.add_argument("--label_threshold", type=float, default=0.0, help="Skip labeling segments <= threshold. Default 0.0")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Validate columns
    for col in [args.phase_col, args.op_col]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'. Existing columns: {list(df.columns)}")

    # Normalize phase
    df[args.phase_col] = df[args.phase_col].astype(str).str.lower().str.strip()
    # Map 'profile' -> 'prefill' if present
    df.loc[df[args.phase_col] == "profile", args.phase_col] = "prefill"

    # Filter phases
    phases = ["prefill", "decode"]
    df = df[df[args.phase_col].isin(phases)].copy()
    if df.empty:
        raise ValueError("No rows left after filtering phases to {prefill, decode}.")

    # Operator filtering (remove unwanted ops)
    exclude_norm = {"k write", "v write", "identity", "all reduce", "allreduce"}
    df["op_norm"] = df[args.op_col].map(norm_text)
    df = df[~df["op_norm"].isin(exclude_norm)].copy()

    # Merge/rename ops
    df["op_group"] = df[args.op_col].map(merge_and_rename_op)

    # Device group
    df["dev_group"] = infer_device_group(df, args.device_type_col, args.device_col)

    if df.empty:
        raise ValueError("No rows left after op/device filtering. Check your CSV content.")

    # Operator order (use total COUNT across both phases for stable x-axis across both plots)
    op_order = (
        df.groupby("op_group")
          .size()
          .sort_values(ascending=False)
          .index
          .tolist()
    )

    # ---- Metric 1: COUNT ----
    weights_c, ratios_c = build_metric_tables(
        df=df,
        phase_col=args.phase_col,
        op_group_col="op_group",
        dev_group_col="dev_group",
        phases=phases,
        op_order=op_order,
        metric="count",
        duration_col=args.duration_col,
    )
    summary_c = weights_c.join(ratios_c.add_suffix("_ratio"))
    summary_c.to_csv(out_dir / "prefill_decode_summary_count.csv", index=True)

    plot_grouped_stacked_ratio(
        ratios=ratios_c,
        op_order=op_order,
        out_png=out_dir / "prefill_decode_gpu_pim_ratio_by_count.png",
        metric_title="Count",
        group_step=args.group_step,
        bar_width=args.bar_width,
        inner_gap=args.inner_gap,
        label_threshold=args.label_threshold,
    )

    # ---- Metric 2: TIME (sum duration) ----
    weights_t, ratios_t = build_metric_tables(
        df=df,
        phase_col=args.phase_col,
        op_group_col="op_group",
        dev_group_col="dev_group",
        phases=phases,
        op_order=op_order,
        metric="time",
        duration_col=args.duration_col,
    )
    summary_t = weights_t.join(ratios_t.add_suffix("_ratio"))
    summary_t.to_csv(out_dir / "prefill_decode_summary_time.csv", index=True)

    plot_grouped_stacked_ratio(
        ratios=ratios_t,
        op_order=op_order,
        out_png=out_dir / "prefill_decode_gpu_pim_ratio_by_time.png",
        metric_title=f"Time (sum {args.duration_col})",
        group_step=args.group_step,
        bar_width=args.bar_width,
        inner_gap=args.inner_gap,
        label_threshold=args.label_threshold,
    )

    print("Done.")
    print("Saved figures:")
    print(" -", (out_dir / "prefill_decode_gpu_pim_ratio_by_count.png").resolve())
    print(" -", (out_dir / "prefill_decode_gpu_pim_ratio_by_time.png").resolve())
    print("Saved summaries:")
    print(" -", (out_dir / "prefill_decode_summary_count.csv").resolve())
    print(" -", (out_dir / "prefill_decode_summary_time.csv").resolve())


if __name__ == "__main__":
    main()

