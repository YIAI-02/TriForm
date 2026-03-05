# -*- coding: utf-8 -*-
"""plot_dev_op_bar.py

Single CSV
--------------------------
python plot_dev_op_bar.py \
  --csv ../../algorithms/output/hw_hardware_1npu_2aim/st64/llama_7b_bf16_b16_s64/algo_hefthint/hefthint_prefill-2048xdecode_256_ops_trace.csv \
  --out_dir ../../figs/dev_op_bar/hw_hardware_1npu_2aim/st64/llama_7b_bf16_b16_s64/algo_hefthint \
  --device_label cpu="NPU" \
  --device_label pim="PIM" \
  --label_threshold 0.03 \
  --group_step 0.52 --bar_width 0.22 --inner_gap 0.04

Batch mode: scan a directory of many ops-trace CSVs
---------------------------------------------------
python plot_dev_op_bar.py \
  --csv_dir ../../algorithms/output/hw_hardware_1gpu_2aim/st64/qwen_1.8b_bf16_b1_s64/algo_hefthint \
  --pattern "*_ops_trace.csv" \
  --out_dir ../../figs/dev_op_bar/hw_hardware_1gpu_2aim/st64/qwen_1.8b_bf16_b1_s64/algo_hefthint


  --device_label cpu="NPU" \
  --device_label pim="PIM"

Default behavior in batch mode is to create one sub-directory per CSV under --out_dir.
Use --flat to put everything in one directory and prefix output names by CSV stem.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import colors as mpl_colors


# =========================
# Text normalization helpers
# =========================
def norm_text(s: str) -> str:
    """Normalize operator name for filtering."""
    s = str(s).strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def merge_and_rename_op(op: str) -> str:
    """Merge/rename operator names to reduce x-axis categories."""
    raw = str(op).strip()
    low = raw.lower()

    if low == "qk":
        return "Score"
    if low == "sv":
        return "Context"
    if low == "o":
        return "Projection"

    if low in {"q", "k", "v"}:
        return "QKV"

    if low in {"ffn1", "ffn2", "ffn3", "ffn_w1", "ffn_w2", "ffn_w3"}:
        return "FFN"

    return raw


# =========================
# Device detection + display
# =========================
def _clean_key(s: str) -> str:
    s = str(s).strip()
    if s.lower() == "nan" or s == "":
        return "UNKNOWN"
    return s


def normalize_device_key(raw: str, *, from_col: str) -> str:
    """Normalize raw device identifier to a compact group key.

    - Prefer semantic keys: CPU/GPU/PIM/NPU/TPU/UNKNOWN
    - Otherwise fall back to uppercased cleaned string.
    """
    s = _clean_key(raw)
    low = s.lower()

    # Common buckets (robust to CPU0 / PIMA0 style)
    if "pim" in low or low.startswith("pim") or low.startswith("pima"):
        return "PIM"
    if "cpu" in low or low.startswith("host"):
        return "CPU"
    if "gpu" in low or "cuda" in low or "nvidia" in low:
        return "GPU"
    if "npu" in low:
        return "NPU"
    if "tpu" in low:
        return "TPU"

    # If we are reading from device_type, often it's already a good semantic token (e.g., "cpu")
    # Keep it as-is but uppercased.
    if from_col == "device_type":
        return low.upper()

    # From `device`, it might be something like "CPU0" or "PIMA1"; we already handled common cases.
    # Fall back.
    return s.upper()


def infer_device_groups(
    df: pd.DataFrame,
    device_type_col: str = "device_type",
    device_col: str = "device",
) -> pd.Series:
    """Infer a normalized device group key per row."""
    if device_type_col in df.columns:
        raw = df[device_type_col].astype(str)
        return raw.map(lambda x: normalize_device_key(x, from_col="device_type"))

    if device_col in df.columns:
        raw = df[device_col].astype(str)
        return raw.map(lambda x: normalize_device_key(x, from_col="device"))

    raise ValueError(
        f"Cannot infer device group: neither '{device_type_col}' nor '{device_col}' exists in CSV."
    )


def parse_device_labels(pairs: List[str]) -> Dict[str, str]:
    """Parse repeated `--device_label key=value` arguments."""
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid --device_label '{item}'. Expected format like cpu=\"Ascend 910B\""
            )
        k, v = item.split("=", 1)
        k = k.strip().upper()
        v = v.strip().strip('"').strip("'")
        if not k:
            raise ValueError(f"Invalid --device_label '{item}': empty key")
        if not v:
            raise ValueError(f"Invalid --device_label '{item}': empty value")
        out[k] = v
    return out


def default_device_order(keys: List[str]) -> List[str]:
    """Stable & readable stacking order."""
    priority = ["CPU", "GPU", "NPU", "TPU", "PIM", "UNKNOWN"]
    keys_u = [k.upper() for k in keys]
    # Preserve original casing is not needed; we use uppercase keys everywhere.
    keys_u = list(dict.fromkeys(keys_u))

    def sort_key(k: str) -> Tuple[int, str]:
        try:
            return (priority.index(k), k)
        except ValueError:
            return (len(priority), k)

    return sorted(keys_u, key=sort_key)


# =========================
# Colors
# =========================
@dataclass(frozen=True)
class PhaseColors:
    prefill: str
    decode: str


def _lighten(hex_color: str, amount: float) -> str:
    """Lighten a hex color by mixing with white. amount in [0, 1]."""
    rgb = np.array(mpl_colors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    out = rgb + (white - rgb) * float(amount)
    return mpl_colors.to_hex(out, keep_alpha=False)


def build_color_map(devices: List[str]) -> Dict[str, PhaseColors]:
    """Return {device_key: PhaseColors(prefill, decode)}."""
    # Keep the original aesthetic for known devices when possible.
    # (These are taken from the original script.)
    known = {
        "GPU": PhaseColors(prefill="#A2D091", decode="#D8E69C"),
        "PIM": PhaseColors(prefill="#ADC1E4", decode="#CED8E8"),
        # A reasonable default for CPU; decode is a lighter shade.
        "CPU": PhaseColors(prefill="#F2BE8D", decode=_lighten("#F2BE8D", 0.30)),
        "NPU": PhaseColors(prefill="#D3A3F4", decode=_lighten("#D3A3F4", 0.30)),
        "TPU": PhaseColors(prefill="#9CC9E8", decode=_lighten("#9CC9E8", 0.30)),
        "UNKNOWN": PhaseColors(prefill="#CFCFCF", decode=_lighten("#CFCFCF", 0.20)),
    }

    cmap = plt.get_cmap("tab20")
    out: Dict[str, PhaseColors] = {}
    extra_i = 0
    for k in devices:
        ku = k.upper()
        if ku in known:
            out[ku] = known[ku]
        else:
            base = mpl_colors.to_hex(cmap(extra_i % cmap.N))
            out[ku] = PhaseColors(prefill=base, decode=_lighten(base, 0.30))
            extra_i += 1
    return out


# =========================
# Aggregation
# =========================
def build_metric_tables(
    df: pd.DataFrame,
    phase_col: str,
    op_group_col: str,
    dev_group_col: str,
    phases: List[str],
    op_order: List[str],
    devices: List[str],
    metric: str,
    duration_col: str = "duration",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weights, ratios) for the given metric."""
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

    # Ensure all device columns exist and have stable order.
    for dev in devices:
        if dev not in weights.columns:
            weights[dev] = 0.0
    weights = weights[devices]

    full_index = pd.MultiIndex.from_product(
        [op_order, phases],
        names=[op_group_col, phase_col],
    )
    weights = weights.reindex(full_index, fill_value=0.0)
    ratios = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)
    return weights, ratios


# =========================
# Plotting
# =========================
def annotate_stack(
    ax: plt.Axes,
    x_positions: np.ndarray,
    vals: pd.DataFrame,
    devices: List[str],
    threshold: float = 0.0,
    fontsize: int = 9,
) -> None:
    """Annotate each stacked segment with a percentage."""
    mat = vals[devices].to_numpy(dtype=float)
    for xi, row in zip(x_positions, mat):
        bottom = 0.0
        for v in row:
            if v > threshold:
                ax.text(xi, bottom + v / 2, f"{v * 100:.0f}%", ha="center", va="center", fontsize=fontsize)
            bottom += v


def plot_grouped_stacked_ratio(
    ratios: pd.DataFrame,
    op_order: List[str],
    devices: List[str],
    device_labels: Dict[str, str],
    color_map: Dict[str, PhaseColors],
    out_png: Path,
    metric_title: str,
    group_step: float = 0.55,
    bar_width: float = 0.22,
    inner_gap: float = 0.04,
    label_threshold: float = 0.0,
    title_suffix: str = "",
) -> None:
    phases = ["prefill", "decode"]

    required_width = 2 * bar_width + inner_gap
    if group_step <= required_width:
        raise ValueError(
            f"group_step ({group_step}) too small: must be > 2*bar_width+inner_gap ({required_width})."
        )

    if not isinstance(ratios.index, pd.MultiIndex):
        raise ValueError("ratios must have a MultiIndex (op, phase).")

    phase_level_name = ratios.index.names[-1]

    def phase_slice(phase: str) -> pd.DataFrame:
        if phase not in ratios.index.get_level_values(phase_level_name):
            return pd.DataFrame(0.0, index=op_order, columns=devices)
        return ratios.xs(phase, level=phase_level_name).reindex(op_order, fill_value=0.0)[devices]

    rp = phase_slice("prefill")
    rd = phase_slice("decode")

    x = np.arange(len(op_order)) * group_step
    offset = (bar_width / 2) + (inner_gap / 2)
    pre_x = x - offset
    dec_x = x + offset

    fig_w = max(10, 6 + 1.2 * len(op_order) * group_step)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    edge = "#2B2B2B"
    lw = 0.7

    # Prefill
    bottom = np.zeros(len(op_order), dtype=float)
    for dev in devices:
        c = color_map[dev].prefill
        ax.bar(
            pre_x,
            rp[dev].to_numpy(),
            width=bar_width,
            bottom=bottom,
            color=c,
            edgecolor=edge,
            linewidth=lw,
            label=f"Prefill {device_labels.get(dev, dev)}",
            zorder=3,
        )
        bottom += rp[dev].to_numpy()

    # Decode
    bottom = np.zeros(len(op_order), dtype=float)
    for dev in devices:
        c = color_map[dev].decode
        ax.bar(
            dec_x,
            rd[dev].to_numpy(),
            width=bar_width,
            bottom=bottom,
            color=c,
            edgecolor=edge,
            linewidth=lw,
            label=f"Decode {device_labels.get(dev, dev)}",
            zorder=3,
        )
        bottom += rd[dev].to_numpy()

    # Separator line inside each group
    ax.vlines(x, 0, 1, colors=edge, linewidth=0.9, alpha=0.7, zorder=4)

    annotate_stack(ax, pre_x, rp, devices, threshold=label_threshold, fontsize=9)
    annotate_stack(ax, dec_x, rd, devices, threshold=label_threshold, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(op_order, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Operator type")
    ax.set_ylabel("Proportion (stacked)")

    devices_title = " vs ".join([device_labels.get(d, d) for d in devices])
    title = f"{devices_title} ratio by operator (Prefill vs Decode) — {metric_title}"
    if str(title_suffix).strip():
        title += f" — {title_suffix}"
    ax.set_title(title)

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    if len(x) > 0:
        pad = group_step * 0.9
        ax.set_xlim(x[0] - pad, x[-1] + pad)

    # Legend can get long; wrap into multiple columns.
    ncols = 2 if len(devices) <= 2 else 3
    ax.legend(ncols=ncols, fontsize=9)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def slug_devices(devices: List[str]) -> str:
    return "_".join([d.lower().replace(" ", "-") for d in devices])


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot grouped stacked ratio bars (prefill/decode) for heterogeneous devices per operator, by count and by time.\n"
            "Device types are auto-detected from the CSV; you can override device display names via --device_label."
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="Input CSV path")
    g.add_argument("--csv_dir", help="Directory to scan for many CSVs")

    p.add_argument("--pattern", default="*_ops_trace.csv", help="Glob pattern under --csv_dir (default: '*_ops_trace.csv')")
    p.add_argument("--recursive", action="store_true", help="Recursively scan --csv_dir")
    p.add_argument(
        "--flat",
        action="store_true",
        help=(
            "Write all outputs directly into --out_dir (file names will be prefixed by CSV stem). "
            "Default: create one subdir per CSV."
        ),
    )
    p.add_argument(
        "--no_case_in_title",
        action="store_true",
        help="Do not append case info parsed from filename to plot title (default: append).",
    )
    p.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on the first CSV that fails (default: continue and report errors).",
    )
    p.add_argument("--out_dir", required=True, help="Output directory")

    # Column name overrides
    p.add_argument("--phase_col", default="phase", help="Phase column name (default: phase)")
    p.add_argument("--op_col", default="op", help="Operator column name (default: op)")
    p.add_argument("--device_type_col", default="device_type", help="Device type column name (default: device_type)")
    p.add_argument("--device_col", default="device", help="Device column name (default: device)")
    p.add_argument("--duration_col", default="duration", help="Duration column name for time metric (default: duration)")

    # Device selection / renaming
    p.add_argument(
        "--devices",
        default="",
        help=(
            "Optional comma-separated device keys to include (after normalization), e.g. 'CPU,PIM'. "
            "If omitted, all detected devices are included."
        ),
    )
    p.add_argument(
        "--device_label",
        action="append",
        default=[],
        help=(
            "Override device display name in the plot, repeatable. "
            "Format: --device_label cpu=\"Ascend 910B\" (keys are case-insensitive)."
        ),
    )

    # Plot layout knobs
    p.add_argument("--group_step", type=float, default=0.55, help="Group spacing step (<1 reduces gaps). Default 0.55")
    p.add_argument("--bar_width", type=float, default=0.22, help="Bar width. Default 0.22")
    p.add_argument("--inner_gap", type=float, default=0.04, help="Gap between prefill and decode within a group. Default 0.04")
    p.add_argument("--label_threshold", type=float, default=0.0, help="Skip labeling segments <= threshold. Default 0.0")
    return p.parse_args()


def _strip_known_suffixes(stem: str) -> str:
    s = str(stem)
    s = re.sub(r"(_ops_trace|_ops|_trace)$", "", s, flags=re.IGNORECASE)
    return s


def prefill_decode_from_stem(stem: str) -> Tuple[int, int]:
    """Try to parse (prefill, decode) ints from a filename stem."""
    s = stem.lower()
    s = _strip_known_suffixes(s)
    m = re.search(r"prefill[-_]?([0-9]+)\s*x\s*decode[-_]?([0-9]+)", s)
    if not m:
        m = re.search(r"prefill[-_]?([0-9]+)xdecode[-_]?([0-9]+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Fallback: push unknown to the end.
    return 10**18, 10**18


def case_title_from_csv(csv_path: Path) -> str:
    """Human-friendly case title suffix derived from CSV filename."""
    stem = _strip_known_suffixes(csv_path.stem)
    low = stem.lower()

    # Prefer extracting the part starting from 'prefill' if present.
    m = re.search(r"prefill.*", low)
    tag = stem[m.start() :] if m else stem

    pd_nums = prefill_decode_from_stem(tag)
    if pd_nums[0] < 10**18:
        return f"prefill={pd_nums[0]}, decode={pd_nums[1]}"
    return tag


def discover_csvs(csv_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if not csv_dir.exists():
        raise FileNotFoundError(f"--csv_dir not found: {csv_dir}")
    if not csv_dir.is_dir():
        raise NotADirectoryError(f"--csv_dir is not a directory: {csv_dir}")

    if recursive:
        paths = list(csv_dir.rglob(pattern))
    else:
        paths = list(csv_dir.glob(pattern))
    paths = [p for p in paths if p.is_file()]

    # Sort by (prefill, decode, name) when possible.
    paths.sort(key=lambda p: (*prefill_decode_from_stem(p.stem), p.name))
    return paths


def process_one_csv(
    csv_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    name_prefix: str = "",
    title_suffix: str = "",
) -> None:
    """Run the original single-CSV pipeline for one file."""
    out_dir.mkdir(parents=True, exist_ok=True)


    df = pd.read_csv(csv_path)

    # Validate columns
    for col in [args.phase_col, args.op_col]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'. Existing columns: {list(df.columns)}")

    # Normalize phase
    df[args.phase_col] = df[args.phase_col].astype(str).str.lower().str.strip()
    df.loc[df[args.phase_col] == "profile", args.phase_col] = "prefill"

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

    # Device groups (auto-detect)
    df["dev_group"] = infer_device_groups(df, args.device_type_col, args.device_col)
    df["dev_group"] = df["dev_group"].astype(str).str.upper().str.strip()

    # Optional device filter
    detected_devices = default_device_order(sorted(df["dev_group"].unique().tolist()))
    if args.devices.strip():
        wanted = [x.strip().upper() for x in args.devices.split(",") if x.strip()]
        devices = [d for d in detected_devices if d in set(wanted)]
        missing = [d for d in wanted if d not in set(detected_devices)]
        if missing:
            raise ValueError(
                f"--devices includes unknown device(s): {missing}. Detected devices: {detected_devices}"
            )
        if not devices:
            raise ValueError(f"After applying --devices, no devices left. Detected: {detected_devices}")
        df = df[df["dev_group"].isin(devices)].copy()
    else:
        devices = detected_devices

    if df.empty:
        raise ValueError("No rows left after op/device filtering. Check your CSV content.")

    # Device labels (for plot)
    device_labels = {k.upper(): v for k, v in parse_device_labels(args.device_label).items()}

    # Colors
    color_map = build_color_map(devices)

    # Operator order (stable x-axis)
    op_order = (
        df.groupby("op_group")
          .size()
          .sort_values(ascending=False)
          .index
          .tolist()
    )

    dev_slug = slug_devices(devices)

    # ---- Metric 1: COUNT ----
    weights_c, ratios_c = build_metric_tables(
        df=df,
        phase_col=args.phase_col,
        op_group_col="op_group",
        dev_group_col="dev_group",
        phases=phases,
        op_order=op_order,
        devices=devices,
        metric="count",
        duration_col=args.duration_col,
    )
    summary_c = weights_c.join(ratios_c.add_suffix("_ratio"))
    prefix = f"{name_prefix}__" if name_prefix else ""

    summary_c.to_csv(out_dir / f"{prefix}prefill_decode_summary_count_{dev_slug}.csv", index=True)

    plot_grouped_stacked_ratio(
        ratios=ratios_c,
        op_order=op_order,
        devices=devices,
        device_labels=device_labels,
        color_map=color_map,
        out_png=out_dir / f"{prefix}prefill_decode_ratio_by_count_{dev_slug}.png",
        metric_title="Count",
        group_step=args.group_step,
        bar_width=args.bar_width,
        inner_gap=args.inner_gap,
        label_threshold=args.label_threshold,
        title_suffix=title_suffix,
    )

    # ---- Metric 2: TIME ----
    weights_t, ratios_t = build_metric_tables(
        df=df,
        phase_col=args.phase_col,
        op_group_col="op_group",
        dev_group_col="dev_group",
        phases=phases,
        op_order=op_order,
        devices=devices,
        metric="time",
        duration_col=args.duration_col,
    )
    summary_t = weights_t.join(ratios_t.add_suffix("_ratio"))
    summary_t.to_csv(out_dir / f"{prefix}prefill_decode_summary_time_{dev_slug}.csv", index=True)

    plot_grouped_stacked_ratio(
        ratios=ratios_t,
        op_order=op_order,
        devices=devices,
        device_labels=device_labels,
        color_map=color_map,
        out_png=out_dir / f"{prefix}prefill_decode_ratio_by_time_{dev_slug}.png",
        metric_title=f"Time (sum {args.duration_col})",
        group_step=args.group_step,
        bar_width=args.bar_width,
        inner_gap=args.inner_gap,
        label_threshold=args.label_threshold,
        title_suffix=title_suffix,
    )

    print("Done:", csv_path)
    print("Detected devices:", detected_devices)
    print("Plotted devices:", devices)
    print("Saved figures:")
    print(" -", (out_dir / f"{prefix}prefill_decode_ratio_by_count_{dev_slug}.png").resolve())
    print(" -", (out_dir / f"{prefix}prefill_decode_ratio_by_time_{dev_slug}.png").resolve())
    print("Saved summaries:")
    print(" -", (out_dir / f"{prefix}prefill_decode_summary_count_{dev_slug}.csv").resolve())
    print(" -", (out_dir / f"{prefix}prefill_decode_summary_time_{dev_slug}.csv").resolve())


def main() -> None:
    args = parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover CSV list
    if args.csv:
        csv_list = [Path(args.csv)]
    else:
        csv_list = discover_csvs(Path(args.csv_dir), args.pattern, args.recursive)
        if not csv_list:
            raise FileNotFoundError(
                f"No CSV matched under {args.csv_dir} with pattern '{args.pattern}' (recursive={args.recursive})"
            )

    total = len(csv_list)
    ok = 0
    failed: List[Tuple[Path, str]] = []

    for i, csv_path in enumerate(csv_list, 1):
        try:
            if args.flat:
                case_out_dir = out_root
                name_prefix = csv_path.stem
            else:
                case_out_dir = out_root / csv_path.stem
                name_prefix = ""

            title_suffix = "" if args.no_case_in_title else case_title_from_csv(csv_path)
            print(f"[{i}/{total}] {csv_path}")
            process_one_csv(
                csv_path=csv_path,
                out_dir=case_out_dir,
                args=args,
                name_prefix=name_prefix,
                title_suffix=title_suffix,
            )
            ok += 1
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failed.append((csv_path, msg))
            print(f"[ERROR] {csv_path}: {msg}")
            if args.fail_fast:
                raise

    print("\n==== Summary ====")
    print(f"Total: {total}, OK: {ok}, Failed: {len(failed)}")
    if failed:
        for p, m in failed:
            print(" -", p, "=>", m)


if __name__ == "__main__":
    main()
