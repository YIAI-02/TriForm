#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single trace:

python plot_trace_gantt.py \
  --base_dir ../algorithms/output/evaluate_single_test/hardware_1npu_2aim/llama_7b_fp16_b16_s64\
  --policy algo_pd \
  --length prefill-2048xdecode_512 \
  --out_dir ../figs/gantt/evaluate_single_test/hardware_1npu_2aim/llama_7b_fp16_b16_s64

python plot_trace_gantt.py \
  --compare \
  ../../algorithms/output/hw_hardware_1npu_2aim/st64/llama_7b_bf16_b16_s64/algo_heft/heft_prefill-256xdecode_256_ops_trace.csv \
  ../../algorithms/output/hw_hardware_1npu_2aim/st64/llama_7b_bf16_b16_s64/algo_pd/pd_prefill-256xdecode_256_comms_trace.csv \
  --out_dir ../../figs/gantt_comparison/hw_hardware_1npu_2aim/st64/llama_7b_bf16_b16_s64 \
  --time_unit ms \
  --fig_w 25 \
  --comm_lane_mode aggregate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ---------------------------
# Style / grouping
# ---------------------------

OP_GROUP_ORDER = ["FFN_W", "ATTN_CORE", "QKVO", "OTHER"]

OP_GROUP_STYLE = {
    "FFN_W": {"label": "FFN_W1/W2/W3", "color": "#2e7277"},
    "ATTN_CORE": {"label": "QK/SV/Softmax/K_write/V_write", "color": "#98d98e"},
    "QKVO": {"label": "Q/K/V/O", "color": "#6C92DA"},
    "OTHER": {"label": "Other", "color": "#dbed93"},
}

DATA_XFER_STYLE = {"label": "data transfer", "color": "#808080", "alpha": 0.40}
DEVICE_TYPE_ORDER = {"npu": 0, "pim": 1, "gpu": 2, "cpu": 3, "comm": 99}


# ---------------------------
# Helpers
# ---------------------------


def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[_\s]+", " ", s)
    return s


def _safe_slug(s: str, *, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


_num_tail_re = re.compile(r"^(.*?)(\d+)$")


def _natural_key(s: str) -> Tuple[str, int]:
    s = str(s)
    m = _num_tail_re.match(s)
    if m:
        return (m.group(1), int(m.group(2)))
    return (s, -1)


LEN_PATTERNS = [
    re.compile(r"prefill-\d+xdecode_\d+", re.IGNORECASE),
    re.compile(r"\d+x\d+", re.IGNORECASE),
]


def infer_pair_paths(any_csv_path: str) -> Tuple[Path, Path]:
    p = Path(any_csv_path).expanduser().resolve()
    name = p.name
    if name.endswith("_ops_trace.csv"):
        ops = p
        comms = p.with_name(name.replace("_ops_trace.csv", "_comms_trace.csv"))
    elif name.endswith("_comms_trace.csv"):
        comms = p
        ops = p.with_name(name.replace("_comms_trace.csv", "_ops_trace.csv"))
    else:
        raise ValueError(f"CSV must end with _ops_trace.csv or _comms_trace.csv: {p}")
    return ops, comms


def infer_label_from_path(any_csv_path: str) -> str:
    p = Path(any_csv_path).expanduser().resolve()
    stem = p.name
    for suf in ["_ops_trace.csv", "_comms_trace.csv", ".csv"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return _safe_slug(stem)


def _infer_type_from_name(name: str) -> str:
    m = re.match(r"([A-Za-z]+)", str(name).strip())
    return m.group(1).lower() if m else ""


def _empty_comms_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["phase", "src", "dst", "start", "end", "duration", "tag", "src_type", "dst_type"]
    )


# ---------------------------
# Data normalization
# ---------------------------


def classify_op_group(op: str) -> str:
    c = _canon(op)
    if re.fullmatch(r"ffn w[123]", c):
        return "FFN_W"
    if c in {"qk", "sv", "softmax", "k write", "v write"}:
        return "ATTN_CORE"
    if c in {"q", "k", "v", "o"}:
        return "QKVO"
    return "OTHER"


@dataclass
class TraceRun:
    label: str
    ops_path: Path
    comms_path: Optional[Path]
    ops_raw: pd.DataFrame
    comms_raw: pd.DataFrame

    @property
    def ops_plot_base(self) -> pd.DataFrame:
        """Compute events only. COMM placeholder rows belong to comms, not compute lanes."""
        if self.ops_raw.empty:
            return self.ops_raw.copy()
        return self.ops_raw[self.ops_raw["device_type"] != "comm"].copy()


@dataclass
class DecodeWindows:
    n_tokens: int
    start_method: str
    active_method: str
    cadence_method: str
    marker_node: Optional[str]
    token_starts: List[float]
    active_windows: List[Tuple[float, float]]
    cadence_windows: List[Tuple[float, float]]
    debug: Dict[str, object] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "n_tokens": self.n_tokens,
            "start_method": self.start_method,
            "active_method": self.active_method,
            "cadence_method": self.cadence_method,
            "marker_node": self.marker_node,
            "token_starts": self.token_starts,
            "active_windows": [[float(a), float(b)] for a, b in self.active_windows],
            "cadence_windows": [[float(a), float(b)] for a, b in self.cadence_windows],
            "debug": self.debug,
        }


@dataclass
class TokenSelection:
    view: str  # decode_token_active or decode_token_latency
    token_index: int
    n_tokens: int
    label: str
    method: str
    window_start: float
    window_end: float
    ops: pd.DataFrame
    comms: pd.DataFrame


REQUIRED_OPS_COLS = ["phase", "op", "device", "device_type", "start", "end"]
REQUIRED_COMMS_COLS = ["phase", "src", "dst", "start", "end"]


def _normalize_ops_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_OPS_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"ops trace missing columns {missing} in {source}")

    out = df.copy()
    out["phase"] = out["phase"].astype(str).str.lower()
    out["op"] = out["op"].astype(str)
    out["device"] = out["device"].astype(str)
    out["device_type"] = out["device_type"].astype(str).str.lower()
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    if "node_id" not in out.columns:
        out["node_id"] = ""
    else:
        out["node_id"] = out["node_id"].astype(str)
    out = out.dropna(subset=["start", "end"]).copy()
    out["duration"] = out["end"] - out["start"]
    out = out[out["duration"] >= 0].copy()  # keep zero-duration rows; they matter for detection / vlines
    out["op_group"] = out["op"].map(classify_op_group)
    return out.reset_index(drop=True)


def _normalize_comms_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COMMS_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"comms trace missing columns {missing} in {source}")

    out = df.copy()
    out["phase"] = out["phase"].astype(str).str.lower()
    out["src"] = out["src"].astype(str)
    out["dst"] = out["dst"].astype(str)
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    if "src_type" not in out.columns:
        out["src_type"] = out["src"].map(_infer_type_from_name)
    else:
        out["src_type"] = out["src_type"].astype(str).str.lower()
    if "dst_type" not in out.columns:
        out["dst_type"] = out["dst"].map(_infer_type_from_name)
    else:
        out["dst_type"] = out["dst_type"].astype(str).str.lower()
    if "tag" not in out.columns:
        out["tag"] = ""
    else:
        out["tag"] = out["tag"].astype(str)
    out = out.dropna(subset=["start", "end"]).copy()
    out["duration"] = out["end"] - out["start"]
    out = out[out["duration"] >= 0].copy()  # keep zero-duration comms as spikes if they exist
    return out.reset_index(drop=True)


def load_trace_run(path: str, *, label: Optional[str] = None, allow_missing_comms: bool = True) -> TraceRun:
    ops_path, comms_path = infer_pair_paths(path)
    if not ops_path.is_file():
        raise FileNotFoundError(f"ops trace not found: {ops_path}")

    ops_raw = _normalize_ops_df(pd.read_csv(ops_path), str(ops_path))
    if comms_path.is_file():
        comms_raw = _normalize_comms_df(pd.read_csv(comms_path), str(comms_path))
        comms_path_val: Optional[Path] = comms_path
    else:
        if allow_missing_comms:
            comms_raw = _empty_comms_df()
            comms_path_val = None
        else:
            raise FileNotFoundError(f"paired comms trace not found: {comms_path}")

    return TraceRun(
        label=label or infer_label_from_path(str(ops_path)),
        ops_path=ops_path,
        comms_path=comms_path_val,
        ops_raw=ops_raw,
        comms_raw=comms_raw,
    )


# ---------------------------
# Stage bounds / slicing
# ---------------------------


def _phase_union_events(run: TraceRun, phase: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    ops = run.ops_plot_base
    if not ops.empty:
        part = ops[ops["phase"] == phase]
        if not part.empty:
            frames.append(part[["start", "end"]].copy())
    if not run.comms_raw.empty:
        part = run.comms_raw[run.comms_raw["phase"] == phase]
        if not part.empty:
            frames.append(part[["start", "end"]].copy())
    if not frames:
        return pd.DataFrame(columns=["start", "end"])
    union = pd.concat(frames, ignore_index=True)
    union["duration"] = union["end"] - union["start"]
    return union


def phase_bounds(run: TraceRun, phase: str) -> Optional[Tuple[float, float]]:
    union = _phase_union_events(run, phase)
    if union.empty:
        return None
    return float(union["start"].min()), float(union["end"].max())


def _clip_time_slice(df: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df[(df["start"] < t1) & (df["end"] >= t0)].copy()
    if out.empty:
        return out
    out["start"] = out["start"].clip(lower=t0, upper=t1) - t0
    out["end"] = out["end"].clip(lower=t0, upper=t1) - t0
    out["duration"] = out["end"] - out["start"]
    out = out[out["duration"] >= 0].copy()
    return out.reset_index(drop=True)


def _shift_rows(df: pd.DataFrame, base: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["start"] = out["start"] - base
    out["end"] = out["end"] - base
    out["duration"] = out["end"] - out["start"]
    return out.reset_index(drop=True)


def select_phase_view(run: TraceRun, phase: str, *, label: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[float, float], str]:
    bounds = phase_bounds(run, phase)
    if bounds is None:
        return run.ops_plot_base.iloc[0:0].copy(), run.comms_raw.iloc[0:0].copy(), (0.0, 0.0), label or phase
    t0, t1 = bounds
    ops = _clip_time_slice(run.ops_plot_base[run.ops_plot_base["phase"] == phase], t0, t1)
    comms = _clip_time_slice(run.comms_raw[run.comms_raw["phase"] == phase], t0, t1)
    return ops, comms, (t0, t1), label or phase


# ---------------------------
# Decode token detection
# ---------------------------


def _detect_marker_token_starts(decode_ops: pd.DataFrame) -> Tuple[List[float], Optional[str], Dict[str, object]]:
    debug: Dict[str, object] = {}
    if decode_ops.empty or "node_id" not in decode_ops.columns:
        return [], None, debug

    nid = decode_ops["node_id"].astype(str)
    op_ln = decode_ops["op"].map(_canon) == "ln"
    layer0 = nid.str.match(r"^L0(?:_|$)", case=False, na=False)

    candidate_masks: List[Tuple[str, pd.Series]] = [
        ("layer0_exact_ln", op_ln & nid.str.match(r"^L0_LN(?:_?\d+)?$", case=False, na=False)),
        ("layer0_op_ln", op_ln & layer0),
        ("layer0_node_contains_ln", layer0 & nid.str.match(r"^L0_.*LN(?:_?\d+)?$", case=False, na=False)),
    ]

    debug_counts: Dict[str, Dict[str, int]] = {}
    for mask_name, mask in candidate_masks:
        vc = decode_ops.loc[mask, "node_id"].value_counts()
        debug_counts[mask_name] = {str(k): int(v) for k, v in vc.head(10).items()}
        vc = vc[vc >= 2]
        if not vc.empty:
            marker_node = str(vc.index[0])
            starts = sorted(map(float, decode_ops.loc[decode_ops["node_id"] == marker_node, "start"].unique().tolist()))
            debug["candidate_counts"] = debug_counts
            return starts, marker_node, debug

    debug["candidate_counts"] = debug_counts
    return [], None, debug


def _detect_rowcount_token_starts(ops_raw: pd.DataFrame) -> Tuple[List[float], Dict[str, object]]:
    debug: Dict[str, object] = {}
    prefill = ops_raw[ops_raw["phase"] == "prefill"]
    decode = ops_raw[ops_raw["phase"] == "decode"]
    if prefill.empty or decode.empty:
        return [], debug

    n_prefill = int(len(prefill))
    n_decode = int(len(decode))
    debug["prefill_rows"] = n_prefill
    debug["decode_rows"] = n_decode

    if n_prefill <= 0 or n_decode <= 0:
        return [], debug
    if n_decode % n_prefill != 0:
        debug["divisible"] = False
        return [], debug

    debug["divisible"] = True
    n_tokens = n_decode // n_prefill
    decode_sorted = decode.sort_values(["start", "end"]).reset_index(drop=True)
    starts: List[float] = []
    for i in range(n_tokens):
        chunk = decode_sorted.iloc[i * n_prefill : (i + 1) * n_prefill]
        if chunk.empty:
            continue
        starts.append(float(chunk["start"].min()))
    debug["n_tokens"] = len(starts)
    return starts, debug


def _detect_gap_token_starts(ops_raw: pd.DataFrame) -> Tuple[List[float], Dict[str, object]]:
    debug: Dict[str, object] = {}
    decode = ops_raw[ops_raw["phase"] == "decode"].sort_values(["start", "end"]).reset_index(drop=True)
    if decode.empty:
        return [], debug
    starts = decode["start"].to_numpy(dtype=float)
    if len(starts) == 1:
        return [float(starts[0])], {"n_tokens": 1, "reason": "single_decode_row"}

    diffs = np.diff(starts)
    med = float(np.median(diffs))
    q99 = float(np.quantile(diffs, 0.99))
    threshold = max(med * 50.0, q99 * 5.0, med + 1e-9)
    boundary_idx = np.where(diffs > threshold)[0]
    starts_out = [float(starts[0])] + [float(starts[i + 1]) for i in boundary_idx.tolist()]
    debug.update(
        {
            "median_diff": med,
            "q99_diff": q99,
            "threshold": threshold,
            "boundary_idx": boundary_idx.tolist(),
            "n_tokens": len(starts_out),
        }
    )
    return starts_out, debug


def _union_decode_events_for_active(run: TraceRun) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    ops = run.ops_plot_base
    if not ops.empty:
        part = ops[ops["phase"] == "decode"]
        if not part.empty:
            frames.append(part[["start", "end"]].assign(source="ops"))
    if not run.comms_raw.empty:
        part = run.comms_raw[run.comms_raw["phase"] == "decode"]
        if not part.empty:
            frames.append(part[["start", "end"]].assign(source="comms"))
    if not frames:
        return pd.DataFrame(columns=["start", "end", "source"])
    union = pd.concat(frames, ignore_index=True)
    union["duration"] = union["end"] - union["start"]
    return union.sort_values(["start", "end"]).reset_index(drop=True)


def _windows_from_starts(token_starts: Sequence[float], stage_end: float, union_events: pd.DataFrame) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    starts = list(map(float, token_starts))
    if not starts:
        return [], []

    cadence: List[Tuple[float, float]] = []
    active: List[Tuple[float, float]] = []

    for i, s in enumerate(starts):
        next_s = float(starts[i + 1]) if i + 1 < len(starts) else float(stage_end)
        cadence.append((float(s), float(next_s)))

        if union_events.empty:
            active.append((float(s), float(next_s)))
            continue

        if i + 1 < len(starts):
            bucket = union_events[(union_events["start"] >= s) & (union_events["start"] < next_s)]
        else:
            bucket = union_events[union_events["start"] >= s]

        if bucket.empty:
            active.append((float(s), float(next_s)))
        else:
            a0 = float(bucket["start"].min())
            a1 = float(bucket["end"].max())
            active.append((a0, a1))

    return active, cadence


def detect_decode_windows(run: TraceRun) -> DecodeWindows:
    decode_ops_raw = run.ops_raw[run.ops_raw["phase"] == "decode"].copy()
    if decode_ops_raw.empty:
        return DecodeWindows(
            n_tokens=0,
            start_method="no-decode",
            active_method="no-decode",
            cadence_method="no-decode",
            marker_node=None,
            token_starts=[],
            active_windows=[],
            cadence_windows=[],
            debug={},
        )

    bounds = phase_bounds(run, "decode")
    if bounds is None:
        stage_end = float(decode_ops_raw["end"].max())
    else:
        _, stage_end = bounds

    marker_starts, marker_node, marker_debug = _detect_marker_token_starts(decode_ops_raw)
    rowcount_starts, rowcount_debug = _detect_rowcount_token_starts(run.ops_raw)
    gap_starts, gap_debug = _detect_gap_token_starts(run.ops_raw)

    # Priority: marker -> rowcount -> gap.
    if marker_starts:
        token_starts = marker_starts
        start_method = f"marker:{marker_node}"
    elif rowcount_starts:
        token_starts = rowcount_starts
        start_method = "rowcount"
    else:
        token_starts = gap_starts
        start_method = "gap"

    union_events = _union_decode_events_for_active(run)
    active_windows, cadence_windows = _windows_from_starts(token_starts, stage_end, union_events)

    return DecodeWindows(
        n_tokens=len(token_starts),
        start_method=start_method,
        active_method=f"{start_method}->bucketed_active",
        cadence_method=f"{start_method}->start_to_next_start",
        marker_node=marker_node,
        token_starts=list(map(float, token_starts)),
        active_windows=active_windows,
        cadence_windows=cadence_windows,
        debug={
            "marker_candidate": {
                "starts": marker_starts,
                "marker_node": marker_node,
                **marker_debug,
            },
            "rowcount_candidate": {"starts": rowcount_starts, **rowcount_debug},
            "gap_candidate": {"starts": gap_starts, **gap_debug},
            "stage_end": stage_end,
        },
    )


# ---------------------------
# Token event selection
# ---------------------------


def _bucket_by_token_start(df: pd.DataFrame, token_starts: Sequence[float], token_index: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    starts = list(map(float, token_starts))
    if not starts:
        return df.iloc[0:0].copy()
    token_index = max(0, min(int(token_index), len(starts) - 1))
    s = starts[token_index]
    if token_index + 1 < len(starts):
        next_s = starts[token_index + 1]
        out = df[(df["start"] >= s) & (df["start"] < next_s)].copy()
    else:
        out = df[df["start"] >= s].copy()
    return out.reset_index(drop=True)


def select_decode_token_view(run: TraceRun, windows: DecodeWindows, token_index: int, view: str) -> Optional[TokenSelection]:
    if windows.n_tokens <= 0:
        return None
    idx = max(0, min(int(token_index), windows.n_tokens - 1))
    token_label = f"token{idx + 1}of{windows.n_tokens}"

    decode_ops = run.ops_plot_base[run.ops_plot_base["phase"] == "decode"].copy()
    decode_comms = run.comms_raw[run.comms_raw["phase"] == "decode"].copy()

    if view == "decode_token_latency":
        t0, t1 = windows.cadence_windows[idx]
        ops = _clip_time_slice(decode_ops, t0, t1)
        comms = _clip_time_slice(decode_comms, t0, t1)
        return TokenSelection(
            view=view,
            token_index=idx,
            n_tokens=windows.n_tokens,
            label=token_label,
            method=windows.cadence_method,
            window_start=float(t0),
            window_end=float(t1),
            ops=ops,
            comms=comms,
        )

    if view == "decode_token_active":
        bucket_ops = _bucket_by_token_start(decode_ops, windows.token_starts, idx)
        bucket_comms = _bucket_by_token_start(decode_comms, windows.token_starts, idx)
        a0, a1 = windows.active_windows[idx]
        bucket_ops = _shift_rows(bucket_ops, a0)
        bucket_comms = _shift_rows(bucket_comms, a0)
        return TokenSelection(
            view=view,
            token_index=idx,
            n_tokens=windows.n_tokens,
            label=token_label,
            method=windows.active_method,
            window_start=float(a0),
            window_end=float(a1),
            ops=bucket_ops,
            comms=bucket_comms,
        )

    raise ValueError(f"Unknown token view: {view}")


# ---------------------------
# Plotting helpers
# ---------------------------


def _draw_segments_or_vlines(
    ax: plt.Axes,
    starts: np.ndarray,
    durations: np.ndarray,
    y0: float,
    lane_h: float,
    *,
    color: str,
    alpha: float,
    tiny_threshold: float,
) -> None:
    if len(starts) == 0:
        return
    mask_tiny = durations <= tiny_threshold
    mask_bar = ~mask_tiny
    if mask_bar.any():
        segs = list(zip(starts[mask_bar], durations[mask_bar]))
        ax.broken_barh(segs, (y0, lane_h), facecolors=color, edgecolors="none", alpha=alpha)
    if mask_tiny.any():
        xs = starts[mask_tiny]
        ax.vlines(xs, y0, y0 + lane_h, colors=[color], alpha=alpha, linewidth=1.0)


def _build_lanes(ops_plot: pd.DataFrame, comms_plot: pd.DataFrame, *, comm_lane_mode: str) -> List[Tuple[str, str]]:
    lanes: List[Tuple[str, str]] = []

    if not ops_plot.empty:
        dev_info = (
            ops_plot[["device", "device_type"]]
            .drop_duplicates()
            .assign(_ord=lambda x: x["device_type"].map(lambda t: DEVICE_TYPE_ORDER.get(str(t), 50)))
            .assign(_nat=lambda x: x["device"].map(_natural_key))
            .sort_values(["_ord", "device"], key=lambda s: s if s.name != "device" else s.map(_natural_key))
        )
        for _, r in dev_info.iterrows():
            lanes.append((f"DEV::{r['device']}", f"{r['device']} ({r['device_type']})"))

    if not comms_plot.empty:
        if comm_lane_mode == "aggregate":
            lanes.append(("XFER::ALL", "data transfer"))
        elif comm_lane_mode == "per_link":
            links = (
                comms_plot[["src", "dst", "src_type", "dst_type"]]
                .drop_duplicates()
                .sort_values(
                    ["src_type", "src", "dst_type", "dst"],
                    key=lambda s: s if s.name in {"src_type", "dst_type"} else s.map(_natural_key),
                )
            )
            for _, r in links.iterrows():
                lanes.append((f"XFER::{r['src']}->{r['dst']}", f"{r['src']}→{r['dst']} (data transfer)"))
        else:
            raise ValueError(f"Unknown comm_lane_mode: {comm_lane_mode}")

    return lanes


def _compute_figsize(
    n_lanes: int,
    *,
    fig_w: float,
    fig_min_h: float,
    fig_h_scale: float,
    lane_h: float,
    lane_gap: float,
) -> Tuple[float, float]:
    h = max(fig_min_h, (n_lanes * (lane_h + lane_gap) + 0.8) * fig_h_scale)
    return float(fig_w), float(h)


def draw_gantt_on_ax(
    ax: plt.Axes,
    ops_s: pd.DataFrame,
    comms_s: pd.DataFrame,
    *,
    title: str,
    time_scale: float,
    time_unit: str,
    comm_lane_mode: str,
    tiny_event_threshold_s: float,
    lane_h: float,
    lane_gap: float,
    show_xlabel: bool,
) -> Tuple[int, float, List[Patch]]:
    ops_plot = ops_s.copy()
    comms_plot = comms_s.copy()

    for df in (ops_plot, comms_plot):
        if not df.empty:
            df["start"] = df["start"] * time_scale
            df["end"] = df["end"] * time_scale
            df["duration"] = df["duration"] * time_scale

    lanes = _build_lanes(ops_plot, comms_plot, comm_lane_mode=comm_lane_mode)
    if not lanes:
        raise ValueError("No lanes to plot.")

    tiny_thr = tiny_event_threshold_s * time_scale
    y = 0.0
    yticks: List[float] = []
    ylabels: List[str] = []

    for lane_key, lane_label in lanes:
        y0 = y
        yticks.append(y0 + lane_h / 2.0)
        ylabels.append(lane_label)

        if lane_key.startswith("DEV::"):
            dev = lane_key.split("DEV::", 1)[1]
            sub = ops_plot[ops_plot["device"] == dev]
            for group_name in OP_GROUP_ORDER:
                grp = sub[sub["op_group"] == group_name]
                if grp.empty:
                    continue
                _draw_segments_or_vlines(
                    ax,
                    grp["start"].to_numpy(dtype=float),
                    grp["duration"].to_numpy(dtype=float),
                    y0,
                    lane_h,
                    color=OP_GROUP_STYLE[group_name]["color"],
                    alpha=0.92,
                    tiny_threshold=tiny_thr,
                )

        elif lane_key == "XFER::ALL":
            _draw_segments_or_vlines(
                ax,
                comms_plot["start"].to_numpy(dtype=float),
                comms_plot["duration"].to_numpy(dtype=float),
                y0,
                lane_h,
                color=DATA_XFER_STYLE["color"],
                alpha=float(DATA_XFER_STYLE["alpha"]),
                tiny_threshold=tiny_thr,
            )

        elif lane_key.startswith("XFER::"):
            link = lane_key.split("XFER::", 1)[1]
            src, dst = link.split("->", 1)
            sub = comms_plot[(comms_plot["src"] == src) & (comms_plot["dst"] == dst)]
            _draw_segments_or_vlines(
                ax,
                sub["start"].to_numpy(dtype=float),
                sub["duration"].to_numpy(dtype=float),
                y0,
                lane_h,
                color=DATA_XFER_STYLE["color"],
                alpha=float(DATA_XFER_STYLE["alpha"]),
                tiny_threshold=tiny_thr,
            )

        y += lane_h + lane_gap

    max_end = 0.0
    if not ops_plot.empty:
        max_end = max(max_end, float(ops_plot["end"].max()))
    if not comms_plot.empty:
        max_end = max(max_end, float(comms_plot["end"].max()))

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel(f"Time ({time_unit}, shifted to window start)")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)

    handles: List[Patch] = []
    present_groups = set(ops_plot["op_group"].dropna().tolist()) if not ops_plot.empty else set()
    for group_name in OP_GROUP_ORDER:
        if group_name in present_groups:
            handles.append(
                Patch(
                    facecolor=OP_GROUP_STYLE[group_name]["color"],
                    edgecolor="none",
                    label=OP_GROUP_STYLE[group_name]["label"],
                )
            )
    if not comms_plot.empty:
        handles.append(
            Patch(
                facecolor=DATA_XFER_STYLE["color"],
                edgecolor="none",
                label=DATA_XFER_STYLE["label"],
                alpha=DATA_XFER_STYLE["alpha"],
            )
        )

    return len(lanes), max_end, handles


# ---------------------------
# Output writers
# ---------------------------


def _save_token_window_summary(path: Path, windows: DecodeWindows) -> None:
    rows = []
    for idx, (a0, a1) in enumerate(windows.active_windows):
        c0, c1 = windows.cadence_windows[idx]
        rows.append(
            {
                "token_index": idx,
                "token_1based": idx + 1,
                "token_start": windows.token_starts[idx],
                "active_start": a0,
                "active_end": a1,
                "active_duration": a1 - a0,
                "cadence_start": c0,
                "cadence_end": c1,
                "cadence_duration": c1 - c0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_single_plot(
    run: TraceRun,
    *,
    view_name: str,
    ops: pd.DataFrame,
    comms: pd.DataFrame,
    out_path: Path,
    title: str,
    time_scale: float,
    time_unit: str,
    comm_lane_mode: str,
    tiny_event_threshold_s: float,
    fig_w: float,
    fig_min_h: float,
    fig_h_scale: float,
    lane_h: float,
    lane_gap: float,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lanes = _build_lanes(ops, comms, comm_lane_mode=comm_lane_mode)
    fig_w_in, fig_h_in = _compute_figsize(
        len(lanes),
        fig_w=fig_w,
        fig_min_h=fig_min_h,
        fig_h_scale=fig_h_scale,
        lane_h=lane_h,
        lane_gap=lane_gap,
    )

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    _, max_end, handles = draw_gantt_on_ax(
        ax,
        ops,
        comms,
        title=title,
        time_scale=time_scale,
        time_unit=time_unit,
        comm_lane_mode=comm_lane_mode,
        tiny_event_threshold_s=tiny_event_threshold_s,
        lane_h=lane_h,
        lane_gap=lane_gap,
        show_xlabel=True,
    )
    ax.set_xlim(0.0, max(1e-12, max_end * 1.02))

    if handles:
        fig.tight_layout(rect=[0.0, 0.12, 1.0, 1.0])
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=min(5, len(handles)),
            fontsize=11,
            frameon=False,
        )
    else:
        fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_compare_plot(
    run_a: TraceRun,
    run_b: TraceRun,
    *,
    ops_a: pd.DataFrame,
    comms_a: pd.DataFrame,
    ops_b: pd.DataFrame,
    comms_b: pd.DataFrame,
    out_path: Path,
    title: str,
    time_scale: float,
    time_unit: str,
    comm_lane_mode: str,
    tiny_event_threshold_s: float,
    fig_w: float,
    fig_min_h: float,
    fig_h_scale: float,
    lane_h: float,
    lane_gap: float,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_a = _build_lanes(ops_a, comms_a, comm_lane_mode=comm_lane_mode)
    lanes_b = _build_lanes(ops_b, comms_b, comm_lane_mode=comm_lane_mode)

    _, h1 = _compute_figsize(len(lanes_a), fig_w=fig_w, fig_min_h=fig_min_h, fig_h_scale=fig_h_scale, lane_h=lane_h, lane_gap=lane_gap)
    _, h2 = _compute_figsize(len(lanes_b), fig_w=fig_w, fig_min_h=fig_min_h, fig_h_scale=fig_h_scale, lane_h=lane_h, lane_gap=lane_gap)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(fig_w, h1 + h2 + 1.4),
        dpi=dpi,
        gridspec_kw={"height_ratios": [max(h1, 1.0), max(h2, 1.0)]},
    )

    _, max1, handles1 = draw_gantt_on_ax(
        ax1,
        ops_a,
        comms_a,
        title=f"{title}\n{run_a.label}",
        time_scale=time_scale,
        time_unit=time_unit,
        comm_lane_mode=comm_lane_mode,
        tiny_event_threshold_s=tiny_event_threshold_s,
        lane_h=lane_h,
        lane_gap=lane_gap,
        show_xlabel=False,
    )
    _, max2, handles2 = draw_gantt_on_ax(
        ax2,
        ops_b,
        comms_b,
        title=run_b.label,
        time_scale=time_scale,
        time_unit=time_unit,
        comm_lane_mode=comm_lane_mode,
        tiny_event_threshold_s=tiny_event_threshold_s,
        lane_h=lane_h,
        lane_gap=lane_gap,
        show_xlabel=True,
    )
    ax1.set_xlim(0.0, max(1e-12, max(max1, max2) * 1.02))

    handle_map: Dict[str, Patch] = {}
    for h in handles1 + handles2:
        handle_map[h.get_label()] = h
    handles = list(handle_map.values())

    if handles:
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
        fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), fontsize=11, frameon=False)
    else:
        fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# CLI execution
# ---------------------------


DEFAULT_VIEWS = ["prefill", "decode_total", "decode_token_latency", "decode_token_active"]


def _parse_views(s: str) -> List[str]:
    if not s:
        return DEFAULT_VIEWS.copy()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts or "all" in {p.lower() for p in parts}:
        return DEFAULT_VIEWS.copy()
    allowed = set(DEFAULT_VIEWS)
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise ValueError(f"Unknown views: {bad}. Allowed: {sorted(allowed)}")
    return parts


def _shared_token_index(w_a: DecodeWindows, w_b: DecodeWindows, requested: Optional[int]) -> Tuple[int, int, int]:
    if w_a.n_tokens <= 0 or w_b.n_tokens <= 0:
        return 0, 0, 0
    shared = min(w_a.n_tokens, w_b.n_tokens)
    if requested is None:
        idx = shared // 2
    else:
        idx = max(0, min(int(requested), shared - 1))
    return idx, idx, shared


def _single_token_index(w: DecodeWindows, requested: Optional[int]) -> int:
    if w.n_tokens <= 0:
        return 0
    if requested is None:
        return w.n_tokens // 2
    return max(0, min(int(requested), w.n_tokens - 1))


def write_trace_metadata(out_dir: Path, run: TraceRun, windows: DecodeWindows) -> None:
    meta = {
        "label": run.label,
        "ops_path": str(run.ops_path),
        "comms_path": (str(run.comms_path) if run.comms_path is not None else None),
        "ops_rows_raw": int(len(run.ops_raw)),
        "ops_rows_plot_base": int(len(run.ops_plot_base)),
        "comms_rows_raw": int(len(run.comms_raw)),
        "phase_bounds": {
            phase: (list(map(float, bounds)) if (bounds := phase_bounds(run, phase)) is not None else None)
            for phase in ["prefill", "decode"]
        },
        "decode_windows": windows.to_jsonable(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{run.label}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _save_token_window_summary(out_dir / f"{run.label}_decode_windows.csv", windows)


def run_single(args: argparse.Namespace) -> None:
    run = load_trace_run(args.csv, allow_missing_comms=True)
    out_dir = Path(args.out_dir).expanduser().resolve()
    views = _parse_views(args.views)
    time_scale = 1000.0 if args.time_unit == "ms" else 1.0

    windows = detect_decode_windows(run)
    write_trace_metadata(out_dir, run, windows)

    if "prefill" in views:
        ops, comms, bounds, _ = select_phase_view(run, "prefill", label="prefill")
        if not ops.empty or not comms.empty:
            out = out_dir / f"{run.label}_prefill.{args.format}"
            save_single_plot(
                run,
                view_name="prefill",
                ops=ops,
                comms=comms,
                out_path=out,
                title=f"{run.label} · prefill",
                time_scale=time_scale,
                time_unit=args.time_unit,
                comm_lane_mode=args.comm_lane_mode,
                tiny_event_threshold_s=args.tiny_event_threshold_s,
                fig_w=args.fig_w,
                fig_min_h=args.fig_min_h,
                fig_h_scale=args.fig_h_scale,
                lane_h=args.lane_h,
                lane_gap=args.lane_gap,
                dpi=args.dpi,
            )
            print(f"[OK] prefill -> {out}")

    if "decode_total" in views:
        ops, comms, bounds, _ = select_phase_view(run, "decode", label="decode_total")
        if not ops.empty or not comms.empty:
            out = out_dir / f"{run.label}_decode_total.{args.format}"
            title = f"{run.label} · decode total"
            if bounds[1] > bounds[0]:
                title += f" · span={((bounds[1]-bounds[0])*time_scale):.6g} {args.time_unit}"
            save_single_plot(
                run,
                view_name="decode_total",
                ops=ops,
                comms=comms,
                out_path=out,
                title=title,
                time_scale=time_scale,
                time_unit=args.time_unit,
                comm_lane_mode=args.comm_lane_mode,
                tiny_event_threshold_s=args.tiny_event_threshold_s,
                fig_w=args.fig_w,
                fig_min_h=args.fig_min_h,
                fig_h_scale=args.fig_h_scale,
                lane_h=args.lane_h,
                lane_gap=args.lane_gap,
                dpi=args.dpi,
            )
            print(f"[OK] decode total -> {out}")

    if windows.n_tokens <= 0:
        print("[WARN] no decode tokens detected; skipping decode_token_* views")
        return

    idx = _single_token_index(windows, args.decode_token)

    if "decode_token_latency" in views:
        sel = select_decode_token_view(run, windows, idx, "decode_token_latency")
        assert sel is not None
        out = out_dir / f"{run.label}_{sel.label}_latency.{args.format}"
        title = (
            f"{run.label} · decode {sel.label} · latency window"
            f" · method={sel.method}"
            f" · span={((sel.window_end-sel.window_start)*time_scale):.6g} {args.time_unit}"
        )
        save_single_plot(
            run,
            view_name=sel.view,
            ops=sel.ops,
            comms=sel.comms,
            out_path=out,
            title=title,
            time_scale=time_scale,
            time_unit=args.time_unit,
            comm_lane_mode=args.comm_lane_mode,
            tiny_event_threshold_s=args.tiny_event_threshold_s,
            fig_w=args.fig_w,
            fig_min_h=args.fig_min_h,
            fig_h_scale=args.fig_h_scale,
            lane_h=args.lane_h,
            lane_gap=args.lane_gap,
            dpi=args.dpi,
        )
        print(f"[OK] token latency -> {out}")

    if "decode_token_active" in views:
        sel = select_decode_token_view(run, windows, idx, "decode_token_active")
        assert sel is not None
        out = out_dir / f"{run.label}_{sel.label}_active.{args.format}"
        title = (
            f"{run.label} · decode {sel.label} · active window"
            f" · method={sel.method}"
            f" · span={((sel.window_end-sel.window_start)*time_scale):.6g} {args.time_unit}"
        )
        save_single_plot(
            run,
            view_name=sel.view,
            ops=sel.ops,
            comms=sel.comms,
            out_path=out,
            title=title,
            time_scale=time_scale,
            time_unit=args.time_unit,
            comm_lane_mode=args.comm_lane_mode,
            tiny_event_threshold_s=args.tiny_event_threshold_s,
            fig_w=args.fig_w,
            fig_min_h=args.fig_min_h,
            fig_h_scale=args.fig_h_scale,
            lane_h=args.lane_h,
            lane_gap=args.lane_gap,
            dpi=args.dpi,
        )
        print(f"[OK] token active -> {out}")


def write_compare_metadata(out_dir: Path, run_a: TraceRun, run_b: TraceRun, w_a: DecodeWindows, w_b: DecodeWindows) -> None:
    meta = {
        "trace_a": {
            "label": run_a.label,
            "ops_path": str(run_a.ops_path),
            "comms_path": (str(run_a.comms_path) if run_a.comms_path is not None else None),
            "decode_windows": w_a.to_jsonable(),
        },
        "trace_b": {
            "label": run_b.label,
            "ops_path": str(run_b.ops_path),
            "comms_path": (str(run_b.comms_path) if run_b.comms_path is not None else None),
            "decode_windows": w_b.to_jsonable(),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(f"{run_a.label}_vs_{run_b.label}")
    with open(out_dir / f"compare_{slug}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_compare(args: argparse.Namespace) -> None:
    a_path, b_path = args.compare
    out_dir = Path(args.out_dir).expanduser().resolve()
    views = _parse_views(args.views)
    time_scale = 1000.0 if args.time_unit == "ms" else 1.0

    run_a = load_trace_run(a_path, label=(args.compare_labels[0] if args.compare_labels else None), allow_missing_comms=True)
    run_b = load_trace_run(b_path, label=(args.compare_labels[1] if args.compare_labels else None), allow_missing_comms=True)
    slug = _safe_slug(f"{run_a.label}_vs_{run_b.label}")

    w_a = detect_decode_windows(run_a)
    w_b = detect_decode_windows(run_b)
    write_compare_metadata(out_dir, run_a, run_b, w_a, w_b)

    if "prefill" in views:
        ops_a, comms_a, _, _ = select_phase_view(run_a, "prefill", label="prefill")
        ops_b, comms_b, _, _ = select_phase_view(run_b, "prefill", label="prefill")
        if (not ops_a.empty or not comms_a.empty) and (not ops_b.empty or not comms_b.empty):
            out = out_dir / f"compare_{slug}_prefill.{args.format}"
            save_compare_plot(
                run_a,
                run_b,
                ops_a=ops_a,
                comms_a=comms_a,
                ops_b=ops_b,
                comms_b=comms_b,
                out_path=out,
                title="prefill",
                time_scale=time_scale,
                time_unit=args.time_unit,
                comm_lane_mode=args.comm_lane_mode,
                tiny_event_threshold_s=args.tiny_event_threshold_s,
                fig_w=args.fig_w,
                fig_min_h=args.fig_min_h,
                fig_h_scale=args.fig_h_scale,
                lane_h=args.lane_h,
                lane_gap=args.lane_gap,
                dpi=args.dpi,
            )
            print(f"[OK] compare prefill -> {out}")

    if "decode_total" in views:
        ops_a, comms_a, _, _ = select_phase_view(run_a, "decode", label="decode_total")
        ops_b, comms_b, _, _ = select_phase_view(run_b, "decode", label="decode_total")
        if (not ops_a.empty or not comms_a.empty) and (not ops_b.empty or not comms_b.empty):
            out = out_dir / f"compare_{slug}_decode_total.{args.format}"
            save_compare_plot(
                run_a,
                run_b,
                ops_a=ops_a,
                comms_a=comms_a,
                ops_b=ops_b,
                comms_b=comms_b,
                out_path=out,
                title="decode total",
                time_scale=time_scale,
                time_unit=args.time_unit,
                comm_lane_mode=args.comm_lane_mode,
                tiny_event_threshold_s=args.tiny_event_threshold_s,
                fig_w=args.fig_w,
                fig_min_h=args.fig_min_h,
                fig_h_scale=args.fig_h_scale,
                lane_h=args.lane_h,
                lane_gap=args.lane_gap,
                dpi=args.dpi,
            )
            print(f"[OK] compare decode total -> {out}")

    if w_a.n_tokens <= 0 or w_b.n_tokens <= 0:
        print("[WARN] compare: one side has no decode tokens; skipping decode_token_* views")
        return

    idx_a, idx_b, shared = _shared_token_index(w_a, w_b, args.decode_token)

    if "decode_token_latency" in views:
        sel_a = select_decode_token_view(run_a, w_a, idx_a, "decode_token_latency")
        sel_b = select_decode_token_view(run_b, w_b, idx_b, "decode_token_latency")
        assert sel_a is not None and sel_b is not None
        out = out_dir / f"compare_{slug}_token{idx_a+1}_latency_A{idx_a+1}of{w_a.n_tokens}_B{idx_b+1}of{w_b.n_tokens}.{args.format}"
        save_compare_plot(
            run_a,
            run_b,
            ops_a=sel_a.ops,
            comms_a=sel_a.comms,
            ops_b=sel_b.ops,
            comms_b=sel_b.comms,
            out_path=out,
            title=(
                f"decode token {idx_a+1} / shared {shared} · latency window\n"
                f"A={sel_a.method}, B={sel_b.method}"
            ),
            time_scale=time_scale,
            time_unit=args.time_unit,
            comm_lane_mode=args.comm_lane_mode,
            tiny_event_threshold_s=args.tiny_event_threshold_s,
            fig_w=args.fig_w,
            fig_min_h=args.fig_min_h,
            fig_h_scale=args.fig_h_scale,
            lane_h=args.lane_h,
            lane_gap=args.lane_gap,
            dpi=args.dpi,
        )
        print(f"[OK] compare token latency -> {out}")

    if "decode_token_active" in views:
        sel_a = select_decode_token_view(run_a, w_a, idx_a, "decode_token_active")
        sel_b = select_decode_token_view(run_b, w_b, idx_b, "decode_token_active")
        assert sel_a is not None and sel_b is not None
        out = out_dir / f"compare_{slug}_token{idx_a+1}_active_A{idx_a+1}of{w_a.n_tokens}_B{idx_b+1}of{w_b.n_tokens}.{args.format}"
        save_compare_plot(
            run_a,
            run_b,
            ops_a=sel_a.ops,
            comms_a=sel_a.comms,
            ops_b=sel_b.ops,
            comms_b=sel_b.comms,
            out_path=out,
            title=(
                f"decode token {idx_a+1} / shared {shared} · active window\n"
                f"A={sel_a.method}, B={sel_b.method}"
            ),
            time_scale=time_scale,
            time_unit=args.time_unit,
            comm_lane_mode=args.comm_lane_mode,
            tiny_event_threshold_s=args.tiny_event_threshold_s,
            fig_w=args.fig_w,
            fig_min_h=args.fig_min_h,
            fig_h_scale=args.fig_h_scale,
            lane_h=args.lane_h,
            lane_gap=args.lane_gap,
            dpi=args.dpi,
        )
        print(f"[OK] compare token active -> {out}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot gantt charts from current ops/comms traces.")
    ap.add_argument("--csv", type=str, default=None, help="Single-trace mode: path to *_ops_trace.csv or *_comms_trace.csv")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("TRACE_A", "TRACE_B"), help="Compare mode: two traces")
    ap.add_argument("--compare_labels", nargs=2, default=None, metavar=("LABEL_A", "LABEL_B"), help="Optional labels for compare mode")
    ap.add_argument("--out_dir", type=str, default="plots", help="Output directory")
    ap.add_argument("--views", type=str, default=",".join(DEFAULT_VIEWS), help=f"Comma-separated views. Allowed: {', '.join(DEFAULT_VIEWS)}")
    ap.add_argument("--decode_token", type=int, default=None, help="0-based decode token index. Default: shared middle token")
    ap.add_argument("--comm_lane_mode", type=str, default="aggregate", choices=["aggregate", "per_link"], help="Communication lane mode")
    ap.add_argument("--time_unit", type=str, default="ms", choices=["s", "ms"], help="Axis time unit")
    ap.add_argument("--tiny_event_threshold_s", type=float, default=5e-6, help="Events shorter than this are drawn as vlines")
    ap.add_argument("--fig_w", type=float, default=13.0, help="Figure width in inches")
    ap.add_argument("--fig_min_h", type=float, default=6.0, help="Minimum figure height in inches")
    ap.add_argument("--fig_h_scale", type=float, default=0.85, help="Figure height scale")
    ap.add_argument("--lane_h", type=float, default=0.85, help="Lane rectangle height")
    ap.add_argument("--lane_gap", type=float, default=0.28, help="Gap between lanes")
    ap.add_argument("--dpi", type=int, default=180, help="Output DPI")
    ap.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"], help="Output format")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if bool(args.csv) == bool(args.compare):
        raise SystemExit("Provide exactly one of --csv or --compare.")
    if args.compare is not None:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
