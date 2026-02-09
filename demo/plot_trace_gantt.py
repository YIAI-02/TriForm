#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python plot_trace_gantt.py \
  --csv evaluate_single_test/hw_scale_down/qwen_7b_int8_b4_s64/algo_weights_on_pim/128x512_ops_trace.csv \
  --out_dir ../figs/evaluate_single_test/hw_scale_down/gatt

python plot_trace_gantt.py \
  --base_dir evaluate_single_test/hw_scale_down/qwen_7b_int8_b4_s64 \
  --policy algo_weights_on_pim \
  --length 128x512 \
  --out_dir ../figs/evaluate_single_test/hw_scale_down/gatt

python plot_trace_gantt.py \
  --base_dir ../algorithms/output/experiment_scale_down/hw_scale_down_pima/st64/qwen_7b_int8_b8_s64 \
  --policy algo_hefthint \
  --length 128x1024 \
  --out_dir ../figs/experiment_scale_down/qwen_7b_int8_b8_s64/gantt

python plot_trace_gantt.py \
  --compare \
  ../algorithms/output/evaluate_single_test/hardware_config_scale_down_11pima/llama_7b_int8_b1_s64/algo_hefthint/hefthint_4096x4096_ops_trace.csv \
  ../algorithms/output/evaluate_single_test/hardware_config_scale_down_11pima/llama_7b_int8_b1_s64/algo_pd/4096x4096_ops_trace.csv \
  --out_dir ../figs/evaluate_single_test/hardware_config_scale_down_11pima/llama_7b_int8_b1_s64/gantt_comparison \
  --time_unit ms \
  --fig_w 25 \
  --comm_lane_mode aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------------------
# Grouping + colors
# ---------------------------

OP_GROUP_ORDER = ["FFN_W", "ATTN_CORE", "QKVO", "OTHER"]

OP_GROUP_STYLE = {
    "FFN_W": {"label": "FFN_W1/W2/W3", "color": "#2e7277"},
    "ATTN_CORE": {"label": "QK/SV/Softmax/K_write/V_write", "color": "#98d98e"},
    "QKVO": {"label": "Q/K/V/O", "color": "#6C92DA"},
    "OTHER": {"label": "Other", "color": "#dbed93"},
}

# Requested rename: COMM -> data transfer
DATA_XFER_STYLE = {"label": "data transfer", "color": "#808080", "alpha": 0.40}


def _canon(s: str) -> str:
    """Canonicalize op/tag string for robust matching."""
    s = str(s).strip().lower()
    s = re.sub(r"[_\s]+", " ", s)
    return s


def classify_op_group(op: str) -> str:
    """Apply requested grouping rules (robust to minor naming variants)."""
    c = _canon(op)

    # FFN_W1/W2/W3
    if re.fullmatch(r"ffn w[123]", c):
        return "FFN_W"

    # QK, SV, Softmax, K write, V write  (also supports K_write / V_write)
    if c in {"qk", "sv", "softmax", "k write", "v write"}:
        return "ATTN_CORE"

    # Q / K / V (NOTE: exact match; avoids catching "k write")
    if c in {"q", "k", "v","o"}:
        return "QKVO"

    return "OTHER"


def add_op_groups(ops_df: pd.DataFrame) -> pd.DataFrame:
    ops_df = ops_df.copy()
    if "op" not in ops_df.columns:
        ops_df["op"] = ""
    ops_df["op_group"] = ops_df["op"].map(classify_op_group)
    ops_df["op_group_label"] = ops_df["op_group"].map(
        lambda g: OP_GROUP_STYLE.get(g, OP_GROUP_STYLE["OTHER"])["label"]
    )
    return ops_df


# ---------------------------
# Path helpers
# ---------------------------


def _normalize_path(p: str) -> str:
    p = str(p).strip()
    p = os.path.expanduser(os.path.expandvars(p))
    return os.path.abspath(p)


def _safe_slug(s: str, *, max_len: int = 80) -> str:
    """Make a filesystem-friendly string."""
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def infer_pair_paths(any_csv_path: str) -> Tuple[str, str]:
    """Given either *_ops_trace.csv or *_comms_trace.csv, infer the paired csv path."""
    if any_csv_path.endswith("_ops_trace.csv"):
        ops_csv = any_csv_path
        comms_csv = any_csv_path.replace("_ops_trace.csv", "_comms_trace.csv")
    elif any_csv_path.endswith("_comms_trace.csv"):
        comms_csv = any_csv_path
        ops_csv = any_csv_path.replace("_comms_trace.csv", "_ops_trace.csv")
    else:
        raise ValueError(
            f"CSV must end with _ops_trace.csv or _comms_trace.csv, got: {any_csv_path}"
        )
    return ops_csv, comms_csv


def parse_policy_and_length_from_path(any_csv_path: str) -> Tuple[str, str]:
    """policy: parent folder name; length: first AxB pattern in filename."""
    abs_path = _normalize_path(any_csv_path)
    policy = os.path.basename(os.path.dirname(abs_path))
    fname = os.path.basename(abs_path)
    m = re.search(r"(\d+x\d+)", fname)
    length = m.group(1) if m else os.path.splitext(fname)[0]
    return policy, length


def _find_csv_in_policy_dir(policy_dir: str, length: str, suffix: str) -> Optional[str]:
    guess = os.path.join(policy_dir, f"{length}_{suffix}")
    if os.path.isfile(guess):
        return guess
    candidates = sorted(glob(os.path.join(policy_dir, f"*{length}*_{suffix}")))
    return candidates[0] if candidates else None


def find_csv_pair_in_policy_dir(policy_dir: str, length: str) -> Tuple[str, Optional[str]]:
    """Locate ops/comms csv under policy_dir. Comms is optional."""
    ops_path = _find_csv_in_policy_dir(policy_dir, length, "ops_trace.csv")
    if not ops_path:
        raise FileNotFoundError(f"Cannot find ops_trace for length={length} in {policy_dir}")
    comms_path = _find_csv_in_policy_dir(policy_dir, length, "comms_trace.csv")
    return ops_path, comms_path


def _debug_path_not_found(p: str) -> str:
    return "\n".join(
        [
            f"File not found: {p}",
            f"  cwd: {os.getcwd()}",
            f"  abs: {_normalize_path(p)}",
        ]
    )


# ---------------------------
# JSON loading helpers
# ---------------------------


def _iter_list_of_dicts(obj):
    """Yield any list-of-dicts found by walking a nested JSON-like object."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            if cur and all(isinstance(x, dict) for x in cur):
                yield cur
            else:
                stack.extend(cur)


def _best_table_from_json(obj, required_cols: Sequence[str]) -> Optional[pd.DataFrame]:
    req = set(required_cols)
    best_df: Optional[pd.DataFrame] = None
    best_score = -1

    for lst in _iter_list_of_dicts(obj):
        df = pd.DataFrame(lst)
        cols = set(map(str, df.columns))
        if not req.issubset(cols):
            continue
        score = int(df.shape[0])
        if score > best_score:
            best_score = score
            best_df = df

    return best_df


def _infer_type_from_name(name: str) -> str:
    """Infer device type from a name like 'npu0', 'pim1', 'GPU_0', ..."""
    m = re.match(r"([A-Za-z]+)", str(name).strip())
    return m.group(1).lower() if m else ""


def _ensure_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "duration" not in df.columns:
        df["duration"] = df["end"] - df["start"]
    else:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").astype(float)
    return df


def _standardize_ops_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    required = ["phase", "op", "device", "device_type", "start", "end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"ops trace missing columns {missing} in {source}")

    df["phase"] = df["phase"].astype(str).str.lower()
    df["op"] = df["op"].astype(str)
    df["device"] = df["device"].astype(str)
    df["device_type"] = df["device_type"].astype(str).str.lower()
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype(float)
    df["end"] = pd.to_numeric(df["end"], errors="coerce").astype(float)

    df = df.dropna(subset=["start", "end"]).copy()
    df = _ensure_duration(df)
    df = df[df["duration"] > 0].copy()
    return df


def _standardize_comms_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    required = ["phase", "src", "dst", "start", "end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"comms trace missing columns {missing} in {source}")

    df["phase"] = df["phase"].astype(str).str.lower()
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype(float)
    df["end"] = pd.to_numeric(df["end"], errors="coerce").astype(float)

    df = df.dropna(subset=["start", "end"]).copy()

    if "tag" not in df.columns:
        df["tag"] = ""
    else:
        df["tag"] = df["tag"].astype(str)

    if "src_type" not in df.columns:
        df["src_type"] = df["src"].map(_infer_type_from_name)
    else:
        df["src_type"] = df["src_type"].astype(str).str.lower()

    if "dst_type" not in df.columns:
        df["dst_type"] = df["dst"].map(_infer_type_from_name)
    else:
        df["dst_type"] = df["dst_type"].astype(str).str.lower()

    df = _ensure_duration(df)
    df = df[df["duration"] > 0].copy()
    return df


def _empty_comms_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["phase", "src", "dst", "start", "end", "duration", "tag", "src_type", "dst_type"])


@dataclass(frozen=True)
class TraceRun:
    label: str
    source: str
    ops: pd.DataFrame
    comms: pd.DataFrame


def load_trace_run(path: str, *, label: Optional[str] = None, allow_missing_comms: bool = True) -> TraceRun:
    """Load a trace run from either a CSV pair (given any of the pair) or JSON."""
    path = _normalize_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(_debug_path_not_found(path))

    run_label = _safe_slug(label) if label else _safe_slug(os.path.splitext(os.path.basename(path))[0])

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ops_req = ["phase", "op", "device", "device_type", "start", "end"]
        comms_req = ["phase", "src", "dst", "start", "end"]

        ops_df: Optional[pd.DataFrame] = None
        comms_df: Optional[pd.DataFrame] = None

        if isinstance(data, dict):
            for k in ["ops", "ops_trace", "op_trace", "operations", "compute", "compute_ops"]:
                if k in data and isinstance(data[k], list) and (not data[k] or isinstance(data[k][0], dict)):
                    ops_df = pd.DataFrame(data[k])
                    break
            for k in ["comms", "comms_trace", "comm_trace", "communications", "transfer", "data_transfer"]:
                if k in data and isinstance(data[k], list) and (not data[k] or isinstance(data[k][0], dict)):
                    comms_df = pd.DataFrame(data[k])
                    break

        if ops_df is None:
            ops_df = _best_table_from_json(data, ops_req)
        if comms_df is None:
            comms_df = _best_table_from_json(data, comms_req)

        if ops_df is None:
            raise ValueError(
                f"Could not find ops table in JSON: {path}. Need columns: {ops_req}"
            )

        ops_df = _standardize_ops_df(ops_df, path)
        if comms_df is None:
            if allow_missing_comms:
                comms_df = _empty_comms_df()
            else:
                raise ValueError(
                    f"Could not find comms table in JSON: {path}. Need columns: {comms_req}"
                )
        else:
            comms_df = _standardize_comms_df(comms_df, path)

        return TraceRun(run_label, path, ops_df, comms_df)

    # CSV mode (pair)
    ops_csv, comms_csv = infer_pair_paths(path)
    ops_csv = _normalize_path(ops_csv)
    comms_csv = _normalize_path(comms_csv)

    if not os.path.isfile(ops_csv):
        raise FileNotFoundError(_debug_path_not_found(ops_csv))

    ops_df = _standardize_ops_df(pd.read_csv(ops_csv), ops_csv)

    if os.path.isfile(comms_csv):
        comms_df = _standardize_comms_df(pd.read_csv(comms_csv), comms_csv)
    else:
        if allow_missing_comms:
            print(f"[WARN] paired comms csv not found, will treat as empty: {comms_csv}")
            comms_df = _empty_comms_df()
        else:
            raise FileNotFoundError(_debug_path_not_found(comms_csv))

    if label is None:
        policy, length = parse_policy_and_length_from_path(ops_csv)
        run_label = _safe_slug(f"{policy}_{length}")

    return TraceRun(run_label, ops_csv, ops_df, comms_df)


# ---------------------------
# Token window detection
# ---------------------------


@dataclass(frozen=True)
class TokenWindows:
    windows: List[Tuple[float, float]]
    method: str


def detect_decode_token_windows(ops_df: pd.DataFrame) -> TokenWindows:
    """Try to split decode phase into per-token time windows.

    Heuristics (in priority order):
      1) Marker node_id: prefer "L0_LN1". If absent, try any node_id matching ^L0_.*LN1$ with count>=2.
      2) If decode rows are an integer multiple of prefill rows:
         split decode events (sorted by time) into equal-size chunks.
      3) Fallback: time-gap heuristic on sorted start times.
    """
    decode = ops_df[ops_df["phase"] == "decode"].copy()
    if decode.empty:
        return TokenWindows([], "no-decode")

    # 1) Marker-based windows
    marker: Optional[str] = None
    if "node_id" in decode.columns:
        if (decode["node_id"] == "L0_LN1").any():
            marker = "L0_LN1"
        else:
            cand = decode.loc[
                decode["node_id"].astype(str).str.match(r"^L0_.*LN1$", na=False),
                "node_id",
            ].value_counts()
            if not cand.empty and int(cand.iloc[0]) >= 2:
                marker = str(cand.index[0])

    if marker is not None:
        starts = decode.loc[decode["node_id"] == marker, "start"].sort_values().to_numpy()
        if len(starts) >= 2:
            windows: List[Tuple[float, float]] = []
            for i in range(len(starts)):
                t0 = float(starts[i])
                t1 = float(starts[i + 1]) if i + 1 < len(starts) else float(decode["end"].max())
                windows.append((t0, t1))
            return TokenWindows(windows, f"marker:{marker}")

    # 2) Uniform row-count windows
    prefill = ops_df[ops_df["phase"] == "prefill"]
    if not prefill.empty and len(decode) % len(prefill) == 0:
        n_tokens = len(decode) // len(prefill)
        decode_sorted = decode.sort_values(["start", "end"]).reset_index(drop=True)
        windows = []
        chunk_size = len(prefill)
        for i in range(n_tokens):
            chunk = decode_sorted.iloc[i * chunk_size : (i + 1) * chunk_size]
            windows.append((float(chunk["start"].min()), float(chunk["end"].max())))
        return TokenWindows(windows, "uniform_by_rowcount")

    # 3) Time-gap heuristic
    decode_sorted = decode.sort_values(["start", "end"])
    starts = decode_sorted["start"].to_numpy()
    if len(starts) <= 1:
        return TokenWindows(
            [(float(decode_sorted["start"].min()), float(decode_sorted["end"].max()))],
            "single-window",
        )

    diffs = np.diff(starts)
    med = float(np.median(diffs))
    threshold = max(med * 50.0, med + 1e-9)
    boundary_idx = np.where(diffs > threshold)[0]
    boundaries = [0] + (boundary_idx + 1).tolist() + [len(decode_sorted)]

    windows = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        chunk = decode_sorted.iloc[a:b]
        if chunk.empty:
            continue
        windows.append((float(chunk["start"].min()), float(chunk["end"].max())))
    return TokenWindows(windows, "gap-heuristic")


# ---------------------------
# Window slicing helpers
# ---------------------------


def _shift_to_zero(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = float(df["start"].min())
    df = df.copy()
    df["start"] = df["start"] - base
    df["end"] = df["end"] - base
    df["duration"] = df["end"] - df["start"]
    return df


def _slice_and_shift(df: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[(df["start"] < t1) & (df["end"] > t0)].copy()
    df["start"] = df["start"] - t0
    df["end"] = df["end"] - t0
    df["duration"] = df["end"] - df["start"]
    return df


# ---------------------------
# Utilization + Other op summary
# ---------------------------


def _merged_interval_sum(starts: np.ndarray, ends: np.ndarray) -> float:
    """Union length of intervals."""
    if len(starts) == 0:
        return 0.0
    order = np.argsort(starts)
    s = starts[order]
    e = ends[order]

    cur_s = float(s[0])
    cur_e = float(e[0])
    total = 0.0
    for i in range(1, len(s)):
        si = float(s[i])
        ei = float(e[i])
        if si <= cur_e:
            if ei > cur_e:
                cur_e = ei
        else:
            total += max(0.0, cur_e - cur_s)
            cur_s, cur_e = si, ei
    total += max(0.0, cur_e - cur_s)
    return float(total)


def compute_utilization_tables(
    ops_window_s: pd.DataFrame,
    comms_window_s: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Compute utilization per device and per device_type for a stage window.

    - ops_window_s/comms_window_s should already be shifted so that stage starts at 0.
    - total_time = max(ops_end, comms_end)
    - busy_time per device = union of that device's compute intervals

    Returns: (per_device_df, per_type_df, total_time_s)
    """
    total_time = 0.0
    if ops_window_s is not None and not ops_window_s.empty:
        total_time = max(total_time, float(ops_window_s["end"].max()))
    if comms_window_s is not None and not comms_window_s.empty:
        total_time = max(total_time, float(comms_window_s["end"].max()))

    if total_time <= 0.0 or ops_window_s is None or ops_window_s.empty:
        per_device = pd.DataFrame(columns=["device", "device_type", "busy_time_s", "total_time_s", "utilization"])
        per_type = pd.DataFrame(columns=["device_type", "n_devices", "busy_time_sum_s", "total_time_s", "utilization_avg"])
        return per_device, per_type, float(total_time)

    rows = []
    for (dev, dev_type), g in ops_window_s.groupby(["device", "device_type"], dropna=False):
        busy = _merged_interval_sum(g["start"].to_numpy(), g["end"].to_numpy())
        util = busy / total_time if total_time > 0 else np.nan
        rows.append(
            {
                "device": str(dev),
                "device_type": str(dev_type),
                "busy_time_s": float(busy),
                "total_time_s": float(total_time),
                "utilization": float(util),
            }
        )

    per_device = pd.DataFrame(rows)
    if not per_device.empty:
        per_device = per_device.sort_values(["device_type", "device"]).reset_index(drop=True)

    type_rows = []
    for dev_type, g in per_device.groupby("device_type", dropna=False):
        n = int(g.shape[0])
        busy_sum = float(g["busy_time_s"].sum())
        util_avg = busy_sum / (total_time * n) if (total_time > 0 and n > 0) else np.nan
        type_rows.append(
            {
                "device_type": str(dev_type),
                "n_devices": n,
                "busy_time_sum_s": busy_sum,
                "total_time_s": float(total_time),
                "utilization_avg": float(util_avg),
            }
        )

    per_type = pd.DataFrame(type_rows)
    if not per_type.empty:
        per_type = per_type.sort_values(["device_type"]).reset_index(drop=True)

    return per_device, per_type, float(total_time)


def save_utilization_tables(
    per_device: pd.DataFrame,
    per_type: pd.DataFrame,
    out_prefix: str,
    *,
    stage: str,
    trace_label: str,
    time_unit: str,
    time_scale: float,
) -> Tuple[str, str]:
    """Save utilization tables as CSV next to the plot."""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    per_device_out = per_device.copy()
    per_type_out = per_type.copy()

    # Attach metadata columns
    per_device_out.insert(0, "trace", trace_label)
    per_device_out.insert(1, "stage", stage)
    per_type_out.insert(0, "trace", trace_label)
    per_type_out.insert(1, "stage", stage)

    # Also provide durations in plotted unit
    unit_col_busy = f"busy_time_{time_unit}"
    unit_col_total = f"total_time_{time_unit}"
    if not per_device_out.empty:
        per_device_out[unit_col_busy] = per_device_out["busy_time_s"] * time_scale
        per_device_out[unit_col_total] = per_device_out["total_time_s"] * time_scale
    if not per_type_out.empty:
        per_type_out[f"busy_time_sum_{time_unit}"] = per_type_out["busy_time_sum_s"] * time_scale
        per_type_out[unit_col_total] = per_type_out["total_time_s"] * time_scale

    dev_csv = f"{out_prefix}_util_device.csv"
    typ_csv = f"{out_prefix}_util_type.csv"

    per_device_out.to_csv(dev_csv, index=False)
    per_type_out.to_csv(typ_csv, index=False)
    return dev_csv, typ_csv


def summarize_other_ops(
    ops_window_s: pd.DataFrame,
    *,
    stage: str,
    trace_label: str,
    time_unit: str,
    time_scale: float,
) -> pd.DataFrame:
    """List what 'Other' contains: op names + counts + total duration."""
    if ops_window_s is None or ops_window_s.empty or "op_group" not in ops_window_s.columns:
        return pd.DataFrame(columns=["trace", "stage", "op", "op_canon", "count", "total_duration_s", f"total_duration_{time_unit}", "duration_share"])

    other = ops_window_s[ops_window_s["op_group"] == "OTHER"].copy()
    if other.empty:
        return pd.DataFrame(columns=["trace", "stage", "op", "op_canon", "count", "total_duration_s", f"total_duration_{time_unit}", "duration_share"])

    other["op_canon"] = other["op"].map(_canon)
    g = other.groupby(["op", "op_canon"], dropna=False)
    out = g["duration"].agg(count="count", total_duration_s="sum").reset_index()

    total_other = float(out["total_duration_s"].sum())
    out["duration_share"] = out["total_duration_s"] / total_other if total_other > 0 else np.nan
    out[f"total_duration_{time_unit}"] = out["total_duration_s"] * time_scale

    out.insert(0, "stage", stage)
    out.insert(0, "trace", trace_label)

    out = out.sort_values(["total_duration_s", "count"], ascending=[False, False]).reset_index(drop=True)
    return out


def save_other_ops_table(other_df: pd.DataFrame, out_prefix: str) -> str:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    out_path = f"{out_prefix}_other_ops.csv"
    other_df.to_csv(out_path, index=False)
    return out_path


# ---------------------------
# Plotting
# ---------------------------


def _draw_segments_or_vlines(
    ax: plt.Axes,
    starts: np.ndarray,
    durations: np.ndarray,
    y0: float,
    lane_h: float,
    *,
    color,
    alpha: float,
    tiny_threshold: float,
) -> None:
    """Very short events can be invisible as rectangles; draw them as vlines."""
    if len(starts) == 0:
        return

    mask_tiny = durations < tiny_threshold
    mask_bar = ~mask_tiny

    if mask_bar.any():
        segs = list(zip(starts[mask_bar], durations[mask_bar]))
        ax.broken_barh(segs, (y0, lane_h), facecolors=color, edgecolors="none", alpha=alpha)

    if mask_tiny.any():
        xs = starts[mask_tiny]
        ax.vlines(xs, y0, y0 + lane_h, colors=[color], alpha=alpha, linewidth=1.0)


def _build_lanes(
    ops_plot: pd.DataFrame,
    comms_plot: pd.DataFrame,
    *,
    comm_lane_mode: str,
) -> List[Tuple[str, str]]:
    """Return list of (lane_key, lane_label)."""
    dtype_order = {"npu": 0, "pim": 1, "gpu": 2, "cpu": 3}
    lanes: List[Tuple[str, str]] = []

    if not ops_plot.empty:
        dev_info = (
            ops_plot[["device", "device_type"]]
            .drop_duplicates()
            .assign(_ord=lambda x: x["device_type"].map(lambda t: dtype_order.get(str(t), 99)))
            .sort_values(["_ord", "device"])
        )
        for _, r in dev_info.iterrows():
            lanes.append((f"DEV::{r['device']}", f"{r['device']} ({r['device_type']})"))

    if not comms_plot.empty:
        if comm_lane_mode == "aggregate":
            lanes.append(("XFER::ALL", "data transfer"))
        elif comm_lane_mode == "per_link":
            cols = ["src", "dst", "src_type", "dst_type"]
            avail_cols = [c for c in cols if c in comms_plot.columns]
            sort_cols = [c for c in ["src_type", "src", "dst_type", "dst"] if c in avail_cols]
            links = comms_plot[avail_cols].drop_duplicates().sort_values(sort_cols)
            for _, r in links.iterrows():
                lanes.append((f"XFER::{r['src']}->{r['dst']}", f"{r['src']}→{r['dst']} (data transfer)"))
        else:
            raise ValueError(f"Unknown comm_lane_mode={comm_lane_mode}")

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
    # Figure height: scale with number of lanes
    # (make it taller by default: fig_h_scale is larger than the original script)
    h = max(fig_min_h, (n_lanes * (lane_h + lane_gap) + 0.8) * fig_h_scale)
    return float(fig_w), float(h)


def draw_gantt_on_ax(
    ax: plt.Axes,
    ops_s: pd.DataFrame,
    comms_s: pd.DataFrame,
    *,
    time_scale: float,
    time_unit: str,
    comm_lane_mode: str,
    tiny_event_threshold_s: float,
    lane_h: float,
    lane_gap: float,
    title: str,
    show_xlabel: bool,
) -> Tuple[int, float, List[Patch]]:
    """Draw gantt on an existing axis.

    Returns:
      (n_lanes, max_end_axis_units, legend_handles)
    """
    # Prepare plot dfs in axis units
    ops_plot = ops_s.copy()
    comms_plot = comms_s.copy()

    for df in (ops_plot, comms_plot):
        if df is not None and not df.empty:
            df["start"] = df["start"] * time_scale
            df["end"] = df["end"] * time_scale
            df["duration"] = df["duration"] * time_scale

    tiny_thr = tiny_event_threshold_s * time_scale

    lanes = _build_lanes(ops_plot, comms_plot, comm_lane_mode=comm_lane_mode)
    if not lanes:
        raise ValueError("No lanes to plot. Check if trace is empty or phase filter is wrong.")

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
            if not sub.empty:
                for g in OP_GROUP_ORDER:
                    grp = sub[sub["op_group"] == g]
                    if grp.empty:
                        continue
                    _draw_segments_or_vlines(
                        ax,
                        grp["start"].to_numpy(),
                        grp["duration"].to_numpy(),
                        y0,
                        lane_h,
                        color=OP_GROUP_STYLE[g]["color"],
                        alpha=0.92,
                        tiny_threshold=tiny_thr,
                    )

        elif lane_key == "XFER::ALL":
            sub = comms_plot
            if not sub.empty:
                _draw_segments_or_vlines(
                    ax,
                    sub["start"].to_numpy(),
                    sub["duration"].to_numpy(),
                    y0,
                    lane_h,
                    color=DATA_XFER_STYLE["color"],
                    alpha=float(DATA_XFER_STYLE["alpha"]),
                    tiny_threshold=tiny_thr,
                )

        elif lane_key.startswith("XFER::"):
            link = lane_key.split("XFER::", 1)[1]
            src, dst = link.split("->")
            sub = comms_plot[(comms_plot["src"] == src) & (comms_plot["dst"] == dst)]
            if not sub.empty:
                _draw_segments_or_vlines(
                    ax,
                    sub["start"].to_numpy(),
                    sub["duration"].to_numpy(),
                    y0,
                    lane_h,
                    color=DATA_XFER_STYLE["color"],
                    alpha=float(DATA_XFER_STYLE["alpha"]),
                    tiny_threshold=tiny_thr,
                )

        y += lane_h + lane_gap

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=16)
    if show_xlabel:
        ax.set_xlabel(f"Time ({time_unit}, shifted to phase/token start)")
    ax.set_title(title)

    max_end = 0.0
    if not ops_plot.empty:
        max_end = max(max_end, float(ops_plot["end"].max()))
    if not comms_plot.empty:
        max_end = max(max_end, float(comms_plot["end"].max()))

    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)

    # Legend handles (only groups that appear in this axis)
    handles: List[Patch] = []
    if not ops_plot.empty:
        present = list(ops_plot["op_group"].dropna().unique())
        present_set = set(present)
        for g in OP_GROUP_ORDER:
            if g in present_set:
                handles.append(Patch(facecolor=OP_GROUP_STYLE[g]["color"], edgecolor="none", label=OP_GROUP_STYLE[g]["label"]))
    if not comms_plot.empty:
        handles.append(Patch(facecolor=DATA_XFER_STYLE["color"], edgecolor="none", label=DATA_XFER_STYLE["label"], alpha=DATA_XFER_STYLE["alpha"]))

    return len(lanes), float(max_end), handles


def plot_single_stage(
    run: TraceRun,
    *,
    stage: str,
    ops_window_s: pd.DataFrame,
    comms_window_s: pd.DataFrame,
    out_path: str,
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
    """Plot a single stage (prefill or one decode-token window) and save PNG + CSVs."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Ensure op_group exists for plotting & Other summary
    ops_window_s = add_op_groups(ops_window_s)

    # Pre-compute lanes count for figsize (need scaled dfs for comm lanes existence)
    # We can build lanes using already-shifted seconds but without scaling; it doesn't matter.
    lanes = _build_lanes(ops_window_s, comms_window_s, comm_lane_mode=comm_lane_mode)
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
        ops_window_s,
        comms_window_s,
        time_scale=time_scale,
        time_unit=time_unit,
        comm_lane_mode=comm_lane_mode,
        tiny_event_threshold_s=tiny_event_threshold_s,
        lane_h=lane_h,
        lane_gap=lane_gap,
        title=run.label,
        show_xlabel=True,
    )

    ax.set_xlim(0.0, max_end * 1.02 if max_end > 0 else 1.0)

    if handles:
        fig.tight_layout(rect=[0.0, 0.14, 1.0, 1.0])
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(5, len(handles)),
            fontsize=16,
            frameon=False,
        )
    else:
        fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # ---- Save utilization + other ops (after each plot) ----
    out_prefix = os.path.splitext(out_path)[0]

    per_device, per_type, _total = compute_utilization_tables(ops_window_s, comms_window_s)
    # Note: always write, even if empty
    save_utilization_tables(
        per_device,
        per_type,
        out_prefix,
        stage=stage,
        trace_label=run.label,
        time_unit=time_unit,
        time_scale=time_scale,
    )

    other_df = summarize_other_ops(
        ops_window_s,
        stage=stage,
        trace_label=run.label,
        time_unit=time_unit,
        time_scale=time_scale,
    )
    save_other_ops_table(other_df, out_prefix)


def plot_compare_stage(
    run_a: TraceRun,
    run_b: TraceRun,
    *,
    stage: str,
    ops_a_s: pd.DataFrame,
    comms_a_s: pd.DataFrame,
    ops_b_s: pd.DataFrame,
    comms_b_s: pd.DataFrame,
    out_path: str,
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
    """Plot A/B in one PNG with a shared time axis (no independent scaling)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ops_a_s = add_op_groups(ops_a_s)
    ops_b_s = add_op_groups(ops_b_s)

    lanes_a = _build_lanes(ops_a_s, comms_a_s, comm_lane_mode=comm_lane_mode)
    lanes_b = _build_lanes(ops_b_s, comms_b_s, comm_lane_mode=comm_lane_mode)

    w1, h1 = _compute_figsize(
        len(lanes_a),
        fig_w=fig_w,
        fig_min_h=fig_min_h,
        fig_h_scale=fig_h_scale,
        lane_h=lane_h,
        lane_gap=lane_gap,
    )
    w2, h2 = _compute_figsize(
        len(lanes_b),
        fig_w=fig_w,
        fig_min_h=fig_min_h,
        fig_h_scale=fig_h_scale,
        lane_h=lane_h,
        lane_gap=lane_gap,
    )

    # Add a little extra space for a shared legend
    fig_h_total = h1 + h2 + 1.6
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(fig_w, fig_h_total),
        dpi=dpi,
        gridspec_kw={"height_ratios": [max(1.0, h1), max(1.0, h2)]},
    )

    ax1.set_title(f"{title}\n{run_a.label}")
    n1, max1, handles1 = draw_gantt_on_ax(
        ax1,
        ops_a_s,
        comms_a_s,
        time_scale=time_scale,
        time_unit=time_unit,
        comm_lane_mode=comm_lane_mode,
        tiny_event_threshold_s=tiny_event_threshold_s,
        lane_h=lane_h,
        lane_gap=lane_gap,
        title="",
        show_xlabel=False,
    )
    # draw_gantt_on_ax will overwrite title; reset to our multi-line title
    ax1.set_title(f"{title}\n{run_a.label}")

    ax2.set_title(run_b.label)
    n2, max2, handles2 = draw_gantt_on_ax(
        ax2,
        ops_b_s,
        comms_b_s,
        time_scale=time_scale,
        time_unit=time_unit,
        comm_lane_mode=comm_lane_mode,
        tiny_event_threshold_s=tiny_event_threshold_s,
        lane_h=lane_h,
        lane_gap=lane_gap,
        title=run_b.label,
        show_xlabel=True,
    )

    # Shared xlim: align on same absolute timeline
    x_max = max(max1, max2)
    ax1.set_xlim(0.0, x_max * 1.02 if x_max > 0 else 1.0)

    # Shared legend: union handles
    def _key(h: Patch) -> str:
        return str(h.get_label())

    handle_map: Dict[str, Patch] = {}
    for h in handles1 + handles2:
        handle_map[_key(h)] = h
    handles = list(handle_map.values())

    if handles:
        fig.tight_layout(rect=[0.0, 0.10, 1.0, 1.0])
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(5, len(handles)),
            fontsize=16,
            frameon=False,
        )
    else:
        fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # ---- Save utilization + other ops (combined CSV with trace column) ----
    out_prefix = os.path.splitext(out_path)[0]

    per_dev_a, per_type_a, _ = compute_utilization_tables(ops_a_s, comms_a_s)
    per_dev_b, per_type_b, _ = compute_utilization_tables(ops_b_s, comms_b_s)

    per_dev = pd.concat([per_dev_a.assign(trace=run_a.label), per_dev_b.assign(trace=run_b.label)], ignore_index=True)
    per_type = pd.concat([per_type_a.assign(trace=run_a.label), per_type_b.assign(trace=run_b.label)], ignore_index=True)

    # Now reuse saver (it will insert trace again if we let it). We'll call a simplified writer here.
    # Keep columns consistent with single-run saver.
    if per_dev.empty:
        per_dev_out = pd.DataFrame(columns=["trace", "stage", "device", "device_type", "busy_time_s", "total_time_s", "utilization", f"busy_time_{time_unit}", f"total_time_{time_unit}"])
    else:
        per_dev_out = per_dev.copy()
        per_dev_out.insert(1, "stage", stage)
        per_dev_out[f"busy_time_{time_unit}"] = per_dev_out["busy_time_s"] * time_scale
        per_dev_out[f"total_time_{time_unit}"] = per_dev_out["total_time_s"] * time_scale

    if per_type.empty:
        per_type_out = pd.DataFrame(columns=["trace", "stage", "device_type", "n_devices", "busy_time_sum_s", "total_time_s", "utilization_avg", f"busy_time_sum_{time_unit}", f"total_time_{time_unit}"])
    else:
        per_type_out = per_type.copy()
        per_type_out.insert(1, "stage", stage)
        per_type_out[f"busy_time_sum_{time_unit}"] = per_type_out["busy_time_sum_s"] * time_scale
        per_type_out[f"total_time_{time_unit}"] = per_type_out["total_time_s"] * time_scale

    per_dev_out.to_csv(f"{out_prefix}_util_device.csv", index=False)
    per_type_out.to_csv(f"{out_prefix}_util_type.csv", index=False)

    other_a = summarize_other_ops(ops_a_s, stage=stage, trace_label=run_a.label, time_unit=time_unit, time_scale=time_scale)
    other_b = summarize_other_ops(ops_b_s, stage=stage, trace_label=run_b.label, time_unit=time_unit, time_scale=time_scale)
    other_all = pd.concat([other_a, other_b], ignore_index=True)
    save_other_ops_table(other_all, out_prefix)


# ---------------------------
# Job builder for batch mode
# ---------------------------


def build_jobs_from_args(args: argparse.Namespace) -> List[Tuple[str, str, str, Optional[str]]]:
    """Returns jobs: (policy_name, length, ops_csv_path, comms_csv_path_or_none)."""
    jobs: List[Tuple[str, str, str, Optional[str]]] = []

    if args.csv:
        ops_csv, comms_csv = infer_pair_paths(args.csv)
        ops_csv = _normalize_path(ops_csv)
        comms_csv = _normalize_path(comms_csv)
        if not os.path.isfile(ops_csv):
            raise FileNotFoundError(_debug_path_not_found(ops_csv))
        jobs.append(("single", "single", ops_csv, comms_csv if os.path.isfile(comms_csv) else None))
        return jobs

    if not args.base_dir:
        raise ValueError("Must provide either --csv, --compare, or --base_dir.")
    if not args.length:
        raise ValueError("--length is required when using --base_dir.")

    base_dir = _normalize_path(args.base_dir)
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"base_dir not found: {base_dir}")

    if args.policy is None or args.policy.lower() == "all":
        for name in sorted(os.listdir(base_dir)):
            pdir = os.path.join(base_dir, name)
            if not os.path.isdir(pdir):
                continue
            try:
                ops_csv, comms_csv = find_csv_pair_in_policy_dir(pdir, args.length)
            except FileNotFoundError:
                continue
            jobs.append((name, args.length, ops_csv, comms_csv))
        if not jobs:
            raise FileNotFoundError(f"No policies found under {base_dir} with length={args.length}")
        return jobs

    policy_dir = os.path.join(base_dir, args.policy)
    if not os.path.isdir(policy_dir):
        raise NotADirectoryError(f"policy dir not found: {policy_dir}")
    ops_csv, comms_csv = find_csv_pair_in_policy_dir(policy_dir, args.length)
    jobs.append((args.policy, args.length, ops_csv, comms_csv))
    return jobs


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    ap = argparse.ArgumentParser()

    # Single/batch
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to either *_ops_trace.csv or *_comms_trace.csv. The paired file will be inferred.",
    )
    ap.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base dir that contains policy sub-folders (e.g., .../qwen_7b_int8_b8_s64/).",
    )
    ap.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Policy folder name. Use 'all' to plot every policy under base_dir.",
    )
    ap.add_argument(
        "--length",
        type=str,
        default=None,
        help="Length key in filename, e.g., 128x512 (used when searching under --base_dir).",
    )

    # Compare
    ap.add_argument(
        "--compare",
        nargs=2,
        default=None,
        metavar=("TRACE_A", "TRACE_B"),
        help="Compare mode: give two traces (CSV ops/comms or JSON). Draw on same time axis in one PNG.",
    )
    ap.add_argument(
        "--compare_labels",
        nargs=2,
        default=None,
        metavar=("LABEL_A", "LABEL_B"),
        help="Optional labels for compare mode.",
    )

    # Output + plot options
    ap.add_argument("--out_dir", type=str, default="plots", help="Where to save images. Default: ./plots")
    ap.add_argument(
        "--decode_token",
        type=int,
        default=None,
        help="Which decode token to plot (0-based). In compare mode, applies to both (will clamp per run).",
    )
    ap.add_argument(
        "--comm_lane_mode",
        type=str,
        default="aggregate",
        choices=["aggregate", "per_link"],
        help="data transfer lane mode: aggregate(all links in one row) or per_link(one row per src->dst).",
    )
    ap.add_argument(
        "--time_unit",
        type=str,
        default="s",
        choices=["s", "ms"],
        help="Time unit shown on axis. s=seconds, ms=milliseconds.",
    )
    ap.add_argument(
        "--tiny_event_threshold_s",
        type=float,
        default=5e-6,
        help="Events shorter than this threshold (seconds) will be drawn as vertical lines (more visible).",
    )

    # Figure geometry (默认更高更窄)
    ap.add_argument("--fig_w", type=float, default=11.5, help="Figure width in inches (default narrower than before).")
    ap.add_argument("--fig_min_h", type=float, default=6.0, help="Minimum figure height in inches.")
    ap.add_argument("--fig_h_scale", type=float, default=0.85, help="Height scale factor (default taller than before).")
    ap.add_argument("--lane_h", type=float, default=0.85, help="Lane rectangle height (axis units).")
    ap.add_argument("--lane_gap", type=float, default=0.28, help="Gap between lanes (axis units).")
    ap.add_argument("--dpi", type=int, default=180, help="PNG dpi.")

    args = ap.parse_args()

    time_scale = 1000.0 if args.time_unit == "ms" else 1.0
    out_dir = _normalize_path(args.out_dir)

    # ---------------- Compare mode ----------------
    if args.compare is not None:
        a_path, b_path = args.compare
        if args.compare_labels is not None:
            la, lb = args.compare_labels
        else:
            la, lb = None, None

        run_a = load_trace_run(a_path, label=la, allow_missing_comms=True)
        run_b = load_trace_run(b_path, label=lb, allow_missing_comms=True)

        # Prefill
        pre_a_ops = _shift_to_zero(run_a.ops[run_a.ops["phase"] == "prefill"].copy())
        pre_a_comms = _shift_to_zero(run_a.comms[run_a.comms["phase"] == "prefill"].copy())
        pre_b_ops = _shift_to_zero(run_b.ops[run_b.ops["phase"] == "prefill"].copy())
        pre_b_comms = _shift_to_zero(run_b.comms[run_b.comms["phase"] == "prefill"].copy())

        slug = _safe_slug(f"{run_a.label}_vs_{run_b.label}")
        pre_out = os.path.join(out_dir, f"compare_{slug}_prefill.pdf")
        plot_compare_stage(
            run_a,
            run_b,
            stage="prefill",
            ops_a_s=pre_a_ops,
            comms_a_s=pre_a_comms,
            ops_b_s=pre_b_ops,
            comms_b_s=pre_b_comms,
            out_path=pre_out,
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
        print(f"[OK] compare prefill -> {pre_out}")

        # Decode (if exists)
        tw_a = detect_decode_token_windows(run_a.ops)
        tw_b = detect_decode_token_windows(run_b.ops)
        if not tw_a.windows or not tw_b.windows:
            print("[WARN] compare: one side has no decode phase; skip decode-token comparison.")
            return

        n_a = len(tw_a.windows)
        n_b = len(tw_b.windows)

        if args.decode_token is None:
            idx_a = n_a // 2
            idx_b = n_b // 2
        else:
            idx_a = max(0, min(int(args.decode_token), n_a - 1))
            idx_b = max(0, min(int(args.decode_token), n_b - 1))

        t0a, t1a = tw_a.windows[idx_a]
        t0b, t1b = tw_b.windows[idx_b]

        dec_a_ops = _slice_and_shift(run_a.ops[run_a.ops["phase"] == "decode"].copy(), t0a, t1a)
        dec_a_comms = _slice_and_shift(run_a.comms[run_a.comms["phase"] == "decode"].copy(), t0a, t1a)
        dec_b_ops = _slice_and_shift(run_b.ops[run_b.ops["phase"] == "decode"].copy(), t0b, t1b)
        dec_b_comms = _slice_and_shift(run_b.comms[run_b.comms["phase"] == "decode"].copy(), t0b, t1b)

        dec_out = os.path.join(
            out_dir,
            f"compare_{slug}_decode_A{idx_a+1}of{n_a}_B{idx_b+1}of{n_b}.pdf",
        )
        plot_compare_stage(
            run_a,
            run_b,
            stage="decode",
            ops_a_s=dec_a_ops,
            comms_a_s=dec_a_comms,
            ops_b_s=dec_b_ops,
            comms_b_s=dec_b_comms,
            out_path=dec_out,
            title=f"decode token  A:{idx_a+1}/{n_a} (method={tw_a.method})  vs  B:{idx_b+1}/{n_b} (method={tw_b.method})",
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
        print(f"[OK] compare decode -> {dec_out}")
        return

    # ---------------- Single / batch mode ----------------
    jobs = build_jobs_from_args(args)

    for policy, length, ops_csv, comms_csv in jobs:
        # Load run with optional comms
        run = load_trace_run(ops_csv, label=(None if policy == "single" else f"{policy}_{length}"), allow_missing_comms=True)

        # -------- prefill plot --------
        pre_ops = _shift_to_zero(run.ops[run.ops["phase"] == "prefill"].copy())
        pre_comms = _shift_to_zero(run.comms[run.comms["phase"] == "prefill"].copy())

        pre_out = os.path.join(out_dir, f"{policy}_{length}_prefill.pdf")
        plot_single_stage(
            run,
            stage="prefill",
            ops_window_s=pre_ops,
            comms_window_s=pre_comms,
            out_path=pre_out,
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

        # -------- decode token plot --------
        tw = detect_decode_token_windows(run.ops)
        if not tw.windows:
            print(f"[WARN] {run.label}: no decode phase found, skip decode-token plot.")
            continue

        n_tokens = len(tw.windows)
        token_idx = args.decode_token if args.decode_token is not None else (n_tokens // 2)
        token_idx = max(0, min(int(token_idx), n_tokens - 1))
        t0, t1 = tw.windows[token_idx]

        dec_ops = _slice_and_shift(run.ops[run.ops["phase"] == "decode"].copy(), t0, t1)
        dec_comms = _slice_and_shift(run.comms[run.comms["phase"] == "decode"].copy(), t0, t1)

        dec_out = os.path.join(out_dir, f"{policy}_{length}_decode_token{token_idx+1}of{n_tokens}.pdf")
        plot_single_stage(
            run,
            stage="decode",
            ops_window_s=dec_ops,
            comms_window_s=dec_comms,
            out_path=dec_out,
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

        print(f"[OK] {policy} {length}")
        print(f"  prefill:      {pre_out}")
        print(f"  decode token: {dec_out}")


if __name__ == "__main__":
    main()
