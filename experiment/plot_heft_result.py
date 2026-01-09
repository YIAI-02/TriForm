# -*- coding: utf-8 -*-
"""
Multi-algorithm trace analysis (prefill/decode) with stride-sampled decode traces.

Usage
-----:
python plot_heft_result.py \
  --baseline_dir /Users/yangjiaqi/WW/project_1/python/TriForm_heft_comm_result/1214_v3_hybrid_true/len_sweep/pima/mixtral_8x7b_int8_b8_s64/algo_attacc\
  --hef_dir      /Users/yangjiaqi/WW/project_1/python/TriForm_heft_comm_result/1214_v3_hybrid_true/len_sweep/pima/mixtral_8x7b_int8_b8_s64/algo_hefthint \
  --prefill 128 --decode 1024 --stride 64 \
  --out_dir /Users/yangjiaqi/WW/project_1/python/TriForm_heft_comm_result/1214_v3_hybrid_true/len_sweep/pima/mixtral_8x7b_int8_b8_s64/plots_heft_vs_attacc

python plot_heft_result.py \
  --baseline_dir /path/to/.../algo_attacc \
  --compare_dir  /path/to/.../algo_mynew \
  --pair_algo mynew \
  --prefill 128 --decode 1024 --stride 64 \
  --out_dir /path/to/out

"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# User-specified palette
# -----------------------------
PALETTE = ["#0024FF", "#0349FF", "#006CFE", "#0092FE",
           "#FF3F04", "#FE5D00", "#FE8000", "#FFBF02"]

# Required by the user for comparisons
COLOR_REF = "#006CFE"     # baseline/reference
COLOR_TARGET = "#FE8000"  # compared algo (e.g., hefthint)

# Phase colors (requested)
COLOR_DECODE = "#0092FE"
COLOR_PREFILL = "#FFBF02"


# -----------------------------
# Utilities
# -----------------------------
def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}. got={list(df.columns)}")


def sanitize_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^0-9a-zA-Z_\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "algo"


def extract_algo_name(algo_dir: Path) -> str:
    """
    Extract algorithm name from directory path.
    If any path segment matches 'algo_xxx', return 'xxx'.
    Otherwise return the directory name.
    """
    for part in algo_dir.parts[::-1]:
        if part.startswith("algo_") and len(part) > len("algo_"):
            return part[len("algo_"):]
    return algo_dir.name


def parse_prefill_decode_from_name(path: Path) -> Optional[Tuple[int, int]]:
    """
    Parse prefill/decode from a filename like '*128x1024*ops_trace.csv' or '*128x2048*comms_trace.csv'
    Returns (prefill, decode) if found, else None.
    """
    m = re.search(r"(\d+)\s*x\s*(\d+)", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def pick_latest(paths: List[Path]) -> Path:
    """Pick the most recently modified file among candidates."""
    if len(paths) == 1:
        return paths[0]
    return max(paths, key=lambda p: p.stat().st_mtime)


def find_ops_trace(algo_dir: Path, prefill: int, decode: int) -> Path:
    """
    Find ops trace:
      - Prefer *{prefill}x{decode}*_ops_trace.csv
      - Fallback: *{prefill}x*_ops_trace.csv choosing closest decode
    """
    patt = f"*{prefill}x{decode}*_ops_trace.csv"
    matches = list(algo_dir.rglob(patt))
    if not matches:
        candidates = list(algo_dir.rglob(f"*{prefill}x*_ops_trace.csv"))
        parsed: List[Tuple[Path, int]] = []
        for p in candidates:
            pdv = parse_prefill_decode_from_name(p)
            if pdv and pdv[0] == prefill:
                parsed.append((p, pdv[1]))
        if not parsed:
            raise FileNotFoundError(f"Cannot find ops trace under {algo_dir} with pattern {patt}")
        parsed.sort(key=lambda x: (abs(x[1] - decode), x[1]))
        best_decode = parsed[0][1]
        best = [p for p, d in parsed if d == best_decode]
        return pick_latest(best)
    return pick_latest(matches)


def find_comms_trace(algo_dir: Path, prefill: int, decode: int) -> Path:
    """
    Find comm trace:
      - Prefer decode==requested decode: *{prefill}x{decode}*_comms_trace.csv
      - Else choose:
          1) smallest decode >= requested
          2) else largest decode available
    """
    exact = list(algo_dir.rglob(f"*{prefill}x{decode}*_comms_trace.csv"))
    if exact:
        return pick_latest(exact)

    candidates = list(algo_dir.rglob(f"*{prefill}x*_comms_trace.csv"))
    parsed: List[Tuple[Path, int]] = []
    for p in candidates:
        pdv = parse_prefill_decode_from_name(p)
        if pdv and pdv[0] == prefill:
            parsed.append((p, pdv[1]))
    if not parsed:
        raise FileNotFoundError(f"Cannot find comms trace under {algo_dir} with prefill={prefill}")

    ge = [(p, d) for p, d in parsed if d >= decode]
    if ge:
        ge.sort(key=lambda x: (x[1], -x[0].stat().st_mtime))
        best_decode = ge[0][1]
        best = [p for p, d in ge if d == best_decode]
        return pick_latest(best)

    parsed.sort(key=lambda x: (x[1], -x[0].stat().st_mtime))
    best_decode = parsed[-1][1]
    best = [p for p, d in parsed if d == best_decode]
    return pick_latest(best)


def interval_union_length(intervals: List[Tuple[float, float]]) -> float:
    """Total length of union of half-open intervals [start,end)."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            total += (cur_e - cur_s)
            cur_s, cur_e = s, e
    total += (cur_e - cur_s)
    return float(total)


# -----------------------------
# Loaders
# -----------------------------
def load_ops(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["phase", "node_id", "op", "device", "device_type", "mode", "start", "end", "duration"], "ops")
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["duration"] = df["duration"].astype(float)
    return df


def load_comms(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["phase", "src", "src_type", "dst", "dst_type", "bytes", "start", "end", "duration", "tag"], "comms")
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["duration"] = df["duration"].astype(float)
    df["bytes"] = df["bytes"].astype(np.int64)
    return df


# -----------------------------
# Core analytics
# -----------------------------
@dataclass
class PhaseWindow:
    start: float
    end: float

    @property
    def makespan(self) -> float:
        return self.end - self.start


@dataclass
class AlgoRun:
    role: str               # baseline_ref / baseline / compare
    algo_dir: Path
    algo_name: str
    ops_path: Path
    comms_path: Path
    ops: pd.DataFrame
    comms: pd.DataFrame
    wins: Dict[str, PhaseWindow]
    decode_win: PhaseWindow
    decode_per_token: float
    decode_iters: pd.DataFrame
    decode_comm: pd.DataFrame
    decode_comm_iters: pd.DataFrame
    n_samples: int
    prefill_tokens: int
    decode_tokens: int


def phase_windows_from_ops(ops: pd.DataFrame) -> Dict[str, PhaseWindow]:
    wins: Dict[str, PhaseWindow] = {}
    for ph, g in ops.groupby("phase"):
        wins[str(ph)] = PhaseWindow(start=float(g["start"].min()), end=float(g["end"].max()))
    return wins


def add_iter_index_ops(ops: pd.DataFrame, phase: str) -> pd.DataFrame:
    sub = ops[ops["phase"] == phase].copy()
    sub = sub.sort_values(["node_id", "start"], kind="mergesort")
    sub["iter"] = sub.groupby("node_id").cumcount()
    return sub


def iter_summary_ops(ops_iter: pd.DataFrame) -> pd.DataFrame:
    g = ops_iter.groupby("iter").agg(
        iter_start=("start", "min"),
        iter_end=("end", "max"),
        n_ops=("duration", "size"),
        compute_sum=("duration", "sum"),
    )
    g["iter_makespan"] = g["iter_end"] - g["iter_start"]
    g["parallelism"] = g["compute_sum"] / g["iter_makespan"]
    return g.reset_index()


def infer_sample_token_indices(iter_starts: np.ndarray, decode_start: float, per_token: float) -> np.ndarray:
    token_idx = np.rint((iter_starts - decode_start) / per_token).astype(int)
    token_idx[token_idx < 0] = 0
    return token_idx


def sampling_pattern(token_idx: np.ndarray, algo: str) -> pd.DataFrame:
    if len(token_idx) <= 1:
        return pd.DataFrame({"algo": [algo], "delta_tokens": [np.nan], "count": [0], "fraction": [np.nan]})
    d = np.diff(token_idx)
    vc = pd.Series(d).value_counts().sort_index()
    out = pd.DataFrame({
        "algo": algo,
        "delta_tokens": vc.index.astype(int),
        "count": vc.values.astype(int),
        "fraction": (vc.values / float(len(d))).round(4),
    })
    return out


def crop_to_window(df: pd.DataFrame, win: PhaseWindow) -> pd.DataFrame:
    return df[(df["start"] < win.end) & (df["end"] > win.start)].copy()


def assign_comm_iter_by_midtime(comm: pd.DataFrame, iter_starts: np.ndarray) -> np.ndarray:
    mid = (comm["start"].values + comm["end"].values) / 2.0
    idx = np.searchsorted(iter_starts, mid, side="right") - 1
    idx[idx < 0] = 0
    idx[idx >= len(iter_starts)] = len(iter_starts) - 1
    return idx.astype(int)


def iter_comm_metrics(comm: pd.DataFrame) -> pd.DataFrame:
    rec = []
    for it, g in comm.groupby("iter"):
        intervals = list(zip(g["start"].values, g["end"].values))
        rec.append({
            "iter": int(it),
            "n_events": int(len(g)),
            "comm_union": interval_union_length(intervals),
            "comm_duration_sum": float(g["duration"].sum()),
            "comm_bytes_sum": int(g["bytes"].sum()),
        })
    out = pd.DataFrame(rec).sort_values("iter")
    return out


def comm_direction_summary(comm: pd.DataFrame) -> pd.DataFrame:
    g = comm.groupby(["phase", "tag", "src_type", "dst_type"]).agg(
        n=("bytes", "size"),
        total_bytes=("bytes", "sum"),
        total_duration_sum=("duration", "sum"),
    ).reset_index()
    g["eff_bytes_per_timeunit"] = g["total_bytes"] / g["total_duration_sum"].replace(0, np.nan)
    return g.sort_values(["phase", "tag", "total_duration_sum"], ascending=[True, True, False])


def comm_total_time_decode(comm: pd.DataFrame) -> float:
    sub = comm[(comm["phase"] == "decode") & (comm["tag"] == "comm")]
    return float(sub["duration"].sum())


def device_type_busy_per_token(ops: pd.DataFrame, phase: str, n_samples: int) -> pd.DataFrame:
    sub = ops[ops["phase"] == phase]
    if sub.empty:
        return pd.DataFrame(columns=["device_type", "sum_duration", "per_token_busy_est"])
    g = sub.groupby("device_type")["duration"].sum().reset_index(name="sum_duration")
    g["per_token_busy_est"] = g["sum_duration"] / float(max(n_samples, 1))
    return g.sort_values("per_token_busy_est", ascending=False)


def node_device_map(ops: pd.DataFrame, phase: str) -> pd.Series:
    sub = ops[ops["phase"] == phase]
    return sub.groupby("node_id")["device"].agg(lambda s: s.mode().iloc[0])


def moved_nodes_table(ref_ops: pd.DataFrame, other_ops: pd.DataFrame, phase: str) -> pd.DataFrame:
    ref_map = node_device_map(ref_ops, phase)
    oth_map = node_device_map(other_ops, phase)

    meta = ref_ops.drop_duplicates("node_id")[["node_id", "op"]].set_index("node_id")

    comp = pd.DataFrame({
        "op": meta["op"],
        "ref_device": ref_map,
        "other_device": oth_map,
    })
    comp["moved"] = comp["ref_device"] != comp["other_device"]
    comp["move"] = comp["ref_device"] + "->" + comp["other_device"]
    return comp.reset_index()


def op_type_speedup_decode(ref_ops: pd.DataFrame, other_ops: pd.DataFrame,
                          ref_n_samples: int, other_n_samples: int) -> pd.DataFrame:
    """
    Compute per-op-type speedup for decode:
      speedup = (ref_per_sample_iter_time) / (other_per_sample_iter_time)
    """
    ref = ref_ops[ref_ops["phase"] == "decode"].groupby("op")["duration"].sum().reset_index(name="ref_sum")
    oth = other_ops[other_ops["phase"] == "decode"].groupby("op")["duration"].sum().reset_index(name="other_sum")
    df = ref.merge(oth, on="op", how="outer").fillna(0.0)
    df["ref_per_iter"] = df["ref_sum"] / float(max(ref_n_samples, 1))
    df["other_per_iter"] = df["other_sum"] / float(max(other_n_samples, 1))

    # Avoid div-by-zero: if other is 0, set inf when ref>0 else NaN
    df["speedup"] = np.where(
        df["other_per_iter"] > 0,
        df["ref_per_iter"] / df["other_per_iter"],
        np.where(df["ref_per_iter"] == 0, np.nan, np.inf)
    )
    df["baseline_weight"] = df["ref_per_iter"]
    df = df.sort_values("baseline_weight", ascending=False)
    return df


# -----------------------------
# Plot helpers (no seaborn)
# -----------------------------
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_decode_iter_and_device_busy(ref_run: AlgoRun, tgt_run: AlgoRun, out: Path) -> None:
    """
    One figure with two subplots:
      (left) decode iter makespan over sampled tokens
      (right) device_type busy time per token for prefill + decode (same axes)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0))

    # Left: iter makespan
    ax = axes[0]
    ax.plot(ref_run.decode_iters["token_idx"], ref_run.decode_iters["iter_makespan"],
            marker="o", label=ref_run.algo_name, color=COLOR_REF)
    ax.plot(tgt_run.decode_iters["token_idx"], tgt_run.decode_iters["iter_makespan"],
            marker="o", label=tgt_run.algo_name, color=COLOR_TARGET)
    ax.set_xlabel("sampled token index")
    ax.set_ylabel("iter makespan (time unit)")
    ax.set_title("Decode iteration makespan (stride-sampled)")
    ax.legend()

    # Right: device busy share by device_type (percentage), both phases
    ax = axes[1]
    phases = [("prefill", ref_run.prefill_tokens), ("decode", ref_run.decode_tokens)]
    busy_rows = []
    for run in [ref_run, tgt_run]:
        for phase, tokens in phases:
            busy = device_type_busy_per_token(run.ops, phase, tokens)
            if busy.empty:
                continue
            busy = busy.copy()
            busy["phase"] = phase
            busy["algo"] = run.algo_name
            total = float(busy["per_token_busy_est"].sum())
            busy["share_pct"] = np.where(total > 0, busy["per_token_busy_est"] / total * 100.0, 0.0)
            busy_rows.append(busy)

    if busy_rows:
        busy_all = pd.concat(busy_rows, ignore_index=True)
    else:
        busy_all = pd.DataFrame(columns=["device_type", "per_token_busy_est", "phase", "algo", "share_pct"])

    device_types = sorted(busy_all["device_type"].unique().tolist()) if not busy_all.empty else []
    x = np.arange(len(device_types))
    width = 0.18

    # Colors requested per algo-phase
    combo_color = {
        (ref_run.algo_name, "prefill"): COLOR_REF,      # pd prefill 006CFE
        (ref_run.algo_name, "decode"): COLOR_DECODE,    # pd decode 0092FE
        (tgt_run.algo_name, "prefill"): COLOR_TARGET,   # hefthint prefill FE8000
        (tgt_run.algo_name, "decode"): COLOR_PREFILL,   # hefthint decode FFBF02
    }

    order = [
        (ref_run.algo_name, "prefill", 0),
        (ref_run.algo_name, "decode", 1),
        (tgt_run.algo_name, "prefill", 2),
        (tgt_run.algo_name, "decode", 3),
    ]

    handles_combo = {}

    for (algo, phase, idx) in order:
        vals = []
        sub = busy_all[(busy_all["algo"] == algo) & (busy_all["phase"] == phase)]
        sub = sub.set_index("device_type") if not sub.empty else pd.DataFrame(columns=["device_type"])
        for dt in device_types:
            vals.append(float(sub.loc[dt, "share_pct"]) if dt in sub.index else 0.0)

        positions = x + (idx - 1.5) * width
        color = combo_color.get((algo, phase), COLOR_DECODE)
        bars = ax.bar(positions, vals, width=width, color=color, edgecolor="black")

        key = f"{algo}-{phase}"
        if key not in handles_combo:
            handles_combo[key] = bars[0]

    ax.set_xticks(x)
    ax.set_xticklabels(device_types, rotation=0)
    ax.set_ylabel("device busy share (%)")
    ax.set_title("Device-type busy share per phase (prefill + decode)")

    ax.legend(handles_combo.values(), handles_combo.keys(), title="algo-phase", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

    savefig(out)


def plot_latency_and_comm_log(ref_run: AlgoRun, tgt_run: AlgoRun,
                                                         metrics_df: pd.DataFrame,
                                                         comm_dir_ref: pd.DataFrame, comm_dir_tgt: pd.DataFrame,
                                                         out: Path) -> None:
    """
    One figure with two subplots (both log-y):
        (left) latency comparison (time only): total/prefill/decode/decode_per_token
        (right) comm comparison (time only): tag=comm, phases prefill+decode, duration sum by direction
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0))

    # Left: latency
    ax = axes[0]
    wanted = ["total_makespan", "prefill_makespan", "decode_makespan", "decode_per_token"]
    sub = metrics_df[metrics_df["metric"].isin(wanted)].copy()
    # Expect exactly two algos rows for each metric
    label_map = {
        "total_makespan": "total",
        "prefill_makespan": "prefill",
        "decode_makespan": "decode",
        "decode_per_token": "decode_per_tok",
    }
    metrics = [m for m in wanted if m in sub["metric"].unique()]
    x = np.arange(len(metrics))
    width = 0.36

    ref_vals = [float(sub[(sub["metric"] == m) & (sub["algo"] == ref_run.algo_name)]["value"].iloc[0]) for m in metrics]
    tgt_vals = [float(sub[(sub["metric"] == m) & (sub["algo"] == tgt_run.algo_name)]["value"].iloc[0]) for m in metrics]

    # Avoid zeros on log scale
    eps = 1e-12
    ref_vals = [v if v > 0 else eps for v in ref_vals]
    tgt_vals = [v if v > 0 else eps for v in tgt_vals]

    ax.bar(x - width/2, ref_vals, width=width, label=ref_run.algo_name, color=COLOR_REF)
    ax.bar(x + width/2, tgt_vals, width=width, label=tgt_run.algo_name, color=COLOR_TARGET)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in metrics], rotation=0)
    ax.set_ylabel("time (log scale)")
    ax.set_yscale("log")
    ax.set_title("Latency comparison (time only, log-y)")
    ax.legend()

    # Right: comm time by direction (prefill + decode, tag=comm)
    ax = axes[1]
    def prep_comm_dir(df: pd.DataFrame, label: str) -> pd.DataFrame:
        sub = df[df["tag"] == "comm"].copy()
        if sub.empty:
            return pd.DataFrame(columns=["algo", "phase", "direction", "total_duration_sum"])
        sub["direction"] = sub["src_type"] + "->" + sub["dst_type"]
        sub["algo"] = label
        return sub[["algo", "phase", "direction", "total_duration_sum"]]

    c_ref = prep_comm_dir(comm_dir_ref, ref_run.algo_name)
    c_tgt = prep_comm_dir(comm_dir_tgt, tgt_run.algo_name)
    c = pd.concat([c_ref, c_tgt], ignore_index=True)

    # Keep top direction-phase combos by max(duration) across both algos
    c["dir_phase"] = c["phase"].astype(str) + ":" + c["direction"].astype(str)
    top_dirs = (
        c.groupby("dir_phase")["total_duration_sum"]
        .max()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    c = c[c["dir_phase"].isin(top_dirs)]
    pivot = c.pivot_table(index="dir_phase", columns="algo", values="total_duration_sum", aggfunc="sum").fillna(0.0)

    for col in [ref_run.algo_name, tgt_run.algo_name]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[[ref_run.algo_name, tgt_run.algo_name]]

    x = np.arange(len(pivot.index))
    phase_color = {"prefill": COLOR_PREFILL, "decode": COLOR_DECODE}
    algo_hatch = {ref_run.algo_name: "", tgt_run.algo_name: "//"}

    ref_vals = [v if v > 0 else eps for v in pivot[ref_run.algo_name].values.astype(float).tolist()]
    tgt_vals = [v if v > 0 else eps for v in pivot[tgt_run.algo_name].values.astype(float).tolist()]

    colors = []
    for name in pivot.index:
        phase = str(name).split(":", 1)[0]
        colors.append(phase_color.get(phase, COLOR_DECODE))

    bars_ref = ax.bar(x - width/2, ref_vals, width=width, color=colors, hatch=algo_hatch[ref_run.algo_name], edgecolor="black")
    bars_tgt = ax.bar(x + width/2, tgt_vals, width=width, color=colors, hatch=algo_hatch[tgt_run.algo_name], edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=20, ha="right")
    ax.set_ylabel("comm time (sum duration, log scale)")
    ax.set_yscale("log")
    ax.set_title("Comm comparison (prefill + decode, tag=comm, log-y)")

    # Legends
    phases_used = []
    for name in pivot.index:
        phase = str(name).split(":", 1)[0]
        if phase not in phases_used:
            phases_used.append(phase)
    phase_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=phase_color.get(p, COLOR_DECODE), edgecolor="black") for p in phases_used]
    phase_labels = phases_used
    if phase_handles:
        phase_legend = ax.legend(phase_handles, phase_labels, title="phase", loc="upper left", bbox_to_anchor=(0.0, 1.02))
        ax.add_artist(phase_legend)
    algo_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=algo_hatch[a]) for a in [ref_run.algo_name, tgt_run.algo_name]]
    ax.legend(algo_handles, [ref_run.algo_name, tgt_run.algo_name], title="algo", loc="upper right", bbox_to_anchor=(1.0, 1.02))

    savefig(out)


def plot_op_type_speedup(op_speed: pd.DataFrame, ref_name: str, other_name: str,
                         out: Path, top_k: int = 25) -> None:
    """
    Plot per-op-type speedup (ref/other) for top_k ops (by ref baseline weight).
    """
    df = op_speed.copy()
    # Keep finite + meaningful baseline weight
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["baseline_weight"] > 0]
    df = df.dropna(subset=["speedup"])
    df = df.head(top_k)

    if df.empty:
        return

    x = np.arange(len(df))
    plt.figure(figsize=(max(10.5, 0.35 * len(df) + 6), 5.0))
    colors = [COLOR_TARGET if v > 1.0 else COLOR_REF for v in df["speedup"].values]
    plt.bar(x, df["speedup"].values, color=colors, edgecolor="black")
    plt.axhline(1.0, linestyle="--", color=PALETTE[6])
    plt.xticks(x, df["op"].tolist(), rotation=35, ha="right")
    plt.ylabel(f"speedup = {ref_name}/{other_name} (per-sampled-iter time)")
    plt.title("Per-op-type speedup on decode (top ops by baseline time)")
    # Legend for color meaning
    over_patch = plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_TARGET, edgecolor="black")
    under_patch = plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_REF, edgecolor="black")
    plt.legend([over_patch, under_patch], [">1 (baseline faster)", "≤1 (baseline slower or equal)"], loc="upper right")
    savefig(out)


# -----------------------------
# Orchestration
# -----------------------------
def build_run(algo_dir: Path, role: str, prefill: int, decode: int) -> Tuple[AlgoRun, Dict[str, int]]:
    """
    Build a run object and return (run, file_prefill_decode_info)
    """
    algo_dir = Path(algo_dir)
    algo_name = extract_algo_name(algo_dir)

    ops_path = find_ops_trace(algo_dir, prefill, decode)
    comms_path = find_comms_trace(algo_dir, prefill, decode)

    ops = load_ops(ops_path)
    comms = load_comms(comms_path)

    wins = phase_windows_from_ops(ops)
    decode_win = wins["decode"]
    decode_per_token = decode_win.makespan / float(decode)

    # decode iters (stride-sampled)
    dec_iter = add_iter_index_ops(ops, "decode")
    iters = iter_summary_ops(dec_iter)
    iters["token_idx"] = infer_sample_token_indices(iters["iter_start"].values, decode_win.start, decode_per_token)
    iters = iters.sort_values("token_idx").reset_index(drop=True)
    n_samples = int(iters.shape[0])

    # comm crop to decode window (fairness)
    comm_decode = crop_to_window(comms, decode_win)
    # assign comm to iter
    if len(comm_decode) > 0 and n_samples > 0:
        comm_decode = comm_decode.copy()
        comm_decode["iter"] = assign_comm_iter_by_midtime(comm_decode, iters["iter_start"].values)
        comm_iters = iter_comm_metrics(comm_decode)
    else:
        comm_iters = pd.DataFrame(columns=["iter", "n_events", "comm_union", "comm_duration_sum", "comm_bytes_sum"])

    pdv_ops = parse_prefill_decode_from_name(ops_path) or (prefill, decode)
    pdv_comm = parse_prefill_decode_from_name(comms_path) or (prefill, decode)

    file_info = {
        "ops_prefill": int(pdv_ops[0]),
        "ops_decode": int(pdv_ops[1]),
        "comm_prefill": int(pdv_comm[0]),
        "comm_decode": int(pdv_comm[1]),
    }

    run = AlgoRun(
        role=role,
        algo_dir=algo_dir,
        algo_name=algo_name,
        ops_path=ops_path,
        comms_path=comms_path,
        ops=ops,
        comms=comms,
        wins=wins,
        decode_win=decode_win,
        decode_per_token=decode_per_token,
        decode_iters=iters,
        decode_comm=comm_decode,
        decode_comm_iters=comm_iters,
        n_samples=n_samples,
        prefill_tokens=int(prefill),
        decode_tokens=int(decode),
    )
    return run, file_info


def choose_target_run(runs: List[AlgoRun], ref_name: str, pair_algo: Optional[str] = None) -> Optional[AlgoRun]:
    """
    Choose a "target" run for pairwise plots (ref vs target).

    Priority:
      0) If pair_algo is provided, pick the run whose algo_name matches it (exact match).
      1) Prefer an algo name containing 'hef' / 'heft' (case-insensitive)
      2) Else prefer first run that is not the reference
    """
    if pair_algo:
        for r in runs:
            if r.algo_name == pair_algo:
                return r

    for r in runs:
        if r.algo_name != ref_name and ("hef" in r.algo_name.lower() or "heft" in r.algo_name.lower()):
            return r
    for r in runs:
        if r.algo_name != ref_name:
            return r
    return None



def analyze(
    baseline_dirs: List[Path],
    compare_dirs: List[Path],
    prefill: int,
    decode: int,
    stride: int,
    out_dir: Path,
    topk_ops: int,
    pair_algo: Optional[str] = None,
) -> None:
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_dirs:
        raise ValueError("Need at least one baseline_dir")

    # Build runs (reference baseline is first baseline_dir)
    runs: List[AlgoRun] = []
    input_rows = []

    # Baselines
    for i, bdir in enumerate(baseline_dirs):
        role = "baseline_ref" if i == 0 else "baseline"
        run, finfo = build_run(Path(bdir), role, prefill, decode)
        runs.append(run)
        input_rows.append({
            "role": role,
            "algo_name": run.algo_name,
            "algo_dir": str(run.algo_dir),
            "ops_trace": str(run.ops_path),
            "comms_trace": str(run.comms_path),
            "ops_file_prefill": finfo["ops_prefill"],
            "ops_file_decode": finfo["ops_decode"],
            "comm_file_prefill": finfo["comm_prefill"],
            "comm_file_decode": finfo["comm_decode"],
        })

    # Compares (e.g., hefthint)
    for cdir in compare_dirs:
        run, finfo = build_run(Path(cdir), "compare", prefill, decode)
        runs.append(run)
        input_rows.append({
            "role": "compare",
            "algo_name": run.algo_name,
            "algo_dir": str(run.algo_dir),
            "ops_trace": str(run.ops_path),
            "comms_trace": str(run.comms_path),
            "ops_file_prefill": finfo["ops_prefill"],
            "ops_file_decode": finfo["ops_decode"],
            "comm_file_prefill": finfo["comm_prefill"],
            "comm_file_decode": finfo["comm_decode"],
        })

    inputs_used = pd.DataFrame(input_rows)
    inputs_used.to_csv(tables_dir / "inputs_used.csv", index=False)

    # Identify reference
    ref_run = runs[0]
    ref_name = ref_run.algo_name

    # Save per-run tables
    samp_rows = []
    comm_dir_rows = []
    dev_busy_rows = []

    for r in runs:
        prefix = sanitize_name(r.algo_name)

        # decode iters + sampling
        r.decode_iters.to_csv(tables_dir / f"decode_iters_{prefix}.csv", index=False)
        samp_rows.append(sampling_pattern(r.decode_iters["token_idx"].values, r.algo_name))

        # decode comm iters
        r.decode_comm_iters.to_csv(tables_dir / f"decode_comm_iters_{prefix}.csv", index=False)

        # comm direction summary (on cropped decode comm)
        comm_dir = comm_direction_summary(r.comms)
        comm_dir["algo"] = r.algo_name
        comm_dir_rows.append(comm_dir)

        # device_type busy per token
        dev_busy = device_type_busy_per_token(r.ops, "decode", r.n_samples)
        dev_busy["algo"] = r.algo_name
        dev_busy_rows.append(dev_busy)

    sampling_df = pd.concat(samp_rows, ignore_index=True)
    sampling_df.to_csv(tables_dir / "sampling_pattern.csv", index=False)

    comm_dir_all = pd.concat(comm_dir_rows, ignore_index=True)
    comm_dir_all.to_csv(tables_dir / "comm_direction_decode.csv", index=False)

    dev_busy_all = pd.concat(dev_busy_rows, ignore_index=True)
    dev_busy_all.to_csv(tables_dir / "device_type_busy_decode.csv", index=False)

    # Metrics (absolute)
    metric_rows = []
    for r in runs:
        total_makespan = float(r.ops["end"].max() - r.ops["start"].min())
        metric_rows.extend([
            {"metric": "total_makespan", "algo": r.algo_name, "value": total_makespan},
            {"metric": "prefill_makespan", "algo": r.algo_name, "value": float(r.wins["prefill"].makespan) if "prefill" in r.wins else np.nan},
            {"metric": "decode_makespan", "algo": r.algo_name, "value": float(r.decode_win.makespan)},
            {"metric": "decode_per_token", "algo": r.algo_name, "value": float(r.decode_per_token)},
            {"metric": "decode_parallelism_mean", "algo": r.algo_name, "value": float(r.decode_iters["parallelism"].mean()) if len(r.decode_iters) else np.nan},
            {"metric": "decode_comm_total_time", "algo": r.algo_name, "value": comm_total_time_decode(r.decode_comm)},
        ])

    metrics_abs = pd.DataFrame(metric_rows)
    metrics_abs.to_csv(tables_dir / "metrics_absolute_long.csv", index=False)

    # Speedups vs reference baseline (first baseline)
    ref_vals = metrics_abs[metrics_abs["algo"] == ref_name].set_index("metric")["value"].to_dict()
    sp_rows = []
    for _, row in metrics_abs.iterrows():
        m = row["metric"]
        algo = row["algo"]
        val = float(row["value"]) if pd.notna(row["value"]) else np.nan
        refv = float(ref_vals.get(m, np.nan))
        sp = (refv / val) if (pd.notna(refv) and pd.notna(val) and val > 0) else np.nan
        sp_rows.append({"metric": m, "algo": algo, "speedup_vs_ref": sp})

    metrics_sp = pd.DataFrame(sp_rows)
    metrics_sp.to_csv(tables_dir / "metrics_speedup_vs_ref_long.csv", index=False)

    # Also output wide forms
    metrics_abs_w = metrics_abs.pivot_table(index="metric", columns="algo", values="value", aggfunc="first")
    metrics_abs_w.to_csv(tables_dir / "metrics_absolute_wide.csv")

    metrics_sp_w = metrics_sp.pivot_table(index="metric", columns="algo", values="speedup_vs_ref", aggfunc="first")
    metrics_sp_w.to_csv(tables_dir / "metrics_speedup_vs_ref_wide.csv")

    # Pairwise analyses (ref vs target for plots and per-op speedup)
    tgt_run = choose_target_run(runs, ref_name, pair_algo=pair_algo)
    if tgt_run is not None:
        tgt_name = tgt_run.algo_name
        prefix_pair = f"{sanitize_name(ref_name)}_vs_{sanitize_name(tgt_name)}"

        # moved nodes (decode)
        moved_decode = moved_nodes_table(ref_run.ops, tgt_run.ops, "decode")
        moved_decode.to_csv(tables_dir / f"moved_nodes_decode_{prefix_pair}.csv", index=False)

        moved_summary_move = moved_decode[moved_decode["moved"]].groupby("move").size().reset_index(name="count").sort_values("count", ascending=False)
        moved_summary_move.to_csv(tables_dir / f"moved_nodes_decode_by_move_{prefix_pair}.csv", index=False)

        moved_summary_op = moved_decode[moved_decode["moved"]].groupby("op").size().reset_index(name="count").sort_values("count", ascending=False)
        moved_summary_op.to_csv(tables_dir / f"moved_nodes_decode_by_op_{prefix_pair}.csv", index=False)

        # op-type speedup
        op_speed = op_type_speedup_decode(ref_run.ops, tgt_run.ops, ref_run.n_samples, tgt_run.n_samples)
        op_speed.to_csv(tables_dir / f"op_type_speedup_decode_{prefix_pair}.csv", index=False)

        # ----- Plots (updated as per requirements) -----
        # (1) decode iter makespan + device busy in one figure
        plot_decode_iter_and_device_busy(
            ref_run, tgt_run,
            plots_dir / f"decode_iter_and_device_busy_{prefix_pair}.png"
        )

        # (2) latency + comm (time only) in one figure, both log-y
        comm_dir_ref = comm_direction_summary(ref_run.decode_comm)
        comm_dir_tgt = comm_direction_summary(tgt_run.decode_comm)
        plot_latency_and_comm_log(
            ref_run, tgt_run,
            metrics_abs,
            comm_dir_ref, comm_dir_tgt,
            plots_dir / f"latency_and_comm_log_{prefix_pair}.png"
        )

        # (3) per-op-type speedup plot
        plot_op_type_speedup(
            op_speed,
            ref_name=ref_run.algo_name,
            other_name=tgt_run.algo_name,
            out=plots_dir / f"op_type_speedup_decode_{prefix_pair}.png",
            top_k=topk_ops,
        )

    # Save run-level metadata (stride, decode/prefill)
    meta = pd.DataFrame([
        {"key": "prefill", "value": int(prefill)},
        {"key": "decode", "value": int(decode)},
        {"key": "stride", "value": int(stride)},
        {"key": "ref_algo", "value": ref_name},
        {"key": "n_algorithms", "value": int(len(runs))},
    ])
    meta.to_csv(tables_dir / "run_meta.csv", index=False)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", nargs="+", required=True,
                   help="One or more baseline algorithm directories. The FIRST one is the reference baseline.")
    p.add_argument("--compare_dir", nargs="*", default=[],
                   help="Optional: directories of algorithms to compare (e.g., hefthint).")
    # Backward-compatible alias
    p.add_argument("--hef_dir", default=None,
                   help="Alias of one --compare_dir (single directory).")
    p.add_argument("--prefill", type=int, required=True)
    p.add_argument("--decode", type=int, required=True)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--topk_ops", type=int, default=25, help="Top-K ops to show in op-type speedup plot.")
    p.add_argument("--pair_algo", type=str, default=None,
                   help="Optional: choose which algo_name to pair with reference for pairwise plots/op-speedup.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dirs = [Path(x) for x in args.baseline_dir]
    compare_dirs = [Path(x) for x in (args.compare_dir or [])]
    if args.hef_dir:
        compare_dirs.append(Path(args.hef_dir))

    analyze(
        baseline_dirs=baseline_dirs,
        compare_dirs=compare_dirs,
        prefill=int(args.prefill),
        decode=int(args.decode),
        stride=int(args.stride),
        out_dir=Path(args.out_dir),
        topk_ops=int(args.topk_ops),
        pair_algo=getattr(args, 'pair_algo', None),
    )


if __name__ == "__main__":
    main()
