#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PYTHONPATH=$PWD/algorithms:$PYTHONPATH \
python ./experiment/plot_roofline.py \
  --devices GPU0 PIM0 \
  --strategies-json ./experiment/roofline_configs/opt_methods.json \
  --outdir ./figs/roofline_multiops

Inputs:
- cost_model.py (imports config.py as "config", located under ./algorithms/config.py)
- optimizations.py (imports task_graph.py as "task_graph")
So we MUST add both repo_root and ./algorithms to sys.path.

Output:
- For each device, phase, batch, seqlen => one PNG.

Bytes mode:
- default: activation+weight (so quantization / weight compression affects intensity)
- optional: activation only (matches "estimate_activation_bytes" only)

PYTHONPATH=$PWD/algorithms:$PYTHONPATH \
python ./experiment/plot_roofline.py \
  --devices GPU0 PIM0 \
  --strategies-json ./experiment/roofline_configs/opt_methods.json \
  --batches 1 32 \
  --prefill-seqlens 4096 \
  --decode-seqlens 8192 \
  --groups QKV_GEN SCORE CONTEXT FFN\
  --phases prefill decode \
  --bytes-mode activation+weight \
  --outdir ./figs/roofline/paper

python ./experiment/plot_roofline.py \
  --hardware-json ./algorithms/examples/hardware_config_scale_down_11pima.json \
  --devices CPU0 PIM0 \
  --strategies-json ./experiment/roofline_configs/opt_methods.json \
  --batches 1\
  --prefill-seqlens 1024 8192 \
  --decode-seqlens 1024 8192 \
  --groups QKV_GEN SCORE CONTEXT FFN \
  --phases prefill decode \
  --bytes-mode activation+weight \
  --outdir ./figs/roofline/hw_hardware_weight

python ./experiment/plot_roofline.py \
  --hardware-json ./algorithms/examples/represent_hardware.json \
  --devices Ascend AiM GA100 TPUv4 \
  --strategies-json ./experiment/roofline_configs/opt_methods.json \
  --batches 1\
  --prefill-seqlens 1024 8192 \
  --decode-seqlens 1024 8192 \
  --groups QKV_GEN SCORE CONTEXT FFN \
  --phases prefill decode \
  --bytes-mode activation+weight \
  --outdir ./figs/roofline/paper_represent
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D


# ---------------------------
# Palette (strategies)
# ---------------------------
# Strategy colors (marker facecolor). Keep a small, soft palette.
# NOTE: Device colors are configured separately below.
PALETTE = [
    "#505180", "#ADC1E4", "#CED8E8", "#DFD5E5",
    "#D5E5D1",
]

# ---------------------------
# Device styles (user specified)
# ---------------------------
# Four devices -> four colors (in the same order as devices are loaded).
DEVICE_EDGE_COLORS = ["black", "#A0C3BB", "#A2D091", "#D8E69C"]

# Keep line styles subtle (color is the primary device differentiator)
DEVICE_LINE_STYLES = [
    (0, (1, 0)),          # solid
    (0, (1, 0)),
    (0, (1, 0)),
    (0, (1, 0)),
]

# Operator groups (keys are CLI-friendly; display_name is what appears in legend)
OP_GROUPS: Dict[str, Dict[str, Any]] = {
    "QKV_GEN": {
        "display": "QKV generation",
        "ops": ["Q", "K", "V"],
        "marker": "o",
    },
    "SCORE": {
        "display": "Score",
        "ops": ["QK"],
        "marker": "^",
    },
    "SOFTMAX": {
        "display": "Softmax",
        "ops": ["SOFTMAX"],
        "marker": "D",
    },
    "CONTEXT": {
        "display": "Context",
        "ops": ["SV"],
        "marker": "v",
    },
    "PROJECTION": {
        "display": "Projection",
        "ops": ["O"],
        "marker": "s",
    },
    "FFN": {
        "display": "FFN",
        "ops": ["FFN_W1", "FFN_W2"],
        "marker": "P",
    },
}

# Underlying ops we need to build nodes for (union of all group members)
UNDERLYING_OPS = sorted({op for g in OP_GROUPS.values() for op in g["ops"]})


# ---------------------------
# Minimal graph/node objects
# ---------------------------
@dataclass
class MiniNode:
    id: str
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    weight_id: Optional[str] = None
    weight_size: int = 0  # bytes
    flops: float = 0.0


@dataclass
class MiniGraph:
    nodes: Dict[str, MiniNode] = field(default_factory=dict)


class DummyCluster:
    """CostModel needs a cluster in __init__, but estimate_* doesn't rely on it."""

    def __init__(self) -> None:
        self.devices = {}

    def devices_by_type(self, _t: str):
        return []

    def get_link_spec(self, _a: str, _b: str):
        class _Spec:
            bw_GBs = 0.0
            flit_size_B = 16
            max_payload_B = 256
            latency_s = 0.0
            overhead_s = 0.0

        return _Spec()


# ---------------------------
# Helpers
# ---------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def import_module_from_path(module_name: str, file_path: Path):
    file_path = Path(file_path)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def setup_syspath(cost_model_path: Path) -> Tuple[Path, Path]:
    """Ensure `import config` (algorithms/config.py) works for cost_model.py.

    Add both repo_root and algorithms/ to sys.path.
    """

    cm_path = Path(cost_model_path).resolve()
    alg_dir = cm_path.parent.resolve()
    repo_root = alg_dir.parent.resolve()

    for p in (str(repo_root), str(alg_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    return repo_root, alg_dir


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return default
        return int(float(s))
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() in ("none", "nan"):
            return default
        return float(s)
    except Exception:
        return default


def sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def format_seqlen(S: int) -> str:
    if S % 1024 == 0:
        return f"{S // 1024}k"
    return str(S)


# ---------------------------
# Device roofline
# ---------------------------
@dataclass
class DeviceInfo:
    name: str
    peak_tflops: float
    mem_bw_GBs: float
    line_style: Tuple[int, Tuple[int, ...]]
    edge_color: str

    @property
    def knee_intensity(self) -> float:
        # peak(TF/s)*1e12 = I*(BW(GB/s)*1e9) => I = peak*1000/BW
        if self.mem_bw_GBs <= 0:
            return float("inf")
        return float(self.peak_tflops * 1000.0 / self.mem_bw_GBs)

    def bound_tflops(self, intensity: float) -> float:
        if intensity <= 0 or not math.isfinite(intensity):
            return 0.0
        return float(min(self.peak_tflops, intensity * self.mem_bw_GBs / 1000.0))


def load_devices(hardware_json: Path, chosen: List[str]) -> List[DeviceInfo]:
    hw = load_json(hardware_json)
    devs = ((hw.get("hardware") or {}).get("devices") or [])
    chosen_list = [x.strip() for x in (chosen or []) if str(x).strip()]
    chosen_set = set(chosen_list)

    # Index devices by name for stable ordering.
    by_name: Dict[str, Dict[str, Any]] = {}
    order_in_json: List[str] = []
    for d in devs:
        name = str(d.get("name") or "").strip()
        if not name:
            continue
        if name in by_name:
            continue
        by_name[name] = d
        order_in_json.append(name)

    # Prefer the CLI order (so device colors are predictable for users).
    if chosen_list:
        names_order = [n for n in chosen_list if n in by_name]
    else:
        names_order = order_in_json

    out: List[DeviceInfo] = []
    for idx, name in enumerate(names_order):
        if chosen_set and name not in chosen_set:
            continue
        d = by_name.get(name, {})
        peak = safe_float(d.get("tflops"), 0.0)
        bw = safe_float(d.get("mem_bw_GBs"), 0.0)
        if peak <= 0.0 or bw <= 0.0:
            continue
        out.append(
            DeviceInfo(
                name=name,
                peak_tflops=peak,
                mem_bw_GBs=bw,
                line_style=DEVICE_LINE_STYLES[idx % len(DEVICE_LINE_STYLES)],
                edge_color=DEVICE_EDGE_COLORS[idx % len(DEVICE_EDGE_COLORS)],
            )
        )
    return out


# ---------------------------
# Build mini nodes from model shape
# ---------------------------

def build_common_dims(shape: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    D = safe_int(shape.get("hidden_dim", shape.get("dim")), 0)
    Hf = safe_int(shape.get("intermediate_dim", shape.get("ffn_dim")), 0)
    qh = safe_int(shape.get("q_head_num", shape.get("q_heads")), 0)
    kvh = safe_int(shape.get("kv_head_num", shape.get("kv_heads")), qh)
    hd = safe_int(shape.get("head_dim"), 0)
    if hd <= 0 and D > 0 and qh > 0:
        hd = D // qh
    return D, Hf, qh, kvh, hd


def estimate_weight_size_bytes(op: str, shape: Dict[str, Any], base_weight_dtype_bytes: int) -> int:
    op = op.upper()
    D, Hf, qh, kvh, hd = build_common_dims(shape)
    q_dim = qh * hd
    kv_dim = kvh * hd
    o_dim = q_dim

    bpe = max(1, int(base_weight_dtype_bytes))

    elems = 0
    if op == "Q":
        elems = D * q_dim
    elif op in ("K", "V"):
        elems = D * kv_dim
    elif op == "O":
        elems = o_dim * D
    elif op in ("FFN_W1", "FFN_W3", "FFN_UP", "FFN_GATE"):
        elems = D * Hf
    elif op in ("FFN_W2", "FFN_DOWN"):
        elems = Hf * D
    else:
        elems = 0

    return int(max(0, elems) * bpe)


def make_node(op: str, shape: Dict[str, Any], kv_len: int, base_weight_dtype_bytes: int) -> MiniNode:
    op_up = op.upper()
    D, Hf, qh, kvh, hd = build_common_dims(shape)
    q_dim = qh * hd
    kv_dim = kvh * hd
    o_dim = q_dim

    attrs: Dict[str, Any] = {
        "layer": 0,
        "dim": D,
        "ffn_dim": Hf,
        "q_heads": qh,
        "kv_heads": kvh,
        "n_kv_heads": kvh,
        "head_dim": hd,
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "o_dim": o_dim,
        "causal": True,
        "kv_len": int(kv_len),  # important for decode attention
    }

    w_bytes = estimate_weight_size_bytes(op_up, shape, base_weight_dtype_bytes)
    weight_id = f"{op_up}_W" if w_bytes > 0 else None

    return MiniNode(
        id=op_up,
        name=op_up,
        attrs=attrs,
        weight_id=weight_id,
        weight_size=int(w_bytes),
        flops=0.0,
    )


def build_base_graph(ops: List[str], shape: Dict[str, Any], kv_len: int, base_weight_dtype_bytes: int) -> MiniGraph:
    g = MiniGraph()
    for op in ops:
        n = make_node(op, shape, kv_len=kv_len, base_weight_dtype_bytes=base_weight_dtype_bytes)
        g.nodes[n.id] = n
    return g


# ---------------------------
# Compute group costs
# ---------------------------
@dataclass
class GroupCost:
    flops: float
    bytes_total: int
    intensity: float


def compute_group_cost(
    cm: Any,
    graph: MiniGraph,
    group_ops: List[str],
    *,
    batch: int,
    seq_len: int,
    phase: str,
    bytes_mode: str,
) -> GroupCost:
    total_flops = 0.0
    total_bytes = 0

    # keep kv_len consistent with current seq_len (especially for decode)
    for node in graph.nodes.values():
        node.attrs["kv_len"] = int(seq_len)

    for op in group_ops:
        node = graph.nodes.get(op.upper())
        if node is None:
            continue

        f = float(cm.estimate_flops(node, int(batch), int(seq_len), str(phase)))
        rd, wr = cm.estimate_activation_bytes(node, int(batch), int(seq_len), str(phase))
        act_bytes = int(rd) + int(wr)

        if bytes_mode == "activation":
            b = act_bytes
        else:
            b = act_bytes + int(getattr(node, "weight_size", 0) or 0)

        total_flops += f
        total_bytes += int(b)

    if total_bytes <= 0:
        intensity = float("inf") if total_flops > 0 else 0.0
    else:
        intensity = float(total_flops / float(total_bytes))

    return GroupCost(flops=total_flops, bytes_total=total_bytes, intensity=intensity)


# ---------------------------
# Opacity mapping
# ---------------------------

def alpha_for_batch_seqlen(
    B: int,
    S: int,
    *,
    batches_sorted: List[int],
    seqlens_sorted: List[int],
    alpha_min: float,
    alpha_max: float,
) -> float:
    """Jointly map (B, S) -> alpha.

    Requirements from user:
    - bigger batch => more opaque
    - bigger seqlen => more opaque
    - prefer different opacities across combinations

    We use a monotone, unique (for all pairs) mapping:
        rank = rank_B * Ns + rank_S
        alpha = lerp(alpha_min, alpha_max, rank/(Nb*Ns-1))
    """

    if not batches_sorted or not seqlens_sorted:
        return float(max(0.0, min(1.0, alpha_max)))

    Nb = len(batches_sorted)
    Ns = len(seqlens_sorted)
    total = Nb * Ns

    try:
        rB = batches_sorted.index(int(B))
    except ValueError:
        # clamp to nearest
        rB = min(range(Nb), key=lambda i: abs(batches_sorted[i] - int(B)))

    try:
        rS = seqlens_sorted.index(int(S))
    except ValueError:
        rS = min(range(Ns), key=lambda i: abs(seqlens_sorted[i] - int(S)))

    if total <= 1:
        return float(max(0.0, min(1.0, alpha_max)))

    rank = rB * Ns + rS
    t = float(rank) / float(total - 1)
    a = float(alpha_min + t * (alpha_max - alpha_min))
    return float(max(0.0, min(1.0, a)))


def annotate_point_bs(
    ax: Any,
    *,
    x: float,
    y: float,
    text: str,
    y_limits: Tuple[float, float],
    offset_pts: float,
    color: str = "saddlebrown",
) -> None:
    """Place B/S annotation above or below the marker with a fixed screen offset."""

    y_min, y_max = y_limits
    # decide above/below based on where the point lies in log space
    if y <= 0 or not math.isfinite(y) or y_min <= 0 or y_max <= 0 or not math.isfinite(y_min) or not math.isfinite(y_max):
        place_below = False
    else:
        logy = math.log10(y)
        logy_min = math.log10(y_min)
        logy_max = math.log10(y_max)
        denom = (logy_max - logy_min)
        frac = (logy - logy_min) / denom if denom > 0 else 0.5
        # if too close to top, put text below
        place_below = frac > 0.85

    if place_below:
        xytext = (0, -offset_pts)
        va = "top"
    else:
        xytext = (0, offset_pts)
        va = "bottom"

    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=7,
        color=color,
        alpha=0.95,
        zorder=4,
        annotation_clip=True,
    )


# ---------------------------
# Plot helpers
# ---------------------------

def nice_log_limits(
    values: List[float],
    pad_decades: float = 0.2,
    min_floor: float = 1e-8,
) -> Tuple[float, float]:
    vs = [v for v in values if v > 0 and math.isfinite(v)]
    if not vs:
        return 1e-3, 1e3
    mn, mx = min(vs), max(vs)
    mn = max(mn, min_floor)
    # Tighter limits: pad in *fractional* log10 decades instead of snapping to full decades.
    lo = 10 ** (math.log10(mn) - float(pad_decades))
    hi = 10 ** (math.log10(mx) + float(pad_decades))
    hi = max(hi, lo * 1.2)
    return lo, hi


def draw_phase(
    *,
    ax: plt.Axes,
    phase: str,
    devices: List[DeviceInfo],
    strategy_names: List[str],
    strategy_colors: Dict[str, str],
    # points[device][strategy][group] = list of (I, bound, batch, seqlen)
    points: Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int, int]]]]],
    groups_to_plot: List[str],
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    annotate_bs: bool = False,
    label_offset_pts: float = 16.0,
    marker_area: float = 220.0,
    point_alpha: float = 0.9,
) -> Tuple[List[Line2D], List[Line2D], List[Line2D]]:
    """Draw a single phase into an existing axis.

    Returns: (device_handles, strategy_handles, operator_handles)
    """

    x_min, x_max = x_limits
    y_min, y_max = y_limits

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", linestyle=":", alpha=0.35)

    # Make each subplot square.
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    # Subplot title: keep only 'prefill' or 'decode'
    ax.set_title(str(phase))

    # Rooflines
    dev_handles: List[Line2D] = []
    for dev in devices:
        knee = dev.knee_intensity
        # slope segment: y = I*BW/1000
        x1 = x_min
        x2 = min(knee, x_max)
        y1 = max(y_min, x1 * dev.mem_bw_GBs / 1000.0)
        y2 = max(y_min, x2 * dev.mem_bw_GBs / 1000.0)
        ax.plot([x1, x2], [y1, y2], color=dev.edge_color, linewidth=2.6, linestyle=dev.line_style, alpha=0.95)

        # flat segment
        if knee < x_max:
            ax.plot(
                [max(knee, x_min), x_max],
                [dev.peak_tflops, dev.peak_tflops],
                color=dev.edge_color,
                linewidth=2.6,
                linestyle=dev.line_style,
                alpha=0.95,
            )

        dev_handles.append(
            Line2D(
                [0],
                [0],
                color=dev.edge_color,
                linestyle=dev.line_style,
                linewidth=2.6,
                label=f"{dev.name}",
            )
        )

    # Points
    for dev in devices:
        dev_name = dev.name
        for s in strategy_names:
            col = strategy_colors[s]
            for g in groups_to_plot:
                mk = OP_GROUPS[g]["marker"]
                pts = points[dev_name][s].get(g, [])
                if not pts:
                    continue

                xs_: List[float] = []
                ys_: List[float] = []
                filtered: List[Tuple[float, float, int, int]] = []
                for (I, bound, B, S) in pts:
                    if not (I > 0 and math.isfinite(I) and bound > 0 and math.isfinite(bound)):
                        continue
                    xs_.append(I)
                    ys_.append(bound)
                    filtered.append((I, bound, int(B), int(S)))

                if not xs_:
                    continue

                ax.scatter(
                    xs_,
                    ys_,
                    s=float(marker_area),
                    marker=mk,
                    facecolors=col,
                    edgecolors=dev.edge_color,
                    linewidths=0.8,
                    alpha=float(point_alpha),
                    zorder=3,
                )

                if annotate_bs:
                    for (x, y, B, S) in filtered:
                        annotate_point_bs(
                            ax,
                            x=x,
                            y=y,
                            text=f"B={B}, S={S}",
                            y_limits=(y_min, y_max),
                            offset_pts=float(label_offset_pts),
                            color="saddlebrown",
                        )

    # Legend handles
    strat_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=strategy_colors[s],
            markeredgecolor="black",
            label=s,
        )
        for s in strategy_names
    ]

    op_handles = [
        Line2D(
            [0],
            [0],
            marker=OP_GROUPS[g]["marker"],
            linestyle="None",
            color="black",
            markersize=8,
            label=OP_GROUPS[g]["display"].replace(" generation", ""),
        )
        for g in groups_to_plot
    ]

    return dev_handles, strat_handles, op_handles


def plot_phases_subplots(
    *,
    phases: List[str],
    devices: List[DeviceInfo],
    strategy_names: List[str],
    strategy_colors: Dict[str, str],
    points_by_phase: Dict[str, Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int, int]]]]]],
    groups_to_plot: List[str],
    out_path: Path,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    annotate_bs: bool = False,
    label_offset_pts: float = 16.0,
    marker_area: float = 220.0,
) -> None:
    """Plot multiple phases into one figure (each phase is one subplot)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    phases = [str(p).strip().lower() for p in phases if str(p).strip()]
    if not phases:
        phases = ["prefill", "decode"]

    # Prefer the common paper layout: prefill | decode (left -> right)
    if set(phases) >= {"prefill", "decode"}:
        phases = ["prefill", "decode"] + [p for p in phases if p not in ("prefill", "decode")]

    n = len(phases)
    ncols = n
    fig_w = 5.2 * ncols
    fig_h = 5.2
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]

    # Draw each phase
    dev_handles: List[Line2D] = []
    strat_handles: List[Line2D] = []
    op_handles: List[Line2D] = []
    for ax, phase in zip(axes, phases):
        pts = points_by_phase.get(phase, {})
        dev_h, strat_h, op_h = draw_phase(
            ax=ax,
            phase=phase,
            devices=devices,
            strategy_names=strategy_names,
            strategy_colors=strategy_colors,
            points=pts,
            groups_to_plot=groups_to_plot,
            x_limits=x_limits,
            y_limits=y_limits,
            annotate_bs=bool(annotate_bs),
            label_offset_pts=float(label_offset_pts),
            marker_area=float(marker_area),
        )
        # Keep handles from the first axis (same mapping across all)
        if not dev_handles:
            dev_handles = dev_h
        if not strat_handles:
            strat_handles = strat_h
        if not op_handles:
            op_handles = op_h

    # Axis labels (cleaner for subplots)
    for ax in axes:
        ax.set_xlabel("Arithmetic intensity (FLOPs / Byte)")
    axes[0].set_ylabel("Roofline upper bound (TFLOPs/s)")

    # Legends: keep them *inside* plot corners (upper-left / upper-right / lower-right).
    # We draw the same legend set on every subplot so each panel is self-contained.
    for ax in axes:
        leg_dev = ax.legend(handles=dev_handles, loc="upper left", fontsize=8, frameon=False)
        ax.add_artist(leg_dev)
        leg_strat = ax.legend(handles=strat_handles, loc="upper right", fontsize=8, frameon=False)
        ax.add_artist(leg_strat)
        ax.legend(handles=op_handles, loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--cost-model", type=str, default="./algorithms/cost_model.py")
    parser.add_argument("--optimizations", type=str, default="./algorithms/optimizations.py")
    parser.add_argument("--model-shape", type=str, default="./configs/llama_7b_shape.json")
    parser.add_argument("--hardware-json", type=str, default="./algorithms/examples/hardware_config_gpu_7aim.json")

    # Strategies:
    # Option A: a single JSON with {"strategies":[{"name":..., "config":{...}}, ...]}  (recommended)
    parser.add_argument("--strategies-json", type=str, default="", help="JSON with a list of strategies (name+config).")
    # Option B: provide multiple --opt-json (each is one strategy)
    parser.add_argument("--opt-json", action="append", default=[], help="One strategy config per file (repeatable).")

    parser.add_argument("--devices", nargs="*", default=["GPU0", "PIM0"], help="Devices to plot on the same figure")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=[],
        help="Which operator groups to plot. Choices: " + ", ".join(OP_GROUPS.keys()),
    )
    parser.add_argument("--outdir", type=str, default="../figs/roofline_bubbles")

    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--base-weight-dtype-bytes", type=int, default=2)

    parser.add_argument("--bytes-mode", type=str, default="activation+weight", choices=["activation", "activation+weight"])

    parser.add_argument("--batches", nargs="*", type=int, default=[1, 8, 32])

    # Backward-compatible seqlens
    parser.add_argument(
        "--seqlens",
        nargs="*",
        type=int,
        default=[1024, 2048, 4096, 8192],
        help="Fallback seqlens used when phase-specific seqlens are not provided.",
    )

    # Phase-specific seqlens
    parser.add_argument(
        "--prefill-seqlens",
        nargs="*",
        type=int,
        default=[],
        help="Prefill seqlens (prompt length). If empty, fall back to --seqlens.",
    )
    parser.add_argument(
        "--decode-seqlens",
        nargs="*",
        type=int,
        default=[],
        help="Decode seqlens (KV/cache length). If empty, fall back to --seqlens.",
    )

    parser.add_argument("--phases", nargs="*", type=str, default=["prefill", "decode"])

    parser.add_argument(
        "--only-strategies",
        nargs="*",
        default=[],
        help="Optional: filter strategy names (works only with --strategies-json).",
    )

    # Annotation (default OFF to reduce clutter)
    ann_group = parser.add_mutually_exclusive_group()
    ann_group.add_argument(
        "--annotate-bs",
        action="store_true",
        help="Enable the brown 'B=..., S=...' annotation next to points.",
    )
    # Legacy flag kept for backward compatibility (now a no-op because the default is already OFF)
    ann_group.add_argument(
        "--no-annotate-bs",
        action="store_true",
        help="(legacy) Disable the brown 'B=..., S=...' annotation next to points.",
    )
    parser.add_argument(
        "--label-offset-pts",
        type=float,
        default=16.0,
        help="Text offset (in points) for the brown annotation above/below each marker.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.25, help="Minimum marker opacity.")
    parser.add_argument("--alpha-max", type=float, default=1.0, help="Maximum marker opacity.")
    parser.add_argument("--marker-area", type=float, default=220.0, help="Marker area for scatter points (points^2).")

    args = parser.parse_args()

    # sys.path fix
    setup_syspath(Path(args.cost_model))

    shape = load_json(Path(args.model_shape))

    # import cost_model + optimizations
    cm_mod = import_module_from_path("_cm_mod", Path(args.cost_model))
    opt_mod = import_module_from_path("_opt_mod", Path(args.optimizations))
    CostModel = getattr(cm_mod, "CostModel")
    apply_optimizations_to_graph = getattr(opt_mod, "apply_optimizations_to_graph")

    # devices
    devices = load_devices(Path(args.hardware_json), args.devices)
    if not devices:
        raise ValueError("No valid devices selected (need tflops>0 and mem_bw_GBs>0).")

    # groups
    groups_to_plot = [g.upper() for g in (args.groups or []) if g.strip()]
    if not groups_to_plot:
        groups_to_plot = list(OP_GROUPS.keys())
    for g in groups_to_plot:
        if g not in OP_GROUPS:
            raise ValueError(f"Unknown group {g}. Choices: {list(OP_GROUPS.keys())}")

    # Phase-specific seqlens resolution
    prefill_seqlens = [int(x) for x in (args.prefill_seqlens or []) if int(x) > 0] or [int(x) for x in args.seqlens]
    decode_seqlens = [int(x) for x in (args.decode_seqlens or []) if int(x) > 0] or [int(x) for x in args.seqlens]

    phases = [p.strip().lower() for p in (args.phases or []) if p.strip()]
    if not phases:
        phases = ["prefill", "decode"]

    seqlens_per_phase: Dict[str, List[int]] = {}
    for ph in phases:
        if ph == "prefill":
            seqlens_per_phase[ph] = prefill_seqlens
        elif ph == "decode":
            seqlens_per_phase[ph] = decode_seqlens
        else:
            seqlens_per_phase[ph] = [int(x) for x in args.seqlens]

    seqlens_all = sorted({s for lst in seqlens_per_phase.values() for s in lst if int(s) > 0})
    if not seqlens_all:
        seqlens_all = [int(x) for x in args.seqlens if int(x) > 0]

    # Build base graph template once (kv_len will be updated per seqlen)
    base_graph = build_base_graph(
        UNDERLYING_OPS,
        shape=shape,
        kv_len=int(max(seqlens_all) if seqlens_all else max(args.seqlens)),
        base_weight_dtype_bytes=int(args.base_weight_dtype_bytes),
    )

    # Load strategies
    strategy_graphs: List[Tuple[str, MiniGraph]] = []

    if args.strategies_json:
        root = load_json(Path(args.strategies_json))
        strategies = root.get("strategies", [])
        if not isinstance(strategies, list) or not strategies:
            raise ValueError("strategies-json must contain key 'strategies': [ {name, config}, ... ]")

        for item in strategies:
            name = str(item.get("name", "")).strip()
            cfg = item.get("config", {})
            if not name or not isinstance(cfg, dict):
                continue
            if args.only_strategies and name not in set(args.only_strategies):
                continue
            g2 = copy.deepcopy(base_graph)
            apply_optimizations_to_graph(g2, cfg, base_weight_dtype_bytes=int(args.base_weight_dtype_bytes), shape=shape)
            strategy_graphs.append((name, g2))

    # Option B: each opt-json is one strategy
    for p in args.opt_json:
        pth = Path(p)
        cfg = load_json(pth)
        g2 = copy.deepcopy(base_graph)
        apply_optimizations_to_graph(g2, cfg, base_weight_dtype_bytes=int(args.base_weight_dtype_bytes), shape=shape)
        strategy_graphs.append((pth.stem, g2))

    if not strategy_graphs:
        # fallback: a no-op strategy
        strategy_graphs = [("no_opt", base_graph)]

    # assign colors (by strategy order)
    strategy_names = [name for name, _ in strategy_graphs]
    strategy_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(strategy_names)}

    # CostModel
    cm = CostModel(cluster=DummyCluster(), dtype=str(args.dtype), pim_fast_mode=True)

    outdir = Path(args.outdir)

    # Compute points for ALL phases first (so we can unify axis limits)
    points_by_phase: Dict[str, Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int, int]]]]]] = {}

    xs_global: List[float] = []
    ys_global: List[float] = []
    for dev in devices:
        xs_global.append(dev.knee_intensity)
        ys_global.append(dev.peak_tflops)

    for phase in phases:
        seqlens_phase = seqlens_per_phase[phase]

        # points dict
        points: Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int, int]]]]] = {
            dev.name: {s: {g: [] for g in groups_to_plot} for s in strategy_names} for dev in devices
        }

        for sname, g_strat in strategy_graphs:
            for S in seqlens_phase:
                for B in args.batches:
                    for gkey in groups_to_plot:
                        group_ops = OP_GROUPS[gkey]["ops"]
                        cost = compute_group_cost(
                            cm,
                            g_strat,
                            group_ops,
                            batch=int(B),
                            seq_len=int(S),
                            phase=str(phase),
                            bytes_mode=str(args.bytes_mode),
                        )
                        I = cost.intensity
                        if not (I > 0 and math.isfinite(I)):
                            continue

                        for dev in devices:
                            bound = dev.bound_tflops(I)
                            points[dev.name][sname][gkey].append((I, bound, int(B), int(S)))
                            xs_global.append(I)
                            ys_global.append(bound)

        points_by_phase[phase] = points

    # Unified axes across phases (tighter than the previous decade-snapped limits)
    x_limits = nice_log_limits(xs_global, min_floor=1e-8)
    y_limits = nice_log_limits(ys_global, min_floor=1e-6)

    # Plot phases into subplots (prefill | decode)
    out_path = outdir / (
        f"roofline_phases-{sanitize_filename('-'.join(phases))}_bytes-{sanitize_filename(args.bytes_mode)}.png"
    )

    plot_phases_subplots(
        phases=phases,
        devices=devices,
        strategy_names=strategy_names,
        strategy_colors=strategy_colors,
        points_by_phase=points_by_phase,
        groups_to_plot=groups_to_plot,
        out_path=out_path,
        x_limits=x_limits,
        y_limits=y_limits,
        annotate_bs=bool(args.annotate_bs),
        label_offset_pts=float(args.label_offset_pts),
        marker_area=float(args.marker_area),
    )
    print(f"[OK] {out_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()

