#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Roofline plotter.
- Device: roofline curve (dashed)
- Operator: arithmetic intensity vertical line (solid), plus optional roofline-bound points
- Multi-strategy overlay from optimization JSONs (enable true/false)

Uses:
- ./algorithms/cost_model.py : CostModel.estimate_flops / estimate_activation_bytes
- ./algorithms/optimizations.py : apply_optimizations_to_graph (tags node.attrs['opt'])

python ./experiment/plot_roofline.py \
  --ops Q K V O QK SOFTMAX SV FFN_W1 FFN_W2 \
  --devices GPU0 PIM0 \
  --opt-json ./experiment/roofline_configs/opt_baseline.json \
  --opt-json ./experiment/roofline_configs/opt_sparse.json \
  --phase prefill \
  --batch 1 \
  --seq-len 4096 \
  --outdir ./figs/roofline_plots

"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------
# Strategy palette (you provided)
# ---------------------------------------------------------
STRATEGY_COLORS = [
    "#505180",  "#ADC1E4",  "#CED8E8", "#DFD5E5",
    "#D5E5D1",  "#A0C3BB", "#A2D091", "#D8E69C",
]

# Device curve styles (keep dashed; vary dash pattern + gray level)
DEVICE_LINE_STYLES = [
    (0, (6, 3)),
    (0, (3, 2)),
    (0, (2, 2)),
    (0, (8, 3, 2, 3)),
    (0, (10, 4)),
]
DEVICE_GRAY = ["black", "dimgray", "gray", "darkgray", "slategray", "black"]

DEVICE_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]


# ---------------------------------------------------------
# Minimal graph/node objects (enough for optimizations + cost_model)
# ---------------------------------------------------------
@dataclass
class MiniNode:
    id: str
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    weight_id: Optional[str] = None
    weight_size: int = 0   # bytes
    flops: float = 0.0     # fallback, CostModel will compute when it knows op+attrs


@dataclass
class MiniGraph:
    nodes: Dict[str, MiniNode] = field(default_factory=dict)


class DummyCluster:
    """
    CostModel wants a 'cluster' object in __init__.
    For estimate_flops/estimate_activation_bytes we don't need real cluster info.
    """
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


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def import_module_from_path(module_name: str, file_path: Path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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


# ---------------------------------------------------------
# Known operators (you can extend freely)
# ---------------------------------------------------------
KNOWN_OPS = [
    "LN",
    "Q", "K", "V",
    "QK", "SOFTMAX", "SV",
    "O",
    "FFN_W1", "FFN_W3", "SWIGLU", "FFN_W2",
    "GELU",
    "ADD",
    # You can add: "FFN_UP", "FFN_GATE", "FFN_DOWN", "IDENTITY", ...
]


def expand_ops(tokens: List[str]) -> List[str]:
    """
    Allow:
    - exact op name (e.g., QK)
    - prefix token ending with '_' (e.g., FFN_) -> all known ops containing that substring
    """
    out: List[str] = []
    for t in tokens:
        t = str(t).strip().upper()
        if not t:
            continue
        if t.endswith("_"):
            # substring match like optimizations.py does
            for k in KNOWN_OPS:
                if t in k:
                    out.append(k)
        else:
            out.append(t)
    # dedup while preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# ---------------------------------------------------------
# Build node attrs/weights from model shape
# ---------------------------------------------------------
def build_common_dims(shape: Dict[str, Any]) -> Tuple[int, int, int, int, int, int, int]:
    """
    Returns:
      D, Hf, qh, kvh, hd, q_dim, kv_dim
    """
    D = safe_int(shape.get("hidden_dim", shape.get("dim")), 0)
    Hf = safe_int(shape.get("intermediate_dim", shape.get("ffn_dim")), 0)
    qh = safe_int(shape.get("q_head_num", shape.get("q_heads")), 0)
    kvh = safe_int(shape.get("kv_head_num", shape.get("kv_heads")), qh)
    hd = safe_int(shape.get("head_dim"), 0)
    if hd <= 0 and D > 0 and qh > 0:
        hd = D // qh
    q_dim = qh * hd
    kv_dim = kvh * hd
    return D, Hf, qh, kvh, hd, q_dim, kv_dim


def estimate_weight_size_bytes(
    op_name: str,
    *,
    shape: Dict[str, Any],
    base_weight_dtype_bytes: int,
) -> int:
    """
    Provide weight_size to make weight sparsity/quantization apply correctly.
    """
    op = op_name.upper()
    D, Hf, qh, kvh, hd, q_dim, kv_dim = build_common_dims(shape)
    o_dim = q_dim

    if base_weight_dtype_bytes <= 0:
        base_weight_dtype_bytes = 2

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

    return int(max(0, elems) * int(base_weight_dtype_bytes))


def make_node(
    op_name: str,
    *,
    shape: Dict[str, Any],
    layer: int,
    seq_len: int,
    base_weight_dtype_bytes: int,
    causal: bool = True,
) -> MiniNode:
    op = op_name.upper()
    D, Hf, qh, kvh, hd, q_dim, kv_dim = build_common_dims(shape)
    o_dim = q_dim

    attrs: Dict[str, Any] = {
        "layer": int(layer),
        "dim": int(D),
        "ffn_dim": int(Hf),
        "q_heads": int(qh),
        "kv_heads": int(kvh),
        "n_kv_heads": int(kvh),
        "head_dim": int(hd),
        "q_dim": int(q_dim),
        "kv_dim": int(kv_dim),
        "o_dim": int(o_dim),
        "causal": bool(causal),
        "kv_len": int(seq_len),   # used by decode attention in cost_model
    }

    w_bytes = estimate_weight_size_bytes(op, shape=shape, base_weight_dtype_bytes=base_weight_dtype_bytes)
    weight_id = None
    if w_bytes > 0:
        weight_id = f"{op}_W"

    return MiniNode(
        id=op,
        name=op,
        attrs=attrs,
        weight_id=weight_id,
        weight_size=int(w_bytes),
        flops=0.0,
    )


def build_base_graph(
    ops: List[str],
    *,
    shape: Dict[str, Any],
    seq_len: int,
    base_weight_dtype_bytes: int,
    layer: int = 0,
    causal: bool = True,
) -> MiniGraph:
    g = MiniGraph()
    for op in ops:
        if op.upper() not in KNOWN_OPS:
            # allow unknown if user extends; still create node with minimal attrs
            pass
        n = make_node(
            op,
            shape=shape,
            layer=layer,
            seq_len=seq_len,
            base_weight_dtype_bytes=base_weight_dtype_bytes,
            causal=causal,
        )
        g.nodes[n.id] = n
    return g


# ---------------------------------------------------------
# Devices / roofline
# ---------------------------------------------------------
@dataclass
class DeviceInfo:
    name: str
    peak_tflops: float
    mem_bw_GBs: float

    @property
    def knee_intensity(self) -> float:
        # FLOPs/Byte where slope meets peak:
        # peak(TF/s)*1e12 = I*(BW(GB/s)*1e9) => I = peak*1000/BW
        if self.mem_bw_GBs <= 0:
            return float("inf")
        return float(self.peak_tflops * 1000.0 / self.mem_bw_GBs)

    def roofline_tflops(self, intensity: float) -> float:
        if intensity <= 0 or self.mem_bw_GBs <= 0:
            return 0.0
        return float(min(self.peak_tflops, intensity * self.mem_bw_GBs / 1000.0))


def load_devices(hardware_json: Path, chosen: List[str]) -> List[DeviceInfo]:
    hw = load_json(hardware_json)
    devs = ((hw.get("hardware") or {}).get("devices") or [])
    chosen_set = set([x.strip() for x in chosen if x.strip()]) if chosen else set()

    out: List[DeviceInfo] = []
    for d in devs:
        name = str(d.get("name") or "").strip()
        if not name:
            continue
        if chosen_set and name not in chosen_set:
            continue
        peak = safe_float(d.get("tflops"), 0.0)
        bw = safe_float(d.get("mem_bw_GBs"), 0.0)
        if peak <= 0.0 or bw <= 0.0:
            continue
        out.append(DeviceInfo(name=name, peak_tflops=peak, mem_bw_GBs=bw))

    # keep stable order (as in json)
    return out


# ---------------------------------------------------------
# Intensity computation
# ---------------------------------------------------------
@dataclass
class OpCost:
    flops: float
    bytes_rw: int
    weight_bytes: int
    intensity: float  # FLOPs/Byte


def compute_op_cost(
    cm: Any,
    node: MiniNode,
    *,
    batch: int,
    seq_len: int,
    phase: str,
    include_weight_bytes: bool,
) -> OpCost:
    flops = float(cm.estimate_flops(node, int(batch), int(seq_len), str(phase)))
    rd, wr = cm.estimate_activation_bytes(node, int(batch), int(seq_len), str(phase))
    bytes_rw = int(rd) + int(wr)

    w_bytes = int(getattr(node, "weight_size", 0) or 0)
    denom = bytes_rw + (w_bytes if include_weight_bytes else 0)

    if denom <= 0:
        intensity = float("inf") if flops > 0 else 0.0
    else:
        intensity = float(flops / float(denom))

    return OpCost(flops=flops, bytes_rw=bytes_rw, weight_bytes=w_bytes, intensity=intensity)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
def auto_axis_ranges(
    *,
    intensities: List[float],
    devices: List[DeviceInfo],
) -> Tuple[float, float, float, float]:
    # X range from operator intensities + device knees
    xs: List[float] = []
    for x in intensities:
        if x > 0 and math.isfinite(x):
            xs.append(x)
    for dev in devices:
        k = dev.knee_intensity
        if k > 0 and math.isfinite(k):
            xs.append(k)

    if not xs:
        x_min, x_max = 1e-4, 1e4
    else:
        mn = min(xs)
        mx = max(xs)
        # expand 1 decade each side
        x_min = 10 ** (math.floor(math.log10(mn)) - 1)
        x_max = 10 ** (math.ceil(math.log10(mx)) + 1)
        x_min = max(x_min, 1e-8)
        x_max = max(x_max, x_min * 10)

    # Y range from device peaks
    peaks = [dev.peak_tflops for dev in devices if dev.peak_tflops > 0 and math.isfinite(dev.peak_tflops)]
    if not peaks:
        y_min, y_max = 1e-3, 1.0
    else:
        pmax = max(peaks)
        y_max = pmax * 1.5
        # set y_min to 1e-3 TFLOPs/s (=1 GFLOPs/s) or smaller if peaks tiny
        y_min = min(1e-3, pmax / 1e6)
        y_min = max(y_min, 1e-6)

    return x_min, x_max, y_min, y_max


def plot_roofline_for_op(
    op_name: str,
    *,
    devices: List[DeviceInfo],
    strategy_costs: List[Tuple[str, OpCost]],  # (label, cost)
    outdir: Path,
    phase: str,
    batch: int,
    seq_len: int,
    include_weight_bytes: bool,
    show_points: bool,
):
    outdir.mkdir(parents=True, exist_ok=True)

    intensities = [c.intensity for _, c in strategy_costs if c.intensity > 0 and math.isfinite(c.intensity)]
    x_min, x_max, y_min, y_max = auto_axis_ranges(intensities=intensities, devices=devices)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", linestyle=":", alpha=0.35)

    # ---- Plot device rooflines (dashed) ----
    dev_handles: List[Line2D] = []
    for i, dev in enumerate(devices):
        dash = DEVICE_LINE_STYLES[i % len(DEVICE_LINE_STYLES)]
        color = DEVICE_GRAY[i % len(DEVICE_GRAY)]

        knee = dev.knee_intensity
        # Piecewise roofline:
        # y = I * BW/1000 for I <= knee, else y = peak
        # slope segment
        if x_min < knee:
            x1 = x_min
            x2 = min(knee, x_max)
            y1 = max(y_min, x1 * dev.mem_bw_GBs / 1000.0)
            y2 = max(y_min, x2 * dev.mem_bw_GBs / 1000.0)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.0, linestyle=dash)
        # flat segment
        if knee < x_max:
            x1 = max(knee, x_min)
            x2 = x_max
            y = max(y_min, dev.peak_tflops)
            ax.plot([x1, x2], [y, y], color=color, linewidth=2.0, linestyle=dash)

        # knee marker + annotation
        if knee > 0 and math.isfinite(knee):
            ax.scatter([knee], [dev.peak_tflops], s=28, color=color)
            ax.text(
                knee,
                dev.peak_tflops,
                f"  {dev.name}\n  peak={dev.peak_tflops:g} TF/s\n  BW={dev.mem_bw_GBs:g} GB/s",
                fontsize=8,
                ha="left",
                va="bottom",
                color=color,
                alpha=0.9,
            )

        dev_handles.append(Line2D([0], [0], color=color, linestyle=dash, linewidth=2.0,
                                  label=f"{dev.name} roofline (peak={dev.peak_tflops:g} TF/s, BW={dev.mem_bw_GBs:g} GB/s)"))

    # ---- Plot strategies: solid vertical intensity + optional points ----
    strat_handles: List[Line2D] = []
    dev_marker_handles: List[Line2D] = []

    # device marker legend (one time)
    if show_points and devices:
        for i, dev in enumerate(devices):
            mk = DEVICE_MARKERS[i % len(DEVICE_MARKERS)]
            dev_marker_handles.append(
                Line2D([0], [0], marker=mk, linestyle="None", color="black", markersize=7, label=f"{dev.name} bound-point")
            )

    for si, (label, cost) in enumerate(strategy_costs):
        color = STRATEGY_COLORS[si % len(STRATEGY_COLORS)]
        I = cost.intensity

        # Handle zero/inf intensity in log x
        if not (I > 0 and math.isfinite(I)):
            # put at left bound for visibility
            I_plot = x_min
            I_str = "0/inf"
        else:
            I_plot = I
            I_str = f"{I:.3g}"

        ax.axvline(I_plot, color=color, linewidth=2.6, linestyle="-", alpha=0.95)

        denom_name = "act_rw" if not include_weight_bytes else "act_rw+weights"
        strat_handles.append(
            Line2D([0], [0], color=color, linewidth=2.6, linestyle="-",
                   label=f"{label}: I={I_str} FLOPs={cost.flops:.3g} Bytes({denom_name})={cost.bytes_rw + (cost.weight_bytes if include_weight_bytes else 0)}")
        )

        # optional: intersection points on each device roofline
        if show_points:
            for di, dev in enumerate(devices):
                mk = DEVICE_MARKERS[di % len(DEVICE_MARKERS)]
                yb = dev.roofline_tflops(I if (I > 0 and math.isfinite(I)) else x_min)
                yb = max(y_min, yb)
                ax.scatter([I_plot], [yb], marker=mk, s=70, color=color, edgecolors="none", alpha=0.95)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)")
    ax.set_ylabel("Performance Upper Bound (TFLOPs/s)")
    ax.set_title(f"Roofline: {op_name} | phase={phase} | B={batch} | S={seq_len}")

    # Legends: strategies then devices, keep separated
    leg1 = ax.legend(handles=strat_handles, loc="upper left", fontsize=8, frameon=False)
    ax.add_artist(leg1)

    # device roofline legend
    leg2 = ax.legend(handles=dev_handles, loc="lower right", fontsize=8, frameon=False)
    ax.add_artist(leg2)

    # marker legend
    if show_points and dev_marker_handles:
        ax.legend(handles=dev_marker_handles, loc="center right", fontsize=8, frameon=False)

    fig.tight_layout()

    fname = sanitize_filename(f"roofline_{op_name}_phase-{phase}_B{batch}_S{seq_len}.png")
    fig.savefig(outdir / fname, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # Paths (defaults = your repo layout)
    parser.add_argument("--cost-model", type=str, default="./algorithms/cost_model.py")
    parser.add_argument("--optimizations", type=str, default="./algorithms/optimizations.py")
    parser.add_argument("--model-shape", type=str, default="./configs/llama_7b_shape.json")
    parser.add_argument("--hardware-json", type=str, default="./algorithms/examples/hardware_config_gpu_7aim.json")

    # What to draw
    parser.add_argument("--ops", nargs="*", default=[], help="Ops to plot (e.g., Q K V O QK SOFTMAX SV FFN_W1 FFN_W2). Supports prefix like FFN_.")
    parser.add_argument("--devices", nargs="*", default=[], help="Device names from hardware json, e.g., GPU0 PIM0 PIM1")

    # Strategies
    parser.add_argument("--opt-json", action="append", default=[],
                        help="Optimization JSON path (repeatable). baseline: all enable=false; sparse: weight.enable=true, etc.")

    # Runtime params
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--phase", type=str, default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--dtype", type=str, default="fp16", help="CostModel activation dtype baseline (fp16/bf16/int8...)")

    # Weight assumptions (needed so weight sparsity can apply)
    parser.add_argument("--base-weight-dtype-bytes", type=int, default=2, help="Base dense weight dtype bytes, fp16=2")
    parser.add_argument("--include-weight-bytes", action="store_true",
                        help="If set: intensity denominator includes activation R+W plus weights (weight_size). Default: activation only.")

    # Plot knobs
    parser.add_argument("--outdir", type=str, default="./roofline_plots")
    parser.add_argument("--no-points", action="store_true", help="Disable plotting strategy bound points on each device roofline.")
    parser.add_argument("--list-ops", action="store_true", help="Print known ops and exit.")

    args = parser.parse_args()

    if args.list_ops:
        for k in KNOWN_OPS:
            print(k)
        return

    ops = expand_ops(args.ops)
    if not ops:
        print("[ERROR] No --ops specified. Try --list-ops.")
        sys.exit(2)

    # Ensure repo root on sys.path (so cost_model.py can import config/task_graph/etc)
    cm_path = Path(args.cost_model).resolve()        # e.g. .../TriForm/algorithms/cost_model.py
    alg_dir = cm_path.parent.resolve()              # .../TriForm/algorithms
    repo_root = alg_dir.parent.resolve()            # .../TriForm

    for p in (str(repo_root), str(alg_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)


    shape = load_json(Path(args.model_shape))

    # Import modules from provided paths
    cm_mod = import_module_from_path("_cm_mod", Path(args.cost_model))
    opt_mod = import_module_from_path("_opt_mod", Path(args.optimizations))

    CostModel = getattr(cm_mod, "CostModel")
    apply_optimizations_to_graph = getattr(opt_mod, "apply_optimizations_to_graph")

    # Create base graph (NO csv/effects.tsv)
    base_graph = build_base_graph(
        ops,
        shape=shape,
        seq_len=int(args.seq_len),
        base_weight_dtype_bytes=int(args.base_weight_dtype_bytes),
        layer=0,
        causal=True,
    )

    # Strategies:
    # If user gives none, still run "no_opt" baseline
    strategy_graphs: List[Tuple[str, MiniGraph]] = [("no_opt", base_graph)]

    for p in args.opt_json:
        pth = Path(p)
        cfg = load_json(pth)
        g2 = copy.deepcopy(base_graph)
        # annotate in-place
        apply_optimizations_to_graph(
            g2,
            cfg,
            base_weight_dtype_bytes=int(args.base_weight_dtype_bytes),
            shape=shape,
        )
        strategy_graphs.append((pth.stem, g2))

    # Devices
    devices = load_devices(Path(args.hardware_json), args.devices)
    if not devices:
        print("[ERROR] No valid devices selected (need tflops>0 and mem_bw_GBs>0). Check --devices and hardware json.")
        sys.exit(2)

    # CostModel
    cm = CostModel(cluster=DummyCluster(), dtype=str(args.dtype), pim_fast_mode=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot per op
    for op in ops:
        op_up = op.upper()
        if op_up not in base_graph.nodes:
            print(f"[WARNING] op {op_up} not in graph nodes, skip.")
            continue

        costs: List[Tuple[str, OpCost]] = []
        for label, g in strategy_graphs:
            node = g.nodes.get(op_up)
            if node is None:
                continue
            c = compute_op_cost(
                cm,
                node,
                batch=int(args.batch),
                seq_len=int(args.seq_len),
                phase=str(args.phase),
                include_weight_bytes=bool(args.include_weight_bytes),
            )
            costs.append((label, c))

        if not costs:
            print(f"[WARNING] No costs for op {op_up}, skip.")
            continue

        plot_roofline_for_op(
            op_name=op_up,
            devices=devices,
            strategy_costs=costs,
            outdir=outdir,
            phase=str(args.phase),
            batch=int(args.batch),
            seq_len=int(args.seq_len),
            include_weight_bytes=bool(args.include_weight_bytes),
            show_points=(not bool(args.no_points)),
        )
        print(f"[OK] saved {op_up} -> {outdir}")

    print("[DONE]")


if __name__ == "__main__":
    main()
