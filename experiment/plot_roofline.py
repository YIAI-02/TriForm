#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PYTHONPATH=$PWD/algorithms:$PYTHONPATH \
python ./experiment/plot_roofline.py \
  --devices GPU0 PIM0 \
  --strategies-json ./experiment/roofline_configs/opt_methods.json \
  --outdir ./figs/roofline_multiops


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
  --hardware-json ./algorithms/examples/represent_hardware.json \
  --devices Ascend AiM GA100 TPUv4 PIMoE HBNMP\
  --strategies-json ./experiment/roofline_configs/opt_methods.json \
  --batches 1 32 \
  --prefill-seqlens 1024 \
  --decode-seqlens 1024\
  --groups QKV_GEN SCORE CONTEXT FFN \
  --phases prefill decode \
  --bytes-mode auto \
  --outdir ./figs/roofline/paper_represent \
  --debug

python ./experiment/plot_roofline.py \
  --hardware-json ./algorithms/examples/represent_hardware.json \
  --devices AiM GA100 \
  --strategies-json ./experiment/roofline_configs/opt_baseline.json \
  --batches 1 32 \
  --prefill-seqlens 1024 \
  --decode-seqlens 1024\
  --groups QKV_GEN\
  --phases prefill\
  --bytes-mode auto \
  --outdir ./figs/roofline/hardware_only \
  --debug
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
# Palette (optimizations & batches)
# ---------------------------
OPTIMIZATION_COLORS = ["#5837A8", "#A83747", "#3791A8", "#A89A37"]
BATCH_DOT_COLORS = ["#39A937", "#3760A9"]

# Backward-compat: resolve_strategy_colors uses this name.
PALETTE = list(OPTIMIZATION_COLORS)

# ---------------------------
# Device styles
# ---------------------------
# Only two devices are expected; represent them using black solid vs black dashed.
DEVICE_EDGE_COLOR = "black"
DEVICE_LINE_STYLES: List[Tuple[int, Tuple[int, ...]]] = [
    (0, (1, 0)),   # solid
    (0, (6, 3)),   # dashed
    (0, (3, 2, 1, 2)),  # dashdot (fallback if >2 devices)
]

# Marker outlines
MARKER_EDGE_COLOR = "black"

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


def _is_valid_color(c: str) -> bool:
    if c is None:
        return False
    s = str(c).strip()
    if not s:
        return False
    try:
        _ = mcolors.to_rgba(s)
        return True
    except Exception:
        return False


def _load_strategy_color_map(path: str) -> Dict[str, str]:
    p = str(path or "").strip()
    if not p:
        return {}
    try:
        obj = load_json(Path(p))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in obj.items():
        kk = str(k).strip()
        vv = str(v).strip()
        if kk and _is_valid_color(vv):
            out[kk] = vv
    return out


def resolve_strategy_colors(
    strategy_names: List[str],
    *,
    color_map_file: str = "",
    colors_list: List[str] | None = None,
    color_hints: Dict[str, str] | None = None,
    palette: List[str] | None = None,
) -> Dict[str, str]:
    """Resolve per-strategy colors.

    Priority:
      1) --strategy-color-map (exact name->color)
      2) per-strategy "color" field in --strategies-json
      3) --strategy-colors (assigned by order)
      4) fallback palette
    """

    palette = palette or PALETTE
    colors_list = list(colors_list or [])
    color_hints = dict(color_hints or {})

    file_map = _load_strategy_color_map(color_map_file)

    out: Dict[str, str] = {}
    for i, name in enumerate(strategy_names):
        # 1) file map
        c = file_map.get(name, "")
        if _is_valid_color(c):
            out[name] = str(c).strip()
            continue

        # 2) strategies-json hint
        c = color_hints.get(name, "")
        if _is_valid_color(c):
            out[name] = str(c).strip()
            continue

        # 3) CLI list
        if colors_list:
            c = colors_list[i % len(colors_list)]
            if _is_valid_color(c):
                out[name] = str(c).strip()
                continue

        # 4) fallback palette
        out[name] = str(palette[i % len(palette)]).strip()

    return out


def parse_strategies_from_json_obj(obj: Any) -> List[Dict[str, Any]]:
    """Parse a 'strategies json' in a few common formats.

    Supported formats:
      1) {"strategies": [ {"name":..., "config":{...}, "color":...}, ... ]}
      2) [ {"name":..., "config":{...}}, ... ]
      3) { "baseline": { ... }, "opt1": { ... }, ... }
         (mapping strategy_name -> config OR -> {"config":{...}, "color":...})

    Returns a normalized list of items with at least: {"name": str, "config": dict}.
    """

    # (2) list form
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    # (1) dict with 'strategies'
    if isinstance(obj, dict):
        # If it looks like a *single* optimization config (opt_baseline.json style),
        # do not interpret keys like "quantization"/"sparsity" as strategy names.
        if any(k in obj for k in ("quantization", "sparsity", "attention_sparsity", "optimizations", "optimization", "optim")):
            return []

        st = obj.get("strategies")
        if isinstance(st, list):
            return [x for x in st if isinstance(x, dict)]

        # Some configs may store strategies as a dict mapping.
        if isinstance(st, dict):
            out: List[Dict[str, Any]] = []
            for name, v in st.items():
                if not str(name).strip() or not isinstance(v, dict):
                    continue
                cfg = v.get("config") if isinstance(v.get("config"), dict) else v
                item = {"name": str(name), "config": cfg}
                if v.get("color") is not None:
                    item["color"] = v.get("color")
                out.append(item)
            return out

        # (3) plain mapping name -> config
        out2: List[Dict[str, Any]] = []
        for name, v in obj.items():
            if not str(name).strip():
                continue
            if not isinstance(v, dict):
                continue
            cfg = v.get("config") if isinstance(v.get("config"), dict) else v
            item = {"name": str(name), "config": cfg}
            if v.get("color") is not None:
                item["color"] = v.get("color")
            out2.append(item)
        return out2

    return []


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
    mem_capacity_GB: float
    line_style: Tuple[int, Tuple[int, ...]]
    edge_color: str
    group: str = "unknown"  # e.g., compute | pim
    device_count: int = 1
    multicore_factor: float = 1.0

    @property
    def scale_factor(self) -> float:
        mc = float(self.multicore_factor) if math.isfinite(float(self.multicore_factor)) else 1.0
        if mc <= 0:
            mc = 1.0
        cnt = int(self.device_count) if int(self.device_count) > 0 else 1
        return float(mc) * float(cnt)

    @property
    def label(self) -> str:
        """Legend label that reflects device stacking/multicore."""
        parts: List[str] = [str(self.name)]
        if int(self.device_count) != 1:
            parts.append(f"×{int(self.device_count)}")
        if abs(float(self.multicore_factor) - 1.0) > 1e-9:
            parts.append(f"mc{float(self.multicore_factor):g}")
        return " ".join(parts)

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


def _infer_device_group(name: str, d: Dict[str, Any]) -> str:
    """Infer a coarse group from device name/metadata.

    Requirement (represent_hardware.json):
      - CPU/NPU/GPU share one group/color
      - PIM share one group/color

    This tries metadata first (type/category fields), then falls back to name heuristics.
    """

    # 1) Metadata keys (best-effort)
    for k in ("type", "device_type", "category", "class", "kind", "family"):
        v = str(d.get(k) or "").strip().lower()
        if not v:
            continue
        if "pim" in v or "aim" in v or "in-memory" in v or "in_memory" in v:
            return "pim"
        if "cpu" in v:
            return "compute"
        if "gpu" in v:
            return "compute"
        if "npu" in v or "tpu" in v or "xpu" in v or "accelerator" in v:
            return "compute"

    # 2) Name heuristics (covers represent_hardware.json typical names)
    n = str(name).strip().lower()
    if any(tok in n for tok in ("pim", "aim", "hbmnp", "hbnmp", "inmem", "in-mem", "in_memory")):
        return "pim"
    # Treat TPU/Ascend/etc. as compute group
    if any(tok in n for tok in ("cpu", "gpu", "npu", "tpu", "ascend", "ga100", "a100", "h100", "tpuv")):
        return "compute"

    # Default to compute if unknown (safer than mislabeling as PIM)
    return "compute"


def load_devices(
    hardware_json: Path,
    chosen: List[str]
) -> List[DeviceInfo]:
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
        peak0 = safe_float(d.get("tflops"), 0.0)
        bw = safe_float(d.get("mem_bw_GBs"), 0.0)
        cap0 = safe_float(d.get("mem_capacity_GB"), 0.0)
        if peak0 <= 0.0 or bw <= 0.0:
            continue

        # ------------------------------------------------------------
        # Device stacking / multi-core (represent_hardware.json)
        # ------------------------------------------------------------
        # Requirements:
        #   - each device can specify a multi-core factor and a device count
        #   - compute scales multiplicatively
        #   - internal bandwidth unchanged
        #   - internal storage scales multiplicatively
        mc = safe_float(
            d.get(
                "multicore_factor",
                d.get("multi_core_factor", d.get("multicore", d.get("multi_core", d.get("mc", 1.0)))),
            ),
            1.0,
        )
        if mc <= 0.0 or not math.isfinite(mc):
            mc = 1.0
        cnt = safe_int(
            d.get(
                "device_count",
                d.get("count", d.get("num_devices", d.get("devices", d.get("n_devices", 1)))),
            ),
            1,
        )
        if cnt <= 0:
            cnt = 1

        scale = float(mc) * float(cnt)
        peak = float(peak0) * float(scale)
        cap = float(cap0) * float(scale) if cap0 > 0 else float(cap0)

        group = _infer_device_group(name, d)

        edge_color = DEVICE_EDGE_COLOR
        line_style = DEVICE_LINE_STYLES[idx % len(DEVICE_LINE_STYLES)]

        out.append(
            DeviceInfo(
                name=name,
                peak_tflops=peak,
                mem_bw_GBs=bw,
                mem_capacity_GB=cap,
                line_style=line_style,
                edge_color=edge_color,
                group=group,
                device_count=int(cnt),
                multicore_factor=float(mc),
            )
        )
    return out


def device_requires_weight_loading(dev: DeviceInfo, bytes_mode: str = "auto") -> bool:
    """Whether the device should count weight bytes in arithmetic intensity.

    User requirement:
      - PIM: assume weights do NOT need to be loaded (activation-only)
      - CPU/GPU/NPU: weights need to be loaded (activation + weight)

    bytes_mode:
      - auto (default): follow the above per-device behavior
      - activation: force activation-only for non-PIM devices
      - activation+weight: same as auto for non-PIM (kept for backward compatibility)
    """

    if str(getattr(dev, "group", "")).strip().lower() == "pim":
        return False

    bm = str(bytes_mode or "auto").strip().lower()
    if bm == "activation":
        return False

    # auto / activation+weight
    return True


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
def _fmt_num(x: float) -> str:
    """Compact float formatting for debug prints."""
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return str(v)
    av = abs(v)
    if av == 0.0:
        return "0"
    # Use scientific notation for very small/large values.
    if av >= 1e6 or av < 1e-2:
        return f"{v:.3e}"
    # Otherwise use a compact fixed-point.
    if av >= 1000:
        return f"{v:,.0f}"
    if av >= 100:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def _fmt_bytes(n: int) -> str:
    """Human readable bytes for debug."""
    try:
        nn = int(n)
    except Exception:
        return str(n)
    if nn < 0:
        return str(nn)
    if nn >= (1 << 30):
        return f"{nn / (1 << 30):.3f} GiB"
    if nn >= (1 << 20):
        return f"{nn / (1 << 20):.3f} MiB"
    if nn >= (1 << 10):
        return f"{nn / (1 << 10):.3f} KiB"
    return f"{nn} B"


def debug_dump_strategy_costs(
    cm: Any,
    strategy_graphs: List[Tuple[str, MiniGraph]],
    *,
    phases: List[str],
    seqlens_per_phase: Dict[str, List[int]],
    batches: List[int],
    groups_to_plot: List[str],
) -> None:
    """Print per-op FLOPs/bytes/intensity to help validate roofline inputs.

    This is especially useful for decode where KV-cache read traffic must be
    included for attention (QK/SV).
    """
    dbg_groups = [g.strip().upper() for g in (groups_to_plot or [])]

    b0 = int((batches or [1])[0])

    for ph in phases:
        ph_l = str(ph).strip().lower()
        seqlens = seqlens_per_phase.get(ph_l, []) or []
        if ph_l == "prefill":
            s0 = int((seqlens or [1])[0])
        elif ph_l == "decode":
            s0 = int((seqlens or [1])[0])
        else:
            s0 = int((seqlens or [1])[0])

        print("\n" + "=" * 88)
        print(f"[DEBUG] phase={ph_l}  B={b0}  S={s0}")
        print("=" * 88)

        for sname, g in strategy_graphs:
            print("\n" + "-" * 88)
            print(f"[DEBUG] strategy={sname}")
            print("-" * 88)

            # Keep kv_len consistent with seq_len for this debug case.
            for n in g.nodes.values():
                try:
                    n.attrs["kv_len"] = int(s0)
                except Exception:
                    pass

            for gkey in dbg_groups:
                if gkey not in OP_GROUPS:
                    continue
                ops = OP_GROUPS[gkey]["ops"]
                print(f"\n  [GROUP {gkey}] ops={ops}")

                # Per-op breakdown
                for op in ops:
                    node = g.nodes.get(str(op).upper())
                    if node is None:
                        continue

                    flops = float(cm.estimate_flops(node, int(b0), int(s0), str(ph_l)))
                    rd, wr = cm.estimate_activation_bytes(node, int(b0), int(s0), str(ph_l))
                    act_bytes = int(rd) + int(wr)

                    kv_read = 0
                    if ph_l == "decode":
                        opu = str(getattr(node, "name", op) or op).strip().upper()
                        if opu in ("QK", "SV"):
                            try:
                                kv_read = int(cm.estimate_kv_cache_read_bytes(node, int(b0), int(s0), str(ph_l)))
                            except Exception:
                                kv_read = 0
                    act_bytes_total = act_bytes + kv_read

                    w_bytes = int(getattr(node, "weight_size", 0) or 0)
                    total_act = int(act_bytes_total)
                    total_aw = int(act_bytes_total) + int(w_bytes)

                    intensity_act = (flops / float(total_act)) if total_act > 0 else float("inf")
                    intensity_aw = (flops / float(total_aw)) if total_aw > 0 else float("inf")

                    optd = (getattr(node, "attrs", {}) or {}).get("opt", {})
                    orig_w = optd.get("orig_weight_size")
                    qspec = optd.get("quantization") if isinstance(optd, dict) else None
                    ws = optd.get("weight_sparsity") if isinstance(optd, dict) else None
                    kv_dtype_b = optd.get("kv_dtype_bytes") if isinstance(optd, dict) else None
                    actq = optd.get("activation_quant") if isinstance(optd, dict) else None

                    # Compact opt summary
                    opt_parts: List[str] = []
                    if isinstance(qspec, dict) and qspec.get("mode"):
                        opt_parts.append(f"q={qspec.get('mode')} wbits={qspec.get('weight_bits')}")
                        if qspec.get("activation_bits") is not None:
                            opt_parts.append(f"abits={qspec.get('activation_bits')}")
                    if isinstance(ws, dict) and ws.get("density") is not None:
                        opt_parts.append(f"ws_density={ws.get('density')}")
                        if ws.get("storage"):
                            opt_parts.append(f"ws_store={ws.get('storage')}")
                    if isinstance(actq, dict) and actq.get("act_dtype_bytes") is not None:
                        opt_parts.append(f"actB={actq.get('act_dtype_bytes')}")
                    if kv_dtype_b is not None:
                        opt_parts.append(f"kvB={kv_dtype_b}")
                    opt_str = ", ".join(opt_parts) if opt_parts else "(no opt tags)"

                    print(
                        "    "
                        + f"op={str(op).upper():<8} "
                        + f"FLOPs={_fmt_num(flops):>10} "
                        + f"act={_fmt_bytes(act_bytes):>12} "
                        + f"kv_read={_fmt_bytes(kv_read):>12} "
                        + f"w={_fmt_bytes(w_bytes):>12} "
                        + (f"(orig_w={_fmt_bytes(int(orig_w))}) " if isinstance(orig_w, int) else "")
                        + f"total_act={_fmt_bytes(total_act):>12} "
                        + f"I_act={_fmt_num(intensity_act):>10} "
                        + f"total_aw={_fmt_bytes(total_aw):>12} "
                        + f"I_aw={_fmt_num(intensity_aw):>10} "
                        + f"[{opt_str}]"
                    )

                # Group summary
                gc_act = compute_group_cost(
                    cm,
                    g,
                    ops,
                    batch=int(b0),
                    seq_len=int(s0),
                    phase=str(ph_l),
                    include_weight=False,
                )
                gc_aw = compute_group_cost(
                    cm,
                    g,
                    ops,
                    batch=int(b0),
                    seq_len=int(s0),
                    phase=str(ph_l),
                    include_weight=True,
                )
                print(
                    f"\n    => GROUP_SUM_ACT  FLOPs={_fmt_num(gc_act.flops)}  bytes={_fmt_bytes(gc_act.bytes_total)}  "
                    f"I={_fmt_num(gc_act.intensity)}"
                )
                print(
                    f"    => GROUP_SUM_A+W  FLOPs={_fmt_num(gc_aw.flops)}  bytes={_fmt_bytes(gc_aw.bytes_total)}  "
                    f"I={_fmt_num(gc_aw.intensity)}"
                )


def compute_group_cost(
    cm: Any,
    graph: MiniGraph,
    group_ops: List[str],
    *,
    batch: int,
    seq_len: int,
    phase: str,
    include_weight: bool,
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

        # ------------------------------------------------------------
        # KV-cache read traffic (decode)
        # ------------------------------------------------------------
        kv_cache_read_bytes = 0
        if str(phase).lower() == "decode":
            opu = str(getattr(node, "name", op) or op).strip().upper()
            if opu in ("QK", "SV"):
                try:
                    kv_cache_read_bytes = int(cm.estimate_kv_cache_read_bytes(node, int(batch), int(seq_len), str(phase)))
                except Exception:
                    kv_cache_read_bytes = 0
        act_bytes += int(kv_cache_read_bytes)

        b = act_bytes
        if bool(include_weight):
            b += int(getattr(node, "weight_size", 0) or 0)

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
    batches_sorted: List[int],
    batch_color_map: Dict[int, str],
    marker_area: float = 220.0,
    point_alpha: float = 0.9,
) -> Tuple[List[Line2D], List[Line2D], List[Line2D], List[Line2D]]:
    """Draw a single phase into an existing axis.

    Returns: (device_handles, strategy_handles, batch_handles, operator_handles)
    """

    x_min, x_max = x_limits
    y_min, y_max = y_limits

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", linestyle=":", alpha=0.35)

    # All outer frames (axis spines) in black
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(colors="black")
    # Do not force a square subplot; keep a landscape layout.

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
                label=str(dev.label),
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

                # Small multiplicative jitter on X to avoid perfect overplotting across batches
                # (especially for PIM where intensity is activation-only and can be batch-invariant).
                def _jitter_intensity(x: float, B: int) -> float:
                    try:
                        if not batches_sorted or len(batches_sorted) <= 1:
                            return float(x)
                        if not (x > 0 and math.isfinite(x)):
                            return float(x)
                        b = int(B)
                        if b not in batches_sorted:
                            return float(x)
                        r = batches_sorted.index(b)
                        center = (len(batches_sorted) - 1) / 2.0
                        off = (float(r) - float(center)) * 0.03
                        x2 = float(x) * float(1.0 + off)
                        return float(x2) if x2 > 0 else float(x)
                    except Exception:
                        return float(x)

                xs_: List[float] = []
                ys_: List[float] = []
                filtered: List[Tuple[float, float, int, int]] = []
                batch_xy: Dict[int, Tuple[List[float], List[float]]] = {}

                for (I, bound, B, S) in pts:
                    if not (I > 0 and math.isfinite(I) and bound > 0 and math.isfinite(bound)):
                        continue
                    x_plot = _jitter_intensity(float(I), int(B))
                    xs_.append(x_plot)
                    ys_.append(float(bound))
                    filtered.append((x_plot, float(bound), int(B), int(S)))

                    bb = int(B)
                    if bb not in batch_xy:
                        batch_xy[bb] = ([], [])
                    batch_xy[bb][0].append(x_plot)
                    batch_xy[bb][1].append(float(bound))

                if not xs_:
                    continue

                ax.scatter(
                    xs_,
                    ys_,
                    s=float(marker_area),
                    marker=mk,
                    facecolors=col,
                    edgecolors=MARKER_EDGE_COLOR,
                    linewidths=0.9,
                    alpha=float(point_alpha),
                    zorder=3,
                )

                # Inner dot indicates batch (two batches expected)
                dot_area = float(marker_area) * 0.18
                for bb, (xx, yy) in batch_xy.items():
                    dot_c = str(batch_color_map.get(int(bb), "black"))
                    ax.scatter(
                        xx,
                        yy,
                        s=dot_area,
                        marker="o",
                        facecolors=dot_c,
                        edgecolors=MARKER_EDGE_COLOR,
                        linewidths=0.6,
                        alpha=float(point_alpha),
                        zorder=4,
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
            markeredgecolor=MARKER_EDGE_COLOR,
            label=s,
        )
        for s in strategy_names
    ]

    batch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=str(batch_color_map.get(int(b), "black")),
            markeredgecolor=MARKER_EDGE_COLOR,
            label=f"B={int(b)}",
        )
        for b in (batches_sorted or [])
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

    return dev_handles, strat_handles, batch_handles, op_handles


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
    batches_sorted: List[int],
    batch_color_map: Dict[int, str],
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
    fig_w = 7.0 * ncols
    fig_h = 4.0
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]

    # Draw each phase
    dev_handles: List[Line2D] = []
    strat_handles: List[Line2D] = []
    batch_handles: List[Line2D] = []
    op_handles: List[Line2D] = []
    for ax, phase in zip(axes, phases):
        pts = points_by_phase.get(phase, {})
        dev_h, strat_h, batch_h, op_h = draw_phase(
            ax=ax,
            phase=phase,
            devices=devices,
            strategy_names=strategy_names,
            strategy_colors=strategy_colors,
            points=pts,
            groups_to_plot=groups_to_plot,
            x_limits=x_limits,
            y_limits=y_limits,
            batches_sorted=list(batches_sorted or []),
            batch_color_map=dict(batch_color_map or {}),
            marker_area=float(marker_area),
        )
        # Keep handles from the first axis (same mapping across all)
        if not dev_handles:
            dev_handles = dev_h
        if not strat_handles:
            strat_handles = strat_h
        if not batch_handles:
            batch_handles = batch_h
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

        leg_batch = ax.legend(handles=batch_handles, loc="lower left", fontsize=8, frameon=False)
        ax.add_artist(leg_batch)

        ax.legend(handles=op_handles, loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_single_phase(
    *,
    phase: str,
    devices: List[DeviceInfo],
    strategy_names: List[str],
    strategy_colors: Dict[str, str],
    points: Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int, int]]]]],
    groups_to_plot: List[str],
    out_path: Path,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    batches_sorted: List[int],
    batch_color_map: Dict[int, str],
    marker_area: float = 220.0,
) -> None:
    """Plot exactly one phase into a single figure.

    Requirement: prefill and decode separated (two separate images).
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
    dev_handles, strat_handles, batch_handles, op_handles = draw_phase(
        ax=ax,
        phase=str(phase).strip().lower(),
        devices=devices,
        strategy_names=strategy_names,
        strategy_colors=strategy_colors,
        points=points,
        groups_to_plot=groups_to_plot,
        x_limits=x_limits,
        y_limits=y_limits,
        batches_sorted=list(batches_sorted or []),
        batch_color_map=dict(batch_color_map or {}),
        marker_area=float(marker_area),
    )

    ax.set_xlabel("Arithmetic intensity (FLOPs / Byte)")
    ax.set_ylabel("Roofline upper bound (TFLOPs/s)")

    leg_dev = ax.legend(handles=dev_handles, loc="upper left", fontsize=8, frameon=False)
    ax.add_artist(leg_dev)
    leg_strat = ax.legend(handles=strat_handles, loc="upper right", fontsize=8, frameon=False)
    ax.add_artist(leg_strat)
    leg_batch = ax.legend(handles=batch_handles, loc="lower left", fontsize=8, frameon=False)
    ax.add_artist(leg_batch)
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
    parser.add_argument("--model-shape", type=str, default="./experiment/roofline_configs/llama_7b_shape.json")
    parser.add_argument("--hardware-json", type=str, default="./algorithms/examples/represent_hardware.json")
    # Strategies:a single JSON with {"strategies":[{"name":..., "config":{...}}, ...]}  (recommended)
    parser.add_argument("--strategies-json", type=str, default="", help="JSON with a list of strategies (name+config).")
    # Strategy colors (optimizations):
    # Priority: --strategy-color-map > per-strategy "color" in strategies-json > --strategy-colors > fallback PALETTE
    parser.add_argument(
        "--strategy-color-map",
        type=str,
        default="",
        help='JSON mapping {"strategy_name": "#RRGGBB", ...}. Overrides other sources.',
    )
    parser.add_argument(
        "--strategy-colors",
        nargs="*",
        default=[],
        help="List of colors (e.g., #RRGGBB) assigned by strategy order.",
    )

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

    # Bytes accounting:
    #   - auto (default): PIM => activation only; CPU/GPU/NPU => activation + weight
    #   - activation: force activation-only for CPU/GPU/NPU as well (PIM is still activation-only)
    #   - activation+weight: same as auto for non-PIM (kept for backward compatibility)
    parser.add_argument(
        "--bytes-mode",
        type=str,
        default="auto",
        choices=["auto", "activation", "activation+weight"],
        help="Bytes accounting mode. Default 'auto': PIM uses activation-only; non-PIM uses activation+weight.",
    )

    parser.add_argument("--batches", nargs="*", type=int, default=[1, 32])

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

    # Plot layout:
    #   - default: one figure per phase (prefill and decode separated)
    #   - --combine-phases: legacy layout (prefill|decode subplots in one figure)
    parser.add_argument(
        "--combine-phases",
        action="store_true",
        help="(legacy) Put all phases into one multi-panel figure instead of separate figures.",
    )

    parser.add_argument(
        "--only-strategies",
        nargs="*",
        default=[],
        help="Optional: filter strategy names (works only with --strategies-json).",
    )

    parser.add_argument("--alpha-min", type=float, default=0.25, help="Minimum marker opacity.")
    parser.add_argument("--alpha-max", type=float, default=1.0, help="Maximum marker opacity.")
    parser.add_argument("--marker-area", type=float, default=220.0, help="Marker area for scatter points (points^2).")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-op FLOPs/bytes breakdown for a representative (phase,B,S). Useful to verify KV-cache read bytes and quantization/sparsity tags.",
    )

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
    devices = load_devices(
        Path(args.hardware_json),
        args.devices
    )
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
    strategy_color_hints: Dict[str, str] = {}

    if args.strategies_json:
        root = load_json(Path(args.strategies_json))

        # Support multiple formats (list, {"strategies":[...]}, name->config mapping)
        items = parse_strategies_from_json_obj(root)

        # Allow passing a *single* optimization config (opt_baseline.json style)
        # via --strategies-json (use the file stem as name).
        if not items and isinstance(root, dict):
            items = [{"name": Path(args.strategies_json).stem, "config": root}]

        if not items:
            raise ValueError(
                "strategies-json format unsupported. Expected one of: "
                "{strategies:[{name,config}]}, a list of {name,config}, or a mapping {name: config}."
            )

        for item in items:
            name = str(item.get("name") or "").strip()
            cfg = item.get("config", {})
            if not name or not isinstance(cfg, dict):
                continue

            # Optional color hint per strategy
            c_hint = str(item.get("color") or "").strip()
            if c_hint and _is_valid_color(c_hint):
                strategy_color_hints[name] = c_hint

            if args.only_strategies and name not in set(args.only_strategies):
                continue

            g2 = copy.deepcopy(base_graph)
            apply_optimizations_to_graph(g2, cfg, base_weight_dtype_bytes=int(args.base_weight_dtype_bytes), shape=shape)
            strategy_graphs.append((name, g2))

    if not strategy_graphs:
        # fallback: a no-op strategy
        strategy_graphs = [("no_opt", base_graph)]

    # assign colors (by strategy order / user mapping)
    strategy_names = [name for name, _ in strategy_graphs]
    strategy_colors = resolve_strategy_colors(
        strategy_names,
        color_map_file=str(args.strategy_color_map),
        colors_list=list(args.strategy_colors or []),
        color_hints=strategy_color_hints,
        palette=PALETTE,
    )

    # CostModel
    cm = CostModel(cluster=DummyCluster(), dtype=str(args.dtype), pim_fast_mode=True)

    # Optional: print a representative FLOPs/bytes breakdown for validation.
    if bool(getattr(args, "debug", False)):
        debug_dump_strategy_costs(
            cm,
            strategy_graphs,
            phases=list(phases),
            seqlens_per_phase=dict(seqlens_per_phase),
            batches=[int(x) for x in (args.batches or [1])],
            groups_to_plot=list(groups_to_plot),
        )

    outdir = Path(args.outdir)

    # Batch -> dot color (inner marker)
    batches_sorted = sorted({int(b) for b in (args.batches or []) if int(b) > 0}) or [1]
    batch_color_map: Dict[int, str] = {
        int(b): str(BATCH_DOT_COLORS[i % len(BATCH_DOT_COLORS)]) for i, b in enumerate(batches_sorted)
    }

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
                for B in batches_sorted:
                    for gkey in groups_to_plot:
                        group_ops = OP_GROUPS[gkey]["ops"]

                        for dev in devices:
                            include_weight = device_requires_weight_loading(dev, str(args.bytes_mode))
                            cost = compute_group_cost(
                                cm,
                                g_strat,
                                group_ops,
                                batch=int(B),
                                seq_len=int(S),
                                phase=str(phase),
                                include_weight=bool(include_weight),
                            )
                            I = cost.intensity
                            if not (I > 0 and math.isfinite(I)):
                                continue

                            bound = dev.bound_tflops(I)
                            points[dev.name][sname][gkey].append((I, bound, int(B), int(S)))
                            xs_global.append(I)
                            ys_global.append(bound)

        points_by_phase[phase] = points

    # Unified axes across phases (tighter than the previous decade-snapped limits)
    x_limits = nice_log_limits(xs_global, min_floor=1e-8)
    y_limits = nice_log_limits(ys_global, min_floor=1e-6)
    if bool(args.combine_phases):
        out_path = outdir / (
            f"roofline_phases-{sanitize_filename('-'.join(phases))}_bytes-{sanitize_filename(args.bytes_mode)}.pdf"
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
            batches_sorted=list(batches_sorted or []),
            batch_color_map=dict(batch_color_map or {}),
            marker_area=float(args.marker_area),
        )
        print(f"[OK] {out_path}")
    else:
        for ph in phases:
            pts = points_by_phase.get(ph, {})
            out_path = outdir / (
                f"roofline_phase-{sanitize_filename(ph)}_bytes-{sanitize_filename(args.bytes_mode)}.pdf"
            )
            plot_single_phase(
                phase=ph,
                devices=devices,
                strategy_names=strategy_names,
                strategy_colors=strategy_colors,
                points=pts,
                groups_to_plot=groups_to_plot,
                out_path=out_path,
                x_limits=x_limits,
                y_limits=y_limits,
                batches_sorted=list(batches_sorted or []),
                batch_color_map=dict(batch_color_map or {}),
                marker_area=float(args.marker_area),
            )
            print(f"[OK] {out_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()

