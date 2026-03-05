#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export TRIFORM_ROOT=/lustre/home/2501111916/workspace/XPUPIM_0226_gpupim_parameter/TriForm
export PYTHONPATH=$TRIFORM_ROOT/algorithms:$TRIFORM_ROOT:$PYTHONPATH
export RAMULATOR2_BIN=../../algorithms/ramulator2

python eval_pim_single_op_fit.py \
  --shape-json /lustre/home/2501111916/workspace/XPUPIM_0226_gpupim_parameter/TriForm/configs/llama_7b_shape.json \
  --pim-config ../../algorithms/aim_simulator/gb.json \
  --ramulator-config ../../algorithms/aim_simulator/example.yaml \
  --pim-tflops 16.0 \
  --pim-mem-bw-gbs 16384.0 \
  --dtype fp16 \
  --phases prefill,decode \
  --seqlens 512,1024,2048,4096 \
  --batch 1,8,16,32 \
  --out pim_roofline_vs_trace.csv \
  --debug
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Repo/module location helpers (same pattern as eval_npu_single_op_fit.py)
# --------------------------------------------------------------------------------------

def _find_repo_module(module_rel_path: str) -> str:
    """Locate a file by walking upward from CWD."""
    here = os.path.abspath(os.getcwd())
    for _ in range(10):
        cand = os.path.join(here, module_rel_path)
        if os.path.isfile(cand):
            return cand
        here = os.path.dirname(here)
    raise FileNotFoundError(f"Could not find '{module_rel_path}' from CWD='{os.getcwd()}'")

def _find_repo_module_any(rel_paths: Sequence[str]) -> str:
    last: Optional[Exception] = None
    for rp in rel_paths:
        try:
            return _find_repo_module(str(rp))
        except Exception as e:
            last = e
    if last:
        raise last
    raise FileNotFoundError(f"Could not find any of: {list(rel_paths)}")

def _import_from_path(mod_name: str, path: str):
    """Import a module from an explicit .py path, after making sibling imports visible."""
    import importlib.util

    path = os.path.abspath(path)
    module_dir = os.path.dirname(path)           # e.g. .../TriForm/algorithms
    repo_root = os.path.dirname(module_dir)      # e.g. .../TriForm

    # Ensure cost_model.py can import sibling modules like task_graph.py, hardware.py, ...
    for p in (module_dir, repo_root):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

def _pretty(obj: Any) -> str:
    try:
        import json as _json
        return _json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)

# --------------------------------------------------------------------------------------
# Minimal device/cluster stubs (duck-typed for CostModel)
# --------------------------------------------------------------------------------------

@dataclass
class SimpleDev:
    name: str
    type: str = 'pim'
    tflops: float = 0.0
    mem_bw_GBs: float = 0.0
    # Optional PIM access-latency model (if you want to bypass bandwidth-only mem_time):
    pim_memory: Optional[Dict[str, Any]] = None
    pim_read_latency_ns: float = 0.0
    pim_write_latency_ns: float = 0.0

class SimpleCluster:
    def __init__(self, devices: Dict[str, Any]):
        self.devices = dict(devices)
        # Optional shared PIM memory config (CostModel._pim_parallel_access_bytes checks this too).
        self.pim_memory: Optional[Dict[str, Any]] = None

    def devices_by_type(self, typ: str) -> List[Any]:
        t = str(typ).lower()
        return [d for d in self.devices.values() if str(getattr(d, 'type', '')).lower() == t]

    # Communication is not needed for this script; keep minimal stubs if CostModel calls them.
    def get_link_spec(self, src: str, dst: str):
        class _Dummy:
            bw_GBs = 0.0
            flit_size_B = 0
            max_payload_B = 0
            latency_s = 0.0
            overhead_s = 0.0
        return _Dummy()

class SimpleNode:
    """Duck-typed TaskNode replacement for single-op evaluation."""
    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attrs = attrs or {}
        self.flops = 0.0

class _DummyLabel:
    """Duck-typed PlanLabel replacement (node_device_cost only uses kv_in_pim)."""
    kv_in_pim = False

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) == 0:
        return float('nan')
    yt = [float(x) for x in y_true]
    yp = [float(x) for x in y_pred]
    mean_y = sum(yt) / len(yt)
    ss_tot = sum((y - mean_y) ** 2 for y in yt)
    ss_res = sum((y - p) ** 2 for y, p in zip(yt, yp))
    if ss_tot <= 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')

def _metrics(y_true_us: Sequence[float], y_pred_us: Sequence[float]) -> Dict[str, float]:
    if not y_true_us:
        return {
            'n': 0,
            'mae_us': float('nan'),
            'mape_pct': float('nan'),
            'mean_err_us': float('nan'),
            'r2': float('nan'),
            'mean_ratio': float('nan'),
        }
    err = [float(p) - float(t) for t, p in zip(y_true_us, y_pred_us)]
    abs_err = [abs(e) for e in err]
    rel = [abs(e) / max(1e-9, float(t)) for t, e in zip(y_true_us, err)]
    ratio = [float(p) / max(1e-9, float(t)) for t, p in zip(y_true_us, y_pred_us)]
    return {
        'n': int(len(y_true_us)),
        'mae_us': float(_mean(abs_err)),
        'mape_pct': float(_mean(rel) * 100.0),
        'mean_err_us': float(_mean(err)),
        'r2': float(r2_score(y_true_us, y_pred_us)),
        'mean_ratio': float(_mean(ratio)),
    }

# --------------------------------------------------------------------------------------
# Shape parsing / model parameter inference
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelParams:
    dim: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: int
    head_dim: int
    q_dim: int
    kv_dim: int
    o_dim: int
    max_seq_len: int

def _iter_items(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _iter_items(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_items(v)

def _find_first_int_by_keys(obj: Any, keys: Sequence[str]) -> Optional[int]:
    keys_l = {str(k).lower() for k in keys}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys_l:
                try:
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, (int, float)) and float(v).is_integer():
                        return int(v)
                    if isinstance(v, str) and v.strip().isdigit():
                        return int(v.strip())
                except Exception:
                    pass
            found = _find_first_int_by_keys(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_first_int_by_keys(v, keys)
            if found is not None:
                return found
    return None

def _collect_ints_by_keys(obj: Any, keys: Sequence[str]) -> List[int]:
    keys_l = {str(k).lower() for k in keys}
    out: List[int] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys_l:
                if isinstance(v, list):
                    for x in v:
                        try:
                            if isinstance(x, bool):
                                continue
                            if isinstance(x, (int, float)) and float(x).is_integer():
                                out.append(int(x))
                            elif isinstance(x, str) and x.strip().isdigit():
                                out.append(int(x.strip()))
                        except Exception:
                            continue
                else:
                    try:
                        if isinstance(v, bool):
                            pass
                        elif isinstance(v, (int, float)) and float(v).is_integer():
                            out.append(int(v))
                        elif isinstance(v, str) and v.strip().isdigit():
                            out.append(int(v.strip()))
                    except Exception:
                        pass
            out.extend(_collect_ints_by_keys(v, keys))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_ints_by_keys(v, keys))
    return out

def infer_model_params(shape_obj: Any, *, overrides: Dict[str, Optional[int]]) -> ModelParams:
    # Common key aliases (robust against different JSON formats)
    dim = overrides.get('dim') or _find_first_int_by_keys(shape_obj, ['dim', 'hidden_size', 'model_dim', 'd_model']) or 4096
    n_heads = overrides.get('n_heads') or _find_first_int_by_keys(shape_obj, ['n_heads', 'num_heads', 'num_attention_heads', 'n_head']) or 32
    n_kv_heads = overrides.get('n_kv_heads') or _find_first_int_by_keys(shape_obj, ['n_kv_heads', 'num_key_value_heads', 'kv_heads', 'n_kv_head']) or n_heads
    ffn_dim = overrides.get('ffn_dim') or _find_first_int_by_keys(shape_obj, ['ffn_dim', 'intermediate_size', 'mlp_dim', 'hidden_dim']) or (11008 if dim == 4096 else 4 * dim)

    head_dim = overrides.get('head_dim') or _find_first_int_by_keys(shape_obj, ['head_dim', 'd_head']) or max(1, dim // max(1, n_heads))
    q_dim = overrides.get('q_dim') or _find_first_int_by_keys(shape_obj, ['q_dim']) or (n_heads * head_dim)
    kv_dim = overrides.get('kv_dim') or _find_first_int_by_keys(shape_obj, ['kv_dim']) or (n_kv_heads * head_dim)
    o_dim = overrides.get('o_dim') or _find_first_int_by_keys(shape_obj, ['o_dim']) or q_dim

    max_seq_len = overrides.get('max_seq_len') or _find_first_int_by_keys(shape_obj, ['max_seq_len', 'max_seq_length', 'context_length']) or 4096

    # Final sanity clamps
    dim = max(1, int(dim))
    n_heads = max(1, int(n_heads))
    n_kv_heads = max(1, int(n_kv_heads))
    ffn_dim = max(1, int(ffn_dim))
    head_dim = max(1, int(head_dim))
    q_dim = max(1, int(q_dim))
    kv_dim = max(1, int(kv_dim))
    o_dim = max(1, int(o_dim))
    max_seq_len = max(1, int(max_seq_len))

    return ModelParams(
        dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, ffn_dim=ffn_dim,
        head_dim=head_dim, q_dim=q_dim, kv_dim=kv_dim, o_dim=o_dim,
        max_seq_len=max_seq_len,
    )

def infer_seqlens(shape_obj: Any, *, user_seqlens: Optional[List[int]] = None) -> List[int]:
    if user_seqlens:
        xs = [int(x) for x in user_seqlens if int(x) > 0]
        return sorted(set(xs))
    # Try to find common lists/values in JSON
    candidates = []
    candidates.extend(_collect_ints_by_keys(shape_obj, ['seqlens', 'seq_lens', 'seq_len', 'seqlen', 'context_lengths', 'context_length']))
    # Filter plausible seqlen values
    xs = [int(x) for x in candidates if 0 < int(x) <= 131072]
    xs = sorted(set(xs))
    if xs:
        # Avoid extreme explosion: keep a compact representative set
        # (smallest 3 + a few mid + largest 3)
        if len(xs) <= 12:
            return xs
        pick = []
        pick.extend(xs[:3])
        mid = xs[len(xs)//2 - 2 : len(xs)//2 + 2]
        pick.extend(mid)
        pick.extend(xs[-3:])
        return sorted(set(pick))
    # Fallback defaults
    return [1, 16, 32, 128, 512, 2048]

# --------------------------------------------------------------------------------------
# Op construction + roofline breakdown
# --------------------------------------------------------------------------------------

SUPPORTED_OPS_DEFAULT = [
    'Q', 'K', 'V', 'O',
    'QK', 'SOFTMAX', 'SV',
    'LN', 'ROPE',
    'FFN_W1', 'FFN_W3', 'FFN_W2',
    'SWIGLU', 'GELU',
    'ADD',
]

def make_node(op_name: str, mp: ModelParams, *, seq_len: int, phase: str, causal: bool = True) -> SimpleNode:
    op = str(op_name).strip().upper()
    D = int(mp.dim)
    H = int(mp.n_heads)
    HKV = int(mp.n_kv_heads)
    hd = int(mp.head_dim)
    q_dim = int(mp.q_dim)
    kv_dim = int(mp.kv_dim)
    o_dim = int(mp.o_dim)
    Hf = int(mp.ffn_dim)

    # Common attrs: always populate heads so the PIM trace backend guard passes.
    attrs: Dict[str, Any] = {
        'dim': D,
        'ffn_dim': Hf,
        'q_heads': H,
        'kv_heads': HKV,
        'n_kv_heads': HKV,
        'head_dim': hd,
        'q_dim': q_dim,
        'kv_dim': kv_dim,
        'o_dim': o_dim,
        'causal': bool(causal),
    }

    # For decode, kv_len matters (CostModel uses it for QK/SOFTMAX/SV attention pairs).
    if str(phase).lower() == 'decode':
        attrs['kv_len'] = int(seq_len)
        attrs['past_kv_len'] = int(seq_len)

    # The CostModel uses node.name matching; keep these canonical names.
    if op in ('Q', 'K', 'V', 'O', 'LN', 'ROPE', 'QK', 'SOFTMAX', 'SV', 'FFN_W1', 'FFN_W2', 'FFN_W3', 'SWIGLU', 'GELU', 'ADD', 'IDENTITY'):
        return SimpleNode(op, attrs=attrs)

    # Allow direct pass-through of already supported names.
    return SimpleNode(op_name, attrs=attrs)

def pim_roofline_breakdown_us(
    cm_obj,
    dev: Any,
    node: SimpleNode,
    *,
    batch: int,
    seq_len: int,
    phase: str,
    cm_mod: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a breakdown of the FAST(PIM) model."""

    rd_B, wr_B = cm_obj.estimate_activation_bytes(node, int(batch), int(seq_len), str(phase))
    mem_s = float(cm_obj.pim_mem_time(int(rd_B), int(wr_B), dev))
    flops = float(cm_obj.estimate_flops(node, int(batch), int(seq_len), str(phase)))

    # Compute time (includes compute-utilization curve from config via effective_tflops())
    compute_s = float(cm_obj.flop_time(flops, dev))

    peak_tflops = float(getattr(dev, 'tflops', 0.0) or 0.0)
    try:
        util = float(getattr(cm_obj, '_compute_utilization')(float(flops), dev))
    except Exception:
        util = 1.0
    try:
        eff_tflops = float(getattr(cm_obj, 'effective_tflops')(float(flops), dev))
    except Exception:
        eff_tflops = float(peak_tflops)

    # Naive (no-util) compute time for comparison.
    if peak_tflops > 0.0 and math.isfinite(float(flops)):
        compute_s_naive = float(flops) / (float(peak_tflops) * 1e12)
    else:
        compute_s_naive = float('inf')

    # Kernel-launch overhead (config-driven)
    raw_key = str(getattr(node, 'name', '') or getattr(node, 'id', '') or '')
    try:
        op_key_ovh = str(getattr(cm_mod, '_normalize_npu_op_key')(raw_key)) if cm_mod is not None else raw_key.strip().lower()
    except Exception:
        op_key_ovh = raw_key.strip().lower()
    try:
        ovh_s = float(getattr(cm_obj, 'kernel_launch_overhead_s')(op_key_ovh, dev, phase=str(phase), time_scale=1.0))
    except Exception:
        ovh_s = 0.0

    roof_s = max(mem_s, compute_s)
    dominant = 'compute' if compute_s >= mem_s else 'memory'

    # "Constrained" total = roofline + launch overhead
    total_s = float(roof_s + max(0.0, float(ovh_s)))

    # "Naive" roofline for sanity: peak TFLOPS (no util curve) and no launch overhead.
    naive_roof_s = max(mem_s, float(compute_s_naive))

    return {
        'bytes_rd_B': int(rd_B),
        'bytes_wr_B': int(wr_B),
        'bytes_total_B': int(rd_B + wr_B),
        'mem_us': float(mem_s * 1e6),
        'flops': float(flops),
        'peak_tflops': float(peak_tflops),
        'utilization': float(util),
        'effective_tflops': float(eff_tflops),
        'compute_us': float(compute_s * 1e6),
        'compute_us_naive': float(compute_s_naive * 1e6),
        'roofline_us': float(roof_s * 1e6),
        'kernel_launch_overhead_us': float(max(0.0, float(ovh_s)) * 1e6),
        'total_us': float(total_s * 1e6),
        'naive_roofline_us': float(naive_roof_s * 1e6),
        'op_key_overhead': str(op_key_ovh),
        'dominant': dominant,
    }

def weight_bytes_for_op(op_name: str, mp: ModelParams, *, dtype_bytes: int) -> Optional[int]:
    """Return the weight matrix size in bytes for weight-bearing ops."""
    op = str(op_name).strip().upper()
    D = int(mp.dim)
    q_dim = int(mp.q_dim)
    kv_dim = int(mp.kv_dim)
    o_dim = int(mp.o_dim)
    Hf = int(mp.ffn_dim)
    b = int(dtype_bytes)

    # Weight shapes follow CostModel's GEMM semantics:
    #   Q:  [D, q_dim]
    #   K/V:[D, kv_dim]
    #   O:  [o_dim, D]
    #   W1/W3: [D, Hf]
    #   W2: [Hf, D]
    if op == 'Q':
        return int(D) * int(q_dim) * b
    if op in ('K', 'V'):
        return int(D) * int(kv_dim) * b
    if op == 'O':
        return int(o_dim) * int(D) * b
    if op in ('FFN_W1', 'FFN_W3'):
        return int(D) * int(Hf) * b
    if op == 'FFN_W2':
        return int(Hf) * int(D) * b
    return None

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    xs = []
    for tok in str(s).split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            xs.append(int(tok))
        else:
            # allow forms like "2048;4096"
            try:
                xs.append(int(float(tok)))
            except Exception:
                pass
    return xs or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape-json', default='/lustre/home/2501111916/workspace/XPUPIM_0226_gpupim_parameter/TriForm/configs/llama_7b_shape.json')
    ap.add_argument('--ops', default=None, help='Comma-separated ops to test (default: all common ops)')
    ap.add_argument('--phases', default='prefill,decode', help='Comma-separated phases: prefill,decode')
    ap.add_argument('--seqlens', default=None, help='Comma-separated seq lens; if omitted, try infer from shape-json')
    ap.add_argument('--batch', default='1', help='Comma-separated batch sizes (default 1)')
    ap.add_argument('--dtype', default='fp16', help='Activation/weight dtype for CostModel (fp16/bf16/fp32/int8/...)')

    ap.add_argument('--pim-tflops', type=float, required=True, help='PIM peak TFLOPS used by roofline (FAST)')
    ap.add_argument('--pim-mem-bw-gbs', type=float, required=True, help='PIM memory bandwidth in GB/s used by roofline (FAST)')
    ap.add_argument(
        '--device-name',
        default='PIM0',
        help=(
            "PIM device name. NOTE: this is used by CostModel to match config.COMPUTE_UTILIZATION / "
            "config.KERNEL_LAUNCH_OVERHEAD (by_device_name) via name-prefix heuristics."
        ),
    )

    ap.add_argument('--pim-config', type=str, required=True, help='PIM memory config JSON for CENT/AiM trace generator')
    ap.add_argument('--ramulator-config', type=str, required=True, help='Ramulator2 config file path')

    ap.add_argument('--gb-config', type=str, default=None, help='Global Buffer memory config JSON (required for --include-weight-load)')

    ap.add_argument('--include-weight-load', action='store_true', help='Also validate weight loading (GB->PIM) using trace backend')
    ap.add_argument('--max-weight-bytes', type=int, default=0, help='Skip weight-load simulations larger than this (0 = no limit)')

    ap.add_argument('--trace-scale-repeats', type=int, default=1, help='1=scale unit-trace by repeats (fast), 0=explicit unroll (slow)')
    ap.add_argument('--disable-trace-cache', action='store_true', help='Disable PIM latency cache (trace backend)')
    ap.add_argument('--keep-traces', action='store_true', help='Keep generated traces (debug)')
    ap.add_argument('--trace-dir', default=None, help='Where to store traces if --keep-traces is set')

    ap.add_argument('--out', default='pim_roofline_vs_trace.csv', help='Output CSV for op-level results')
    ap.add_argument('--out-weight', default='pim_weightload_roofline_vs_trace.csv', help='Output CSV for weight-load results')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--debug-print-n', type=int, default=8, help='Print detailed breakdown for first N op-cases')

    # Optional: explicit module paths (if your repo layout differs)
    ap.add_argument('--cost-model-py', default=None)
    ap.add_argument('--pim-backend-py', default=None)
    ap.add_argument('--config-py', default=None)

    # Optional overrides for model params (if shape-json does not contain them)
    ap.add_argument('--dim', type=int, default=None)
    ap.add_argument('--n-heads', type=int, default=None)
    ap.add_argument('--n-kv-heads', type=int, default=None)
    ap.add_argument('--ffn-dim', type=int, default=None)
    ap.add_argument('--head-dim', type=int, default=None)
    ap.add_argument('--q-dim', type=int, default=None)
    ap.add_argument('--kv-dim', type=int, default=None)
    ap.add_argument('--o-dim', type=int, default=None)
    ap.add_argument('--max-seq-len', type=int, default=None)

    args = ap.parse_args()

    # IMPORTANT: PIM trace backend reads this env var at import time.
    os.environ['PIM_TRACE_SCALE_REPEATS'] = '1' if int(args.trace_scale_repeats) != 0 else '0'

    # Locate modules
    cost_model_path = args.cost_model_py or _find_repo_module_any([os.path.join('algorithms', 'cost_model.py')])
    pim_backend_path = args.pim_backend_py or _find_repo_module_any([os.path.join('algorithms', 'cost_model_pim_backend.py')])

    # Import config first (CostModel depends on it). Prefer the config next to cost_model.py.
    if args.config_py:
        cfg_path = os.path.abspath(args.config_py)
    else:
        cand_cfg = os.path.join(os.path.dirname(cost_model_path), 'config.py')
        if os.path.isfile(cand_cfg):
            cfg_path = cand_cfg
        else:
            cfg_path = _find_repo_module_any([os.path.join('algorithms', 'config.py')])

    cfg_mod = _import_from_path('config', cfg_path)

    # Enable debug logging early
    if bool(args.debug):
        try:
            log_dir = os.path.join(os.path.abspath(os.path.dirname(args.out) or '.'), 'debug_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'pim_roofline_vs_trace_debug.txt')
            if hasattr(cfg_mod, 'setup_logging'):
                cfg_mod.setup_logging(True, log_file=str(log_file))
            print('[DEBUG] Logging enabled:', log_file)
        except Exception as e:
            print('[WARN] config.setup_logging failed:', e)

    cm_mod = _import_from_path('cost_model', cost_model_path)
    pim_mod = _import_from_path('cost_model_pim_backend', pim_backend_path)

    # Load shape json (best-effort; script can run without it)
    shape_path = Path(str(args.shape_json))
    shape_obj: Any = {}
    if shape_path.is_file():
        try:
            shape_obj = json.loads(shape_path.read_text(encoding='utf-8'))
        except Exception as e:
            print('[WARN] Failed to parse shape-json:', e)
            shape_obj = {}
    else:
        print('[WARN] shape-json not found:', shape_path)

    # Parse model params
    overrides = {
        'dim': args.dim,
        'n_heads': args.n_heads,
        'n_kv_heads': args.n_kv_heads,
        'ffn_dim': args.ffn_dim,
        'head_dim': args.head_dim,
        'q_dim': args.q_dim,
        'kv_dim': args.kv_dim,
        'o_dim': args.o_dim,
        'max_seq_len': args.max_seq_len,
    }
    mp = infer_model_params(shape_obj, overrides=overrides)

    # Seqlens and batches
    user_seqlens = _parse_int_list(args.seqlens)
    seqlens = infer_seqlens(shape_obj, user_seqlens=user_seqlens)
    batches = _parse_int_list(args.batch) or [1]
    phases = [p.strip().lower() for p in str(args.phases).split(',') if p.strip()]
    if not phases:
        phases = ['prefill', 'decode']

    # Ops
    ops = [o.strip().upper() for o in (str(args.ops).split(',') if args.ops else SUPPORTED_OPS_DEFAULT)]
    ops = [o for o in ops if o]

    # Device + cluster
    pim_dev = SimpleDev(
        name=str(args.device_name),
        type='pim',
        tflops=float(args.pim_tflops),
        mem_bw_GBs=float(args.pim_mem_bw_gbs),
    )
    cluster = SimpleCluster({'PIM0': pim_dev})

    # Build CostModel for FAST predictions (roofline)
    cm_fast = cm_mod.CostModel(cluster=cluster, dtype=str(args.dtype), pim_fast_mode=True, npu_backend='fast')

    # Build model_dict for TRACE backend (CENT needs it). Must have cache sized >= max seqlen.
    max_seq = int(max(seqlens) if seqlens else mp.max_seq_len)
    try:
        # Prefer the helper shipped in cost_model_pim_backend.
        model_dict = pim_mod._make_shared_model_dict(int(mp.dim), int(mp.n_heads), int(mp.n_kv_heads), int(mp.ffn_dim), int(max_seq))
    except Exception as e:
        print('[ERROR] Failed to build model_dict (torch missing or helper changed):', e)
        print('Hint: ensure torch is available and CENT submodule is correctly installed.')
        raise

    # Normalize + validate paths
    pim_cfg = Path(str(args.pim_config)).expanduser()
    ram_cfg = Path(str(args.ramulator_config)).expanduser()
    if not pim_cfg.exists():
        raise FileNotFoundError(f'PIM config not found: {pim_cfg}')
    if not ram_cfg.exists():
        raise FileNotFoundError(f'Ramulator config not found: {ram_cfg}')

    # NOTE: We call trace backend directly, but we still create a CostModel object in TRACE mode
    # so you can reuse its internal dtype/settings if you want to extend later.
    cm_trace = cm_mod.CostModel(
        cluster=cluster,
        dtype=str(args.dtype),
        pim_fast_mode=False,
        pim_config_path=pim_cfg,
        ramulator_config_path=ram_cfg,
        gb_config_path=Path(str(args.gb_config)).expanduser() if args.gb_config else None,
        model_dict=model_dict,
    )
    cm_trace.set_model_dict(model_dict)

    # Cache policy
    use_cache = not bool(args.disable_trace_cache)

    # Debug prints
    if bool(args.debug):
        print('\n=== [DEBUG] Resolved modules ===')
        print('config.py                 :', getattr(cfg_mod, '__file__', None))
        print('cost_model.py             :', getattr(cm_mod, '__file__', None))
        print('cost_model_pim_backend.py :', getattr(pim_mod, '__file__', None))
        print('cost_model._config         :', getattr(getattr(cm_mod, '_config', None), '__file__', None))
        print('\n=== [DEBUG] Model params ===')
        print(_pretty(mp.__dict__))
        print('\n=== [DEBUG] Cases ===')
        print('phases :', phases)
        print('batches:', batches)
        print('seqlens:', seqlens)
        print('ops    :', ops)

        # ---- Verify FAST(PIM) is bound to config constraints ----
        try:
            print('\n=== [DEBUG] FAST(PIM) config binding ===')
            print('PIM device name:', getattr(pim_dev, 'name', None))
            if hasattr(cm_mod, '_device_family_key_from_name'):
                try:
                    print('device_family_key:', cm_mod._device_family_key_from_name(str(getattr(pim_dev, 'name', '') or '')))
                except Exception:
                    pass

            util_sel = None
            try:
                util_sel = cm_mod._lookup_cfg_by_device_name(getattr(cfg_mod, 'COMPUTE_UTILIZATION', {}), pim_dev)
            except Exception as e:
                util_sel = f'error: {e}'
            print('COMPUTE_UTILIZATION selected:\n', _pretty(util_sel))

            kl_sel = None
            try:
                kl_sel = cm_fast._kernel_launch_cfg(pim_dev)
            except Exception as e:
                kl_sel = f'error: {e}'
            print('KERNEL_LAUNCH_OVERHEAD selected:\n', _pretty(kl_sel))

            # One concrete example to prove the numbers are actually used.
            ex_phase = str(phases[0] if phases else 'prefill')
            ex_seq = int(seqlens[0] if seqlens else 1)
            ex_batch = int(batches[0] if batches else 1)
            ex_node = make_node('Q', mp, seq_len=ex_seq, phase=ex_phase)
            ex_flops = float(cm_fast.estimate_flops(ex_node, ex_batch, ex_seq, ex_phase))
            ex_util = float(cm_fast._compute_utilization(ex_flops, pim_dev))
            ex_op_key = str(cm_mod._normalize_npu_op_key(getattr(ex_node, 'name', 'q')))
            ex_ovh_us = float(cm_fast.kernel_launch_overhead_s(ex_op_key, pim_dev, phase=ex_phase, time_scale=1.0) * 1e6)
            print(
                f"Example[Q]: phase={ex_phase} B={ex_batch} S={ex_seq} "
                f"flops={ex_flops:.3e} util={ex_util:.4f} "
                f"launch_overhead_us={ex_ovh_us:.4f} (op_key={ex_op_key})"
            )
        except Exception as e:
            print('[WARN] FAST(PIM) config binding print failed:', e)

    # -------------------------
    # Op-level comparison
    # -------------------------
    op_rows: List[Dict[str, Any]] = []
    dbg_left = int(args.debug_print_n)

    for phase in phases:
        for seq_len in seqlens:
            for batch in batches:
                for op in ops:
                    node = make_node(op, mp, seq_len=int(seq_len), phase=str(phase))
                    # FAST prediction
                    fast_s = float(cm_fast.node_device_cost(node, pim_dev, _DummyLabel(), batch=int(batch), seq_len=int(seq_len), phase=str(phase)))
                    bd = pim_roofline_breakdown_us(
                        cm_fast,
                        pim_dev,
                        node,
                        batch=int(batch),
                        seq_len=int(seq_len),
                        phase=str(phase),
                        cm_mod=cm_mod,
                    )

                    # TRACE prediction (if op is traceable, otherwise mark as skipped)
                    op_norm = str(pim_mod._normalize_pim_op(op.lower() if isinstance(op, str) else str(op))).strip().lower()
                    traceable = bool(op_norm) and (op_norm in getattr(pim_mod, 'PIM_TRACE_SUPPORTED_OPS', set()))

                    trace_s = float('nan')
                    trace_err = None
                    if traceable:
                        try:
                            trace_s = float(
                                pim_mod._get_pim_latency_via_trace(
                                    op=str(op_norm),
                                    pim_config=pim_cfg,
                                    ramulator_config=ram_cfg,
                                    dim=int(mp.dim),
                                    n_heads=int(mp.n_heads),
                                    n_kv_heads=int(mp.n_kv_heads),
                                    ffn_dim=int(mp.ffn_dim),
                                    seqlen=int(seq_len),
                                    batch=int(batch),
                                    head_dim=int(mp.head_dim),
                                    q_dim=int(mp.q_dim),
                                    kv_dim=int(mp.kv_dim),
                                    o_dim=int(mp.o_dim),
                                    phase=str(phase),
                                    model_dict=model_dict,
                                    use_cache=bool(use_cache),
                                    keep_traces=bool(args.keep_traces),
                                    trace_dir=Path(str(args.trace_dir)).expanduser() if args.trace_dir else None,
                                    trace_prefix=f'{op}_{phase}_B{batch}_S{seq_len}',
                                )
                            )
                        except Exception as e:
                            trace_err = str(e)
                            trace_s = float('nan')

                    fast_us = float(fast_s * 1e6) if math.isfinite(fast_s) else float('nan')
                    trace_us = float(trace_s * 1e6) if math.isfinite(trace_s) else float('nan')

                    abs_err_us = float('nan')
                    rel_err_pct = float('nan')
                    ratio = float('nan')
                    if math.isfinite(fast_us) and math.isfinite(trace_us):
                        abs_err_us = abs(fast_us - trace_us)
                        rel_err_pct = abs_err_us / max(1e-9, trace_us) * 100.0
                        ratio = fast_us / max(1e-9, trace_us)

                    fast_total_bd_us = float(bd.get('total_us', float('nan')))
                    fast_breakdown_delta_us = float('nan')
                    if math.isfinite(fast_us) and math.isfinite(fast_total_bd_us):
                        fast_breakdown_delta_us = float(fast_us - fast_total_bd_us)

                    row = {
                        'op': str(op),
                        'op_norm': str(op_norm),
                        'traceable': bool(traceable),
                        'phase': str(phase),
                        'batch': int(batch),
                        'seq_len': int(seq_len),
                        'dim': int(mp.dim),
                        'n_heads': int(mp.n_heads),
                        'n_kv_heads': int(mp.n_kv_heads),
                        'ffn_dim': int(mp.ffn_dim),
                        'fast_us': fast_us,
                        'fast_total_us_from_breakdown': fast_total_bd_us,
                        'fast_breakdown_delta_us': fast_breakdown_delta_us,
                        'trace_us': trace_us,
                        'abs_err_us': abs_err_us,
                        'rel_err_pct': rel_err_pct,
                        'fast_overall_dominant': bd.get('dominant'),
                        'fast_mem_us': bd.get('mem_us'),
                        'fast_compute_us': bd.get('compute_us'),
                        'fast_compute_us_naive': bd.get('compute_us_naive'),
                        'fast_roofline_us': bd.get('roofline_us'),
                        'fast_naive_roofline_us': bd.get('naive_roofline_us'),
                        'fast_kernel_launch_overhead_us': bd.get('kernel_launch_overhead_us'),
                        'fast_utilization': bd.get('utilization'),
                        'fast_peak_tflops': bd.get('peak_tflops'),
                        'fast_effective_tflops': bd.get('effective_tflops'),
                        'fast_op_key_overhead': bd.get('op_key_overhead'),
                        'fast_bytes_total_B': bd.get('bytes_total_B'),
                        'trace_error': trace_err,
                    }
                    op_rows.append(row)

                    # Debug print (first N successful trace cases)
                    if bool(args.debug) and dbg_left > 0 and traceable and math.isfinite(trace_us):
                        print('\n--- [DEBUG][op-case] ---')
                        print(f'op={op} (norm={op_norm})  phase={phase}  B={batch}  S={seq_len}')
                        print(f'FAST(us)={fast_us:.4f}   TRACE(us)={trace_us:.4f}   ratio(F/T)={ratio:.4f}')
                        print(
                            "FAST breakdown: "
                            f"dominant={bd.get('dominant')}  "
                            f"util={float(bd.get('utilization', 1.0)):.4f}  "
                            f"peak_tflops={float(bd.get('peak_tflops', 0.0)):.3f}  "
                            f"eff_tflops={float(bd.get('effective_tflops', 0.0)):.3f}  "
                            f"compute_us={float(bd.get('compute_us', float('nan'))):.4f}  "
                            f"compute_naive_us={float(bd.get('compute_us_naive', float('nan'))):.4f}  "
                            f"mem_us={float(bd.get('mem_us', float('nan'))):.4f}  "
                            f"roofline_us={float(bd.get('roofline_us', float('nan'))):.4f}  "
                            f"launch_ovh_us={float(bd.get('kernel_launch_overhead_us', 0.0)):.4f}  "
                            f"total_us={float(bd.get('total_us', float('nan'))):.4f}  "
                            f"bytes={bd.get('bytes_total_B')}  "
                            f"op_key_ovh={bd.get('op_key_overhead')}"
                        )
                        dbg_left -= 1

    # Summary per op
    summary_rows: List[Dict[str, Any]] = []
    for op in ops:
        xs = [r for r in op_rows if r['op'] == op and bool(r.get('traceable')) and math.isfinite(float(r.get('trace_us', float('nan'))))]
        y_true = [float(r['trace_us']) for r in xs]
        y_pred = [float(r['fast_us']) for r in xs]
        met = _metrics(y_true, y_pred)
        summary_rows.append({'op': op, **met})

    # Save CSVs
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(op_rows)
        df.to_csv(args.out, index=False)

        df_sum = pd.DataFrame(summary_rows).sort_values(['op']).reset_index(drop=True)

        print('\n=== PIM FAST(roofline) vs TRACE(sim) summary (per op) ===')
        print(df_sum.to_string(index=False))
        print('\nSaved per-case CSV :', os.path.abspath(args.out))

        # Also show an overall aggregate
        xs_all = [r for r in op_rows if bool(r.get('traceable')) and math.isfinite(float(r.get('trace_us', float('nan'))))]
        y_true_all = [float(r['trace_us']) for r in xs_all]
        y_pred_all = [float(r['fast_us']) for r in xs_all]
        met_all = _metrics(y_true_all, y_pred_all)
        print('\n=== Overall ===')
        print(_pretty(met_all))

    except Exception as e:
        print('[WARN] pandas not available or failed:', e)
        # Fallback CSV writer
        import csv
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(op_rows[0].keys()) if op_rows else [])
            if op_rows:
                w.writeheader()
                for r in op_rows:
                    w.writerow(r)
        print('Saved per-case CSV :', os.path.abspath(args.out))

    # -------------------------
    # Optional: weight load comparison
    # -------------------------
    if bool(args.include_weight_load):
        if not args.gb_config:
            raise ValueError('--include-weight-load requires --gb-config')
        gb_cfg = Path(str(args.gb_config)).expanduser()
        if not gb_cfg.exists():
            raise FileNotFoundError(f'GB config not found: {gb_cfg}')

        dtype_bytes = int(cm_mod.DTYPE_BYTES.get(str(args.dtype).lower(), 2))
        weight_rows: List[Dict[str, Any]] = []

        for op in ops:
            wbytes = weight_bytes_for_op(op, mp, dtype_bytes=dtype_bytes)
            if wbytes is None:
                continue
            if int(args.max_weight_bytes or 0) > 0 and int(wbytes) > int(args.max_weight_bytes):
                weight_rows.append({
                    'op': op,
                    'weight_bytes': int(wbytes),
                    'fast_total_us': float(cm_fast.weight_load_time_pim(int(wbytes)) * 1e6),
                    'trace_total_us': float('nan'),
                    'trace_read_us': float('nan'),
                    'trace_write_us': float('nan'),
                    'skipped': True,
                    'skip_reason': f'weight_bytes>{int(args.max_weight_bytes)}',
                })
                continue

            # FAST model: current implementation approximates with PIM mem model only
            fast_us = float(cm_fast.weight_load_time_pim(int(wbytes)) * 1e6)

            # TRACE model: GB READ + PIM WRITE simulated via Ramulator2
            try:
                rd_s, wr_s = pim_mod._simulate_weight_loading_latency(
                    int(wbytes),
                    pim_config_path=pim_cfg,
                    gb_config_path=gb_cfg,
                    ramulator_config_path=ram_cfg,
                    dtype_bytes=int(dtype_bytes),
                    use_cache=bool(use_cache),
                    keep_traces=bool(args.keep_traces),
                    model_dict=model_dict,
                )
                trace_read_us = float(rd_s * 1e6)
                trace_write_us = float(wr_s * 1e6)
                trace_total_us = float((float(rd_s) + float(wr_s)) * 1e6)
                weight_rows.append({
                    'op': op,
                    'weight_bytes': int(wbytes),
                    'fast_total_us': fast_us,
                    'trace_total_us': trace_total_us,
                    'trace_read_us': trace_read_us,
                    'trace_write_us': trace_write_us,
                    'skipped': False,
                    'skip_reason': '',
                })
                if bool(args.debug):
                    print('\n--- [DEBUG][weight-load] ---')
                    print(f'op={op} weight_bytes={wbytes}')
                    print(f'FAST_total_us={fast_us:.3f}  TRACE_total_us={trace_total_us:.3f} (read={trace_read_us:.3f}, write={trace_write_us:.3f})')
            except Exception as e:
                weight_rows.append({
                    'op': op,
                    'weight_bytes': int(wbytes),
                    'fast_total_us': fast_us,
                    'trace_total_us': float('nan'),
                    'trace_read_us': float('nan'),
                    'trace_write_us': float('nan'),
                    'skipped': False,
                    'skip_reason': f'error: {e}',
                })
                if bool(args.debug):
                    print('[WARN] weight-load sim failed for', op, ':', e)

        # Save weight CSV
        try:
            import pandas as pd  # type: ignore
            dfw = pd.DataFrame(weight_rows)
            dfw.to_csv(args.out_weight, index=False)

            xs = [r for r in weight_rows if (not bool(r.get('skipped'))) and math.isfinite(float(r.get('trace_total_us', float('nan'))))]
            y_true = [float(r['trace_total_us']) for r in xs]
            y_pred = [float(r['fast_total_us']) for r in xs]
            print('\n=== Weight-load FAST vs TRACE summary ===')
            print(_pretty(_metrics(y_true, y_pred)))
            print('Saved weight CSV:', os.path.abspath(args.out_weight))
        except Exception:
            import csv
            with open(args.out_weight, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(weight_rows[0].keys()) if weight_rows else [])
                if weight_rows:
                    w.writeheader()
                    for r in weight_rows:
                        w.writerow(r)
            print('Saved weight CSV:', os.path.abspath(args.out_weight))

if __name__ == '__main__':
    main()
