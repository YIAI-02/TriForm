#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export TRIFORM_MMAD_LUT=/path/to/mmad_lookup_table.json
export TRIFORM_SOFTMAX_LUT=/path/to/softmax_lookup_table.json
export TRIFORM_GELU_LUT=/path/to/gelu_lookup_table.json
export TRIFORM_NORM_LUT=/path/to/layernorm_lookup_table.json

export TRIFORM_ROOT=/lustre/home/2501111916/workspace/XPUPIM_0226_gpupim_parameter/TriForm
export PYTHONPATH=$TRIFORM_ROOT/algorithms:$TRIFORM_ROOT:$PYTHONPATH

python eval_npu_single_op_fit.py \
  --npu-tflops 280.0 \
  --npu-mem-bw-gbs 819.2 \
  --device-name Ascend_910B_NPU0 \
  --dtype fp16 \
  --debug \
  --debug-print-n 10 \
  --debug-log-file ./output/debug_log.txt

"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

def _pretty(obj: Any) -> str:
    """Best-effort pretty formatter for debug prints."""
    try:
        import json
        return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)

# --------------------------------------------------------------------------------------
# Repo/module location helpers
# --------------------------------------------------------------------------------------

def _find_repo_module(module_rel_path: str) -> str:
    """Locate a file by walking upward from CWD."""
    here = os.path.abspath(os.getcwd())
    for _ in range(8):
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


# --------------------------------------------------------------------------------------
# Minimal device/cluster stubs
# --------------------------------------------------------------------------------------

@dataclass
class SimpleDev:
    name: str = 'Ascend_910B_NPU0'
    type: str = 'npu'
    tflops: float = 0.0
    mem_bw_GBs: float = 0.0


class SimpleCluster:
    def __init__(self, devices: Dict[str, Any]):
        self.devices = dict(devices)

    def devices_by_type(self, typ: str) -> List[Any]:
        t = str(typ).lower()
        return [d for d in self.devices.values() if str(getattr(d, 'type', '')).lower() == t]


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


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    if not y_true:
        return {
            'n': 0,
            'mae_us': float('nan'),
            'mape_pct': float('nan'),
            'mean_err_us': float('nan'),
            'r2': float('nan'),
        }
    err = [float(p) - float(t) for t, p in zip(y_true, y_pred)]
    abs_err = [abs(e) for e in err]
    rel = [abs(e) / max(1e-9, float(t)) for t, e in zip(y_true, err)]
    return {
        'n': int(len(y_true)),
        'mae_us': float(_mean(abs_err)),
        'mape_pct': float(_mean(rel) * 100.0),
        'mean_err_us': float(_mean(err)),
        'r2': float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------------------
def _make_single_op_node(op_kind: str, dims: Tuple[int, ...]):
    op = (op_kind or '').lower().strip()

    phase = 'prefill'
    batch = 1
    seq_len = 1

    if op == 'mmad':
        # dims = (M, N, K)
        M, N, K = map(int, dims)
        node = SimpleNode('Q', attrs={'dim': K, 'q_dim': N, 'q_heads': 1})
        phase = 'decode'
        batch = M
        seq_len = 1
        return node, batch, seq_len, phase

    if op == 'softmax':
        M, K = map(int, dims)
        node = SimpleNode('SOFTMAX', attrs={'q_heads': M, 'kv_len': K, 'causal': False})
        phase = 'decode'
        batch = 1
        seq_len = 1
        return node, batch, seq_len, phase

    if op == 'gelu':
        L = int(dims[0])
        node = SimpleNode('GELU', attrs={'ffn_dim': L})
        phase = 'decode'
        batch = 1
        seq_len = 1
        return node, batch, seq_len, phase

    if op in ('layernorm', 'norm'):
        rows, width = map(int, dims)
        node = SimpleNode('LN', attrs={'dim': width})
        phase = 'decode'
        batch = rows
        seq_len = 1
        return node, batch, seq_len, phase

    return None, batch, seq_len, phase

def _fast_cost_model_predict_us(cm_mod, cm_obj, dev: Any, *, op_kind: str, dims: Tuple[int, ...], dtype: str) -> Optional[float]:
    """CostModel fast-backend prediction using node_device_cost() (includes util + overhead)."""
    node, batch, seq_len, phase = _make_single_op_node(op_kind, dims)
    if node is None:
        return None

    cm_obj.dtype = dtype
    pred_s = float(cm_obj.node_device_cost(node, dev, _DummyLabel(), batch=batch, seq_len=seq_len, phase=phase))
    if not math.isfinite(pred_s):
        return None
    return float(pred_s * 1e6)


def _fast_roofline_breakdown_us(cm_mod, cm_obj, dev: Any, *, node: Any, batch: int, seq_len: int, phase: str) -> Dict[str, Any]:
    """Return a detailed breakdown for the FAST(roofline) path.

    This is meant purely for debug/verification:
      - FLOPs / util curve / effective TFLOPS
      - compute lower bound vs memory lower bound
      - kernel launch overhead (from config.KERNEL_LAUNCH_OVERHEAD)
    """
    attrs = getattr(node, 'attrs', {}) or {}

    # Mem lower bound
    rd_B, wr_B = cm_obj.estimate_activation_bytes(node, batch, seq_len, phase)
    mem_s = float(cm_obj.mem_time(int(rd_B + wr_B), dev))

    # Compute lower bound (with utilization correction)
    flops = float(cm_obj.estimate_flops(node, batch, seq_len, phase))
    util = float(getattr(cm_obj, '_compute_utilization')(flops, dev))
    eff_tflops = float(cm_obj.effective_tflops(flops, dev))
    compute_s = float(cm_obj.flop_time(flops, dev))

    # FAST backend uses max(compute, mem)
    roofline_s = max(compute_s, mem_s)
    dominant = 'compute' if compute_s >= mem_s else 'memory'

    # Time scale hint (normally 1.0)
    try:
        time_scale = float(getattr(cm_obj, '_time_scale_hint')(node, getattr(dev, 'type', '')))
    except Exception:
        time_scale = 1.0

    # Kernel launch overhead (config-gated)
    raw_key = str(getattr(node, 'name', '') or getattr(node, 'id', '') or '')
    op_key = cm_mod._normalize_npu_op_key(raw_key)
    overhead_s = 0.0
    overhead_scaled_s = 0.0
    try:
        overhead_s = float(cm_obj.kernel_launch_overhead_s(op_key, dev, phase=str(phase)))
        kl_cfg = getattr(cm_obj, '_kernel_launch_cfg')(dev)
        if bool(kl_cfg.get('scale_by_time_scale', False)):
            overhead_scaled_s = overhead_s * time_scale
        else:
            overhead_scaled_s = overhead_s
    except Exception:
        overhead_s = 0.0
        overhead_scaled_s = 0.0

    total_s = float(roofline_s) * float(time_scale) + float(overhead_scaled_s)

    return {
        'node_name': getattr(node, 'name', '?'),
        'op_key': str(op_key),
        'phase': str(phase),
        'batch': int(batch),
        'seq_len': int(seq_len),
        'attrs': dict(attrs),
        'bytes_rd_B': int(rd_B),
        'bytes_wr_B': int(wr_B),
        'bytes_total_B': int(rd_B + wr_B),
        'mem_us': float(mem_s * 1e6),
        'flops': float(flops),
        'util': float(util),
        'peak_tflops': float(getattr(dev, 'tflops', 0.0) or 0.0),
        'eff_tflops': float(eff_tflops),
        'compute_us': float(compute_s * 1e6),
        'roofline_us': float(roofline_s * 1e6),
        'dominant': dominant,
        'time_scale': float(time_scale),
        'overhead_us': float(overhead_s * 1e6),
        'overhead_scaled_us': float(overhead_scaled_s * 1e6),
        'total_us': float(total_s * 1e6),
    }


def _extract_samples_from_lut(lut: Dict[str, Any]) -> List[Tuple[Tuple[int, ...], float]]:
    """Return list of (dims, gt_us) using LUT's index as the ground truth."""
    idx = lut.get('index', None)
    samples: List[Tuple[Tuple[int, ...], float]] = []
    if isinstance(idx, dict) and idx:
        for k, v in idx.items():
            try:
                dims = tuple(int(x) for x in k)
                gt = float(v)
                samples.append((dims, gt))
            except Exception:
                continue
    else:
        # Fallback: use points (may include duplicates)
        pts = lut.get('points', []) or []
        for p in pts:
            try:
                dims = tuple(int(x) for x in p[:-1])
                gt = float(p[-1])
                samples.append((dims, gt))
            except Exception:
                continue
    # Stable ordering for reproducible prints
    samples.sort(key=lambda t: (len(t[0]),) + tuple(t[0]))
    return samples


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dtype', default='fp16', help='Activation dtype used in CostModel (fp16/fp32/bf16/int8...)')
    ap.add_argument('--npu-tflops', type=float, required=True, help='NPU peak TFLOPS used as peak throughput')
    ap.add_argument('--npu-mem-bw-gbs', type=float, required=True, help='NPU memory bandwidth in GB/s used for mem_time')
    ap.add_argument(
        '--device-name',
        default='Ascend_910B_NPU0',
        help='Device name string used for config.by_device_name matching (e.g., Ascend_910B_NPU0)',
    )
    ap.add_argument('--out', default='npu_single_op_fit_summary.csv', help='Output CSV path')
    ap.add_argument('--debug', action='store_true', help='Enable verbose debug prints + CostModel DEBUG logs')
    ap.add_argument('--debug-log-file', default='./eval_npu_single_op_fit_output/debug_log.txt', help='Debug log path used by config.setup_logging')
    ap.add_argument('--debug-print-n', type=int, default=8, help='Print detailed per-sample breakdown for first N samples per op')
    # Optional: explicit module paths (if your repo layout differs)
    ap.add_argument('--cost-model-py', default=None)
    ap.add_argument('--ascend-backend-py', default=None)
    ap.add_argument('--config-py', default=None)

    args = ap.parse_args()

    # Locate modules (support both "algorithms/" and "algorithm/" layouts)
    cost_model_path = args.cost_model_py or _find_repo_module_any([
        os.path.join('algorithms', 'cost_model.py')
    ])
    ascend_path = args.ascend_backend_py or _find_repo_module_any([
        os.path.join('algorithms', 'cost_model_npu_ascend_backend.py')
    ])

    # Import config first (CostModel depends on it). Prefer the config next to cost_model.py.
    if args.config_py:
        cfg_path = os.path.abspath(args.config_py)
    else:
        cand_cfg = os.path.join(os.path.dirname(cost_model_path), 'config.py')
        if os.path.isfile(cand_cfg):
            cfg_path = cand_cfg
        else:
            cfg_path = _find_repo_module_any([
                os.path.join('algorithms', 'config.py'),
            ])

    cfg_mod = _import_from_path('config', cfg_path)

    # Enable debug logging before importing CostModel (so its module-level loggers are configured).
    try:
        if bool(args.debug):
            # Ensure parent dirs exist (config.setup_logging uses FileHandler directly).
            log_dir = os.path.dirname(os.path.abspath(args.debug_log_file))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        if hasattr(cfg_mod, 'setup_logging'):
            cfg_mod.setup_logging(bool(args.debug), log_file=str(args.debug_log_file))
    except Exception as e:
        # Don't fail the evaluation just because logging setup failed.
        print('[WARN] config.setup_logging failed:', e)

    cm_mod = _import_from_path('cost_model', cost_model_path)
    ascend_mod = _import_from_path('cost_model_npu_ascend_backend', ascend_path)

    # Load LUTs
    lut_mmad = ascend_mod._load_mmad_lut()
    lut_softmax = ascend_mod._load_softmax_lut()
    lut_gelu = ascend_mod._load_gelu_lut()
    lut_norm = ascend_mod._load_layernorm_lut()

    missing = []
    if not lut_mmad: missing.append('MMAD')
    if not lut_softmax: missing.append('SOFTMAX')
    if not lut_gelu: missing.append('GELU')
    if not lut_norm: missing.append('NORM/LN')

    if missing:
        print('ERROR: Some LUTs could not be loaded:', ', '.join(missing))
        print('  - Ensure LUT files exist in cost_model_npu_ascend_backend search paths, or set env vars:')
        print('      TRIFORM_MMAD_LUT, TRIFORM_SOFTMAX_LUT, TRIFORM_GELU_LUT, TRIFORM_NORM_LUT')
        sys.exit(2)

    # Prepare datasets from LUT indices (treat as ground truth)
    datasets: Dict[str, Dict[str, Any]] = {
        'mmad': {
            'lut': lut_mmad,
            'samples': _extract_samples_from_lut(lut_mmad),
            'lut_predict': lambda dims: ascend_mod._lut_query(tag='MMAD', lut=lut_mmad, dims=dims),
        },
        'softmax': {
            'lut': lut_softmax,
            'samples': _extract_samples_from_lut(lut_softmax),
            'lut_predict': lambda dims: ascend_mod._lut_query(tag='SOFTMAX', lut=lut_softmax, dims=dims),
        },
        'gelu': {
            'lut': lut_gelu,
            'samples': _extract_samples_from_lut(lut_gelu),
            'lut_predict': lambda dims: ascend_mod._lut_query(tag='GELU', lut=lut_gelu, dims=dims),
        },
        'layernorm': {
            'lut': lut_norm,
            'samples': _extract_samples_from_lut(lut_norm),
            'lut_predict': lambda dims: ascend_mod._lut_query(tag='NORM', lut=lut_norm, dims=dims),
        },
    }

    # Build CostModel (duck-typed cluster + dev)
    dev = SimpleDev(
        name=str(args.device_name),
        type='npu',
        tflops=float(args.npu_tflops),
        mem_bw_GBs=float(args.npu_mem_bw_gbs),
    )
    cluster = SimpleCluster({'NPU0': dev})
    cm_obj = cm_mod.CostModel(cluster=cluster, dtype=str(args.dtype), pim_fast_mode=True, npu_backend='fast')
    cm_obj._npu_backend_impl_name = 'fast'  # for overhead gating

    # ------------------------------------------------------------------
    # Debug: verify config overrides are actually picked up
    # ------------------------------------------------------------------
    if bool(args.debug):
        print('\n=== [DEBUG] Module + config wiring ===')
        print('config.py path        :', getattr(cfg_mod, '__file__', None))
        # CostModel internally imports `config as _config`.
        try:
            print('cost_model sees config:', getattr(cm_mod._config, '__file__', None))
            print('config identity match :', bool(cm_mod._config is cfg_mod))
        except Exception:
            pass

        print('\n=== [DEBUG] Device used for matching ===')
        print('dev.name      :', getattr(dev, 'name', None))
        print('dev.type      :', getattr(dev, 'type', None))
        print('dev.tflops    :', getattr(dev, 'tflops', None))
        print('dev.mem_bw_GBs:', getattr(dev, 'mem_bw_GBs', None))
        try:
            fam = cm_mod._device_family_key_from_name(getattr(dev, 'name', ''))
            print('device family :', fam)
        except Exception:
            pass

        # Show per-device overrides found in config
        try:
            cu_cfg = getattr(cfg_mod, 'COMPUTE_UTILIZATION', None)
            kl_cfg = getattr(cfg_mod, 'KERNEL_LAUNCH_OVERHEAD', None)
            cu_hit = cm_mod._lookup_cfg_by_device_name(cu_cfg, dev)
            kl_hit = cm_mod._lookup_cfg_by_device_name(kl_cfg, dev)
            print('\n=== [DEBUG] COMPUTE_UTILIZATION per-device override ===')
            print(_pretty(cu_hit))
            print('\n=== [DEBUG] KERNEL_LAUNCH_OVERHEAD per-device override ===')
            print(_pretty(kl_hit))
        except Exception as e:
            print('[WARN] failed to resolve per-device overrides:', e)

        # Show merged kernel launch cfg used by CostModel
        try:
            merged_kl = cm_obj._kernel_launch_cfg(dev)
            print('\n=== [DEBUG] CostModel merged kernel launch cfg ===')
            print(_pretty(merged_kl))
        except Exception:
            pass

        # Probe utilization curve at a few FLOP points
        print('\n=== [DEBUG] Utilization probe (u = f(flops)) ===')
        probe_flops = [5e7, 1e8, 1e9, 1e10, 1.5e11, 5e11, 1e12, 5e12]
        for f in probe_flops:
            try:
                u = float(cm_obj._compute_utilization(float(f), dev))
                eff = float(cm_obj.effective_tflops(float(f), dev))
                print(f'flops={f:.3e}  util={u:.4f}  eff_tflops={eff:.3f}')
            except Exception as e:
                print(f'flops={f:.3e}  util=<err> ({e})')

    rows: List[Dict[str, Any]] = []

    # Per-op evaluation
    for op_kind, ds in datasets.items():
        samples: List[Tuple[Tuple[int, ...], float]] = ds['samples']
        lut_predict = ds['lut_predict']

        y_true: List[float] = []
        y_pred_lut: List[float] = []
        y_pred_fast: List[float] = []

        dbg_printed = 0
        for dims, gt in samples:
            pred_lut = lut_predict(tuple(int(x) for x in dims))
            if pred_lut is None or not math.isfinite(float(pred_lut)):
                continue

            pred_fast = _fast_cost_model_predict_us(cm_mod, cm_obj, dev, op_kind=op_kind, dims=dims, dtype=str(args.dtype))
            if pred_fast is None or not math.isfinite(float(pred_fast)):
                continue

            # Extra debug: print a detailed breakdown for the first N samples per op.
            if bool(args.debug) and dbg_printed < int(args.debug_print_n):
                node, batch, seq_len, phase = _make_single_op_node(op_kind, dims)
                if node is not None:
                    bd = _fast_roofline_breakdown_us(
                        cm_mod, cm_obj, dev,
                        node=node, batch=batch, seq_len=seq_len, phase=phase,
                    )
                    print('\n--- [DEBUG][sample] ---')
                    print(f'op={op_kind}  dims={tuple(int(x) for x in dims)}  phase={phase}')
                    print(f'gt_us={float(gt):.4f}  lut_us={float(pred_lut):.4f}  fast_us={float(pred_fast):.4f}')
                    print(f"roofline breakdown: dominant={bd.get('dominant')}  util={bd.get('util'):.4f}  eff_tflops={bd.get('eff_tflops'):.3f}")
                    print(f"compute_us={bd.get('compute_us'):.4f}  mem_us={bd.get('mem_us'):.4f}  roofline_us={bd.get('roofline_us'):.4f}")
                    print(f"overhead_us={bd.get('overhead_us'):.4f}  overhead_scaled_us={bd.get('overhead_scaled_us'):.4f}  time_scale={bd.get('time_scale'):.3f}")
                    print(f"total_us(breakdown)={bd.get('total_us'):.4f}  delta_vs_fast_us={float(bd.get('total_us') - float(pred_fast)):.6f}")
                    dbg_printed += 1

            y_true.append(float(gt))
            y_pred_lut.append(float(pred_lut))
            y_pred_fast.append(float(pred_fast))

        rows.append({'op': op_kind, 'method': 'lut_idw', **_metrics(y_true, y_pred_lut)})
        rows.append({'op': op_kind, 'method': 'fast_roofline', **_metrics(y_true, y_pred_fast)})


    # Print + save
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        df['op'] = df['op'].astype(str)
        df['method'] = df['method'].astype(str)
        df = df.sort_values(['op', 'method']).reset_index(drop=True)

        print('\n=== NPU single-op fit summary (per-op) ===')
        print(df.to_string(index=False))

        df.to_csv(args.out, index=False)
        print(f"\nSaved CSV: {os.path.abspath(args.out)}")
    except Exception:
        print('\n=== NPU single-op fit summary (per-op) ===')
        for r in rows:
            print(r)
        # best-effort CSV
        try:
            import csv
            with open(args.out, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"\nSaved CSV: {os.path.abspath(args.out)}")
        except Exception as e:
            print('WARN: failed to write CSV:', e)


if __name__ == '__main__':
    main()
