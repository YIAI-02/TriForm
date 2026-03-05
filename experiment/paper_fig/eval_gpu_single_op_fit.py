#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export TRIFORM_ROOT="${TRIFORM_ROOT:-$(cd ../.. && pwd)}"
export PYTHONPATH=$TRIFORM_ROOT/algorithms:$TRIFORM_ROOT:$PYTHONPATH

python algorithms/eval_gpu_single_op_fit.py \
  --run-gpu \
  --verify-script verify/schedule_deploy_verify.py \
  --out-dir ./eval_gpu_single_op_fit_output \
  --prefix a100_single_op \
  --llama-shape-json ./algorithms/llama_7b_shape.json \
  --batches 1,2,4 \
  --prefill-lens 128,512,1024 \
  --decode-context-lens 128,256,512,1024,2048,4096 \
  --phases decode \
  --gpu-device cuda \
  --gpu-dtype fp16 \
  --dtype fp16 \
  --device-name A100_GPU0 \
  --warmup 3 --iters 10 \
  --debug
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------------------
# Pretty helper
# --------------------------------------------------------------------------------------
def _pretty(obj: Any) -> str:
    try:
        import json as _json
        return _json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


# --------------------------------------------------------------------------------------
# Repo/module location helpers (same style as eval_npu_single_op_fit.py) 
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

    # Ensure sibling imports are visible
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
# Minimal device/cluster stubs (duck-typed for CostModel) 
# --------------------------------------------------------------------------------------
@dataclass
class SimpleDev:
    name: str = "A100_GPU0"
    type: str = "npu"  # IMPORTANT: CostModel uses 'npu' codepath for FAST-mode + per-device overrides
    tflops: float = 0.0
    mem_bw_GBs: float = 0.0


class SimpleCluster:
    def __init__(self, devices: Dict[str, Any]):
        self.devices = dict(devices)

    def devices_by_type(self, typ: str) -> List[Any]:
        t = str(typ).lower()
        return [d for d in self.devices.values() if str(getattr(d, "type", "")).lower() == t]


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
# Metrics (same as eval_npu_single_op_fit.py) 
# --------------------------------------------------------------------------------------
def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) == 0:
        return float("nan")
    yt = [float(x) for x in y_true]
    yp = [float(x) for x in y_pred]
    mean_y = sum(yt) / len(yt)
    ss_tot = sum((y - mean_y) ** 2 for y in yt)
    ss_res = sum((y - p) ** 2 for y, p in zip(yt, yp))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    if not y_true:
        return {
            "n": 0,
            "mae_us": float("nan"),
            "mape_pct": float("nan"),
            "mean_err_us": float("nan"),
            "r2": float("nan"),
        }
    err = [float(p) - float(t) for t, p in zip(y_true, y_pred)]
    abs_err = [abs(e) for e in err]
    rel = [abs(e) / max(1e-9, float(t)) for t, e in zip(y_true, err)]
    return {
        "n": int(len(y_true)),
        "mae_us": float(_mean(abs_err)),
        "mape_pct": float(_mean(rel) * 100.0),
        "mean_err_us": float(_mean(err)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------------------
# Shape helpers (mirror verify/schedule_deploy_verify.py logic) 
# --------------------------------------------------------------------------------------
def decode_token_index_from_sample_step(step_sample: int, stride: int) -> int:
    s = int(step_sample)
    st = int(stride) if stride is not None else 1
    if st <= 1:
        return s
    if s <= 0:
        return 0
    if s == 1:
        return 1
    return int((s - 1) * st - 1)


def _infer_query_key_len(
    *,
    phase: str,
    step: int,
    prefill_len: int,
    decode_context_lens: Optional[List[int]],
    decode_stride: int,
) -> Tuple[int, int]:
    """Return (T, K) as used by schedule_deploy_verify.GPUBackend shape inference."""
    ph = str(phase).strip().lower()
    if ph == "prefill":
        T = int(prefill_len)
        K = int(prefill_len)
        return max(1, T), max(1, K)

    # decode
    T = 1
    if decode_context_lens:
        if 0 <= int(step) < len(decode_context_lens):
            K = int(decode_context_lens[int(step)])
        else:
            K = int(decode_context_lens[-1])
        return 1, max(1, K)

    # fallback: inferred from step+stride
    tok = int(decode_token_index_from_sample_step(int(step), int(decode_stride)))
    K = int(prefill_len + 1 + tok)
    return 1, max(1, K)


def _parse_int_list(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    ss = str(s).strip()
    if not ss:
        return []
    out: List[int] = []
    for part in ss.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _parse_str_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    ss = str(s).strip()
    if not ss:
        return []
    out: List[str] = []
    for part in ss.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(p)
    return out


def _decode_lens_tag(decode_context_lens: List[int]) -> str:
    if not decode_context_lens:
        return "Kauto"
    xs = [int(x) for x in decode_context_lens if int(x) > 0]
    if not xs:
        return "Kauto"
    mn = min(xs)
    mx = max(xs)
    n = len(xs)
    if mn == mx:
        return f"K{mn}"
    return f"K{mn}-{mx}x{n}"


# --------------------------------------------------------------------------------------
# Task JSON generation (consumed by schedule_deploy_verify.py run-gpu) 
# --------------------------------------------------------------------------------------
_DEFAULT_OPS = [
    "LN",
    "Q", "K", "V", "O",
    "FFN_W1", "FFN_W3", "SwiGLU", "FFN_W2",
    "Add",
    "QK", "Softmax", "SV",
]

_ATTENTION_OPS = {"QK", "Softmax", "SV"}

# --------------------------------------------------------------------------------------
# LLaMA shape loader
# --------------------------------------------------------------------------------------
def load_llama_shape_json(path: str) -> Dict[str, int]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"llama shape json not found: {p}")

    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"llama shape json must be a JSON object, got {type(obj)}")

    def _pick_int(keys: Sequence[str], default: Optional[int] = None) -> Optional[int]:
        for k in keys:
            if k in obj and obj[k] is not None:
                try:
                    return int(obj[k])
                except Exception:
                    pass
        return default

    dim = _pick_int(["hidden_dim", "hidden_size", "dim"])  # hidden size
    ffn_dim = _pick_int(["intermediate_dim", "ffn_dim", "mlp_dim"])  # FFN/MLP intermediate
    n_heads = _pick_int(["q_head_num", "num_attention_heads", "n_heads", "head_num"])  # Q heads
    n_kv_heads = _pick_int(["kv_head_num", "num_key_value_heads", "n_kv_heads"], default=n_heads)

    if dim is None or int(dim) <= 0:
        raise ValueError(f"Invalid hidden_dim in shape json: {dim}")
    if ffn_dim is None or int(ffn_dim) <= 0:
        raise ValueError(f"Invalid intermediate_dim in shape json: {ffn_dim}")
    if n_heads is None or int(n_heads) <= 0:
        raise ValueError(f"Invalid q_head_num in shape json: {n_heads}")
    if n_kv_heads is None or int(n_kv_heads) <= 0:
        n_kv_heads = int(n_heads)

    if int(dim) % int(n_heads) != 0:
        raise ValueError(f"hidden_dim {dim} must be divisible by q_head_num {n_heads}")
    head_dim = int(dim) // int(n_heads)

    return {
        "dim": int(dim),
        "ffn_dim": int(ffn_dim),
        "n_heads": int(n_heads),
        "n_kv_heads": int(n_kv_heads),
        "head_dim": int(head_dim),
    }


def build_single_op_tasks_json(
    *,
    out_path: Path,
    prefix: str,
    dim: int,
    ffn_dim: int,
    n_heads: int,
    n_kv_heads: int,
    batch: int,
    prefill_len: int,
    decode_context_lens: List[int],
    decode_stride: int,
    phases: List[str],
    ops: List[str],
    device: str,
    gpu_dtype: str,
) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ph_list = [str(p).strip().lower() for p in phases if str(p).strip()]
    if not ph_list:
        ph_list = ["decode"]

    # Normalize ops list
    ops_list = [str(o).strip() for o in ops if str(o).strip()]
    if not ops_list:
        ops_list = list(_DEFAULT_OPS)

    D = int(dim)
    F = int(ffn_dim)
    H = int(n_heads)
    Hkv = int(n_kv_heads)
    shards = 1
    if H <= 0:
        raise ValueError("n_heads must be > 0")
    if D <= 0 or F <= 0 or prefill_len <= 0:
        raise ValueError("dim/ffn_dim/prefill_len must be positive")
    if D % H != 0:
        raise ValueError(f"dim={D} must be divisible by n_heads={H}")
    Hd = int(D // H)

    if not decode_context_lens:
        # Keep at least one decode length so attention ops can vary K.
        decode_context_lens = [prefill_len]

    # Config dict must match WorkloadConfig fields in schedule_deploy_verify.py 
    cfg = {
        "dim": int(D),
        "ffn_dim": int(F),
        "n_heads": int(H),
        "n_kv_heads": int(Hkv),
        "prefill_len": int(prefill_len),
        "decode_context_lens": [int(x) for x in decode_context_lens],
        "decode_stride": int(decode_stride),
        "device": str(device),
        "gpu_dtype": str(gpu_dtype),
        "batch": int(batch),
        "segment_scope": "single_op",
        # keep defaults for the rest
    }

    tasks: List[Dict[str, Any]] = []

    def add_task(op: str, phase: str, step: int, shard_idx: int = 0):
        T, K = _infer_query_key_len(
            phase=phase,
            step=step,
            prefill_len=prefill_len,
            decode_context_lens=decode_context_lens,
            decode_stride=decode_stride,
        )
        key = f"{prefix}|{phase}|{op}|B{batch}|T{T}|K{K}|D{D}|F{F}|H{H}|Hkv{Hkv}|Hd{Hd}"
        tasks.append({
            "key": key,
            "sig": {"device_type": "npu", "phase": str(phase), "step": int(step)},
            "ops": [{"op": str(op), "shard": int(shard_idx)}],
            "ops_repr": f"{op}:{shard_idx}",
            "count_hint": 1,
            # extra metadata (ignored by run-gpu, but useful for debugging)
            "meta": {
                "op": str(op),
                "phase": str(phase),
                "step": int(step),
                "batch": int(batch),
                "query_len": int(T),
                "key_len": int(K),
                "dim": int(D),
                "ffn_dim": int(F),
                "n_heads": int(H),
                "n_kv_heads": int(Hkv),
                "head_dim": int(Hd),
            },
        })

    # Build tasks
    for ph in ph_list:
        if ph == "prefill":
            step = -1
            for op in ops_list:
                add_task(op, "prefill", step, shard_idx=0)
        elif ph == "decode":
            # Non-attention ops: just one sample at step=0
            for op in ops_list:
                if op in _ATTENTION_OPS:
                    # One task per decode_context_len (vary K)
                    for step in range(len(decode_context_lens)):
                        add_task(op, "decode", step, shard_idx=0)
                else:
                    add_task(op, "decode", 0, shard_idx=0)
        else:
            raise ValueError(f"Unknown phase: {ph!r} (expected prefill/decode)")

    obj = {
        "version": 3,
        "task_type": "segment",
        "backend": "gpu",
        "segment_scope": "single_op",
        "schedules": [],
        "weight_load_s": 0.0,
        "weight_load_meta": {"gpu_s": 0.0, "pim_s": 0.0, "unknown_s": 0.0, "total_s": 0.0},
        "weight_load_by_schedule": {},
        "config": cfg,
        "tasks": sorted(tasks, key=lambda x: x["key"]),
    }

    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------------------
# CostModel FAST prediction
# --------------------------------------------------------------------------------------
def _make_cost_model_node_for_op(
    op: str,
    *,
    dim: int,
    ffn_dim: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    kv_len: int,
    phase: str,
) -> SimpleNode:
    op = str(op)
    ph = str(phase).strip().lower()

    # Common attrs
    attrs: Dict[str, Any] = {"dim": int(dim)}

    if op in ("Q", "K", "V", "O"):
        attrs.update({
            "q_heads": int(q_heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
        })

    if op in ("FFN_W1", "FFN_W2", "FFN_W3", "SwiGLU"):
        attrs["ffn_dim"] = int(ffn_dim)

    if op in ("QK", "Softmax", "SV"):
        attrs.update({
            "q_heads": int(q_heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
            "kv_len": int(kv_len),
            # IMPORTANT: GPUBackend benchmark is dense (no causal masking). 
            "causal": False,
        })

    if op == "Add":
        # nothing extra
        pass

    if op == "LN":
        # nothing extra
        pass

    return SimpleNode(op, attrs=attrs)


def _fast_cost_model_predict_us(cm_mod, cm_obj, dev: Any, *, node: Any, batch: int, seq_len: int, phase: str) -> Optional[float]:
    """CostModel fast-backend prediction using node_device_cost() (includes util + overhead)."""
    pred_s = float(cm_obj.node_device_cost(node, dev, _DummyLabel(), batch=int(batch), seq_len=int(seq_len), phase=str(phase)))
    if not math.isfinite(pred_s):
        return None
    return float(pred_s * 1e6)


def _fast_roofline_breakdown_us(cm_mod, cm_obj, dev: Any, *, node: Any, batch: int, seq_len: int, phase: str) -> Dict[str, Any]:
    """Detailed breakdown for FAST(roofline) path (for debugging)."""
    attrs = getattr(node, "attrs", {}) or {}

    # Memory lower bound
    rd_B, wr_B = cm_obj.estimate_activation_bytes(node, int(batch), int(seq_len), str(phase))
    mem_s = float(cm_obj.mem_time(int(rd_B + wr_B), dev))

    # Compute lower bound (with utilization correction)
    flops = float(cm_obj.estimate_flops(node, int(batch), int(seq_len), str(phase)))
    util = float(getattr(cm_obj, "_compute_utilization")(flops, dev))
    eff_tflops = float(cm_obj.effective_tflops(flops, dev))
    compute_s = float(cm_obj.flop_time(flops, dev))

    roofline_s = max(compute_s, mem_s)
    dominant = "compute" if compute_s >= mem_s else "memory"

    # Time scale hint (normally 1.0)
    try:
        time_scale = float(getattr(cm_obj, "_time_scale_hint")(node, getattr(dev, "type", "")))
    except Exception:
        time_scale = 1.0

    # Kernel launch overhead
    raw_key = str(getattr(node, "name", "") or getattr(node, "id", "") or "")
    op_key = cm_mod._normalize_npu_op_key(raw_key)
    overhead_s = 0.0
    overhead_scaled_s = 0.0
    try:
        overhead_s = float(cm_obj.kernel_launch_overhead_s(op_key, dev, phase=str(phase)))
        kl_cfg = getattr(cm_obj, "_kernel_launch_cfg")(dev)
        overhead_scaled_s = overhead_s * time_scale if bool(kl_cfg.get("scale_by_time_scale", False)) else overhead_s
    except Exception:
        overhead_s = 0.0
        overhead_scaled_s = 0.0

    total_s = float(roofline_s) * float(time_scale) + float(overhead_scaled_s)

    return {
        "node_name": getattr(node, "name", "?"),
        "op_key": str(op_key),
        "phase": str(phase),
        "batch": int(batch),
        "seq_len": int(seq_len),
        "attrs": dict(attrs),
        "bytes_rd_B": int(rd_B),
        "bytes_wr_B": int(wr_B),
        "bytes_total_B": int(rd_B + wr_B),
        "mem_us": float(mem_s * 1e6),
        "flops": float(flops),
        "util": float(util),
        "peak_tflops": float(getattr(dev, "tflops", 0.0) or 0.0),
        "eff_tflops": float(eff_tflops),
        "compute_us": float(compute_s * 1e6),
        "roofline_us": float(roofline_s * 1e6),
        "dominant": dominant,
        "time_scale": float(time_scale),
        "overhead_us": float(overhead_s * 1e6),
        "overhead_scaled_us": float(overhead_scaled_s * 1e6),
        "total_us": float(total_s * 1e6),
    }


# --------------------------------------------------------------------------------------
# GPU benchmark runner (via schedule_deploy_verify.py run-gpu) 
# --------------------------------------------------------------------------------------
def _resolve_verify_script(path_hint: Optional[str]) -> str:
    if path_hint:
        p = os.path.abspath(os.path.expanduser(path_hint))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"--verify-script not found: {p}")
        return p
    return _find_repo_module_any([
        os.path.join("verify", "schedule_deploy_verify_gpu.py"),
        os.path.join("verify", "schedule_deploy_verify.py"),
        "schedule_deploy_verify_gpu.py",
        "schedule_deploy_verify.py",
    ])


def _resolve_llama_shape_json(path_hint: Optional[str]) -> str:
    if path_hint:
        p = Path(path_hint).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--llama-shape-json not found: {p}")
        return str(p)

    envp = os.environ.get("LLAMA_SHAPE_JSON", "").strip()
    if envp:
        p = Path(envp).expanduser().resolve()
        if p.is_file():
            return str(p)

    cands: List[Path] = []
    cwd = Path(os.getcwd()).resolve()
    cands += [cwd / "llama_shape.json", cwd / "llama_7b_shape.json"]
    try:
        here = Path(__file__).resolve().parent
        cands += [here / "llama_shape.json", here / "llama_7b_shape.json"]
    except Exception:
        pass

    for p in cands:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        "Could not locate llama_shape.json. Provide --llama-shape-json or set env LLAMA_SHAPE_JSON.\n"
        f"Tried: {[str(x) for x in cands]}"
    )


def run_gpu_benchmark(
    *,
    verify_script: str,
    tasks_json: str,
    out_json: str,
    warmup: int,
    iters: int,
    device: Optional[str],
    gpu_dtype: Optional[str],
    batch: Optional[int],
    debug: bool,
    debug_txt: Optional[str],
) -> None:
    cmd = [sys.executable, str(verify_script), "run-gpu", "--tasks", str(tasks_json), "--out", str(out_json),
           "--warmup", str(int(warmup)), "--iters", str(int(iters))]
    if device:
        cmd += ["--device", str(device)]
    if gpu_dtype:
        cmd += ["--gpu-dtype", str(gpu_dtype)]
    if batch is not None:
        cmd += ["--batch", str(int(batch))]
    if debug:
        cmd += ["--debug"]
        if debug_txt:
            cmd += ["--debug-txt", str(debug_txt)]
    print("[eval][run-gpu] Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def _save_rows_csv(rows: List[Dict[str, Any]], path: Path, *, sort_by: Optional[List[str]] = None) -> None:
    """Best-effort CSV writer (pandas if available, else csv module)."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to save: {path}")
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        if sort_by:
            cols = [c for c in sort_by if c in df.columns]
            if cols:
                df = df.sort_values(cols).reset_index(drop=True)
        df.to_csv(path, index=False)
        return
    except Exception:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)


def evaluate_one_case(
    *,
    cm_mod,
    cm_obj,
    dev: Any,
    tasks_json: Path,
    results_json: Path,
    decode_context_lens_fallback: List[int],
    decode_stride_fallback: int,
    debug: bool,
    debug_print_n: int,
    case_meta: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate one (batch, prefill_len) case; return (summary_rows, detail_rows)."""
    res_obj = json.loads(Path(results_json).read_text(encoding="utf-8"))
    gt_map = res_obj.get("results", {}) or {}
    if not isinstance(gt_map, dict) or not gt_map:
        raise ValueError(f"Malformed results json (missing 'results'): {results_json}")

    task_obj = json.loads(Path(tasks_json).read_text(encoding="utf-8"))
    cfg = task_obj.get("config", {}) or {}
    tasks = task_obj.get("tasks", []) or []
    if not tasks:
        raise ValueError(f"tasks json has no tasks: {tasks_json}")

    # Shapes from tasks config
    D = int(cfg.get("dim", 0) or 0)
    F = int(cfg.get("ffn_dim", 0) or 0)
    H = int(cfg.get("n_heads", 0) or 0)
    Hkv = int(cfg.get("n_kv_heads", H) or H)
    batch = int(cfg.get("batch", 1) or 1)
    prefill_len = int(cfg.get("prefill_len", 1) or 1)

    if D <= 0 or F <= 0 or H <= 0:
        raise ValueError(f"Invalid shape in tasks config: dim={D}, ffn_dim={F}, n_heads={H}")
    if D % H != 0:
        raise ValueError(f"dim={D} must be divisible by n_heads={H}")
    Hd = int(D // H)

    decode_context_lens_cfg = cfg.get("decode_context_lens", None)
    if isinstance(decode_context_lens_cfg, list) and decode_context_lens_cfg:
        decode_context_lens = [int(x) for x in decode_context_lens_cfg]
    else:
        decode_context_lens = list(decode_context_lens_fallback)
    decode_stride = int(cfg.get("decode_stride", decode_stride_fallback) or decode_stride_fallback)

    detail_rows: List[Dict[str, Any]] = []
    by_group_true: Dict[Tuple[str, str], List[float]] = {}
    by_group_pred: Dict[Tuple[str, str], List[float]] = {}
    dbg_count: Dict[Tuple[str, str], int] = {}

    for t in tasks:
        key = str(t.get("key", ""))
        sig = t.get("sig", {}) or {}
        ops_l = t.get("ops", []) or []
        if not key or not ops_l:
            continue

        op = str(ops_l[0].get("op", ""))
        phase = str(sig.get("phase", "decode")).strip().lower()
        step = int(sig.get("step", 0))
        shard_idx = int(ops_l[0].get("shard", 0))

        if key not in gt_map:
            continue
        gt_s = float(gt_map[key])
        if not math.isfinite(gt_s) or gt_s <= 0:
            continue
        gt_us = gt_s * 1e6

        T, K = _infer_query_key_len(
            phase=phase,
            step=step,
            prefill_len=prefill_len,
            decode_context_lens=decode_context_lens,
            decode_stride=decode_stride,
        )

        node = _make_cost_model_node_for_op(
            op,
            dim=int(D),
            ffn_dim=int(F),
            q_heads=int(H),
            kv_heads=int(Hkv),
            head_dim=int(Hd),
            kv_len=int(K),
            phase=phase,
        )

        pred_us = _fast_cost_model_predict_us(
            cm_mod,
            cm_obj,
            dev,
            node=node,
            batch=int(batch),
            seq_len=int(T),
            phase=str(phase),
        )
        if pred_us is None or not math.isfinite(float(pred_us)):
            continue

        bd = _fast_roofline_breakdown_us(
            cm_mod,
            cm_obj,
            dev,
            node=node,
            batch=int(batch),
            seq_len=int(T),
            phase=str(phase),
        )

        gk = (op, phase)
        if bool(debug) and dbg_count.get(gk, 0) < int(debug_print_n):
            dbg_count[gk] = dbg_count.get(gk, 0) + 1
            print("\n--- [DEBUG][sample] ---")
            print(f"case={case_meta.get('case_id','?')}  key={key}")
            print(f"op={op} phase={phase} step={step} shard={shard_idx}  B={batch} T={T} K={K}  D={D} F={F} H={H} Hkv={Hkv} Hd={Hd}")
            print(f"gt_us={gt_us:.4f}  fast_us={float(pred_us):.4f}")
            print(f"roofline breakdown: dominant={bd.get('dominant')} util={bd.get('util'):.4f} eff_tflops={bd.get('eff_tflops'):.3f}")
            print(f"compute_us={bd.get('compute_us'):.4f}  mem_us={bd.get('mem_us'):.4f}  roofline_us={bd.get('roofline_us'):.4f}")
            print(f"overhead_us={bd.get('overhead_us'):.4f}  overhead_scaled_us={bd.get('overhead_scaled_us'):.4f}  time_scale={bd.get('time_scale'):.3f}")
            print(f"total_us(breakdown)={bd.get('total_us'):.4f}  delta_vs_pred_us={float(bd.get('total_us') - float(pred_us)):.6f}")

        by_group_true.setdefault(gk, []).append(float(gt_us))
        by_group_pred.setdefault(gk, []).append(float(pred_us))

        row = {
            **dict(case_meta),
            "key": key,
            "op": op,
            "phase": phase,
            "step": int(step),
            "batch": int(batch),
            "T_query_len": int(T),
            "K_key_len": int(K),
            "dim": int(D),
            "ffn_dim": int(F),
            "n_heads": int(H),
            "n_kv_heads": int(Hkv),
            "head_dim": int(Hd),
            "gt_us": float(gt_us),
            "pred_fast_us": float(pred_us),
            # breakdown
            "dominant": bd.get("dominant"),
            "mem_us": bd.get("mem_us"),
            "compute_us": bd.get("compute_us"),
            "roofline_us": bd.get("roofline_us"),
            "util": bd.get("util"),
            "eff_tflops": bd.get("eff_tflops"),
            "overhead_us": bd.get("overhead_us"),
            "total_breakdown_us": bd.get("total_us"),
        }
        detail_rows.append(row)

    if not detail_rows:
        raise RuntimeError(
            f"No valid samples were evaluated for case={case_meta.get('case_id','?')}.\n"
            "Common causes:\n"
            "  - results json keys do not match tasks json keys\n"
            "  - run-gpu failed or produced empty results\n"
            "  - your ops list is empty\n"
        )

    summary_rows: List[Dict[str, Any]] = []
    for (op, phase), y_true in sorted(by_group_true.items(), key=lambda x: (x[0][0], x[0][1])):
        y_pred = by_group_pred.get((op, phase), [])
        summary_rows.append({
            **dict(case_meta),
            "op": op,
            "phase": phase,
            **_metrics(y_true, y_pred),
        })

    return summary_rows, detail_rows


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # I/O
    ap.add_argument("--out-dir", default="./eval_gpu_single_op_fit_output", help="Directory to store json/csv outputs")
    ap.add_argument("--prefix", default="a100_single_op", help="Prefix for output files under --out-dir")
    ap.add_argument("--tasks-json", default=None, help="Override tasks json path (default: <out-dir>/<prefix>.gpu_tasks.json)")
    ap.add_argument("--gpu-results-json", default=None, help="Override gpu results json path (default: <out-dir>/<prefix>.gpu_results.json)")

    # GPU benchmark control
    ap.add_argument("--run-gpu", action="store_true", help="Run GPU benchmark via schedule_deploy_verify.py run-gpu")
    ap.add_argument("--verify-script", default=None, help="Path to schedule_deploy_verify(_gpu).py (optional)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--gpu-device", default="cuda", help="Passed to run-gpu --device (default: cuda)")
    ap.add_argument("--gpu-dtype", default="fp16", help="Passed to run-gpu --gpu-dtype (fp16/bf16/fp32)")
    # Workload shapes
    ap.add_argument(
        "--llama-shape-json",
        default="/lustre/home/2501111916/workspace/XPUPIM_0226_gpupim_parameter/TriForm/configs/llama_7b_shape.json",
        help="Path to llama_shape.json (e.g. llama_7b_shape.json). If omitted, we try to auto-locate it.",
    )

    # Sweep controls
    ap.add_argument(
        "--batches",
        default="1",
        help="Comma-separated batch sizes to sweep (e.g. 1,2,4). Default: 1",
    )
    ap.add_argument(
        "--prefill-lens",
        default="1024",
        help="Comma-separated prefill lengths to sweep (e.g. 128,512,1024). Default: 1024",
    )
    # Backward-compatible single-value flags (optional)
    ap.add_argument("--batch", type=int, default=None, help="(deprecated) single batch; prefer --batches")
    ap.add_argument("--prefill-len", type=int, default=None, help="(deprecated) single prefill; prefer --prefill-lens")

    ap.add_argument("--decode-context-lens", default="128,256,512,1024,2048,4096,8192",
                    help="Comma-separated K lengths for decode attention ops (used as decode_context_lens list)")
    ap.add_argument("--decode-stride", type=int, default=1, help="Only used if decode_context_lens is empty")

    ap.add_argument("--phases", default="decode", help="Comma-separated phases: decode,prefill (default decode)")
    ap.add_argument("--ops", default=None, help=f"Comma-separated ops to benchmark. Default: {','.join(_DEFAULT_OPS)}")

    # CostModel device params
    ap.add_argument("--dtype", default="fp16", help="CostModel dtype (fp16/fp32/bf16/int8...)")
    ap.add_argument("--device-name", default="A100_GPU0",
                    help="Device name string used for config.by_device_name matching (e.g., A100_GPU0) ")
    ap.add_argument("--a100-tflops", type=float, default=312.0, help="Peak TFLOPS used by CostModel")
    ap.add_argument("--a100-mem-bw-gbs", type=float, default=1555.0, help="HBM bandwidth (GB/s) used by CostModel")

    # Debug
    ap.add_argument("--debug", action="store_true", help="Verbose debug prints + CostModel DEBUG logs")
    ap.add_argument("--debug-log-file", default=None, help="Path to debug log file (default: <out-dir>/<prefix>.debug_log.txt)")
    ap.add_argument("--debug-print-n", type=int, default=8, help="Print detailed breakdown for first N samples per (op,phase)")

    # Optional explicit module paths
    ap.add_argument("--cost-model-py", default=None)
    ap.add_argument("--config-py", default=None)

    # Output CSVs
    ap.add_argument("--out-summary", default=None, help="Summary CSV path (default: <out-dir>/<prefix>.summary.csv)")
    ap.add_argument("--out-detail", default=None, help="Detail CSV path (default: <out-dir>/<prefix>.detail.csv)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(args.prefix).strip()
    if not prefix:
        prefix = "a100_single_op"

    # Aggregated sweep outputs
    out_sweep_summary = out_dir / f"{prefix}.sweep_summary.csv"
    out_sweep_detail = out_dir / f"{prefix}.sweep_detail.csv"

    # Optional single-case overrides (mainly for re-evaluating existing jsons)
    tasks_json_override = Path(args.tasks_json).expanduser().resolve() if args.tasks_json else None
    results_json_override = Path(args.gpu_results_json).expanduser().resolve() if args.gpu_results_json else None
    out_summary_override = Path(args.out_summary).expanduser().resolve() if args.out_summary else None
    out_detail_override = Path(args.out_detail).expanduser().resolve() if args.out_detail else None

    # Debug logs
    base_debug_log_file = args.debug_log_file or str(out_dir / f"{prefix}.debug_log.txt")

    phases = _parse_str_list(args.phases) or ["decode"]
    ops = _parse_str_list(args.ops) if args.ops else list(_DEFAULT_OPS)
    decode_context_lens = _parse_int_list(args.decode_context_lens)

    # ----------------------------------------------------------------------------------
    # Import config + cost_model (so device-name overrides work)  
    # ----------------------------------------------------------------------------------
    cost_model_path = args.cost_model_py or _find_repo_module_any([
        os.path.join("algorithms", "cost_model.py"),
        "cost_model.py",
    ])

    # Import config next to cost_model.py by default (ensures same module instance inside cost_model) 
    if args.config_py:
        cfg_path = os.path.abspath(os.path.expanduser(args.config_py))
    else:
        cand_cfg = os.path.join(os.path.dirname(cost_model_path), "config.py")
        if os.path.isfile(cand_cfg):
            cfg_path = cand_cfg
        else:
            cfg_path = _find_repo_module_any([
                os.path.join("algorithms", "config.py"),
                "config.py",
            ])

    cfg_mod = _import_from_path("config", cfg_path)

    # Enable debug logging before importing cost_model (mirrors eval_npu_single_op_fit.py) 
    try:
        if bool(args.debug):
            Path(base_debug_log_file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        if hasattr(cfg_mod, "setup_logging"):
            cfg_mod.setup_logging(bool(args.debug), log_file=str(base_debug_log_file))
    except Exception as e:
        print("[WARN] config.setup_logging failed:", e)

    cm_mod = _import_from_path("cost_model", cost_model_path)

    # ----------------------------------------------------------------------------------
    # Resolve model shape + sweep lists
    # ----------------------------------------------------------------------------------
    shape_json_path = _resolve_llama_shape_json(args.llama_shape_json)
    shape = load_llama_shape_json(shape_json_path)
    D = int(shape["dim"])
    F = int(shape["ffn_dim"])
    H = int(shape["n_heads"])
    Hkv = int(shape.get("n_kv_heads", H))
    Hd = int(shape["head_dim"])

    # Sweep lists
    if args.batch is not None:
        batches = [int(args.batch)]
    else:
        batches = _parse_int_list(args.batches) or [1]
    if args.prefill_len is not None:
        prefill_lens = [int(args.prefill_len)]
    else:
        prefill_lens = _parse_int_list(args.prefill_lens) or [1024]

    # Keep deterministic ordering
    batches = [int(x) for x in batches]
    prefill_lens = [int(x) for x in prefill_lens]

    # "decode length" is represented by this list (K values) for attention ops.
    decode_tag = _decode_lens_tag(decode_context_lens)
    decode_min = min(decode_context_lens) if decode_context_lens else None
    decode_max = max(decode_context_lens) if decode_context_lens else None
    decode_n = len(decode_context_lens)

    # ----------------------------------------------------------------------------------
    # Build CostModel with A100 overrides via device-name matching in config.py
    # ----------------------------------------------------------------------------------
    dev = SimpleDev(
        name=str(args.device_name),
        type="npu",
        tflops=float(args.a100_tflops),
        mem_bw_GBs=float(args.a100_mem_bw_gbs),
    )
    cluster = SimpleCluster({"GPU0": dev})
    cm_obj = cm_mod.CostModel(cluster=cluster, dtype=str(args.dtype), pim_fast_mode=True, npu_backend="fast")
    try:
        cm_obj._npu_backend_impl_name = "fast"
    except Exception:
        pass

    # ----------------------------------------------------------------------------------
    # Debug: verify config overrides are picked up for A100
    # ----------------------------------------------------------------------------------
    if bool(args.debug):
        print("\n=== [DEBUG] Module + config wiring ===")
        print("config.py path        :", getattr(cfg_mod, "__file__", None))
        try:
            print("cost_model sees config:", getattr(cm_mod._config, "__file__", None))
            print("config identity match :", bool(cm_mod._config is cfg_mod))
        except Exception:
            pass

        print("\n=== [DEBUG] Device used for matching ===")
        print("dev.name      :", getattr(dev, "name", None))
        print("dev.type      :", getattr(dev, "type", None))
        print("dev.tflops    :", getattr(dev, "tflops", None))
        print("dev.mem_bw_GBs:", getattr(dev, "mem_bw_GBs", None))
        try:
            fam = cm_mod._device_family_key_from_name(getattr(dev, "name", ""))
            print("device family :", fam)
        except Exception:
            pass

        print("\n=== [DEBUG] Shape (from llama_shape.json) ===")
        print(f"shape_json={shape_json_path}")
        print(f"dim={D} ffn_dim={F} n_heads={H} n_kv_heads={Hkv} head_dim={Hd}")

        try:
            cu_cfg = getattr(cfg_mod, "COMPUTE_UTILIZATION", None)
            kl_cfg = getattr(cfg_mod, "KERNEL_LAUNCH_OVERHEAD", None)
            cu_hit = cm_mod._lookup_cfg_by_device_name(cu_cfg, dev)
            kl_hit = cm_mod._lookup_cfg_by_device_name(kl_cfg, dev)
            print("\n=== [DEBUG] COMPUTE_UTILIZATION per-device override ===")
            print(_pretty(cu_hit))
            print("\n=== [DEBUG] KERNEL_LAUNCH_OVERHEAD per-device override ===")
            print(_pretty(kl_hit))
        except Exception as e:
            print("[WARN] failed to resolve per-device overrides:", e)

        try:
            merged_kl = cm_obj._kernel_launch_cfg(dev)
            print("\n=== [DEBUG] CostModel merged kernel launch cfg ===")
            print(_pretty(merged_kl))
        except Exception:
            pass

        print("\n=== [DEBUG] Utilization probe (u = f(flops)) ===")
        probe_flops = [1e6, 1e7, 5e7, 1e8, 1e9, 1e10, 1e11, 1e12, 5e12]
        for f in probe_flops:
            try:
                u = float(cm_obj._compute_utilization(float(f), dev))
                eff = float(cm_obj.effective_tflops(float(f), dev))
                print(f"flops={f:.3e}  util={u:.4f}  eff_tflops={eff:.3f}")
            except Exception as e:
                print(f"flops={f:.3e}  util=<err> ({e})")

    # ----------------------------------------------------------------------------------
    # Single-case override mode
    # ----------------------------------------------------------------------------------
    if tasks_json_override is not None or results_json_override is not None or out_summary_override is not None or out_detail_override is not None:
        if len(batches) != 1 or len(prefill_lens) != 1:
            raise ValueError(
                "--tasks-json/--gpu-results-json/--out-summary/--out-detail are only supported for a single case.\n"
                f"Got batches={batches}, prefill_lens={prefill_lens}"
            )
        b0 = int(batches[0])
        p0 = int(prefill_lens[0])
        case_prefix = prefix
        tasks_json = tasks_json_override or (out_dir / f"{case_prefix}.gpu_tasks.json")
        results_json = results_json_override or (out_dir / f"{case_prefix}.gpu_results.json")
        out_summary = out_summary_override or (out_dir / f"{case_prefix}.summary.csv")
        out_detail = out_detail_override or (out_dir / f"{case_prefix}.detail.csv")
        gpu_debug_txt = out_dir / f"{case_prefix}.gpu_debug.txt"

        build_single_op_tasks_json(
            out_path=tasks_json,
            prefix=case_prefix,
            dim=int(D),
            ffn_dim=int(F),
            n_heads=int(H),
            n_kv_heads=int(Hkv),
            batch=int(b0),
            prefill_len=int(p0),
            decode_context_lens=decode_context_lens,
            decode_stride=int(args.decode_stride),
            phases=phases,
            ops=ops,
            device=str(args.gpu_device),
            gpu_dtype=str(args.gpu_dtype),
        )
        print(f"[eval] wrote tasks json: {tasks_json}")

        if bool(args.run_gpu):
            verify_script = _resolve_verify_script(args.verify_script)
            print(f"[eval] verify script: {verify_script}")
            run_gpu_benchmark(
                verify_script=verify_script,
                tasks_json=str(tasks_json),
                out_json=str(results_json),
                warmup=int(args.warmup),
                iters=int(args.iters),
                device=str(args.gpu_device) if args.gpu_device else None,
                gpu_dtype=str(args.gpu_dtype) if args.gpu_dtype else None,
                batch=int(b0),
                debug=bool(args.debug),
                debug_txt=str(gpu_debug_txt) if bool(args.debug) else None,
            )

        if not Path(results_json).is_file():
            raise FileNotFoundError(
                f"gpu_results_json not found: {results_json}\n"
                f"Hint: run with --run-gpu on a GPU node, or provide an existing --gpu-results-json."
            )

        case_meta = {
            "case_id": f"B{b0}_P{p0}_{decode_tag}",
            "case_prefix": case_prefix,
            "shape_json": str(shape_json_path),
            "sweep_batch": int(b0),
            "sweep_prefill_len": int(p0),
            "decode_tag": str(decode_tag),
            "decode_min": decode_min,
            "decode_max": decode_max,
            "decode_n": int(decode_n),
        }

        summary_rows, detail_rows = evaluate_one_case(
            cm_mod=cm_mod,
            cm_obj=cm_obj,
            dev=dev,
            tasks_json=Path(tasks_json),
            results_json=Path(results_json),
            decode_context_lens_fallback=decode_context_lens,
            decode_stride_fallback=int(args.decode_stride),
            debug=bool(args.debug),
            debug_print_n=int(args.debug_print_n),
            case_meta=case_meta,
        )
        _save_rows_csv(summary_rows, out_summary, sort_by=["op", "phase"])
        _save_rows_csv(detail_rows, out_detail, sort_by=["op", "phase", "K_key_len", "T_query_len"])
        _save_rows_csv(summary_rows, out_sweep_summary, sort_by=["sweep_batch", "sweep_prefill_len", "op", "phase"])
        _save_rows_csv(detail_rows, out_sweep_detail, sort_by=["sweep_batch", "sweep_prefill_len", "op", "phase", "K_key_len", "T_query_len"])

        print("\n=== A100 single-op fit summary (single case) ===")
        try:
            import pandas as pd  # type: ignore
            print(pd.DataFrame(summary_rows).sort_values(["op", "phase"]).to_string(index=False))
        except Exception:
            pass
        print(f"\nSaved summary CSV: {out_summary}")
        print(f"Saved detail  CSV: {out_detail}")
        print(f"Saved sweep summary: {out_sweep_summary}")
        print(f"Saved sweep detail : {out_sweep_detail}")
        return

    # ----------------------------------------------------------------------------------
    # Sweep mode (batch x prefill_len)
    # ----------------------------------------------------------------------------------
    verify_script: Optional[str] = None
    if bool(args.run_gpu):
        verify_script = _resolve_verify_script(args.verify_script)
        print(f"[eval] verify script: {verify_script}")

    all_summary: List[Dict[str, Any]] = []
    all_detail: List[Dict[str, Any]] = []

    for b in batches:
        for p in prefill_lens:
            case_id = f"B{int(b)}_P{int(p)}_{decode_tag}"
            case_prefix = f"{prefix}_{case_id}"
            tasks_json = out_dir / f"{case_prefix}.gpu_tasks.json"
            results_json = out_dir / f"{case_prefix}.gpu_results.json"
            out_summary = out_dir / f"{case_prefix}.summary.csv"
            out_detail = out_dir / f"{case_prefix}.detail.csv"
            gpu_debug_txt = out_dir / f"{case_prefix}.gpu_debug.txt"

            build_single_op_tasks_json(
                out_path=tasks_json,
                prefix=case_prefix,
                dim=int(D),
                ffn_dim=int(F),
                n_heads=int(H),
                n_kv_heads=int(Hkv),
                batch=int(b),
                prefill_len=int(p),
                decode_context_lens=decode_context_lens,
                decode_stride=int(args.decode_stride),
                phases=phases,
                ops=ops,
                device=str(args.gpu_device),
                gpu_dtype=str(args.gpu_dtype),
            )
            print(f"[eval] wrote tasks json: {tasks_json}")

            if bool(args.run_gpu):
                assert verify_script is not None
                run_gpu_benchmark(
                    verify_script=verify_script,
                    tasks_json=str(tasks_json),
                    out_json=str(results_json),
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    device=str(args.gpu_device) if args.gpu_device else None,
                    gpu_dtype=str(args.gpu_dtype) if args.gpu_dtype else None,
                    batch=int(b),
                    debug=bool(args.debug),
                    debug_txt=str(gpu_debug_txt) if bool(args.debug) else None,
                )

            if not results_json.is_file():
                raise FileNotFoundError(
                    f"gpu_results_json not found: {results_json}\n"
                    f"Hint: run with --run-gpu on a GPU node, or provide an existing --gpu-results-json."
                )

            case_meta = {
                "case_id": str(case_id),
                "case_prefix": str(case_prefix),
                "shape_json": str(shape_json_path),
                "sweep_batch": int(b),
                "sweep_prefill_len": int(p),
                "decode_tag": str(decode_tag),
                "decode_min": decode_min,
                "decode_max": decode_max,
                "decode_n": int(decode_n),
            }

            summary_rows, detail_rows = evaluate_one_case(
                cm_mod=cm_mod,
                cm_obj=cm_obj,
                dev=dev,
                tasks_json=tasks_json,
                results_json=results_json,
                decode_context_lens_fallback=decode_context_lens,
                decode_stride_fallback=int(args.decode_stride),
                debug=bool(args.debug),
                debug_print_n=int(args.debug_print_n),
                case_meta=case_meta,
            )

            _save_rows_csv(summary_rows, out_summary, sort_by=["op", "phase"])
            _save_rows_csv(detail_rows, out_detail, sort_by=["op", "phase", "K_key_len", "T_query_len"])

            all_summary.extend(summary_rows)
            all_detail.extend(detail_rows)

    # Aggregated
    _save_rows_csv(all_summary, out_sweep_summary, sort_by=["sweep_batch", "sweep_prefill_len", "op", "phase"])
    _save_rows_csv(all_detail, out_sweep_detail, sort_by=["sweep_batch", "sweep_prefill_len", "op", "phase", "K_key_len", "T_query_len"])

    print("\n=== A100 single-op fit sweep summary (grouped by case/op/phase) ===")
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(all_summary)
        cols = ["case_id", "op", "phase", "n", "mae_us", "mape_pct", "mean_err_us", "r2"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].sort_values(["case_id", "op", "phase"]).to_string(index=False))
    except Exception:
        pass

    print(f"\nSaved sweep summary CSV: {out_sweep_summary}")
    print(f"Saved sweep detail  CSV: {out_sweep_detail}")


if __name__ == "__main__":
    main()
