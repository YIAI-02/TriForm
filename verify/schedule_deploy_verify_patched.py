#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

1) export
   - 输入：一个或多个 schedule CSV
   - 输出：
     * <prefix>.gpu_tasks.json
     * <prefix>.pim_tasks.json

python ./verify/schedule_deploy_verify.py export \
  --schedule  ./algorithms/output/evaluate_single_test/hardware_1gpu_4aim/llama_7b_int8_b1_s128/algo_pd/pd_prefill-8192xdecode_128_ops_trace.csv \
  --comms     ./algorithms/output/evaluate_single_test/hardware_1gpu_4aim/llama_7b_int8_b1_s128/algo_pd/pd_prefill-8192xdecode_128_comms_trace.csv \
  --out-dir   ./verify/out \
  --prefix    pd_seg \
  --segment-scope layer \
  --prefill-len 8192 \
  --decode-stride 128 \
  --cfg ./configs/llama_7b_shape.json

python ./verify/schedule_deploy_verify.py export \
  --schedule  ./algorithms/output/evaluate_single_test/hardware_1gpu_4aim/llama_7b_int8_b1_s128/algo_hefthint/hefthint_prefill-8192xdecode_128_ops_trace.csv \
  --comms     ./algorithms/output/evaluate_single_test/hardware_1gpu_4aim/llama_7b_int8_b1_s128/algo_hefthint/hefthint_prefill-8192xdecode_128_comms_trace.csv \
  --out-dir   ./verify/out \
  --prefix    hefthint_seg \
  --segment-scope layer \
  --prefill-len 8192 \
  --decode-stride 128 \
  --cfg ./configs/llama_7b_shape.json

2) run-gpu
   - 读取 gpu_tasks.json，把每个 segment 的算子序列连续执行并计时
   - 输出：gpu_results.json （key -> sec_per_call）
   python ./verify/schedule_deploy_verify.py run-gpu \
  --tasks ./verify/out/hefthint_seg.gpu_tasks.json \
  --out ./verify/out/hefthint_seg.gpu_results.json \
  --warmup 3 --iters 10


3) run-pim
   - 读取 pim_tasks.json，调用 aim_sim.PIM 的 only_trace primitive，
     把 segment 内算子连续仿真并计时
   - 输出：pim_results.json （key -> sec_per_call）
  python ./verify/schedule_deploy_verify.py run-pim \
  --tasks ./verify/out/hefthint_seg.pim_tasks.json \
  --out   ./verify/out/hefthint_seg.pim_results.json \
  --pim-ramulator-config ./algorithms/aim_simulator/example.yaml \
  --pim-hw-json          ./algorithms/aim_simulator/PIM_AiM.json \
  --pim-ramulator-bin ./algorithms/ramulator2\
  --pim-num-devices 4

4) merge
   - 输入：schedule CSV + gpu_results.json + pim_results.json
   - 重新按同样规则切 segment，统计每个 segment 出现次数，做求和/聚合，给出总 latency + speedup 对比。
   python ./verify/schedule_deploy_verify.py merge \
  --schedule ./algorithms/output/evaluate_single_test/hardware_1gpu_4aim/llama_7b_int8_b1_s128/algo_hefthint/hefthint_prefill-8192xdecode_128_ops_trace.csv \
  --gpu-results ./verify/out/hefthint_seg.gpu_results.json \
  --pim-results ./verify/out/hefthint_seg.pim_results.json \
  --comm-model schedule \
  --decode-stride 128 \
  --out-csv ./verify/out/merge_report_hefthint.csv
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any
import os
import re
import json
import sys
import time
import hashlib
import subprocess
import tempfile
import shutil
import uuid
import yaml
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

_COMM_OPS = {
    # KV cache writebacks
    "k_write",
    "v_write",
    "kv_write",

    # Collectives / comm
    "allreduce",
    "all_reduce",
    "reduce",
    "scatter",
    "reducescatter",
    "reduce_scatter",
    "allgather",
    "all_gather",
    "broadcast",
    "alltoall",
    "all_to_all",

    # Point-to-point / misc
    "send",
    "recv",
    "identity",
}


# ==========================================================
# GPU env / spec report helpers
# ==========================================================
def _try_run(cmd: List[str]) -> Tuple[bool, str]:
    """Run a command and return (ok, stdout_or_error)."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return True, out.strip()
    except Exception as e:
        return False, str(e)

def collect_gpu_info(device_str: str) -> Dict[str, Any]:
    """Collect GPU model/spec info for the device this script will use.

    The returned dict is JSON-serializable.
    """
    info: Dict[str, Any] = {
        "requested_device": str(device_str),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    if torch is None:
        info["torch_available"] = False
        return info

    info["torch_available"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    info["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    # ROCm/HIP environments (best effort)
    info["torch_hip_version"] = getattr(getattr(torch, "version", None), "hip", None)

    # cuDNN
    try:
        info["cudnn_available"] = bool(torch.backends.cudnn.is_available())
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        info["cudnn_available"] = None
        info["cudnn_version"] = None

    # CUDA device properties via torch
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["cuda_available"] = False

    if info.get("cuda_available"):
        try:
            info["cuda_device_count"] = int(torch.cuda.device_count())
        except Exception:
            info["cuda_device_count"] = None

    try:
        dev = torch.device(device_str)
    except Exception as e:
        info["torch_device_parse_error"] = str(e)
        dev = None

    if dev is not None and dev.type == "cuda" and info.get("cuda_available"):
        try:
            idx = dev.index
            if idx is None:
                idx = int(torch.cuda.current_device())
            info["device_index"] = int(idx)
            props = torch.cuda.get_device_properties(int(idx))

            info["name"] = getattr(props, "name", None)
            info["multi_processor_count"] = int(getattr(props, "multi_processor_count", -1))
            info["compute_capability"] = f"{int(getattr(props, 'major', -1))}.{int(getattr(props, 'minor', -1))}"

            # Best-effort extra specs (present in most PyTorch builds)
            for attr, key in [
                ("clock_rate", "clock_rate_khz"),
                ("memory_clock_rate", "memory_clock_rate_khz"),
                ("memory_bus_width", "memory_bus_width_bits"),
                ("l2_cache_size", "l2_cache_size_bytes"),
                ("warp_size", "warp_size"),
                ("max_threads_per_block", "max_threads_per_block"),
                ("max_threads_per_multi_processor", "max_threads_per_multiprocessor"),
                ("shared_memory_per_block", "shared_memory_per_block_bytes"),
            ]:
                if hasattr(props, attr):
                    try:
                        info[key] = int(getattr(props, attr))
                    except Exception:
                        pass

        except Exception as e:
            info["torch_cuda_props_error"] = str(e)

    # Best-effort NVIDIA driver info (if nvidia-smi exists)
    smi_path = shutil.which("nvidia-smi")
    if smi_path:
        ok, out = _try_run([smi_path, "-L"])
        if ok:
            info["nvidia_smi_L"] = out
        else:
            info["nvidia_smi_L_error"] = out

        # Minimal, generally-supported query fields
        ok, out = _try_run([
            smi_path,
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ])
        if ok:
            info["nvidia_smi_query"] = out
        else:
            info["nvidia_smi_query_error"] = out
    else:
        info["nvidia_smi"] = "not_found"

    return info


def print_gpu_info(info: Dict[str, Any], *, print_fn=print) -> None:
    """Pretty-print GPU info for logs."""
    dev = info.get("requested_device")
    if not info.get("torch_available", False):
        print_fn(f"[run-gpu] torch not available; cannot query GPU properties. requested_device={dev}")
        return

    if dev and str(dev).startswith("cuda") and info.get("cuda_available"):
        name = info.get("name") or "<unknown>"
        mem = info.get("total_memory_gib")
        cc = info.get("compute_capability")
        sm = info.get("multi_processor_count")
        idx = info.get("device_index")
        extra = []
        if mem is not None:
            extra.append(f"{mem:.2f} GiB")
        if sm is not None and sm != -1:
            extra.append(f"SM={sm}")
        if cc:
            extra.append(f"CC={cc}")
        extra_s = ("; " + ", ".join(extra)) if extra else ""

        print_fn(f"[run-gpu] GPU[{idx}] {name}{extra_s}")
        if info.get("nvidia_smi_L"):
            print_fn(f"[run-gpu] nvidia-smi -L: {info['nvidia_smi_L']}")
        if info.get("nvidia_smi_query"):
            print_fn(f"[run-gpu] nvidia-smi query: {info['nvidia_smi_query']}")
    else:
        print_fn(
            f"[run-gpu] running on non-CUDA device={dev} (cuda_available={info.get('cuda_available')})."
        )

# cxl model is optional
try:
    import cxl_latency  # type: ignore
except Exception:
    cxl_latency = None  # type: ignore
# ==========================================================
# Path helpers (workdir-robust)
# ==========================================================

def _script_dir() -> Path:
    return Path(__file__).resolve().parent

def _add_sys_path(p: Path) -> None:
    p = p.resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

def resolve_existing_path(path_str: str, *, extra_roots: Optional[List[Path]] = None) -> Path:
    p = Path(path_str).expanduser()
    if p.exists():
        return p.resolve()

    cand = (Path.cwd() / p)
    if cand.exists():
        return cand.resolve()

    cand = (_script_dir().parent / p)
    if cand.exists():
        return cand.resolve()

    if extra_roots:
        for r in extra_roots:
            cand = (r / p)
            if cand.exists():
                return cand.resolve()

    raise FileNotFoundError(f"Path not found: {path_str}")


# ==========================================================
# CENT / aim_sim
# ==========================================================

def find_cent_sim_root(start: Optional[Path] = None) -> Optional[Path]:
    here = (start or _script_dir()).resolve()
    for p in [here] + list(here.parents):
        cand = p / "submodules" / "CENT" / "cent_simulation"
        if cand.exists():
            return cand.resolve()
    return None

def import_aim_transformer_block(cent_sim_root: Optional[str] = None):
    """Import TransformerBlockLlama (preferred) from CENT/AiM simulator.
    """
    root = None
    if cent_sim_root:
        root = Path(cent_sim_root).expanduser().resolve()
    elif os.environ.get("CENT_SIM_ROOT"):
        root = Path(os.environ["CENT_SIM_ROOT"]).expanduser().resolve()
    else:
        root = find_cent_sim_root()

    if root is None or not root.exists():
        raise RuntimeError(
            "Cannot locate cent_simulation. Please pass --cent-sim-root "
        )
    _add_sys_path(root)

    import importlib.util
    prev_utils = sys.modules.get("utils", None)
    cent_utils_path = (Path(root) / "utils.py").resolve()
    cent_utils_mod = None

    if cent_utils_path.exists():
        spec = importlib.util.spec_from_file_location("_cent_utils_for_aim", str(cent_utils_path))
        if spec and spec.loader:
            cent_utils_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cent_utils_mod)  # type: ignore[attr-defined]
            sys.modules["utils"] = cent_utils_mod

    try:
        # Prefer Llama wrapper (has memory_mapping()) but accept different casings.
        for mod_name in ("Llama", "llama"):
            try:
                mod = __import__(mod_name, fromlist=["TransformerBlockLlama"])
                if hasattr(mod, "TransformerBlockLlama"):
                    return getattr(mod, "TransformerBlockLlama")
            except Exception:
                pass

        # Fallback to raw TransformerBlock (also accept different casings)
        for mod_name in ("TransformerBlock", "transformerblock"):
            try:
                mod = __import__(mod_name, fromlist=["TransformerBlock"])
                if hasattr(mod, "TransformerBlock"):
                    return getattr(mod, "TransformerBlock")
            except Exception:
                pass

        raise RuntimeError(f"Neither TransformerBlockLlama nor TransformerBlock could be imported from {root}")
    finally:
        if prev_utils is not None:
            sys.modules["utils"] = prev_utils
        else:
            if "utils" in sys.modules and sys.modules["utils"] is cent_utils_mod:
                try:
                    del sys.modules["utils"]
                except Exception:
                    pass


def parse_layer(node_id: str) -> int:
    m = re.match(r"L(\d+)_", str(node_id))
    return int(m.group(1)) if m else -1

def parse_shard(node_id: str) -> int:
    m = re.search(r"_(?:s|shard)_?(\d+)", str(node_id), flags=re.IGNORECASE)
    return int(m.group(1)) if m else -1

def strip_shard_suffix(node_id: str) -> str:
    """Strip trailing shard suffix from node_id.

    Example: 'L0_Q_s3' -> 'L0_Q'.
    """
    s = str(node_id)
    return re.sub(r"_(?:s|shard)_?\d+$", "", s, flags=re.IGNORECASE)


def parse_prefill_decode_len_from_path(path: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        base = os.path.basename(str(path))
    except Exception:
        base = str(path)

    prefill_len: Optional[int] = None
    decode_len: Optional[int] = None

    m = re.search(r"prefill[_-]?(\d+)", base, flags=re.IGNORECASE)
    if m:
        try:
            prefill_len = int(m.group(1))
        except Exception:
            prefill_len = None

    # Most common: decode_1024 / decode-1024
    m = re.search(r"decode[_-]?(\d+)", base, flags=re.IGNORECASE)
    if m:
        try:
            decode_len = int(m.group(1))
        except Exception:
            decode_len = None

    # Fallback: prefill-1024xdecode_1024 style (prefill sometimes not matched above)
    if prefill_len is None:
        m = re.search(r"prefill[_-]?(\d+)\s*[xX]\s*decode", base, flags=re.IGNORECASE)
        if m:
            try:
                prefill_len = int(m.group(1))
            except Exception:
                prefill_len = None

    return prefill_len, decode_len


def decode_token_index_from_sample_step(step_sample: int, stride: int) -> int:
    s = int(step_sample)
    st = int(stride) if stride is not None else 1
    if st <= 1:
        # stride==1: no sampling, each step is exactly one token.
        return s
    if s <= 0:
        return 0
    if s == 1:
        return 1
    return int((s - 1) * st - 1)

def compute_decode_step_scale_map(
    schedule_df: pd.DataFrame,
    schedule_path: Optional[str],
    decode_stride: int,
) -> Tuple[Dict[int, int], str, Optional[int]]:
    st = int(decode_stride) if decode_stride is not None else 1
    if st < 1:
        st = 1

    if schedule_df is None or schedule_df.empty or "phase" not in schedule_df.columns:
        return ({}, "no_decode", None)

    df = schedule_df
    if "step" not in df.columns:
        try:
            df = add_step_column(df)
        except Exception:
            return ({}, "unknown", None)

    dec = df[df["phase"].astype(str) == "decode"]
    if dec.empty:
        return ({}, "no_decode", None)

    dec_steps = (
        pd.to_numeric(dec["step"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .unique()
        .tolist()
    )
    dec_steps = sorted([int(s) for s in dec_steps if int(s) >= 0])

    # stride==1 -> no expansion
    if st == 1:
        return ({int(s): 1 for s in dec_steps}, "stride1", None)
    decode_len: Optional[int] = None
    try:
        _, decode_len = parse_prefill_decode_len_from_path(schedule_path or "")
    except Exception:
        decode_len = None
    if decode_len is None:
        raise ValueError(
            "decode_len must be parsed from schedule filename (e.g., '*decode_1024*'). "
            f"Got schedule_path={schedule_path!r}."
        )
    decode_len_i = int(decode_len)
    if decode_len_i <= 0:
        raise ValueError(f"decode_len must be > 0 (got {decode_len_i}) from {schedule_path!r}")

    # Latest sampling/scaling policy (always):
    #   step 0 -> token 0, scale 1
    #   step 1 -> token 1, scale (stride-1)
    #   step >=2 -> sample at token (k*stride-1), scale stride (last block clipped)
    mode = "token0_token1"
    scale_by_step: Dict[int, int] = {}

    # Optional sanity check: warn if sample count doesn't match the expected pattern.
    try:
        if decode_len_i > 1 and st > 1:
            n_blocks = int((decode_len_i + st - 1) // st)
            expected = int(n_blocks + 1)
            if len(dec_steps) != expected or (dec_steps and dec_steps[0] != 0) or (len(dec_steps) > 1 and dec_steps[1] != 1):
                print(
                    f"[warn] decode sampling steps mismatch for {os.path.basename(str(schedule_path or ''))}: "
                    f"decode_len={decode_len_i} stride={st} expected_steps={expected} got_steps={len(dec_steps)} "
                    f"(first_steps={dec_steps[:4]})",
                    file=sys.stderr,
                )
    except Exception:
        pass

    for s in dec_steps:
        si = int(s)
        if si <= 0:
            scale_by_step[si] = 1
            continue
        if si == 1:
            # token 1 represents the remaining tokens in the first block.
            scale_by_step[si] = max(0, min(int(st - 1), int(decode_len_i - 1)))
            continue
        # step>=2: block index b = step-1, represents tokens [b*stride .. (b+1)*stride-1].
        b = int(si - 1)
        block_start = int(b * st)
        remaining = int(decode_len_i - block_start)
        scale_by_step[si] = max(0, min(int(st), remaining))

    return scale_by_step, mode, decode_len_i

def infer_decode_steps(df: pd.DataFrame) -> int:
    #TODO
    # infer decode step from the lines of op.csv "
    d = df[df["phase"] == "decode"]
    if len(d) == 0:
        return 0
    nodes_per_step = d["node_id"].nunique()
    if nodes_per_step == 0:
        return 0
    # In the provided trace format, decode contains repeated blocks of same node_id set.
    if len(d) % nodes_per_step != 0:
        # fallback: treat as 1 step
        return 1
    return len(d) // nodes_per_step

def add_step_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'step' column: prefill=-1, decode=0..(steps-1)."""
    df = df.copy()
    df["step"] = -1
    dmask = df["phase"] == "decode"
    if dmask.any():
        d = df[dmask]
        nodes_per_step = d["node_id"].nunique()
        if nodes_per_step > 0:
            # assume rows grouped by token, one token = nodes_per_step rows
            df.loc[dmask, "step"] = (d.groupby("phase").cumcount() // nodes_per_step).astype(int).values
    return df


def normalize_device_type(v: Any) -> str:
    s = str(v).strip().lower()
    # Common aliases
    if s in ("gpu", "cuda"):
        return "npu"
    if s in ("aim",):
        return "pim"
    if s in ("host", "x86", "arm"):
        return "cpu"
    return s

def load_schedule_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"phase","node_id","op","device","device_type","start","end","duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"schedule csv missing columns: {sorted(missing)}")
    df = df.copy()
    df["_row"] = range(len(df))
    df["device_type"] = df["device_type"].apply(normalize_device_type)
    # best-effort numeric conversion (avoid string dtype surprises later)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = add_step_column(df)
    df["layer"] = df["node_id"].apply(parse_layer)
    df["shard"] = df["node_id"].apply(parse_shard)
    return df


# ==========================================================
# Comms trace (optional) helpers
# ==========================================================
def load_comms_csv(path: str) -> pd.DataFrame:
    """Load a comms trace CSV."""
    df = pd.read_csv(path)
    # best-effort normalization: keep original columns, only ensure duration is numeric
    if "duration" in df.columns:
        df = df.copy()
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    return df


def extract_weight_load_seconds(
    comms_paths: List[str],
    *,
    decode_stride: int = 1,
) -> Dict[str, float]:
    """Aggregate `weight_load` durations from comms trace(s)."""
    gpu_s = 0.0
    pim_s = 0.0
    unknown_s = 0.0

    if not comms_paths:
        return {"gpu_s": 0.0, "pim_s": 0.0, "total_s": 0.0, "unknown_s": 0.0}

    rows: List[pd.DataFrame] = []

    def _norm_str(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").replace("nan", "", regex=False).str.strip()

    for cp in comms_paths:
        p = resolve_existing_path(cp)
        df = load_comms_csv(str(p))
        if df is None or df.empty:
            continue
        if "tag" not in df.columns or "duration" not in df.columns:
            continue

        tag_l = df["tag"].astype(str).str.strip().str.lower()
        wl = df[tag_l == "weight_load"].copy()
        if wl.empty:
            continue

        wl["duration"] = pd.to_numeric(wl["duration"], errors="coerce").fillna(0.0).astype(float)

        # destination-based classification
        dst = _norm_str(wl["dst"]) if "dst" in wl.columns else pd.Series([""] * len(wl), index=wl.index)
        dst_type = _norm_str(wl["dst_type"]) if "dst_type" in wl.columns else pd.Series([""] * len(wl), index=wl.index)

        dst_u = dst.astype(str).str.upper().str.strip()
        dst_norm = dst_u.str.replace("_", "", regex=False)
        dst_norm = dst_norm.str.replace(r"^NPU", "GPU", regex=True)

        dst_type_l = dst_type.astype(str).str.strip().str.lower()

        pim_mask = dst_norm.str.startswith("PIM") | dst_type_l.str.contains("pim")
        gpu_mask = (
            dst_norm.str.startswith("GPU")
            | dst_norm.str.startswith("NPU")
            | dst_type_l.str.contains("gpu")
            | dst_type_l.str.contains("npu")
            | dst_type_l.isin(["other"])
        )
        gpu_mask = gpu_mask & (~pim_mask)

        cls = np.where(pim_mask.to_numpy(), "pim", np.where(gpu_mask.to_numpy(), "gpu", "unknown"))
        wkey = pd.Series([""] * len(wl), index=wl.index)
        if "node_id" in wl.columns:
            nid = _norm_str(wl["node_id"])
            wkey = wkey.where(wkey != "", nid)
        if "weight_id" in wl.columns:
            wid = _norm_str(wl["weight_id"])
            wkey = wkey.where(wkey != "", wid)
        if "op" in wl.columns:
            op = _norm_str(wl["op"])
            wkey = wkey.where(wkey != "", op)
        # last resort: stable per-row id inside this comms trace
        wkey = wkey.where(wkey != "", pd.Series([f"ROW{i}" for i in range(len(wl))], index=wl.index))

        rows.append(pd.DataFrame({
            "class": cls,
            "dst": dst_norm.to_numpy(),
            "wkey": wkey.to_numpy(),
            "duration": wl["duration"].to_numpy(dtype=np.float64),
        }))

    if not rows:
        return {"gpu_s": 0.0, "pim_s": 0.0, "total_s": 0.0, "unknown_s": 0.0}

    all_wl = pd.concat(rows, ignore_index=True)

    uniq = (
        all_wl.groupby(["class", "dst", "wkey"], dropna=False)["duration"]
        .max()
        .reset_index()
    )
    per_dst = (
        uniq.groupby(["class", "dst"], dropna=False)["duration"]
        .sum()
        .reset_index()
    )

    pim_by = per_dst[per_dst["class"] == "pim"].set_index("dst")["duration"].to_dict()
    gpu_by = per_dst[per_dst["class"] == "gpu"].set_index("dst")["duration"].to_dict()
    unk_by = per_dst[per_dst["class"] == "unknown"].set_index("dst")["duration"].to_dict()

    pim_part = max(pim_by.values()) if pim_by else 0.0
    gpu_part = max(gpu_by.values()) if gpu_by else 0.0
    unk_part = sum(unk_by.values()) if unk_by else 0.0

    pim_s = float(pim_part)
    gpu_s = float(gpu_part + unk_part)
    unknown_s = float(unk_part)

    total_s = max(float(gpu_s), float(pim_s))
    return {"gpu_s": float(gpu_s), "pim_s": float(pim_s), "total_s": total_s, "unknown_s": float(unknown_s)}


# ==========================================================
# Layer-scope comm accumulation (kv_load)
# ==========================================================

def extract_layer_comm_seconds(
    comms_paths: List[str],
    schedule_df: pd.DataFrame,
    *,
    tags: Tuple[str, ...] = ("kv_load",),
) -> pd.DataFrame:
    """Extract per-(phase, step, layer) comm seconds for selected comm tags.
    Returned columns:
      phase, step, layer, comm_s, weight_load_s, kv_load_s
    """

    if not comms_paths:
        return pd.DataFrame(columns=["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"])

    # Load + concat
    parts = []
    for p in comms_paths:
        if not p:
            continue
        p = resolve_existing_path(p)
        if p is None:
            continue
        try:
            c = load_comms_csv(str(p))
        except Exception:
            continue
        if c is None or c.empty:
            continue
        parts.append(c)

    if not parts:
        return pd.DataFrame(columns=["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"])

    df = pd.concat(parts, ignore_index=True)

    if "tag" not in df.columns or "duration" not in df.columns:
        return pd.DataFrame(columns=["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"])

    df = df.copy()
    df["tag_l"] = (
        df["tag"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    tags_l = [str(t).strip().lower().replace(" ", "_") for t in tags]
    df = df[df["tag_l"].isin(tags_l)].copy()
    if df.empty:
        return pd.DataFrame(columns=["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"])

    # Phase
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype(str).str.strip()
    else:
        df["phase"] = "prefill"

    # Duration
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).astype(float)

    # Layer
    if "layer" in df.columns:
        df["layer"] = pd.to_numeric(df["layer"], errors="coerce").fillna(-1).astype(int)
    else:
        src_col = None
        for c in ("node_id", "prod_node", "cons_node"):
            if c in df.columns:
                src_col = c
                break
        if src_col is None:
            df["layer"] = -1
        else:
            df["layer"] = df[src_col].apply(parse_layer).astype(int)

    # Step
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(-1).astype(int)
    else:
        # Build decode step start times from schedule
        d = schedule_df
        if d is None or d.empty or "phase" not in d.columns or "step" not in d.columns:
            df["step"] = -1
        else:
            dph = d["phase"].astype(str)
            ddec = d[dph == "decode"]
            if ddec.empty:
                df["step"] = -1
            else:
                bounds = (
                    ddec.groupby("step", as_index=True)
                    .agg(start_min=("start", "min"), end_max=("end", "max"))
                    .sort_values("start_min")
                )
                step_ids = bounds.index.to_numpy(dtype=int)
                step_starts = bounds["start_min"].to_numpy(dtype=float)

                # Use comm start time; fall back to end
                st = pd.to_numeric(df.get("start"), errors="coerce")
                en = pd.to_numeric(df.get("end"), errors="coerce")
                t = st.fillna(en)

                # Default -1 for non-decode
                df["step"] = -1
                mask_dec = df["phase"].astype(str) == "decode"
                if mask_dec.any() and step_starts.size > 0:
                    tt = t[mask_dec].to_numpy(dtype=float)
                    import numpy as np
                    idx = np.searchsorted(step_starts, tt, side="right") - 1
                    idx[idx < 0] = 0
                    idx[idx >= len(step_ids)] = len(step_ids) - 1
                    df.loc[mask_dec, "step"] = step_ids[idx]

    # Aggregate
    g = (
        df.groupby(["phase", "step", "layer", "tag_l"], as_index=False)["duration"]
        .sum()
        .rename(columns={"duration": "comm_s"})
    )

    wide = (
        g.pivot_table(index=["phase", "step", "layer"], columns="tag_l", values="comm_s", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )

    # Ensure columns
    if "weight_load" not in wide.columns:
        wide["weight_load"] = 0.0
    if "kv_load" not in wide.columns:
        wide["kv_load"] = 0.0

    wide = wide.rename(columns={"weight_load": "weight_load_s", "kv_load": "kv_load_s"})
    wide["comm_s"] = wide["weight_load_s"].astype(float) + wide["kv_load_s"].astype(float)

    # Order
    cols = ["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"]
    for c in cols:
        if c not in wide.columns:
            wide[c] = 0
    wide = wide[cols]

    return wide


def estimate_kv_load_seconds_from_ops_csv(
    schedule_df: pd.DataFrame,
    cfg: WorkloadConfig,
    *,
    decode_stride: int,
) -> Dict[Tuple[str, int, int], float]:
    """Estimate kv_load time per (phase, step, layer) by scanning QK/SV ops.

    Model (decode only):
        bytes(K or V) = batch * kv_heads_shard * head_dim * kv_len * kv_dtype_bytes
        time_s        = bytes / (kv_load_bw_gbs * 1e9) + kv_load_overhead_us * 1e-6
    """

    if schedule_df is None or schedule_df.empty:
        return {}

    bw = float(getattr(cfg, "kv_load_bw_gbs", 0.0) or 0.0)
    if bw <= 0.0:
        return {}

    kv_dtype_bytes = float(getattr(cfg, "kv_dtype_bytes", 2.0) or 2.0)
    overhead_s = float(getattr(cfg, "kv_load_overhead_us", 0.0) or 0.0) * 1e-6
    batch = int(getattr(cfg, "batch", 1) or 1)
    n_kv_heads = getattr(cfg, "n_kv_heads", None)
    if n_kv_heads is None:
        n_kv_heads = int(getattr(cfg, "n_heads", 0) or 0)
    else:
        n_kv_heads = int(n_kv_heads)
    dim = int(getattr(cfg, "dim", 0) or 0)
    q_heads = int(getattr(cfg, "n_heads", 0) or 0)
    head_dim = max(1, int(dim // max(1, q_heads)))
    shards = int(getattr(cfg, "shards", 1) or 1)
    shards = max(1, shards)

    def _kv_heads_for_shard(shard_id: int) -> int:
        """Distribute KV heads across shards (supports non-divisible cases)."""
        if shard_id < 0:
            return int(n_kv_heads)
        base = int(n_kv_heads) // int(shards)
        rem = int(n_kv_heads) % int(shards)
        # First `rem` shards carry one extra head.
        return int(base + (1 if int(shard_id) < int(rem) else 0))

    # decode context length lookup
    dcl = getattr(cfg, "decode_context_lens", None)
    dcl_list = list(dcl) if isinstance(dcl, (list, tuple)) else None

    prefill_len = int(getattr(cfg, "prefill_len", 0) or 0)
    ds = int(decode_stride or getattr(cfg, "decode_stride", 1) or 1)
    if ds <= 0:
        ds = 1

    def _kv_len_for_step(step: int) -> int:
        if dcl_list is not None and 0 <= int(step) < len(dcl_list):
            try:
                return int(dcl_list[int(step)])
            except Exception:
                pass
        tok = int(decode_token_index_from_sample_step(int(step), int(ds)))
        return int(prefill_len + 1 + tok)

    # Filter QK/SV ops in decode.
    ph = schedule_df.get("phase")
    op = schedule_df.get("op")
    if ph is None or op is None:
        return {}

    ph_s = ph.astype(str).str.strip()
    op_u = op.astype(str).str.strip().str.upper()
    mask = (ph_s == "decode") & (op_u.isin(["QK", "SV"]))
    if not bool(mask.any()):
        return {}

    sub = schedule_df.loc[mask, ["phase", "step", "layer", "device", "shard", "op"]].copy()
    # best-effort numeric
    sub["step"] = pd.to_numeric(sub["step"], errors="coerce").fillna(-1).astype(int)
    sub["layer"] = pd.to_numeric(sub["layer"], errors="coerce").fillna(-1).astype(int)
    sub["shard"] = pd.to_numeric(sub["shard"], errors="coerce").fillna(-1).astype(int)

    per_dev: Dict[Tuple[str, int, int, str], float] = {}
    for r in sub.itertuples(index=False):
        phase = str(r.phase)
        step = int(r.step)
        layer = int(r.layer)
        device = str(r.device)
        shard_id = int(r.shard)

        if step < 0:
            # safety: skip malformed decode rows
            continue

        kv_len = _kv_len_for_step(step)
        kvh = _kv_heads_for_shard(shard_id)
        if kv_len <= 0 or kvh <= 0:
            continue

        elems = float(batch) * float(kvh) * float(head_dim) * float(kv_len)
        bytes_ = float(elems) * float(kv_dtype_bytes)
        t = float(bytes_ / (float(bw) * 1e9) + float(overhead_s))

        k = (phase, step, layer, device)
        per_dev[k] = float(per_dev.get(k, 0.0) + t)

    # collapse device dimension: layer-scope kv_load is the max across devices
    layer_map: Dict[Tuple[str, int, int], float] = {}
    for (phase, step, layer, _dev), t in per_dev.items():
        kk = (phase, int(step), int(layer))
        layer_map[kk] = max(float(layer_map.get(kk, 0.0)), float(t))

    return layer_map



# ==========================================================
# Workload config + shape inference (same as v2)
# ==========================================================

@dataclass
class WorkloadConfig:
    # model
    dim: int = 4096
    ffn_dim: int = 11008
    n_heads: int = 32
    shards: int = 4
    # schedule lengths
    prefill_len: int = 128
    decode_context_lens: Optional[List[int]] = None  # length = decode steps
    decode_stride: int = 1
    # gpu bench
    device: str = "cuda"
    gpu_dtype: str = "fp16"

    # KV-cache load modeling (verification-time)
    kv_load_bw_gbs: float = 0.0
    kv_dtype_bytes: float = 2.0
    kv_load_overhead_us: float = 0.0
    n_kv_heads: Optional[int] = None
    batch: int = 1

    # pim sim config (defaults roughly match aim_sim)
    pim_dram_column: int = 256
    pim_dram_row: int = 64
    pim_burst_length: int = 16
    pim_num_banks: int = 8
    pim_num_channels: int = 4
    pim_threads: int = 1
    pim_reuse_size: int = 32
    pim_num_devices: int = 1  # DIMM/device count (model-parallel)

    # AiM-Ramulator integration
    pim_ramulator_bin: Optional[str] = None
    pim_ramulator_config: Optional[str] = None  # required to enable ramulator mode
    pim_freq_ghz: float = 1.0
    pim_ramulator_timeout_s: int = 1500
    pim_keep_traces: bool = False
    pim_trace_dir: Optional[str] = None
    pim_hw_json_path: Optional[str] = None
    debug: bool = False

    # segmenting
    segment_scope: str = "layer"

    # shard placement granularity:
    shard_policy: str = "coarse_majority"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorkloadConfig":
        cfg = WorkloadConfig()
        for k,v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass(frozen=True)
class OpSig:
    """Internal operator signature for shape inference and execution."""
    device_type: str
    phase: str
    step: int
    op: str
    shard: int

@dataclass(frozen=True)
class OpShape:
    dim: int
    shard_dim: int
    ffn_shard_dim: int
    query_len: int
    key_len: int
    heads_per_shard: int
    head_dim: int

def infer_op_shape(sig: OpSig, cfg: WorkloadConfig) -> OpShape:
    D = int(cfg.dim)
    Sd = int(D // max(1, cfg.shards))
    Fd = int(cfg.ffn_dim)
    Fsd = int(Fd // max(1, cfg.shards))
    Hd = int(D // max(1, cfg.n_heads))
    H = int(cfg.n_heads // max(1, cfg.shards))

    if sig.phase == "prefill":
        T = int(cfg.prefill_len)
        K = int(cfg.prefill_len)
    else:
        if cfg.decode_context_lens is None:
            stride = int(getattr(cfg, "decode_stride", 1) or 1)
            if stride <= 0:
                stride = 1
            tok = int(decode_token_index_from_sample_step(int(sig.step), int(stride)))
            K = int(cfg.prefill_len + 1 + tok)
        else:
            if sig.step < 0 or sig.step >= len(cfg.decode_context_lens):
                if cfg.decode_context_lens:
                    K = int(cfg.decode_context_lens[-1])
                else:
                    stride = int(getattr(cfg, "decode_stride", 1) or 1)
                    if stride <= 0:
                        stride = 1
                    tok = int(decode_token_index_from_sample_step(int(sig.step), int(stride)))
                    K = int(cfg.prefill_len + 1 + tok)
            else:
                K = int(cfg.decode_context_lens[sig.step])
        T = 1

    return OpShape(
        dim=D,
        shard_dim=Sd,
        ffn_shard_dim=Fsd,
        query_len=T,
        key_len=K,
        heads_per_shard=H,
        head_dim=Hd,
    )


# ==========================================================
# Segment signature + extraction
# ==========================================================

@dataclass(frozen=True)
class SegmentSig:
    """
    Segment signature:
    - device_type: 'npu' or 'pim' (measured backends).
      Other device types (e.g. 'cpu') are treated as trace-only: we skip exporting
      them as tasks and keep their schedule durations directly in merge().
    - phase: 'prefill' or 'decode'
    - step: prefill=-1, decode step id
    - ops: ordered list of (op, shard) inside the segment
    """
    device_type: str
    phase: str
    step: int
    ops: Tuple[Tuple[str,int], ...]

    def ops_repr(self) -> str:
        return ",".join([f"{op}:{shard}" for op,shard in self.ops])

    def to_key(self) -> str:
        # Key is stable and short, but collision-resistant
        h = hashlib.md5(self.ops_repr().encode("utf-8")).hexdigest()
        return f"{self.device_type}|{self.phase}|{self.step}|{h}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type,
            "phase": self.phase,
            "step": int(self.step),
            "ops": [{"op": op, "shard": int(shard)} for op, shard in self.ops],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SegmentSig":
        ops = tuple((str(x["op"]), int(x.get("shard", -1))) for x in d["ops"])
        return SegmentSig(device_type=str(d["device_type"]), phase=str(d["phase"]), step=int(d["step"]), ops=ops)

def extract_segments(df: pd.DataFrame, segment_scope: str) -> Tuple[Dict[str, SegmentSig], Counter]:
    """Extract unique segments from a schedule.

    We always segment by **layer**: one segment per (phase, step, layer, device).

    Notes:
      - Ops in `_COMM_OPS` are treated as communication/synchronization and are **not** exported
        as compute segments (GPU/PIM benchmarking skips them).
      - Device-scope segmentation (splitting a device timeline by comm ops) has been removed.

    Returns:
      - uniq: dict key -> SegmentSig
      - ctr:  Counter over keys (each occurrence corresponds to one (phase, step, layer, device) group)
    """

    scope = str(segment_scope).lower().strip()
    if scope != "layer":
        raise ValueError(
            f"Only segment_scope='layer' is supported (got {segment_scope!r}). "
            "Device-scope segmentation has been removed."
        )

    uniq: Dict[str, SegmentSig] = {}
    ctr: Counter = Counter()

    group_cols = ["phase", "step", "layer", "device"]
    for (phase, step, layer, device), g in df.groupby(group_cols, sort=False):
        g = g.sort_values(["start", "_row"], kind="mergesort")
        dev_type = str(g["device_type"].iloc[0])
        ops: List[Tuple[str, int]] = []
        for _, r in g.iterrows():
            op = str(r["op"]).strip()
            if op.lower() in _COMM_OPS:
                continue
            shard = int(r["shard"]) if pd.notna(r["shard"]) else -1
            ops.append((op, shard))

        if not ops:
            continue

        seg = SegmentSig(device_type=dev_type, phase=str(phase), step=int(step), ops=tuple(ops))
        k = seg.to_key()
        uniq[k] = seg
        ctr[k] += 1

    return uniq, ctr


def _iter_layer_segments_coarse_majority(
    df: pd.DataFrame,
    *,
    total_shards: int,
) -> Iterable[Tuple[str, int, int, str, Tuple[Tuple[str, int], ...]]]:
    """Yield coarse-grained compute segments for each (phase,step,layer).

    We only consider compute rows with device_type in {npu, pim}, and we ignore comm ops in _COMM_OPS.

    Placement rule for sharded ops (node_id like *_s0..*_sN):
      - if (# unique shards on GPU) > total_shards/2 -> put the whole op on GPU (npu)
      - else -> put the whole op on PIM (pim)

    """
    total_shards = int(total_shards)
    if total_shards <= 0:
        total_shards = 1

    # Restrict to compute on GPU/PIM and drop comm ops.
    comp = df[df["device_type"].astype(str).str.lower().isin(["npu", "pim"])].copy()
    if comp.empty:
        return
    op_l = comp["op"].astype(str).str.strip().str.lower()
    comp = comp[~op_l.isin(_COMM_OPS)].copy()
    if comp.empty:
        return

    comp["_base_node"] = comp["node_id"].apply(strip_shard_suffix)

    # group by layer occurrence
    for (phase, step, layer), g_layer in comp.groupby(["phase", "step", "layer"], sort=False):
        # group by op instance (base node id) to avoid merging LN vs LN2, Add1 vs Add2, etc.
        items: List[Tuple[float, int, str, str, bool]] = []
        # (t0, row0, op, chosen_dev, is_sharded)
        for base, g_op in g_layer.groupby("_base_node", sort=False):
            # ordering keys (stable)
            try:
                t0 = float(pd.to_numeric(g_op["start"], errors="coerce").min())
            except Exception:
                t0 = 0.0
            try:
                row0 = int(pd.to_numeric(g_op["_row"], errors="coerce").min())
            except Exception:
                row0 = 0

            op = str(g_op["op"].iloc[0]).strip()
            dev0 = str(g_op["device_type"].iloc[0]).strip().lower()

            sh_series = pd.to_numeric(g_op.get("shard"), errors="coerce").fillna(-1).astype(int)
            is_sharded = bool((sh_series >= 0).any())

            chosen = dev0
            if is_sharded:
                # Count unique shard indices scheduled on GPU (npu)
                gtmp = g_op.copy()
                gtmp["_shard_i"] = sh_series.values
                gpu_mask = gtmp["device_type"].astype(str).str.lower().eq("npu") & (gtmp["_shard_i"] >= 0)
                gpu_shards = set(gtmp.loc[gpu_mask, "_shard_i"].tolist())
                num_gpu = len(gpu_shards)

                if num_gpu > (float(total_shards) / 2.0):
                    chosen = "npu"
                else:
                    chosen = "pim"

            items.append((t0, row0, op, chosen, is_sharded))

        # stable order inside the layer
        items.sort(key=lambda x: (x[0], x[1]))

        gpu_ops: List[Tuple[str, int]] = []
        pim_ops: List[Tuple[str, int]] = []

        for _, _, op, chosen, is_sharded in items:
            if chosen == "npu":
                if is_sharded and total_shards > 1:
                    for s in range(total_shards):
                        gpu_ops.append((op, int(s)))
                else:
                    gpu_ops.append((op, -1))
            elif chosen == "pim":
                # In coarse mode, sharded ops are emitted once; PIMBackend handles model-parallel via FC_devices.
                pim_ops.append((op, -1))
            else:
                # not npu/pim -> ignore (trace-only in merge)
                pass

        if gpu_ops:
            yield (str(phase), int(step), int(layer), "npu", tuple(gpu_ops))
        if pim_ops:
            yield (str(phase), int(step), int(layer), "pim", tuple(pim_ops))


def extract_segments_coarse_majority(df: pd.DataFrame, segment_scope: str, *, total_shards: int) -> Tuple[Dict[str, SegmentSig], Counter]:
    """Extract unique segments using coarse-majority shard placement.

    Returns:
      - uniq: dict key -> SegmentSig
      - ctr:  Counter over keys (each occurrence corresponds to one (phase, step, layer, device_type) group)
    """
    scope = str(segment_scope).lower().strip()
    if scope != "layer":
        raise ValueError(
            f"Only segment_scope='layer' is supported (got {segment_scope!r}). "
            "Device-scope segmentation has been removed."
        )

    uniq: Dict[str, SegmentSig] = {}
    ctr: Counter = Counter()

    for phase, step, layer, dev, ops in _iter_layer_segments_coarse_majority(df, total_shards=total_shards):
        if not ops:
            continue
        seg = SegmentSig(device_type=str(dev), phase=str(phase), step=int(step), ops=tuple(ops))
        k = seg.to_key()
        uniq[k] = seg
        ctr[k] += 1

    return uniq, ctr


# ==========================================================
# GPU backend (segment-level benchmark)
# ==========================================================

class GPUBackend:
    def __init__(self, cfg: WorkloadConfig):
        if torch is None or F is None:
            raise RuntimeError("PyTorch not available; cannot run GPUBackend.")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cfg.device=cuda but torch.cuda.is_available() == False on this machine.")
        self.dtype = self._parse_dtype(cfg.gpu_dtype)
        self._buf: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._wbuf: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}

    @staticmethod
    def _parse_dtype(name: str):
        name = name.lower()
        if name == "fp16":
            return torch.float16
        if name == "bf16":
            return torch.bfloat16
        if name == "fp32":
            return torch.float32
        raise ValueError(f"unknown gpu_dtype: {name}")

    def _rand(self, shape: Tuple[int, ...]) -> torch.Tensor:
        if shape not in self._buf:
            t = torch.randn(shape, device=self.device, dtype=self.dtype)
            self._buf[shape] = t
        return self._buf[shape]

    def _weight(self, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        key = (name, shape)
        if key not in self._wbuf:
            w = torch.randn(shape, device=self.device, dtype=self.dtype)
            self._wbuf[key] = w
        return self._wbuf[key]

    @staticmethod
    def _cuda_sync():
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _bench(self, fn, warmup: int = 3, iters: int = 10) -> float:
        """Return average latency in seconds for fn()."""
        if self.device.type == "cuda":
            self._cuda_sync()
            # warmup
            for _ in range(max(0, warmup)):
                fn()
            self._cuda_sync()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(max(1, iters)):
                fn()
            ender.record()
            self._cuda_sync()
            ms = starter.elapsed_time(ender) / max(1, iters)
            return float(ms) / 1000.0
        else:
            # CPU fallback
            for _ in range(max(0, warmup)):
                fn()
            t0 = time.perf_counter()
            for _ in range(max(1, iters)):
                fn()
            t1 = time.perf_counter()
            return float(t1 - t0) / max(1, iters)

    def rmsnorm(self, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x: [T, D], w: [D]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
        return x * w

    def execute_one(self, sig: OpSig, sh: OpShape) -> None:
        """Execute one op once (no timing)."""
        D = sh.dim
        Sd = sh.shard_dim
        Fsd = sh.ffn_shard_dim
        T = sh.query_len
        H = sh.heads_per_shard
        Hd = sh.head_dim
        K = sh.key_len

        op = sig.op
        s = sig.shard

        # Use no_grad to avoid autograd overhead
        if op == "LN":
            x = self._rand((T, D))
            w = self._weight("rms_w", (D,))
            _ = self.rmsnorm(x, w)
            return

        elif op in ("Q","K","V"):
            x = self._rand((T, D))
            w = self._weight(f"{op}_w_{s}", (Sd, D))
            _ = F.linear(x, w)
            return

        elif op == "O":
            x = self._rand((T, Sd))
            w = self._weight(f"O_w_{s}", (D, Sd))
            _ = F.linear(x, w)
            return

        elif op in ("FFN_W1","FFN_W3"):
            x = self._rand((T, D))
            w = self._weight(f"{op}_w_{s}", (Fsd, D))
            _ = F.linear(x, w)
            return

        elif op == "SwiGLU":
            a = self._rand((T, Fsd))
            b = self._rand((T, Fsd))
            _ = F.silu(a) * b
            return

        elif op == "FFN_W2":
            x = self._rand((T, Fsd))
            w = self._weight(f"FFN_W2_w_{s}", (D, Fsd))
            _ = F.linear(x, w)
            return

        elif op == "Add":
            a = self._rand((T, D))
            b = self._rand((T, D))
            _ = a + b
            return

        elif op == "QK":
            q = self._rand((H, T, Hd))
            k = self._rand((H, Hd, K))
            scale = 1.0 / (Hd ** 0.5)
            _ = torch.matmul(q, k) * scale
            return

        elif op == "Softmax":
            scores = self._rand((H, T, K))
            _ = torch.softmax(scores, dim=-1)
            return

        elif op == "SV":
            p = self._rand((H, T, K))
            v = self._rand((H, K, Hd))
            _ = torch.matmul(p, v)
            return

        # Unknown op:
        else:
            raise ValueError(f"Unknown op for GPUBackend: {op}")
            return

    def benchmark_segment(
        self,
        seg: SegmentSig,
        warmup: int = 3,
        iters: int = 10,
        *,
        debug: bool = False,
        debug_logger: Optional[Any] = None,
        seg_key: Optional[str] = None,
        task_meta: Optional[Dict[str, Any]] = None,
        debug_store: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Benchmark one segment.

        When debug=True, prints per-op shapes + a per-op timing breakdown (very noisy).
        """

        # Pre-build execution plan to avoid shape inference in inner loop
        plan: List[Tuple[OpSig, OpShape]] = []
        for op, shard in seg.ops:
            sig = OpSig(device_type=seg.device_type, phase=seg.phase, step=seg.step, op=op, shard=shard)
            plan.append((sig, infer_op_shape(sig, self.cfg)))

        def fn():
            with torch.no_grad():
                for sig, sh in plan:
                    self.execute_one(sig, sh)

        if not bool(debug):
            return self._bench(fn, warmup=warmup, iters=iters)

        # ------------------------------
        # Debug path
        # ------------------------------
        log = (getattr(debug_logger, "log", None) if debug_logger is not None else None)
        if not callable(log):
            log = print

        # Segment meta
        key_s = seg_key or seg.to_key()
        meta = task_meta or {}
        try:
            cnt = int(meta.get("count_hint", 0))
        except Exception:
            cnt = 0
        ops_repr = meta.get("ops_repr", None)

        log("=" * 88)
        log(f"[gpu-debug] segment_key={key_s}")
        log(f"[gpu-debug] sig: device_type={seg.device_type} phase={seg.phase} step={seg.step} (count_hint={cnt})")
        if ops_repr:
            log(f"[gpu-debug] ops_repr: {ops_repr}")
        log(f"[gpu-debug] cfg: device={self.device} dtype={self.dtype} dim={self.cfg.dim} ffn_dim={self.cfg.ffn_dim} "
            f"n_heads={self.cfg.n_heads} shards={self.cfg.shards}")

        # Helper formatters
        def _fmt_shape(shape: Tuple[int, ...]) -> str:
            return "x".join(str(int(x)) for x in shape)

        def _fmt_bytes(n: float) -> str:
            n = float(n)
            if n < 1024:
                return f"{n:.0f} B"
            if n < 1024**2:
                return f"{n/1024:.2f} KiB"
            if n < 1024**3:
                return f"{n/1024**2:.2f} MiB"
            return f"{n/1024**3:.2f} GiB"

        def _fmt_flops(n: float) -> str:
            n = float(n)
            if n < 1e3:
                return f"{n:.0f}"
            if n < 1e6:
                return f"{n/1e3:.3f} K"
            if n < 1e9:
                return f"{n/1e6:.3f} M"
            if n < 1e12:
                return f"{n/1e9:.3f} G"
            return f"{n/1e12:.3f} T"

        # Op I/O shapes + rough cost model
        try:
            dtype_bytes = int(torch.tensor([], dtype=self.dtype).element_size())
        except Exception:
            dtype_bytes = 0

        def _op_io(op: str, sh: OpShape) -> Tuple[List[Tuple[str, Tuple[int, ...]]], Tuple[str, Tuple[int, ...]]]:
            D = int(sh.dim)
            Sd = int(sh.shard_dim)
            Fsd = int(sh.ffn_shard_dim)
            T = int(sh.query_len)
            H = int(sh.heads_per_shard)
            Hd = int(sh.head_dim)
            K = int(sh.key_len)

            if op == "LN":
                return [("x", (T, D)), ("w", (D,))], ("y", (T, D))
            if op in ("Q", "K", "V"):
                return [("x", (T, D)), ("w", (Sd, D))], ("y", (T, Sd))
            if op == "O":
                return [("x", (T, Sd)), ("w", (D, Sd))], ("y", (T, D))
            if op in ("FFN_W1", "FFN_W3"):
                return [("x", (T, D)), ("w", (Fsd, D))], ("y", (T, Fsd))
            if op == "SwiGLU":
                return [("a", (T, Fsd)), ("b", (T, Fsd))], ("y", (T, Fsd))
            if op == "FFN_W2":
                return [("x", (T, Fsd)), ("w", (D, Fsd))], ("y", (T, D))
            if op == "Add":
                return [("a", (T, D)), ("b", (T, D))], ("y", (T, D))
            if op == "QK":
                return [("q", (H, T, Hd)), ("k", (H, Hd, K))], ("y", (H, T, K))
            if op == "Softmax":
                return [("scores", (H, T, K))], ("y", (H, T, K))
            if op == "SV":
                return [("p", (H, T, K)), ("v", (H, K, Hd))], ("y", (H, T, Hd))
            return [("?", tuple())], ("?", tuple())

        def _op_flops(op: str, sh: OpShape) -> float:
            """Very rough FLOPs estimate (multiply-add counted as 2 FLOPs)."""
            D = float(sh.dim)
            Sd = float(sh.shard_dim)
            Fsd = float(sh.ffn_shard_dim)
            T = float(sh.query_len)
            H = float(sh.heads_per_shard)
            Hd = float(sh.head_dim)
            K = float(sh.key_len)

            if op == "LN":
                # mean(x^2), rsqrt, mul: ~5 ops per element (rough)
                return 5.0 * T * D
            if op in ("Q", "K", "V"):
                return 2.0 * T * Sd * D
            if op == "O":
                return 2.0 * T * D * Sd
            if op in ("FFN_W1", "FFN_W3"):
                return 2.0 * T * Fsd * D
            if op == "SwiGLU":
                # silu + mul: very rough
                return 6.0 * T * Fsd
            if op == "FFN_W2":
                return 2.0 * T * D * Fsd
            if op == "Add":
                return 1.0 * T * D
            if op == "QK":
                return 2.0 * H * T * K * Hd
            if op == "Softmax":
                # exp + sum + div ~ 5 ops per element (rough)
                return 5.0 * H * T * K
            if op == "SV":
                return 2.0 * H * T * K * Hd
            return 0.0

        def _op_bytes(op: str, sh: OpShape) -> float:
            ins, out = _op_io(op, sh)
            total_elems = 0.0
            for _, shape in ins:
                if shape:
                    elems = 1
                    for v in shape:
                        elems *= int(v)
                    total_elems += float(elems)
            _, oshape = out
            if oshape:
                elems = 1
                for v in oshape:
                    elems *= int(v)
                total_elems += float(elems)
            return float(total_elems) * float(dtype_bytes)

        # Print op list + inferred shapes
        log("[gpu-debug] op plan (with inferred tensor shapes):")
        for j, (sig, sh) in enumerate(plan):
            ins, out = _op_io(sig.op, sh)
            ins_s = " + ".join([f"{n}=[{_fmt_shape(shape)}]" for n, shape in ins if shape])
            out_s = f"{out[0]}=[{_fmt_shape(out[1])}]" if out[1] else out[0]
            log(f"  - op#{j:02d}  op={sig.op:<8} shard={sig.shard:<2d}  {ins_s}  ->  {out_s}  (T={sh.query_len}, K={sh.key_len})")

        # GPU memory stats (best effort)
        mem_before = None
        if self.device.type == "cuda":
            try:
                mem_before = {
                    "alloc": int(torch.cuda.memory_allocated()),
                    "reserved": int(torch.cuda.memory_reserved()),
                }
            except Exception:
                mem_before = None

        # Warmup once (avoid re-alloc / cuBLAS autotune costs in timing loops)
        if warmup and warmup > 0:
            with torch.no_grad():
                for _ in range(int(warmup)):
                    for sig, sh in plan:
                        self.execute_one(sig, sh)
            if self.device.type == "cuda":
                self._cuda_sync()

        mem_after_warmup = None
        if self.device.type == "cuda":
            try:
                mem_after_warmup = {
                    "alloc": int(torch.cuda.memory_allocated()),
                    "reserved": int(torch.cuda.memory_reserved()),
                }
            except Exception:
                mem_after_warmup = None

        # Accurate segment timing (no extra per-op events)
        seg_s = self._bench(fn, warmup=0, iters=iters)

        # Per-op timing breakdown
        per_op_s: List[float] = [0.0 for _ in range(len(plan))]
        if len(plan) == 0:
            log("[gpu-debug] empty plan; segment time is 0")
            return float(seg_s)

        if self.device.type == "cuda":
            try:
                # Reset peak stats so we can report per-segment peaks
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

                # Allocate events once (reused each iter)
                ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(len(plan))]
                ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(len(plan))]

                it = max(1, int(iters))
                with torch.no_grad():
                    for _ in range(it):
                        for j, (sig, sh) in enumerate(plan):
                            ev_s[j].record()
                            self.execute_one(sig, sh)
                            ev_e[j].record()
                        self._cuda_sync()
                        for j in range(len(plan)):
                            per_op_s[j] += float(ev_s[j].elapsed_time(ev_e[j])) / 1000.0
                per_op_s = [float(x) / float(it) for x in per_op_s]

                mem_peak = None
                try:
                    mem_peak = {
                        "max_alloc": int(torch.cuda.max_memory_allocated()),
                        "max_reserved": int(torch.cuda.max_memory_reserved()),
                    }
                except Exception:
                    mem_peak = None

            except Exception as e:
                log(f"[gpu-debug] per-op CUDA timing failed: {e}")
                mem_peak = None
        else:
            import time as _time
            it = max(1, int(iters))
            with torch.no_grad():
                for _ in range(it):
                    for j, (sig, sh) in enumerate(plan):
                        t0 = _time.perf_counter()
                        self.execute_one(sig, sh)
                        t1 = _time.perf_counter()
                        per_op_s[j] += float(t1 - t0)
            per_op_s = [float(x) / float(it) for x in per_op_s]
            mem_peak = None

        # Print per-op timing table
        log("[gpu-debug] per-op timing breakdown (avg over iters):")
        hdr = f"  {'idx':>3}  {'op':<8} {'sh':>2}  {'io':<46}  {'time':>10}  {'est_flops':>10}  {'TFLOPs':>8}  {'bytes':>10}  {'GB/s':>8}"
        log(hdr)
        log("  " + "-" * (len(hdr) - 2))
        total_ops_s = float(sum(per_op_s))
        for j, ((sig, sh), t_s) in enumerate(zip(plan, per_op_s)):
            ins, out = _op_io(sig.op, sh)
            ins_s = "+".join([f"{n}[{_fmt_shape(shape)}]" for n, shape in ins if shape])
            out_s = f"{out[0]}[{_fmt_shape(out[1])}]" if out[1] else out[0]
            io_s = f"{ins_s}->{out_s}"
            if len(io_s) > 46:
                io_s = io_s[:43] + "..."

            fl = _op_flops(sig.op, sh)
            by = _op_bytes(sig.op, sh)
            t_ms = float(t_s) * 1000.0
            tflops = (fl / float(t_s) / 1e12) if (t_s and t_s > 0 and fl > 0) else 0.0
            gbs = (by / float(t_s) / 1e9) if (t_s and t_s > 0 and by > 0) else 0.0

            log(
                f"  {j:3d}  {sig.op:<8} {sig.shard:2d}  {io_s:<46}  {t_ms:9.3f}ms  {_fmt_flops(fl):>10}  {tflops:8.2f}  {_fmt_bytes(by):>10}  {gbs:8.2f}"
            )

        # Optional structured debug payload (stored into gpu_results.json when enabled)
        if isinstance(debug_store, dict):
            try:
                per_op_dbg: List[Dict[str, Any]] = []
                for j, ((sig, sh), t_s) in enumerate(zip(plan, per_op_s)):
                    ins, out = _op_io(sig.op, sh)
                    per_op_dbg.append(
                        {
                            "idx": int(j),
                            "op": str(sig.op),
                            "shard": int(sig.shard),
                            "phase": str(seg.phase),
                            "step": int(seg.step),
                            "T": int(sh.query_len),
                            "K": int(sh.key_len),
                            "dim": int(sh.dim),
                            "shard_dim": int(sh.shard_dim),
                            "ffn_shard_dim": int(sh.ffn_shard_dim),
                            "heads_per_shard": int(sh.heads_per_shard),
                            "head_dim": int(sh.head_dim),
                            "inputs": [{"name": n, "shape": [int(x) for x in shape]} for n, shape in ins if shape],
                            "output": {"name": out[0], "shape": [int(x) for x in out[1]]} if out[1] else {"name": out[0], "shape": []},
                            "latency_s": float(t_s),
                            "est_flops": float(_op_flops(sig.op, sh)),
                            "est_bytes": float(_op_bytes(sig.op, sh)),
                        }
                    )

                debug_store[str(key_s)] = {
                    "segment_key": str(key_s),
                    "device_type": str(seg.device_type),
                    "phase": str(seg.phase),
                    "step": int(seg.step),
                    "count_hint": int(cnt),
                    "ops_repr": str(ops_repr) if ops_repr is not None else seg.ops_repr(),
                    "segment_latency_s": float(seg_s),
                    "sum_per_op_s": float(total_ops_s),
                    "per_op": per_op_dbg,
                }
            except Exception:
                pass

        lbl = "cuda_events" if self.device.type == "cuda" else "wall_time"
        log(f"[gpu-debug] segment avg latency ({lbl}): {seg_s*1000.0:.3f} ms")
        log(f"[gpu-debug] sum(per-op)               : {total_ops_s*1000.0:.3f} ms (ratio={total_ops_s/seg_s if seg_s>0 else 0.0:.3f})")
        log("[gpu-debug] NOTE: sum(per-op) may differ from segment latency due to Python overhead, event/launch overhead, and allocator/cache effects.")

        if mem_before is not None or mem_after_warmup is not None:
            if mem_before is not None:
                log(f"[gpu-debug] cuda mem before warmup: alloc={_fmt_bytes(mem_before['alloc'])} reserved={_fmt_bytes(mem_before['reserved'])}")
            if mem_after_warmup is not None:
                log(f"[gpu-debug] cuda mem after warmup : alloc={_fmt_bytes(mem_after_warmup['alloc'])} reserved={_fmt_bytes(mem_after_warmup['reserved'])}")
        if self.device.type == "cuda" and 'mem_peak' in locals() and mem_peak is not None:
            log(f"[gpu-debug] cuda mem peak (this segment): max_alloc={_fmt_bytes(mem_peak['max_alloc'])} max_reserved={_fmt_bytes(mem_peak['max_reserved'])}")

        return float(seg_s)



class _DebugLogger:
    """Lightweight logger for verbose debug runs.

    When enabled, logs are printed to stdout and optionally duplicated to a
    text file.
    """

    def __init__(self, enabled: bool, out_path: Optional[str] = None):
        self.enabled = bool(enabled)
        self.out_path: Optional[Path] = None
        self._fp = None
        if self.enabled and out_path:
            p = Path(out_path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.out_path = p
            self._fp = p.open('w', encoding='utf-8')

    def log(self, msg: str) -> None:
        if not self.enabled:
            return
        s = str(msg)
        print(s)
        if self._fp is not None:
            try:
                self._fp.write(s + "\n")
                self._fp.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None


def _resolve_ramulator_bin_path(bin_str: str) -> Path:
    """Resolve ramulator2 executable path from a CLI/config string."""
    if not bin_str:
        raise ValueError("pim_ramulator_bin is empty")

    # Best-effort: treat as path first, then fall back to PATH lookup.
    p: Optional[Path] = None
    try:
        # resolve_existing_path handles cwd and script-root relative lookups.
        p = resolve_existing_path(bin_str)
    except Exception:
        which = shutil.which(bin_str)
        if which:
            p = Path(which)

    if p is None:
        raise FileNotFoundError(f"ramulator2 executable not found: {bin_str!r}")

    if p.is_dir():
        cand = p / 'ramulator2'
        if cand.exists():
            p = cand
        else:
            raise FileNotFoundError(f"ramulator2 executable not found in directory: {p}")

    return p.expanduser().resolve()


def _infer_max_seqlen_from_tasks(tasks: List[Dict[str, Any]], cfg: WorkloadConfig) -> int:
    """Infer a safe max_seq_len for trace args from the task list."""
    mx = int(max(1, cfg.prefill_len))
    # Prefer explicit decode_context_lens if present in config.
    if cfg.decode_context_lens:
        try:
            mx = max(mx, int(max(cfg.decode_context_lens)))
        except Exception:
            pass
    # Scan tasks as fallback.
    for t in tasks:
        try:
            ph = str(t.get('sig', {}).get('phase', 'decode'))
            st = int(t.get('sig', {}).get('step', 0))
            ops = t.get('ops', []) or []
            if not ops:
                continue
            op0 = str(ops[0].get('op', ''))
            sh0 = int(ops[0].get('shard', -1))
            shp = infer_op_shape(OpSig(device_type='pim', phase=ph, step=st, op=op0, shard=sh0), cfg)
            mx = max(mx, int(shp.key_len))
        except Exception:
            continue
    return int(max(1, mx))


class PIMBackendViaCostModel:
    """PIM latency backend that delegates to cost_model_pim_backend.

    This unifies op normalization and trace emission so schedule verification
    and CostModel use the exact same implementation (no duplicated emitters).
    """

    def __init__(
        self,
        cfg: WorkloadConfig,
        tasks: List[Dict[str, Any]],
        *,
        debug_logger: _DebugLogger,
        use_cache: bool = True,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for AiM simulator.")
        self.cfg = cfg
        self.debug = bool(getattr(cfg, 'debug', False))
        self._dbg = debug_logger
        self.use_cache = bool(use_cache)

        # Lazy import so GPU-only flows don't depend on CostModel modules.
        try:
            # Make import robust even when invoked from ./verify.
            try:
                _add_sys_path(_script_dir().parent)
            except Exception:
                pass
            import cost_model_pim_backend as cm_pim_backend  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import cost_model_pim_backend. "
                "Ensure schedule_deploy_verify is run from the repo root (so PYTHONPATH works). "
                f"Import error: {e}"
            ) from e
        self.cm = cm_pim_backend

        # Resolve configs
        if not cfg.pim_ramulator_config:
            raise ValueError("pim_ramulator_config must be provided")
        self.ramulator_config = resolve_existing_path(str(cfg.pim_ramulator_config))

        if not cfg.pim_ramulator_bin:
            raise ValueError("pim_ramulator_bin must be provided")
        self.ramulator_bin = _resolve_ramulator_bin_path(str(cfg.pim_ramulator_bin))

        # cost_model_pim_backend historically uses Path.cwd()/ramulator2;
        # we standardize it via env var.
        os.environ['RAMULATOR2_BIN'] = str(self.ramulator_bin)

        # PIM hw config (pim_config)
        self._tmp_pim_cfg_dir: Optional[Path] = None
        self.pim_config = self._resolve_or_synthesize_pim_config(tasks)

        # Model dict cache (keyed by (dim, heads, kv_heads, ffn_dim, seqlen)).
        self._model_dict_cache: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
        self.debug_segments: Optional[Dict[str, Any]] = {} if self.debug else None

        self._print_debug_header()

    def _resolve_or_synthesize_pim_config(self, tasks: List[Dict[str, Any]]) -> Path:
        # If user provided a PIM hw JSON path, use it directly.
        if getattr(self.cfg, 'pim_hw_json_path', None):
            return resolve_existing_path(str(self.cfg.pim_hw_json_path))

        # Otherwise synthesize a minimal JSON compatible with cost_model backend.
        max_seq_len = _infer_max_seqlen_from_tasks(tasks, self.cfg)
        pim_cfg = {
            'DRAM_column': int(self.cfg.pim_dram_column),
            'DRAM_row': int(self.cfg.pim_dram_row),
            'burst_length': int(self.cfg.pim_burst_length),
            'num_banks': int(self.cfg.pim_num_banks),
            'num_channels': int(self.cfg.pim_num_channels),
            'threads': int(self.cfg.pim_threads),
            'reuse_size': int(self.cfg.pim_reuse_size),
            'channels_per_block': int(self.cfg.pim_num_channels),
            'max_seq_len': int(max_seq_len),
        }

        self._tmp_pim_cfg_dir = Path(tempfile.mkdtemp(prefix='pim_hw_synth_')).resolve()
        p = self._tmp_pim_cfg_dir / 'pim_hw_synth.json'
        p.write_text(json.dumps(pim_cfg, indent=2), encoding='utf-8')
        if self.debug:
            self._dbg.log(f"[run-pim][debug] synthesized pim_config JSON: {p}")
        return p

    def close(self) -> None:
        if self._tmp_pim_cfg_dir is not None:
            try:
                shutil.rmtree(self._tmp_pim_cfg_dir, ignore_errors=True)
            except Exception:
                pass
            self._tmp_pim_cfg_dir = None

    def _print_debug_header(self) -> None:
        if not self.debug:
            return

        # Try to surface the CostModel constant used for cycles->sec.
        try:
            from config import PIM_FREQ_GHZ as CM_PIM_FREQ_GHZ  # type: ignore
        except Exception:
            CM_PIM_FREQ_GHZ = None  # type: ignore

        self._dbg.log("=" * 80)
        self._dbg.log("[run-pim][debug] Using cost_model_pim_backend._get_pim_latency_via_trace")
        self._dbg.log(f"[run-pim][debug] shard_policy={self.cfg.shard_policy!r} shards={int(self.cfg.shards)}")
        self._dbg.log(f"[run-pim][debug] model: dim={int(self.cfg.dim)} ffn_dim={int(self.cfg.ffn_dim)} n_heads={int(self.cfg.n_heads)} n_kv_heads={(int(self.cfg.n_kv_heads) if self.cfg.n_kv_heads is not None else 'auto->n_heads')}")
        self._dbg.log(f"[run-pim][debug] lengths: prefill_len={int(self.cfg.prefill_len)} decode_stride={int(getattr(self.cfg,'decode_stride',1) or 1)} decode_context_lens_len={(len(self.cfg.decode_context_lens) if self.cfg.decode_context_lens else 0)}")
        self._dbg.log(f"[run-pim][debug] pim_config={self.pim_config} (from {'pim_hw_json_path' if self.cfg.pim_hw_json_path else 'synth'})")
        self._dbg.log(f"[run-pim][debug] ramulator_config={self.ramulator_config}")
        self._dbg.log(f"[run-pim][debug] ramulator_bin={self.ramulator_bin} (exported as env RAMULATOR2_BIN)")
        self._dbg.log(f"[run-pim][debug] ramulator_timeout_s={int(getattr(self.cfg,'pim_ramulator_timeout_s',0) or 0)}")
        self._dbg.log(f"[run-pim][debug] cache_enabled={self.use_cache}")
        self._dbg.log(f"[run-pim][debug] schedule cfg.pim_freq_ghz={float(getattr(self.cfg,'pim_freq_ghz',0.0) or 0.0)}")
        self._dbg.log(f"[run-pim][debug] cost_model config.PIM_FREQ_GHZ={CM_PIM_FREQ_GHZ}")
        self._dbg.log("=" * 80)


    def _pim_params_for_op(
        self,
        *,
        op: str,
        shard: int,
        phase: str,
        step: int,
    ) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        sig = OpSig(device_type='pim', phase=str(phase), step=int(step), op=str(op), shard=int(shard))
        sh = infer_op_shape(sig, self.cfg)

        shards = int(max(1, getattr(self.cfg, 'shards', 1) or 1))
        shard_policy = str(getattr(self.cfg, 'shard_policy', 'fine')).strip().lower()
        fine = shard_policy in ('fine',)

        n_kv_total = int(self.cfg.n_kv_heads) if self.cfg.n_kv_heads is not None else int(self.cfg.n_heads)
        dim = int(sh.dim)

        if fine and int(shard) >= 0 and shards > 1:
            n_heads = int(max(1, sh.heads_per_shard))
            ffn_dim = int(max(1, sh.ffn_shard_dim))
            n_kv_heads = int(max(1, n_kv_total // shards))
        else:
            n_heads = int(max(1, self.cfg.n_heads))
            ffn_dim = int(max(1, self.cfg.ffn_dim))
            n_kv_heads = int(max(1, n_kv_total))

        head_dim = int(max(1, sh.head_dim))
        q_dim = int(max(1, int(n_heads) * int(head_dim)))
        kv_dim = int(max(1, int(n_kv_heads) * int(head_dim)))
        o_dim = int(max(1, int(q_dim)))

        seqlen = int(max(1, sh.key_len))
        qlen = int(max(1, sh.query_len))
        return dim, n_heads, n_kv_heads, ffn_dim, seqlen, qlen, head_dim, q_dim, kv_dim, o_dim

    def _get_model_dict(self, *, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: int) -> Dict[str, Any]:
        key = (int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim), int(seqlen))
        md = self._model_dict_cache.get(key)
        if md is not None:
            return md

        # Use CostModel's shared model_dict generator to avoid drift.
        md = self.cm._make_shared_model_dict(int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim), int(seqlen))
        self._model_dict_cache[key] = md
        return md

    def benchmark_segment(
        self,
        seg: SegmentSig,
        *,
        seg_key: Optional[str] = None,
        task_meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        # Try to surface the CostModel constant used for cycles->sec.
        try:
            from config import PIM_FREQ_GHZ as CM_PIM_FREQ_GHZ  # type: ignore
        except Exception:
            CM_PIM_FREQ_GHZ = None  # type: ignore

        key_s = seg_key or seg.to_key()
        meta = task_meta or {}
        try:
            cnt = int(meta.get("count_hint", 0))
        except Exception:
            cnt = 0
        ops_repr = meta.get("ops_repr", None)

        # Local helper: format logical tensor shapes (not the internal CENT tensors).
        def _fmt_shape(shape: Tuple[int, ...]) -> str:
            return "x".join(str(int(x)) for x in shape)

        def _op_io_logical(op: str, sh: OpShape) -> Tuple[List[Tuple[str, Tuple[int, ...]]], Tuple[str, Tuple[int, ...]]]:
            """Logical op I/O shapes (aligned with infer_op_shape)."""
            D = int(sh.dim)
            Sd = int(sh.shard_dim)
            Fsd = int(sh.ffn_shard_dim)
            T = int(sh.query_len)
            H = int(sh.heads_per_shard)
            Hd = int(sh.head_dim)
            K = int(sh.key_len)

            if op == "LN":
                return [("x", (T, D)), ("w", (D,))], ("y", (T, D))
            if op in ("Q", "K", "V"):
                return [("x", (T, D)), ("w", (Sd, D))], ("y", (T, Sd))
            if op == "O":
                return [("x", (T, Sd)), ("w", (D, Sd))], ("y", (T, D))
            if op in ("FFN_W1", "FFN_W3"):
                return [("x", (T, D)), ("w", (Fsd, D))], ("y", (T, Fsd))
            if op == "SwiGLU":
                return [("a", (T, Fsd)), ("b", (T, Fsd))], ("y", (T, Fsd))
            if op == "FFN_W2":
                return [("x", (T, Fsd)), ("w", (D, Fsd))], ("y", (T, D))
            if op == "Add":
                return [("a", (T, D)), ("b", (T, D))], ("y", (T, D))
            if op == "QK":
                return [("q", (H, T, Hd)), ("k", (H, Hd, K))], ("y", (H, T, K))
            if op == "Softmax":
                return [("scores", (H, T, K))], ("y", (H, T, K))
            if op == "SV":
                return [("p", (H, T, K)), ("v", (H, K, Hd))], ("y", (H, T, Hd))
            return [("?", tuple())], ("?", tuple())

        # Debug segment header + op list (logical shapes)
        if self.debug:
            self._dbg.log("=" * 88)
            self._dbg.log(f"[pim-debug] segment_key={key_s}")
            self._dbg.log(f"[pim-debug] sig: device_type={seg.device_type} phase={seg.phase} step={seg.step} (count_hint={cnt})")
            if ops_repr:
                self._dbg.log(f"[pim-debug] ops_repr: {ops_repr}")
            self._dbg.log(
                f"[pim-debug] cfg: shard_policy={self.cfg.shard_policy!r} shards={int(self.cfg.shards)} "
                f"dim={int(self.cfg.dim)} ffn_dim={int(self.cfg.ffn_dim)} n_heads={int(self.cfg.n_heads)} "
                f"n_kv_heads={(int(self.cfg.n_kv_heads) if self.cfg.n_kv_heads is not None else 'auto->n_heads')}"
            )
            self._dbg.log(f"[pim-debug] pim_config={self.pim_config}")
            self._dbg.log(f"[pim-debug] ramulator_config={self.ramulator_config}")
            self._dbg.log(f"[pim-debug] cost_model config.PIM_FREQ_GHZ={CM_PIM_FREQ_GHZ}")
            self._dbg.log("[pim-debug] op plan (logical shapes inferred from schedule cfg):")
            for j, (op, shard) in enumerate(seg.ops):
                if str(op).strip().lower() in _COMM_OPS:
                    continue
                sig = OpSig(device_type="pim", phase=str(seg.phase), step=int(seg.step), op=str(op), shard=int(shard))
                sh = infer_op_shape(sig, self.cfg)
                ins, out = _op_io_logical(str(op), sh)
                ins_s = " + ".join([f"{n}=[{_fmt_shape(shape)}]" for n, shape in ins if shape])
                out_s = f"{out[0]}=[{_fmt_shape(out[1])}]" if out[1] else out[0]
                self._dbg.log(
                    f"  - op#{j:02d}  op={str(op):<8} shard={int(shard):<2d}  {ins_s}  ->  {out_s}  (T={int(sh.query_len)}, K={int(sh.key_len)})"
                )

            self._dbg.log(
                "[pim-debug] NOTE: CENT(AiM) trace backend internally uses x shape (1,1,dim) and handles prefill by repeating ops;\n"
                "           the table below shows the *effective parameters* passed into cost_model_pim_backend for each op."
            )

        total = 0.0
        per_op_dbg: List[Dict[str, Any]] = []
        base_dim = int(self.cfg.dim)
        base_n_heads = int(self.cfg.n_heads)
        base_n_kv_heads = int(self.cfg.n_kv_heads) if self.cfg.n_kv_heads is not None else int(self.cfg.n_heads)
        base_ffn_dim = int(self.cfg.ffn_dim)

        for j, (op, shard) in enumerate(seg.ops):
            if str(op).strip().lower() in _COMM_OPS:
                continue

            dim, n_heads, n_kv_heads, ffn_dim, seqlen, qlen, head_dim, q_dim, kv_dim, o_dim = self._pim_params_for_op(
                op=str(op), shard=int(shard), phase=str(seg.phase), step=int(seg.step)
            )
            op_norm = self.cm._normalize_pim_op(str(op))

            cache_hit = None
            if self.debug and self.use_cache:
                try:
                    cache_hit = self.cm._pim_cache.get(
                        op_norm,
                        str(seg.phase),
                        int(dim),
                        int(n_heads),
                        int(n_kv_heads),
                        int(ffn_dim),
                        int(seqlen),
                        int(head_dim),
                        int(q_dim),
                        int(kv_dim),
                        int(o_dim),
                        self.pim_config,
                        self.ramulator_config,
                    )
                except Exception:
                    cache_hit = None

            model_dict = self._get_model_dict(dim=base_dim, n_heads=base_n_heads, n_kv_heads=base_n_kv_heads, ffn_dim=base_ffn_dim, seqlen=seqlen)

            # Provide a stable prefix so traces are easy to correlate (if kept).
            trace_prefix = None
            if self.debug:
                trace_prefix = f"{seg.phase}_step{seg.step}_sh{shard}_{op_norm}"

            sec = float(
                self.cm._get_pim_latency_via_trace(
                    op=op_norm,
                    pim_config=self.pim_config,
                    ramulator_config=self.ramulator_config,
                    dim=int(dim),
                    n_heads=int(n_heads),
                    n_kv_heads=int(n_kv_heads),
                    ffn_dim=int(ffn_dim),
                    seqlen=int(seqlen),
                    phase=str(seg.phase),
                    model_dict=model_dict,
                    use_cache=bool(self.use_cache),
                    head_dim=int(head_dim),
                    q_dim=int(q_dim),
                    kv_dim=int(kv_dim),
                    o_dim=int(o_dim),
                    ramulator_timeout_s=int(getattr(self.cfg, 'pim_ramulator_timeout_s', 300) or 300),
                    keep_traces=bool(getattr(self.cfg, 'pim_keep_traces', False)),
                    trace_dir=(Path(self.cfg.pim_trace_dir).expanduser().resolve() if getattr(self.cfg, 'pim_trace_dir', None) else None),
                    trace_prefix=trace_prefix,
                )
            )

            total += sec

            cycles_est = None
            try:
                if CM_PIM_FREQ_GHZ is not None and float(CM_PIM_FREQ_GHZ) > 0.0:
                    cycles_est = int(round(float(sec) * float(CM_PIM_FREQ_GHZ) * 1e9))
            except Exception:
                cycles_est = None

            per_op_dbg.append(
                {
                    "idx": int(j),
                    "op": str(op),
                    "op_norm": str(op_norm),
                    "shard": int(shard),
                    "phase": str(seg.phase),
                    "step": int(seg.step),
                    "dim": int(dim),
                    "n_heads": int(n_heads),
                    "head_dim": int(head_dim),
                    "q_dim": int(q_dim),
                    "kv_dim": int(kv_dim),
                    "o_dim": int(o_dim),
                    "n_kv_heads": int(n_kv_heads),
                    "ffn_dim": int(ffn_dim),
                    "seqlen": int(seqlen),
                    "qlen": int(qlen),
                    "cache_hit": bool(cache_hit is not None) if self.use_cache else False,
                    "latency_s": float(sec),
                    "cycles_est": int(cycles_est) if cycles_est is not None else None,
                    "trace_prefix": str(trace_prefix) if trace_prefix else None,
                }
            )

            if self.debug:
                extra = f" cycles≈{cycles_est}" if cycles_est is not None else ""
                self._dbg.log(
                    f"[pim-debug] op#{j:02d} op={str(op):<8} norm={op_norm:<10} sh={int(shard):<2d} "
                    f"dim={int(dim):<5d} heads={int(n_heads):<3d} hd={int(head_dim):<3d} q_dim={int(q_dim):<5d} kv_heads={int(n_kv_heads):<3d} kv_dim={int(kv_dim):<5d} o_dim={int(o_dim):<5d} ffn_dim={int(ffn_dim):<6d} "
                    f"seqlen={int(seqlen):<6d} qlen={int(qlen):<3d} cache={'Y' if (cache_hit is not None and self.use_cache) else 'N'} "
                    f"lat={sec*1e3:9.3f}ms{extra}"
                )

        if self.debug:
            self._dbg.log(f"[pim-debug] segment total latency: {total*1e3:.3f} ms")

        if self.debug_segments is not None:
            self.debug_segments[key_s] = {
                "segment_key": key_s,
                "device_type": str(seg.device_type),
                "phase": str(seg.phase),
                "step": int(seg.step),
                "count_hint": int(cnt),
                "ops_repr": str(ops_repr) if ops_repr is not None else seg.ops_repr(),
                "total_s": float(total),
                "per_op": per_op_dbg,
            }

        return float(total)
# ==========================================================
# JSON helpers
# ==========================================================
def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _load_json(path: str) -> Dict[str, Any]:
    p = resolve_existing_path(path)
    return json.loads(p.read_text(encoding="utf-8"))


# ==========================================================
# Model shape JSON helpers
# ==========================================================
_MODEL_DIM_KEYS = [
    "hidden_dim",
    "hidden_size",
    "dim",
    "d_model",
    "model_dim",
    "n_embd",
]
_MODEL_FFN_KEYS = [
    "intermediate_dim",
    "intermediate_size",
    "ffn_dim",
    "ffn_hidden_dim",
    "mlp_dim",
]
_MODEL_NHEAD_KEYS = [
    "q_head_num",
    "num_attention_heads",
    "n_heads",
    "num_heads",
    "head_num",
    "n_head",
]


def load_model_shape_json(path: str) -> Dict[str, int]:
    """Load a model shape/config JSON and return {dim, ffn_dim, n_heads}."""
    p = resolve_existing_path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"model cfg must be a JSON object (dict), got {type(obj)}: {p}")

    lower = {str(k).lower(): v for k, v in obj.items()}

    def _pick_int(keys: List[str]) -> Optional[int]:
        for k in keys:
            if k.lower() in lower:
                v = lower[k.lower()]
                try:
                    return int(v)
                except Exception:
                    continue
        return None

    dim = _pick_int(_MODEL_DIM_KEYS)
    ffn = _pick_int(_MODEL_FFN_KEYS)
    n_heads = _pick_int(_MODEL_NHEAD_KEYS)

    missing: List[str] = []
    if dim is None:
        missing.append("dim")
    if ffn is None:
        missing.append("ffn_dim")
    if n_heads is None:
        missing.append("n_heads")
    if missing:
        raise ValueError(
            f"model cfg missing required field(s): {missing}. "
            f"Got keys={sorted(list(lower.keys()))}. File={p}"
        )
    return {"dim": int(dim), "ffn_dim": int(ffn), "n_heads": int(n_heads)}

def load_pim_hw_json(path: str) -> Dict[str, Any]:
    """Load and normalize a minimal PIM HW spec JSON.
    Required keys (any alias below is accepted):
      - DRAM_column / dram_column / column / columns
      - DRAM_row / dram_row / row / rows
      - burst_length / burst / bl
      - num_banks / banks
      - num_channels / channels
      - pim_threads / threads
      - pim_reuse_size / reuse_size
      - pim_freq_ghz / freq_ghz / frequency_ghz
    """
    p = resolve_existing_path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"PIM HW json must be a JSON object (dict), got {type(obj)}: {p}")

    # case-insensitive lookup table
    lower = {str(k).lower(): v for k, v in obj.items()}

    def _get_any(keys: List[str], *, required: bool = True) -> Any:
        for k in keys:
            lk = str(k).lower()
            if lk in lower and lower[lk] is not None:
                return lower[lk]
        if required:
            raise KeyError(f"Missing required key in {p}: one of {keys}")
        return None

    def _as_pos_int(name: str, v: Any) -> int:
        try:
            iv = int(v)
        except Exception as e:
            raise ValueError(f"{name} must be an integer, got {v!r}") from e
        if iv <= 0:
            raise ValueError(f"{name} must be > 0, got {iv}")
        return iv

    def _as_pos_float(name: str, v: Any) -> float:
        try:
            fv = float(v)
        except Exception as e:
            raise ValueError(f"{name} must be a float, got {v!r}") from e
        if fv <= 0:
            raise ValueError(f"{name} must be > 0, got {fv}")
        return fv

    hw: Dict[str, Any] = {
        "pim_dram_column": _as_pos_int("pim_dram_column", _get_any(["dram_column", "dram_col", "column", "columns", "dram_column".upper()])),
        "pim_dram_row": _as_pos_int("pim_dram_row", _get_any(["dram_row", "row", "rows", "dram_row".upper()])),
        "pim_burst_length": _as_pos_int("pim_burst_length", _get_any(["burst_length", "burst", "bl"])),
        "pim_num_banks": _as_pos_int("pim_num_banks", _get_any(["num_banks", "banks"])),
        "pim_num_channels": _as_pos_int("pim_num_channels", _get_any(["num_channels", "channels"])),
        "pim_threads": None,
        "pim_reuse_size": None,
        "pim_freq_ghz": None,
    }

    v = _get_any(["pim_num_devices", "num_devices", "devices", "fc_devices"], required=False)
    if v is not None:
        raise ValueError(
            f"{p}: device-count keys (pim_num_devices/num_devices/devices/FC_devices) are no longer supported in --pim-hw-json. "
            "Pass device count via CLI: run-pim --pim-num-devices N."
        )

    v = _get_any(["pim_threads", "threads"], required=False)
    if v is not None:
        hw["pim_threads"] = _as_pos_int("pim_threads", v)

    v = _get_any(["pim_reuse_size", "reuse_size"], required=False)
    if v is not None:
        hw["pim_reuse_size"] = _as_pos_int("pim_reuse_size", v)

    v = _get_any(["pim_freq_ghz", "freq_ghz", "frequency_ghz"], required=False)
    if v is not None:
        hw["pim_freq_ghz"] = _as_pos_float("pim_freq_ghz", v)

    return hw

# ==========================================================
# Export / Run / Merge
# ==========================================================
def export_tasks(
    schedule_paths: List[str],
    cfg: WorkloadConfig,
    out_dir: str,
    prefix: str,
    *,
    comms_paths: Optional[List[str]] = None,
) -> Tuple[Path, Path]:
    outp = Path(out_dir).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)

    # load schedules
    dfs: List[pd.DataFrame] = []
    resolved_paths: List[Path] = []
    max_decode_steps = 0
    max_shard_seen = -1
    for sp in schedule_paths:
        p = resolve_existing_path(sp)
        resolved_paths.append(p)
        df = load_schedule_csv(str(p))
        dfs.append(df)
        max_decode_steps = max(max_decode_steps, infer_decode_steps(df))

        # infer max shard index across all schedules
        if "shard" in df.columns:
            ms = pd.to_numeric(df["shard"], errors="coerce").fillna(-1).astype(int).max()
            max_shard_seen = max(max_shard_seen, int(ms))

    inferred = int(max_shard_seen + 1) if int(max_shard_seen) >= 0 else 1
    if int(getattr(cfg, "shards", 0) or 0) != inferred:
        print(f"[export] inferred shards={inferred} from schedule (max_shard={max_shard_seen}), override cfg.shards={cfg.shards}")
    cfg.shards = inferred

    # optional comms traces (for weight_load accounting)
    resolved_comms: List[Path] = []
    if comms_paths:
        for cp in comms_paths:
            if not cp:
                continue
            resolved_comms.append(resolve_existing_path(cp))
    # Global sum across all provided comms traces.
    wl = extract_weight_load_seconds(
        [str(p) for p in resolved_comms],
        decode_stride=int(getattr(cfg, "decode_stride", 1) or 1),
    )
    weight_load_gpu_s = float(wl.get("gpu_s", 0.0))
    weight_load_pim_s = float(wl.get("pim_s", 0.0))
    weight_load_unknown_s = float(wl.get("unknown_s", 0.0))

    # Optional per-schedule mapping when user provides paired schedules + comms traces.
    # This avoids over/under-counting when merge() is called with multiple schedules.
    weight_load_by_schedule: Dict[str, Dict[str, float]] = {}
    if comms_paths and len(resolved_comms) == len(resolved_paths):
        stride = int(getattr(cfg, "decode_stride", 1) or 1)
        for sp, cp in zip(resolved_paths, resolved_comms):
            w = extract_weight_load_seconds([str(cp)], decode_stride=stride)
            weight_load_by_schedule[str(sp)] = {
                "gpu_s": float(w.get("gpu_s", 0.0)),
                "pim_s": float(w.get("pim_s", 0.0)),
                "unknown_s": float(w.get("unknown_s", 0.0)),
                "total_s": float(w.get("total_s", 0.0)),
            }

    # auto-fill decode_context_lens if None
    if cfg.decode_context_lens is None:
        stride = int(getattr(cfg, "decode_stride", 1) or 1)
        if stride <= 0:
            stride = 1
        cfg.decode_context_lens = (
            [
                int(cfg.prefill_len + 1 + int(decode_token_index_from_sample_step(int(i), int(stride))))
                for i in range(max_decode_steps)
            ]
            if max_decode_steps > 0
            else []
        )

    # collect unique segments over all schedules
    uniq: Dict[str, SegmentSig] = {}
    ctr: Counter = Counter()

    shard_policy = str(getattr(cfg, "shard_policy", "fine")).strip().lower()
    use_coarse = shard_policy in ("coarse", "coarse_majority", "coarse-majority", "majority", "coarsemajority")

    for df in dfs:  # all schedules merge into one uniq dict and update counter
        if use_coarse:
            u, c = extract_segments_coarse_majority(df, cfg.segment_scope, total_shards=int(cfg.shards))
        else:
            u, c = extract_segments(df, cfg.segment_scope)
        uniq.update(u)
        ctr.update(c)

    gpu_tasks: List[Dict[str, Any]] = []
    pim_tasks: List[Dict[str, Any]] = []
    skipped_tasks: List[Dict[str, Any]] = []
    for k, seg in uniq.items():
        item = {
            "key": k,
            "sig": {"device_type": seg.device_type, "phase": seg.phase, "step": int(seg.step)},
            "ops": [{"op": op, "shard": int(shard)} for op, shard in seg.ops],
            "ops_repr": seg.ops_repr(),
            "count_hint": int(ctr.get(k, 0)),
        }
        dev_type = str(seg.device_type).strip().lower()
        if dev_type == "npu":
            gpu_tasks.append(item)
        elif dev_type == "pim":
            pim_tasks.append(item)
        else:
            skipped_tasks.append(item)


    gpu_json = {
        "version": 3,
        "task_type": "segment",
        "backend": "gpu",
        "segment_scope": cfg.segment_scope,
        "schedules": [str(p) for p in resolved_paths],
        "weight_load_s": float(weight_load_gpu_s),
        "weight_load_meta": {
            "gpu_s": float(weight_load_gpu_s),
            "pim_s": float(weight_load_pim_s),
            "unknown_s": float(weight_load_unknown_s),
            "total_s": float(weight_load_gpu_s + weight_load_pim_s),
        },
        "weight_load_by_schedule": weight_load_by_schedule,
        "config": cfg.to_dict(),
        "tasks": sorted(gpu_tasks, key=lambda x: x["key"]),
    }
    pim_json = {
        "version": 3,
        "task_type": "segment",
        "backend": "pim",
        "segment_scope": cfg.segment_scope,
        "schedules": [str(p) for p in resolved_paths],
        "comms_traces": [str(p) for p in resolved_comms],
        "weight_load_s": float(weight_load_pim_s),
        "weight_load_meta": {
            "gpu_s": float(weight_load_gpu_s),
            "pim_s": float(weight_load_pim_s),
            "unknown_s": float(weight_load_unknown_s),
            "total_s": float(weight_load_gpu_s + weight_load_pim_s),
        },
        "weight_load_by_schedule": weight_load_by_schedule,
        "config": cfg.to_dict(),
        "tasks": sorted(pim_tasks, key=lambda x: x["key"]),
    }

    gpu_path = outp / f"{prefix}.gpu_tasks.json"
    pim_path = outp / f"{prefix}.pim_tasks.json"
    _save_json(gpu_path, gpu_json)
    _save_json(pim_path, pim_json)

    print(f"[export] segment_scope={cfg.segment_scope}")
    if resolved_comms:
        print(
            f"[export] weight_load_s: gpu={weight_load_gpu_s:.6f}s pim={weight_load_pim_s:.6f}s "
            f"(unknown={weight_load_unknown_s:.6f}s) from {len(resolved_comms)} comms trace(s)"
        )
    print(f"[export] wrote {gpu_path} ({len(gpu_tasks)} segments)")
    print(f"[export] wrote {pim_path} ({len(pim_tasks)} segments)")
    if skipped_tasks:
        by_type: Counter = Counter([str(t.get("sig", {}).get("device_type", "")).strip().lower() for t in skipped_tasks])
        by_type_s = ", ".join([f"{k or '<empty>'}={v}" for k, v in by_type.items()])
        print(
            f"[export] skipped {len(skipped_tasks)} trace-only segment(s) (not npu/pim). "
            f"They will be kept via schedule durations in merge(). by_device_type: {by_type_s}"
        )
    return gpu_path, pim_path



def run_gpu(
    tasks_json: str,
    out_json: str,
    warmup: int = 3,
    iters: int = 10,
    device: Optional[str] = None,
    gpu_dtype: Optional[str] = None,
    *,
    debug: bool = False,
    debug_txt: Optional[str] = None,
) -> Path:
    data = _load_json(tasks_json)
    cfg = WorkloadConfig.from_dict(data.get("config", {}))
    # Apply CLI/runtime overrides (take precedence over gpu_tasks.json config)
    if device:
        cfg.device = device
    if gpu_dtype:
        cfg.gpu_dtype = gpu_dtype
    seg_scope = data.get("segment_scope", cfg.segment_scope)

    try:
        weight_load_s = float(data.get("weight_load_s", 0.0) or 0.0)
    except Exception:
        weight_load_s = 0.0

    weight_load_by_schedule = data.get("weight_load_by_schedule", {}) or {}

    tasks = data.get("tasks", [])

    # Debug output + (optional) tee to file
    cfg.debug = bool(debug or getattr(cfg, "debug", False))
    dbg = _DebugLogger(enabled=bool(cfg.debug), out_path=debug_txt)
    log = dbg.log if dbg.enabled else print

    gpu_info = collect_gpu_info(cfg.device)

    results: Dict[str, float] = {}
    log(f"[run-gpu] tasks={len(tasks)} warmup={warmup} iters={iters} device={cfg.device} dtype={cfg.gpu_dtype} segment_scope={seg_scope}")
    if weight_load_s:
        log(f"[run-gpu] extra weight_load_s={weight_load_s:.6f}s (will be added in merge)")
    print_gpu_info(gpu_info, print_fn=log)

    # Extra backend/flags snapshot (debug only)
    if dbg.enabled and torch is not None:
        try:
            log(f"[run-gpu][debug] torch_default_dtype={torch.get_default_dtype()}")
        except Exception:
            pass
        try:
            # matmul / TF32 knobs (may not exist on some builds)
            mm = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
            if mm is not None and hasattr(mm, "allow_tf32"):
                log(f"[run-gpu][debug] torch.backends.cuda.matmul.allow_tf32={mm.allow_tf32}")
        except Exception:
            pass
        try:
            cd = getattr(torch.backends, "cudnn", None)
            if cd is not None:
                log(
                    f"[run-gpu][debug] cudnn.enabled={getattr(cd,'enabled',None)} "
                    f"benchmark={getattr(cd,'benchmark',None)} deterministic={getattr(cd,'deterministic',None)} "
                    f"allow_tf32={getattr(cd,'allow_tf32',None)}"
                )
        except Exception:
            pass

        # One-time nvidia-smi snapshot (utilization / mem usage) if available.
        smi_path = shutil.which("nvidia-smi")
        if smi_path:
            ok, out = _try_run([
                smi_path,
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ])
            if ok and out:
                log(f"[run-gpu][debug] nvidia-smi snapshot: {out}")

    backend = GPUBackend(cfg)
    debug_segments: Optional[Dict[str, Any]] = {} if dbg.enabled else None
    try:
        for i, t in enumerate(tasks):
            key = t["key"]
            ops = tuple((x["op"], int(x.get("shard", -1))) for x in t.get("ops", []))
            sig = SegmentSig(device_type="npu", phase=str(t["sig"]["phase"]), step=int(t["sig"]["step"]), ops=ops)

            sec = backend.benchmark_segment(
                sig,
                warmup=warmup,
                iters=iters,
                debug=bool(dbg.enabled),
                debug_logger=dbg,
                seg_key=key,
                task_meta=t,
                debug_store=debug_segments,
            )
            results[key] = float(sec)

            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                log(f"  progress {i+1}/{len(tasks)}")
    finally:
        try:
            dbg.close()
        except Exception:
            pass

    outp = Path(out_json).expanduser().resolve()
    out = {
        "version": 3,
        "task_type": "segment",
        "backend": "gpu",
        "segment_scope": seg_scope,
        "config": cfg.to_dict(),
        "weight_load_s": float(weight_load_s),
        "weight_load_by_schedule": weight_load_by_schedule,
        "env": {"gpu": gpu_info},
        "results": results,
    }
    if dbg.enabled and debug_segments:
        out["debug_segments"] = debug_segments
    _save_json(outp, out)
    log(f"[run-gpu] wrote {outp}")
    return outp


def run_pim(
    tasks_json: str,
    out_json: str,
    *,
    cent_sim_root: Optional[str] = None,
    ramulator_config: Optional[str] = None,
    pim_hw_json: Optional[str] = None,
    pim_num_devices: Optional[int] = None,
    ramulator_bin: Optional[str] = None,
    ramulator_timeout_s: Optional[int] = None,
    keep_traces: bool = False,
    trace_dir: Optional[str] = None,
    debug: bool = False,
    debug_txt: Optional[str] = None,
    no_cache: bool = False,
) -> Path:

    data = _load_json(tasks_json)
    cfg_dict = data.get("config", {}) or {}
    cfg = WorkloadConfig.from_dict(cfg_dict)
    seg_scope = data.get("segment_scope", cfg.segment_scope)
    tasks = data.get("tasks", [])

    try:
        weight_load_s = float(data.get("weight_load_s", 0.0) or 0.0)
    except Exception:
        weight_load_s = 0.0

    weight_load_by_schedule = data.get("weight_load_by_schedule", {}) or {}

    if ramulator_config is not None:
        cfg.pim_ramulator_config = ramulator_config
    if ramulator_bin is not None:
        cfg.pim_ramulator_bin = ramulator_bin
    if ramulator_timeout_s is not None:
        cfg.pim_ramulator_timeout_s = int(ramulator_timeout_s)

    if keep_traces:
        cfg.pim_keep_traces = True
    if trace_dir is not None:
        cfg.pim_trace_dir = trace_dir
    # ------------------------------
    # PIM HW spec: read from a single JSON (PIM_AiM.json)
    # ------------------------------
    if pim_hw_json is not None:
        hw = load_pim_hw_json(pim_hw_json)
        # Required topology for aim_sim
        cfg.pim_dram_column = int(hw["pim_dram_column"])
        cfg.pim_dram_row = int(hw["pim_dram_row"])
        cfg.pim_burst_length = int(hw["pim_burst_length"])
        cfg.pim_num_banks = int(hw["pim_num_banks"])
        cfg.pim_num_channels = int(hw["pim_num_channels"])

        # Optional overrides (if present in JSON)
        if hw.get("pim_threads") is not None:
            cfg.pim_threads = int(hw["pim_threads"])  # type: ignore[arg-type]
        if hw.get("pim_reuse_size") is not None:
            cfg.pim_reuse_size = int(hw["pim_reuse_size"])  # type: ignore[arg-type]
        if hw.get("pim_freq_ghz") is not None:
            cfg.pim_freq_ghz = float(hw["pim_freq_ghz"])  # type: ignore[arg-type]
        # Keep the original JSON path so the CostModel trace backend can consume it.
        cfg.pim_hw_json_path = str(resolve_existing_path(pim_hw_json))
    if pim_num_devices is not None:
        try:
            iv = int(pim_num_devices)
        except Exception as e:
            raise ValueError(f"pim_num_devices must be an integer, got {pim_num_devices!r}") from e
        if iv <= 0:
            raise ValueError(f"pim_num_devices must be > 0, got {iv}")
        cfg.pim_num_devices = iv

    # In fine mode, we do NOT use cfg.pim_num_devices/--pim-num-devices to enable FC_devices.
    # Sharding is modeled explicitly via schedule rows and per-device segments.
    shard_policy = str(getattr(cfg, 'shard_policy', 'fine')).strip().lower()
    use_coarse = shard_policy in ("coarse", "coarse_majority", "coarse-majority", "majority", "coarsemajority")
    if (not use_coarse) and int(getattr(cfg, 'pim_num_devices', 1) or 1) != 1:
        print(
            f"[run-pim] shard_policy={cfg.shard_policy!r}: ignoring pim_num_devices={cfg.pim_num_devices} for trace generation; "
            "fine-mode tracing always uses a single PIM device (FC_devices=1) and follows per-shard placement from the schedule."
        )
    # For ramulator config, we require it to be actually present (not empty/None).
    # This can be provided via pim_tasks.json or CLI override.
    if not cfg.pim_ramulator_bin:
        raise ValueError(
            "pim_ramulator_bin must be provided to run ramulator.\n"
            "Provide it either when exporting tasks (export/all) or at run time: "
            "run-pim --pim-ramulator-bin <path>."
        )
    if not cfg.pim_ramulator_config:
        raise ValueError(
            "pim_ramulator_config must be provided to run ramulator.\n"
            "Provide it either when exporting tasks (export/all) or at run time: "
            "run-pim --pim-ramulator-config <path> --pim-hw-json <path>."
        )

    # Debug output + (optional) tee to file
    cfg.debug = bool(debug or getattr(cfg, 'debug', False))
    dbg = _DebugLogger(enabled=bool(cfg.debug), out_path=debug_txt)
    backend = PIMBackendViaCostModel(cfg, tasks, debug_logger=dbg, use_cache=(not bool(no_cache)))

    results: Dict[str, float] = {}
    print(f"[run-pim] tasks={len(tasks)} segment_scope={seg_scope}")
    if weight_load_s:
        print(f"[run-pim] extra weight_load_s={weight_load_s:.6f}s (will be added in merge)")
    try:
        for i, t in enumerate(tasks):
            key = t["key"]
            ops = tuple((x["op"], int(x.get("shard",-1))) for x in t.get("ops", []))
            sig = SegmentSig(device_type="pim", phase=str(t["sig"]["phase"]), step=int(t["sig"]["step"]), ops=ops)
            sec = backend.benchmark_segment(sig, seg_key=str(key), task_meta=t)
            results[key] = float(sec)
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                print(f"  progress {i+1}/{len(tasks)}")
    finally:
        try:
            backend.close()
        except Exception:
            pass
        try:
            dbg.close()
        except Exception:
            pass

    outp = Path(out_json).expanduser().resolve()
    out = {
        "version": 3,
        "task_type": "segment",
        "backend": "pim",
        "segment_scope": seg_scope,
        "config": cfg.to_dict(),
        "weight_load_s": float(weight_load_s),
        "weight_load_by_schedule": weight_load_by_schedule,
        "results": results,
    }
    if getattr(cfg, "debug", False) and getattr(backend, "debug_segments", None):
        out["debug_segments"] = backend.debug_segments
    _save_json(outp, out)
    print(f"[run-pim] wrote {outp}")
    return outp


def _comm_latency_from_df(df: pd.DataFrame, comm_model: str, pcie_lanes: int = 16) -> float:
    comm_model = comm_model.lower()
    comm_df = df[df["op"].astype(str).str.strip().str.lower().isin(_COMM_OPS)]
    if comm_model == "none":
        return 0.0
    if comm_model == "schedule":
        return float(comm_df["duration"].sum())
    if comm_model == "cxl":
        if cxl_latency is None:
            raise RuntimeError("cxl_latency.py not importable; cannot use --comm-model cxl")
        # crude: each comm node duration is estimated based on bytes inferred from node_id/op
        total = 0.0
        for _, r in comm_df.iterrows():
            op = str(r["op"])
            # Heuristic: KV write ~ shard_dim * dtype_bytes; Identity ~ dim * dtype_bytes
            # This is rough; adjust if you have a better mapping.
            layer = int(r.get("layer", -1))
            shard = int(r.get("shard", -1))
            # assume fp16 = 2 bytes
            dtype_bytes = 2
            # use cfg? we don't have cfg here; fallback to reading from trace? use 4096/4
            dim = 4096
            shards = 4
            shard_dim = dim // shards
            if op in ("K_write","V_write"):
                bytes_ = shard_dim * dtype_bytes
            else:
                bytes_ = dim * dtype_bytes
            total += float(cxl_latency.cxl_memcpy_latency(bytes_, lanes=pcie_lanes))
        return total
    raise ValueError(f"Unknown comm_model: {comm_model}")


def _try_lookup_segment_latency(seg: SegmentSig, res: Dict[str, float]) -> Optional[float]:
    k = seg.to_key()
    v = res.get(k, None)
    if v is not None:
        return float(v)
    return None


def _build_row_durations_layer_scope(
    df: pd.DataFrame,
    *,
    gpu_res: Dict[str, float],
    pim_res: Dict[str, float],
    comm_model: str,
    pcie_lanes: int,
    decode_stride: int,
    dim: int,
    shards: int,
    allow_missing: bool,
) -> Tuple[np.ndarray, int]:
    n = len(df)
    dur_s = np.zeros(n, dtype=np.float64)

    op_l = df["op"].astype(str).str.strip().str.lower()
    is_comm = op_l.isin(_COMM_OPS).to_numpy()

    # 1) COMM rows
    if comm_model.lower() == "schedule":
        dur_s[is_comm] = df.loc[is_comm, "duration"].astype(float).to_numpy()

    # 2) Compute rows (layer scope)
    compute_df = df.loc[~is_comm, ["phase", "step", "layer", "device", "device_type", "op", "shard", "start", "_row", "duration"]].copy()

    # Sort to match extract_segments() ordering: per group sorted by (start, _row)
    compute_df = compute_df.sort_values(["phase", "step", "layer", "device", "start", "_row"], kind="mergesort")

    missing = 0
    group_cols = ["phase", "step", "layer", "device"]
    for (phase, step, layer, device), g in compute_df.groupby(group_cols, sort=False):
        dev_type = str(g["device_type"].iloc[0])
        dev_type_n = dev_type.strip().lower()

        ops: List[Tuple[str, int]] = []
        for _, rr in g.iterrows():
            op = str(rr["op"]).strip()
            shard = int(rr["shard"]) if pd.notna(rr["shard"]) else -1
            ops.append((op, shard))

        idxs = g.index.to_numpy()
        if idxs.size == 0:
            continue

        w = pd.to_numeric(g["duration"], errors="coerce").fillna(0.0).astype(float).to_numpy()
        if dev_type_n not in ("npu", "pim"):
            dur_s[idxs] = w
            continue

        # Segment signature lookup (latest convention):
        # - `step` is the sampled step id from ops_trace.csv (prefill=-1, decode=0..N-1)
        step_i = int(step)
        seg = SegmentSig(device_type=dev_type, phase=str(phase), step=int(step_i), ops=tuple(ops))
        if dev_type_n == "npu":
            total_lat = _try_lookup_segment_latency(seg, gpu_res)
        elif dev_type_n == "pim":
            total_lat = _try_lookup_segment_latency(seg, pim_res)
        else:
            total_lat = None

        if total_lat is None:
            if allow_missing:
                total_lat = 0.0
            else:
                missing += 1
                total_lat = 0.0

        wsum = float(np.nansum(w.astype(np.float64)))
        if wsum > 0:
            dur_s[idxs] = float(total_lat) * (w / wsum)
        else:
            dur_s[idxs] = float(total_lat) / float(len(idxs))

    return dur_s, int(missing)


def _simulate_block_overlap(df_block: pd.DataFrame, dur_s: np.ndarray) -> float:
    """
    Simulate one "block" (either prefill or one decode sampled step) with:
      - layer-by-layer sequential dependency (Transformer layers are sequential)
      - within each layer: sharded ops (S0..Sk) can overlap; shardless ops (S=-1) sync with all shards
      - per-resource serialization: each device is a resource; COMM is a separate resource

    df_block must be in the original trace order (strictly follow ops_trace.csv row order).
    """
    if df_block.empty:
        return 0.0

    # Precompute comm mask for block
    op_l = df_block["op"].astype(str).str.strip().str.lower()
    is_comm = op_l.isin(_COMM_OPS).to_numpy()

    idxs = df_block.index.to_numpy()
    layers = df_block["layer"].astype(int).to_numpy()
    shards = df_block["shard"].astype(int).to_numpy()
    devices = df_block["device"].astype(str).to_numpy()

    # Resource timelines
    res_t: Dict[str, float] = {}

    def get_res_time(res: str) -> float:
        return float(res_t.get(res, 0.0))

    def set_res_time(res: str, t: float) -> None:
        res_t[res] = float(t)

    prev_layer_done = 0.0
    cur_layer: Optional[int] = None
    layer_max = 0.0
    global_chain = 0.0
    shard_chain: Dict[int, float] = {}

    for i in range(len(idxs)):
        idx = int(idxs[i])
        layer = int(layers[i])
        shard = int(shards[i])

        # layer boundary
        if cur_layer is None:
            cur_layer = layer
            global_chain = prev_layer_done
            shard_chain.clear()
            layer_max = prev_layer_done
        elif layer != cur_layer:
            prev_layer_done = layer_max
            cur_layer = layer
            global_chain = prev_layer_done
            shard_chain.clear()
            layer_max = prev_layer_done

        # resource
        res = "COMM" if bool(is_comm[i]) else str(devices[i])

        d = float(dur_s[idx])

        # dependency inside layer
        if shard < 0:
            shard_max = max(shard_chain.values()) if shard_chain else prev_layer_done
            deps = max(global_chain, shard_max)
        else:
            deps = max(global_chain, shard_chain.get(shard, prev_layer_done))

        st = max(get_res_time(res), deps)
        et = st + d
        set_res_time(res, et)

        if shard < 0:
            global_chain = et
        else:
            shard_chain[shard] = et

        if et > layer_max:
            layer_max = et

    # finish last layer
    prev_layer_done = max(prev_layer_done, layer_max)
    return float(prev_layer_done)

def compute_trace_times_from_ops_csv(df: pd.DataFrame, *, decode_stride: int = 1) -> Dict[str, float]:
    """Compute prefill/decode e2e time from the original ops_trace.csv timeline."""

    if df is None or df.empty:
        return {
            "trace_prefill_s": 0.0,
            "trace_decode_s": 0.0,
            "trace_decode_scaled_s": 0.0,
            "trace_total_raw_s": 0.0,
            "trace_total_scaled_s": 0.0,
        }

    # Best-effort numeric conversion
    start = pd.to_numeric(df.get("start"), errors="coerce")
    end = pd.to_numeric(df.get("end"), errors="coerce")

    def _span(mask: np.ndarray) -> float:
        if mask is None or mask.size == 0 or (not mask.any()):
            return 0.0
        st = float(np.nanmin(start[mask].to_numpy(dtype=np.float64)))
        en = float(np.nanmax(end[mask].to_numpy(dtype=np.float64)))
        if not np.isfinite(st) or not np.isfinite(en):
            return 0.0
        return float(max(0.0, en - st))

    ph = df.get("phase")
    if ph is None:
        pre = 0.0
        dec = 0.0
    else:
        ph_s = ph.astype(str)
        pre = _span((ph_s == "prefill").to_numpy())
        dec = _span((ph_s == "decode").to_numpy())

    total_raw = _span(np.ones(len(df), dtype=bool))

    ds = int(decode_stride) if decode_stride is not None else 1
    if ds <= 0:
        ds = 1
    total = float(pre + dec)

    return {
        "trace_prefill_s": float(pre),
        "trace_decode_s": float(dec),
        "trace_decode_scaled_s": float(dec),
        "trace_total_raw_s": float(total_raw),
        "trace_total_scaled_s": float(total),
        "trace_total_s": float(total),
    }

def merge(
    schedule_paths: List[str],
    gpu_results_json: Optional[str],
    pim_results_json: Optional[str],
    *,
    comms_paths: Optional[List[str]] = None,
    non_overlap: float = 0.0,
    comm_model: str = "schedule",
    pcie_lanes: int = 16,
    decode_stride: int = 1,
    kv_load_bw_gbs_override: Optional[float] = None,
    kv_dtype_bytes_override: Optional[float] = None,
    kv_load_overhead_us_override: Optional[float] = None,
    n_kv_heads_override: Optional[int] = None,
    batch_override: Optional[int] = None,
    out_csv: str,
    out_steps_csv: Optional[str] = None,
    allow_missing: bool = False,
    debug: bool = False,
    debug_txt: Optional[str] = None,
) -> None:
    """Merge schedule trace(s) + measured segment latencies into a layer-scope estimate.

    verification logic:
      1) Schedule comm ops such as K_write/V_write, allreduce, reduce, scatter, etc are NOT exported.
         During merge we account for them directly from the schedule, combined using the schedule overlap
         dimension (device-wise max within a layer).
      2) From comms trace(s) (optional):
           - weight_load: treated as a one-time cost and added into prefill_time_s directly (no NON_OVERLAP).
           - kv_load: accumulated per (phase,step,layer) and added with coarse-grained overlap:
                 layer_time = base_layer_time + NON_OVERLAP * kv_load_time
         where base_layer_time is the overlapped makespan across compute + schedule comm ops inside that layer.
         If comm traces are not provided (or do not contain kv_load), we can optionally *estimate* kv_load from
         the schedule's QK/SV ops using cfg.kv_load_bw_gbs/kv_dtype_bytes/kv_load_overhead_us.
      3) GPU and PIM compute are not summed. For each layer, compute time is the maximum across devices
         (typically max(GPU, PIM)).

    decode_stride is applied as the sampling stride: each sampled decode step represents decode_stride tokens.
    """

    if decode_stride < 1:
        raise ValueError(f"decode_stride must be >=1 (got {decode_stride})")
    if non_overlap < 0.0:
        raise ValueError(f"non_overlap must be >=0 (got {non_overlap})")

    # If caller enabled --debug but did not provide --debug-txt, default to <out_csv>.debug.txt
    if debug and (not debug_txt):
        try:
            oc = Path(out_csv).expanduser()
            if str(oc).lower().endswith(".csv"):
                debug_txt = str(oc)[:-4] + ".debug.txt"
            else:
                debug_txt = str(oc) + ".debug.txt"
        except Exception:
            debug_txt = str(out_csv) + ".debug.txt"

    # ---------- load benchmark results ----------
    gpu_res: Dict[str, float] = {}
    pim_res: Dict[str, float] = {}
    cfg_meta: Dict[str, Any] = {}

    gpu_data = _load_json(gpu_results_json)
    pim_data = _load_json(pim_results_json)

    # Optional structured debug payloads from run-gpu/run-pim
    gpu_dbg_map: Dict[str, Any] = {}
    pim_dbg_map: Dict[str, Any] = {}
    try:
        if isinstance(gpu_data.get("debug_segments"), dict):
            gpu_dbg_map = gpu_data.get("debug_segments") or {}
    except Exception:
        gpu_dbg_map = {}
    try:
        if isinstance(pim_data.get("debug_segments"), dict):
            pim_dbg_map = pim_data.get("debug_segments") or {}
    except Exception:
        pim_dbg_map = {}

    def _coerce_results(d: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in (d.get("results") or {}).items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

    gpu_res = _coerce_results(gpu_data)
    pim_res = _coerce_results(pim_data)

    # Prefer config from whichever results file exists.
    if isinstance(gpu_data.get("config"), dict):
        cfg_meta = gpu_data["config"]
    elif isinstance(pim_data.get("config"), dict):
        cfg_meta = pim_data["config"]

    dim = int(cfg_meta.get("dim", 4096))
    shards = int(cfg_meta.get("shards", 1))
    shard_policy = str(cfg_meta.get("shard_policy", "fine")).strip().lower()
    use_coarse = shard_policy in ("coarse", "coarse_majority", "coarse-majority", "majority", "coarsemajority")

    # Optional merge-time overrides for kv_load estimate knobs.
    # These are useful when you want to re-merge with a different KV load bandwidth/dtype
    # without regenerating benchmark results.
    if kv_load_bw_gbs_override is not None:
        try:
            cfg_meta["kv_load_bw_gbs"] = float(kv_load_bw_gbs_override)
        except Exception:
            pass
    if kv_dtype_bytes_override is not None:
        try:
            cfg_meta["kv_dtype_bytes"] = float(kv_dtype_bytes_override)
        except Exception:
            pass
    if kv_load_overhead_us_override is not None:
        try:
            cfg_meta["kv_load_overhead_us"] = float(kv_load_overhead_us_override)
        except Exception:
            pass
    if n_kv_heads_override is not None:
        try:
            cfg_meta["n_kv_heads"] = int(n_kv_heads_override)
        except Exception:
            pass
    if batch_override is not None:
        try:
            cfg_meta["batch"] = int(batch_override)
        except Exception:
            pass

    # ---------- helpers ----------
    comms_paths = [p for p in (comms_paths or []) if p]

    def _pick_comms_for_schedule(i: int) -> List[str]:
        if not comms_paths:
            return []
        if len(comms_paths) == 1:
            return comms_paths
        if len(comms_paths) == len(schedule_paths):
            return [comms_paths[i]]
        # fallback: apply all
        return comms_paths
    def _lookup_weight_load_from_results(schedule_abs: str) -> float:
        """Best-effort total weight_load seconds for a given schedule, from results JSON meta."""
        for d in (gpu_data, pim_data):
            try:
                m = d.get("weight_load_by_schedule") or {}
            except Exception:
                m = {}
            rec = m.get(schedule_abs)
            if isinstance(rec, dict):
                try:
                    if rec.get("total_s") is not None:
                        return float(rec["total_s"])
                except Exception:
                    pass
                try:
                    return float(max(float(rec.get("gpu_s", 0.0) or 0.0), float(rec.get("pim_s", 0.0) or 0.0)))
                except Exception:
                    return 0.0

        g = 0.0
        p = 0.0
        try:
            g = float(gpu_data.get("weight_load_s", 0.0) or 0.0)
        except Exception:
            g = 0.0
        try:
            p = float(pim_data.get("weight_load_s", 0.0) or 0.0)
        except Exception:
            p = 0.0
        return float(max(g, p))

    def _lookup_latency(dev_type: str, phase: str, step_sample: int, ops: Tuple[Tuple[str, int], ...],) -> Optional[float]:
        dev = dev_type.strip().lower()
        if dev == "npu":
            table = gpu_res
        elif dev == "pim":
            table = pim_res
        else:
            return None

        k = SegmentSig(device_type=dev, phase=phase, step=int(step_sample), ops=ops).to_key()
        if k in table:
            return float(table[k])
        return None

    def _lookup_latency_with_key(
        dev_type: str,
        phase: str,
        step_sample: int,
        ops: Tuple[Tuple[str, int], ...],
    ) -> Tuple[Optional[float], Optional[str], Optional[int], str]:
        """Like _lookup_latency(), but also returns the matched key + matched step.

        Returns: (lat_s, key, step_used, mode)
          mode in {'sample_step','miss','unknown_dev'}
        """
        dev = dev_type.strip().lower()
        if dev == "npu":
            table = gpu_res
        elif dev == "pim":
            table = pim_res
        else:
            return (None, None, None, "unknown_dev")

        k = SegmentSig(device_type=dev, phase=phase, step=int(step_sample), ops=ops).to_key()
        if k in table:
            return (float(table[k]), str(k), int(step_sample), "sample_step")
        return (None, None, None, "miss")

    def _devicewise_max_sum(g: pd.DataFrame, value_col: str, *, group_cols: List[str], device_col: str = "device",) -> Dict[Tuple[str, int, int], float]:
        """Sum within each device, then take max across devices for each (phase,step,layer)."""
        if g.empty:
            return {}
        by_dev = (g.groupby(group_cols + [device_col], sort=False)[value_col].sum().reset_index())
        by_key = by_dev.groupby(group_cols, sort=False)[value_col].max()
        return {(ph, int(st), int(ly)): float(v) for (ph, st, ly), v in by_key.items()}

    # ---------- main loop ----------
    rows_out: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    # Optional: debug CSV outputs (segment-level + row-level) emitted when --debug.
    seg_debug_frames: List[pd.DataFrame] = []
    row_debug_frames: List[pd.DataFrame] = []

    dbg_f = None
    if debug and debug_txt:
        dbg_f = open(debug_txt, "w", encoding="utf-8")

    for i, sp in enumerate(schedule_paths):
        p = resolve_existing_path(sp)
        df = load_schedule_csv(str(p))

        # Decode stride expansion policy (matches simulator token0 special-case when present)
        decode_scale_by_step, decode_scale_mode, decode_len_hint = compute_decode_step_scale_map(
            df, str(p), int(decode_stride)
        )

        comms_for_sched = _pick_comms_for_schedule(i)
        weight_load_prefill_s = 0.0
        if comms_for_sched:
            wl = extract_weight_load_seconds(comms_for_sched, decode_stride=int(decode_stride))
            weight_load_prefill_s = float(wl.get("total_s", 0.0) or 0.0)
        else:
            weight_load_prefill_s = _lookup_weight_load_from_results(str(p))

        # kv_load: per-layer comm; overlap controlled by NON_OVERLAP
        comm_tbl = extract_layer_comm_seconds(comms_for_sched, df, tags=("kv_load",))
        kv_map_trace = {
            (str(r.phase), int(r.step), int(r.layer)): float(getattr(r, "kv_load_s", 0.0))
            for r in comm_tbl.itertuples(index=False)
        }

        # Optional: estimate kv_load from QK/SV ops when comm trace is missing.
        kv_map_est: Dict[Tuple[str, int, int], float] = {}
        try:
            cfg_kv = WorkloadConfig.from_dict(cfg_meta or {})
            cfg_kv.decode_stride = int(decode_stride)
            kv_map_est = estimate_kv_load_seconds_from_ops_csv(df, cfg_kv, decode_stride=int(decode_stride))
        except Exception:
            kv_map_est = {}

        # Merge policy: use comm-trace kv_load when present (>0), otherwise fall back to estimate.
        kv_map: Dict[Tuple[str, int, int], float] = dict(kv_map_trace)
        if kv_map_est:
            for kk, vv in kv_map_est.items():
                if float(kv_map.get(kk, 0.0)) <= 0.0:
                    kv_map[kk] = float(vv)

        # Compute times per (phase,step,layer) for GPU/PIM.
        gpu_layer: Dict[Tuple[str, int, int], float] = {}
        pim_layer: Dict[Tuple[str, int, int], float] = {}
        missing_segments = 0

        if use_coarse:
            # Coarse-grained shard placement: decide GPU vs PIM per sharded op by majority,
            # then build one GPU segment and/or one PIM segment per (phase,step,layer).
            for phase, step, layer, dev_type, ops in _iter_layer_segments_coarse_majority(df, total_shards=int(shards)):
                lat = _lookup_latency(str(dev_type), str(phase), int(step), tuple(ops))
                if lat is None:
                    missing_segments += 1
                    if not allow_missing and dbg_f:
                        dbg_f.write(
                            f"[MISSING] {p.name} phase={phase} step={step} layer={layer} dev={dev_type} ops={len(ops)} (coarse)\n"
                        )
                    lat = 0.0

                key = (str(phase), int(step), int(layer))
                if str(dev_type).strip().lower() == "npu":
                    gpu_layer[key] = max(gpu_layer.get(key, 0.0), float(lat))
                elif str(dev_type).strip().lower() == "pim":
                    pim_layer[key] = max(pim_layer.get(key, 0.0), float(lat))
        else:
            comp = df[df["device_type"].astype(str).str.lower().isin(["npu", "pim"])].copy()
            for (phase, step, layer, device), g in comp.groupby(["phase", "step", "layer", "device"], sort=False):
                dev_type = str(g["device_type"].iloc[0]).strip().lower()
                g = g.sort_values(["start", "_row"], kind="mergesort")
                ops: List[Tuple[str, int]] = []
                for _, r in g.iterrows():
                    op = str(r["op"]).strip()
                    if op.lower() in _COMM_OPS:
                        continue
                    ops.append((op, int(r["shard"]) if pd.notna(r["shard"]) else -1))
                if not ops:
                    continue

                lat = _lookup_latency(dev_type, str(phase), int(step), tuple(ops))
                if lat is None:
                    missing_segments += 1
                    if not allow_missing and dbg_f:
                        dbg_f.write(f"[MISSING] {p.name} phase={phase} step={step} layer={layer} dev={dev_type} ops={len(ops)}\n")
                    lat = 0.0

                key = (str(phase), int(step), int(layer))
                if dev_type == "npu":
                    gpu_layer[key] = max(gpu_layer.get(key, 0.0), float(lat))
                elif dev_type == "pim":
                    pim_layer[key] = max(pim_layer.get(key, 0.0), float(lat))

        # ---------- debug: segment-level and row-level diffs ----------
        if debug:
            try:
                # Segment-level diff: compare schedule's own compute durations (ops.csv) vs measured latency.
                # Note: schedule durations are for sampled decode steps; we include both raw and scaled (x decode_stride).
                op_l = df["op"].astype(str).str.strip().str.lower()
                is_comm = op_l.isin(_COMM_OPS)
                comp_all = df[~is_comm].copy()
                comp_all["duration_s"] = pd.to_numeric(comp_all["duration"], errors="coerce").fillna(0.0).astype(float)

                seg_rows: List[Dict[str, Any]] = []

                if use_coarse:
                    # Rebuild coarse-majority segments with trace sums.
                    total_shards = int(shards)
                    comp_np = comp_all[comp_all["device_type"].astype(str).str.lower().isin(["npu", "pim"])].copy()
                    for (phase, step, layer), g_all in comp_np.groupby(["phase", "step", "layer"], sort=False):
                        # Group by (node_id, op) to find per-op majority device.
                        g_all = g_all.sort_values(["start", "_row"], kind="mergesort")
                        items: List[Tuple[Tuple[Any, Any], pd.DataFrame, int, int]] = []
                        for (nid, op), gg in g_all.groupby(["node_id", "op"], sort=False):
                            dtypes = gg["device_type"].astype(str).str.lower().tolist()
                            npu_ct = sum(1 for x in dtypes if x == "npu")
                            pim_ct = sum(1 for x in dtypes if x == "pim")
                            items.append(((nid, op), gg, npu_ct, pim_ct))

                        # Accumulate per pseudo segment.
                        gpu_ops: List[Tuple[str, int]] = []
                        pim_ops: List[Tuple[str, int]] = []
                        gpu_trace_s = 0.0
                        pim_trace_s = 0.0
                        for (nid_op, gg, npu_ct, pim_ct) in items:
                            # Choose by majority: GPU if > half shards, else PIM.
                            chosen = "npu" if npu_ct > (total_shards / 2.0) else "pim"
                            gg = gg.sort_values(["start", "_row"], kind="mergesort")
                            op_name = str(gg["op"].iloc[0]).strip()
                            # In coarse mode, we attribute the *sum across shards* to the chosen device.
                            trace_sum = float(pd.to_numeric(gg["duration"], errors="coerce").fillna(0.0).sum())
                            shard_vals = gg["shard"].tolist()
                            shard_set = sorted({int(s) for s in shard_vals if pd.notna(s)})
                            is_sharded = (len(shard_set) >= 2)
                            if chosen == "npu":
                                gpu_trace_s += trace_sum
                                if is_sharded:
                                    for s in range(total_shards):
                                        gpu_ops.append((op_name, int(s)))
                                else:
                                    gpu_ops.append((op_name, -1))
                            else:
                                pim_trace_s += trace_sum
                                pim_ops.append((op_name, -1))

                        for dev_type, ops, trace_s in (("npu", gpu_ops, gpu_trace_s), ("pim", pim_ops, pim_trace_s)):
                            if not ops:
                                continue
                            ops_t = tuple(ops)
                            lat_s, lat_key, step_used, mode = _lookup_latency_with_key(dev_type, str(phase), int(step), ops_t)
                            if str(phase) == "decode":
                                scale = float(decode_scale_by_step.get(int(step), int(decode_stride)))
                                tok_idx = int(decode_token_index_from_sample_step(int(step), int(decode_stride)))
                            else:
                                scale = 1.0
                                tok_idx = int(step)
                            seg_rows.append(
                                {
                                    "schedule": p.name,
                                    "phase": str(phase),
                                    "sample_step": int(step),
                                    "token_index": int(tok_idx),
                                    "layer": int(layer),
                                    "device": "<coarse>",
                                    "device_type": str(dev_type),
                                    "n_ops": int(len(ops_t)),
                                    "ops_repr": SegmentSig(device_type=str(dev_type), phase=str(phase), step=int(step), ops=ops_t).ops_repr(),
                                    "schedule_sum_s": float(trace_s),
                                    "measured_s": float(lat_s) if lat_s is not None else None,
                                    "measured_key": lat_key,
                                    "measured_step": step_used,
                                    "measured_mode": mode,
                                    "schedule_sum_scaled_s": float(trace_s) * scale,
                                    "measured_scaled_s": float(lat_s) * scale if lat_s is not None else None,
                                }
                            )
                else:
                    # Fine mode: one segment per (phase,step,layer,device) for npu/pim
                    comp_np = comp_all[comp_all["device_type"].astype(str).str.lower().isin(["npu", "pim"])].copy()
                    for (phase, step, layer, device), g in comp_np.groupby(["phase", "step", "layer", "device"], sort=False):
                        dev_type = str(g["device_type"].iloc[0]).strip().lower()
                        g = g.sort_values(["start", "_row"], kind="mergesort")
                        ops: List[Tuple[str, int]] = []
                        for _, r in g.iterrows():
                            op = str(r["op"]).strip()
                            if op.lower() in _COMM_OPS:
                                continue
                            ops.append((op, int(r["shard"]) if pd.notna(r["shard"]) else -1))
                        if not ops:
                            continue
                        ops_t = tuple(ops)
                        trace_s = float(pd.to_numeric(g["duration"], errors="coerce").fillna(0.0).sum())
                        lat_s, lat_key, step_used, mode = _lookup_latency_with_key(dev_type, str(phase), int(step), ops_t)
                        if str(phase) == "decode":
                            scale = float(decode_scale_by_step.get(int(step), int(decode_stride)))
                            tok_idx = int(decode_token_index_from_sample_step(int(step), int(decode_stride)))
                        else:
                            scale = 1.0
                            tok_idx = int(step)
                        seg_rows.append(
                            {
                                "schedule": p.name,
                                "phase": str(phase),
                                "sample_step": int(step),
                                "token_index": int(tok_idx),
                                "layer": int(layer),
                                "device": str(device),
                                "device_type": str(dev_type),
                                "n_ops": int(len(ops_t)),
                                "ops_repr": SegmentSig(device_type=str(dev_type), phase=str(phase), step=int(step), ops=ops_t).ops_repr(),
                                "schedule_sum_s": float(trace_s),
                                "measured_s": float(lat_s) if lat_s is not None else None,
                                "measured_key": lat_key,
                                "measured_step": step_used,
                                "measured_mode": mode,
                                "schedule_sum_scaled_s": float(trace_s) * scale,
                                "measured_scaled_s": float(lat_s) * scale if lat_s is not None else None,
                            }
                        )

                if seg_rows:
                    seg_df = pd.DataFrame(seg_rows)
                    # Derived deltas/ratios
                    seg_df["delta_s"] = seg_df["measured_s"] - seg_df["schedule_sum_s"]
                    seg_df["ratio"] = seg_df["measured_s"] / seg_df["schedule_sum_s"].replace({0.0: np.nan})
                    seg_df["delta_scaled_s"] = seg_df["measured_scaled_s"] - seg_df["schedule_sum_scaled_s"]
                    seg_df["ratio_scaled"] = seg_df["measured_scaled_s"] / seg_df["schedule_sum_scaled_s"].replace({0.0: np.nan})
                    seg_debug_frames.append(seg_df)

                    if dbg_f:
                        dbg_f.write(f"[DEBUG] {p.name}: segment diff (top 20 by |delta_scaled_s|)\n")
                        tmp = seg_df.copy()
                        tmp = tmp[tmp["measured_scaled_s"].notna() & (tmp["schedule_sum_scaled_s"] > 0.0)]
                        tmp["abs_delta"] = (tmp["delta_scaled_s"]).abs()
                        tmp = tmp.sort_values(["abs_delta"], ascending=False).head(20)

                        def _summ_ops_breakdown(dev: str, key: Optional[str]) -> str:
                            if not key:
                                return ""
                            m = gpu_dbg_map if str(dev).lower() == "npu" else pim_dbg_map
                            rec = m.get(str(key)) if isinstance(m, dict) else None
                            if not isinstance(rec, dict):
                                return ""
                            per = rec.get("per_op")
                            if not isinstance(per, list) or not per:
                                return ""
                            agg: Dict[str, float] = {}
                            for it in per:
                                try:
                                    nm = str(it.get("op_norm") or it.get("op") or "?")
                                    agg[nm] = agg.get(nm, 0.0) + float(it.get("latency_s", 0.0) or 0.0)
                                except Exception:
                                    continue
                            top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:5]
                            return ", ".join([f"{k}={v*1e3:.2f}ms" for k, v in top])

                        for _, rr in tmp.iterrows():
                            dbg_f.write(
                                f"  phase={rr['phase']} step={int(rr['sample_step'])} tok={int(rr['token_index'])} layer={int(rr['layer'])} "
                                f"dev={rr['device_type']} device={rr['device']} n_ops={int(rr['n_ops'])} "
                                f"sched={float(rr['schedule_sum_scaled_s']):.6f}s meas={(float(rr['measured_scaled_s']) if pd.notna(rr['measured_scaled_s']) else float('nan')):.6f}s "
                                f"delta={float(rr['delta_scaled_s']):+.6f}s ratio={float(rr['ratio_scaled'] if pd.notna(rr['ratio_scaled']) else float('nan')):.3f} "
                                f"mode={rr['measured_mode']}\n"
                            )
                            bd = _summ_ops_breakdown(str(rr["device_type"]), rr.get("measured_key"))
                            if bd:
                                dbg_f.write(f"    per-op(top5): {bd}\n")

                # Row-level diff (fine only): compare each ops.csv row duration vs merged row duration.
                if (not use_coarse):
                    try:
                        dur_s, missing_row = _build_row_durations_layer_scope(
                            df,
                            gpu_res=gpu_res,
                            pim_res=pim_res,
                            comm_model=str(comm_model),
                            pcie_lanes=int(pcie_lanes),
                            decode_stride=int(decode_stride),
                            dim=int(dim),
                            shards=int(shards),
                            allow_missing=True,
                        )
                        raw_s = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).astype(float)
                        phase_s = df["phase"].astype(str)
                        step_s = pd.to_numeric(df.get("step"), errors="coerce").fillna(-1).astype(int)
                        scale = np.ones(len(df), dtype=float)
                        dec_mask = (phase_s == "decode").to_numpy(dtype=bool)
                        if dec_mask.any():
                            st_arr = step_s.to_numpy(dtype=int)
                            scale[dec_mask] = np.array([
                                float(decode_scale_by_step.get(int(s), int(decode_stride))) for s in st_arr[dec_mask]
                            ], dtype=float)
                        sched_scaled = raw_s.to_numpy(dtype=float) * scale
                        merged_scaled = np.asarray(dur_s, dtype=float) * scale

                        row_df = df.copy()
                        row_df["schedule"] = p.name
                        row_df["schedule_duration_s"] = raw_s
                        row_df["schedule_duration_scaled_s"] = sched_scaled
                        row_df["merged_duration_scaled_s"] = merged_scaled
                        row_df["delta_scaled_s"] = row_df["merged_duration_scaled_s"] - row_df["schedule_duration_scaled_s"]
                        denom = row_df["schedule_duration_scaled_s"].replace({0.0: np.nan})
                        row_df["ratio_scaled"] = row_df["merged_duration_scaled_s"] / denom
                        row_df["missing_segments_row_scope"] = int(missing_row)
                        row_debug_frames.append(row_df)

                        if dbg_f:
                            dbg_f.write(
                                f"[DEBUG] {p.name}: row diff (top 20 by |delta_scaled_s|, fine mode only; missing_segments_row_scope={int(missing_row)})\n"
                            )
                            tmp2 = row_df.copy()
                            tmp2["abs_delta"] = tmp2["delta_scaled_s"].abs()
                            tmp2 = tmp2.sort_values(["abs_delta"], ascending=False).head(20)
                            for _, rr in tmp2.iterrows():
                                dbg_f.write(
                                    f"  phase={rr['phase']} step={int(rr['step'])} layer={int(rr['layer'])} dev={rr['device_type']} device={rr['device']} "
                                    f"op={str(rr['op']).strip()} shard={int(rr['shard']) if pd.notna(rr['shard']) else -1} "
                                    f"sched={float(rr['schedule_duration_scaled_s']):.6f}s merged={float(rr['merged_duration_scaled_s']):.6f}s "
                                    f"delta={float(rr['delta_scaled_s']):+.6f}s ratio={float(rr['ratio_scaled'] if pd.notna(rr['ratio_scaled']) else float('nan')):.3f}\n"
                                )
                    except Exception as e:
                        if dbg_f:
                            dbg_f.write(f"[DEBUG] {p.name}: row diff skipped due to error: {e}\n")
            except Exception as e:
                if dbg_f:
                    dbg_f.write(f"[DEBUG] {p.name}: segment/row diff generation failed: {e}\n")

        # Schedule comm ops (kv_write / allreduce / reduce / scatter / etc)
        op_l = df["op"].astype(str).str.strip().str.lower()
        comm_ops_df = df[op_l.isin(_COMM_OPS)].copy()
        if comm_model.lower() == "none" or comm_ops_df.empty:
            comm_ops_df["lat_s"] = 0.0
        elif comm_model.lower() == "schedule":
            comm_ops_df["lat_s"] = pd.to_numeric(comm_ops_df["duration"], errors="coerce").fillna(0.0).astype(float)
        else:
            raise ValueError(f"unknown comm_model: {comm_model}")
        comm_ops_layer = _devicewise_max_sum(comm_ops_df, "lat_s", group_cols=["phase", "step", "layer"])

        # Schedule other (non-GPU/PIM) ops that are NOT classified as comm ops.
        other_df = df[(~df["device_type"].astype(str).str.lower().isin(["npu", "pim"])) & (~op_l.isin(_COMM_OPS))].copy()
        other_layer = _devicewise_max_sum(other_df, "duration", group_cols=["phase", "step", "layer"])

        # Aggregate per-phase/per-step times by summing layers (layers are sequential).
        prefill_time_s = 0.0
        decode_time_s = 0.0

        gpu_busy_s = 0.0
        pim_busy_s = 0.0
        sched_comm_s = 0.0
        sched_other_s = 0.0
        comm_added_s = 0.0  # after NON_OVERLAP and scaling
        comm_kv_s = 0.0

        # Prefill (step = -1)
        pre_layers = sorted(df[df["phase"] == "prefill"]["layer"].unique())
        for ly in pre_layers:
            k = ("prefill", -1, int(ly))
            gpu_s = float(gpu_layer.get(k, 0.0))
            pim_s = float(pim_layer.get(k, 0.0))
            compute_max = max(gpu_s, pim_s)
            comm_s = float(comm_ops_layer.get(k, 0.0))
            oth_s = float(other_layer.get(k, 0.0))
            base = max(compute_max, comm_s, oth_s)
            kv_trace = float(kv_map.get(k, 0.0))
            add_s = non_overlap * kv_trace

            layer_s = base + add_s
            prefill_time_s += layer_s

            gpu_busy_s += gpu_s
            pim_busy_s += pim_s
            sched_comm_s += comm_s
            sched_other_s += oth_s
            comm_added_s += add_s
            comm_kv_s += add_s

         # weight_load is charged to prefill as a one-time overhead
        prefill_time_s += weight_load_prefill_s

        dec = df[df["phase"] == "decode"]
        dec_steps = sorted(dec["step"].unique())
        dec_layers = sorted(dec["layer"].unique())

        for st in dec_steps:
            step_s_per_token = 0.0
            gpu_step = 0.0
            pim_step = 0.0
            comm_step = 0.0
            oth_step = 0.0
            add_step = 0.0

            for ly in dec_layers:
                k = ("decode", int(st), int(ly))
                gpu_s = float(gpu_layer.get(k, 0.0))
                pim_s = float(pim_layer.get(k, 0.0))
                compute_max = max(gpu_s, pim_s)
                comm_s = float(comm_ops_layer.get(k, 0.0))
                oth_s = float(other_layer.get(k, 0.0))
                base = max(compute_max, comm_s, oth_s)

                kv_trace = float(kv_map.get(k, 0.0))
                add_s = non_overlap * kv_trace

                layer_s = base + add_s
                step_s_per_token += layer_s

                gpu_step += gpu_s
                pim_step += pim_s
                comm_step += comm_s
                oth_step += oth_s
                add_step += add_s

                # per-tag breakdown (after NON_OVERLAP, before scaling)
                comm_kv_s += add_s

            scale_tokens = int(decode_scale_by_step.get(int(st), int(decode_stride)))
            step_s = step_s_per_token * float(scale_tokens)
            decode_time_s += step_s

            gpu_busy_s += gpu_step * float(scale_tokens)
            pim_busy_s += pim_step * float(scale_tokens)
            sched_comm_s += comm_step * float(scale_tokens)
            sched_other_s += oth_step * float(scale_tokens)
            comm_added_s += add_step * float(scale_tokens)

            step_rows.append(
                {
                    "schedule": p.name,
                    "phase": "decode",
                    "sample_step": int(st),
                    "token_index": int(decode_token_index_from_sample_step(int(st), int(decode_stride))),
                    "tokens": int(scale_tokens),
                    "step_time_s": float(step_s),
                    "step_time_per_token_s": float(step_s_per_token),
                }
            )

        total_time_s = prefill_time_s + decode_time_s

        trace_t = compute_trace_times_from_ops_csv(df, decode_stride=decode_stride)
        trace_prefill_s = float(trace_t.get("trace_prefill_s", 0.0))
        trace_decode_s = float(trace_t.get("trace_decode_scaled_s", trace_t.get("trace_decode_s", 0.0)))
        trace_total_s = float(trace_t.get("trace_total_scaled_s", trace_t.get("trace_total_s", 0.0)))

        row = {
            "schedule": p.name,
            "dim": dim,
            "shards": shards,
            "decode_stride": int(decode_stride),
            "non_overlap": float(non_overlap),
            "prefill_time_s": float(prefill_time_s),
            "decode_time_s": float(decode_time_s),
            "total_time_s": float(total_time_s),
            "trace_prefill_s": float(trace_prefill_s),
            "trace_decode_s": float(trace_decode_s),
            "trace_total_s": float(trace_total_s),
            "delta_prefill_s": float(prefill_time_s - trace_prefill_s),
            "delta_decode_s": float(decode_time_s - trace_decode_s),
            "delta_total_s": float(total_time_s - trace_total_s),
            "gpu_busy_s": float(gpu_busy_s),
            "pim_busy_s": float(pim_busy_s),
            "schedule_comm_s": float(sched_comm_s),
            "schedule_other_s": float(sched_other_s),
            "comm_added_s": float(comm_added_s),
            "weight_load_prefill_s": float(weight_load_prefill_s),
            "comm_kv_load_added_s": float(comm_kv_s),
            "missing_segments": int(missing_segments),
        }
        rows_out.append(row)

        if dbg_f:
            dbg_f.write(
                f"[SCHEDULE] {p.name} prefill={prefill_time_s:.6f} (weight_load={weight_load_prefill_s:.6f}) decode={decode_time_s:.6f} total={total_time_s:.6f} "
                f"missing_segments={missing_segments}\n"
            )

    if dbg_f:
        dbg_f.close()

    # Write debug CSV outputs (only when --debug)
    if debug:
        try:
            out_base = str(Path(out_csv).expanduser())
        except Exception:
            out_base = str(out_csv)
        if out_base.lower().endswith(".csv"):
            seg_path = out_base[:-4] + ".seg_debug.csv"
            row_path = out_base[:-4] + ".row_debug.csv"
        else:
            seg_path = out_base + ".seg_debug.csv"
            row_path = out_base + ".row_debug.csv"
        try:
            if seg_debug_frames:
                os.makedirs(os.path.dirname(seg_path) or ".", exist_ok=True)
                pd.concat(seg_debug_frames, ignore_index=True).to_csv(seg_path, index=False)
        except Exception:
            pass
        try:
            if row_debug_frames:
                os.makedirs(os.path.dirname(row_path) or ".", exist_ok=True)
                pd.concat(row_debug_frames, ignore_index=True).to_csv(row_path, index=False)
        except Exception:
            pass

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)

    if out_steps_csv:
        os.makedirs(os.path.dirname(out_steps_csv) or ".", exist_ok=True)
        pd.DataFrame(step_rows).to_csv(out_steps_csv, index=False)

def collect_merge_csv(list_tsv: str, *, out_csv: str, skip_missing: bool = False) -> None:
    """Concatenate multiple per-run merge.csv files into one combined CSV."""
    lp = resolve_existing_path(list_tsv)
    try:
        lst = pd.read_csv(str(lp), sep="\t")
    except Exception as e:
        raise ValueError(f"Failed to read list TSV: {lp}") from e

    if lst is None or lst.empty:
        raise ValueError(f"merge list is empty: {lp}")

    # Find merge_csv column (recommended name)
    merge_col = None
    if "merge_csv" in lst.columns:
        merge_col = "merge_csv"
    else:
        # best effort: any column containing 'merge' and 'csv'
        for c in lst.columns:
            cl = str(c).lower()
            if "merge" in cl and "csv" in cl:
                merge_col = c
                break
    if merge_col is None:
        raise ValueError(f"{lp}: missing required column 'merge_csv' (columns={list(lst.columns)})")

    frames: List[pd.DataFrame] = []
    missing: List[str] = []

    for _, r in lst.iterrows():
        m = str(r.get(merge_col, "")).strip()
        if not m or m.lower() in ("nan", "none", "-"):
            continue
        mp = Path(m).expanduser()
        if not mp.is_absolute():
            mp = (lp.parent / mp).resolve()

        if not mp.exists():
            if skip_missing:
                missing.append(str(mp))
                continue
            raise FileNotFoundError(f"merge_csv not found: {mp}")

        df = pd.read_csv(str(mp))
        # Always keep the physical path for traceability
        if "merge_csv" not in df.columns:
            df["merge_csv"] = str(mp)

        # Attach metadata columns from list file
        for c in lst.columns:
            if c == merge_col:
                continue
            meta_name = c if c not in df.columns else f"meta_{c}"
            df[meta_name] = r.get(c, "")
        frames.append(df)

    if not frames:
        raise ValueError(f"No merge CSVs found to collect from: {lp}")

    out = pd.concat(frames, ignore_index=True)

    outp = Path(out_csv).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(str(outp), index=False)

    if missing:
        print(f"[collect-merge] skipped {len(missing)} missing merge csv(s) (skip_missing=1)")


# ==========================================================
# CLI
# ==========================================================

def add_common_model_args(p: argparse.ArgumentParser) -> None:

    p.add_argument("--cfg", type=str, default=None, help="model shape json, e.g. ./configs/llama_7b_shape.json")
    p.add_argument("--prefill-len", type=int, default=128, dest="prefill_len")
    p.add_argument("--decode-context-lens", type=str, nargs="?", const="auto", default=None,)

    p.add_argument("--shard-policy", type=str, default="coarse_majority", choices=["fine","coarse_majority"],
                   help="shard placement policy for verification: fine keeps per-shard placement; coarse_majority chooses GPU if > half shards on GPU else PIM")

    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu (GPU benchmark side)")
    p.add_argument("--gpu-dtype", type=str, default="fp16", dest="gpu_dtype")

    p.add_argument(
        "--kv-load-bw-gbs",
        type=float,
        default=0.0,
        help="Estimated KV-cache load bandwidth in GB/s (decode). Set >0 to enable implicit kv_load before QK/SV.",
    )
    p.add_argument(
        "--kv-dtype-bytes",
        type=float,
        default=2.0,
        dest="kv_dtype_bytes",
        help="Bytes per KV element for kv_load estimate (per K or per V). Default=2 (fp16/bf16).",
    )
    p.add_argument(
        "--kv-load-overhead-us",
        type=float,
        default=0.0,
        dest="kv_load_overhead_us",
        help="Fixed overhead per KV-load event (per shard/op), in microseconds.",
    )
    p.add_argument(
        "--n-kv-heads",
        type=int,
        default=None,
        dest="n_kv_heads",
        help="Optional KV head count for GQA/MQA. If omitted, defaults to n_heads.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        dest="batch",
        help="Optional batch size for KV-load bytes estimate (default=1).",
    )

    # PIM config
    p.add_argument("--pim-dram-column", type=int, default=256)
    p.add_argument("--pim-dram-row", type=int, default=64)
    p.add_argument("--pim-burst-length", type=int, default=16)
    p.add_argument("--pim-num-banks", type=int, default=8)
    p.add_argument("--pim-num-channels", type=int, default=4)
    p.add_argument("--pim-threads", type=int, default=1)
    p.add_argument("--pim-reuse-size", type=int, default=32)

    p.add_argument("--pim-num-devices", type=int, default=1, help="PIM device/DIMM count")
    p.add_argument("--pim-ramulator-bin", type=str, default=None, help="path/name of AiM-enabled ramulator2 executable (required for run-pim)",)
    p.add_argument("--pim-ramulator-config", type=str, default=None, help="ramulator2 config file (required for run-pim)")
    p.add_argument("--pim-freq-ghz", type=float, default=1.0, help="PIM clock frequency in GHz for cycles->seconds")
    p.add_argument("--pim-ramulator-timeout-s", type=int, default=3000, help="ramulator2 timeout per trace (seconds)")
    p.add_argument("--pim-keep-traces", action="store_true", help="keep generated AiM traces (debug)")
    p.add_argument("--pim-trace-dir", type=str, default=None, help="trace output dir when keeping traces")
    p.add_argument("--pim-no-override-ramulator-config", action="store_true",
                   help="do not rewrite channels/banks/devices into ramulator config (NOT recommended)")
    # segmenting
    p.add_argument("--segment-scope", type=str, default="layer", choices=["layer"],
                   help="segment granularity")

def build_cfg_from_args(args: argparse.Namespace) -> WorkloadConfig:
    cfg = WorkloadConfig()

    # ------------------------------
    # Model shape from --cfg
    # ------------------------------
    cfg_path = getattr(args, "cfg", None)
    if cfg_path:
        shape = load_model_shape_json(cfg_path)
        cfg.dim = int(shape["dim"])
        cfg.ffn_dim = int(shape["ffn_dim"])
        cfg.n_heads = int(shape["n_heads"])
    # ------------------------------
    # Schedule lengths / dtype / device / PIM params
    # ------------------------------
    cfg.prefill_len = int(getattr(args, "prefill_len", cfg.prefill_len))
    cfg.decode_context_lens = None
    cfg.device = str(getattr(args, "device", cfg.device))
    cfg.gpu_dtype = str(getattr(args, "gpu_dtype", cfg.gpu_dtype))

    cfg.kv_load_bw_gbs = float(getattr(args, "kv_load_bw_gbs", cfg.kv_load_bw_gbs))
    cfg.kv_dtype_bytes = float(getattr(args, "kv_dtype_bytes", cfg.kv_dtype_bytes))
    cfg.kv_load_overhead_us = float(getattr(args, "kv_load_overhead_us", cfg.kv_load_overhead_us))
    if hasattr(args, "n_kv_heads"):
        v = getattr(args, "n_kv_heads")
        cfg.n_kv_heads = None if v is None else int(v)
    if hasattr(args, "batch"):
        cfg.batch = int(getattr(args, "batch"))

    cfg.pim_dram_column = int(getattr(args, "pim_dram_column", cfg.pim_dram_column))
    cfg.pim_dram_row = int(getattr(args, "pim_dram_row", cfg.pim_dram_row))
    cfg.pim_burst_length = int(getattr(args, "pim_burst_length", cfg.pim_burst_length))
    cfg.pim_num_banks = int(getattr(args, "pim_num_banks", cfg.pim_num_banks))
    cfg.pim_num_channels = int(getattr(args, "pim_num_channels", cfg.pim_num_channels))
    cfg.pim_threads = int(getattr(args, "pim_threads", cfg.pim_threads))
    cfg.pim_reuse_size = int(getattr(args, "pim_reuse_size", cfg.pim_reuse_size))
    cfg.pim_num_devices = int(getattr(args, "pim_num_devices", cfg.pim_num_devices))

    cfg.pim_ramulator_bin = getattr(args, "pim_ramulator_bin", cfg.pim_ramulator_bin)
    cfg.pim_ramulator_config = getattr(args, "pim_ramulator_config", cfg.pim_ramulator_config)
    cfg.pim_freq_ghz = float(getattr(args, "pim_freq_ghz", cfg.pim_freq_ghz))
    cfg.pim_ramulator_timeout_s = int(getattr(args, "pim_ramulator_timeout_s", cfg.pim_ramulator_timeout_s))
    cfg.pim_keep_traces = bool(getattr(args, "pim_keep_traces", cfg.pim_keep_traces))
    cfg.pim_trace_dir = getattr(args, "pim_trace_dir", cfg.pim_trace_dir)

    cfg.segment_scope = str(getattr(args, "segment_scope", cfg.segment_scope))

    cfg.shard_policy = str(getattr(args, "shard_policy", cfg.shard_policy))

    # decode stride (used to generate decode-context-lens / shape inference)
    if hasattr(args, "decode_stride") and getattr(args, "decode_stride") is not None:
        cfg.decode_stride = int(getattr(args, "decode_stride"))

    # decode context lengths
    dcl = getattr(args, "decode_context_lens", None)
    if dcl is not None:
        dcl_s = str(dcl).strip()
        if dcl_s and dcl_s.lower() not in ("auto", "none"):
            cfg.decode_context_lens = [int(x) for x in dcl_s.split(",") if str(x).strip()]
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    subcmds = {"export","run-gpu","run-pim","merge","collect-merge","all"}

    # Backward compatibility: if user calls without subcommand, treat as "all"
    if not argv or argv[0].startswith("-") or argv[0] not in subcmds:
        argv = ["all"] + argv

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # export
    p_exp = sub.add_parser("export", help="export gpu_tasks.json and pim_tasks.json (segment-level) from schedule csv(s)")
    p_exp.add_argument("--schedule", type=str, default=None, help="single schedule csv path")
    p_exp.add_argument("--schedules", type=str, nargs="+", default=None, help="schedule csv paths (one or more)")
    p_exp.add_argument("--comms", type=str, default=None, help="(optional) single comms trace csv path (weight_load/kv_load, layer mapping)")
    p_exp.add_argument("--comms-traces", type=str, nargs="+", default=None,
                       help="(optional) comms trace csv paths (one or more); if same count as --schedules, treated as paired")
    p_exp.add_argument("--out-dir", type=str, default=".", help="output directory")
    p_exp.add_argument("--prefix", type=str, default="tasks", help="output file prefix")
    p_exp.add_argument("--decode-stride", type=int, default=1)
    add_common_model_args(p_exp)

    # run-gpu
    p_gpu = sub.add_parser("run-gpu", help="run GPU segment benchmark from gpu_tasks.json -> gpu_results.json")
    p_gpu.add_argument("--tasks", type=str, required=True, help="gpu_tasks.json")
    p_gpu.add_argument("--out", type=str, required=True, help="gpu_results.json")
    p_gpu.add_argument("--warmup", type=int, default=3)
    p_gpu.add_argument("--iters", type=int, default=10)
    p_gpu.add_argument("--device", type=str, default=None, help="override device in tasks config (cuda|cpu)")
    p_gpu.add_argument("--gpu-dtype", type=str, default=None, dest="gpu_dtype", help="override dtype (fp16|bf16|fp32)")

    p_gpu.add_argument("--debug", action="store_true",
                       help="Verbose GPU debug: print per-segment per-op shapes/timings/memory stats (very noisy)")
    p_gpu.add_argument("--debug-txt", type=str, default=None,
                       help="(optional) also write the --debug log to this file")

    # run-pim
    # Only requires two files to drive ramulator:
    #   1) ramulator config (YAML/JSON) e.g. example.yaml
    #   2) PIM HW spec JSON           e.g. PIM_AiM.json
    p_pim = sub.add_parser("run-pim", help="run PIM segment simulation from pim_tasks.json -> pim_results.json")
    p_pim.add_argument("--tasks", type=str, required=True, help="pim_tasks.json")
    p_pim.add_argument("--out", type=str, required=True, help="pim_results.json")
    p_pim.add_argument("--cent-sim-root", type=str, default=None,
                       help="path to .../submodules/CENT/cent_simulation (or set env CENT_SIM_ROOT)")
    p_pim.add_argument("--pim-ramulator-bin",type=str, default=None, help="path/name of AiM-enabled ramulator2 executable (required unless present in pim_tasks.json config)",)
    p_pim.add_argument("--pim-ramulator-config", type=str, required=True,
                       help="ramulator2 config file (YAML/JSON), e.g. example.yaml")
    p_pim.add_argument("--pim-hw-json", type=str, required=True,
                       help="PIM HW spec JSON, e.g. PIM_AiM.json")
    p_pim.add_argument("--pim-num-devices", type=int, default=None,
                       help="Override PIM device/DIMM count (used as FC_devices for model-parallel mapping). "
                            "This overrides the value stored in pim_tasks.json.")
    p_pim.add_argument("--debug", action="store_true",
                       help="Verbose debug: print all resolved configs + per-op parameters/latencies (very noisy)")
    p_pim.add_argument("--debug-txt", type=str, default=None,
                       help="(optional) also write the --debug log to this file")
    p_pim.add_argument("--no-cache", action="store_true",
                       help="Disable CostModel PIM latency cache (force ramulator runs for every op)")
    # merge
    p_m = sub.add_parser("merge", help="merge schedule(s) with gpu_results.json + pim_results.json (segment-level)")
    p_m.add_argument("--schedule", type=str, default=None, help="single schedule csv path")
    p_m.add_argument("--schedules", type=str, nargs="+", default=None)
    p_m.add_argument("--gpu-results", type=str, default=None)
    p_m.add_argument("--pim-results", type=str, default=None)
    p_m.add_argument("--comm-model", type=str, default="schedule", choices=["schedule","cxl","none"])
    p_m.add_argument("--pcie-lanes", type=int, default=16)
    p_m.add_argument("--decode-stride", type=int, required=True,
                     help="decode stride used in simulation; decode_token_index = 1 + step*stride")
    p_m.add_argument("--comms", type=str, default=None,
                     help="(optional) single comms trace csv path; used for weight_load/kv_load accumulation")
    p_m.add_argument("--comms-traces", type=str, nargs="+", default=None,
                     help="(optional) comms trace csv paths; if same count as --schedules, treated as paired")
    p_m.add_argument("--non-overlap", type=float, default=1.0,
                     help="NON_OVERLAP factor for kv_load comm time per layer: layer = base_layer_time + NON_OVERLAP*kv_load_time (weight_load is always added to prefill)")
    # KV-load estimate override knobs (optional). If provided, they override the values stored in *_results.json config.
    p_m.add_argument("--kv-load-bw-gbs", type=float, default=None,
                     help="Override config.kv_load_bw_gbs for implicit kv_load estimate (GB/s).")
    p_m.add_argument("--kv-dtype-bytes", type=float, default=None, dest="kv_dtype_bytes",
                     help="Override config.kv_dtype_bytes for implicit kv_load estimate.")
    p_m.add_argument("--kv-load-overhead-us", type=float, default=None, dest="kv_load_overhead_us",
                     help="Override config.kv_load_overhead_us for implicit kv_load estimate.")
    p_m.add_argument("--n-kv-heads", type=int, default=None, dest="n_kv_heads",
                     help="Override config.n_kv_heads for implicit kv_load estimate.")
    p_m.add_argument("--batch", type=int, default=None,
                     help="Override config.batch for implicit kv_load estimate.")
    p_m.add_argument("--out-csv", type=str, required=True,
                     help="where to save merged latency report (csv)")
    p_m.add_argument("--out-steps-csv", type=str, default=None,
                     help="(optional) save per-decode-step block latency (csv)")
    p_m.add_argument("--allow-missing", action="store_true",
                     help="if a segment key is missing in results, treat its cost as 0 instead of error")
    p_m.add_argument("--debug", action="store_true",
                     help="write per-segment (measured vs schedule-trace) comparison into a txt file")
    p_m.add_argument("--debug-txt", type=str, default=None,
                     help="(optional) output path for --debug; default: <out-csv>.debug.txt")

    # collect-merge
    p_c = sub.add_parser("collect-merge", help="collect multiple merge.csv into a single CSV")
    p_c.add_argument("--list", type=str, required=True,
                     help="TSV file listing merge csv paths. Recommended column name: merge_csv")
    p_c.add_argument("--out-csv", type=str, required=True,
                     help="output combined CSV path")
    p_c.add_argument("--skip-missing", action="store_true",
                     help="skip missing merge csvs instead of failing")

    # all (single-machine end-to-end)
    p_all = sub.add_parser("all", help="single-machine mode: export -> run both -> merge")
    p_all.add_argument("--schedule", type=str, default=None, help="single schedule csv path")
    p_all.add_argument("--schedules", type=str, nargs="+", default=None)
    p_all.add_argument("--comms", type=str, default=None, help="(optional) single comms trace csv path (weight_load/kv_load, layer-scope)")
    p_all.add_argument("--comms-traces", type=str, nargs="+", default=None,
                       help="(optional) comms trace csv paths (one or more); if same count as --schedules, treated as paired")
    p_all.add_argument("--out-dir", type=str, default=".", help="where to place tasks/results")
    p_all.add_argument("--prefix", type=str, default="run", help="prefix for tasks/results files")
    p_all.add_argument("--warmup", type=int, default=3)
    p_all.add_argument("--iters", type=int, default=10)
    p_all.add_argument("--cent-sim-root", type=str, default=None)
    p_all.add_argument("--comm-model", type=str, default="schedule", choices=["schedule","cxl","none"])
    p_all.add_argument("--pcie-lanes", type=int, default=16)
    p_all.add_argument("--decode-stride", type=int, required=True,
                       help="decode stride used in simulation; decode_token_index = 1 + step*stride")
    p_all.add_argument("--non-overlap", type=float, default=1.0,
                       help="NON_OVERLAP factor for kv_load comm time per layer (weight_load is always added to prefill)")
    p_all.add_argument("--merge-out-csv", type=str, default=None,
                       help="where to save merged latency report (csv); default: <out-dir>/<prefix>.merge.csv")
    p_all.add_argument("--merge-out-steps-csv", type=str, default=None,
                       help="(optional) save per-decode-step block latency (csv)")
    p_all.add_argument("--allow-missing", action="store_true")
    p_all.add_argument("--debug", action="store_true",
                       help="write per-segment (measured vs schedule-trace) comparison into a txt file during merge")
    p_all.add_argument("--debug-txt", type=str, default=None,
                       help="(optional) output path for --debug; default: alongside merge_out_csv")
    add_common_model_args(p_all)

    args = parser.parse_args(argv)

    if args.cmd == "export":
        cfg = build_cfg_from_args(args)
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        comms_paths = ([args.comms] if getattr(args, "comms", None) else (getattr(args, "comms_traces", None) or []))
        if not schedule_paths:
            raise SystemExit("export requires --schedule or --schedules")
        export_tasks(schedule_paths, cfg, args.out_dir, args.prefix, comms_paths=comms_paths)
        return

    if args.cmd == "run-gpu":
        run_gpu(args.tasks, args.out, warmup=args.warmup, iters=args.iters, device=args.device, gpu_dtype=args.gpu_dtype, debug=bool(getattr(args, "debug", False)), debug_txt=getattr(args, "debug_txt", None))
        return

    if args.cmd == "run-pim":
        run_pim(
            args.tasks,
            args.out,
            cent_sim_root=args.cent_sim_root,
            ramulator_bin=args.pim_ramulator_bin,
            ramulator_config=args.pim_ramulator_config,
            pim_hw_json=args.pim_hw_json,
            pim_num_devices=args.pim_num_devices,
            debug=bool(getattr(args, 'debug', False)),
            debug_txt=getattr(args, 'debug_txt', None),
            no_cache=bool(getattr(args, 'no_cache', False)),
        )
        return

    if args.cmd == "merge":
        if (args.gpu_results is None and args.pim_results is None):
            raise SystemExit("merge requires --gpu-results and/or --pim-results")
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        if not schedule_paths:
            raise SystemExit("merge requires --schedule or --schedules")
        comms_paths = ([args.comms] if getattr(args, "comms", None) else (getattr(args, "comms_traces", None) or []))
        merge(schedule_paths, args.gpu_results, args.pim_results,
              comms_paths=comms_paths, non_overlap=args.non_overlap,
              comm_model=args.comm_model, pcie_lanes=args.pcie_lanes,
              decode_stride=args.decode_stride,
              kv_load_bw_gbs_override=getattr(args, "kv_load_bw_gbs", None),
              kv_dtype_bytes_override=getattr(args, "kv_dtype_bytes", None),
              kv_load_overhead_us_override=getattr(args, "kv_load_overhead_us", None),
              n_kv_heads_override=getattr(args, "n_kv_heads", None),
              batch_override=getattr(args, "batch", None),
              out_csv=args.out_csv,
              out_steps_csv=args.out_steps_csv,
              allow_missing=args.allow_missing,
              debug=args.debug, debug_txt=args.debug_txt)
        return

    if args.cmd == "collect-merge":
        collect_merge_csv(args.list, out_csv=args.out_csv, skip_missing=args.skip_missing)
        return

    if args.cmd == "all":
        cfg = build_cfg_from_args(args)
        out_dir = args.out_dir
        prefix = args.prefix
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        comms_paths = ([args.comms] if getattr(args, "comms", None) else (getattr(args, "comms_traces", None) or []))
        if not schedule_paths:
            raise SystemExit("all requires --schedule or --schedules")

        # 1) export
        gpu_tasks, pim_tasks = export_tasks(schedule_paths, cfg, out_dir, prefix, comms_paths=comms_paths)

        # 2) run both
        gpu_res_path = None
        pim_res_path = None
        gpu_data = _load_json(str(gpu_tasks))
        if len(gpu_data.get("tasks", [])) > 0:
            gpu_res_path = run_gpu(str(gpu_tasks), str(Path(out_dir)/f"{prefix}.gpu_results.json"),
                                   warmup=args.warmup, iters=args.iters, device=cfg.device, gpu_dtype=cfg.gpu_dtype)

        pim_data = _load_json(str(pim_tasks))
        if len(pim_data.get("tasks", [])) > 0:
            pim_res_path = run_pim(str(pim_tasks), str(Path(out_dir)/f"{prefix}.pim_results.json"),
                                   cent_sim_root=args.cent_sim_root,
                                   ramulator_bin=getattr(args, "pim_ramulator_bin", None))

        # 3) merge
        merge(schedule_paths,
              str(gpu_res_path) if gpu_res_path else None,
              str(pim_res_path) if pim_res_path else None,
              comms_paths=comms_paths, non_overlap=args.non_overlap,
              comm_model=args.comm_model, pcie_lanes=args.pcie_lanes,
              decode_stride=args.decode_stride,
              out_csv=(args.merge_out_csv or str(Path(out_dir)/f"{prefix}.merge.csv")),
              out_steps_csv=(args.merge_out_steps_csv),
              allow_missing=args.allow_missing,
              debug=args.debug, debug_txt=args.debug_txt)
        return

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
