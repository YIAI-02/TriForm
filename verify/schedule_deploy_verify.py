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

decode 有多少个 token（多少个 step）：由 schedule 的 decode 行数决定
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
    "k_write",
    "v_write",
    "identity",
    "allreduce",
    "allgather",
    "reducescatter",
    "send",
    "recv",
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


def print_gpu_info(info: Dict[str, Any]) -> None:
    """Pretty-print GPU info for logs."""
    dev = info.get("requested_device")
    if not info.get("torch_available", False):
        print(f"[run-gpu] torch not available; cannot query GPU properties. requested_device={dev}")
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

        print(f"[run-gpu] GPU[{idx}] {name}{extra_s}")
        if info.get("nvidia_smi_L"):
            print(f"[run-gpu] nvidia-smi -L: {info['nvidia_smi_L']}")
        if info.get("nvidia_smi_query"):
            print(f"[run-gpu] nvidia-smi query: {info['nvidia_smi_query']}")
    else:
        print(
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
    m = re.search(r"_[sS]_?(\d+)", str(node_id))
    return int(m.group(1)) if m else -1

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

    # segmenting
    segment_scope: str = "layer"  # layer | device_step

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
            K = int(cfg.prefill_len + 1 + int(sig.step) * stride)
        else:
            if sig.step < 0 or sig.step >= len(cfg.decode_context_lens):
                K = int(cfg.decode_context_lens[-1]) if cfg.decode_context_lens else int(cfg.prefill_len + 1 + sig.step)
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
    """
    Return:
      - uniq segments: key -> SegmentSig
      - ctr: Counter over keys (each occurrence = one (phase,step,layer,device) group in scope=layer, etc.)

    Only 2 segment scope is supported 
        - layer: phase, step, layer, device 同一个segment中不会出现不同的layer和不同的device
        - device: depart by device 按照communication ops 进行分段
    """
    uniq: Dict[str, SegmentSig] = {}
    ctr: Counter = Counter()

    segment_scope = str(segment_scope).lower() 
    if segment_scope not in ("layer", "device_step"):
        raise ValueError(f"Unknown segment_scope: {segment_scope}")

    if segment_scope == "layer":
        # One segment per (phase, step, layer, device) excluding comm ops
        group_cols = ["phase","step","layer","device"]
        for (phase, step, layer, device), g in df.groupby(group_cols, sort=False): #groupby(group_cols) divide by different cols
            g = g.sort_values(["start","_row"])
            dev_type = str(g["device_type"].iloc[0])
            ops: List[Tuple[str,int]] = []
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

    else:
        # device_step: segments on each device timeline, split by comm ops
        group_cols = ["phase","step","device"]
        for (phase, step, device), g in df.groupby(group_cols, sort=False):
            g = g.sort_values(["start","_row"])
            dev_type = str(g["device_type"].iloc[0])
            cur: List[Tuple[str,int]] = []
            for _, r in g.iterrows():
                op = str(r["op"]).strip()
                if op.lower() in _COMM_OPS:

                    if cur:
                        seg = SegmentSig(device_type=dev_type, phase=str(phase), step=int(step), ops=tuple(cur))
                        k = seg.to_key()
                        uniq[k] = seg
                        ctr[k] += 1
                        cur = []
                    continue
                shard = int(r["shard"]) if pd.notna(r["shard"]) else -1
                cur.append((op, shard))
            if cur:
                seg = SegmentSig(device_type=dev_type, phase=str(phase), step=int(step), ops=tuple(cur))
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

    def benchmark_segment(self, seg: SegmentSig, warmup: int = 3, iters: int = 10) -> float:
        # Pre-build execution plan to avoid shape inference in inner loop
        plan: List[Tuple[OpSig, OpShape]] = []
        for op, shard in seg.ops:
            sig = OpSig(device_type=seg.device_type, phase=seg.phase, step=seg.step, op=op, shard=shard)
            plan.append((sig, infer_op_shape(sig, self.cfg)))

        def fn():
            with torch.no_grad():
                for sig, sh in plan:
                    self.execute_one(sig, sh)

        return self._bench(fn, warmup=warmup, iters=iters)


# ==========================================================
# PIM backend (segment-level simulation)
# ==========================================================

class PIMBackend:
    """
    PIM backend (trace-based):
      1) Use CENT(AiM) simulator's TransformerBlock trace generators to emit an AiM instruction trace.
      2) Run AiM-enabled ramulator2 on that trace and parse "memory_system_cycles".
    """

    def __init__(self, cfg: WorkloadConfig, *, cent_sim_root: Optional[str] = None):
        if torch is None:
            raise RuntimeError("PyTorch is required for AiM simulator.")

        self.cfg = cfg
        self._TB_cls = import_aim_transformer_block(cent_sim_root)

        if not cfg.pim_ramulator_bin:
            raise RuntimeError(
                "pim_ramulator_bin must be provided (path/name of AiM-enabled ramulator2 executable).\n"
                "Provide it via cfg (pim_tasks.json config) or at run time: "
                "run-pim --pim-ramulator-bin <path>."
            )

        if not cfg.pim_ramulator_config:
            raise RuntimeError(
                "PIM ramulator mode requires a config file.\n"
            )
        self._ramulator_config_base = Path(cfg.pim_ramulator_config).expanduser().resolve()
        if not self._ramulator_config_base.exists():
            raise FileNotFoundError(f"Ramulator config not found: {self._ramulator_config_base}")

        self._ramulator_config = self._ramulator_config_base

    # ---------------------------------------------------------------------
    # Config / helpers
    # ---------------------------------------------------------------------

    def _ramulator_cmd(self, trace_path: Path) -> List[str]:
        exe_s = str(self.cfg.pim_ramulator_bin)
        if ("/" in exe_s) or ("\\" in exe_s) or exe_s.startswith("~"):
            exe = str(Path(exe_s).expanduser().resolve())
        else:
            exe = exe_s
            cand = (Path.cwd() / exe_s)
            if cand.exists():
                exe = str(cand.resolve())
        return [exe, "-f", str(self._ramulator_config), "-t", str(trace_path)]

    def _run_ramulator(self, trace_path: Path) -> int:
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        cmd = self._ramulator_cmd(trace_path)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(self.cfg.pim_ramulator_timeout_s),
                cwd=str(trace_path.parent),
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Ramulator timed out after {self.cfg.pim_ramulator_timeout_s}s: {cmd}") from e

        if result.returncode != 0:
            out = (result.stdout or "") + "\n" + (result.stderr or "")
            raise RuntimeError(
                f"Ramulator failed (rc={result.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"Trace: {trace_path}\n"
                f"Config: {self._ramulator_config}\n"
                f"Output:\n{out}"
            )

        out = (result.stdout or "") + "\n" + (result.stderr or "")
        m = re.search(r"(?mi)^\s*memory_system_cycles\s*:\s*([0-9]+)\s*$", out)
        if not m:
            raise RuntimeError(
                "Failed to parse ramulator output for memory_system_cycles.\n"
                f"Command: {' '.join(cmd)}\nOutput:\n{out}"
            )
        cycles = int(m.group(1))
        if cycles <= 0:
            raise RuntimeError(f"Invalid cycle count from ramulator: {cycles}")
        return cycles

    def _alloc_trace_dir(self) -> Tuple[Path, bool]:
        if self.cfg.pim_keep_traces:
            base = Path(self.cfg.pim_trace_dir or "./pim_traces").expanduser().resolve()
            base.mkdir(parents=True, exist_ok=True)
            return base, False
        tmp = Path(tempfile.mkdtemp(prefix="pim_trace_")).resolve()
        return tmp, True

    def _max_seq_len(self) -> int:
        # Prefer decoded context lens from tasks.json if available; else fall back.
        m = 0
        if self.cfg.decode_context_lens:
            try:
                m = int(max(self.cfg.decode_context_lens))
            except Exception:
                m = 0
        if m <= 0:
            # prefill_len + decode_steps is a safe upper bound if schedule is standard.
            try:
                m = int(self.cfg.prefill_len) + int(self.cfg.decode_steps)
            except Exception:
                m = int(self.cfg.prefill_len)
        return max(1, int(m))

    def _make_tb_args(self, *, trace_file: str, seqlen: int) -> Any:
        import types
        args = types.SimpleNamespace()

        args.DRAM_column = int(self.cfg.pim_dram_column)
        args.DRAM_row = int(self.cfg.pim_dram_row)
        args.burst_length = int(self.cfg.pim_burst_length)
        args.num_banks = int(self.cfg.pim_num_banks)
        args.num_channels = int(self.cfg.pim_num_channels)
        args.threads = int(self.cfg.pim_threads)
        args.reuse_size = int(self.cfg.pim_reuse_size)

        # Most CENT traces assume channels_per_block exists.
        args.channels_per_block = int(getattr(self.cfg, 'pim_channels_per_block', 0) or args.num_channels)

        # Device count
        args.model_parallel = bool(int(self.cfg.pim_num_devices) > 1)
        args.FC_devices = int(self.cfg.pim_num_devices)

        # PIM simulator flags
        args.only_trace = True
        args.op_trace = True   # ensure trace_* flags are enabled if any path checks them
        args.trace_file = str(trace_file)
        args.pim_compute = True

        # Model meta (not used by trace-only ops heavily, but required by the class)
        args.model = 'llama_like'
        args.embedding = 'rope'
        args.seqlen = int(seqlen)
        args.max_seq_len = int(self._max_seq_len())

        # Parallelism flags
        args.pipeline_parallel = False
        args.inter_device_attention = False
        args.only_FC = False

        # Trace knobs
        args.trace_prepare = True
        args.trace_norm = True
        args.trace_fc_kqvo = True
        args.trace_attention = True
        args.trace_softmax = True
        args.trace_fc_ffn = True
        args.trace_activation = True

        # GEMV mode used by TransformerBlock's trace generator
        args.GEMV = 'reuse-GB'

        return args

    def _make_dummy_model_dict(self, *, dim: int, n_heads: int, ffn_dim: int, seqlen: int) -> Dict[str, Any]:
        """Create a minimal model_dict compatible with TransformerBlockLlama.memory_mapping()."""
        D = int(dim)
        H = int(max(1, n_heads))
        F = int(ffn_dim)
        S = int(max(1, seqlen))

        hd = int(max(1, D // H))
        nkv = H

        model_dict: Dict[str, Any] = {
            'TP_param': torch.tensor(1),
            'dim': torch.tensor(D),
            'n_heads': torch.tensor(H),
            'n_kv_heads': torch.tensor(nkv),

            'x': torch.zeros((1, 1, D)),
            'SANorm': torch.zeros((D,)),
            'FFNNorm': torch.zeros((D,)),

            'sa': torch.zeros((1, 1, D)),
            'h': torch.zeros((1, 1, D)),
            'out': torch.zeros((1, 1, D)),

            'wq': torch.zeros((D, D)),
            'wk': torch.zeros((D, D)),
            'wv': torch.zeros((D, D)),
            'xq': torch.zeros((1, 1, D)),
            'xk': torch.zeros((1, 1, D)),
            'xv': torch.zeros((1, 1, D)),

            'start_pos': torch.tensor(S - 1),
            'cache_k': torch.zeros((1, S, nkv, hd)),
            'cache_v': torch.zeros((1, S, nkv, hd)),
            'scores': torch.zeros((1, H, 1, S)),
            'output': torch.zeros((1, 1, D)),
            'wo': torch.zeros((D, D)),

            'w1': torch.zeros((F, D)),
            'w3': torch.zeros((F, D)),
            'w2': torch.zeros((D, F)),
            'ffn': torch.zeros((1, 1, D)),
        }
        return model_dict

    def _calc_channels(self, block) -> Tuple[int, List[int], int]:
        total_banks = int(block.total_banks)
        if getattr(block, 'model_parallel', False):
            FC_total_banks = int(total_banks * int(getattr(block, 'FC_devices', 1))) #todo
            channels_required = int(getattr(block, 'num_channels', self.cfg.pim_num_channels))
        else:
            FC_total_banks = int(total_banks)
            channels_required = int(getattr(block, 'channels_per_block', self.cfg.pim_num_channels))

        num_channels = int(getattr(block, 'num_channels', channels_required))
        channel_multi_required = (num_channels // channels_required) * channels_required if channels_required > 0 else num_channels
        channel_lst_multi = [c for c in range(max(1, channel_multi_required))]
        return channels_required, channel_lst_multi, FC_total_banks

    def _time_add(self, block, key: str, delta: float) -> None:
        if not hasattr(block, 'time'):
            return
        try:
            block.time[key] = float(block.time.get(key, 0.0)) + float(delta)
        except Exception:
            try:
                block.time[key] = block.time.get(key, 0) + delta
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Trace emitters
    # ---------------------------------------------------------------------
    def _emit_rmsnorm(self, block, channels_required: int, channel_lst: List[int]) -> None:
        dim = int(block.dim)
        burst = int(block.burst_length)
        total_banks = int(block.total_banks)

        input_len = (dim - 1) // max(1, (total_banks // 2)) + 1
        block.WR_BIAS_only_trace(channel_lst)
        block.MAC_ABK_only_trace(channel_lst, block.x_row_index, (input_len - 1) // burst + 1, 'breakdown_sa_pow')
        block.RD_MAC_only_trace(channel_lst)

        ew_len = (dim - 1) // max(1, (total_banks // 4)) + 1
        ew_banks = (dim - 1) // ew_len + 1

        self._time_add(block, 'WR_SBK', float(getattr(block, 'timing_constant', {}).get('WR_SBK', 0)) + dim / burst)
        block.store_for_EWMUL_input_only_trace(channels_required, ew_banks, 1, block.x_copy_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.x_copy_row_index, (ew_len - 1) // burst + 1)

        for bank in range(int(block.num_banks)):
            if bank % 4 == 2:
                block.COPY_BK_GB_only_trace(channel_lst, bank, block.x_copy_row_index, (ew_len - 1) // burst + 1)
                block.COPY_GB_BK_only_trace(channel_lst, bank - 1, block.SANorm_row_index, (ew_len - 1) // burst + 1)

        block.EWMUL_only_trace(channel_lst, block.SANorm_row_index, (ew_len - 1) // burst + 1)

        self._time_add(block, 'RD_SBK', float(getattr(block, 'timing_constant', {}).get('RD_SBK', 0)) + dim / burst)
        block.load_from_EWMUL_input_only_trace(channels_required, ew_banks, 2, block.SANorm_row_index, ew_len)
        block.SYNC_only_trace()

    def _emit_softmax(self, block, channels_required: int, channel_lst: List[int], seqlen: int) -> None:
        burst = int(block.burst_length)
        num_banks = int(block.num_banks)
        total_banks = int(block.total_banks)
        S = int(max(1, seqlen))

        rows_per_score = (S - 1) // int(block.DRAM_column) + 1
        input_vector_EWMUL_length = (S - 1) // max(1, (total_banks // 4)) + 1
        input_vector_EWMUL_utilized_banks = (S - 1) // input_vector_EWMUL_length + 1

        for row in range(rows_per_score):
            if row == rows_per_score - 1:
                ew_len = (S - row * int(block.DRAM_column) - 1) // max(1, (total_banks // 4)) + 1
                ew_banks = (S - row * int(block.DRAM_column) - 1) // ew_len + 1
            else:
                ew_len = (int(block.DRAM_column) - 1) // max(1, (total_banks // 4)) + 1
                ew_banks = (int(block.DRAM_column) - 1) // ew_len + 1

            self._time_add(block, 'WR_SBK', float(getattr(block, 'timing_constant', {}).get('WR_SBK', 0)) + ew_banks * ew_len / burst)
            block.store_for_EWMUL_score_only_trace(channels_required, block.scores_row_index, input_vector_EWMUL_utilized_banks, 1,  input_vector_EWMUL_length)
            block.EWMUL_only_trace(channel_lst, block.scores_row_index, (ew_len - 1) // burst + 1)

            for bank in range(num_banks):
                if bank % 4 == 2:
                    block.COPY_BK_GB_only_trace(channel_lst, bank, block.scores_row_index, (ew_len - 1) // burst + 1)
                    block.COPY_GB_BK_only_trace(channel_lst, bank - 1, block.xk_row_index, (ew_len - 1) // burst + 1)

            block.EWMUL_only_trace(channel_lst, block.xk_row_index, (ew_len - 1) // burst + 1)

            self._time_add(block, 'RD_SBK', float(getattr(block, 'timing_constant', {}).get('RD_SBK', 0)) + ew_banks * ew_len / burst)
            block.load_from_EWMUL_score_only_trace(channels_required, block.xk_row_index, input_vector_EWMUL_utilized_banks, 2,  input_vector_EWMUL_length)
            block.SYNC_only_trace()

    def _emit_silu(self, block, channel_lst: List[int]) -> None:
        ffn_dim = int(block.w1.shape[0])
        burst = int(block.burst_length)
        total_banks = int(block.total_banks)

        ew_len = (ffn_dim - 1) // max(1, (total_banks // 4)) + 1
        ew_banks = (ffn_dim - 1) // ew_len + 1

        self._time_add(block, 'WR_SBK', float(getattr(block, 'timing_constant', {}).get('WR_SBK', 0)) + ffn_dim / burst)
        block.store_for_EWMUL_input_only_trace(int(block.channels_per_block), ew_banks, 1, block.ffn_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.ffn_row_index, (ew_len - 1) // burst + 1)

        for bank in range(int(block.num_banks)):
            if bank % 4 == 2:
                block.COPY_BK_GB_only_trace(channel_lst, bank, block.ffn_row_index, (ew_len - 1) // burst + 1)
                block.COPY_GB_BK_only_trace(channel_lst, bank - 1, block.SANorm_row_index, (ew_len - 1) // burst + 1)

        block.EWMUL_only_trace(channel_lst, block.SANorm_row_index, (ew_len - 1) // burst + 1)

        self._time_add(block, 'RD_SBK', float(getattr(block, 'timing_constant', {}).get('RD_SBK', 0)) + ffn_dim / burst)
        block.SYNC_only_trace()

    def _emit_residual(self, block) -> None:
        op_size = int(block.dim) // int(block.burst_length)
        block.EWADD_only_trace(op_size)

    def _emit_weight_gemv(self, block, channel_lst: List[int], row_index: int, V: int, N: int, FC_total_banks: int, timing: str, *, with_af: bool = False) -> None:
        if with_af and hasattr(block, 'Vector_Matrix_Mul_weight_af_pim_only_trace'):
            block.Vector_Matrix_Mul_weight_af_pim_only_trace(channel_lst, row_index, int(V), int(N), int(FC_total_banks), timing)
        else:
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, row_index, int(V), int(N), int(FC_total_banks), timing)

    # ---------------------------------------------------------------------
    # Public: simulate a whole segment as one continuous trace
    # ---------------------------------------------------------------------
    def benchmark_segment(self, seg: SegmentSig) -> float:
        # Find a representative shape to build the TB
        first = None
        for op, shard in seg.ops:
            if op in _COMM_OPS:
                continue
            first = OpSig(device_type=seg.device_type, phase=seg.phase, step=seg.step, op=op, shard=shard)
            break
        if first is None:
            return 0.0

        sh0 = infer_op_shape(first, self.cfg)

        # LOCAL dims (single-shard modeling)
        dim_local = int(sh0.shard_dim if int(self.cfg.shards) > 1 else sh0.dim)
        n_heads_local = int(sh0.heads_per_shard if int(self.cfg.shards) > 1 else int(self.cfg.n_heads))
        ffn_local = int(sh0.ffn_shard_dim if int(self.cfg.shards) > 1 else int(self.cfg.ffn_dim))
        seqlen = int(sh0.key_len)

        trace_dir, cleanup = self._alloc_trace_dir()
        trace_name = f"{seg.phase}_step{seg.step}_{uuid.uuid4().hex}.trace"
        trace_path = trace_dir / trace_name

        args = self._make_tb_args(trace_file=str(trace_path), seqlen=seqlen)
        model_dict = self._make_dummy_model_dict(dim=dim_local, n_heads=n_heads_local, ffn_dim=ffn_local, seqlen=seqlen)

        block = self._TB_cls(model_dict, args)

        try:
            if hasattr(block, 'memory_mapping'):
                block.memory_mapping()

            channels_required, channel_lst, FC_total_banks = self._calc_channels(block)

            for op, shard in seg.ops:
                if op in _COMM_OPS:
                    continue

                sig = OpSig(device_type=seg.device_type, phase=seg.phase, step=seg.step, op=op, shard=shard)
                sh = infer_op_shape(sig, self.cfg)
                T = int(max(1, sh.query_len))

                if seg.phase == 'prefill' and T > 1: #todo
                    seqlens_list = list(range(1, int(sh.key_len) + 1))
                else:
                    seqlens_list = [int(sh.key_len)]

                if op == 'LN':
                    for _ in range(T):
                        self._emit_rmsnorm(block, channels_required, channel_lst)

                elif op == 'Q':
                    for _ in range(T):
                        self._emit_weight_gemv(block, channel_lst, int(block.wq_row_index), dim_local, dim_local, FC_total_banks, 'breakdown_sa_weight')

                elif op == 'K':
                    for _ in range(T):
                        self._emit_weight_gemv(block, channel_lst, int(block.wk_row_index), dim_local, dim_local, FC_total_banks, 'breakdown_sa_weight')

                elif op == 'V':
                    for _ in range(T):
                        self._emit_weight_gemv(block, channel_lst, int(block.wv_row_index), dim_local, dim_local, FC_total_banks, 'breakdown_sa_weight')

                elif op == 'O':
                    for _ in range(T):
                        self._emit_weight_gemv(block, channel_lst, int(block.wo_row_index), dim_local, dim_local, FC_total_banks, 'breakdown_sa_weight')

                elif op == 'FFN_W1':
                    for _ in range(T):
                        self._emit_weight_gemv(block, channel_lst, int(block.w1_row_index), dim_local, ffn_local, FC_total_banks, 'breakdown_ffn_weight', with_af=True)

                elif op == 'FFN_W3':
                    for _ in range(T):
                        self._emit_weight_gemv(block, channel_lst, int(block.w3_row_index), dim_local, ffn_local, FC_total_banks, 'breakdown_ffn_weight')

                elif op == 'FFN_W2':
                    for _ in range(T):
                        self._emit_weight_gemv(block, channel_lst, int(block.w2_row_index), ffn_local, dim_local, FC_total_banks, 'breakdown_ffn_weight')

                elif op == 'SwiGLU':
                    for _ in range(T):
                        self._emit_silu(block, channel_lst)

                elif op == 'Add':
                    for _ in range(T):
                        self._emit_residual(block)

                elif op == 'QK':
                    for S in seqlens_list:
                        block.Vector_Matrix_Mul_score_pim_only_trace(int(block.cache_k_row_index), int(S), 'breakdown_sa_score')

                elif op == 'Softmax':
                    for S in seqlens_list:
                        self._emit_softmax(block, channels_required, channel_lst, int(S))

                elif op == 'SV':
                    for S in seqlens_list:
                        block.Vector_Matrix_Mul_output_pim_only_trace(int(block.cache_v_row_index), int(S), 'breakdown_sa_output')

                else:
                    raise ValueError(f"Unknown / unsupported PIM op in schedule: {op}")

            if hasattr(block, 'finish'):
                block.finish()
            else:
                if getattr(block, 'file', None):
                    try:
                        block.file.write('AiM EOC\n')
                    except Exception:
                        pass

            if getattr(block, 'file', None):
                try:
                    block.file.flush()
                except Exception:
                    pass
                try:
                    block.file.close()
                except Exception:
                    pass

            cycles = self._run_ramulator(trace_path)
            sec = float(cycles) / (float(self.cfg.pim_freq_ghz) * 1_000_000_000.0)
            return sec

        finally:
            if cleanup:
                try:
                    shutil.rmtree(trace_dir, ignore_errors=True)
                except Exception:
                    pass
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
        cfg.decode_context_lens = [int(cfg.prefill_len + 1 + i * stride) for i in range(max_decode_steps)] if max_decode_steps > 0 else []

    # collect unique segments over all schedules
    uniq: Dict[str, SegmentSig] = {}
    ctr: Counter = Counter()
    for df in dfs: # all schedules merge into one uniq dict and update counter
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


def run_gpu(tasks_json: str, out_json: str, warmup: int = 3, iters: int = 10,
            device: Optional[str] = None, gpu_dtype: Optional[str] = None) -> Path:
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
    gpu_info = collect_gpu_info(cfg.device)

    results: Dict[str, float] = {}
    print(f"[run-gpu] tasks={len(tasks)} warmup={warmup} iters={iters} device={cfg.device} dtype={cfg.gpu_dtype} segment_scope={seg_scope}")
    if weight_load_s:
        print(f"[run-gpu] extra weight_load_s={weight_load_s:.6f}s (will be added in merge)")
    print_gpu_info(gpu_info)

    backend = GPUBackend(cfg)
    for i, t in enumerate(tasks):
        key = t["key"]
        ops = tuple((x["op"], int(x.get("shard",-1))) for x in t.get("ops", []))
        sig = SegmentSig(device_type="npu", phase=str(t["sig"]["phase"]), step=int(t["sig"]["step"]), ops=ops)
        sec = backend.benchmark_segment(sig, warmup=warmup, iters=iters)
        results[key] = float(sec)
        if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
            print(f"  progress {i+1}/{len(tasks)}")

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
    _save_json(outp, out)
    print(f"[run-gpu] wrote {outp}")
    return outp


def run_pim(tasks_json: str, out_json: str, *,
            cent_sim_root: Optional[str] = None,
            ramulator_config: Optional[str] = None,
            pim_hw_json: Optional[str] = None,
            pim_num_devices: Optional[int] = None,
            ramulator_bin: Optional[str] = None,
            ramulator_timeout_s: Optional[int] = None,
            keep_traces: bool = False,
            trace_dir: Optional[str] = None) -> Path:

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
    if pim_num_devices is not None:
        try:
            iv = int(pim_num_devices)
        except Exception as e:
            raise ValueError(f"pim_num_devices must be an integer, got {pim_num_devices!r}") from e
        if iv <= 0:
            raise ValueError(f"pim_num_devices must be > 0, got {iv}")
        cfg.pim_num_devices = iv
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

    backend = PIMBackend(cfg, cent_sim_root=cent_sim_root)

    results: Dict[str, float] = {}
    print(f"[run-pim] tasks={len(tasks)} segment_scope={seg_scope}")
    if weight_load_s:
        print(f"[run-pim] extra weight_load_s={weight_load_s:.6f}s (will be added in merge)")
    for i, t in enumerate(tasks):
        key = t["key"]
        ops = tuple((x["op"], int(x.get("shard",-1))) for x in t.get("ops", []))
        sig = SegmentSig(device_type="pim", phase=str(t["sig"]["phase"]), step=int(t["sig"]["step"]), ops=ops)
        sec = backend.benchmark_segment(sig)
        results[key] = float(sec)
        if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
            print(f"  progress {i+1}/{len(tasks)}")

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



def _comm_latency_from_row(r: Any, comm_model: str, *, pcie_lanes: int = 16,
                           dim: int = 4096, shards: int = 4, dtype_bytes: int = 2) -> float:
    """Per-row COMM latency (seconds)."""
    comm_model = str(comm_model).lower()
    if comm_model == "none":
        return 0.0
    if comm_model == "schedule":
        return float(r["duration"])
    if comm_model == "cxl":
        if cxl_latency is None:
            raise RuntimeError("cxl_latency.py not importable; cannot use --comm-model cxl")
        op = str(r["op"])
        shard = int(r.get("shard", -1)) if pd.notna(r.get("shard", -1)) else -1
        shard_dim = max(1, int(dim) // max(1, int(shards)))
        if op in ("K_write", "V_write"):
            bytes_ = shard_dim * dtype_bytes
        else:
            bytes_ = int(dim) * dtype_bytes
        return float(cxl_latency.cxl_memcpy_latency(bytes_, lanes=pcie_lanes))
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
    """
    Build per-row durations (seconds) for the whole trace, aligned to df.index.

    - COMM rows: per-row comm model latency
    - compute rows: segment total latency (from *_results.json) distributed to ops
      proportionally to their original trace 'duration' within each (phase, step, layer, device).

    Returns:
      (dur_s: np.ndarray, missing_segments: int)
    """
    n = len(df)
    dur_s = np.zeros(n, dtype=np.float64)

    op_l = df["op"].astype(str).str.strip().str.lower()
    is_comm = op_l.isin(_COMM_OPS).to_numpy()

    # 1) COMM rows
    if comm_model.lower() == "schedule":
        dur_s[is_comm] = df.loc[is_comm, "duration"].astype(float).to_numpy()
    elif comm_model.lower() == "none":
        pass
    else:
        # cxl (or any future row-level model)
        comm_idx = df.index[is_comm].to_list()
        for i in comm_idx:
            r = df.loc[i]
            dur_s[i] = _comm_latency_from_row(r, comm_model, pcie_lanes=pcie_lanes, dim=dim, shards=shards)

    # 2) Compute rows (layer scope)
    compute_df = df.loc[~is_comm, ["phase", "step", "layer", "device", "device_type", "op", "shard", "start", "_row", "duration"]].copy()

    # Sort to match extract_segments() ordering: per group sorted by (start, _row)
    compute_df = compute_df.sort_values(["phase", "step", "layer", "device", "start", "_row"], kind="mergesort")

    missing = 0
    group_cols = ["phase", "step", "layer", "device"]
    for (phase, step, layer, device), g in compute_df.groupby(group_cols, sort=False):
        # build segment signature (NOTE: step uses *token index* when decode_stride > 1)
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

        step_key = int(step)
        if str(phase) == "decode":
            # decode token index modeling: token_idx = 1 + step*stride
            # (step here is the sampled-step id computed from the trace)
            step_key = 1 + int(step) * int(decode_stride)

        seg_tok = SegmentSig(device_type=dev_type, phase=str(phase), step=step_key, ops=tuple(ops))

        # lookup latency
        total_lat = None
        if dev_type == "npu":
            total_lat = _try_lookup_segment_latency(seg_tok, gpu_res)
        elif dev_type == "pim":
            total_lat = _try_lookup_segment_latency(seg_tok, pim_res)

        # backward compatible: if results were generated with step=0..N-1 (sample step),
        # try the legacy key as fallback.
        if total_lat is None and str(phase) == "decode" and int(decode_stride) != 1:
            seg_legacy = SegmentSig(device_type=dev_type, phase=str(phase), step=int(step), ops=tuple(ops))
            if dev_type == "npu":
                total_lat = _try_lookup_segment_latency(seg_legacy, gpu_res)
            elif dev_type == "pim":
                total_lat = _try_lookup_segment_latency(seg_legacy, pim_res)

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

    # 3) decode scaling
    ds = int(decode_stride) if decode_stride is not None else 1
    dmask = (df["phase"].astype(str) == "decode").to_numpy()
    dur_s[dmask & (~is_comm)] *= float(ds)

    return dur_s, int(missing)

def _build_row_durations_device_step_scope(
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
    elif comm_model.lower() == "none":
        pass
    else:
        # cxl (or any future row-level model)
        comm_idx = df.index[is_comm].to_list()
        for i in comm_idx:
            r = df.loc[i]
            dur_s[i] = _comm_latency_from_row(r, comm_model, pcie_lanes=pcie_lanes, dim=dim, shards=shards)

    # 2) Compute rows (device_step scope: split by comm ops on each device timeline)
    cols = ["phase", "step", "device", "device_type", "op", "shard", "start", "_row", "duration"]
    sub = df.loc[:, [c for c in cols if c in df.columns]].copy()

    # Ensure required columns exist (schedule csv schema contract)
    for req in ["phase", "step", "device", "device_type", "op", "start", "_row", "duration"]:
        if req not in sub.columns:
            raise ValueError(f"schedule is missing required column: {req}")

    missing = 0
    group_cols = ["phase", "step", "device"]
    for (phase, step, device), g in sub.groupby(group_cols, sort=False):
        # Match extract_segments() ordering: per group sorted by (start, _row)
        g = g.sort_values(["start", "_row"], kind="mergesort")
        dev_type = str(g["device_type"].iloc[0])
        dev_type_n = dev_type.strip().lower()

        cur_ops: List[Tuple[str, int]] = []
        cur_idxs: List[int] = []

        def flush_segment() -> None:
            nonlocal missing, cur_ops, cur_idxs
            if not cur_ops:
                cur_ops = []
                cur_idxs = []
                return

            if dev_type_n not in ("npu", "pim"):
                idxs = np.array(cur_idxs, dtype=int)
                if idxs.size:
                    dur_s[idxs] = pd.to_numeric(df.loc[idxs, "duration"], errors="coerce").fillna(0.0).astype(float).to_numpy()
                cur_ops = []
                cur_idxs = []
                return

            ph = str(phase)
            st = int(step)

            # NOTE: Try token-index key first (newer convention), then fall back to sample-step key.
            total_lat: Optional[float] = None
            if ph == "decode" and int(decode_stride) != 1:
                st_tok = 1 + int(st) * int(decode_stride)
                seg_tok = SegmentSig(device_type=dev_type, phase=ph, step=int(st_tok), ops=tuple(cur_ops))
                if dev_type == "npu":
                    total_lat = _try_lookup_segment_latency(seg_tok, gpu_res)
                elif dev_type == "pim":
                    total_lat = _try_lookup_segment_latency(seg_tok, pim_res)

                if total_lat is None:
                    seg_legacy = SegmentSig(device_type=dev_type, phase=ph, step=int(st), ops=tuple(cur_ops))
                    if dev_type == "npu":
                        total_lat = _try_lookup_segment_latency(seg_legacy, gpu_res)
                    elif dev_type == "pim":
                        total_lat = _try_lookup_segment_latency(seg_legacy, pim_res)
            else:
                seg = SegmentSig(device_type=dev_type, phase=ph, step=int(st), ops=tuple(cur_ops))
                if dev_type == "npu":
                    total_lat = _try_lookup_segment_latency(seg, gpu_res)
                elif dev_type == "pim":
                    total_lat = _try_lookup_segment_latency(seg, pim_res)

            if total_lat is None:
                if allow_missing:
                    total_lat = 0.0
                else:
                    missing += 1
                    total_lat = 0.0

            # Distribute segment latency back to its rows proportionally to trace duration.
            idxs = np.array(cur_idxs, dtype=int)
            if idxs.size == 0:
                cur_ops = []
                cur_idxs = []
                return

            # Pull weights from original df to avoid any dtype surprises.
            w = pd.to_numeric(df.loc[idxs, "duration"], errors="coerce").fillna(0.0).astype(float).to_numpy()
            wsum = float(np.nansum(w.astype(np.float64)))
            if wsum > 0:
                dur_s[idxs] = float(total_lat) * (w / wsum)
            else:
                dur_s[idxs] = float(total_lat) / float(idxs.size)

            cur_ops = []
            cur_idxs = []

        for idx, r in g.iterrows():
            op = str(r["op"]).strip()
            if op.lower() in _COMM_OPS:
                flush_segment()
                continue
            shard = int(r["shard"]) if ("shard" in r and pd.notna(r["shard"])) else -1
            cur_ops.append((op, shard))
            cur_idxs.append(int(idx))

        flush_segment()

    # 3) decode scaling
    ds = int(decode_stride) if decode_stride is not None else 1
    if ds <= 0:
        ds = 1
    if ds != 1:
        dmask = (df["phase"].astype(str) == "decode").to_numpy()
        # measured segment latencies are per-sampled-step
        dur_s[dmask & (~is_comm)] *= float(ds)

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
        "trace_total_s": float(total),
    }

def _lookup_segment_latency_for_debug(
    dev_type: str,
    phase: str,
    step_sample: int,
    ops: List[Tuple[str, int]],
    *,
    decode_stride: int,
    gpu_res: Dict[str, float],
    pim_res: Dict[str, float],
) -> Tuple[Optional[float], str, int, str]:
    """Lookup a segment latency in results dicts."""
    dev_type = str(dev_type)
    phase = str(phase)

    # Candidate step encodings
    cands: List[Tuple[str, int]] = []
    ds = int(decode_stride) if decode_stride is not None else 1
    if ds <= 0:
        ds = 1
    if phase == "decode" and ds != 1:
        cands.append(("token_index", 1 + int(step_sample) * ds))
    cands.append(("sample_step", int(step_sample)))

    res = gpu_res if dev_type == "npu" else pim_res if dev_type == "pim" else {}

    # Try in order
    for kind, st in cands:
        seg = SegmentSig(device_type=dev_type, phase=phase, step=int(st), ops=tuple(ops))
        k = seg.to_key()
        if k in res:
            try:
                return float(res[k]), k, int(st), str(kind)
            except Exception:
                return res[k], k, int(st), str(kind)  # type: ignore[return-value]

    # Not found: return the first candidate key as a reference
    kind0, st0 = cands[0]
    seg0 = SegmentSig(device_type=dev_type, phase=phase, step=int(st0), ops=tuple(ops))
    return None, seg0.to_key(), int(st0), str(kind0)


def _collect_segment_debug_rows(
    df: pd.DataFrame,
    *,
    seg_scope: str,
    decode_stride: int,
    gpu_res: Dict[str, float],
    pim_res: Dict[str, float],
    allow_missing: bool,
) -> List[Dict[str, Any]]:
    """Build per-unique-segment comparison rows: (measured vs trace duration)."""
    seg_scope = str(seg_scope).lower()
    if seg_scope not in ("layer", "device_step"):
        raise ValueError(f"Unknown segment_scope: {seg_scope}")

    # normalize op type
    op_l = df["op"].astype(str).str.strip().str.lower()
    is_comm = op_l.isin(_COMM_OPS).to_numpy()

    # Aggregation dict: key_used -> stats
    stats: Dict[str, Dict[str, Any]] = {}

    def _accum(
        *,
        phase: str,
        step_sample: int,
        dev_type: str,
        ops: List[Tuple[str, int]],
        trace_sum_s_raw: float,
    ) -> None:
        dev_type_n = str(dev_type).strip().lower()
        if dev_type_n not in ("npu", "pim"):
            seg = SegmentSig(device_type=str(dev_type), phase=str(phase), step=int(step_sample), ops=tuple(ops))
            key_used = seg.to_key()
            step_used = int(step_sample)
            step_kind = "trace"
            found = True
            measured_s_raw = 0.0  # will be overwritten to trace_avg_s_raw (running average) below
        else:
            lat, key_used, step_used, step_kind = _lookup_segment_latency_for_debug(
                dev_type, phase, step_sample, ops,
                decode_stride=int(decode_stride),
                gpu_res=gpu_res, pim_res=pim_res,
            )
            found = lat is not None
            measured_s_raw = float(lat) if lat is not None else 0.0  # match merge() behavior when missing

        ops_repr = ",".join([f"{op}:{sh}" for op, sh in ops])

        rec = stats.get(key_used)
        if rec is None:
            rec = {
                "key": key_used,
                "device_type": str(dev_type),
                "phase": str(phase),
                "step_sample": int(step_sample),
                "step_used": int(step_used),
                "step_kind": str(step_kind),
                "ops_repr": ops_repr,
                "count": 0,
                "trace_sum_s_raw": 0.0,
                "measured_s_raw": float(measured_s_raw),
                "found_in_results": bool(found),
                "missing_count": 0,
            }
            stats[key_used] = rec

        rec["count"] = int(rec.get("count", 0)) + 1
        rec["trace_sum_s_raw"] = float(rec.get("trace_sum_s_raw", 0.0)) + float(trace_sum_s_raw)

        # For trace-only segments, keep "measured" equal to the running trace average.
        if dev_type_n not in ("npu", "pim"):
            cnt = max(1, int(rec.get("count", 0)))
            rec["measured_s_raw"] = float(rec.get("trace_sum_s_raw", 0.0)) / float(cnt)
            rec["found_in_results"] = True
            rec["missing_count"] = 0

        # If any occurrence is missing, record it (helps spot bad keys)
        if not found:
            rec["missing_count"] = int(rec.get("missing_count", 0)) + 1
            if not allow_missing:
                # In non-allow-missing mode we'd still set cost=0 in merge(), but flag it.
                rec["found_in_results"] = False

    if seg_scope == "layer":
        # Compute rows only (COMM handled separately)
        compute_df = df.loc[~is_comm, ["phase", "step", "layer", "device", "device_type", "op", "shard", "start", "_row", "duration"]].copy()
        compute_df = compute_df.sort_values(["phase", "step", "layer", "device", "start", "_row"], kind="mergesort")

        for (phase, step, layer, device), g in compute_df.groupby(["phase", "step", "layer", "device"], sort=False):
            if g.empty:
                continue
            dev_type = str(g["device_type"].iloc[0])
            ops: List[Tuple[str, int]] = []
            for _, rr in g.iterrows():
                op = str(rr["op"]).strip()
                shard = int(rr["shard"]) if pd.notna(rr["shard"]) else -1
                ops.append((op, shard))
            if not ops:
                continue
            trace_sum = float(pd.to_numeric(g["duration"], errors="coerce").fillna(0.0).astype(float).sum())
            _accum(phase=str(phase), step_sample=int(step), dev_type=dev_type, ops=ops, trace_sum_s_raw=trace_sum)

    else:
        # device_step: per-device timeline, split by COMM ops
        cols = ["phase", "step", "device", "device_type", "op", "shard", "start", "_row", "duration"]
        sub = df.loc[:, [c for c in cols if c in df.columns]].copy()
        sub = sub.sort_values(["phase", "step", "device", "start", "_row"], kind="mergesort")

        for (phase, step, device), g in sub.groupby(["phase", "step", "device"], sort=False):
            if g.empty:
                continue
            dev_type = str(g["device_type"].iloc[0])

            cur_ops: List[Tuple[str, int]] = []
            cur_durs: List[float] = []

            def flush() -> None:
                nonlocal cur_ops, cur_durs
                if not cur_ops:
                    cur_ops = []
                    cur_durs = []
                    return
                trace_sum = float(np.nansum(np.array(cur_durs, dtype=np.float64)))
                _accum(phase=str(phase), step_sample=int(step), dev_type=dev_type, ops=cur_ops, trace_sum_s_raw=trace_sum)
                cur_ops = []
                cur_durs = []

            for _, rr in g.iterrows():
                op = str(rr["op"]).strip()
                if op.lower() in _COMM_OPS:
                    flush()
                    continue
                shard = int(rr["shard"]) if ("shard" in rr and pd.notna(rr["shard"])) else -1
                cur_ops.append((op, shard))
                d = rr.get("duration", 0.0)
                try:
                    cur_durs.append(float(d))
                except Exception:
                    cur_durs.append(0.0)
            flush()

    # finalize rows + derived columns
    out_rows: List[Dict[str, Any]] = []
    ds = int(decode_stride) if decode_stride is not None else 1
    if ds <= 0:
        ds = 1

    for k, rec in stats.items():
        cnt = max(1, int(rec.get("count", 0) or 0))
        trace_sum_raw = float(rec.get("trace_sum_s_raw", 0.0) or 0.0)
        trace_avg_raw = trace_sum_raw / float(cnt)

        phase = str(rec.get("phase", ""))
        measured_scale = float(ds) if phase == "decode" else 1.0

        measured_raw = float(rec.get("measured_s_raw", 0.0) or 0.0)
        measured_scaled = measured_raw * measured_scale
        trace_avg_scaled = trace_avg_raw

        delta_raw = measured_raw - trace_avg_raw
        delta_scaled = measured_scaled - trace_avg_scaled

        def _pct(delta: float, base: float) -> float:
            try:
                if float(base) == 0.0:
                    return float("nan")
                return float(delta) / float(base) * 100.0
            except Exception:
                return float("nan")

        out_rows.append({
            **rec,
            "trace_avg_s_raw": float(trace_avg_raw),
            "trace_avg_s_scaled": float(trace_avg_scaled),
            "measured_s_scaled": float(measured_scaled),
            "delta_s_raw": float(delta_raw),
            "delta_pct_raw": _pct(delta_raw, trace_avg_raw),
            "delta_s_scaled": float(delta_scaled),
            "delta_pct_scaled": _pct(delta_scaled, trace_avg_scaled),
        })

    return out_rows


def merge(
    schedule_paths: List[str],
    gpu_results_json: Optional[str],
    pim_results_json: Optional[str],
    *,
    comm_model: str = "schedule",
    pcie_lanes: int = 16,
    decode_stride: int,
    out_csv: str,
    out_steps_csv: Optional[str] = None,
    allow_missing: bool = False,
    segment_scope: Optional[str] = None,
    debug: bool = False,
    debug_txt: Optional[str] = None,
) -> None:
    """
    Merge schedule trace(s) + measured segment latencies into an overlapped end-to-end latency.
    """
    if decode_stride is None:
        raise ValueError("--decode-stride is required")
    decode_stride = int(decode_stride)
    if decode_stride <= 0:
        raise ValueError("--decode-stride must be positive")

    # load results
    gpu_res: Dict[str, float] = {}
    pim_res: Dict[str, float] = {}
    gpu_weight_load_s_global: float = 0.0
    pim_weight_load_s_global: float = 0.0
    gpu_weight_load_by_schedule: Dict[str, Any] = {}
    pim_weight_load_by_schedule: Dict[str, Any] = {}
    scope_from_results: Optional[str] = None
    cfg_dim = 4096
    cfg_shards = 4
    cfg_prefill_len = None

    if gpu_results_json:
        data = _load_json(gpu_results_json)
        gpu_res = {k: float(v) for k, v in data.get("results", {}).items()}
        try:
            gpu_weight_load_s_global = float(data.get("weight_load_s", 0.0) or 0.0)
        except Exception:
            gpu_weight_load_s_global = 0.0
        gpu_weight_load_by_schedule = data.get("weight_load_by_schedule", {}) or {}
        scope_from_results = data.get("segment_scope", scope_from_results)
        cfg = data.get("config", {}) or {}
        cfg_dim = int(cfg.get("dim", cfg_dim))
        cfg_shards = int(cfg.get("shards", cfg_shards))
        if "prefill_len" in cfg:
            cfg_prefill_len = int(cfg.get("prefill_len"))

    if pim_results_json:
        data = _load_json(pim_results_json)
        pim_res = {k: float(v) for k, v in data.get("results", {}).items()}
        try:
            pim_weight_load_s_global = float(data.get("weight_load_s", 0.0) or 0.0)
        except Exception:
            pim_weight_load_s_global = 0.0
        pim_weight_load_by_schedule = data.get("weight_load_by_schedule", {}) or {}
        scope_from_results = data.get("segment_scope", scope_from_results)
        cfg = data.get("config", {}) or {}
        cfg_dim = int(cfg.get("dim", cfg_dim))
        cfg_shards = int(cfg.get("shards", cfg_shards))
        if cfg_prefill_len is None and "prefill_len" in cfg:
            cfg_prefill_len = int(cfg.get("prefill_len"))

    # choose segment scope
    seg_scope = (segment_scope or scope_from_results or "layer")
    seg_scope = str(seg_scope).lower()
    if seg_scope not in ("layer", "device_step"):
        raise ValueError(f"Unknown segment_scope: {seg_scope}")

    results_rows: List[Dict[str, Any]] = []
    steps_rows: List[Dict[str, Any]] = []

    dbg_f = None
    dbg_path = None
    if debug:
        dbg_path = Path(debug_txt) if debug_txt else Path(out_csv).with_suffix(".debug.txt")
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        dbg_f = open(str(dbg_path), "w", encoding="utf-8")
        dbg_f.write(f"# merge debug\n")
        dbg_f.write(f"# segment_scope={seg_scope} comm_model={comm_model} pcie_lanes={pcie_lanes} decode_stride={decode_stride} allow_missing={allow_missing}\n")

    for sp in schedule_paths:
        p = resolve_existing_path(sp)
        df = load_schedule_csv(str(p))

        # build per-row durations (seconds)
        if seg_scope == "layer":
            dur_s, missing_cnt = _build_row_durations_layer_scope(
                df,
                gpu_res=gpu_res,
                pim_res=pim_res,
                comm_model=comm_model,
                pcie_lanes=pcie_lanes,
                decode_stride=decode_stride,
                dim=cfg_dim,
                shards=cfg_shards,
                allow_missing=allow_missing,
            )
        else:
            dur_s, missing_cnt = _build_row_durations_device_step_scope(
                df,
                gpu_res=gpu_res,
                pim_res=pim_res,
                comm_model=comm_model,
                pcie_lanes=pcie_lanes,
                decode_stride=decode_stride,
                dim=cfg_dim,
                shards=cfg_shards,
                allow_missing=allow_missing,
            )

        if dbg_f is not None:
            dbg_f.write(f"\n# schedule={Path(p).name}\n")
            dbg_rows = _collect_segment_debug_rows(
                df,
                seg_scope=seg_scope,
                decode_stride=int(decode_stride),
                gpu_res=gpu_res,
                pim_res=pim_res,
                allow_missing=bool(allow_missing),
            )
            miss_keys = sum(1 for r in dbg_rows if not bool(r.get("found_in_results", False)))
            dbg_f.write(f"# unique_segments={len(dbg_rows)} missing_keys={miss_keys} missing_instances_in_merge={missing_cnt}\n")

            cols = [
                "phase","device_type","step_sample","step_used","step_kind",
                "count","measured_s_raw","trace_avg_s_raw","delta_s_raw","delta_pct_raw",
                "measured_s_scaled","trace_avg_s_scaled","delta_s_scaled","delta_pct_scaled",
                "found_in_results","missing_count","key","ops_repr",
            ]
            dbg_f.write("\t".join(cols) + "\n")

            def _row_line(r: Dict[str, Any]) -> str:
                out: List[str] = []
                for c in cols:
                    v = r.get(c, "")
                    if isinstance(v, float):
                        if np.isfinite(v):
                            out.append(f"{v:.9g}")
                        else:
                            out.append("nan")
                    else:
                        out.append(str(v))
                return "\t".join(out)

            # Top-N by absolute delta (scaled)
            dbg_f.write("# -- top_by_abs_delta_scaled --\n")
            def _abs_delta_scaled(x: Dict[str, Any]) -> float:
                try:
                    v = float(x.get("delta_s_scaled", 0.0) or 0.0)
                    return float(abs(v)) if np.isfinite(v) else float("inf")
                except Exception:
                    return float("inf")
            for r in sorted(dbg_rows, key=_abs_delta_scaled, reverse=True)[:50]:
                dbg_f.write(_row_line(r) + "\n")

            # Full listing in a stable order (easy to diff between runs)
            dbg_f.write("# -- all_segments_sorted --\n")
            def _stable_key(x: Dict[str, Any]) -> Tuple[Any, ...]:
                try:
                    step_u = int(x.get("step_used", 0) or 0)
                except Exception:
                    step_u = 0
                return (str(x.get("phase","")), str(x.get("device_type","")), step_u, str(x.get("ops_repr","")))
            for r in sorted(dbg_rows, key=_stable_key):
                dbg_f.write(_row_line(r) + "\n")

        # busy-time breakdown (already includes decode stride scaling)
        op_l = df["op"].astype(str).str.strip().str.lower()
        is_comm = op_l.isin(_COMM_OPS).to_numpy()
        gpu_busy = float(dur_s[(~is_comm) & (df["device_type"].astype(str) == "npu").to_numpy()].sum())
        pim_busy = float(dur_s[(~is_comm) & (df["device_type"].astype(str) == "pim").to_numpy()].sum())
        comm_busy = float(dur_s[is_comm].sum())

        # prefill block
        prefill_df = df[df["phase"].astype(str) == "prefill"]
        prefill_time_no_weight_load = _simulate_block_overlap(prefill_df, dur_s)

        # --------------------------------------------------
        # Weight load overhead (optional)
        # --------------------------------------------------
        sched_k = str(p)
        weight_load_gpu_s = float(gpu_weight_load_s_global)
        weight_load_pim_s = float(pim_weight_load_s_global)
        weight_load_unknown_s = 0.0

        if isinstance(gpu_weight_load_by_schedule, dict) and sched_k in gpu_weight_load_by_schedule:
            try:
                v = gpu_weight_load_by_schedule.get(sched_k) or {}
                if isinstance(v, dict):
                    weight_load_gpu_s = float(v.get("gpu_s", weight_load_gpu_s) or 0.0)
                    weight_load_unknown_s += float(v.get("unknown_s", 0.0) or 0.0)
            except Exception:
                pass

        if isinstance(pim_weight_load_by_schedule, dict) and sched_k in pim_weight_load_by_schedule:
            try:
                v = pim_weight_load_by_schedule.get(sched_k) or {}
                if isinstance(v, dict):
                    weight_load_pim_s = float(v.get("pim_s", weight_load_pim_s) or 0.0)
                    weight_load_unknown_s += float(v.get("unknown_s", 0.0) or 0.0)
            except Exception:
                pass

        weight_load_total_s = float(weight_load_gpu_s + weight_load_pim_s)
        prefill_time = float(prefill_time_no_weight_load + weight_load_total_s)

        # decode blocks (one per sampled step), strictly sequential across steps
        decode_time = 0.0
        decode_df = df[df["phase"].astype(str) == "decode"]
        steps = sorted(decode_df["step"].unique().tolist()) if len(decode_df) else []
        for s in steps:
            blk = decode_df[decode_df["step"] == s]
            t = _simulate_block_overlap(blk, dur_s)
            decode_time += float(t)

            # step detail row (token index + context len for traceability)
            tok_idx = 1 + int(s) * int(decode_stride)
            ctx_len = (int(cfg_prefill_len) + tok_idx) if cfg_prefill_len is not None else None
            steps_rows.append({
                "schedule": str(Path(p).name),
                "phase": "decode",
                "sample_step": int(s),
                "decode_stride": int(decode_stride),
                "decode_token_index": int(tok_idx),
                "decode_context_len": int(ctx_len) if ctx_len is not None else "",
                "block_time_s": float(t),
            })

        total_time_no_weight_load = float(prefill_time_no_weight_load + decode_time)
        total_time = float(prefill_time + decode_time)

        # Busy-time breakdown: add weight-load to the respective device buckets.
        gpu_busy_with_weight_load = float(gpu_busy + weight_load_gpu_s)
        pim_busy_with_weight_load = float(pim_busy + weight_load_pim_s)

        # --------------------------------------------------
        # Compare with original ops_trace.csv (timeline in the schedule CSV)
        # --------------------------------------------------
        trace_t = compute_trace_times_from_ops_csv(df, decode_stride=decode_stride)
        trace_prefill_s = float(trace_t.get("trace_prefill_s", 0.0) or 0.0)
        trace_decode_s = float(trace_t.get("trace_decode_s", 0.0) or 0.0)
        trace_decode_scaled_s = float(trace_t.get("trace_decode_scaled_s", 0.0) or 0.0)
        trace_total_raw_s = float(trace_t.get("trace_total_raw_s", 0.0) or 0.0)
        trace_total_scaled_s = float(trace_t.get("trace_total_scaled_s", 0.0) or 0.0)

        def _safe_ratio(delta: float, base: float) -> float:
            try:
                base_f = float(base)
                if base_f == 0.0:
                    return float("nan")
                return float(delta) / base_f
            except Exception:
                return float("nan")

        delta_prefill_s = float(prefill_time_no_weight_load - trace_prefill_s)
        delta_decode_s = float(decode_time - trace_decode_scaled_s)
        delta_total_s = float(total_time_no_weight_load - trace_total_scaled_s)

        delta_prefill_ratio = _safe_ratio(delta_prefill_s, trace_prefill_s)
        delta_decode_ratio = _safe_ratio(delta_decode_s, trace_decode_scaled_s)
        delta_total_ratio = _safe_ratio(delta_total_s, trace_total_scaled_s)

        delta_prefill_pct = float(delta_prefill_ratio * 100.0) if np.isfinite(delta_prefill_ratio) else float("nan")
        delta_decode_pct = float(delta_decode_ratio * 100.0) if np.isfinite(delta_decode_ratio) else float("nan")
        delta_total_pct = float(delta_total_ratio * 100.0) if np.isfinite(delta_total_ratio) else float("nan")

        results_rows.append({
            "schedule": str(Path(p).name),
            "segment_scope": seg_scope,
            "comm_model": str(comm_model),
            "pcie_lanes": int(pcie_lanes),
            "decode_stride": int(decode_stride),
            "weight_load_gpu_s": float(weight_load_gpu_s),
            "weight_load_pim_s": float(weight_load_pim_s),
            "weight_load_total_s": float(weight_load_total_s),
            "weight_load_unknown_s": float(weight_load_unknown_s),
            "prefill_time_no_weight_load_s": float(prefill_time_no_weight_load),
            "prefill_time_s": float(prefill_time),
            "decode_time_s": float(decode_time),
            "total_time_no_weight_load_s": float(total_time_no_weight_load),
            "total_time_s": float(total_time),
            "gpu_busy_s": float(gpu_busy_with_weight_load),
            "pim_busy_s": float(pim_busy_with_weight_load),
            "comm_busy_s": float(comm_busy),
            "missing_segments": int(missing_cnt),
            # ---- schedule trace timeline baseline (ops_trace.csv) ----
            "trace_prefill_s": float(trace_prefill_s),
            "trace_decode_s": float(trace_decode_s),
            "trace_decode_scaled_s": float(trace_decode_scaled_s),
            "trace_total_raw_s": float(trace_total_raw_s),
            "trace_total_scaled_s": float(trace_total_scaled_s),
            # ---- delta = (merged_no_weight_load - trace_scaled) ----
            "delta_prefill_s": float(delta_prefill_s),
            "delta_prefill_ratio": float(delta_prefill_ratio),
            "delta_prefill_pct": float(delta_prefill_pct),
            "delta_decode_s": float(delta_decode_s),
            "delta_decode_ratio": float(delta_decode_ratio),
            "delta_decode_pct": float(delta_decode_pct),
            "delta_total_s": float(delta_total_s),
            "delta_total_ratio": float(delta_total_ratio),
            "delta_total_pct": float(delta_total_pct),
        })

    # save CSV (no console print)
    out_csv_p = Path(out_csv)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_rows).to_csv(str(out_csv_p), index=False)

    if out_steps_csv:
        out_steps_p = Path(out_steps_csv)
        out_steps_p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(steps_rows).to_csv(str(out_steps_p), index=False)

    if dbg_f is not None:
        dbg_f.close()

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

    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu (GPU benchmark side)")
    p.add_argument("--gpu-dtype", type=str, default="fp16", dest="gpu_dtype")

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
    p.add_argument("--segment-scope", type=str, default="layer", choices=["layer","device_step"],
                   help="how to form segments from operator-level schedule")

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
    p_exp.add_argument("--comms", type=str, default=None, help="(optional) single comms trace csv path (for weight_load accounting)")
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
    p_m.add_argument("--out-csv", type=str, required=True,
                     help="where to save merged latency report (csv)")
    p_m.add_argument("--out-steps-csv", type=str, default=None,
                     help="(optional) save per-decode-step block latency (csv)")
    p_m.add_argument("--allow-missing", action="store_true",
                     help="if a segment key is missing in results, treat its cost as 0 instead of error")
    p_m.add_argument("--segment-scope", type=str, default=None, choices=["layer","device_step"],
                     help="override segment scope used to parse schedules (default: inferred from results or 'layer')")
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
    p_all.add_argument("--comms", type=str, default=None, help="(optional) single comms trace csv path (for weight_load accounting)")
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
        run_gpu(args.tasks, args.out, warmup=args.warmup, iters=args.iters, device=args.device, gpu_dtype=args.gpu_dtype)
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
        )
        return

    if args.cmd == "merge":
        if (args.gpu_results is None and args.pim_results is None):
            raise SystemExit("merge requires --gpu-results and/or --pim-results")
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        if not schedule_paths:
            raise SystemExit("merge requires --schedule or --schedules")
        merge(schedule_paths, args.gpu_results, args.pim_results,
              comm_model=args.comm_model, pcie_lanes=args.pcie_lanes,
              decode_stride=args.decode_stride,
              out_csv=args.out_csv,
              out_steps_csv=args.out_steps_csv,
              allow_missing=args.allow_missing, segment_scope=args.segment_scope,
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
              comm_model=args.comm_model, pcie_lanes=args.pcie_lanes,
              decode_stride=args.decode_stride,
              out_csv=(args.merge_out_csv or str(Path(out_dir)/f"{prefix}.merge.csv")),
              out_steps_csv=(args.merge_out_steps_csv),
              allow_missing=args.allow_missing, segment_scope=cfg.segment_scope,
              debug=args.debug, debug_txt=args.debug_txt)
        return

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
