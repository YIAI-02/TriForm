#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

1) export
   - 输入：一个或多个 schedule CSV
   - 输出：
     * <prefix>.gpu_tasks.json
     * <prefix>.pim_tasks.json

    python ./verify/schedule_deploy_verify.py export \
        --schedule ./algorithms/output/evaluate_single_test/hardware_config_scale_down_11pima/llama_7b_fp16_b8_s64/algo_hefthint/hefthint_128x128_ops_trace.csv \
        --out-dir ./verify/out \
        --prefix hefthint_seg \
        --segment-scope layer \
        --prefill-len 128 \
        --dim 4096 --ffn-dim 11008 --n-heads 32 --shards 4

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
  --out ./verify/out/hefthint_seg.pim_results.json \
  --cent-sim-root /Users/yangjiaqi/WW/project_1/python/TriForm/submodules/CENT/cent_simulation \
  --pim-ramulator-config ./algorithms/aim_simulator/example.yaml \
  --pim-ramulator-bin ./algorithms/ramulator2 \
  --pim-freq-ghz 1.6 \
  --pim-num-channels 4 --pim-num-banks 8 --pim-num-devices 1

  python ./verify/schedule_deploy_verify.py run-pim \
  --tasks ./verify/out/hefthint_seg.pim_tasks.json \
  --out   ./verify/out/hefthint_seg.pim_results.json \
  --pim-ramulator-config ./algorithms/aim_simulator/example.yaml \
  --pim-hw-json          ./algorithms/aim_simulator/PIM_AiM.json

4) merge
   - 输入：schedule CSV + gpu_results.json + pim_results.json
   - 重新按同样规则切 segment，统计每个 segment 出现次数，做求和/聚合，给出总 latency + speedup 对比。
   python ./verify/schedule_deploy_verify.py merge \
  --schedule ./algorithms/output/evaluate_single_test/hardware_config_scale_down_11pima/llama_7b_fp16_b8_s64/algo_hefthint/hefthint_128x128_ops_trace.csv \
  --gpu-results ./verify/out/hefthint_seg.gpu_results.json \
  --pim-results ./verify/out/hefthint_seg.pim_results.json \
  --comm-model schedule \
  --agg sum

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
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


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

def import_aim_pim(cent_sim_root: Optional[str] = None):
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
    try:
        from aim_sim import PIM  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import aim_sim.PIM from {root}: {e}") from e
    return PIM
    
# ==========================================================
# Schedule parsing
# ==========================================================

_COMM_OPS = {"Identity", "K_write", "V_write"} #communication ops will be skip

def parse_layer(node_id: str) -> int:
    m = re.match(r"L(\d+)_", str(node_id))
    return int(m.group(1)) if m else -1

def parse_shard(node_id: str) -> int:
    m = re.search(r"_S(\d+)", str(node_id))
    return int(m.group(1)) if m else -1

def infer_decode_steps(df: pd.DataFrame) -> int:
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

def load_schedule_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"phase","node_id","op","device","device_type","start","end","duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"schedule csv missing columns: {sorted(missing)}")
    df = df.copy()
    df["_row"] = range(len(df))  # stable tie-breaker
    df = add_step_column(df)
    df["layer"] = df["node_id"].apply(parse_layer)
    df["shard"] = df["node_id"].apply(parse_shard)
    return df


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
    pim_ramulator_bin: str = "ramulator2"
    pim_ramulator_config: Optional[str] = None  # required to enable ramulator mode
    pim_freq_ghz: float = 1.0
    pim_ramulator_timeout_s: int = 300
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
            # default: growing by 1
            K = int(cfg.prefill_len + 1 + sig.step)
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
    - device_type: 'npu' or 'pim'
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
        for (phase, step, layer, device), g in df.groupby(group_cols, sort=False):
            g = g.sort_values(["start","_row"])
            dev_type = str(g["device_type"].iloc[0])
            ops: List[Tuple[str,int]] = []
            for _, r in g.iterrows():
                op = str(r["op"])
                if op in _COMM_OPS:
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
                op = str(r["op"])
                if op in _COMM_OPS:
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
    PIM backend:
      1) Use aim_sim.PIM only_trace primitives to emit an AiM instruction trace.
      2) Run AiM-enabled ramulator2 on that trace and parse "memory_system_cycles".
    """
    def __init__(self, cfg: WorkloadConfig, *, cent_sim_root: Optional[str] = None):
        PIM = import_aim_pim(cent_sim_root)
        if torch is None:
            raise RuntimeError("PyTorch is required for aim_sim (even on CPU).")
        self.cfg = cfg
        self._PIM_cls = PIM

        self._validate_hw_cfg()

        if not cfg.pim_ramulator_config:
            raise RuntimeError(
                "PIM ramulator mode requires a config file.\n"
            )
        self._ramulator_config_base = Path(cfg.pim_ramulator_config).expanduser().resolve()
        if not self._ramulator_config_base.exists():
            raise FileNotFoundError(f"Ramulator config not found: {self._ramulator_config_base}")

        # Optionally rewrite the ramulator config so channels/banks/devices are explicitly specified
        self._ramulator_config = self._ramulator_config_base

    def _validate_hw_cfg(self) -> None:
        cfg = self.cfg
        required_pos_int = [
            ("pim_dram_column", cfg.pim_dram_column),
            ("pim_dram_row", cfg.pim_dram_row),
            ("pim_burst_length", cfg.pim_burst_length),
            ("pim_num_banks", cfg.pim_num_banks),
            ("pim_num_channels", cfg.pim_num_channels),
            ("pim_threads", cfg.pim_threads),
            ("pim_reuse_size", cfg.pim_reuse_size),
            ("pim_num_devices", cfg.pim_num_devices),
        ]
        for name, val in required_pos_int:
            if val is None or int(val) <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val!r}")

        if cfg.pim_freq_ghz is None or float(cfg.pim_freq_ghz) <= 0:
            raise ValueError(f"pim_freq_ghz must be > 0, got {cfg.pim_freq_ghz!r}")

    def _make_pim(self, trace_file: str):
        import types
        args = types.SimpleNamespace()
        # topology (explicit)
        args.DRAM_column = int(self.cfg.pim_dram_column)
        args.DRAM_row = int(self.cfg.pim_dram_row)
        args.burst_length = int(self.cfg.pim_burst_length)
        args.num_banks = int(self.cfg.pim_num_banks)
        args.num_channels = int(self.cfg.pim_num_channels)
        args.threads = int(self.cfg.pim_threads)
        args.reuse_size = int(self.cfg.pim_reuse_size)

        # device count (explicit)
        args.model_parallel = bool(int(self.cfg.pim_num_devices) > 1)
        args.FC_devices = int(self.cfg.pim_num_devices)

        # tracing
        args.only_trace = True
        args.op_trace = False
        args.trace_file = str(trace_file)
        return self._PIM_cls(args)


    def _ramulator_cmd(self, trace_path: Path) -> List[str]:
        exe = str(self.cfg.pim_ramulator_bin or "ramulator2")
        # Prefer local ./ramulator2 if user passes plain name and it exists
        if ("/" not in exe and "\\" not in exe) and (Path.cwd() / exe).exists():
            exe = str((Path.cwd() / exe).resolve())
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
            raise RuntimeError(f"Failed to parse ramulator output for memory_system_cycles.\nCommand: {' '.join(cmd)}\nOutput:\n{out}")
        cycles = int(m.group(1))
        if cycles <= 0:
            raise RuntimeError(f"Invalid cycle count from ramulator: {cycles}")
        return cycles

    def _alloc_trace_dir(self) -> Tuple[Path, bool]:
        """
        Returns (trace_dir, should_cleanup).
        If cfg.pim_keep_traces is True, traces are kept and should_cleanup is False.
        """
        if self.cfg.pim_keep_traces:
            base = Path(self.cfg.pim_trace_dir or "./pim_traces").expanduser().resolve()
            base.mkdir(parents=True, exist_ok=True)
            return base, False
        tmp = Path(tempfile.mkdtemp(prefix="pim_trace_")).resolve()
        return tmp, True

    # ---- internal approximation (still used to generate AiM trace) ----
    def _channels(self) -> List[int]:
        return list(range(int(self.cfg.pim_num_channels)))

    def _simulate_gemv_once(self, pim, vector_dim: int, matrix_col: int, timing_key: str = "MAC_ABK") -> None:
        ch = self._channels()
        if timing_key not in pim.time:
            pim.time[timing_key] = 0

        total_banks = pim.num_channels * pim.num_banks
        burst = pim.burst_length

        col_per_bank = (matrix_col + total_banks - 1) // total_banks
        utilized_banks = (matrix_col + col_per_bank - 1) // col_per_bank

        op_size = (vector_dim + burst - 1) // burst
        # write vector to GB (rough)
        pim.WR_GB_only_trace(ch, op_size)

        pim.WR_BIAS_only_trace(ch)
        pim.MAC_ABK_only_trace(ch, row_index=0, op_size=op_size, timing=timing_key)
        pim.RD_MAC_only_trace(ch)

        # read / store output (rough): proportionate to utilized banks
        pim.time["RD_SBK"] += pim.timing_constant["RD_SBK"] + (utilized_banks * col_per_bank) / burst
        pim.time["WR_SBK"] += pim.timing_constant["WR_SBK"] + (utilized_banks * col_per_bank) / burst

    def _simulate_elementwise(self, pim, num_elements: int, kind: str) -> None:
        ch = self._channels()
        burst = pim.burst_length
        op_size = (num_elements + burst - 1) // burst
        if kind == "mul":
            pim.EWMUL_only_trace(ch, row_index=0, op_size=op_size)
        elif kind == "add":
            pim.EWADD_only_trace(op_size)
        else:
            pim.EWMUL_only_trace(ch, row_index=0, op_size=op_size)

    def _simulate_softmax_row(self, pim, klen: int) -> None:
        # Simplified: two EWMUL passes + some RD/WR (trace only includes EWMUL lines)
        ch = self._channels()
        burst = pim.burst_length
        op_size = (klen + burst - 1) // burst
        pim.EWMUL_only_trace(ch, row_index=0, op_size=op_size)
        pim.time["RD_SBK"] += pim.timing_constant["RD_SBK"] + klen / burst
        pim.time["WR_SBK"] += pim.timing_constant["WR_SBK"] + klen / burst
        pim.EWMUL_only_trace(ch, row_index=0, op_size=op_size)

    def simulate_one(self, pim, sig: OpSig) -> None:
        sh = infer_op_shape(sig, self.cfg)
        D = sh.dim
        Sd = sh.shard_dim
        Fsd = sh.ffn_shard_dim
        T = sh.query_len
        H = sh.heads_per_shard
        Hd = sh.head_dim
        K = sh.key_len

        op = sig.op
        # LN
        if op == "LN":
            # rmsnorm roughly: pow + ew mul
            self._simulate_gemv_once(pim, vector_dim=D, matrix_col=D, timing_key="breakdown_sa_pow")
            self._simulate_elementwise(pim, D, "mul")
            return

        # linear projections
        elif op in ("Q", "K", "V"):
            self._simulate_gemv_once(pim, vector_dim=D, matrix_col=Sd, timing_key="breakdown_sa_weight")
            return

        elif op == "O":
            self._simulate_gemv_once(pim, vector_dim=Sd, matrix_col=D, timing_key="breakdown_sa_weight")
            return

        elif op in ("FFN_W1", "FFN_W3"):
            self._simulate_gemv_once(pim, vector_dim=D, matrix_col=Fsd, timing_key="breakdown_ffn_weight")
            return

        elif op == "FFN_W2":
            self._simulate_gemv_once(pim, vector_dim=Fsd, matrix_col=D, timing_key="breakdown_ffn_weight")
            return

        elif op == "SwiGLU":
            self._simulate_elementwise(pim, T * Fsd, "mul")
            return

        elif op == "Add":
            self._simulate_elementwise(pim, T * D, "add")
            return

        elif op == "QK":
            # q: [H,T,Hd], k: [H,Hd,K] => [H,T,K]
            for _ in range(max(1, H * T)):
                self._simulate_gemv_once(pim, vector_dim=Hd, matrix_col=K, timing_key="breakdown_sa_score")
            return

        elif op == "Softmax":
            for _ in range(max(1, H * T)):
                self._simulate_softmax_row(pim, K)
            return

        elif op == "SV":
            # score: [H,T,K], v: [H,K,Hd] => [H,T,Hd]
            for _ in range(max(1, H * T)):
                self._simulate_gemv_once(pim, vector_dim=K, matrix_col=Hd, timing_key="breakdown_sa_output")
            return

        # Identity / K/V writes are comm-only in schedules; skip
        else:
            raise ValueError(f"Unknown op for GPUBackend: {op}")
            return

    def benchmark_segment(self, seg: SegmentSig) -> float:
        trace_dir, cleanup = self._alloc_trace_dir()
        trace_name = f"{seg.phase}_step{seg.step}_{uuid.uuid4().hex}.trace"
        trace_path = trace_dir / trace_name

        pim = self._make_pim(str(trace_path))
        try:
            for op, shard in seg.ops:
                if op in _COMM_OPS:
                    continue
                sig = OpSig(device_type=seg.device_type, phase=seg.phase, step=seg.step, op=op, shard=shard)
                self.simulate_one(pim, sig)

            # Finish trace
            if hasattr(pim, "finish"):
                pim.finish()
            if getattr(pim, "file", None):
                try:
                    pim.file.flush()
                except Exception:
                    pass
                try:
                    pim.file.close()
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

def load_pim_hw_json(path: str) -> Dict[str, Any]:
    """Load and normalize a minimal PIM HW spec JSON.
    Required keys (any alias below is accepted):
      - DRAM_column / dram_column / column / columns
      - DRAM_row / dram_row / row / rows
      - burst_length / burst / bl
      - num_banks / banks
      - num_channels / channels

    Optional keys:
      - pim_num_devices / num_devices / devices / FC_devices
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
        "pim_num_devices": None,
        "pim_threads": None,
        "pim_reuse_size": None,
        "pim_freq_ghz": None,
    }

    v = _get_any(["pim_num_devices", "num_devices", "devices", "fc_devices"], required=False)
    if v is not None:
        hw["pim_num_devices"] = _as_pos_int("pim_num_devices", v)

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
def export_tasks(schedule_paths: List[str], cfg: WorkloadConfig, out_dir: str, prefix: str) -> Tuple[Path, Path]:
    outp = Path(out_dir).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)

    # load schedules
    dfs: List[pd.DataFrame] = []
    resolved_paths: List[Path] = []
    max_decode_steps = 0
    for sp in schedule_paths:
        p = resolve_existing_path(sp)
        resolved_paths.append(p)
        df = load_schedule_csv(str(p))
        dfs.append(df)
        max_decode_steps = max(max_decode_steps, infer_decode_steps(df))

    # auto-fill decode_context_lens if None
    if cfg.decode_context_lens is None:
        cfg.decode_context_lens = [cfg.prefill_len + 1 + i for i in range(max_decode_steps)] if max_decode_steps > 0 else []

    # collect unique segments over all schedules
    uniq: Dict[str, SegmentSig] = {}
    ctr: Counter = Counter()
    for df in dfs:
        u, c = extract_segments(df, cfg.segment_scope)
        uniq.update(u)
        ctr.update(c)

    gpu_tasks: List[Dict[str, Any]] = []
    pim_tasks: List[Dict[str, Any]] = []
    for k, seg in uniq.items():
        item = {
            "key": k,
            "sig": {"device_type": seg.device_type, "phase": seg.phase, "step": int(seg.step)},
            "ops": [{"op": op, "shard": int(shard)} for op, shard in seg.ops],
            "ops_repr": seg.ops_repr(),
            "count_hint": int(ctr.get(k, 0)),
        }
        if seg.device_type == "npu":
            gpu_tasks.append(item)
        elif seg.device_type == "pim":
            pim_tasks.append(item)
        else:
            raise ValueError(f"Unknown device_type {seg.device_type!r}, expected 'npu' or 'pim'")


    gpu_json = {
        "version": 3,
        "task_type": "segment",
        "backend": "gpu",
        "segment_scope": cfg.segment_scope,
        "schedules": [str(p) for p in resolved_paths],
        "config": cfg.to_dict(),
        "tasks": sorted(gpu_tasks, key=lambda x: x["key"]),
    }
    pim_json = {
        "version": 3,
        "task_type": "segment",
        "backend": "pim",
        "segment_scope": cfg.segment_scope,
        "schedules": [str(p) for p in resolved_paths],
        "config": cfg.to_dict(),
        "tasks": sorted(pim_tasks, key=lambda x: x["key"]),
    }

    gpu_path = outp / f"{prefix}.gpu_tasks.json"
    pim_path = outp / f"{prefix}.pim_tasks.json"
    _save_json(gpu_path, gpu_json)
    _save_json(pim_path, pim_json)

    print(f"[export] segment_scope={cfg.segment_scope}")
    print(f"[export] wrote {gpu_path} ({len(gpu_tasks)} segments)")
    print(f"[export] wrote {pim_path} ({len(pim_tasks)} segments)")
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

    tasks = data.get("tasks", [])
    gpu_info = collect_gpu_info(cfg.device)

    results: Dict[str, float] = {}
    print(f"[run-gpu] tasks={len(tasks)} warmup={warmup} iters={iters} device={cfg.device} dtype={cfg.gpu_dtype} segment_scope={seg_scope}")
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
            ramulator_bin: Optional[str] = None,
            ramulator_timeout_s: Optional[int] = None,
            keep_traces: bool = False,
            trace_dir: Optional[str] = None) -> Path:

    data = _load_json(tasks_json)
    cfg_dict = data.get("config", {}) or {}
    cfg = WorkloadConfig.from_dict(cfg_dict)
    seg_scope = data.get("segment_scope", cfg.segment_scope)
    tasks = data.get("tasks", [])

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
        if hw.get("pim_num_devices") is not None:
            cfg.pim_num_devices = int(hw["pim_num_devices"])  # type: ignore[arg-type]
        if hw.get("pim_threads") is not None:
            cfg.pim_threads = int(hw["pim_threads"])  # type: ignore[arg-type]
        if hw.get("pim_reuse_size") is not None:
            cfg.pim_reuse_size = int(hw["pim_reuse_size"])  # type: ignore[arg-type]
        if hw.get("pim_freq_ghz") is not None:
            cfg.pim_freq_ghz = float(hw["pim_freq_ghz"])  # type: ignore[arg-type]

    # For ramulator config, we require it to be actually present (not empty/None).
    # This can be provided via pim_tasks.json or CLI override.
    if not cfg.pim_ramulator_config:
        raise ValueError(
            "pim_ramulator_config must be provided to run ramulator.\n"
            "Provide it either when exporting tasks (export/all) or at run time: "
            "run-pim --pim-ramulator-config <path> --pim-hw-json <path>."
        )

    backend = PIMBackend(cfg, cent_sim_root=cent_sim_root)

    results: Dict[str, float] = {}
    print(f"[run-pim] tasks={len(tasks)} segment_scope={seg_scope}")
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
        "results": results,
    }
    _save_json(outp, out)
    print(f"[run-pim] wrote {outp}")
    return outp


def _comm_latency_from_df(df: pd.DataFrame, comm_model: str, pcie_lanes: int = 16) -> float:
    comm_model = comm_model.lower()
    comm_df = df[df["op"].isin(_COMM_OPS)]
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


def merge(schedule_paths: List[str], gpu_results_json: Optional[str], pim_results_json: Optional[str],
          *, comm_model: str = "schedule", pcie_lanes: int = 16, agg: str = "sum",
          allow_missing: bool = False, segment_scope: Optional[str] = None) -> None:
    # load results
    gpu_res: Dict[str, float] = {}
    pim_res: Dict[str, float] = {}
    scope_from_results: Optional[str] = None

    if gpu_results_json:
        data = _load_json(gpu_results_json)
        gpu_res = {k: float(v) for k,v in data.get("results", {}).items()}
        scope_from_results = data.get("segment_scope", scope_from_results)

    if pim_results_json:
        data = _load_json(pim_results_json)
        pim_res = {k: float(v) for k,v in data.get("results", {}).items()}
        scope_from_results = data.get("segment_scope", scope_from_results)

    # choose segment scope
    seg_scope = (segment_scope or scope_from_results or "layer")
    print(f"[merge] segment_scope={seg_scope} comm_model={comm_model} agg={agg}")

    totals: List[Tuple[str, Dict[str, float]]] = []
    for sp in schedule_paths:
        p = resolve_existing_path(sp)
        df = load_schedule_csv(str(p))

        if agg == "trace_end":
            total = float(df["end"].max())
            totals.append((str(p), {"TOTAL": total, "GPU": 0.0, "PIM": 0.0, "COMM": 0.0}))
            continue

        # segment counters
        _, ctr = extract_segments(df, seg_scope)

        gpu_t = 0.0
        pim_t = 0.0
        missing: List[str] = []
        for k, c in ctr.items():
            dev = k.split("|", 1)[0]  # device_type
            if dev == "npu":
                if k in gpu_res:
                    gpu_t += float(c) * float(gpu_res[k])
                else:
                    if not allow_missing:
                        missing.append(k)
            elif dev == "pim":
                if k in pim_res:
                    pim_t += float(c) * float(pim_res[k])
                else:
                    if not allow_missing:
                        missing.append(k)

        if missing and not allow_missing:
            raise RuntimeError(
                f"Missing {len(missing)} segment keys in results. "
                f"Example: {missing[0]}. "
                f"Hint: export with union of schedules, then run-gpu/run-pim again; or use --allow-missing."
            )

        comm_t = _comm_latency_from_df(df, comm_model, pcie_lanes=pcie_lanes)

        if agg == "sum":
            total = gpu_t + pim_t + comm_t
        elif agg == "parallel_max":
            total = max(gpu_t, pim_t) + comm_t
        else:
            raise ValueError(f"Unknown agg: {agg}")

        totals.append((str(p), {"GPU": gpu_t, "PIM": pim_t, "COMM": comm_t, "TOTAL": total}))

    # print report
    print("\n=== Merge Report (seconds) ===")
    for path, d in totals:
        print(f"- {Path(path).name}")
        print(f"    GPU  : {d['GPU']:.6f}")
        print(f"    PIM  : {d['PIM']:.6f}")
        print(f"    COMM : {d['COMM']:.6f}")
        print(f"    TOTAL: {d['TOTAL']:.6f}")

    if len(totals) >= 2:
        base = totals[0][1]["TOTAL"]
        print(f"\n=== Speedup vs {Path(totals[0][0]).name} ===")
        for path, d in totals[1:]:
            spd = (base / d["TOTAL"]) if d["TOTAL"] > 0 else float("inf")
            print(f"  {Path(path).name}: {spd:.3f} x")


# ==========================================================
# CLI
# ==========================================================

def add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--ffn-dim", type=int, default=11008, dest="ffn_dim")
    p.add_argument("--n-heads", type=int, default=32, dest="n_heads")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--prefill-len", type=int, default=128, dest="prefill_len")
    p.add_argument("--decode-context-lens", type=str, default=None,
                   help="comma-separated list, e.g. 129,130,131. If omitted, inferred from schedule decode steps.")

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
    p.add_argument("--pim-ramulator-bin", type=str, default="ramulator2", help="path/name of AiM-enabled ramulator2 executable")
    p.add_argument("--pim-ramulator-config", type=str, default=None, help="ramulator2 config file (required for run-pim)")
    p.add_argument("--pim-freq-ghz", type=float, default=1.0, help="PIM clock frequency in GHz for cycles->seconds")
    p.add_argument("--pim-ramulator-timeout-s", type=int, default=300, help="ramulator2 timeout per trace (seconds)")
    p.add_argument("--pim-keep-traces", action="store_true", help="keep generated AiM traces (debug)")
    p.add_argument("--pim-trace-dir", type=str, default=None, help="trace output dir when keeping traces")
    p.add_argument("--pim-no-override-ramulator-config", action="store_true",
                   help="do not rewrite channels/banks/devices into ramulator config (NOT recommended)")


    # segmenting
    p.add_argument("--segment-scope", type=str, default="layer", choices=["layer","device_step"],
                   help="how to form segments from operator-level schedule")

def build_cfg_from_args(args: argparse.Namespace) -> WorkloadConfig:
    cfg = WorkloadConfig(
        dim=args.dim,
        ffn_dim=args.ffn_dim,
        n_heads=args.n_heads,
        shards=args.shards,
        prefill_len=args.prefill_len,
        decode_context_lens=None,
        device=args.device,
        gpu_dtype=args.gpu_dtype,
        pim_dram_column=args.pim_dram_column,
        pim_dram_row=args.pim_dram_row,
        pim_burst_length=args.pim_burst_length,
        pim_num_banks=args.pim_num_banks,
        pim_num_channels=args.pim_num_channels,
        pim_threads=args.pim_threads,
        pim_reuse_size=args.pim_reuse_size,
        pim_num_devices=args.pim_num_devices,
        pim_ramulator_bin=args.pim_ramulator_bin,
        pim_ramulator_config=args.pim_ramulator_config,
        pim_freq_ghz=args.pim_freq_ghz,
        pim_ramulator_timeout_s=args.pim_ramulator_timeout_s,
        pim_keep_traces=bool(args.pim_keep_traces),
        pim_trace_dir=args.pim_trace_dir,
        segment_scope=args.segment_scope,
    )
    if args.decode_context_lens:
        cfg.decode_context_lens = [int(x) for x in str(args.decode_context_lens).split(",") if x.strip()]
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    subcmds = {"export","run-gpu","run-pim","merge","all"}

    # Backward compatibility: if user calls without subcommand, treat as "all"
    if not argv or argv[0].startswith("-") or argv[0] not in subcmds:
        argv = ["all"] + argv

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # export
    p_exp = sub.add_parser("export", help="export gpu_tasks.json and pim_tasks.json (segment-level) from schedule csv(s)")
    p_exp.add_argument("--schedule", type=str, default=None, help="single schedule csv path")
    p_exp.add_argument("--schedules", type=str, nargs="+", default=None, help="schedule csv paths (one or more)")
    p_exp.add_argument("--out-dir", type=str, default=".", help="output directory")
    p_exp.add_argument("--prefix", type=str, default="tasks", help="output file prefix")
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
    p_pim.add_argument("--pim-ramulator-config", type=str, required=True,
                       help="ramulator2 config file (YAML/JSON), e.g. example.yaml")
    p_pim.add_argument("--pim-hw-json", type=str, required=True,
                       help="PIM HW spec JSON, e.g. PIM_AiM.json")


    # merge
    p_m = sub.add_parser("merge", help="merge schedule(s) with gpu_results.json + pim_results.json (segment-level)")
    p_m.add_argument("--schedule", type=str, default=None, help="single schedule csv path")
    p_m.add_argument("--schedules", type=str, nargs="+", default=None)
    p_m.add_argument("--gpu-results", type=str, default=None)
    p_m.add_argument("--pim-results", type=str, default=None)
    p_m.add_argument("--comm-model", type=str, default="schedule", choices=["schedule","cxl","none"])
    p_m.add_argument("--pcie-lanes", type=int, default=16)
    p_m.add_argument("--agg", type=str, default="sum", choices=["sum","parallel_max","trace_end"])
    p_m.add_argument("--allow-missing", action="store_true",
                     help="if a segment key is missing in results, treat its cost as 0 instead of error")
    p_m.add_argument("--segment-scope", type=str, default=None, choices=["layer","device_step"],
                     help="override segment scope used to parse schedules (default: inferred from results or 'layer')")

    # all (single-machine end-to-end)
    p_all = sub.add_parser("all", help="single-machine mode: export -> run both -> merge")
    p_all.add_argument("--schedule", type=str, default=None, help="single schedule csv path")
    p_all.add_argument("--schedules", type=str, nargs="+", default=None)
    p_all.add_argument("--out-dir", type=str, default=".", help="where to place tasks/results")
    p_all.add_argument("--prefix", type=str, default="run", help="prefix for tasks/results files")
    p_all.add_argument("--warmup", type=int, default=3)
    p_all.add_argument("--iters", type=int, default=10)
    p_all.add_argument("--cent-sim-root", type=str, default=None)
    p_all.add_argument("--comm-model", type=str, default="schedule", choices=["schedule","cxl","none"])
    p_all.add_argument("--pcie-lanes", type=int, default=16)
    p_all.add_argument("--agg", type=str, default="sum", choices=["sum","parallel_max","trace_end"])
    p_all.add_argument("--allow-missing", action="store_true")
    add_common_model_args(p_all)

    args = parser.parse_args(argv)

    if args.cmd == "export":
        cfg = build_cfg_from_args(args)
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        if not schedule_paths:
            raise SystemExit("export requires --schedule or --schedules")
        export_tasks(schedule_paths, cfg, args.out_dir, args.prefix)
        return

    if args.cmd == "run-gpu":
        run_gpu(args.tasks, args.out, warmup=args.warmup, iters=args.iters, device=args.device, gpu_dtype=args.gpu_dtype)
        return

    if args.cmd == "run-pim":
        run_pim(
            args.tasks,
            args.out,
            cent_sim_root=args.cent_sim_root,
            ramulator_config=args.pim_ramulator_config,
            pim_hw_json=args.pim_hw_json,
        )
        return

    if args.cmd == "merge":
        if args.agg != "trace_end" and (args.gpu_results is None and args.pim_results is None):
            raise SystemExit("merge requires --gpu-results and/or --pim-results unless --agg trace_end")
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        if not schedule_paths:
            raise SystemExit("merge requires --schedule or --schedules")
        merge(schedule_paths, args.gpu_results, args.pim_results,
              comm_model=args.comm_model, pcie_lanes=args.pcie_lanes, agg=args.agg,
              allow_missing=args.allow_missing, segment_scope=args.segment_scope)
        return

    if args.cmd == "all":
        cfg = build_cfg_from_args(args)
        out_dir = args.out_dir
        prefix = args.prefix
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        if not schedule_paths:
            raise SystemExit("all requires --schedule or --schedules")

        # 1) export
        gpu_tasks, pim_tasks = export_tasks(schedule_paths, cfg, out_dir, prefix)

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
                                   cent_sim_root=args.cent_sim_root)

        # 3) merge
        merge(schedule_paths,
              str(gpu_res_path) if gpu_res_path else None,
              str(pim_res_path) if pim_res_path else None,
              comm_model=args.comm_model, pcie_lanes=args.pcie_lanes, agg=args.agg,
              allow_missing=args.allow_missing, segment_scope=cfg.segment_scope)
        return

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
