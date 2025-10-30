# cost_model.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import subprocess
import tempfile
import re
import sys
import shutil
import hashlib
import pickle
from threading import Lock
import time
from datetime import datetime
from collections import defaultdict

from task_graph import TaskGraph, TaskNode
from hardware import Cluster, DeviceSpec
from plan_label import PlanLabel
from config import (
    HOST_NAME, DEVICE_PREFERRED_FORMAT,
    FORMAT_SIZE_MULTIPLIER, FORMAT_CONV_BW_GBs,PIM_FREQ_GHZ,
)

DTYPE_BYTES: Dict[str, int] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
}

# =========================
# CENT trace generation helpers (from 01_gentrace.py)
# =========================

def _ensure_cent_on_path(start: Optional[Path] = None) -> Tuple[Path, Path]:
    here = (start or Path(__file__)).resolve()
    for p in [here.parent] + list(here.parents):
        cand = p / "submodules" / "CENT" / "cent_simulation"
        if cand.exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand, p
    raise RuntimeError(f"Cannot find 'submodules/CENT/cent_simulation' above {here}")

def _load_pim_config(path: Path) -> Dict[str, any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    std: Dict[str, any] = {}
    
    alias = {
        "DRAM_column": ["dram_column", "DRAMCol", "dramCol", "dram_col"],
        "DRAM_row": ["dram_row", "DRAMRow", "dramRow"],
        "burst_length": ["burst", "burstLength", "BL"],
        "num_banks": ["banks", "numBanks"],
        "num_channels": ["channels", "numChannels"],
        "threads": ["thread", "nThreads"],
        "reuse_size": ["reuseSize", "reuse", "RS"],
        "channels_per_block": ["channelsPerBlock", "cpb"],
        "max_seq_len": ["maxSeqLen", "max_seq_length"],
    }
    
    for k, v in cfg.items():
        matched = False
        for stdk, alist in alias.items():
            if k in alist or k == stdk:
                std[stdk] = v
                matched = True
                break
        if not matched:
            std[k] = v
    
    std.setdefault("DRAM_column", 256)
    std.setdefault("DRAM_row", 64)
    std.setdefault("burst_length", 16)
    std.setdefault("num_banks", 8)
    std.setdefault("num_channels", 4)
    std.setdefault("threads", 1)
    std.setdefault("reuse_size", 32)
    std.setdefault("channels_per_block", None)
    std.setdefault("max_seq_len", 4096)
    
    return std

def _make_tb_args_from_pim(cfg: Dict[str, any], trace_file: str):
    """创建 TransformerBlock 参数"""
    from types import SimpleNamespace
    
    cpb = cfg["channels_per_block"]
    if cpb is None:
        cpb = cfg["num_channels"]
    
    return SimpleNamespace(
        DRAM_column        = int(cfg["DRAM_column"]),
        DRAM_row           = int(cfg["DRAM_row"]),
        burst_length       = int(cfg["burst_length"]),
        num_banks          = int(cfg["num_banks"]),
        num_channels       = int(cfg["num_channels"]),
        threads            = int(cfg["threads"]),
        reuse_size         = int(cfg["reuse_size"]),
        channels_per_block = int(cpb),
        max_seq_len        = int(cfg["max_seq_len"]),
        only_trace         = True,
        op_trace           = False,
        trace_file         = trace_file,
        pim_compute        = True,
        model              = "llama_like",
        embedding          = "rope",
        seqlen             = 16,
        model_parallel     = False,
        FC_devices         = 1,
        pipeline_parallel  = False,
        inter_device_attention = False,
        only_FC            = False,
        trace_prepare      = False,
        trace_norm         = False,
        trace_fc_kqvo      = False,
        trace_attention    = False,
        trace_softmax      = False,
        trace_fc_ffn       = False,
        trace_activation   = False,
        GEMV               = "reuse-GB",
    )

def _calc_channels(block):
    """计算通道列表"""
    if getattr(block, "model_parallel", False):
        FC_total_banks = int(block.total_banks) * int(block.FC_devices)
        channels_required = int(block.num_channels)
    else:
        FC_total_banks = int(block.total_banks)
        channels_required = int(block.channels_per_block)

    num_channels = int(block.num_channels)
    channels_required = int(channels_required)

    channel_multi_tb_required = int((num_channels // channels_required) * channels_required)
    channel_lst = [channel for channel in range(channel_multi_tb_required)]
    
    return channel_lst, FC_total_banks, channels_required

def _emit_single_op_trace(
    block,
    op: str,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_dim: int,
    seqlens: Optional[List[int]]
):

    channel_lst, FC_total_banks, channels_required = _calc_channels(block)
    head_dim = dim // max(1, n_heads)

    if op in ("q_proj", "k_proj", "v_proj", "wo_proj", "ffn_up", "ffn_gate", "ffn_down"):
        if op == "q_proj":
            row_tag, V, N = "wq_row_index", dim, dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_sa_weight")
        elif op == "k_proj":
            row_tag, V, N = "wk_row_index", dim, n_kv_heads * head_dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_sa_weight")
        elif op == "v_proj":
            row_tag, V, N = "wv_row_index", dim, n_kv_heads * head_dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_sa_weight")
        elif op == "wo_proj":
            row_tag, V, N = "wo_row_index", dim, dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_sa_weight")
        elif op == "ffn_up":
            row_tag, V, N = "w1_row_index", dim, ffn_dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_ffn_weight")
        elif op == "ffn_gate":
            row_tag, V, N = "w3_row_index", dim, ffn_dim
            block.Vector_Matrix_Mul_weight_af_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_ffn_weight")
        elif op == "ffn_down":
            row_tag, V, N = "w2_row_index", ffn_dim, dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_ffn_weight")

    elif op in ("score", "softmax", "output"):
        for S in (seqlens or [1]):
            if op == "score":
                block.Vector_Matrix_Mul_score_pim_only_trace(block.cache_k_row_index, S, "breakdown_sa_score")
            elif op == "output":
                block.Vector_Matrix_Mul_output_pim_only_trace(block.cache_v_row_index, S, "breakdown_sa_output")
            elif op == "softmax":
                rows_per_score = (S - 1) // block.DRAM_column + 1
                for r in range(rows_per_score):
                    op_size = (block.DRAM_column // block.burst_length
                               if r < rows_per_score - 1
                               else (S - block.DRAM_column * r - 1) // block.burst_length + 1)
                    block.EWMUL_only_trace(channel_lst, block.scores_row_index + r, op_size)

                block.time["RD_SBK"] += block.timing_constant["RD_SBK"] + (S * block.n_heads) // block.burst_length
                block.load_from_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 2, S)

                block.time["WR_SBK"] += block.timing_constant["WR_SBK"] + (S * block.n_heads) // block.burst_length
                block.store_for_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 0, S)

                rows_per_score = (S - 1) // block.DRAM_column + 1
                for r in range(rows_per_score):
                    op_size = (block.DRAM_column // block.burst_length
                               if r < rows_per_score - 1
                               else (S - block.DRAM_column * r - 1) // block.burst_length + 1)
                    block.EWMUL_only_trace(channel_lst, block.scores_row_index + r, op_size)

                block.load_from_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 2, S)

    elif op == "rmsnorm":
        input_len = (dim - 1) // (block.total_banks // 2) + 1
        block.WR_BIAS_only_trace(channel_lst)
        block.MAC_ABK_only_trace(channel_lst, block.x_row_index, (input_len - 1) // block.burst_length + 1, "breakdown_sa_pow")
        block.RD_MAC_only_trace(channel_lst)

        ew_len = (dim - 1) // (block.total_banks // 4) + 1
        ew_banks = (dim - 1) // ew_len + 1

        block.time["WR_SBK"] += block.timing_constant["WR_SBK"] + dim // block.burst_length
        block.store_for_EWMUL_input_only_trace(channels_required, ew_banks, 1, block.x_copy_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.x_copy_row_index, (ew_len - 1) // block.burst_length + 1)

        for bank in range(block.num_banks):
            if bank % 4 == 2:
                block.COPY_BK_GB_only_trace(channel_lst, bank, block.x_copy_row_index, (ew_len - 1) // block.burst_length + 1)
                block.COPY_GB_BK_only_trace(channel_lst, bank - 1, block.SANorm_row_index, (ew_len - 1) // block.burst_length + 1)

        block.EWMUL_only_trace(channel_lst, block.SANorm_row_index, (ew_len - 1) // block.burst_length + 1)
        block.time["RD_SBK"] += block.timing_constant["RD_SBK"] + block.dim // block.burst_length
        block.load_from_EWMUL_input_only_trace(channels_required, ew_banks, 2, block.SANorm_row_index, ew_len)
        block.SYNC_only_trace()

    elif op == "rope":
        ew_len = (head_dim - 1) // (block.total_banks // 4) + 1
        ew_size = (ew_len - 1) // block.burst_length + 1
        block.store_for_EWMUL_input_only_trace(block.channels_per_block, (dim - 1) // ew_len + 1, 1, block.xq_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.xq_row_index, ew_size)
        block.store_for_EWMUL_input_only_trace(block.channels_per_block, (dim - 1) // ew_len + 1, 1, block.xk_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.xk_row_index, ew_size)

    elif op in ("silu", "gelu"):
        ew_len = (ffn_dim - 1) // (block.total_banks // 4) + 1
        ew_banks = (ffn_dim - 1) // ew_len + 1
        block.time["WR_SBK"] += block.timing_constant["WR_SBK"] + ffn_dim // block.burst_length
        block.store_for_EWMUL_input_only_trace(block.channels_per_block, ew_banks, 1, block.ffn_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.ffn_row_index, (ew_len - 1) // block.burst_length + 1)
        for bank in range(block.num_banks):
            if bank % 4 == 2:
                block.COPY_BK_GB_only_trace(channel_lst, bank, block.ffn_row_index, (ew_len - 1) // block.burst_length + 1)
                block.COPY_GB_BK_only_trace(channel_lst, bank - 1, block.ffn_row_index, (ew_len - 1) // block.burst_length + 1)
        block.EWMUL_only_trace(channel_lst, block.ffn_row_index, (ew_len - 1) // block.burst_length + 1)
        block.time["RD_SBK"] += block.timing_constant["RD_SBK"] + ffn_dim // block.burst_length
        block.SYNC_only_trace()

    elif op == "residual":
        op_size = block.dim // block.burst_length
        block.EWADD_only_trace(op_size)

    else:
        raise ValueError(f"Unsupported op: {op}")

    if hasattr(block, "file") and block.file:
        block.file.write("AiM EOC\n")
        block.file.flush()


def _generate_pim_trace(
    op: str,
    pim_config: Path,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_dim: int,
    seqlen: Optional[int],
    trace_file: Path
) -> None:
    try:
        _ensure_cent_on_path()
        from Llama import TransformerBlockLlama as TransformerBlock
    except Exception:
        try:
            from TransformerBlock import TransformerBlock
        except ImportError:
            raise RuntimeError("Cannot import TransformerBlock from CENT")

    pim_cfg = _load_pim_config(pim_config)
    args = _make_tb_args_from_pim(pim_cfg, str(trace_file))
    args.op_trace = True
    args.seqlen = int(seqlen or args.seqlen or 16)


    def _make_dic_model(dim_: int, n_heads_: int, n_kv_heads_: int, seqlen_: int, ffn_dim_: int):
        import torch
        head_dim_ = dim_ // max(1, n_heads_)
        TP_param = 1
        return {
            "TP_param": torch.tensor(TP_param),
            "dim": torch.tensor(dim_),
            "n_heads": torch.tensor(n_heads_),
            "n_kv_heads": torch.tensor(n_kv_heads_),
            "x": torch.zeros((1, 1, dim_)),
            "SANorm": torch.zeros(dim_),
            "FFNNorm": torch.zeros(dim_),
            "sa": torch.zeros((1, 1, dim_)),
            "h": torch.zeros((1, 1, dim_)),
            "out": torch.zeros((1, 1, dim_)),
            "wq": torch.zeros((dim_ // TP_param, dim_)),
            "wk": torch.zeros((head_dim_ * n_kv_heads_), dim_),
            "wv": torch.zeros((head_dim_ * n_kv_heads_), dim_),
            "xq": torch.zeros((1, 1, dim_)),
            "xk": torch.zeros((1, 1, head_dim_ * n_heads_)),
            "xv": torch.zeros((1, 1, head_dim_ * n_heads_)),
            "start_pos": torch.tensor(max(1, seqlen_) - 1),
            "cache_k": torch.zeros((1, seqlen_, n_kv_heads_, head_dim_)),
            "cache_v": torch.zeros((1, seqlen_, n_kv_heads_, head_dim_)),
            "scores": torch.zeros((1, n_heads_, 1, seqlen_)),
            "output": torch.zeros((1, 1, dim_)),
            "wo": torch.zeros((dim_ // TP_param, dim_)),
            "w1": torch.zeros((ffn_dim_ // TP_param, dim_)),
            "w3": torch.zeros((ffn_dim_ // TP_param, dim_)),
            "w2": torch.zeros((dim_ // TP_param, ffn_dim_)),
            "ffn": torch.zeros((1, 1, dim_)),
        }

    dic_model = _make_dic_model(dim, n_heads, n_kv_heads, args.seqlen, ffn_dim)
    block = TransformerBlock(dic_model, args)
    if hasattr(block, "memory_mapping"):
        block.memory_mapping()

    seqlens_list = [seqlen] if seqlen else None
    _emit_single_op_trace(block, op, dim, n_heads, n_kv_heads, ffn_dim, seqlens_list)

    if hasattr(block, "file") and block.file:
        block.file.flush()
        block.file.close()

    if not trace_file.exists():
        raise RuntimeError(f"Trace file not generated: {trace_file}")
    if trace_file.stat().st_size == 0:
        raise RuntimeError(f"Trace file is empty: {trace_file}")

# =========================
# Ramulator runner
# =========================

def _run_ramulator(trace_path: Path, ramulator_config: Path, timeout: int = 300) -> int:
    cmd = f"./ramulator2 -f {ramulator_config} -t {trace_path}"
    
    print(f"[PIM] Running ramulator: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout 
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Ramulator execution timeout after {timeout}s for {trace_path}")
    
    if result.returncode != 0:
        raise RuntimeError(f"Ramulator failed with return code {result.returncode}:\n{result.stderr}")

    pattern = r"memory_system_cycles:\s*([0-9]+)"
    match = re.search(pattern, result.stdout)
    
    if not match:
        print(f"[PIM] Ramulator stdout:\n{result.stdout}")
        print(f"[PIM] Ramulator stderr:\n{result.stderr}")
        raise RuntimeError(f"Could not parse cycles from ramulator output")
    
    cycles = int(match.group(1))
    print(f"[PIM] Ramulator completed: {cycles} cycles")
    
    return cycles

# =========================
# PIM latency cache
# =========================

class PIMLatencyCache:
    """缓存 PIM 延迟结果，避免重复仿真"""
    
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or Path(".pim_latency_cache.pkl")
        self.cache: Dict[str, float] = {}
        self.lock = Lock()
        self._load_cache()
    
    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"[PIM Cache] Loaded {len(self.cache)} entries from {self.cache_file}")
            except Exception as e:
                print(f"[PIM Cache] Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"[PIM Cache] Failed to save cache: {e}")
    
    def _make_key(self, op: str, dim: int, n_heads: int, n_kv_heads: int, 
                  ffn_dim: int, seqlen: Optional[int],
                  pim_config: Path, ramulator_config: Path) -> str:
        params = f"{op}_{dim}_{n_heads}_{n_kv_heads}_{ffn_dim}_{seqlen}"
        configs = f"{pim_config.name}_{ramulator_config.name}"
        key = f"{params}_{configs}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, op: str, dim: int, n_heads: int, n_kv_heads: int,
            ffn_dim: int, seqlen: Optional[int],
            pim_config: Path, ramulator_config: Path) -> Optional[float]:
        key = self._make_key(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen, 
                            pim_config, ramulator_config)
        with self.lock:
            return self.cache.get(key)
    
    def set(self, op: str, dim: int, n_heads: int, n_kv_heads: int,
            ffn_dim: int, seqlen: Optional[int],
            pim_config: Path, ramulator_config: Path, latency: float):
        key = self._make_key(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen,
                            pim_config, ramulator_config)
        with self.lock:
            self.cache[key] = latency
            self._save_cache()

_pim_cache = PIMLatencyCache()

# =========================
# Combined PIM trace-based latency
# =========================

# 全局统计和日志
class SimulationLogger:
    """仿真日志和统计"""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("pim_simulation.log")
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # 统计不重复的算子配置
        self.simulated_ops: Dict[str, set] = defaultdict(set)
        self.lock = Lock()
        
        # 确保目录存在
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 打开日志文件
        self._log_handle = open(self.log_file, 'w', encoding='utf-8')
        self._log(f"{'='*80}")
        self._log(f"PIM Simulation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"{'='*80}\n")
    
    def _log(self, message: str):
        """写入日志文件并打印到控制台"""
        print(message)
        self._log_handle.write(message + '\n')
        self._log_handle.flush()
    
    def start_simulation(self):
        """开始计时"""
        self.start_time = time.time()
        self._log(f"\n{'='*80}")
        self._log(f"Simulation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self._log(f"{'='*80}\n")
    
    def end_simulation(self):
        """结束计时并统计"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time if self.start_time else 0.0
        
        self._log(f"\n{'='*80}")
        self._log(f"Simulation Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self._log(f"Total Simulation Time: {elapsed:.3f} seconds ({elapsed/60:.2f} minutes)")
        self._log(f"{'='*80}\n")
        
        self._print_statistics()
    
    def record_simulation(self, op: str, dim: int, n_heads: int, n_kv_heads: int, 
                         ffn_dim: int, seqlen: Optional[int]):
        """记录一次仿真"""
        with self.lock:
            config = (dim, n_heads, n_kv_heads, ffn_dim, seqlen or 0)
            self.simulated_ops[op].add(config)
    
    def _print_statistics(self):
        """打印统计信息"""
        self._log(f"\n{'='*80}")
        self._log("Simulated Operations Summary")
        self._log(f"{'='*80}")
        
        total_unique = sum(len(configs) for configs in self.simulated_ops.values())
        self._log(f"\nTotal unique operations simulated: {total_unique}")
        self._log(f"Total operation types: {len(self.simulated_ops)}\n")
        
        for op in sorted(self.simulated_ops.keys()):
            configs = self.simulated_ops[op]
            self._log(f"\n{op.upper()}:")
            self._log(f"  - Unique configurations: {len(configs)}")
            
            for config in sorted(configs):
                dim, n_heads, n_kv_heads, ffn_dim, seqlen = config
                self._log(f"    * dim={dim}, heads={n_heads}, kv_heads={n_kv_heads}, "
                         f"ffn_dim={ffn_dim}, seqlen={seqlen if seqlen > 0 else 'None'}")
        
        self._log(f"\n{'='*80}\n")
    
    def close(self):
        """关闭日志文件"""
        if self._log_handle and not self._log_handle.closed:
            self._log_handle.close()

# 全局日志实例
_sim_logger: Optional[SimulationLogger] = None

def get_simulation_logger(log_file: Optional[Path] = None) -> SimulationLogger:
    """获取全局日志实例"""
    global _sim_logger
    if _sim_logger is None:
        _sim_logger = SimulationLogger(log_file)
    return _sim_logger

def _get_pim_latency_via_trace(
    op: str,
    pim_config: Path,
    ramulator_config: Path,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_dim: int,
    seqlen: Optional[int],
    use_cache: bool = True
) -> float:
    """
    Generate trace and run Ramulator, returning the latency (in seconds).
    This function blocks until Ramulator finishes the simulation and returns the result.
    """
    logger = get_simulation_logger()
    logger.record_simulation(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen)
    
    if use_cache:
        cached = _pim_cache.get(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen,
                               pim_config, ramulator_config)
        if cached is not None:
            msg = f"[PIM Cache] Hit for {op} (dim={dim}, heads={n_heads}, seq={seqlen})"
            logger._log(msg)
            return cached
    
    msg = f"[PIM] Computing latency for {op} (dim={dim}, heads={n_heads}, seq={seqlen})"
    logger._log(msg)
    
    # temp dir
    temp_dir = Path(tempfile.mkdtemp(prefix="pim_trace_"))
    
    try:
        # generate trace
        trace_path = temp_dir / f"{op}_trace.trace"
        print(f"[PIM] Generating trace: {trace_path}")
        _generate_pim_trace(
            op=op,
            pim_config=pim_config,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_dim=ffn_dim,
            seqlen=seqlen,
            trace_file=trace_path
        )
        print(f"[PIM] Trace generation completed")
        print(f"[PIM] Starting ramulator simulation...")
        cycles = _run_ramulator(trace_path, ramulator_config)
        
        # convert cycles to latency
        if PIM_FREQ_GHZ > 0.0:
            latency = float(cycles) / (PIM_FREQ_GHZ * 1e9)
        else:
            latency = 0.0
        
        print(f"[PIM] Latency computed: {latency:.6e} seconds ({cycles} cycles)")
        
        if use_cache:
            _pim_cache.set(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen,
                          pim_config, ramulator_config, latency)
        
        return latency
    
    except Exception as e:
        print(f"[PIM] Error during latency computation: {e}")
        raise
    
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"[PIM] Warning: Failed to cleanup temp dir {temp_dir}: {e}")

# =========================
# CostModel
# =========================

class CostModel:
    def __init__(
        self,
        cluster: Cluster,
        dtype: str = "fp16",
        pim_config_path: Optional[Path] = None,
        ramulator_config_path: Optional[Path] = None,
        pim_cache_enabled: bool = True,
        simulation_log_file: Optional[Path] = None
    ):
        self.cluster = cluster
        self.dtype = dtype
        self.pim_config_path = pim_config_path
        self.ramulator_config_path = ramulator_config_path
        self.pim_cache_enabled = pim_cache_enabled
        self.logger = get_simulation_logger(simulation_log_file)
        
        # PIM 模式需要的配置检查
        if pim_config_path:
            if not pim_config_path.exists():
                raise ValueError(f"PIM config not found: {pim_config_path}")
        
        if ramulator_config_path:
            if not ramulator_config_path.exists():
                raise ValueError(f"Ramulator config not found: {ramulator_config_path}")

    # --------------------------
    # Basic times
    # --------------------------
    def flop_time(self, flops: float, dev: DeviceSpec) -> float:
        if dev.tflops <= 0:
            return 0.0
        return flops / (dev.tflops * 1e12)

    def mem_time(self, bytes_amount: int, dev: DeviceSpec) -> float:
        bw = dev.mem_bw_GBs * 1e9
        return 0.0 if bw <= 0 else bytes_amount / bw

    def link_time(self, bytes_amount: int, src: DeviceSpec, dst: DeviceSpec) -> float:
        bw = self.cluster.get_link_bw(src.name, dst.name) * 1e9
        return 0.0 if bw <= 0 else bytes_amount / bw

    def comm_cost(self, src: DeviceSpec, dst: DeviceSpec, bytes_amount: int) -> float:
        if src.name == dst.name:
            return 0.0
        return self.link_time(bytes_amount, src, dst)

    # --------------------------
    # Format helpers
    # --------------------------
    def get_host_device(self) -> DeviceSpec:
        if HOST_NAME in self.cluster.devices:
            return self.cluster.devices[HOST_NAME]
        cpus = self.cluster.devices_by_type("cpu")
        return cpus[0] if cpus else next(iter(self.cluster.devices.values()))

    def device_preferred_fmt(self, dev: DeviceSpec) -> str:
        return DEVICE_PREFERRED_FORMAT.get(dev.type, "ND")

    def format_size(self, size_bytes: int, fmt: str) -> int:
        m = float(FORMAT_SIZE_MULTIPLIER.get(fmt, 1.0))
        return int(size_bytes * m)

    def format_conversion_time(self, size_src_bytes: int, src_fmt: str, dst_fmt: str, dev: DeviceSpec) -> float:
        if src_fmt == dst_fmt:
            return 0.0
        bw_gbs = float(FORMAT_CONV_BW_GBs.get(dev.type, FORMAT_CONV_BW_GBs.get("default", 50.0)))
        bw = bw_gbs * 1e9
        return 0.0 if bw <= 0 else size_src_bytes / bw

    def gb_move_and_format(self, dev: DeviceSpec, size_src_bytes: int, src_fmt: str, dst_fmt: str) -> float:
        host = self.get_host_device()
        t_move = self.link_time(size_src_bytes, host, dev)
        t_conv = self.format_conversion_time(size_src_bytes, src_fmt, dst_fmt, dev)
        return max(t_move, t_conv)

    # --------------------------
    # PIM op -> key mapping
    # --------------------------
    def _resolve_pim_key(self, node) -> List[str]:
        """将 node 映射到 PIM op key"""
        keys: List[str] = []
        name = (node.name or "").upper()
        
        # Attention projections
        if name in ("Q", "Q_PROJ"):
            keys.append("q_proj")
        elif name in ("K", "K_PROJ"):
            keys.append("k_proj")
        elif name in ("V", "V_PROJ"):
            keys.append("v_proj")
        elif name in ("O", "WO", "WO_PROJ", "O_PROJ"):
            keys.append("wo_proj")
        
        # FFN layers
        elif name in ("FFN_W1", "FFN_UP"):
            keys.append("ffn_up")
        elif name in ("FFN_W3", "FFN_GATE"):
            keys.append("ffn_gate")
        elif name in ("FFN_W2", "FFN_DOWN"):
            keys.append("ffn_down")
        
        # Attention operations
        elif ("QK" in name) or ("SCORE" in name):
            keys.append("score")
        elif ("SV" in name) or ("OUTPUT" in name):
            keys.append("output")
        elif "SOFTMAX" in name:
            keys.append("softmax")
        
        # Other ops
        elif "RMSNORM" in name or name == "LN":
            keys.append("rmsnorm")
        elif "ROPE" in name:
            keys.append("rope")
        elif "SILU" in name or "SWIGLU" in name:
            keys.append("silu")
        elif "GELU" in name:
            keys.append("gelu")
        elif name in ("ADD", "RESIDUAL"):
            keys.append("residual")
        
        return keys

    # --------------------------
    # Dynamic flop estimation (保持不变)
    # --------------------------
    def estimate_flops(self, node, batch: int, seq_len: int, phase: str) -> float:
        attrs = getattr(node, "attrs", {}) or {}
        default = float(getattr(node, "flops", 0.0) or 0.0)

        b = int(batch or attrs.get("batch", 0) or 0)
        if b <= 0:
            return default

        D   = int(attrs.get("dim", 0) or 0)
        Hf  = int(attrs.get("ffn_dim", attrs.get("hidden_dim", 0)) or 0)
        qh  = int(attrs.get("q_heads", attrs.get("n_head", attrs.get("kv_heads", 0))) or 0)
        kvh = int(attrs.get("kv_heads", attrs.get("n_kv_heads", qh)) or 0)
        hd  = int(attrs.get("head_dim", D // max(qh, 1)) or 0)

        q_dim = int(attrs.get("q_dim", qh * hd) or 0)
        kv_dim = int(attrs.get("kv_dim", kvh * hd) or 0)
        o_dim  = int(attrs.get("o_dim", qh * hd) or 0)
        q_len  = seq_len if phase == "prefill" else 1
        kv_len = int(attrs.get("kv_len", attrs.get("past_kv_len", seq_len)) or seq_len)
        causal = bool(attrs.get("causal", True))
        
        def tri(n: int) -> int:
            return n * (n + 1) // 2

        C_MATMUL  = 2.0
        C_LN      = 5.0
        C_SOFTMAX = 5.0
        C_GELU    = 6.0
        C_SILU    = 5.0

        name = (getattr(node, "name", "") or "").upper()

        if name in ("LN") and D > 0:
            return float(b * q_len * D * C_LN)

        if name in ("Q", "K", "V") and D > 0:
            out_dim = q_dim if name == "Q" else kv_dim
            if out_dim <= 0:
                return default
            return float(C_MATMUL * D * out_dim * b * q_len)

        if name in ("QK") and qh > 0 and hd > 0:
            if phase == "prefill":
                pairs = tri(q_len) if causal else q_len * q_len
            else:
                pairs = kv_len
            return float(C_MATMUL * b * qh * hd * pairs)

        if name in ("SOFTMAX") and qh > 0:
            if phase == "prefill":
                elems = tri(q_len) if causal else q_len * q_len
            else:
                elems = kv_len
            return float(b * qh * elems * C_SOFTMAX)

        if name in ("SV") and qh > 0 and hd > 0:
            if phase == "prefill":
                pairs = tri(q_len) if causal else q_len * q_len
            else:
                pairs = kv_len
            return float(C_MATMUL * b * qh * hd * pairs)

        if name in ("O") and D > 0 and o_dim > 0:
            return float(C_MATMUL * o_dim * D * b * q_len)

        if name in ("FFN_W1", "FFN_W3", "FFN_UP", "FFN_GATE") and D > 0 and Hf > 0:
            return float(C_MATMUL * D * Hf * b * q_len)

        if name in ("FFN_W2", "FFN_DOWN") and D > 0 and Hf > 0:
            return float(C_MATMUL * Hf * D * b * q_len)

        if name in ("SWIGLU", "SILU_GLU") and Hf > 0:
            return float(b * q_len * Hf * (C_SILU + 1.0))

        if name in ("GELU",) and Hf > 0:
            return float(b * q_len * Hf * C_GELU)

        if name == "ADD" and D > 0:
            return float(b * q_len * D)

        if name in ("IDENTITY", "RESIDUAL", "DROPOUT") and D > 0:
            return float(b * q_len * D)

        if name in ("KV_READ", "KV_WRITE", "ROPE", "ALIBI"):
            return 0.0

        return default

    def estimate_activation_bytes(self, node, batch: int, seq_len: int, phase: str):
        attrs = getattr(node, "attrs", {}) or {}
        dtype_bytes = int(DTYPE_BYTES.get(self.dtype, 2))

        def to_bytes(elems: float) -> int:
            return int(max(0.0, float(elems))) * dtype_bytes

        b = int(batch or attrs.get("batch", 0) or 1)
        T = int(seq_len or 0)
        if T <= 0:
            return 0, 0

        kv_len = int(attrs.get("kv_len", attrs.get("past_kv_len", T)) or T)
        active_tokens = T if phase == "prefill" else 1
        causal = bool(attrs.get("causal", True))
        
        def tri(n: int) -> int:
            return n * (n + 1) // 2

        D   = int(attrs.get("dim", attrs.get("hidden_size", 0)) or 0)
        Hf  = int(attrs.get("ffn_dim", attrs.get("mlp_dim", 0)) or 0)
        hd  = int(attrs.get("head_dim", 0) or 0)
        qh  = int(attrs.get("q_heads", attrs.get("n_heads", 0)) or 0)
        kvh = int(attrs.get("n_kv_heads", attrs.get("kv_heads", qh)) or 0)
        q_dim   = int(attrs.get("q_dim", qh * hd) or 0)
        kv_dim  = int(attrs.get("kv_dim", kvh * hd) or 0)
        o_dim   = int(attrs.get("o_dim", qh * hd) or 0)

        name = (getattr(node, "name", attrs.get("op", "")) or "").upper()

        if phase == "prefill":
            attn_pairs = tri(T) if causal else T * T
        else:
            attn_pairs = kv_len

        if name in ("LN") and D > 0:
            elems = b * active_tokens * D
            return to_bytes(elems), to_bytes(elems)

        if name == "Q" and D > 0:
            out_dim = q_dim if q_dim > 0 else D
            return to_bytes(b * active_tokens * D), to_bytes(b * active_tokens * out_dim)

        if name in ("K", "V") and D > 0:
            out_dim = kv_dim if kv_dim > 0 else D
            write_tokens = active_tokens
            return to_bytes(b * active_tokens * D), to_bytes(b * write_tokens * out_dim)

        if name in ("O") and D > 0:
            inp_dim = o_dim if o_dim > 0 else D
            return to_bytes(b * active_tokens * inp_dim), to_bytes(b * active_tokens * D)

        if name in ("FFN_W1", "FFN_W3") and D > 0 and Hf > 0:
            return to_bytes(b * active_tokens * D), to_bytes(b * active_tokens * Hf)

        if name in ("FFN_W2",) and D > 0 and Hf > 0:
            return to_bytes(b * active_tokens * Hf), to_bytes(b * active_tokens * D)

        if name in ("SWIGLU", "SILU_GLU") and Hf > 0:
            return to_bytes(b * active_tokens * (2 * Hf)), to_bytes(b * active_tokens * Hf)

        if name in ("GELU", "RELU"):
            width = Hf if Hf > 0 else D
            return to_bytes(b * active_tokens * width), to_bytes(b * active_tokens * width)

        if name == "ADD" and D > 0:
            read_elems = b * active_tokens * D * 2
            write_elems = b * active_tokens * D
            return to_bytes(read_elems), to_bytes(write_elems)

        if name in ("IDENTITY",):
            elems = b * active_tokens * D
            return to_bytes(elems), to_bytes(elems)

        if name in ("QK") and qh > 0 and hd > 0:
            q_read = b * active_tokens * q_dim
            k_read = b * (T if phase == "prefill" else kv_len) * kv_dim
            write_elems = b * qh * attn_pairs
            return to_bytes(q_read + k_read), to_bytes(write_elems)

        if name in ("SOFTMAX", "ATTN_SOFTMAX") and qh > 0:
            elems = b * qh * attn_pairs
            return to_bytes(elems), to_bytes(elems)

        if name in ("SV") and qh > 0 and hd > 0:
            attn_read = b * qh * attn_pairs
            v_read    = b * (T if phase == "prefill" else kv_len) * kv_dim
            out_elems = b * qh * active_tokens * hd
            return to_bytes(attn_read + v_read), to_bytes(out_elems)

        if name in ("KV_READ", "KV_WRITE"):
            read = 2 * batch * kvh * hd * kv_len
            write = 2 * batch * kvh * hd * active_tokens
            return read, write

        if D > 0:
            elems = b * active_tokens * D
            return to_bytes(elems), to_bytes(elems)

        return 0, 0

    # --------------------------
    # Node device cost
    # --------------------------
    def node_device_cost(
        self,
        node: TaskNode,
        dev: DeviceSpec,
        label: PlanLabel,
        batch: int,
        seq_len: int,
        phase: str
    ) -> float:
        # NPU
        if dev.type == "npu":
            flops = self.estimate_flops(node, batch, seq_len, phase)
            rd, wr = self.estimate_activation_bytes(node, batch, seq_len, phase)
            return max(self.flop_time(flops, dev), self.mem_time(rd + wr, dev))

        # PIM - use aim simulator
        if dev.type == "pim":
            if not self.pim_config_path or not self.ramulator_config_path:
                print(f"[PIM] Warning: PIM configs not set, returning 0 for {node.name}")
                return 0.0
            
            attrs = getattr(node, "attrs", {}) or {}
            dim = int(attrs.get("dim", 0) or 0)
            n_heads = int(attrs.get("n_heads", attrs.get("q_heads", attrs.get("n_head", 0))) or 0)
            n_kv_heads = int(attrs.get("n_kv_heads", attrs.get("kv_heads", n_heads)) or n_heads)
            ffn_dim = int(attrs.get("ffn_dim", 0) or 0)
            ffn_dim_mul = float(attrs.get("ffn_dim_mul", 4.0))
   
            if ffn_dim == 0 and dim > 0:
                ffn_dim = int(ffn_dim_mul * dim)
            
            compute_time = 0.0

            keys = self._resolve_pim_key(node)
            op_key = keys[0] if keys else None

            if node.name.upper() in ("KV_READ", "KV_WRITE"):
                compute_time = 0.0
            elif op_key and dim > 0 and n_heads > 0:
                try:
                    # aim simulator 
                    compute_time = _get_pim_latency_via_trace(
                        op=op_key,
                        pim_config=self.pim_config_path,
                        ramulator_config=self.ramulator_config_path,
                        dim=dim,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        ffn_dim=ffn_dim,
                        seqlen=seq_len if seq_len > 0 else None,
                        use_cache=self.pim_cache_enabled
                    )
                except Exception as e:
                    print(f"[PIM] ERROR: Failed to compute latency for {node.name}: {e}")
                    # 不使用 fallback，直接返回 0 或抛出异常
                    raise RuntimeError(f"PIM latency computation failed for {node.name}: {e}")
            else:
                print(f"[PIM] Warning: Insufficient parameters for {node.name} "
                      f"(op={op_key}, dim={dim}, heads={n_heads})")
            
            # 考虑 KV cache 在 PIM 中的情况
            kv_in_pim = getattr(label, "kv_in_pim", False)
            if kv_in_pim:
                mem_time = 0.0
            else:
                rd, wr = self.estimate_activation_bytes(node, batch, seq_len, phase)
                mem_time = self.mem_time((rd + wr), dev)
            
            return compute_time + mem_time
        
        return 0.0