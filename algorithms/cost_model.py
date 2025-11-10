from __future__ import annotations
from config import attach_local_debug_filter
import json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from functools import lru_cache
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
from config import HOST_NAME, DEVICE_PREFERRED_FORMAT, FORMAT_SIZE_MULTIPLIER, FORMAT_CONV_BW_GBs, PIM_FREQ_GHZ, GB_FREQ_GHZ
import logging
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger)
DTYPE_BYTES: Dict[str, int] = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1, 'fp8': 1}

def _ensure_cent_on_path(start: Optional[Path]=None) -> Tuple[Path, Path]:
    here = (start or Path(__file__)).resolve()
    for p in [here.parent] + list(here.parents):
        cand = p / 'submodules' / 'CENT' / 'cent_simulation'
        if cand.exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return (cand, p)
    raise RuntimeError(f"Cannot find 'submodules/CENT/cent_simulation' above {here}")

def _initialize_cent_module():
    try:
        cent_path, _ = _ensure_cent_on_path()
        from Llama import TransformerBlockLlama as TransformerBlock
        return TransformerBlock
    except Exception:
        try:
            from TransformerBlock import TransformerBlock
            return TransformerBlock
        except ImportError:
            raise RuntimeError('Cannot import TransformerBlock from CENT')
_TransformerBlock = None

def _get_transformer_block():
    global _TransformerBlock
    if _TransformerBlock is None:
        _TransformerBlock = _initialize_cent_module()
    return _TransformerBlock

def _load_memory_config(path: Path) -> Dict[str, any]:
    cfg = json.loads(path.read_text(encoding='utf-8'))
    std: Dict[str, any] = {}
    alias = {'DRAM_column': ['dram_column', 'DRAMCol', 'dramCol', 'dram_col'], 'DRAM_row': ['dram_row', 'DRAMRow', 'dramRow'], 'burst_length': ['burst', 'burstLength', 'BL'], 'num_banks': ['banks', 'numBanks'], 'num_channels': ['channels', 'numChannels'], 'threads': ['thread', 'nThreads'], 'reuse_size': ['reuseSize', 'reuse', 'RS'], 'channels_per_block': ['channelsPerBlock', 'cpb'], 'max_seq_len': ['maxSeqLen', 'max_seq_length']}
    for k, v in cfg.items():
        matched = False
        for stdk, alist in alias.items():
            if k in alist or k == stdk:
                std[stdk] = v
                matched = True
                break
        if not matched:
            std[k] = v
    std.setdefault('DRAM_column', 256)
    std.setdefault('DRAM_row', 64)
    std.setdefault('burst_length', 16)
    std.setdefault('num_banks', 8)
    std.setdefault('num_channels', 4)
    std.setdefault('threads', 1)
    std.setdefault('reuse_size', 32)
    std.setdefault('channels_per_block', None)
    std.setdefault('max_seq_len', 4096)
    return std

def _make_tb_args_from_pim(cfg: Dict[str, any], trace_file: str):
    from types import SimpleNamespace
    cpb = cfg['channels_per_block']
    if cpb is None:
        cpb = cfg['num_channels']
    return SimpleNamespace(DRAM_column=int(cfg['DRAM_column']), DRAM_row=int(cfg['DRAM_row']), burst_length=int(cfg['burst_length']), num_banks=int(cfg['num_banks']), num_channels=int(cfg['num_channels']), threads=int(cfg['threads']), reuse_size=int(cfg['reuse_size']), channels_per_block=int(cpb), max_seq_len=int(cfg['max_seq_len']), only_trace=True, op_trace=False, trace_file=trace_file, pim_compute=True, model='llama_like', embedding='rope', seqlen=16, model_parallel=False, FC_devices=1, pipeline_parallel=False, inter_device_attention=False, only_FC=False, trace_prepare=False, trace_norm=False, trace_fc_kqvo=False, trace_attention=False, trace_softmax=False, trace_fc_ffn=False, trace_activation=False, GEMV='reuse-GB')

def _calc_channels(block):
    if getattr(block, 'model_parallel', False):
        FC_total_banks = int(block.total_banks) * int(block.FC_devices)
        channels_required = int(block.num_channels)
    else:
        FC_total_banks = int(block.total_banks)
        channels_required = int(block.channels_per_block)
    num_channels = int(block.num_channels)
    channels_required = int(channels_required)
    channel_multi_tb_required = int(num_channels // channels_required * channels_required)
    channel_lst = [channel for channel in range(channel_multi_tb_required)]
    return (channel_lst, FC_total_banks, channels_required)

def _emit_single_op_trace(block, op: str, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlens: Optional[List[int]]):
    channel_lst, FC_total_banks, channels_required = _calc_channels(block)
    head_dim = dim // max(1, n_heads)
    if op in ('q_proj', 'k_proj', 'v_proj', 'wo_proj', 'ffn_up', 'ffn_gate', 'ffn_down'):
        if op == 'q_proj':
            row_tag, V, N = ('wq_row_index', dim, dim)
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_sa_weight')
        elif op == 'k_proj':
            row_tag, V, N = ('wk_row_index', dim, n_kv_heads * head_dim)
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_sa_weight')
        elif op == 'v_proj':
            row_tag, V, N = ('wv_row_index', dim, n_kv_heads * head_dim)
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_sa_weight')
        elif op == 'wo_proj':
            row_tag, V, N = ('wo_row_index', dim, dim)
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_sa_weight')
        elif op == 'ffn_up':
            row_tag, V, N = ('w1_row_index', dim, ffn_dim)
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_ffn_weight')
        elif op == 'ffn_gate':
            row_tag, V, N = ('w3_row_index', dim, ffn_dim)
            block.Vector_Matrix_Mul_weight_af_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_ffn_weight')
        elif op == 'ffn_down':
            row_tag, V, N = ('w2_row_index', ffn_dim, dim)
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_ffn_weight')
    elif op in ('score', 'softmax', 'output'):
        for S in seqlens or [1]:
            if op == 'score':
                block.Vector_Matrix_Mul_score_pim_only_trace(block.cache_k_row_index, S, 'breakdown_sa_score')
            elif op == 'output':
                block.Vector_Matrix_Mul_output_pim_only_trace(block.cache_v_row_index, S, 'breakdown_sa_output')
            elif op == 'softmax':
                rows_per_score = (S - 1) // block.DRAM_column + 1
                for r in range(rows_per_score):
                    op_size = block.DRAM_column // block.burst_length if r < rows_per_score - 1 else (S - block.DRAM_column * r - 1) // block.burst_length + 1
                    block.EWMUL_only_trace(channel_lst, block.scores_row_index + r, op_size)
                block.time['RD_SBK'] += block.timing_constant['RD_SBK'] + S * block.n_heads // block.burst_length
                block.load_from_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 2, S)
                block.time['WR_SBK'] += block.timing_constant['WR_SBK'] + S * block.n_heads // block.burst_length
                block.store_for_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 0, S)
                rows_per_score = (S - 1) // block.DRAM_column + 1
                for r in range(rows_per_score):
                    op_size = block.DRAM_column // block.burst_length if r < rows_per_score - 1 else (S - block.DRAM_column * r - 1) // block.burst_length + 1
                    block.EWMUL_only_trace(channel_lst, block.scores_row_index + r, op_size)
                block.load_from_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 2, S)
    elif op == 'rmsnorm':
        input_len = (dim - 1) // (block.total_banks // 2) + 1
        block.WR_BIAS_only_trace(channel_lst)
        block.MAC_ABK_only_trace(channel_lst, block.x_row_index, (input_len - 1) // block.burst_length + 1, 'breakdown_sa_pow')
        block.RD_MAC_only_trace(channel_lst)
        ew_len = (dim - 1) // (block.total_banks // 4) + 1
        ew_banks = (dim - 1) // ew_len + 1
        block.time['WR_SBK'] += block.timing_constant['WR_SBK'] + dim // block.burst_length
        block.store_for_EWMUL_input_only_trace(channels_required, ew_banks, 1, block.x_copy_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.x_copy_row_index, (ew_len - 1) // block.burst_length + 1)
        for bank in range(block.num_banks):
            if bank % 4 == 2:
                block.COPY_BK_GB_only_trace(channel_lst, bank, block.x_copy_row_index, (ew_len - 1) // block.burst_length + 1)
                block.COPY_GB_BK_only_trace(channel_lst, bank - 1, block.SANorm_row_index, (ew_len - 1) // block.burst_length + 1)
        block.EWMUL_only_trace(channel_lst, block.SANorm_row_index, (ew_len - 1) // block.burst_length + 1)
        block.time['RD_SBK'] += block.timing_constant['RD_SBK'] + block.dim // block.burst_length
        block.load_from_EWMUL_input_only_trace(channels_required, ew_banks, 2, block.SANorm_row_index, ew_len)
        block.SYNC_only_trace()
    elif op == 'rope':
        ew_len = (head_dim - 1) // (block.total_banks // 4) + 1
        ew_size = (ew_len - 1) // block.burst_length + 1
        block.store_for_EWMUL_input_only_trace(block.channels_per_block, (dim - 1) // ew_len + 1, 1, block.xq_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.xq_row_index, ew_size)
        block.store_for_EWMUL_input_only_trace(block.channels_per_block, (dim - 1) // ew_len + 1, 1, block.xk_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.xk_row_index, ew_size)
    elif op in ('silu', 'gelu'):
        ew_len = (ffn_dim - 1) // (block.total_banks // 4) + 1
        ew_banks = (ffn_dim - 1) // ew_len + 1
        block.time['WR_SBK'] += block.timing_constant['WR_SBK'] + ffn_dim // block.burst_length
        block.store_for_EWMUL_input_only_trace(block.channels_per_block, ew_banks, 1, block.ffn_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.ffn_row_index, (ew_len - 1) // block.burst_length + 1)
        for bank in range(block.num_banks):
            if bank % 4 == 2:
                block.COPY_BK_GB_only_trace(channel_lst, bank, block.ffn_row_index, (ew_len - 1) // block.burst_length + 1)
                block.COPY_GB_BK_only_trace(channel_lst, bank - 1, block.ffn_row_index, (ew_len - 1) // block.burst_length + 1)
        block.EWMUL_only_trace(channel_lst, block.ffn_row_index, (ew_len - 1) // block.burst_length + 1)
        block.time['RD_SBK'] += block.timing_constant['RD_SBK'] + ffn_dim // block.burst_length
        block.SYNC_only_trace()
    elif op == 'residual':
        op_size = block.dim // block.burst_length
        block.EWADD_only_trace(op_size)
    else:
        raise ValueError(f'Unsupported op: {op}')
    if hasattr(block, 'file') and block.file:
        block.file.write('AiM EOC\n')
        block.file.flush()

def _generate_pim_trace(op: str, pim_config: Path, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: Optional[int], trace_file: Path, model_dict: Optional[Dict]=None) -> None:
    if model_dict is None:
        raise ValueError('Model dictionary must be provided for PIM trace generation')
    TransformerBlock = _get_transformer_block()
    pim_cfg = _load_memory_config(pim_config)
    args = _make_tb_args_from_pim(pim_cfg, str(trace_file))
    args.op_trace = True
    args.seqlen = int(seqlen or args.seqlen or 16)
    block = TransformerBlock(model_dict, args)
    if hasattr(block, 'memory_mapping'):
        block.memory_mapping()
    seqlens_list = [seqlen] if seqlen else None
    _emit_single_op_trace(block, op, dim, n_heads, n_kv_heads, ffn_dim, seqlens_list)
    if hasattr(block, 'file') and block.file:
        block.file.flush()
        block.file.close()
    if not trace_file.exists():
        raise RuntimeError(f'Trace file not generated: {trace_file}')
    if trace_file.stat().st_size == 0:
        raise RuntimeError(f'Trace file is empty: {trace_file}')

def _make_shared_model_dict(dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: int) -> Dict:
    import torch
    head_dim = dim // max(1, n_heads)
    TP_param = 1
    return {'TP_param': torch.tensor(TP_param), 'dim': torch.tensor(dim), 'n_heads': torch.tensor(n_heads), 'n_kv_heads': torch.tensor(n_kv_heads), 'x': torch.zeros((1, 1, dim)), 'SANorm': torch.zeros(dim), 'FFNNorm': torch.zeros(dim), 'sa': torch.zeros((1, 1, dim)), 'h': torch.zeros((1, 1, dim)), 'out': torch.zeros((1, 1, dim)), 'wq': torch.zeros((dim // TP_param, dim)), 'wk': torch.zeros(head_dim * n_kv_heads, dim), 'wv': torch.zeros(head_dim * n_kv_heads, dim), 'xq': torch.zeros((1, 1, dim)), 'xk': torch.zeros((1, 1, head_dim * n_heads)), 'xv': torch.zeros((1, 1, head_dim * n_heads)), 'start_pos': torch.tensor(max(1, seqlen) - 1), 'cache_k': torch.zeros((1, seqlen, n_kv_heads, head_dim)), 'cache_v': torch.zeros((1, seqlen, n_kv_heads, head_dim)), 'scores': torch.zeros((1, n_heads, 1, seqlen)), 'output': torch.zeros((1, 1, dim)), 'wo': torch.zeros((dim // TP_param, dim)), 'w1': torch.zeros((ffn_dim // TP_param, dim)), 'w3': torch.zeros((ffn_dim // TP_param, dim)), 'w2': torch.zeros((dim // TP_param, ffn_dim)), 'ffn': torch.zeros((1, 1, dim))}

def _generate_weight_read_trace(trace_path: Path, weight_bytes: int, dtype_bytes: int=2, gb_config: Optional[Dict[str, any]]=None, model_dict: Optional[Dict]=None) -> None:
    """
    生成从 Global Buffer 读取权重的 trace
    使用独立的 Global Buffer DRAM 配置（不是 PIM 配置）
    """
    if gb_config is None:
        raise ValueError('Global Buffer config (gb_config) must be provided for weight read trace generation')
    if model_dict is None:
        raise ValueError('Model dictionary (model_dict) must be provided for weight read trace generation')
    num_elements = max(0, weight_bytes // max(1, dtype_bytes))
    if num_elements == 0:
        with trace_path.open('w', encoding='utf-8') as f:
            f.write('AiM EOC\n')
        return
    TransformerBlock = _get_transformer_block()
    cfg: Dict[str, any] = dict(gb_config)
    temp_trace = trace_path.parent / f'temp_{trace_path.name}'
    args = _make_tb_args_from_pim(cfg, str(temp_trace))
    args.only_trace = True
    block = TransformerBlock(model_dict, args)
    if hasattr(block, 'memory_mapping'):
        block.memory_mapping()
    block.file = trace_path.open('w', encoding='utf-8')
    block.trace_file = str(trace_path)
    DRAM_column = int(getattr(block, 'DRAM_column', cfg['DRAM_column']))
    burst_length = int(getattr(block, 'burst_length', cfg['burst_length']))
    num_banks = int(getattr(block, 'num_banks', cfg['num_banks']))
    num_channels = int(getattr(block, 'num_channels', cfg['num_channels']))
    total_banks = num_channels * num_banks
    elements_per_bank = (num_elements + total_banks - 1) // total_banks
    rows_per_bank = (elements_per_bank + DRAM_column - 1) // DRAM_column
    bursts_per_full_row = max(1, DRAM_column // max(1, burst_length))
    for channel in range(num_channels):
        for bank in range(num_banks):
            for row in range(rows_per_bank):
                if row < rows_per_bank - 1:
                    bursts_in_row = bursts_per_full_row
                else:
                    remaining_elems = elements_per_bank - (rows_per_bank - 1) * DRAM_column
                    bursts_in_row = (remaining_elems + burst_length - 1) // burst_length
                    if bursts_in_row <= 0:
                        continue
                size_elems = int(bursts_in_row * burst_length)
                if hasattr(block, 'R_MEM_only_trace'):
                    block.R_MEM_only_trace(channel, bank, row, size_elems)
                else:
                    raise RuntimeError('R_MEM_only_trace not available on TransformerBlock/PIM')
    if hasattr(block, 'file') and block.file:
        block.file.write('AiM EOC\n')
        block.file.flush()
        block.file.close()

def _generate_weight_write_trace_to_pim(trace_path: Path, weight_bytes: int, pim_config: Dict[str, any], dtype_bytes: int=2, model_dict: Optional[Dict]=None) -> None:
    if pim_config is None:
        raise ValueError('PIM config must be provided for weight write trace generation')
    if model_dict is None:
        raise ValueError('Model dictionary (model_dict) must be provided for weight write trace generation')
    TransformerBlock = _get_transformer_block()
    temp_trace = trace_path.parent / f'temp_{trace_path.name}'
    args = _make_tb_args_from_pim(pim_config, str(temp_trace))
    args.only_trace = True
    block = TransformerBlock(model_dict, args)
    if hasattr(block, 'memory_mapping'):
        block.memory_mapping()
    num_elements = weight_bytes // dtype_bytes
    DRAM_column = int(pim_config.get('DRAM_column', 256))
    burst_length = int(pim_config.get('burst_length', 16))
    num_banks = int(pim_config.get('num_banks', 8))
    num_channels = int(pim_config.get('num_channels', 4))
    total_banks = num_banks * num_channels
    elements_per_bank = (num_elements + total_banks - 1) // total_banks
    rows_per_bank = (elements_per_bank + DRAM_column - 1) // DRAM_column
    block.file = trace_path.open('w', encoding='utf-8')
    block.trace_file = str(trace_path)
    channel_lst = list(range(num_channels))
    for channel in channel_lst:
        for bank in range(num_banks):
            row_start = 0
            for row in range(rows_per_bank):
                if row < rows_per_bank - 1:
                    bursts_in_row = DRAM_column // burst_length
                else:
                    remaining_elems = elements_per_bank - (rows_per_bank - 1) * DRAM_column
                    bursts_in_row = (remaining_elems + burst_length - 1) // burst_length
                    if bursts_in_row <= 0:
                        continue
                size_elems = int(bursts_in_row * burst_length)
                if hasattr(block, 'W_MEM_only_trace'):
                    block.W_MEM_only_trace(channel, bank, row, size_elems)
                else:
                    raise RuntimeError('W_MEM_only_trace not available on TransformerBlock/PIM')
    if hasattr(block, 'file') and block.file:
        block.file.write('AiM EOC\n')
        block.file.flush()
        block.file.close()
    if temp_trace.exists():
        temp_trace.unlink()

def _simulate_weight_loading_latency(weight_bytes: int, pim_config_path: Path, gb_config_path: Path, ramulator_config_path: Path, dtype_bytes: int=2, use_cache: bool=True, keep_traces: bool=False, model_dict: Optional[Dict]=None) -> Tuple[float, float]:
    """
    仿真 weight 加载延迟：
    1. 从 Global Buffer 读取 (READ trace) - 使用 GB 配置
    2. 写入 PIM banks (WRITE trace) - 使用 PIM 配置
    
    返回: (read_latency, write_latency) in seconds
    """
    if model_dict is None:
        raise ValueError('Model dictionary must be provided for weight loading simulation')
    cache_key = f'weight_load_{weight_bytes}_{pim_config_path.name}_{gb_config_path.name}_{ramulator_config_path.name}'
    if use_cache:
        cached = _pim_cache.cache.get(hashlib.md5(cache_key.encode()).hexdigest())
        if cached is not None:
            logger.debug(str(f'[Weight Load Cache] Hit for {weight_bytes} bytes'))
            return cached
    logger.debug(str(f'[Weight Load] Simulating loading {weight_bytes} bytes to PIM'))
    if keep_traces:
        temp_dir = Path('./debug_weight_traces')
        temp_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trace_dir = temp_dir / f'weight_{weight_bytes}_{timestamp}'
        trace_dir.mkdir(exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix='weight_trace_'))
        trace_dir = temp_dir
    try:
        pim_cfg = _load_memory_config(pim_config_path)
        gb_cfg = _load_memory_config(gb_config_path)
        read_trace = trace_dir / 'weight_read.trace'
        logger.debug(str(f'[Weight Load] Generating READ trace from Global Buffer: {read_trace}'))
        _generate_weight_read_trace(read_trace, weight_bytes, dtype_bytes, gb_cfg, model_dict)
        if not read_trace.exists():
            raise RuntimeError(f'Failed to generate READ trace: {read_trace}')
        trace_size = read_trace.stat().st_size
        logger.debug(str(f'[Weight Load] READ trace size: {trace_size} bytes'))
        if trace_size < 1000:
            with read_trace.open('r') as f:
                lines = f.readlines()[:10]
                logger.debug(str(f"[Weight Load] First lines of READ trace:\n{''.join(lines)}"))
        logger.debug(str(f'[Weight Load] Running READ simulation...'))
        read_cycles = _run_ramulator(read_trace, ramulator_config_path)
        f_gb = GB_FREQ_GHZ if GB_FREQ_GHZ and GB_FREQ_GHZ > 0 else PIM_FREQ_GHZ
        read_latency = float(read_cycles) / (f_gb * 1000000000.0) if f_gb > 0 else 0.0
        write_trace = trace_dir / 'weight_write.trace'
        logger.debug(str(f'[Weight Load] Generating WRITE trace to PIM: {write_trace}'))
        _generate_weight_write_trace_to_pim(write_trace, weight_bytes, pim_cfg, dtype_bytes, model_dict)
        if not write_trace.exists():
            raise RuntimeError(f'Failed to generate WRITE trace: {write_trace}')
        trace_size = write_trace.stat().st_size
        logger.debug(str(f'[Weight Load] WRITE trace size: {trace_size} bytes'))
        if trace_size < 1000:
            with write_trace.open('r') as f:
                lines = f.readlines()[:10]
                logger.debug(str(f"[Weight Load] First lines of WRITE trace:\n{''.join(lines)}"))
        logger.debug(str(f'[Weight Load] Running WRITE simulation...'))
        write_cycles = _run_ramulator(write_trace, ramulator_config_path)
        f_pim = PIM_FREQ_GHZ
        write_latency = float(write_cycles) / (f_pim * 1000000000.0) if f_pim > 0 else 0.0
        total_latency = (read_latency, write_latency)
        logger.debug(str(f'[Weight Load] Read: {read_latency:.6e}s ({read_cycles} cycles)'))
        logger.debug(str(f'[Weight Load] Write: {write_latency:.6e}s ({write_cycles} cycles)'))
        if keep_traces:
            logger.debug(str(f'[Weight Load] Traces saved to: {trace_dir}'))
        if use_cache:
            key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            _pim_cache.cache[key_hash] = total_latency
            _pim_cache._save_cache()
        return total_latency
    except Exception as e:
        logger.debug(str(f'[Weight Load] Error: {e}'))
        if keep_traces:
            logger.debug(str(f'[Weight Load] Debug traces preserved at: {trace_dir}'))
        raise
    finally:
        if not keep_traces:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.debug(str(f'[Weight Load] Warning: Failed to cleanup {temp_dir}: {e}'))

def _run_ramulator(trace_path: Path, ramulator_config: Path, timeout: int=300) -> int:
    """Run Ramulator2 on a trace file and return cycle count."""
    if not trace_path.exists():
        raise FileNotFoundError(f'Trace file not found: {trace_path}')
    if not ramulator_config.exists():
        raise FileNotFoundError(f'Ramulator config not found: {ramulator_config}')
    ramulator_exe = Path.cwd() / 'ramulator2'
    cmd = [str(ramulator_exe), '-f', str(ramulator_config), '-t', str(trace_path)]
    logger.debug(str(f"[PIM] Running ramulator: {' '.join(cmd)}"))
    logger.debug(str(f'[PIM] Working directory: {trace_path.parent}'))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            error_msg = f'Ramulator failed with return code {result.returncode}:\n'
            error_msg += f"Command: {' '.join(cmd)}\n"
            error_msg += f'STDOUT:\n{result.stdout}\n'
            error_msg += f'STDERR:\n{result.stderr}\n'
            error_msg += f'Trace file: {trace_path}\n'
            error_msg += f'Config file: {ramulator_config}\n'
            raise RuntimeError(error_msg)
        cycles = 0
        for line in result.stdout.splitlines():
            if 'cpu.total_cycles' in line.lower() or 'total cycles' in line.lower():
                match = re.search('(\\d+)', line)
                if match:
                    cycles = int(match.group(1))
                    break
        if cycles == 0:
            for line in result.stdout.splitlines():
                if re.search('\\d+', line):
                    match = re.search('(\\d+)', line)
                    if match and int(match.group(1)) > 0:
                        cycles = int(match.group(1))
                        break
        return cycles
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'Ramulator timed out after {timeout}s')
    except Exception as e:
        raise RuntimeError(f'Ramulator execution failed: {e}')

class PIMLatencyCache:

    def __init__(self, cache_file: Optional[Path]=None):
        self.cache_file = cache_file or Path('./output/pim_latency_cache.pkl')
        self.cache: Dict[str, float] = {}
        self.lock = Lock()
        self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.debug(str(f'[PIM Cache] Loaded {len(self.cache)} entries from {self.cache_file}'))
            except Exception as e:
                logger.debug(str(f'[PIM Cache] Failed to load cache: {e}'))
                self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.debug(str(f'[PIM Cache] Failed to save cache: {e}'))

    def _make_key(self, op: str, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: Optional[int], pim_config: Path, ramulator_config: Path) -> str:
        params = f'{op}_{dim}_{n_heads}_{n_kv_heads}_{ffn_dim}_{seqlen}'
        configs = f'{pim_config.name}_{ramulator_config.name}'
        key = f'{params}_{configs}'
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, op: str, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: Optional[int], pim_config: Path, ramulator_config: Path) -> Optional[float]:
        key = self._make_key(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen, pim_config, ramulator_config)
        with self.lock:
            return self.cache.get(key)

    def set(self, op: str, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: Optional[int], pim_config: Path, ramulator_config: Path, latency: float):
        key = self._make_key(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen, pim_config, ramulator_config)
        with self.lock:
            self.cache[key] = latency
            self._save_cache()
_pim_cache = PIMLatencyCache()
_DEF_RUNTIME_REL = os.path.join(os.path.dirname(__file__), 'runtime_models', 'mmad_latency_model.json')
_DEF_SOFTMAX_MODEL_REL = os.path.join(os.path.dirname(__file__), 'runtime_models', 'softmax_latency_model.json')
_DEF_GELU_MODEL_REL    = os.path.join(os.path.dirname(__file__), 'runtime_models', 'gelu_latency_linear_model.json')
_DEF_LAYERNORM_MODEL_REL  = os.path.join(os.path.dirname(__file__), 'runtime_models', 'layernorm_latency_model.json')

def _candidate_model_paths() -> list:
    env_path = os.environ.get('TRIFORM_MMAD_MODEL')
    cand = []
    if env_path:
        cand.append(env_path)
    cand.extend([_DEF_RUNTIME_REL])
    return cand
def _candidate_softmax_model_paths() -> list:
    cand = []
    env_path = os.environ.get('TRIFORM_SOFTMAX_MODEL')
    if env_path: cand.append(env_path)
    cand.append(_DEF_SOFTMAX_MODEL_REL)
    return cand

def _candidate_gelu_model_paths() -> list:
    cand = []
    env_path = os.environ.get('TRIFORM_GELU_MODEL')
    if env_path: cand.append(env_path)
    cand.append(_DEF_GELU_MODEL_REL)
    return cand

def _candidate_layernorm_model_paths() -> list:
    cand = []
    env_path = os.environ.get('TRIFORM_LAYERNORM_MODEL')
    if env_path: cand.append(env_path)
    cand.append(_DEF_LAYERNORM_MODEL_REL)
    return cand

@lru_cache(maxsize=1)
def _load_mmad_model_json() -> Optional[Dict]:
    for p in _candidate_model_paths():
        try:
            if p and os.path.isfile(p):
                with open(p, 'r') as f:
                    obj = json.load(f)
                assert 'block_size' in obj and 'coefficients' in obj
                return obj
        except Exception as e:
            logger.debug(str(f"[MMAD-Model] Failed to load '{p}': {e}"))
    return None


@lru_cache(maxsize=1)
def _load_softmax_model_json() -> Optional[Dict]:
    for p in _candidate_softmax_model_paths():
        try:
            if p and os.path.isfile(p):
                with open(p, 'r') as f:
                    obj = json.load(f)
                if 'coefficients' in obj:
                    return obj
        except Exception as e:
            logger.debug(str(f"[SOFTMAX-Model] Failed to load '{p}': {e}"))
    return None

@lru_cache(maxsize=1)
def _load_gelu_model_json() -> Optional[Dict]:
    for p in _candidate_gelu_model_paths():
        try:
            if p and os.path.isfile(p):
                with open(p, 'r') as f:
                    obj = json.load(f)
                if ('alpha' in obj) and ('beta' in obj):
                    return obj
        except Exception as e:
            logger.debug(str(f"[GELU-Model] Failed to load '{p}': {e}"))
    return None

@lru_cache(maxsize=1)
def _load_layernorm_model_json() -> Optional[Dict]:
    for p in _candidate_layernorm_model_paths():
        try:
            if p and os.path.isfile(p):
                with open(p, 'r') as f:
                    obj = json.load(f)
                # 支持 hinge（首选）或线性回退
                if (obj.get('family') == 'hinge') or (('alpha' in obj) and ('beta' in obj)):
                    return obj
        except Exception as e:
            logger.debug(str(f"[LN-Model] Failed to load '{p}': {e}"))
    return None
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def _compute_feature_vector(M: int, N: int, K: int, block_size: int, feature_names: list) -> Tuple[list, list]:
    """
    根据 JSON 中的 feature 名称（如 ["tiles","mn","sum_b"]）计算对应特征向量。
    """
    MB = _ceil_div(M, block_size)
    NB = _ceil_div(N, block_size)
    KB = _ceil_div(K, block_size)
    base = {'MB': MB, 'NB': NB, 'KB': KB, 'tiles': MB * NB * KB, 'mn': MB * NB, 'sum_b': MB + NB + KB, 'M': M, 'N': N, 'K': K}
    feats = [float(base[name]) for name in feature_names]
    return (feats, [MB, NB, KB])

def _predict_mmad_latency_us_from_json(M: int, N: int, K: int) -> Optional[float]:
    """
    用运行时 JSON 模型预测单次 MMAD 的总延迟（μs），包含 SplitA/SplitB/SplitBias/Compute 四阶段。
    """
    logger.debug(str(f'\n[MMAD-MODEL] Called with M={M}, N={N}, K={K}'))
    model = _load_mmad_model_json()
    if model is None:
        logger.debug(str('[MMAD-MODEL] ✗ Model loading failed, returning None'))
        return None
    logger.debug(str(f'[MMAD-MODEL] ✓ Model loaded successfully'))
    block_size = int(model.get('block_size', 16))
    feature_names = model.get('features', ['tiles', 'mn', 'sum_b'])
    coefs = model['coefficients']
    logger.debug(str(f'[MMAD-MODEL] block_size={block_size}'))
    logger.debug(str(f'[MMAD-MODEL] features={feature_names}'))
    feats, blocks = _compute_feature_vector(M, N, K, block_size, feature_names)
    logger.debug(str(f'[MMAD-MODEL] Block counts: MB={blocks[0]}, NB={blocks[1]}, KB={blocks[2]}'))
    logger.debug(str(f'[MMAD-MODEL] Feature vector: {dict(zip(feature_names, feats))}'))
    y = float(coefs.get('b0', 0.0))
    logger.debug(str(f'[MMAD-MODEL] b0 = {y}'))
    for name, val in zip(feature_names, feats):
        coef = float(coefs.get(f'b_{name}', 0.0))
        contribution = coef * val
        y += contribution
        logger.debug(str(f'[MMAD-MODEL] b_{name} = {coef:.6f}, feat = {val:.2f}, contribution = {contribution:.6f}'))
    result = max(0.0, y)
    logger.debug(str(f'[MMAD-MODEL] ✓ Final prediction: {result:.4f} μs'))
    return result

def _map_op_to_mmad_dims(op: str, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: int) -> Optional[Tuple[int, int, int, int]]:
    """
    线性/FFN/注意力算子 → MMAD (M,N,K,reps)
    """
    logger.debug(str(f"\n[MAP-MMAD] Mapping op='{op}' to MMAD dims"))
    logger.debug(str(f'[MAP-MMAD] Input: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, ffn_dim={ffn_dim}, seqlen={seqlen}'))
    if not op:
        logger.debug(str(f'[MAP-MMAD] ✗ op is None/empty, returning None'))
        return None
    op = op.lower()
    head_dim = dim // max(1, n_heads)
    logger.debug(str(f'[MAP-MMAD] Computed head_dim={head_dim}'))
    result = None
    if op in ('q_proj', 'k_proj', 'v_proj', 'wo_proj'):
        result = (1, dim, dim, max(1, seqlen))
        logger.debug(str(f'[MAP-MMAD] ✓ Matched linear projection: {result}'))
    elif op in ('ffn_up', 'ffn_gate'):
        result = (1, ffn_dim if ffn_dim > 0 else 4 * dim, dim, max(1, seqlen))
        logger.debug(str(f'[MAP-MMAD] ✓ Matched FFN up/gate: {result}'))
    elif op == 'ffn_down':
        result = (1, dim, ffn_dim if ffn_dim > 0 else 4 * dim, max(1, seqlen))
        logger.debug(str(f'[MAP-MMAD] ✓ Matched FFN down: {result}'))
    elif op == 'score' and seqlen and head_dim:
        result = (1, seqlen, head_dim, max(1, n_heads * seqlen))
        logger.debug(str(f'[MAP-MMAD] ✓ Matched attention score: {result}'))
    elif op == 'output' and seqlen and head_dim:
        result = (1, head_dim, seqlen, max(1, n_heads * seqlen))
        logger.debug(str(f'[MAP-MMAD] ✓ Matched attention output: {result}'))
    else:
        logger.debug(str(f"[MAP-MMAD] ✗ No match for op='{op}'"))
    return result

def _predict_softmax_latency_us_from_json(M: int, K: int, *, phase: str='decode', causal: bool=True) -> Optional[float]:
    """
    JSON: T_us = a*MK + b*M + d*blocks + e*k_tail + c
    blocks = M * ceil(K_eff / K_ALIGN), k_tail = K_eff % K_ALIGN
    prefill+causal 使用三角平均：MK *= 0.5 且 K_eff = (K+1)//2
    """
    model = _load_softmax_model_json()
    if model is None:
        return None
    coefs = model.get('coefficients', {})
    a = float(coefs.get('a_MK', 0.0))
    b = float(coefs.get('b_M', 0.0))
    d = float(coefs.get('d_blk', 0.0))
    e = float(coefs.get('e_ktl', 0.0))
    c = float(coefs.get('c_bias', 0.0))
    K_ALIGN = int(model.get('knobs', {}).get('K_ALIGN', 128))

    mk_factor = 0.5 if (phase == 'prefill' and causal) else 1.0
    K_eff = int(K) if mk_factor == 1.0 else int((K + 1) // 2)
    M = int(max(1, M)); K_eff = int(max(1, K_eff))
    blocks_per_row = (K_eff + K_ALIGN - 1) // K_ALIGN
    blocks = M * blocks_per_row
    k_tail = K_eff % K_ALIGN
    MK_eff = float(M) * float(K) * mk_factor
    T_us = a * MK_eff + b * float(M) + d * float(blocks) + e * float(k_tail) + c
    return float(max(0.0, T_us))

def _predict_gelu_latency_us_from_json(length: int) -> Optional[float]:
    """GELU: time_us = alpha * dataLength + beta"""
    model = _load_gelu_model_json()
    if model is None:
        return None
    alpha = float(model.get('alpha', 0.0)); beta = float(model.get('beta', 0.0))
    us = alpha * float(max(0, int(length))) + beta
    return float(max(0.0, us))

def _predict_layernorm_latency_us_from_json(rows: int, width: int) -> Optional[float]:
    """
    LayerNorm（首选 hinge）:
      T_us = a*x + b*B + d*D + t*max(0, x - c) + bias
      x = rows*width, B=rows, D=width
    若非 hinge，线性回退: time_us = alpha*x + beta
    """
    model = _load_layernorm_model_json()
    if model is None:
        return None
    rows = int(max(1, rows)); width = int(max(1, width)); x = float(rows * width)
    if model.get('family') == 'hinge':
        state = model.get('state', {})
        coef = state.get('coef', [])
        a, b, d, t, bias = [float(v) for v in coef] if len(coef) >= 5 else (0.0, 0.0, 0.0, 0.0, 0.0)
        c = float(state.get('c', 0.0))
        B = float(rows); D = float(width)
        T_us = a*x + b*B + d*D + t*max(0.0, x - c) + bias
        return float(max(0.0, T_us))
    # 线性回退
    alpha = float(model.get('alpha', 0.0)); beta = float(model.get('beta', 0.0))
    T_us = alpha * x + beta
    return float(max(0.0, T_us))

class SimulationLogger:

    def __init__(self, log_file: Optional[Path]=None):
        self.log_file = log_file or Path('pim_simulation.log')
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.simulated_ops: Dict[str, set] = defaultdict(set)
        self.lock = Lock()
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(self.log_file, 'w', encoding='utf-8')
        self._log(f"{'=' * 80}")
        self._log(f"PIM Simulation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"{'=' * 80}\n")

    def _log(self, message: str):
        """写入日志文件并打印到控制台"""
        logger.debug(str(message))
        self._log_handle.write(message + '\n')
        self._log_handle.flush()

    def start_simulation(self):
        """开始计时"""
        self.start_time = time.time()
        self._log(f"\n{'=' * 80}")
        self._log(f"Simulation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self._log(f"{'=' * 80}\n")

    def end_simulation(self):
        """结束计时并统计"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time if self.start_time else 0.0
        self._log(f"\n{'=' * 80}")
        self._log(f"Simulation Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self._log(f'Total Simulation Time: {elapsed:.3f} seconds ({elapsed / 60:.2f} minutes)')
        self._log(f"{'=' * 80}\n")
        self._print_statistics()

    def record_simulation(self, op: str, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: Optional[int]):
        """记录一次仿真"""
        with self.lock:
            config = (dim, n_heads, n_kv_heads, ffn_dim, seqlen or 0)
            self.simulated_ops[op].add(config)

    def _print_statistics(self):
        """打印统计信息"""
        self._log(f"\n{'=' * 80}")
        self._log('Simulated Operations Summary')
        self._log(f"{'=' * 80}")
        total_unique = sum((len(configs) for configs in self.simulated_ops.values()))
        self._log(f'\nTotal unique operations simulated: {total_unique}')
        self._log(f'Total operation types: {len(self.simulated_ops)}\n')
        for op in sorted(self.simulated_ops.keys()):
            configs = self.simulated_ops[op]
            self._log(f'\n{op.upper()}:')
            self._log(f'  - Unique configurations: {len(configs)}')
            for config in sorted(configs):
                dim, n_heads, n_kv_heads, ffn_dim, seqlen = config
                self._log(f"    * dim={dim}, heads={n_heads}, kv_heads={n_kv_heads}, ffn_dim={ffn_dim}, seqlen={(seqlen if seqlen > 0 else 'None')}")
        self._log(f"\n{'=' * 80}\n")

    def close(self):
        """关闭日志文件"""
        if self._log_handle and (not self._log_handle.closed):
            self._log_handle.close()
_sim_logger: Optional[SimulationLogger] = None

def get_simulation_logger(log_file: Optional[Path]=None) -> SimulationLogger:
    """获取全局日志实例"""
    global _sim_logger
    if _sim_logger is None:
        _sim_logger = SimulationLogger(log_file)
    return _sim_logger

def _get_pim_latency_via_trace(op: str, pim_config: Path, ramulator_config: Path, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: Optional[int], model_dict: Optional[Dict]=None, use_cache: bool=True) -> float:
    """
    Generate trace and run Ramulator, returning the latency (in seconds).
    This function blocks until Ramulator finishes the simulation and returns the result.
    """
    if model_dict is None:
        raise ValueError('Model dictionary must be provided for PIM latency computation')
    sim_logger = get_simulation_logger()
    sim_logger.record_simulation(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen)
    if use_cache:
        cached = _pim_cache.get(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen, pim_config, ramulator_config)
        if cached is not None:
            msg = f'[PIM Cache] Hit for {op} (dim={dim}, heads={n_heads}, seq={seqlen})'
            sim_logger._log(msg)
            return cached
    msg = f'[PIM] Computing latency for {op} (dim={dim}, heads={n_heads}, seq={seqlen})'
    sim_logger._log(msg)
    temp_dir = Path(tempfile.mkdtemp(prefix='pim_trace_'))
    try:
        trace_path = temp_dir / f'{op}_trace.trace'
        sim_logger._log(f'[PIM] Generating trace: {trace_path}')
        _generate_pim_trace(op=op, pim_config=pim_config, dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, ffn_dim=ffn_dim, seqlen=seqlen, trace_file=trace_path, model_dict=model_dict)
        sim_logger._log(f'[PIM] Trace generation completed')
        sim_logger._log(f'[PIM] Starting ramulator simulation...')
        cycles = _run_ramulator(trace_path, ramulator_config)
        if PIM_FREQ_GHZ > 0.0:
            latency = float(cycles) / (PIM_FREQ_GHZ * 1000000000.0)
        else:
            latency = 0.0
        sim_logger._log(f'[PIM] Latency computed: {latency:.6e} seconds ({cycles} cycles)')
        if use_cache:
            _pim_cache.set(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen, pim_config, ramulator_config, latency)
        return latency
    except Exception as e:
        sim_logger._log(f'[PIM] Error during latency computation: {e}')
        raise
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            sim_logger._log(f'[PIM] Warning: Failed to cleanup temp dir {temp_dir}: {e}')

class CostModel:

    def __init__(self, cluster: Cluster, dtype: str='fp16', pim_config_path: Optional[Path]=None, gb_config_path: Optional[Path]=None, ramulator_config_path: Optional[Path]=None, simulation_log_file: Optional[Path]=None, debug_traces: bool=False, model_dict: Optional[Dict]=None):
        self.cluster = cluster
        self.dtype = dtype
        self.pim_config_path = pim_config_path
        self.gb_config_path = gb_config_path
        self.ramulator_config_path = ramulator_config_path
        self.debug_traces = debug_traces
        global _sim_logger
        if _sim_logger is None:
            _sim_logger = get_simulation_logger(simulation_log_file)
        self.logger = _sim_logger
        self.pim_cache_enabled = True
        self._shared_model_dict: Optional[Dict] = model_dict
        if pim_config_path:
            if not pim_config_path.exists():
                raise ValueError(f'PIM config not found: {pim_config_path}')
            if model_dict is None:
                logger.debug(str('[WARNING] PIM config provided but model_dict is None. Call set_model_dict() before using PIM operations.'))
        if gb_config_path:
            if not gb_config_path.exists():
                raise ValueError(f'Global Buffer config not found: {gb_config_path}')
        if ramulator_config_path:
            if not ramulator_config_path.exists():
                raise ValueError(f'Ramulator config not found: {ramulator_config_path}')

    def set_model_dict(self, model_dict: Dict):
        """设置统一的模型字典"""
        if model_dict is None:
            raise ValueError('model_dict cannot be None')
        self._shared_model_dict = model_dict
        logger.debug(str(f'[CostModel] Model dictionary set with keys: {list(model_dict.keys())[:5]}...'))

    def get_model_dict(self) -> Dict:
        """获取统一的模型字典"""
        if self._shared_model_dict is None:
            raise RuntimeError('Model dictionary not set. You must call set_model_dict() or provide model_dict during initialization before using PIM operations.')
        return self._shared_model_dict

    def has_model_dict(self) -> bool:
        """检查是否已设置模型字典"""
        return self._shared_model_dict is not None

    def flop_time(self, flops: float, dev: DeviceSpec) -> float:
        if dev.tflops <= 0:
            return 0.0
        return flops / (dev.tflops * 1000000000000.0)

    def mem_time(self, bytes_amount: int, dev: DeviceSpec) -> float:
        bw = dev.mem_bw_GBs * 1000000000.0
        return 0.0 if bw <= 0 else bytes_amount / bw

    def link_time(self, bytes_amount: int, src: DeviceSpec, dst: DeviceSpec) -> float:
        bw = self.cluster.get_link_bw(src.name, dst.name) * 1000000000.0
        return 0.0 if bw <= 0 else bytes_amount / bw

    def comm_cost(self, src: DeviceSpec, dst: DeviceSpec, bytes_amount: int) -> float:
        if src.name == dst.name:
            return 0.0
        return self.link_time(bytes_amount, src, dst)

    def get_host_device(self) -> DeviceSpec:
        if HOST_NAME in self.cluster.devices:
            return self.cluster.devices[HOST_NAME]
        cpus = self.cluster.devices_by_type('cpu')
        return cpus[0] if cpus else next(iter(self.cluster.devices.values()))

    def device_preferred_fmt(self, dev: DeviceSpec) -> str:
        return DEVICE_PREFERRED_FORMAT.get(dev.type, 'ND')

    def format_size(self, size_bytes: int, fmt: str) -> int:
        m = float(FORMAT_SIZE_MULTIPLIER.get(fmt, 1.0))
        return int(size_bytes * m)

    def format_conversion_time(self, size_src_bytes: int, src_fmt: str, dst_fmt: str, dev: DeviceSpec) -> float:
        if src_fmt == dst_fmt:
            return 0.0
        bw_gbs = float(FORMAT_CONV_BW_GBs.get(dev.type, FORMAT_CONV_BW_GBs.get('default', 50.0)))
        bw = bw_gbs * 1000000000.0
        return 0.0 if bw <= 0 else size_src_bytes / bw

    def gb_move_and_format(self, dev: DeviceSpec, size_src_bytes: int, src_fmt: str, dst_fmt: str) -> float:
        host = self.get_host_device()
        t_move = self.link_time(size_src_bytes, host, dev)
        t_conv = self.format_conversion_time(size_src_bytes, src_fmt, dst_fmt, dev)
        return max(t_move, t_conv)

    def _resolve_pim_key(self, node) -> List[str]:
        """将 node 映射到 PIM op key"""
        keys: List[str] = []
        name = (node.name or '').upper()
        if name in ('Q', 'Q_PROJ'):
            keys.append('q_proj')
        elif name in ('K', 'K_PROJ'):
            keys.append('k_proj')
        elif name in ('V', 'V_PROJ'):
            keys.append('v_proj')
        elif name in ('O', 'WO', 'WO_PROJ', 'O_PROJ'):
            keys.append('wo_proj')
        elif name in ('FFN_W1', 'FFN_UP'):
            keys.append('ffn_up')
        elif name in ('FFN_W3', 'FFN_GATE'):
            keys.append('ffn_gate')
        elif name in ('FFN_W2', 'FFN_DOWN'):
            keys.append('ffn_down')
        elif 'QK' in name or 'SCORE' in name:
            keys.append('score')
        elif 'SV' in name or 'OUTPUT' in name:
            keys.append('output')
        elif 'SOFTMAX' in name:
            keys.append('softmax')
        elif 'RMSNORM' in name or name == 'LN':
            keys.append('rmsnorm')
        elif 'ROPE' in name:
            keys.append('rope')
        elif 'SILU' in name or 'SWIGLU' in name:
            keys.append('silu')
        elif 'GELU' in name:
            keys.append('gelu')
        elif name in ('ADD', 'RESIDUAL'):
            keys.append('residual')
        return keys

    def estimate_flops(self, node, batch: int, seq_len: int, phase: str) -> float:
        attrs = getattr(node, 'attrs', {}) or {}
        default = float(getattr(node, 'flops', 0.0) or 0.0)
        b = int(batch or attrs.get('batch', 0) or 0)
        if b <= 0:
            return default
        D = int(attrs.get('dim', 0) or 0)
        Hf = int(attrs.get('ffn_dim', attrs.get('hidden_dim', 0)) or 0)
        qh = int(attrs.get('q_heads', attrs.get('n_head', attrs.get('kv_heads', 0))) or 0)
        kvh = int(attrs.get('kv_heads', attrs.get('n_kv_heads', qh)) or 0)
        hd = int(attrs.get('head_dim', D // max(qh, 1)) or 0)
        q_dim = int(attrs.get('q_dim', qh * hd) or 0)
        kv_dim = int(attrs.get('kv_dim', kvh * hd) or 0)
        o_dim = int(attrs.get('o_dim', qh * hd) or 0)
        q_len = seq_len if phase == 'prefill' else 1
        kv_len = int(attrs.get('kv_len', attrs.get('past_kv_len', seq_len)) or seq_len)
        causal = bool(attrs.get('causal', True))

        def tri(n: int) -> int:
            return n * (n + 1) // 2
        C_MATMUL = 2.0
        C_LN = 5.0
        C_SOFTMAX = 5.0
        C_GELU = 6.0
        C_SILU = 5.0
        name = (getattr(node, 'name', '') or '').upper()
        if name in 'LN' and D > 0:
            return float(b * q_len * D * C_LN)
        if name in ('Q', 'K', 'V') and D > 0:
            out_dim = q_dim if name == 'Q' else kv_dim
            if out_dim <= 0:
                return default
            return float(C_MATMUL * D * out_dim * b * q_len)
        if name in 'QK' and qh > 0 and (hd > 0):
            if phase == 'prefill':
                pairs = tri(q_len) if causal else q_len * q_len
            else:
                pairs = kv_len
            return float(C_MATMUL * b * qh * hd * pairs)
        if name in 'SOFTMAX' and qh > 0:
            if phase == 'prefill':
                elems = tri(q_len) if causal else q_len * q_len
            else:
                elems = kv_len
            return float(b * qh * elems * C_SOFTMAX)
        if name in 'SV' and qh > 0 and (hd > 0):
            if phase == 'prefill':
                pairs = tri(q_len) if causal else q_len * q_len
            else:
                pairs = kv_len
            return float(C_MATMUL * b * qh * hd * pairs)
        if name in 'O' and D > 0 and (o_dim > 0):
            return float(C_MATMUL * o_dim * D * b * q_len)
        if name in ('FFN_W1', 'FFN_W3', 'FFN_UP', 'FFN_GATE') and D > 0 and (Hf > 0):
            return float(C_MATMUL * D * Hf * b * q_len)
        if name in ('FFN_W2', 'FFN_DOWN') and D > 0 and (Hf > 0):
            return float(C_MATMUL * Hf * D * b * q_len)
        if name in ('SWIGLU', 'SILU_GLU') and Hf > 0:
            return float(b * q_len * Hf * (C_SILU + 1.0))
        if name in ('GELU',) and Hf > 0:
            return float(b * q_len * Hf * C_GELU)
        if name == 'ADD' and D > 0:
            return float(b * q_len * D)
        if name in ('IDENTITY', 'RESIDUAL', 'DROPOUT') and D > 0:
            return float(b * q_len * D)
        if name in ('KV_READ', 'KV_WRITE', 'ROPE', 'ALIBI'):
            return 0.0
        return default

    def estimate_activation_bytes(self, node, batch: int, seq_len: int, phase: str):
        attrs = getattr(node, 'attrs', {}) or {}
        dtype_bytes = int(DTYPE_BYTES.get(self.dtype, 2))

        def to_bytes(elems: float) -> int:
            return int(max(0.0, float(elems))) * dtype_bytes
        b = int(batch or attrs.get('batch', 0) or 1)
        T = int(seq_len or 0)
        if T <= 0:
            return (0, 0)
        kv_len = int(attrs.get('kv_len', attrs.get('past_kv_len', T)) or T)
        active_tokens = T if phase == 'prefill' else 1
        causal = bool(attrs.get('causal', True))

        def tri(n: int) -> int:
            return n * (n + 1) // 2
        D = int(attrs.get('dim', attrs.get('hidden_size', 0)) or 0)
        Hf = int(attrs.get('ffn_dim', attrs.get('mlp_dim', 0)) or 0)
        hd = int(attrs.get('head_dim', 0) or 0)
        qh = int(attrs.get('q_heads', attrs.get('n_heads', 0)) or 0)
        kvh = int(attrs.get('n_kv_heads', attrs.get('kv_heads', qh)) or 0)
        q_dim = int(attrs.get('q_dim', qh * hd) or 0)
        kv_dim = int(attrs.get('kv_dim', kvh * hd) or 0)
        o_dim = int(attrs.get('o_dim', qh * hd) or 0)
        name = (getattr(node, 'name', attrs.get('op', '')) or '').upper()
        if phase == 'prefill':
            attn_pairs = tri(T) if causal else T * T
        else:
            attn_pairs = kv_len
        if name in 'LN' and D > 0:
            elems = b * active_tokens * D
            return (to_bytes(elems), to_bytes(elems))
        if name == 'Q' and D > 0:
            out_dim = q_dim if q_dim > 0 else D
            return (to_bytes(b * active_tokens * D), to_bytes(b * active_tokens * out_dim))
        if name in ('K', 'V') and D > 0:
            out_dim = kv_dim if kv_dim > 0 else D
            write_tokens = active_tokens
            return (to_bytes(b * active_tokens * D), to_bytes(b * write_tokens * out_dim))
        if name in 'O' and D > 0:
            inp_dim = o_dim if o_dim > 0 else D
            return (to_bytes(b * active_tokens * inp_dim), to_bytes(b * active_tokens * D))
        if name in ('FFN_W1', 'FFN_W3') and D > 0 and (Hf > 0):
            return (to_bytes(b * active_tokens * D), to_bytes(b * active_tokens * Hf))
        if name in ('FFN_W2',) and D > 0 and (Hf > 0):
            return (to_bytes(b * active_tokens * Hf), to_bytes(b * active_tokens * D))
        if name in ('SWIGLU', 'SILU_GLU') and Hf > 0:
            return (to_bytes(b * active_tokens * (2 * Hf)), to_bytes(b * active_tokens * Hf))
        if name in ('GELU', 'RELU'):
            width = Hf if Hf > 0 else D
            return (to_bytes(b * active_tokens * width), to_bytes(b * active_tokens * width))
        if name == 'ADD' and D > 0:
            read_elems = b * active_tokens * D * 2
            write_elems = b * active_tokens * D
            return (to_bytes(read_elems), to_bytes(write_elems))
        if name in ('IDENTITY',):
            elems = b * active_tokens * D
            return (to_bytes(elems), to_bytes(elems))
        if name in 'QK' and qh > 0 and (hd > 0):
            q_read = b * active_tokens * q_dim
            k_read = b * (T if phase == 'prefill' else kv_len) * kv_dim
            write_elems = b * qh * attn_pairs
            return (to_bytes(q_read + k_read), to_bytes(write_elems))
        if name in ('SOFTMAX', 'ATTN_SOFTMAX') and qh > 0:
            elems = b * qh * attn_pairs
            return (to_bytes(elems), to_bytes(elems))
        if name in 'SV' and qh > 0 and (hd > 0):
            attn_read = b * qh * attn_pairs
            v_read = b * (T if phase == 'prefill' else kv_len) * kv_dim
            out_elems = b * qh * active_tokens * hd
            return (to_bytes(attn_read + v_read), to_bytes(out_elems))
        if name in ('KV_READ', 'KV_WRITE'):
            read = 2 * batch * kvh * hd * kv_len
            write = 2 * batch * kvh * hd * active_tokens
            return (read, write)
        if D > 0:
            elems = b * active_tokens * D
            return (to_bytes(elems), to_bytes(elems))
        return (0, 0)

    def node_device_cost(self, node: TaskNode, dev: DeviceSpec, label: PlanLabel, batch: int, seq_len: int, phase: str) -> float:
        attrs = getattr(node, 'attrs', {}) or {}
        dim = int(attrs.get('dim', 0) or 0)
        n_heads = int(attrs.get('n_heads', attrs.get('q_heads', attrs.get('n_head', 0))) or 0)
        n_kv_heads = int(attrs.get('n_kv_heads', attrs.get('kv_heads', n_heads)) or n_heads)
        ffn_dim = int(attrs.get('ffn_dim', 0) or 0)
        ffn_dim_mul = float(attrs.get('ffn_dim_mul', 4.0))
        if dev.type == 'npu':
            logger.debug(str(f'[NPU-PATH] ✓ Entering NPU branch'))
            if ffn_dim == 0 and dim > 0:
                ffn_dim = int(ffn_dim_mul * dim)
            keys = self._resolve_pim_key(node)
            op_key = (keys[0].lower() if keys else None)
            # 预先算好内存时间；所有分支都与其取 max
            rd, wr = self.estimate_activation_bytes(node, batch, seq_len, phase)
            mem_t = self.mem_time(rd + wr, dev)
    
            ACT_KEYS = {
                'gelu','relu','silu','swish','mish','tanh','sigmoid','relu6','leaky_relu','elu','hardtanh','selu','prelu',
                'geglu','swiglu','glu_act','activation'
            }
            NORM_KEYS = {
                'layernorm','layer_norm','ln','rmsnorm','rms_norm','norm','groupnorm','group_norm','instancenorm','instance_norm','batchnorm','batch_norm'
            }
    
            # 1) Softmax → Softmax JSON
            if op_key == 'softmax':
                b = int(batch or attrs.get('batch', 1) or 1)
                qh = int(attrs.get('q_heads', attrs.get('n_heads', attrs.get('n_head', n_heads))) or n_heads)
                causal = bool(attrs.get('causal', True))
                active_tokens = int(seq_len if phase == 'prefill' else 1)
                M_rows = max(1, b * max(1, qh) * max(1, active_tokens))
                K_cols = max(1, int(seq_len if phase == 'prefill' else attrs.get('kv_len', attrs.get('past_kv_len', seq_len))))
                us = _predict_softmax_latency_us_from_json(M_rows, K_cols, phase=phase, causal=causal)
                logger.debug(str(f'[NPU-SOFTMAX] Inputs: M={M_rows}, K={K_cols}, phase={phase}, causal={causal}; us={us}'))
                if us is not None:
                    return max(us * 1e-06, mem_t)
    
            # 2) Activation（统一走 GELU）→ GELU JSON
            if op_key in ACT_KEYS:
                b = int(batch or 1)
                active_tokens = int(seq_len if phase == 'prefill' else 1)
                width = int(attrs.get('ffn_dim', attrs.get('hidden_dim', ffn_dim)) or 0)
                if not width or width <= 0: width = dim
                data_len = max(1, b) * max(1, active_tokens) * max(1, int(width))
                us = _predict_gelu_latency_us_from_json(data_len)
                logger.debug(str(f'[NPU-ACT] Inputs: data_len={data_len}; us={us}'))
                if us is not None:
                    return max(us * 1e-06, mem_t)
    
            # 3) Norm（统一走 LayerNorm）→ LayerNorm JSON
            if op_key in NORM_KEYS:
                b = int(batch or 1)
                rows = max(1, b) * (seq_len if phase == 'prefill' else 1)
                width = int(attrs.get('dim', attrs.get('hidden_dim', dim)) or dim)
                us = _predict_layernorm_latency_us_from_json(rows, width)
                logger.debug(str(f'[NPU-NORM] Inputs: rows={rows}, width={width}; us={us}'))
                if us is not None:
                    return max(us * 1e-06, mem_t)
    
            # 4) 线性/FFN/注意力 → MMAD JSON
            dims = _map_op_to_mmad_dims(op_key, dim, n_heads, n_kv_heads, ffn_dim, seq_len) if op_key else None
            if dims is not None:
                M, N, K, reps = dims
                us = _predict_mmad_latency_us_from_json(M, N, K)
                logger.debug(str(f'[NPU-MMAD] Inputs: M={M}, N={N}, K={K}, reps={reps}; us={us}'))
                if us is not None:
                    t_us = us * max(1, reps) * max(1, batch)
                    return max(t_us * 1e-06, mem_t)
    
            # 5) 回退 FLOPs
            flops = self.estimate_flops(node, batch, seq_len, phase)
            return max(self.flop_time(flops, dev), mem_t)
 
        if dev.type == 'pim':
            if not self.pim_config_path or not self.ramulator_config_path:
                logger.debug(str(f'[PIM] Warning: PIM configs not set, returning 0 for {node.name}'))
                return 0.0
            if ffn_dim == 0 and dim > 0:
                ffn_dim = int(ffn_dim_mul * dim)
            compute_time = 0.0
            keys = self._resolve_pim_key(node)
            op_key = keys[0] if keys else None
            if node.name.upper() in ('KV_READ', 'KV_WRITE'):
                compute_time = 0.0
            elif op_key and dim > 0 and (n_heads > 0):
                try:
                    model_dict = self.get_model_dict()
                    compute_time = _get_pim_latency_via_trace(op=op_key, pim_config=self.pim_config_path, ramulator_config=self.ramulator_config_path, dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, ffn_dim=ffn_dim, seqlen=seq_len if seq_len > 0 else None, model_dict=model_dict, use_cache=self.pim_cache_enabled)
                except Exception as e:
                    logger.debug(str(f'[PIM] ERROR: Failed to compute latency for {node.name}: {e}'))
                    raise RuntimeError(f'PIM latency computation failed for {node.name}: {e}')
            else:
                logger.debug(str(f'[PIM] Warning: Insufficient parameters for {node.name} (op={op_key}, dim={dim}, heads={n_heads})'))
            kv_in_pim = getattr(label, 'kv_in_pim', False)
            if kv_in_pim:
                mem_time = 0.0
            else:
                rd, wr = self.estimate_activation_bytes(node, batch, seq_len, phase)
                mem_time = self.mem_time(rd + wr, dev)
            return compute_time + mem_time
        return 0.0

    def weight_load_time_pim(self, weight_bytes: int) -> float:
        """
        使用trace仿真计算PIM的weight加载时间
        返回总延迟（读取+写入）
        """
        if not self.pim_config_path or not self.gb_config_path or (not self.ramulator_config_path):
            raise ValueError('PIM config, GB config, and Ramulator config must be set for weight loading simulation')
        dtype_bytes = DTYPE_BYTES.get(self.dtype, 2)
        try:
            model_dict = self.get_model_dict()
            read_lat, write_lat = _simulate_weight_loading_latency(weight_bytes, self.pim_config_path, self.gb_config_path, self.ramulator_config_path, dtype_bytes, use_cache=self.pim_cache_enabled, keep_traces=self.debug_traces, model_dict=model_dict)
            return read_lat + write_lat
        except Exception as e:
            logger.debug(str(f'[Weight Load] Falling back to bandwidth estimation due to: {e}'))
            pim_devs = self.cluster.devices_by_type('pim')
            if pim_devs:
                return self.mem_time(weight_bytes, pim_devs[0])
            return 0.0