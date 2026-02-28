from __future__ import annotations
"""PIM backend (CENT/AiM simulator + Ramulator trace)

This module is extracted from the original monolithic cost_model.py to keep
the core CostModel logic independent from backend-specific simulators.

Public entry points used by CostModel:
- _simulate_weight_loading_latency
- _get_pim_latency_via_trace
"""

import json
import os
import subprocess
import tempfile
import re
import sys
import shutil
import hashlib
import pickle
import contextlib
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from threading import Lock
from datetime import datetime
import logging

PIM_TRACE_SCALE_REPEATS: bool = str(os.environ.get("PIM_TRACE_SCALE_REPEATS", "1")).strip().lower() not in {
    "0", "false", "no", "off",
}


def _ensure_repo_root_on_syspath() -> None:
    try:
        here = Path(__file__).resolve()
    except Exception:
        return

    # Search current dir and ancestors for config.py
    for parent in (here.parent, *here.parents):
        try:
            if (parent / 'config.py').is_file():
                sp = str(parent)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
                return
        except Exception:
            continue


try:
    from config import attach_local_debug_filter, PIM_FREQ_GHZ, GB_FREQ_GHZ  # type: ignore
    from stats_recorder import get_simulation_logger  # type: ignore
except ModuleNotFoundError:
    _ensure_repo_root_on_syspath()
    from config import attach_local_debug_filter, PIM_FREQ_GHZ, GB_FREQ_GHZ  # type: ignore
    from stats_recorder import get_simulation_logger  # type: ignore

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)

def _file_signature(p: Path) -> str:
    try:
        rp = Path(p).expanduser().resolve()
    except Exception:
        rp = Path(p)
    try:
        st = rp.stat()
        return f"{rp}|{int(st.st_size)}|{int(st.st_mtime_ns)}"
    except Exception:
        return f"{rp}|missing"

_PIM_NORM_ALIASES = {
    'ln', 'layernorm', 'layer_norm',
    'rmsnorm', 'rms_norm',
    'norm',
    'groupnorm', 'group_norm',
    'instancenorm', 'instance_norm',
    'batchnorm', 'batch_norm',
}

_PIM_NORM_TOKEN_RE = re.compile(r'(^|_)(ln|layernorm|layer_norm|rmsnorm|rms_norm|groupnorm|group_norm|instancenorm|instance_norm|batchnorm|batch_norm|norm)($|_)')

# Map graph node labels / aliases to canonical ops supported by the PIM trace backend.
_PIM_OP_ALIASES = {
    # Attention projections
    'q': 'q_proj',
    'k': 'k_proj',
    'v': 'v_proj',
    'o': 'wo_proj',
    'wo': 'wo_proj',

    # Attention core
    'qk': 'score',
    'score': 'score',
    'softmax': 'softmax',
    'sv': 'output',
    'output': 'output',

    # FFN / MLP
    'ffn_w1': 'ffn_gate',
    'ffn_w3': 'ffn_up',
    'ffn_w2': 'ffn_down',
    'w1': 'ffn_gate',
    'w3': 'ffn_up',
    'w2': 'ffn_down',
    'mlp_w1': 'ffn_gate',
    'mlp_w3': 'ffn_up',
    'mlp_w2': 'ffn_down',

    # Residual / elementwise
    'add': 'residual',
    'residual': 'residual',

    # Activations
    'gelu': 'gelu',
    'silu': 'silu',
    'swiglu': 'silu',
    'swi_glu': 'silu',
    'act': 'silu',

    # Positional
    'rope': 'rope',
}

# Canonical ops currently supported by the trace generator / CENT(AiM) simulator.
PIM_TRACE_SUPPORTED_OPS = frozenset({
    'q_proj', 'k_proj', 'v_proj', 'wo_proj',
    'ffn_up', 'ffn_gate', 'ffn_down',
    'score', 'softmax', 'output',
    'rmsnorm', 'rope',
    'silu', 'gelu',
    'residual',
})


def _normalize_pim_op(op: str) -> str:
    """Normalize op names for the PIM trace backend.

    If an op label looks like a normalization operator (LN / LayerNorm /
    RMSNorm / *Norm / fused-*norm), return 'rmsnorm' since that is the only
    norm op currently traceable by the CENT/AiM simulator.
    """
    s = (op or '').strip().lower()
    if not s:
        return s
    s = s.replace('-', '_')
    if s in _PIM_NORM_ALIASES:
        return 'rmsnorm'
    # Common embedded/fused names: add_rmsnorm, rmsnorm_add, skip_layernorm, ...
    if ('rmsnorm' in s) or ('layernorm' in s) or s.endswith('norm'):
        return 'rmsnorm'
    # Some graphs may name norms like ln1 / ln2
    if s.startswith('ln') and (len(s) == 2 or s[2:].isdigit()):
        return 'rmsnorm'
    if _PIM_NORM_TOKEN_RE.search(s):
        return 'rmsnorm'
    # Non-norm ops: map aliases to canonical trace ops (if known)
    s = _PIM_OP_ALIASES.get(s, s)
    return s

# ---- AiM simulator intergeration (git submodule: submodules/CENT) ----
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
    cent_path, _ = _ensure_cent_on_path()

    # --- Patch module name collision: `utils` ---
    import importlib.util
    old_utils = sys.modules.get('utils', None)
    cent_utils_path = (cent_path / 'utils.py')
    cent_utils_mod = None
    if cent_utils_path.exists():
        try:
            spec = importlib.util.spec_from_file_location('_cent_utils_temp', str(cent_utils_path))
            if spec is not None and spec.loader is not None:
                cent_utils_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cent_utils_mod)  # type: ignore[attr-defined]
        except Exception:
            cent_utils_mod = None

    try:
        if cent_utils_mod is not None:
            sys.modules['utils'] = cent_utils_mod

        try:
            from Llama import TransformerBlockLlama as TransformerBlock
            return TransformerBlock
        except Exception:
            from TransformerBlock import TransformerBlock
            return TransformerBlock
    except Exception as e:
        raise RuntimeError(f'Cannot import TransformerBlock from CENT: {e}')
    finally:
        # Restore previous `utils` to avoid breaking other submodules (e.g., LLMCompass).
        if old_utils is not None:
            sys.modules['utils'] = old_utils
        else:
            sys.modules.pop('utils', None)

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
    return SimpleNamespace(
        DRAM_column=int(cfg['DRAM_column']), 
        DRAM_row=int(cfg['DRAM_row']), 
        burst_length=int(cfg['burst_length']), 
        num_banks=int(cfg['num_banks']), 
        num_channels=int(cfg['num_channels']), 
        threads=int(cfg['threads']), 
        reuse_size=int(cfg['reuse_size']), 
        channels_per_block=int(cpb), 
        max_seq_len=int(cfg['max_seq_len']), 
        only_trace=True, 
        op_trace=False, 
        trace_file=trace_file, 
        pim_compute=True, 
        model='llama_like', 
        embedding='rope', 
        seqlen=16, 
        model_parallel=False, 
        FC_devices=1, 
        pipeline_parallel=False, 
        inter_device_attention=False, 
        only_FC=False, trace_prepare=False, 
        trace_norm=False, 
        trace_fc_kqvo=False, 
        trace_attention=False, 
        trace_softmax=False, 
        trace_fc_ffn=False, 
        trace_activation=False, 
        GEMV='reuse-GB')

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


def _emit_single_op_trace(
    block,
    op: str,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_dim: int,
    seqlens: Optional[List[int]],
    *,
    head_dim: Optional[int] = None,
    q_dim: Optional[int] = None,
    kv_dim: Optional[int] = None,
    o_dim: Optional[int] = None,
) -> None:

    channel_lst, FC_total_banks, channels_required = _calc_channels(block)
    hd = int(head_dim) if (head_dim is not None and int(head_dim) > 0) else int(dim // max(1, int(n_heads)))
    q_dim_eff = int(q_dim) if (q_dim is not None and int(q_dim) > 0) else int(max(1, int(n_heads)) * max(1, int(hd)))
    kv_dim_eff = int(kv_dim) if (kv_dim is not None and int(kv_dim) > 0) else int(max(1, int(n_kv_heads)) * max(1, int(hd)))
    o_dim_eff = int(o_dim) if (o_dim is not None and int(o_dim) > 0) else int(q_dim_eff)

    @contextlib.contextmanager
    def _temp_block_attrs(**updates):
        old: Dict[str, Any] = {}
        for k, v in updates.items():
            if hasattr(block, k):
                try:
                    old[k] = getattr(block, k)
                    setattr(block, k, v)
                except Exception:
                    pass
        try:
            yield
        finally:
            for k, v in old.items():
                try:
                    setattr(block, k, v)
                except Exception:
                    pass
    if op in ('q_proj', 'k_proj', 'v_proj', 'wo_proj', 'ffn_up', 'ffn_gate', 'ffn_down'):
        if op == 'q_proj':
            row_tag, V, N = ('wq_row_index', int(dim), int(q_dim_eff))
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_sa_weight')
        elif op == 'k_proj':
            row_tag, V, N = ('wk_row_index', int(dim), int(kv_dim_eff))
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_sa_weight')
        elif op == 'v_proj':
            row_tag, V, N = ('wv_row_index', int(dim), int(kv_dim_eff))
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_sa_weight')
        elif op == 'wo_proj':
            row_tag, V, N = ('wo_row_index', int(o_dim_eff), int(dim))
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_sa_weight')
        elif op == 'ffn_up':
            row_tag, V, N = ('w3_row_index', dim, ffn_dim)
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_ffn_weight')
        elif op == 'ffn_gate':
            row_tag, V, N = ('w1_row_index', dim, ffn_dim)
            block.Vector_Matrix_Mul_weight_af_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_ffn_weight')
        elif op == 'ffn_down':
            row_tag, V, N = ('w2_row_index', ffn_dim, dim)
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, 'breakdown_ffn_weight')
    elif op in ('score', 'softmax', 'output'):
        for S in seqlens or [1]:
            if op == 'score':
                with _temp_block_attrs(n_heads=int(n_heads), n_kv_heads=int(n_kv_heads), head_dim=int(hd), dim=int(q_dim_eff)):
                    block.Vector_Matrix_Mul_score_pim_only_trace(block.cache_k_row_index, S, 'breakdown_sa_score')
            elif op == 'output':
                with _temp_block_attrs(n_heads=int(n_heads), n_kv_heads=int(n_kv_heads), head_dim=int(hd), dim=int(q_dim_eff)):
                    block.Vector_Matrix_Mul_output_pim_only_trace(block.cache_v_row_index, S, 'breakdown_sa_output')
            elif op == 'softmax':
                rows_per_score = (S - 1) // block.DRAM_column + 1
                for r in range(rows_per_score):
                    op_size = block.DRAM_column // block.burst_length if r < rows_per_score - 1 else (S - block.DRAM_column * r - 1) // block.burst_length + 1
                    block.EWMUL_only_trace(channel_lst, block.scores_row_index + r, op_size)
                with _temp_block_attrs(n_heads=int(n_heads)):
                    block.time['RD_SBK'] += block.timing_constant['RD_SBK'] + S * int(n_heads) // block.burst_length
                    block.load_from_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 2, S)
                    block.time['WR_SBK'] += block.timing_constant['WR_SBK'] + S * int(n_heads) // block.burst_length
                    block.store_for_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 0, S)
                rows_per_score = (S - 1) // block.DRAM_column + 1
                for r in range(rows_per_score):
                    op_size = block.DRAM_column // block.burst_length if r < rows_per_score - 1 else (S - block.DRAM_column * r - 1) // block.burst_length + 1
                    block.EWMUL_only_trace(channel_lst, block.scores_row_index + r, op_size)
                with _temp_block_attrs(n_heads=int(n_heads)):
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
        ew_len = (hd - 1) // (block.total_banks // 4) + 1
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


def _generate_pim_trace(
    op: str,
    pim_config: Path,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_dim: int,
    seqlen: Optional[int],
    head_dim: Optional[int],
    q_dim: Optional[int],
    kv_dim: Optional[int],
    o_dim: Optional[int],
    phase: str,
    trace_file: Path,
    model_dict: Optional[Dict] = None,
    batch: int = 1,
) -> None:
    op = _normalize_pim_op(op)
    if model_dict is None:
        raise ValueError('Model dictionary must be provided for PIM trace generation')
    TransformerBlock = _get_transformer_block()
    pim_cfg = _load_memory_config(pim_config)
    args = _make_tb_args_from_pim(pim_cfg, str(trace_file))
    args.op_trace = True
    S = int(seqlen or args.seqlen or 16)
    block = TransformerBlock(model_dict, args)
    if hasattr(block, 'memory_mapping'):
        block.memory_mapping()
    ph = str(phase or '').strip().lower()
    is_prefill = (ph == 'prefill')

    try:
        B = max(1, int(batch or 1))
    except Exception:
        B = 1

    # Prefill: query_len == key_len == S. Decode: query_len == 1.
    T = int(S if is_prefill else 1)
    if op in ('score', 'softmax', 'output'):
        if is_prefill and T > 1:
            seqlens_list: List[int] = list(range(1, S + 1))
        else:
            seqlens_list = [S]
        for _ in range(B):
            _emit_single_op_trace(
                block, op, int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim), seqlens_list,
                head_dim=head_dim, q_dim=q_dim, kv_dim=kv_dim, o_dim=o_dim,
            )
    else:
        for _ in range(B):
            for _ in range(T):
                _emit_single_op_trace(
                    block, op, int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim), None,
                    head_dim=head_dim, q_dim=q_dim, kv_dim=kv_dim, o_dim=o_dim,
                )

    # Finalize trace once.
    if hasattr(block, 'finish'):
        try:
            block.finish()
        except Exception:
            if getattr(block, 'file', None):
                block.file.write('AiM EOC\n')
    else:
        if getattr(block, 'file', None):
            block.file.write('AiM EOC\n')

    if getattr(block, 'file', None):
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
    if model_dict is None:
        raise ValueError('Model dictionary must be provided for weight loading simulation')
    cache_key = f"weight_load|{int(weight_bytes)}|{int(dtype_bytes)}|{_file_signature(pim_config_path)}|{_file_signature(gb_config_path)}|{_file_signature(ramulator_config_path)}"
    if use_cache:
        cached = _pim_cache.cache.get(hashlib.md5(cache_key.encode()).hexdigest())
        if cached is not None:
            logger.debug(str(f'[Weight Load Cache] Hit for {weight_bytes} bytes'))
            return cached

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
        read_latency = float(read_cycles) / (f_gb * 1000000000) if f_gb > 0 else 0.0 #us
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
        write_latency = float(write_cycles) / (f_pim * 1000000000) if f_pim > 0 else 0.0 #us
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

def _run_ramulator(trace_path: Path, ramulator_config: Path, timeout: int = 3000) -> int:
    """Run Ramulator2 on a trace file and return cycle count."""
    if not trace_path.exists():
        raise FileNotFoundError(f'Trace file not found: {trace_path}')
    if not ramulator_config.exists():
        raise FileNotFoundError(f'Ramulator config not found: {ramulator_config}')

    def _resolve_ramulator2_exe() -> Path:
        for k in ("RAMULATOR2_BIN", "PIM_RAMULATOR_BIN"):
            v = (os.environ.get(k) or "").strip()
            if not v:
                continue

            if ("/" not in v) and ("\\" not in v):
                w = shutil.which(v)
                if w:
                    return Path(w)

            p = Path(v).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.exists():
                return p

        p = (Path(__file__).resolve().parent / "ramulator2")
        if p.exists():
            return p

        w = shutil.which("ramulator2")
        if w:
            return Path(w)

        return Path.cwd() / "ramulator2"

    ramulator_exe = _resolve_ramulator2_exe()

    if not ramulator_exe.exists():
        raise FileNotFoundError(
            "Ramulator2 executable not found.\n"
            f"  resolved={ramulator_exe}\n"
            f"  CWD={Path.cwd()}\n"
            f"  env.RAMULATOR2_BIN={os.environ.get('RAMULATOR2_BIN')}\n"
            f"  env.PIM_RAMULATOR_BIN={os.environ.get('PIM_RAMULATOR_BIN')}\n"
            "Hint: pass --pim-ramulator-bin /abs/path/to/ramulator2 (or export RAMULATOR2_BIN)."
        )
    if not os.access(str(ramulator_exe), os.X_OK):
        raise PermissionError(f"Ramulator2 is not executable: {ramulator_exe} (try chmod +x)")

    cmd = [str(ramulator_exe), '-f', str(ramulator_config), '-t', str(trace_path)]
    logger.debug(str(f"[PIM] Running ramulator: {' '.join(cmd)}"))
    logger.debug(str(f"[PIM] CWD: {Path.cwd()}"))

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

        out = (result.stdout or '') + '\n' + (result.stderr or '')
        m = re.search(r'(?mi)^\s*memory_system_cycles\s*:\s*([0-9]+)\s*$', out)
        if not m:
            raise RuntimeError(f'Failed to parse Ramulator output for cycles:\n{out}')
        cycles = int(m.group(1))
        if cycles <= 0:
            raise RuntimeError(f'Invalid cycle count from Ramulator: {cycles}')
        logger.debug(str(f'[PIM] Ramulator cycles: {cycles}'))
        return cycles

    except subprocess.TimeoutExpired:
        raise RuntimeError(f'Ramulator timed out after {timeout}s')
    except Exception as e:
        raise RuntimeError(f'Ramulator execution failed: {e}')

class PIMLatencyCache:
    def __init__(self, cache_file: Optional[Path]=None):
        self.cache_file = cache_file or Path('./pkl/pim_latency_cache.pkl')
        self.cache: Dict[str, Any] = {}
        self.lock = Lock()
        self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                logger.debug(str(f'[PIM Cache] Failed to load cache: {e}'))
                self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.debug(str(f'[PIM Cache] Failed to save cache: {e}'))

    def _make_key(
        self,
        op: str,
        phase: str,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        seqlen: Optional[int],
        head_dim: Optional[int],
        q_dim: Optional[int],
        kv_dim: Optional[int],
        o_dim: Optional[int],
        pim_config: Path,
        ramulator_config: Path,
        batch: int = 1,
    ) -> str:
        ph = str(phase or '').strip().lower() or 'decode'

        try:
            b = max(1, int(batch or 1))
        except Exception:
            b = 1

        params_base = (
            f'{op}|{ph}|{int(dim)}|{int(n_heads)}|{int(n_kv_heads)}|{int(ffn_dim)}|{int(seqlen) if seqlen is not None else -1}'
            f'|hd={int(head_dim) if head_dim is not None else -1}'
            f'|q={int(q_dim) if q_dim is not None else -1}'
            f'|kv={int(kv_dim) if kv_dim is not None else -1}'
            f'|o={int(o_dim) if o_dim is not None else -1}'
        )
        cfgs = f'{_file_signature(pim_config)}|{_file_signature(ramulator_config)}'

        if int(b) == 1:
            key = f'v2|{params_base}|{cfgs}'
        else:
            key = f'v3|{params_base}|b={int(b)}|{cfgs}'
        return hashlib.md5(key.encode()).hexdigest()

    def get(
        self,
        op: str,
        phase: str,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        seqlen: Optional[int],
        head_dim: Optional[int],
        q_dim: Optional[int],
        kv_dim: Optional[int],
        o_dim: Optional[int],
        pim_config: Path,
        ramulator_config: Path,
        batch: int = 1,
    ) -> Optional[float]:
        key = self._make_key(op, phase, dim, n_heads, n_kv_heads, ffn_dim, seqlen, head_dim, q_dim, kv_dim, o_dim, pim_config, ramulator_config, batch=batch)
        with self.lock:
            v = self.cache.get(key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def set(
        self,
        op: str,
        phase: str,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        seqlen: Optional[int],
        head_dim: Optional[int],
        q_dim: Optional[int],
        kv_dim: Optional[int],
        o_dim: Optional[int],
        pim_config: Path,
        ramulator_config: Path,
        latency: float,
        batch: int = 1,
    ):
        key = self._make_key(op, phase, dim, n_heads, n_kv_heads, ffn_dim, seqlen, head_dim, q_dim, kv_dim, o_dim, pim_config, ramulator_config, batch=batch)
        with self.lock:
            self.cache[key] = float(latency)
            self._save_cache()

_pim_cache = PIMLatencyCache()


def _get_pim_latency_via_trace(
    op: str,
    pim_config: Path,
    ramulator_config: Path,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_dim: int,
    seqlen: Optional[int],
    batch: int = 1,
    head_dim: Optional[int] = None,
    q_dim: Optional[int] = None,
    kv_dim: Optional[int] = None,
    o_dim: Optional[int] = None,
    phase: str = "decode",
    model_dict: Optional[Dict] = None,
    use_cache: bool = True,
    ramulator_timeout_s: int = 3000,
    keep_traces: bool = False,
    trace_dir: Optional[Path] = None,
    trace_prefix: Optional[str] = None,
    **_unused_kwargs,
) -> float:
    orig_op = op
    op = _normalize_pim_op(op)

    if model_dict is None:
        raise ValueError('Model dictionary must be provided for PIM latency computation')

    try:
        b = max(1, int(batch or 1))
    except Exception:
        b = 1

    ph = str(phase or '').strip().lower() or 'decode'
    sim_logger = get_simulation_logger()
    sim_logger.record_simulation(op, dim, n_heads, n_kv_heads, ffn_dim, seqlen)
    if orig_op != op:
        sim_logger._log(f"[PIM] normalize op '{orig_op}' -> '{op}'")

    # ------------------------------------------------------------------
    # Fast(er) mode: simulate one instance, then scale by batch/prefill.
    # ------------------------------------------------------------------
    if PIM_TRACE_SCALE_REPEATS:
        # Scale factor requested by caller.
        scale = 1
        try:
            scale *= max(1, int(b))
        except Exception:
            scale *= 1

        if ph == 'prefill':
            try:
                s = int(seqlen) if seqlen is not None else 1
                scale *= max(1, s)
            except Exception:
                scale *= 1

        # We generate a "unit" trace that corresponds to one token in decode
        # (batch=1). Prefill is approximated by multiplying by prefill length.
        base_ph = 'decode'
        base_b = 1

        if use_cache:
            cached_base = _pim_cache.get(
                op, base_ph,
                int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim),
                (int(seqlen) if seqlen is not None else None),
                (int(head_dim) if head_dim is not None else None),
                (int(q_dim) if q_dim is not None else None),
                (int(kv_dim) if kv_dim is not None else None),
                (int(o_dim) if o_dim is not None else None),
                pim_config, ramulator_config,
                batch=int(base_b),
            )
            if cached_base is not None:
                return float(cached_base) * float(scale)

        msg = (
            f"[PIM] Computing latency for {op} phase={ph} "
            f"(scaled: base_phase={base_ph}, base_batch={base_b}, scale={scale}; "
            f"dim={dim}, heads={n_heads}, seq={seqlen}, batch={b})"
        )
        sim_logger._log(msg)

        # Trace directory handling (keep_traces/trace_dir are best-effort).
        if keep_traces or trace_dir is not None:
            out_dir = Path(trace_dir) if trace_dir is not None else Path('./debug_pim_traces')
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir = out_dir
            cleanup = False
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix='pim_trace_'))
            cleanup = True

        try:
            prefix = str(trace_prefix or '').strip() or op
            trace_path = tmp_dir / f'{prefix}_{base_ph}_unit.trace'
            sim_logger._log(f'[PIM] Generating unit trace: {trace_path}')
            _generate_pim_trace(
                op=op,
                pim_config=pim_config,
                dim=int(dim),
                n_heads=int(n_heads),
                n_kv_heads=int(n_kv_heads),
                ffn_dim=int(ffn_dim),
                seqlen=int(seqlen) if seqlen is not None else None,
                head_dim=int(head_dim) if head_dim is not None else None,
                q_dim=int(q_dim) if q_dim is not None else None,
                kv_dim=int(kv_dim) if kv_dim is not None else None,
                o_dim=int(o_dim) if o_dim is not None else None,
                phase=base_ph,
                trace_file=trace_path,
                model_dict=model_dict,
                batch=int(base_b),
            )
            sim_logger._log('[PIM] Unit trace generation completed')
            sim_logger._log('[PIM] Starting ramulator simulation (unit trace)...')
            timeout_s = int(ramulator_timeout_s) if ramulator_timeout_s else 3000
            cycles = _run_ramulator(trace_path, ramulator_config, timeout=timeout_s)
            if PIM_FREQ_GHZ > 0.0:
                base_latency = float(cycles) / (PIM_FREQ_GHZ * 1000000000.0)
            else:
                base_latency = 0.0
            total_latency = float(base_latency) * float(scale)
            sim_logger._log(
                f'[PIM] Latency computed: base={base_latency:.6e}s ({cycles} cycles), '
                f'scale={scale} => total={total_latency:.6e}s'
            )
            if use_cache:
                _pim_cache.set(
                    op, base_ph,
                    int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim),
                    (int(seqlen) if seqlen is not None else None),
                    (int(head_dim) if head_dim is not None else None),
                    (int(q_dim) if q_dim is not None else None),
                    (int(kv_dim) if kv_dim is not None else None),
                    (int(o_dim) if o_dim is not None else None),
                    pim_config, ramulator_config,
                    float(base_latency),
                    batch=int(base_b),
                )
            return float(total_latency)
        except Exception as e:
            sim_logger._log(f'[PIM] Error during latency computation (scaled mode): {e}')
            raise
        finally:
            if cleanup:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception as e:
                    sim_logger._log(f'[PIM] Warning: Failed to cleanup temp dir {tmp_dir}: {e}')

    # ------------------------------------------------------------------
    # Original mode: explicitly unroll batch/prefill into the trace.
    # ------------------------------------------------------------------
    if use_cache:
        cached = _pim_cache.get(
            op, ph,
            int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim),
            (int(seqlen) if seqlen is not None else None),
            (int(head_dim) if head_dim is not None else None),
            (int(q_dim) if q_dim is not None else None),
            (int(kv_dim) if kv_dim is not None else None),
            (int(o_dim) if o_dim is not None else None),
            pim_config, ramulator_config,
            batch=int(b),
        )
        if cached is not None:
            return float(cached)

    msg = f"[PIM] Computing latency for {op} phase={ph} (dim={dim}, heads={n_heads}, seq={seqlen}, batch={b})"
    sim_logger._log(msg)

    temp_dir = Path(tempfile.mkdtemp(prefix='pim_trace_'))
    try:
        trace_path = temp_dir / f'{op}_{ph}_trace.trace'
        sim_logger._log(f'[PIM] Generating trace: {trace_path}')
        _generate_pim_trace(
            op=op,
            pim_config=pim_config,
            dim=int(dim),
            n_heads=int(n_heads),
            n_kv_heads=int(n_kv_heads),
            ffn_dim=int(ffn_dim),
            seqlen=int(seqlen) if seqlen is not None else None,
            head_dim=int(head_dim) if head_dim is not None else None,
            q_dim=int(q_dim) if q_dim is not None else None,
            kv_dim=int(kv_dim) if kv_dim is not None else None,
            o_dim=int(o_dim) if o_dim is not None else None,
            phase=ph,
            trace_file=trace_path,
            model_dict=model_dict,
            batch=int(b),
        )
        sim_logger._log('[PIM] Trace generation completed')
        sim_logger._log('[PIM] Starting ramulator simulation...')
        timeout_s = int(ramulator_timeout_s) if ramulator_timeout_s else 3000
        cycles = _run_ramulator(trace_path, ramulator_config, timeout=timeout_s)
        if PIM_FREQ_GHZ > 0.0:
            latency = float(cycles) / (PIM_FREQ_GHZ * 1000000000.0)
        else:
            latency = 0.0
        sim_logger._log(f'[PIM] Latency computed: {latency:.6e} seconds ({cycles} cycles)')
        if use_cache:
            _pim_cache.set(
                op, ph,
                int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim),
                (int(seqlen) if seqlen is not None else None),
                (int(head_dim) if head_dim is not None else None),
                (int(q_dim) if q_dim is not None else None),
                (int(kv_dim) if kv_dim is not None else None),
                (int(o_dim) if o_dim is not None else None),
                pim_config, ramulator_config,
                float(latency),
                batch=int(b),
            )
        return float(latency)
    except Exception as e:
        sim_logger._log(f'[PIM] Error during latency computation: {e}')
        raise
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            sim_logger._log(f'[PIM] Warning: Failed to cleanup temp dir {temp_dir}: {e}')
