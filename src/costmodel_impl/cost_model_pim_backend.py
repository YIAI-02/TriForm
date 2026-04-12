from __future__ import annotations
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
from dataclasses import dataclass, replace
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
    import config as _config  # type: ignore
    from config import attach_local_debug_filter  # type: ignore
    from stats_recorder import get_simulation_logger  # type: ignore
except ModuleNotFoundError:
    _ensure_repo_root_on_syspath()
    import config as _config  # type: ignore
    from config import attach_local_debug_filter  # type: ignore
    from stats_recorder import get_simulation_logger  # type: ignore

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)


def _require_pim_freq_ghz(pim_freq_ghz: Any) -> float:
    try:
        freq = float(pim_freq_ghz)
    except Exception as exc:
        raise RuntimeError(
            'PIM trace backend requires device freq_ghz to be set in the hardware JSON for each PIM device.'
        ) from exc
    if not (freq > 0.0):
        raise RuntimeError(
            f'PIM trace backend requires device freq_ghz > 0 in hardware JSON, got {pim_freq_ghz!r}.'
        )
    return float(freq)


@dataclass(frozen=True)
class PimWeightDesc:
    weight_id: str
    storage_fmt: str  # ND / PIM-OPT / NZ(host-side NPU layout, must be unpacked before ND-style PIM write)
    op: str           # q_proj/k_proj/v_proj/wo_proj/ffn_gate/ffn_up/ffn_down
    rows: int         # logical matrix rows (already shard-local)
    cols: int         # logical matrix cols
    row_index_attr: str


_PIM_WEIGHT_WRITE_OPS = frozenset({
    'q_proj', 'k_proj', 'v_proj', 'wo_proj',
    'ffn_gate', 'ffn_up', 'ffn_down',
})

_PIM_WEIGHT_ROW_INDEX_ATTR: Dict[str, str] = {
    'q_proj': 'wq_row_index',
    'k_proj': 'wk_row_index',
    'v_proj': 'wv_row_index',
    'wo_proj': 'wo_row_index',
    'ffn_gate': 'w1_row_index',
    'ffn_up': 'w3_row_index',
    'ffn_down': 'w2_row_index',
}


def _require_positive_int(value: Any, *, field: str, node_name: str, weight_id: str) -> int:
    out = int(value)
    if out <= 0:
        raise RuntimeError(
            f"Invalid PIM weight metadata: field='{field}' must be > 0 for node='{node_name}' weight_id='{weight_id}', got {value!r}."
        )
    return out


def _require_node_attrs_dict(node: Any) -> Dict[str, Any]:
    attrs = getattr(node, 'attrs', None)
    if not isinstance(attrs, dict):
        raise RuntimeError(
            f"PIM weight descriptor requires node.attrs to be a dict, but node='{getattr(node, 'name', '?')}' got {type(attrs).__name__}."
        )
    return attrs


def _node_attr_int(attrs: Dict[str, Any], key: str, *, node_name: str, weight_id: str) -> int:
    if key not in attrs:
        raise RuntimeError(
            f"Missing required attrs['{key}'] for node='{node_name}' weight_id='{weight_id}' when building PimWeightDesc."
        )
    return _require_positive_int(attrs[key], field=key, node_name=node_name, weight_id=weight_id)


def make_pim_weight_desc_from_node(node: Any, storage_fmt: str) -> PimWeightDesc:
    node_name = str(getattr(node, 'name', '') or '').strip()
    weight_id = str(getattr(node, 'weight_id', '') or '').strip()
    if not weight_id:
        raise RuntimeError(
            f"PIM weight descriptor requires a non-empty weight_id, but node='{node_name}' does not have one."
        )

    fmt = str(storage_fmt or '').strip().upper()
    if fmt not in {'ND', 'PIM-OPT', 'NZ'}:
        raise RuntimeError(
            f"Unsupported PIM weight storage format '{storage_fmt}' for weight_id='{weight_id}'. Allowed=['ND', 'NZ', 'PIM-OPT']."
        )

    op = _normalize_pim_op(node_name)
    if op not in _PIM_WEIGHT_WRITE_OPS:
        raise RuntimeError(
            f"Unsupported PIM weight op '{node_name}' normalized='{op}' for weight_id='{weight_id}'."
        )

    attrs = _require_node_attrs_dict(node)
    dim = _node_attr_int(attrs, 'dim', node_name=node_name, weight_id=weight_id)
    q_dim = int(attrs.get('q_dim', 0) or 0)
    kv_dim = int(attrs.get('kv_dim', 0) or 0)
    o_dim = int(attrs.get('o_dim', 0) or 0)
    ffn_dim = int(attrs.get('ffn_dim', 0) or 0)

    if op == 'q_proj':
        rows, cols = _require_positive_int(q_dim, field='q_dim', node_name=node_name, weight_id=weight_id), dim
    elif op == 'k_proj':
        rows, cols = _require_positive_int(kv_dim, field='kv_dim', node_name=node_name, weight_id=weight_id), dim
    elif op == 'v_proj':
        rows, cols = _require_positive_int(kv_dim, field='kv_dim', node_name=node_name, weight_id=weight_id), dim
    elif op == 'wo_proj':
        rows, cols = dim, _require_positive_int(o_dim, field='o_dim', node_name=node_name, weight_id=weight_id)
    elif op == 'ffn_gate':
        rows, cols = _require_positive_int(ffn_dim, field='ffn_dim', node_name=node_name, weight_id=weight_id), dim
    elif op == 'ffn_up':
        rows, cols = _require_positive_int(ffn_dim, field='ffn_dim', node_name=node_name, weight_id=weight_id), dim
    elif op == 'ffn_down':
        rows, cols = dim, _require_positive_int(ffn_dim, field='ffn_dim', node_name=node_name, weight_id=weight_id)
    else:
        raise RuntimeError(
            f"Internal error: no PimWeightDesc shape rule for op='{op}' weight_id='{weight_id}'."
        )

    return PimWeightDesc(
        weight_id=weight_id,
        storage_fmt=fmt,
        op=op,
        rows=int(rows),
        cols=int(cols),
        row_index_attr=_PIM_WEIGHT_ROW_INDEX_ATTR[op],
    )

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
    'wq': 'q_proj',
    'wk': 'k_proj',
    'wv': 'v_proj',
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

_PIM_OP_TOKENS: Tuple[str, ...] = (
    'ffn_w1', 'ffn_w2', 'ffn_w3',
    'mlp_w1', 'mlp_w2', 'mlp_w3',
    'q_proj', 'k_proj', 'v_proj', 'wo_proj',
    'ffn_gate', 'ffn_up', 'ffn_down',
    'softmax', 'score', 'output',
    'swi_glu', 'swiglu', 'silu', 'gelu',
    'rope',
    'residual', 'add',
    'wq', 'wk', 'wv', 'wo',
    'qk', 'sv',
    'q', 'k', 'v', 'o',
    'w1', 'w2', 'w3',
)

_PIM_OP_TOKEN_RE = re.compile(r'(^|_)(' + '|'.join(re.escape(t) for t in _PIM_OP_TOKENS) + r')($|_)')


def _extract_pim_op_token(s: str) -> Optional[str]:
    if not s:
        return None
    # Avoid accidental matches on comm-like ops (e.g., k_write / v_write).
    ss = str(s).strip().lower()
    if 'write' in ss and ('k_write' in ss or 'v_write' in ss or ss.endswith('_write')):
        return None
    m = _PIM_OP_TOKEN_RE.search(ss)
    if not m:
        return None
    try:
        return str(m.group(2))
    except Exception:
        return None

def _normalize_pim_op(op: str) -> str:
    """Normalize op names for the PIM trace backend."""
    s = (op or '').strip().lower()
    if not s:
        return s
    s = s.replace('-', '_').replace('.', '_').replace('/', '_').replace('\\', '_')
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

    if s in _PIM_OP_ALIASES:
        return _PIM_OP_ALIASES[s]
    tok = _extract_pim_op_token(s)
    if tok:
        s = tok

    # Final alias mapping to canonical trace ops (if known).
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

        p = (Path(__file__).resolve().parent.parent / "ramulator2")
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
        env_path = str(os.environ.get('PIM_LATENCY_CACHE_FILE', '') or '').strip()
        self.cache_file = cache_file or (Path(env_path) if env_path else Path('./pkl/pim_latency_cache.pkl'))
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
        pim_freq_ghz: Optional[float] = None,
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
        try:
            freq_tag = float(pim_freq_ghz) if pim_freq_ghz is not None else -1.0
        except Exception:
            freq_tag = -1.0

        scale_flag = 1 if PIM_TRACE_SCALE_REPEATS else 0

        if int(b) == 1:
            key = f'v6|{params_base}|scale={int(scale_flag)}|freq={freq_tag}|{cfgs}'
        else:
            key = f'v6|{params_base}|b={int(b)}|scale={int(scale_flag)}|freq={freq_tag}|{cfgs}'
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
        pim_freq_ghz: Optional[float] = None,
    ) -> Optional[float]:
        key = self._make_key(op, phase, dim, n_heads, n_kv_heads, ffn_dim, seqlen, head_dim, q_dim, kv_dim, o_dim, pim_config, ramulator_config, batch=batch, pim_freq_ghz=pim_freq_ghz)
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
        pim_freq_ghz: Optional[float] = None,
    ):
        key = self._make_key(op, phase, dim, n_heads, n_kv_heads, ffn_dim, seqlen, head_dim, q_dim, kv_dim, o_dim, pim_config, ramulator_config, batch=batch, pim_freq_ghz=pim_freq_ghz)
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
    pim_freq_ghz: Optional[float] = None,
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
    freq_ghz = _require_pim_freq_ghz(pim_freq_ghz)
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
                pim_freq_ghz=float(freq_ghz),
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
            base_latency = float(cycles) / (float(freq_ghz) * 1000000000.0)
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
                    pim_freq_ghz=float(freq_ghz),
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
            pim_freq_ghz=float(freq_ghz),
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
        latency = float(cycles) / (float(freq_ghz) * 1000000000.0)
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
                pim_freq_ghz=float(freq_ghz),
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
