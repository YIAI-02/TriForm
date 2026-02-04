from __future__ import annotations
"""NPU backend: Ascend 310B JSON runtime models.

This module loads latency model JSONs (linear / hinge regressions) and provides
predictors for:
- MMAD/GEMM (M,N,K)
- Softmax
- GeLU
- LayerNorm
"""
import json
import os
import re
from functools import lru_cache
from typing import Optional, Dict, Tuple
import logging

from config import attach_local_debug_filter

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)

# --------------------------------------------------- NPU 310B JSON --------------------------------------------------
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
                if (obj.get('family') == 'hinge') or (('alpha' in obj) and ('beta' in obj)):
                    return obj
        except Exception as e:
            logger.debug(str(f"[LN-Model] Failed to load '{p}': {e}"))
    return None
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def _compute_feature_vector(M: int, N: int, K: int, block_size: int, feature_names: list) -> Tuple[list, list]:
    MB = _ceil_div(M, block_size)
    NB = _ceil_div(N, block_size)
    KB = _ceil_div(K, block_size)
    base = {'MB': MB, 'NB': NB, 'KB': KB, 'tiles': MB * NB * KB, 'mn': MB * NB, 'sum_b': MB + NB + KB, 'M': M, 'N': N, 'K': K}
    feats = [float(base[name]) for name in feature_names]
    return (feats, [MB, NB, KB])

def _predict_mmad_latency_us_from_json(M: int, N: int, K: int) -> Optional[float]:
    model = _load_mmad_model_json()
    if model is None:
        logger.error(str('[MMAD-MODEL] ✗ Model loading failed, returning None'))
        return None
    block_size = int(model.get('block_size', 16))
    feature_names = model.get('features', ['tiles', 'mn', 'sum_b'])
    coefs = model['coefficients']
    feats, blocks = _compute_feature_vector(M, N, K, block_size, feature_names)
    y = float(coefs.get('b0', 0.0))
    for name, val in zip(feature_names, feats):
        coef = float(coefs.get(f'b_{name}', 0.0))
        contribution = coef * val
        y += contribution
    result = max(0.0, y)
    return result

def _map_op_to_mmad_dims(op: str, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: int) -> Optional[Tuple[int, int, int, int]]:
    if not op:
        logger.debug(str(f'[MAP-MMAD] ✗ op is None/empty, returning None'))
        return None
    op = op.lower()
    head_dim = dim // max(1, n_heads)
    result = None
    if op in ('q_proj', 'k_proj', 'v_proj', 'wo_proj'):
        result = (1, dim, dim, max(1, seqlen))
    elif op in ('ffn_up', 'ffn_gate'):
        result = (1, ffn_dim if ffn_dim > 0 else 4 * dim, dim, max(1, seqlen))
    elif op == 'ffn_down':
        result = (1, dim, ffn_dim if ffn_dim > 0 else 4 * dim, max(1, seqlen))
    elif op == 'score' and seqlen and head_dim:
        result = (1, seqlen, head_dim, max(1, n_heads * seqlen))
    elif op == 'output' and seqlen and head_dim:
        result = (1, head_dim, seqlen, max(1, n_heads * seqlen))
    else:
        logger.debug(str(f"[MAP-MMAD] ✗ No match for op='{op}'"))
    return result

def _predict_softmax_latency_us_from_json(M: int, K: int, *, phase: str='decode', causal: bool=True) -> Optional[float]:
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
    alpha = float(model.get('alpha', 0.0)); beta = float(model.get('beta', 0.0))
    T_us = alpha * x + beta
    return float(max(0.0, T_us))

