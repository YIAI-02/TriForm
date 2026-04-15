from __future__ import annotations
"""NPU backend: LLMCompass integration.

This module lazily imports LLMCompass (git submodule: submodules/LLMCompass)
and provides single-op latency estimators.

Public entry points used by CostModel:
- _normalize_npu_backend
- _llmcompass_guess_device_key
- _llmcompass_simulate_matmul_s / _softmax_s / _layernorm_s / _gelu_s
"""

import sys
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import logging
import math
from config import attach_local_debug_filter
import config as _config
from hardware import DeviceSpec
from dtype_utils import normalize_dtype_token

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)

# ---- LLMCompass integration (git submodule: submodules/LLMCompass) ----
def _ensure_llmcompass_on_path(start: Optional[Path]=None) -> Tuple[Path, Path]:
    """Add submodules/LLMCompass to sys.path so we can import `hardware_model` / `software_model`."""
    here = (start or Path(__file__)).resolve()
    for p in [here.parent] + list(here.parents):
        cand = p / 'submodules' / 'LLMCompass'
        if cand.exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return (cand, p)
    raise RuntimeError(f"Cannot find 'submodules/LLMCompass' above {here}")

def _normalize_npu_backend(backend: Optional[str]) -> Optional[str]:
    if backend is None:
        return None
    b = str(backend).strip().lower().replace('-', '_')
    b = b.replace(' ', '_')
    if b in ('fast', 'fastmode', 'fast_mode'):
        return 'fast'
    if b in ('ascend_310b_json', 'ascend310b_json', 'ascend_310b'):
        return 'ascend_310b_json'
    if b in ('llmcompass',):
        return 'llmcompass'
    raise ValueError(f"Unknown npu_backend='{backend}'. Expected one of: fast, ascend_310b_json, llmcompass")


_LLMCOMPASS_MODS: Optional[Dict[str, Any]] = None
_LLMCOMPASS_DEVICE_CACHE: Dict[str, Any] = {}
_LLMCOMPASS_LAT_CACHE_S: Dict[Tuple[Any, ...], float] = {}
_LLMCOMPASS_DEVICE_OVERRIDE: Dict[str, Any] = {}  # user-registered Device objects


_LLMCOMPASS_DEVICE_ALIASES: Dict[str, str] = {
    # LLMCompass' built-in database uses full device keys. Accept shorter names
    # that are commonly used in hardware JSON files.
    'a100': 'A100_80GB_fp16',
    'a100_80gb': 'A100_80GB_fp16',
    'a100_80gb_fp16': 'A100_80GB_fp16',
    'ga100': 'A100_80GB_fp16',
    'ga100_fp16': 'A100_80GB_fp16',
    'tpuv3': 'TPUv3',
    'tpu_v3': 'TPUv3',
    'tpu-v3': 'TPUv3',
    'mi210': 'MI210',
    'amd_mi210': 'MI210',
    'tpuv3_new': 'TPUv3_new',
    'tpu_v3_new': 'TPUv3_new',
}


def _initialize_llmcompass_modules() -> Dict[str, Any]:
    """Lazy-import LLMCompass modules; only used when npu_backend == 'llmcompass'."""
    llm_root, _ = _ensure_llmcompass_on_path()
    try:
        # Software models (single-op simulators)
        from software_model.matmul import Matmul, BatchedMatmul
        from software_model.softmax import Softmax
        from software_model.layernorm import LayerNorm
        from software_model.gelu import GeLU
        from software_model.utils import Tensor, DataType, data_type_dict

        # Hardware models / device database
        import hardware_model.device as device_mod
    except Exception as e:
        raise RuntimeError(
            "Failed to import LLMCompass. "
            "Make sure you added it as a git submodule at submodules/LLMCompass and installed its dependencies (e.g., scalesim). "
            f"Inner error: {e}"
        )
    return {
        'llm_root': llm_root,
        'Matmul': Matmul,
        'BatchedMatmul': BatchedMatmul,
        'Softmax': Softmax,
        'LayerNorm': LayerNorm,
        'GeLU': GeLU,
        'Tensor': Tensor,
        'DataType': DataType,
        'data_type_dict': data_type_dict,
        'device_mod': device_mod,
    }

def _get_llmcompass_mods() -> Dict[str, Any]:
    global _LLMCOMPASS_MODS
    if _LLMCOMPASS_MODS is None:
        _LLMCOMPASS_MODS = _initialize_llmcompass_modules()
    return _LLMCOMPASS_MODS

def _llmcompass_dtype_key(dtype: str) -> str:
    try:
        key = normalize_dtype_token(dtype, default='fp16')
    except Exception:
        key = str(dtype or '').strip().lower()
    return key

def _llmcompass_dtype(dtype: str):
    mods = _get_llmcompass_mods()
    dt_dict = mods.get('data_type_dict', None)
    if not isinstance(dt_dict, dict) or not dt_dict:
        raise RuntimeError("LLMCompass data_type_dict not found or empty (software_model/utils.py).")

    key = _llmcompass_dtype_key(dtype)
    if key not in dt_dict:
        supported = ', '.join(sorted(dt_dict.keys()))
        raise RuntimeError(
            f"LLMCompass does not support dtype='{dtype}' (normalized='{key}'). "
            f"Supported: {supported}."
        )
    return dt_dict[key]

def _llmcompass_guess_device_key(dev: DeviceSpec) -> str:
    # Allow explicit override in hardware json -> DeviceSpec.
    for k in (
        'llmcompass_kind',
        'llmcompass_device',
        'llmcompass_device_name',
        'llmcompass_arch',
        'arch',
    ):
        v = getattr(dev, k, None)
        if v:
            return str(v)
    return str(getattr(dev, 'name', ''))

def _llmcompass_normalize_device_key(device_key: str) -> str:
    s = (device_key or '').strip()
    sl = s.lower()
    sl_norm = sl.replace('-', '_').replace(' ', '_')
    if sl_norm in _LLMCOMPASS_DEVICE_ALIASES:
        return _LLMCOMPASS_DEVICE_ALIASES[sl_norm]
    if 'a100' in sl_norm or 'ga100' in sl_norm:
        return 'A100_80GB_fp16'
    if 'mi210' in sl_norm:
        return 'MI210'
    if 'tpu' in sl_norm and 'v3' in sl_norm:
        return 'TPUv3_new' if 'new' in sl_norm else 'TPUv3'
    if not s:
        raise ValueError(
            "LLMCompass device_key is empty. "
            "Please set dev.arch/dev.llmcompass_device (hardware JSON) or pass CostModel(llmcompass_device=...)."
        )
    return s

def _llmcompass_get_device(device_key: str):
    device_key = _llmcompass_normalize_device_key(device_key)
    if device_key in _LLMCOMPASS_DEVICE_OVERRIDE:
        v = _LLMCOMPASS_DEVICE_OVERRIDE[device_key]
        _LLMCOMPASS_DEVICE_CACHE[device_key] = v
        return v

    if device_key in _LLMCOMPASS_DEVICE_CACHE:
        return _LLMCOMPASS_DEVICE_CACHE[device_key]
    mods = _get_llmcompass_mods()
    device_mod = mods['device_mod']

    # Try a few common registries / factories in LLMCompass.
    dict_candidates = [
        'device_dict',
        'devices',
        'DEVICE_DICT',
        'device_module_dict',
        'DEVICE_MODULE_DICT',
    ]
    key_l = device_key.lower()
    for dname in dict_candidates:
        d = getattr(device_mod, dname, None)
        if isinstance(d, dict) and d:
            # Exact / case-insensitive
            for k, v in d.items():
                if str(k).lower() == key_l:
                    _LLMCOMPASS_DEVICE_CACHE[device_key] = v
                    return v
            # Fuzzy match (e.g., 'gpu_a100_80gb')
            for k, v in d.items():
                kl = str(k).lower()
                if key_l in kl or kl in key_l:
                    _LLMCOMPASS_DEVICE_CACHE[device_key] = v
                    return v

    # Factory functions
    for fname in ('get_device', 'get', 'get_device_by_name', 'create_device', 'make_device'):
        fn = getattr(device_mod, fname, None)
        if callable(fn):
            try:
                v = fn(device_key)
                if v is not None:
                    _LLMCOMPASS_DEVICE_CACHE[device_key] = v
                    return v
            except Exception:
                pass

    available: list[str] = []
    for dname in dict_candidates:
        d = getattr(device_mod, dname, None)
        if isinstance(d, dict) and d:
            available.extend([str(k) for k in d.keys()])
    available_msg = ', '.join(sorted(set(available))) if available else 'unknown'
    raise RuntimeError(
        f"Unable to obtain LLMCompass Device for device_key='{device_key}'. "
        f"Available built-in device keys: {available_msg}. "
        "Add `llmcompass_device` to the NPU entry in the hardware JSON, "
        "for example `\"llmcompass_device\": \"A100_80GB_fp16\"`, `\"TPUv3\"`, or `\"MI210\"`."
    )

def _llmcompass_default_compile_mode(device_key: str) -> str:
    dk = (device_key or '').lower()
    if 'tpu' in dk:
        return 'heuristic-TPU'
    return 'heuristic-GPU'

def _llmcompass_simulate_matmul_s(
    device_key: str,
    dtype: str,
    M: int,
    N: int,
    K: int,
    compile_mode: Optional[str] = None,
    *,
    batch: int = 1,
    batched: bool = False,
) -> float:
    device_key = _llmcompass_normalize_device_key(device_key)
    compile_mode = str(compile_mode or _llmcompass_default_compile_mode(device_key))
    M = int(M); N = int(N); K = int(K)
    batch = int(max(1, int(batch or 1)))
    batched = bool(batched)
    key = ('matmul', device_key, _llmcompass_dtype_key(dtype), compile_mode, int(M), int(N), int(K), int(batch), int(batched))
    if key in _LLMCOMPASS_LAT_CACHE_S:
        return _LLMCOMPASS_LAT_CACHE_S[key]
    mods = _get_llmcompass_mods()
    Matmul = mods['Matmul']
    BatchedMatmul = mods.get('BatchedMatmul', None)
    Tensor = mods['Tensor']
    dt = _llmcompass_dtype(dtype)
    dev_obj = _llmcompass_get_device(device_key)
    if batched:
        if BatchedMatmul is None:
            raise RuntimeError("LLMCompass BatchedMatmul not available (software_model/matmul.py missing BatchedMatmul).")
        op = BatchedMatmul(dt)
        a = Tensor([int(batch), int(M), int(K)], dt)
        b = Tensor([int(batch), int(K), int(N)], dt)
    else:
        op = Matmul(dt)
        a = Tensor([int(M), int(K)], dt)
        b = Tensor([int(K), int(N)], dt)
    op(a, b)

    try:
        lat = op.compile_and_simulate(dev_obj, compile_mode=compile_mode)
    except TypeError:
        # Older signature may not accept keyword.
        lat = op.compile_and_simulate(dev_obj, compile_mode)
    except Exception as e:
        # Best-effort fallback: roofline (still LLMCompass-native).
        try:
            lat = op.roofline_model(dev_obj)
        except Exception:
            raise RuntimeError(f"LLMCompass matmul simulation failed: {e}")

    lat_s = float(lat)
    _LLMCOMPASS_LAT_CACHE_S[key] = lat_s
    return lat_s

def _llmcompass_simulate_softmax_s(device_key: str, dtype: str, M: int, N: int, compile_mode: Optional[str]=None) -> float:
    device_key = _llmcompass_normalize_device_key(device_key)
    compile_mode = str(compile_mode or _llmcompass_default_compile_mode(device_key))
    key = ('softmax', device_key, _llmcompass_dtype_key(dtype), compile_mode, int(M), int(N))
    if key in _LLMCOMPASS_LAT_CACHE_S:
        return _LLMCOMPASS_LAT_CACHE_S[key]
    mods = _get_llmcompass_mods()
    Softmax = mods['Softmax']
    Tensor = mods['Tensor']
    dt = _llmcompass_dtype(dtype)
    dev_obj = _llmcompass_get_device(device_key)
    op = Softmax(dt)
    x = Tensor([int(M), int(N)], dt)
    op(x)
    try:
        lat = op.compile_and_simulate(dev_obj, compile_mode=compile_mode)
    except TypeError:
        lat = op.compile_and_simulate(dev_obj)
    except Exception as e:
        try:
            lat = op.roofline_model(dev_obj)
        except Exception:
            raise RuntimeError(f"LLMCompass softmax simulation failed: {e}")
    lat_s = float(lat)
    _LLMCOMPASS_LAT_CACHE_S[key] = lat_s
    return lat_s

def _llmcompass_simulate_layernorm_s(device_key: str, dtype: str, M: int, N: int, compile_mode: Optional[str]=None) -> float:
    device_key = _llmcompass_normalize_device_key(device_key)
    compile_mode = str(compile_mode or _llmcompass_default_compile_mode(device_key))
    key = ('layernorm', device_key, _llmcompass_dtype_key(dtype), compile_mode, int(M), int(N))
    if key in _LLMCOMPASS_LAT_CACHE_S:
        return _LLMCOMPASS_LAT_CACHE_S[key]
    mods = _get_llmcompass_mods()
    LayerNorm = mods['LayerNorm']
    Tensor = mods['Tensor']
    dt = _llmcompass_dtype(dtype)
    dev_obj = _llmcompass_get_device(device_key)
    op = LayerNorm(dt)
    x = Tensor([int(M), int(N)], dt)
    op(x)
    try:
        lat = op.compile_and_simulate(dev_obj, compile_mode)
    except Exception as e:
        try:
            lat = op.roofline_model(dev_obj)
        except Exception:
            raise RuntimeError(f"LLMCompass layernorm simulation failed: {e}")
    lat_s = float(lat)
    _LLMCOMPASS_LAT_CACHE_S[key] = lat_s
    return lat_s

def _llmcompass_simulate_gelu_s(device_key: str, dtype: str, data_len: int, compile_mode: Optional[str]=None) -> float:
    device_key = _llmcompass_normalize_device_key(device_key)
    compile_mode = str(compile_mode or _llmcompass_default_compile_mode(device_key))
    key = ('gelu', device_key, _llmcompass_dtype_key(dtype), compile_mode, int(data_len))
    if key in _LLMCOMPASS_LAT_CACHE_S:
        return _LLMCOMPASS_LAT_CACHE_S[key]
    mods = _get_llmcompass_mods()
    GeLU = mods['GeLU']
    Tensor = mods['Tensor']
    dt = _llmcompass_dtype(dtype)
    dev_obj = _llmcompass_get_device(device_key)
    op = GeLU(dt)
    x = Tensor([int(data_len)], dt)
    op(x)
    try:
        lat = op.compile_and_simulate(dev_obj, compile_mode)
    except Exception as e:
        try:
            lat = op.roofline_model(dev_obj)
        except Exception:
            raise RuntimeError(f"LLMCompass GeLU simulation failed: {e}")
    lat_s = float(lat)
    _LLMCOMPASS_LAT_CACHE_S[key] = lat_s
    return lat_s

