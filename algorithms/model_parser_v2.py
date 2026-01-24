from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from pathlib import Path
import json
from model_definition import ModelShape, make_model_def
from cost_model import DTYPE_BYTES
from optimizations import apply_optimizations_to_graph
from onnx_graph_parser import build_task_graph_from_onnx
DEFAULT_SHAPE_DIR = Path(__file__).resolve().parent.parent / "configs"

FILE_MAP = {
    ("llama","7b"):  "llama_7b_shape.json",
    ("llama","13b"): "llama_13b_shape.json",
    ("llama","70b"): "llama_70b_shape.json",
    ("mpt","7b"):    "mpt_7b_shape.json",
    ("mpt","30b"):   "mpt_30b_shape.json",
    ("palm","8b"):   "palm_8b_shape.json",
    ("palm","62b"):  "palm_62b_shape.json",
    ("palm","540b"): "palm_540b_shape.json",
    ("mixtral", "8x7b"):  "mixtral_8x7b_shape.json",
    ("qwen", "1.8b"): "qwen_1.8b_shape.json",
    ("qwen", "7b"):   "qwen_7b_shape.json",
    ("qwen", "14b"):  "qwen_14b_shape.json",
}

def load_shape_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def parse_model_shape_from_file(family: str, variant: str, batch: int, max_seq_len: int, override: Dict[str,Any]) -> ModelShape:
    # 0) Direct shape dict override (useful for ONNX/custom models).
    shape_dict = override.get('model_shape') or override.get('shape') or override.get('shape_override') or override.get('shape_params')
    if isinstance(shape_dict, dict) and shape_dict:
        def _gi(*keys, default=None):
            for k in keys:
                if k in shape_dict and shape_dict.get(k) is not None:
                    try:
                        return int(shape_dict.get(k))
                    except Exception:
                        pass
            return default
        layer_num0 = _gi('layer_num','num_layers','n_layers')
        dim0       = _gi('dim','hidden_dim','hidden_size')
        n_heads0    = _gi('n_heads','q_head_num','q_heads','num_heads')
        n_kv0       = _gi('n_kv_heads','kv_head_num','kv_heads','num_kv_heads', default=n_heads0)
        ffn0       = _gi('ffn_dim','intermediate_dim','mlp_dim', default=(int(dim0)*4 if dim0 else None))
        if layer_num0 and dim0 and n_heads0:
            shape = ModelShape(
                layer_num=int(layer_num0),
                dim=int(dim0),
                ffn_dim=int(ffn0 or (int(dim0)*4)),
                n_heads=int(n_heads0),
                n_kv_heads=int(n_kv0 or n_heads0),
                batch=int(batch or 1),
                max_seq_len=int(max_seq_len or 1),
            )
            # Optional MoE knobs
            for k in ('experts_per_layer','experts_top_k','moe_imbalance_factor'):
                if k in shape_dict and shape_dict.get(k) is not None:
                    try:
                        setattr(shape, k, int(shape_dict.get(k)))
                    except Exception:
                        try:
                            setattr(shape, k, float(shape_dict.get(k)))
                        except Exception:
                            pass
            return shape

    shape_file = Path(override.get("shape_file", ""))
    if shape_file and shape_file.is_file():
        data = load_shape_json(shape_file)
    else:
        fname = FILE_MAP.get((family.lower(), variant.lower()))
        if fname is None:
            raise ValueError(f"No shape file mapping for ({family},{variant}). Provide --config with 'shape_file' explicitly.")
        data = load_shape_json(DEFAULT_SHAPE_DIR / fname)

    hidden_dim = data.get("hidden_dim")
    layer_num = data.get("layer_num")
    intermediate_dim = data.get("intermediate_dim")
    q_head_num = data.get("q_head_num")
    kv_head_num = data.get("kv_head_num")

    shape =  ModelShape(
        layer_num=layer_num,
        dim=hidden_dim,
        ffn_dim=intermediate_dim,
        n_heads=q_head_num,
        n_kv_heads=kv_head_num,
        batch=batch,
        max_seq_len=max_seq_len,
    )
    experts_per_layer = data.get("experts_per_layer")
    if experts_per_layer is not None:
        setattr(shape, "experts_per_layer", experts_per_layer)
    experts_top_k = data.get("experts_top_k")
    if experts_top_k is not None:
        setattr(shape, "experts_top_k", experts_top_k)
    moe_imbalance_factor = data.get("moe_imbalance_factor")
    if moe_imbalance_factor is not None:
        setattr(shape, "moe_imbalance_factor", moe_imbalance_factor)
    return shape

def build_graph(cfg: Dict[str, Any], pim_shards: int = None, split_by: str = None):
    """
    Build a unified task graph that works for both prefill and decode.
    The graph structure is phase-independent; only runtime costs vary.
    """
    family = cfg.get("model_family", cfg.get("model_type","llama"))
    variant = cfg.get("model_variant", "7b")
    batch = cfg.get("batch", 1)
    # Use prefill_len + decode_len as max_seq_len for graph structure
    max_seq_len = cfg.get("prefill_len", 128) + cfg.get("decode_len", 32)
    
    shape = parse_model_shape_from_file(family, variant, batch, max_seq_len, cfg)
    dtype_bytes = DTYPE_BYTES.get(cfg.get('dtype','fp16'), 2)
    

    # Partition controls (optional): used for building head-sharded graphs.
    part_dim = str(
        split_by if split_by is not None
        else cfg.get('split_by', 'head') or 'head'
    ).strip().lower()

    if pim_shards is None:
        try:
            pim_shards = int(cfg.get('num_pim', cfg.get('pim_count', 1)) or 1)
        except Exception:
            pim_shards = 1
    shape.split_by = part_dim  # 'head_num' 会在 model_definition 里 normalize 成 'head'
    if "head" in part_dim:
        shape.split_shards = int(pim_shards or 0)


    onnx_path = cfg.get('onnx_path') or cfg.get('onnx_model') or cfg.get('onnx') or cfg.get('onnx_file')
    if isinstance(onnx_path, str) and onnx_path.strip():
        g = build_task_graph_from_onnx(str(onnx_path), cfg, dtype_bytes=int(dtype_bytes))
    else:
        md = make_model_def(family)
        g = md.build(shape, dtype_bytes=dtype_bytes)
    opt_cfg = cfg.get("optimizations") or cfg.get("optimization") or cfg.get("optim") or cfg
    apply_optimizations_to_graph(
        g,
        opt_cfg,
        base_weight_dtype_bytes=int(dtype_bytes),
        shape=shape,
    )
    return g, shape