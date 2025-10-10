from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from pathlib import Path
import json

from model_definition import ModelShape, make_model_def
from cost_model import DTYPE_BYTES

DEFAULT_SHAPE_DIR = Path("../configs")

FILE_MAP = {
    ("llama","7b"):  "llama_7b_shape.json",
    ("llama","13b"): "llama_13b_shape.json",
    ("llama","70b"): "llama_70b_shape.json",
    ("mpt","7b"):    "mpt_7b_shape.json",
    ("mpt","30b"):   "mpt_30b_shape.json",
    ("palm","8b"):   "palm_8b_shape.json",
    ("palm","62b"):  "palm_62b_shape.json",
    ("palm","540b"): "palm_540b_shape.json",
}

def load_shape_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def parse_model_shape_from_file(family: str, variant: str, batch: int, seq_len: int, override: Dict[str,Any]) -> ModelShape:
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

    return ModelShape(
        layer_num=layer_num,
        dim=hidden_dim,
        ffn_dim=intermediate_dim,
        n_heads=q_head_num,
        n_kv_heads=kv_head_num,
        batch=batch,
        seq_len=seq_len,
    )

def build_graph(cfg: Dict[str, Any], seq_len: int, phase: str):
    family = cfg.get("model_family", cfg.get("model_type","llama"))
    variant = cfg.get("model_variant", "7b")
    batch = cfg.get("batch", 1)
    shape = parse_model_shape_from_file(family, variant, batch, seq_len, cfg)
    md = make_model_def(family)
    dtype_bytes = DTYPE_BYTES.get(cfg.get('dtype','fp16'), 2)
    return md.build(shape, phase=phase, dtype_bytes=dtype_bytes), shape
