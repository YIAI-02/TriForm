from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time
from pathlib import Path
import json
import os
import logging
from model_definition import ModelShape, make_model_def
from cost_model import DTYPE_BYTES
from optimizations import apply_optimizations_to_graph
from task_graph import TaskGraph, TaskNode
from config import attach_local_debug_filter
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)
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
def _jsonable(x: Any) -> Any:
    """Best-effort conversion to JSON-serializable objects."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    # Common small objects
    try:
        import pathlib

        if isinstance(x, pathlib.Path):
            return str(x)
    except Exception:
        pass
    # Fallback
    try:
        return _jsonable(vars(x))
    except Exception:
        return str(x)
        
def _edges(g: TaskGraph) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for u in g.nodes.keys():
        for v in g.successors(u):
            out.append((str(u), str(v)))
    return out


def summarize_graph(g: TaskGraph) -> Dict[str, Any]:
    node_cnt = len(g.nodes)
    edge_cnt = sum(len(list(g.successors(u))) for u in g.nodes.keys())

    total_weight_new = 0
    total_weight_orig = 0
    weighted_nodes = 0

    q_cnt = 0
    ws_cnt = 0
    a_cnt = 0
    attn_cnt = 0

    for n in g.nodes.values():
        optd = {}
        try:
            optd = (n.attrs or {}).get("opt") or {}
        except Exception:
            optd = {}

        if "quantization" in optd:
            q_cnt += 1
        if "weight_sparsity" in optd:
            ws_cnt += 1
        if "activation_sparsity" in optd:
            a_cnt += 1
        if "attention_sparsity" in optd:
            attn_cnt += 1

        has_w = bool(getattr(n, "weight_id", None)) and int(getattr(n, "weight_size", 0) or 0) > 0
        if has_w:
            weighted_nodes += 1
            total_weight_new += int(getattr(n, "weight_size", 0) or 0)
            total_weight_orig += int(optd.get("orig_weight_size", getattr(n, "weight_size", 0) or 0) or 0)

    ratio = (float(total_weight_new) / float(total_weight_orig)) if total_weight_orig > 0 else 1.0
    return {
        "node_count": int(node_cnt),
        "edge_count": int(edge_cnt),
        "weighted_node_count": int(weighted_nodes),
        "total_weight_size_bytes": int(total_weight_new),
        "total_orig_weight_size_bytes": int(total_weight_orig),
        "weight_size_ratio_new_over_orig": float(ratio),
        "opt_tagged_counts": {
            "quantization": int(q_cnt),
            "weight_sparsity": int(ws_cnt),
            "activation_sparsity": int(a_cnt),
            "attention_sparsity": int(attn_cnt),
        },
    }


def dump_task_graph(
    g: TaskGraph,
    *,
    out_dir: str,
    tag: str = "graph",
    shape: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    include_edges: bool = True,
    write_effects_tsv: bool = True,
) -> Dict[str, str]:
    """Dump the graph to disk.

    Returns a dict of written file paths:
      {"full_json": ..., "effects_tsv": ...}
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = str(time.time_ns())
    base = os.path.join(out_dir, f"{tag}_{stamp}")

    # ---- Full JSON dump ----
    nodes: List[Dict[str, Any]] = []
    for nid, n in g.nodes.items():
        nodes.append(
            {
                "id": str(getattr(n, "id", nid)),
                "name": str(getattr(n, "name", "")),
                "flops": float(getattr(n, "flops", 0.0) or 0.0),
                "bytes_read": float(getattr(n, "bytes_read", 0.0) or 0.0),
                "bytes_write": float(getattr(n, "bytes_write", 0.0) or 0.0),
                "weight_id": getattr(n, "weight_id", None),
                "weight_size": int(getattr(n, "weight_size", 0) or 0),
                "allowed": _jsonable(getattr(n, "allowed", {}) or {}),
                "attrs": _jsonable(getattr(n, "attrs", {}) or {}),
                "pred": [str(x) for x in (g.predecessors(str(nid)) or [])],
                "succ": [str(x) for x in (g.successors(str(nid)) or [])],
            }
        )

    payload: Dict[str, Any] = {
        "summary": summarize_graph(g),
        "shape": _jsonable(shape) if shape is not None else None,
        "cfg": _jsonable(cfg) if cfg is not None else None,
        "nodes": nodes,
    }
    if include_edges:
        payload["edges"] = _jsonable(_edges(g))

    full_json = base + "_full.json"
    with open(full_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    written = {"full_json": full_json}

    # ---- Compact TSV: see optimization effects quickly ----
    if write_effects_tsv:
        effects_tsv = base + "_effects.tsv"
        with open(effects_tsv, "w", encoding="utf-8") as f:
            f.write(
                "id\tname\tlayer\thead_shard\tweight_id\torig_weight_size\tnew_weight_size\t"
                "q_mode\tq_wbits\tq_gs\tws_pattern\tws_density\tws_storage\t"
                "act_density\tattn_pattern\tattn_density\n"
            )
            for nid, n in g.nodes.items():
                attrs = getattr(n, "attrs", {}) or {}
                optd = (attrs.get("opt") or {}) if isinstance(attrs, dict) else {}

                layer = attrs.get("layer") if isinstance(attrs, dict) else None
                head_shard = attrs.get("head_shard") if isinstance(attrs, dict) else None

                orig_w = optd.get("orig_weight_size")
                if orig_w is None:
                    orig_w = int(getattr(n, "weight_size", 0) or 0)
                new_w = int(getattr(n, "weight_size", 0) or 0)

                q = optd.get("quantization") or {}
                ws = optd.get("weight_sparsity") or {}
                a = optd.get("activation_sparsity") or {}
                att = optd.get("attention_sparsity") or {}

                f.write(
                    f"{str(getattr(n,'id',nid))}\t{str(getattr(n,'name',''))}\t{layer}\t{head_shard}\t"
                    f"{getattr(n,'weight_id',None)}\t{orig_w}\t{new_w}\t"
                    f"{q.get('mode','')}\t{q.get('weight_bits','')}\t{q.get('group_size','')}\t"
                    f"{ws.get('pattern','')}\t{ws.get('density','')}\t{ws.get('storage','')}\t"
                    f"{a.get('density','')}\t{att.get('pattern','')}\t{att.get('density','')}\n"
                )

        written["effects_tsv"] = effects_tsv

    return written

def load_shape_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def parse_model_shape_from_file(family: str, variant: str, batch: int, max_seq_len: int, override: Dict[str,Any]) -> ModelShape:
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

def build_graph(cfg: Dict[str, Any]):
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
    md = make_model_def(family)
    dtype_bytes = DTYPE_BYTES.get(cfg.get('dtype','fp16'), 2)

    g = md.build(shape, dtype_bytes=dtype_bytes)

    try:
        apply_optimizations_to_graph(
            g,
            cfg,
            base_weight_dtype_bytes=int(dtype_bytes),
            shape=shape,
        )
    except Exception as e:
        # Only be loud when user explicitly wants debug/dump.
        if bool(cfg.get("debug", False)) or bool(cfg.get("dump_graph", False)):
            logger.exception("apply_optimizations_to_graph failed: %s", e)

    # Optional: dump the final graph (after optimizations) for inspection.
    if bool(cfg.get("dump_graph", False)) or bool(cfg.get("dump_task_graph", False)):
        out_dir = str(
            cfg.get("dump_graph_dir")
            or cfg.get("result_dir")
            or os.path.join("./output", "graph_dumps")
        )
        tag = str(
            cfg.get("dump_graph_tag")
            or f"{family}_{variant}_"\
               f"B{batch}_S{int(cfg.get('prefill_len', 0) or 0)}_T{int(cfg.get('decode_len', 0) or 0)}_"
               f"{cfg.get('dtype','fp16')}"
        ).replace(" ", "")
        written = dump_task_graph(g, out_dir=out_dir, tag=tag, shape=shape, cfg=cfg)
        # Print paths so user can find them even when logging is off.
        try:
            logger.debug(f"[GRAPH-DUMP] full_json: {written.get('full_json')}")
            if written.get("effects_tsv"):
                logger.debug(f"[GRAPH-DUMP] effects_tsv: {written.get('effects_tsv')}")
        except Exception:
            pass

    return g, shape