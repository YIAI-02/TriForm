from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time
from pathlib import Path
import json
import os
import logging
from model_definition import ModelShape, make_model_def
from dtype_utils import dtype_bytes, normalize_dtype_token
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
    ("llama","405b"): "llama_405b_shape.json",
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
        family_key = str(family or "").lower()
        variant_key = str(variant or "").lower().replace("*", "x").replace("×", "x")
        fname = FILE_MAP.get((family_key, variant_key))
        if fname is None:
            raise ValueError(f"No shape file mapping for ({family},{variant}). Provide --config with 'shape_file' explicitly.")
        data = load_shape_json(DEFAULT_SHAPE_DIR / fname)

    def _pick(*keys: str, default: Any = None) -> Any:
        for src in (override, data):
            if not isinstance(src, dict):
                continue
            for key in keys:
                if key in src and src.get(key) is not None:
                    return src.get(key)
        return default

    hidden_dim = _pick("hidden_dim", "hidden_size")
    layer_num = _pick("layer_num", "num_hidden_layers")
    intermediate_dim = _pick("intermediate_dim", "intermediate_size", "ffn_dim")
    q_head_num = _pick("q_head_num", "num_attention_heads", "n_heads")
    kv_head_num = _pick("kv_head_num", "num_key_value_heads", "n_kv_heads", default=q_head_num)

    shape =  ModelShape(
        layer_num=layer_num,
        dim=hidden_dim,
        ffn_dim=intermediate_dim,
        n_heads=q_head_num,
        n_kv_heads=kv_head_num,
        batch=batch,
        max_seq_len=max_seq_len,
    )

    experts_per_layer = _pick("experts_per_layer", "num_local_experts", "num_experts")
    if experts_per_layer is not None:
        setattr(shape, "experts_per_layer", int(experts_per_layer))

    experts_top_k = _pick("experts_top_k", "num_experts_per_tok", "top_k")
    if experts_top_k is not None:
        setattr(shape, "experts_top_k", int(experts_top_k))

    active_experts_per_layer = _pick("active_experts_per_layer")
    if active_experts_per_layer is not None:
        setattr(shape, "active_experts_per_layer", int(active_experts_per_layer))

    moe_imbalance_factor = _pick("moe_imbalance_factor", default=1.0)
    if moe_imbalance_factor is not None:
        setattr(shape, "moe_imbalance_factor", float(moe_imbalance_factor))

    router_aux_loss_coef = _pick("router_aux_loss_coef")
    if router_aux_loss_coef is not None:
        setattr(shape, "router_aux_loss_coef", float(router_aux_loss_coef))

    router_jitter_noise = _pick("router_jitter_noise")
    if router_jitter_noise is not None:
        setattr(shape, "router_jitter_noise", float(router_jitter_noise))

    vocab_size = _pick("vocab_size")
    if vocab_size is not None:
        setattr(shape, "vocab_size", int(vocab_size))

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
    cfg['dtype'] = normalize_dtype_token(cfg.get('dtype', 'fp16'), default='fp16')
    dtype_bytes_value = dtype_bytes(cfg.get('dtype', 'fp16'), default='fp16')

    # ----------------------------
    # Validate TP sharding params
    # ----------------------------
    family_key = str(family or "").lower()
    is_mixtral = family_key == 'mixtral'

    # QKV TP (column parallel): shard by head groups to minimize cross-PIM traffic.
    # For Mixtral, `tp` is reserved for MoE total shard count, so attention only
    # follows an explicit `tp_qkv` override.
    try:
        Hq = int(getattr(shape, 'n_heads', getattr(shape, 'n_head', 0)) or 0)
    except Exception:
        Hq = 0
    try:
        Hkv = int(getattr(shape, 'n_kv_heads', getattr(shape, 'n_kv_head', Hq)) or Hq)
    except Exception:
        Hkv = Hq
    try:
        Hf = int(getattr(shape, 'ffn_dim', getattr(shape, 'hidden_dim', 0)) or 0)
    except Exception:
        Hf = 0

    tp_qkv_raw = cfg.get('tp_qkv', 1 if is_mixtral else cfg.get('tp', 1))
    try:
        tp_qkv = max(1, int(tp_qkv_raw or 1))
    except Exception:
        tp_qkv = 1

    tp_qkv_eff: int
    if tp_qkv <= 1:
        tp_qkv_eff = 1
    elif tp_qkv <= max(Hq, 1) and tp_qkv <= max(Hkv, 1):
        if (Hq % tp_qkv) != 0 or (Hkv % tp_qkv) != 0:
            raise ValueError(
                f"Invalid tp_qkv={tp_qkv}: require Hq%tp_qkv==0 and Hkv%tp_qkv==0 "
                f"(Hq={Hq}, Hkv={Hkv})."
            )
        tp_qkv_eff = tp_qkv
    elif tp_qkv > max(Hq, 1):
        # If tp_qkv > Hq, split by KV heads (kv-head baseline) to avoid cross-PIM.
        tp_qkv_eff = max(1, int(Hkv) if Hkv else 1)
    else:
        # tp_qkv between Hkv and Hq (e.g., Hkv < tp_qkv <= Hq) is not supported.
        raise ValueError(
            f"Invalid tp_qkv={tp_qkv}: unsupported when Hkv < tp_qkv <= Hq "
            f"(Hq={Hq}, Hkv={Hkv})."
        )

    # Dense tp_ffn is only for non-MoE models. Mixtral uses `tp` to control MoE total shards.
    tp_ffn_raw = cfg.get('tp_ffn', 1)
    try:
        tp_ffn = max(1, int(tp_ffn_raw or 1))
    except Exception:
        tp_ffn = 1
    if not is_mixtral and tp_ffn > 1:
        if Hf <= 0:
            raise ValueError(f"Invalid tp_ffn={tp_ffn}: unknown ffn_dim (Hf={Hf}).")
        if (Hf % tp_ffn) != 0:
            raise ValueError(
                f"Invalid tp_ffn={tp_ffn}: require ffn_dim%tp_ffn==0 (ffn_dim={Hf})."
            )

    tp_moe_total = 1
    tp_moe_expert_ffn = 1
    if is_mixtral:
        try:
            top_k = int(getattr(shape, 'experts_top_k', 2) or 2)
        except Exception:
            top_k = 2
        top_k = max(1, int(top_k))

        try:
            experts_total = int(getattr(shape, 'experts_per_layer', 0) or 0)
        except Exception:
            experts_total = 0
        if experts_total <= 0:
            raise ValueError(f"Invalid Mixtral shape: unknown experts_per_layer (E={experts_total}).")
        if top_k > experts_total:
            raise ValueError(
                f"Invalid Mixtral shape: experts_top_k={top_k} exceeds experts_per_layer={experts_total}."
            )

        # New Mixtral semantics:
        # - `tp` = total number of MoE FFN shards across the selected top-k experts.
        # - if tp <= top_k: selected experts are distributed across tp shards and each
        #   expert FFN remains unsplit.
        # - if tp > top_k: each selected expert is split into tp / top_k FFN shards.
        tp_moe_total_raw = cfg.get('tp', cfg.get('tp_ffn', cfg.get('tp_moe', 1)))
        try:
            tp_moe_total = max(1, int(tp_moe_total_raw or 1))
        except Exception:
            tp_moe_total = 1

        if tp_moe_total > top_k:
            if (tp_moe_total % top_k) != 0:
                raise ValueError(
                    f"Invalid Mixtral tp={tp_moe_total}: when tp > top_k, require tp%top_k==0 "
                    f"(top_k={top_k})."
                )
            tp_moe_expert_ffn = int(tp_moe_total // top_k)
            if Hf <= 0:
                raise ValueError(
                    f"Invalid Mixtral tp={tp_moe_total}: unknown ffn_dim (Hf={Hf})."
                )
            if (Hf % tp_moe_expert_ffn) != 0:
                raise ValueError(
                    f"Invalid Mixtral tp={tp_moe_total}: require ffn_dim%(tp/top_k)==0 "
                    f"(ffn_dim={Hf}, top_k={top_k}, per_expert_tp={tp_moe_expert_ffn})."
                )
        else:
            tp_moe_expert_ffn = 1
    else:
        tp_moe_total = 1
        tp_moe_expert_ffn = 1

    # Stash validated effective values for downstream components.
    cfg['tp_qkv_effective'] = int(tp_qkv_eff)
    cfg['tp_ffn_effective'] = int(1 if is_mixtral else tp_ffn)
    cfg['tp_moe_effective'] = int(tp_moe_total)
    cfg['tp_moe_total_effective'] = int(tp_moe_total)
    cfg['tp_moe_expert_ffn_effective'] = int(tp_moe_expert_ffn)

    g = md.build(shape, dtype_bytes=float(dtype_bytes_value), cfg=cfg)

    try:
        apply_optimizations_to_graph(
            g,
            cfg,
            base_weight_dtype_bytes=float(dtype_bytes_value),
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
            or f"{family}_{variant}_"f"B{batch}_S{int(cfg.get('prefill_len', 0) or 0)}_T{int(cfg.get('decode_len', 0) or 0)}_"
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

