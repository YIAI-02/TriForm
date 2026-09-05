import json
from pathlib import Path

from export_hetinfer_tiny_suite import WORKLOADS, experiment_config
from model_parser import build_graph


def test_seven_workloads_keep_native_decode_snapshots_and_hardware():
    assert WORKLOADS == ((1, 16), (2, 16), (4, 16), (8, 16),
                         (1, 64), (1, 256), (1, 1024))
    for model in ("mixtral", "qwen"):
        for batch, prefill in WORKLOADS:
            cfg = experiment_config(model, batch, prefill)
            assert cfg["batch"] == batch and cfg["prefill_len"] == prefill
            assert cfg["decode_len"] == 32 and cfg["decode_sample_stride"] == 1
            assert cfg["max_seq_len"] == prefill + 32
            assert cfg["pim_trace_strict"] and not cfg["pim_fast_mode"]
            hardware = json.loads(Path(cfg["hardware_json"]).read_text())["hardware"]
            assert [(d["type"], d["mem_capacity_GB"]) for d in hardware["devices"]
                    if d["type"] != "cpu"] == [("npu", 16), ("pim", 16), ("pim", 16)]


def test_only_router_and_combine_get_explicit_analytic_npu_timing():
    cfg = experiment_config("mixtral", 1, 16)
    graph, shape = build_graph(cfg)
    assert (shape.dim, shape.ffn_dim, shape.layer_num, shape.n_heads, shape.n_kv_heads) == (1024, 4096, 4, 8, 2)
    marked = [node for node in graph.nodes.values() if "timing_source" in node.attrs]
    assert len(marked) == 8
    assert {node.name.upper() for node in marked} == {"MOE_ROUTER", "MOE_COMBINE"}
    assert all(node.allowed == {"cpu": False, "npu": True, "pim": False} for node in marked)
    assert sum(node.weight_size for node in graph.nodes.values()) == 826343424
    cfg.pop("moe_control_timing")
    unmodified, _ = build_graph(cfg)
    assert all("timing_source" not in node.attrs for node in unmodified.nodes.values())
