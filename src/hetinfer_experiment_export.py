"""Build Tiny-MoE experiment bundles from native DOPS exports and exact costs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import sys

from cost_model import CostModel
from hardware import demo_cluster
from model_parser import build_graph
from plan_label import PlanLabel
from scheduler.scheduler_base import SchedulerBase
from hetinfer_camc_profile_export import build_expert_service_lut, export_camc_bundle
from hetinfer_prior import validate_prior_artifact
from hetinfer_tensor_bindings_export import export_tensor_bindings_manifest_from_artifacts

NPU = "Ascend_910B_NPU0"
DEVICES = (NPU, "PIM0", "PIM1")


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")) + "\n")


def _service(cost: CostModel, node, device, batch: int, sequence: int, phase: str) -> float:
    label = PlanLabel(kv_in_pim=True, kv_place="pim")
    if node.weight_size:
        return float(cost.weighted_compute_stage(
            node, device, label, batch, sequence, phase,
            resident_weight_fmt=cost.weight_resident_format("ND", device)).total_s)
    return float(cost.node_device_cost(node, device, label, batch, sequence, phase))


def _expert_lut(cost: CostModel, nodes: dict, phase: str, maximum: int,
                output: Path, shard: int = 0, shards: int = 1) -> dict:
    result = {}
    anchor_counts = sorted({1, maximum, *(2 ** k for k in range(maximum.bit_length())
                                         if 2 ** k <= maximum)})
    for family, original in nodes.items():
        node = deepcopy(original)
        node.attrs["moe_token_fraction"] = 1.0
        npu = {}
        pim = {device: {} for device in DEVICES[1:]}
        for n_e in range(1 + shard, maximum + 1, shards):
            batch, sequence = (1, n_e) if phase == "prefill" else (n_e, 1)
            if n_e in anchor_counts:
                npu[n_e] = _service(cost, node, cost.cluster.devices[NPU], batch, sequence, phase)
            for device in pim:
                pim[device][n_e] = _service(cost, node, cost.cluster.devices[device],
                                           batch, sequence, phase)
            if (n_e - 1 - shard) % (64 * shards) == 0 or maximum - n_e < shards:
                print(f"AIM_EXACT {phase} {family} n_e={n_e}/{maximum} shard={shard}/{shards}", flush=True)
        if shards == 1:
            result[family] = build_expert_service_lut(
                max_tokens=maximum, activation_bytes_per_token=int(node.attrs["dim"]) * 2,
                npu_anchors={NPU: npu}, pim_measurements=pim)
        else:
            result[family] = {"activation_bytes_per_token": int(node.attrs["dim"]) * 2,
                              "npu": npu, "pim": pim}
    _write(output, {"phase": phase, "max_tokens": maximum,
                    "raw_measurements": shards > 1, "shard": shard, "shards": shards,
                    "pim_trace_scale_repeats": 0,
                    "expert_token_fraction": 1.0,
                    "mapping": "prefill:(batch=1,seqlen=n_e); decode:(batch=n_e,seqlen=1)",
                    "npu_anchor_counts": anchor_counts, "operators": result})
    return result


def _order(operators: list[dict]) -> list[str]:
    by_id = {op["op_id"]: op for op in operators}
    pending = {op_id: set(op["dependencies"]) for op_id, op in by_id.items()}
    result = []
    while pending:
        ready = sorted((op_id for op_id, deps in pending.items() if not deps),
                       key=lambda op_id: by_id[op_id]["operator_index"])
        if not ready:
            raise ValueError("Native DOPS graph is not a DAG")
        for op_id in ready:
            result.append(op_id)
            del pending[op_id]
        for deps in pending.values():
            deps.difference_update(ready)
    return result


def _experiment_cost(cfg):
    graph, shape = build_graph(cfg)
    cluster = demo_cluster(cfg)
    cost = CostModel(cluster, dtype="fp16", npu_backend="ascend_310b_lut", npu_lut_strict=True,
                     pim_config_path=Path(cfg["pim_config_path"]),
                     ramulator_config_path=Path(cfg["ramulator_config_path"]),
                     pim_trace_strict=True, pim_fast_mode=False,
                     pim_ramulator_timeout_s=1800)
    # Reuse one backing tensor set; trace length, batch, and phase stay exact.
    maximum = int(cfg["batch"]) * int(cfg["prefill_len"])
    cost.set_model_dict(cost.get_or_make_pim_model_dict(
        dim=shape.dim, n_heads=shape.n_heads, n_kv_heads=shape.n_kv_heads,
        ffn_dim=shape.ffn_dim, seqlen=max(maximum, cfg["max_seq_len"])))
    return graph, shape, cluster, cost


def _parallel_expert_lut(config_path, phase, maximum, output):
    families = ("ffn_w1", "ffn_w3", "swiglu", "ffn_w2")
    base_cache = Path(os.environ["PIM_LATENCY_CACHE_FILE"])
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    shards = min(maximum, max(1, workers // len(families)))
    tasks = [(family, shard) for family in families for shard in range(shards)]

    def measure(task):
        family, shard = task
        suffix = family + (f"_s{shard}" if shards > 1 else "")
        family_cache = base_cache.with_name(f"{base_cache.stem}.{family}.pkl")
        worker_cache = (base_cache.with_name(f"{base_cache.stem}.{family}.s{shard}.pkl")
                        if shards > 1 else family_cache)
        seed_cache = family_cache if family_cache.exists() else base_cache
        if not worker_cache.exists() and seed_cache.exists():
            shutil.copy2(seed_cache, worker_cache)
        env = {**os.environ, "PIM_LATENCY_CACHE_FILE": str(worker_cache)}
        log = output.parent / f"expert_{phase}_{suffix}.{os.environ['SLURM_JOB_ID']}.log"
        with log.open("w") as handle:
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--config", str(config_path), "--lut-phase", phase,
                            "--lut-family", family, "--lut-shard", str(shard),
                            "--lut-shards", str(shards)],
                           env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(f"AIM_SHARD_OK {phase} {suffix} max_n_e={maximum}", flush=True)
        return json.loads((output.parent / f"expert_lut_{phase}_{suffix}.json").read_text())

    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        parts = list(pool.map(measure, tasks))
    operators = {}
    for family in families:
        selected = [part["operators"][family] for task, part in zip(tasks, parts) if task[0] == family]
        if shards == 1:
            operators[family] = selected[0]
            continue
        npu, pim = {}, {device: {} for device in DEVICES[1:]}
        for part in selected:
            npu.update({int(n): value for n, value in part["npu"].items()})
            for device in pim:
                pim[device].update({int(n): value for n, value in part["pim"][device].items()})
        operators[family] = build_expert_service_lut(
            max_tokens=maximum, activation_bytes_per_token=selected[0]["activation_bytes_per_token"],
            npu_anchors={NPU: npu}, pim_measurements=pim)
    _write(output, {**parts[0], "operators": operators, "raw_measurements": False,
                    "shard": None, "shards": shards,
                    "npu_anchor_origin": "DOPS ascend_310b_lut evaluation, including existing internal LUT interpolation"})
    return operators


def build_experiment_bundle(config_path: Path) -> Path:
    if os.environ.get("PIM_TRACE_SCALE_REPEATS") != "0":
        raise RuntimeError("Set PIM_TRACE_SCALE_REPEATS=0 before importing DOPS")
    cfg = json.loads(config_path.read_text())
    output = config_path.parent
    graph, shape, cluster, cost = _experiment_cost(cfg)
    maximum = int(cfg["batch"]) * int(cfg["prefill_len"])
    prior = json.loads(Path(cfg["hetinfer_prior_out"]).read_text())
    manifest = json.loads(Path(cfg["hetinfer_network_out"]).read_text())
    is_moe = cfg["model_family"] == "mixtral"
    luts = {}
    if is_moe:
        for phase, limit in (("prefill", maximum), ("decode", int(cfg["batch"]))):
            luts[phase] = _parallel_expert_lut(config_path, phase, limit,
                                              output / f"expert_lut_{phase}.json")

    prior_ops = {op["op_id"]: op for op in prior["operators"]}
    default = {op["op_id"]: op["device_id"] for op in prior["expert_placement"]}
    service = {(item["op_id"], item["device_id"]): item["duration_s"]
               for item in prior["t_service"]}
    move_keys = {tuple(item[key] for key in ("tensor_id", "source_device_id",
                                            "destination_device_id", "bytes", "layout"))
                 for item in prior["legal_movement_routes"]}
    route_context = SimpleNamespace(cost=cost, cluster=cluster)

    def add_route(tensor: str, source: str, destination: str, size: int,
                  layout: str, duration: float) -> None:
        key = (tensor, source, destination, size, layout)
        if key in move_keys:
            return
        move_keys.add(key)
        item = dict(zip(("tensor_id", "source_device_id", "destination_device_id", "bytes", "layout"), key))
        prior["legal_movement_routes"].append(item)
        prior["t_move"].append({**item, "duration_s": duration})

    weight_catalog = {}
    metadata = {}
    spec_layers = []
    projections = []
    for network_index, network in enumerate(manifest["networks"]):
        phase = network["phase"]
        work = network["workload"]
        batch = work["batch"]
        sequence = work["sequence_length"]
        network_ids = {op["op_id"] for op in network["operators"]}
        kv_homes = {raw["layer_index"]: default[raw["op_id"]]
                    for raw in network["operators"] if raw["op_role"] == "KV_WRITE"}
        node_specs = []
        for raw in network["operators"]:
            op_id = raw["op_id"]
            if raw["op_role"] == "KV_WRITE":
                node = SimpleNamespace(name=raw["canonical_op_slot"], attrs={},
                                       weight_id=None, weight_size=0)
            else:
                node = graph.nodes[op_id.split(":", 2)[2]]
            family = node.name.lower()
            expert = node.attrs.get("expert_id")
            layer = raw["layer_index"]
            supernode = f"{phase}:{network_index}:L{layer}:expert:{expert}" if expert else op_id
            metadata[op_id] = {"family": family,
                               "timing_source": node.attrs.get("timing_source", "npu_lut_or_aim"),
                               "weight_id": node.weight_id}
            if node.weight_size:
                wid = str(node.weight_id)
                size = int(node.weight_size)
                load = {}
                for device_id in prior_ops[op_id]["legal_devices"]:
                    device = cluster.devices[device_id]
                    comm = float(cost.comm_cost(cluster.devices["CPU0"], device, size))
                    local = (float(cost.pim_local_weight_load_cost(size, "ND", dev=device).total_s)
                             if device.type == "pim" else
                             float(cost.npu_local_weight_load_cost(size, "ND",
                                   cost.weight_resident_format("ND", device), dev=device).total_s))
                    load[device_id] = {"transfer_s": comm, "local_write_format_s": local,
                                       "total_s": comm + local}
                    add_route(f"weight:{wid}", "CPU0", device_id, size, "ND", comm + local)
                weight_catalog[wid] = {"bytes": size, "load": load}
                prior["inputs"].append({
                    "consumer_op_id": op_id, "producer_op_id": None,
                    "tensor_id": f"weight:{wid}", "semantics": "data", "bytes": size,
                    "source_residencies": [{"device_id": "CPU0", "layout": "ND"}],
                    "destination_devices": list(prior_ops[op_id]["legal_devices"])})
            node_specs.append({
                "op_id": op_id, "layer_index": layer, "operator_index": raw["operator_index"],
                "operator_family": family, "placement_supernode": supernode,
                "parallel_group_hint": (f"{phase}:{network_index}:L{layer}:experts" if expert else None),
                "weight_home": "CPU0" if node.weight_size else None,
                "kv_home": (kv_homes[layer] if raw["canonical_op_slot"]
                            in {"k", "v", "qk", "sv", "k_write", "v_write"} else None),
                "expert_id": expert,
                "expert_service_buckets": luts[phase][family] if expert else [],
            })

        # Native scheduling is per operator; contract the default to the same
        # expert placement unit that Het-Infer is allowed to select.
        groups = {}
        for node in node_specs:
            if node["expert_id"]:
                groups.setdefault(node["placement_supernode"], []).append(node)
        for group, members in groups.items():
            legal = prior_ops[members[0]["op_id"]]["legal_devices"]
            selected = min(legal, key=lambda dev: sum(service[(node["op_id"], dev)] for node in members))
            for node in members:
                op_id = node["op_id"]
                if default[op_id] != selected:
                    projections.append({"op_id": op_id, "native_device": default[op_id],
                                        "supernode_device": selected})
                default[op_id] = selected

        by_spec = {node["op_id"]: node for node in node_specs}
        roles = {raw["op_id"]: raw["op_role"] for raw in network["operators"]}
        if is_moe:
            for entry in list(prior["inputs"]):
                producer, consumer = entry["producer_op_id"], entry["consumer_op_id"]
                if consumer not in network_ids or producer not in network_ids:
                    continue
                expert_node = (consumer if roles[producer] == "ROUTER" and roles[consumer] == "EXPERT"
                               else producer if roles[producer] == "EXPERT" and roles[consumer] == "COMBINE"
                               else None)
                if expert_node is None:
                    continue
                for bucket in by_spec[expert_node]["expert_service_buckets"]:
                    size = bucket["activation_bytes"]
                    for residency in entry["source_residencies"]:
                        for destination in entry["destination_devices"]:
                            duration = SchedulerBase._hetinfer_route_time_s(
                                route_context, cluster.devices[residency["device_id"]],
                                cluster.devices[destination], size, source_layout=residency["layout"])
                            add_route(entry["tensor_id"], residency["device_id"], destination,
                                      size, residency["layout"], duration)

        reference = next(node for node in graph.nodes.values() if node.name.upper() == "FFN_W1")
        reference = deepcopy(reference)
        reference.attrs["moe_token_fraction"] = 1.0
        flops = float(cost.estimate_flops(reference, batch, sequence, phase))
        capabilities = {}
        for domain, device_ids in (("NPU", (NPU,)), ("PIM", DEVICES[1:])):
            capabilities[domain] = {
                "effective_compute_flops_per_s": sum(flops / _service(cost, reference, cluster.devices[d], batch, sequence, phase) for d in device_ids),
                "effective_bandwidth_bytes_per_s": sum(float(cluster.devices[d].mem_bw_GBs) * 1e9 for d in device_ids),
                "queue_count": len(device_ids),
            }
        spec_layers.append({
            "network_index": network_index, "layer_class": "moe" if is_moe else "dense",
            "phase": phase, "batch_size": batch, "sequence_length": sequence,
            "past_kv_len": work["past_kv_len"], "query_len": work["query_len"],
            "router_top_k": 2 if is_moe else None, "sd_component": "none",
            "shape_bucket": f"b{batch}-past{work['past_kv_len']}-q{work['query_len']}",
            "capability_basis": "compute", "domain_capabilities": capabilities,
            "default_order": _order(network["operators"]),
            "nodes": [by_spec[op_id] for op_id in _order(network["operators"])],
        })

    for item in prior["expert_placement"]:
        item["device_id"] = default[item["op_id"]]
    spec = {"graph_id": prior["graph_id"], "workload_id": prior["workload_id"],
            "device_domains": {NPU: "NPU", "PIM0": "PIM", "PIM1": "PIM"},
            "layers": spec_layers}
    prior_artifact = validate_prior_artifact(prior)
    bindings_path = output / "experiment_tensor_bindings.json"
    export_tensor_bindings_manifest_from_artifacts(
        prior_artifact=prior, network_manifest=manifest, output=bindings_path)
    bindings = json.loads(bindings_path.read_text())
    bundle = output / "bundle"
    export_camc_bundle(prior_artifact=prior_artifact, network_manifest=manifest,
                       tensor_bindings=bindings, layer_spec=spec, output_dir=bundle)
    _write(bundle / "experiment.json", {
        "model": cfg["model_family"], "layer_count": int(shape.layer_num),
        "batch": cfg["batch"], "prefill": cfg["prefill_len"], "decode_rounds": int(cfg["decode_len"]),
        "weights": weight_catalog, "operators": metadata,
        "weight_backing_store": "CPU0", "weight_capacity_ratio": 0.95,
        "weight_capacity_bytes": {device: int(cluster.devices[device].mem_capacity_GB
                                             * 1024 ** 3 * 0.95) for device in DEVICES},
        "hardware_json": cfg["hardware_json"],
        "pim_trace_scale_repeats": 0, "moe_control_timing": cfg.get("moe_control_timing"),
        "default_placement_projection": projections,
        "produced_tensor": False, "produced_token": False})
    print(f"BUNDLE_OK {bundle}", flush=True)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--lut-phase", choices=("prefill", "decode"))
    parser.add_argument("--lut-family", choices=("ffn_w1", "ffn_w3", "swiglu", "ffn_w2"))
    parser.add_argument("--lut-shard", type=int, default=0)
    parser.add_argument("--lut-shards", type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the exporter on a Slurm compute node")
    if (args.lut_phase is None) != (args.lut_family is None):
        parser.error("--lut-phase and --lut-family must be supplied together")
    if args.lut_phase:
        if len(args.config) != 1:
            parser.error("One config is required for a LUT worker")
        cfg = json.loads(args.config[0].read_text())
        graph, _, _, cost = _experiment_cost(cfg)
        node = next(node for node in graph.nodes.values()
                    if "expert" in node.attrs and node.name.lower() == args.lut_family)
        maximum = int(cfg["batch"]) * (int(cfg["prefill_len"]) if args.lut_phase == "prefill" else 1)
        suffix = args.lut_family + (f"_s{args.lut_shard}" if args.lut_shards > 1 else "")
        _expert_lut(cost, {args.lut_family: node}, args.lut_phase, maximum,
                    args.config[0].parent / f"expert_lut_{args.lut_phase}_{suffix}.json",
                    args.lut_shard, args.lut_shards)
    else:
        for config in args.config:
            build_experiment_bundle(config)


if __name__ == "__main__":
    main()
